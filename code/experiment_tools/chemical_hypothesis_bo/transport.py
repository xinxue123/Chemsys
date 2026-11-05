"""Immutable fresh-agent requests; no credentials, API calls or fake outputs."""
from pathlib import Path
from datetime import datetime,timezone
import json
import time
from filelock import FileLock
from kernel_contrast_qlogei.study import read,write as _write,sha,ROOT
from . import VARIANT,MODEL,EFFORT,BACKEND
from .hypotheses import digest

BASE=ROOT/'experiments/runs'/VARIANT
INSTRUCTIONS='''Act as the language model for ONE fresh independent scientific request.
Read ONLY the assigned UTF-8 JSON input in full:
{input_path}
It contains system and public. Follow system and respond using the supplied experimental
definitions, molecular identities and this trajectory's own measurements. Do not read any
other files, inspect benchmark sources or other trials, browse the internet, retrieve outcome
tables, infer rankings from IDs, or delegate. Do not compute observations or run an optimizer.
Write your OWN requested JSON response, without Markdown fences, to:
{output_path}
Tools may only read this input and write that output. Do not implement a heuristic generator,
edit a response after validation or seek hidden targets. Give concise public scientific
rationales, not private reasoning. Finish after writing with a short completion message.'''


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with FileLock(str(path)+'.write.lock',timeout=60):_write(path,value)


def request(task,kind,query,public,validator):
    from .prompts import SYSTEM
    value=dict(system=SYSTEM,public=public,model=MODEL,effort=EFFORT,backend=BACKEND)
    key=digest(value);folder=BASE/'requests'/task/key;write(folder/'context.json',value)
    if (folder/'validated.json').exists():
        saved=read(folder/'validated.json');assert saved['context_sha256']==key
        assert saved['plan']==validator(saved['raw']);return saved
    for attempt in range(1,3):
        af=folder/f'attempt{attempt}'
        if attempt==1:payload=public
        else:payload=dict(original=public,schema_repair_only=read(folder/'attempt1/error.json'))
        req=dict(system=SYSTEM,public=payload,model=MODEL,effort=EFFORT,backend=BACKEND,
            fork_turns='none',task=task,kind=kind,query=query,attempt=attempt,context_sha256=key)
        req['request_sha256']=digest(req);write(af/'request.json',req)
        opaque=digest(dict(request=req['request_sha256'],namespace=VARIANT))
        io=ROOT/'experiments/agent_io'/VARIANT/opaque
        write(io/'input.json',dict(system=SYSTEM,public=payload))
        export=dict(input=(io/'input.json').relative_to(ROOT).as_posix(),
            output=(io/'output.json').relative_to(ROOT).as_posix(),input_sha256=sha(io/'input.json'))
        write(af/'export.json',export)
        invocation=dict(task_name=f'chem_bg_{task}_{kind}_{query:02d}_a{attempt}',fork_turns='none',model=MODEL,reasoning_effort=EFFORT,
            message=INSTRUCTIONS.format(input_path=(io/'input.json').as_posix(),output_path=(io/'output.json').as_posix()))
        write(af/'invocation.json',invocation)
        started=time.monotonic()
        while not (af/'response_receipt.json').exists():
            if time.monotonic()-started>7200:raise TimeoutError('Independent LLM response not supplied within2h')
            time.sleep(.25)
        receipt=read(af/'response_receipt.json')
        assert receipt['request_sha256']==req['request_sha256']
        assert receipt['raw_sha256']==sha(af/'response.raw.json')
        assert receipt['assignment_sha256']==sha(af/'assignment.json')
        raw=read(af/'response.raw.json')
        try:plan=validator(raw)
        except (ValueError,KeyError,TypeError) as exc:
            write(af/'error.json',dict(error=str(exc),repair_policy='schema only; no experiment outcome obtained'))
            continue
        saved=dict(context_sha256=key,raw=raw,plan=plan,receipt=receipt,request=(af/'request.json').relative_to(ROOT).as_posix())
        write(folder/'validated.json',saved);return saved
    raise ValueError('Two fresh-agent schema attempts failed')


def pending():
    result=[]
    for path in sorted((BASE/'requests').glob('*/*/attempt*/invocation.json')):
        folder=path.parent
        if (folder/'response_receipt.json').exists():continue
        export=read(folder/'export.json')
        result.append(dict(request=str(folder/'request.json'),invocation=read(path),
            assigned=(folder/'assignment.json').exists(),output_exists=(ROOT/export['output']).exists()))
    return result


def assign(request_path,result):
    folder=Path(request_path).resolve().parent;req=read(folder/'request.json');inv=read(folder/'invocation.json')
    assert folder.is_relative_to(BASE)
    assert req['request_sha256']==digest({k:v for k,v in req.items() if k!='request_sha256'})
    agent=result.get('task_name',result.get('agent_id'))
    if agent!='/root/'+inv['task_name']:raise ValueError('Actual spawn result does not match requested fresh agent')
    for other in (BASE/'requests').rglob('assignment.json'):
        if other!=folder/'assignment.json' and read(other)['agent_id']==agent:raise ValueError('A request must have a fresh isolated agent')
    write(folder/'assignment.json',dict(agent_id=agent,spawn_request=inv,spawn_result=result,
        backend=BACKEND,provider_attestation=False))


def seal(request_path):
    folder=Path(request_path).resolve().parent;req=read(folder/'request.json');export=read(folder/'export.json')
    assignment=read(folder/'assignment.json')
    assert assignment['spawn_request']==read(folder/'invocation.json')
    assert assignment['agent_id']=='/root/'+assignment['spawn_request']['task_name']
    assert sha(ROOT/export['input'])==export['input_sha256']
    if (folder/'response_receipt.json').exists():
        receipt=read(folder/'response_receipt.json');assert receipt['raw_sha256']==sha(folder/'response.raw.json')
        return dict(status='ALREADY_SEALED',request=str(request_path),agent=assignment['agent_id'])
    data=(ROOT/export['output']).read_bytes();raw=json.loads(data.decode('utf-8-sig'))
    write(folder/'response.raw.json',raw)
    write(folder/'response_receipt.json',dict(request_sha256=req['request_sha256'],raw_sha256=sha(folder/'response.raw.json'),
        assignment_sha256=sha(folder/'assignment.json'),agent_id=assignment['agent_id'],model=MODEL,effort=EFFORT,
        usage_available=False,sealed_UTC=datetime.now(timezone.utc).isoformat()))
    return dict(status='SEALED',request=str(request_path),agent=assignment['agent_id'])
