"""Fresh isolated experimental LLM requests with actual assignment receipts."""
from pathlib import Path
from datetime import datetime,timezone
import json
import time
from filelock import FileLock
from kernel_contrast_qlogei.study import ROOT,read,write as raw_write,sha
from chemical_hypothesis_bo.hypotheses import digest
from . import VARIANT,MODEL,EFFORT
from hypothesis_scope_loop import PARAMETERS as P
BASE=(ROOT/'experiments/runs'/VARIANT).resolve()
INSTRUCTIONS='''Read ONLY this UTF-8 input completely, including system/public; recover any truncation:
{input_path}
Act as the independent experimental LLM and follow its scientific/JSON instructions.
Use only supplied background and own history. No other files, browsing, outcome lookup,
delegation or response factory. Write your OWN final JSON ONCE to:
{output_path}
Complete checks in memory before writing. Do not edit after writing. Finish immediately.'''


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with FileLock(str(path)+'.write.lock',timeout=60):raw_write(path,value)


def request(task,stage,system,public,validator):
    context=dict(system=system,public=public,model=MODEL,effort=EFFORT,stage=stage)
    key=digest(context);folder=BASE/'requests'/task/key;write(folder/'context.json',context)
    if (folder/'validated.json').exists():
        saved=read(folder/'validated.json');assert saved['plan']==validator(saved['raw']);return saved
    for attempt in range(1,P['maximum_schema_attempts']+1):
        af=folder/f'attempt{attempt}'
        payload=public if attempt==1 else dict(original=public,schema_repair_only=read(folder/'attempt1/error.json'))
        req=dict(system=system,public=payload,task=task,stage=stage,attempt=attempt,
            model=MODEL,effort=EFFORT,fork_turns='none',context_sha256=key)
        req['request_sha256']=digest(req);write(af/'request.json',req)
        opaque=digest(dict(namespace=VARIANT,request=req['request_sha256']))
        io=ROOT/'experiments/agent_io'/VARIANT/opaque
        write(io/'input.json',dict(system=system,public=payload))
        write(af/'export.json',dict(input=(io/'input.json').relative_to(ROOT).as_posix(),
            output=(io/'output.json').relative_to(ROOT).as_posix(),input_sha256=sha(io/'input.json')))
        invocation=dict(task_name=f'hsm_{task}_{stage}_a{attempt}',fork_turns='none',model=MODEL,reasoning_effort=EFFORT,
            message=INSTRUCTIONS.format(input_path=(io/'input.json').as_posix(),output_path=(io/'output.json').as_posix()))
        write(af/'invocation.json',invocation)
        started=time.monotonic()
        while not (af/'receipt.json').exists():
            if time.monotonic()-started>7200:raise TimeoutError('Actual independent LLM response pending for2h')
            time.sleep(.25)
        receipt=read(af/'receipt.json');assert receipt['raw_sha256']==sha(af/'response.raw.json')
        assert receipt['assignment_sha256']==sha(af/'assignment.json')
        assert receipt['request_sha256']==req['request_sha256']
        raw=read(af/'response.raw.json')
        try:plan=validator(raw)
        except (ValueError,TypeError,KeyError) as exc:
            write(af/'error.json',dict(error=str(exc),repair_scope='schema/input validity only; no new observations'))
            continue
        saved=dict(context_sha256=key,raw=raw,plan=plan,receipt=receipt,request=(af/'request.json').relative_to(ROOT).as_posix())
        write(folder/'validated.json',saved);return saved
    raise ValueError('Two actual fresh-agent schema attempts failed')


def pending():
    rows=[]
    for path in sorted((BASE/'requests').glob('*/*/attempt*/invocation.json')):
        af=path.parent
        if (af/'receipt.json').exists():continue
        export=read(af/'export.json')
        rows.append(dict(request=str(af/'request.json'),invocation=read(path),export=export,
            assigned=(af/'assignment.json').exists(),output_exists=(ROOT/export['output']).exists()))
    return rows


def assign(request_path,result):
    af=Path(request_path).resolve().parent;assert af.is_relative_to(BASE)
    inv=read(af/'invocation.json');agent=result['task_name']
    assert agent=='/root/'+inv['task_name']
    for other in BASE.glob('requests/*/*/attempt*/assignment.json'):
        if other!=af/'assignment.json':assert read(other)['agent_id']!=agent
    write(af/'assignment.json',dict(agent_id=agent,spawn_request=inv,spawn_result=result,provider_attestation=False))


def complete(request_path):
    af=Path(request_path).resolve().parent;assert af.is_relative_to(BASE)
    assignment=read(af/'assignment.json');export=read(af/'export.json')
    assert (ROOT/export['output']).exists()
    write(af/'completion.json',dict(agent_id=assignment['agent_id'],
        source='Actual collaboration FINAL_ANSWER notification acknowledged by coordinator',
        output_sha256=sha(ROOT/export['output']),acknowledged_UTC=datetime.now(timezone.utc).isoformat()))


def seal(request_path):
    af=Path(request_path).resolve().parent;assert af.is_relative_to(BASE)
    req=read(af/'request.json');assignment=read(af/'assignment.json');export=read(af/'export.json');completion=read(af/'completion.json')
    assert completion['agent_id']==assignment['agent_id'] and completion['output_sha256']==sha(ROOT/export['output'])
    assert assignment['spawn_request']==read(af/'invocation.json')
    assert sha(ROOT/export['input'])==export['input_sha256']
    raw=read(ROOT/export['output']);write(af/'response.raw.json',raw)
    write(af/'receipt.json',dict(agent_id=assignment['agent_id'],request_sha256=req['request_sha256'],
        raw_sha256=sha(af/'response.raw.json'),assignment_sha256=sha(af/'assignment.json'),
        model=MODEL,effort=EFFORT,usage_available=False,completion_sha256=sha(af/'completion.json'),sealed_UTC=datetime.now(timezone.utc).isoformat()))
