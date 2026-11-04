"""Exact public packets to fresh experimental decision agents; no model heuristic."""
from pathlib import Path
import json
import time
from paper_v12_subagent.transport import read,digest,file_sha,write_once,write_once_bytes
from . import MODEL,EFFORT,VARIANT

CURRENT=None
INSTRUCTIONS='''Answer one experimental-decision request as the language model.
This is a fresh independent request. Read ONLY the assigned UTF-8 JSON in full:
{input}
It contains exactly system and public. Follow those instructions using only
supplied observations and public variable meanings. Do not inspect other files,
search the repository/web, recall benchmark outcome tables, infer yield from
candidate IDs, or delegate. Tools may only read this input and write your own
requested JSON response, with no Markdown fences, to:
{output}
Do not run experiments, inspect validation, amend an existing response, or write
a response generator/scoring heuristic. Supply concise public rationales, never
private reasoning. After writing output, finish with a brief completion message.'''


def request(root,tag,kind,query,system,public,validator,*,selected_validator=None):
    root=Path(root);job=CURRENT
    if job is None:raise ValueError('No registered job')
    context=dict(system=system,public=public)
    folder=root/'experiments/runs'/VARIANT/'requests'/job/f'{query:02}_{kind}'
    write_once(folder/'context.json',context)
    error=prior=lock=None
    for attempt in range(1,4):
        af=folder/f'a{attempt}'
        payload=public if error is None else dict(original=public,schema_repair_only=error,original_response=prior,selected_id_must_remain=lock)
        packet=dict(system=system,public=payload)
        opaque=digest(dict(job=job,query=query,kind=kind,attempt=attempt,context=packet))[:24]
        io=root/'experiments/agent_io'/VARIANT/opaque
        write_once(io/'input.json',packet)
        meta=dict(job=job,kind=kind,query=query,attempt=attempt,context_sha256=digest(context),
            input_sha256=file_sha(io/'input.json'),input=(io/'input.json').relative_to(root).as_posix(),
            output=(io/'output.json').relative_to(root).as_posix(),model=MODEL,effort=EFFORT,fork_turns='none')
        meta['request_sha256']=digest(meta)
        write_once(af/'request.json',meta)
        invocation=dict(task_name=f'lv_{job}_{query:02}_{kind}_{attempt}',fork_turns='none',model=MODEL,reasoning_effort=EFFORT,
            message=INSTRUCTIONS.format(input=(io/'input.json').as_posix(),output=(io/'output.json').as_posix()))
        write_once(af/'invocation.json',invocation)
        while not (af/'response_receipt.json').exists():time.sleep(.15)
        receipt=read(af/'response_receipt.json')
        if receipt['request_sha256']!=meta['request_sha256'] or receipt['raw_sha256']!=file_sha(af/'response.raw.json'):
            raise ValueError('Response custody mismatch')
        try:
            raw=read(af/'response.raw.json')
            if selected_validator:
                selected=selected_validator(raw)
                if lock is not None and lock!=selected:raise ValueError('Repair changed valid selected point')
                lock=selected
            plan=validator(raw)
        except (ValueError,KeyError,TypeError) as exc:
            error=str(exc);prior=raw if 'raw' in locals() else None
            write_once(af/'validation.json',dict(status='SCHEMA_REJECTED',error=error,selected_lock=lock));continue
        result=dict(raw=raw,plan=plan,context_sha256=digest(context),attempts=attempt,response_receipt=receipt,
            request_relpath=(af/'request.json').relative_to(root).as_posix())
        write_once(af/'validation.json',dict(status='VALIDATED',plan_sha256=digest(plan)))
        return result
    raise ValueError('Three schema attempts exhausted; no new oracle query')


def pending(root):
    base=Path(root)/'experiments/runs'/VARIANT/'requests';rows=[]
    for p in sorted(base.glob('*/*/a*/request.json')):
        if not (p.parent/'response_receipt.json').exists():
            rows.append(dict(request=str(p),invocation=read(p.parent/'invocation.json'),assigned=(p.parent/'assignment.json').exists()))
    return rows


def assign(root,path,result):
    p=Path(path);req=read(p);inv=read(p.parent/'invocation.json')
    actual=result.get('task_name',result.get('agent_id'))
    if actual!='/root/'+inv['task_name']:raise ValueError('Actual agent differs')
    write_once(p.parent/'assignment.json',dict(spawn_request=inv,spawn_result=result,agent_id=actual,request_sha256=req['request_sha256']))


def seal(root,path):
    p=Path(path);req=read(p);a=read(p.parent/'assignment.json')
    if a['request_sha256']!=req['request_sha256']:raise ValueError('Assignment changed')
    if file_sha(Path(root)/req['input'])!=req['input_sha256']:raise ValueError('Public packet changed')
    data=(Path(root)/req['output']).read_bytes();data.decode('utf-8')
    write_once_bytes(p.parent/'response.raw.json',data)
    write_once(p.parent/'response_receipt.json',dict(request_sha256=req['request_sha256'],raw_sha256=file_sha(p.parent/'response.raw.json'),
        assignment_sha256=file_sha(p.parent/'assignment.json'),agent_id=a['agent_id'],usage_available=False))
