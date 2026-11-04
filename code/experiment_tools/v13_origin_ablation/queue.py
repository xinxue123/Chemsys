"""Couple responses only for identical scientific inputs within one start."""
from pathlib import Path
import json
import time

from filelock import FileLock
from paper_v12_subagent.transport import read,digest,file_sha,write_once,write_once_bytes
from . import VARIANT,MODEL,EFFORT

CURRENT=None
INSTRUCTIONS='''Answer one chemical experimental-decision request as the LLM.
This is a fresh independent request. Read ONLY this assigned UTF-8 JSON in full:
{input}
It contains exactly system and public. Follow those instructions using only the
supplied own observations and domain information. Never read any other files,
search the repository or web, recall benchmark outcome tables, infer outcomes
from identifiers, or delegate. Tools may only read this input and write your own
requested JSON response, once, without Markdown fences, to:
{output}
Do not run experiments, inspect validation, amend that response, or implement a
scoring heuristic/response generator. Give concise public rationales, not private
reasoning. After writing the response, finish with a short completion message.'''


def base(root):return (Path(root)/'experiments/runs'/VARIANT).resolve()
def rel(root,p):return Path(p).resolve().relative_to(Path(root).resolve()).as_posix()
def context(system,public):return dict(system=system,public=public,model=MODEL,effort=EFFORT)


def response_dir(root,cell,kind,ctx):
    # Full SHA256 is verified in every entry; compact filenames also work with
    # Windows atomic-write temporary suffixes and the resolved data junction.
    return base(root)/'responses'/cell/digest(dict(kind=kind,context=ctx))[:24]


def seed(root,cell,kind,original,raw_path,provenance):
    ctx=context(original['system'],original['public'])
    if original.get('model',MODEL)!=MODEL or original.get('effort',EFFORT)!=EFFORT:
        raise ValueError('Historical requested model differs')
    target=response_dir(root,cell,kind,ctx)
    data=Path(raw_path).read_bytes();json.loads(data)
    entry=dict(mode='historical_frozen_response',cell=cell,kind=kind,context_sha256=digest(ctx),
               raw_path=rel(root,target/'seeded.raw.json'),raw_sha256=file_sha(raw_path),provenance=provenance)
    write_once(target/'context.json',ctx)
    if (target/'entry.json').exists():
        old=read(target/'entry.json')
        if old['raw_sha256']!=entry['raw_sha256']:raise ValueError('Conflicting original response within one start')
        return False
    write_once_bytes(target/'seeded.raw.json',data);write_once(target/'entry.json',entry)
    return True


def request(root,tag,kind,query,system,public,validator,*,selected_validator=None):
    job=CURRENT
    if job is None:raise ValueError('No registered active job')
    root=Path(root).resolve();ctx=context(system,public)
    folder=response_dir(root,job['cell'],kind,ctx)
    call=base(root)/'calls'/job['id']/f'{query:02}_{kind}'
    write_once(call/'context.json',ctx)
    folder.mkdir(parents=True,exist_ok=True)
    with FileLock(str(folder/'.response.lock'),timeout=43200):
        if not (folder/'entry.json').exists():
            if job.get('arm')=='full':
                raise ValueError('Full replay requested an unarchived scientific input')
            _fresh(root,job,folder,ctx,kind,query,validator,selected_validator)
        entry=read(folder/'entry.json')
        if entry['context_sha256']!=digest(ctx) or file_sha(root/entry['raw_path'])!=entry['raw_sha256']:
            raise ValueError('Response-map custody mismatch')
        raw=read(root/entry['raw_path']);plan=validator(raw)
        meta=dict(job=job['id'],cell=job['cell'],kind=kind,query=query,context_sha256=digest(ctx),
                  response_map_entry=rel(root,folder/'entry.json'),entry_sha256=file_sha(folder/'entry.json'),
                  raw_sha256=entry['raw_sha256'],origin=entry['mode'],model=MODEL,effort=EFFORT)
        meta['request_sha256']=digest(meta);write_once(call/'request.json',meta)
        receipt=dict(mode='exact_input_response_map',origin=entry['mode'],raw_sha256=entry['raw_sha256'],
                     entry_path=rel(root,folder/'entry.json'),entry_sha256=file_sha(folder/'entry.json'))
        result=dict(raw=raw,plan=plan,context_sha256=digest(ctx),attempts=entry.get('attempts',0),
                    response_receipt=receipt,request_relpath=rel(root,call/'request.json'))
        write_once(call/'validated.json',dict(plan_sha256=digest(plan),response=receipt))
        return result


def _fresh(root,job,folder,ctx,kind,query,validator,selected_validator):
    write_once(folder/'context.json',ctx)
    error=prior=lock=None
    for attempt in range(1,4):
        af=folder/f'a{attempt}'
        public=ctx['public'] if error is None else dict(original=ctx['public'],schema_repair_only=error,
            original_response=prior,selected_id_must_remain=lock)
        packet=dict(system=ctx['system'],public=public)
        opaque=digest(dict(cell=job['cell'],folder=rel(root,folder),attempt=attempt))[:24]
        io=root/'experiments/agent_io'/VARIANT/opaque
        write_once(io/'input.json',packet)
        meta=dict(job=job['id'],cell=job['cell'],kind=kind,query=query,attempt=attempt,
            context_sha256=digest(ctx),input=rel(root,io/'input.json'),output=rel(root,io/'output.json'),
            input_sha256=file_sha(io/'input.json'),model=MODEL,effort=EFFORT,fork_turns='none')
        meta['request_sha256']=digest(meta);write_once(af/'request.json',meta)
        invocation=dict(task_name=f'oa_{opaque}_{kind}_{attempt}',fork_turns='none',model=MODEL,
            reasoning_effort=EFFORT,message=INSTRUCTIONS.format(input=(io/'input.json').as_posix(),output=(io/'output.json').as_posix()))
        write_once(af/'invocation.json',invocation)
        while not (af/'response_receipt.json').exists():time.sleep(.15)
        receipt=read(af/'response_receipt.json')
        if receipt['request_sha256']!=meta['request_sha256'] or receipt['raw_sha256']!=file_sha(af/'response.raw.json'):
            raise ValueError('Fresh response custody mismatch')
        raw=None
        try:
            raw=read(af/'response.raw.json')
            if selected_validator:
                selected=selected_validator(raw)
                if lock is not None and selected!=lock:raise ValueError('Repair changed valid selected point')
                lock=selected
            plan=validator(raw)
        except (ValueError,KeyError,TypeError) as exc:
            error=str(exc);prior=raw
            write_once(af/'validation.json',dict(status='SCHEMA_REJECTED',error=error,selected_lock=lock));continue
        write_once(af/'validation.json',dict(status='VALIDATED',plan_sha256=digest(plan)))
        write_once(folder/'entry.json',dict(mode='fresh_subagent',cell=job['cell'],kind=kind,
            context_sha256=digest(ctx),raw_path=rel(root,af/'response.raw.json'),raw_sha256=file_sha(af/'response.raw.json'),
            request_path=rel(root,af/'request.json'),receipt_path=rel(root,af/'response_receipt.json'),attempts=attempt))
        return
    raise ValueError('Three schema attempts exhausted; no new benchmark outcome')


def pending(root):
    result=[]
    for p in sorted((base(root)/'responses').glob('*/*/a*/request.json')):
        if not (p.parent/'response_receipt.json').exists() and (p.parent/'invocation.json').exists():
            r=read(p)
            result.append(dict(request=str(p),invocation=read(p.parent/'invocation.json'),
                assigned=(p.parent/'assignment.json').exists(),output_exists=(Path(root)/r['output']).exists()))
    return result


def assign(root,path,result):
    p=Path(path).resolve()
    if not p.is_relative_to(base(root)):raise ValueError('Request outside new namespace')
    req=read(p);inv=read(p.parent/'invocation.json')
    actual=result.get('task_name',result.get('agent_id'))
    if actual!='/root/'+inv['task_name']:raise ValueError('Actual agent differs from prepared invocation')
    write_once(p.parent/'assignment.json',dict(spawn_request=inv,spawn_result=result,agent_id=actual,
        request_sha256=req['request_sha256']))


def seal(root,path):
    p=Path(path).resolve()
    if not p.is_relative_to(base(root)):raise ValueError('Request outside new namespace')
    req=read(p);a=read(p.parent/'assignment.json')
    if a['request_sha256']!=req['request_sha256'] or file_sha(Path(root)/req['input'])!=req['input_sha256']:
        raise ValueError('Assignment or packet changed')
    data=(Path(root)/req['output']).read_bytes();data.decode('utf-8')
    write_once_bytes(p.parent/'response.raw.json',data)
    write_once(p.parent/'response_receipt.json',dict(request_sha256=req['request_sha256'],
        raw_sha256=file_sha(p.parent/'response.raw.json'),assignment_sha256=file_sha(p.parent/'assignment.json'),
        agent_id=a['agent_id'],usage_available=False))


def status(root):
    p=read(base(root)/'protocol.json');jobs=[]
    for j in p['jobs']:
        sf=base(root)/'runs'/j['id']/'state.json';s=read(sf) if sf.exists() else {}
        jobs.append(dict(id=j['id'],arm=j['arm'],status=s.get('status','WAITING'),queries=len(s.get('history',[]))))
    ps=pending(root)
    return dict(complete=sum(j['status']=='COMPLETE' for j in jobs),total=len(jobs),
        full_replay_complete=sum(j['arm']=='full' and j['status']=='COMPLETE' for j in jobs),
        control_complete=sum(j['arm']!='full' and j['status']=='COMPLETE' for j in jobs),
        control_queries=sum(j['queries'] for j in jobs if j['arm']!='full'),pending_requests=len(ps),
        assigned_pending=sum(j['assigned'] for j in ps),active=[j for j in jobs if j['status'] not in ('WAITING','COMPLETE')])
