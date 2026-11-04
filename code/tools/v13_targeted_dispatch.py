"""Metadata-only dispatch for the explicitly reduced four-cell study."""
from datetime import datetime,timezone
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools'),str(ROOT/'code/tools')]
from v13_origin_ablation import queue
import v13_origin_nested_dispatch as original_dispatch


def base():return queue.base(ROOT)/'targeted'
def protocol():return queue.read(base()/'protocol.json')


def pending():
    p=protocol();ids={j['id'] for j in p['jobs']}
    result=[]
    for row in queue.pending(ROOT):
        if queue.read(row['request'])['job'] in ids:result.append(row)
    return result


def status():
    p=protocol();rs=[]
    for j in p['jobs']:
        sf=base()/'runs'/j['id']/'state.json';s=queue.read(sf) if sf.exists() else {}
        rs.append(dict(id=j['id'],arm=j['arm'],status=s.get('status','WAITING'),queries=len(s.get('history',[]))))
    ps=pending()
    return dict(complete=sum(r['status']=='COMPLETE' for r in rs),total=16,
        full_replay_complete=sum(r['arm']=='full' and r['status']=='COMPLETE' for r in rs),
        control_complete=sum(r['arm']!='full' and r['status']=='COMPLETE' for r in rs),control_total=12,
        control_queries=sum(r['queries'] for r in rs if r['arm']!='full'),control_query_budget=240,
        pending_requests=len(ps),assigned_pending=sum(r['assigned'] for r in ps),
        active=[r for r in rs if r['status'] not in ('WAITING','COMPLETE')])


def main():
    action=sys.argv[1]
    if action=='register':
        p=protocol()
        value=dict(kind='targeted_dispatch_scope',registered_utc=datetime.now(timezone.utc).isoformat(),
            helper_path=Path(__file__).resolve().relative_to(ROOT).as_posix(),helper_sha256=queue.file_sha(__file__),
            targeted_protocol_sha256=p['protocol_sha256'],allowed_job_ids=[j['id'] for j in p['jobs']],
            allowed_agent_parent='/root/oa_dispatch',no_notes_cancelled=True,
            original_namespace_registration_sha256=queue.file_sha(queue.base(ROOT)/'operational_dispatch_registration.json'))
        queue.write_once(base()/'dispatch_registration.json',value);print(json.dumps(value));return
    reg=queue.read(base()/'dispatch_registration.json')
    if reg['helper_sha256']!=queue.file_sha(__file__) or reg['targeted_protocol_sha256']!=protocol()['protocol_sha256']:
        raise ValueError('Reduced dispatch scope changed')
    if action=='pending':print(json.dumps(pending()));return
    if action=='status':print(json.dumps(status()));return
    payload=json.loads(sys.stdin.read())
    ids=set(reg['allowed_job_ids'])
    if action=='assign':
        for row in payload:
            req=queue.read(row['request'])
            if req['job'] not in ids:raise ValueError('Cancelled job cannot be assigned')
            queue.write_once(Path(row['request']).parent/'targeted_scope.json',dict(
                targeted_protocol_sha256=reg['targeted_protocol_sha256'],scope_registration_sha256=queue.file_sha(base()/'dispatch_registration.json')))
        # Original helper checks and records actual nested spawn receipts.
        old_stdin=sys.stdin
        import io
        try:sys.stdin=io.StringIO(json.dumps(payload));original_dispatch.main()
        finally:sys.stdin=old_stdin
    elif action=='seal':
        for path in payload:
            if queue.read(path)['job'] not in ids:raise ValueError('Cancelled request cannot be sealed by this scheduler')
            queue.seal(ROOT,Path(path))
        print(json.dumps(dict(status='OK',count=len(payload))))
    else:raise ValueError(action)


if __name__=='__main__':main()
