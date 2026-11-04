"""Metadata-only dispatch; scientific packets stay with fresh leaf responders."""
from datetime import datetime,timezone
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from program_kf_weight import study,MODEL,EFFORT
from v13_origin_ablation import queue

PARENT='/root/program_dispatch'


def pending():
    with study.transport_scope():return queue.pending(ROOT)


def status():
    p=queue.read(study.base()/'protocol.json');rows=[]
    for job in p['jobs']:
        folder=study.base()/('prefix/'+job['cell'] if job['phase']=='prefix' else 'runs/'+job['id'])
        s=queue.read(folder/'state.json') if (folder/'state.json').exists() else {}
        rows.append(dict(id=job['id'],phase=job['phase'],status=s.get('status','WAITING'),
            queries=s.get('query',len(s.get('history',[])))))
    ps=pending()
    return dict(rows=rows,complete=sum(r['status']=='COMPLETE' for r in rows),
        new_experiment_queries=sum(r['queries'] for r in rows if r['phase']=='traces'),
        historical_prefixes=sum(r['queries'] for r in rows if r['phase']=='prefix'),
        pending=len(ps),assigned_pending=sum(x['assigned'] for x in ps))


def main():
    action=sys.argv[1]
    if action=='register':
        p=study.check()
        queue.write_once(study.base()/'dispatch_registration.json',dict(parent=PARENT,
            helper_sha256=queue.file_sha(__file__),protocol_sha256=p['protocol_sha256'],
            registered_utc=datetime.now(timezone.utc).isoformat()))
        print(json.dumps(dict(status='REGISTERED',parent=PARENT)));return
    reg=queue.read(study.base()/'dispatch_registration.json')
    if reg['helper_sha256']!=queue.file_sha(__file__):raise ValueError('Dispatcher changed')
    if action=='pending':print(json.dumps(pending()));return
    if action=='status':print(json.dumps(status()));return
    payload=json.loads(sys.stdin.read())
    with study.transport_scope():
        if action=='assign':
            for row in payload:
                path=Path(row['request']).resolve();req=queue.read(path);inv=queue.read(path.parent/'invocation.json')
                if not path.is_relative_to(study.base().resolve()):raise ValueError('Wrong namespace')
                actual=row['spawn_result'].get('task_name',row['spawn_result'].get('agent_id'))
                if actual!=PARENT+'/'+inv['task_name']:raise ValueError('Actual nested agent differs')
                if (inv['model'],inv['reasoning_effort'],inv['fork_turns'])!=(MODEL,EFFORT,'none'):
                    raise ValueError('Experimental model changed')
                queue.write_once(path.parent/'assignment.json',dict(spawn_request=inv,spawn_result=row['spawn_result'],
                    agent_id=actual,request_sha256=req['request_sha256'],registration_sha256=queue.file_sha(study.base()/'dispatch_registration.json')))
        elif action=='seal':
            for path in payload:queue.seal(ROOT,Path(path))
        else:raise ValueError(action)
    print(json.dumps(dict(status='OK',count=len(payload))))


if __name__=='__main__':main()
