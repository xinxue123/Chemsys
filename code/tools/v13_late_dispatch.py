"""Fast metadata-only driver for the registered fresh experimental LLM queue."""
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from v13_late_validation import queue


def main():
    action=sys.argv[1]
    if action=='pending':
        rows=queue.pending(ROOT)
        print(json.dumps(rows));return
    if action=='status':
        b=ROOT/'experiments/runs/v13_late_validation_20261103';p=queue.read(b/'protocol.json');rows=[]
        for j in p['jobs']:
            sf=b/j['id']/'runs'/j['id']/'state.json'
            s=queue.read(sf) if sf.exists() else {}
            rows.append(dict(id=j['id'],status=s.get('status','WAITING'),queries=len(s.get('history',[])),new_queries=max(0,len(s.get('history',[]))-j['prefix_queries'])))
        print(json.dumps(dict(complete=sum(x['status']=='COMPLETE' for x in rows),total=len(rows),
            new_queries=sum(x['new_queries'] for x in rows),pending_requests=len(queue.pending(ROOT)),
            active=[x for x in rows if x['status'] not in ('WAITING','COMPLETE')])));return
    value=json.loads(sys.stdin.read())
    if action=='assign':
        for row in value if isinstance(value,list) else [value]:
            path=Path(row['request']);result=row['spawn_result'];inv=queue.read(path.parent/'invocation.json')
            actual=result.get('task_name',result.get('agent_id'))
            if actual=='/root/'+inv['task_name']:queue.assign(ROOT,path,result)
            elif actual=='/root/late_llm_dispatch/'+inv['task_name']:
                req=queue.read(path)
                if req['request_sha256']!=queue.digest({k:v for k,v in req.items() if k!='request_sha256'}):raise ValueError('Changed request')
                queue.write_once(path.parent/'assignment.json',dict(spawn_request=inv,spawn_result=result,agent_id=actual,request_sha256=req['request_sha256']))
                queue.write_once(path.parent/'dispatch_namespace_receipt.json',dict(
                    purpose='Actual canonical child path for dedicated metadata dispatcher; input/output bytes and model policy unchanged',
                    dispatcher='/root/late_llm_dispatch',actual_agent_id=actual,
                    adapter_sha256=queue.file_sha(__file__),registered_scientific_sources_modified=False))
            else:raise ValueError('Actual agent is outside the registered dispatch namespaces')
    elif action=='seal':
        for path in value['requests']:queue.seal(ROOT,Path(path))
    else:raise ValueError(action)
    print('OK')


if __name__=='__main__':main()
