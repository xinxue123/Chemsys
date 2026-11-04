"""Operational agent namespace adapter; no science, prompts or policy changes."""
from pathlib import Path
from datetime import datetime,timezone
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from v13_origin_ablation import queue

PARENT='/root/oa_dispatch'


def registration():
    p=queue.read(queue.base(ROOT)/'operational_dispatch_registration.json')
    if p['helper_sha256']!=queue.file_sha(__file__) or p['allowed_agent_parent']!=PARENT:
        raise ValueError('Operational dispatcher registration changed')
    return p


def main():
    action=sys.argv[1];base=queue.base(ROOT)
    if action=='register':
        protocol=queue.read(base/'protocol.json')
        p=dict(kind='nested_dispatch_namespace_only',registered_utc=datetime.now(timezone.utc).isoformat(),
            allowed_agent_parent=PARENT,helper_path=Path(__file__).resolve().relative_to(ROOT).as_posix(),
            helper_sha256=queue.file_sha(__file__),scientific_protocol_sha256=protocol['protocol_sha256'],
            algorithm_prompts_GP_budgets_unchanged=True,fresh_children_fork_none=True,
            reason='Continuous metadata-only dispatch; coordinator must never inspect scientific packets or outcome data',
            first_root_wrapper_note='First probe dispatch reflowed line breaks in the external wrapper only; exact system/public input unchanged; subsequent dispatches copy prepared invocation')
        queue.write_once(base/'operational_dispatch_registration.json',p)
        print(json.dumps(p));return
    registration()
    if action=='status':print(json.dumps(queue.status(ROOT)));return
    if action=='pending':
        limit=int(sys.argv[2]) if len(sys.argv)>2 else 12
        print(json.dumps(queue.pending(ROOT)[:limit]));return
    payload=json.loads(sys.stdin.read())
    if action=='seal':
        for path in payload:queue.seal(ROOT,Path(path))
    elif action=='assign':
        for row in payload:
            path=Path(row['request']).resolve()
            if not path.is_relative_to(base):raise ValueError('Request outside registered namespace')
            req=queue.read(path);inv=queue.read(path.parent/'invocation.json');result=row['spawn_result']
            actual=result.get('task_name',result.get('agent_id'))
            if actual!=PARENT+'/'+inv['task_name']:raise ValueError('Actual child differs from allowed prepared request')
            if inv['model']!='gpt-6.1-terra' or inv['reasoning_effort']!='medium' or inv['fork_turns']!='none':
                raise ValueError('Requested experimental model changed')
            queue.write_once(path.parent/'assignment.json',dict(spawn_request=inv,spawn_result=result,
                agent_id=actual,request_sha256=req['request_sha256'],
                operational_registration_sha256=queue.file_sha(base/'operational_dispatch_registration.json')))
    else:raise ValueError(action)
    print(json.dumps(dict(status='OK',count=len(payload))))


if __name__=='__main__':main()
