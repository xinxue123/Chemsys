"""Metadata-only convenience wrapper; actual tool-returned names are CLI inputs."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from llm_bo_handoff import study
from cold_start_v13 import transport


def main():
    action=sys.argv[1]
    study.write(study.BASE/'operations_helper.json',dict(path=Path(__file__).relative_to(ROOT).as_posix(),
        sha256=study.sha(__file__),role='metadata only; no scientific inputs or outputs generated'))
    with study.transport_scope():
        pending=transport.pending(ROOT)
        if action=='assign':
            names=set(sys.argv[2:])
            results=[]
            for row in pending:
                canonical='/root/'+row['invocation']['task_name']
                if canonical in names:
                    assert not row['assigned']
                    results.append(transport.record_spawn(ROOT,row['request'],dict(task_name=canonical)))
                    names.remove(canonical)
            assert not names,names
            print(json.dumps(dict(assigned=results)))
            return
        if action!='cycle':
            raise ValueError(action)
        sealed=[]
        for row in pending:
            if row['assigned'] and row['output_exists']:
                sealed.append(transport.seal(ROOT,row['request']))
        pending=transport.pending(ROOT)
        unassigned=[r for r in pending if not r['assigned']]
        proposals=[]
        for r in unassigned[:3]:
            req=study.read(r['request'])
            export=study.read(Path(r['request']).parent/'export.json')
            proposals.append(dict(task_name=r['invocation']['task_name'],model=r['invocation']['model'],
                effort=r['invocation']['reasoning_effort'],input=export['input'],output=export['output'],
                request=r['request'],query=req['query'],attempt=req['attempt']))
        states=[]
        for sf in (study.BASE/'runs').glob('*/state.json'):
            s=study.read(sf)
            states.append(dict(job=s['job']['id'],queries=len(s['history']),status=s['status'],
                calls=s['knowledge_calls']))
        print(json.dumps(dict(sealed=len(sealed),pending=len(pending),
            assigned_running=[r['invocation']['task_name'] for r in pending if r['assigned']],
            next=proposals,states=states)))


if __name__=='__main__':
    main()
