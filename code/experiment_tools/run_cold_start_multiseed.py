"""Registered multi-start cold comparison and actual-agent response queue."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/experiment_tools'),str(ROOT / 'code/chemist_system')]
from cold_start_v13_seeds import study


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=('prepare','preflight','batch','next','status','finish','record','audit'))
    parser.add_argument('agents',nargs='*')
    parser.add_argument('--workers',type=int,default=3)
    args = parser.parse_args()
    if args.action=='prepare':
        p = study.prepare(ROOT)
        print(json.dumps(dict(protocol_sha256=p['protocol_sha256'],jobs=len(study.JOBS),new_queries=1440,starting_seeds=list(study.SEEDS))))
    elif args.action=='preflight':
        print(json.dumps(study.preflight(ROOT,args.workers)))
    elif args.action=='batch':
        study.batch(ROOT,args.workers)
    elif args.action=='audit':
        from cold_start_v13_seeds.results import audit
        print(json.dumps(audit(ROOT),ensure_ascii=False))
    elif args.action=='record':
        for agent in args.agents:
            study.record_schema(ROOT,agent)
        print('Actual schema-only collaborator message recorded.')
    else:
        if args.action=='finish':
            study.finish(ROOT,args.agents)
        status = study.status(ROOT)
        pending = study.pending(ROOT)
        compact = [dict(task=r['invocation']['task_name'],seed=r['seed'],
            input=str((ROOT / study.transport.read(Path(r['request']).parent / 'export.json')['input']).as_posix()),
            assigned=r['assigned'],output_exists=r['output_exists']) for r in pending]
        print(json.dumps(dict(complete=status['complete'],queries=status['queries'],
            **({'rows':status['rows']} if args.action=='status' else {}),pending=compact)))


if __name__=='__main__':
    main()
