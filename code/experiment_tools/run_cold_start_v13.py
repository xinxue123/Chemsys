"""Command-line cold-start runner and actual-agent receipt queue."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/experiment_tools'),str(ROOT / 'code/chemist_system')]
from cold_start_v13 import study, transport


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=('prepare','preflight','batch','status','next','finish','audit'))
    parser.add_argument('agents',nargs='*')
    parser.add_argument('--workers',type=int,default=3)
    args = parser.parse_args()
    if args.action=='prepare':
        value = study.prepare(ROOT)
        print(json.dumps(dict(protocol_sha256=value['protocol_sha256'],initial=0,queries=20,jobs=len(study.JOBS))))
    elif args.action=='preflight':
        from cold_start_v13.validation import preflight
        print(json.dumps(preflight(ROOT,args.workers)))
    elif args.action=='batch':
        study.batch(ROOT,args.workers)
    elif args.action=='audit':
        from cold_start_v13.validation import audit
        print(json.dumps(audit(ROOT),ensure_ascii=False))
    else:
        if args.action=='finish':
            pending = {'/root/'+r['invocation']['task_name']:r for r in transport.pending(ROOT)}
            for agent in args.agents:
                row = pending[agent]
                transport.record_spawn(ROOT,row['request'],dict(task_name=agent))
                transport.seal(ROOT,row['request'])
        status = study.status(ROOT)
        pending = transport.pending(ROOT)
        compact = [dict(task=r['invocation']['task_name'],
            input=str((ROOT / transport.read(Path(r['request']).parent/'export.json')['input']).as_posix()),
            assigned=r['assigned'],output_exists=r['output_exists']) for r in pending]
        print(json.dumps(dict(complete=status['complete'],queries=status['queries'],
            **({'rows':status['rows']} if args.action=='status' else {}),pending=compact),ensure_ascii=False))


if __name__=='__main__':
    main()
