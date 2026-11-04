"""V14 register/preflight/run/replay and actual fresh-agent queue CLI."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/experiment_tools'),str(ROOT/'code/chemist_system')]
from reason_cards_v14 import study,transport


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('prepare','preflight','batch','next','status','finish','audit'))
    parser.add_argument('agents',nargs='*');parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if args.action=='prepare':
        p=study.prepare(ROOT);result=dict(protocol_sha256=p['protocol_sha256'],jobs=p['jobs'],new_queries=120)
    elif args.action=='preflight':result=study.preflight(ROOT,args.workers)
    elif args.action=='batch':study.batch(ROOT,args.workers);return
    elif args.action=='audit':
        from reason_cards_v14.results import audit
        result=audit(ROOT)
    else:
        if args.action=='finish':transport.finish(ROOT,args.agents)
        result=study.status(ROOT)
        if args.action!='status':result.pop('rows')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
