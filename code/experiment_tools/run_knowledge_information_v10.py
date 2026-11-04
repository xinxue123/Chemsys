"""V10 standalone entry; all sealed historical control files stay read-only."""
from pathlib import Path
import argparse,json,os,sys


def main():
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root/'code/chemist_system'));sys.path.insert(0,str(root/'code/experiment_tools'))
    os.chdir(root/'experiments/chemist_system')
    from knowledge_information_v10 import study
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['preflight','prepare','check','run','status','report'])
    ap.add_argument('--workers',type=int,default=8);args=ap.parse_args()
    if args.action=='run':study.batch(root,args.workers);return
    if args.action=='preflight':value=study.preflight(root,args.workers)
    elif args.action=='status':value=[study.summary(study.old.read(p)) for p in sorted((study.directory(root)/'runs').glob('*/state.json'))]
    elif args.action=='report':
        from knowledge_information_v10.report import build
        value=build(root)
    else:
        p=study.prepare(root) if args.action=='prepare' else study.check(root)
        value={k:p[k] for k in ('variant','protocol_sha256','tasks','readonly_controls','initial','queries','model','reasoning_effort')}
    print(json.dumps(value,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
