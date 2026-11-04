"""Standalone V9 entrypoint, preserving all V8 production source hashes."""
from pathlib import Path
import argparse,json,os,sys


def main():
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root/'code/chemist_system'));sys.path.insert(0,str(root/'code/experiment_tools'))
    os.chdir(root/'experiments/chemist_system')
    from knowledge_rank_v9 import study
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','check','run','status','report'])
    ap.add_argument('--arm',choices=study.ARMS,default='kf_rank');ap.add_argument('--workers',type=int,default=8)
    args=ap.parse_args()
    if args.action=='run':study.batch(root,args.arm,args.workers);return
    if args.action=='report':
        from knowledge_rank_v9.report import build
        print(json.dumps(build(root),ensure_ascii=False));return
    if args.action=='status':
        value=[study.summary(study.old.read(p)) for p in sorted((study.directory(root)/'runs').glob('*/state.json'))]
    else:
        p=study.prepare(root) if args.action=='prepare' else study.check(root)
        value={key:p[key] for key in ('variant','protocol_sha256','tasks','arms','initial','queries','model','reasoning_effort')}
    print(json.dumps(value,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
