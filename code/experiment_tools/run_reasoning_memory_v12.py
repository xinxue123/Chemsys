"""V12 frozen targeted experiment; completed trajectories are never rerun."""
from pathlib import Path
import argparse,json,os,sys


def main():
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root/'code/chemist_system'));sys.path.insert(0,str(root/'code/experiment_tools'))
    os.chdir(root/'experiments/chemist_system')
    from reasoning_memory_v12 import study
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['preflight','prepare','check','run','status'])
    ap.add_argument('--workers',type=int,default=3);args=ap.parse_args()
    if args.action=='run':study.batch(root,args.workers);return
    if args.action=='preflight':value=study.preflight(root,args.workers)
    elif args.action=='status':value=[study.summary(study.old.read(p)) for p in sorted((study.directory(root)/'runs').glob('*/state.json'))]
    else:
        p=study.prepare(root) if args.action=='prepare' else study.check(root)
        value={k:p[k] for k in ('variant','protocol_sha256','runs','readonly_controls','initial','queries','model','reasoning_effort')}
    print(json.dumps(value,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
