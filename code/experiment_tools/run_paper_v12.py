"""Single entry point for the frozen-V12 publication supplement."""
from pathlib import Path
import argparse,json,os,sys

def main():
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root/'code/chemist_system'));sys.path.insert(0,str(root/'code/experiment_tools'))
    os.chdir(root/'experiments/chemist_system')
    from paper_v12 import study
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','check','preflight','run','status']);ap.add_argument('--workers',type=int,default=9);args=ap.parse_args()
    if args.action=='run':study.batch(root,args.workers);return
    if args.action=='preflight':r=study.preflight(root,args.workers);out=dict(status=r['status'],pairs=len(r['rows']),copied=r['GP_cache_files_copied'])
    elif args.action=='status':out=study.status(root)
    else:
        p=study.prepare(root) if args.action=='prepare' else study.check(root,True)
        out={k:p[k] for k in ('variant','protocol_sha256','maximum_new_trajectories','maximum_new_queries','maximum_logical_LLM_calls','model','reasoning_effort')}
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
