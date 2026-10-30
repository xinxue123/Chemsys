"""Standalone rerun; does not mutate the sealed Perera/Doyle entrypoint."""
import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_name]='1'
import argparse
import json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['qualify','prepare','preflight','run','status','validate','report','release'])
    parser.add_argument('--config',default='../config.txt');args=parser.parse_args()
    from chem.benchmarks import stable_suzuki_study as study
    if args.action in ('qualify','status'):value=getattr(study,args.action)()
    elif args.action in ('prepare','preflight','run'):value=getattr(study,args.action)(args.config)
    else:
        from chem.benchmarks import stable_suzuki_report as report
        value=getattr(report,args.action)()
    print(json.dumps(value,ensure_ascii=False,default=str),flush=True)


if __name__=='__main__':main()
