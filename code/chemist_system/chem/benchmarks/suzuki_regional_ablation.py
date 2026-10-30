import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse
import json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['qualify','prepare','static','event','status','report','verify','release'])
    parser.add_argument('--config',default='../config.txt');args=parser.parse_args()
    from . import regional_ablation_suzuki_study as s
    if args.action=='qualify':result=s.qualify()
    elif args.action=='prepare':result=s.prepare(args.config)
    elif args.action in ('static','event'):result=s.run(args.action=='event',args.config)
    elif args.action=='status':result=s.status()
    else:
        from . import regional_ablation_suzuki_report as r
        result=getattr(r,args.action)()
    print(json.dumps(result,ensure_ascii=False,separators=(',',':')))


if __name__=='__main__':main()
