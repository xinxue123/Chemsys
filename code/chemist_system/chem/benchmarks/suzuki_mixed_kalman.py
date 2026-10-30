"""Mixed categorical GP replacement pilot; prior experiments remain immutable."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
import argparse
import json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['qualify','prepare','run','controls','report','verify','release']);parser.add_argument('--config',default='../config.txt');args=parser.parse_args()
    from . import mixed_kalman_suzuki_study as s
    if args.action=='qualify':r=s.qualification()
    elif args.action=='prepare':r=s.prepare(args.config)
    elif args.action in ('run','controls'):r=s.run(args.config,args.action=='controls')
    else:
        from . import mixed_kalman_suzuki_report as report
        r=getattr(report,args.action)(args.config) if args.action=='verify' else getattr(report,args.action)()
    print(json.dumps(r,ensure_ascii=False,separators=(',',':')))


if __name__=='__main__':main()
