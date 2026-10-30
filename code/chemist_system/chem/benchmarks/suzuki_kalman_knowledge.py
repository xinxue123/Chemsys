"""Independent Kalman knowledge pilot; all earlier studies are read-only."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
import argparse
import json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['qualify','prepare','run','frozen','report','verify','release']);parser.add_argument('--config',default='../config.txt');args=parser.parse_args()
    from . import kalman_knowledge_suzuki_study as s
    if args.action=='qualify':result=s.qualification()
    elif args.action=='prepare':result=s.prepare(args.config)
    elif args.action in ('run','frozen'):result=s.run(args.config,args.action=='frozen')
    else:
        from . import kalman_knowledge_suzuki_report as report
        result=getattr(report,args.action)(args.config) if args.action=='verify' else getattr(report,args.action)()
    print(json.dumps(result,ensure_ascii=False,separators=(',',':')))


if __name__=='__main__':main()
