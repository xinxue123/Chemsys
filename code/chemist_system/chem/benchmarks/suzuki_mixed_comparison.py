"""python -m chem.benchmarks.suzuki_mixed_comparison qualify|prepare|regional|llinbo|report|verify|release."""
import os
for variable in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[variable]='1'
import argparse
import json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['qualify','prepare','regional','llinbo','status','report','verify','release'])
    parser.add_argument('--config',default='../config.txt')
    parser.add_argument('--provider',choices=['deepseek-flash','current_assistant'],default='deepseek-flash')
    args=parser.parse_args()
    from . import mixed_comparison_suzuki_study as study
    if args.action=='qualify':result=study.qualification()
    elif args.action=='prepare':result=study.prepare(args.config,args.provider)
    elif args.action in ('regional','llinbo'):result=study.run(args.action,args.config)
    elif args.action=='status':result=study.status()
    else:
        from . import mixed_comparison_suzuki_report as report
        result=getattr(report,args.action)()
    print(json.dumps(result,ensure_ascii=False,separators=(',',':')))


if __name__=='__main__':main()
