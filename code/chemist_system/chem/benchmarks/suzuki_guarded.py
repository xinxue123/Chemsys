"""Matched B4/LGBO study: python -m chem.benchmarks.suzuki_guarded."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
import argparse
import json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','develop','freeze','test','report','verify'])
    args=parser.parse_args()
    from . import guarded_suzuki_study as study
    if args.action=='prepare':value=study.prepare()
    elif args.action=='develop':value=study.run_stage('development')
    elif args.action=='freeze':value=study.select_budget()
    elif args.action=='test':value=study.run_stage('test')
    else:
        from . import guarded_suzuki_report as report
        value=report.report() if args.action=='report' else report.verify()
    print(json.dumps(value,ensure_ascii=False,separators=(',',':')))


if __name__=='__main__':main()
