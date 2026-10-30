"""python -m chem.benchmarks.suzuki_llinbo_guarded prepare|run|view|submit|report|verify."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
import argparse
import json

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','status','view','submit','report','verify'])
    p.add_argument('--initial',action='store_true');p.add_argument('--answers')
    a=p.parse_args()
    from . import llinbo_guarded_study as study
    if a.action=='prepare':result=study.prepare()
    elif a.action=='run':result=study.run()
    elif a.action=='status':result=study.status()
    elif a.action=='view':
        for row in study.view(a.initial):print(json.dumps(row,separators=(',',':')))
        return
    elif a.action=='submit':result=study.submit(a.answers)
    else:
        from . import llinbo_guarded_report as report
        result=report.report() if a.action=='report' else report.verify()
    print(json.dumps(result,ensure_ascii=False,separators=(',',':')))

if __name__=='__main__':main()
