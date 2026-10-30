"""Run controlled acquisition study from chemist_system with python -m."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
import argparse
import json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['qualify','prepare','develop','freeze','main','report','audit','verify','release'])
    action=parser.parse_args().action
    from . import acquisition_suzuki_study as s
    if action in ('qualify','prepare'):result=getattr(s,action)()
    elif action in ('develop','main'):result=s.run_stage('development' if action=='develop' else 'main')
    elif action=='freeze':result=s.select_kappa()
    else:
        from . import acquisition_suzuki_report as r
        result=getattr(r,action)()
    print(json.dumps(result,ensure_ascii=False,separators=(',',':')))


if __name__=='__main__':main()
