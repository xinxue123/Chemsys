"""python -m chem.benchmarks.suzuki_agent_score prepare|view|submit file|frozen|report|verify|release"""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
import argparse
import json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','view','submit','frozen','report','verify','release']);parser.add_argument('file',nargs='?');a=parser.parse_args()
    from . import agent_score_suzuki_study as s
    if a.action in ('prepare','view'):result=getattr(s,a.action)()
    elif a.action=='submit':result=s.submit(a.file)
    elif a.action=='frozen':result=s.frozen_control()
    else:
        from . import agent_score_suzuki_report as r
        result=getattr(r,a.action)()
    print(json.dumps(result,ensure_ascii=False,separators=(',',':')))


if __name__=='__main__':main()
