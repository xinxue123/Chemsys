"""Reduced independent-API benchmark: four cases, seed0, unchanged query budget."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
import argparse
import json


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','frozen','report','verify','release']);p.add_argument('--config',default='../config.txt');a=p.parse_args()
    from . import deepseek_small_suzuki_study as s
    if a.action in ('prepare','run'):r=getattr(s,a.action)(a.config)
    elif a.action=='frozen':r=s.frozen()
    else:
        from . import deepseek_small_suzuki_report as report
        r=getattr(report,a.action)()
    print(json.dumps(r,ensure_ascii=False,separators=(',',':')))


if __name__=='__main__':main()
