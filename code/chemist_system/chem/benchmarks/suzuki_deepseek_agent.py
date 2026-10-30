"""Independent DeepSeek API scorecard study; config contains credentials only."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
import argparse
import json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','frozen','report','verify','release']);parser.add_argument('--config',default='../config.txt');args=parser.parse_args()
    from . import deepseek_agent_suzuki_study as s
    if args.action in ('prepare','run'):result=getattr(s,args.action)(args.config)
    elif args.action=='frozen':result=s.frozen_control()
    else:
        from . import deepseek_agent_suzuki_report as r
        result=getattr(r,args.action)()
    print(json.dumps(result,ensure_ascii=False,separators=(',',':')))


if __name__=='__main__':main()
