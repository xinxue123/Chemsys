"""Single entry point for the protected-prefix R5 SnAr/HPLC study."""
import argparse
import json
import os
from pathlib import Path


def main():
    os.chdir(Path(__file__).resolve().parent)
    from chem.benchmarks import continuous_r5_study as study
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['qualify','prepare','preflight','run','status','audit','report','inspect'])
    p.add_argument('--config',default='../config.txt');p.add_argument('--task',choices=study.TASKS,default='SnAr');p.add_argument('--round',type=int,default=1)
    args=p.parse_args()
    if args.action in ('audit','report','inspect'):
        from chem.benchmarks import continuous_r5_report as report
        result=report.inspect(args.task,args.round) if args.action=='inspect' else getattr(report,args.action)()
    elif args.action in ('prepare','preflight','run'):result=getattr(study,args.action)(args.config)
    else:result=getattr(study,args.action)()
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
