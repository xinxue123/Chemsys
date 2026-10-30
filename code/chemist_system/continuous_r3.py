"""Single entry point for the two protected-prefix SnAr/HPLC R3 trajectories."""
import argparse
import json
import os
from pathlib import Path


def main():
    os.chdir(Path(__file__).resolve().parent)
    from chem.benchmarks import continuous_r3_study as study
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['qualify','prepare','preflight','run','status','audit','report','inspect'])
    parser.add_argument('--config',default='../config.txt');parser.add_argument('--task',choices=study.TASKS,default='SnAr')
    parser.add_argument('--round',type=int,default=1);args=parser.parse_args()
    if args.action in ('audit','report','inspect'):
        from chem.benchmarks import continuous_r3_report as report
        result=report.inspect(args.task,args.round) if args.action=='inspect' else getattr(report,args.action)()
    elif args.action in ('prepare','preflight','run'):result=getattr(study,args.action)(args.config)
    else:result=getattr(study,args.action)()
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
