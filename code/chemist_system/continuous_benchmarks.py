"""Run / resume / inspect the fixed continuous-domain comparison."""
import argparse
import json
import os
from pathlib import Path

def main():
    os.chdir(Path(__file__).resolve().parent)
    from chem.benchmarks import continuous_benchmark_study as s
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['qualify','prepare','preflight','initialize','run','status','audit','report','inspect'])
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--config',default='../config.txt')
    parser.add_argument('--method',choices=s.e.METHODS)
    parser.add_argument('--task',choices=s.TASKS,default='SnAr')
    parser.add_argument('--round',type=int,default=1)
    a=parser.parse_args()
    if a.action in ('qualify','prepare','preflight','initialize','run','status'):
        if a.action=='run':out=s.run(a.config,a.workers,[a.method] if a.method else None)
        elif a.action in ('prepare','preflight','initialize'):out=getattr(s,a.action)(a.config)
        else:out=getattr(s,a.action)()
    else:
        from chem.benchmarks import continuous_benchmark_report as report
        if a.action=='inspect':out=report.inspect(a.task,a.method or 'New_Agent',a.round)
        else:out=getattr(report,a.action)()
    print(json.dumps(out,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
