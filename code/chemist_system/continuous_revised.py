"""Single entry point for the two revised-agent SnAr/HPLC trajectories."""
import argparse,json,os
from pathlib import Path

def main():
    os.chdir(Path(__file__).resolve().parent)
    from chem.benchmarks import revised_continuous_study as s
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['qualify','prepare','preflight','run','status','audit','report','inspect'])
    p.add_argument('--config',default='../config.txt');p.add_argument('--task',choices=s.TASKS,default='SnAr');p.add_argument('--round',type=int,default=1)
    a=p.parse_args()
    if a.action in ('audit','report','inspect'):
        from chem.benchmarks import revised_continuous_report as report
        result=report.inspect(a.task,a.round) if a.action=='inspect' else getattr(report,a.action)()
    elif a.action in ('prepare','preflight','run'):result=getattr(s,a.action)(a.config)
    else:result=getattr(s,a.action)()
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
