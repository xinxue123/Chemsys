"""Wait for every registered trajectory, then report, audit, and seal once."""
from datetime import datetime,timezone
from pathlib import Path
import json
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'experiments/runs/grounded_baselines_20261105'
OPS=ROOT/'experiments/operations/grounded_baselines_20261105'


def write(value):
    OPS.mkdir(parents=True,exist_ok=True)
    path=OPS/'finalization_status.json';temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');temp.replace(path)


def main():
    if (BASE/'completion_receipt.json').exists():
        print('Already sealed; outputs retained.',flush=True);return
    p=json.loads((BASE/'protocol.json').read_text(encoding='utf-8'))
    short=dict(zip(p['tasks'],('c1','c2','c3','c4','perera','doyle','snar','hplc')))
    folders=[BASE/'runs'/f's{seed}'/(short[task]+'_'+arm) for task,seed,arm in p['jobs']]
    last=None
    while True:
        errors=list(BASE.glob('runs/*/*/execution_error.json'))
        if errors:
            value=dict(status='EXECUTION_ERROR_REQUIRES_INSPECTION',errors=[x.relative_to(ROOT).as_posix() for x in errors])
            write(value);raise RuntimeError(json.dumps(value))
        done=sum((folder/'completion.json').exists() for folder in folders)
        if done!=last:
            value=dict(status='WAITING_FOR_COMPLETE_POPULATION',completed=done,planned=64,updated_UTC=datetime.now(timezone.utc).isoformat())
            write(value);print(json.dumps(value),flush=True);last=done
        if done==64:break
        time.sleep(20)
    for tool,args in [('report_grounded_baselines.py',[]),('audit_grounded_baselines.py',['--final']),('seal_grounded_baselines.py',[])]:
        write(dict(status='FINALIZING',tool=tool))
        subprocess.run([sys.executable,str(ROOT/'code/tools'/tool),*args],cwd=str(ROOT),check=True)
    write(dict(status='SEALED',completed=64,planned=64,updated_UTC=datetime.now(timezone.utc).isoformat()))
    print('FINALIZATION_COMPLETE',flush=True)


if __name__=='__main__':main()
