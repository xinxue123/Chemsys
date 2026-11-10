"""Small entry point for the paper-specific research project. Defaults to help."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
if hasattr(sys.stdout,'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('verify',help='Check files, 13 pools and saved results without model calls or fits.')
    sub.add_parser('datasets',help='List the frozen candidate pools.')
    sub.add_parser('results',help='Print the model-specific held-out comparison table.')
    s=sub.add_parser('inspect',help='Inspect one archived development controller decision.')
    s.add_argument('--task',choices=['c1','c2','c3','c4','perera','doyle','snar','hplc'],required=True)
    s.add_argument('--seed',type=int,choices=range(4),default=0)
    s.add_argument('--query',type=int,choices=range(1,21),required=True)
    s=sub.add_parser('rebuild-results',help='Rebuild saved-evidence analyses; never start an experiment.')
    s.add_argument('part',choices=['development','heldout','screen','first_points','knowledge','all'],default='all',nargs='?')
    a=p.parse_args()
    if a.command=='verify':
        sys.path.insert(0,str(ROOT/'tools'));from verify_release import main as verify;verify()
    elif a.command=='datasets':
        with (ROOT/'data/tasks.csv').open(encoding='utf-8',newline='') as f:
            for r in csv.DictReader(f):print(f"{r['task']:7s} {int(r['candidates']):5d} candidates  {r['pool']}")
    elif a.command=='results':
        print((ROOT/'results/deepseek_table4_20261110/table4_model_extension.csv').read_text(encoding='utf-8'))
    elif a.command=='inspect':
        filename='run__hypothesis_scope_loop_20261105.zip' if a.seed==0 else 'run__hypothesis_scope_multiseed_20261105.zip'
        prefix=f'experiments/runs/hypothesis_scope_loop_20261105/runs/{a.task}' if a.seed==0 else f'experiments/runs/hypothesis_scope_multiseed_20261105/runs/s{a.seed}/{a.task}_hyp'
        if not (ROOT/'archives'/filename).exists():
            raise SystemExit('Decision inspection requires a local archive. See LOCAL_DATA_REQUIREMENTS.md.')
        with zipfile.ZipFile(ROOT/'archives'/filename) as z:record=json.loads(z.read(f'{prefix}/rounds/{a.query:02d}_commit.json').decode('utf-8-sig'))
        print(json.dumps(record,ensure_ascii=False,indent=2))
    else:
        subprocess.run([sys.executable,str(ROOT/'tools/rebuild_results.py'),a.part],cwd=ROOT,check=True)

if __name__=='__main__':main()
