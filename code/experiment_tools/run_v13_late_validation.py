from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from v13_late_validation import study,queue

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','preflight','batch','pending','assign','seal'))
    p.add_argument('--request',type=Path);p.add_argument('--spawn-result',type=Path);p.add_argument('--workers',type=int,default=6)
    a=p.parse_args()
    if a.action=='prepare':x=study.prepare();print(json.dumps({k:x[k] for k in ('variant','total_trajectories','new_paid_queries','protocol_sha256')}))
    elif a.action=='preflight':print(json.dumps(study.preflight()))
    elif a.action=='batch':print(json.dumps(study.batch(a.workers)))
    elif a.action=='pending':print(json.dumps(queue.pending(ROOT)))
    elif a.action=='assign':queue.assign(ROOT,a.request,json.loads(a.spawn_result.read_text(encoding='utf-8-sig')));print('ASSIGNED')
    else:queue.seal(ROOT,a.request);print('SEALED')
