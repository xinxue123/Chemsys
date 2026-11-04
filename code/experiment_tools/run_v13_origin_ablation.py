from pathlib import Path
import argparse
import json
import os
import sys

os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=('prepare','preflight','replay','controls','probes','pending','status','assign','seal'))
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--limit',type=int,default=9)
    a=p.parse_args()
    from v13_origin_ablation import queue
    if a.action=='pending':x=queue.pending(ROOT)[:a.limit]
    elif a.action=='status':x=queue.status(ROOT)
    elif a.action in ('assign','seal'):
        payload=json.loads(sys.stdin.read())
        if a.action=='assign':
            for r in payload:queue.assign(ROOT,Path(r['request']),r['spawn_result'])
        else:
            for path in payload:queue.seal(ROOT,Path(path))
        x=dict(status='OK',count=len(payload))
    else:
        from v13_origin_ablation import study
        if a.action=='prepare':
            v=study.prepare();x={k:v[k] for k in ('variant','full_replay_trajectories','new_control_trajectories',
                'new_control_queries','targeted_one_step_probes','seeded_responses','protocol_sha256')}
        elif a.action=='preflight':x=study.preflight()
        elif a.action=='probes':x=study.probes(a.workers)
        else:x=study.batch('replay' if a.action=='replay' else 'controls',a.workers)
    print(json.dumps(x,ensure_ascii=True),flush=True)


if __name__=='__main__':main()
