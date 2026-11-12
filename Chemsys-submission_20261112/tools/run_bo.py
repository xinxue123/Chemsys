"""Start a new 0+20 BO campaign using the frozen GP and exact-tie policy."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(name,'1')
from pathlib import Path
import argparse
import csv
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]

def load_pool(short):
    from chem.benchmarks.knowledge_gp import Pool
    with (ROOT/'data/tasks.csv').open(encoding='utf-8',newline='') as f:
        row=next(r for r in csv.DictReader(f) if r['task']==short)
    path=ROOT/row['pool'];p=json.loads(path.read_text(encoding='utf-8'))
    axes=tuple(json.loads(row['axes']));kinds=tuple(json.loads(row['kinds']))
    if short in ('c1','c2','c3','c4'):
        features=tuple(tuple(float(x) for x in z[:3])+(point['catalyst'],) for z,point in zip(p['z'],p['points']))
        group,scale,task='P0',100.,f'Reizman_Suzuki_Case_{short[-1]}'
    elif short in ('snar','hplc'):
        features=tuple(tuple(map(float,z)) for z in p['z'])
        group,scale,task='P2',float(p['objective_scale']),('SnAr' if short=='snar' else 'HPLC')
    else:
        features=tuple(tuple(point[a] for a in axes) for point in p['points'])
        group,scale,task='P1',float(p.get('objective_scale',100.)),p.get('task_id',short)
    # Match the native held-out adapters: construct the public Pool, then remove
    # its input-only initialization placeholders for a 0+20 campaign.
    initial=p.get('initial_ids',p['ids'][:4])
    if isinstance(initial,dict):initial=initial['0']
    pool=Pool(task,group,tuple(p['ids']),axes,kinds,tuple(p['points']),features,tuple(initial),p.get('objective_unit','simulated yield percent'),scale,p,ROOT,path)
    pool.initial_ids=()
    return pool

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=['c1','c2','c3','c4','perera','doyle','snar','hplc','aryl','cpa','dfl','bor','chx'],required=True)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    import numpy as np
    from chem.benchmarks import knowledge_gp as native
    from hypothesis_scope_multiseed.study import SeededGP
    from kernel_contrast_qlogei.study import write,digest
    pool=load_pool(args.task)
    out=(args.out or ROOT/'output/new_bo'/f'{args.task}_s{args.seed}').resolve()
    assert not out.is_relative_to(ROOT/'data') and not out.is_relative_to(ROOT/'experiments')
    out.mkdir(parents=True,exist_ok=True)
    file=out/'state.json'
    state=json.loads(file.read_text(encoding='utf-8')) if file.exists() else dict(task=pool.task,short=args.task,seed=args.seed,initial=0,queries=20,status='RUNNING',history=[],decisions=[])
    assert state['short']==args.task and state['seed']==args.seed and state['queries']==20
    native.warm()  # Exactly once per process; all rounds reuse the same native operators.
    gp=SeededGP(pool,out/'cache',tie_seed=args.seed)
    while len(state['history'])<20:
        query=len(state['history'])+1
        context=gp.context(state['history'],query)
        selected=pool.ids[context['bo_index']]
        decision=dict(query=query,selected_id=selected,reason='uniform_prior_order' if query==1 else 'native_qLogEI',
                      observed_prefix=[h['candidate_id'] for h in state['history']],tie_seed=args.seed,
                      fit_key=context['stats']['fit_key'],acquisition=context['acquisition_meta'])
        decision['commit_sha256']=digest(decision)
        # Durably record the selection before crossing the oracle boundary.
        write(out/'rounds'/f'{query:02d}_commit.json',decision)
        observation=native.read_observation(ROOT,pool,selected)
        state['history'].append(observation)
        state['decisions'].append(dict(query=query,**observation,decision=decision,best=max(h['raw_value'] for h in state['history'])))
        state['status']='COMPLETE' if query==20 else 'RUNNING'
        write(file,state,replace=True)
        print(json.dumps(dict(query=query,candidate_id=selected,value=observation['raw_value'],best=state['decisions'][-1]['best'])),flush=True)
        gp.release()
    # Full-pool outcomes are inspected only by this post-campaign evaluator.
    with np.load(pool.source_path.with_name('oracle.npz'),allow_pickle=False) as z:
        top=float((z['raw_values'] if 'raw_values' in z else z['values']*pool.scale).max())
    best=np.maximum.accumulate([h['raw_value'] for h in state['history']])
    write(out/'metrics.json',dict(N=20,SearchAUC=float(best.mean()),SearchAUC_pct_of_max=float(100*best.mean()/top),final_best=float(best[-1]),final_best_pct_of_max=float(100*best[-1]/top)),replace=True)

if __name__=='__main__':main()
