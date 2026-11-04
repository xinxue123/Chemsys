"""User-scoped four-cell factorial, preserving the cancelled broad phase."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from contextlib import contextmanager,ExitStack
from datetime import datetime,timezone
from pathlib import Path
from unittest.mock import patch
import argparse
import json
import os
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from v13_origin_ablation import study as original,queue

VARIANT='v13_targeted_ablation_20261103'
DESIGN='docs/v13_targeted_ablation_20261103.txt'
CELLS=(('Doyle_Buchwald_Hartwig',0),('Doyle_Buchwald_Hartwig',1),('HPLC',3),('Perera_Suzuki',2))
ARMS=('full','fixed','no_refresh','neither')
_CHECKED=None


def base():return original.base()/'targeted'
def rel(p):return queue.rel(ROOT,p)
def read(p):return queue.read(p)
def write(p,x):return queue.write_once(p,x)


def jobs():
    return [dict(id=f't_{original.SHORT[t]}{s}_{a}',cell=f'{original.SHORT[t]}{s}',task=t,seed=s,arm=a)
            for t,s in CELLS for a in ARMS]


def edit_event(arm,pool,state,ordered,q):
    if arm in ('no_refresh','neither') and state['notes']:
        return dict(allowed=False,reasons=['semantic_editing_stopped_after_first_accepted_note'])
    return original.EDIT_EVENT(pool,state,ordered,q)


def prepare():
    if (base()/'protocol.json').exists():return check(True)
    p=original.check(True);base().mkdir(parents=True,exist_ok=True)
    cancelled={}
    for path in (original.base()/'runs').glob('*/state.json'):cancelled[rel(path)]=queue.file_sha(path)
    sources=dict(p['sources_sha256'])
    for name in (DESIGN,'code/experiment_tools/v13_targeted_ablation.py'):
        sources[name]=queue.file_sha(ROOT/name)
    for job in jobs():
        if job['arm']!='full' and (original.base()/'runs'/f"{job['cell']}_{job['arm']}"/'state.json').exists():
            raise ValueError('Targeted control already has broad-phase outcomes')
    protocol=dict(variant=VARIANT,registered_utc=datetime.now(timezone.utc).isoformat(),jobs=jobs(),
        source_origin_protocol_sha256=p['protocol_sha256'],initial=0,queries=20,numerical_seed=0,
        controls=12,new_control_queries=240,full_replays=4,model='gpt-6.1-terra',effort='medium',
        sources_sha256=sources,cancelled_phase_states_sha256=cancelled,
        user_scope_changes=['Cancel no_notes','Do not ablate all eight tasks'],
        design=(ROOT/DESIGN).read_text(encoding='utf-8'),targeted_new_outcomes_at_registration=0)
    protocol['protocol_sha256']=original.control.digest(protocol)
    write(base()/'protocol.json',protocol)
    write(original.base()/'broad_phase_cancelled.json',dict(status='CANCELLED_BY_USER_SCOPE_CHANGE',
        cancelled_groups=['no_notes','all non-target task/start combinations'],
        new_protocol=rel(base()/'protocol.json'),new_protocol_sha256=protocol['protocol_sha256'],
        existing_states_preserved=cancelled))
    for task in {t for t,s in CELLS}:
        src=original.base()/'cache'/task;dst=base()/'cache'/task;dst.mkdir(parents=True,exist_ok=True)
        for path in src.glob('*'):
            if path.suffix in ('.pt','.npz'):
                target=dst/path.name
                if not target.exists():shutil.copy2(path,target)
                if queue.file_sha(path)!=queue.file_sha(target):raise ValueError('Targeted exact-prefix cache copy differs')
    for name,h in sources.items():
        dst=base()/'frozen_source'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dst)
        if queue.file_sha(dst)!=h:raise ValueError('Source snapshot mismatch')
    return check(True)


def check(protected=False):
    p=read(base()/'protocol.json')
    if p['protocol_sha256']!=original.control.digest({k:v for k,v in p.items() if k!='protocol_sha256'}):
        raise ValueError('Targeted protocol changed')
    for name,h in p['sources_sha256'].items():
        if queue.file_sha(ROOT/name)!=h:raise ValueError('Targeted scientific source changed: '+name)
    old=original.check(protected)
    if old['protocol_sha256']!=p['source_origin_protocol_sha256']:raise ValueError('Original protocol changed')
    if protected:
        for name,h in p['cancelled_phase_states_sha256'].items():
            if queue.file_sha(ROOT/name)!=h:raise ValueError('Cancelled phase state changed')
    return p


@contextmanager
def redirected(job,p):
    with original.redirected(job,p),ExitStack() as stack:
        stack.enter_context(patch.object(original.native,'directory',return_value=base()))
        stack.enter_context(patch.object(original.native,'run_path',return_value=base()/'runs'/job['id']))
        stack.enter_context(patch.object(original,'edit_event',side_effect=edit_event))
        yield


def run_job(job):
    global _CHECKED
    if _CHECKED is None:_CHECKED=check()
    os.chdir(ROOT/'experiments/chemist_system')
    with redirected(job,_CHECKED):result=original.native.run_one(ROOT,job['task'],'v13')
    folder=base()/'runs'/job['id'];s=read(folder/'state.json')
    first_note=None
    for d in s['decisions']:
        c=read(folder/'rounds'/f"{d['query']:02}_commit.json")
        if c['edited_notes'] and first_note is None:first_note=d['query']
    if job['arm']=='full':
        old=read(ROOT/original.original(job['task'],job['seed'])['state_path'])
        if s['history']!=old['history'] or s['notes']!=old['notes']:raise ValueError('Targeted full replay differs')
    if job['arm'] in ('no_refresh','neither') and first_note is not None and any(q>first_note for q in s['edit_calls']):
        raise ValueError('Editing continued after accepted initial knowledge')
    write(folder/'intervention_complete.json',dict(job=job,first_accepted_note_query=first_note,
        editor_queries=s['edit_calls'],history_sha256=original.control.digest(s['history'])))
    return result


def batch(stage='controls',workers=6):
    p=check(True)
    if stage=='controls' and read(base()/'replay_complete.json')['status']!='COMPLETE':
        raise ValueError('Exact full replay required first')
    todo=[j for j in p['jobs'] if (j['arm']=='full')==(stage=='replay')];errors=[]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        fs={ex.submit(run_job,j):j for j in todo}
        for f in as_completed(fs):
            try:f.result()
            except Exception as e:
                error=dict(job=fs[f]['id'],type=type(e).__name__,message=str(e));errors.append(error)
                print('JOB_ERROR '+json.dumps(error),flush=True)
    check(True);result=dict(status='COMPLETE' if not errors else 'INCOMPLETE',stage=stage,jobs=len(todo),errors=errors)
    write(base()/(stage+'_complete.json'),result);return result


def main():
    a=argparse.ArgumentParser();a.add_argument('action',choices=('prepare','replay','controls'));a.add_argument('--workers',type=int,default=6)
    v=a.parse_args()
    if v.action=='prepare':
        p=prepare();result={k:p[k] for k in ('variant','controls','new_control_queries','full_replays','protocol_sha256')}
    else:result=batch(v.action,v.workers)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
