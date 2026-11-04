"""Registered coverage, matched controls and two removal ablations for V12."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager, ExitStack
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
import json, shutil, subprocess
import numpy as np
from filelock import FileLock
from v12_initialization_seeds import study as seeds

v12=seeds.v12; v9=seeds.v9; old=seeds.old
digest=v12.digest; write=v12.write
NATIVE_LOAD=seeds.NATIVE_LOAD_TASK
NATIVE_KF=v12.engine.kf.update
TASKS=tuple(v12.gpmod.TASKS)
FOCUS=tuple(v12.TASKS)
OTHER=tuple(t for t in TASKS if t not in FOCUS)
DESIGN='docs/paper_v12_supplement_protocol_20261101.txt'
SHORT={t:(f'c{t[-1]}' if 'Case_' in t else {'Perera_Suzuki':'perera','Doyle_Buchwald_Hartwig':'doyle','SnAr':'snar','HPLC':'hplc'}[t]) for t in TASKS}
JOBS=tuple([(t,0,'memory') for t in OTHER]+[(t,s,a) for s in (1,2,3) for t in TASKS for a in (('bo','memory','reasoning_bo') if t in OTHER else ('reasoning_bo',))]+[(t,s,a) for s in (0,3) for t in FOCUS for a in ('no_kf','no_refresh')])

def directory(root):return Path(root)/'experiments/runs/paper_v12'
def native_arm(arm):return arm if arm in ('bo','reasoning_bo') else 'memory'
def base(root,task,seed,arm):
    if (task,seed,arm) not in JOBS:raise ValueError('Unregistered job')
    return directory(root)/f's{seed}'/SHORT[task]/arm
def run_path(root,task,seed,arm):return base(root,task,seed,arm)/'runs'/f'{task}_{native_arm(arm)}'

def initialization(pool,seed):
    if seed==0:return pool.initial_ids
    order=sorted(range(len(pool.ids)),key=lambda j:pool.features[j])
    if len(set(pool.features))!=len(pool.ids):raise ValueError('Ambiguous physical input ordering')
    ix=np.random.default_rng(np.random.SeedSequence([seed,0])).choice(len(order),4,replace=False)
    return tuple(pool.ids[order[int(j)]] for j in ix)

def sources(root):
    root=Path(root);files=seeds.sources(root)
    paths=list((root/'code/experiment_tools/paper_v12').glob('*.py'))+[root/'code/experiment_tools/run_paper_v12.py']
    files.update({p.relative_to(root).as_posix():old.sha(p) for p in paths});return files

def prepare(root):
    root=Path(root).resolve();d=directory(root)
    if (d/'protocol.json').exists():return check(root,True)
    if any(d.glob('s[0-9]*')):raise ValueError('Register before new activity')
    prior=seeds.check(root,verify_controls=True);frozen=v12.check(root);initial=[]
    for task in TASKS:
        pool=NATIVE_LOAD(root,task,seed=0)
        for seed in (0,1,2,3):
            ids=initialization(pool,seed)
            if seed in (1,2) and ids!=NATIVE_LOAD(root,task,seed=seed).initial_ids:raise ValueError('Archived seed inputs differ')
            initial.append(dict(task=task,initialization_seed=seed,initial_ids=list(ids),input_pool=pool.source_path.relative_to(root).as_posix(),input_sha256=old.sha(pool.source_path),candidate_count=len(pool.ids),axes=list(pool.axes),kinds=list(pool.kinds),unit=pool.unit))
    protected=deepcopy(prior['protected_controls_sha256'])
    paths=[p for p in seeds.directory(root).rglob('*') if p.is_file() and p.suffix!='.lock']
    paths += list((root/'experiments/reports/knowledge_loop_20261101').glob('v12_seeds_*'))
    paths += [root/'docs/v12_initialization_seeds_results_20261101.txt',root/'docs/v12_innovation_summary_20261101.txt']
    protected.update({p.relative_to(root).as_posix():old.sha(p) for p in paths if p.is_file()})
    p=dict(variant='paper_v12',created_utc=datetime.now(timezone.utc).isoformat(),jobs=[list(j) for j in JOBS],
        tasks=list(TASKS),runs=[[t,a] for t in TASKS for a in ('bo','reasoning_bo','memory')],arms=['bo','reasoning_bo','memory'],
        initializations=initial,inputs=deepcopy(frozen['inputs']),initial=4,queries=16,seed=0,trial_seed=0,tie_seed=0,numerical_seed=0,menu_seed=0,
        model=frozen['model'],reasoning_effort=frozen['reasoning_effort'],parameters=deepcopy(frozen['parameters']),prompt_sha256=deepcopy(frozen['prompt_sha256']),
        frozen_V12_protocol_sha256=frozen['protocol_sha256'],frozen_seed_protocol_sha256=prior['protocol_sha256'],
        source_sha256=sources(root),protected_controls_sha256=protected,design_path=DESIGN,design_sha256=old.sha(root/DESIGN),
        code_milestone=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        maximum_new_trajectories=71,maximum_new_queries=1136,maximum_logical_LLM_calls=1006,
        initial_outcomes_accessed=False,LLM_calls_at_registration=0,GP_fits_at_registration=0,no_performance_rerolls=True,
        execution='unmodified frozen loops; process-local input/I-O adapter; two explicitly registered module-removal wrappers')
    assert len(JOBS)==71 and len(set(JOBS))==71
    p['protocol_sha256']=digest(p);write(d/'protocol.json',p)
    snapshot=root/'runtime/knowledge_loop/snapshots/paper_v12'
    for name,sha in p['source_sha256'].items():
        dest=snapshot/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,dest)
        if old.sha(dest)!=sha:raise ValueError('Snapshot mismatch')
    dest=snapshot/DESIGN;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/DESIGN,dest)
    shutil.copy2(d/'protocol.json',snapshot/'protocol.json');return check(root,True)

def check(root,verify_controls=False):
    root=Path(root);p=old.read(directory(root)/'protocol.json');f=v12.check(root);seeds.check(root)
    if p['protocol_sha256']!=digest({k:v for k,v in p.items() if k!='protocol_sha256'}):raise ValueError('Protocol drift')
    if p['source_sha256']!=sources(root) or p['design_sha256']!=old.sha(root/DESIGN):raise ValueError('Frozen supplement source drift')
    if p['jobs']!=[list(j) for j in JOBS] or p['frozen_V12_protocol_sha256']!=f['protocol_sha256']:raise ValueError('Frozen scope drift')
    if verify_controls:
        for name,sha in p['protected_controls_sha256'].items():
            if old.sha(root/name)!=sha:raise ValueError('Historical artifact drift: '+name)
    return p

def pool_for(root,task,seed,p):
    pool=NATIVE_LOAD(root,task,seed=0);row=next(x for x in p['initializations'] if x['task']==task and x['initialization_seed']==seed)
    ids=initialization(pool,seed)
    if list(ids)!=row['initial_ids'] or old.sha(pool.source_path)!=row['input_sha256']:raise ValueError('Input registration changed')
    return replace(pool,initial_ids=ids)

def no_kf_update(state,prob,g,success,t,*,prior_transport=False):
    # Preserve direct measured forecast diagnostics, not the adaptive filter.
    _,event=NATIVE_KF(state,prob,g,success,t,prior_transport=prior_transport)
    after=deepcopy(state)
    after.update(updates=state['updates']+1,local_updates=state['local_updates']+int(not prior_transport))
    event.update(after=deepcopy(after),reason='fixed_calibration_ablation')
    return after,event

def no_refresh_event(pool,state,ordered,t):
    return dict(allowed=not state['edit_calls'],reasons=['initial_knowledge'] if not state['edit_calls'] else ['refresh_disabled_ablation'])

@contextmanager
def redirected(root,task,seed,arm,p,redirect_gp=True):
    root=Path(root);b=base(root,task,seed,arm);pool=pool_for(root,task,seed,p);native=v12.gpmod.GPAdapter
    def loader(r,name,*,seed=0):
        if Path(r).resolve()!=root.resolve() or name!=task or seed!=0:raise ValueError('Unexpected loader')
        return pool
    def adapter(obj,path,seed=0):
        if obj is not pool or seed!=0:raise ValueError('Unexpected GP input')
        return native(obj,directory(root)/'cache'/task,seed=0)
    with ExitStack() as stack:
        for module in (v12,v9):
            stack.enter_context(patch.object(module,'directory',return_value=b))
            stack.enter_context(patch.object(module,'check',return_value=deepcopy(p)))
        stack.enter_context(patch.object(v12.gpmod,'load_task',side_effect=loader))
        if redirect_gp:stack.enter_context(patch.object(v12.gpmod,'GPAdapter',side_effect=adapter))
        if arm=='no_kf':stack.enter_context(patch.object(v12.engine.kf,'update',side_effect=no_kf_update))
        if arm=='no_refresh':stack.enter_context(patch.object(v12.engine,'edit_event',side_effect=no_refresh_event))
        yield pool

def copy_cache(root):
    root=Path(root);count=0
    for source_base in (seeds.directory(root)/'cache',v12.directory(root)/'cache',v12.v11.directory(root)/'cache',v9.directory(root)/'cache'):
        for source in source_base.rglob('*'):
            if not source.is_file() or source.suffix not in ('.json','.npz','.pt'):continue
            target=directory(root)/'cache'/source.relative_to(source_base)
            if not target.exists():target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);count+=1
            elif old.sha(source)!=old.sha(target):raise ValueError('Native prefix cache conflict')
    return count

def preflight_one(root,task,seed):
    root=Path(root);p=check(root);pool=pool_for(root,task,seed,p);folder=directory(root)/'preflight'/f's{seed}'/SHORT[task]
    old.initial_commit(folder,pool,p,'memory','agent_initial_terra',create=True)
    hist=[v12.gpmod.read_observation(root,pool,k) for k in pool.initial_ids]
    gp=v12.gpmod.GPAdapter(pool,directory(root)/'cache'/task,seed=0);c=gp.context(hist,1)
    pre=dict(history=hist,decisions=[],edit_calls=[],hypotheses=[],insights=[],arm='memory',category_menu_insertions=0)
    a,b,display,_=v12.engine.menu(pool,pre,c,1,'memory');bo,bd=v9.engine.candidates(pool,c,1)
    np.testing.assert_array_equal(a,bo);np.testing.assert_array_equal(display,bd)
    paths=[]
    if seed==0:paths=[v9.directory(root)/'runs'/f'{task}_bo/rounds/01_GP.npz']
    elif task in FOCUS:paths=[seeds.run_path(root,task,seed,'bo')/'rounds/01_GP.npz']
    for path in paths:
        with np.load(path,allow_pickle=False) as z:
            for k in ('mu_y','sd_y','sd_predictive_y'):np.testing.assert_array_equal(c['stats'][k],z[k])
            np.testing.assert_array_equal(c['base_logei'],z['base_logei'])
    np.savez_compressed(folder/'01_GP.npz',mu_y=c['stats']['mu_y'],sd_y=c['stats']['sd_y'],sd_predictive_y=c['stats']['sd_predictive_y'],base_logei=c['base_logei'],available=c['available'],tie_priority=c['tie_priority'])
    r=dict(task=task,initialization_seed=seed,status='PASS',initial_history=hist,initial_ids=list(pool.initial_ids),GP_sha256=v12.engine.previous.gp_hash(c),historical_first_GP_comparisons=len(paths),counters=gp.counters)
    gp.release();write(folder/'receipt.json',r);return r

def preflight(root,workers=9):
    p=check(root,True);n=copy_cache(root);pairs=sorted({(t,s) for t,s,a in JOBS})
    with ProcessPoolExecutor(max_workers=min(workers,9)) as ex:rows=list(ex.map(preflight_one,[str(root)]*len(pairs),[x[0] for x in pairs],[x[1] for x in pairs]))
    r=dict(status='PASS',protocol_sha256=p['protocol_sha256'],GP_cache_files_copied=n,rows=rows,new_queries=0,LLM_calls=0)
    write(Path(root)/'experiments/reports/paper_v12/preflight.json',r);return r

def run_one(root,task,seed,arm):
    root=Path(root).resolve();p=check(root);b=base(root,task,seed,arm)
    pf=old.read(directory(root)/'preflight'/f's{seed}'/SHORT[task]/'receipt.json')
    if pf['status']!='PASS':raise ValueError('Preflight required')
    b.mkdir(parents=True,exist_ok=True)
    with FileLock(str(b/'run.lock'),timeout=900):
        launch=b/'launch.json'
        if not launch.exists():
            if (b/'runs').exists() or (b/'requests').exists():raise ValueError('Fresh namespace required')
            write(launch,dict(task=task,initialization_seed=seed,arm=arm,native_arm=native_arm(arm),protocol_sha256=p['protocol_sha256'],started_utc=datetime.now(timezone.utc).isoformat(),LLM_cache_initially_empty=True,response_files_copied=0))
        elif old.read(launch)['protocol_sha256']!=p['protocol_sha256']:raise ValueError('Launch drift')
        with redirected(root,task,seed,arm,p):
            runner=v9.run_one if arm in ('bo','reasoning_bo') else v12.run_one
            result=dict(runner(root,task,native_arm(arm)),initialization_seed=seed,study_arm=arm)
        check(root);write(b/'completion.json',dict(result=result,protocol_sha256=p['protocol_sha256'],finished_utc=datetime.now(timezone.utc).isoformat()))
        print('FINISH '+json.dumps(result),flush=True);return result

def batch(root,workers=9):
    check(root,True);errors=[]
    with ProcessPoolExecutor(max_workers=min(workers,9)) as ex:
        fs={ex.submit(run_one,str(root),*job):job for job in JOBS}
        for f in as_completed(fs):
            try:f.result()
            except Exception as e:
                errors.append(dict(job=fs[f],kind=type(e).__name__,message=str(e)));print('JOB_ERROR '+json.dumps(errors[-1]),flush=True)
    write(Path(root)/'experiments/reports/paper_v12/batch_receipt.json',dict(errors=errors,status='INCOMPLETE' if errors else 'COMPLETE',jobs=71))
    if errors:raise RuntimeError('Preserve failed prefixes and resume: '+json.dumps(errors))

def status(root):
    rows=[]
    for t,s,a in JOBS:
        path=run_path(root,t,s,a)/'state.json'
        if path.exists():
            st=old.read(path);r=dict(queries=len(st['decisions']),status=st['status'])
            if st['status']=='COMPLETE':r.update((v9 if a in ('bo','reasoning_bo') else v12).summary(st))
        else:r=dict(status='NOT_STARTED',queries=0)
        rows.append(dict(r,task=t,initialization_seed=s,study_arm=a))
    return rows
