"""External from-zero factorial; native selection, GP and feedback stay frozen."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from contextlib import ExitStack,contextmanager
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
from unittest.mock import patch
import csv
import json
import os
import shutil

import numpy as np
from cold_start_v13 import study as native,control,policy,prompts,adapter
from cold_start_v13.gp import ColdGP
from reasoning_memory_v12.llm import validate_insight,compiled_extractor
from . import VARIANT,MODEL,EFFORT,ARMS,queue

ROOT=Path(__file__).resolve().parents[3]
DESIGN='docs/v13_origin_ablation_20261103.txt'
SHORT=dict(native.SHORT)
CALIBRATION=control.calibration.update
EDIT_EVENT=policy.edit_event
INITIAL=native.initial_state
PRIORITIES=native.native_gp.semantic_priorities
POOL=native.pool_for
_CHECKED=None


def base():return queue.base(ROOT)
def rel(p):return Path(p).resolve().relative_to(ROOT).as_posix()
def read(p):return queue.read(p)
def write(p,x):return queue.write_once(p,x)
def sha(p):return queue.file_sha(p)


def jobs():
    return [dict(id=f'{short}{seed}_{arm}',cell=f'{short}{seed}',task=task,seed=seed,arm=arm)
            for task,short in SHORT.items() for seed in (0,1,2,3) for arm in ARMS]


def historical():
    with (ROOT/'paper/v13_late/data/metrics.csv').open(encoding='utf-8-sig',newline='') as f:
        return [r for r in csv.DictReader(f) if r['regime']=='cold']


def original(task,seed,method='v13'):
    return next(r for r in historical() if r['task']==SHORT[task] and int(r['seed'])==seed and r['method']==method)


def seed_responses():
    count=0;sources={}
    for m in historical():
        if m['method']!='v13':continue
        sf=ROOT/m['state_path']
        if sha(sf)!=m['state_sha256']:raise ValueError('Frozen full state changed')
        state=read(sf);cell=f"{m['task']}{m['seed']}"
        for d in state['decisions']:
            cp=sf.parent/'rounds'/f"{d['query']:02}_commit.json";c=read(cp)
            sources[rel(cp)]=sha(cp)
            for key,kind in (('reasoning','choose'),('memory','edit')):
                value=c[key]
                if value is None:continue
                req=ROOT/value['request_relpath'];ctx=req.parent.parent/'context.json';raw=req.parent/'response.raw.json'
                if read(raw)!=value['raw']:raise ValueError('Accepted original raw response changed')
                for p in (req,ctx,raw):sources[rel(p)]=sha(p)
                count+=queue.seed(ROOT,cell,kind,read(ctx),raw,dict(source_request=rel(req),
                    source_request_sha256=sha(req),source_context=rel(ctx),source_context_sha256=sha(ctx),
                    source_raw=rel(raw),source_raw_sha256=sha(raw),source_commit=rel(cp),
                    source_commit_sha256=sha(cp),source_attempts=value['attempts']))
    write(base()/'seeded_responses.json',dict(entries=count,source_files=sources,
        selection='all accepted cold V13 responses; no outcome-based selection; exact input within same start'))
    return count


def prepare():
    if (base()/'protocol.json').exists():return check(True)
    base().mkdir(parents=True,exist_ok=True)
    graph={};inputs={};protected={}
    for name in ('cold0_20261102','cold_s123_20261102'):
        p=read(ROOT/'experiments/runs'/name/'protocol.json')
        graph.update(p['sources_sha256'])
        inputs.update(p.get('inputs_sha256',{}))
    for path in Path(__file__).parent.glob('*.py'):graph[rel(path)]=sha(path)
    for name in (DESIGN,'code/experiment_tools/run_v13_origin_ablation.py'):
        graph[name]=sha(ROOT/name)
    for m in historical():
        p=ROOT/m['state_path']
        if sha(p)!=m['state_sha256']:raise ValueError('Historical state hash mismatch')
        protected[rel(p)]=sha(p)
    for name in ('paper/v13_source.zip','paper/v13/main.pdf','paper/v13/supplement.pdf',
                 'paper/v13_late_source.zip','paper/v13_late/main.pdf','paper/v13_late/supplement.pdf',
                 'paper/v13_late/data/metrics.csv','paper/v13_late/data/rounds.csv'):
        protected[name]=sha(ROOT/name)
    for task in SHORT:
        pool=POOL(ROOT,task);inputs[rel(pool.source_path)]=sha(pool.source_path)
        path=pool.source_path.parent/'oracle.npz'
        if not path.exists():
            with (ROOT/'paper/v13_late/data/datasets.csv').open(encoding='utf-8-sig') as f:
                row=next(r for r in csv.DictReader(f) if r['task']==SHORT[task]);path=ROOT/row['oracle_path']
        inputs[rel(path)]=sha(path)
    count=seed_responses()
    probes=[dict(id=f'probe_doyle{s}_q{q}_{method}_r{rep}',cell=f'probe_doyle{s}_q{q}_{method}_r{rep}',
                 task='Doyle_Buchwald_Hartwig',seed=s,query=q,template=method,rep=rep,arm='probe')
            for s,q in ((0,2),(1,9)) for method in ('v13','rbo') for rep in range(3)]
    p=dict(variant=VARIANT,registered_utc=datetime.now(timezone.utc).isoformat(),jobs=jobs(),probes=probes,
        model=MODEL,effort=EFFORT,backend='fresh_subagents_with_exact_input_coupling',fork_turns='none',
        initial=0,queries=20,starting_seeds=[0,1,2,3],numerical_seed=0,MC_paths=512,
        full_replay_trajectories=32,new_control_trajectories=128,new_control_queries=2560,
        targeted_one_step_probes=12,seeded_responses=count,sources_sha256=graph,
        inputs_sha256=inputs,protected_sha256=protected,design=(ROOT/DESIGN).read_text(encoding='utf-8'),
        full_replay_maximum_fresh_LLM_calls=0,new_outcomes_before_registration=0)
    p['protocol_sha256']=control.digest(p);write(base()/'protocol.json',p)
    snap=base()/'frozen_source'
    for name,h in graph.items():
        dst=snap/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dst)
        if sha(dst)!=h:raise ValueError('Scientific source snapshot mismatch')
    return check(True)


def check(protected=False):
    p=read(base()/'protocol.json')
    if p['protocol_sha256']!=control.digest({k:v for k,v in p.items() if k!='protocol_sha256'}):
        raise ValueError('Registered protocol changed')
    for group in ('sources_sha256','inputs_sha256'):
        for name,h in p[group].items():
            if sha(ROOT/name)!=h:raise ValueError('Frozen source/input changed: '+name)
    if protected:
        for name,h in p['protected_sha256'].items():
            if sha(ROOT/name)!=h:raise ValueError('Historical result changed: '+name)
        for name,h in read(base()/'seeded_responses.json')['source_files'].items():
            if sha(ROOT/name)!=h:raise ValueError('Original response evidence changed: '+name)
    return p


def fixed_update(state,p,g,s):
    proposed,e=CALIBRATION(state,p,g,s);actual=deepcopy(state);actual['updates']+=1
    diagnostic=dict(before=state,p_LLM=p,p_GP=g,improved=s,native_proposed=proposed,
        native_diagnostic=e,actual=actual,intervention='fixed_birth_calibration')
    if queue.CURRENT is not None:
        key=control.digest(dict(state=state,p=p,g=g,s=s))
        write(base()/'numerical_diagnostics'/queue.CURRENT['id']/(key+'.json'),diagnostic)
    # Do not leak un-deployed counterfactual weights or treatment labels into
    # recent_feedback shown to the chooser. Record them in the private sidecar.
    e=deepcopy(e);e.update(after=deepcopy(actual),kalman_gain=0.,predicted_variance=state['variance'])
    return actual,e


def edit_event(arm,pool,state,ordered,q):
    if arm=='no_notes':return dict(allowed=False,reasons=['structured_notes_disabled'])
    if arm in ('no_refresh','neither') and state['edit_calls']:
        return dict(allowed=False,reasons=['refresh_disabled_after_initial_editor'])
    return EDIT_EVENT(pool,state,ordered,q)


@contextmanager
def redirected(job,p):
    def initial(task,arm,protocol,commitment):
        s=INITIAL(task,arm,protocol,commitment)
        s.update(seed=job['seed'],starting_seed=job['seed'],numerical_seed=0)
        return s
    def priorities(pool,q,seed=0):
        if seed!=0:raise ValueError('Numerical GP/tie seed changed')
        return PRIORITIES(pool,q,job['seed'] if q==1 else 0)
    with ExitStack() as stack:
        stack.enter_context(patch.object(native,'check',return_value=p))
        stack.enter_context(patch.object(native,'directory',return_value=base()))
        stack.enter_context(patch.object(native,'run_path',return_value=base()/'runs'/job['id']))
        stack.enter_context(patch.object(native,'tag',return_value=job['id']))
        stack.enter_context(patch.object(native,'initial_state',side_effect=initial))
        stack.enter_context(patch.object(native.transport,'request',side_effect=queue.request))
        stack.enter_context(patch.object(native.native_gp,'semantic_priorities',side_effect=priorities))
        stack.enter_context(patch.object(policy,'edit_event',side_effect=lambda pool,state,ordered,q:
            edit_event(job['arm'],pool,state,ordered,q)))
        if job['arm'] in ('fixed','neither'):
            stack.enter_context(patch.object(control.calibration,'update',side_effect=fixed_update))
        queue.CURRENT=job
        try:yield
        finally:queue.CURRENT=None


def run_job(job):
    global _CHECKED
    if _CHECKED is None:_CHECKED=check()
    os.chdir(ROOT/'experiments/chemist_system')
    with redirected(job,_CHECKED):result=native.run_one(ROOT,job['task'],'v13')
    state=read(base()/'runs'/job['id']/'state.json')
    if job['arm']=='full':
        old=read(ROOT/original(job['task'],job['seed'])['state_path'])
        if state['history']!=old['history'] or state['notes']!=old['notes']:
            raise ValueError('Full replay changed original trajectory or knowledge state')
    if job['arm'] in ('no_refresh','neither') and len(state['edit_calls'])!=1:
        raise ValueError('Initial editor was not retained exactly once')
    if job['arm']=='no_notes' and (state['notes'] or state['edit_calls'] or any(d['feedback'] for d in state['decisions'])):
        raise ValueError('Structured Note ablation leaked a Note')
    write(base()/'runs'/job['id']/'intervention_complete.json',dict(job=job,
        history_sha256=control.digest(state['history']),notes_sha256=control.digest(state['notes'])))
    return result


def copy_numeric_caches():
    count=0;total=0
    for variant in ('cold0_20261102','cold_s123_20261102'):
        for task in SHORT:
            source=ROOT/'experiments/runs'/variant/'cache'/task
            target=base()/'cache'/task;target.mkdir(parents=True,exist_ok=True)
            for path in source.glob('*'):
                if path.suffix not in ('.pt','.npz'):continue
                dst=target/path.name
                if not dst.exists():shutil.copy2(path,dst);count+=1;total+=path.stat().st_size
                elif sha(path)!=sha(dst):
                    if path.suffix!='.npz':raise ValueError('Conflicting native model cache: '+path.name)
                    with np.load(path,allow_pickle=False) as a,np.load(dst,allow_pickle=False) as b:
                        if a.files!=b.files:raise ValueError('Numerical cache schema conflict')
                        for k in a.files:np.testing.assert_array_equal(a[k],b[k])
    write(base()/'numeric_cache_transport.json',dict(copied=count,bytes=total,
        sources=['cold0_20261102','cold_s123_20261102'],only_exact_prefix_keys=True))


def preflight():
    p=check(True);copy_numeric_caches();proofs=[]
    os.chdir(ROOT/'experiments/chemist_system');compiled_extractor(str(ROOT))
    for task in SHORT:
        pool=POOL(ROOT,task);gp=ColdGP(pool,base()/'cache'/task,seed=0)
        for seed in (0,1,2,3):
            job=dict(task=task,seed=seed,arm='full',id='preflight',cell='preflight')
            sf=ROOT/original(task,seed)['state_path'];old=read(sf)
            with redirected(job,p):
                for q in (1,2,5):
                    c=read(sf.parent/'rounds'/f'{q:02}_commit.json')
                    ctx=gp.context(old['history'][:q-1],q)
                    if native.native.gp_hash(ctx)!=c['GP_sha256']:raise ValueError('Native numerical replay differs')
                    _,display=native.native.candidates(pool,ctx,q)
                    if [pool.ids[int(j)] for j in display]!=c['display_ids']:raise ValueError('Shared menu changed')
                    gp.release()
            proofs.append(dict(task=task,seed=seed,exact_GP_queries=[1,2,5]))
        if gp.counters['operator_compiles_in_loop']!=0:raise ValueError('Loop operator compilation')
        gp.release()
    result=dict(status='PASS',matched_contexts=32,proofs=proofs,old_results_unchanged=True,
                newly_evaluated_outcomes=0,new_LLM_calls=0)
    write(base()/'preflight.json',result);return result


def batch(stage='controls',workers=6):
    p=check(True)
    if read(base()/'preflight.json')['status']!='PASS':raise ValueError('Preflight required')
    todo=[j for j in p['jobs'] if (j['arm']=='full')==(stage=='replay')]
    if stage=='controls':
        if not (base()/'replay_batch.json').exists() or read(base()/'replay_batch.json')['status']!='COMPLETE':
            raise ValueError('All full replays must complete before new model calls')
    errors=[]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures={ex.submit(run_job,j):j for j in todo}
        for f in as_completed(futures):
            try:f.result()
            except Exception as e:
                row=dict(job=futures[f]['id'],type=type(e).__name__,message=str(e));errors.append(row)
                print('JOB_ERROR '+json.dumps(row),flush=True)
    result=dict(status='INCOMPLETE' if errors else 'COMPLETE',stage=stage,jobs=len(todo),errors=errors)
    if stage=='replay' and queue.pending(ROOT):raise ValueError('Full replay made a fresh LLM request')
    check(True);write(base()/(stage+'_batch.json'),result);return result


def run_probe(job):
    p=check();sf=ROOT/original(job['task'],job['seed'])['state_path'];old=read(sf)
    state=deepcopy(old);state.update(status='READY',history=[],notes=[],decisions=[],insights=[],edit_calls=[],review_queries=[])
    for q in range(1,job['query']):
        c=read(sf.parent/'rounds'/f'{q:02}_commit.json')
        state=native.feedback(state,c,old['history'][q-1])
    os.chdir(ROOT/'experiments/chemist_system')
    pool=POOL(ROOT,job['task']);gp=ColdGP(pool,base()/'cache'/job['task'],seed=0)
    ctx=gp.context(state['history'],job['query']);c=read(sf.parent/'rounds'/f"{job['query']:02}_commit.json")
    if native.native.gp_hash(ctx)!=c['GP_sha256']:raise ValueError('Probe GP differs')
    ids=c['display_ids'];state['notes']=[]
    if job['template']=='v13':
        system=prompts.REASONING_SYSTEM;public=prompts.reasoning_public(ROOT,pool,state,[],ids,job['query'])
        validator=lambda raw:policy.validate_decision(ROOT,pool,[],state,ctx,ids,job['query'],raw)
        selected=lambda raw:validate_insight(ROOT,raw['insight'],ids)
    else:
        system=prompts.RBO_SYSTEM;public=prompts.rbo_public(ROOT,pool,state,ids,job['query'])
        validator=lambda raw:dict(selected_id=validate_insight(ROOT,raw,ids))
        selected=lambda raw:validate_insight(ROOT,raw,ids)
    queue.CURRENT=job
    try:r=queue.request(ROOT,job['id'],'probe',job['query'],system,public,validator,selected_validator=selected)
    finally:queue.CURRENT=None
    chosen=r['plan']['selected_id'];folder=base()/'probes'/job['id']
    write(folder/'commit.json',dict(job=job,response=r,selected_id=chosen,GP_sha256=c['GP_sha256'],
        observed_prefix=state['history'],shared_Insight_history=state['insights'],menu=ids,protocol_sha256=p['protocol_sha256']))
    observation=native.native_gp.read_observation(ROOT,pool,chosen)
    write(folder/'outcome.json',dict(observation=observation,
        incumbent=max(h['raw_value'] for h in state['history']),counts_as_full_trajectory=False))
    gp.release();return dict(id=job['id'],status='COMPLETE')


def probes(workers=6):
    p=check(True)
    with ProcessPoolExecutor(max_workers=workers) as ex:result=list(ex.map(run_probe,p['probes']))
    write(base()/'probes_complete.json',dict(status='COMPLETE',probes=result));return result
