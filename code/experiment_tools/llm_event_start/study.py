"""Unrestricted first-point comparison and matched-start refresh control."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
from unittest.mock import patch
import argparse
import json
import os
import shutil
import time
import numpy as np
from chem.benchmarks import knowledge_gp
from cold_start_v13 import transport
from fast_hit_bo import study as frozen,PARAMETERS as GP_PARAMETERS
from fast_hit_bo.model import SharedGP,digest
from llm_screen_qlogei import knowledge
from . import VARIANT,MODEL,EFFORT,PARAMETERS as P,policy,prompts

ROOT=frozen.ROOT
BASE=ROOT/'experiments/runs'/VARIANT
REPORT=ROOT/'experiments/reports'/VARIANT
REFERENCE=ROOT/'experiments/runs/fast_hit_bo_20261104'
DESIGN=ROOT/'docs'/f'{VARIANT}.txt'
read,sha=frozen.read,frozen.sha
GP=POOL=None


def write(path,value,replace=False):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and not replace:
        assert read(path)==value,'Immutable evidence differs: '+str(path)
        return
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for attempt in range(40):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt==39:
                raise
            time.sleep(.025)


@contextmanager
def transport_scope():
    with patch.object(transport,'VARIANT',VARIANT):
        yield


def check(protected=False):
    p=read(BASE/'protocol.json')
    assert p['protocol_sha256']==digest({k:v for k,v in p.items() if k!='protocol_sha256'})
    for group in ('sources_sha256','inputs_sha256')+(('protected_sha256',) if protected else ()):
        for name,h in p[group].items():
            if sha(ROOT/name)!=h:
                raise ValueError('Frozen file changed: '+name)
    return p


def prepare():
    if (BASE/'protocol.json').exists():
        return check(True)
    pool=knowledge_gp.load_task(ROOT,'Reizman_Suzuki_Case_1',seed=0)
    local_sources={}
    for directory in ('code/chemist_system/chem','code/experiment_tools'):
        for source in (ROOT/directory).rglob('*.py'):
            local_sources[source.relative_to(ROOT).as_posix()]=sha(source)
    old=dict(sources_sha256=local_sources,inputs_sha256={
        source.relative_to(ROOT).as_posix():sha(source)
        for source in (pool.source_path,pool.source_path.with_name('oracle.npz'))})
    jobs=[dict(id=f'{arm}_s{seed}',seed=seed,arm=arm,
        phase='control' if arm=='llm_start_bo' else 'main')
        for seed in P['seeds'] for arm in ('bo','event','llm_start_bo')]
    sources=dict(old['sources_sha256'])
    paths=list(Path(__file__).parent.glob('*.py'))+[
        ROOT/'code/experiment_tools/run_llm_event_start.py',DESIGN,
        ROOT/'code/experiment_tools/llm_screen_qlogei/__init__.py',
        ROOT/'code/experiment_tools/llm_screen_qlogei/knowledge.py',
        ROOT/'code/experiment_tools/reasoning_memory_v13/calibration.py',
        ROOT/'code/experiment_tools/cold_start_v13/transport.py',
        ROOT/'code/experiment_tools/paper_v12_subagent/transport.py',
        ROOT/'code/chemist_system/chem/benchmarks/knowledge_domain_context.py',
        ROOT/'code/tools/event_start_operations.py']
    for path in paths:
        sources[path.relative_to(ROOT).as_posix()]=sha(path)
    protected={}
    for variant in ('fast_hit_bo_20261104','fast_hit_qlogei_rollout_20261104','llm_screen_qlogei_20261104'):
        for directory in (ROOT/'experiments/runs'/variant/'runs',ROOT/'experiments/reports'/variant):
            for path in directory.rglob('*'):
                if path.is_file() and path.suffix!='.tmp':
                    protected[path.relative_to(ROOT).as_posix()]=sha(path)
        path=ROOT/'experiments/runs'/variant/'protocol.json'
        if path.exists():
            protected[path.relative_to(ROOT).as_posix()]=sha(path)
    p=dict(variant=VARIANT,registered_utc=datetime.now(timezone.utc).isoformat(),jobs=jobs,
        parameters=P,GP_parameters=GP_PARAMETERS,model=MODEL,effort=EFFORT,
        backend='fresh_subagent_per_request',fork_turns='none',initial=1,total_queries=8,
        total_trajectories=30,trajectory_observations=240,LLM_calls_per_trajectory_maximum=3,
        initial_yield_conditioning=False,sources_sha256=sources,inputs_sha256=old['inputs_sha256'],
        protected_sha256=protected,design=DESIGN.read_text(encoding='utf-8'),new_outcomes_at_registration=0)
    p['protocol_sha256']=digest(p)
    write(BASE/'protocol.json',p)
    write(BASE/'evaluation_endpoint.json',read(ROOT/'results'/VARIANT/'source_evaluation_endpoint.json'))
    (BASE/'cache').mkdir(exist_ok=True)
    for path in (REFERENCE/'cache').glob('*'):
        if path.suffix in ('.pt','.npz'):
            shutil.copyfile(path,BASE/'cache'/path.name)
    return check(True)


def initialize():
    global GP,POOL
    check()
    os.chdir(ROOT/'experiments/chemist_system')
    POOL=knowledge_gp.load_task(ROOT,'Reizman_Suzuki_Case_1',seed=0)
    GP=SharedGP(POOL,BASE/'cache')


def initial_state(job):
    return dict(job=job,history=[],decisions=[],notes=[],knowledge_archive=[],
        knowledge_calls=[],refresh_queries=[],status='RUNNING')


def run_job(job):
    folder=BASE/'runs'/job['id']
    sf=folder/'state.json'
    state=read(sf) if sf.exists() else initial_state(job)
    if state['status']=='COMPLETE':
        return dict(job=job['id'],status='COMPLETE',queries=8)
    while len(state['history'])<8:
        q=len(state['history'])+1
        cp=folder/'rounds'/f'{q:02}_commit.json'
        if cp.exists():
            commit=read(cp)
        else:
            context=None if q==1 else GP.context(state['history'],q)
            event=dict(allowed=False,reasons=['normal_full_pool_qLogEI'])
            llm=None
            notes=deepcopy(state['notes'])
            proposal=None
            source=None
            if job['arm']=='event':
                event=dict(allowed=True,reasons=['zero_observation_initial_proposal']) if q==1 else knowledge.refresh_event(state,q)
                if event['allowed']:
                    with transport_scope():
                        llm=transport.request(ROOT,job['id'],'knowledge',q,prompts.SYSTEM,
                            prompts.public(POOL,state,q,event),
                            lambda raw:policy.validate(raw,POOL,state['notes'],q))
                    notes=llm['plan']['notes']
                    proposal=llm['plan']['proposal']
            packet=policy.decide(POOL,context,notes,proposal,q,job['seed'],job['arm'])
            if q==1 and job['arm']=='llm_start_bo':
                path=BASE/'runs'/f"event_s{job['seed']}"/'rounds/01_commit.json'
                original=read(path)
                packet.update(selected_id=original['decision']['selected_id'],rule='exact_shared_LLM_initial_point')
                source=dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path))
            selected_gp=None
            fit_key=None
            if context is not None:
                j=POOL.ix[packet['selected_id']]
                selected_gp=dict(mean=float(context['mu'][j])*100,sd=float(context['variance'][j]**.5)*100,
                    qLogEI=float(context['logei'][j]),hyperparameters_fitted=context['hyperparameters_fitted'])
                fit_key=context['key']
                array=folder/'rounds'/f'{q:02}_GP.npz'
                array.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(array,mu=context['mu'],variance=context['variance'],logei=context['logei'],
                    available=context['available'],priorities=context['priorities'])
                packet.update(GP_arrays_sha256=sha(array),checkpoint=context['checkpoint'].relative_to(ROOT).as_posix(),
                    checkpoint_sha256=sha(context['checkpoint']))
            commit=dict(query=q,prestate_sha256=digest(state),history_sha256=digest(state['history']),
                protocol_sha256=read(BASE/'protocol.json')['protocol_sha256'],decision=packet,
                selected_GP=selected_gp,fit_key=fit_key,knowledge_event=event,LLM=llm,
                notes_at_commit=notes,shared_initial_source=source)
            commit['commit_sha256']=digest(commit)
            write(cp,commit)
        assert commit['prestate_sha256']==digest(state)
        assert commit['commit_sha256']==digest({k:v for k,v in commit.items() if k!='commit_sha256'})
        assert commit==read(cp)
        cid=commit['decision']['selected_id']
        assert cid not in {h['candidate_id'] for h in state['history']}
        observed=knowledge_gp.read_observation(ROOT,POOL,cid)
        notes,events=knowledge.feedback(commit['notes_at_commit'],commit['decision']['forecasts'],observed,q)
        if commit['LLM'] is not None:
            state['knowledge_archive'].append(dict(query=q,notes=deepcopy(state['notes'])))
            state['knowledge_calls'].append(q)
            if q>1:
                state['refresh_queries'].append(q)
        state['notes']=notes
        state['history'].append(observed)
        state['decisions'].append(commit)
        state['status']='COMPLETE' if q==8 else 'RUNNING'
        write(folder/'rounds'/f'{q:02}_feedback.json',dict(query=q,events=events,notes_after=notes))
        write(sf,state,replace=True)
        assert sum(knowledge_gp.COMPILE_COUNTS.values())==0
        write(folder/'runtime.json',dict(operator_compiles_in_loop=0,**GP.counters),replace=True)
        print(json.dumps(dict(job=job['id'],query=q,yield_percent=observed['raw_value'],
            best=max(h['raw_value'] for h in state['history']),LLM_call=commit['LLM'] is not None,
            rule=commit['decision']['rule'],notes_scored=len(events))),flush=True)
    return dict(job=job['id'],status='COMPLETE',queries=8)


def preflight():
    from .validation import qualify
    p=check(True)
    initialize()
    result=qualify(GP,POOL,read(ROOT/'results'/VARIANT/'initial_GP_reference.json'))
    result['protocol_sha256']=p['protocol_sha256']
    write(BASE/'preflight.json',result)
    return result


def batch(phase,workers):
    p=check(True)
    assert read(BASE/'preflight.json')['status']=='PASS'
    if phase=='control':
        assert read(BASE/'main_receipt.json')['status']=='COMPLETE'
    jobs=[j for j in p['jobs'] if j['phase']==phase]
    results,errors=[],[]
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize) as executor:
        futures={executor.submit(run_job,j):j for j in jobs}
        for f in as_completed(futures):
            try:
                results.append(f.result())
            except Exception as error:
                failure=dict(job=futures[f]['id'],type=type(error).__name__,error=str(error))
                errors.append(failure)
                print(json.dumps(failure),flush=True)
    check(True)
    result=dict(status='FAILED' if errors else 'COMPLETE',results=results,errors=errors)
    write(BASE/f'{phase}_receipt.json',result,replace=True)
    if errors:
        raise RuntimeError(errors)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('prepare','preflight','run'))
    parser.add_argument('--phase',choices=('main','control'),default='main')
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    result=prepare() if args.action=='prepare' else preflight() if args.action=='preflight' else batch(args.phase,args.workers)
    if args.action=='prepare':
        result=dict(status='REGISTERED',jobs=len(result['jobs']),protocol_sha256=result['protocol_sha256'])
    print(json.dumps(result))
