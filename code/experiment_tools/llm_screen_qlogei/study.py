"""Matched Case 1 screening pilot, with immutable prospective decisions."""
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
import numpy as np

from chem.benchmarks import knowledge_gp
from cold_start_v13 import transport
from fast_hit_bo import study as reference,PARAMETERS as GP_PARAMETERS
from fast_hit_bo.model import SharedGP,digest
from . import VARIANT,MODEL,EFFORT,PARAMETERS as P
from . import knowledge,policy,prompts

ROOT=reference.ROOT
BASE=ROOT/'experiments/runs'/VARIANT
REPORT=ROOT/'experiments/reports'/VARIANT
REFERENCE=ROOT/'experiments/runs/fast_hit_bo_20261104'
DESIGN=ROOT/'docs'/f'{VARIANT}.txt'
read,write,sha=reference.read,reference.write,reference.sha
GP=None
POOL=None


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
    old=read(REFERENCE/'protocol.json')
    controls=[j for j in old['jobs'] if j['arm']=='qlogei']
    jobs=[dict(j,id=j['id'].replace('_qlogei','_llm_kf_screen'),arm='llm_kf_screen') for j in controls]
    assert len(jobs)==10
    sources=dict(old['sources_sha256'])
    paths=list(Path(__file__).parent.glob('*.py'))+[
        ROOT/'code/experiment_tools/run_llm_screen_qlogei.py',DESIGN,
        ROOT/'code/experiment_tools/reasoning_memory_v13/calibration.py',
        ROOT/'code/experiment_tools/cold_start_v13/transport.py',
        ROOT/'code/experiment_tools/paper_v12_subagent/transport.py',
        ROOT/'code/chemist_system/chem/benchmarks/knowledge_domain_context.py',
        ROOT/'code/tools/llm_screen_dispatch.py']
    for path in paths:
        sources[path.relative_to(ROOT).as_posix()]=sha(path)
    protected={}
    for variant in ('fast_hit_bo_20261104','fast_hit_qlogei_rollout_20261104'):
        for directory in (ROOT/'experiments/runs'/variant/'runs',ROOT/'experiments/reports'/variant):
            for path in directory.rglob('*'):
                if path.is_file():
                    protected[path.relative_to(ROOT).as_posix()]=sha(path)
        path=ROOT/'experiments/runs'/variant/'protocol.json'
        protected[path.relative_to(ROOT).as_posix()]=sha(path)
    p=dict(variant=VARIANT,registered_utc=datetime.now(timezone.utc).isoformat(),
        jobs=jobs,reference_jobs=controls,parameters=P,GP_parameters=GP_PARAMETERS,
        model=MODEL,effort=EFFORT,backend='fresh_subagent_per_request',fork_turns='none',
        total_new_trajectories=10,total_new_observations=80,initial=1,total_queries=8,
        sources_sha256=sources,inputs_sha256=old['inputs_sha256'],protected_sha256=protected,
        design=DESIGN.read_text(encoding='utf-8'),new_outcomes_at_registration=0)
    p['protocol_sha256']=digest(p)
    write(BASE/'protocol.json',p)
    write(BASE/'evaluation_endpoint.json',read(REFERENCE/'evaluation_endpoint.json'))
    (BASE/'cache').mkdir(exist_ok=True)
    for path in (REFERENCE/'cache').iterdir():
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
            llm=None
            event=dict(allowed=False,reasons=['controlled_initial_point'])
            notes=deepcopy(state['notes'])
            if q==1:
                packet=dict(selected_id=job['initial_id'],rule='controlled_shared_initial_point',forecasts=[])
                selected_gp=None
                fit_key=None
            else:
                context=GP.context(state['history'],q)
                event=dict(allowed=True,reasons=['initial_knowledge']) if not state['knowledge_calls'] else knowledge.refresh_event(state,q)
                if event['allowed']:
                    public=prompts.public(POOL,state,q,event)
                    with transport_scope():
                        llm=transport.request(ROOT,job['id'],'knowledge',q,prompts.SYSTEM,public,
                            lambda raw:knowledge.install(raw,POOL,state['notes'],q))
                    notes=llm['plan']
                packet,screen=policy.decision(POOL,context,notes,q)
                j=POOL.ix[packet['selected_id']]
                selected_gp=dict(mean=float(context['mu'][j])*100,
                    sd=float(context['variance'][j]**.5)*100,qLogEI=float(context['logei'][j]),
                    hyperparameters_fitted=context['hyperparameters_fitted'])
                fit_key=context['key']
                array=folder/'rounds'/f'{q:02}_GP.npz'
                array.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(array,mu=context['mu'],variance=context['variance'],
                    logei=context['logei'],available=context['available'],priorities=context['priorities'],**screen)
                packet.update(GP_arrays_sha256=sha(array),
                    checkpoint=context['checkpoint'].relative_to(ROOT).as_posix(),
                    checkpoint_sha256=sha(context['checkpoint']))
            commit=dict(query=q,prestate_sha256=digest(state),history_sha256=digest(state['history']),
                protocol_sha256=read(BASE/'protocol.json')['protocol_sha256'],
                decision=packet,selected_GP=selected_gp,fit_key=fit_key,
                knowledge_event=event,LLM=llm,notes_at_commit=notes)
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
            if q>2:
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
            screen=commit['decision'].get('screen_mode'),notes_scored=len(events))),flush=True)
    return dict(job=job['id'],status='COMPLETE',queries=8)


def preflight():
    from .validation import qualify
    p=check(True)
    initialize()
    result=qualify(GP,POOL,read(REFERENCE/'runs/s4_middle_qlogei/state.json'))
    result['protocol_sha256']=p['protocol_sha256']
    write(BASE/'preflight.json',result)
    return result


def batch(workers):
    p=check(True)
    assert read(BASE/'preflight.json')['status']=='PASS'
    results,errors=[],[]
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize) as executor:
        futures={executor.submit(run_job,j):j for j in p['jobs']}
        for f in as_completed(futures):
            try:
                results.append(f.result())
            except Exception as error:
                failure=dict(job=futures[f]['id'],type=type(error).__name__,error=str(error))
                errors.append(failure)
                print(json.dumps(failure),flush=True)
    check(True)
    result=dict(status='FAILED' if errors else 'COMPLETE',results=results,errors=errors)
    write(BASE/'batch_receipt.json',result,replace=True)
    if errors:
        raise RuntimeError(errors)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('prepare','preflight','run'))
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    result=prepare() if args.action=='prepare' else preflight() if args.action=='preflight' else batch(args.workers)
    if args.action=='prepare':
        result=dict(status='REGISTERED',jobs=len(result['jobs']),protocol_sha256=result['protocol_sha256'])
    print(json.dumps(result))
