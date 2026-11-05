"""Full trajectories from frozen LLM initial points, with prospective feedback."""
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
from fast_hit_bo.model import SharedGP,digest
from llm_event_start import study as previous,policy as original_policy
from llm_screen_qlogei import knowledge
from . import VARIANT,PARAMETERS as P,policy,prompts

ROOT=previous.ROOT
BASE=ROOT/'experiments/runs'/VARIANT
REPORT=ROOT/'experiments/reports'/VARIANT
REFERENCE=ROOT/'experiments/runs/llm_event_start_qlogei_20261104'
DESIGN=ROOT/'docs/llm_bo_handoff_study_20261104.txt'
MODEL='gpt-6.1-terra'
EFFORT='medium'
read,write,sha=previous.read,previous.write,previous.sha
GP=POOL=None


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
    old=previous.check(True)
    jobs=[dict(id=f'handoff_s{seed}',seed=seed,arm='handoff',initial_source_job=f'event_s{seed}')
          for seed in range(10)]
    sources=dict(old['sources_sha256'])
    for source in list(Path(__file__).parent.glob('*.py'))+[
        ROOT/'code/experiment_tools/run_llm_bo_handoff.py',
        ROOT/'code/tools/handoff_operations.py',DESIGN,
        ROOT/'docs/llm_bo_handoff_20261104.txt']:
        sources[source.relative_to(ROOT).as_posix()]=sha(source)
    protected={}
    for directory in (REFERENCE/'runs',REFERENCE/'cache',
        ROOT/'experiments/reports/llm_event_start_qlogei_20261104',
        ROOT/'experiments/reports/llm_initialization_audit_20261104'):
        for path in directory.rglob('*'):
            if path.is_file() and path.suffix not in ('.lock','.tmp'):
                protected[path.relative_to(ROOT).as_posix()]=sha(path)
    protected[(REFERENCE/'protocol.json').relative_to(ROOT).as_posix()]=sha(REFERENCE/'protocol.json')
    p=dict(variant=VARIANT,registered_utc=datetime.now(timezone.utc).isoformat(),jobs=jobs,
        references=[j for j in old['jobs'] if j['arm'] in ('bo','llm_start_bo','event')],
        parameters=P,GP_parameters=old['GP_parameters'],model=MODEL,effort=EFFORT,
        backend='fresh_subagent_per_request',fork_turns='none',initial=1,total_queries=8,
        new_trajectories=10,new_prospective_queries=70,reused_initial_observations=10,
        initial_yield_conditioning=False,initial_LLM_replies_reused=True,
        sources_sha256=sources,inputs_sha256=old['inputs_sha256'],protected_sha256=protected,
        design=DESIGN.read_text(encoding='utf-8'),new_prospective_outcomes_at_registration=0)
    p['protocol_sha256']=digest(p)
    write(BASE/'protocol.json',p)
    write(BASE/'evaluation_endpoint.json',read(REFERENCE/'evaluation_endpoint.json'))
    (BASE/'cache').mkdir(exist_ok=True)
    copied=0
    for path in (REFERENCE/'cache').iterdir():
        if path.suffix in ('.pt','.npz'):
            destination=BASE/'cache'/path.name
            shutil.copyfile(path,destination)
            assert sha(path)==sha(destination)
            copied+=1
    write(BASE/'cache_copy.json',dict(files=copied,key_policy='Exact observed prefix only'))
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


def event_for(state,query):
    if query==P['total_queries']:
        return dict(allowed=False,reasons=['pure_BO_endpoint_no_LLM_authority'])
    return knowledge.refresh_event(state,query)


def bootstrap(job,state,folder):
    sf=folder/'state.json'
    cp=folder/'rounds/01_commit.json'
    source_folder=REFERENCE/'runs'/job['initial_source_job']
    source_state=read(source_folder/'state.json')
    source_commit=read(source_folder/'rounds/01_commit.json')
    if cp.exists():
        c=read(cp)
    else:
        notes=deepcopy(source_commit['notes_at_commit'])
        proposal=source_commit['decision']['proposal']
        packet,_=policy.decide(POOL,None,notes,proposal,1,job['seed'])
        assert packet['selected_id']==source_state['history'][0]['candidate_id']
        c=dict(query=1,prestate_sha256=digest(state),history_sha256=digest([]),
            protocol_sha256=read(BASE/'protocol.json')['protocol_sha256'],decision=packet,
            selected_GP=None,fit_key=None,LLM=None,notes_at_commit=notes,
            knowledge_event=dict(allowed=False,reasons=['frozen_LLM_initialization_reused']),
            initial_source=dict(state_path=(source_folder/'state.json').relative_to(ROOT).as_posix(),
                state_sha256=sha(source_folder/'state.json'),
                commit_path=(source_folder/'rounds/01_commit.json').relative_to(ROOT).as_posix(),
                commit_sha256=sha(source_folder/'rounds/01_commit.json'),
                original_LLM_request_path=source_commit['LLM']['request_relpath'],
                original_LLM_raw_sha256=source_commit['LLM']['response_receipt']['raw_sha256'],
                observation_sha256=digest(source_state['history'][0]),
                reused_prior_observation=True,new_LLM_call=False,new_oracle_query=False))
        c['commit_sha256']=digest(c)
        write(cp,c)
    assert c['prestate_sha256']==digest(state)
    assert c['commit_sha256']==digest({k:v for k,v in c.items() if k!='commit_sha256'})
    state['history'].append(deepcopy(source_state['history'][0]))
    state['decisions'].append(c)
    state['notes']=deepcopy(c['notes_at_commit'])
    write(folder/'rounds/01_feedback.json',dict(query=1,events=[],notes_after=state['notes'],
        reused_initial_observation=True))
    write(sf,state,replace=True)
    return state


def run_job(job):
    folder=BASE/'runs'/job['id']
    sf=folder/'state.json'
    state=read(sf) if sf.exists() else initial_state(job)
    if not state['history']:
        state=bootstrap(job,state,folder)
    if state['status']=='COMPLETE':
        return dict(job=job['id'],status='COMPLETE',queries=8)
    while len(state['history'])<P['total_queries']:
        q=len(state['history'])+1
        cp=folder/'rounds'/f'{q:02}_commit.json'
        if cp.exists():
            c=read(cp)
        else:
            context=GP.context(state['history'],q)
            notes=deepcopy(state['notes'])
            event=event_for(state,q)
            llm=None
            proposal=None
            if event['allowed']:
                with transport_scope():
                    llm=transport.request(ROOT,job['id'],'knowledge',q,prompts.SYSTEM,
                        prompts.public(POOL,state,q,event,policy.authority(q)),
                        lambda raw:original_policy.validate(raw,POOL,state['notes'],q))
                notes=llm['plan']['notes']
                proposal=llm['plan']['proposal']
            packet,screen=policy.decide(POOL,context,notes,proposal,q,job['seed'])
            j=POOL.ix[packet['selected_id']]
            array=folder/'rounds'/f'{q:02}_GP.npz'
            array.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(array,mu=context['mu'],variance=context['variance'],
                logei=context['logei'],available=context['available'],priorities=context['priorities'],**screen)
            packet.update(GP_arrays_sha256=sha(array),checkpoint=context['checkpoint'].relative_to(ROOT).as_posix(),
                checkpoint_sha256=sha(context['checkpoint']))
            c=dict(query=q,prestate_sha256=digest(state),history_sha256=digest(state['history']),
                protocol_sha256=read(BASE/'protocol.json')['protocol_sha256'],decision=packet,
                selected_GP=dict(mean=float(context['mu'][j])*100,sd=float(context['variance'][j]**.5)*100,
                    qLogEI=float(context['logei'][j]),hyperparameters_fitted=context['hyperparameters_fitted']),
                fit_key=context['key'],LLM=llm,notes_at_commit=notes,knowledge_event=event,initial_source=None)
            c['commit_sha256']=digest(c)
            write(cp,c)
        assert c['prestate_sha256']==digest(state)
        assert c['history_sha256']==digest(state['history'])
        assert c['commit_sha256']==digest({k:v for k,v in c.items() if k!='commit_sha256'})
        assert c==read(cp)
        cid=c['decision']['selected_id']
        assert cid not in {h['candidate_id'] for h in state['history']}
        observed=knowledge_gp.read_observation(ROOT,POOL,cid)
        notes,events=knowledge.feedback(c['notes_at_commit'],c['decision']['forecasts'],observed,q)
        if c['LLM'] is not None:
            state['knowledge_archive'].append(dict(query=q,notes=deepcopy(state['notes'])))
            state['knowledge_calls'].append(q)
            state['refresh_queries'].append(q)
        state['notes']=notes
        state['history'].append(observed)
        state['decisions'].append(c)
        state['status']='COMPLETE' if q==8 else 'RUNNING'
        write(folder/'rounds'/f'{q:02}_feedback.json',dict(query=q,events=events,notes_after=notes))
        write(sf,state,replace=True)
        assert sum(knowledge_gp.COMPILE_COUNTS.values())==0
        write(folder/'runtime.json',dict(operator_compiles_in_loop=0,**GP.counters),replace=True)
        print(json.dumps(dict(job=job['id'],query=q,yield_percent=observed['raw_value'],
            best=max(h['raw_value'] for h in state['history']),
            authority=c['decision']['handoff_authority_cap'],epsilon=c['decision']['effective_logei_gap'],
            differs_from_current_BO=c['decision']['knowledge_used'],refresh=c['LLM'] is not None)),flush=True)
    return dict(job=job['id'],status='COMPLETE',queries=8)


def preflight():
    from .check import run
    p=check(True)
    run()
    initialize()
    validated=[]
    for job in p['jobs']:
        source=read(REFERENCE/'runs'/job['initial_source_job']/'state.json')
        c=source['decisions'][0]
        plan=original_policy.validate(c['LLM']['raw'],POOL,[],1)
        initial,_=policy.decide(POOL,None,plan['notes'],plan['proposal'],1,job['seed'])
        assert initial['selected_id']==source['history'][0]['candidate_id']
        context=GP.context(source['history'][:1],2)
        with np.load(REFERENCE/'runs'/job['initial_source_job']/'rounds/02_GP.npz',allow_pickle=False) as z:
            for key in ('mu','variance','logei','available','priorities'):
                np.testing.assert_array_equal(context[key],z[key])
        endpoint,_=policy.decide(POOL,context,plan['notes'],None,8,job['seed'])
        assert endpoint['selected_id']==POOL.ids[policy.choose(context['logei'],context['available'],context['priorities'])]
        validated.append(dict(job=job['id'],initial_point_and_knowledge_reused_exactly=True,
            first_GP_and_scores_match=True,terminal_native_BO_rule_exact=True))
    assert GP.counters['fits']==0
    result=dict(status='PASS',protocol_sha256=p['protocol_sha256'],checks=validated,
        new_LLM_calls=0,new_GP_fits=0,new_prospective_queries=0,operator_compiles_in_loop=0)
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
