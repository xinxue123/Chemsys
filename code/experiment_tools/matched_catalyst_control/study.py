"""Precommitted Case 1 single-action intervention with sealed references."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import shutil
import warnings

import numpy as np

from chem.benchmarks import knowledge_gp
from fast_hit_bo.model import SharedGP, digest
from fast_hit_bo.policy import choose
from llm_bo_handoff import study as original
from llm_screen_qlogei import knowledge
from . import VARIANT, PARAMETERS as P

ROOT = original.ROOT
BASE = ROOT/'experiments/runs'/VARIANT
REPORT = ROOT/'experiments/reports'/VARIANT
SOURCE = original.BASE
BO_SOURCE = original.REFERENCE
DESIGN = ROOT/'docs'/f'{VARIANT}.txt'
read, write, sha = original.read, original.write, original.sha
POOL = GP = None


def matched_choice(pool, history, context):
    """Inputs and existing observations only; never read candidate outcomes."""
    continuous = [axis for axis, kind in zip(pool.axes, pool.kinds) if kind=='continuous']
    for observation in reversed(history):
        anchor = pool.ix[observation['candidate_id']]
        point = pool.points[anchor]
        mask = context['available'].copy()
        mask &= np.array([p['catalyst']!=point['catalyst'] for p in pool.points])
        for axis in continuous:
            mask &= np.array([abs(p[axis]-point[axis])<=P['exact_continuous_tolerance']
                              for p in pool.points])
        if not mask.any():
            continue
        j = choose(context['logei'], mask, context['priorities'])
        return dict(anchor_id=pool.ids[anchor], selected_id=pool.ids[j],
                    matched_candidates=int(mask.sum())), mask
    return None, np.zeros(len(pool.ids), dtype=bool)


def check(protected=False):
    p=read(BASE/'protocol.json')
    assert p['protocol_sha256']==digest({k:v for k,v in p.items() if k!='protocol_sha256'})
    for group in ('sources_sha256','inputs_sha256')+(('protected_sha256',) if protected else ()):
        for path,h in p[group].items():
            assert sha(ROOT/path)==h, path
    return p


def prepare():
    if (BASE/'protocol.json').exists():
        return check(True)
    old=original.check(True)
    pool=knowledge_gp.load_task(ROOT,'Reizman_Suzuki_Case_1',seed=0)
    jobs, exclusions=[],[]
    for source_job in old['jobs']:
        folder=SOURCE/'runs'/source_job['id']
        state=read(folder/'state.json')
        replacement=None
        actions=[]
        for q in range(2,9):
            commit=read(folder/'rounds'/f'{q:02}_commit.json')
            d=commit['decision']
            if not d['knowledge_used']:
                continue
            actions.append(q)
            with np.load(folder/'rounds'/f'{q:02}_GP.npz',allow_pickle=False) as z:
                context={key:z[key].copy() for key in ('logei','available','priorities')}
            pair,_=matched_choice(pool,state['history'][:q-1],context)
            if pair is None or pair['selected_id']==d['selected_id']:
                continue
            replacement=dict(**pair,query=q,original_selected_id=d['selected_id'],
                original_BO_id=d['BO_reference_id'],original_epsilon=d['effective_logei_gap'],
                native_logei_gap=float(context['logei'][pool.ix[d['BO_reference_id']]]-
                                      context['logei'][pool.ix[pair['selected_id']]]))
            break
        if replacement is None:
            exclusions.append(dict(seed=source_job['seed'],source_job=source_job['id'],
                original_knowledge_actions=actions,
                reason='no exact observed-anchor contrast at an original knowledge action'
                       if actions else 'no postinitial knowledge action'))
            continue
        for q in range(replacement['query']+1,9):
            d=state['decisions'][q-1]['decision']
            assert d['selected_id']==d['BO_reference_id'] and not d['knowledge_used']
        prefix=state['history'][:replacement['query']-1]
        cluster=digest(dict(prefix=prefix,selected_id=replacement['selected_id']))
        jobs.append(dict(id=f"contrast_s{source_job['seed']}",seed=source_job['seed'],
            source_job=source_job['id'],replacement=replacement,prefix_count=len(prefix),
            physical_cluster=cluster))
    assert [j['seed'] for j in jobs]==[0,2,3,6,7]
    assert all(j['replacement']['query']==3 for j in jobs)
    primary={}
    for job in jobs:
        job['physical_primary']=primary.setdefault(job['physical_cluster'],job['id'])
    sources=dict(old['sources_sha256'])
    for path in list(Path(__file__).parent.glob('*.py'))+[
            ROOT/'code/experiment_tools/run_matched_catalyst_control.py',
            ROOT/'code/analysis/analyze_matched_catalyst_control.py',
            ROOT/'code/analysis/check_matched_catalyst_control_native.py',DESIGN]:
        sources[path.relative_to(ROOT).as_posix()]=sha(path)
    protected=dict(old['protected_sha256'])
    for directory in (SOURCE/'runs',SOURCE/'cache',original.REPORT):
        for path in directory.rglob('*'):
            if path.is_file() and path.suffix not in ('.lock','.tmp'):
                protected[path.relative_to(ROOT).as_posix()]=sha(path)
    protected[(SOURCE/'protocol.json').relative_to(ROOT).as_posix()]=sha(SOURCE/'protocol.json')
    p=dict(variant=VARIANT,registered_utc=datetime.now(timezone.utc).isoformat(),
        jobs=jobs,exclusions=exclusions,parameters=P,GP_parameters=old['GP_parameters'],
        jobs_count=len(jobs),physical_clusters=len(primary),total_queries=8,
        reused_prefix_observations=sum(j['prefix_count'] for j in jobs),
        new_policy_queries=sum(8-j['prefix_count'] for j in jobs),
        subsequent_LLM_randomness_removed=True,new_LLM_calls=0,
        original_actual_suffix_rule_verified=True,design=DESIGN.read_text(encoding='utf-8'),
        sources_sha256=sources,inputs_sha256=old['inputs_sha256'],protected_sha256=protected,
        prospective_outcomes_in_this_experiment_at_registration=0)
    p['protocol_sha256']=digest(p)
    write(BASE/'protocol.json',p)
    write(BASE/'evaluation_endpoint.json',read(SOURCE/'evaluation_endpoint.json'))
    (BASE/'cache').mkdir(exist_ok=True)
    copied=0
    for path in (SOURCE/'cache').iterdir():
        if path.suffix in ('.pt','.npz'):
            target=BASE/'cache'/path.name
            shutil.copyfile(path,target)
            assert sha(path)==sha(target)
            copied+=1
    write(BASE/'cache_copy.json',dict(files=copied,policy='exact observed prefix only'))
    REPORT.mkdir(parents=True,exist_ok=True)
    return check(True)


def initialize():
    global POOL,GP
    check()
    os.chdir(ROOT/'experiments/chemist_system')
    POOL=knowledge_gp.load_task(ROOT,'Reizman_Suzuki_Case_1',seed=0)
    GP=SharedGP(POOL,BASE/'cache')


def initial_state(job):
    return dict(job=job,history=[],decisions=[],notes=[],status='RUNNING')


def reuse_prefix(job,state,folder):
    source_folder=SOURCE/'runs'/job['source_job']
    source_state=read(source_folder/'state.json')
    for q in range(len(state['history'])+1,job['prefix_count']+1):
        old=read(source_folder/'rounds'/f'{q:02}_commit.json')
        after=read(source_folder/'rounds'/f'{q:02}_feedback.json')
        commit=dict(query=q,prestate_sha256=digest(state),history_sha256=digest(state['history']),
            protocol_sha256=read(BASE/'protocol.json')['protocol_sha256'],
            decision=deepcopy(old['decision']),selected_GP=old['selected_GP'],fit_key=old['fit_key'],
            notes_at_commit=old['notes_at_commit'],kind='reused_prefix',
            source_commit_path=(source_folder/'rounds'/f'{q:02}_commit.json').relative_to(ROOT).as_posix(),
            source_commit_sha256=sha(source_folder/'rounds'/f'{q:02}_commit.json'),
            reused_observation=True,new_oracle_query=False)
        commit['commit_sha256']=digest(commit)
        write(folder/'rounds'/f'{q:02}_commit.json',commit)
        state['history'].append(deepcopy(source_state['history'][q-1]))
        state['decisions'].append(commit)
        state['notes']=deepcopy(after['notes_after'])
        write(folder/'rounds'/f'{q:02}_feedback.json',dict(query=q,events=after['events'],
            notes_after=state['notes'],reused_prefix_feedback=True))
        write(folder/'state.json',state,replace=True)
    return state


def forecasts(notes,context,j):
    from scipy.special import ndtr
    gp_probability=float(ndtr((context['mu'][j]-context['incumbent'])/
                              np.sqrt(max(context['variance'][j],1e-18))))
    return [dict(id=n['id'],version=n['version'],incumbent_y=context['incumbent'],
        p_LLM=knowledge.forecast(n,context['incumbent']),p_GP=gp_probability,
        yield_mean=n['yield_mean'],yield_sd=n['yield_sd'],promoted_selection=False)
        for n in notes if knowledge.scope_mask(POOL,n['scope'])[j]]


def run_job(job):
    folder=BASE/'runs'/job['id']
    path=folder/'state.json'
    state=read(path) if path.exists() else initial_state(job)
    if len(state['history'])<job['prefix_count']:
        state=reuse_prefix(job,state,folder)
    while len(state['history'])<8:
        q=len(state['history'])+1
        cp=folder/'rounds'/f'{q:02}_commit.json'
        if cp.exists():
            c=read(cp)
        else:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                context=GP.context(state['history'],q)
            baseline=choose(context['logei'],context['available'],context['priorities'])
            is_override=q==job['replacement']['query']
            if is_override:
                pair,mask=matched_choice(POOL,state['history'],context)
                assert pair is not None and pair['selected_id']==job['replacement']['selected_id']
                j=POOL.ix[pair['selected_id']]
            else:
                pair,mask=None,np.zeros(len(POOL.ids),dtype=bool)
                j=baseline
            array=folder/'rounds'/f'{q:02}_GP.npz'
            array.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(array,mu=context['mu'],variance=context['variance'],logei=context['logei'],
                available=context['available'],priorities=context['priorities'],matched_mask=mask)
            gap=float(context['logei'][baseline]-context['logei'][j])
            decision=dict(query=q,selected_id=POOL.ids[j],BO_reference_id=POOL.ids[baseline],
                rule='single_matched_catalyst_override' if is_override else 'frozen_native_BO_suffix',
                design_override=is_override,knowledge_used=False,anchor_id=pair['anchor_id'] if pair else None,
                original_replaced_id=job['replacement']['original_selected_id'] if is_override else None,
                actual_logei_gap=gap,retained_exp_qLogEI=float(np.exp(-gap)),
                original_budget_epsilon=job['replacement']['original_epsilon'] if is_override else None,
                passes_original_budget=bool(gap<=job['replacement']['original_epsilon']) if is_override else None,
                forecasts=forecasts(state['notes'],context,j),
                checkpoint=context['checkpoint'].relative_to(ROOT).as_posix(),
                checkpoint_sha256=sha(context['checkpoint']),GP_arrays_sha256=sha(array))
            c=dict(query=q,prestate_sha256=digest(state),history_sha256=digest(state['history']),
                protocol_sha256=read(BASE/'protocol.json')['protocol_sha256'],kind='new_choice',
                decision=decision,selected_GP=dict(mean=float(context['mu'][j])*100,
                    sd=float(context['variance'][j]**.5)*100,qLogEI=float(context['logei'][j]),
                    hyperparameters_fitted=context['hyperparameters_fitted']),
                fit_key=context['key'],notes_at_commit=deepcopy(state['notes']),
                native_warnings=[str(w.message) for w in caught],LLM=None)
            c['commit_sha256']=digest(c)
            write(cp,c)
        assert c['prestate_sha256']==digest(state)
        assert c['history_sha256']==digest(state['history'])
        assert c['commit_sha256']==digest({k:v for k,v in c.items() if k!='commit_sha256'})
        assert c==read(cp)
        cid=c['decision']['selected_id']
        assert cid not in {h['candidate_id'] for h in state['history']}
        # New outcome access occurs only after this round's immutable commitment.
        observed=knowledge_gp.read_observation(ROOT,POOL,cid)
        notes,events=knowledge.feedback(c['notes_at_commit'],c['decision']['forecasts'],observed,q)
        state['history'].append(observed)
        state['decisions'].append(c)
        state['notes']=notes
        state['status']='COMPLETE' if q==8 else 'RUNNING'
        write(folder/'rounds'/f'{q:02}_feedback.json',dict(query=q,events=events,notes_after=notes,
            diagnostic_only_KF=True,LLM_refresh_disabled_for_matched_suffix=True))
        write(path,state,replace=True)
        assert sum(knowledge_gp.COMPILE_COUNTS.values())==0
        write(folder/'runtime.json',dict(operator_compiles_in_loop=0,**GP.counters),replace=True)
        print(json.dumps(dict(job=job['id'],query=q,yield_percent=observed['raw_value'],
            best=max(h['raw_value'] for h in state['history']),override=c['decision']['design_override'])),flush=True)
    return dict(job=job['id'],status='COMPLETE',queries=8)


def preflight():
    p=check(True)
    initialize()
    tested=[]
    for job in p['jobs']:
        source=read(SOURCE/'runs'/job['source_job']/'state.json')
        q=job['replacement']['query']
        context=GP.context(source['history'][:q-1],q)
        with np.load(SOURCE/'runs'/job['source_job']/'rounds'/f'{q:02}_GP.npz') as z:
            for key in ('mu','variance','logei','available','priorities'):
                np.testing.assert_array_equal(context[key],z[key])
        pair,mask=matched_choice(POOL,source['history'][:q-1],context)
        assert pair['selected_id']==job['replacement']['selected_id']
        assert pair['anchor_id']==job['replacement']['anchor_id']
        assert int(mask.sum())==7
        # Candidate storage order and identifier order cannot choose the match.
        order=np.arange(len(POOL.ids))[::-1]
        j=choose(context['logei'][order],mask[order],context['priorities'][order])
        assert POOL.ids[order[j]]==pair['selected_id']
        tested.append(dict(job=job['id'],exact_prefix_and_GP=True,exact_physical_match=True,
            selection_ignores_candidate_storage_order=True,paired_candidates=7,
            original_acquisition_gate_pass=job['replacement']['native_logei_gap']<=job['replacement']['original_epsilon']))
    assert GP.counters['fits']==0
    result=dict(status='PASS',checks=tested,new_GP_fits=0,new_oracle_queries=0,new_LLM_calls=0,
        protocol_sha256=p['protocol_sha256'],operator_compiles_in_loop=0)
    write(BASE/'preflight.json',result)
    check(True)
    return result


def batch(workers):
    p=check(True)
    assert read(BASE/'preflight.json')['status']=='PASS'
    jobs=sorted(p['jobs'],key=lambda j:(j['physical_primary']!=j['id'],j['seed']))
    results,errors=[],[]
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize) as executor:
        fs={executor.submit(run_job,j):j for j in jobs}
        for f in as_completed(fs):
            try:
                results.append(f.result())
            except Exception as error:
                errors.append(dict(job=fs[f]['id'],type=type(error).__name__,error=str(error)))
    check(True)
    result=dict(status='FAILED' if errors else 'COMPLETE',results=results,errors=errors)
    write(BASE/'batch_receipt.json',result,replace=True)
    assert not errors,errors
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('prepare','preflight','run'))
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    result=prepare() if args.action=='prepare' else preflight() if args.action=='preflight' else batch(args.workers)
    if args.action=='prepare':
        result={k:result[k] for k in ('jobs_count','physical_clusters','new_policy_queries','protocol_sha256')}
    print(json.dumps(result))
