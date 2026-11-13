"""Zero observations, a genuinely LLM-led first point, then19 native BO queries."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import argparse
import json
import os
import shutil
import numpy as np
import pandas as pd
from filelock import FileLock
from chem.benchmarks import knowledge_gp as native
from cold_start_v13.gp import ColdGP
from kernel_contrast_qlogei import study as old
from chemical_hypothesis_bo.hypotheses import digest
from . import VARIANT,PARAMETERS as P,MODEL,EFFORT,proposal,transport
ROOT=old.ROOT;BASE=ROOT/'experiments/runs'/VARIANT;REPORT=ROOT/'experiments/reports'/VARIANT
COLD=ROOT/'experiments/runs/cold0_20261102';SHORT=old.SHORT
read,write,sha=old.read,old.write,old.sha


def pool_for(task):
    pool=native.load_task(ROOT,task,seed=0);pool.initial_ids=();return pool


def check(protected=False):
    p=read(BASE/'protocol.json')
    assert p['protocol_sha256']==digest({k:v for k,v in p.items() if k!='protocol_sha256'})
    for group in ('sources_sha256','inputs_sha256')+(('protected_sha256',) if protected else ()):
        for name,h in p[group].items():
            if sha(ROOT/name)!=h:raise ValueError('Registered artifact changed: '+name)
    return p


def prepare():
    if (BASE/'protocol.json').exists():return check(True)
    assert not (BASE/'runs').exists() and not (BASE/'requests').exists()
    cold=read(COLD/'protocol.json')
    assert (cold['initial'],cold['queries'],cold['seed'])==(0,20,0)
    # Verify the entire historical source recipe, not just a summary table.
    for name,h in cold['sources_sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError('Cold BO source recipe changed: '+name)
    prior=read(ROOT/'experiments/runs/chemical_hypothesis_bo_doyle_identityfix_20261105/protocol.json')
    protected=dict(prior['protected_sha256']);protected.update(prior['sources_sha256'])
    protected.update(cold['sources_sha256']);protected[(COLD/'protocol.json').relative_to(ROOT).as_posix()]=sha(COLD/'protocol.json')
    for folder in (ROOT/'experiments/runs/chemical_hypothesis_bo_doyle_identityfix_20261105/runs',
        ROOT/'experiments/runs/chemical_hypothesis_bo_doyle_identityfix_20261105/requests'):
        for path in folder.rglob('*.json'):protected[path.relative_to(ROOT).as_posix()]=sha(path)
    controls={};cards=[]
    for task in native.TASKS:
        pool=pool_for(task);path=COLD/'runs'/(SHORT[task]+'_bo')/'state.json';state=read(path)
        assert state['status']=='COMPLETE' and len(state['history'])==20 and state['history'][0]['candidate_id']==pool.ids[int(np.argmin(native.semantic_priorities(pool,1,0)))]
        controls[task]=dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path))
        for item in path.parent.rglob('*.json'):protected[item.relative_to(ROOT).as_posix()]=sha(item)
        public=proposal.public(pool);write(REPORT/'initial_inputs'/f'{SHORT[task]}.json',public)
        cards.append(dict(task=task,public_sha256=digest(public),candidate_count=len(pool.ids),initial_ids=[]))
    files=list(Path(__file__).parent.glob('*.py'))+[ROOT/'code/experiment_tools/run_grounded_first_start.py',ROOT/'docs/grounded_first_start_20261105.txt']
    sources={p.relative_to(ROOT).as_posix():sha(p) for p in files}
    for name in ('code/experiment_tools/chemical_hypothesis_bo/background.py','code/experiment_tools/chemical_hypothesis_bo/hypotheses.py',
        'code/experiment_tools/cold_start_v13/gp.py','code/experiment_tools/cold_start_v13/__init__.py'):
        sources[name]=sha(ROOT/name)
    inputs=dict(cold['inputs_sha256'])
    p=dict(variant=VARIANT,registered_UTC=datetime.now(timezone.utc).isoformat(),parameters=P,
        initial=0,queries=20,seed=0,tasks=list(native.TASKS),model=MODEL,effort=EFFORT,fork_turns='none',
        controls=controls,backgrounds=cards,sources_sha256=sources,inputs_sha256=inputs,protected_sha256=protected,
        cold_GP_source_sha256=sha(ROOT/'code/experiment_tools/cold_start_v13/gp.py'),
        maximum_new_experimental_queries=160,maximum_logical_LLM_requests=16,maximum_actual_LLM_attempts=32,
        first_point_no_GP_or_incumbent=True,first_actual_candidate_explicitly_selected_by_LLM=True,
        input_projection_confirmation_required_unless_exact=True,following_rounds_exact_native_BO=True,
        historical_4plus16_results_preserved=True,development_not_held_out=True)
    p['protocol_sha256']=digest(p);write(BASE/'protocol.json',p)
    for source in files:
        target=BASE/'source_snapshot'/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    return check(True)


def initialize():
    os.chdir(ROOT/'experiments/chemist_system');native.warm()


def folder_for(task):return BASE/'runs'/SHORT[task]


def run_one(task):
    p=check();pool=pool_for(task);folder=folder_for(task);folder.mkdir(parents=True,exist_ok=True)
    with FileLock(str(folder/'.worker.lock'),timeout=1):
        sf=folder/'state.json'
        state=read(sf) if sf.exists() else dict(task=task,protocol_sha256=p['protocol_sha256'],initial_ids=[],
            initial=0,queries=20,seed=0,history=[],decisions=[],status='RUNNING')
        assert state['protocol_sha256']==p['protocol_sha256']
        if state['status']=='COMPLETE':return dict(task=SHORT[task],status='COMPLETE')
        write(folder/'initial_commit.json',dict(task=task,protocol_sha256=p['protocol_sha256'],initial_ids=[],history=[]))
        if not state['history']:
            cp=folder/'rounds/01_commit.json'
            if cp.exists():commit=read(cp)
            else:
                first=transport.request(SHORT[task],'propose',proposal.PROPOSE_SYSTEM,proposal.public(pool),lambda raw:proposal.validate_proposal(raw,pool))
                write(folder/'first_proposal_commit.json',first)
                confirmed=None;candidate=first['plan']['exact_candidate_id']
                if candidate is None:
                    confirmed=transport.request(SHORT[task],'confirm',proposal.CONFIRM_SYSTEM,proposal.confirmation(pool,first),
                        lambda raw:proposal.validate_confirmation(raw,pool,first['plan']['neighbors']))
                    candidate=confirmed['plan']['choice']['candidate_id']
                actual=confirmed['plan']['choice'] if confirmed else first['plan']['choice']
                match=next(row for row in first['plan']['neighbors'] if row['candidate_id']==candidate)
                decision=dict(query=1,selected_id=candidate,conditions=deepcopy(pool.points[pool.ix[candidate]]),
                    reason='independent_grounded_LLM_first_choice',GP_fitted=False,qLogEI_defined=False,
                    rationale=actual['rationale'],key_assumptions=actual['key_assumptions'],
                    forecast=actual['forecast'],forecast_unit=pool.unit,falsification=actual['falsification'],
                    initial_ideal_conditions=first['plan']['choice']['conditions'],
                    normalized_input_RMS_distance=match['normalized_input_RMS_distance'],
                    projection_confirmed_by_LLM=confirmed is not None,
                    proposal_request=first['request'],confirmation_request=confirmed['request'] if confirmed else None,
                    all_candidate_inputs_allowed=True,GP_prediction_or_rank_exposed=False)
                commit=dict(query=1,prestate_sha256=digest(state),decision=decision,
                    phase_order=['zero_data_background','LLM_preference_sealed','input_only_mapping',
                        'LLM_executable_confirmation_if_needed','first_choice_committed_before_observation'])
                commit['commit_sha256']=digest(commit);write(cp,commit)
            apply_outcome(pool,folder,state,commit)
        # No observation-conditioned GP is constructed before the LLM first
        # choice and the first durably committed real observation are complete.
        gp=ColdGP(pool,BASE/'cache'/SHORT[task],seed=0)
        while len(state['history'])<20:
            q=len(state['history'])+1;cp=folder/'rounds'/f'{q:02d}_commit.json'
            if cp.exists():commit=read(cp)
            else:
                context=gp.context(state['history'],q);j=context['bo_index']
                stats=context['stats'];ap=folder/'rounds'/f'{q:02d}_GP.npz'
                if not ap.exists():
                    with ap.open('xb') as f:np.savez_compressed(f,**{k:stats[k] for k in ('mu_y','sd_y','sd_predictive_y')},
                        **{k:context[k] for k in ('base_logei','available','tie_priority')})
                decision=dict(query=q,selected_id=pool.ids[j],conditions=deepcopy(pool.points[j]),reason='native_full_pool_qLogEI',
                    GP_fitted=True,qLogEI_defined=True,base_logEI=float(context['base_logei'][j]),
                    GP_mean_raw=float(stats['mu_y'][j]*pool.scale),GP_SD_raw=float(stats['sd_y'][j]*pool.scale),
                    GP_stage=context['stage'],observations_before=q-1,fit_key=stats['fit_key'],
                    operator_compiles_in_loop=gp.counters['operator_compiles_in_loop'])
                commit=dict(query=q,prestate_sha256=digest(state),decision=decision,GP_arrays_sha256=sha(ap),
                    GP_parameters=stats['fit_parameters'],acquisition=context['acquisition_meta'],
                    output_scale_convention=stats.get('cold_scale_convention'))
                commit['commit_sha256']=digest(commit);write(cp,commit)
            apply_outcome(pool,folder,state,commit);gp.release()
        return dict(task=SHORT[task],status='COMPLETE')


def apply_outcome(pool,folder,state,commit):
    assert commit['prestate_sha256']==digest(state)
    assert commit['commit_sha256']==digest({k:v for k,v in commit.items() if k!='commit_sha256'})
    q=commit['query'];d=commit['decision'];assert q==len(state['history'])+1
    assert d['selected_id'] not in {r['candidate_id'] for r in state['history']}
    observed=native.read_observation(ROOT,pool,d['selected_id'])
    write(folder/'rounds'/f'{q:02d}_feedback.json',dict(commit_sha256=commit['commit_sha256'],observation=observed))
    state['history'].append(observed)
    state['decisions'].append(dict(query=q,**observed,decision=d,best=max(r['raw_value'] for r in state['history'])))
    state['status']='COMPLETE' if q==20 else 'RUNNING';write(folder/'state.json',state,replace=True)
    print(json.dumps(dict(task=SHORT[pool.task],query=q,value=observed['raw_value'],best=state['decisions'][-1]['best'],phase=d['reason'])),flush=True)


def metrics(state,task,method):
    assert len(state['history'])==20 and state['status']=='COMPLETE'
    raw=np.asarray([r['raw_value'] for r in state['history']]);best=np.maximum.accumulate(raw)
    return dict(task=SHORT[task],method=method,SearchAUC=float(best.mean()),final_best=float(best[-1]),
        first_value=float(raw[0]),first_final_best_query=int(np.flatnonzero(raw==best[-1])[0])+1)


def report():
    p=check(True);rows=[];firsts=[]
    for task in native.TASKS:
        state=read(folder_for(task)/'state.json');bo=read(ROOT/p['controls'][task]['path']);pool=pool_for(task)
        b=metrics(bo,task,'pure_BO');m=metrics(state,task,'grounded_LLM_first_then_BO')
        rows+=[b,dict(m,delta_SearchAUC=m['SearchAUC']-b['SearchAUC'])]
        d=state['decisions'][0]['decision']
        firsts.append(dict(task=SHORT[task],candidate_id=d['selected_id'],actual_first_value=m['first_value'],
            BO_first_value=b['first_value'],conditions=json.dumps(d['conditions']),rationale=d['rationale'],
            forecast_mean_raw=d['forecast']['mean_raw'],forecast_SD_raw=d['forecast']['sd_raw'],
            unit=pool.unit,confirmed_after_projection=d['projection_confirmed_by_LLM'],
            projection_RMS=d['normalized_input_RMS_distance'],assumptions=json.dumps(d['key_assumptions']),
            proposal_request=d['proposal_request'],confirmation_request=d['confirmation_request']))
    frame=pd.DataFrame(rows);frame.to_csv(REPORT/'results.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(firsts).to_csv(REPORT/'first_choices.csv',index=False,encoding='utf-8-sig')
    text=['Grounded LLM first choice from zero observations','',
        'All tasks have0 initial observations and20 paid queries. The first actual experiment is independently chosen by a grounded LLM. '
        'It receives no GP prediction, acquisition ranking, incumbent or objective table. A nonexact physical proposal is mapped by input '
        'distance, then a second fresh LLM chooses the actual executable input and locks its forecast. All subsequent19 choices are exact native qLogEI.','',
        'The historical cold-start BO controls use the identical frozen GP, pools, seed and20-query budget. Their full source recipe '
        'and input hashes are verified. This isolates the first-choice change; it does not test KF, event refresh or the preceding4+16 mixed-expert policy. '
        'That earlier evidence remains frozen. SearchAUC averages20 post-query incumbents and includes the paid first point.','',
        'Task | BO SearchAUC | LLM-first SearchAUC | BO final | LLM-first final | BO first | LLM first']
    for task in SHORT.values():
        f=frame[frame.task==task].set_index('method');a=f.loc['grounded_LLM_first_then_BO'];b=f.loc['pure_BO']
        text.append(f'{task} | {b.SearchAUC:.6f} | {a.SearchAUC:.6f} | {b.final_best:.6f} | {a.final_best:.6f} | {b.first_value:.6f} | {a.first_value:.6f}')
    text+=['','A single seed and one independent LLM realization per task do not establish stability. Forecasts are subjective '
        'uncertain point estimates, not observations or calibrated probabilities, and are never inserted into GP training targets. '
        'First-choice proposals, confirmations, actual physical inputs, forecasts and all native posterior arrays are retained. '
        'The inputs include the verified MTBD annotation for Doyle and recovered Suzuki substrate/ligand identities.']
    (REPORT/'report.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(frame.to_string(index=False));return dict(status='REPORT_COMPLETE',rows=len(rows))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','preflight','run','pending','dispatch','report'))
    parser.add_argument('--workers',type=int,default=8);parser.add_argument('--spawned',nargs='*',default=[]);args=parser.parse_args()
    if args.action=='prepare':p=prepare();value=dict(protocol_sha256=p['protocol_sha256'],tasks=len(p['tasks']))
    elif args.action=='preflight':
        from .validation import preflight
        value=preflight()
    elif args.action=='run':
        check(True)
        with ProcessPoolExecutor(max_workers=min(8,args.workers),initializer=initialize) as executor:
            for future in as_completed([executor.submit(run_one,t) for t in native.TASKS]):print(json.dumps(future.result()),flush=True)
        value=dict(status='COMPLETE')
    elif args.action=='report':value=report()
    else:
        for item in transport.pending():
            name=item['invocation']['task_name']
            if name in args.spawned and not item['assigned']:transport.assign(item['request'],dict(task_name='/root/'+name))
            if (Path(item['request']).parent/'assignment.json').exists() and item['output_exists']:transport.seal(item['request'])
        value=[dict(name=r['invocation']['task_name'],assigned=r['assigned'],ready=r['output_exists'],export=r['export']) for r in transport.pending()]
    print(json.dumps(value,ensure_ascii=True))


if __name__=='__main__':main()
