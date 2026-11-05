"""Paired zero-data start, explicit scope restrictions and feedback-led refresh."""
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
from grounded_first_start import study as first
from chemical_hypothesis_bo.hypotheses import digest
from . import VARIANT,PARAMETERS as P,MODEL,EFFORT,knowledge as k,policy,prompts,transport
ROOT=first.ROOT;BASE=ROOT/'experiments/runs'/VARIANT;REPORT=ROOT/'experiments/reports'/VARIANT
SHORT=first.SHORT;read,write,sha=first.read,first.write,first.sha
pool_for=first.pool_for


def locked_first(task):
    d=read(first.folder_for(task)/'state.json')['decisions'][0]['decision']
    # Whitelist only the zero-data input and its public explanation; never the
    # feedback, later choices or the global-optimum post-hoc comparison files.
    return dict(candidate_id=d['selected_id'],conditions=deepcopy(d['conditions']),
        rationale=d['rationale'],assumptions=deepcopy(d['key_assumptions']))


def check(protected=False):
    p=read(BASE/'protocol.json')
    assert p['protocol_sha256']==digest({a:b for a,b in p.items() if a!='protocol_sha256'})
    for group in ('sources_sha256','inputs_sha256')+(('protected_sha256',) if protected else ()):
        for name,h in p[group].items():
            if sha(ROOT/name)!=h:raise ValueError('Registered artifact changed: '+name)
    return p


def prepare():
    if (BASE/'protocol.json').exists():return check(True)
    assert not (BASE/'runs').exists() and not (BASE/'requests').exists()
    previous=first.check(True);qa=read(BASE/'preflight.json')
    assert qa['status']=='PASS_SCOPE_LOOP_PREFLIGHT'
    files=list(Path(__file__).parent.glob('*.py'))+[ROOT/'code/experiment_tools/run_hypothesis_scope_loop.py',ROOT/'docs/hypothesis_scope_loop_20261105.txt']
    protected=dict(previous['protected_sha256']);protected.update(previous['sources_sha256'])
    for folder in (first.BASE/'runs',first.BASE/'requests'):
        for path in folder.rglob('*.json'):protected[path.relative_to(ROOT).as_posix()]=sha(path)
    protected[(first.BASE/'protocol.json').relative_to(ROOT).as_posix()]=sha(first.BASE/'protocol.json')
    sources={path.relative_to(ROOT).as_posix():sha(path) for path in files}
    for name in ('code/experiment_tools/grounded_first_start/proposal.py','code/experiment_tools/grounded_first_start/study.py',
        'code/experiment_tools/chemical_hypothesis_bo/hypotheses.py','code/experiment_tools/chemical_hypothesis_bo/background.py',
        'code/experiment_tools/reasoning_memory_v13/calibration.py','code/experiment_tools/cold_start_v13/gp.py'):
        sources[name]=sha(ROOT/name)
    controls={}
    for task in native.TASKS:
        paths=dict(pure_BO=ROOT/previous['controls'][task]['path'],LLM_first_then_BO=first.folder_for(task)/'state.json')
        controls[task]={key:dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path)) for key,path in paths.items()}
    p=dict(variant=VARIANT,registered_UTC=datetime.now(timezone.utc).isoformat(),parameters=P,
        initial=0,queries=20,seed=0,tasks=list(native.TASKS),model=MODEL,effort=EFFORT,fork_turns='none',
        sources_sha256=sources,inputs_sha256=previous['inputs_sha256'],protected_sha256=protected,controls=controls,
        preflight_sha256=sha(BASE/'preflight.json'),maximum_logical_LLM_requests=32,maximum_actual_LLM_attempts=64,
        maximum_experimental_queries=160,paired_first_input_from_frozen_independent_LLM=True,
        new_initial_hypotheses_blind_to_first_outcome=True,current_GP_ranking_not_given_to_LLM=True,
        all_own_history_given_on_refresh=True,no_GP_mean_SD_or_acquisition_modification=True,
        hypothesis_constraints_have_no_global_qLogEI_gap_gate=True,development_not_held_out=True)
    p['protocol_sha256']=digest(p);write(BASE/'protocol.json',p)
    for path in files:
        target=BASE/'source_snapshot'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    return check(True)


def initialize():
    os.chdir(ROOT/'experiments/chemist_system');native.warm()


def folder_for(task):return BASE/'runs'/SHORT[task]


def apply_outcome(pool,folder,state,commit):
    assert commit['prestate_sha256']==digest(state)
    assert commit['commit_sha256']==digest({a:b for a,b in commit.items() if a!='commit_sha256'})
    q=commit['query'];d=commit['decision'];assert q==len(state['history'])+1
    assert d['selected_id'] not in {h['candidate_id'] for h in state['history']}
    observed=native.read_observation(ROOT,pool,d['selected_id'])
    notes,feedback=k.feedback(commit['working_knowledge']['notes'],d['locked_forecasts'],observed,q)
    write(folder/'rounds'/f'{q:02d}_feedback.json',dict(commit_sha256=commit['commit_sha256'],observation=observed,
        knowledge_feedback=feedback,notes_after_feedback=notes))
    state.update(deepcopy(commit['working_knowledge']));state['notes']=notes
    state['history'].append(observed)
    state['decisions'].append(dict(query=q,**observed,decision=d,knowledge_feedback=feedback,
        best=max(r['raw_value'] for r in state['history'])))
    state['status']='COMPLETE' if q==20 else 'RUNNING';write(folder/'state.json',state,replace=True)
    print(json.dumps(dict(task=SHORT[pool.task],query=q,value=observed['raw_value'],best=state['decisions'][-1]['best'],
        phase=d['reason'],scope_count=d.get('scope_count'),LLM_calls=len(state['calls']),refresh=d['refresh_event']['call'])),flush=True)


def run_one(task):
    p=check();pool=pool_for(task);folder=folder_for(task);folder.mkdir(parents=True,exist_ok=True);scopes=k.Scopes(pool)
    with FileLock(str(folder/'.worker.lock'),timeout=1):
        sf=folder/'state.json'
        state=read(sf) if sf.exists() else dict(task=task,protocol_sha256=p['protocol_sha256'],initial_ids=[],
            initial=0,queries=20,seed=0,history=[],decisions=[],notes=[],archive=[],calls=[],permission_events=[],status='RUNNING')
        assert state['protocol_sha256']==p['protocol_sha256']
        write(folder/'initial_commit.json',dict(task=task,protocol_sha256=p['protocol_sha256'],initial_ids=[],history=[]))
        gp=None
        while len(state['history'])<20:
            q=len(state['history'])+1;cp=folder/'rounds'/f'{q:02d}_commit.json'
            if cp.exists():commit=read(cp)
            else:
                work=deepcopy(state);inc=max((r['raw_value'] for r in state['history']),default=None)
                available=~np.isin(pool.ids,[r['candidate_id'] for r in state['history']])
                work['notes'],permission_events=k.pause_for_recommendation(work['notes'],inc)
                work['permission_events']+=permission_events
                event=policy.event(work,scopes,available,inc,q);proposal=None
                if event['call']:
                    public=prompts.public(pool,work,event,locked_first(task))
                    proposal=transport.request(SHORT[task],f'query{q:02d}',prompts.SYSTEM,public,
                        lambda raw:prompts.validate(raw,pool,work,locked_first(task)))
                    previous=deepcopy(work['notes'])
                    work['notes'],edits=k.install(proposal['plan']['drafts'],previous,work['archive'],scopes,q,event['stagnated'])
                    work['archive']+=previous
                    work['calls'].append(dict(query=q,event=event,request=proposal['request'],
                        preferred=proposal['plan']['preferred'],projection=proposal['plan']['projection'],edits=edits))
                    work['notes'],permission_events=k.pause_for_recommendation(work['notes'],inc)
                    work['permission_events']+=permission_events
                decision=dict(query=q,conditions=None,incumbent_before=inc,refresh_event=event,
                    active_hypothesis_ids=[],LLM_preferred=proposal['plan']['preferred'] if proposal else None,
                    LLM_projection=proposal['plan']['projection'] if proposal else None,
                    selection_is_native_qLogEI_after_first=True,global_qLogEI_gap_gate=False)
                gp_meta={};context=None;causes={}
                if q==1:
                    j=pool.ix[locked_first(task)['candidate_id']]
                    decision.update(reason='locked_independent_LLM_first_with_blinded_hypotheses',GP_fitted=False,
                        qLogEI_defined=False,scope_count=int(k.filter_mask(scopes,work['notes'],available,inc)[0].sum()),
                        active_hypothesis_ids=[n['id'] for n in work['notes'] if k.eligible(n,inc)])
                else:
                    if gp is None:gp=ColdGP(pool,BASE/'cache'/SHORT[task],seed=0)
                    context=gp.context(state['history'],q);choice=policy.select(context,scopes,work['notes'],inc)
                    j=choice['index'];causes=choice['causes'];stats=context['stats']
                    arrays={a:stats[a] for a in ('mu_y','sd_y','sd_predictive_y')}
                    arrays.update({a:context[a] for a in ('base_logei','available','tie_priority')})
                    arrays.update(final_mask=choice['mask'],proposed_scope_mask=choice['proposed_scope_mask'])
                    for n in work['notes']:arrays['scope_'+n['semantic_key']]=scopes.mask(n['scope'])
                    ap=folder/'rounds'/f'{q:02d}_GP.npz'
                    if not ap.exists():
                        with ap.open('xb') as f:np.savez_compressed(f,**arrays)
                    table=pd.DataFrame(dict(candidate_id=pool.ids,**{a:[point[a] for point in pool.points] for a in pool.axes},
                        GP_mean_raw=stats['mu_y']*pool.scale,GP_SD_raw=stats['sd_y']*pool.scale,
                        GP_predictive_SD_raw=stats['sd_predictive_y']*pool.scale,base_qLogEI=context['base_logei'],
                        unqueried=context['available'],inside_effective_scope=choice['mask'],
                        selected=np.arange(len(pool.ids))==j,unrestricted_BO=np.arange(len(pool.ids))==context['bo_index']))
                    for n in work['notes']:table['in_scope_'+n['id']]=scopes.mask(n['scope'])
                    table.to_csv(folder/'rounds'/f'{q:02d}_candidates.csv',index=False,encoding='utf-8-sig')
                    decision.update(reason=choice['reason'],GP_fitted=True,qLogEI_defined=True,
                        scope_count=int(choice['mask'].sum()),proposed_scope_count=int(choice['proposed_scope_mask'].sum()),
                        active_hypothesis_ids=[n['id'] for n in choice['active']],fallback=choice['fallback'],
                        unrestricted_BO_id=pool.ids[context['bo_index']],changed_from_unrestricted_BO=j!=context['bo_index'],
                        base_logEI=float(context['base_logei'][j]),qLogEI_opportunity_gap=choice['opportunity_gap'],
                        GP_mean_raw=float(stats['mu_y'][j]*pool.scale),GP_SD_raw=float(stats['sd_y'][j]*pool.scale),
                        GP_predictive_SD_raw=float(stats['sd_predictive_y'][j]*pool.scale),GP_stage=context['stage'],
                        causal_diagnostics=choice['causal_diagnostics'],fit_key=stats['fit_key'],
                        operator_compiles_in_loop=gp.counters['operator_compiles_in_loop'])
                    gp_meta=dict(GP_arrays_sha256=sha(ap),GP_parameters=stats['fit_parameters'],
                        acquisition=context['acquisition_meta'],output_scale_convention=stats.get('cold_scale_convention'))
                decision.update(selected_id=pool.ids[j],conditions=deepcopy(pool.points[j]),
                    locked_forecasts=k.freeze_forecasts(pool,scopes,work['notes'],j,q,inc,context,causes))
                commit=dict(query=q,prestate_sha256=digest(state),decision=decision,
                    working_knowledge={a:work[a] for a in ('notes','archive','calls','permission_events')},
                    phase_order=['own_history_only','permission_review','event_triggered_independent_LLM_if_needed',
                        'unchanged_native_GP_after_first','scope_filter','native_qLogEI_argmax','forecasts_locked','selection_committed_before_oracle'],**gp_meta)
                commit['commit_sha256']=digest(commit);write(cp,commit)
            apply_outcome(pool,folder,state,commit)
            if gp is not None:gp.release()
        return dict(task=SHORT[task],status='COMPLETE')


def report():
    p=check(True);rows=[];trace=[]
    for task in native.TASKS:
        for method,c in p['controls'][task].items():rows.append(first.metrics(read(ROOT/c['path']),task,method))
        state=read(folder_for(task)/'state.json');m=first.metrics(state,task,'hypothesis_scope_loop')
        rows.append(dict(m,LLM_calls=len(state['calls']),changed_from_native_BO=sum(r['decision'].get('changed_from_unrestricted_BO',False) for r in state['decisions']),
            scoped_queries=sum(r['decision']['reason']=='hypothesis_constrained_native_qLogEI' for r in state['decisions']),
            KF_updates=sum(e['KF'] is not None for r in state['decisions'] for e in r['knowledge_feedback'])))
        for r in state['decisions']:
            d=r['decision'];trace.append(dict(task=SHORT[task],query=r['query'],candidate_id=r['candidate_id'],objective_raw=r['raw_value'],
                best=r['best'],selector=d['reason'],scope_count=d['scope_count'],active_hypotheses=';'.join(d['active_hypothesis_ids']),
                changed_from_BO=d.get('changed_from_unrestricted_BO',False),qLogEI_gap=d.get('qLogEI_opportunity_gap'),
                refresh=d['refresh_event']['call'],event_reasons=';'.join(d['refresh_event']['reasons']),
                events=json.dumps(r['knowledge_feedback'],ensure_ascii=False)))
    REPORT.mkdir(parents=True,exist_ok=True);frame=pd.DataFrame(rows)
    frame.to_csv(REPORT/'results.csv',index=False,encoding='utf-8-sig');pd.DataFrame(trace).to_csv(REPORT/'decision_trace.csv',index=False,encoding='utf-8-sig')
    print(frame.to_string(index=False));return dict(status='REPORT_COMPLETE',rows=len(rows))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','preflight','run','pending','dispatch','report'))
    parser.add_argument('--workers',type=int,default=8);parser.add_argument('--spawned',nargs='*',default=[]);args=parser.parse_args()
    if args.action=='prepare':value=dict(protocol_sha256=prepare()['protocol_sha256'])
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
