"""Reconstruct every actual decision, input prompt, native posterior and KF update."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
import json
import numpy as np
import pandas as pd
from hypothesis_scope_loop import study as s,knowledge as k,policy,prompts


def audit_one(task):
    p=s.check();pool=s.pool_for(task);folder=s.folder_for(task);scopes=k.Scopes(pool)
    final=s.read(folder/'state.json')
    state=dict(task=task,protocol_sha256=p['protocol_sha256'],initial_ids=[],initial=0,queries=20,seed=0,
        history=[],decisions=[],notes=[],archive=[],calls=[],permission_events=[],status='RUNNING')
    gp=s.ColdGP(pool,s.BASE/'cache'/s.SHORT[task],seed=0);rows=[];maxerr=0.;request_paths=[]
    for q in range(1,21):
        c=s.read(folder/'rounds'/f'{q:02d}_commit.json');d=c['decision']
        assert c['prestate_sha256']==s.digest(state)
        assert c['commit_sha256']==s.digest({a:b for a,b in c.items() if a!='commit_sha256'})
        work=deepcopy(state);inc=max((h['raw_value'] for h in state['history']),default=None)
        available=~np.isin(pool.ids,[h['candidate_id'] for h in state['history']])
        work['notes'],events=k.pause_for_recommendation(work['notes'],inc);work['permission_events']+=events
        event=policy.event(work,scopes,available,inc,q);assert event==d['refresh_event']
        if event['call']:
            call=c['working_knowledge']['calls'][-1];rp=ROOT/call['request'];request_paths.append(str(rp))
            context=s.read(rp.parent.parent/'context.json')
            expected=prompts.public(pool,work,event,s.locked_first(task))
            assert context['public']==expected and context['system']==prompts.SYSTEM
            assert len(expected['own_observations'])==q-1 and not expected['current_GP_predictions_or_ranking_supplied']
            saved=s.read(rp.parent.parent/'validated.json')
            plan=prompts.validate(saved['raw'],pool,work,s.locked_first(task));assert plan==saved['plan']
            previous=deepcopy(work['notes'])
            work['notes'],edits=k.install(plan['drafts'],previous,work['archive'],scopes,q,event['stagnated'])
            work['archive']+=previous
            work['calls'].append(dict(query=q,event=event,request=saved['request'],preferred=plan['preferred'],projection=plan['projection'],edits=edits))
            work['notes'],events=k.pause_for_recommendation(work['notes'],inc);work['permission_events']+=events
        assert {a:work[a] for a in ('notes','archive','calls','permission_events')}==c['working_knowledge']
        causes={};context=None
        if q==1:
            j=pool.ix[s.locked_first(task)['candidate_id']]
            assert not d['GP_fitted'] and not d['qLogEI_defined']
        else:
            context=gp.context(state['history'],q);choice=policy.select(context,scopes,work['notes'],inc)
            j=choice['index'];ap=folder/'rounds'/f'{q:02d}_GP.npz';assert s.sha(ap)==c['GP_arrays_sha256']
            with np.load(ap) as arrays:
                for a in ('mu_y','sd_y','sd_predictive_y'):
                    err=float(np.max(abs(arrays[a]-context['stats'][a])));maxerr=max(maxerr,err)
                    assert np.allclose(arrays[a],context['stats'][a],rtol=1e-10,atol=1e-10)
                err=float(np.max(abs(arrays['base_logei']-context['base_logei'])));maxerr=max(maxerr,err)
                assert np.allclose(arrays['base_logei'],context['base_logei'],rtol=1e-10,atol=1e-10)
                for a in ('available','tie_priority'):np.testing.assert_array_equal(arrays[a],context[a])
                np.testing.assert_array_equal(arrays['final_mask'],choice['mask'])
                np.testing.assert_array_equal(arrays['proposed_scope_mask'],choice['proposed_scope_mask'])
                for note in work['notes']:np.testing.assert_array_equal(arrays['scope_'+note['semantic_key']],scopes.mask(note['scope']))
            assert d['reason']==choice['reason'] and d['scope_count']==int(choice['mask'].sum())
            assert d['causal_diagnostics']==choice['causal_diagnostics']
            assert d['unrestricted_BO_id']==pool.ids[context['bo_index']]
            assert context['stats']['fit_parameters']==c['GP_parameters']
            assert gp.counters['operator_compiles_in_loop']==0;causes=choice['causes']
        assert d['selected_id']==pool.ids[j] and d['conditions']==pool.points[j]
        forecasts=k.freeze_forecasts(pool,scopes,work['notes'],j,q,inc,context,causes)
        assert forecasts==d['locked_forecasts']
        feedback=s.read(folder/'rounds'/f'{q:02d}_feedback.json');obs=feedback['observation']
        assert feedback['commit_sha256']==c['commit_sha256'] and obs['candidate_id']==pool.ids[j]
        notes,events=k.feedback(work['notes'],forecasts,obs,q)
        assert notes==feedback['notes_after_feedback'] and events==feedback['knowledge_feedback']
        assert obs==s.native.read_observation(ROOT,pool,obs['candidate_id'])
        state.update(deepcopy(c['working_knowledge']));state['notes']=notes;state['history'].append(obs)
        state['decisions'].append(dict(query=q,**obs,decision=d,knowledge_feedback=events,best=max(h['raw_value'] for h in state['history'])))
        state['status']='COMPLETE' if q==20 else 'RUNNING'
        rows.append(dict(task=s.SHORT[task],query=q,choice=pool.ids[j],replayed=True,refresh=event['call'],
            changed_from_BO=d.get('changed_from_unrestricted_BO',False),KF_updates=sum(e['KF'] is not None for e in events)))
        gp.release()
    assert state==final and len(set(r['candidate_id'] for r in state['history']))==20
    baseline=s.read(s.first.folder_for(task)/'state.json');assert state['history'][0]==baseline['history'][0]
    return dict(task=s.SHORT[task],rows=rows,maximum_numeric_error=maxerr,requests=request_paths)


def receipts():
    result=[];ids=set();incidents=[]
    for rp in sorted((s.BASE/'requests').glob('*/*/attempt*/request.json')):
        af=rp.parent;req=s.read(rp);receipt=s.read(af/'receipt.json');assignment=s.read(af/'assignment.json');export=s.read(af/'export.json')
        assert req['request_sha256']==s.digest({a:b for a,b in req.items() if a!='request_sha256'})
        assert receipt['request_sha256']==req['request_sha256'] and receipt['raw_sha256']==s.sha(af/'response.raw.json')
        assert receipt['assignment_sha256']==s.sha(af/'assignment.json') and s.sha(ROOT/export['input'])==export['input_sha256']
        agent=assignment['agent_id'];inv=s.read(af/'invocation.json')
        assert agent=='/root/'+inv['task_name'] and agent not in ids;ids.add(agent)
        assert assignment['spawn_request']==inv and assignment['spawn_result']==dict(task_name=agent)
        assert inv['model']=='gpt-6.1-terra' and inv['reasoning_effort']=='medium' and inv['fork_turns']=='none'
        scratch=s.read(ROOT/export['output']);sealed=s.read(af/'response.raw.json')
        export_unchanged=scratch==sealed
        if not export_unchanged:
            # The authoritative experimental response is the immutable sealed
            # raw file and receipt. Preserve and disclose scratch-file changes;
            # never replace the response retrospectively or erase the incident.
            task=next(t for t,v in s.SHORT.items() if v==req['task'])
            c=s.read(s.folder_for(task)/'rounds'/f"{int(req['stage'][5:]):02d}_commit.json")
            call=c['working_knowledge']['calls'][-1]
            assert scratch['hypotheses']==sealed['hypotheses']
            assert scratch['summary']==sealed['summary']
            sc=scratch['preferred_experiment'];se=sealed['preferred_experiment']
            assert sc['basis_ids']==se['basis_ids'] and sc['rationale']==se['rationale']
            changes={a:dict(sealed=se['conditions'][a],export=sc['conditions'][a],
                absolute_difference=abs(sc['conditions'][a]-se['conditions'][a])) for a in sc['conditions'] if sc['conditions'][a]!=se['conditions'][a]}
            assert all(v['absolute_difference']<1e-8 for v in changes.values())
            assert req['stage']!='query01' # Preferred prototypes do not select post-first points.
            incidents.append(dict(agent_id=agent,request=rp.relative_to(ROOT).as_posix(),changes=changes,
                authoritative_response='response.raw.json sealed in receipt before use',
                hypotheses_and_forecasts_unchanged=True,post_first_preferred_prototype_not_a_selection_input=True,
                actual_decision_and_KF_replay_unaffected=True,sealed_UTC=receipt['sealed_UTC'],
                export_mtime_ns=(ROOT/export['output']).stat().st_mtime_ns))
        public=s.read(ROOT/export['input'])['public']
        if req['attempt']>1:public=public['original']
        assert public['experimental_budget']['observed']==len(public['own_observations'])
        result.append(dict(agent_id=agent,task=req['task'],stage=req['stage'],attempt=req['attempt'],
            own_observations=len(public['own_observations']),request=rp.relative_to(ROOT).as_posix(),
            provider_usage_available=False,receipt_verified=True,agent_export_unchanged=export_unchanged))
    return result,incidents


def main():
    p=s.check(True);allrows=[];diagnostics=[]
    with ProcessPoolExecutor(max_workers=8,initializer=s.initialize) as executor:
        for future in as_completed([executor.submit(audit_one,t) for t in s.native.TASKS]):
            value=future.result();allrows+=value.pop('rows');diagnostics.append(value)
            print(json.dumps(dict(task=value['task'],status='REPLAY_PASS',maximum_numeric_error=value['maximum_numeric_error'])),flush=True)
    records,incidents=receipts();s.check(True)
    pd.DataFrame(allrows).sort_values(['task','query']).to_csv(s.REPORT/'trajectory_audit.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(records).to_csv(s.REPORT/'LLM_receipts.csv',index=False,encoding='utf-8-sig')
    value=dict(status='PASS_DECISION_REPLAY_WITH_EXPORT_INCIDENT' if incidents else 'PASS_SCOPE_LOOP_AUDIT',protocol_sha256=p['protocol_sha256'],actual_queries=len(allrows),
        actual_LLM_attempts=len(records),trajectories=8,maximum_numeric_error=max(v['maximum_numeric_error'] for v in diagnostics),
        all_refresh_prompts_match_complete_own_prefix=True,paired_first_observations_match=True,
        GP_parameters_predictions_and_qLogEI_unchanged=True,protected_artifacts=len(p['protected_sha256']),
        no_operator_compilation_in_loop=True,agent_export_revision_incidents=incidents,
        efficacy_or_stability_not_established_by_this_audit=True,details=diagnostics)
    s.write(s.REPORT/'audit.json',value,replace=True);print(json.dumps({a:b for a,b in value.items() if a!='details'}))


if __name__=='__main__':main()
