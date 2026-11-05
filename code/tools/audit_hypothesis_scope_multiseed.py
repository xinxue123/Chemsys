"""Replay seeded decisions/GP arrays, sealed refresh contexts and KF states."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
import json
import numpy as np
import pandas as pd
from hypothesis_scope_multiseed import study as s


def audit_one(task,seed,arm):
    p=s.check();pool=s.frozen.pool_for(task);folder=s.folder_for(task,seed,arm);final=s.read(folder/'state.json')
    state=s.initial_state(task,seed,arm,p);scopes=s.knowledge.Scopes(pool)
    gp=s.SeededGP(pool,s.BASE/'cache'/s.SHORT[task],tie_seed=seed);rows=[];maxerr=0.
    for q in range(1,21):
        c=s.read(folder/'rounds'/f'{q:02d}_commit.json');d=c['decision'];f=s.read(folder/'rounds'/f'{q:02d}_feedback.json')
        assert c['prestate_sha256']==s.digest(state)
        assert c['commit_sha256']==s.digest({a:b for a,b in c.items() if a!='commit_sha256'}) and f['commit_sha256']==c['commit_sha256']
        inc=max((h['raw_value'] for h in state['history']),default=None);work=deepcopy(state)
        if arm=='hyp':
            work['notes'],events=s.knowledge.pause_for_recommendation(work['notes'],inc);work['permission_events']+=events
            available=~np.isin(pool.ids,[h['candidate_id'] for h in state['history']])
            event=s.policy.event(work,scopes,available,inc,q);assert event==d['refresh_event']
            if event['call']:
                call=c['working_knowledge']['calls'][-1]
                if q==1:
                    saved=s.initial_prior(task);assert call['request']==saved['request']
                else:
                    rp=ROOT/call['request'];saved=s.read(rp.parent.parent/'validated.json');context=s.read(rp.parent.parent/'context.json')
                    assert context['public']==s.prompts.public(pool,work,event,s.frozen.locked_first(task)) and context['system']==s.prompts.SYSTEM
                    assert len(context['public']['own_observations'])==q-1
                plan=s.prompts.validate(saved['raw'],pool,work,s.frozen.locked_first(task));assert plan==saved['plan']
                previous=deepcopy(work['notes']);work['notes'],edits=s.knowledge.install(plan['drafts'],previous,work['archive'],scopes,q,event['stagnated'])
                work['archive']+=previous
                work['calls'].append(dict(query=q,event=event,request=saved['request'],preferred=plan['preferred'],projection=plan['projection'],edits=edits))
                work['notes'],events=s.knowledge.pause_for_recommendation(work['notes'],inc);work['permission_events']+=events
            assert {a:work[a] for a in ('notes','archive','calls','permission_events')}==c['working_knowledge']
        ctx=None;causes={}
        if q==1:
            j=int(np.argmin(s.frozen.native.semantic_priorities(pool,1,seed))) if arm=='bo' else pool.ix[s.frozen.locked_first(task)['candidate_id']]
            assert not d['GP_fitted'] and not d['qLogEI_defined']
        else:
            ctx=gp.context(state['history'],q);ap=folder/'rounds'/f'{q:02d}_GP.npz'
            assert s.sha(ap)==c['GP_arrays_sha256'] and ctx['stats']['fit_parameters']==c['GP_parameters']
            j=ctx['bo_index']
            if arm=='hyp':
                choice=s.policy.select(ctx,scopes,work['notes'],inc);j=choice['index'];causes=choice['causes']
                assert d['reason']==choice['reason'] and d['unrestricted_BO_id']==pool.ids[ctx['bo_index']]
                assert d['scope_count']==int(choice['mask'].sum()) and d['causal_diagnostics']==choice['causal_diagnostics']
            with np.load(ap) as a:
                for key in ('mu_y','sd_y','sd_predictive_y','base_logei'):
                    target=ctx[key] if key=='base_logei' else ctx['stats'][key]
                    error=float(np.max(abs(a[key]-target)));maxerr=max(error,maxerr)
                    assert np.allclose(a[key],target,rtol=1e-10,atol=1e-10)
                for key in ('available','tie_priority'):np.testing.assert_array_equal(a[key],ctx[key])
                if arm=='hyp':
                    np.testing.assert_array_equal(a['final_mask'],choice['mask'])
                    for n in work['notes']:np.testing.assert_array_equal(a['scope_'+n['semantic_key']],scopes.mask(n['scope']))
            assert gp.counters['operator_compiles_in_loop']==0
        assert d['selected_id']==pool.ids[j] and d['conditions']==pool.points[j]
        obs=f['observation'];assert obs['candidate_id']==pool.ids[j] and obs==s.frozen.native.read_observation(ROOT,pool,pool.ids[j])
        feedback=[]
        if arm=='hyp':
            forecasts=s.knowledge.freeze_forecasts(pool,scopes,work['notes'],j,q,inc,ctx,causes);assert forecasts==d['locked_forecasts']
            notes,feedback=s.knowledge.feedback(work['notes'],forecasts,obs,q)
            assert feedback==f['knowledge_feedback'] and notes==f['notes_after_feedback']
            state.update(deepcopy(c['working_knowledge']));state['notes']=notes
        state['history'].append(obs);row=dict(query=q,**obs,decision=d,best=max(h['raw_value'] for h in state['history']))
        if arm=='hyp':row['knowledge_feedback']=feedback
        state['decisions'].append(row);state['status']='COMPLETE' if q==20 else 'RUNNING'
        rows.append(dict(task=s.SHORT[task],seed=seed,arm=arm,query=q,replayed=True,
            KF_updates=sum(e['KF'] is not None for e in feedback),changed_from_BO=d.get('changed_from_unrestricted_BO',False)))
        gp.release()
    assert state==final and len({h['candidate_id'] for h in state['history']})==20
    assert gp.counters.get('fits',0)==0,'Audit unexpectedly fitted a new GP'
    if arm!='bo':assert final['history'][0]==s.read(s.frozen.first.folder_for(task)/'state.json')['history'][0]
    return dict(task=s.SHORT[task],seed=seed,arm=arm,rows=rows,maximum_numeric_error=maxerr)


def receipts():
    rows=[];agents=set()
    for rp in sorted((s.BASE/'requests').glob('*/*/attempt*/request.json')):
        af=rp.parent;req=s.read(rp);rec=s.read(af/'receipt.json');assignment=s.read(af/'assignment.json');done=s.read(af/'completion.json')
        inv=s.read(af/'invocation.json');export=s.read(af/'export.json');agent=assignment['agent_id']
        assert req['request_sha256']==s.digest({a:b for a,b in req.items() if a!='request_sha256'})
        assert agent not in agents and agent=='/root/'+inv['task_name'];agents.add(agent)
        assert assignment['spawn_result']==dict(task_name=agent) and assignment['spawn_request']==inv
        assert rec['request_sha256']==req['request_sha256'] and rec['assignment_sha256']==s.sha(af/'assignment.json')
        assert rec['raw_sha256']==s.sha(af/'response.raw.json') and rec['completion_sha256']==s.sha(af/'completion.json')
        assert done['agent_id']==agent and done['output_sha256']==s.sha(ROOT/export['output'])
        assert s.read(ROOT/export['output'])==s.read(af/'response.raw.json') and s.sha(ROOT/export['input'])==export['input_sha256']
        assert inv['fork_turns']=='none' and inv['model']=='gpt-6.1-terra' and inv['reasoning_effort']=='medium'
        public=s.read(ROOT/export['input'])['public'];public=public.get('original',public)
        assert public['experimental_budget']['observed']==len(public['own_observations']) and len(public['own_observations'])>0
        rows.append(dict(agent_id=agent,task_seed=req['task'],stage=req['stage'],attempt=req['attempt'],observed=len(public['own_observations']),
            receipt_verified=True,completed_before_seal=True,request=rp.relative_to(ROOT).as_posix()))
    return rows


def main():
    p=s.check(True);records=[];diagnostics=[]
    with ProcessPoolExecutor(max_workers=8,initializer=s.initialize) as executor:
        for future in as_completed([executor.submit(audit_one,*job) for job in s.JOBS]):
            value=future.result();records+=value.pop('rows');diagnostics.append(value);print(json.dumps(value),flush=True)
    llm=receipts();s.check(True)
    pd.DataFrame(records).sort_values(['seed','task','arm','query']).to_csv(s.REPORT/'trajectory_audit.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(llm).to_csv(s.REPORT/'LLM_receipts.csv',index=False,encoding='utf-8-sig')
    value=dict(status='PASS_MULTISEED_AUDIT',new_trajectories=len(diagnostics),new_queries_replayed=len(records),
        maximum_numeric_error=max(r['maximum_numeric_error'] for r in diagnostics),actual_new_LLM_attempts=len(llm),
        reused_frozen_initial_priors=24,all_refresh_contexts_match_own_complete_prefix=True,all_GP_replays_cache_only=True,
        no_operator_compilation_in_loop=True,protected_artifacts=len(p['protected_sha256']),
        seed0_status='Imported unchanged; prior export-custody incident remains disclosed in original audit',
        protocol_sha256=p['protocol_sha256'],details=diagnostics)
    s.write(s.REPORT/'audit.json',value,replace=True);print(json.dumps({a:b for a,b in value.items() if a!='details'}))


if __name__=='__main__':main()
