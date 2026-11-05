"""Replay the frozen study; no new experimental choices or LLM calls."""
from pathlib import Path
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
import sys
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from chemical_hypothesis_bo import study as s,experts,policy,prompts,hypotheses as h,transport


def one(task):
    p=s.check();pool=s.native.load_task(ROOT,task,seed=0);out=[]
    bo=s.read(ROOT/p['controls'][task]['path'])
    for arm in ('fixed','adaptive'):
        folder=s.folder_for(task,arm);state=s.read(folder/'state.json')
        assert state['status']=='COMPLETE' and len(state['history'])==20 and len(state['decisions'])==16
        initial=s.read(folder/'initial_commit.json')
        assert initial['commit_sha256']==h.digest({k:v for k,v in initial.items() if k!='commit_sha256'})
        assert initial['initial_ids']==list(pool.initial_ids)
        replay=dict(task=task,arm=arm,protocol_sha256=p['protocol_sha256'],status='RUNNING',
            history=deepcopy(state['history'][:4]),notes=[],archive=[],decisions=[],calls=[],ledger=[],
            log_weights=dict(base=0.),initial_notes=None,initial_log_weights=None)
        assert replay['history']==bo['history'][:4]
        gp=s.native.GPAdapter(pool,s.BASE/'cache'/s.SHORT[task],seed=0)
        max_error=0.;changes=credit_changes=informative=weight_updates=0;mixture_mass=[]
        for q in range(1,17):
            cp=s.read(folder/'rounds'/f'{q:02d}_commit.json');d=cp['decision']
            assert cp['commit_sha256']==h.digest({k:v for k,v in cp.items() if k!='commit_sha256'})
            assert cp['prestate_sha256']==h.digest(replay)
            context=gp.context(replay['history'],q)
            assert cp['native_state_sha256']==s.previous.models.state_hash(context['stats']['model'])
            event=policy.refresh_event(pool,replay,context,q);assert cp['refresh']==event
            if cp['LLM_request']:
                af=(ROOT/cp['LLM_request']).parent;req=s.read(af/'request.json')
                supplied=req['public'] if req['attempt']==1 else req['public']['original']
                assert supplied==prompts.public(pool,replay['history'],replay['notes'],replay['ledger'],event,credit=replay['log_weights'])
                plan=s.read(af.parent/'validated.json')['plan']
                expected=policy.initial(plan['notes']) if not replay['calls'] else policy.edit_weights(replay['log_weights'],plan['notes'])
                assert cp['knowledge_state']['log_weights']==expected
            else:assert not event['allowed'] and cp['knowledge_state']['log_weights']==replay['log_weights']
            values=experts.evaluate(gp,context,cp['knowledge_state']['notes'],cp['knowledge_state']['log_weights'])
            j=policy.choose(context,values,q);assert pool.ids[j]==d['selected_id']
            assert pool.ids[context['bo_index']]==d['BO_reference_id']
            assert d['selected_id'] not in {v['candidate_id'] for v in replay['history']}
            assert context['eligible'][j]
            assert d['native_logEI_gap']<=context['epsilon']+1e-12
            if q==16:assert not d['differs_from_BO']
            with np.load(folder/'rounds'/f'{q:02d}_arrays.npz',allow_pickle=False) as saved:
                for key in ('mu_y','sd_y','sd_predictive_y'):np.testing.assert_allclose(saved[key],context['stats'][key],rtol=1e-12,atol=1e-12)
                for key in ('base_logei','available','eligible','tie_priority'):np.testing.assert_array_equal(saved[key],context[key])
                np.testing.assert_allclose(saved['expert_mu'],values['mu'],rtol=1e-12,atol=1e-12)
                np.testing.assert_allclose(saved['expert_logEI'],values['scores'],rtol=1e-12,atol=1e-12)
                np.testing.assert_allclose(saved['mixture_logEI'],values['mixture'],rtol=1e-12,atol=1e-12)
                assert saved['expert_ids'].tolist()==values['ids']
            if arm=='adaptive':
                prior=policy.initial(cp['knowledge_state']['notes'])
                equal=dict(values,mixture=logsumexp(values['scores']+np.asarray([prior[i] for i in values['ids']])[:,None],axis=0))
                eq=pool.ids[policy.choose(context,equal,q)]
                assert eq==d['equal_prior_shadow_id']
                assert d['credit_changes_choice']==bool(q>1 and eq!=d['selected_id'])
            for k,pred in enumerate(d['predictions']):
                assert pred['id']==values['ids'][k]
                assert abs(pred['mu_y']-values['mu'][k,j])<1e-12
            observed=s.native.read_observation(ROOT,pool,d['selected_id'])
            assert observed==state['history'][q+3]
            fb=policy.feedback(d,observed,pool.points[j]);saved_fb=s.read(folder/'rounds'/f'{q:02d}_feedback.json')
            assert saved_fb==dict(commit_sha256=cp['commit_sha256'],observation=observed,feedback=fb)
            previous_best=max(v['raw_value'] for v in replay['history'])
            replay.update(deepcopy(cp['knowledge_state']));replay['log_weights']=fb['log_weights_after']
            replay['ledger']+=fb['ledger'];replay['history'].append(observed)
            replay['decisions'].append(dict(query=q,**observed,decision=d,incumbent_before=previous_best,
                incumbent_after=max(previous_best,observed['raw_value']),feedback=fb,refresh=event))
            replay['status']='COMPLETE' if q==16 else 'RUNNING'
            max_error=max(max_error,values['native_replay_error'])
            changes+=d['differs_from_BO'];credit_changes+=d['credit_changes_choice']
            informative+=sum(e['informative'] for e in fb['ledger'])
            weight_updates+=fb['log_weights_after']!=d['log_weights_before']
            mixture_mass.append(1-math.exp(d['log_weights_before']['base']))
            assert cp['counters']['operator_compiles_in_loop']==0
            gp.release()
        assert replay==state
        out.append(dict(task=s.SHORT[task],arm=arm,rounds=16,decisions_replayed=True,initial_conditions_matched=True,
            maximum_native_logEI_replay_error=max_error,interventions=changes,credit_changes_choice=credit_changes,
            informative_expert_forecasts=informative,weight_update_rounds=weight_updates,
            mean_knowledge_mass=float(np.mean(mixture_mass)),final_knowledge_mass=1-math.exp(state['log_weights']['base']),
            LLM_call_queries=','.join(map(str,state['calls'])),operator_compiles_in_loop=0))
    fixed=s.read(s.folder_for(task,'fixed')/'state.json');adaptive=s.read(s.folder_for(task,'adaptive')/'state.json')
    assert fixed['initial_notes']==adaptive['initial_notes'] and fixed['initial_log_weights']==adaptive['initial_log_weights']
    assert fixed['decisions'][0]['candidate_id']==adaptive['decisions'][0]['candidate_id']
    assert s.read(s.folder_for(task,'fixed')/'rounds/01_commit.json')['LLM_request']==s.read(s.folder_for(task,'adaptive')/'rounds/01_commit.json')['LLM_request']
    return out


def main():
    p=s.check(True);assert not transport.pending()
    receipts=[];agents=set();logical=set()
    for path in sorted((s.BASE/'requests').glob('*/*/attempt*/response_receipt.json')):
        af=path.parent;receipt=s.read(path);req=s.read(af/'request.json');assignment=s.read(af/'assignment.json');export=s.read(af/'export.json')
        assert receipt['raw_sha256']==s.sha(af/'response.raw.json')
        assert receipt['assignment_sha256']==s.sha(af/'assignment.json')
        assert receipt['request_sha256']==req['request_sha256']
        assert req['request_sha256']==h.digest({k:v for k,v in req.items() if k!='request_sha256'})
        assert assignment['spawn_request']==s.read(af/'invocation.json')
        agent=assignment['agent_id'];assert agent not in agents;agents.add(agent)
        assert assignment['spawn_result']==dict(task_name=agent)
        assert assignment['spawn_request']['fork_turns']=='none'
        assert assignment['spawn_request']['model']=='gpt-6.1-terra' and assignment['spawn_request']['reasoning_effort']=='medium'
        assert s.sha(ROOT/export['input'])==export['input_sha256']
        assert s.read(ROOT/export['output'])==s.read(af/'response.raw.json')
        assert receipt['usage_available'] is False
        logical.add((req['task'],req['context_sha256']))
        receipts.append(dict(task=req['task'],kind=req['kind'],query=req['query'],attempt=req['attempt'],
            agent=agent,request_sha256=req['request_sha256'],raw_sha256=receipt['raw_sha256']))
    assert len(logical)<=32
    with ProcessPoolExecutor(max_workers=8,initializer=s.initialize) as executor:
        rows=[row for group in executor.map(one,s.native.TASKS) for row in group]
    pd.DataFrame(rows).to_csv(s.REPORT/'audit_trajectories.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(receipts).to_csv(s.REPORT/'LLM_receipts.csv',index=False,encoding='utf-8-sig')
    verified=s.check(True)
    value=dict(status='PASS_REPLAY_AND_CUSTODY',protocol_sha256=p['protocol_sha256'],trajectories=len(rows),
        rounds=sum(r['rounds'] for r in rows),fresh_agents=len(agents),logical_LLM_calls=len(logical),
        independent_initial_requests_shared=8,old_artifacts_hash_verified=len(verified['protected_sha256']),
        current_sources_hash_verified=len(verified['sources_sha256']),native_GP_unchanged=True,
        zero_shift_native_logEI_error=max(r['maximum_native_logEI_replay_error'] for r in rows),
        selection_committed_before_observation=True,all_observations_from_selected_candidates=True,
        oracle_data_in_LLM_packets=False,operator_compiles_in_loop=0,provider_attestation_available=False,
        token_usage_available=False,
        windows_operator_adaptation='Resolve both sides of junction containment in the dispatch operator only; frozen sources unchanged.',
        audit_script_sha256=s.sha(__file__),operator_script_sha256=s.sha(ROOT/'code/tools/chemical_hypothesis_operations.py'))
    s.write(s.REPORT/'audit.json',value,replace=True)
    print(json.dumps(value))


if __name__=='__main__':main()
