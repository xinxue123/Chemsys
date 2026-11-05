"""Replay the0+20 first-choice study and verify actual independent LLM receipts."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path
import os
import json
import sys
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
import numpy as np
import pandas as pd
from grounded_first_start import study as s,proposal,transport
from chemical_hypothesis_bo.hypotheses import digest


def one(task):
    p=s.check();pool=s.pool_for(task);folder=s.folder_for(task);state=s.read(folder/'state.json')
    assert state['status']=='COMPLETE' and len(state['history'])==20 and state['initial_ids']==[]
    assert len({v['candidate_id'] for v in state['history']})==20
    first=s.read(folder/'first_proposal_commit.json')
    req=s.read(ROOT/first['request']);public=req['public'] if req['attempt']==1 else req['public']['original']
    assert public==proposal.public(pool) and public['own_observations']==[]
    assert req['system']==proposal.PROPOSE_SYSTEM
    assert first['plan']==proposal.validate_proposal(first['raw'],pool)
    first_decision=state['decisions'][0]['decision'];confirmation=first_decision['confirmation_request']
    if confirmation:
        req2=s.read(ROOT/confirmation);af=(ROOT/confirmation).parent
        value=s.read(af.parent/'validated.json');public2=req2['public'] if req2['attempt']==1 else req2['public']['original']
        assert public2==proposal.confirmation(pool,first)
        assert value['plan']==proposal.validate_confirmation(value['raw'],pool,first['plan']['neighbors'])
        choice=value['plan']['choice'];assert choice['candidate_id']==first_decision['selected_id']
    else:
        choice=first['plan']['choice'];assert first['plan']['exact_candidate_id']==first_decision['selected_id']
    assert choice['forecast']==first_decision['forecast'] and choice['rationale']==first_decision['rationale']
    assert first_decision['conditions']==pool.points[pool.ix[first_decision['selected_id']]]
    assert first_decision['GP_fitted'] is False and first_decision['qLogEI_defined'] is False
    assert first_decision['GP_prediction_or_rank_exposed'] is False
    assert not (folder/'rounds/01_GP.npz').exists()
    replay=dict(task=task,protocol_sha256=p['protocol_sha256'],initial_ids=[],initial=0,queries=20,seed=0,
        history=[],decisions=[],status='RUNNING')
    gp=None;maximum_error=0.;decisions=[]
    for q in range(1,21):
        cp=s.read(folder/'rounds'/f'{q:02d}_commit.json');d=cp['decision']
        assert cp['commit_sha256']==digest({k:v for k,v in cp.items() if k!='commit_sha256'})
        assert cp['prestate_sha256']==digest(replay)
        j=pool.ix[d['selected_id']]
        assert d['conditions']==pool.points[j]
        if q>1:
            if gp is None:gp=s.ColdGP(pool,s.BASE/'cache'/s.SHORT[task],seed=0)
            context=gp.context(replay['history'],q)
            assert d['selected_id']==pool.ids[context['bo_index']]
            assert d['reason']=='native_full_pool_qLogEI' and d['observations_before']==q-1
            ap=folder/'rounds'/f'{q:02d}_GP.npz';assert s.sha(ap)==cp['GP_arrays_sha256']
            with np.load(ap,allow_pickle=False) as a:
                for key in ('mu_y','sd_y','sd_predictive_y'):np.testing.assert_allclose(a[key],context['stats'][key],rtol=1e-12,atol=1e-12)
                for key in ('available','tie_priority'):np.testing.assert_array_equal(a[key],context[key])
                maximum_error=max(maximum_error,float(np.max(np.abs(a['base_logei']-context['base_logei']))))
                js=np.flatnonzero(a['available']);js=js[a['base_logei'][js]==np.max(a['base_logei'][js])]
                assert int(js[np.argmin(a['tie_priority'][js])])==j
            assert abs(d['GP_mean_raw']-context['stats']['mu_y'][j]*pool.scale)<1e-10
            assert abs(d['GP_SD_raw']-context['stats']['sd_y'][j]*pool.scale)<1e-10
            assert d['operator_compiles_in_loop']==0
            gp.release()
        obs=s.native.read_observation(ROOT,pool,d['selected_id']);assert obs==state['history'][q-1]
        assert s.read(folder/'rounds'/f'{q:02d}_feedback.json')==dict(commit_sha256=cp['commit_sha256'],observation=obs)
        replay['history'].append(obs)
        replay['decisions'].append(dict(query=q,**obs,decision=d,best=max(v['raw_value'] for v in replay['history'])))
        replay['status']='COMPLETE' if q==20 else 'RUNNING'
        decisions.append(dict(task=s.SHORT[task],query=q,candidate_id=d['selected_id'],conditions=json.dumps(d['conditions']),
            objective_raw=obs['raw_value'],best_raw=replay['decisions'][-1]['best'],unit=pool.unit,
            selector=d['reason'],GP_mean_raw=d.get('GP_mean_raw'),GP_SD_raw=d.get('GP_SD_raw'),
            native_logEI=d.get('base_logEI'),GP_fitted=d['GP_fitted']))
    assert state==replay
    examples={r['candidate_id'] for r in public['input_only_examples']}
    actual=float(state['history'][0]['raw_value']);forecast=first_decision['forecast']
    return dict(task=s.SHORT[task],queries=20,all19_following_choices_exact_native_BO=True,
        maximum_saved_logEI_replay_error=maximum_error,zero_observations_before_LLM=True,
        first_GP_fit_after_first_real_observation=True,projection_confirmation=bool(confirmation),
        first_chosen_from_input_examples=first_decision['selected_id'] in examples,
        forecast_error_raw=actual-forecast['mean_raw'],forecast_standardized_error=(actual-forecast['mean_raw'])/forecast['sd_raw'],
        declared_subjective_95_interval_contains_outcome=abs(actual-forecast['mean_raw'])<=1.96*forecast['sd_raw'],
        operator_compiles_in_loop=0,decisions=decisions)


def main():
    p=s.check(True);assert not transport.pending()
    for task in s.native.TASKS:assert s.read(s.folder_for(task)/'state.json')['status']=='COMPLETE'
    receipts=[];agents=set()
    for path in sorted(transport.BASE.glob('requests/*/*/attempt*/receipt.json')):
        af=path.parent;receipt=s.read(path);req=s.read(af/'request.json');assignment=s.read(af/'assignment.json');export=s.read(af/'export.json')
        assert req['request_sha256']==digest({k:v for k,v in req.items() if k!='request_sha256'})
        assert receipt['request_sha256']==req['request_sha256']
        assert s.sha(af/'response.raw.json')==receipt['raw_sha256']
        assert s.sha(af/'assignment.json')==receipt['assignment_sha256']
        assert s.read(ROOT/export['output'])==s.read(af/'response.raw.json')
        assert s.sha(ROOT/export['input'])==export['input_sha256']
        agent=assignment['agent_id'];assert agent not in agents;agents.add(agent)
        assert assignment['spawn_request']==s.read(af/'invocation.json')
        assert assignment['spawn_result']==dict(task_name=agent)
        assert assignment['spawn_request']['fork_turns']=='none'
        assert assignment['spawn_request']['model']=='gpt-6.1-terra' and assignment['spawn_request']['reasoning_effort']=='medium'
        public=req['public'] if req['attempt']==1 else req['public']['original']
        assert public['own_observations']==[] and public['experimental_budget']['observed']==0
        assert public['GP_available'] is False and public['qLogEI_available'] is False and public['incumbent'] is None
        receipts.append(dict(task=req['task'],stage=req['stage'],attempt=req['attempt'],agent=agent,
            request_sha256=req['request_sha256'],response_sha256=receipt['raw_sha256']))
    assert len(receipts)<=32
    with ProcessPoolExecutor(max_workers=8,initializer=s.initialize) as executor:results=list(executor.map(one,s.native.TASKS))
    pd.DataFrame([r for row in results for r in row.pop('decisions')]).to_csv(s.REPORT/'decision_trace.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(results).to_csv(s.REPORT/'trajectory_audit.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(receipts).to_csv(s.REPORT/'LLM_receipts.csv',index=False,encoding='utf-8-sig')
    s.check(True)
    value=dict(status='PASS_ZERO_DATA_LLM_FIRST_AUDIT',tasks=8,paid_queries=160,initial_observations=0,
        native_BO_choices_replayed=152,actual_independent_LLM_calls=len(receipts),
        zero_GP_predictions_or_outcomes_in_all_LLM_requests=True,first_choices_all_explicitly_selected_by_LLM=True,
        operator_compiles_in_loop=0,maximum_saved_logEI_replay_error=max(r['maximum_saved_logEI_replay_error'] for r in results),
        source_sha256=s.sha(__file__),protocol_sha256=p['protocol_sha256'],historical_controls_unchanged=True,
        control_budget='0+20',historical_protected_artifacts=len(p['protected_sha256']),
        uncertainty='Single seed and one independent first-choice realization per task. Input-only archetype presentation is not separately ablated.')
    s.write(s.REPORT/'audit.json',value,replace=True);print(json.dumps(value))


if __name__=='__main__':main()
