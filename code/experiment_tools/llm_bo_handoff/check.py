"""Mechanism checks only; no new LLM calls, GP fitting or oracle lookups."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
from reasoning_memory_v13 import calibration
from llm_screen_qlogei import knowledge
from . import policy,PARAMETERS as P,VARIANT

ROOT=Path(__file__).resolve().parents[3]


def note(scope,reliability=.4):
    result=dict(id='h',version=1,claim='Qualification-only preference',direction='favour',
        scope=scope,yield_mean=80.,yield_sd=20.,permission='provisional',KF=calibration.initial(),
        negative_streak=0,deployed_failures=0,feedback=[],installed_query=1,evidence_inherited_from=None)
    result['KF']['reliability']=reliability
    return result


def run():
    pool=SimpleNamespace(ids=('a','b','c','d'),axes=('x',),kinds=('continuous',),
        points=({'x':0.},{'x':.4},{'x':.7},{'x':1.}))
    context=dict(mu=np.full(4,.4),variance=np.full(4,.1),logei=np.array([0.,-.2,-.8,-2.]),
        available=np.ones(4,dtype=bool),priorities=np.arange(4),incumbent=.2)
    broad=note({'x':[.3,.8]})
    early,_=policy.decide(pool,context,[broad],None,2)
    assert early['selected_id']=='b' and early['BO_reference_id']=='a'
    updated,events=knowledge.feedback([broad],early['forecasts'],
        dict(candidate_id='b',y=0.,raw_value=0.),2)
    assert events[0]['Brier_gain']<0 and updated[0]['permission']=='paused'
    following,_=policy.decide(pool,context,updated,None,3)
    assert following['selected_id']=='a'
    late,_=policy.decide(pool,context,[broad],None,7)
    assert late['selected_id']=='a'
    narrow=note({'x':[.6,.8]})
    high,_=policy.decide(pool,context,[narrow],None,2)
    low_note=note({'x':[.6,.8]},.21)
    low,_=policy.decide(pool,context,[low_note],None,2)
    assert high['selected_id']=='c' and low['selected_id']=='a'
    paused=deepcopy(broad)
    paused['permission']='paused'
    paused['KF']['reliability']=.8
    blocked,_=policy.decide(pool,context,[paused],None,2)
    assert blocked['selected_id']=='a'
    failure=deepcopy(broad)
    failure.update(yield_mean=0.,yield_sd=5.)
    reliable_failure,_=policy.decide(pool,context,[failure],None,2)
    assert reliable_failure['selected_id']=='a'
    tied=dict(context,logei=np.array([0.,0.,0.,-2.]))
    endpoint,_=policy.decide(pool,tied,[broad],None,8)
    assert endpoint['selected_id']=='a' and endpoint['phase']=='pure_BO'
    refreshed=deepcopy(broad)
    refreshed['installed_query']=7
    refresh_packet,_=policy.decide(pool,context,[refreshed],None,7)
    assert refresh_packet['handoff_authority_cap']==late['handoff_authority_cap']
    schedule=[]
    for q in range(2,9):
        packet,_=policy.decide(pool,context,[broad],None,q)
        schedule.append(dict(query=q,LLM_authority_cap=packet['handoff_authority_cap'],
            logei_gap=packet['effective_logei_gap'],minimum_exp_qLogEI_retained=packet['minimum_retained_exp_qLogEI']))
    assert all(a['logei_gap']>=b['logei_gap'] for a,b in zip(schedule,schedule[1:]))
    assert schedule[-1]['logei_gap']==0
    # A real already-observed prefix, with predictions fixed and no oracle read.
    from chem.benchmarks import knowledge_gp
    from llm_event_start import policy as original
    base=ROOT/'experiments/runs/llm_event_start_qlogei_20261104/runs/event_s0'
    state_path=base/'state.json'
    if not state_path.exists():
        state_path=ROOT/'results/llm_event_start_qlogei_20261104/event_reference.json'
    state=json.loads(state_path.read_text(encoding='utf-8'))
    real=knowledge_gp.load_task(ROOT,'Reizman_Suzuki_Case_1',seed=0)
    initial=state['decisions'][0]
    plan=original.validate(initial['LLM']['raw'],real,[],1)
    first,_=policy.decide(real,None,plan['notes'],plan['proposal'],1)
    assert first['selected_id']==state['history'][0]['candidate_id']
    array_path=base/'rounds/02_GP.npz'
    if array_path.exists():
        with np.load(array_path,allow_pickle=False) as z:
            held={key:z[key].copy() for key in z.files}
    else:
        from fast_hit_bo.model import SharedGP
        gp=SharedGP(real,ROOT/'experiments/runs'/VARIANT/'numerical_check_cache')
        context=gp.context(state['history'][:1],2)
        held={key:context[key].copy() for key in ('mu','variance','logei','available','priorities')}
        assert gp.counters['fits']==0
    held['incumbent']=state['history'][0]['y']
    shadow,_=policy.decide(real,held,initial['notes_at_commit'],None,2)
    final,_=policy.decide(real,held,initial['notes_at_commit'],None,8)
    assert final['selected_id']==shadow['BO_reference_id']
    result=dict(status='PASS',test_scope='controller qualification; no performance experiment',
        first_point_is_LLM_selected=True,knowledge_can_change_early_choice=True,
        transfer_starts_at_query2=True,time_ceiling_contracts=True,
        KF_confidence_can_reduce_choice_authority=True,paused_knowledge_is_blocked=True,
        prospective_negative_KF_feedback_changes_following_choice=True,
        accurate_failure_does_not_become_positive_recommendation=True,
        refresh_does_not_reset_handoff_clock=True,terminal_native_BO_tie_rule_exact=True,
        GP_arrays_unchanged=True,schedule=schedule,real_fixed_prefix_shadow=shadow,
        new_LLM_calls=0,new_GP_fits=0,new_oracle_experiments=0)
    out=ROOT/'experiments/reports'/VARIANT/'mechanism_checks.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))
