"""No-outcome qualification of the first-point and sparse-event rules."""
from copy import deepcopy
import numpy as np
from llm_screen_qlogei.validation import qualify as qualify_knowledge
from . import policy


def qualify(gp,pool,reference):
    prior=qualify_knowledge(gp,pool,reference)
    c=gp.context(reference['history'][:1],2)
    raw=dict(proposal=dict(conditions=pool.points[0],basis_ids=['h'],rationale='Input-only projection check'),
        hypotheses=[dict(id='h',claim='Input-only test hypothesis',direction='favour',
            scope={'temperature':[30.,110.]},yield_mean=50.,yield_sd=25.)],summary='Numerical test only')
    plan=policy.validate(raw,pool,[],1)
    first=policy.decide(pool,None,plan['notes'],plan['proposal'],1,0,'event')
    assert first['selected_id']==pool.ids[0] and first['proposal_accepted']
    assert first['forecasts']==[]
    normal=policy.decide(pool,c,plan['notes'],None,2,0,'event')
    baseline=policy.choose(c['logei'],c['available'],c['priorities'])
    assert normal['selected_id']==pool.ids[baseline] and normal['continuous_screening_used'] is False
    proposal=deepcopy(plan['proposal'])
    proposal['conditions']=pool.points[baseline]
    accepted=policy.decide(pool,c,plan['notes'],proposal,2,0,'event')
    assert accepted['proposal_accepted'] and accepted['selected_id']==pool.ids[baseline]
    js=np.flatnonzero(c['available'])
    worst=int(js[np.argmin(c['logei'][js])])
    assert c['logei'][baseline]-c['logei'][worst]>.7
    proposal['conditions']=pool.points[worst]
    rejected=policy.decide(pool,c,plan['notes'],proposal,2,0,'event')
    assert not rejected['proposal_accepted'] and rejected['selected_id']==pool.ids[baseline]
    assert rejected['rule']=='event_rejected_qLogEI_budget'
    paused=deepcopy(plan['notes'])
    paused[0]['permission']='paused'
    proposal['conditions']=pool.points[baseline]
    denied=policy.decide(pool,c,paused,proposal,2,0,'event')
    assert not denied['proposal_accepted']
    return dict(status='PASS',first_proposal_is_input_only_and_reaches_actual_pool=True,
        normal_rounds_are_full_pool_qLogEI=True,event_proposal_review_accepts_and_rejects=True,
        paused_knowledge_cannot_bypass_review=True,random_first_point_has_no_yield_condition=True,
        knowledge_mechanisms=prior,LLM_calls=0,new_experiments=0,operator_compiles_in_loop=0)
