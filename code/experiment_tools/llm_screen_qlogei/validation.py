"""Mechanism qualification without LLM calls or new oracle experiments."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
from reasoning_memory_v13 import calibration
from . import knowledge,policy


def qualify(gp,pool,reference):
    context=gp.context(reference['history'][:1],2)
    expected=reference['decisions'][1]
    assert context['key']==expected['fit_key']
    assert pool.ids[policy.choose(context['logei'],context['available'],context['priorities'])]==expected['decision']['selected_id']
    toy=SimpleNamespace(ids=('a','b','c'),axes=('x',),kinds=('continuous',),
        points=({'x':0.},{'x':.5},{'x':1.}))
    note=dict(id='local',version=1,claim='test',direction='favour',scope={'x':[.4,.6]},
        yield_mean=80.,yield_sd=10.,KF=calibration.initial(),permission='provisional',
        negative_streak=0,deployed_failures=0,feedback=[],evidence_inherited_from=None,installed_query=2)
    c=dict(mu=np.full(3,.1),variance=np.full(3,.09),logei=np.array([0.,-.1,-.2]),
        available=np.ones(3,dtype=bool),priorities=np.arange(3),incumbent=.2)
    packet,arrays=policy.decision(toy,c,[note],2)
    assert packet['selected_id']=='b' and packet['BO_reference_id']=='a' and not packet['baseline_survives']
    assert arrays['eligible'].tolist()==[False,True,False]
    after,events=knowledge.feedback([note],packet['forecasts'],dict(candidate_id='b',y=0.,raw_value=0.),2)
    assert events[0]['Brier_gain']<0 and after[0]['KF']['reliability']<.2 and after[0]['permission']=='paused'
    subsequent,_=policy.decision(toy,c,after,3)
    assert subsequent['selected_id']=='a'
    event=knowledge.refresh_event(dict(notes=after,refresh_queries=[]),3)
    assert event['allowed']
    avoided=deepcopy(after[0])
    avoided['direction']='avoid'
    avoided['permission']='paused'
    avoided['yield_mean']=0.
    pre=dict(id=avoided['id'],version=avoided['version'],p_LLM=.1,p_GP=.9,
        incumbent_y=.2,promoted_selection=False)
    recovered,feedback=knowledge.feedback([avoided],[pre],dict(candidate_id='b',y=0.,raw_value=0.),3)
    assert feedback[0]['Brier_gain']>0 and recovered[0]['permission']=='paused'
    # Rewording a failed rule at the same physical scope retains its evidence.
    real_scope={'temperature':[70.,110.]}
    old=dict(note,scope=real_scope,KF=deepcopy(after[0]['KF']),permission='paused')
    draft={k:deepcopy(old[k]) for k in ('id','claim','direction','scope','yield_mean','yield_sd')}
    draft.update(id='renamed',claim='rephrased')
    installed=knowledge.install(dict(hypotheses=[draft],summary='test'),pool,[old],3)
    assert installed[0]['KF']==old['KF'] and installed[0]['permission']=='paused'
    bounded=dict(c,logei=np.array([0.,-5.,-.2]))
    bounded_decision,_=policy.decision(toy,bounded,[note],2)
    assert bounded_decision['selected_id']=='a'
    assert gp.counters['fits']==0
    return dict(status='PASS',native_GP_and_qLogEI_reference_match=True,
        LLM_screen_can_change_selection=True,negative_feedback_reduces_KF_and_changes_next_choice=True,
        degraded_knowledge_triggers_refresh=True,accurate_failure_does_not_restore_permission=True,
        rewording_inherits_evidence=True,screen_cannot_escape_qLogEI_budget=True,
        GP_arrays_unchanged=True,LLM_calls=0,new_experiments=0,operator_compiles_in_loop=0)
