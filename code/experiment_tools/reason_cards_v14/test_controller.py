"""Meaningful protocol and feedback tests; no scientific oracle/LLM calls."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from cold_start_v13 import control as old,adapter as old_adapter
from knowledge_rank_v9 import engine
from reasoning_memory_v12.llm import compiled_extractor
from . import controller as c,policy

ROOT=Path(__file__).resolve().parents[3]


def fixture(empty=False,gap=.1):
    ids=('old','a','b','c','d','e')
    pool=SimpleNamespace(ids=ids,ix={k:i for i,k in enumerate(ids)},axes=('electrophile','boron'),
        kinds=('categorical','categorical'),features=(('Br','acid'),('I','acid'),('I','ester'),('Br','ester'),('Cl','acid'),('Cl','ester')),scale=100.)
    state=dict(history=[] if empty else [dict(candidate_id='old',y=.2,raw_value=20.)],
        notes=[],decisions=[],decision_channel=c.initial_channel(),review_queries=[],edit_calls=[])
    n=len(ids)
    context=dict(stats=dict(mu_y=np.full(n,np.nan if empty else .2),sd_y=np.full(n,np.nan if empty else .2),
        sd_predictive_y=np.full(n,np.nan if empty else .2)),
        base_logei=np.array([0.,0.,-gap,-.8,-.9,-1.]) if not empty else np.array([-1.,0.,0.,0.,0.,0.]),
        available=np.ones(n,bool) if empty else np.array([False,True,True,True,True,True]),tie_priority=np.arange(n,dtype=float))
    ids5=['a','b','c','d','e']
    raw=dict(insight=dict(comment='Test the joint candidate prospectively.',keywords='test',hypotheses=[
        dict(strategy=f'Test {i}',rationale='Own observations do not establish a causal effect.',confidence='low',parameter_sets=[dict(candidate_id='b')]) for i in range(3)]),
        knowledge_uses=[],assessments=[],decision_card=dict(claim='Test this distinct joint regime.',focus=['boron'],purpose='improve',
        predictions=[dict(candidate_id=cid,p_improve_incumbent=None if empty else (.7 if cid=='b' else .4)) for cid in ids5]))
    return pool,state,context,ids5,raw


class V14Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):compiled_extractor(str(ROOT))

    def test_zero_observation_no_fabricated_probability_or_KF(self):
        pool,s,ctx,ids,raw=fixture(empty=True)
        packet=c.decide(ROOT,pool,[],s,ctx,ids,1,raw)
        self.assertEqual(packet['selected_id'],'b');self.assertIsNone(packet['actual_p_GP'])
        _,channel,_,event=c.feedback(pool,[],s['decision_channel'],[],packet,dict(candidate_id='b',raw_value=10.,y=.1))
        self.assertEqual(channel['calibration']['updates'],0)
        self.assertEqual(event['kind'],'no_incumbent_no_calibration')

    def test_all_five_forecasts_required_and_unmeasured_not_updated(self):
        pool,s,ctx,ids,raw=fixture()
        bad=deepcopy(raw);bad['decision_card']['predictions'].pop()
        with self.assertRaises(ValueError):c.decide(ROOT,pool,[],s,ctx,ids,2,bad)
        packet=c.decide(ROOT,pool,[],s,ctx,ids,2,raw)
        _,channel,_,_=c.feedback(pool,[],s['decision_channel'],s['history'],packet,dict(candidate_id='b',raw_value=70.,y=.7))
        self.assertEqual(channel['calibration']['updates'],1)
        self.assertEqual(len(channel['events']),1)

    def test_gate_is_numeric_and_GP_unchanged(self):
        pool,s,ctx,ids,raw=fixture(gap=.7);before=engine.gp_hash(ctx)
        packet=c.decide(ROOT,pool,[],s,ctx,ids,2,raw)
        self.assertEqual(packet['proposed_id'],'b');self.assertEqual(packet['selected_id'],'a')
        self.assertEqual(packet['deployment']['reason'],'opportunity_cost_exceeds_calibrated_budget')
        self.assertEqual(before,engine.gp_hash(ctx))

    def test_fallback_calibrates_executed_point_not_unmeasured_proposal(self):
        pool,s,ctx,ids,raw=fixture(gap=.7)
        packet=c.decide(ROOT,pool,[],s,ctx,ids,2,raw)
        self.assertEqual(packet['actual_p_LLM'],.4)
        _,channel,_,event=c.feedback(pool,[],s['decision_channel'],s['history'],packet,dict(candidate_id='a',raw_value=12.,y=.12))
        self.assertAlmostEqual(event['calibration']['Brier_gain'],.5**2-.4**2)
        self.assertEqual(channel['facts'],[])

    def test_uniform_background_has_no_positive_discrimination(self):
        pool,s,ctx,ids,raw=fixture(gap=.0)
        for p in raw['decision_card']['predictions']:p['p_improve_incumbent']=.4
        packet=c.decide(ROOT,pool,[],s,ctx,ids,2,raw)
        self.assertEqual(packet['selected_id'],'a')
        self.assertEqual(packet['deployment']['reason'],'no_candidate_discrimination_beyond_GP')

    def test_wrong_focus_can_calibrate_but_not_gain_direction_permission(self):
        pool,s,ctx,ids,raw=fixture()
        s['history'][0].update(candidate_id='a')
        note=old.new_note('H001',1,'promote',{},['electrophile'],created_query=2)
        f=old.commit_forecast(note,s['history'],2,'b',old_adapter.conditions(pool,'b'),.7,.5,use='tentative_test')
        legacy=dict(forecasts=[f],selected_id='b')
        rows=c.effective_forecasts(pool,legacy,[note],s['history'])
        self.assertFalse(rows[0]['credit']['eligible']);self.assertEqual(rows[0]['effective']['use'],'shadow')
        after,event=old.feedback(note,rows[0]['effective'],s['history'],dict(candidate_id='b',raw_value=70.,y=.7))
        self.assertTrue(event['improved']);self.assertEqual(after['calibration']['updates'],1)
        self.assertEqual(after['recommendation']['positive_tests'],[])

    def test_accurate_failure_does_not_release_local_recommendation(self):
        pool,s,ctx,ids,raw=fixture()
        fact=dict(fact_id='risk',conditions=old_adapter.conditions(pool,'b'),query=1,severity='large_shortfall',
            status='suppressed',review_successes=[],review_failures=0)
        s['decision_channel']['facts']=[fact]
        risk=c.risk_context(pool,s,'b',2,['boron'])
        self.assertTrue(risk['blocked']);self.assertFalse(risk['eligible_review'])
        packet=c.decide(ROOT,pool,[],s,ctx,ids,2,raw)
        self.assertEqual(packet['selected_id'],'a')
        _,channel,_,event=c.feedback(pool,[],s['decision_channel'],s['history'],packet,dict(candidate_id='a',raw_value=12.,y=.12))
        self.assertTrue(event['accurate_failure_forecast'])
        self.assertEqual(channel['facts'][0]['status'],'suppressed')

    def test_true_material_change_review_is_bounded_and_consumes_on_execution(self):
        pool,s,ctx,ids,raw=fixture()
        s['decision_channel']['facts']=[dict(fact_id='risk',conditions=old_adapter.conditions(pool,'a'),
            query=1,status='suppressed',review_successes=[],review_failures=0)]
        raw['decision_card']['purpose']='review'
        packet=c.decide(ROOT,pool,[],s,ctx,ids,2,raw)
        self.assertEqual(packet['selected_id'],'b');self.assertIsNotNone(packet['deployment']['review'])
        _,channel,_,_=c.feedback(pool,[],s['decision_channel'],s['history'],packet,dict(candidate_id='b',raw_value=70.,y=.7))
        self.assertEqual(len(channel['review_queries']),1)
        self.assertEqual(channel['facts'][0]['status'],'suppressed')
        s['decision_channel']['review_queries']=[2,3]
        self.assertFalse(c.risk_context(pool,s,'b',2,['boron'])['eligible_review'])

    def test_serious_local_failure_is_recorded_with_physical_conditions(self):
        pool,s,ctx,ids,raw=fixture()
        s['history'][0].update(raw_value=80.,y=.8)
        pool.ids+=('prior2',);pool.ix['prior2']=6;pool.features+=(('F','acid'),)
        s['history'].append(dict(candidate_id='prior2',raw_value=60.,y=.6))
        for key in ctx['stats']:ctx['stats'][key]=np.append(ctx['stats'][key],.2)
        ctx['base_logei']=np.append(ctx['base_logei'],-2.)
        ctx['available']=np.append(ctx['available'],False);ctx['tie_priority']=np.append(ctx['tie_priority'],6.)
        packet=c.decide(ROOT,pool,[],s,ctx,ids,3,raw)
        _,channel,_,event=c.feedback(pool,[],s['decision_channel'],s['history'],packet,dict(candidate_id='b',raw_value=10.,y=.1))
        self.assertTrue(event['local_suppression']);self.assertEqual(channel['facts'][0]['severity'],'large_shortfall')
        self.assertEqual(channel['facts'][0]['conditions'],old_adapter.conditions(pool,'b'))

    def test_two_distinct_improving_review_queries_release_only_local_fact(self):
        pool,s,ctx,ids,raw=fixture()
        pool.ids+=('newx',);pool.ix['newx']=6;pool.features+=(('I','salt'),)
        for key in ctx['stats']:ctx['stats'][key]=np.append(ctx['stats'][key],.2)
        ctx['base_logei']=np.append(ctx['base_logei'],-1.5)
        ctx['available']=np.append(ctx['available'],True);ctx['tie_priority']=np.append(ctx['tie_priority'],6.)
        s['decision_channel']['facts']=[dict(fact_id='risk',conditions=old_adapter.conditions(pool,'a'),
            query=1,status='suppressed',review_successes=[],review_failures=0)]
        raw['decision_card']['purpose']='review'
        packet=c.decide(ROOT,pool,[],s,ctx,ids,2,raw)
        observed=dict(candidate_id='b',raw_value=70.,y=.7)
        _,channel,_,_=c.feedback(pool,[],s['decision_channel'],s['history'],packet,observed)
        self.assertEqual(channel['facts'][0]['status'],'suppressed')
        s['history'].append(observed);s['decision_channel']=channel
        ctx['available'][2]=False;ctx['base_logei'][6]=-.1
        for key in ctx['stats']:ctx['stats'][key][:]=.7 if key=='mu_y' else .2
        ids2=['a','c','d','e','newx'];raw2=deepcopy(raw)
        for h in raw2['insight']['hypotheses']:h['parameter_sets']=[dict(candidate_id='newx')]
        raw2['decision_card']['predictions']=[dict(candidate_id=cid,p_improve_incumbent=.7 if cid=='newx' else .4) for cid in ids2]
        packet2=c.decide(ROOT,pool,[],s,ctx,ids2,3,raw2)
        _,channel,_,_=c.feedback(pool,[],s['decision_channel'],s['history'],packet2,dict(candidate_id='newx',raw_value=90.,y=.9))
        self.assertEqual(channel['facts'][0]['status'],'released')
        self.assertEqual(channel['facts'][0]['review_successes'],[2,3])
        self.assertEqual(len(channel['review_queries']),2)

    def test_measured_only_note_revision_cannot_erase_old_version(self):
        pool,s,ctx,ids,raw=fixture()
        oldnote=old.new_note('H001',1,'promote',{},['boron'])
        newnote=old.new_note('H001',2,'promote',{'electrophile':{'allowed':['Br']}},['boron'])
        # Br has two pool witnesses; one is already measured, so only one is executable.
        s['notes']=[oldnote]
        plan=dict(notes=[old.withdraw(oldnote,2,'superseded'),newnote],reviews=[dict(accepted=True,outcome='revise',id='H001',version=2)])
        with patch.object(policy.old_policy,'apply_edits',return_value=plan):
            result=policy.apply_edits({},pool,s,2)
        self.assertFalse(result['reviews'][0]['accepted']);self.assertEqual(result['notes'],[oldnote])


if __name__=='__main__':unittest.main()
