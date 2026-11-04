from copy import deepcopy
from types import SimpleNamespace
from pathlib import Path
import json
import unittest
import numpy as np
from chem.benchmarks.knowledge_llm import digest
from knowledge_rank_v9 import kf,policy,engine,llm

ROOT=Path(__file__).resolve().parents[4]


def setup():
    ids=[f'p{i}' for i in range(30)]
    pool=SimpleNamespace(ids=ids,ix={key:i for i,key in enumerate(ids)},axes=('x','category'),
        kinds=('continuous','categorical'),features=tuple((i/29,'A' if i%2 else 'B') for i in range(30)),
        points=[dict(x=i/29,category='A' if i%2 else 'B') for i in range(30)],scale=100.,task='synthetic',unit='percent')
    state=dict(task='synthetic',arm='kf_rank',seed=0,protocol_sha256='registered',initial_commit_sha256='initial',
        history=[dict(candidate_id=ids[j],y=.1*(j+1),raw_value=10.*(j+1)) for j in range(4)],
        hypotheses=[],decisions=[],insights=[],edit_calls=[],status='READY')
    return pool,state


def context(pool,state):
    available=~np.isin(pool.ids,[r['candidate_id'] for r in state['history']])
    return dict(stats=dict(mu_y=np.full(30,.3),sd_y=np.full(30,.2),sd_predictive_y=np.full(30,.25),fit_key='fit'),
        base_logei=-np.arange(30,dtype=float),available=available,tie_priority=np.arange(30),eligible=available,epsilon=.2)


def action(**changes):
    a=dict(op='add',id='H001',claim='Local x trend',focus=['x'],effect='promote',scope={},reference_id='p0',reason='test')
    a.update(changes);return a


def planned():
    pool,state=setup();ctx=context(pool,state);ordered,display=engine.candidates(pool,ctx,1)
    ids=[pool.ids[j] for j in display];refresh=engine.edit_event(pool,state,ordered,1)
    probs={key:.5 for key in ids};probs['p4']=.1;probs['p5']=.9
    raw=dict(summary='synthetic',actions=[action()],assessments=[dict(id='H001',probabilities=probs)])
    plan=llm.validate_kf(raw,pool,state,ids,1,refresh)
    receipt=dict(context_sha256='context',raw=raw,plan=plan)
    return pool,state,ctx,engine.plan(pool,state,ctx,1,'kf_rank',receipt,refresh)


class KFTests(unittest.TestCase):
    def test_hand_computed_update_joseph_and_negative_weight(self):
        state=kf.initial();after,event=kf.update(state,.9,.5,False,1)
        z=.25-.81;r=4*.4**2*.25;pp=.2525;gain=pp/(pp+r)
        self.assertAlmostEqual(event['Brier_gain'],z)
        self.assertAlmostEqual(after['mean'],gain*z)
        self.assertAlmostEqual(after['variance'],(1-gain)**2*pp+gain*gain*r)
        self.assertLess(after['weight'],state['weight']);self.assertEqual(state,kf.initial())

    def test_automatic_pause_retract_and_no_retired_feedback(self):
        state=kf.initial()
        for t in range(1,5):
            state,event=kf.update(state,.95,.5,False,t)
            if t==2:self.assertEqual(event['reason'],'automatic_pause');self.assertEqual(kf.deployment_weight(state),0.)
        self.assertEqual(event['reason'],'automatic_retract');self.assertEqual(state['status'],'retracted')
        with self.assertRaises(ValueError):kf.update(state,.9,.5,True,5)

    def test_pause_requires_new_positive_feedback_for_recovery(self):
        state=kf.initial()
        for t in (1,2):state,event=kf.update(state,.2,.1,False,t)
        self.assertEqual(state['status'],'paused');before=deepcopy(state)
        self.assertEqual(kf.deployment_weight(state),0.);self.assertEqual(state,before)
        for t in range(3,10):
            state,event=kf.update(state,.9,.1,True,t)
            if state['status']=='active':break
        self.assertEqual(event['reason'],'automatic_resume');self.assertGreater(kf.deployment_weight(state),0)

    def test_no_new_credit_without_probability_difference(self):
        state=kf.initial();after,event=kf.update(state,.5,.5,True,1)
        self.assertEqual(event['Brier_gain'],0.);self.assertEqual(after['weight'],.4)
        for bad in (float('nan'),float('inf'),True,'0.5'):
            with self.assertRaises(ValueError):kf.update(state,bad,.5,True,1)

    def test_tiny_negative_scores_reduce_weight_without_failure_streak(self):
        state=kf.initial()
        for t in range(1,7):state,event=kf.update(state,.95,.999,True,t)
        self.assertEqual(state['status'],'active');self.assertEqual(state['bad_streak'],0)
        self.assertLess(state['weight'],.4)


class ClosedLoopTests(unittest.TestCase):
    def test_ranking_changes_choice_without_modifying_gp(self):
        pool,state,ctx,c=planned()
        self.assertEqual(c['selected_id'],'p5');self.assertEqual(c['mode'],'KF_rank')
        self.assertEqual(c['GP_sha256'],c['GP_sha256_after_ranking']);before=engine.gp_hash(ctx)
        engine.feedback(state,c,dict(candidate_id='p5',y=.05,raw_value=5.))
        self.assertEqual(len(state['hypotheses'][0]['events']),1);self.assertEqual(engine.gp_hash(ctx),before)
        self.assertLess(state['hypotheses'][0]['kf']['weight'],.4)

    def test_repeated_or_tampered_feedback_is_atomic(self):
        pool,state,ctx,c=planned();before=deepcopy(state)
        for field in ('commit','forecast','semantics','objective'):
            bad=deepcopy(c);obs=dict(candidate_id=c['selected_id'],y=.4,raw_value=40.)
            if field=='commit':bad['selected_id']='p6'
            if field=='forecast':bad['predictions'][0]['prediction']['p_LLM']=.2
            if field=='semantics':bad['hypotheses_at_commit'][0]['claim']='rewritten'
            if field=='objective':obs['y']=float('nan')
            with self.assertRaises(ValueError):engine.feedback(state,bad,obs)
            self.assertEqual(state,before)
        engine.feedback(state,c,dict(candidate_id=c['selected_id'],y=.4,raw_value=40.));saved=deepcopy(state)
        with self.assertRaises(ValueError):engine.feedback(state,c,dict(candidate_id=c['selected_id'],y=.4,raw_value=40.))
        self.assertEqual(state,saved)

    def test_revision_preserves_negative_prior_and_cannot_retain_resume(self):
        pool,state,ctx,c=planned();engine.feedback(state,c,dict(candidate_id='p5',y=.05,raw_value=5.))
        ordered,_=engine.candidates(pool,context(pool,state),2)
        hs,reviews=policy.apply(pool,state['hypotheses'],[action(op='revise',claim='Reworded x relationship')],state['history'],ordered,2,True)
        self.assertTrue(reviews[0]['accepted']);self.assertEqual(hs[-1]['version'],2)
        self.assertEqual(hs[-1]['kf']['status'],'paused');self.assertEqual(hs[-1]['kf']['local_updates'],0)
        self.assertEqual(len(hs[-1]['inherited_negative']),1);self.assertLess(hs[-1]['kf']['weight'],.4)
        retain=action(op='retain',claim=None,focus=[],effect=None,scope={},reference_id=None)
        kept,_=policy.apply(pool,hs,[retain],state['history'],ordered,3,True)
        self.assertEqual(kept[-1]['kf']['status'],'paused')
        with self.assertRaises(ValueError):policy.apply(pool,hs,[retain],state['history'],ordered,3,False)

    def test_tied_llm_probabilities_and_category_labels_are_not_ranks(self):
        pool,state,ctx,c=planned();hs=c['hypotheses_at_commit'];ordered,_=engine.candidates(pool,ctx,1)
        p={'H001':{pool.ids[j]:.5 for j in ordered}}
        selected,r=engine.rank_select(pool,ctx,ordered,hs,p)
        self.assertEqual(selected,ordered[0]);self.assertTrue(r['terms'][0]['no_rank_information'])
        hs[0]['scope']={'category':{'allowed':['A']}}
        got=policy.scope_indices(pool,hs[0],ordered)
        self.assertTrue(all(pool.features[j][1]=='A' for j in got))
        np.testing.assert_array_equal(engine.borda([.2,.5,.5,.9,.1]),[.25,.625,.625,1.,0.])

    def test_refresh_quota_reservation_and_conflict(self):
        pool,state=setup();ordered,_=engine.candidates(pool,context(pool,state),1)
        state['edit_calls']=[1,4,7]
        self.assertFalse(engine.edit_event(pool,state,ordered,10)['allowed'])
        self.assertTrue(engine.edit_event(pool,state,ordered,11)['allowed'])
        state['edit_calls']+=[11];self.assertFalse(engine.edit_event(pool,state,ordered,16)['allowed'])

    def test_official_confidence_sort_and_candidate_boundary(self):
        ids=['a','b','c','d','e']
        def h(key,confidence):return dict(strategy='test',rationale='test',confidence=confidence,parameter_sets=[dict(candidate_id=key)])
        raw=dict(comment='test',keywords='test',hypotheses=[h('a','low'),h('b','high'),h('c','high')])
        self.assertEqual(llm.validate_rbo(ROOT,raw,ids)['selected_id'],'b')
        raw['hypotheses'][0]['parameter_sets'][0]['candidate_id']='unlisted'
        with self.assertRaises(ValueError):llm.validate_rbo(ROOT,raw,ids)


if __name__=='__main__':unittest.main()
