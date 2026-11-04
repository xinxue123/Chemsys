"""Synthetic controller tests and real-prefix prompt-blinding checks."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import csv
import unittest
import numpy as np
from cold_start_v13 import study as frozen,control
from reasoning_memory_v13 import calibration
from . import policy,prompts,study

ROOT=Path(__file__).resolve().parents[3]


def fixture():
    pool=SimpleNamespace(ids=np.array(['a','b','c','d','e']),ix={c:i for i,c in enumerate('abcde')},
        axes=['x'],kinds=['continuous'],features=[(i/4,) for i in range(5)],scale=1.)
    ctx=dict(stats=dict(mu_y=np.full(5,.2),sd_y=np.full(5,.1),sd_predictive_y=np.full(5,.2)),
        base_logei=np.arange(4,-1,-1,dtype=float),available=np.ones(5,bool),tie_priority=np.array([3,0,4,2,1]))
    return pool,ctx


class ControllerTests(unittest.TestCase):
    def test_hand_calculated_rank_weights(self):
        pool,ctx=fixture();p=dict(zip(pool.ids,[.1,.2,.3,.4,.9]));h=[{'candidate_id':'old','y':0.,'raw_value':0.}]
        fixed=policy.rank(pool,ctx,pool.ids,p,.4,'a',h)
        dynamic=policy.rank(pool,ctx,pool.ids,p,.8,'a',h)
        np.testing.assert_allclose(fixed['combined_scores'],[.6,.55,.5,.45,.4])
        np.testing.assert_allclose(dynamic['combined_scores'],[.2,.35,.5,.65,.8])
        self.assertEqual((fixed['selected_id'],dynamic['selected_id']),('a','e'))

    def test_true_GP_ties_are_not_artificial_order(self):
        pool,ctx=fixture();ctx['base_logei'][:]=7.;p=dict(zip(pool.ids,[.1,.2,.3,.4,.9]));h=[{}]
        for w in (.01,.4,.8):
            r=policy.rank(pool,ctx,pool.ids,p,w,'a',h)
            self.assertEqual(r['selected_id'],'e');self.assertEqual(r['GP_rank_scores'],[.5]*5)

    def test_uniform_LLM_cannot_perturb_GP(self):
        pool,ctx=fixture();p={c:.3 for c in pool.ids}
        for w in (0.,.4,.8):
            r=policy.rank(pool,ctx,pool.ids,p,w,'e',[{}]);self.assertEqual(r['selected_id'],'a')

    def test_zero_data_has_no_fabricated_probability(self):
        pool,ctx=fixture();p={c:None for c in pool.ids}
        r=policy.rank(pool,ctx,pool.ids,p,.4,'d',[])
        self.assertEqual(r['selected_id'],'d');self.assertIsNone(r['GP_rank_scores'])

    def test_KF_state_never_enters_both_prompt_surfaces(self):
        with (ROOT/'paper/v13_late/data/metrics.csv').open(encoding='utf-8-sig') as f:
            row=next(r for r in csv.DictReader(f) if r['regime']=='cold' and r['task']=='hplc'
                     and r['seed']=='3' and r['method']=='v13')
        path=ROOT/row['state_path'];old=study.read(path);q=6
        c=study.read(path.parent/'rounds/06_commit.json');pool=frozen.pool_for(ROOT,'HPLC')
        state=study.held_state(old,q);state['ranker_KF']=calibration.initial();notes=c['edited_notes'];ids=c['display_ids']
        other=deepcopy(state);other['ranker_KF']['reliability']=.8
        other_notes=deepcopy(notes)
        for n in other_notes:n['calibration'].update(mean=99.,variance=88.,reliability=.01)
        self.assertEqual(prompts.reasoning_public(ROOT,pool,state,notes,ids,q),
                         prompts.reasoning_public(ROOT,pool,other,other_notes,ids,q))
        state['notes']=notes;other['notes']=other_notes;refresh={'allowed':True,'reasons':['test']}
        self.assertEqual(prompts.memory_public(pool,state,ids,q,refresh),prompts.memory_public(pool,other,ids,q,refresh))

    def test_candidate_predictions_reject_future_invented_first_probabilities(self):
        pool,_=fixture();card=dict(claim='synthetic',focus=['x'],purpose='improve',
            predictions=[dict(candidate_id=c,p_improve_incumbent=None) for c in pool.ids])
        raw=dict(insight={},knowledge_uses=[],assessments=[],decision_card=card)
        self.assertTrue(all(v is None for v in policy.predictions(raw,pool,pool.ids,[]).values()))
        raw['decision_card']['predictions'][0]['p_improve_incumbent']=.4
        with self.assertRaises(ValueError):policy.predictions(raw,pool,pool.ids,[])

    def test_only_actual_point_updates_dense_KF(self):
        state=dict(history=[dict(candidate_id='old',y=.1,raw_value=.1)],notes=[],ranker_KF=calibration.initial(),
                   edit_calls=[],review_queries=[],insights=[],decisions=[],status='READY')
        packet=dict(query=2,selected_id='executed',proposed_id='unexecuted',BO_reference_id='bo',
            history_sha256=control.digest(state['history']),notes_before_sha256=control.digest([]),
            ranker_before=deepcopy(state['ranker_KF']),notes_at_commit=[],note_forecasts=[],knowledge_uses=[],
            actual_p_LLM=.2,actual_p_GP=.5)
        packet['commit_sha256']=control.digest(packet)
        commit=dict(query=2,selected_id='executed',prestate_sha256=control.digest(state),decision=packet,
                    edited_notes=[],memory=None,reasoning=dict(raw=dict(insight={'comment':'proposal'})))
        commit['commit_sha256']=control.digest(commit)
        result=policy.feedback(state,commit,dict(candidate_id='executed',y=0.,raw_value=0.))
        self.assertEqual(result['ranker_KF']['updates'],1)
        self.assertEqual(result['decisions'][0]['ranker_feedback']['candidate_id'],'executed')
        self.assertEqual(result['decisions'][0]['feedback'],[])
        self.assertEqual(len(state['history']),1)


if __name__=='__main__':unittest.main()
