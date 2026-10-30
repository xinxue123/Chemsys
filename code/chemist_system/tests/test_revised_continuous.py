"""Behavioral safeguards for the four explicitly requested R1 changes."""
import unittest
from copy import deepcopy
import numpy as np
from chem.benchmarks import revised_continuous_engine as e
from chem.benchmarks import revised_continuous_llm as llm
from chem.benchmarks import continuous_benchmark_study as old

class RevisedContinuousTests(unittest.TestCase):
    def test_HPLC_original_region_is_not_executable(self):
        pool=old.pool('HPLC');ready=old.data.read(old.ROOT/'initialization/HPLC/ready.json')
        a=e.admission(pool,ready['knowledge'],ready['history'])
        self.assertEqual(a['pool_witness_count'],1);self.assertEqual(a['unqueried_witness_count'],0);self.assertFalse(a['admitted'])
        empty={'confidence':.7,'clauses':[]};self.assertTrue(e.admission(pool,empty,ready['history'])['admitted'])
    def test_validator_rejects_retaining_exhausted_initial_region(self):
        pool=old.pool('HPLC');ready=old.data.read(old.ROOT/'initialization/HPLC/ready.json');state={**ready,'decisions':[]}
        obs=old.llm.observations(pool,state)[3]
        raw={'actions':[{'id':h['id'],'operation':'retain','bounds':[],'reason':'uncertain existing preference',
            'support':[{'index':4,'axis':h['axis'],'value':obs['point'][h['axis']],'objective':obs['objective']}]} for h in ready['knowledge']['clauses']],
            'summary':'retain all'}
        with self.assertRaisesRegex(ValueError,'ZERO UNQUERIED'):llm.validate_region(raw,pool,state)
    def test_bad_deployed_prediction_strictly_reduces_coefficient(self):
        bank=e.new_bank(6);bank.update(mean=1.,last_round=5)
        prior=e.forecast(bank,6)
        after,m=e.update_bank(bank,prior,6,-746.0862380015271/13000,7358.386266250042/13000,
            1171.4172862184678/13000,7258.288591640539/13000,True)
        self.assertLess(m['actual_paired_SE_gain'],0.);self.assertLess(after['mean'],prior['mean'])
        self.assertAlmostEqual(m['innovation'],m['observed_y']-m['actual_deployed_mean_y'])
    def test_negative_gain_property_and_no_hidden_unbounded_state(self):
        rng=np.random.default_rng(0);checked=0
        for _ in range(300):
            bank=e.new_bank(1);bank['mean']=float(rng.random());prior=e.forecast(bank,1)
            after,m=e.update_bank(bank,prior,1,float(rng.normal(0,.1)),.2,.03,float(rng.normal(.2,.2)),True)
            self.assertTrue(0<=after['mean']<=1)
            if m['actual_paired_SE_gain']< -1e-12:
                self.assertLessEqual(after['mean'],prior['mean']+1e-12);checked+=1
        self.assertGreater(checked,50)
    def test_closed_gate_does_not_count_shadow_as_deployed_success(self):
        bank=e.new_bank(1);prior=e.forecast(bank,1)
        after,m=e.update_bank(bank,prior,1,.1,.2,.03,.8,False)
        self.assertEqual(after['mean'],prior['mean']);self.assertEqual(after['informative_updates'],0)
        self.assertEqual(m['actual_paired_SE_gain'],0.);self.assertEqual(m['feature'],0.)
        self.assertAlmostEqual(after['variance'],prior['variance'])
    def test_late_allowance_is_preserved(self):
        state={'calls':[{'iteration':1},{'iteration':3}]}
        for t in (5,6,7,8):
            p=e.refresh_permission(state,t);self.assertFalse(p['allowed']);self.assertEqual(p['reason'],'preserve_at_least_one_late_refresh')
        self.assertTrue(e.refresh_permission(state,9)['allowed'])
        state['calls'].append({'iteration':9});self.assertFalse(e.refresh_permission(state,11)['allowed'])
    def test_exact_and_scattered_joint_options_respect_gate(self):
        for task in ('SnAr','HPLC'):
            pool=old.pool(task);ready=old.data.read(old.ROOT/'initialization'/task/'ready.json');state={**ready,'interaction_tests':[]}
            cache=type('FeatureCache',(),{})();cache.pool=pool;cache.ix={k:i for i,k in enumerate(pool['ids'])}
            cache.X=e.old.torch.tensor(pool['z'],dtype=e.old.torch.double)
            n=len(pool['ids']);available=~np.isin(pool['ids'],[h['candidate_id'] for h in ready['history']])
            eligible=available.copy();eligible[::2]=False
            d={'available':available,'eligible':eligible,'tie_priority':e.old.priority(n,2,0),'base':np.linspace(0,1,n),
               'reference':int(np.flatnonzero(available)[-1]),'budget':2.,'shift_std':np.zeros(n),
               'stats':{'mu_y':np.zeros(n),'sd_y':np.ones(n)*.1,'std':.1}}
            prepared=e.interaction_candidates(cache,state,d,2);self.assertGreater(len(prepared['candidates']),0)
            for row in prepared['candidates']:
                self.assertTrue(eligible[cache.ix[row['candidate_id']]])
                if task=='SnAr':self.assertEqual(len(row['all_changed_axes']),2)
                else:
                    self.assertEqual(row['geometry'],'matched_background_joint_conditions')
                    self.assertLessEqual(row['background_RMS_normalized'],.2)
                    self.assertGreater(len(row['all_changed_axes']),2)

if __name__=='__main__':unittest.main()
