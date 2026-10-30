"""Behavioral checks for prospective deployment and protected-prefix continuation."""
from copy import deepcopy
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
import numpy as np
from chem.benchmarks import continuous_r2_engine as e
from chem.benchmarks import continuous_r2_llm as llm
from chem.benchmarks import continuous_r2_study as s


def fixture():
    pool={'task':'fixture','ids':['p0','p1','p2','p3','p4'],'axes':['x'],
          'z':[[0.],[.25],[.5],[.75],[1.]]}
    spec={'allowed':{'x':[.2,.8]},'confidence':.7}
    key=e.old.trust.geometry_key(spec)
    bank=e.r1.new_bank(1);bank.update(mean=.1,variance=.12,last_round=1)
    state={'history':[{'candidate_id':'p0','y':.2}], 'knowledge':{'confidence':.7,'clauses':[{'id':'H1','axis':'x','bounds':[.2,.8]}]},
           'banks':{key:bank}, 'geometry_ledger':{key:{'region':spec,'negative_actual_count':1}},
           'region_trust':{},'calls':[],'pending':None,'active_suspended':False}
    return pool,state,spec,key


class ContinuousR2Tests(unittest.TestCase):
    def test_narrowing_cannot_reset_negative_actual_evidence(self):
        pool,state,spec,key=fixture();new={'allowed':{'x':[.4,.6]},'confidence':.7}
        bank,receipt=e.inherit_bank(pool,state,new,3)
        self.assertEqual(bank['mean'],.1);self.assertGreaterEqual(bank['variance'],.25)
        self.assertEqual(receipt['sources'][0]['overlap_of_smaller_set'],1.)
        self.assertEqual(receipt['sources'][0]['geometry_key'],key)

    def test_disjoint_region_and_shadow_only_evidence_do_not_inherit_failure(self):
        pool,state,spec,key=fixture();new={'allowed':{'x':[.9,1.]},'confidence':.7}
        bank,receipt=e.inherit_bank(pool,state,new,3)
        self.assertEqual(bank['mean'],.5);self.assertEqual(receipt['sources'],[])
        state['geometry_ledger'][key]['negative_actual_count']=0
        bank,receipt=e.inherit_bank(pool,state,{'allowed':{'x':[.4,.6]},'confidence':.7},3)
        self.assertEqual(bank['mean'],.5);self.assertEqual(receipt['sources'],[])

    def test_existing_geometry_reuses_actual_bank_without_renaming_reset(self):
        pool,state,spec,key=fixture();bank,receipt=e.inherit_bank(pool,state,spec,3)
        self.assertEqual(bank,state['banks'][key]);self.assertTrue(receipt['existing_geometry'])

    def staged(self,mean=.1):
        pool,state,spec,key=fixture();state['banks'][key]['mean']=mean
        knowledge={'confidence':.7,'clauses':[{'id':'H1','axis':'x','bounds':[.4,.6]}]}
        answer={'plan':{'knowledge':knowledge}}
        e.stage_refresh(pool,state,3,answer,'context',['bad_predictions'],{'allowed':True})
        d={'selected':0,'reference':0,'gate_open':False,'stats':{'mu_y':np.array([.2]),'std':1.}}
        frozen={'region':state['pending']['region'],'raw_shift_std':np.array([.1]),
                'shadow_probe_coefficient':mean if mean>0 else .5,'inherited_deployment_coefficient':mean}
        return pool,state,d,frozen

    def test_refresh_does_not_replace_active_knowledge_before_feedback(self):
        pool,state,d,frozen=self.staged();self.assertTrue(state['active_suspended'])
        self.assertEqual(state['knowledge']['clauses'][0]['bounds'],[.2,.8])
        self.assertEqual(state['pending']['knowledge']['clauses'][0]['bounds'],[.4,.6])
        self.assertEqual(state['pending']['initial_bank']['mean'],.1)
        self.assertIsNone(e.promote(state,3))

    def test_positive_shadow_requires_later_round_and_does_not_raise_actual_KF(self):
        pool,state,d,frozen=self.staged();before=deepcopy(state['banks'])
        check=e.finish_pending(state,frozen,d,3,.4,'p0')
        self.assertTrue(check['positive_evidence']);self.assertEqual(state['banks'],before)
        with self.assertRaises(AssertionError):e.promote(state,3)
        promotion=e.promote(state,4);self.assertEqual(promotion['query'],4)
        key=e.old.trust.geometry_key(frozen['region'])
        self.assertEqual(state['banks'][key]['mean'],.1);self.assertFalse(state['active_suspended'])

    def test_failed_shadow_stays_suspended_and_zero_inheritance_stays_zero(self):
        pool,state,d,frozen=self.staged();e.finish_pending(state,frozen,d,3,.1,'p0')
        self.assertEqual(state['pending']['status'],'rejected');self.assertIsNone(e.promote(state,4))
        self.assertTrue(state['active_suspended'])
        pool,state,d,frozen=self.staged(0.);e.finish_pending(state,frozen,d,3,.4,'p0');e.promote(state,4)
        self.assertEqual(state['banks'][e.old.trust.geometry_key(frozen['region'])]['mean'],0.)

    def test_subset_cannot_restore_from_one_positive_shadow_or_negative_latest(self):
        state={'subset_evidence':{'g':{'observations':[{'paired_SE_gain':1.}]}}};d={'bank_key':'g'}
        self.assertFalse(e.subset_permission(state,d)['allowed'])
        state['subset_evidence']['g']['observations'].append({'paired_SE_gain':2.})
        self.assertTrue(e.subset_permission(state,d)['allowed'])
        state['subset_evidence']['g']['observations'].append({'paired_SE_gain':-.1})
        self.assertFalse(e.subset_permission(state,d)['allowed'])

    def test_optional_joint_may_keep_BO_outside_joint_pool(self):
        raw={'action':'keep_BO','candidate_id':'BO','prediction':100.,'reason':'preserve the acquisition opportunity','support':[1]}
        state={'history':[{'candidate_id':'a','y':.2}]};prepared={'candidates':[]}
        plan=llm.validate_joint(raw,prepared,state,{'candidate_id':'BO'})
        self.assertEqual(plan['geometry'],'BO_abstention')
        raw['candidate_id']='notBO'
        with self.assertRaises(ValueError):llm.validate_joint(raw,prepared,state,{'candidate_id':'BO'})

    def test_both_variants_preserve_HPLC_first_three_full_transitions(self):
        source=s.data.read(s.parent.ROOT/'runs/HPLC_New_Agent_R1/state.json')
        temporary_root=Path('outputs/r2_behavioral_tmp').resolve()
        temporary_root.mkdir(parents=True,exist_ok=True)
        for method in e.METHODS:
            state=s.initial('HPLC',method,{'protocol_sha256':'behavioral_test'})
            with tempfile.TemporaryDirectory(dir=temporary_root) as tmp, redirect_stdout(io.StringIO()):
                self.assertTrue(Path(tmp).resolve().is_relative_to(temporary_root))
                path=Path(tmp)/'state.json'
                for t in range(1,4):s.bootstrap(state,path,t)
            self.assertEqual(state['history'],source['history'][:7])
            for actual,expected in zip(state['decisions'],source['decisions'][:3]):
                for name in ['selected_id','measurement','bank_before','bank_prior','bank_after','knowledge','region_trust_after']:
                    self.assertEqual(actual[name],expected[name])
            self.assertEqual(len(state['calls']),1)
            self.assertAlmostEqual(state['decisions'][-1]['incumbent_after']*10000,2317.81641)

    def test_real_prefix_refresh_round_executes_BO_and_keeps_old_active_knowledge(self):
        s.old.warm()
        temporary_root=Path('outputs/r2_behavioral_tmp').resolve()
        temporary_root.mkdir(parents=True,exist_ok=True)
        for task,t in [('SnAr',3),('HPLC',4)]:
            source=s.data.read(s.parent.ROOT/'runs'/(task+'_New_Agent_R1')/'state.json')
            state=s.initial(task,e.METHODS[0],{'protocol_sha256':'behavioral_test'})
            with tempfile.TemporaryDirectory(dir=temporary_root) as tmp, redirect_stdout(io.StringIO()):
                self.assertTrue(Path(tmp).resolve().is_relative_to(temporary_root))
                for query in range(1,t):s.bootstrap(state,Path(tmp)/'state.json',query)
            old_knowledge=deepcopy(state['knowledge'])
            saved=next(c for c in source['calls'] if c['iteration']==t)
            cache=e.Cache(s.frozen.pool(task))
            def no_joint(*args):raise AssertionError('No joint request is due at this round')
            plan=s.decide(state,cache,t,lambda *args:(saved['answer'],'fixture_context'),no_joint)
            self.assertEqual(plan['d']['selected'],plan['d']['reference'])
            self.assertFalse(plan['d']['gate_open']);self.assertEqual(plan['d']['amplitude'],0.)
            self.assertEqual(state['knowledge'],old_knowledge)
            self.assertIsNotNone(plan['pending_forecast']);self.assertTrue(state['active_suspended'])
            self.assertEqual(cache.counters['fits'],0)


if __name__=='__main__':unittest.main()
