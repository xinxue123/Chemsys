"""Synthetic independent-auditor contracts; no API, oracle, or model fitting."""
import copy
import math
from pathlib import Path
import unittest
from tempfile import TemporaryDirectory

import numpy as np
import audit_knowledge_numeric_v8 as audit


class NumericAuditTests(unittest.TestCase):
    def protocol(self):
        p=dict(tasks=list(audit.TASKS),inputs=[dict(task=t) for t in audit.TASKS],initial=4,queries=16,seed=0)
        p['protocol_sha256']=audit.digest(p)
        paths=[Path(t+'_'+a+'_agent_initial_terra/state.json') for t in audit.TASKS for a in ('bo','full')]
        return p,paths

    def test_complete_manifest_and_missing_run_guard(self):
        p,paths=self.protocol();audit.manifest_guard(p,paths)
        with self.assertRaisesRegex(AssertionError,'missing'):audit.manifest_guard(p,paths[:-1])

    def test_manifest_hash_and_duplicate_guard(self):
        p,paths=self.protocol()
        with self.assertRaisesRegex(AssertionError,'duplicate'):audit.manifest_guard(p,paths+[paths[0]])
        p['seed']=1
        with self.assertRaisesRegex(AssertionError,'hash'):audit.manifest_guard(p,paths)

    def test_pending_contract_never_passes(self):
        checks={k:'PASS' for k in audit.REQUIRED_CHECKS}
        self.assertEqual(audit.status_from_checks(checks),'PASS_LISTED_CHECKS')
        del checks['birth_conditioning']
        self.assertEqual(audit.status_from_checks(checks),'PENDING_UNAUDITED_CONTRACTS')
        self.assertEqual(audit.status_from_checks(checks,['bad']),'FAIL')

    def test_development_gate_does_not_promote_no_weighted_choice(self):
        rows=[dict(task=t,mode=a,status='COMPLETE',SearchAUC=1.+(a=='full'),selection_modes={}) for t in audit.TASKS for a in ('bo','full')]
        self.assertFalse(audit.stage_gate(rows)['trigger_next_stage'])
        rows[1]['selection_modes']={'knowledge_guided':1}
        self.assertTrue(audit.stage_gate(rows)['trigger_next_stage'])
        self.assertTrue(audit.stage_gate(rows)['technical_PASS_required_separately'])
        self.assertIsNone(audit.stage_gate(rows[:-1])['trigger_next_stage'])

    def test_signed_tempering_cap_and_exact_noop(self):
        values=audit.independent_tempering(np.zeros(4),[1.,1.,1.,0.],[0.,10.,-10.,1.],[1.,1.,1.,1.],0.)
        np.testing.assert_allclose(values['offset'],[0.,math.log(4),math.log(.25),0.],atol=1e-12)
        self.assertEqual(values['tempering'][0],0.)
        self.assertEqual(values['tempering'][3],0.)
        self.assertGreater(values['tempering'][1],0.)
        self.assertLess(values['tempering'][1],1.)
        self.assertAlmostEqual(values['tempering'][2],.75)

    def test_scalar_tempering_and_invalid_variance(self):
        self.assertEqual(float(audit.independent_tempering(0.,1.,0.,1.,0.)['offset']),0.)
        with self.assertRaisesRegex(AssertionError,'variance'):audit.independent_tempering(0.,-1.,0.,1.,0.)

    def test_independent_birth_matches_augmented_covariance(self):
        mean=np.array([.2,.3,.4]);feature=np.array([0.,.5,1.]);C=np.array([[.4,.1,.05],[.1,.5,.2],[.05,.2,.6]])
        m,P,noise=.2,.3,.04;z=[1];q=[2];y=[.8]
        result=audit.independent_birth_condition(mean,np.diag(C),feature,m,P,noise,z,q,y,C[np.ix_(z,z)],C[np.ix_(q,z)])
        joint=C+P*np.outer(feature,feature);v=joint[2,1]/(joint[1,1]+noise)
        expected=mean[2]+m*feature[2]+v*(y[0]-mean[1]-m*feature[1])
        self.assertAlmostEqual(result['full']['mean'][0],expected)
        self.assertAlmostEqual(result['full']['latent_variance'][0],joint[2,2]-v*joint[1,2])
        self.assertLess(result['beta_variance'],P)

    def test_cut_feedback_preserves_prior_and_actual_shadow_variance(self):
        args=([0.,0.],[1.,1.],[0.,1.],.2,.4,.1,[0],[1],[.6],np.array([[1.]]),np.array([[.2]]))
        r=audit.independent_birth_condition(*args,calibrate=False)
        self.assertEqual(r['beta_mean'],.2);self.assertEqual(r['beta_variance'],.4)
        np.testing.assert_array_equal(r['actual']['mean'],r['cut_feedback']['mean'])
        np.testing.assert_array_equal(r['variance_only']['latent_variance'],r['actual']['latent_variance'])

    def test_covariance_diagonal_and_order_fail_closed(self):
        with self.assertRaisesRegex(AssertionError,'diagonal'):
            audit.independent_birth_condition([0.,0.],[1.,1.],[0.,1.],0.,1.,.1,[0],[1],[0.],[[.9]],[[.1]])

    def test_four_scores_independent_density_formula(self):
        g=dict(mean=0.,latent_variance=1.,observation_variance=1.1)
        e=dict(mean=.2,latent_variance=1.2,observation_variance=1.3)
        n=dict(mean=.1,latent_variance=.8,observation_variance=.9)
        pred=dict(models=dict(current=g,raw_expert=e,birth_null=n,cut_feedback=e,variance_only=dict(e,mean=.1),full_expert=e),forecast=dict(log_tempering=math.log(.4)))
        score=audit.independent_four_scores(pred,.7)
        def density(d):return math.exp(-.5*(.7-d['mean'])**2/d['observation_variance'])/math.sqrt(2*math.pi*d['observation_variance'])
        self.assertAlmostEqual(score['gp_gain'],math.log(.6+.4*density(e)/density(g)))
        self.assertAlmostEqual(score['null_gain'],math.log((.6*density(g)+.4*density(e))/(.6*density(g)+.4*density(n))))
        pred['forecast']['log_tempering']=None
        self.assertEqual(audit.independent_four_scores(pred,.7)['gp_gain'],0.)

    def test_evidence_deduplicates_one_point_and_preserves_negative_bound(self):
        gains={k:.5 for k in audit.GAINS};own=dict(query_round=1,point_id='a',**gains)
        inherited=dict(query_round=1,point_id='a',applicable=True,**{k:-.2 for k in audit.GAINS})
        h=dict(ledger=dict(events=[own],trusted_weight=.4),inherited_negative_evidence=[inherited,copy.deepcopy(inherited)])
        s=audit.evidence_summary(h)
        self.assertEqual(s['effective_scores']['gp_gain'],-.2);self.assertEqual(s['trusted_weight'],0.)

    def test_any_failed_utility_cannot_restore_credit(self):
        gains={k:1. for k in audit.GAINS};gains['null_mean_gain']=-1e-12
        self.assertEqual(audit.recovery_weight(.1,.4,gains),(.1,0.))
        gains['null_mean_gain']=1e-9
        after,limit=audit.recovery_weight(.1,.4,gains)
        self.assertLess(after,.100000001);self.assertAlmostEqual(after-.1,limit)

    def selector(self):
        h=dict(id='H1',version=1,status='active',ledger=dict(events=[],trusted_weight=0.),inherited_negative_evidence=[])
        c=dict(query=1,epsilon=.4,hypotheses_at_commit=[h])
        arrays=dict(base_logei=np.array([0.,-.02,-.03]),eligible=np.ones(3,bool),tie_priority=np.array([2,1,0]))
        rows=[dict(id='H1',version=1,weight=0.,can_change_rank_at_cap=True,eligible_inside_scope=3)]
        calc={('H1',1):dict(indices=np.arange(3),positions={0:0,1:1,2:2},offsets=np.array([.1,.2,.3]),information=np.array([.01,.05,.1]))}
        params=dict(maximum_deployment_weight=.5,last_verification_round=10,maximum_optional_verifications=1,verification_maximum_gap=math.log(1.1),minimum_incremental_information=.01)
        return c,arrays,rows,calc,params

    def test_optional_global_information_argmax_and_quota(self):
        c,a,r,v,p=self.selector()
        result=audit.independent_selection(c,a,r,v,'full',0,p)
        self.assertEqual((result['selected'],result['mode']),(2,'optional_verification'))
        self.assertEqual(audit.independent_selection(c,a,r,v,'full',1,p)['selected'],0)
        c['query']=11
        self.assertEqual(audit.independent_selection(c,a,r,v,'full',0,p)['selected'],0)

    def test_optional_negative_credit_and_gap_gates(self):
        c,a,r,v,p=self.selector();c['hypotheses_at_commit'][0]['ledger']['events']=[dict(query_round=1,point_id='past',**{k:-.01 for k in audit.GAINS})]
        self.assertEqual(audit.independent_selection(c,a,r,v,'full',0,p)['selected'],0)
        c['hypotheses_at_commit'][0]['ledger']['events']=[];a['base_logei'][1:]=-.2
        self.assertEqual(audit.independent_selection(c,a,r,v,'full',0,p)['selected'],0)

    def test_negative_expert_cannot_route_to_uncovered_point(self):
        c,a,r,v,p=self.selector();r[0]['weight']=.5
        v['H1',1].update(indices=np.array([0]),offsets=np.array([-1.,0.,0.]))
        result=audit.independent_selection(c,a,r,v,'full',1,p)
        self.assertEqual(result['selected'],0)
        self.assertEqual(result['fallback'],'uncovered_or_no_effect_weighted_winner')

    def test_verification_arm_never_deploys_weight(self):
        c,a,r,v,p=self.selector();r[0]['weight']=.5
        self.assertEqual(audit.independent_selection(c,a,r,v,'verification',0,p)['selected'],0)
        self.assertEqual(audit.independent_selection(c,a,r,v,'verification',0,p)['deployed'],[])

    def test_pre_call_empty_scope_is_preserved_without_birth_read(self):
        pool=dict(ids=['a','b'],axes=['x'],kinds=['continuous'],features=[(0.,),(1.,)])
        h=dict(id='H1',version=1,status='active',scope=dict(axes=dict(x=dict(kind='continuous',bounds=[0.,.5]))),ledger=dict(events=[],trusted_weight=0.))
        birth=audit.BirthAudit(Path('unused'),Path('unused'),pool,{})
        a=dict(base_logei=np.array([0.,1.]),eligible=np.array([False,True]),tie_priority=np.array([0,1]))
        rows=birth.pre_call_coverage([h],[],a,True)
        self.assertEqual(rows[0]['eligible_inside_scope'],0);self.assertFalse(rows[0]['can_change_rank_at_cap'])

    def test_pre_call_missing_old_snapshot_block_fails_closed(self):
        pool=dict(ids=['a','b'],axes=['x'],kinds=['continuous'],features=[(0.,),(1.,)])
        h=dict(id='H1',version=1,status='active',scope=dict(axes=dict(x=dict(kind='continuous',bounds=[0.,1.]))),ledger=dict(events=[],trusted_weight=0.))
        birth=audit.BirthAudit(Path('unused'),Path('unused'),pool,{})
        birth.snapshot=lambda *_:dict(meta=dict(snapshot_sha256='frozen'),arrays={},folder=Path('definitely_absent_birth_audit_folder'))
        a=dict(base_logei=np.array([0.,1.]),eligible=np.array([False,True]),tie_priority=np.array([0,1]))
        with self.assertRaises(FileNotFoundError):birth.pre_call_coverage([h],[],a,True)


if __name__=='__main__':unittest.main()
