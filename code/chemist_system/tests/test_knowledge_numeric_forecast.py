"""Density/acquisition agreement and prospective integrity, no benchmark run."""
from copy import deepcopy
import math
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from chem.benchmarks import knowledge_numeric_forecast as m


def gaussian(mu,vf=.04,noise=.01):
    return dict(mean=mu,latent_variance=vf,observation_variance=vf+noise)


def frozen(**changes):
    null=gaussian(.2);expert=gaussian(.6,.06)
    args=dict(hypothesis_id='H001',version=1,point_id='plus',query_round=2,
        prefix_sha256='prefix',current_gp_sha256='gp',birth_snapshot_sha256='birth',semantic_key='scope',
        current=gaussian(.25),birth_null=null,raw_expert=expert,cut_feedback=gaussian(.3,.08),
        variance_only=dict(expert,mean=null['mean']),best=.5,beta_variance=.2,residual_feature=.4,
        birth_noise_variance=.01,reference_id='known',reference_y=.1,actual_calibrate=True)
    args.update(changes)
    return m.freeze_forecast(**args)


def normal_ei(mu,var,best):
    if var==0:return max(0.,mu-best)
    sd=math.sqrt(var);z=(mu-best)/sd
    return (mu-best)*norm.cdf(z)+sd*norm.pdf(z)


class NumericForecastTests(unittest.TestCase):
    def test_both_caps_match_the_actual_mixture_ei(self):
        for em in [2.,-2.,.2]:
            got=m.vector_forecast(.1,.04,.05,em,.04,.06,.5)
            ratio=normal_ei(em,.04,.5)/normal_ei(.1,.04,.5)
            lam=float(got['tempering'])
            expected=(1-lam)+lam*ratio
            self.assertAlmostEqual(math.exp(float(got['offset'])),expected,places=12)
            self.assertGreaterEqual(expected,.25-1e-13);self.assertLessEqual(expected,4.+1e-13)
        self.assertLess(float(m.vector_forecast(.1,.04,.05,-2.,.04,.06,.5)['offset']),0)

    def test_zero_expert_ei_is_negative_effect_not_noop(self):
        got=m.vector_forecast(.1,.04,.05,-1.,0.,.01,.5)
        self.assertEqual(float(got['raw_log_ratio']),-np.inf)
        self.assertFalse(bool(got['zero_effect_policy']))
        self.assertAlmostEqual(float(got['tempering']),.75)
        self.assertAlmostEqual(float(got['offset']),math.log(.25))

    def test_current_zero_and_identical_ei_are_exact_noops(self):
        for gf,em,ef in [(0.,4.,.1),(.04,.1,.04)]:
            got=m.vector_forecast(.1,gf,gf+.01,em,ef,ef+.01,.5)
            self.assertTrue(bool(got['zero_effect_policy']))
            self.assertEqual(float(got['offset']),0.)
            self.assertEqual(float(got['log_tempering']),-np.inf)
            self.assertEqual(float(got['tempered_mean_y']),.1)
            self.assertAlmostEqual(float(got['tempered_var_y']),gf+.01)

    def test_density_normalization_and_scores_match_direct_mixture(self):
        pred=frozen();models=pred['models'];lam=pred['forecast']['tempering'];y=.47
        def pdf(name,z):
            c=models[name]
            return norm.pdf(z,c['mean'],math.sqrt(c['observation_variance']))
        def mix(name,z):return (1-lam)*pdf('current',z)+lam*pdf(name,z)
        area=quad(lambda z:mix('raw_expert',z),-np.inf,np.inf,epsabs=1e-12)[0]
        self.assertAlmostEqual(area,1.,places=11)
        receipt=m.score_forecast(pred,y,expected_prediction_sha256=pred['prediction_sha256'])
        self.assertAlmostEqual(receipt['gp_gain'],math.log(mix('raw_expert',y)/pdf('current',y)),places=13)
        self.assertAlmostEqual(receipt['null_gain'],math.log(mix('raw_expert',y)/mix('birth_null',y)),places=13)
        exmean=pred['forecast']['tempered_mean_y'];g=models['current']
        self.assertAlmostEqual(receipt['mean_gain'],((y-g['mean'])**2-(y-exmean)**2)/(2*g['observation_variance']),places=13)
        nm=receipt['tempered_birth_null_mean'];nv=receipt['tempered_birth_null_observation_variance']
        self.assertAlmostEqual(receipt['null_mean_gain'],((y-nm)**2-(y-exmean)**2)/(2*nv),places=13)

    def test_wrong_mean_can_have_positive_density_gain_from_variance(self):
        g=gaussian(0.,.001,.001);null=deepcopy(g);expert=gaussian(-.01,.02,.001)
        pred=frozen(current=g,birth_null=null,raw_expert=expert,full_expert=expert,
                    cut_feedback=gaussian(.3,.08,.001),
                    variance_only=dict(expert,mean=0.),birth_noise_variance=.001,best=.01)
        result=m.score_forecast(pred,.2)
        self.assertGreater(result['gp_gain'],0.)
        self.assertLess(result['mean_gain'],0.)
        self.assertLess(result['null_mean_gain'],0.)

    def test_cut_is_actual_distribution_and_full_is_only_shadow(self):
        cut=gaussian(.35,.03);full=gaussian(.8,.08)
        pred=frozen(actual_calibrate=False,raw_expert=cut,cut_feedback=cut,full_expert=full,
                    variance_only=dict(cut,mean=.2))
        got=m.score_forecast(pred,.7)
        self.assertEqual(got['gp_gain'],got['cut_feedback_gp_gain'])
        self.assertNotEqual(got['gp_gain'],got['full_expert_gp_gain'])
        with self.assertRaises(ValueError):frozen(actual_calibrate=False)

    def test_hash_and_rehashed_derived_tampering_are_rejected(self):
        pred=frozen();original=pred['prediction_sha256']
        bad=deepcopy(pred);bad['point_id']='different'
        with self.assertRaises(ValueError):m.score_forecast(bad,.4)
        bad=deepcopy(pred);bad['forecast']['tempering']*=.5
        bad['prediction_sha256']=m._digest({k:v for k,v in bad.items() if k!='prediction_sha256'})
        with self.assertRaises(ValueError):m.score_forecast(bad,.4)
        changed=frozen(point_id='another')
        with self.assertRaises(ValueError):m.score_forecast(changed,.4,expected_prediction_sha256=original)

    def test_information_exact_gaussian_formula_and_degenerate_contract(self):
        got=m.calibration_information(.2,.4,.03,.01)
        self.assertAlmostEqual(float(got),.5*math.log1p(.2*.4**2/.04),places=14)
        self.assertEqual(float(m.calibration_information(0.,.4,0.,0.)),0.)
        self.assertEqual(float(m.calibration_information(.2,0.,0.,0.)),0.)
        with self.assertRaises(ValueError):m.calibration_information(.2,.4,0.,0.)
        with self.assertRaises(ValueError):m.calibration_information(-.2,.4,.03,.01)

    def test_observation_variance_and_frozen_noise_are_not_silently_repaired(self):
        for vf,vy in [(0.,0.),(.1,.09),(-.1,.1)]:
            with self.assertRaises(ValueError):m.vector_forecast(0.,vf,vy,.2,.02,.03,0.)
        with self.assertRaises(ValueError):frozen(birth_noise_variance=.02)
        with self.assertRaises(ValueError):frozen(current=gaussian(np.nan))
        with self.assertRaises(ValueError):m.score_forecast(frozen(),np.inf)
        with self.assertRaises(ValueError):frozen(variance_only=gaussian(.3,.06))
        with self.assertRaises(ValueError):frozen(cut_feedback=gaussian(.3,.08,.02))

    def test_same_variance_far_tail_score_avoids_subtracting_huge_log_densities(self):
        g=gaussian(0.,.04,.01);null=deepcopy(g);e=gaussian(.01,.04,.01)
        pred=frozen(current=g,birth_null=null,raw_expert=e,variance_only=dict(e,mean=0.),best=-1.)
        y=1e8
        gain=.01*(2*y-.01)/(.1)
        lam=pred['forecast']['tempering']
        expected=float(np.logaddexp(math.log1p(-lam) if lam<1 else -np.inf,math.log(lam)+gain))
        self.assertAlmostEqual(m.score_forecast(pred,y)['gp_gain'],expected,places=7)

    def test_noop_has_zero_credit_in_all_ledgers(self):
        pred=frozen(current=gaussian(.2,0.,.01))
        got=m.score_forecast(pred,.5)
        for key in ('gp_gain','null_gain','mean_gain','null_mean_gain','birth_null_gp_gain',
                    'cut_feedback_gp_gain','full_expert_gp_gain','variance_only_gp_gain'):
            self.assertEqual(got[key],0.)

    def test_tiny_positive_variance_and_extreme_cap_have_no_variance_floor(self):
        got=m.vector_forecast(0.,1e-16,2e-16,.01,1e-16,2e-16,.001)
        self.assertTrue(np.isfinite(got['log_tempering']))
        self.assertLess(float(got['log_tempering']),-1e6)
        self.assertFalse(bool(got['zero_effect_policy']))
        self.assertAlmostEqual(float(got['offset']),math.log(4.),places=6)
        self.assertGreaterEqual(float(got['tempered_var_y']),0.)

    def test_extreme_log_ratio_retains_cap_normalizer_in_two_parts(self):
        for vf in (1e-8,1e-10,1e-12,1e-14,1e-16):
            got=m.vector_forecast(0.,vf,vf+.001,1.,.1,.101,.8)
            self.assertAlmostEqual(float(got['offset']),math.log(4.),places=14)
            # If a density ratio equals the EI ratio, the same lambda must
            # produce exactly the same bounded mixture ratio, even at 1e15.
            score=m._mixture_gain(0.,float(got['raw_log_ratio']),float(got['log_tempering']),
                                  float(got['log_tempering_base']),float(got['log_tempering_remainder']))
            self.assertAlmostEqual(score,math.log(4.),places=14)

    def test_vectorization_and_inputs_are_immutable(self):
        mu=np.array([.1,.2]);copy=mu.copy()
        got=m.vector_forecast(mu,.04,.05,[.3,-1.],.04,.05,.5)
        self.assertEqual(got['offset'].shape,(2,))
        np.testing.assert_array_equal(mu,copy)
        pred=frozen();before=deepcopy(pred)
        m.score_forecast(pred,.4)
        self.assertEqual(pred,before)


if __name__=='__main__':unittest.main()
