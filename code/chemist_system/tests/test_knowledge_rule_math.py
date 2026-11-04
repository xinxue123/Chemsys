"""Independent-integral and contract checks; no GP fitting, API, or oracle."""
from copy import deepcopy
from hashlib import sha256
import json
from math import exp, log, sqrt
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import log_ndtr, ndtr, xlogy
from scipy.stats import norm

from chem.benchmarks import knowledge_rule_math as m


def prediction(mu=.5, vF=.04, vY=.05, reference=.55, best=.6, **overrides):
    values = m.vector_scores(mu, vF, vY, reference, best)
    args = dict(hyp_id='H001', version=1, point_id='new-plus', t=1,
                mu=mu, vF=vF, vY=vY, reference_id='observed-reference',
                reference_y=reference, tempering=float(values['tempering']),
                log_tempering=float(values['log_tempering']), semantic_key='rule-semantic-key')
    args.update(overrides)
    return m.freeze_prediction(**args)


def direct_integrals(mu, vF, vY, reference, best, tau=.06):
    """Integrate directly over F, unlike implementation's Z quantile GH."""
    sd = sqrt(vF)
    alpha = (mu-reference)/sqrt(vY+tau*tau)
    prior = ndtr(alpha)
    lower = (best-mu)/sd
    denominator = quad(lambda z: (mu+sd*z-best)*norm.pdf(z), lower, np.inf,
                       epsabs=1e-16, epsrel=2e-11, limit=150)[0]
    numerator = quad(lambda z: (mu+sd*z-best)*norm.pdf(z)
                     *ndtr((mu+sd*z-reference)/sqrt(vY-vF+tau*tau))/prior,
                     lower, np.inf, epsabs=1e-16, epsrel=2e-11, limit=150)[0]
    return denominator, numerator


class KnowledgeRuleMathTests(unittest.TestCase):
    def test_ei_ratios_against_independent_f_integral(self):
        cases = [(.5, .04, .05, .55, .6), (.1, .01, .015, .5, .65),
                 (.7, .003, .004, .5, .8), (.5, .04, .05, .5, .95),
                 (.1, .1, .11, .5, 2.)]
        for args in cases:
            with self.subTest(args=args):
                values = m.vector_scores(*args)
                base, conditional = direct_integrals(*args)
                self.assertAlmostEqual(exp(float(values['base_normal_log_ei']))/base, 1., places=8)
                ratio = conditional/base
                self.assertAlmostEqual(exp(float(values['pure_log_ei_ratio']))/ratio, 1., places=6)
                self.assertGreaterEqual(float(values['offset']), 0.)
                self.assertLessEqual(float(values['offset']), log(4.))
                t = float(values['tempering'])
                # Verify the actual tempered *density's* EI, not just a
                # clipped acquisition number produced by the same code.
                actual_ratio = (1-t)+t*ratio
                self.assertAlmostEqual(actual_ratio/exp(float(values['offset'])), 1., places=6)
                for weight in (0., .2, .5):
                    deployed_ratio = (1-weight)+weight*actual_ratio
                    self.assertAlmostEqual(deployed_ratio, 1+weight*t*(ratio-1), places=10)

    def test_tempering_is_exact_mixture_not_mean_shift_clipping(self):
        values = m.vector_scores(.1, .01, .015, .5, .65)
        ratio = exp(float(values['pure_log_ei_ratio']))
        t = float(values['tempering'])
        self.assertGreater(ratio, 4.)
        self.assertAlmostEqual(t, 3/(ratio-1), places=13)
        self.assertEqual(float(values['offset']), log(4.))
        pred = prediction(.1, .01, .015, .5, .65)
        for y in (.2, .5, .8):
            pure = float(log_ndtr((y-.5)/.06)-log_ndtr((.1-.5)/sqrt(.015+.06**2)))
            expected = log((1-t)+t*exp(pure))
            self.assertAlmostEqual(m.score_prediction(pred, y), expected, places=12)

    def test_observation_density_normalization_and_moments(self):
        for args in [(.5, .04, .05, .55, .6), (.1, .01, .015, .5, .65)]:
            with self.subTest(args=args):
                pred = prediction(*args)
                mu, _, vy, _, _ = args
                sd = sqrt(vy)
                def integrand(z, degree):
                    y = mu+sd*z
                    return norm.pdf(z)*exp(m.score_prediction(pred, y))*y**degree
                mass = quad(lambda z: integrand(z, 0), -12., 12., epsabs=2e-11)[0]
                first = quad(lambda z: integrand(z, 1), -12., 12., epsabs=2e-11)[0]
                second = quad(lambda z: integrand(z, 2), -12., 12., epsabs=2e-11)[0]
                self.assertAlmostEqual(mass, 1., places=9)
                self.assertAlmostEqual(first, pred['knowledge_mean_y'], places=9)
                self.assertAlmostEqual(second-first*first, pred['knowledge_var_y'], places=9)

    def test_sign_information_is_binary_event_entropy_difference(self):
        def entropy(p):
            return -xlogy(p, p)-xlogy(1-p, 1-p)
        # Large vY/tau**2 checks that GH does not miss the narrow entropy
        # transition; continuous Gaussian MI would exceed log(2) here.
        for vy, reference in [(.001, .5), (.05, .55), (.2, .5), (.2, .75), (2., .5)]:
            with self.subTest(vy=vy, reference=reference):
                mu, tau = .5, .06
                got = float(m.vector_scores(mu, vy, vy, reference, .7)['sign_information'])
                prior_entropy = entropy(ndtr((mu-reference)/sqrt(vy+tau*tau)))
                conditional_entropy = quad(lambda z: entropy(ndtr((mu+sqrt(vy)*z-reference)/tau))*norm.pdf(z),
                                           -14., 14., epsabs=2e-12, limit=150)[0]
                self.assertAlmostEqual(got, prior_entropy-conditional_entropy, places=9)
                self.assertGreaterEqual(got, 0.)
                self.assertLessEqual(got, log(2.))

    def test_zero_latent_covariance_policy_is_bitwise_gp(self):
        for mu in (.2, .9):
            values = m.vector_scores(mu, 0., .02, .7, .8)
            self.assertEqual(float(values['offset']), 0.)
            self.assertEqual(float(values['tempering']), 0.)
            self.assertTrue(bool(values['noop']))
            self.assertEqual(float(values['tempered_mean_y']), mu)
            self.assertEqual(float(values['tempered_var_y']), .02)
            # The observation may still contain information about its
            # measurement noise; the explicit no-latent-effect policy
            # prevents earning credit by forecasting this noise alone.
            pred = prediction(mu, 0., .02, .7, .8)
            self.assertTrue(pred['zero_effect_policy'])
            self.assertIsNone(pred['log_tempering'])
            for y in (-10., .5, 10.):
                self.assertEqual(m.score_prediction(pred, y), 0.)
                self.assertEqual(m.deployed_score_prediction(pred, y, .5), 0.)
        with self.assertRaises(ValueError):
            prediction(.2, 0., .02, .7, .8, tempering=1., log_tempering=0.)

    def test_zero_tempering_and_zero_weight_are_exact_noops(self):
        pred = prediction(tempering=0., log_tempering=None)
        self.assertEqual(pred['knowledge_mean_y'], pred['gp_mean_y'])
        self.assertEqual(pred['knowledge_var_y'], pred['gp_var_y'])
        self.assertEqual(m.score_prediction(pred, .1), 0.)
        active = prediction()
        self.assertEqual(m.deployed_score_prediction(active, -.5, 0.), 0.)
        base = np.array([-.3, -.3, -.3])
        offsets = m.vector_scores([.2]*3, 0., .02, .7, .8)['offset']
        self.assertTrue(np.array_equal(base+offsets, base))

    def test_event_probability_bounds_ratio_and_prevents_certain_event_artifact(self):
        for mu, vf, vy, reference, best in [(.8, .01, .01, 0., 1.5),
                                           (.7, .001, .001, -.7, 1.),
                                           (.1, .01, .015, .5, .65)]:
            values = m.vector_scores(mu, vf, vy, reference, best)
            self.assertLessEqual(float(values['pure_log_ei_ratio']),
                                 -float(values['event_log_probability']))
        values = m.vector_scores(.7, .001, .001, -.7, 1.)
        self.assertEqual(float(values['offset']), 0.)
        self.assertTrue(bool(values['noop']))

    def test_extreme_tail_preserves_log_tempering_not_false_noop(self):
        values = m.vector_scores(-5., .01, .01, 1., 1.)
        self.assertGreater(float(values['pure_log_ei_ratio']), 745.)
        self.assertEqual(float(values['tempering']), 0.)  # Float underflow.
        self.assertFalse(bool(values['noop']))
        self.assertTrue(np.isfinite(values['log_tempering']))
        self.assertEqual(float(values['offset']), log(4.))
        pred = prediction(-5., .01, .01, 1., 1.)
        self.assertFalse(pred['zero_effect_policy'])
        for y in (-5., 1., 2.):
            self.assertTrue(np.isfinite(m.score_prediction(pred, y)))
        self.assertGreater(m.score_prediction(pred, 1.), 0.)
        # JSON does not encode -Infinity/NaN: a true zero-effect uses null.
        json.dumps(pred, allow_nan=False)
        json.dumps(prediction(.2, 0., .02, .7, .8), allow_nan=False)

    def test_deployed_predictor_uses_weight_times_tempering(self):
        pred = prediction(.1, .01, .015, .5, .65)
        for y in (-2., .4, .8):
            for weight in (.1, .5):
                t = pred['tempering']
                pure = float(log_ndtr((y-.5)/.06)-pred['event_log_probability'])
                expected = float(np.logaddexp(np.log1p(-weight*t), log(weight*t)+pure))
                got = m.deployed_score_prediction(pred, y, weight)
                self.assertAlmostEqual(got, expected, places=12)
                self.assertGreaterEqual(got, log(1-weight)-1e-12)

    def test_broadcast_permutation_and_identifier_relabel_invariance(self):
        mu = np.array([.2, .4, .6])
        result = m.vector_scores(mu, .02, .025, .5, .7)
        permutation = np.array([2, 0, 1])
        shuffled = m.vector_scores(mu[permutation], .02, .025, .5, .7)
        for field in ('offset', 'tempering', 'sign_information', 'tempered_mean_y'):
            np.testing.assert_array_equal(shuffled[field], result[field][permutation])
            for j in range(3):
                self.assertEqual(float(result[field][j]), float(m.vector_scores(mu[j], .02, .025, .5, .7)[field]))
        original = prediction()
        relabeled = prediction(hyp_id='中文规则', point_id='a-new-real-label', reference_id='another-label')
        self.assertNotEqual(original['prediction_sha256'], relabeled['prediction_sha256'])
        self.assertEqual(m.score_prediction(original, .63), m.score_prediction(relabeled, .63))

    def test_frozen_prediction_rejects_tampering_even_rehashed_diagnostics(self):
        pred = prediction()
        for field in ('reference_y', 'gp_mean_y', 'version', 'tempering'):
            altered = deepcopy(pred)
            altered[field] += .1
            with self.assertRaises(ValueError):
                m.score_prediction(altered, .7)
        altered = deepcopy(pred)
        altered['knowledge_mean_y'] += .1
        altered['prediction_sha256'] = sha256(json.dumps(
            {k: v for k, v in altered.items() if k != 'prediction_sha256'},
            sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')).hexdigest()
        with self.assertRaises(ValueError):
            m.score_prediction(altered, .7)
        self.assertEqual(pred, deepcopy(pred))

    def test_invalid_variances_inputs_and_probe_identity(self):
        bad = [(0., -.1, .2, .5, .6), (0., .2, .1, .5, .6),
               (np.nan, .1, .2, .5, .6), (.2, .1, np.inf, .5, .6)]
        for args in bad:
            with self.subTest(args=args), self.assertRaises(ValueError):
                m.vector_scores(*args)
        with self.assertRaises(ValueError):
            m.vector_scores(.5, .1, .2, .5, .6, tau=0.)
        for overrides in ({'point_id': 'observed-reference'}, {'t': 0}, {'version': True},
                          {'tempering': 1.2}, {'log_tempering': 1.}, {'log_tempering': -5.}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                prediction(**overrides)
        with self.assertRaises(ValueError):
            m.score_prediction(prediction(), np.nan)

    def test_logscore_is_not_the_directional_fact(self):
        # A failed factual comparison can nevertheless improve prediction
        # relative to an even more pessimistic GP. Policy must keep two
        # ledgers rather than labeling this score "rule confirmed".
        pred = prediction(.2, .04, .04, .6, .7)
        self.assertLess(.5, pred['reference_y'])
        self.assertGreater(m.score_prediction(pred, .5), 0.)
        # Conversely, beating the reference may be below what GP expected.
        pred = prediction(.8, .01, .01, .6, .85)
        self.assertGreater(.61, pred['reference_y'])
        self.assertLess(m.score_prediction(pred, .61), 0.)


if __name__ == '__main__':
    unittest.main()
