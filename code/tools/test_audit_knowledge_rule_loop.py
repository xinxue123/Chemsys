"""Synthetic tests for the independent V4 auditor; no production imports."""
from copy import deepcopy
import unittest

import numpy as np

import audit_knowledge_rule_loop as a


def example():
    pool = dict(ids=['ref-low', 'ref-high', 'new-a', 'new-b', 'wrong-category'],
        ix={'ref-low': 0, 'ref-high': 1, 'new-a': 2, 'new-b': 3, 'wrong-category': 4},
        axes=['temperature', 'catalyst'], kinds=['continuous', 'categorical'],
        features=[(.1, 'L-A'), (.2, 'L-A'), (.5, 'L-A'), (.7, 'L-A'), (.8, 'L-Z')])
    h = dict(id='H001', version=1, status='active', focus={'temperature': {'direction': 'increase'}},
        scope={'axes': {'temperature': {'kind': 'continuous', 'bounds': [0., 1.]},
                        'catalyst': {'kind': 'categorical', 'allowed': ['L-A']}}},
        categorical_preference={}, inherited_negative_evidence=[],
        ledger=dict(prediction_score=0., score=0., successes=0, failures=0,
                    trusted_weight=0., weight=0., events=[], seen=[]))
    h['semantic_key'] = a.digest(a.contract(h))
    prefix = [dict(candidate_id='ref-low', y=.4), dict(candidate_id='ref-high', y=.6)]
    return pool, h, prefix


class IndependentRuleAuditTests(unittest.TestCase):
    def test_quad_validates_gh_including_far_tail_and_low_threshold(self):
        cases = [(.5, .04, .05, .55, .6), (.1, .01, .015, .5, .65),
                 (-5., .01, .01, 1., 1.), (.9, .0001, .0001, .8, .5),
                 (.9, 1e-8, 1e-8, .8, .5)]
        for args in cases:
            with self.subTest(args=args):
                r = a.math_fields(*args)
                self.assertLess(a.adaptive_ei_check(*args, float(r['pure_log_ei_ratio'])), 2e-5)
        with self.assertRaises(AssertionError):
            a.adaptive_ei_check(*cases[0], 1.2)

    def test_reference_is_highest_known_compatible_and_id_invariant(self):
        pool, h, prefix = example()
        eligible = np.array([False, False, True, True, True])
        got = a.probes(pool, h, prefix, eligible)
        self.assertEqual(set(got), {2, 3})
        self.assertEqual({r['reference_id'] for r in got.values()}, {'ref-high'})
        tied = [dict(r, y=.6) for r in prefix]
        got = a.probes(pool, h, tied, eligible)
        self.assertEqual({r['reference_id'] for r in got.values()}, {'ref-low'})
        renamed = deepcopy(pool)
        renamed['ids'][0], renamed['ids'][1] = 'zzz', 'aaa'
        renamed['ix'] = {name: j for j, name in enumerate(renamed['ids'])}
        tied = [dict(candidate_id='zzz', y=.6), dict(candidate_id='aaa', y=.6)]
        self.assertEqual({r['reference_id'] for r in a.probes(renamed, h, tied, eligible).values()}, {'zzz'})

    def test_wrong_direction_or_unobserved_reference_is_not_executable(self):
        pool, h, prefix = example()
        h['focus']['temperature']['direction'] = 'decrease'
        h['semantic_key'] = a.digest(a.contract(h))
        self.assertEqual(a.probes(pool, h, prefix, np.array([False, False, True, True, True])), {})
        self.assertEqual(a.probes(pool, h, [], np.ones(5, bool)), {})

    def test_same_physical_negative_comparison_does_not_double_count(self):
        _, h, _ = example()
        common = dict(query_round=1, point_id='new-a', reference_id='ref-high',
                      fact_success=False, applicable=True)
        h['inherited_negative_evidence'] = [dict(common, delta_log_score=-.2), dict(common, delta_log_score=-.6)]
        summary = a.evidence(h)
        self.assertEqual(summary['failures'], 1)
        self.assertEqual(summary['effective_prediction_score'], -.6)
        self.assertEqual(summary['trusted_weight'], 0.)

    def test_tempered_logscore_has_normalized_density_and_half_gp_bound(self):
        from scipy.integrate import quad
        from scipy.stats import norm
        mu, vf, vy, ref, best = .1, .01, .015, .5, .65
        fields = a.math_fields(mu, vf, vy, ref, best)
        pred = dict(gp_mean_y=mu, gp_var_y=vy, reference_y=ref, preference_noise_sd=.06,
            tempering=float(fields['tempering']), log_tempering=float(fields['log_tempering']),
            event_log_probability=float(fields['event_log_probability']))
        mass = quad(lambda z: norm.pdf(z)*np.exp(a.score(pred, mu+np.sqrt(vy)*z)), -12, 12)[0]
        self.assertAlmostEqual(mass, 1., places=9)
        for y in (-10., .3, .8):
            mixed = np.logaddexp(np.log(.5), np.log(.5)+a.score(pred, y))
            self.assertGreaterEqual(mixed, -np.log(2)-1e-12)

    def test_mutated_semantic_contract_rejected(self):
        pool, h, prefix = example()
        h['scope']['axes']['temperature']['bounds'][1] = .8
        with self.assertRaises(AssertionError):
            a.probes(pool, h, prefix, np.array([False, False, True, True, False]))


if __name__ == '__main__':
    unittest.main()
