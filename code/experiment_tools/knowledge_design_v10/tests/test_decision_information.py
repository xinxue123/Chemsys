"""Numerical/design checks, deliberately not a benchmark-performance test."""
from math import exp, pi, sqrt
import unittest

import numpy as np
from scipy.integrate import quad

from knowledge_design_v10.decision_information import (
    expected_best_value_gain as kg, knowledge_probe_priority,
)


class DecisionInformationTests(unittest.TestCase):
    def test_two_equal_alternatives_analytic_value(self):
        self.assertAlmostEqual(kg([0, 0], [1, 0], 1), 1 / sqrt(2 * pi), places=14)
        self.assertAlmostEqual(kg([0, 0], [1, -1], 1), sqrt(2 / pi), places=14)

    def test_common_mode_uncertainty_does_not_invent_decision_information(self):
        self.assertEqual(kg([5, 3, -2], [100, 100, 100], 10000), 0.0)
        self.assertEqual(kg([5], [100], 10000), 0.0)

    def test_zero_information_and_invalid_degenerate_covariance(self):
        self.assertEqual(kg([1, 2], [0, 0], 0), 0)
        with self.assertRaises(ValueError):
            kg([1, 2], [0, 1], 0)

    def test_shift_scale_permutation_and_immutability(self):
        mu = np.array([0.2, 0.3, 0.1, 0.4])
        cov = np.array([0.1, 0.25, -0.01, 0.02])
        original_mu, original_cov = mu.copy(), cov.copy()
        value = kg(mu, cov, 0.3)
        self.assertAlmostEqual(kg(mu + 3, cov, 0.3), value, places=13)
        self.assertAlmostEqual(kg(mu * 7, cov * 49, 0.3 * 49), 7 * value, places=13)
        perm = [3, 1, 0, 2]
        self.assertAlmostEqual(kg(mu[perm], cov[perm], 0.3), value, places=14)
        np.testing.assert_array_equal(mu, original_mu)
        np.testing.assert_array_equal(cov, original_cov)

    def test_noise_reduces_information_for_fixed_latent_covariance(self):
        clean = kg([0.5, 0.4], [0.1, 0], 0.1)
        noisy = kg([0.5, 0.4], [0.1, 0], 0.5)
        self.assertGreater(clean, noisy)
        self.assertGreaterEqual(noisy, 0)

    def test_dominated_uncertain_point_is_less_useful_than_near_optimal_point(self):
        useful = kg([0.8, 0.79, -10], [0, 0.1, 0], 0.1)
        irrelevant = kg([0.8, 0.79, -10], [0, 0, 0.1], 0.1)
        self.assertGreater(useful, irrelevant)
        self.assertGreater(useful, 0.1)

    def test_independent_quadrature_on_random_valid_gaussian_posteriors(self):
        rng = np.random.default_rng(902)
        for _ in range(32):
            n = int(rng.integers(2, 10))
            means = rng.normal(size=n)
            basis = rng.normal(size=(n, n))
            covariance = basis @ basis.T / n
            j = int(rng.integers(n))
            variance = covariance[j, j] + 0.05
            slopes = covariance[:, j] / sqrt(variance)
            # Integrate max directly, without the production upper-envelope
            # algorithm. Pairwise crossings partition the nonsmooth integrand.
            cuts = []
            for a in range(n):
                for b in range(a):
                    if slopes[a] != slopes[b]:
                        z = (means[b] - means[a]) / (slopes[a] - slopes[b])
                        if -12 < z < 12:
                            cuts.append(float(z))
            integrand = lambda z: (np.max(means + slopes * z) - np.max(means)) * exp(-z * z / 2) / sqrt(2 * pi)
            expected, _ = quad(integrand, -12, 12, points=sorted(set(cuts)),
                               epsabs=2e-12, epsrel=2e-12, limit=200)
            self.assertAlmostEqual(kg(means, covariance[:, j], variance), expected, places=11)

    def test_paused_knowledge_cannot_deploy_and_no_value_means_no_override(self):
        self.assertEqual(knowledge_probe_priority(1, 0.1, 0), 0)
        self.assertEqual(knowledge_probe_priority(0.1, 1, 0.8), 0)
        self.assertAlmostEqual(knowledge_probe_priority(0.3, 0.1, 0.4), 0.08)

    def test_invalid_inputs_are_rejected(self):
        for mu, covariance, variance in [([], [], 1), ([1, 2], [1], 1),
                                         ([1, np.nan], [0, 0], 1), ([1], [0], -1),
                                         ([1], [np.inf], 1)]:
            with self.assertRaises(ValueError):
                kg(mu, covariance, variance)


if __name__ == '__main__':
    unittest.main()
