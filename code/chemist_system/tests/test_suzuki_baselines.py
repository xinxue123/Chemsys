"""Numerical identities and custody gates, not additional benchmark tasks."""
from copy import deepcopy

import numpy as np
import pytest
from scipy.stats import norm
from threadpoolctl import threadpool_limits

from chem.benchmarks.suzuki_domain import (CATALYSTS, NUMERIC, encode, initial_design,
    validate_point, validate_region, region_grid, same_point)
from chem.benchmarks.suzuki_policies import MixedGP, justify_select, lgbo_select, log_expected_improvement
from chem.benchmarks.suzuki_runner import validate_response


def test_domain_encoding_has_valid_one_hot_categories():
    rows = initial_design(1, 0)
    x = encode(rows)
    assert x.shape == (4, 11)
    assert (x[:, 3:].sum(axis=1) == 1).all()
    assert len({r['catalyst'] for r in rows}) == 4
    assert rows == initial_design(1, 0)
    assert rows != initial_design(1, 1)
    for name, value in [('catalyst', 'unknown'), ('temperature', 111), ('t_res', float('nan'))]:
        bad = dict(rows[0], **{name: value})
        with pytest.raises(ValueError):
            validate_point(bad)


def test_regional_grid_uses_equal_categorical_mass():
    region = {'catalysts': list(CATALYSTS[:3]), 't_res': [200, 400],
              'temperature': [70, 100], 'catalyst_loading': [1, 2]}
    rows, a = region_grid(region, 51)
    assert len(rows) == 512 and a.sum() == pytest.approx(1)
    for c in region['catalysts']:
        assert sum(w for r, w in zip(rows, a) if r['catalyst'] == c) == pytest.approx(1/3)
    for r in rows:
        validate_point(r)
        assert all(region[n][0] <= r[n] <= region[n][1] for n in NUMERIC)
    with pytest.raises(ValueError):
        validate_region(dict(region, catalysts=[]))


@pytest.fixture(scope='module')
def gp():
    # A covariance-identity fixture on valid Suzuki coordinates, not an optimization study.
    with threadpool_limits(1):
        return MixedGP(initial_design(1, 0), [.1, .4, .2, .3], 99)


def test_covariance_includes_normalized_target_scale(gp):
    rows = initial_design(1, 1)
    _, actual = gp.model.predict(encode(rows), return_cov=True)
    assert np.allclose(gp.covariance(rows, rows), actual, atol=1e-13)
    a = np.array([.1, .2, .3, .4])
    assert np.allclose(gp.weighted_covariance(rows, rows, a), actual @ a, atol=1e-13)


def test_exponential_gaussian_tilt_identity_and_unchanged_covariance(gp):
    rows = initial_design(1, 1)
    grid = initial_design(1, 2)
    a = np.ones(4) / 4
    mu, std = gp.moments(rows)
    before = gp.covariance(rows, rows)
    cross = gp.covariance(rows, grid)
    regional_var = a @ gp.covariance(grid, grid) @ a
    c = .7
    lam = c / np.sqrt(regional_var)
    shifted = mu + lam * gp.weighted_covariance(rows, grid, a)
    assert np.allclose(shifted, mu + lam * cross @ a, atol=1e-12)
    assert lam * regional_var == pytest.approx(c * np.sqrt(regional_var))
    assert np.array_equal(before, gp.covariance(rows, rows))
    assert np.array_equal(std, gp.moments(rows)[1])


def test_log_ei_exact_q1_and_finite_negative_tail():
    z = np.linspace(-8, 8, 61)
    expected = norm.pdf(z) + z * norm.cdf(z)
    actual = np.exp(log_expected_improvement(z, np.ones_like(z), 0))
    assert np.allclose(actual, expected, rtol=2e-10, atol=1e-25)
    assert np.isfinite(log_expected_improvement(np.array([-1e6, -50., -21., 0]), np.ones(4), 0)).all()


def test_justify_accept_reject_and_sqrt_beta():
    class Stub:
        def moments(self, rows):
            return np.array([r['mu'] for r in rows]), np.array([r['sd'] for r in rows])
    gp_point = {'mu': .9, 'sd': .2}
    llm_point = {'mu': .1, 'sd': .01}
    point, receipt = justify_select(Stub(), [gp_point], llm_point, t=2, psi0=.01)
    assert point == gp_point and not receipt['accepted_llm']
    assert receipt['ucb_sigma_multiplier'] ** 2 == pytest.approx(receipt['beta_argument'])
    assert receipt['psi'] == .005
    point, receipt = justify_select(Stub(), [gp_point], llm_point, t=2, psi0=100.)
    assert point == llm_point and receipt['accepted_llm']


def test_zero_confidence_lgbo_is_exact_base_ei(gp):
    region = {'catalysts': list(CATALYSTS), 't_res': [60, 600],
              'temperature': [30, 110], 'catalyst_loading': [.5, 2.5]}
    candidates = initial_design(1, 3)
    with threadpool_limits(1):
        selected, receipt = lgbo_select(gp, candidates, {'region': region, 'confidence': 0}, best=.4, grid_seed=22)
    mu, sigma = gp.moments(candidates)
    assert selected == candidates[np.argmax(log_expected_improvement(mu, sigma, .4))]
    assert receipt['lambda'] == 0 and receipt['max_abs_mean_shift'] == 0


def test_assistant_response_is_context_bound_and_citations_are_prefix_only():
    rows = initial_design(1, 0)
    request = {'run_id': 'c1_s0_LLINBO', 'iteration': 1, 'context_sha256': 'abc',
               'method': 'LLINBO', 'history': [{'point': row} for row in rows]}
    answer = {'run_id': request['run_id'], 'iteration': 1, 'context_sha256': 'abc',
              'reasoning': 'Observed catalyst-specific response warrants a comparison.',
              'supporting_queries': [1], 'point': initial_design(1, 1)[0]}
    validate_response(answer, request)
    for changed in ({'context_sha256': 'def'}, {'supporting_queries': [5]}, {'point': rows[0]}):
        with pytest.raises(ValueError):
            validate_response(dict(answer, **changed), request)
