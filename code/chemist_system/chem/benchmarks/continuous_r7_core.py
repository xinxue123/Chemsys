"""Small, auditable numerical helpers for R7 passive preference propagation.

The event is the *single anchored* R6 contrast Z=f(x+)-f(x-)+noise>0.
No global monotonicity, independent pseudo-pairs, or new GP fit is implied.
"""

from __future__ import annotations

from functools import lru_cache
from math import sqrt

import numpy as np
from scipy.special import ndtr, xlogy


# Numerical independence test, not a tunable chemical or evidence gate.
MIN_RHO_SQUARED = 1e-10


def squared_correlation(variance_y: float, contrast_variance: float,
                        covariance: float, preference_noise_sd: float) -> float:
    """Squared correlation of a future observation and noisy GP contrast."""
    vy = float(variance_y)
    vz = float(contrast_variance) + float(preference_noise_sd) ** 2
    cov = float(covariance)
    if not np.isfinite((vy, vz, cov)).all() or vy <= 0 or vz <= 0:
        raise ValueError('invalid predictive or contrast moments')
    rho2 = cov * cov / (vy * vz)
    if rho2 > 1 + 1e-8:
        raise ValueError('joint covariance is not positive semidefinite')
    return float(np.clip(rho2, 0., 1.))


@lru_cache(maxsize=2)
def _gauss_hermite(n: int):
    if n < 8:
        raise ValueError('at least 8 quadrature nodes required')
    nodes, weights = np.polynomial.hermite.hermgauss(n)
    return nodes, weights / sqrt(np.pi)


def _binary_entropy(p):
    return -xlogy(p, p) - xlogy(1. - p, 1. - p)


def sign_information_nats(gp_mean_y: float, gp_var_y: float,
                          contrast_mean: float, contrast_var: float,
                          cov_y_contrast: float, preference_noise_sd: float,
                          quadrature: int = 64) -> float:
    """I(Y; 1[Z>0]) for a joint-Gaussian (Y,Z); not I(Y;Z).

    Y is an actually measurable noisy target, while Z is the noisy latent
    preference contrast. Its mean matters: a nearly certain sign has little
    binary information even if the Gaussian contrast correlation is large.
    """
    rho2 = squared_correlation(gp_var_y, contrast_var, cov_y_contrast,
                               preference_noise_sd)
    if rho2 <= 1e-15:
        return 0.
    vy = float(gp_var_y)
    vz = float(contrast_var) + float(preference_noise_sd) ** 2
    remaining = max(vz * (1. - rho2), vz * 1e-15)
    p_prior = float(ndtr(float(contrast_mean) / sqrt(vz)))
    nodes, weights = _gauss_hermite(quadrature)
    # Y=mu_y+sqrt(2*var_y)*node.  mu_y cancels algebraically.
    conditional_mean = float(contrast_mean) + (float(cov_y_contrast) / sqrt(vy)) * sqrt(2.) * nodes
    p_after = ndtr(conditional_mean / sqrt(remaining))
    information = float(_binary_entropy(p_prior) - np.dot(weights, _binary_entropy(p_after)))
    if not np.isfinite(information):
        raise ValueError('nonfinite binary contrast information')
    return float(np.clip(information, 0., _binary_entropy(p_prior)))


def sign_information_for_shadow(prediction) -> float:
    return sign_information_nats(
        prediction.gp_mean_y, prediction.gp_var_y,
        prediction.contrast_mean, prediction.contrast_var,
        prediction.cov_y_contrast, prediction.preference_noise_sd)


def flip_directions(proposal: dict) -> dict:
    """Negative control: invert one valid physical preference and its signs."""
    import copy
    out = copy.deepcopy(proposal)
    if out.get('action') != 'compare':
        return out
    out['better_id'], out['worse_id'] = out['worse_id'], out['better_id']
    opposites = {'increase': 'decrease', 'decrease': 'increase'}
    out['focus_directions'] = {axis: opposites[d] for axis, d in out['focus_directions'].items()}
    out['reason'] = 'R7 sign-flipped negative control (original rationale archived separately)'
    return out
