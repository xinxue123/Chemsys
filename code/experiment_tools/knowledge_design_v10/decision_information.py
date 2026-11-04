"""Read-only, finite-set one-step knowledge-gradient calculation.

For fixed GP parameters and scalar future observation, posterior means are
mu(a) + Cov(f(a), Y_x) / sd(Y_x) * Z, Z~N(0,1).  Integrating the upper
envelope of these affine functions gives the expected improvement of the
best posterior mean.  It is not EI, entropy, a GP refit, or a SearchAUC gain.

This module has no oracle, LLM, fitted-model mutation, or experiment runner.
The caller must supply means/covariances in consistent objective units.
"""
from math import exp, fsum, inf, isfinite, pi, sqrt

import numpy as np
from scipy.special import ndtr


def _normal_pdf(z):
    return exp(-0.5 * z * z) / sqrt(2.0 * pi)


def _normal_mass(left, right):
    # Survival probabilities avoid subtracting two numbers rounded to one.
    if left >= 0:
        return float(ndtr(-left) - ndtr(-right))
    return float(ndtr(right) - ndtr(left))


def expected_best_value_gain(means, query_cross_covariance, observation_variance):
    """Return KG on a fixed finite recommendation set, in objective units.

    ``query_cross_covariance[a]`` is the latent posterior covariance between
    f(a) and f(x); independent future measurement noise contributes only to
    ``observation_variance``.  All entries must come from the same GP prefix.
    The caller owns PSD/covariance consistency checks.  No input is mutated.

    Identical common-mode uncertainty has zero decision value. Observing a
    redundant point can have positive value when its measurement is noisy;
    geometry alone cannot rule that out.
    """
    mu = np.asarray(means, dtype=float)
    cov = np.asarray(query_cross_covariance, dtype=float)
    var = float(observation_variance)
    if (mu.ndim != 1 or not mu.size or cov.shape != mu.shape
            or not np.isfinite(mu).all() or not np.isfinite(cov).all()
            or not isfinite(var) or var < 0):
        raise ValueError('Expected finite aligned means/covariance and nonnegative observation variance')
    if var == 0:
        if np.any(cov != 0):
            raise ValueError('Zero observation variance requires zero cross-covariance')
        return 0.0

    best = int(np.argmax(mu))
    intercepts = mu - mu[best]
    slopes = cov / sqrt(var)
    slopes = slopes - slopes[best]  # A common random shift has zero expected value.
    if not np.isfinite(intercepts).all() or not np.isfinite(slopes).all():
        raise ValueError('Input scale cannot be represented safely')

    # Each segment records (intercept, slope, left endpoint). Parallel lines
    # contribute only their largest intercept. Neither IDs nor input row order
    # are used to break a numerical tie.
    distinct = {}
    for a, b in zip(intercepts, slopes):
        distinct[float(b)] = max(float(a), distinct.get(float(b), -inf))
    hull = []
    for b, a in sorted(distinct.items()):
        left = -inf
        while hull:
            previous_a, previous_b, previous_left = hull[-1]
            left = (previous_a - a) / (b - previous_b)
            if left > previous_left:
                break
            hull.pop()
        if not hull:
            left = -inf
        hull.append((a, b, left))

    terms = []
    for i, (a, b, left) in enumerate(hull):
        right = hull[i + 1][2] if i + 1 < len(hull) else inf
        terms.append(a * _normal_mass(left, right))
        terms.append(b * (_normal_pdf(left) - _normal_pdf(right)))
    gain = fsum(terms)
    rounding_scale = 64 * np.finfo(float).eps * max(1.0, fsum(map(abs, terms)))
    if not isfinite(gain) or gain < -rounding_scale:
        raise ArithmeticError('Invalid expected best-value gain')
    return max(0.0, gain)


def knowledge_probe_priority(probe_kg, fallback_kg, deployment_weight):
    """Prototype priority: credit times positive *additional* decision value.

    Weight zero (paused/retracted) has no deployment authority. A new neutral
    prior may fund a budgeted pilot; positive feedback is not required before
    the first possible validation. Budget/window/scope checks belong to the
    future registered policy, which is deliberately not implemented here.
    """
    values = (probe_kg, fallback_kg, deployment_weight)
    if any(not isfinite(float(v)) or v < 0 for v in values) or deployment_weight > 1:
        raise ValueError('Expected nonnegative values and weight in [0,1]')
    return float(deployment_weight * max(0.0, probe_kg - fallback_kg))
