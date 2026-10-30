"""LLINBO-Justify, analytic LGBO regional tilt, and unmodified official HEBO."""
from __future__ import annotations

import random
import time
import warnings

import numpy as np
from scipy.linalg import cho_solve
from scipy.special import log_ndtr
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from .suzuki_domain import CATALYSTS, NUMERIC, LOW, HIGH, encode, candidate_grid, region_grid, same_point


class MixedGP:
    """Normalized numerics + complete one-hot catalyst; no ordered category distance."""
    def __init__(self, rows, values, seed):
        self.X = encode(rows)
        kernel = ConstantKernel(1., (1e-3, 1e3)) * Matern(np.ones(11), (1e-2, 1e2), nu=2.5)
        self.model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-6,
                                             n_restarts_optimizer=5, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.model.fit(self.X, np.array(values))
        self.scale2 = float(np.asarray(self.model._y_train_std).ravel()[0]) ** 2

    def moments(self, rows):
        return self.model.predict(encode(rows), return_std=True)

    def covariance(self, a, b):
        x, z = encode(a), encode(b)
        kx = self.model.kernel_(x, self.X)
        kz = self.model.kernel_(self.X, z)
        return (self.model.kernel_(x, z) - kx @ cho_solve((self.model.L_, True), kz)) * self.scale2

    def weighted_covariance(self, rows, grid, weights):
        # Compute Cov(X,G) a without materializing a large posterior joint matrix.
        z = encode(grid)
        back = cho_solve((self.model.L_, True), self.model.kernel_(self.X, z) @ weights)
        result = []
        for start in range(0, len(rows), 256):
            x = encode(rows[start:start + 256])
            value = self.model.kernel_(x, z) @ weights - self.model.kernel_(x, self.X) @ back
            result.append(value * self.scale2)
        return np.concatenate(result)


def log_expected_improvement(mu, sigma, best):
    """Exact q=1 Gaussian log EI, with a stable negative-tail expansion."""
    mu, sigma = np.asarray(mu), np.asarray(sigma)
    s = np.maximum(sigma, np.finfo(float).tiny)
    z = (mu - best) / s
    logphi = -.5 * z ** 2 - .5 * np.log(2 * np.pi)
    logfactor = np.empty_like(z)
    positive = z >= 0
    logfactor[positive] = np.logaddexp(logphi[positive],
                                      np.log(np.maximum(z[positive], np.finfo(float).tiny)) + log_ndtr(z[positive]))
    middle = (z < 0) & (z >= -20)
    q = np.log(-z[middle]) + log_ndtr(z[middle]) - logphi[middle]
    logfactor[middle] = logphi[middle] + np.log(-np.expm1(np.minimum(q, -np.finfo(float).eps)))
    tail = z < -20
    r = 1 / z[tail] ** 2
    logfactor[tail] = logphi[tail] - 2 * np.log(-z[tail]) + np.log(1 - 3*r + 15*r**2 - 105*r**3 + 945*r**4)
    return np.log(s) + logfactor


def available_grid(case, seed, iteration, observed):
    return [p for p in candidate_grid(case, seed, iteration) if not any(same_point(p, row) for row in observed)]


def justify_select(gp, candidates, llm_point, *, t, psi0):
    """Upstream code schedule: beta=2 log(t*d*pi²/.1*6); psi=max initial variance/t."""
    rows = [*candidates, llm_point]
    mu, sigma = gp.moments(rows)
    beta = float(2 * np.log(t * 4 * np.pi ** 2 / .1 * 6))
    score = mu + np.sqrt(beta) * sigma
    gp_index = int(np.argmax(score))
    gap = float(score[gp_index] - score[-1])
    psi = float(psi0 / t)
    accepted = gap <= psi
    i = len(rows) - 1 if accepted else gp_index
    return rows[i], {"accepted_llm": accepted, "ucb_gap": gap, "psi": psi, "psi0": psi0,
                     "beta_argument": beta, "ucb_sigma_multiplier": float(np.sqrt(beta)),
                     "llm_ucb": float(score[-1]), "max_searched_ucb": float(score[gp_index]),
                     "selected_mu": float(mu[i]), "selected_sigma": float(sigma[i]),
                     "searched_candidates": len(rows), "gate_comparator": "gap <= psi"}


def lgbo_select(gp, candidates, preference, *, best, grid_seed):
    """Paper Eq. (lambda): c / sqrt(a^T Cov(G,G) a), unchanged covariance."""
    grid, weights = region_grid(preference["region"], grid_seed, 512)
    region_variance = float(weights @ gp.weighted_covariance(grid, grid, weights))
    if region_variance < -1e-10:
        raise ValueError("regional posterior variance is negative")
    confidence = float(preference["confidence"])
    lam = confidence / np.sqrt(max(region_variance, 1e-12))
    mu, sigma = gp.moments(candidates)
    shift = lam * gp.weighted_covariance(candidates, grid, weights)
    base_score = log_expected_improvement(mu, sigma, best)
    score = log_expected_improvement(mu + shift, sigma, best)
    i, base_i = int(np.argmax(score)), int(np.argmax(base_score))
    return candidates[i], {"lambda": float(lam), "confidence": confidence,
                           "regional_variance": max(region_variance, 0.), "grid_size": len(grid),
                           "selected_base_mu": float(mu[i]), "selected_mean_shift": float(shift[i]),
                           "selected_tilted_mu": float(mu[i] + shift[i]), "selected_sigma": float(sigma[i]),
                           "covariance_unchanged": True, "selected_log_ei": float(score[i]),
                           "unguided_argmax": candidates[base_i], "guidance_changed_choice": i != base_i,
                           "max_abs_mean_shift": float(np.max(np.abs(shift))),
                           "searched_candidates": len(candidates), "grid_seed": grid_seed}


def hebo_select(rows, values, seed):
    """Official HEBO 0.3.6 suggest/observe; preserve MACE and 100x100 evolution."""
    import pandas as pd
    import torch
    from unittest.mock import patch
    from hebo.design_space.design_space import DesignSpace
    from hebo.optimizers.hebo import HEBO, EvolutionOpt
    from hebo.models.gp.gp import GP
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)
    space = DesignSpace().parse([
        *[{"name": n, "type": "num", "lb": float(l), "ub": float(h)} for n, l, h in zip(NUMERIC, LOW, HIGH)],
        {"name": "catalyst", "type": "cat", "categories": list(CATALYSTS)}])
    optimizer = HEBO(space, rand_sample=4, scramble_seed=seed)
    optimizer.observe(pd.DataFrame(rows), -np.array(values).reshape(-1, 1))
    counts = {"gp_fit_calls": 0, "evolution_calls": 0, "evolution_population": None, "evolution_generations": None}
    original_fit, original_opt = GP.fit, EvolutionOpt.optimize

    def counted_fit(self, *args, **kwargs):
        counts["gp_fit_calls"] += 1
        counts["gp_epochs"] = self.num_epochs
        return original_fit(self, *args, **kwargs)

    def counted_opt(self, *args, **kwargs):
        counts["evolution_calls"] += 1
        counts["evolution_population"] = self.pop
        counts["evolution_generations"] = self.iter
        counts["acquisition_objectives"] = self.acq.num_obj
        return original_opt(self, *args, **kwargs)

    started = time.perf_counter()
    with patch.object(GP, "fit", counted_fit), patch.object(EvolutionOpt, "optimize", counted_opt):
        point = optimizer.suggest(n_suggestions=1).iloc[0].to_dict()
    if counts["evolution_calls"] != 1 or counts["gp_fit_calls"] < 1:
        raise ValueError("official HEBO did not run its GP/evolution path")
    return point, {**counts, "native_continuous_suggest": True, "official_hebo": "0.3.6",
                   "seed": seed, "suggest_seconds": time.perf_counter() - started}
