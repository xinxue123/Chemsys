"""Two-step expected improvement with a full-pool second-step search."""
import numpy as np
import torch
from botorch.acquisition.analytic import _log_ei_helper

from . import PARAMETERS as P


def log_ei(mean, variance, incumbent):
    sd = variance.clamp_min(1e-18).sqrt()
    return sd.log() + _log_ei_helper((mean-incumbent)/sd)


def choose(values, available, priorities):
    js = np.flatnonzero(available)
    top = js[values[js] == np.max(values[js])]
    return int(top[np.argmin(priorities[top])])


def coverage_points(gp):
    points = gp.X.numpy()
    selected = []
    for category in sorted(set(points[:, 3])):
        js = np.flatnonzero(points[:, 3] == category)
        ordered = sorted(map(int, js), key=lambda j: gp.pool.features[j])
        first = ordered[0]
        distance = ((points[js, :3]-points[first, :3])**2).sum(1)
        candidates = js[distance == distance.max()]
        second = min(map(int, candidates), key=lambda j: gp.pool.features[j])
        selected.extend([first, second])
    return selected


def decision(gp, context, query, arm):
    available = context['available']
    priority = context['priorities']
    baseline = choose(context['logei'], available, priority)
    mean = torch.from_numpy(context['mu'])
    variance = torch.from_numpy(context['variance'])
    one_step_log = log_ei(mean, variance, context['incumbent']).numpy()
    analytic_shadow = choose(one_step_log, available, priority)
    common = dict(BO_reference_id=gp.pool.ids[baseline],
                  analytic_EI_shadow_id=gp.pool.ids[analytic_shadow],
                  one_step_estimators_disagree=baseline!=analytic_shadow)
    if arm == 'qlogei' or query == P['total_queries']:
        return dict(selected_id=gp.pool.ids[baseline], rule='full_pool_qLogEI', **common,
                    outer_ids=[], outer_values=[], fantasy_seed=None)
    js = np.flatnonzero(available)
    ei_order = js[np.lexsort((priority[js], -context['logei'][js]))][:P['outer_qlogei']]
    sd_order = js[np.lexsort((priority[js], -context['variance'][js]))][:P['outer_uncertainty']]
    outer = sorted(set(map(int, ei_order)) | set(map(int, sd_order)) |
                   {j for j in coverage_points(gp) if available[j]},
                   key=lambda j: gp.pool.features[j])
    assert baseline in outer
    covariance = gp.conditional_cross_covariance(context, outer)
    fantasy_seed = 4112026+query
    engine = torch.quasirandom.SobolEngine(1, scramble=True, seed=fantasy_seed)
    uniform = engine.draw(P['fantasy_samples']).double().reshape(-1).clamp(1e-12, 1-1e-12)
    normals = torch.distributions.Normal(0., 1.).icdf(uniform)
    utilities = []
    with torch.no_grad():
        for column, j in enumerate(outer):
            vj = variance[j].clamp_min(1e-18)
            fantasy_y = mean[j] + vj.sqrt()*normals
            denominator = vj+P['noise_variance']
            cross = covariance[:, column]
            new_mean = mean[None, :] + (fantasy_y-mean[j])[:, None]*cross[None, :]/denominator
            new_variance = (variance-cross.square()/denominator).clamp_min(1e-18)
            new_incumbent = fantasy_y.clamp_min(context['incumbent'])
            future_logei = log_ei(new_mean, new_variance[None, :], new_incumbent[:, None])
            valid = available.copy()
            valid[j] = False
            future_logei[:, ~torch.from_numpy(valid)] = -torch.inf
            future_best = future_logei.max(dim=1).values.exp()
            first_ei = one_step_log[j]
            utilities.append(float(np.exp(first_ei)+future_best.mean()))
    scores = np.full(len(gp.pool.ids), -np.inf)
    scores[outer] = utilities
    winner = choose(scores, np.isin(np.arange(len(scores)), outer), priority)
    return dict(selected_id=gp.pool.ids[winner], rule='restricted_outer_two_step_EI_rollout',
                **common, outer_ids=[gp.pool.ids[j] for j in outer], outer_values=utilities,
                fantasy_seed=fantasy_seed)
