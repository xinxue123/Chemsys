"""Receding two-step rollout with the qLogEI baseline at both stages."""
import numpy as np
import torch

from fast_hit_bo import PARAMETERS as P
from fast_hit_bo.policy import choose, coverage_points
from chem.benchmarks import knowledge_gp
from .acquisition import maximum_qlogei


def decision(gp, context, query, arm='rollout_qlogei'):
    available = context['available']
    priorities = context['priorities']
    baseline = choose(context['logei'], available, priorities)
    common = dict(BO_reference_id=gp.pool.ids[baseline],
                  base_policy='qLogExpectedImprovement_MC512',
                  analytic_EI_used=False)
    if query==P['total_queries'] or arm=='qlogei':
        return dict(selected_id=gp.pool.ids[baseline], rule='full_pool_qLogEI',
                    **common, outer_ids=[], outer_values=[], fantasy_seed=None,
                    future_seed=None, future_ids=[], future_logei=[], evaluated_counts=[])
    js = np.flatnonzero(available)
    first = js[np.lexsort((priorities[js],-context['logei'][js]))][:P['outer_qlogei']]
    uncertain = js[np.lexsort((priorities[js],-context['variance'][js]))][:P['outer_uncertainty']]
    outer = sorted(set(map(int,first)) | set(map(int,uncertain)) |
                   {j for j in coverage_points(gp) if available[j]},
                   key=lambda j:gp.pool.features[j])
    assert baseline in outer
    cross = gp.conditional_cross_covariance(context, outer)
    mu = torch.from_numpy(context['mu'])
    variance = torch.from_numpy(context['variance'])
    fantasy_seed = 4112026+query
    future_seed = 38112000+query+1
    future_priorities = torch.from_numpy(knowledge_gp.semantic_priorities(gp.pool, query+1, 0))
    engine = torch.quasirandom.SobolEngine(1,scramble=True,seed=fantasy_seed)
    u = engine.draw(P['fantasy_samples']).double().reshape(-1).clamp(1e-12,1-1e-12)
    normal = torch.distributions.Normal(0.,1.).icdf(u)
    utilities, future_ids, future_values, evaluated = [], [], [], []
    with torch.no_grad():
        for column,j in enumerate(outer):
            v = variance[j].clamp_min(1e-18)
            y = mu[j]+v.sqrt()*normal
            denominator = v+P['noise_variance']
            cov = cross[:,column]
            next_mu = mu[None,:]+(y-mu[j])[:,None]*cov[None,:]/denominator
            next_variance = (variance-cov.square()/denominator).clamp_min(1e-18)
            # SharedGP passes a Python float as best_f; BoTorch first stores it
            # in the default float32 dtype, then casts it to the samples' dtype.
            # Reproduce that conversion separately for each virtual history.
            incumbent = y.clamp_min(context['incumbent']).float().double()
            valid = torch.from_numpy(available.copy())
            valid[j] = False
            log_best,winners,counts = maximum_qlogei(
                next_mu,next_variance,incumbent,valid,
                future_priorities,future_seed)
            value = np.exp(context['logei'][j])+float(log_best.exp().mean())
            utilities.append(float(value))
            future_ids.append([gp.pool.ids[k] for k in winners])
            future_values.append(log_best.tolist())
            evaluated.append(counts)
    scores = np.full(len(gp.pool.ids),-np.inf)
    scores[outer] = utilities
    winner = choose(scores,np.isin(np.arange(len(scores)),outer),priorities)
    return dict(selected_id=gp.pool.ids[winner],rule='two_step_qLogEI_rollout',
                **common,outer_ids=[gp.pool.ids[j] for j in outer],outer_values=utilities,
                fantasy_seed=fantasy_seed,future_seed=future_seed,
                first_logei=[float(context['logei'][j]) for j in outer],
                future_ids=future_ids,future_logei=future_values,evaluated_counts=evaluated)
