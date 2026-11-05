"""Exact q=1 MC qLogEI maximization with bounded candidate elimination."""
import math
import torch

from botorch.acquisition.logei import _log_improvement, TAU_MAX, TAU_RELU
from botorch.sampling.normal import draw_sobol_normal_samples
from botorch.utils.safe_math import fatmax, fatplus, logmeanexp


def base_samples(seed):
    return draw_sobol_normal_samples(d=1, n=512, device=torch.device('cpu'),
                                    dtype=torch.double, seed=seed).reshape(-1)


def marginal_scores(mean, sd, incumbent, normals):
    """Use the same sample-level transform and reductions as BoTorch."""
    samples = mean[None, :, None]+sd[None, :, None]*normals[:, None, None]
    improvement = _log_improvement(samples, incumbent, tau=TAU_RELU, fat=True)
    return logmeanexp(fatmax(improvement, dim=-1, tau=TAU_MAX), dim=0)


def maximum_qlogei(mean, variance, incumbent, valid, priorities, seed):
    """Return each fantasy's full-pool qLogEI winner.

    For q=1 a Gaussian posterior is determined by its marginal mean and SD.
    The sampler collapses t-batch dimensions, sharing 512 normal base samples.
    A finite-sum hard improvement bounds the smooth value within
    tau*(log(2)+0.1). Jensen and maximum-sample bounds tighten this interval.
    Only candidates whose upper bound can meet the best lower bound need
    the native sample-level evaluation. Conservative roundoff allowances
    widen the bounds without changing any acquisition value.
    """
    normals = base_samples(seed)
    ordered = normals.sort().values
    count_samples = len(normals)
    suffix = torch.cat([ordered.flip(0).cumsum(0).flip(0), ordered.new_zeros(1)])
    sd = variance.clamp_min(1e-10).sqrt()
    if sd.ndim == 1:
        sd = sd.unsqueeze(0).expand_as(mean)
    delta = mean-incumbent[:, None]
    index = torch.searchsorted(ordered, (-delta/sd).contiguous(), right=True)
    count = count_samples-index
    hard = ((count*delta+sd*suffix[index])/count_samples).clamp_min(0.)
    epsilon = torch.finfo(mean.dtype).eps*4096
    roundoff = epsilon*(delta.abs()+sd*normals.abs().max())*count/count_samples
    max_sample = fatplus(delta+sd*normals.max(), tau=TAU_RELU)
    jensen = fatplus(delta+sd*normals.mean(), tau=TAU_RELU)
    lower = torch.maximum(hard-roundoff, jensen*(1-epsilon)).clamp_min(0.)
    upper = torch.minimum(
        hard+TAU_RELU*(math.log(2)+.1)+roundoff,
        max_sample*(1+epsilon))
    available = valid.unsqueeze(0).expand_as(mean)
    best_lower = lower.masked_fill(~available, -torch.inf).max(dim=1).values
    keep = available & (upper>=best_lower[:, None])
    pairs = keep.nonzero()
    if len(pairs)==0:
        raise ValueError('All candidates were eliminated')
    log_values = torch.full_like(mean, -torch.inf)
    for batch in pairs.split(256):
        row, col = batch[:, 0], batch[:, 1]
        evaluated = marginal_scores(mean[row, col], sd[row, col], incumbent[row], normals)
        log_values[row, col] = evaluated
    best_values = log_values.max(dim=1).values
    winners = []
    for row in range(len(mean)):
        ties = (log_values[row]==best_values[row]).nonzero().flatten()
        winner = ties[torch.argmin(priorities[ties])]
        winners.append(int(winner))
    if not torch.isfinite(best_values).all():
        raise ValueError('Non-finite future qLogEI')
    return best_values, winners, keep.sum(dim=1).tolist()
