"""Input-only mathematics for a prospective rule at a new proposed point.

The observed reference is a *fixed number*, not another uncertain GP value.
For F ~ N(mu, vF), Y = F + independent noise, Var(Y) = vY, the soft
event is Z = Y - reference_y + eta > 0, eta ~ N(0, tau**2).

Acquisition uses GH64 over a quantile representation of Z | Z > 0 and
analytic EI for F | Z. This intentionally replaces the legacy hard-hinge
GH ratio. It never replaces or recomputes the caller's native MC qLogEI:
``offset`` is an approximate log multiplier for that unchanged acquisition.
The prediction density and its feedback score are analytic, not Gaussian
moment approximations. Tempering acts on this density as well as its EI.

No model, oracle, file, or network access occurs in this module. Arrays use
the same engineering units as the GP targets. All result arrays broadcast
to the inputs' common shape (scalar inputs produce zero-dimensional arrays).
"""
from functools import lru_cache
from hashlib import sha256
import json
from math import log

import numpy as np
from scipy.special import log_ndtr, logsumexp, ndtri_exp


GH_NODES = 64
MAX_EI_RATIO = 4.0
NUMERICAL_NOOP_ATOL = 1e-12
DEFAULT_TAU = .06
_LOG_2PI = log(2. * np.pi)
_LOG_2 = log(2.)


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             allow_nan=False, separators=(',', ':')).encode('utf-8')).hexdigest()


@lru_cache(maxsize=1)
def _quadrature():
    nodes, weights = np.polynomial.hermite.hermgauss(GH_NODES)
    z = np.sqrt(2.) * nodes
    log_weights = np.log(weights / np.sqrt(np.pi))
    # Returning readonly arrays prevents a caller changing the shared rule.
    z.setflags(write=False)
    log_weights.setflags(write=False)
    return z, log_weights


def _inputs(mu, vF, vY, reference_y, tau, best=None):
    values = [mu, vF, vY, reference_y, tau]
    if best is not None:
        values.append(best)
    try:
        arrays = np.broadcast_arrays(*[np.asarray(x, dtype=float) for x in values])
    except (ValueError, TypeError) as error:
        raise ValueError('Rule inputs must be broadcastable real numbers') from error
    if not all(np.all(np.isfinite(x)) for x in arrays):
        raise ValueError('Rule inputs must be finite')
    _, latent, observed, _, noise = arrays[:5]
    if np.any(latent < 0.) or np.any(observed < latent) or np.any(noise <= 0.):
        raise ValueError('Require 0 <= vF <= vY and tau > 0')
    if not np.all(np.isfinite(observed + noise * noise)):
        raise ValueError('Event variance overflow')
    return arrays


def _log_normal_ei(mu, variance, best):
    """Log analytic Gaussian EI, including deterministic and far-tail cases."""
    mu, variance, best = np.broadcast_arrays(mu, variance, best)
    result = np.full(mu.shape, -np.inf)
    deterministic = variance == 0.
    positive = deterministic & (mu > best)
    result[positive] = np.log(mu[positive] - best[positive])
    random = ~deterministic
    if not np.any(random):
        return result
    sd = np.sqrt(variance[random])
    z = (mu[random] - best[random]) / sd
    log_phi = -.5 * z * z - .5 * _LOG_2PI
    helper = np.empty(z.shape)
    upper = z >= 0.
    with np.errstate(divide='ignore'):
        helper[upper] = np.logaddexp(log_phi[upper], np.log(z[upper]) + log_ndtr(z[upper]))
    middle = (z < 0.) & (z >= -10.)
    # phi(z) + z Phi(z), evaluated as a stable positive subtraction.
    ratio_log = np.log(-z[middle]) + log_ndtr(z[middle]) - log_phi[middle]
    helper[middle] = log_phi[middle] + np.log(-np.expm1(ratio_log))
    tail = z < -10.
    if np.any(tail):
        # Mills expansion for phi(z)+z*Phi(z), avoiding cancellation at
        # z=-1e4, where a subtraction of two nearly equal terms is unsafe.
        inv = 1. / (z[tail] * z[tail])
        series = np.ones(inv.shape)
        term = np.ones(inv.shape)
        for k in range(1, 13):
            term *= -(2*k + 1) * inv
            series += term
        helper[tail] = log_phi[tail] - 2. * np.log(-z[tail]) + np.log(series)
    result[random] = np.log(sd) + helper
    return result


def _entropy_probit(z):
    lp, lq = log_ndtr(z), log_ndtr(-z)
    # log_ndtr is finite for the supported finite inputs; exp underflow is
    # harmless and means the corresponding entropy contribution is zero.
    return -np.exp(lp)*lp - np.exp(lq)*lq


def _conditional_moments(mu, vF, vY, reference_y, tau):
    vz = vY + tau*tau
    alpha = (mu-reference_y)/np.sqrt(vz)
    log_prior = log_ndtr(alpha)
    lam = np.exp(-.5*alpha*alpha - .5*_LOG_2PI - log_prior)
    reduction = np.clip(lam*(lam+alpha), 0., 1.)
    shift_y = vY/np.sqrt(vz)*lam
    shift_f = vF/np.sqrt(vz)*lam
    return dict(event_log_probability=log_prior,
                conditional_mean_y=mu+shift_y,
                conditional_var_y=np.maximum(0., vY-vY*vY/vz*reduction),
                conditional_mean_f=mu+shift_f,
                conditional_var_f=np.maximum(0., vF-vF*vF/vz*reduction))


def _mixture_moments(mu, variance, component_mean, component_variance, fraction):
    shift = component_mean-mu
    return (mu+fraction*shift,
            (1.-fraction)*variance + fraction*component_variance + fraction*(1.-fraction)*shift*shift)


def vector_scores(mu, vF, vY, reference_y, best, tau=DEFAULT_TAU):
    """Return bounded EI multipliers and sign-event information, using GH64.

    ``sign_information`` is I(Y; 1[Z>0]), not the continuous Gaussian
    contrast MI. Its small quadrature error is clipped to [0, log(2)].
    No-effect policy: vF=0 or log(R)<=1e-12 gives t=0, exact offset=0,
    and an exactly unchanged *observed* predictive density. Otherwise
    t=min(1, 3/(R-1)), so the tempered expert's EI ratio is min(R,4).
    ``noop`` identifies this policy separately from the general t formula.
    """
    mu, vF, vY, ref, tau, best = _inputs(mu, vF, vY, reference_y, tau, best)
    nodes, logw = _quadrature()
    vz = vY+tau*tau
    alpha = (mu-ref)/np.sqrt(vz)
    log_prior = log_ndtr(alpha)
    base_log_ei = _log_normal_ei(mu, vF, best)
    # If U=Phi(N(0,1)), U is uniform. Under the positive event, the
    # survival probability of its standardized Z is (1-U)*Phi(alpha).
    # ndtri_exp avoids subtraction from one in rare-event tails.
    qz = -ndtri_exp(log_ndtr(-nodes) + log_prior[..., None])
    conditional_mu = mu[..., None] + (vF/np.sqrt(vz))[..., None]*qz
    conditional_var = np.maximum(0., vF*(1.-vF/vz))
    log_ei_by_z = _log_normal_ei(conditional_mu, conditional_var[..., None], best[..., None])
    conditional_log_ei = logsumexp(log_ei_by_z+logw, axis=-1)
    raw = np.zeros(mu.shape)
    random = vF > 0.
    raw[random] = conditional_log_ei[random]-base_log_ei[random]
    if not np.all(np.isfinite(raw)):
        raise ValueError('Non-finite conditional EI ratio')
    # Positive covariance ensures improvement. Also P(E|F)<=1 gives the
    # exact upper bound R<=1/P(E). In particular a virtually certain event
    # must not manufacture a tiny ranking change from quadrature error.
    pure_log_ratio = np.clip(raw, 0., -log_prior)
    noop = (~random) | (pure_log_ratio <= NUMERICAL_NOOP_ATOL)
    pure_log_ratio = np.where(noop, 0., pure_log_ratio)
    log_tempering = np.zeros(mu.shape)
    clipped = pure_log_ratio > log(MAX_EI_RATIO)
    # 3/(exp(logR)-1), evaluated without exp(logR) overflow.
    log_tempering[clipped] = (log(MAX_EI_RATIO-1.)-pure_log_ratio[clipped]
                               -np.log1p(-np.exp(-pure_log_ratio[clipped])))
    log_tempering[noop] = -np.inf
    tempering = np.exp(log_tempering)
    offset = np.minimum(pure_log_ratio, log(MAX_EI_RATIO))
    y_draws = mu[..., None]+np.sqrt(vY)[..., None]*nodes
    expected_entropy = np.sum(np.exp(logw)*_entropy_probit((y_draws-ref[..., None])/tau[..., None]), axis=-1)
    # When sd(Y) >> tau, the entropy transition is narrow and direct GH
    # nodes under Y can miss it. Integrate in s=(Y-reference)/tau instead;
    # H(Phi(s))/phi(s) is smooth. This is still the same GH64 entropy
    # integral, with a deterministic change of variable, not extra nodes.
    narrow = vY > tau*tau
    if np.any(narrow):
        sd = np.sqrt(vY[narrow])
        z = ((ref[narrow]-mu[narrow])[..., None]+tau[narrow][..., None]*nodes)/sd[..., None]
        transformed = (logw + np.log(tau[narrow]/sd)[..., None]
                       -.5*z*z + .5*nodes*nodes + np.log(_entropy_probit(nodes)))
        expected_entropy = np.array(expected_entropy, copy=True)
        expected_entropy[narrow] = np.exp(logsumexp(transformed, axis=-1))
    sign_information = np.clip(_entropy_probit(alpha)-expected_entropy, 0., _LOG_2)
    sign_information = np.where(vY == 0., 0., sign_information)
    moments = _conditional_moments(mu, vF, vY, ref, tau)
    mean_y, var_y = _mixture_moments(mu, vY, moments['conditional_mean_y'], moments['conditional_var_y'], tempering)
    mean_f, var_f = _mixture_moments(mu, vF, moments['conditional_mean_f'], moments['conditional_var_f'], tempering)
    return dict(offset=np.asarray(offset), log_tempered_ei_ratio=np.asarray(offset),
                tempering=np.asarray(tempering), log_tempering=np.asarray(log_tempering),
                pure_log_ei_ratio=np.asarray(pure_log_ratio),
                raw_quadrature_log_ei_ratio=np.asarray(raw),
                base_normal_log_ei=np.asarray(base_log_ei), conditional_normal_log_ei=np.asarray(conditional_log_ei),
                sign_information=np.asarray(sign_information), noop=np.asarray(noop),
                tempered_mean_y=np.asarray(mean_y), tempered_var_y=np.asarray(var_y),
                tempered_mean_f=np.asarray(mean_f), tempered_var_f=np.asarray(var_f), **moments)


def _scalar_inputs(mu, vF, vY, reference_y, tau):
    values = _inputs(mu, vF, vY, reference_y, tau)
    if any(x.ndim != 0 for x in values):
        raise ValueError('A frozen prediction must describe exactly one point')
    return tuple(float(x) for x in values)


def freeze_prediction(hyp_id, version, point_id, t, mu, vF, vY,
                      reference_id, reference_y, *, tempering,
                      tau=DEFAULT_TAU, semantic_key=None, log_tempering=None):
    """Build a self-contained forecast; caller persists it before observing y."""
    if (not all(isinstance(x, str) and x for x in (hyp_id, point_id, reference_id))
            or point_id == reference_id or type(version) is not int or version < 1
            or type(t) is not int or t < 1
            or (semantic_key is not None and not isinstance(semantic_key, str))):
        raise ValueError('Invalid prospective identity/version/reference')
    mu, vF, vY, reference_y, tau = _scalar_inputs(mu, vF, vY, reference_y, tau)
    if not np.isscalar(tempering) or not np.isfinite(tempering) or not 0. <= tempering <= 1.:
        raise ValueError('Tempering must be in [0,1]')
    tempering = float(tempering)
    if log_tempering is None or np.isneginf(log_tempering):
        log_fraction = None if tempering == 0. else log(tempering)
    elif not np.isscalar(log_tempering) or not np.isfinite(log_tempering) or log_tempering > 0.:
        raise ValueError('Log tempering must be <=0 or absent for a no-op')
    else:
        log_fraction = float(log_tempering)
        if not np.isclose(np.exp(log_fraction), tempering, rtol=1e-12, atol=0.):
            raise ValueError('Tempering and its logarithm disagree')
    if vF == 0. and log_fraction is not None:
        raise ValueError('Zero latent variance requires the no-effect policy t=0')
    moments = _conditional_moments(mu, vF, vY, reference_y, tau)
    mean_y, var_y = _mixture_moments(mu, vY, moments['conditional_mean_y'], moments['conditional_var_y'], tempering)
    prediction = dict(hypothesis_id=hyp_id, version=version, point_id=point_id,
        query_round=t, reference_id=reference_id, reference_y=reference_y, semantic_key=semantic_key,
        gp_mean_y=mu, gp_var_latent_y=vF, gp_var_y=vY, preference_noise_sd=tau,
        tempering=tempering, log_tempering=log_fraction,
        event='Y-reference_y+independent_Normal(0,tau^2)>0',
        distribution='(1-tempering)*GP + tempering*one-sign conditional observation density',
        knowledge_mean_y=float(mean_y), knowledge_var_y=float(var_y),
        conditional_mean_y=float(moments['conditional_mean_y']),
        conditional_var_y=float(moments['conditional_var_y']),
        event_log_probability=float(moments['event_log_probability']),
        zero_effect_policy=bool(log_fraction is None))
    prediction['prediction_sha256'] = _digest(prediction)
    return prediction


def _validate_prediction(prediction):
    if not isinstance(prediction, dict):
        raise ValueError('Prediction must be a frozen mapping')
    try:
        fingerprint = _digest({k: v for k, v in prediction.items() if k != 'prediction_sha256'})
        if prediction.get('prediction_sha256') != fingerprint:
            raise ValueError('Frozen prediction was modified')
        expected = freeze_prediction(prediction['hypothesis_id'], prediction['version'], prediction['point_id'],
            prediction['query_round'], prediction['gp_mean_y'], prediction['gp_var_latent_y'], prediction['gp_var_y'],
            prediction['reference_id'], prediction['reference_y'], tempering=prediction['tempering'],
            tau=prediction['preference_noise_sd'], semantic_key=prediction['semantic_key'],
            log_tempering=prediction['log_tempering'])
        if expected != prediction:
            raise ValueError('Prediction fields do not match the declared probability model')
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError('Invalid frozen prediction') from error


def score_prediction(prediction, y):
    """Exact tempered log-density improvement over GP for one frozen forecast.

    The caller owns exactly-once, version, time, and history validation.
    This function neither updates credit nor equates prediction skill with
    the separate factual comparison y > reference_y.
    """
    _validate_prediction(prediction)
    if not np.isscalar(y) or not np.isfinite(y):
        raise ValueError('Observation must be a finite scalar')
    fraction = prediction['tempering']
    log_fraction = prediction['log_tempering']
    if log_fraction is None:
        return 0.
    pure = float(log_ndtr((float(y)-prediction['reference_y'])/prediction['preference_noise_sd'])
                 -prediction['event_log_probability'])
    if fraction == 1.:
        return pure
    return float(np.logaddexp(np.log1p(-fraction), log_fraction+pure))


def deployed_score_prediction(prediction, y, weight):
    """Actual one-expert deployment score, including its external weight.

    For several experts, the engine must combine all frozen expert density
    ratios in one mixture; summing these log scores would be incorrect.
    """
    if not np.isscalar(weight) or not np.isfinite(weight) or not 0. <= weight <= .5:
        raise ValueError('Deployment weight must be in [0,.5]')
    expert_score = score_prediction(prediction, y)
    if weight == 0. or prediction['zero_effect_policy']:
        return 0.
    return float(np.logaddexp(np.log1p(-weight), log(weight)+expert_score))
