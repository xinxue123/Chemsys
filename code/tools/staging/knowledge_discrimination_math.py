"""Information distinguishing the actual tempered expert from its base GP.

For g=N(mu,vY), Z=Y-reference+eta and p=P_g(Z>0), the expert ratio is
L(y)=(1-t)+t*Phi((y-reference)/tau)/p. The .5/.5 design comparison
has JS(g,e)=E_g[.5*(L*log(L)-(1+L)*log((1+L)/2))]. This is model-label
information, not the GP sign-event MI or a posterior rule-truth probability.

Adaptive quadrature integrates the actual non-Gaussian density. Its reported
error is an estimate, not a mathematical interval certificate. Scalar tuples
are cached; SciPy's precompiled QUADPACK is used, with no per-round JIT.
"""
from functools import lru_cache
from math import exp, expm1, hypot, isfinite, log, log1p, pi, sqrt
import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.special import log_ndtr

PRIOR = .5
EPSABS = 1e-12
EPSREL = 2e-11
NUMERICAL_TOLERANCE = 1e-12
_LOG2 = log(2.)
_LOG2PI = log(2.*pi)


def _integrand(z, mu, sd, reference, tau, log_prior, log_t, log_one_minus_t):
    log_g = -.5*z*z-.5*_LOG2PI
    log_l = float(np.logaddexp(log_one_minus_t,
        log_t + log_ndtr((mu-reference+sd*z)/tau)-log_prior))
    if abs(log_l) < 1e-3:
        # F(1+d) has no constant/linear term. This avoids cancellation when
        # the expert and GP nearly coincide; the omitted term is O(d**9).
        d = expm1(log_l)
        factor = sum(.5*((-1.)**(n+1))*(1.-2.**(-n))*d**(n+1)/(n*(n+1))
                     for n in range(1,8))
        return exp(log_g)*max(0.,factor)
    # Evaluate the two density-weighted log ratios without exp(log_l), which
    # may overflow for rare positive events although the density is finite.
    gp_term = exp(log_g)*(_LOG2-float(np.logaddexp(0.,log_l)))
    expert_term = exp(log_g+log_l)*(_LOG2-float(np.logaddexp(0.,-log_l)))
    return max(0.,.5*(gp_term+expert_term))


@lru_cache(maxsize=8192)
def _cached_js(mu, vY, reference, log_t, tau):
    if log_t == -np.inf or vY == 0.:
        return 0.,0.,0
    sd = sqrt(vY)
    sd_z = hypot(sd,tau)
    alpha = (mu-reference)/sd_z
    log_prior = float(log_ndtr(alpha))
    log_one_minus_t = log(-expm1(log_t)) if log_t < 0. else -np.inf
    ratio = tau/sd
    transition = (reference-mu)/sd
    # The extra intervals locate a sharp probit transition and the positive
    # event's tail mass even when ordinary Gaussian nodes would miss them.
    lam = exp(-.5*alpha*alpha-.5*_LOG2PI-log_prior)
    rho = sd/sd_z
    center = rho*lam
    spread = (sqrt(max(0.,1.-rho*rho))+rho/(1.+abs(alpha))) if alpha < 0. else 1.
    breaks = {-10.,-6.,-2.,0.,2.,6.,10.,transition,
              transition-8.*ratio,transition+8.*ratio,center}
    breaks.update(center+k*spread for k in (-12.,-4.,-1.,1.,4.,12.))
    if not all(isfinite(x) for x in breaks):
        raise ValueError('Non-finite discrimination integration coordinates')
    bounds = [-np.inf,*sorted(breaks),np.inf]
    total = error = 0.
    evaluations = 0
    with warnings.catch_warnings():
        warnings.simplefilter('error',IntegrationWarning)
        for lower,upper in zip(bounds,bounds[1:]):
            value,estimate,info = quad(_integrand,lower,upper,
                args=(mu,sd,reference,tau,log_prior,log_t,log_one_minus_t),
                epsabs=EPSABS/len(bounds),epsrel=EPSREL,limit=160,full_output=1)[:3]
            total += value
            error += estimate
            evaluations += info['neval']
    if (not isfinite(total) or not isfinite(error) or error > 1e-8
            or total < -NUMERICAL_TOLERANCE or total > _LOG2+max(error,NUMERICAL_TOLERANCE)):
        raise ValueError('Model discrimination integral failed its numerical checks')
    return min(_LOG2,max(0.,total)),error,evaluations


def model_discrimination(mu, vY, reference_y, log_tempering, tau=.06):
    """Return scalar JS, estimated quadrature error and evaluation count."""
    mu,vY,reference_y,tau = map(float,(mu,vY,reference_y,tau))
    log_t = -np.inf if log_tempering is None else float(log_tempering)
    if (not all(isfinite(x) for x in (mu,vY,reference_y,tau)) or vY < 0. or tau <= 0.
            or not isfinite(vY+tau*tau) or np.isnan(log_t) or log_t > 0.):
        raise ValueError('Require finite GP mean/reference, vY>=0, tau>0 and log tempering<=0')
    score,error,evaluations = _cached_js(mu,vY,reference_y,log_t,tau)
    return dict(js=score,error=error,evaluations=evaluations)


def vector_discrimination(mu, vY, reference_y, log_tempering, tau=.06):
    arrays = np.broadcast_arrays(*[np.asarray(x,dtype=float)
        for x in (mu,vY,reference_y,log_tempering,tau)])
    scores = np.empty(arrays[0].shape)
    errors = np.empty_like(scores)
    for index in np.ndindex(scores.shape):
        value = model_discrimination(*(float(a[index]) for a in arrays))
        scores[index],errors[index] = value['js'],value['error']
    return dict(js=scores,error=errors)


def warm():
    return model_discrimination(0.,1.,0.,0.)
