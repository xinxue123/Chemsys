"""Deterministic conditional-EI ratio; not connected to any experiment runner.

For F~Normal(mu,vf), Y=F+independent noise with Var(Y)=vy, and
E={Y-reference+Normal(0,tau**2)>0}, evaluate
R=E[(F-best)+|E]/E[(F-best)+]. Native BO acquisition is not evaluated here.

Use a monotonicity bound first. Otherwise integrate the positive F-space
integrand, centered on its unique log-concave mode. Tangent envelopes bound
omitted tails. QUADPACK's interior errors and the floating-point allowance are
ESTIMATES, not interval-arithmetic certificates. A discrepancy or unresolved
no-op boundary raises NumericalResolutionError rather than silently clipping.

Inputs broadcast. Scalar inputs return Python scalars; arrays return independent
ndarrays. ``log_tempering=-inf`` denotes an actual no-op; finite log tempering
is retained even when exp(log_tempering) underflows to zero.

Supported numerical envelope: finite real inputs, 0<=vf<=vy, tau>0, no
intermediate overflow, |event z|<=1000 and |threshold-event z|<=1e6. The
integration fallback additionally requires |(best-mu)/sqrt(vf)|<=1e6 and
sqrt(vf)/sqrt(vy-vf+tau**2)<=1e6. Requested absolute log tolerance must be
1e-12..1e-4. Inputs inside this envelope can still be rejected if roundoff,
integration error, or threshold ambiguity prevents meeting that tolerance.
No JIT, files, network, model fits, stochastic integration, or target access.
"""
from dataclasses import asdict, dataclass
from functools import lru_cache
from math import exp, isfinite, log, log1p, sqrt
from threading import Lock
import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.optimize import brentq
from scipy.special import erfcx, log_ndtr

NOOP_ATOL = 1e-12
MAX_EI_RATIO = 4.
DEFAULT_ATOL = 1e-10
_LOG_CAP = log(MAX_EI_RATIO)
_EPS = np.finfo(float).eps
_LOG2PI = log(2*np.pi)
_WARM_LOCK = Lock()
_WARMED = False


class NumericalResolutionError(ArithmeticError):
    """Inputs or numerical diagnostics do not resolve the declared precision."""


@dataclass(frozen=True)
class _Integral:
    mode: float
    log_scaled_area: float
    relative_quad_error: float
    relative_tail_bound: float
    evaluations: int


@dataclass(frozen=True)
class _Result:
    log_ratio: float
    raw_log_ratio: float
    offset: float
    tempering: float
    log_tempering: float
    noop: bool
    event_log_probability: float
    log_ratio_error_estimate: float
    analytic_lower: float
    analytic_upper: float
    resolved_lower: float
    resolved_upper: float
    method: str
    quad_error_estimate: float
    tail_error_bound: float
    roundoff_allowance: float
    evaluations: int


def _mills(z):
    if z <= 0:
        return sqrt(2/np.pi)/float(erfcx(-z/sqrt(2)))
    return exp(-.5*z*z-.5*_LOG2PI-float(log_ndtr(z)))


def _checked_coordinates(mu, vf, vy, reference, best, tau, atol):
    if not all(isfinite(x) for x in (mu, vf, vy, reference, best, tau, atol)):
        raise ValueError('All inputs must be finite real values')
    if vf < 0 or vy < vf or tau <= 0:
        raise ValueError('Require 0<=vf<=vy and tau>0')
    if not 1e-12 <= atol <= 1e-4:
        raise ValueError('atol must be between 1e-12 and 1e-4 in log-ratio units')
    vz, vn = vy+tau*tau, vy-vf+tau*tau
    if not isfinite(vz) or not isfinite(vn) or vz <= 0 or vn <= 0:
        raise NumericalResolutionError('Event variance overflow or underflow')
    alpha, c = (mu-reference)/sqrt(vz), (best-reference)/sqrt(vn)
    if not isfinite(alpha) or not isfinite(c) or abs(alpha)>1000 or abs(c)>1e6:
        raise NumericalResolutionError('Standardized event outside supported envelope')
    lp = float(log_ndtr(alpha))
    if not isfinite(lp):
        raise NumericalResolutionError('Nonfinite event log probability')
    return lp, c, sqrt(vn)


@lru_cache(maxsize=8192)
def _integral(h, c, k, conditional, tolerance):
    def derivative(u):
        return 1/u-h-u+(k*_mills(c+k*u) if conditional else 0.)

    hi=max(1., -h+1., -c/k+1. if conditional and k else 1.)
    for _ in range(80):
        if derivative(hi)<0: break
        hi*=2
    else: raise NumericalResolutionError('Could not bracket the integral mode')
    lo=min(1., hi)*1e-14
    for _ in range(80):
        if derivative(lo)>0: break
        lo*=.1
    else: raise NumericalResolutionError('Could not bracket the lower integral mode')
    mode=brentq(derivative, lo, hi, xtol=5e-324, rtol=8*_EPS)
    z=c+k*mode
    curvature=1+1/(mode*mode)
    if conditional:
        mills=_mills(z)
        # A coordinate choice only: curvature rounding cannot authorize a
        # result, because tail bounds and final diagnostics are separate.
        curvature+=k*k*max(0., mills*(z+mills))
    width=1/sqrt(curvature)
    lpm=float(log_ndtr(z)) if conditional else 0.
    if not isfinite(width) or width<=0 or not isfinite(lpm):
        raise NumericalResolutionError('Unrepresentable mode or integration width')

    def relative_log(u):
        if u<=0: return -np.inf
        value=log(u/mode)-.5*(u-mode)*(2*h+u+mode)
        if conditional: value+=float(log_ndtr(c+k*u))-lpm
        if not isfinite(value):
            raise NumericalResolutionError('Nonfinite scaled integrand')
        return value

    def tail(v, left):
        u=mode+width*v
        if u<=0: return 0.
        d=derivative(u)
        if (left and d<=0) or (not left and d>=0): return np.inf
        return exp(relative_log(u))/(abs(d)*width)

    left, right=max(-1., -mode/width), 1.
    for _ in range(80):
        lt=tail(left, True)
        if lt<=tolerance*.1: break
        left=max(2*left, -mode/width)
    else: raise NumericalResolutionError('Left tail bound did not converge')
    for _ in range(80):
        rt=tail(right, False)
        if rt<=tolerance*.1: break
        right*=2
    else: raise NumericalResolutionError('Right tail bound did not converge')
    calls=0
    def integrand(v):
        nonlocal calls
        calls+=1
        return exp(relative_log(mode+width*v))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', IntegrationWarning)
            a, ea=quad(integrand, left, 0., epsabs=tolerance*.1,
                       epsrel=tolerance, limit=250)
            b, eb=quad(integrand, 0., right, epsabs=tolerance*.1,
                       epsrel=tolerance, limit=250)
    except IntegrationWarning as error:
        raise NumericalResolutionError('Interior quadrature failed: '+str(error)) from error
    total=a+b
    if not isfinite(total) or total<=0 or not all(isfinite(x) and x>=0 for x in (ea, eb, lt, rt)):
        raise NumericalResolutionError('Invalid integral or error estimate')
    return _Integral(mode, log(width)+log(total), (ea+eb)/total,
                     (lt+rt)/total, calls)


def _finish(value, raw, lower, upper, lp, err, alo, ahi, method,
            qerr=0., tail=0., rounding=0., evaluations=0):
    if upper<=NOOP_ATOL:
        return _Result(0., raw, 0., 0., -np.inf, True, lp, err,
                       alo, ahi, lower, upper, method, qerr, tail, rounding, evaluations)
    if lower<=NOOP_ATOL:
        raise NumericalResolutionError('Numerical interval straddles the no-op threshold')
    if not 0.<value<=ahi or not isfinite(value):
        raise NumericalResolutionError('Resolved log ratio violates probability bounds')
    if value>_LOG_CAP:
        log_t=log(MAX_EI_RATIO-1)-value-log(-np.expm1(-value))
        t=exp(log_t)
    else:
        log_t, t=0., 1.
    return _Result(value, raw, min(value, _LOG_CAP), t, log_t, False, lp, err,
                   alo, ahi, lower, upper, method, qerr, tail, rounding, evaluations)


@lru_cache(maxsize=8192)
def _scalar(mu, vf, vy, reference, best, tau, atol):
    lp, c, noise_sd=_checked_coordinates(mu, vf, vy, reference, best, tau, atol)
    if vf==0:
        return _Result(0., 0., 0., 0., -np.inf, True, lp, 0., 0., 0., 0., 0.,
                       'zero_latent', 0., 0., 0., 0)
    upper=-lp
    log_at_best=float(log_ndtr(c))
    # Width=min(-log P(E),-log P(E|F=best)) avoids subtracting two nearly
    # equal bounds. The lower bound also uses positive association R>=1.
    bound_width=min(upper, -log_at_best)
    lower=upper-bound_width
    rounding=64*_EPS*(1+upper)
    bound_error=.5*bound_width+rounding
    if bound_error<=atol:
        resolution_lower=max(0., lower-rounding)
        resolution_upper=upper+rounding
        if resolution_upper<=NOOP_ATOL or resolution_lower>NOOP_ATOL:
            return _finish(lower+.5*bound_width, lower+.5*bound_width,
                           resolution_lower, resolution_upper, lp, bound_error,
                           lower, upper, 'analytic_bound', rounding=rounding)

    sd=sqrt(vf); h=(best-mu)/sd; k=sd/noise_sd
    if not isfinite(h) or not isfinite(k) or abs(h)>1e6 or k>1e6:
        raise NumericalResolutionError('F integral outside supported numerical envelope')
    if rounding>=atol:
        raise NumericalResolutionError('Requested tolerance below floating-point allowance')
    work_tolerance=max(4e-14, atol*.01)
    base=_integral(h, 0., 0., False, work_tolerance)
    cond=_integral(h, c, k, True, work_tolerance)
    mb, mc=base.mode, cond.mode
    mode_log_ratio=log(mc/mb)
    normal_difference=-.5*(mc-mb)*(2*h+mc+mb)
    conditional_log_probability=float(log_ndtr(c+k*mc))
    raw=(mode_log_ratio+normal_difference+conditional_log_probability
         +cond.log_scaled_area-base.log_scaled_area-lp)
    # The common, potentially enormous h^2/2 term was removed algebraically.
    rounding=64*_EPS*(1+abs(mode_log_ratio)+abs(normal_difference)
                     +abs(conditional_log_probability)+abs(cond.log_scaled_area)
                     +abs(base.log_scaled_area)+abs(lp))
    qerr=base.relative_quad_error+cond.relative_quad_error
    tail=base.relative_tail_bound+cond.relative_tail_bound
    estimate=qerr+tail+rounding
    if not isfinite(raw) or not isfinite(estimate) or estimate>atol:
        raise NumericalResolutionError(f'Unresolved log ratio: estimated error {estimate:g} exceeds {atol:g}')
    # Intersect the measured numerical interval with the analytic constraint.
    # A disagreement beyond the error allowance is an error, never a clamp.
    resolved_lower=max(lower, raw-estimate)
    resolved_upper=min(upper, raw+estimate)
    if resolved_lower>resolved_upper:
        raise NumericalResolutionError('Positive integral disagrees with analytic bounds')
    value=resolved_lower+.5*(resolved_upper-resolved_lower)
    error=max(abs(value-raw)+estimate, .5*(resolved_upper-resolved_lower)+rounding)
    if error>atol:
        raise NumericalResolutionError('Bound refinement exceeds requested error budget')
    # Include rounding in threshold resolution, but preserve the stated raw
    # analytic bounds separately for audit and future stricter reconstruction.
    return _finish(value, raw, max(0., resolved_lower-rounding),
                   resolved_upper+rounding, lp, error, lower, upper,
                   'positive_f_integral', qerr, tail, rounding,
                   base.evaluations+cond.evaluations)


def conditional_ei_ratio(mu, vf, vy, reference, best, tau=.06, atol=DEFAULT_ATOL):
    """Broadcast inputs and return ratio, density tempering, and diagnostics.

    ``log_ratio_error_estimate`` bounds/estimates error of the *uncensored*
    mathematical ratio calculation, not the deliberate no-op policy. No-op
    sets log_ratio/offset/tempering to exactly zero after resolving its interval.
    The fixed cap changes EI via density tempering, not an unrelated score clip.
    A successful result is an estimated tolerance check, not a formal proof.
    """
    try:
        raw_inputs=(mu, vf, vy, reference, best, tau, atol)
        if any(np.iscomplexobj(x) for x in raw_inputs):
            raise ValueError('Complex inputs are not supported')
        arrays=np.broadcast_arrays(*[np.asarray(x, dtype=float) for x in raw_inputs])
    except (TypeError, ValueError) as error:
        raise ValueError('Inputs must be broadcastable real numbers') from error
    shape=arrays[0].shape
    if not all(np.all(np.isfinite(x)) for x in arrays):
        raise ValueError('All inputs must be finite')
    # Exact float keys; returned dictionaries/arrays never expose cached objects.
    rows=[asdict(_scalar(*map(float, values)))
          for values in zip(*(x.flat for x in arrays))]
    names=tuple(_Result.__dataclass_fields__)
    if shape==(): return rows[0]
    result={}
    for name in names:
        dtype=str if name=='method' else bool if name=='noop' else int if name=='evaluations' else float
        result[name]=np.asarray([r[name] for r in rows], dtype=dtype).reshape(shape)
    return result


def cache_info():
    """Exact-input scalar and integral cache counters, no hidden rounding."""
    return dict(scalar=_scalar.cache_info()._asdict(), integral=_integral.cache_info()._asdict(),
                warmed=_WARMED, operator_compiles=0)


def warm():
    """Once per process, warm analytic and integral paths; no JIT compilation."""
    global _WARMED
    with _WARM_LOCK:
        if not _WARMED:
            _scalar(0., .04, .05, 0., 0., .06, DEFAULT_ATOL)
            _scalar(0., .04, .05, -1., 1., .06, DEFAULT_ATOL)
            _WARMED=True
    return cache_info()
