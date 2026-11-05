"""UNSELECTED input-only positive-integral prototype; no production imports.

Review aid, not a replacement for the frozen V7 algorithm.  All validation here
uses synthetic scalar normal distributions, never benchmark outcomes or fits.

F=mu+sigma*X, h=(best-mu)/sigma, u=X-h>0, and s^2=vY-vF+tau^2.
The improvement numerator is sigma/sqrt(2*pi) times the integral of
  exp(L(u)), L(u)=log(u)-(h+u)^2/2+log Phi(c+k*u),
where c=(best-reference)/s and k=sigma/s.  The ordinary EI denominator
uses the same expression without log Phi.  Both integrands are positive and
strictly log concave.  Their unique modes are found before quadrature, so a
narrow or distant peak cannot be missed by an uncentered infinite integral.

At endpoints a<mode<b, concavity gives left tail <= exp(L(a))/L'(a)
and right tail <= exp(L(b))/(-L'(b)). These are mathematical tail bounds;
the QUADPACK interior estimate and floating-point evaluations are NOT rigorous
interval arithmetic. Refine independently before treating a near tie as resolved.
"""
from dataclasses import dataclass, asdict
import argparse
import json
from math import exp, isfinite, log, sqrt
from pathlib import Path
import time

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import erfcx, log_ndtr

LOG2PI = log(2*np.pi)


def inverse_mills(z):
    # Scaled erfc avoids cancellation of two O(z^2) logarithms in the tail.
    if z <= 0:
        return sqrt(2/np.pi)/float(erfcx(-z/sqrt(2)))
    return exp(-.5*z*z-.5*LOG2PI-float(log_ndtr(z)))


@dataclass
class Integral:
    log_value: float
    log_scaled_area: float
    relative_error_estimate: float
    relative_tail_bound: float
    mode: float
    width: float
    evaluations: int


def positive_integral(h, c=0., k=0., conditional=False, tol=1e-10):
    """Log integral and separated numerical/tail error diagnostics."""
    def derivative(u):
        return 1/u-h-u+(k*inverse_mills(c+k*u) if conditional else 0.)

    # Derivative is strictly decreasing, diverges positively at zero, and is
    # eventually negative. Bracket using only the input distribution.
    hi=max(1., -h+1., (-c/k+1.) if conditional and k>0 else 1.)
    for _ in range(100):
        if derivative(hi)<0: break
        hi*=2
    else: raise ArithmeticError('Mode not bracketed')
    lo=min(1., hi)*1e-14
    while derivative(lo)<=0: lo*=.1
    mode=brentq(derivative,lo,hi,xtol=5e-324,rtol=8*np.finfo(float).eps)
    zm=c+k*mode
    mills=inverse_mills(zm) if conditional else 0.
    # This width is only a numerical coordinate choice, not an error bound.
    curvature=1+1/(mode*mode)
    if conditional:
        curvature+=k*k*max(0.,mills*(zm+mills))
    width=1/sqrt(curvature)
    logphi_mode=float(log_ndtr(zm)) if conditional else 0.
    lm=log(mode)-.5*(h+mode)**2+logphi_mode

    def relative_log(u):
        if u<=0:return -np.inf
        value=log(u/mode)-.5*(u-mode)*(2*h+u+mode)
        if conditional:value+=float(log_ndtr(c+k*u))-logphi_mode
        return value

    def boundary(v,left):
        u=mode+width*v
        if u<=0:return 0.
        d=derivative(u)
        if (left and d<=0) or (not left and d>=0):return np.inf
        # Scaled integration is dv, not du.
        return exp(relative_log(u))/(abs(d)*width)

    left=max(-1.,-mode/width);right=1.
    for _ in range(100):
        bl=boundary(left,True)
        if bl<=tol*.01:break
        left=max(2*left,-mode/width)
    else:raise ArithmeticError('Left tail failed')
    for _ in range(100):
        br=boundary(right,False)
        if br<=tol*.01:break
        right*=2
    else:raise ArithmeticError('Right tail failed')
    evaluations=0
    def f(v):
        nonlocal evaluations
        evaluations+=1
        return exp(relative_log(mode+width*v))
    a,ea=quad(f,left,0.,epsabs=tol*.01,epsrel=tol*.1,limit=250)
    b,eb=quad(f,0.,right,epsabs=tol*.01,epsrel=tol*.1,limit=250)
    total=a+b
    if total<=0 or not isfinite(total):raise ArithmeticError('Nonpositive integral')
    return Integral(lm+log(width)+log(total),log(width)+log(total),(ea+eb)/total,(bl+br)/total,
                    mode,width,evaluations)


def conditional_ei_ratio(mu,vf,vy,reference,best,tau=.06,tol=1e-10):
    values=(mu,vf,vy,reference,best,tau)
    if not all(isfinite(x) for x in values) or vf<0 or vy<vf or tau<=0:
        raise ValueError('Finite scalar inputs and 0<=vF<=vY,tau>0 required')
    if vf==0:
        return dict(log_ratio=0.,unclipped_log_ratio=0.,offset=0.,
                    log_tempering=None,tempering=0.,numerical_noop=True,
                    exact_zero_latent=True,error_estimate=0.,tail_bound=0.)
    sd=sqrt(vf);s=sqrt(vy-vf+tau*tau)
    h=(best-mu)/sd;c=(best-reference)/s;k=sd/s
    base=positive_integral(h,tol=tol)
    cond=positive_integral(h,c,k,True,tol=tol)
    lp=float(log_ndtr((mu-reference)/sqrt(vy+tau*tau)))
    # Subtract mode heights algebraically, rather than subtracting two log
    # integrals containing the same potentially enormous -h^2/2 term.
    mb,mc=base.mode,cond.mode
    mode_difference=(log(mc/mb)-.5*(mc-mb)*(2*h+mc+mb)
                     +float(log_ndtr(c+k*mc)))
    raw=mode_difference+cond.log_scaled_area-base.log_scaled_area-lp
    lr=min(-lp,max(0.,raw))
    noop=lr<=1e-12
    if noop:lr=0.;lt=None;t=0.
    elif lr>log(4):lt=log(3)-lr-log(-np.expm1(-lr));t=exp(lt)
    else:lt=0.;t=1.
    return dict(log_ratio=lr,unclipped_log_ratio=raw,offset=min(lr,log(4)),
                log_tempering=lt,tempering=t,numerical_noop=noop,
                exact_zero_latent=False,event_log_probability=lp,
                error_estimate=base.relative_error_estimate+cond.relative_error_estimate,
                tail_bound=base.relative_tail_bound+cond.relative_tail_bound,
                base=asdict(base),conditional=asdict(cond))


def analytic_log_ratio_bounds(mu,vf,vy,reference,best,tau=.06):
    """Monotone event likelihood on F>best; no numerical integration.

    This function is deliberately not selected as a production fast path.
    For positive latent variance, max(1,P(E|F=best)/P(E))<=R<=1/P(E).
    Zero latent variance uses the original declared zero-effect policy.
    """
    if not all(isfinite(x) for x in (mu,vf,vy,reference,best,tau)) or vf<0 or vy<vf or tau<=0:
        raise ValueError('Invalid scalar normal inputs')
    if vf==0:return dict(lower=0.,upper=0.,width=0.,zero_latent=True)
    lp=float(log_ndtr((mu-reference)/sqrt(vy+tau*tau)))
    lower=max(0.,float(log_ndtr((best-reference)/sqrt(vy-vf+tau*tau)))-lp)
    upper=-lp
    return dict(lower=lower,upper=upper,width=max(0.,upper-lower),zero_latent=False)


def synthetic_review():
    cases=[]
    # Include high correlation, far-below/above incumbent, rare event and
    # near-certain event. Values are fabricated, not benchmark distributions.
    for vf,extra in [(1e-8,0.),(.0001,.001),(.01,0.),(1.,0.),(100.,0.),(1.,2.)]:
        for best in [-10.,-.1,0.,.1,10.]:
            for reference in [-10.,-.1,0.,.1,10.]:
                cases.append((0.,vf,vf+extra,reference,best))
    cases.extend([(0.,1.,1.,100.,100.),(0.,1.,1.,100.,-100.),
                  (0.,1.,1.,-100.,100.),(0.,1.,1.,0.,100.),
                  (0.,1.,1.,0.,-100.),(0.,0.,.1,2.,3.)])
    started=time.perf_counter();rows=[]
    for values in cases:
        a=conditional_ei_ratio(*values,tol=1e-9)
        b=conditional_ei_ratio(*values,tol=1e-12)
        bounds=analytic_log_ratio_bounds(*values)
        violation=max(0.,bounds['lower']-b['log_ratio'],b['log_ratio']-bounds['upper'])
        rows.append(dict(inputs=values,log_ratio=b['log_ratio'],
                         refinement_difference=abs(a['log_ratio']-b['log_ratio']),
                         reported_error=b['error_estimate'],tail_bound=b['tail_bound'],
                         analytic_bounds=bounds,bound_violation=violation))
    analytic=[]
    for vf,extra in [(1e-8,0.),(.0001,.001),(.01,0.),(1.,0.),(100.,0.),(1.,2.)]:
        result=conditional_ei_ratio(0.,vf,vf+extra,0.,0.,tol=1e-12)
        # At mu=best=reference, an independent integration-by-parts identity
        # gives R=1+sqrt(vF/(vY+tau^2)).
        exact=float(np.log1p(sqrt(vf/(vf+extra+.06**2))))
        analytic.append(dict(vf=vf,extra=extra,error=abs(result['log_ratio']-exact)))
    high_precision=[]
    try:
        import mpmath as mp
        mp.mp.dps=60
        for values in [(0.,1.,1.,.1,.1),(0.,1.,1.,2.,-1.),(0.,1.,1.,-2.,1.),
                       (0.,.0001,.0011,.1,.1),(0.,1.,3.,2.,5.),(0.,100.,100.,.1,1.)]:
            mu,vf,vy,ref,b=map(mp.mpf,values);tau=mp.mpf('.06')
            sd=mp.sqrt(vf);s=mp.sqrt(vy-vf+tau*tau)
            h=(b-mu)/sd;c=(b-ref)/s;k=sd/s
            probability=mp.erfc(-(mu-ref)/mp.sqrt(2*(vy+tau*tau)))/2
            cuts=sorted(set([h]+[z for z in [mp.mpf(0),(ref-mu)/sd,h+1,h+5] if z>h]))+[mp.inf]
            numerator=mp.quad(lambda z:(z-h)*mp.exp(-z*z/2)/mp.sqrt(2*mp.pi)*mp.erfc(-(c+k*(z-h))/mp.sqrt(2))/2,cuts)
            denominator=mp.exp(-h*h/2)/mp.sqrt(2*mp.pi)-h*mp.erfc(h/mp.sqrt(2))/2
            exact=float(mp.log(numerator/denominator/probability))
            result=conditional_ei_ratio(*values,tol=1e-12)
            high_precision.append(dict(inputs=values,error=abs(result['log_ratio']-exact)))
    except ImportError:
        high_precision=None
    return dict(status='SYNTHETIC_REVIEW_ONLY_NOT_SELECTED',cases=len(rows),
                elapsed_seconds=time.perf_counter()-started,
                max_refinement_difference=max(x['refinement_difference'] for x in rows),
                max_reported_error=max(x['reported_error'] for x in rows),
                max_tail_bound=max(x['tail_bound'] for x in rows),rows=rows,
                max_analytic_bound_violation=max(x['bound_violation'] for x in rows),
                analytic_fast_path_width_at_most_1e_10=sum(x['analytic_bounds']['width']<=1e-10 for x in rows),
                analytic_identity_checks=analytic,mpmath_60_digit_checks=high_precision,
                limitation='QUADPACK estimates are not interval proofs; logcdf floating-point cancellation can dominate extreme tails. No experiment or oracle accessed.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--synthetic',action='store_true')
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if not args.synthetic:p.error('Only --synthetic is provided; no experimental CLI')
    result=synthetic_review()
    if args.output:args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
