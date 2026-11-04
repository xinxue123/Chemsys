"""Prospective Gaussian birth-expert forecasts, not an optimization policy.

The native MC acquisition is never recomputed. ``offset`` is the log analytic
EI ratio of the *same tempered density* subsequently scored. Latent variances
may be zero; every scored observation variance must be strictly positive.
Unrepresentable arithmetic fails explicitly; there is no variance floor.

Hashes detect corruption and inconsistent derived fields, not authenticity.
The caller must store the original hash in its immutable query commit and pass
it as ``expected_prediction_sha256`` when scoring after receiving an outcome.
Scope, version transitions, chronological evidence and repeated observations
are the caller's responsibility. No target lookup, GP fit or network I/O.
"""
from copy import deepcopy
from hashlib import sha256
import json
from math import fsum, isfinite

import numpy as np
from .knowledge_rule_math import _log_normal_ei

MIN_EI_RATIO = .25
MAX_EI_RATIO = 4.
NOOP_LOG_RATIO_ATOL = 1e-12
_LOG2PI = float(np.log(2*np.pi))
_LOGMIN = float(np.log(MIN_EI_RATIO))
_LOGMAX = float(np.log(MAX_EI_RATIO))


def _digest(value):
    return sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,
                            allow_nan=False,separators=(',',':')).encode('utf-8')).hexdigest()


def _broadcast(*values):
    if any(np.iscomplexobj(v) for v in values):
        raise ValueError('Forecast inputs must be real')
    try:
        arrays=np.broadcast_arrays(*[np.asarray(v,dtype=float) for v in values])
    except (ValueError,TypeError,OverflowError) as error:
        raise ValueError('Forecast inputs must be broadcastable numbers') from error
    if not all(np.all(np.isfinite(a)) for a in arrays):
        raise ValueError('Forecast inputs must be finite')
    return arrays


def _gaussian(mean,latent,observed):
    mean,latent,observed=_broadcast(mean,latent,observed)
    if np.any(latent<0) or np.any(observed<=0) or np.any(observed<latent):
        raise ValueError('Require 0 <= latent variance <= strictly positive observation variance')
    return mean,latent,observed


def _log_one_minus(log_weight):
    with np.errstate(divide='ignore',invalid='ignore',under='ignore'):
        return np.where(log_weight==0.,-np.inf,np.log(-np.expm1(log_weight)))


def _weighted(log_base,log_remainder,value):
    with np.errstate(divide='ignore',under='ignore',over='ignore',invalid='ignore'):
        return np.sign(value)*np.exp((log_base+np.log(np.abs(value)))+log_remainder)


def _moments(gmean,gvar,emean,evar,log_weight,log_base=None,log_remainder=None):
    if log_base is None:log_base=np.zeros(np.shape(log_weight))
    if log_remainder is None:log_remainder=log_weight
    log_other=_log_one_minus(log_weight)
    with np.errstate(over='ignore',invalid='ignore',divide='ignore',under='ignore'):
        mean=_weighted(0.,log_other,gmean)+_weighted(log_base,log_remainder,emean)
        difference=emean-gmean
        log_difference=np.log(np.abs(difference))
        # Opposite, very large finite means can overflow their subtraction.
        huge=~np.isfinite(difference)
        log_difference=np.where(huge,np.logaddexp(np.log(np.abs(emean)),np.log(np.abs(gmean))),log_difference)
        variance=(_weighted(0.,log_other,gvar)+_weighted(log_base,log_remainder,evar)
                  +np.exp(((log_base+2*log_difference)+log_remainder)+log_other))
    mean=np.where(log_weight==-np.inf,gmean,np.where(log_weight==0.,emean,mean))
    variance=np.where(log_weight==-np.inf,gvar,np.where(log_weight==0.,evar,variance))
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(variance)) or np.any(variance<0):
        raise ArithmeticError('Unrepresentable Gaussian-mixture moments')
    return mean,variance


def vector_forecast(current_mean,current_latent_var,current_observation_var,
                    expert_mean,expert_latent_var,expert_observation_var,best):
    """Return arrays for a proper two-sided EI-bounded Gaussian mixture.

    An exact zero current latent variance is an explicit no-op policy. A zero
    expert latent EI gives raw_log_ratio=-inf and lambda=.75, not a no-op.
    The base/remainder pair represents log lambda without losing the O(1)
    normalizer at extreme log EI ratios. Scalar log_tempering is diagnostic;
    both parts are authoritative even when exp(log_tempering) underflows.
    """
    gm,gf,gy,em,ef,ey,inc=_broadcast(current_mean,current_latent_var,current_observation_var,
                                  expert_mean,expert_latent_var,expert_observation_var,best)
    _gaussian(gm,gf,gy);_gaussian(em,ef,ey)
    current_zero=gf==0.
    with np.errstate(over='ignore',divide='ignore',invalid='ignore',under='ignore'):
        base_ei=_log_normal_ei(gm,gf,inc)
        raw_ei=_log_normal_ei(em,ef,inc)
    if np.any((~current_zero)&(~np.isfinite(base_ei))):
        raise ArithmeticError('Current Gaussian log EI is not representable')
    if np.any(np.isnan(raw_ei)) or np.any(np.isposinf(raw_ei)):
        raise ArithmeticError('Expert Gaussian log EI is not representable')
    # If positive variance nevertheless underflows in log space, this is not
    # an exact zero-EI expert and cannot silently use the deterministic branch.
    if np.any((ef>0)&(~np.isfinite(raw_ei))):
        raise ArithmeticError('Positive-variance expert log EI is not representable')
    with np.errstate(invalid='ignore',over='ignore'):
        raw_ratio=np.where(current_zero,0.,raw_ei-base_ei)
    if np.any(np.isnan(raw_ratio)) or np.any(np.isposinf(raw_ratio)):
        raise ArithmeticError('Gaussian EI ratio is not representable')
    noop=current_zero|(np.abs(raw_ratio)<=NOOP_LOG_RATIO_ATOL)
    log_base=np.zeros(gm.shape);log_remainder=np.zeros(gm.shape)
    above=(raw_ratio>_LOGMAX)&(~noop)
    below=(raw_ratio<_LOGMIN)&(~noop)
    with np.errstate(divide='ignore',invalid='ignore',under='ignore',over='ignore'):
        log_base=np.where(above,-raw_ratio,log_base)
        log_remainder=np.where(above,np.log(3.)-np.log1p(-np.exp(-raw_ratio)),log_remainder)
        log_remainder=np.where(below,np.log(.75)-np.log1p(-np.exp(raw_ratio)),log_remainder)
    log_base=np.where(noop,0.,log_base)
    log_remainder=np.where(noop,-np.inf,log_remainder)
    log_t=log_base+log_remainder
    with np.errstate(under='ignore',invalid='ignore'):
        offset=np.logaddexp(_log_one_minus(log_t),(log_base+raw_ratio)+log_remainder)
    offset=np.where(noop,0.,offset)
    if not np.all(np.isfinite(offset)) or np.any(offset<_LOGMIN-1e-13) or np.any(offset>_LOGMAX+1e-13):
        raise ArithmeticError('Tempered EI ratio violates its density-defined bounds')
    mf,vf=_moments(gm,gf,em,ef,log_t,log_base,log_remainder)
    my,vy=_moments(gm,gy,em,ey,log_t,log_base,log_remainder)
    return dict(raw_log_ratio=np.asarray(raw_ratio),log_ratio=np.asarray(offset),offset=np.asarray(offset),
                tempering=np.exp(log_t),log_tempering=np.asarray(log_t),
                log_tempering_base=np.asarray(log_base),log_tempering_remainder=np.asarray(log_remainder),
                zero_effect_policy=np.asarray(noop),current_zero_latent_variance=np.asarray(current_zero),
                base_normal_log_ei=np.asarray(base_ei),raw_normal_log_ei=np.asarray(raw_ei),
                tempered_mean_y=my,tempered_var_y=vy,tempered_mean_f=mf,tempered_var_f=vf)


def calibration_information(beta_variance,residual_feature,null_latent_var,birth_noise_variance):
    """I(beta;Y) within the birth model, not improvement over the current GP.

    A nonzero signal with zero residual observation noise has infinite mutual
    information and is unsupported here; it is rejected rather than capped.
    """
    p,psi,v,n=_broadcast(beta_variance,residual_feature,null_latent_var,birth_noise_variance)
    if np.any(p<0) or np.any(v<0) or np.any(n<0):
        raise ValueError('Information variances must be nonnegative')
    signal=(p>0)&(psi!=0)
    denominator=v+n
    if not np.all(np.isfinite(denominator)) or np.any(signal&(denominator<=0)):
        raise ValueError('Nonzero coefficient signal requires positive finite residual observation variance')
    with np.errstate(divide='ignore',invalid='ignore',under='ignore'):
        log_ratio=np.log(p)+2*np.log(np.abs(psi))-np.log(denominator)
        value=np.where(signal,.5*np.logaddexp(0.,log_ratio),0.)
    if not np.all(np.isfinite(value)):
        raise ArithmeticError('Calibration information is not representable')
    return np.asarray(value)


def _model(value,name):
    if not isinstance(value,dict) or set(value)!={'mean','latent_variance','observation_variance'}:
        raise ValueError(name+' must contain exactly mean, latent_variance, observation_variance')
    arrays=_gaussian(value['mean'],value['latent_variance'],value['observation_variance'])
    if any(a.shape!=() for a in arrays):raise ValueError('Frozen forecasts describe exactly one point')
    return dict(zip(('mean','latent_variance','observation_variance'),map(float,arrays)))


def _json_vector(value):
    result={}
    for key,x in value.items():
        if x.shape!=():raise ValueError('Frozen forecasts describe exactly one point')
        if x.dtype.kind=='b':result[key]=bool(x)
        else:
            scalar=float(x)
            # -inf has explicit meanings: zero EI or exact no-op log lambda.
            result[key]=None if scalar==-np.inf else scalar
    return result


def freeze_forecast(*,hypothesis_id,version,point_id,query_round,prefix_sha256,
                    current_gp_sha256,birth_snapshot_sha256,semantic_key,
                    current,birth_null,raw_expert,cut_feedback,variance_only,best,
                    beta_variance,residual_feature,birth_noise_variance,
                    reference_id,reference_y,actual_calibrate,full_expert=None):
    identities=dict(hypothesis_id=hypothesis_id,version=version,point_id=point_id,query_round=query_round,
                    prefix_sha256=prefix_sha256,current_gp_sha256=current_gp_sha256,
                    birth_snapshot_sha256=birth_snapshot_sha256,semantic_key=semantic_key,
                    reference_id=reference_id)
    for k,v in identities.items():
        if k in ('version','query_round'):
            if type(v)is not int or v<1:raise ValueError(k+' must be a positive integer')
        elif not isinstance(v,str) or not v:raise ValueError(k+' must be a nonempty string')
    if point_id==reference_id:raise ValueError('A new point must differ from its known reference')
    if type(actual_calibrate)is not bool:raise ValueError('actual_calibrate must be boolean')
    models={k:_model(v,k) for k,v in dict(current=current,birth_null=birth_null,raw_expert=raw_expert,
                                       cut_feedback=cut_feedback,variance_only=variance_only).items()}
    if not actual_calibrate and full_expert is None:
        raise ValueError('The cut control must retain an explicit full_expert shadow')
    models['full_expert']=_model(full_expert,'full_expert') if full_expert is not None else deepcopy(models['raw_expert'])
    if not actual_calibrate and models['raw_expert']!=models['cut_feedback']:
        raise ValueError('A noncalibrating forecast must deploy cut_feedback as raw_expert')
    if actual_calibrate and models['raw_expert']!=models['full_expert']:
        raise ValueError('A calibrating forecast must deploy full_expert as raw_expert')
    scalar_arrays=_broadcast(best,reference_y,beta_variance,residual_feature,birth_noise_variance)
    if any(x.shape!=() for x in scalar_arrays):raise ValueError('Frozen scalar metadata must be scalar')
    inc,ref,p,psi,noise=map(float,scalar_arrays)
    g=models['current'];e=models['raw_expert'];n=models['birth_null']
    if noise<0:raise ValueError('Birth noise variance must be nonnegative')
    for name,x in models.items():
        if name=='current':continue
        noise_error=abs(x['observation_variance']-x['latent_variance']-noise)
        if noise_error>64*np.finfo(float).eps*max(x['observation_variance'],noise,np.finfo(float).tiny):
            raise ValueError(name+' observation variance must include the frozen birth noise')
    variance_shadow=models['variance_only']
    if (variance_shadow['mean']!=n['mean'] or any(variance_shadow[key]!=e[key]
            for key in ('latent_variance','observation_variance'))):
        raise ValueError('variance_only must use birth-null mean and actual raw-expert variances')
    info=float(calibration_information(p,psi,n['latent_variance'],noise))
    forecast=_json_vector(vector_forecast(g['mean'],g['latent_variance'],g['observation_variance'],
                                        e['mean'],e['latent_variance'],e['observation_variance'],inc))
    result=dict(**identities,actual_calibrate=actual_calibrate,reference_y=ref,best=inc,
                beta_variance=p,residual_feature=psi,birth_noise_variance=noise,
                information=info,information_kind='birth_model_I_beta_Y',models=models,forecast=forecast,
                schema='birth_gaussian_forecast_v1')
    result['prediction_sha256']=_digest(result)
    return result


def _validate(pred,expected_prediction_sha256):
    if not isinstance(pred,dict):raise ValueError('Frozen forecast must be a dictionary')
    try:
        fingerprint=_digest({k:v for k,v in pred.items() if k!='prediction_sha256'})
        if fingerprint!=pred.get('prediction_sha256'):
            raise ValueError('Frozen forecast hash mismatch')
        if expected_prediction_sha256 is not None and fingerprint!=expected_prediction_sha256:
            raise ValueError('Forecast differs from the independently committed hash')
        kwargs={k:pred[k] for k in ('hypothesis_id','version','point_id','query_round','prefix_sha256',
            'current_gp_sha256','birth_snapshot_sha256','semantic_key','best','beta_variance','residual_feature',
            'birth_noise_variance','reference_id','reference_y','actual_calibrate')}
        rebuilt=freeze_forecast(**kwargs,**pred['models'])
        if rebuilt!=pred:raise ValueError('Frozen forecast derived fields changed')
    except (KeyError,TypeError,OverflowError) as error:
        raise ValueError('Malformed frozen forecast') from error


def _log_density(model,y):
    with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
        z=(y-model['mean'])/np.sqrt(model['observation_variance'])
        value=-.5*(_LOG2PI+np.log(model['observation_variance'])+z*z)
    if not isfinite(float(value)):raise ArithmeticError('Gaussian log density is not representable')
    return float(value)


def _gaussian_gain(component,current,y):
    if component['observation_variance']==current['observation_variance']:
        return _mean_gain(y,current['mean'],component['mean'],current['observation_variance'])
    return _log_density(component,y)-_log_density(current,y)


def _mixture_gain(g,component,log_t,log_base=0.,log_remainder=None):
    if log_t==-np.inf:return 0.
    if log_remainder is None:log_remainder=log_t
    contribution=fsum((log_base,log_remainder,component,-g))
    return float(np.logaddexp(float(_log_one_minus(np.asarray(log_t))),contribution))


def _mean_gain(y,base_mean,expert_mean,variance):
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        result=(expert_mean-base_mean)*(2*y-expert_mean-base_mean)/(2*variance)
    if not isfinite(float(result)):raise ArithmeticError('Same-variance mean score is not representable')
    return float(result)


def score_forecast(pred,y,*,expected_prediction_sha256=None):
    """Score before updating any coefficient; the four gains are not independent likelihoods."""
    _validate(pred,expected_prediction_sha256)
    if isinstance(y,(bool,str)) or not isfinite(float(y)):raise ValueError('Observed outcome must be finite')
    y=float(y);models=pred['models'];f=pred['forecast'];g=models['current']
    log_t=-np.inf if f['log_tempering'] is None else f['log_tempering']
    log_base=f['log_tempering_base']
    log_remainder=-np.inf if f['log_tempering_remainder'] is None else f['log_tempering_remainder']
    gains=({k:0. for k in models} if log_t==-np.inf else
           {k:_mixture_gain(0.,_gaussian_gain(v,g,y),log_t,log_base,log_remainder) for k,v in models.items()})
    def moments(name):
        x=models[name]
        return _moments(np.asarray(g['mean']),np.asarray(g['observation_variance']),
                        np.asarray(x['mean']),np.asarray(x['observation_variance']),np.asarray(log_t),
                        np.asarray(log_base),np.asarray(log_remainder))
    expert_mean=float(f['tempered_mean_y'])
    null_mean,null_var=map(float,moments('birth_null'))
    # Null comparison uses the observation variance of the matched tempered
    # null distribution, not a mismatched raw-null or current-GP variance.
    result=dict(gp_gain=gains['raw_expert'],null_gain=gains['raw_expert']-gains['birth_null'],
        mean_gain=_mean_gain(y,g['mean'],expert_mean,g['observation_variance']),
        null_mean_gain=_mean_gain(y,null_mean,expert_mean,null_var),
        birth_null_gp_gain=gains['birth_null'],cut_feedback_gp_gain=gains['cut_feedback'],
        variance_only_gp_gain=gains['variance_only'],full_expert_gp_gain=gains['full_expert'],
        tempered_birth_null_mean=null_mean,tempered_birth_null_observation_variance=null_var,
        fact_success=bool(y>pred['reference_y']),prediction_sha256=pred['prediction_sha256'],
        delta_log_score=gains['raw_expert'])
    if not all(isfinite(v) for v in result.values() if isinstance(v,float)):
        raise ArithmeticError('Forecast score is not representable')
    return result


__all__=['vector_forecast','calibration_information','freeze_forecast','score_forecast',
         'MIN_EI_RATIO','MAX_EI_RATIO','NOOP_LOG_RATIO_ATOL']
