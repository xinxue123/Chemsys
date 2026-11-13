"""Dynamic predictive-skill filter and automatic knowledge lifecycle."""
from copy import deepcopy
from math import isfinite, sqrt
from scipy.special import ndtr

PARAMETERS=dict(prior_mean=0.,prior_variance=.25,process_variance=.0025,
    measurement_variance_floor=.0025,maximum_weight=.8,pause_bad_streak=2,
    retract_bad_streak=4,pause_probability=.20,retract_probability=.05,
    resume_probability=.60,resume_good_streak=2,minimum_pause_queries=2,streak_gain_threshold=.02,atol=1e-12)


def initial():
    return dict(mean=0.,variance=.25,weight=.4,probability_positive=.5,
        bad_streak=0,good_streak=0,updates=0,local_updates=0,
        status='active',paused_round=None,retracted_round=None)


def update(state,p,g,success,t,*,prior_transport=False):
    if (type(t) is not int or t<1 or type(success) is not bool
        or any(type(x) not in (int,float) or not isfinite(x) or not 0<=x<=1 for x in (p,g))):
        raise ValueError('Invalid prospective probability or feedback')
    if state['status']=='retracted' and not prior_transport:raise ValueError('Retired knowledge cannot receive feedback')
    before=deepcopy(state);out=deepcopy(state);s=float(success)
    z=(s-g)**2-(s-p)**2
    r=max(PARAMETERS['measurement_variance_floor'],4*(p-g)**2*g*(1-g))
    pp=state['variance']+PARAMETERS['process_variance'];gain=pp/(pp+r)
    mean=state['mean']+gain*(z-state['mean'])
    variance=(1-gain)**2*pp+gain*gain*r
    if not isfinite(mean) or not isfinite(variance) or variance<=0:raise ArithmeticError('Invalid KF posterior')
    probability=float(ndtr(mean/sqrt(variance)));raw=PARAMETERS['maximum_weight']*probability
    weight=min(raw,state['weight']) if z<=PARAMETERS['atol'] else raw
    bad=state['bad_streak']+1 if z<-PARAMETERS['streak_gain_threshold'] else 0
    good=state['good_streak']+1 if z>PARAMETERS['streak_gain_threshold'] else 0
    out.update(mean=mean,variance=variance,weight=weight,probability_positive=probability,
        bad_streak=bad,good_streak=good,updates=state['updates']+1,
        local_updates=state['local_updates']+int(not prior_transport))
    reason='negative_feedback_downgrade' if z<-PARAMETERS['atol'] else 'prospective_update'
    if not prior_transport:
        n=out['local_updates']
        if bad>=4 or n>=4 and probability<=PARAMETERS['retract_probability']:
            out.update(status='retracted',retracted_round=t,weight=0.);reason='automatic_retract'
        elif state['status']=='active' and (bad>=2 or n>=2 and probability<=PARAMETERS['pause_probability']):
            out.update(status='paused',paused_round=t);reason='automatic_pause'
        elif (state['status']=='paused' and t-state['paused_round']>=PARAMETERS['minimum_pause_queries']
              and good>=2 and probability>=PARAMETERS['resume_probability']):
            out.update(status='active',paused_round=None);reason='automatic_resume'
    return out,dict(query=t,p_LLM=p,p_GP=g,success=success,Brier_gain=z,
        measurement_variance=r,predicted_variance=pp,kalman_gain=gain,
        before=before,after=deepcopy(out),reason=reason,prior_transport=prior_transport)


def deployment_weight(state):
    return state['weight'] if state['status']=='active' else 0.
