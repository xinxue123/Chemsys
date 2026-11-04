"""Prospective utility ledgers for the frozen-birth numeric expert.

Four correlated utility comparisons are safety constraints, not independent
likelihoods to multiply. New versions inherit only applicable negative evidence;
one physical observation is counted once even if multiple old rules scored it.
"""
from copy import deepcopy
from math import isfinite, tanh

from .knowledge_llm import digest

GAINS = ('gp_gain','null_gain','mean_gain','null_mean_gain')
MAX_WEIGHT = .5


def initial_ledger():
    return dict(events=[],seen=[],trusted_weight=0.,weight=0.)


def evidence_summary(h):
    ledger=h['ledger'];groups={}
    for source,events in (('local',ledger['events']),
                          ('inherited',[e for e in h.get('inherited_negative_evidence',[]) if e.get('applicable')])):
        for event in events:
            key=(event['query_round'],event['point_id'])
            groups.setdefault(key,{'local':[],'inherited':[]})[source].append(event)
    local={name:0. for name in GAINS};total=local.copy()
    for group in groups.values():
        for name in GAINS:
            own=[float(e.get(name,0.)) for e in group['local']]
            negative=[min(0.,float(e.get(name,0.))) for e in group['inherited']]
            local[name]+=min(own,default=0.)
            total[name]+=min(own+negative,default=0.)
    if not all(isfinite(x) for x in list(local.values())+list(total.values())):
        raise ValueError('Nonfinite evidence utility')
    formula=min([MAX_WEIGHT]+[max(0.,tanh(total[name]/2)) for name in GAINS])
    trusted=min(formula,max(0.,float(ledger['trusted_weight'])))
    return dict(local_scores=local,effective_scores=total,formula_weight=formula,
        trusted_weight=trusted,observations=len(groups),local_observations=len(ledger['seen']),
        inherited_constraint_events=sum(bool(g['inherited']) for g in groups.values()),
        interpretation='Correlated utility safety bounds; never multiplied as independent likelihoods.')


def weight(h):
    return evidence_summary(h)['trusted_weight'] if h['status']=='active' else 0.


def record_feedback(h,pred,observation,*,expected_prediction_sha256=None):
    from .knowledge_numeric_forecast import score_forecast
    if h['status']=='retracted':raise ValueError('Cannot update a retired version')
    if (pred['hypothesis_id']!=h['id'] or pred['version']!=h['version']
        or pred['semantic_key']!=h['semantic_key']):raise ValueError('Forecast belongs to another rule/version')
    t=pred['query_round'];point=pred['point_id'];ledger=h['ledger']
    if (type(t)is not int or t<h['created_round'] or point in ledger['seen']
        or point in {x['candidate_id'] for x in h['birth_prefix']}
        or any(e['query_round']>=t for e in ledger['events'])):
        raise ValueError('Repeated, retrospective, or out-of-order feedback')
    if observation['candidate_id']!=point or observation.get('query_round',t)!=t:
        raise ValueError('Observation does not match frozen prediction')
    y=float(observation['y'])
    if not isfinite(y):raise ValueError('Nonfinite experiment result')
    scored=score_forecast(pred,y,expected_prediction_sha256=expected_prediction_sha256)
    gains={name:float(scored[name]) for name in GAINS}
    if not all(isfinite(x) for x in gains.values()):raise ValueError('Nonfinite prospective utility')
    before=evidence_summary(h);deployed=weight(h)
    event=dict(hypothesis_id=h['id'],version=h['version'],point_id=point,
        reference_id=pred['reference_id'],reference_y=pred['reference_y'],query_round=t,
        prediction=deepcopy(pred),observed_y=y,comparison_success=bool(y>pred['reference_y']),
        **gains,forecast_scores=deepcopy(scored))
    event['event_id']=digest(event)
    # All validation/scoring precedes mutation. Calibration itself is performed
    # by the joint model on the next unique post-birth prefix, not this ledger.
    ledger['events'].append(event);ledger['seen']=sorted(set(ledger['seen'])|{point})
    formula=evidence_summary(h)['formula_weight']
    positive=all(gains[name]>0 for name in GAINS)
    increment=min(max(0.,tanh(gains[name]/2)) for name in GAINS) if positive else 0.
    ledger['trusted_weight']=min(formula,before['trusted_weight']+increment)
    ledger['weight']=weight(h)
    after=evidence_summary(h)
    return dict(event_id=event['event_id'],hypothesis_id=h['id'],version=h['version'],
        point_id=point,reference_id=pred['reference_id'],reference_y=pred['reference_y'],query_round=t,
        **gains,delta_log_score=gains['gp_gain'],weight_before=deployed,weight_after=ledger['weight'],
        trusted_weight_before=before['trusted_weight'],trusted_weight_after=after['trusted_weight'],
        effective_scores=after['effective_scores'],formula_weight=after['formula_weight'],
        recovery_credit_limit=increment,comparison_success=event['comparison_success'],
        comparison_is_credit_gate=False,transition='increment_limited_four_positive_utilities' if positive else 'nonincreasing_after_failed_utility')
