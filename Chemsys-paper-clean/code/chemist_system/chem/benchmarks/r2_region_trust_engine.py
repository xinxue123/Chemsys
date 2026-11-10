"""Only change: experimental-evidence latch for previously conflicted regions."""
from copy import deepcopy
import math
import numpy as np

from . import pointwise_hte_engine as original
from .suzuki_domain import digest

PARAMETERS={'gain_tolerance_y_squared':1e-12,'shift_tolerance_y':1e-12,
    'rule':'same canonical geometry stays blocked after a conflict until a later own experiment improves frozen guided squared error over base, then B4 is clear',
    'recovery_prediction':'prequery clipped KF coefficient times raw regional field; nuisance b excluded as in deployment',
    'other_mechanisms':'unchanged full pool, Mixed GP, qLogEI512, KF equations, B4 and event schedule'}
Cache=original.Cache


def geometry_key(spec):
    if spec is None:return None
    # Confidence, clause names, and category order cannot erase a geometry's risk.
    return digest({axis:sorted(set(values)) for axis,values in sorted(spec['allowed'].items())})


def permission(state,d,iteration):
    """Pure gate state transition; no outcomes, GP fitting or LLM calls."""
    key=geometry_key(d['region']);n=len(state['history']);book=state.setdefault('region_trust',{})
    before=deepcopy(book.get(key));entry=deepcopy(before)
    b4=bool(d['fallback']);recovered=False
    if key is not None and b4:
        if entry is None:entry={'blocked':True,'blocked_observation_count':n,'blocked_query':iteration,
            'pending_evidence':None,'recoveries':0,'block_events':[]}
        entry['blocked']=True
        if n>entry['blocked_observation_count']:
            entry.update(blocked_observation_count=n,blocked_query=iteration,pending_evidence=None)
        marker={'query':iteration,'observation_count':n}
        if not entry['block_events'] or entry['block_events'][-1]!=marker:entry['block_events'].append(marker)
    elif key is not None and entry is not None and entry['blocked']:
        ev=entry.get('pending_evidence')
        if ev is not None and entry['blocked_observation_count']<ev['after_observation_count']<=n:
            entry['blocked']=False;entry['recoveries']+=1;entry['recovered_query']=iteration
            entry['recovery_evidence']=deepcopy(ev);entry['pending_evidence']=None;recovered=True
    if key is not None and entry is not None:book[key]=entry
    blocked=bool(entry and entry['blocked'])
    alpha=0. if blocked else float(d['amplitude'])
    return {'geometry_key':key,'observation_count':n,'b4_fallback':b4,'trust_blocked':blocked,
        'recovered':recovered,'candidate_amplitude_before_trust':float(d['amplitude']),
        'amplitude_after_trust':alpha,'entry_before':before,'entry_after':deepcopy(entry)}


def setup(cache,state,iteration):
    d=original.setup(cache,state,iteration);gate=permission(state,d,iteration)
    if gate['trust_blocked'] and d['amplitude']!=0.:
        d['amplitude']=0.
        d['score'],d['shift_std'],d['guided_meta']=cache.score(state['history'])
        d['selected']=d['reference']
    d['fallback']=bool(d['fallback'] or gate['trust_blocked']);d['trust']=gate
    return d


def feedback(state,d,iteration,y,updated_bank):
    """Score a prediction frozen BEFORE the observed outcome; defer recovery."""
    key=d['trust']['geometry_key'];entry=state.setdefault('region_trust',{}).get(key)
    if entry is None or not entry['blocked']:return None
    j=d['selected'];mu=float(d['stats']['mu_y'][j])
    # KF measurement stores the physical field before any deployment/fallback.
    raw=float(d['_measurement']['raw_region_shift_y'])
    alpha=float(d['prior']['amplitude']);probe=alpha*raw
    gain=(float(y)-mu)**2-(float(y)-mu-probe)**2
    after_n=len(state['history'])+1
    valid=(after_n>entry['blocked_observation_count'] and abs(probe)>PARAMETERS['shift_tolerance_y']
        and gain>PARAMETERS['gain_tolerance_y_squared'] and float(updated_bank['mean'][1])>0.)
    receipt={'query':iteration,'geometry_key':key,'before_observation_count':len(state['history']),
        'after_observation_count':after_n,'GP_mean_y':mu,'raw_region_shift_y':raw,
        'frozen_probe_amplitude':alpha,'frozen_probe_shift_y':probe,'observed_y':float(y),
        'squared_error_gain_y_squared':float(gain),'positive_evidence':bool(valid),
        'KF_coefficient_after_feedback':float(updated_bank['mean'][1])}
    entry['last_feedback']=deepcopy(receipt)
    if valid:entry['pending_evidence']=deepcopy(receipt)
    # A non-supporting later observation invalidates earlier pending evidence.
    else:entry['pending_evidence']=None
    return receipt


event_reasons=original.event_reasons
