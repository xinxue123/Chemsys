"""Frozen molecular GP decisions, with an exposed native identifiability diagnostic."""
from .stable_structure_engine import Cache,original,feedback,priority,ordered,competition_ranks,summary
from . import stable_structure_engine as fixed
from . import stable_structure_fit as fit


def setup(cache,state,t):
    d=fixed.setup(cache,state,t);record=d['GP_health']
    if record.get('model_kind')=='molecular_categorical':diag=record['identification']
    else:
        _,v=cache.model(state['history']);diag=fit.diagnose(v['model'],cache.pool['axes'])
    pool=cache.pool;ix={k:i for i,k in enumerate(pool['ids'])}
    seen={a:{pool['points'][ix[h['candidate_id']]][a] for h in state['history']} for a in pool['axes']}
    d['model_identifiability']={**diag,'extreme_unsupported':diag['trigger'],
        'unseen_levels':{a:[v for v in pool['levels'][a] if v not in seen[a]] for a in pool['axes']}}
    return d


def event_reasons(state,d,t):
    reasons=fixed.event_reasons(state,d,t)
    recent=next((c for c in state['recombination_calls'] if c['iteration']==t-1),None)
    if recent and len(state['calls'])<3 and not any(c['iteration']==t for c in state['calls']):
        if recent['validated']['plan']['violated_preferences'] and recent['observed_percent']>=recent['incumbent_before_percent']+.5:
            return list(dict.fromkeys(reasons+['successful_joint_counterexample']))
    return reasons
