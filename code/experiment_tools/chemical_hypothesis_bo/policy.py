"""Expected improvement, explicit prediction credit and event-driven knowledge."""
from copy import deepcopy
import math
import numpy as np
from scipy.special import logsumexp
from . import PARAMETERS as P
from .hypotheses import masks


def normalize(values):
    values=np.asarray(values,float)
    if not np.isfinite(values).all():raise ValueError('Nonfinite predictive credit')
    return values-logsumexp(values)


def initial(notes):
    if not notes:return dict(base=0.)
    return dict(base=math.log(.5),**{h['id']:math.log(.5/len(notes)) for h in notes})


def edit_weights(previous,notes):
    ids={h['id'] for h in notes};old={k:math.exp(v) for k,v in previous.items()}
    base=old.get('base',0.)+sum(v for k,v in old.items() if k!='base' and k not in ids)
    kept={k:v for k,v in old.items() if k in ids};new=[h['id'] for h in notes if h['id'] not in kept]
    mass=P['new_hypothesis_mass'] if new else 0.
    values=dict(base=base*(1-mass),**{k:v*(1-mass) for k,v in kept.items()})
    values.update({k:mass/len(new) for k in new})
    return dict(zip(values,normalize([math.log(max(v,1e-300)) for v in values.values()]).tolist()))


def refresh_event(pool,state,context,query):
    if not state['calls']:return dict(allowed=True,reasons=['initial_chemical_hypotheses'])
    if state['arm']=='fixed':return dict(allowed=False,reasons=['frozen_initial_hypotheses'])
    if len(state['calls'])>=P['maximum_LLM_calls']:return dict(allowed=False,reasons=['call_budget'])
    if query>P['last_refresh_query']:return dict(allowed=False,reasons=['last_refresh_limit'])
    if query-state['calls'][-1]<P['minimum_refresh_gap']:return dict(allowed=False,reasons=['minimum_gap'])
    if query<11 and len(state['calls'])>=3:return dict(allowed=False,reasons=['reserved_late_call'])
    if context['eligible'].sum()<2:return dict(allowed=False,reasons=['native_budget_has_one_candidate'])
    reasons=[]
    if len(state['decisions'])>=3:
        recent=state['decisions'][-3:]
        if recent[-1]['incumbent_after']<=recent[0]['incumbent_before']+1e-12:reasons.append('three_query_stagnation')
    for note in state['notes']:
        rows=[e for e in state['ledger'] if e['id']==note['id'] and e['informative']][-2:]
        if len(rows)==2 and all(e['relative_log_score']<-.01 for e in rows):reasons.append('two_informative_negative_forecasts')
    coverage=any((masks(pool,h)[0]&context['eligible']).any() for h in state['notes'])
    if not coverage:reasons.append('no_preferred_coverage')
    if query>=11 and not any(q>=11 for q in state['calls']):reasons.append('reserved_late_review')
    return dict(allowed=bool(reasons),reasons=sorted(set(reasons)))


def choose(context,values,query):
    if query==16:return int(context['bo_index'])
    js=np.flatnonzero(context['eligible']);scores=values['mixture']
    return int(js[np.lexsort((context['tie_priority'][js],-context['base_logei'][js],-scores[js]))][0])


def commit(pool,state,context,values,query):
    j=choose(context,values,query);base=context['bo_index']
    return dict(query=query,arm=state['arm'],selected_id=pool.ids[j],BO_reference_id=pool.ids[base],
        differs_from_BO=bool(j!=base),epsilon=context['epsilon'],
        native_logEI_gap=float(context['base_logei'][base]-context['base_logei'][j]),
        mixture_logEI=float(values['mixture'][j]),budget_size=int(context['eligible'].sum()),
        log_weights_before=deepcopy(values['log_weights']),notes=deepcopy(state['notes']),
        predictions=[dict(id=i,mu_y=float(values['mu'][k,j]),sd_observation_y=float(context['stats']['sd_predictive_y'][j]),
            mean_shift_y=float(values['mu'][k,j]-context['stats']['mu_y'][j])) for k,i in enumerate(values['ids'])],
        native_replay_error=values['native_replay_error'],field_metadata=values['field_metadata'])


def feedback(decision,observation,conditions):
    rows=[]
    for p in decision['predictions']:
        sd=max(p['sd_observation_y'],1e-12);z=(observation['y']-p['mu_y'])/sd
        rows.append(dict(id=p['id'],log_score=-.5*z*z-math.log(sd)-.5*math.log(2*math.pi),
            mean_shift_y=p['mean_shift_y']))
    reference=rows[0]['log_score'];ledger=[]
    for row in rows[1:]:
        ledger.append(dict(id=row['id'],query=decision['query'],candidate_id=observation['candidate_id'],
            conditions=deepcopy(conditions),observed_y=observation['y'],raw_value=observation['raw_value'],
            relative_log_score=row['log_score']-reference,
            informative=bool(abs(row['mean_shift_y'])>P['informative_shift_tolerance']),
            mean_shift_y=row['mean_shift_y']))
    before=decision['log_weights_before']
    if decision['arm']=='adaptive':
        logs=normalize([before[r['id']]+r['log_score'] for r in rows]);after=dict(zip((r['id'] for r in rows),logs.tolist()))
    else:after=deepcopy(before)
    return dict(log_weights_after=after,ledger=ledger,predictive_scores=rows)
