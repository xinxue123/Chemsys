"""Event scheduling and native qLogEI constrained only by eligible hypotheses."""
import numpy as np
from . import PARAMETERS as P
from . import knowledge as k


def stagnated(state):
    tail=state['decisions'][-P['stagnation_window']:]
    return len(tail)==P['stagnation_window'] and all(r['decision']['incumbent_before'] is not None and
        r['raw_value']<=r['decision']['incumbent_before']+1e-12 for r in tail)


def event(state,scopes,available,incumbent,query):
    if not state['calls']:return dict(call=True,reasons=['initial_blinded_hypotheses'],stagnated=False)
    mask,active=k.filter_mask(scopes,state['notes'],available,incumbent);reasons=[]
    if not active:reasons.append('no_eligible_guidance')
    elif not mask.any():reasons.append('scope_exhausted_or_conflicting')
    if any(n['permission']=='paused' for n in state['notes']):reasons.append('paused_hypotheses_require_review')
    if stagnated(state):reasons.append('three_query_stagnation')
    hold=None
    if not reasons:hold='no_trigger'
    elif len(state['calls'])>=P['maximum_LLM_calls']:hold='LLM_budget_exhausted'
    elif query>P['last_refresh_query']:hold='past_last_refresh_query'
    elif query-state['calls'][-1]['query']<P['minimum_refresh_gap']:hold='minimum_refresh_gap'
    elif len(state['calls'])==P['maximum_LLM_calls']-1 and query<P['reserved_late_query']:hold='reserved_late_refresh'
    return dict(call=hold is None,reasons=reasons,hold=hold,stagnated=stagnated(state))


def argmax(context,mask):
    js=np.flatnonzero(mask)
    return int(js[np.lexsort((context['tie_priority'][js],-context['base_logei'][js]))[0]])


def select(context,scopes,notes,incumbent):
    mask,active=k.filter_mask(scopes,notes,context['available'],incumbent)
    fallback=not active or not mask.any()
    j=context['bo_index'] if fallback else argmax(context,mask)
    causes={};details=[]
    for note in active:
        own=scopes.mask(note['scope']);b=context['bo_index']
        excludes=not own[b] if note['direction']=='favour' else bool(own[b])
        removed,_=k.filter_mask(scopes,notes,context['available'],incumbent,note['semantic_key'])
        alternate=argmax(context,removed) if removed.any() else b
        changes=not fallback and j!=b and (excludes or alternate!=j)
        causes[note['semantic_key']]=bool(changes)
        details.append(dict(id=note['id'],semantic_key=note['semantic_key'],excludes_unrestricted_BO=bool(excludes),
            choice_without_this_rule=scopes.pool.ids[alternate],caused_selection=bool(changes)))
    gap=float(context['base_logei'][context['bo_index']]-context['base_logei'][j])
    return dict(index=int(j),mask=mask if not fallback else context['available'].copy(),
        proposed_scope_mask=mask,active=active,causes=causes,causal_diagnostics=details,
        reason='native_full_pool_qLogEI_no_eligible_hypothesis' if not active else
            'native_full_pool_qLogEI_empty_scope_fallback' if fallback else 'hypothesis_constrained_native_qLogEI',
        fallback=fallback,opportunity_gap=gap)
