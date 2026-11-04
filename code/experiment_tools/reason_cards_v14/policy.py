"""Bounded event editing, executable notes, and V14 proposal adjudication."""
from copy import deepcopy
from cold_start_v13 import policy as old_policy, control as old, adapter as old_adapter
from . import controller

PARAMETERS=dict(old_policy.PARAMETERS,**controller.PARAMETERS)
PARAMETERS['late_edit_round']=controller.PARAMETERS['late_edit_query']


def edit_event(pool,state,ordered,query):
    calls=state['edit_calls']
    if len(state['history'])<controller.PARAMETERS['minimum_editor_observations']:
        return dict(allowed=False,reasons=['insufficient_comparative_observations'])
    if not calls:
        return dict(allowed=True,reasons=['initial_comparative_knowledge'])
    if len(calls)>=4:
        return dict(allowed=False,reasons=['edit_budget_exhausted'])
    if query-calls[-1]<3:
        return dict(allowed=False,reasons=['minimum_edit_gap'])
    recent=state['decision_channel']['events'][-2:]
    severe=any(e.get('local_suppression') for e in recent)
    if len(calls)>=3 and query<controller.PARAMETERS['late_edit_query'] and not severe:
        return dict(allowed=False,reasons=['reserved_late_edit'])
    reasons=[]
    if severe:reasons.append('new_local_adverse_evidence')
    if len(recent)==2 and all(e.get('calibration',{}).get('Brier_gain',0)<0 for e in recent):
        reasons.append('consecutive_negative_prediction_skill')
    if old.stagnation_event(state['history'],query):reasons.append('stagnation')
    live=[n for n in state['notes'] if n['recommendation']['permission']!='withdrawn']
    ids=[pool.ids[int(j)] for j in ordered]
    if live and not any(old.in_scope(n['scope'],old_adapter.conditions(pool,cid)) for n in live for cid in ids):
        reasons.append('live_notes_have_no_executable_menu_coverage')
    return dict(allowed=bool(reasons),reasons=sorted(reasons))


def apply_edits(raw,pool,state,query):
    plan=old_policy.apply_edits(raw,pool,state,query)
    measured={h['candidate_id'] for h in state['history']}
    for review in plan['reviews']:
        if not review.get('accepted') or review.get('outcome') not in ('add','revise'):
            continue
        identity=review['id'],review['version']
        note=next(n for n in plan['notes'] if (n['id'],n['version'])==identity)
        witnesses=[cid for cid in pool.ids if cid not in measured and old.in_scope(note['scope'],old_adapter.conditions(pool,cid))]
        if len(witnesses)<2:
            plan['notes']=[n for n in plan['notes'] if n['id']!=review['id']]
            plan['notes'].extend(deepcopy([n for n in state['notes'] if n['id']==review['id']]))
            review.update(accepted=False,reason='Need at least two unqueried executable witnesses; prior version preserved')
        else:
            review.update(unqueried_witnesses=len(witnesses))
    return plan


def validate_decision(root,pool,notes,state,context,ids,query,raw):
    return controller.decide(root,pool,notes,state,context,ids,query,raw)
