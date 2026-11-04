"""Independent proposals include all-five forecasts and closed-loop reasons."""
from copy import deepcopy
from cold_start_v13 import prompts as old_prompts, policy as old_policy
from . import controller, policy

MEMORY_SYSTEM=old_prompts.MEMORY_SYSTEM+'''
V14: the decision-card ledger tracks every selected experiment's prospective
forecast, including free reasoning without a regional Note. Distinguish a
productive experiment from evidence for a named chemical axis. A direction test
needs an actual focus change, unchanged categorical background and bounded
continuous background drift. An unchanged focus cannot earn direction credit.
Approximate continuous contrasts remain local predictive evidence, not causality.
New/revised Notes must cover at least TWO still-unqueried physical inputs.
A scoped warning can remain historical without controlling unrelated choices.
Continuous multivariable observations may support a local JOINT predictive
hypothesis; do not claim isolated causal effects from confounded measurements.
Inspect rejected edits and the last actual card feedback; avoid spending a
refresh merely rephrasing a warning that covers no executable candidates.
'''

REASONING_SYSTEM=old_prompts.REASONING_SYSTEM.replace(
    '{"insight":object,"knowledge_uses":list,"assessments":list}',
    '{"insight":object,"knowledge_uses":list,"assessments":list,"decision_card":object}')+'''
V14 overrides the response envelope: return exactly insight, knowledge_uses,
assessments, decision_card. The first three retain their supplied V13 meanings:
knowledge_uses/assessments refer to your OFFICIAL PROPOSED candidate, not a
possible backend replacement. Do not invent a note or cite a uniform-background
note as evidence that distinguishes otherwise different candidates.
decision_card is exactly:
{"claim":"a concise testable reason for this proposed combination",
 "focus":["one or two real named physical axes"],
 "purpose":"improve or learn or review",
 "predictions":[{"candidate_id":"...","p_improve_incumbent":0.3}, ...]}.
Include each of the FIVE menu candidates exactly once. Predict STRICT
improvement over the current incumbent independently for each, even when no
regional Note applies. On query1 there is no incumbent: all five probabilities
must be null. Never fabricate a first-round improvement probability.
Make an actual comparison of alternatives, not three generic arguments for a
single arbitrary combination. Only supplied observations support conclusions.
focus names the main axes the proposal is trying to investigate; it is not a
claim that their effects have been isolated. Small-step comparisons can learn
chemistry but have an opportunity cost when the incumbent itself is poor.
purpose=learn is a limited information test (at most two overrides per run).
purpose=review is only a bounded eligible test of materially changed conditions
or true stagnation after adverse evidence. A right failure forecast provides a
warning; it is never evidence of positive recommendation utility.
The backend retains GP predictions unchanged and may execute its GP reference
when a proposal lacks discriminating value, exceeds its opportunity budget,
or repeats a locally suppressed pattern. Previously proposed points were NOT
measured when the ledger says a different point was executed. Never fabricate
feedback for an unexecuted proposal or unmeasured alternatives.
'''


def channel_public(pool,state,ids,query):
    channel=state['decision_channel']
    cards=[]
    for d in state['decisions'][-6:]:
        cards.append(dict(query=d['query'],proposed_id=d['proposed_id'],executed_id=d['selected_id'],
            decision_card=d['decision_card'],observed=d['raw_value'],
            deployment_reason=d['deployment']['reason'],feedback=d['card_feedback']))
    # Current numerical GP means, SDs, ranks and probabilities stay hidden.
    return dict(prediction_calibration=deepcopy(channel['calibration']),recent_closed_cards=cards,
        local_adverse_facts=deepcopy(channel['facts'][-8:]),
        learning_budget_remaining=max(0,2-len(channel['learning_queries'])),
        review_budget_remaining=max(0,2-len(state['review_queries'])-len(channel['review_queries'])),
        review_availability=[dict(candidate_id=cid,**controller.risk_context(pool,state,cid,query,list(pool.axes))) for cid in ids],
        instructions='Prediction skill does not restore recommendation permission; no feedback exists for unexecuted proposals.')


def memory_public(pool,state,ids,query,refresh):
    public=old_prompts.memory_public(pool,state,ids,query,refresh)
    public.update(decision_channel=channel_public(pool,state,ids,query),parameters=policy.PARAMETERS)
    return public


def reasoning_public(root,pool,state,notes,ids,query):
    public=old_prompts.reasoning_public(root,pool,state,notes,ids,query)
    public['decision_channel']=channel_public(pool,dict(state,notes=notes),ids,query)
    remaining=max(0,2-len(state['review_queries'])-len(state['decision_channel']['review_queries']))
    public['retrieved_notes']['review_query_budget_remaining']=remaining
    if not remaining:
        for note in public['retrieved_notes']['knowledge']:
            note['eligible_review_candidates']=[]
    return public
