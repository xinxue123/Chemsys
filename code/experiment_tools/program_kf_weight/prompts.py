"""The forecaster sees observations and knowledge, never a KF or arm value."""
from copy import deepcopy
from cold_start_v13 import prompts as original, policy as note_policy

PRIVATE_KEYS = {'calibration', 'prediction_reliability', 'Brier_gain',
                'KF_weight', 'calibration_weight', 'reliability_mean',
                'reliability_variance', 'p_GP', 'GP_probabilities'}


def public_only(value):
    if isinstance(value, dict):
        return {k: public_only(v) for k,v in value.items() if k not in PRIVATE_KEYS}
    if isinstance(value, list):
        return [public_only(v) for v in value]
    return deepcopy(value)


MEMORY_SYSTEM = original.MEMORY_SYSTEM + '''
Current numerical prediction calibration is private and absent. Do not invent it.
Use supplied physical observations, adverse facts, scopes and recommendation
permissions. This request does not disclose the program's arm or rank weight.
'''

REASONING_SYSTEM = original.REASONING_SYSTEM.replace(
    '{"insight":object,"knowledge_uses":list,"assessments":list}',
    '{"insight":object,"knowledge_uses":list,"assessments":list,"decision_card":object}') + '''
Candidate-forecast experiment: return exactly insight, knowledge_uses,
assessments, decision_card. The first three describe the OFFICIAL PROPOSED point,
selected by the existing confidence sort, and retain their supplied meanings.
decision_card is exactly {"claim":string,"focus":[one or two real axis names],
"purpose":"improve or learn or review","predictions":[
{"candidate_id":"a displayed ID","p_improve_incumbent":number}, ...]}.
Forecast EACH of the FIVE displayed candidates exactly once, independently,
for strict improvement over the current incumbent. On query1 there is no
incumbent, so all five probabilities must be null. Explain comparisons using
your concise public Insight. Equal probabilities are allowed when the supplied
evidence does not distinguish candidates; do not invent discrimination.
The backend chooses from this same menu by a sealed program rule and may execute
a different point from your official proposal. Only executed points are measured.
Previous proposal rationales are hypotheses, not observations. The supplied
executed_decisions ledger identifies what was actually measured.
No current KF state, numerical GP predictions, rank weight, treatment label or
scoring diagnostics are provided. Do not guess them or manipulate probabilities
to control a hidden gate. Prediction credibility is separate from recommendation
permission; a uniform-background note does not explain a candidate distinction.
'''


def memory_public(pool,state,ids,q,refresh):
    return public_only(original.memory_public(pool,state,ids,q,refresh))


def reasoning_public(root,pool,state,notes,ids,q):
    clean = dict(state, insights=public_only(state['insights']))
    public = original.common_public(pool,clean,ids,q)
    memory = public_only(note_policy.public_context(pool,notes,state['history'],ids,q,state['review_queries']))
    ref = note_policy.incumbent(pool,state['history']) if state['history'] else None
    public.update(retrieved_notes=memory,
        current_incumbent=dict(candidate_id=ref['candidate_id'],objective=ref['raw_value']) if ref else None,
        forecast_target='Selected objective strictly greater than current_incumbent.objective' if ref else None,
        executed_decisions=[dict(query=d['query'],proposed_id=d.get('proposed_id',d['selected_id']),
            executed_id=d['selected_id'],observed=d['raw_value']) for d in state['decisions']])
    return original._official(root,pool,clean,public,memory)
