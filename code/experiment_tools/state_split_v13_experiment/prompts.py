"""Public, own-prefix-only requests. No numerical GP output is shown."""
from copy import deepcopy
import json
from pathlib import Path

from knowledge_rank_v9.llm import common_public
from knowledge_rank_v9.policy import next_ids
from reasoning_memory_v12.llm import MEMORY_SYSTEM as LEGACY_EDITOR
from reasoning_memory_v13.adapter import SYSTEM_APPENDIX
from . import policy

MEMORY_SYSTEM = LEGACY_EDITOR + '''
This experiment separates prediction_reliability from recommendation.permission.
There is no combined lifecycle/weight. Preserve and use supplied adverse facts.
Retain cannot reopen suppression. Revisions create new versions and inherit
applicable adverse facts and used review counts; renaming cannot erase evidence.
A suppressed promote note can only be positively retested through a supplied
stagnation/material-condition-change review opportunity, inside the experiment
budget. Do not relabel its positive reasoning as a fresh tentative hypothesis.
Withdrawal ends new recommendations and forecasts but preserves factual warnings.
Every new note starts tentative (or suppressed when inheriting adverse facts).
Forecast credibility is not the utility of recommending the region.
'''

REASONING_SYSTEM = '''Execute the supplied OFFICIAL Reasoning BO optimization
prompt. Use complete own results, previous Insights and relevant evidence Notes
to choose ONE of the five supplied BO experiments. Return exactly
{"insight":object,"knowledge_uses":list,"assessments":list}.
insight has exactly comment,keywords,hypotheses; hypotheses contains 3..5 objects
with exactly strategy,rationale,confidence,parameter_sets. Text fields are nonempty
strings. confidence is high/medium/low. Each parameter_sets entry is exactly
{"candidate_id":"a supplied ID"}. The original stable confidence sort picks the
first point (n=1). IDs and display order are not ranks. No sixth point, GP changes,
benchmark outcome recall, invented molecular identities or other-arm information.
Current GP numerical values are hidden. Public concise rationales only.
''' + SYSTEM_APPENDIX + '''
The trajectory-wide review budget is also binding. A high calibration score
does not make a promote direction useful. Explicitly cite any supplied note
that materially supports or cautions your selected experiment. Do not evade
suppression by omitting a citation or presenting the same claim as fresh reasoning.
You may choose any menu point based on an independent, stated hypothesis.
All applicable live notes require an assessment even when not cited (shadow).
'''


def memory_public(pool, state, ids, query, refresh):
    public = common_public(pool, state, ids, query)
    ref = policy.incumbent(pool, state['history'])
    public.update(refresh=refresh, knowledge=[policy.public_note(n) for n in state['notes']],
                  next_available_ids=next_ids(state['notes']), incumbent_id=ref['candidate_id'],
                  incumbent_objective=ref['raw_value'], official_insight_history=deepcopy(state['insights']),
                  review_queries_used=list(state['review_queries']), parameters=policy.PARAMETERS)
    return public


def reasoning_public(root, pool, state, notes, ids, query):
    public = common_public(pool, state, ids, query)
    memory = policy.public_context(pool, notes, state['history'], ids, query, state['review_queries'])
    ref = policy.incumbent(pool, state['history'])
    public.update(retrieved_notes=memory, current_incumbent=dict(candidate_id=ref['candidate_id'], objective=ref['raw_value']),
                  forecast_target='Selected objective strictly greater than current_incumbent.objective')
    template = json.loads((Path(root) / 'code/chemist_system/third_party/reasoning_bo_official/src/prompts/test_prompts.json').read_text(encoding='utf-8'))['optimization_loop']['en']
    public['official_optimization_prompt'] = template.format(iteration=len(state['history']),
        target=public['domain_context']['objective'], retrieved_context=json.dumps(memory, ensure_ascii=False),
        insight_history=json.dumps(state['insights'], ensure_ascii=False),
        trial_data=json.dumps(public['own_observations'], ensure_ascii=False),
        bo_recommendations=json.dumps(public['candidate_menu'], ensure_ascii=False),
        constraint='Choose exactly ONE displayed candidate; parameter_sets contain only candidate_id references.',
        parameter_definitions=json.dumps(public['axes'], ensure_ascii=False))
    return public
