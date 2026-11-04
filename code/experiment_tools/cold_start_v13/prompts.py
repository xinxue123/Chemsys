"""Own-prefix-only cold-start prompts, with explicit first-round semantics."""
from copy import deepcopy
import json
from pathlib import Path

from chem.benchmarks.knowledge_domain_context import task_context
from knowledge_rank_v9.llm import point, RBO_SYSTEM as WARM_RBO_SYSTEM
from knowledge_rank_v9.policy import next_ids
from state_split_v13_experiment.prompts import MEMORY_SYSTEM
from state_split_v13_experiment.prompts import REASONING_SYSTEM as WARM_REASONING_SYSTEM
from . import QUERIES, policy

COLD_APPENDIX = '''
This is a ZERO-observation start. There are no free initial experiments.
All20 experiment queries, including the first, count toward the budget.
On query1 the five menu inputs are sampled by a shared data-free random order;
there is no fitted GP, qLogEI or incumbent. Use cautious domain hypotheses only.
The original stable confidence sort still chooses one supplied point.
For V13 on query1, knowledge_uses=[] and assessments=[] are required because no
versioned note or incumbent exists yet. Improvement forecasts begin only after
an own observation exists. Never invent bootstrap results or access old runs.
'''
REASONING_SYSTEM = WARM_REASONING_SYSTEM + COLD_APPENDIX
RBO_SYSTEM = WARM_RBO_SYSTEM.replace(
    'Initial4 points are already measured and shared with BO.\nNo new warm-start experiment may be suggested.',
    'There are zero initial observations. All proposed experiments are paid queries.') + COLD_APPENDIX


def common_public(pool, state, ids, query):
    domain = task_context(pool.task, pool)
    domain.pop('source_references', None)
    return dict(task=pool.task, query=query, remaining_queries=QUERIES + 1 - query,
        total_query_budget=QUERIES, free_initial_observations=0,
        objective_unit=pool.unit, domain_context=domain,
        axes=[dict(name=a, kind=k) for a,k in zip(pool.axes,pool.kinds)],
        own_observations=[dict(point(pool,r['candidate_id']), objective=r['raw_value']) for r in state['history']],
        candidate_menu=[point(pool,key) for key in ids],
        interface='shared5candidate_sequential1; neutral physical display order',
        candidate_source='data_free_random_order' if not state['history'] else 'native_GP_qLogEI_top5')


def memory_public(pool, state, ids, query, refresh):
    if not state['history']:
        raise ValueError('Versioned incumbent-relative notes need at least one observation')
    public = common_public(pool,state,ids,query)
    ref = policy.incumbent(pool,state['history'])
    public.update(refresh=refresh, knowledge=[policy.public_note(n) for n in state['notes']],
        next_available_ids=next_ids(state['notes']), incumbent_id=ref['candidate_id'],
        incumbent_objective=ref['raw_value'], official_insight_history=deepcopy(state['insights']),
        review_queries_used=list(state['review_queries']), parameters=policy.PARAMETERS)
    return public


def _official(root, pool, state, public, memory):
    template = json.loads((Path(root) / 'code/chemist_system/third_party/reasoning_bo_official/src/prompts/test_prompts.json').read_text(encoding='utf-8'))['optimization_loop']['en']
    public['official_optimization_prompt'] = template.format(iteration=len(state['history']),
        target=public['domain_context']['objective'], retrieved_context=json.dumps(memory,ensure_ascii=False),
        insight_history=json.dumps(state['insights'],ensure_ascii=False),
        trial_data=json.dumps(public['own_observations'],ensure_ascii=False),
        bo_recommendations=json.dumps(public['candidate_menu'],ensure_ascii=False),
        constraint='Choose exactly ONE displayed candidate; parameter_sets contain only candidate_id references.',
        parameter_definitions=json.dumps(public['axes'],ensure_ascii=False))
    return public


def reasoning_public(root, pool, state, notes, ids, query):
    public = common_public(pool,state,ids,query)
    memory = policy.public_context(pool,notes,state['history'],ids,query,state['review_queries'])
    ref = policy.incumbent(pool,state['history']) if state['history'] else None
    public.update(retrieved_notes=memory,
        current_incumbent=dict(candidate_id=ref['candidate_id'],objective=ref['raw_value']) if ref else None,
        forecast_target='Selected objective strictly greater than current_incumbent.objective' if ref else None)
    return _official(root,pool,state,public,memory)


def rbo_public(root, pool, state, ids, query):
    public = common_public(pool,state,ids,query)
    return _official(root,pool,state,public,None)
