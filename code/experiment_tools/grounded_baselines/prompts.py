"""Shared scientific background and blinded priors; method-specific interfaces."""
from copy import deepcopy
import json
from grounded_first_start import proposal
from knowledge_rank_v9.llm import point
from . import PARAMETERS as P

LGBO_SYSTEM='''You are the scientific preference module in an LGBO optimization loop.
Use only the supplied verified chemistry/process background, BLINDED initial
hypotheses, and this trajectory's own observations. Do not browse, inspect other
files, recall benchmark outcomes, infer ordinal meaning from candidate IDs, or
invent unmeasured outcomes. Explain scientific choices concisely in English,
not private reasoning. Original physical units and molecular identities matter.

Return exactly {"summary":string,"mode":"point"|"region"|"none",
"confidence":number,"conditions":object|null,"scope":object,"rationale":string}.
confidence must be strictly between0 and1. Supply one preference each round.
For mode=region: conditions=null; scope maps one or more actual input axes to
{"bounds":[lower,upper]} for continuous physical values or
{"allowed":[exact listed categorical strings/SMILES]} for categories. Omitted
axes are free. The region must contain an unqueried executable input. For
mode=point: specify all physical conditions and scope={}; software maps to the
nearest unqueried executable input preserving categorical identities, without
using outcomes. For mode=none: conditions=null and scope={}; native BO is used.

The official region-lift rule turns confidence into delta=3*Phi_inverse(confidence)
and changes the surrogate mean by lambda*posterior_covariance_to_region,
leaving its covariance unchanged. Values above0.5 favour the region, below0.5
suppress it, and0.5 yields no lift. The unchanged qLogEI implementation chooses
the next point over the ENTIRE unqueried pool, not just inside the preferred
region. There is no KF, event gate or hard filtering in this comparator.
You receive no current GP mean, uncertainty or acquisition ranking. Reconsider
the prior preference using measured feedback. Initial hypotheses are uncertain
scientific priors rather than empirical findings. Keep summary/rationale brief.'''

RBO_SYSTEM='''Execute the supplied official Reasoning BO optimization prompt in
this5-to1 sequential interface. Use only the supplied verified scientific
background, blinded initial knowledge, own observations and exact displayed
candidates. Do not recall benchmark outcomes or inspect other files. Candidate
membership comes from GP; display order and IDs do not reveal acquisition rank.
Return exactly comment,keywords,hypotheses. Provide3..5 hypotheses, each with
strategy,rationale,confidence,parameter_sets. confidence=high/medium/low.
Each parameter_sets entry must be exactly {"candidate_id":"one displayed ID"}.
The official stable confidence sort chooses the FIRST point (n=1). No sixth
candidate, invented measurement or other-arm information is permitted.
Give concise English scientific explanations, not private reasoning.
There are0 free observations. The common independent LLM first experiment
has already consumed paid query1; select one more point per round until20.
This comparator has no KF, automatic knowledge permissions or event-gated edits.'''

def common(pool,state,prior):
    return dict(background=proposal.background(pool),objective_unit=pool.unit,
        experimental_budget=dict(initial=0,total_paid_queries=20,
            observed=len(state['history']),remaining=20-len(state['history'])),
        own_observations=[dict(query=i+1,candidate_id=h['candidate_id'],
            conditions=deepcopy(pool.points[pool.ix[h['candidate_id']]]),objective_raw=h['raw_value'])
            for i,h in enumerate(state['history'])],
        blinded_initial_hypotheses=deepcopy(prior),
        incumbent_raw=max(h['raw_value'] for h in state['history']),
        current_GP_predictions_or_ranking_supplied=False)

def lgbo(pool,state,prior):
    public=common(pool,state,prior)
    seen={h['candidate_id'] for h in state['history']}
    public.update(input_only_examples=[p for p in proposal.examples(pool) if p['candidate_id'] not in seen],
        previous_preferences=deepcopy(state['preferences']),all_unqueried_pool_inputs_allowed=True)
    return public

def rbo(root,pool,state,prior,ids):
    public=common(pool,state,prior)
    public.update(candidate_menu=[point(pool,key) for key in ids],
        axes=[dict(name=a,kind=k) for a,k in zip(pool.axes,pool.kinds)],
        candidate_source='native_GP_qLogEI_top5; neutral display order')
    template=json.loads((root/'code/chemist_system/third_party/reasoning_bo_official/src/prompts/test_prompts.json').read_text(encoding='utf-8'))['optimization_loop']['en']
    public['official_optimization_prompt']=template.format(iteration=len(state['history']),
        target=public['background']['objective'],retrieved_context=json.dumps(prior),
        insight_history=json.dumps(state['insights'],ensure_ascii=False),
        trial_data=json.dumps(public['own_observations'],ensure_ascii=False),
        bo_recommendations=json.dumps(public['candidate_menu'],ensure_ascii=False),
        constraint='Choose exactly ONE displayed candidate; parameter_sets contain only candidate_id references.',
        parameter_definitions=json.dumps(public['axes'],ensure_ascii=False))
    return public
