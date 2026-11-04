"""Separate independent proposal calls from current-menu forecasting calls."""
from copy import deepcopy
from math import isfinite
import numpy as np
from knowledge_rank_v9 import llm as previous,policy
from . import proposals

request=previous.request
PROPOSAL_SYSTEM='''Propose locally testable chemistry/process knowledge for sequential
optimization. Use ONLY public input semantics and your own measured prefix. No GP
scores, ranks or predictions are shown. Do not recall benchmark outcomes or infer
molecular identities behind opaque labels. Candidate IDs convey no ranking.

Return exactly {"summary":string,"actions":list}. At most3 actions,4 live rules.
An action has exactly op,id,claim,focus,effect,scope,reason,probe.
op=add/retain/revise/retract; use supplied next_available_ids for add.
For add/revise, focus contains1..2 real axes; effect=promote/avoid; scope maps
optional axes to {"bounds":[0..1,0..1]} or {"allowed":[exact category strings]}.
Omitted axes apply globally. Prefer hypotheses that distinguish promising choices,
such as an uncertain conditional trend/interaction, not a generic high-yield claim.
probe={"target":{ALL axis names: values},"why_informative":string,
"if_above":string,"if_below":string}. Continuous target values are normalized0..1;
categorical targets are exact supplied labels. The input-only coverage examples
are aids, not a whitelist: your target projects to the nearest unqueried point
in the FULL frozen pool, with exact categorical match, inside scope. Invent no new
variable or molecular identity. Explain which future condition choices each
outcome would favor. This is an observational comparison, not causal proof.
The engine selects the highest known comparable reference (same non-focus
categories and continuous distance<=.25); if none, it uses the known incumbent
and labels the comparison as unmatched. You will see the actual fixed reference
before forecasting. Never invent its value. At least2 unused points must be in
the declared scope. Broad scopes are allowed but require a defensible claim.
retain/retract use claim:null,focus:[],effect:null,scope:{},probe:null and a reason.
Revisions retire old versions; negative evidence and task probe budgets survive.
A paused rule cannot resume by retain. You may abstain with actions:[].
Each text field<=700 characters, summary<=1000. JSON only; concise public
rationales, no private reasoning trace. Do not make proposals follow a GP shortlist.'''

FORECAST_SYSTEM='''Forecast candidate outcomes using the supplied immutable local
chemistry/process hypotheses and ONLY your own measured observations. No GP
numerical values, ordering or membership labels are supplied. The menu mixes
BO and independently proposed candidates; IDs/order have no chemical meaning.
Do not edit rules in this call. Return exactly {"summary":string,"assessments":list}.
For EACH latest non-retracted rule (including paused), supply exactly
{"id":rule_id,"probabilities":{EVERY displayed candidate_id:p}} where p in[0,1]
is P(the NEXT measured objective is STRICTLY GREATER than that rule's FIXED known
reference). No yield magnitudes or unobserved values. Equal probabilities and
uncertainty are allowed; don't manufacture confident distinctions. Forecast all
menu points, but only selected in-scope points update that rule's KF. KF measures
incremental Brier skill vs GP, not chemical causality. Do not recall benchmark
targets, invent molecular identities, treat IDs as ranks, or use other arms.
summary<=1000 characters. JSON only with concise public explanation.'''


def knowledge(hs):
    return [dict(id=h['id'],version=h['version'],claim=h['claim'],focus=h['focus'],effect=h['effect'],
        scope=h['scope'],reference_id=h['reference_id'],reference_y=h['reference_y'],probe=h['probe'],
        kf=deepcopy(h['kf']),reference_selection=h['reference_selection'],
        recent_feedback=[{k:e[k] for k in ('query','point_id','success','p_LLM','Brier_gain','reason')} for e in h['events'][-4:]],
        inherited_negative_count=len(h['inherited_negative'])) for h in hs]


def proposal_public(pool,state,t,refresh,ties):
    menu=proposals.coverage_menu(pool,state['history'],ties)
    public=previous.common_public(pool,state,[],t)
    public.pop('candidate_menu');public['interface']='input_only_full_pool_target_proposals; no_GP_shortlist'
    public['domain_context']['candidate_policy']='Targets map into the full frozen pool; no new conditions or objective lookup.'
    public.update(refresh=refresh,knowledge=knowledge(state['hypotheses']),
        next_available_ids=policy.next_ids(state['hypotheses']),
        input_coverage_examples=[previous.point(pool,pool.ids[j]) for j in menu],
        target_axes={axis:dict(kind=kind,**({'levels':sorted({row[k] for row in pool.features})} if kind=='categorical'
            else {'normalized_bounds':[0.,1.]})) for k,(axis,kind) in enumerate(zip(pool.axes,pool.kinds))},
        probe_budget=dict(used=state['information_used'],maximum=3,window_last_query=8,dissent_used=state['dissent_used']))
    # All objective references in a public payload use the task's raw units.
    for h in public['knowledge']:h['reference_objective']=h.pop('reference_y')*pool.scale
    return public


def validate_proposal(raw,pool,state,t,available,ties):
    if (not isinstance(raw,dict) or set(raw)!={'summary','actions'} or type(raw['summary']) is not str
            or not 1<=len(raw['summary'])<=1000):raise ValueError('Expected summary and actions')
    return proposals.apply(pool,state,raw['actions'],t,available,ties)


def forecast_public(pool,state,hs,ids,t):
    public=previous.common_public(pool,state,ids,t)
    public['interface']='neutral_union_menu; no_GP_ranks_or_source_labels'
    public['knowledge']=knowledge(hs)
    for h in public['knowledge']:h['reference_objective']=h.pop('reference_y')*pool.scale
    return public


def validate_forecast(raw,hs,ids):
    if (not isinstance(raw,dict) or set(raw)!={'summary','assessments'} or type(raw['summary']) is not str
        or not 1<=len(raw['summary'])<=1000 or not isinstance(raw['assessments'],list)):
        raise ValueError('Invalid probability envelope')
    live={h['id'] for h in hs if h['kf']['status']!='retracted'};result={}
    for row in raw['assessments']:
        if not isinstance(row,dict) or set(row)!={'id','probabilities'} or row['id'] not in live or row['id'] in result:raise ValueError('Assess each live rule once')
        values=row['probabilities']
        if not isinstance(values,dict) or set(values)!=set(ids):raise ValueError('Assess exactly the displayed menu')
        if any(type(p) not in (int,float) or not isfinite(p) or not 0<=p<=1 for p in values.values()):raise ValueError('Probabilities must be finite0..1')
        result[row['id']]={key:float(p) for key,p in values.items()}
    if set(result)!=live:raise ValueError('Missing live hypothesis')
    return dict(probabilities=result)
