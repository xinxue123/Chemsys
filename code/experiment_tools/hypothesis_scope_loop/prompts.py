"""Grounded hypotheses, blinded paired first point, and complete own history."""
from copy import deepcopy
import math
import numpy as np
from chem.benchmarks import knowledge_gp as native
from grounded_first_start import proposal as first
from . import PARAMETERS as P
from .knowledge import Scopes

SYSTEM='''Propose a concrete preferred experiment AND falsifiable chemical/process hypotheses.
Use only the supplied scientific background and this trajectory's observed experiments.
Do not browse, recall benchmark outcomes, infer objectives from IDs, inspect other files,
or assume an unmeasured input has an observed objective. Give concise English scientific
explanations, not private reasoning. Interpret original physical units and exact molecular
identities. Background unknowns remain unknown (especially the HPLC response scale).

Return exactly {"summary":string,"preferred_experiment":{
"conditions":object,"basis_ids":list of hypothesis IDs,"rationale":string} or null,
"hypotheses":[{"id":short string,"claim":string,"direction":"favour" or "avoid",
"scope":{axis:{"allowed":[exact categorical values]} or {"bounds":[lo,hi]}},
"forecast":{"mean_raw":number,"sd_raw":positive number},"falsification":string,
"supporting_observation_ids":[observed IDs],"review_action":"new" or "keep" or "revise" or "retest"}]}.
At most 3 hypotheses. Each scope must contain at least 4 executable inputs, constrain at
least one real axis and exclude some pool inputs. All favour scopes are INTERSECTED;
all avoid scopes are SUBTRACTED. Choose compatible, useful, not needlessly narrow scopes;
their combined unmeasured region must contain at least 4 inputs. Scope bounds use original
units; categorical strings must match supplied identities exactly. Omitted axes are free.
forecast is a subjective absolute objective distribution for experiments INSIDE that scope,
including avoid scopes. Percentage means must lie in [0,100]; all means nonnegative.
Do not give a positive effect size as an absolute objective forecast. sd_raw must be >0.
Supply a concrete testable claim and operational falsification criterion. An unqueried
preferred condition must specify EVERY input axis and lie in the combined region, using
exact categorical identities. Examples are input-only, not an objective-ranked menu;
other feasible inputs may be proposed. The software maps a physical prototype to the
nearest unqueried executable input with identical categories, using INPUT DISTANCE only.

INITIAL: this paired experiment locks a previously independent, zero-data LLM first input.
Return that exact preferred input and at least one favour hypothesis justifying it. You
have NO measured outcomes or current GP scores. Your hypotheses are locked BEFORE the
first observation is accessed. Their new forecasts need not copy a previous estimate.

AFTER THE FIRST: the unchanged native GP maximizes native qLogEI INSIDE eligible scopes.
Your preferred prototype is a scientific proposal, NOT guaranteed to be executed; the GP
chooses the best executable point within your constraints. There is NO global qLogEI-gap
gate in this experiment. GP predictions and training targets are never edited. A favour
scope may guide only while its forecast probability of improving the current incumbent
is >=0.5 and its GP-relative KF credibility is >=0.2 (initial credibility 0.4). An avoid
scope may suppress only while its improvement probability is <=0.5 and credibility >=0.2.
Only predictions locked before an actual in-scope observation receive feedback. KF uses
relative Brier score of improvement against the GP; it is distinct from permission to
recommend. Two negative relative forecasts, two >2SD misses, low credibility, or two
nonimprovements genuinely caused by favour constraints pause guidance. These operational
failures do not establish that the chemical claim is universally false.

ON REFRESH: you receive ALL own experiments, locked predictions, KF updates, permissions,
and event reasons. Retain, revise or withdraw claims from this evidence and propose a new
preferred point. Return only hypotheses you want to keep; omissions withdraw old claims.
You may abstain with hypotheses=[] and preferred_experiment=null. Renaming/rewording the
same finite-input region and direction inherits its failure evidence, KF and retest count.
A paused rule never resumes merely because it predicts failure correctly. Only explicit
review_action='retest' during >=3-query stagnation permits ONE test of that semantic rule;
restoration then requires an actual incumbent improvement and positive GP-relative skill.
New regions are new provisional claims, not validated discoveries. Supporting IDs must
refer only to provided own observations. Current GP rankings/scores are deliberately
withheld so your chemical hypotheses and preferred proposal are independent of them.'''


def public(pool,state,event,locked_first):
    observations=[]
    for row in state['decisions']:
        d=row['decision']
        observations.append(dict(query=row['query'],candidate_id=row['candidate_id'],
            conditions=deepcopy(d['conditions']),objective_raw=row['raw_value'],unit=pool.unit,
            incumbent_before=d['incumbent_before'],incumbent_after=row['best'],
            selection_reason=d['reason'],applied_hypothesis_ids=d.get('active_hypothesis_ids',[]),
            locked_predictions=deepcopy(d.get('locked_forecasts',[])),
            hypothesis_feedback=deepcopy(row.get('knowledge_feedback',[]))))
    seen={r['candidate_id'] for r in state['history']}
    return dict(background=first.background(pool),objective_unit=pool.unit,
        experimental_budget=dict(initial=0,total_paid_queries=20,observed=len(observations),remaining=20-len(observations)),
        own_observations=observations,current_GP_predictions_or_ranking_supplied=False,
        incumbent_raw=max((r['raw_value'] for r in state['history']),default=None),
        input_only_examples=[v for v in first.examples(pool) if v['candidate_id'] not in seen],
        all_original_unqueried_pool_inputs_allowed=True,
        locked_initial_preferred_experiment=deepcopy(locked_first) if not observations else None,
        current_hypotheses=deepcopy(state['notes']),withdrawn_or_revised_history=deepcopy(state['archive']),
        refresh_event=deepcopy(event),automatic_permission_events=deepcopy(state.get('permission_events',[])),
        scope_combination='intersection of eligible favour regions minus union of eligible avoid regions',
        GP_values_are_not_modified=True,no_global_qLogEI_gap_gate=True)


def validate(raw,pool,state,locked_first):
    if not isinstance(raw,dict) or set(raw)!={'summary','preferred_experiment','hypotheses'} or not isinstance(raw['summary'],str):
        raise ValueError('Expected summary, preferred_experiment, hypotheses')
    drafts=raw['hypotheses'];scopes=Scopes(pool);seen=set();ids=set();history=state['history']
    if not isinstance(drafts,list) or len(drafts)>P['maximum_hypotheses']:raise ValueError('At most3 hypotheses')
    available=~np.isin(pool.ids,[r['candidate_id'] for r in history]);mask=available.copy()
    fields={'id','claim','direction','scope','forecast','falsification','supporting_observation_ids','review_action'}
    for note in drafts:
        if not isinstance(note,dict) or set(note)!=fields:raise ValueError('Invalid hypothesis fields')
        if any(not isinstance(note[k],str) or not 1<=len(note[k])<=1800 for k in ('id','claim','falsification')):raise ValueError('Missing hypothesis explanation')
        if note['id'] in ids or len(note['id'])>80:raise ValueError('Use unique short hypothesis IDs')
        ids.add(note['id'])
        if note['direction'] not in ('favour','avoid') or note['review_action'] not in ('new','keep','revise','retest'):raise ValueError('Invalid direction/review action')
        region=scopes.mask(note['scope']);key=scopes.semantic_key(note['direction'],note['scope'])
        if key in seen:raise ValueError('Duplicate finite-input semantic region')
        seen.add(key)
        if not P['minimum_scope_candidates']<=int(region.sum())<len(region):raise ValueError('Scope must include >=4 inputs and exclude some inputs')
        mask &= region if note['direction']=='favour' else ~region
        support=note['supporting_observation_ids']
        if not isinstance(support,list) or not set(support)<={r['candidate_id'] for r in history}:raise ValueError('Only own observed support IDs allowed')
        f=note['forecast']
        if not isinstance(f,dict) or set(f)!={'mean_raw','sd_raw'} or any(type(v) not in (int,float) or not math.isfinite(v) for v in f.values()):raise ValueError('Finite raw-unit forecast required')
        if f['mean_raw']<0 or f['sd_raw']<=0 or (pool.group in ('P0','P1') and f['mean_raw']>100):raise ValueError('Invalid raw-unit forecast')
    if drafts and mask.sum()<P['minimum_scope_candidates']:raise ValueError('Combined unqueried scope must contain >=4 inputs; widen or abstain')
    preferred=raw['preferred_experiment'];projection=None
    if preferred is not None:
        if not isinstance(preferred,dict) or set(preferred)!={'conditions','basis_ids','rationale'}:raise ValueError('Invalid preferred experiment fields')
        first.validate_conditions(pool,preferred['conditions'])
        if not isinstance(preferred['basis_ids'],list) or not set(preferred['basis_ids'])<=ids or not isinstance(preferred['rationale'],str):raise ValueError('Preferred basis IDs must refer to returned hypotheses')
        for note in drafts:
            inside=True
            for axis,spec in note['scope'].items():
                v=preferred['conditions'][axis]
                inside &= v in spec['allowed'] if 'allowed' in spec else spec['bounds'][0]-1e-10<=v<=spec['bounds'][1]+1e-10
            if inside!=(note['direction']=='favour'):raise ValueError('Preferred condition must satisfy all returned constraints')
        valid,distance=first.distances(pool,preferred['conditions']);js=np.flatnonzero(valid&mask)
        if not len(js):raise ValueError('No executable unqueried projection with identical categories inside scope')
        priorities=native.semantic_priorities(pool,len(history)+1,0)
        j=int(js[np.lexsort((priorities[js],distance[js]))[0]])
        projection=dict(candidate_id=pool.ids[j],conditions=deepcopy(pool.points[j]),normalized_input_RMS_distance=float(distance[j]),not_automatically_executed=bool(history))
    if not history:
        if preferred is None or not any(n['direction']=='favour' for n in drafts):raise ValueError('Initial request requires preferred input and favour hypothesis')
        if preferred['conditions']!=locked_first['conditions'] or projection['candidate_id']!=locked_first['candidate_id']:raise ValueError('Initial preferred input must match locked paired LLM first choice exactly')
    elif drafts and preferred is None:raise ValueError('Nonempty hypotheses require a concrete preferred experiment')
    return dict(summary=raw['summary'],drafts=deepcopy(drafts),preferred=deepcopy(preferred),projection=projection)
