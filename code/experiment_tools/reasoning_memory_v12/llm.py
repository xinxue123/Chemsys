"""Notes are evidence annotations, never an additional gate on official choices."""
from copy import deepcopy
import json
from math import isfinite
from pathlib import Path
from knowledge_rank_v9 import llm as previous
from reasoning_memory_v11.llm import compiled_extractor,official_extract
from . import policy

request=previous.request
MEMORY_SYSTEM='''Maintain experiment-local evidence Notes for Reasoning BO.
Use only own measured outcomes and public process semantics. No benchmark recall,
invented molecular identities, candidate-ID order or other-arm information.
Return exactly {"summary":string,"actions":list}; at most3 actions,4 live notes.
Each action has exactly op,id,claim,focus,effect,scope,reference_id,reason,evidence_queries.
op=add/retain/revise/retract; add uses supplied next_available_ids.
For add/revise: claim<=900chars, focus1..2 actual axes, effect=promote/avoid;
scope uses normalized continuous {"bounds":[lo,hi]} or categorical {"allowed":[exact
labels]}; omitted axes cover full domain. At least2 POOL inputs must be in scope;
there is NO requirement to cover today's BO menu. reference_id is current incumbent.
For retain/retract: claim,effect,reference_id=null; focus=[],scope={}.
reason<=800chars; evidence_queries lists relevant completed query numbers (may be
empty except retract). Retract requires at least one supplied adverse forecast
query for that note lineage. Missing current coverage means dormant, not disproven.
Do not repeatedly narrow notes to fit today's menu. Retain/abstain when there is
no new evidence. actions=[] is allowed, summary1..1000chars.
Version birth reference is immutable historical context. Future probabilities
are assessed against EACH ROUND'S current incumbent. KF measures predictive
calibration, not causal truth or optimization utility. A correct low-success
forecast is risk evidence, not validation that promote is beneficial. New notes
and revisions remain tentative; a single result cannot establish reliable support.
Historical adverse facts survive focus/ID edits, without replaying their numeric
Brier gains into a different forecasting target. Preserve factual cautions.
All five BO candidates remain executable, including novel out-of-scope hypotheses.
Provide concise public reasoning and valid JSON only.'''

REASONING_SYSTEM='''Execute the supplied OFFICIAL Reasoning BO optimization prompt.
Use complete own results, previous Insights and relevant evidence Notes to choose
ONE of5 supplied BO experiments. Return an official Insight with3..5 hypotheses.
Its ORIGINAL stable high/medium/low confidence sort selects the first point n=1.
All candidates are executable: there is no extra knowledge gate, rank blend or
GP modification. IDs and display order are not ranks. No sixth point or edits.
Use public chemistry/process reasoning, not benchmark yield recall, invented
identities behind opaque labels or other-arm data. Current GP numbers are hidden.

Return {"insight":object,"assessments":list,"memory_references":list,"decision_purpose":object}.
insight has exactly comment,keywords,hypotheses. Hypotheses have exactly strategy,
rationale,confidence,parameter_sets; each parameter_sets entry is {"candidate_id":ID}.
Use high/medium/low confidence. Public concise rationales only, no private trace.

Memory is advisory. Distinguish calibration_weight from directional support:
correctly anticipating failure does not establish an improvement direction.
support requires positive_support_eligible=true AND the selected point in scope.
Paused/retracted or untested notes cannot supply established positive evidence.
They may suggest exploratory_transfer; counterevidence means an actual caution.
memory_references contains {"id":ID,"version":integer,"use":"support"|
"counterevidence"|"exploratory_transfer"} for supplied relevant note versions.
History-based fresh hypotheses can use no note. Out-of-scope exploration remains
allowed; explain what uncertainty it tests instead of claiming validated support.

AFTER identifying the official selected point, assess ONLY supplied non-retracted
notes whose scope contains THAT point (paused notes can be assessed in shadow).
Each assessment is {"id":ID,"version":integer,"p_improve_incumbent":number0..1}:
probability that the selected measured objective STRICTLY EXCEEDS the supplied
CURRENT incumbent, NOT a note's birth reference. [] when no notes apply.
These are prospective audit forecasts, never scores that override official choice.
decision_purpose={"purpose":"improve"|"learn","uncertainty":string,
"if_improved":string,"if_not":string}. State intended information and how outcomes
will affect the next decision. Learning is not automatically failure when below
incumbent. No forced exploitation, scope filter or numerical opportunity gate.
Citation/prediction annotations are audited separately; keep the official choice
and output valid JSON.'''


def note(pool,h,ids):
    return dict(id=h['id'],version=h['version'],claim=h['claim'],focus=h['focus'],effect=h['effect'],scope=deepcopy(h['scope']),
        birth_reference_id=h['reference_id'],birth_reference_objective=h['reference_y']*pool.scale,
        lifecycle=h['kf']['status'],calibration_weight=h['kf']['weight'],local_feedback=h['kf']['local_updates'],
        positive_improvement_evidence=policy.positive_count(h),positive_support_eligible=policy.support_eligible(h),
        coverage='relevant' if len(policy.scope_indices(pool,h,[pool.ix[key] for key in ids])) else 'dormant',
        reliability_mean=h['kf']['mean'],reliability_variance=h['kf']['variance'],carried_fact_ids=h['carried_fact_ids'],
        recent_feedback=[{k:e[k] for k in ('query','point_id','reference_id','success','p_LLM','p_GP','Brier_gain','reason')} for e in h['events'][-3:]],
        retirement_reason=h.get('retirement_reason'))


def retrieve(pool,hs,ids):
    found=[]
    for h in hs:
        policy.validate_existing(h)
        if len(policy.scope_indices(pool,h,[pool.ix[key] for key in ids])):found.append(note(pool,h,ids))
    found.sort(key=lambda h:(h['id'],h['version']))
    relevant={(h['id'],h['version']) for h in found}
    carried={key for h in found for key in h['carried_fact_ids']}
    facts=[f for f in policy.negative_facts(hs) if (f['id'],f['version']) in relevant or f['event_id'] in carried][-6:]
    facts=[dict(f,conditions=previous.point(pool,f['point_id'])['conditions'],
        interpretation='Historical measured fact, not a numeric prior for this round') for f in facts]
    return dict(notes=found,adverse_facts=facts,interpretation='Relevant evidence only; all BO candidates remain executable')


def memory_public(pool,state,ids,t,refresh):
    public=previous.common_public(pool,state,ids,t);ref=policy.incumbent(pool,state['history'])
    public.update(refresh=refresh,knowledge=[note(pool,h,ids) for h in state['hypotheses']],
        adverse_facts=policy.negative_facts(state['hypotheses'])[-8:],next_available_ids=policy.next_ids(state['hypotheses']),
        incumbent_id=ref['candidate_id'],incumbent_objective=ref['raw_value'],official_insight_history=deepcopy(state['insights']))
    return public


def validate_memory(raw,pool,state,ids,t):
    if (not isinstance(raw,dict) or set(raw)!={'summary','actions'} or type(raw['summary']) is not str
        or not 1<=len(raw['summary'])<=1000):raise ValueError('Invalid memory envelope')
    hs,reviews=policy.apply(pool,state['hypotheses'],raw['actions'],state['history'],t)
    return dict(hypotheses=hs,reviews=reviews)


def reasoning_public(root,pool,state,hs,ids,t):
    public=previous.common_public(pool,state,ids,t);memory=retrieve(pool,hs,ids);ref=policy.incumbent(pool,state['history'])
    public.update(retrieved_notes=memory,current_incumbent=dict(candidate_id=ref['candidate_id'],objective=ref['raw_value']),
        forecast_target='Selected objective strictly greater than current_incumbent.objective')
    template=json.loads((Path(root)/'code/chemist_system/third_party/reasoning_bo_official/src/prompts/test_prompts.json').read_text(encoding='utf-8'))['optimization_loop']['en']
    public['official_optimization_prompt']=template.format(iteration=len(state['history']),target=public['domain_context']['objective'],
        retrieved_context=json.dumps(memory,ensure_ascii=False),insight_history=json.dumps(state['insights'],ensure_ascii=False),
        trial_data=json.dumps(public['own_observations'],ensure_ascii=False),bo_recommendations=json.dumps(public['candidate_menu'],ensure_ascii=False),
        constraint='Choose exactly ONE displayed candidate; parameter_sets contain only candidate_id references.',
        parameter_definitions=json.dumps(public['axes'],ensure_ascii=False))
    return public


def validate_insight(root,insight,ids):
    if not isinstance(insight,dict) or set(insight)!={'comment','keywords','hypotheses'}:raise ValueError('Invalid official Insight')
    if any(type(insight[k]) is not str or not insight[k] for k in ('comment','keywords')):raise ValueError('Missing public Insight text')
    if not isinstance(insight['hypotheses'],list) or not 3<=len(insight['hypotheses'])<=5:raise ValueError('Need3..5 official hypotheses')
    for h in insight['hypotheses']:
        if not isinstance(h,dict) or set(h)!={'strategy','rationale','confidence','parameter_sets'}:raise ValueError('Invalid official hypothesis')
        if any(type(h[k]) is not str or not h[k] for k in ('strategy','rationale','confidence')) or h['confidence'].lower() not in ('high','medium','low'):raise ValueError('Invalid confidence')
        if not isinstance(h['parameter_sets'],list) or not h['parameter_sets']:raise ValueError('Missing conditions')
        for p in h['parameter_sets']:
            if not isinstance(p,dict) or set(p)!={'candidate_id'} or p['candidate_id'] not in ids:raise ValueError('Only five supplied points allowed')
    return official_extract(root,insight)


def validate_reasoning(root,raw,pool,hs,ids,retrieved):
    if not isinstance(raw,dict) or 'insight' not in raw:raise ValueError('Missing official Insight')
    chosen=validate_insight(root,raw['insight'],ids);j=pool.ix[chosen];warnings=[];forecasts={};refs=[]
    visible={(h['id'],h['version']) for h in retrieved['notes']}
    notes={(h['id'],h['version']):h for h in hs}
    expected={key for key,h in notes.items() if h['kf']['status']!='retracted' and len(policy.scope_indices(pool,h,[j]))}
    assessments=raw.get('assessments',[])
    if not isinstance(assessments,list):warnings.append('assessments_not_list');assessments=[]
    seen=set()
    for a in assessments:
        if not isinstance(a,dict) or type(a.get('id')) is not str or type(a.get('version')) is not int:
            warnings.append('invalid_assessment_identity');continue
        key=(a['id'],a['version']);p=a.get('p_improve_incumbent')
        if key not in expected or key not in visible or key in seen or type(p) not in (int,float) or not isfinite(p) or not 0<=p<=1:
            warnings.append('inapplicable_duplicate_or_invalid_forecast:'+str(key));continue
        seen.add(key);forecasts[a['id']]=dict(version=a['version'],p=float(p))
    for key in sorted(expected-seen):warnings.append('missing_scoped_forecast:'+str(key))
    raw_refs=raw.get('memory_references',[])
    if not isinstance(raw_refs,list):warnings.append('references_not_list');raw_refs=[]
    cited=set()
    for ref in raw_refs:
        if (not isinstance(ref,dict) or type(ref.get('id')) is not str or type(ref.get('version')) is not int
            or ref.get('use') not in ('support','counterevidence','exploratory_transfer')):
            warnings.append('malformed_citation');continue
        key=(ref['id'],ref['version']);h=notes.get(key);effective=ref['use'];reason='valid_annotation'
        if key not in visible or key in cited:effective='unsupported';reason='not_retrieved_or_duplicate'
        elif effective=='support' and (not policy.support_eligible(h) or not len(policy.scope_indices(pool,h,[j]))):
            effective='exploratory_transfer';reason='no_mature_active_in_scope_support'
        refs.append(dict(id=ref['id'],version=ref['version'],requested_use=ref['use'],use=effective,reason=reason))
        cited.add(key)
    purpose=raw.get('decision_purpose')
    if (not isinstance(purpose,dict) or purpose.get('purpose') not in ('improve','learn')
        or any(type(purpose.get(k)) is not str or not purpose[k] for k in ('uncertainty','if_improved','if_not'))):
        warnings.append('missing_or_invalid_decision_purpose');purpose=None
    # A bad sidecar never sends an otherwise valid choice through a fresh LLM
    # call. Preserve raw annotations; no invented probability or changed point.
    return dict(selected_id=chosen,forecasts=forecasts,memory_references=refs,decision_purpose=deepcopy(purpose),
        warnings=warnings,expected_forecasts=len(expected),valid_forecasts=len(forecasts))
