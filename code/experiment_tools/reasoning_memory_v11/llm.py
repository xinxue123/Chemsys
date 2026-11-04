"""Official confidence extraction and experiment-local, outcome-checked memory."""
import ast
from contextlib import redirect_stdout
from copy import deepcopy
from functools import lru_cache
import io,json,re
from math import isfinite
from pathlib import Path
from chem.benchmarks.knowledge_llm import digest
from knowledge_rank_v9 import llm as previous,policy

request=previous.request
MEMORY_SYSTEM='''Maintain experiment-local Notes for a Reasoning BO optimizer.
Use only supplied own measured outcomes, previous own Insights and public
chemistry/process semantics. Treat proposed trends as uncertain observational
hypotheses, not verified causal laws. Never recall benchmark yields or invent
molecular identities behind opaque labels. Candidate IDs have no ranking.
Return exactly {"summary":string,"actions":list}. At most3 actions,4 live notes.
Action keys exactly op,id,claim,focus,effect,scope,reference_id,reason.
op=add/retain/revise/retract. Use supplied next_available_ids for add.
For add/revise: focus contains1..2 actual axes; effect=promote/avoid; scope
contains optional {"bounds":[lo,hi]} in normalized0..1 for continuous axes or
{"allowed":[exact categorical labels]}; omitted axes mean full domain. At least2
current BO candidates must fall in scope. The reference_id MUST be the supplied
current incumbent ID; its already observed objective stays fixed for this version.
Describe a useful trend or interaction that could distinguish the current BO
choices. Retain/retract cannot rewrite semantics: claim:null,focus:[],effect:null,
scope:{},reference_id:null. Every action includes reason. A revision creates an
immutable version; applicable negative feedback survives. Paused notes are
cautions, not erased facts. retain cannot revive them. Abstaining is allowed.
These notes provide context to the official reasoning model, not a hard candidate
filter or a substitute acquisition function. All BO choices remain available.
summary<=1000,claim<=900,reason<=800 characters. Concise public JSON only.'''

REASONING_SYSTEM='''Execute the supplied OFFICIAL Reasoning BO optimization prompt.
The Bayesian optimizer has provided5 candidate experiments. Reason about their
tradeoffs using the complete own experimental history, previous official Insights,
and retrieved experiment-local evidence Notes. Produce the official Insight with
3..5 scientific hypotheses, high/medium/low confidence and suggested parameter_sets.
The ORIGINAL official stable confidence sort selects the FIRST candidate, n=1.
You decide the confidence levels and ordering. No numerical blending or external
gate overrides your choice, and every displayed BO candidate remains executable.

Return exactly {"insight":object,"assessments":list,"memory_references":list}.
insight has exactly comment,keywords,hypotheses. Each hypothesis has exactly
strategy,rationale,confidence,parameter_sets, matching the official schema.
Each parameter_sets entry is exactly {"candidate_id":"a displayed ID"}; IDs
resolve to supplied full conditions. Candidate order and IDs are not GP ranks.
Use confidence high/medium/low. Do not generate a sixth point or change a variable.

Notes' KF weight and lifecycle are advisory evidence about previous prospective
Brier skill versus GP, NOT proof of chemical truth and NOT mandatory candidate
confidence. Consider failed predictions and successful corrections; don't repeat
a refuted explanation as established fact. You may select any of the5 points
for a fresh, evidence-grounded hypothesis even if an older note was suspended.
Current GP numerical predictions are not supplied. No invented measurements, other-arm
information, recalled benchmark targets or identities behind opaque labels.

For EACH latest non-retracted note, including paused, assessments contains exactly
{"id":note_id,"probabilities":{ALL5 displayed candidate IDs:p}}. Each p is the
probability the next measured objective is STRICTLY GREATER than that note's
FIXED observed reference. These are prospective audit forecasts, not selection
scores; they never mechanically override the official Insight selection. Equal
probabilities are allowed. Only selected in-scope points update that note.
memory_references is a possibly empty list of {"id":note_id,"version":integer,
"use":"support" or "counterevidence"} describing notes actually considered
in the selected experiment's public rationale. Refer only to retrieved notes.
All statements must be concise public reasoning; no private reasoning trace.
Return valid JSON only.'''


@lru_cache(maxsize=1)
def compiled_extractor(root):
    path=Path(root)/'code/chemist_system/third_party/reasoning_bo_official/src/bo/reasoner/base.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BaseReasoner')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_extract_candidates_from_insight')
    namespace={'json':json,'re':re}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[method.name]


def official_extract(root,insight):
    with redirect_stdout(io.StringIO()):points=compiled_extractor(str(Path(root).resolve()))(None,json.dumps(insight),n=1)
    if len(points)!=1:raise ValueError('Official extraction must produce one point')
    return points[0]['candidate_id']


def note(pool,h):
    return dict(id=h['id'],version=h['version'],claim=h['claim'],focus=h['focus'],effect=h['effect'],scope=deepcopy(h['scope']),
        reference_id=h['reference_id'],reference_objective=h['reference_y']*pool.scale,
        lifecycle=h['kf']['status'],KF_weight=h['kf']['weight'],local_feedback=h['kf']['local_updates'],
        prior_only=h['kf']['local_updates']==0,reliability_mean=h['kf']['mean'],reliability_variance=h['kf']['variance'],
        inherited_negative_count=len(h['inherited_negative']),
        recent_feedback=[{k:e[k] for k in ('query','point_id','success','p_LLM','p_GP','Brier_gain','reason')} for e in h['events'][-4:]],
        retirement_reason=h.get('retirement_reason'))


def retrieve(pool,hs,ids):
    js=[pool.ix[key] for key in ids];support=[];caution=[]
    for h in hs:
        policy.validate_existing(h)
        if not len(policy.scope_indices(pool,h,js)):continue
        value=note(pool,h)
        if h['kf']['status']=='active' and h['kf']['weight']>0:support.append(value)
        else:caution.append(value)
    support.sort(key=lambda h:(-h['KF_weight'],h['id'],h['version']))
    caution.sort(key=lambda h:(h['id'],h['version']))
    return dict(supporting_notes=support,cautionary_notes=caution,
        interpretation='Predictive evidence only. Confidence and experiment selection remain with official Reasoning BO.')


def memory_public(pool,state,ids,t,refresh):
    value=previous.common_public(pool,state,ids,t)
    incumbent=min(state['history'],key=lambda r:(-r['y'],pool.features[pool.ix[r['candidate_id']]]))
    value.update(refresh=refresh,knowledge=[note(pool,h) for h in state['hypotheses']],
        next_available_ids=policy.next_ids(state['hypotheses']),incumbent_id=incumbent['candidate_id'],
        official_insight_history=deepcopy(state['insights']))
    return value


def validate_memory(raw,pool,state,ids,t):
    if (not isinstance(raw,dict) or set(raw)!={'summary','actions'} or type(raw['summary']) is not str
        or not 1<=len(raw['summary'])<=1000 or not isinstance(raw['actions'],list) or len(raw['actions'])>3):
        raise ValueError('Invalid evidence-note envelope')
    incumbent=min(state['history'],key=lambda r:(-r['y'],pool.features[pool.ix[r['candidate_id']]]))['candidate_id']
    # Invalid scientific actions are reported, not silently rewritten or treated
    # as a reason to change the current BO menu.
    hs=deepcopy(state['hypotheses']);reviews=[];touched=set()
    for n,a in enumerate(raw['actions']):
        if isinstance(a,dict) and a.get('id') in touched:
            reviews.append(dict(action=n,accepted=False,reason='Repeated note action'));continue
        if isinstance(a,dict) and a.get('op') in ('add','revise') and a.get('reference_id')!=incumbent:
            reviews.append(dict(action=n,accepted=False,id=a.get('id'),op=a.get('op'),reason='New version must use the known incumbent'));continue
        updated,rr=policy.apply(pool,hs,[a],state['history'],[pool.ix[key] for key in ids],t,True)
        review=rr[0];review['action']=n;reviews.append(review)
        if review['accepted']:hs=updated;touched.add(a['id'])
    return dict(hypotheses=hs,reviews=reviews)


def reasoning_public(root,pool,state,hs,ids,t):
    public=previous.common_public(pool,state,ids,t);memory=retrieve(pool,hs,ids)
    template=json.loads((Path(root)/'code/chemist_system/third_party/reasoning_bo_official/src/prompts/test_prompts.json').read_text(encoding='utf-8'))['optimization_loop']['en']
    public.update(retrieved_notes=memory,forecast_notes=[note(pool,h) for h in hs if h['kf']['status']!='retracted'])
    public['official_optimization_prompt']=template.format(iteration=len(state['history']),target=public['domain_context']['objective'],
        retrieved_context=json.dumps(memory,ensure_ascii=False),insight_history=json.dumps(state['insights'],ensure_ascii=False),
        trial_data=json.dumps(public['own_observations'],ensure_ascii=False),bo_recommendations=json.dumps(public['candidate_menu'],ensure_ascii=False),
        constraint='Choose exactly ONE displayed candidate; parameter_sets contain only candidate_id references.',
        parameter_definitions=json.dumps(public['axes'],ensure_ascii=False))
    public['adaptation']='Official prompt and confidence extractor n=1; sequential matched initialization; own evidence Notes only'
    return public


def validate_reasoning(root,raw,hs,ids,retrieved):
    if not isinstance(raw,dict) or set(raw)!={'insight','assessments','memory_references'}:raise ValueError('Expected official Insight plus audit sidecars')
    insight=raw['insight']
    if not isinstance(insight,dict) or set(insight)!={'comment','keywords','hypotheses'}:raise ValueError('Invalid official Insight')
    if any(type(insight[k]) is not str or not insight[k] for k in ('comment','keywords')):raise ValueError('Missing public Insight text')
    if not isinstance(insight['hypotheses'],list) or not 3<=len(insight['hypotheses'])<=5:raise ValueError('Official Insight requires3..5 hypotheses')
    for h in insight['hypotheses']:
        if not isinstance(h,dict) or set(h)!={'strategy','rationale','confidence','parameter_sets'}:raise ValueError('Invalid official hypothesis schema')
        if any(type(h[k]) is not str or not h[k] for k in ('strategy','rationale','confidence')) or h['confidence'].lower() not in ('high','medium','low'):raise ValueError('Invalid confidence')
        if not isinstance(h['parameter_sets'],list) or not h['parameter_sets']:raise ValueError('Missing candidate conditions')
        for point in h['parameter_sets']:
            if not isinstance(point,dict) or set(point)!={'candidate_id'} or point['candidate_id'] not in ids:raise ValueError('Only current5 BO candidates allowed')
    chosen=official_extract(root,insight)
    live={h['id'] for h in hs if h['kf']['status']!='retracted'};probabilities={}
    if not isinstance(raw['assessments'],list):raise ValueError('Assessments must be a list')
    for a in raw['assessments']:
        if not isinstance(a,dict) or set(a)!={'id','probabilities'} or a['id'] not in live or a['id'] in probabilities:raise ValueError('Assess each live note once')
        p=a['probabilities']
        if not isinstance(p,dict) or set(p)!=set(ids):raise ValueError('Forecast all current5 candidates')
        if any(type(v) not in (int,float) or not isfinite(v) or not 0<=v<=1 for v in p.values()):raise ValueError('Finite probabilities required')
        probabilities[a['id']]={key:float(v) for key,v in p.items()}
    if set(probabilities)!=live:raise ValueError('Missing note forecast')
    possible={(h['id'],h['version']) for group in ('supporting_notes','cautionary_notes') for h in retrieved[group]}
    # Annotation-only technical amendment: forecast_notes are also explicitly
    # supplied to the model. A citation to that visible evidence must not veto
    # an otherwise valid official Insight. The audit marks scope membership.
    possible.update((h['id'],h['version']) for h in hs if h['kf']['status']!='retracted')
    if not isinstance(raw['memory_references'],list):raise ValueError('Memory references must be a list')
    seen=set()
    for ref in raw['memory_references']:
        if (not isinstance(ref,dict) or set(ref)!={'id','version','use'} or type(ref['version']) is not int
            or (ref['id'],ref['version']) not in possible or ref['use'] not in ('support','counterevidence')
            or (ref['id'],ref['version']) in seen):raise ValueError('Cite supplied note versions once')
        seen.add((ref['id'],ref['version']))
    return dict(selected_id=chosen,probabilities=probabilities,memory_references=deepcopy(raw['memory_references']))
