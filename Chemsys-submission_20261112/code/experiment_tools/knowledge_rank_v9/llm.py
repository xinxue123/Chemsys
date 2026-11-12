"""Secret-safe exact-context Terra requests; official RBO selection AST."""
import ast
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import re
from filelock import FileLock
from chem.benchmarks.knowledge_llm import digest,write
from chem.benchmarks.knowledge_domain_context import task_context
from chem.benchmarks import knowledge_openai_client as api
from . import policy

KF_SYSTEM='''You guide a sequential chemistry/process experiment by ranking five
real BO candidates. Use supplied public domain information and ONLY your own
measured observations. Do not recall benchmark yields, invent measurements or
identities behind opaque labels, or treat IDs as ranks. No GP numerical values
are provided. Keep physical units distinct. Give concise public rationales.

Return exactly {"summary":string,"actions":list,"assessments":list}.
Knowledge edits are allowed ONLY when refresh.allowed is true. Otherwise actions
must be []. At most3 actions and4 live rules. An action has exactly
op,id,claim,focus,effect,scope,reference_id,reason. op=add/retain/revise/retract.
Use next_available_ids for add and existing IDs for other operations. For add or
revise: claim is a concise uncertain scientific relationship; focus contains1-2
actual axis names; effect is promote or avoid; reference_id is an observed point;
scope maps optional axes to {"bounds":[lo,hi]} for normalized continuous inputs
or {"allowed":[exact real categorical labels]}. Omitted axes mean full domain.
Cover at least2 current candidates. Scope is applicability, not proof of a
causal law. An identical structural rule requires explicit revision, not a new
ID. For retain/retract use claim:null,focus:[],effect:null,scope:{},reference_id:null.
Every action includes a short reason. Revisions create a new immutable version;
old applicable negative evidence survives. A paused rule cannot be revived by
retain or rewording; it needs new successful prospective feedback. You may add
no rules if chemistry or observations do not support a defensible statement.

After actions, assessments must contain exactly ONE entry for EACH latest
non-retracted rule (including paused rules). Entry has exactly id,probabilities.
probabilities maps ALL5 displayed candidate IDs to numerical probabilities in
[0,1] of their NEXT measured objective being STRICTLY GREATER than that rule's
FIXED observed reference objective. No yield magnitude prediction is required.
Equal probabilities are allowed. Even outside a rule scope provide a probability
to define its5-point ranking, but only scope-covered selected points can validate
that rule. New rule IDs must match their add actions. Do not assess retired rules.

KF measures each rule's prospective Brier-score improvement versus GP, updates
its reliability and controls ranking weight. Two consecutive failed predictions
can pause it; four can retire it. This is predictive utility, not causal proof.
Ranking is requested every round; only event rounds may rewrite knowledge.
Use latest feedback to avoid repeating refuted claims. summary<=1000 characters,
claim<=900,reason<=800. Output valid JSON only, no private reasoning trace.'''

RBO_SYSTEM='''Execute the supplied official Reasoning BO optimization prompt in
this5-to1 sequential interface. Use only own observations and exact displayed
candidates, never recalled benchmark outcomes or invented molecular identities.
Return exactly comment,keywords,hypotheses. Provide3..5 hypotheses; each has
strategy,rationale,confidence,parameter_sets. confidence=high/medium/low.
For this finite-pool I/O adapter each parameter_sets entry must be exactly
{"candidate_id":"one displayed ID"}; that ID resolves to its full physical
conditions. The official stable confidence sort chooses the FIRST point (n=1).
No sixth candidate, previous observation, ordinal ID interpretation, hidden
reasoning, yield lookup or other-arm information is permitted. Hypothesis
rationales are concise. Initial4 points are already measured and shared with BO.
No new warm-start experiment may be suggested.'''


def point(pool,key):
    j=pool.ix[key]
    return dict(candidate_id=key,conditions=deepcopy(pool.points[j]),
        normalized_continuous={a:float(pool.features[j][k]) for k,(a,kind) in enumerate(zip(pool.axes,pool.kinds)) if kind=='continuous'})


def common_public(pool,state,ids,t):
    domain=task_context(pool.task,pool);domain.pop('source_references',None)
    return dict(task=pool.task,query=t,remaining_queries=17-t,objective_unit=pool.unit,
        domain_context=domain,axes=[dict(name=a,kind=k) for a,k in zip(pool.axes,pool.kinds)],
        own_observations=[dict(point(pool,row['candidate_id']),objective=row['raw_value']) for row in state['history']],
        candidate_menu=[point(pool,key) for key in ids],
        interface='shared5candidate_sequential1; GP-derived membership, neutral physical display order')


def kf_public(pool,state,ids,t,refresh):
    public=common_public(pool,state,ids,t)
    public.update(refresh=refresh,next_available_ids=policy.next_ids(state['hypotheses']),
        knowledge=[dict(id=h['id'],version=h['version'],claim=h['claim'],focus=h['focus'],
            effect=h['effect'],scope=h['scope'],reference_id=h['reference_id'],
            reference_objective=h['reference_y']*pool.scale,kf=deepcopy(h['kf']),
            recent_feedback=[{key:e[key] for key in ('query','point_id','success','p_LLM','Brier_gain','reason')} for e in h['events'][-4:]],
            inherited_negative_count=len(h['inherited_negative'])) for h in state['hypotheses']])
    return public


def validate_kf(raw,pool,state,ids,t,refresh):
    if (not isinstance(raw,dict) or set(raw)!={'summary','actions','assessments'}
        or type(raw['summary']) is not str or not 1<=len(raw['summary'])<=1000):raise ValueError('Invalid KF-Rank JSON envelope')
    hs,reviews=policy.apply(pool,state['hypotheses'],raw['actions'],state['history'],[pool.ix[k] for k in ids],t,refresh['allowed'])
    live={h['id']:h for h in hs if h['kf']['status']!='retracted'}
    if not isinstance(raw['assessments'],list):raise ValueError('Assessments must be a list')
    out={}
    for row in raw['assessments']:
        if not isinstance(row,dict) or set(row)!={'id','probabilities'} or row['id'] not in live or row['id'] in out:
            raise ValueError('Assess exactly each latest live rule once; rejected actions do not create IDs')
        p=row['probabilities']
        if not isinstance(p,dict) or set(p)!=set(ids):raise ValueError('Assess exactly all5 current candidates')
        if any(type(v) not in (int,float) or not 0<=v<=1 for v in p.values()):raise ValueError('Probabilities must be finite real values in [0,1]')
        out[row['id']]={key:float(v) for key,v in p.items()}
    if set(out)!=set(live):raise ValueError('Missing live-rule assessment')
    return dict(hypotheses=hs,reviews=reviews,probabilities=out)


def official_extract(root,raw):
    path=Path(root)/'code/chemist_system/third_party/reasoning_bo_official/src/bo/reasoner/base.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BaseReasoner')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_extract_candidates_from_insight')
    namespace={'json':json,'re':re}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),namespace)
    with redirect_stdout(io.StringIO()):return namespace[method.name](None,json.dumps(raw),n=1)


def validate_rbo(root,raw,ids):
    if not isinstance(raw,dict) or set(raw)!={'comment','keywords','hypotheses'}:raise ValueError('Invalid official Insight envelope')
    if type(raw['comment']) is not str or not raw['comment'] or type(raw['keywords']) is not str:raise ValueError('Invalid official Insight text')
    if not isinstance(raw['hypotheses'],list) or not 3<=len(raw['hypotheses'])<=5:raise ValueError('Provide3..5 hypotheses')
    for h in raw['hypotheses']:
        if not isinstance(h,dict) or set(h)!={'strategy','rationale','confidence','parameter_sets'}:raise ValueError('Invalid official hypothesis')
        if any(type(h[k]) is not str or not h[k] for k in ('strategy','rationale','confidence')) or h['confidence'].lower() not in ('high','medium','low'):
            raise ValueError('Invalid confidence')
        if not isinstance(h['parameter_sets'],list) or not h['parameter_sets']:raise ValueError('Missing points')
        for p in h['parameter_sets']:
            if not isinstance(p,dict) or set(p)!={'candidate_id'} or p['candidate_id'] not in ids:raise ValueError('Candidate must be in the current5')
    points=official_extract(root,raw)
    if len(points)!=1:raise ValueError('Official extraction did not return1 point')
    return dict(selected_id=points[0]['candidate_id'])


def rbo_public(root,pool,state,ids,t):
    public=common_public(pool,state,ids,t)
    template=json.loads((Path(root)/'code/chemist_system/third_party/reasoning_bo_official/src/prompts/test_prompts.json').read_text(encoding='utf-8'))['optimization_loop']['en']
    prompt=template.format(iteration=len(state['history']),target=public['domain_context']['objective'],
        retrieved_context=None,insight_history=json.dumps(state['insights'],ensure_ascii=False),
        trial_data=json.dumps(public['own_observations'],ensure_ascii=False),
        bo_recommendations=json.dumps(public['candidate_menu'],ensure_ascii=False),
        constraint='Choose exactly ONE displayed candidate; parameter_sets contain only candidate_id references.',
        parameter_definitions=json.dumps(public['axes'],ensure_ascii=False))
    return dict(public,official_optimization_prompt=prompt,adaptation='5-to1; actual observation count; candidate_id transport; no warmstart or Notes')


def request(client,system,public,directory,validator):
    context=dict(system=system,public=public,model=api.MODEL,reasoning_effort=api.REASONING_EFFORT)
    key=digest(context);folder=Path(directory)/key;folder.mkdir(parents=True,exist_ok=True)
    with FileLock(str(folder/'.lock'),timeout=900):
        target=folder/'validated.json'
        if target.exists():
            saved=json.loads(target.read_text(encoding='utf-8'))
            if saved['context_sha256']!=key or saved['plan']!=validator(saved['raw']):raise ValueError('Cached response does not validate against exact context')
            return saved
        write(folder/'context.json',context);error=None
        for attempt in range(3):
            af=folder/f'attempt{attempt+1}';af.mkdir(exist_ok=True)
            response_path=af/'response.json'
            payload=public if error is None else dict(original=public,schema_repair_only=error)
            write(af/'request.json',dict(system=system,public=payload))
            if response_path.exists():response=json.loads(response_path.read_text(encoding='utf-8'))
            else:
                response=client.complete(system,payload,json_output=True,max_output_tokens=6000)
                write(response_path,response)
            try:
                raw=json.loads(response['output_text']);plan=validator(raw)
            except (ValueError,TypeError,KeyError) as exc:
                error=str(exc);write(af/'schema_error.json',dict(error=error));continue
            saved=dict(context_sha256=key,raw=raw,plan=plan,response_id=response['id'],usage=response['usage'],attempts=attempt+1)
            write(target,saved);return saved
        raise ValueError('LLM failed the bounded schema-only repair policy')
