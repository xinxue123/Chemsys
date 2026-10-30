"""Reviewed official Reasoner methods, isolated from telemetry and services.

The original confidence sorting and prompt formatting execute unchanged. Only
I/O, a syntax-only formatter and the configured provider are injected. The
paper's P_t subset C_t requirement is explicitly checked at the boundary.
"""
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Dict
import ast
import csv
import glob
import io
import json
import os
import re

from . import reasoning_bo_data as data
from . import reaction_transfer_llm as nominal
from .suzuki_domain import digest

ROOT = Path('data/runs/reasoning_bo_comparison')
SYSTEM = '''You are the Reasoning BO chemistry reasoner. Execute the supplied official prompt.
Return only valid JSON. Give concise scientific hypotheses/rationales, not a private reasoning trace.
Use only the exact supplied nominal categories and observed data. Never recall benchmark outcomes.
For overview return {"overview":"<the requested overview>"}; otherwise use the prompt's Insight schema.
For iterative optimization EVERY parameter_set MUST be one of the five BO recommendations.
Do not invent a sixth point or resubmit an observed experiment. The confidence-sorted first three
points must be distinct. This implements the paper's P_t subset C_t selection rule.
During initialization choose three distinct executable nominal combinations.
Category integer codes have no ordinal meaning. No unlisted temperature, time, or condition may be added.
The metadata field replicate_seed labels an independent repeat; it is not experimental evidence.'''


def _load_reviewed():
    sources = {}
    namespace = dict(json=json, re=re, os=os, glob=glob, Dict=Dict,
        Experiment=object, Trial=object, BOModel=object, __file__=str(data.UPSTREAM/'src/prompts/base.py'))
    for relative, names in (
        ('src/utils/jsonl.py', {'add_to_jsonl','concatenate_jsonl'}),
        ('src/prompts/base.py', {'PromptManager'}),
        ('src/bo/reasoner/base.py', {'BaseReasoner'})):
        path=data.UPSTREAM/relative
        tree=ast.parse(path.read_text(encoding='utf8'))
        nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
        assert {n.name for n in nodes} == names
        # No upstream imports or module-level Config/client construction executes.
        code=ast.Module(body=nodes,type_ignores=[])
        exec(compile(code,str(path),'exec'),namespace)
        sources[relative]={'sha256':data.filehash(path),'AST_sha256':digest(ast.dump(code,include_attributes=False))}
    return namespace,sources


NATIVE, SOURCE_RECEIPT = _load_reviewed()


def extract(insight):
    with redirect_stdout(io.StringIO()):
        return NATIVE['BaseReasoner']._extract_candidates_from_insight(None,json.dumps(insight),n=3)


def index_point(pool,point):
    if not isinstance(point,dict) or set(point)!=set(pool['axes']):
        raise ValueError('all and only exact nominal parameter names are required')
    if any(not isinstance(point[a],str) or point[a] not in pool['levels'][a] for a in pool['axes']):
        raise ValueError('unknown nominal option; use exact source spelling')
    key=tuple(point[a] for a in pool['axes'])
    index={tuple(p[a] for a in pool['axes']):i for i,p in enumerate(pool['points'])}
    if key not in index:
        raise ValueError('combination is absent from measured executable domain')
    return index[key]


def validate_insight(raw,pool,history,shortlist=None):
    if not isinstance(raw,dict) or set(raw)!={'comment','keywords','hypotheses'}:
        raise ValueError('Insight requires comment,keywords,hypotheses')
    if not isinstance(raw['comment'],str) or not raw['comment'].strip():
        raise ValueError('concise comment required')
    if not isinstance(raw['keywords'],str): raise ValueError('keywords must be a string')
    hs=raw['hypotheses']
    if not isinstance(hs,list) or not 3<=len(hs)<=5:
        raise ValueError('provide3..5 hypotheses as requested')
    queried={h['candidate_id'] for h in history}
    allowed=set(shortlist) if shortlist is not None else set(pool['ids'])
    for h in hs:
        if not isinstance(h,dict) or set(h)!={'strategy','rationale','confidence','parameter_sets'}:
            raise ValueError('invalid hypothesis schema')
        if not all(isinstance(h[k],str) and h[k].strip() for k in ('strategy','rationale','confidence')):
            raise ValueError('strategy,rationale,confidence must be nonempty strings')
        if h['confidence'].lower() not in ('high','medium','low'):
            raise ValueError('confidence is high/medium/low')
        if not isinstance(h['parameter_sets'],list) or not 1<=len(h['parameter_sets'])<=5:
            raise ValueError('each hypothesis requires1..5 parameter sets')
        for point in h['parameter_sets']:
            i=index_point(pool,point);key=pool['ids'][i]
            if key not in allowed: raise ValueError('iterative points must belong to the five BO recommendations')
            if key in queried: raise ValueError('already observed experiment is forbidden')
    points=extract(raw)
    ids=[pool['ids'][index_point(pool,p)] for p in points]
    if len(ids)!=3 or len(set(ids))!=3:
        raise ValueError('the official confidence-sorted first3 points must be distinct')
    return {'insight':deepcopy(raw),'selected_ids':ids}


class SyntaxFormatter:
    def __init__(self,client): self.client=client
    def step(self,prompt):
        # The provider already produced checked JSON. No confidence/hypothesis
        # rewriting by a second model is performed.
        raw=self.client.last['raw']
        return SimpleNamespace(msg=SimpleNamespace(content=json.dumps(raw,ensure_ascii=False)))


class NativeClient:
    def __init__(self,api_client,pool,seed,protocol_sha256):
        self.api_client,self.pool,self.seed,self.protocol_sha256=api_client,pool,seed,protocol_sha256
        self.messages=[];self.last=None;self.mode='overview';self.history=[];self.shortlist=None
        self.records=[]
    def generate(self,user_prompt,**unused):
        public={'official_prompt':user_prompt,'replicate_seed':self.seed,'phase':self.mode}
        context={'public':public,'task':self.pool['task_id'],'mode':self.mode,
            'observed_prefix':deepcopy(self.history),'shortlist_ids':self.shortlist,
            'protocol_sha256':self.protocol_sha256}
        context['context_sha256']=digest(context)
        folder=ROOT/'requests'/context['context_sha256']
        data.write(folder/'context.json',context)
        def validator(raw):
            if self.mode=='overview':
                if not isinstance(raw,dict) or set(raw)!={'overview'} or not isinstance(raw['overview'],str) or not raw['overview'].strip():
                    raise ValueError('overview JSON object required')
                return raw
            return validate_insight(raw,self.pool,self.history,self.shortlist)
        answer=nominal.get(self.api_client,context,folder,SYSTEM,validator)
        self.last=answer
        if self.mode=='overview':content=answer['raw']['overview']
        else:content=json.dumps(answer['raw'],ensure_ascii=False)
        self.messages=[{'role':'user','content':user_prompt},{'role':'assistant','content':content}]
        self.records.append({'mode':self.mode,'context_sha256':context['context_sha256'],
            'response_id':answer['response_id'],'selected_ids':answer['plan'].get('selected_ids')})
        return content,''
    def save_messages(self,path):
        NATIVE['add_to_jsonl'](path,{'messages':self.messages,'retained_raw_reasoning':False})


def save_trial_prefix(experiment,trial,save_dir):
    path=Path(save_dir);path.mkdir(parents=True,exist_ok=True)
    # Replace with the actual completed prefix. The native append-only helper
    # duplicates initialization, and lags the immediately preceding batch.
    # Correcting that I/O gives all methods the same truthful observed prefix.
    with (path/'Yield.csv').open('w',encoding='utf8',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['observed_index',*experiment.pool['axes'],'yield_percent'])
        lookup={key:i for i,key in enumerate(experiment.pool['ids'])}
        for i,h in enumerate(experiment.history,1):
            p=experiment.pool['points'][lookup[h['candidate_id']]]
            writer.writerow([i,*[p[a] for a in experiment.pool['axes']],round(h['y']*100,3)])


def make_reasoner(api_client,pool,seed,directory,protocol_sha256):
    cls=NATIVE['BaseReasoner'];r=cls.__new__(cls)
    r.exp_config=data.read(data.ROOT/pool['task_id']/'compass.json')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    r.result_dir=str(directory)+os.sep
    r.trial_data_dir=str(directory/'trial_data')+os.sep
    r.messages_file_path=str(directory/'messages.jsonl')
    r.insight_history_file_path=str(directory/'insight_history.jsonl')
    r.experiment_analysis_file_path=str(directory/'experiment_analysis.jsonl')
    manager=NATIVE['PromptManager'].__new__(NATIVE['PromptManager'])
    manager.prompts=data.read(data.UPSTREAM/'src/prompts/test_prompts.json')
    r.prompt_manager=manager;r.overview='';r.keywords='';r.experiment_analysis={};r.summary='';r.report=''
    r.client=NativeClient(api_client,pool,seed,protocol_sha256)
    r.chat_agent=SyntaxFormatter(r.client)
    NATIVE['save_trial_data']=save_trial_prefix
    # Functions' globals reference NATIVE, so the reviewed body uses this I/O.
    Path(r.insight_history_file_path).touch(exist_ok=True)
    return r


def initial_reasoning(reasoner):
    reasoner.client.mode='overview'
    with redirect_stdout(io.StringIO()):
        overview=reasoner.generate_overview()
    if not overview: raise RuntimeError('official overview failed')
    reasoner.client.mode='initial_sampling'
    with redirect_stdout(io.StringIO()):
        insight=reasoner.initial_sampling()
        points=reasoner.optimization_first_round(insight)
    if not points: raise RuntimeError('official initialization failed')
    return {'overview':overview,'insight':json.loads(insight),
        'selected_ids':[reasoner.client.pool['ids'][index_point(reasoner.client.pool,p)] for p in points],
        'requests':deepcopy(reasoner.client.records)}


def iteration_reasoning(reasoner,state,shortlist,iteration):
    pool=reasoner.client.pool
    # Rebuild the real visible history deterministically on resume. This also
    # avoids the native one-batch-late append-only CSV bug.
    experiment=SimpleNamespace(pool=pool,history=state['history'])
    trial=SimpleNamespace(index=iteration-1)
    save_trial_prefix(experiment,trial,reasoner.trial_data_dir)
    rows=[{'trial_index':0,'insight':json.dumps(state['initial_insight'],ensure_ascii=False)}]
    rows += [{'trial_index':d['iteration'],'insight':json.dumps(d['insight'],ensure_ascii=False)} for d in state['decisions']]
    Path(reasoner.insight_history_file_path).write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf8')
    reasoner.client.mode=f'optimization_{iteration:02d}'
    reasoner.client.history=deepcopy(state['history']);reasoner.client.shortlist=list(shortlist['ids'])
    indices={key:i for i,key in enumerate(pool['ids'])}
    bo=SimpleNamespace(gen=lambda n:SimpleNamespace(arms=[SimpleNamespace(parameters=pool['points'][indices[key]]) for key in shortlist['ids']]))
    with redirect_stdout(io.StringIO()):
        points=reasoner.optimization_loop(experiment,trial,bo,retrieval_context=None,n=5)
    if not points: raise RuntimeError('official reasoner returned no batch')
    selected=[pool['ids'][index_point(pool,p)] for p in points]
    assert selected == reasoner.client.last['plan']['selected_ids']
    return selected,reasoner.client.last['raw'],reasoner.client.records[-1]


REGION_SYSTEM = nominal.REGION_GROUNDED.replace('Fixed substrates must remain fixed.',
    'Substrates are varying categorical optimization variables in this full-table task.') + '''
This experiment uses simultaneous batches of3 chosen from a shared BO shortlist of5.
No within-batch outcome is available until all3 choices have been committed.
Maintain at most FOUR active clauses across the supplied fixed_slots (Suzuki has5 possible slots).
If adding a fifth, revoke another in the same response. An empty knowledge set is allowed.
joint_innovation_z is the RMS Mahalanobis batch innovation, not a single-point signed z.
The budget constrains JOINT batch qLogNEI opportunity loss. B4 requires a common knowledge witness
in an eligible batch. A conflict forces the base best batch. Do not infer chemical falsity from this.
'''


def regional_request(pool,state,decision,reasons,initial=False):
    lookup={key:i for i,key in enumerate(pool['ids'])}
    observed=[{'index':i,'point':nominal.public_point(pool,pool['points'][lookup[h['candidate_id']]]),
        'yield_percent':round(h['y']*100,3)} for i,h in enumerate(state['history'],1)]
    public={'task':pool['task_id'],'reaction_class':pool['reaction_class'],'domain':nominal.domain(pool),
        'fixed_slots':{f'H{i+1}':a for i,a in enumerate(pool['axes'])},'objective_unit':pool['objective_unit'],
        'phase':'initial_knowledge' if initial else 'event_refresh','reasons':reasons,
        'iteration':len(state['decisions'])+1,'remaining_batches':30-len(state['decisions']),
        'observations':observed,'current_knowledge':nominal.public_knowledge(pool,state['knowledge']),
        'BO_shortlist':[nominal.public_point(pool,pool['points'][j]) for j in decision['shortlist']['indices']],
        'batch_diagnosis':decision['diagnosis'].to_dict(),'calibration_prior':decision['prior'],
        'deployed_amplitude':decision['amplitude'],'conflict_fallback':decision['fallback'],
        'recent_batches':[{k:v for k,v in d.items() if k in ('iteration','selected_ids','incumbent_gain','measurement')}
            for d in state['decisions'][-3:]]}
    context={'public':public,'seed':state['seed'],'protocol_sha256':state['protocol_sha256']}
    context['context_sha256']=digest(context)
    def validator(raw):
        value=nominal.validate_plan(raw,pool,state['knowledge'],observed,grounded=True)
        if len(value['knowledge']['clauses'])>4: raise ValueError('at most4 active clauses')
        # Do not accept an entirely non-executable preference before diagnostics.
        if value['knowledge']['clauses']:
            if not any(all(p[h['axis']] in h['allowed'] for h in value['knowledge']['clauses']) for p in pool['points']):
                raise ValueError('preferred intersection has no measured feature-domain witness')
        return value
    return context,validator


def regional_call(client,pool,state,decision,reasons,initial=False):
    context,validator=regional_request(pool,state,decision,reasons,initial)
    folder=ROOT/'requests'/context['context_sha256']
    data.write(folder/'context.json',context)
    answer=nominal.get(client,context,folder,REGION_SYSTEM,validator)
    return answer,context['context_sha256']
