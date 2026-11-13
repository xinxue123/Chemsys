"""Observed-only point proposals, corrected counters and native RBO ordering."""
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace,MethodType
import io
import json

from . import reasoning_bo_llm as native_rbo
from . import reaction_transfer_llm as nominal
from . import reasoning_bo_data as data
from .suzuki_domain import digest

ROOT=Path('data/runs/hte_pointwise_seed0')
RBO_SYSTEM='''You are a chemistry Reasoning BO agent in a sequential experiment.
Return the supplied official Insight JSON schema, with concise scientific rationales.
Use only exact known nominal options and your own observed data. Do not recall benchmark outcomes.
EVERY parameter_set must be one of the five supplied BO recommendations. The official stable
high/medium/low confidence sort selects ONE next point (5-to1 adaptation), never three.
There are4 common initial experiments and16 sequential queries. No other trajectory or future yields are available.
Category IDs are unordered identifiers. Do not add unlisted temperature/time or other conditions.
Shared initial preferences are uncertain observed-only guidance, not established chemical truth.
Do not provide a private reasoning trace. Output only the JSON object.'''
REGION_SYSTEM=nominal.REGION_GROUNDED.replace('Fixed substrates must remain fixed.',
    'The supplied axes are all varying categorical choices, including substrates when listed.')+'''
The candidate domain is the ENTIRE measured feature pool; there is no preliminary BO shortlist restriction.
This is4 initialization observations plus16 sequential queries. At most FOUR clauses may be active
across all supplied fixed_slots. Revoke one if adding a fifth. An empty knowledge set is allowed.
Scalar KF innovation_z is signed. b is a nuisance and is never deployed. Units for KF are yield fractions.
There is no batch3 decision or within-batch feedback. Query_round and observation_count are explicit.
'''
POINT_SYSTEM=nominal.POINT_SYSTEM.replace('Fixed substrates must remain fixed.',
    'Every supplied axis is a varying nominal choice; no substrate is fixed unless explicitly stated.')


def request(client,public,state,system,validator,phase):
    context={'public':public,'task':state['task'],'method':state['method'],
        'observed_prefix':deepcopy(state['history']),'phase':phase,'seed':0,
        'protocol_sha256':state['protocol_sha256']}
    context['context_sha256']=digest(context)
    folder=ROOT/'requests'/context['context_sha256'];data.write(folder/'context.json',context)
    answer=nominal.get(client,context,folder,system,validator)
    return answer,context['context_sha256']


def regional_call(client,pool,state,d=None,reasons=(),initial=False):
    observed=nominal.observations(pool,state,include_gp=False)
    gp=[]
    if d is not None:
        indices=sorted(__import__('numpy').flatnonzero(d['available']),
            key=lambda j:(-float(d['base'][j]),pool['ids'][j]))[:16]
        gp=[{'point':nominal.public_point(pool,pool['points'][j]),
            'GP_mean_percent':round(float(d['stats']['mu_y'][j])*100,3),
            'GP_latent_SD_pp':round(float(d['stats']['sd_y'][j])*100,3),
            'base_qLogEI':float(d['base'][j])} for j in indices]
    public={'task':'reaction_A' if pool['task_id']=='Perera_Suzuki' else 'reaction_B',
        'reaction_class':pool['reaction_class'],'objective_unit':pool['objective_unit'],
        'domain':nominal.domain(pool),'fixed_slots':{f'H{i+1}':a for i,a in enumerate(pool['axes'])},
        'current_knowledge':nominal.public_knowledge(pool,state['knowledge']),
        'observations':observed,'observation_count':len(observed),'query_round':len(state['decisions'])+1,
        'remaining_queries':16-len(state['decisions']),'reasons':list(reasons),'GP_summary':gp,
        'B4_diagnosis':d['diagnosis'].to_dict() if d is not None else None,
        'calibration_prior':d['prior'] if d is not None else None,
        'recent_measurements':[x.get('measurement') for x in state['decisions'][-3:]],
        'phase':'initial_knowledge' if initial else 'refresh'}
    def validator(raw):
        checked=nominal.validate_plan(raw,pool,state['knowledge'],observed,grounded=True)
        if len(checked['knowledge']['clauses'])>4:raise ValueError('at most4 active clauses')
        if checked['knowledge']['clauses'] and not any(all(p[h['axis']] in h['allowed']
                for h in checked['knowledge']['clauses']) for p in pool['points']):
            raise ValueError('preferred region has no measured feature-domain witness')
        return checked
    return request(client,public,state,REGION_SYSTEM,validator,public['phase'])


def point_call(client,pool,state,prefix,original_prompt,iteration):
    public={'task':'reaction_A' if pool['task_id']=='Perera_Suzuki' else 'reaction_B',
        'reaction_class':pool['reaction_class'],'domain':nominal.domain(pool),
        'objective_unit':pool['objective_unit'],'query_round':iteration,'observation_count':len(prefix),
        'remaining_queries':17-iteration,
        'observations':nominal.observations(pool,{'history':prefix},include_gp=False),
        'uncertain_preferences':nominal.public_knowledge(pool,state['initial_knowledge']),
        'original_LLINBO_prompt':original_prompt,
        'transport_note':'Numeric code bounds in the legacy prompt do not apply to nominal IDs; use only the named domain.'}
    return request(client,public,{**state,'history':prefix},POINT_SYSTEM,
        lambda raw:nominal.validate_point(raw,pool,prefix),'LLINBO_proposal')


def extract_one(raw):
    with redirect_stdout(io.StringIO()):
        return native_rbo.NATIVE['BaseReasoner']._extract_candidates_from_insight(None,json.dumps(raw),n=1)


def validate_rbo(raw,pool,history,shortlist_ids):
    if not isinstance(raw,dict) or set(raw)!={'comment','keywords','hypotheses'}:
        raise ValueError('expected Insight comment,keywords,hypotheses')
    if not isinstance(raw['comment'],str) or not raw['comment'].strip() or not isinstance(raw['keywords'],str):
        raise ValueError('invalid concise comment/keywords')
    hs=raw['hypotheses']
    if not isinstance(hs,list) or not 3<=len(hs)<=5:raise ValueError('3..5 hypotheses required')
    queried={h['candidate_id'] for h in history}
    for h in hs:
        if not isinstance(h,dict) or set(h)!={'strategy','rationale','confidence','parameter_sets'}:
            raise ValueError('invalid hypothesis schema')
        if any(not isinstance(h[k],str) or not h[k].strip() for k in ('strategy','rationale','confidence')):
            raise ValueError('nonempty hypothesis text required')
        if h['confidence'].lower() not in ('high','medium','low'):raise ValueError('invalid confidence')
        if not isinstance(h['parameter_sets'],list) or not 1<=len(h['parameter_sets'])<=5:
            raise ValueError('each hypothesis requires1..5 candidate points')
        for point in h['parameter_sets']:
            j=native_rbo.index_point(pool,point);key=pool['ids'][j]
            if key not in shortlist_ids or key in queried:raise ValueError('point outside unqueried BO top5')
    point=extract_one(raw)[0];key=pool['ids'][native_rbo.index_point(pool,point)]
    return {'selected_id':key,'insight':deepcopy(raw)}


def optimization_template():
    text=data.read(data.UPSTREAM/'src/prompts/test_prompts.json')['optimization_loop']['en']
    text=text.replace('at iteration {iteration}.','at query round {query_round}.')
    return text+'\nThis sequential adaptation selects exactly ONE of the five BO recommendations.\n'


class RBOClient:
    def __init__(self,client,pool):self.api=client;self.pool=pool;self.messages=[];self.last=None;self.state=None;self.top=None
    def generate(self,user_prompt,**unused):
        state=self.state
        public={'official_prompt_with_pointwise_count_fix':user_prompt,
            'query_round':len(state['decisions'])+1,'observation_count':len(state['history']),
            'remaining_queries':16-len(state['decisions']),
            'shared_initial_preferences':state['initial_knowledge']}
        answer,key=request(self.api,public,state,RBO_SYSTEM,
            lambda raw:validate_rbo(raw,self.pool,state['history'],self.top),'ReasoningBO_point')
        self.last=answer;self.context_sha256=key
        content=json.dumps(answer['raw'],ensure_ascii=False)
        self.messages=[{'role':'user','content':user_prompt},{'role':'assistant','content':content}]
        return content,''
    def save_messages(self,path):native_rbo.NATIVE['add_to_jsonl'](path,{'messages':self.messages,'retained_raw_reasoning':False})


def make_reasoner(client,pool,directory,protocol_sha256):
    r=native_rbo.make_reasoner(client,pool,0,directory,protocol_sha256)
    r.client=RBOClient(client,pool);r.chat_agent=native_rbo.SyntaxFormatter(r.client)
    r.prompt_manager.prompts=deepcopy(r.prompt_manager.prompts)
    r.prompt_manager.prompts['optimization_loop']['en']=optimization_template()
    def extract(self,insight,n=1):return extract_one(json.loads(insight))
    r._extract_candidates_from_insight=MethodType(extract,r)
    return r


def rbo_call(reasoner,pool,state,top_indices):
    t=len(state['decisions'])+1;ids=[pool['ids'][i] for i in top_indices]
    reasoner.client.state=state;reasoner.client.top=ids;reasoner.exp_config['query_round']=t
    experiment=SimpleNamespace(pool=pool,history=state['history'])
    # Actual completed observation count, not0-based batch index.
    trial=SimpleNamespace(index=len(state['history']))
    native_rbo.save_trial_prefix(experiment,trial,reasoner.trial_data_dir)
    history=[{'trial_index':0,'insight':json.dumps(state['initial_knowledge'],ensure_ascii=False)}]
    history += [{'trial_index':d['iteration'],'insight':json.dumps(d['insight'],ensure_ascii=False)} for d in state['decisions']]
    Path(reasoner.insight_history_file_path).write_text(''.join(json.dumps(x)+'\n' for x in history),encoding='utf8')
    bo=SimpleNamespace(gen=lambda n:SimpleNamespace(arms=[SimpleNamespace(parameters=pool['points'][i]) for i in top_indices]))
    with redirect_stdout(io.StringIO()):points=reasoner.optimization_loop(experiment,trial,bo,retrieval_context=None,n=5)
    if not isinstance(points,list) or len(points)!=1:raise RuntimeError('pointwise official reasoner failed')
    selected=pool['ids'][native_rbo.index_point(pool,points[0])]
    assert selected==reasoner.client.last['plan']['selected_id']
    return selected,reasoner.client.last['raw'],reasoner.client.context_sha256
