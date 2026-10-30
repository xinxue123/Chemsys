"""Observed-prefix-only DeepSeek calls, grounded regions and native RBO ordering."""
from copy import deepcopy
from pathlib import Path
import json
import math
import numpy as np
from . import continuous_benchmark_data as data
from . import continuous_benchmark_engine as engine
from . import reaction_transfer_llm as transport
from . import reasoning_bo_llm as rbo_native
from .suzuki_domain import digest

ROOT=Path('data/runs/continuous_20261029')
COMMON='''You assist one sequential chemical/process experiment optimization task.
Use ONLY the supplied domain, your OWN observed experiments and explicitly labelled GP predictions.
Never identify a benchmark, recall published outcomes, use other trajectories or invent measurements.
All named input axes are continuous physical quantities. Internal IDs and row order are not rankings.
Optimize the supplied single objective and units; STY and HPLC peak area are not reaction yield percent.
There are four common initial experiments plus sixteen sequential experimental queries, with feedback.
Shared initial preferences are uncertain hypotheses. A concise rationale is sufficient; no private reasoning trace.
No extra conditions or variables may be added. Output only the required JSON object.'''
REGION_SYSTEM=COMMON+'''
Update an uncertain preferred region, an intersection of at most FOUR normalized intervals.
fixed_slots maps H1,H2,... to physical input axes. Each axis may have only one clause.
Confidence is fixed at0.7. Every interval is in normalized [0,1], not physical units.
Return EXACT JSON {"actions":[{"id":"H1","operation":"add","bounds":[0.2,0.8],
"reason":"short testable preference","support":[{"index":1,"axis":"axis_name","value":0.0,"objective":0.0}]}],"summary":"..."}.
Operations: add for vacant slot, revise for active slot, retain or revoke for active slot.
retain/revoke use bounds=[]; every active slot MUST receive one action. Vacant slots may be omitted.
add/revise must have 0<=lower<upper<=1 and width<1; a revise must actually change the interval.
Support cites1..3 supplied OWN observation indices with the exact PHYSICAL axis value and displayed objective.
Do not cite a GP prediction as an observation. Empty knowledge is allowed; joint region must contain a candidate.
The GP supplies mean and uncertainty; the official LGBO covariance field modifies only its mean.
KF [b,a] calibrates residual y-mu_GP against [1,raw regional field]. b is a nuisance, not deployed.
a is clipped to[0,1]. B4 detects acquisition-budget incompatibility, not chemical falsity.
Conflict blocks deployment but shadow KF still learns. Same geometric region cannot regain trust by renaming:
it needs a new experiment that improves its FROZEN prequery guided squared error over GP, and B4 clears.
All normalized objective fields equal raw objective/objective_scale. Keep physical and normalized units distinct.
reason<=1000 characters; summary<=1600. Prefer parsimonious claims, retain/revoke when uncertainty is high.'''
POINT_SYSTEM=COMMON+'''
Propose ONE next point before BO review; you have no hidden objective or GP table.
The original LLINBO prompt is supplied. Return EXACT JSON {"point":{"axis_name":0.0,...},
"reason":"short rationale","support":[1,...]}. point must contain EVERY named PHYSICAL axis in domain.
Your proposal is projected to the nearest unqueried fixed-pool member in normalized Euclidean coordinates.
Independent randomized priority resolves exactly equal projection distances. Then the unchanged official
Justify branch reviews the shared qLogEI loss budget. support must cite your own observed indices.'''
RBO_SYSTEM=COMMON+'''
Apply the supplied official Reasoning BO optimization prompt and Insight schema.
Return EXACT {"comment":"...","keywords":"...","hypotheses":[{"strategy":"...","rationale":"...",
"confidence":"high","parameter_sets":[{"axis_name":0.0,...}]},...]}.
There must be3..5 hypotheses. confidence must be high/medium/low. EVERY parameter_set must be an EXACT
PHYSICAL point from the five supplied BO recommendations. No sixth point and no observed point.
The official stable high/medium/low confidence ordering selects ONE point (sequential5-to1 adaptation).
All five BO candidates may be considered, and the first valid maximum-confidence point is executed.'''
TEST_SYSTEM=COMMON+'''
Select ONE real experiment from counterexample_pool. Each point is an EXACT one-axis neighbor of the own best
observed experiment and violates an uncertain current preference. This is an allocated controlled test.
Use GP uncertainty and chemistry to compare potential improvement and the challenged preference.
Return EXACT {"candidate_id":"...","prediction":0.0,"reason":"short rationale","support":[1,...]}.
prediction is an expected objective in the supplied RAW units, not a measurement or calibrated probability.
support cites own observed1-based indices. No approximate neighbor, generated condition or extra query.
This test uses one of the same16 experimental queries and may exceed the ordinary acquisition loss gate.'''

def observations(pool,state):
    ix={k:i for i,k in enumerate(pool['ids'])};out=[]
    for i,h in enumerate(state['history'],1):
        row={'index':i,'point':{a:round(v,8) for a,v in pool['points'][ix[h['candidate_id']]].items()},
             'objective':round(h['y']*pool['objective_scale'],6),'objective_normalized':h['y']}
        if i>4 and i-5<len(state['decisions']):
            m=state['decisions'][i-5]['measurement']
            row['prequery']={k:m[k] for k in ('prequery_mu_y','prequery_predictive_sd_y','raw_region_shift_y','innovation_z')}
        out.append(row)
    return out

def public(pool,state,t):
    return {'task':'continuous_process_A' if pool['task']=='SnAr' else 'continuous_process_B',
        'description':pool['description'],'domain':pool['bounds'],'objective_unit':pool['objective_unit'],
        'objective_scale':pool['objective_scale'],'objective_name':pool['objective_name'],
        'query_round':t,'observation_count':len(state['history']),'remaining_queries':17-t,
        'observations':observations(pool,state),'shared_initial_knowledge':state.get('initial_knowledge'),
        'fixed_slots':{f'H{i+1}':a for i,a in enumerate(pool['axes'])}}

def request(client,pool,state,t,phase,system,payload,validator):
    context={'public':payload,'observed_prefix':deepcopy(state['history']),'task':pool['task'],'phase':phase,
        'protocol_sha256':state.get('protocol_sha256'),'system_sha256':digest(system)}
    context['context_sha256']=digest(context);key=context['context_sha256'];folder=ROOT/'requests'/key
    data.write(folder/'context.json',context)
    answer=transport.get(client,context,folder,system,validator)
    return answer,key

def support_indices(raw,history):
    s=raw['support']
    if not isinstance(s,list) or not s or len(set(s))!=len(s) or any(type(i)!=int or not 1<=i<=len(history) for i in s):raise ValueError('own observation indices required')

def validate_region(raw,pool,state):
    if not isinstance(raw,dict) or set(raw)!={'actions','summary'} or not isinstance(raw['actions'],list):raise ValueError('actions and summary required')
    if not isinstance(raw['summary'],str) or not 1<=len(raw['summary'])<=1600:raise ValueError('short summary required')
    old={h['id']:h for h in state['knowledge']['clauses']};slots={f'H{i+1}':a for i,a in enumerate(pool['axes'])}
    seen=set();result={};checked=[];obs={r['index']:r for r in observations(pool,state)}
    for row in raw['actions']:
        if not isinstance(row,dict) or set(row)!={'id','operation','bounds','reason','support'}:raise ValueError('invalid action fields')
        key=row['id'];op=row['operation']
        if key not in slots or key in seen:raise ValueError('unknown or duplicate fixed slot')
        seen.add(key);axis=slots[key]
        if not isinstance(row['reason'],str) or not 1<=len(row['reason'])<=1000:raise ValueError('short reason required')
        if not isinstance(row['support'],list) or not row['support']:raise ValueError('typed observed support required')
        cited=[]
        for fact in row['support']:
            if not isinstance(fact,dict) or set(fact)!={'index','axis','value','objective'}:raise ValueError('support requires index,axis,value,objective')
            i=fact['index']
            if type(i)!=int or i not in obs or i in cited or fact['axis']!=axis:raise ValueError('unobserved or inconsistent fact')
            if any(type(fact[k]) not in (int,float) or not math.isfinite(fact[k]) for k in ('value','objective')):raise ValueError('finite fact values required')
            if abs(fact['value']-obs[i]['point'][axis])>1e-8 or abs(fact['objective']-obs[i]['objective'])>1e-6:raise ValueError('observed axis value or objective mismatch')
            checked.append(deepcopy(fact));cited.append(i)
        if op in ('retain','revoke'):
            if key not in old or row['bounds']!=[]:raise ValueError('retain/revoke require active slot and empty bounds')
            if op=='retain':result[key]=deepcopy(old[key])
        elif op in ('add','revise'):
            if (op=='add')!=(key not in old):raise ValueError('add vacant, revise active')
            bounds=row['bounds']
            if not isinstance(bounds,list) or len(bounds)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) for v in bounds):raise ValueError('finite interval required')
            lo,hi=bounds
            if not 0<=lo<hi<=1 or hi-lo>=1:raise ValueError('proper normalized interval required')
            clause={'id':key,'axis':axis,'bounds':[float(lo),float(hi)]}
            if op=='revise' and clause==old[key]:raise ValueError('unchanged geometry must retain')
            result[key]=clause
        else:raise ValueError('invalid operation')
    if not set(old)<=seen:raise ValueError('account for each active preference')
    if len(result)>4:raise ValueError('at most4 active preferences')
    knowledge={'confidence':.7,'clauses':[result[k] for k in sorted(result)]}
    if result and not engine.membership(pool,knowledge).all(0).any():raise ValueError('region has no input-domain candidate witness')
    return {'knowledge':knowledge,'actions':deepcopy(raw['actions']),'summary':raw['summary'],'checked_facts':checked}

def region_call(client,pool,state,t,d=None,reasons=None):
    p=public(pool,state,t);p.update(current_knowledge=state['knowledge'],reasons=reasons or ['initial_shared_knowledge'])
    if d is not None:
        js=sorted(np.flatnonzero(d['available']),key=lambda j:(-d['base'][j],d['tie_priority'][j]))[:16]
        p['GP_summary']=[{'candidate_id':pool['ids'][j],'point':pool['points'][j],'GP_mean':float(d['stats']['mu_y'][j])*pool['objective_scale'],
            'GP_latent_SD':float(d['stats']['sd_y'][j])*pool['objective_scale'],'base_qLogEI':float(d['base'][j])} for j in js]
        p.update(B4_diagnosis=d['diagnosis'].to_dict(),calibration_bank=d['bank'],calibration_forecast=d['prior'],
            deployed_amplitude=d['amplitude'],trust_gate=d['trust'])
        # A tied class is never silently truncated in the summary.
        cutoff=d['base'][js[-1]];mask=d['available']&(d['base']>=cutoff)
        p['GP_summary_coverage']={'displayed':len(js),'total_at_or_above_cutoff':int(mask.sum()),'complete':int(mask.sum())==len(js),
             'cutoff_tied_count':int(np.sum(d['available']&(d['base']==cutoff))),'note':'unshown tied points remain eligible; summary is a presentation sample only'}
        classes=[];covered=0;X=np.asarray(pool['z'])
        for score in np.unique(d['base'][d['available']])[::-1]:
            members=np.flatnonzero(d['available']&(d['base']==score))
            classes.append({'base_qLogEI':float(score),'candidate_count':len(members),
                'GP_mean_range':[float(d['stats']['mu_y'][members].min())*pool['objective_scale'],float(d['stats']['mu_y'][members].max())*pool['objective_scale']],
                'GP_latent_SD_range':[float(d['stats']['sd_y'][members].min())*pool['objective_scale'],float(d['stats']['sd_y'][members].max())*pool['objective_scale']],
                'normalized_ranges':{a:[float(X[members,k].min()),float(X[members,k].max())] for k,a in enumerate(pool['axes'])}})
            covered+=len(members)
            if covered>=16:break
        p['complete_highest_score_classes']=classes
    return request(client,pool,state,t,'initial' if d is None else 'refresh',REGION_SYSTEM,p,lambda r:validate_region(r,pool,state))

def validate_point(raw,pool,history):
    if not isinstance(raw,dict) or set(raw)!={'point','reason','support'}:raise ValueError('point,reason,support required')
    point=raw['point']
    if not isinstance(point,dict) or set(point)!=set(pool['axes']):raise ValueError('all physical axes required')
    for a in pool['axes']:
        v=point[a];lo,hi=pool['bounds'][a]
        if type(v) not in (int,float) or not math.isfinite(v) or not lo<=v<=hi:raise ValueError('physical point outside domain')
    support_indices(raw,history)
    if not isinstance(raw['reason'],str) or not raw['reason'].strip():raise ValueError('concise reason required')
    return deepcopy(raw)

def point_call(client,pool,state,t,original_prompt):
    p=public(pool,state,t);p['original_LLINBO_prompt']=original_prompt
    return request(client,pool,state,t,'LLINBO_proposal',POINT_SYSTEM,p,lambda r:validate_point(r,pool,state['history']))

def project(pool,state,t,point):
    z=np.array([(point[a]-pool['bounds'][a][0])/(pool['bounds'][a][1]-pool['bounds'][a][0]) for a in pool['axes']])
    available=~np.isin(pool['ids'],[h['candidate_id'] for h in state['history']]);dist=((np.asarray(pool['z'])-z)**2).sum(1)
    js=np.flatnonzero(available);js=js[dist[js]==dist[js].min()];ranks=engine.priority(len(dist),t,0);j=int(js[np.argmin(ranks[js])])
    return {'candidate_id':pool['ids'][j],'z':pool['z'][j],'raw_physical_point':point,'normalized_distance':float(np.sqrt(dist[j]))}

def validate_rbo(raw,pool,state,shortlist):
    if not isinstance(raw,dict) or set(raw)!={'comment','keywords','hypotheses'}:raise ValueError('official Insight fields required')
    if not isinstance(raw['comment'],str) or not isinstance(raw['keywords'],str) or not isinstance(raw['hypotheses'],list) or not 3<=len(raw['hypotheses'])<=5:raise ValueError('3..5 hypotheses required')
    options={tuple(pool['points'][j][a] for a in pool['axes']):pool['ids'][j] for j in shortlist}
    for h in raw['hypotheses']:
        if not isinstance(h,dict) or set(h)!={'strategy','rationale','confidence','parameter_sets'} or h['confidence'] not in ('high','medium','low'):raise ValueError('invalid official hypothesis')
        if not all(isinstance(h[k],str) and h[k].strip() for k in ('strategy','rationale')):raise ValueError('short scientific rationale required')
        if not isinstance(h['parameter_sets'],list) or not h['parameter_sets']:raise ValueError('nonempty parameter sets required')
        for p in h['parameter_sets']:
            if not isinstance(p,dict) or set(p)!=set(pool['axes']) or tuple(p[a] for a in pool['axes']) not in options:raise ValueError('point outside exact five BO recommendations')
    point=rbo_native.NATIVE['BaseReasoner']._extract_candidates_from_insight(None,json.dumps(raw),n=1)[0]
    return {'selected_id':options[tuple(point[a] for a in pool['axes'])],'insight':deepcopy(raw)}

def rbo_call(client,pool,state,t,d):
    top=sorted(np.flatnonzero(d['available']),key=lambda j:(-d['base'][j],d['tie_priority'][j]))[:5]
    p=public(pool,state,t);p['BO_recommendations']=[{'point':pool['points'][j],'GP_mean':float(d['stats']['mu_y'][j])*pool['objective_scale'],
        'GP_latent_SD':float(d['stats']['sd_y'][j])*pool['objective_scale'],'base_qLogEI':float(d['base'][j])} for j in top]
    template=data.read('third_party/reasoning_bo_official/src/prompts/test_prompts.json')['optimization_loop']['en']
    p['official_optimization_template']=template
    p['count_adaptation']='Use query_round and observation_count above; recommendations are five, select one, feedback each round. Named continuous physical axes replace source-specific nominal metadata.'
    p['own_previous_insights']=[e['llm_insight'] for e in state['decisions'] if e.get('llm_insight')]
    return request(client,pool,state,t,'ReasoningBO',RBO_SYSTEM,p,lambda r:validate_rbo(r,pool,state,top))

def test_call(client,pool,state,t,d,prepared):
    p=public(pool,state,t);p['counterexample_pool']=prepared;p['current_knowledge']=state['knowledge']
    options={r['candidate_id']:r for r in prepared['candidates']}
    def validate(raw):
        if not isinstance(raw,dict) or set(raw)!={'candidate_id','prediction','reason','support'} or raw['candidate_id'] not in options:raise ValueError('select supplied exact one-axis neighbor')
        if type(raw['prediction']) not in (int,float) or not math.isfinite(raw['prediction']) or raw['prediction']<0:raise ValueError('finite nonnegative expected raw objective required')
        support_indices(raw,state['history'])
        if not isinstance(raw['reason'],str) or not raw['reason'].strip():raise ValueError('short rationale required')
        return {**deepcopy(raw),**{k:options[raw['candidate_id']][k] for k in ('changed_axis','challenged_clause_ids')}}
    return request(client,pool,state,t,'controlled_test',TEST_SYSTEM,p,validate)
