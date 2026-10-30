"""Observed-supported one-factor recombination, inside the original BO gate."""
from copy import deepcopy
import numpy as np
from . import pointwise_hte_llm as llm
from . import stable_r2_engine as engine

SYSTEM='''You choose ONE next chemistry experiment from an observed-supported recombination pool.
Use only supplied structures, frozen GP predictions and your own observed experiments.
Do not identify the benchmark, recall its outcomes, or treat candidate IDs as rankings.
The GP recently had a severe fit collapse. Its fitted marginal likelihood was improved,
but tiny acquisition differences are not evidence that an unfamiliar condition is chemically superior.
Each option is a real unqueried point within the original BO acquisition-loss budget.
Every factor level has already occurred in a high-performing own observation, and each
option differs in exactly one factor from at least one such observation. Other axes are fixed
relative to that anchor. Select the most useful plausible improvement, combining supported
chemistry with GP uncertainty. State which observation/axis comparison your choice tests.
These are uncertain local comparisons, not general causal facts. GP predictions are not yields.
Return ONLY JSON {"candidate_id":"...","prediction":"...","reason":"...","support":[<observed1-based indices>]}.
Use a short testable prediction and concise scientific rationale, not private reasoning.'''


def candidates(pool,state,d,t):
    empty={'candidates':[]}
    if (not state.get('fit_recovery_seen') or t>6 or len(state['recombination_calls'])>=2
            or not .95<=max(h['y'] for h in state['history'])<.99):return empty
    best=max(h['y'] for h in state['history']);ix={k:i for i,k in enumerate(pool['ids'])}
    high=[(i,h) for i,h in enumerate(state['history'],1) if h['y']>=best-.05]
    if len(high)<2:return empty
    levels={axis:{pool['points'][ix[h['candidate_id']]][axis] for _,h in high} for axis in pool['axes']}
    order=engine.ordered(d['base'],d['eligible'],t,state['tie_seed']);rows=[]
    ranks=engine.competition_ranks(d['base'],d['available'])
    for j in order:
        point=pool['points'][j]
        if any(point[a] not in levels[a] for a in pool['axes']):continue
        anchors=[]
        for i,h in high:
            old=pool['points'][ix[h['candidate_id']]];changed=[a for a in pool['axes'] if old[a]!=point[a]]
            if len(changed)==1:anchors.append({'observation_index':i,'changed_axis':changed[0]})
        if not anchors:continue
        rows.append({'candidate_id':pool['ids'][j],'point':llm.nominal.public_point(pool,point),
            'controlled_anchors':anchors,'GP_mean_percent':round(float(d['stats']['mu_y'][j])*100,3),
            'GP_latent_SD_pp':round(float(d['stats']['sd_y'][j])*100,3),'base_qLogEI':float(d['base'][j]),
            'base_rank':int(ranks[j]),'base_opportunity_loss':float(d['base'][d['reference']]-d['base'][j])})
    return {'candidates':rows,'high_observation_indices':[i for i,_ in high],
        'rule':'Observed levels from within5pp of incumbent; one changed axis relative to a high observation; original BO gate; no future truths.'}


def validate(raw,prepared,history):
    if not isinstance(raw,dict) or set(raw)!={'candidate_id','prediction','reason','support'}:raise ValueError('recombination schema')
    options={c['candidate_id']:c for c in prepared['candidates']}
    if raw['candidate_id'] not in options:raise ValueError('not an available gated recombination')
    for k in ('prediction','reason'):
        if not isinstance(raw[k],str) or not raw[k].strip() or len(raw[k])>1600:raise ValueError('short scientific explanation required')
    if not isinstance(raw['support'],list) or not raw['support'] or len(set(raw['support']))!=len(raw['support']):raise ValueError('evidence indices required')
    if any(type(i)!=int or not 1<=i<=len(history) for i in raw['support']):raise ValueError('unobserved evidence')
    anchors=options[raw['candidate_id']]['controlled_anchors']
    if not any(a['observation_index'] in raw['support'] for a in anchors):raise ValueError('cite a controlled anchor')
    return deepcopy(raw)


def call(client,request,pool,state,d,t):
    prepared=candidates(pool,state,d,t)
    if not prepared['candidates']:return None
    public={'task':'reaction_A' if pool['task_id']=='Perera_Suzuki' else 'reaction_B',
        'reaction_class':pool['reaction_class'],'objective_unit':pool['objective_unit'],'domain':llm.nominal.domain(pool),
        'observations':llm.nominal.observations(pool,state,include_gp=False),'observation_count':len(state['history']),
        'query_round':t,'remaining_queries':17-t,'current_knowledge':llm.nominal.public_knowledge(pool,state['knowledge']),
        'recombination_pool':prepared,'normal_BO_reference_id':pool['ids'][d['reference']],
        'normal_R2_proposal_id':pool['ids'][d['selected']],'normal_gate_budget':d['budget'],
        'previous_recombination_results':deepcopy(state['recombination_calls'])}
    answer,key=request(client,public,state,SYSTEM,lambda raw:validate(raw,prepared,state['history']),'empirical_recombination')
    return {'prepared':prepared,'validated':answer,'context_sha256':key}
