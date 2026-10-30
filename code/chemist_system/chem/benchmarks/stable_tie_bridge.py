"""Early chemical arbitration over a complete maximum-acquisition tie class."""
from copy import deepcopy
import numpy as np
from . import pointwise_hte_llm as llm
from . import stable_r2_engine as engine

SYSTEM='''Choose ONE next chemistry experiment from the COMPLETE maximum-base-qLogEI tie class.
All supplied points are actual unqueried measured-domain candidates and have exactly the
same maximum base acquisition score. The GP cannot rank them by acquisition. Your task
is to use supplied chemistry and own observed experiments to choose a plausible productive
experiment that also resolves uncertainty in the early search. Consider the entire class,
not the first rows or smallest IDs. IDs and category integers are lookup symbols only;
neither numerical distances nor rankings nor chemical reactivity follow from their order.
Current knowledge preferences are fallible inferences from a few confounded observations.
Do not require a choice to satisfy those preferences. Low yield in one multi-factor setting
does not establish that a category is intrinsically poor. You may change multiple factors,
but do not claim their effects are isolated. Do not identify the benchmark or recall its outcomes.
Use only supplied structures, current observed results and frozen predictions; never unseen yields.
Return only JSON {"candidate_id":"...","prediction":"...","reason":"...","support":[<own observed1-based indices>]}.
Provide a short testable prediction and concise scientific rationale, not private reasoning.'''


def candidates(pool,state,d,t):
    if t>2 or len(state['recombination_calls'])>=2 or max(h['y'] for h in state['history'])>=.99:return {'candidate_ids':[],'candidates':[]}
    top=float(d['base'][d['available']].max());mask=d['available']&(d['base']==top)
    if mask.sum()<5:return {'candidate_ids':[],'candidates':[]}
    indices=engine.ordered(d['base'],mask,t,state['tie_seed'])
    levels={axis:list(pool['levels'][axis]) for axis in pool['axes']}
    mapping={a:{v:i for i,v in enumerate(levels[a])} for a in pool['axes']}
    structure_codes={a:{str(i):value for i,value in enumerate(levels[a])} for a in pool['axes']}
    rows=[[pool['ids'][j]]+[mapping[a][pool['points'][j][a]] for a in pool['axes']] for j in indices]
    return {'candidate_ids':[pool['ids'][j] for j in indices],'candidates':rows,
        'columns':['candidate_id']+pool['axes'],'category_structures':structure_codes,
        'shared_max_base_qLogEI':top,'count':len(rows),
        'GP_mean_range_percent':[float(d['stats']['mu_y'][indices].min()*100),float(d['stats']['mu_y'][indices].max()*100)],
        'GP_latent_SD_range_pp':[float(d['stats']['sd_y'][indices].min()*100),float(d['stats']['sd_y'][indices].max()*100)],
        'rule':'Complete maximum-qLogEI class, no ID cutoff; row order from isolated tie RNG. Category integers are nominal labels only.'}


def validate(raw,prepared,history):
    if not isinstance(raw,dict) or set(raw)!={'candidate_id','prediction','reason','support'}:raise ValueError('tie arbitration schema')
    if raw['candidate_id'] not in prepared['candidate_ids']:raise ValueError('not in the complete current GP maximum class')
    for k in ('prediction','reason'):
        if not isinstance(raw[k],str) or not raw[k].strip() or len(raw[k])>1600:raise ValueError('short explanation required')
    s=raw['support']
    if not isinstance(s,list) or not s or len(set(s))!=len(s):raise ValueError('unique evidence indices required')
    if any(type(i)!=int or not 1<=i<=len(history) for i in s):raise ValueError('unobserved evidence')
    return deepcopy(raw)


def call(client,request,pool,state,d,t):
    prepared=candidates(pool,state,d,t)
    if not prepared['candidate_ids']:return None
    public={'task':'reaction_A' if pool['task_id']=='Perera_Suzuki' else 'reaction_B',
        'reaction_class':pool['reaction_class'],'objective_unit':pool['objective_unit'],
        'observations':llm.nominal.observations(pool,state,include_gp=False),'observation_count':len(state['history']),
        'query_round':t,'remaining_queries':17-t,'uncertain_preferences':llm.nominal.public_knowledge(pool,state['knowledge']),
        'complete_tie_class':prepared,'normal_BO_reference_id':pool['ids'][d['reference']],
        'normal_R2_proposal_id':pool['ids'][d['selected']],'normal_gate_budget':d['budget']}
    answer,key=request(client,public,state,SYSTEM,lambda raw:validate(raw,prepared,state['history']),'early_GP_tie_arbitration')
    return {'prepared':prepared,'validated':answer,'context_sha256':key}
