"""Empirical challenge to an unidentifiable extreme ARD category effect."""
from copy import deepcopy
import numpy as np
from . import pointwise_hte_llm as llm
from . import stable_r2_engine as engine

SYSTEM='''Design ONE chemistry experiment to challenge an unverified global category effect in the GP.
The GP has an extreme ARD lengthscale for one axis, but there are NO observed pairs that
isolate that axis. Its very low predicted yield and small uncertainty for other categories
can therefore be an extrapolation assumption, not established chemistry. One confounded
low-yield experiment does not show that an entire substrate/catalyst class is intrinsically poor.
Choose one of the supplied real unqueried joint-test candidates, using chemical structures,
own observed results and mechanism. The challenged category is combined with a new
context category; remaining factor levels are supported by own high-performing experiments.
You may change several factors: interpret the result as a test of the joint hypothesis,
NOT a separated causal effect. Prefer plausible productive combinations that resolve this
specific model ambiguity. Consider every option; row order, category indices and IDs are
not ranks or distances. Do not identify this benchmark or recall its outcomes.
This allocated test may be outside the usual acquisition-loss gate and is counted within
the same total experiment budget. The supplied GP predictions are NOT observed yields.
Return only JSON {"candidate_id":"...","prediction":"...","reason":"...","support":[<own observed1-based indices>]}.
Give concise testable chemistry and evidence, not private reasoning.'''


def candidates(pool,state,d,t):
    empty={'candidates':[]}
    if t>2 or len(state['recombination_calls'])>=2 or max(h['y'] for h in state['history'])>=.99:return empty
    lookup={k:i for i,k in enumerate(pool['ids'])};prior=state['recombination_calls']
    if prior:
        plan=prior[0]['validated']['plan'];critical=plan['critical_axis'];context=plan['context_axis']
        critical_levels=plan['critical_levels'];context_levels=plan['context_levels'];diagnosis=plan['diagnosis']
    else:
        diagnosis=d['model_identifiability']
        if not diagnosis['extreme_unsupported']:return empty
        critical=diagnosis['critical_axis'];possible=[a for a in pool['axes'] if a!=critical and diagnosis['isolated_pairs'][a]==0 and diagnosis['unseen_levels'][a]]
        if not possible:return empty
        context=min(possible,key=lambda a:(-len(diagnosis['unseen_levels'][a]),a))
        best=max(state['history'],key=lambda h:h['y']);bestpoint=pool['points'][lookup[best['candidate_id']]]
        counts={v:sum(pool['points'][lookup[h['candidate_id']]][critical]==v for h in state['history']) for v in pool['levels'][critical]}
        critical_levels=[v for v in pool['levels'][critical] if counts[v]<=1 and v!=bestpoint[critical]]
        context_levels=diagnosis['unseen_levels'][context]
    if not critical_levels or not context_levels:return empty
    best=max(h['y'] for h in state['history']);high=[h for h in state['history'] if h['y']>=best-.05]
    allowed={a:{pool['points'][lookup[h['candidate_id']]][a] for h in high} for a in pool['axes']}
    allowed[critical]=set(critical_levels);allowed[context]=set(context_levels)
    priority=engine.priority(len(pool['ids']),t,state['tie_seed']);indices=sorted(np.flatnonzero(d['available']),key=lambda j:priority[j]);rows=[]
    for j in indices:
        point=pool['points'][j]
        if not all(point[a] in allowed[a] for a in pool['axes']):continue
        violated=[c['id'] for c in state['knowledge']['clauses'] if point[c['axis']] not in c['allowed']]
        rows.append({'candidate_id':pool['ids'][j],'point':llm.nominal.public_point(pool,point),
            'GP_mean_percent':round(float(d['stats']['mu_y'][j])*100,3),'GP_latent_SD_pp':round(float(d['stats']['sd_y'][j])*100,3),
            'base_qLogEI':float(d['base'][j]),'base_opportunity_loss':float(d['base'][d['reference']]-d['base'][j]),
            'normal_gate_eligible':bool(d['eligible'][j]),'violated_preferences':violated})
    return {'candidates':rows,'critical_axis':critical,'context_axis':context,'critical_levels':critical_levels,
        'context_levels':context_levels,'diagnosis':diagnosis,'scope':'Joint context challenge; not isolated causality. Acquisition gate exceptions explicitly allocated.'}


def validate(raw,prepared,history):
    if not isinstance(raw,dict) or set(raw)!={'candidate_id','prediction','reason','support'}:raise ValueError('model challenge schema')
    options={c['candidate_id']:c for c in prepared['candidates']}
    if raw['candidate_id'] not in options:raise ValueError('not in current joint challenge pool')
    for k in ('prediction','reason'):
        if not isinstance(raw[k],str) or not raw[k].strip() or len(raw[k])>1600:raise ValueError('short explanation required')
    support=raw['support']
    if not isinstance(support,list) or not support or len(set(support))!=len(support):raise ValueError('unique evidence required')
    if any(type(i)!=int or not 1<=i<=len(history) for i in support):raise ValueError('unobserved evidence')
    return {**deepcopy(raw),**{k:deepcopy(prepared[k]) for k in ('critical_axis','context_axis','critical_levels','context_levels','diagnosis')},
        'violated_preferences':options[raw['candidate_id']]['violated_preferences']}


def call(client,request,pool,state,d,t):
    prepared=candidates(pool,state,d,t)
    if not prepared['candidates']:return None
    public={'task':'reaction_A' if pool['task_id']=='Perera_Suzuki' else 'reaction_B','reaction_class':pool['reaction_class'],
        'objective_unit':pool['objective_unit'],'domain':llm.nominal.domain(pool),
        'observations':llm.nominal.observations(pool,state,include_gp=False),'observation_count':len(state['history']),
        'query_round':t,'remaining_queries':17-t,'uncertain_preferences':llm.nominal.public_knowledge(pool,state['knowledge']),
        'model_challenge_pool':prepared,'previous_joint_tests':deepcopy(state['recombination_calls'])}
    answer,key=request(client,public,state,SYSTEM,lambda raw:validate(raw,prepared,state['history']),'model_challenge')
    return {'prepared':prepared,'validated':answer,'context_sha256':key}
