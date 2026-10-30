"""Complete counterexamples and semantic arbitration inside exact GP tie classes."""
from copy import deepcopy
import math
import numpy as np
from . import stable_r2_policy as base
from . import stable_structure_engine as engine
from . import pointwise_hte_llm as llm

SYSTEM='''Rate EVERY supplied chemistry experiment for expected objective value using molecular
structures and your own actual observations. These candidates change exactly ONE axis from
the supplied best observed anchor. Current chemical preferences are uncertain, not facts.
GP means and acquisitions are uncertain predictions, not measured outcomes. Consider both
productive chemistry and the uncertainty of extrapolating confounded observations. Prioritize
the best chance of improving the objective, rather than only a nearby informative low-yield test.
No unlisted water, time, temperature or other reagent/condition can be changed or added.
Do not identify the benchmark or recall its outcomes. Candidate IDs and presentation order
are not ranks or distances. All candidates must be considered; there is no top5 truncation.
Return EXACT JSON {"ratings":[{"candidate_id":"...","expected_yield_percent":0.0,
"reason":"short testable chemistry"},...],"preference_order":["...",...],
"support":[<own observed1-based indices>],"summary":"short comparison"}.
Every candidate must occur exactly once in ratings and exactly once in preference_order.
Expected values are finite numbers in[0,100]. Preference_order states your chemistry-based
preference and only resolves equal expected values. Reasons<=360 chars, summary<=1200 chars.
Do not supply private reasoning or invent observations.'''
TIE_SYSTEM=SYSTEM.replace('These candidates change exactly ONE axis from\nthe supplied best observed anchor.',
    'Every candidate is in the COMPLETE exact highest guided-GP-acquisition tie class.\nThey may differ in several axes; choose one using chemistry without claiming isolated causality.\nThe GP is exactly indifferent on its acquisition score, so this is a chemical tie review.')


def canonical_rows(pool,rows):
    ix={k:i for i,k in enumerate(pool['ids'])}
    return sorted(rows,key=lambda r:tuple(pool['points'][ix[r['candidate_id']]][a] for a in pool['axes']))


def build_candidates(pool,state,d):
    p=base.focused_candidates(pool,state,d)
    if not p['candidates']:return p
    rank=engine.competition_ranks(d['base'],d['available']);ix={k:i for i,k in enumerate(pool['ids'])}
    for row in p['candidates']:row['base_rank']=int(rank[ix[row['candidate_id']]])
    p['candidates']=canonical_rows(pool,p['candidates'])
    p['shortlist_rule']='Complete focused axis; no qLogEI top-k or candidate-ID truncation.'
    return p


def ratings(raw,prepared,history):
    if not isinstance(raw,dict) or set(raw)!={'ratings','preference_order','support','summary'}:raise ValueError('complete rating/order schema required')
    options={c['candidate_id']:c for c in prepared['candidates']};score={}
    if not isinstance(raw['ratings'],list) or len(raw['ratings'])!=len(options):raise ValueError('rate all candidates')
    for row in raw['ratings']:
        if not isinstance(row,dict) or set(row)!={'candidate_id','expected_yield_percent','reason'}:raise ValueError('rating row schema')
        k=row['candidate_id'];v=row['expected_yield_percent']
        if k not in options or k in score:raise ValueError('unknown/duplicate candidate')
        if type(v) not in (float,int) or not math.isfinite(v) or not 0<=v<=100:raise ValueError('finite0..100 expected percent')
        if not isinstance(row['reason'],str) or not row['reason'].strip() or len(row['reason'])>360:raise ValueError('short reason')
        score[k]=row
    order=raw['preference_order']
    if not isinstance(order,list) or len(order)!=len(options) or len(set(order))!=len(order) or set(order)!=set(options):raise ValueError('complete unique preference order')
    support=raw['support']
    if not isinstance(support,list) or not support or any(type(i)!=int or not 1<=i<=len(history) for i in support) or len(set(support))!=len(support):raise ValueError('own observed support')
    if not isinstance(raw['summary'],str) or not raw['summary'].strip() or len(raw['summary'])>1200:raise ValueError('short summary')
    chosen=max(order,key=lambda k:score[k]['expected_yield_percent']);r=score[chosen]
    return {**deepcopy(raw),'candidate_id':chosen,'prediction':r['expected_yield_percent'],'reason':r['reason']}


def validate(raw,prepared,history):
    plan=ratings(raw,prepared,history);row=next(r for r in prepared['candidates'] if r['candidate_id']==plan['candidate_id'])
    return {**plan,'clause_id':row['challenged_clause_ids'][0],'changed_axis':row['changed_axis'],
        'anchor_observation_index':prepared['anchor_observation_index'],'anchor_candidate_id':prepared['anchor_candidate_id']}


def tie_candidates(pool,state,d,t):
    if state.get('fit_recovery_seen',False) or max(h['y'] for h in state['history'])>=.95 or len(state['tie_calls'])>=2:return {'candidates':[]}
    top=float(d['score'][d['eligible']].max());indices=np.flatnonzero(d['eligible']&(d['score']==top))
    if len(indices)<2:return {'candidates':[]}
    rows=[{'candidate_id':pool['ids'][j],'point':llm.nominal.public_point(pool,pool['points'][j]),
        'GP_mean_percent':round(float(d['stats']['mu_y'][j])*100,3),'GP_latent_SD_pp':round(float(d['stats']['sd_y'][j])*100,3),
        'base_qLogEI':float(d['base'][j]),'guided_qLogEI':float(d['score'][j])} for j in indices]
    return {'candidates':canonical_rows(pool,rows),'guided_qLogEI':top,'scope':'Complete exact maximum tie class; no lower-score candidate may cross.'}


def tie_call(client,request,pool,state,d,t):
    prepared=tie_candidates(pool,state,d,t)
    if not prepared['candidates']:return None
    public={'task':'reaction_A' if pool['task_id']=='Perera_Suzuki' else 'reaction_B','reaction_class':pool['reaction_class'],
        'objective_unit':pool['objective_unit'],'domain':llm.nominal.domain(pool),'observations':llm.nominal.observations(pool,state,include_gp=False),
        'observation_count':len(state['history']),'query_round':t,'remaining_queries':17-t,'tie_pool':prepared,
        'current_knowledge':llm.nominal.public_knowledge(pool,state['knowledge'])}
    answer,key=request(client,public,state,TIE_SYSTEM,lambda raw:ratings(raw,prepared,state['history']),'tie_review')
    return {'prepared':prepared,'validated':answer,'context_sha256':key}


def grouped_summary(cache,state):
    pool=cache.pool;base_scores,_,_=cache.score(state['history']);stats=cache.stats(state['history'])
    available=~np.isin(pool['ids'],[h['candidate_id'] for h in state['history']]);result=[];represented=0
    for score in np.unique(base_scores[available])[::-1]:
        indices=np.flatnonzero(available&(base_scores==score));frequencies={}
        for axis in pool['axes']:
            counts={v:sum(pool['points'][j][axis]==v for j in indices) for v in pool['levels'][axis]}
            frequencies[axis]={llm.nominal.public_point(pool,{a:v if a==axis else pool['levels'][a][0] for a in pool['axes']})[axis]:n for v,n in counts.items() if n}
        result.append({'base_qLogEI':float(score),'candidate_count':len(indices),
            'GP_mean_percent_range':[float(stats['mu_y'][indices].min()*100),float(stats['mu_y'][indices].max()*100)],
            'GP_latent_SD_pp_range':[float(stats['sd_y'][indices].min()*100),float(stats['sd_y'][indices].max()*100)],
            'category_frequencies':frequencies,'scope':'Full tied score class; marginal frequencies do not encode all joint combinations.'})
        represented+=len(indices)
        if represented>=16:break
    return result
