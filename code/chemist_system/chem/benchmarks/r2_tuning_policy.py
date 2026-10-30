"""Feature-only controlled counterexamples around the best observed experiment."""
from copy import deepcopy

from . import pointwise_hte_llm as shared_llm

SYSTEM='''You design ONE controlled chemistry experiment to test uncertain chemical knowledge.
Use only supplied structures, own observed experiments, and frozen GP predictions.
Do not identify this benchmark, recall outcomes, or use candidate IDs as numerical ranks.
Every supplied candidate is an actual unqueried measured-domain point differing in exactly
ONE axis from the best observed anchor, and tests a category excluded by a current preference.
Choose a scientifically useful counterexample with a realistic chance of improving yield.
Low yield in a confounded earlier run does not prove that category is intrinsically poor.
Contrast the chosen category with the anchor while keeping all other axes fixed.
GP predictions and knowledge mean shifts are uncertain; they are NOT observed yields.
This is an allocated validation experiment counted within the same16 queries. It may be
outside the usual acquisition-loss gate; the candidate must still be in the supplied list.
Return exactly {"candidate_id":"...","clause_id":"...","prediction":"...",
"reason":"...","support":[<own observed1-based indices>]}.
clause_id must be one of that candidate's challenged_clause_ids.
Give a short testable prediction and concise scientific rationale, not private reasoning.
Do not claim a general causal conclusion from one comparison. Output only JSON.'''


def build_candidates(pool,state,d):
    history=state['history'];ids=pool['ids'];lookup={k:i for i,k in enumerate(ids)}
    anchor_position=min(range(len(history)),key=lambda i:(-history[i]['y'],i))
    anchor=pool['points'][lookup[history[anchor_position]['candidate_id']]]
    queried={h['candidate_id'] for h in history};clauses=state['knowledge']['clauses'];rows=[]
    ranking=sorted((i for i,k in enumerate(ids) if k not in queried),key=lambda j:(-float(d['base'][j]),ids[j]))
    rank={j:i+1 for i,j in enumerate(ranking)}
    for j,point in enumerate(pool['points']):
        if ids[j] in queried:continue
        changed=[a for a in pool['axes'] if point[a]!=anchor[a]]
        if len(changed)!=1:continue
        axis=changed[0]
        tested=[h['id'] for h in clauses if h['axis']==axis and point[axis] not in h['allowed']]
        if not tested:continue
        rows.append({'candidate_id':ids[j],'point':shared_llm.nominal.public_point(pool,point),
            'changed_axis':axis,'challenged_clause_ids':tested,'base_rank':rank[j],
            'GP_mean_percent':round(float(d['stats']['mu_y'][j])*100,3),
            'GP_latent_SD_pp':round(float(d['stats']['sd_y'][j])*100,3),
            'base_qLogEI':float(d['base'][j]),'normal_gate_eligible':bool(d['eligible'][j]),
            'base_opportunity_loss':float(d['base'][d['reference']]-d['base'][j])})
    return {'anchor_candidate_id':history[anchor_position]['candidate_id'],'anchor_observation_index':anchor_position+1,
        'anchor_point':shared_llm.nominal.public_point(pool,anchor),
        'anchor_yield_percent':round(history[anchor_position]['y']*100,3),'candidates':rows}


def validate(raw,prepared,history):
    expected={'candidate_id','clause_id','prediction','reason','support'}
    if not isinstance(raw,dict) or set(raw)!=expected:raise ValueError('expected controlled-test JSON schema')
    candidates={c['candidate_id']:c for c in prepared['candidates']}
    if not isinstance(raw['candidate_id'],str) or raw['candidate_id'] not in candidates:raise ValueError('candidate not in controlled counterexample pool')
    c=candidates[raw['candidate_id']]
    if raw['clause_id'] not in c['challenged_clause_ids']:raise ValueError('clause does not concern changed axis')
    for name in ('prediction','reason'):
        if not isinstance(raw[name],str) or not raw[name].strip() or len(raw[name])>1600:raise ValueError('short prediction/reason required')
    support=raw['support']
    if not isinstance(support,list) or not support:raise ValueError('nonempty unique evidence indices')
    if any(type(i)!=int or not 1<=i<=len(history) for i in support):raise ValueError('unobserved evidence')
    if len(set(support))!=len(support):raise ValueError('duplicate evidence indices')
    return {**deepcopy(raw),'anchor_candidate_id':prepared['anchor_candidate_id'],
        'anchor_observation_index':prepared['anchor_observation_index'],'changed_axis':c['changed_axis']}
