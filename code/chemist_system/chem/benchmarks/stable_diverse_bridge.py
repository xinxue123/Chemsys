"""Contrasts, not observation counts, determine alternative-category admissibility."""
from copy import deepcopy
from functools import lru_cache
from . import stable_structure_features as features
from . import reasoning_bo_data as data
import math
import numpy as np
from . import pointwise_hte_llm as llm
from . import stable_structure_engine as engine

SIMILARITY_CUTOFF=.5

@lru_cache(maxsize=2)
def catalog(task):return data.read(features.ROOT/(task+'.json'))


SYSTEM='''Rate ALL supplied real unqueried experiments for expected objective value.
The GP's most sensitive category effect is extreme and lacks an observed isolated contrast;
a confounded low result does not prove that a substrate or catalyst class is intrinsically poor.
The candidates change that category and one new context category while every other variable
is fixed at the best own observed experiment. These are JOINT tests, not isolated causal tests.
Use supplied molecular structures and own actual observations. Compare productive catalytic
cycles and reagent activation; prioritize a plausible improvement ABOVE the current incumbent,
not merely an informative low-yield demonstration. Explicitly consider every candidate before
rating. No GP candidate means are supplied because that extrapolation is what is being tested.
Do not identify the benchmark or recall its outcomes. IDs and row order are not ranks or chemical
distances. Do not assume that unlisted water, temperature, time, concentration or extra reagents
can be changed or added; distinguish an uncertain activation mechanism from an observed fact.
Return EXACT JSON {"ratings":[{"candidate_id":"...","expected_yield_percent":0.0,
"reason":"short testable chemistry"},...],"support":[<own observed1-based indices>],
"summary":"concise cross-candidate comparison"}. Include every candidate exactly once.
Expected values must be finite numbers in [0,100], not intervals. Reasons<=360 chars;
summary<=1200 chars. Use appropriately uncertain expectations, not invented measurements.
The software selects the highest expected objective. This allocated test is one of the same16
queries and may exceed the ordinary acquisition-loss gate. No private reasoning is requested.'''


def candidates(pool,state,d,t):
    empty={'candidates':[]}
    if t>2 or len(state['recombination_calls'])>=2 or max(h['y'] for h in state['history'])>=.99:return empty
    ix={k:i for i,k in enumerate(pool['ids'])};diag=d['model_identifiability']
    if state['recombination_calls']:
        first=state['recombination_calls'][0]['validated']['plan']
        critical,context=first['critical_axis'],first['context_axis']
    else:
        if not diag['extreme_unsupported']:return empty
        critical=diag['critical_axis']
        options=[a for a in pool['axes'] if a!=critical and diag['isolated_pairs'][a]==0 and diag['unseen_levels'][a]]
        if not options:return empty
        context=min(options,key=lambda a:(-len(diag['unseen_levels'][a]),a))
    anchor_position=max(range(len(state['history'])),key=lambda i:state['history'][i]['y'])
    anchor=pool['points'][ix[state['history'][anchor_position]['candidate_id']]]
    counts={v:sum(pool['points'][ix[h['candidate_id']]][critical]==v for h in state['history']) for v in pool['levels'][critical]}
    weak={v for v in pool['levels'][critical] if v!=anchor[critical]}
    seen={pool['points'][ix[h['candidate_id']]][context] for h in state['history']}
    new=set(pool['levels'][context])-seen
    excluded=[]
    if state['recombination_calls']:
        table=catalog(pool['task_id'])['axes'][context]['Tanimoto'];levels=pool['levels'][context]
        tested=[pool['points'][ix[c['selected_id']]][context] for c in state['recombination_calls']]
        excluded=[v for v in levels if v in new and max(table[levels.index(v)][levels.index(w)] for w in tested)>=SIMILARITY_CUTOFF]
        new-=set(excluded)
    priority=engine.priority(len(pool['ids']),t,state['tie_seed']);rows=[]
    for j in sorted(np.flatnonzero(d['available']),key=lambda j:priority[j]):
        point=pool['points'][j]
        if point[critical] not in weak or point[context] not in new:continue
        if any(point[a]!=anchor[a] for a in pool['axes'] if a not in (critical,context)):continue
        rows.append({'candidate_id':pool['ids'][j],'point':llm.nominal.public_point(pool,point),
            'violated_preferences':[c['id'] for c in state['knowledge']['clauses'] if point[c['axis']] not in c['allowed']]})
    prepared={'candidates':rows,'critical_axis':critical,'context_axis':context,'diagnosis':deepcopy(diag),
        'anchor_observation_index':anchor_position+1,'anchor_candidate_id':state['history'][anchor_position]['candidate_id'],
        'anchor_yield_percent':state['history'][anchor_position]['y']*100,
        'scope':'Two-axis productive counterexample; all other conditions fixed at own best actual observation.'}
    if state['recombination_calls']:prepared['context_diversity']={'max_similarity_exclusive':SIMILARITY_CUTOFF,'excluded_context_levels':excluded,'rule':'Unqueried contexts must also be structurally distinct from prior joint tests. Repeated confounded observations do not globally exclude a critical category.'}
    return prepared


def validate(raw,prepared,history):
    if not isinstance(raw,dict) or set(raw)!={'ratings','support','summary'}:raise ValueError('expected complete ratings/support/summary')
    options={c['candidate_id']:c for c in prepared['candidates']};ratings={}
    if not isinstance(raw['ratings'],list) or len(raw['ratings'])!=len(options):raise ValueError('every candidate must be rated')
    for row in raw['ratings']:
        if not isinstance(row,dict) or set(row)!={'candidate_id','expected_yield_percent','reason'}:raise ValueError('rating schema')
        key=row['candidate_id'];value=row['expected_yield_percent']
        if key not in options or key in ratings:raise ValueError('unknown or repeated candidate')
        if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=100:raise ValueError('finite expected percent in0..100 required')
        if not isinstance(row['reason'],str) or not row['reason'].strip() or len(row['reason'])>360:raise ValueError('short reason required')
        ratings[key]=row
    if set(ratings)!=set(options):raise ValueError('incomplete ratings')
    if not isinstance(raw['summary'],str) or not raw['summary'].strip() or len(raw['summary'])>1200:raise ValueError('short comparison required')
    support=raw['support']
    if not isinstance(support,list) or not support or any(type(i)!=int or not 1<=i<=len(history) for i in support) or len(set(support))!=len(support):raise ValueError('own observed support only')
    # Candidate list already follows round-specific random priority, independent of IDs.
    chosen=max(options,key=lambda key:ratings[key]['expected_yield_percent'])
    row=ratings[chosen]
    return {**deepcopy(raw),'candidate_id':chosen,'prediction':row['expected_yield_percent'],'reason':row['reason'],
        **{k:deepcopy(prepared[k]) for k in ('critical_axis','context_axis','diagnosis','anchor_observation_index','anchor_candidate_id')},
        'violated_preferences':deepcopy(options[chosen]['violated_preferences'])}


def call(client,request,pool,state,d,t):
    prepared=candidates(pool,state,d,t)
    if not prepared['candidates']:return None
    public={'task':'reaction_A' if pool['task_id']=='Perera_Suzuki' else 'reaction_B','reaction_class':pool['reaction_class'],
        'objective_unit':pool['objective_unit'],'domain':llm.nominal.domain(pool),
        'observations':llm.nominal.observations(pool,state,include_gp=False),'observation_count':len(state['history']),
        'query_round':t,'remaining_queries':17-t,'model_challenge_pool':prepared,
        'uncertain_preferences':llm.nominal.public_knowledge(pool,state['knowledge']),
        'previous_joint_tests':deepcopy(state['recombination_calls'])}
    answer,key=request(client,public,state,SYSTEM,lambda raw:validate(raw,prepared,state['history']),'model_challenge')
    return {'prepared':prepared,'validated':answer,'context_sha256':key}
