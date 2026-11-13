"""Chemical first choices without any GP, rank or unmeasured objective input."""
from copy import deepcopy
from itertools import product
import math
import numpy as np
from chem.benchmarks import knowledge_gp as native
from chemical_hypothesis_bo import background as original
from chemical_hypothesis_bo.hypotheses import digest


PROPOSE_SYSTEM='''Select ONE promising first experiment from ZERO observed outcomes.
You receive verified chemical/process background, exact molecular identities and original
physical units. Reason from this background, rather than candidate labels. No GP, incumbent,
acquisition score or measured objective exists in this request. Do not recall benchmark
results, search sources, infer yields from IDs or invent known experimental successes.

Return exactly {"summary":string,"first_choice":{"conditions":object,"rationale":string,
"key_assumptions":list of1..3 strings,"forecast":{"mean_raw":number,"sd_raw":number},
"falsification":string}}. All explanations should be concise English public explanations,
not private reasoning. conditions must specify EVERY declared input axis. Use EXACT listed
categorical strings/SMILES and continuous values in the declared ORIGINAL units.

The examples are input-only experimental archetypes, NOT a scored menu. You may copy one
actual example or propose another in-domain condition. No initial yield group is imposed.
All original pool candidates are allowed. For a nonexact proposal, software finds the five
nearest executable inputs by physical input distance while preserving ALL categorical
identities. A SECOND independent request then selects the actual first candidate and locks
its forecast. This projection does not use GP predictions or objective values.

forecast is a subjective uncertain objective estimate in the supplied objective unit, never
a measurement or calibrated confidence. sd_raw must be positive. For percentage objectives
mean_raw must be in [0,100]. Do not imply unsupported precision. Give a falsifiable prediction
and state uncertainty. This study tests this first choice; subsequent19 paid experiments
use unchanged native qLogEI and no further LLM intervention. Distinguish reaction yield,
chromatographic UV area, space-time yield and HPLC peak area as defined in the background.'''

CONFIRM_SYSTEM='''Choose the actual FIRST executable experiment from ZERO observations.
Use the supplied chemical/process background and the preceding independent preferred
condition. The supplied candidate list consists ONLY of the nearest physical inputs with
the exact proposed categorical identities. It contains no GP/objective rankings. You may
select any ONE of these candidates. Interpret their actual full conditions and explain why
your chosen point is preferable. Do not assume the preceding proposed forecast is correct.
No GP, incumbent, measured yield, benchmark outcome or acquisition score is supplied.
Do not seek other files or published results. Return concise English public explanations.

Return exactly {"summary":string,"first_choice":{"candidate_id":one exact offered ID,
"rationale":string,"key_assumptions":list of1..3 strings,
"forecast":{"mean_raw":number,"sd_raw":number},"falsification":string}}.
Lock the uncertain objective forecast for the ACTUAL chosen conditions in original units.
It is a subjective forecast, not a measurement. sd_raw must be positive; for percentage
objectives mean_raw must lie in [0,100]. Subsequent19 paid choices use native qLogEI.'''


def background(pool):
    card=original.context(pool)
    for variable in card['inputs']:
        value=variable.get('chemical_identities',{}).get('CN1CCCN2C1=NCCC2')
        if value:
            assert value['graph']['formula']=='C8H15N3' and value['graph']['atom_counts']['N']==3
            value['name']='7-methyl-1,5,7-triazabicyclo[4.4.0]dec-5-ene (MTBD)'
    return card


def validate_conditions(pool,conditions):
    if not isinstance(conditions,dict) or set(conditions)!=set(pool.axes):raise ValueError('Specify every actual input axis')
    domain={v['name']:v for v in background(pool)['inputs']}
    for axis,kind in zip(pool.axes,pool.kinds):
        value=conditions[axis]
        if kind=='categorical':
            if not isinstance(value,str) or value not in domain[axis]['levels']:raise ValueError('Use exact supplied categorical identities')
        else:
            limits=domain[axis]['declared_input_bounds']
            if type(value) not in (int,float) or not math.isfinite(value) or not limits[0]<=value<=limits[1]:raise ValueError('Use finite original-unit inputs inside declared bounds')


def distances(pool,conditions):
    validate_conditions(pool,conditions)
    valid=np.ones(len(pool.ids),bool);squared=np.zeros(len(pool.ids));dimensions=0
    domain={v['name']:v for v in background(pool)['inputs']}
    for axis,kind in zip(pool.axes,pool.kinds):
        if kind=='categorical':valid &= np.asarray([p[axis]==conditions[axis] for p in pool.points])
        else:
            lo,hi=domain[axis]['declared_input_bounds'];dimensions+=1
            squared+=(np.asarray([p[axis] for p in pool.points])-conditions[axis])**2/(hi-lo)**2
    if not valid.any():raise ValueError('This categorical combination is not present in the supplied executable pool')
    return valid,np.sqrt(squared/max(dimensions,1))


def neighbors(pool,conditions,count=5):
    valid,distance=distances(pool,conditions);js=np.flatnonzero(valid)
    priorities=native.semantic_priorities(pool,1,0)
    js=js[np.lexsort((priorities[js],distance[js]))][:count]
    # Display order is independent of IDs and physical-distance rank.
    display=sorted(map(int,js),key=lambda j:int(priorities[j]))
    return [dict(candidate_id=pool.ids[j],conditions=deepcopy(pool.points[j]),
        normalized_input_RMS_distance=float(distance[j])) for j in display]


def examples(pool):
    card=background(pool);domain={v['name']:v for v in card['inputs']}
    continuous=[a for a,k in zip(pool.axes,pool.kinds) if k=='continuous']
    categorical=[a for a,k in zip(pool.axes,pool.kinds) if k=='categorical']
    chosen=set()
    if continuous:
        categories=list(product(*(domain[a]['levels'] for a in categorical))) if categorical else [()]
        vertices=list(product((0.,1.),repeat=len(continuous)))+[(.5,)*len(continuous)]
        for values in categories:
            for vertex in vertices:
                point=dict(zip(categorical,values))
                for axis,fraction in zip(continuous,vertex):
                    lo,hi=domain[axis]['declared_input_bounds'];point[axis]=lo+fraction*(hi-lo)
                match=neighbors(pool,point,1)[0];chosen.add(pool.ix[match['candidate_id']])
    else:
        order=np.argsort(native.semantic_priorities(pool,1,0));chosen.update(map(int,order[:20]))
    priorities=native.semantic_priorities(pool,1,0)
    return [dict(candidate_id=pool.ids[j],conditions=deepcopy(pool.points[j])) for j in sorted(chosen,key=lambda j:int(priorities[j]))]


def public(pool):
    return dict(background=background(pool),experimental_budget=dict(initial=0,total_paid_queries=20,observed=0),
        own_observations=[],GP_available=False,qLogEI_available=False,incumbent=None,
        input_only_examples=examples(pool),example_selection='Nearest pool inputs to domain corners/center for continuous tasks; feature-keyed prior samples for categorical tasks. No target values.',
        all_original_pool_candidates_allowed=True,objective_unit=pool.unit,
        task='Independently propose one chemically meaningful first condition before any GP fit or experimental feedback.')


def _common(raw,pool,candidate_fields):
    if not isinstance(raw,dict) or set(raw)!={'summary','first_choice'} or not isinstance(raw['summary'],str):raise ValueError('Expected summary and first_choice')
    choice=raw['first_choice']
    if not isinstance(choice,dict) or set(choice)!=set(candidate_fields)|{'rationale','key_assumptions','forecast','falsification'}:raise ValueError('Invalid first-choice fields')
    for name in ('rationale','falsification'):
        if not isinstance(choice[name],str) or not 1<=len(choice[name])<=1800:raise ValueError('Supply concise rationale and falsification')
    assumptions=choice['key_assumptions']
    if not isinstance(assumptions,list) or not 1<=len(assumptions)<=3 or any(not isinstance(a,str) or not 1<=len(a)<=1200 for a in assumptions):raise ValueError('Supply1..3 explicit assumptions')
    forecast=choice['forecast']
    if not isinstance(forecast,dict) or set(forecast)!={'mean_raw','sd_raw'} or any(type(v) not in (int,float) or not math.isfinite(v) for v in forecast.values()):raise ValueError('Finite original-unit mean and SD required')
    if forecast['mean_raw']<0 or forecast['sd_raw']<=0 or (pool.group in ('P0','P1') and forecast['mean_raw']>100):raise ValueError('Invalid objective forecast bounds')
    return deepcopy(choice)


def validate_proposal(raw,pool):
    choice=_common(raw,pool,{'conditions'});validate_conditions(pool,choice['conditions'])
    offered=neighbors(pool,choice['conditions']);exact=[r for r in offered if r['normalized_input_RMS_distance']<=1e-12]
    return dict(choice=choice,neighbors=offered,exact_candidate_id=exact[0]['candidate_id'] if exact else None)


def confirmation(pool,first):
    return dict(background=background(pool),experimental_budget=dict(initial=0,total_paid_queries=20,observed=0),
        own_observations=[],GP_available=False,qLogEI_available=False,incumbent=None,
        preceding_independent_choice=first['raw'],executable_candidates=first['plan']['neighbors'],
        objective_unit=pool.unit,projection_uses_only_input_conditions=True)


def validate_confirmation(raw,pool,offered):
    choice=_common(raw,pool,{'candidate_id'})
    if choice['candidate_id'] not in {v['candidate_id'] for v in offered}:raise ValueError('Select one exact offered executable candidate ID')
    return dict(choice=choice)
