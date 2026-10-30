"""Exact fixed-pool single-axis tests; no nearest-neighbor causal claims."""
from copy import deepcopy
import numpy as np
from . import stable_suzuki_engine as engine
from . import grounded_regional_suzuki_engine as grounded
from . import r2_tuning_policy as source
from .suzuki_domain import CATALYSTS

SYSTEM=source.SYSTEM.replace('supplied structures','supplied named conditions').replace(
    'and tests a category excluded by a current preference.',
    'and tests a condition excluded by a current preference. Numeric axes are continuous; catalyst names are anonymous nominal labels with unknown ligand structures.').replace(
    'Contrast the chosen category with the anchor', 'Contrast the chosen condition with the anchor')
REFRESH_SYSTEM=grounded.SYSTEM_PROMPT+'''
This transport also uses the revised persistent region-trust gate. The same
geometric region remains blocked after a B4 conflict until a later OWN result
improves its frozen prequery guided squared error over the base GP and B4 clears.
Renaming clauses cannot restore trust. This is evidence-based deployment control,
not a truth probability. Single-axis controlled tests are possible only when an
EXACT neighbor exists in the supplied fixed pool, at rounds2/4/6 within16 queries.
No molecular structural recovery or multi-category joint tests are available for
anonymous catalysts. Named prequery fields ending _y use yield fractions; GP
display percentages and yield_percent use 0..100. Do not mix these units.'''


def build_candidates(pool,state,d):
    history=state['history'];ix={k:i for i,k in enumerate(pool['ids'])}
    anchor_position=max(range(len(history)),key=lambda i:history[i]['y']);anchor=ix[history[anchor_position]['candidate_id']]
    X=np.asarray(pool['z']);diff=X!=X[anchor]
    mem=engine.original.membership(pool,state['knowledge'])
    ranks=engine.competition_ranks(d['base'],d['available']);rows=[]
    if max(h['y'] for h in history)>=.95:
        return {'candidates':[],'skip_reason':'own incumbent >=95 percent'}
    for j in np.flatnonzero(d['available']&(diff.sum(1)==1)):
        axis=int(np.flatnonzero(diff[j])[0]);tested=[h['id'] for k,h in enumerate(state['knowledge']['clauses']) if h['coordinate']==axis and not mem[k,j]]
        if not tested:continue
        rows.append({'candidate_id':pool['ids'][j],'features':grounded.named_features(pool,j),
            'changed_axis':engine.original.VARIABLES[axis],'challenged_clause_ids':tested,
            'GP_mean_percent':round(float(d['stats']['mu_y'][j])*100,3),
            'GP_latent_SD_pp':round(float(d['stats']['sd_y'][j])*100,3),
            'base_qLogEI':float(d['base'][j]),'base_rank':int(ranks[j]),
            'normal_gate_eligible':bool(d['eligible'][j]),'base_opportunity_loss':float(d['base'][d['reference']]-d['base'][j])})
    focus=None
    if state['validations'] and rows:
        groups={}
        for row in rows:
            a=engine.original.VARIABLES.index(row['changed_axis']);j=ix[row['candidate_id']]
            if X[j,a] not in set(X[[ix[h['candidate_id']] for h in history],a]):groups.setdefault(a,[]).append(row)
        if groups:
            a=min(groups,key=lambda a:(-len(groups[a]),-max(r['GP_latent_SD_pp'] for r in groups[a]),engine.original.VARIABLES[a]))
            rows=groups[a];focus={'axis':engine.original.VARIABLES[a],'criterion':'most unobserved excluded values, then latent SD, then axis name'}
    # Presentation is by physical conditions, never an internal ID preference.
    rows.sort(key=lambda r:tuple(pool['points'][ix[r['candidate_id']]][a] for a in ('catalyst','t_res','temperature','catalyst_loading')))
    return {'anchor_candidate_id':history[anchor_position]['candidate_id'],'anchor_observation_index':anchor_position+1,
        'anchor_features':grounded.named_features(pool,anchor),'anchor_yield_percent':round(history[anchor_position]['y']*100,3),
        'candidates':rows,'coverage_focus':focus,'skip_reason':None if rows else 'no exact single-axis excluded neighbor in unchanged pool',
        'rule':'complete focused pool; no top5 truncation, no generated or approximate neighbors'}


def validate(raw,prepared,history):return source.validate(raw,prepared,history)


def observations(pool,state):
    ix={k:i for i,k in enumerate(pool['ids'])};out=[]
    for i,h in enumerate(state['history'],1):
        row={'index':i,**grounded.named_features(pool,ix[h['candidate_id']]),'yield_percent':round(h['y']*100,3)}
        if i>4:
            m=state['decisions'][i-5]['measurement']
            row['prequery']={k:m[k] for k in ('prequery_mu_y','prequery_predictive_sd_y','raw_region_shift_y','innovation_z')}
        out.append(row)
    return out


def grouped_summary(pool,d):
    """All highest acquisition classes covering >=16 points; no tie truncation."""
    rows=[];count=0;X=np.asarray(pool['z'])
    for score in np.unique(d['base'][d['available']])[::-1]:
        js=np.flatnonzero(d['available']&(d['base']==score))
        rows.append({'base_qLogEI':float(score),'candidate_count':len(js),
            'GP_mean_percent_range':[float(np.min(d['stats']['mu_y'][js])*100),float(np.max(d['stats']['mu_y'][js])*100)],
            'GP_latent_SD_pp_range':[float(np.min(d['stats']['sd_y'][js])*100),float(np.max(d['stats']['sd_y'][js])*100)],
            'normalized_numeric_ranges':{a:[float(X[js,k].min()),float(X[js,k].max())] for k,a in enumerate(engine.original.VARIABLES[:3])},
            'catalyst_counts':{c:sum(pool['points'][j]['catalyst']==c for j in js) for c in CATALYSTS},
            'single_candidate_features':grounded.named_features(pool,int(js[0])) if len(js)==1 else None})
        count+=len(js)
        if count>=16:break
    return rows


def public_request(pool,state,d,reasons,phase):
    dg=d['diagnosis'].to_dict();dg.pop('candidate_id',None);dg.pop('candidate_index',None)
    return {'task':state['task'],'phase':phase,'iteration':len(state['decisions'])+1,'remaining_queries':16-len(state['decisions']),
        'domain':{'normalized_continuous':{a:[0.,1.] for a in engine.original.VARIABLES[:3]},
            'physical':{'residence_time_seconds':[60.,600.],'temperature_C':[30.,110.],'catalyst_loading_mol_percent':[.5,2.5]},
            'catalysts':list(CATALYSTS)},
        'fixed_slots':{k:engine.original.VARIABLES[v] for k,v in state['slots'].items()},
        'reasons':reasons,'observations':observations(pool,state),'current_knowledge':{
            'confidence':state['knowledge']['confidence'],'clauses':[{**{k:v for k,v in h.items() if k!='coordinate'},
                'axis':engine.original.VARIABLES[h['coordinate']]} for h in state['knowledge']['clauses']]},
        'GP_summary':grouped_summary(pool,d),'GP_summary_schema':'complete exact acquisition classes covering >=16 points',
        'B4_diagnosis':dg,'calibration_bank':d['bank'],'calibration_forecast':d['prior'],
        'deployed_amplitude':d['amplitude'],'trust_gate':deepcopy(d['trust']),
        'controlled_validation_results':deepcopy(state['validations']),
        'calibration_units':{'b':'yield fraction','a':'dimensionless','prequery_fields_ending_y':'yield fraction','yield_percent':'0..100'},
        'known_limits':'anonymous catalyst labels; no molecular context or structural-kernel recovery; only exact pool neighbors may be controlled tests'}
