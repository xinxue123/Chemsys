"""V8 fixed numerical hypotheses, birth features, and conservative revision.

Only the original example pair uses V5 geometric comparability. A predeclared
background scope can transfer the birth feature across backgrounds; no later
highest-reference reanchoring or pairwise probe constraint is imposed here.
Four-account credit lives in knowledge_numeric_evidence, not this geometry file.
"""
from copy import deepcopy
from math import isfinite

import numpy as np

from . import knowledge_transfer_policy as geometry

MAX_ACTIONS=3
MAX_LIVE_CLAIMS=4
BACKGROUND_RADIUS=geometry.BACKGROUND_RADIUS
MAX_WEIGHT=.5
GAIN_FIELDS=('gp_gain','null_gain','mean_gain','null_mean_gain')
ACTION_FIELDS={'op','id','plus_id','minus_id','focus','background','numeric_prior','reason'}
_digest=geometry._digest
_get=geometry._get
scope_indices=geometry.scope_indices


def initial_ledger():
    from .knowledge_numeric_evidence import initial_ledger as function
    return function()


def evidence_summary(h):
    from .knowledge_numeric_evidence import evidence_summary as function
    return function(h)


def weight(h):
    from .knowledge_numeric_evidence import weight as function
    return function(h)


def record_feedback(h,pred,observation):
    if h['semantic_key']!=semantic_key(h) or h['family_key']!=family_key(h):
        raise ValueError('Frozen numeric hypothesis semantics changed before feedback')
    if h['birth_prefix_sha256']!=_digest(h['birth_prefix']):
        raise ValueError('Birth prefix changed before feedback')
    from .knowledge_numeric_evidence import record_feedback as function
    return function(h,pred,observation)


def validate_numeric_prior(prior):
    """Return only the raw-unit prior contract; None is explicit abstention."""
    if prior is None:return None
    if not isinstance(prior,dict) or set(prior)!={'delta_mean_raw','delta_sd_raw'}:
        raise ValueError('numeric_prior requires delta_mean_raw and delta_sd_raw, or null')
    if not all(type(x) in (int,float) and isfinite(x) and x>0 for x in prior.values()):
        raise ValueError('Numerical prior mean and SD must be finite positive numbers, not confidence levels')
    return {key:float(prior[key]) for key in ('delta_mean_raw','delta_sd_raw')}


def direction_family(h):
    return dict(directions={axis:spec['direction'] for axis,spec in h['focus'].items()},
                categorical_preference=deepcopy(h['categorical_preference']))


def family_key(h):return _digest(direction_family(h))


def feature_vector(pool,h):
    """Frozen physical feature, equal mean of one or two focus components."""
    axes=list(_get(pool,'axes'));kinds=geometry._kinds(pool)
    rows=np.asarray(_get(pool,'features'),dtype=object)
    result=np.zeros(len(_get(pool,'ids')),dtype=float)
    inside=scope_indices(pool,h,np.arange(len(result)))
    for axis in h['focus']:
        j=axes.index(axis)
        if kinds[axis]=='continuous':
            values=np.asarray(rows[:,j],dtype=float)
            if not np.all(np.isfinite(values)) or np.any(values<0) or np.any(values>1):
                raise ValueError('Continuous feature inputs must be finite normalized values')
            plus,minus=float(h['plus_features'][j]),float(h['minus_features'][j])
            if plus==minus:raise ValueError('A continuous feature requires distinct anchor coordinates')
            component=np.clip((values[inside]-minus)/(plus-minus),-1.,1.)
        else:
            if not all(isinstance(x,str) for x in rows[:,j]):
                raise ValueError('Categorical features require true string labels')
            component=(rows[inside,j]==h['categorical_preference'][axis]['plus']).astype(float)
        result[inside]+=component/len(h['focus'])
    if result[_get(pool,'ix')[h['plus_id']]]!=1. or result[_get(pool,'ix')[h['minus_id']]]!=0.:
        raise ValueError('Numerical feature must be one at plus and zero at reference')
    return result.tolist()


def semantic_key(h):
    """Anchors, quantitative declaration and actual feature all identify a version."""
    fields=('plus_id','minus_id','plus_features','minus_features','focus','scope',
            'categorical_preference','numeric_prior','reference_y','feature_definition','feature_vector')
    return _digest({key:h[key] for key in fields})


def validate_hypothesis(pool,h):
    if h['semantic_key']!=semantic_key(h) or h['family_key']!=family_key(h):
        raise ValueError('Frozen numeric hypothesis semantics were rewritten')
    if h['feature_vector']!=feature_vector(pool,h):
        raise ValueError('Frozen input feature differs from its physical definition')
    if h['birth_prefix_sha256']!=_digest(h['birth_prefix']):
        raise ValueError('Birth prefix was rewritten')


def _source_rule(h):
    return {key:deepcopy(h[key]) for key in ('focus','scope','categorical_preference',
            'plus_id','minus_id','plus_features','minus_features','family_key')}


def _applicable(pool,new,row):
    source=row['source_rule']
    if family_key(source)!=family_key(new):return False
    ids=[row['point_id'],row['reference_id']]
    if any(key not in _get(pool,'ix') for key in ids):return False
    indices=[_get(pool,'ix')[key] for key in ids]
    # Intentionally no V5 make_probes/_compatible_mask: a declared background
    # transfer must retain adverse evidence across those backgrounds.
    return len(scope_indices(pool,new,indices))==2


def _inherit_negative(pool,sources,new):
    records={}
    def add(row):
        key=row['event_id'];candidate=deepcopy(row)
        for name in GAIN_FIELDS:
            value=float(candidate.get(name,0.))
            if not isfinite(value):raise ValueError('Inherited scores must be finite')
            candidate[name]=min(0.,value)
        if not any(candidate[name]<0 for name in GAIN_FIELDS):return
        if key in records:
            previous=records[key]
            if any(previous[name]!=candidate[name] for name in ('point_id','reference_id','query_round')):
                raise ValueError('One inherited event ID refers to different experiments')
            if family_key(previous['source_rule'])!=family_key(candidate['source_rule']):
                raise ValueError('One inherited event ID changed structural direction family')
            for name in GAIN_FIELDS:previous[name]=min(previous[name],candidate[name])
        else:records[key]=candidate
    for old in sources:
        for row in old.get('inherited_negative_evidence',[]):add(row)
        for event in old['ledger']['events']:
            gains={name:float(event.get(name,0.)) for name in GAIN_FIELDS}
            if not any(value<0 for value in gains.values()):continue
            add(dict(event_id=event['event_id'],source_id=old['id'],source_version=old['version'],
                source_rule=_source_rule(old),family_key=family_key(old),
                point_id=event['point_id'],reference_id=event.get('reference_id',old['minus_id']),
                query_round=event['query_round'],source_gains=gains,**gains))
    for row in records.values():
        row['applicable']=_applicable(pool,new,row)
        row['interpretation']='inherited safety constraint, not a new likelihood or positive credit'
    return sorted(records.values(),key=lambda row:row['event_id'])


def review_and_apply(pool,hypotheses,proposal,menu_ids,t,history):
    """Review bounded actions, retaining immutable prior versions and failures."""
    if type(t) is not int or t<1:raise ValueError('Creation round must be positive')
    if (not isinstance(proposal,dict) or set(proposal)!={'actions','summary'} or
            not isinstance(proposal['summary'],str) or not 1<=len(proposal['summary'].strip())<=1600 or
            not isinstance(proposal['actions'],list) or len(proposal['actions'])>MAX_ACTIONS):
        raise ValueError('Proposal requires summary and at most three actions')
    observed=geometry._history(pool,history)
    menu_ids=list(menu_ids)
    if len(set(menu_ids))!=len(menu_ids) or set(menu_ids)-set(_get(pool,'ids')) or set(menu_ids)&set(observed):
        raise ValueError('Menu must contain unique real unqueried IDs')
    for h in hypotheses:validate_hypothesis(pool,h)
    result,events,touched=deepcopy(hypotheses),[],set()
    for position,action in enumerate(proposal['actions']):
        event=dict(round=t,action_index=position,accepted=False,created_expert=False)
        try:
            if not isinstance(action,dict) or set(action)!=ACTION_FIELDS:
                raise ValueError('Action requires the exact numeric-prior schema')
            op,hid=action['op'],action['id'];event.update(op=op,id=hid)
            if op not in ('add','retain','revise','retract'):raise ValueError('Unknown operation')
            if not isinstance(action['reason'],str) or not 1<=len(action['reason'].strip())<=1000:
                raise ValueError('A concise scientific reason is required')
            prior=validate_numeric_prior(action['numeric_prior'])
            matches=[h for h in result if h['id']==hid]
            parent=max(matches,key=lambda h:h['version']) if matches else None
            if op=='add' and hid is not None:raise ValueError('Add requires id=null')
            if op!='add' and (not isinstance(hid,str) or parent is None):raise ValueError('Name an existing hypothesis')
            if hid is not None and hid in touched:raise ValueError('At most one action per existing hypothesis')
            if parent and any(e['query_round']>=t for e in parent['ledger']['events']):
                raise ValueError('Operation must precede current-round feedback')
            if op in ('retain','retract'):
                if action['plus_id'] is not None or action['minus_id'] is not None or action['focus']!={} or action['background']!={} or prior is not None:
                    raise ValueError('Retain/retract must have null anchors/prior and empty geometry')
                if parent['status']=='retracted':raise ValueError('Retracted hypotheses require explicit revision')
                if op=='retract':
                    parent.update(status='retracted',retired_round=t,retirement_reason=action['reason'])
                    parent['ledger']['weight']=0.
                event.update(accepted=True,outcome=op,version=parent['version']);touched.add(hid)
                events.append(event);continue
            contract=geometry._build_contract(pool,action,set(menu_ids),observed)
            if prior is None:
                event.update(accepted=True,outcome='numeric_abstention',reason='No numerical prior; no expert created or replaced')
                if hid is not None:touched.add(hid)
                events.append(event);continue
            scale=float(_get(pool,'scale'))
            if not isfinite(scale) or scale<=0:raise ValueError('Objective scale must be positive and finite')
            prior.update(delta_mean_y=prior['delta_mean_raw']/scale,delta_sd_y=prior['delta_sd_raw']/scale)
            if not all(isfinite(x) and x>0 for x in prior.values()) or not isfinite(prior['delta_sd_y']**2) or prior['delta_sd_y']**2==0:
                raise ValueError('Numerical prior cannot be represented in GP target units')
            contract.update(numeric_prior=prior,reference_y=float(observed[contract['minus_id']]),
                feature_definition='scope-gated equal mean: continuous clipped anchor contrast [-1,1]; categorical plus-label indicator')
            contract['feature_vector']=feature_vector(pool,contract)
            semantic=semantic_key(contract)
            equal=[h for h in result if h['semantic_key']==semantic]
            existing=max(equal,key=lambda h:(h['created_round'],h['version'])) if equal else None
            if existing is not None and not (op=='revise' and existing['status']=='retracted'):
                event.update(accepted=True,outcome='duplicate_semantics_preserved',id=existing['id'],
                             version=existing['version'],status=existing['status'])
                if hid is not None:touched.add(hid)
                events.append(event);continue
            live=sum(h['status']!='retracted' for h in result)
            if live-int(parent is not None and parent['status']!='retracted')>=MAX_LIVE_CLAIMS:
                raise ValueError('At most four live numeric hypotheses, including dormant ones')
            if parent is None:
                number=1;used={h['id'] for h in result}
                while f'H{number:03d}' in used:number+=1
                hid=f'H{number:03d}'
            new=dict(contract,id=hid,version=parent['version']+1 if parent else 1,
                parent={'id':parent['id'],'version':parent['version']} if parent else None,
                created_round=t,status='active',reason=action['reason'],semantic_key=semantic,
                family_key=family_key(contract),birth=None,birth_prefix=deepcopy(history),
                birth_prefix_sha256=_digest(history),ledger=initial_ledger(),
                excluded_observed_ids=sorted(observed),inherited_negative_evidence=[])
            new['inherited_negative_evidence']=_inherit_negative(pool,result,new)
            if parent:
                parent.update(status='retracted',retired_round=t,retirement_reason='superseded by explicit numeric revision')
                parent['ledger']['weight']=0.
            result.append(new);touched.add(hid)
            event.update(accepted=True,created_expert=True,outcome=op,id=hid,version=new['version'],
                         inherited_negative_count=len(new['inherited_negative_evidence']))
        except (ValueError,TypeError,KeyError,OverflowError) as error:event['reason']=str(error)
        events.append(event)
    return result,events
