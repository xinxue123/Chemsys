"""Input-only proposals, immutable probe semantics, and full-pool projection."""
from copy import deepcopy
from math import isfinite
import numpy as np
from chem.benchmarks.knowledge_llm import digest
from knowledge_rank_v9 import policy

ACTION_KEYS={'op','id','claim','focus','effect','scope','reason','probe'}
PROBE_KEYS={'target','why_informative','if_above','if_below'}


def target(pool,value):
    if not isinstance(value,dict) or set(value)!=set(pool.axes):raise ValueError('Target must specify every real input axis')
    result={}
    for k,(axis,kind) in enumerate(zip(pool.axes,pool.kinds)):
        x=value[axis]
        if kind=='categorical':
            if type(x) is not str or x not in {r[k] for r in pool.features}:raise ValueError('Use exact categorical labels')
            result[axis]=x
        else:
            if type(x) not in (int,float) or not isfinite(x) or not 0<=x<=1:raise ValueError('Continuous target must be normalized0..1')
            result[axis]=float(x)
    return result


def project(pool,h,available,ties):
    """Exact categorical match, nearest normalized continuous point inside scope."""
    js=policy.scope_indices(pool,h,np.flatnonzero(available))
    if not len(js):return None
    value=h['probe']['target'];distance=np.zeros(len(js))
    valid=np.ones(len(js),dtype=bool);continuous=0
    for k,(axis,kind) in enumerate(zip(pool.axes,pool.kinds)):
        if kind=='categorical':valid&=np.array([pool.features[j][k]==value[axis] for j in js])
        else:
            distance+=np.array([abs(float(pool.features[j][k])-value[axis]) for j in js]);continuous+=1
    if not valid.any():return None
    ds=distance[valid]/max(continuous,1);indices=js[valid]
    best=np.lexsort((ties[indices],ds))[0];j=int(indices[best])
    return dict(index=j,candidate_id=pool.ids[j],projection_distance=float(ds[best]))


def reference(pool,history,focus,probe):
    compatible=[]
    for row in history:
        feature=pool.features[pool.ix[row['candidate_id']]];ok=True
        for k,(axis,kind) in enumerate(zip(pool.axes,pool.kinds)):
            if axis in focus:continue
            ok &= (feature[k]==probe[axis]) if kind=='categorical' else abs(float(feature[k])-probe[axis])<=.25+1e-12
        if ok:compatible.append(row)
    candidates=compatible or history
    chosen=min(candidates,key=lambda r:(-r['y'],pool.features[pool.ix[r['candidate_id']]]))
    return chosen['candidate_id'],'highest_comparable_observed' if compatible else 'highest_observed_no_local_match'


def metadata_hash(h):
    return digest({k:h[k] for k in ('probe','probe_signature','reference_selection')})


def validate(h):
    policy.validate_existing(h)
    if h['probe_sha256']!=metadata_hash(h):raise ValueError('Probe semantics changed without a new version')


def apply(pool,state,actions,t,available,ties):
    if not isinstance(actions,list) or len(actions)>3:raise ValueError('At most3 actions')
    hs=deepcopy(state['hypotheses']);reviews=[];touched=set()
    for h in hs:validate(h)
    for n,a in enumerate(actions):
        review=dict(action=n,accepted=False)
        try:
            if not isinstance(a,dict) or set(a)!=ACTION_KEYS:raise ValueError('Invalid proposal action keys')
            review.update(op=a['op'],id=a['id'])
            if a['id'] in touched:raise ValueError('Repeated rule action');
            if a['op'] in ('add','revise'):
                probe=deepcopy(a['probe'])
                if not isinstance(probe,dict) or set(probe)!=PROBE_KEYS:raise ValueError('Probe requires target and public decision consequences')
                if any(type(probe[k]) is not str or not 1<=len(probe[k])<=700 for k in PROBE_KEYS-{'target'}):raise ValueError('Explain the test and both outcomes')
                probe['target']=target(pool,probe['target'])
                if not isinstance(a['focus'],list) or not 1<=len(a['focus'])<=2 or set(a['focus'])-set(pool.axes):raise ValueError('Focus must use1..2 real axes')
                scope=policy.normalize_scope(pool,a['scope'])
                template=dict(scope=scope,probe=probe)
                mapped=project(pool,template,available,ties)
                if mapped is None:raise ValueError('No unqueried candidate matches categorical target and scope')
                ref,why=reference(pool,state['history'],a['focus'],probe['target'])
                base={k:v for k,v in a.items() if k!='probe'};base['reference_id']=ref
            else:
                if a['probe'] is not None:raise ValueError('Retain/retract must not rewrite the probe')
                base={k:v for k,v in a.items() if k!='probe'};base['reference_id']=None
            old_latest=max([h for h in hs if h['id']==a['id']],key=lambda h:h['version'],default=None)
            if a['op']=='revise' and old_latest is not None:
                same_base=all(base[k]==old_latest[k] for k in ('claim','focus','effect','scope','reference_id'))
                if same_base and probe!=old_latest['probe']:raise ValueError('Probe-only changes require an explicit changed scientific claim')
            updated,rr=policy.apply(pool,hs,[base],state['history'],np.flatnonzero(available),t,True)
            one=rr[0];review.update(one);review['action']=n
            if one['accepted'] and a['op'] in ('add','revise') and one['outcome']!='unchanged_version_preserved':
                h=next(h for h in updated if h['id']==a['id'] and h['version']==one['version'])
                h.update(probe=probe,reference_selection=why,
                    probe_signature=digest(dict(focus=sorted(h['focus']),effect=h['effect'],target=probe['target'])))
                h['probe_sha256']=metadata_hash(h);review['projected_at_birth']=mapped['candidate_id']
            if one['accepted']:hs=updated;touched.add(a['id'])
        except (ValueError,TypeError,KeyError) as error:review['reason']=str(error)
        reviews.append(review)
    return dict(hypotheses=hs,reviews=reviews)


def coverage_menu(pool,history,ties,count=24):
    """Farthest-first Gower input coverage. No GP score, targets, or ID distance."""
    known=[pool.ix[r['candidate_id']] for r in history];minimum=np.full(len(pool.ids),np.inf)
    def incorporate(j):
        distances=np.zeros(len(pool.ids))
        for k,kind in enumerate(pool.kinds):
            if kind=='categorical':distances+=np.array([row[k]!=pool.features[j][k] for row in pool.features])
            else:distances+=np.array([abs(row[k]-pool.features[j][k]) for row in pool.features])
        np.minimum(minimum,distances/len(pool.axes),out=minimum)
    for j in known:incorporate(j)
    used=set(known);chosen=[]
    for _ in range(min(count,len(pool.ids)-len(known))):
        indices=np.array([j for j in range(len(pool.ids)) if j not in used])
        j=int(indices[np.lexsort((ties[indices],-minimum[indices]))[0]])
        chosen.append(j);used.add(j);incorporate(j)
    return chosen
