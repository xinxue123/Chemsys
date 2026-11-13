"""Validate and compile only supplied physical input scopes."""
from copy import deepcopy
import hashlib
import json
import numpy as np
from . import PARAMETERS as P


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def region(pool,scope):
    if not isinstance(scope,dict) or not set(scope)<=set(pool.axes):raise ValueError('Unknown physical scope axis')
    mask=np.ones(len(pool.ids),bool)
    for name,spec in scope.items():
        axis=pool.axes.index(name)
        if not isinstance(spec,dict):raise ValueError('Each scope axis requires a dictionary')
        if pool.kinds[axis]=='categorical':
            allowed=spec.get('allowed') if isinstance(spec,dict) else None
            if set(spec)!={'allowed'} or not isinstance(allowed,list) or not allowed or not set(allowed)<={p[name] for p in pool.points}:
                raise ValueError('Category scopes must use exact supplied identities')
            mask &= np.asarray([point[name] in allowed for point in pool.points])
        else:
            bounds=spec.get('bounds') if isinstance(spec,dict) else None
            if set(spec)!={'bounds'} or not isinstance(bounds,list) or len(bounds)!=2 or any(type(v) not in (int,float) or not np.isfinite(v) for v in bounds) or bounds[0]>bounds[1]:
                raise ValueError('Continuous scopes require finite physical bounds')
            from chem.benchmarks.knowledge_domain_context import task_context
            limits=next(v['declared_input_bounds'] for v in task_context(pool.task)['inputs'] if v['name']==name)
            if bounds[0]<limits[0]-1e-10 or bounds[1]>limits[1]+1e-10:raise ValueError('Scope outside declared physical input domain')
            mask &= np.asarray([bounds[0]-1e-10<=point[name]<=bounds[1]+1e-10 for point in pool.points])
    return mask


def canonical(scope):
    return {k:dict(allowed=sorted(set(v['allowed']))) if 'allowed' in v else
        dict(bounds=list(map(float,v['bounds']))) for k,v in sorted(scope.items())}


def structural_key(note):
    return digest(dict(preferred=canonical(note['preferred']),comparison=canonical(note['comparison'])))[:20]


def validate(raw,pool,history):
    if not isinstance(raw,dict) or set(raw)!={'summary','hypotheses'} or not isinstance(raw['summary'],str):raise ValueError('Expected summary and hypotheses')
    if not isinstance(raw['hypotheses'],list) or len(raw['hypotheses'])>P['maximum_hypotheses']:raise ValueError('At most3 hypotheses')
    notes=[];inactive=[];seen=set();observed={h['candidate_id'] for h in history}
    fields={'claim','rationale','basis','focus','preferred','comparison','supporting_observation_ids','falsification'}
    for item in raw['hypotheses']:
        if not isinstance(item,dict) or set(item)!=fields:raise ValueError('Invalid hypothesis fields')
        if any(not isinstance(item[k],str) or not item[k].strip() or len(item[k])>1800 for k in ('claim','rationale','falsification')):raise ValueError('Missing concise hypothesis explanation')
        if item['basis'] not in ('mechanistic_prior','observed_association','both'):raise ValueError('Unknown evidence basis')
        focus=item['focus']
        if not isinstance(focus,list) or not 1<=len(focus)<=2 or len(set(focus))!=len(focus) or not set(focus)<=set(pool.axes):raise ValueError('Focus must contain1-2 actual axes')
        support=item['supporting_observation_ids']
        if not isinstance(support,list) or not set(support)<=observed:raise ValueError('Support must refer to current own measurements')
        if item['basis'] in ('observed_association','both') and not support:raise ValueError('Observed association needs own evidence IDs')
        plus=region(pool,item['preferred']);minus=region(pool,item['comparison'])
        preferred,comparison=canonical(item['preferred']),canonical(item['comparison'])
        for axis in pool.axes:
            if axis not in focus and preferred.get(axis)!=comparison.get(axis):raise ValueError('Nonfocus context must agree on both sides')
        note=deepcopy(item);note['id']='chem_'+structural_key(note)
        if note['id'] in seen:raise ValueError('Duplicate structural hypothesis')
        seen.add(note['id']);shared=plus&minus;plus &= ~shared;minus &= ~shared
        note['preferred_count']=int(plus.sum());note['comparison_count']=int(minus.sum())
        if min(plus.sum(),minus.sum())<2:
            inactive.append(dict(hypothesis=note,reason='fewer than2 executable exclusive inputs per contrast side'));continue
        notes.append(note)
    return dict(summary=raw['summary'],notes=notes,inactive=inactive)


def masks(pool,note):
    plus=region(pool,note['preferred']);minus=region(pool,note['comparison']);common=plus&minus
    return plus&~common,minus&~common
