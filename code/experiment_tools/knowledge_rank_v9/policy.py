"""Immutable knowledge versions, real-input scopes, and negative prior custody."""
from copy import deepcopy
from math import isfinite
import numpy as np
from chem.benchmarks.knowledge_llm import digest
from . import kf

ACTION_KEYS={'op','id','claim','focus','effect','scope','reference_id','reason'}


def scope_indices(pool,h,indices):
    answer=[]
    for j in indices:
        row=pool.features[int(j)];valid=True
        for axis,spec in h['scope'].items():
            x=row[pool.axes.index(axis)]
            if 'allowed' in spec:valid=valid and x in spec['allowed']
            else:valid=valid and spec['bounds'][0]-1e-12<=float(x)<=spec['bounds'][1]+1e-12
        if valid:answer.append(int(j))
    return np.asarray(answer,dtype=int)


def semantics(h):
    return digest({key:h[key] for key in ('claim','focus','effect','scope','reference_id','reference_y')})


def structural(h):return digest({key:h[key] for key in ('focus','effect','scope')})


def validate_existing(h):
    if h['semantic_sha256']!=semantics(h) or h['structure_sha256']!=structural(h):
        raise ValueError('Knowledge version was overwritten')


def next_ids(hypotheses):
    used={h['id'] for h in hypotheses};ids=[];n=1
    while len(ids)<3:
        key=f'H{n:03d}'
        if key not in used:ids.append(key)
        n+=1
    return ids


def normalize_scope(pool,scope):
    if not isinstance(scope,dict) or set(scope)-set(pool.axes):raise ValueError('Unknown scope variable')
    result={}
    for axis in sorted(scope):
        spec=scope[axis];j=pool.axes.index(axis)
        if pool.kinds[j]=='categorical':
            if (not isinstance(spec,dict) or set(spec)!={'allowed'} or not isinstance(spec['allowed'],list)
                or not spec['allowed'] or any(type(x) is not str for x in spec['allowed'])):
                raise ValueError('Category scope requires true string labels')
            labels=set(row[j] for row in pool.features)
            if set(spec['allowed'])-labels or len(set(spec['allowed']))!=len(spec['allowed']):raise ValueError('Invalid category scope')
            result[axis]={'allowed':sorted(spec['allowed'])}
        else:
            if not isinstance(spec,dict) or set(spec)!={'bounds'} or len(spec['bounds'])!=2:raise ValueError('Continuous scope requires bounds')
            lo,hi=spec['bounds']
            if any(type(x) not in (int,float) or not isfinite(x) for x in (lo,hi)) or not 0<=lo<=hi<=1:
                raise ValueError('Continuous bounds must be normalized 0..1')
            result[axis]={'bounds':[float(lo),float(hi)]}
    return result


def inherited_negatives(pool,hs,new):
    rows={}
    for old in hs:
        if old['focus']!=new['focus'] or old['effect']!=new['effect']:continue
        events=list(old.get('inherited_negative',[]))+list(old['events'])
        for event in events:
            if event['Brier_gain']>=-kf.PARAMETERS['atol']:continue
            if len(scope_indices(pool,new,[pool.ix[event['point_id']],pool.ix[event['reference_id']]]))!=2:continue
            key=(event['query'],event['point_id'])
            if key not in rows or event['Brier_gain']<rows[key]['Brier_gain']:rows[key]=deepcopy(event)
    return [rows[key] for key in sorted(rows)]


def apply(pool,hs,actions,history,shortlist,t,allow_refresh):
    if not isinstance(actions,list) or len(actions)>3:raise ValueError('At most three knowledge actions')
    if actions and not allow_refresh:raise ValueError('Knowledge edits require an event')
    result=deepcopy(hs);reviews=[];touched=set();allowed_new=set(next_ids(hs))
    known={row['candidate_id']:row for row in history}
    for h in result:validate_existing(h)
    for pos,a in enumerate(actions):
        review=dict(action=pos,accepted=False)
        try:
            if not isinstance(a,dict) or set(a)!=ACTION_KEYS:raise ValueError('Incorrect knowledge action schema')
            op,hid=a['op'],a['id'];review.update(op=op,id=hid)
            if op not in ('add','retain','revise','retract') or type(hid) is not str or hid in touched:raise ValueError('Invalid or repeated action')
            if type(a['reason']) is not str or not 1<=len(a['reason'])<=800:raise ValueError('Reason required')
            versions=[h for h in result if h['id']==hid];old=max(versions,key=lambda h:h['version']) if versions else None
            if op=='add' and (old is not None or hid not in allowed_new):raise ValueError('Use a supplied unused ID')
            if op!='add' and old is None:raise ValueError('Unknown existing knowledge')
            if op in ('retain','retract'):
                if any(a[k] is not None for k in ('claim','effect','reference_id')) or a['focus']!=[] or a['scope']!={}:
                    raise ValueError('Retain/retract cannot rewrite a claim')
                if old['kf']['status']=='retracted':raise ValueError('Retired versions require explicit revision')
                if op=='retract':
                    old['kf'].update(status='retracted',weight=0.,retracted_round=t)
                    old['retirement_reason']='LLM: '+a['reason']
                touched.add(hid);review.update(accepted=True,version=old['version'],outcome=op)
                reviews.append(review);continue
            if (type(a['claim']) is not str or not 1<=len(a['claim'])<=900 or a['effect'] not in ('promote','avoid')
                or not isinstance(a['focus'],list) or not 1<=len(a['focus'])<=2 or len(set(a['focus']))!=len(a['focus'])
                or set(a['focus'])-set(pool.axes) or a['reference_id'] not in known):raise ValueError('Invalid local knowledge declaration')
            new=dict(id=hid,version=old['version']+1 if old else 1,claim=a['claim'],focus=sorted(a['focus']),
                effect=a['effect'],scope=normalize_scope(pool,a['scope']),reference_id=a['reference_id'],
                reference_y=float(known[a['reference_id']]['y']),created_round=t,events=[],inherited_negative=[],
                birth_history_sha256=digest(history),kf=kf.initial(),reason=a['reason'])
            if len(scope_indices(pool,new,shortlist))<2:raise ValueError('Knowledge must cover at least two current candidates')
            new['semantic_sha256']=semantics(new);new['structure_sha256']=structural(new)
            duplicates=[h for h in result if structural(h)==structural(new)]
            if any(h is not old for h in duplicates):raise ValueError('Structural duplicate: explicitly revise its existing ID')
            if old is not None and old['semantic_sha256']==new['semantic_sha256']:
                review.update(accepted=True,outcome='unchanged_version_preserved',version=old['version']);touched.add(hid)
                reviews.append(review);continue
            live=sum(h['kf']['status']!='retracted' for h in result)
            if live-int(old is not None and old['kf']['status']!='retracted')>=4:raise ValueError('At most four live rules')
            negatives=inherited_negatives(pool,result,new);new['inherited_negative']=negatives
            for event in negatives:
                new['kf'],_=kf.update(new['kf'],event['p_LLM'],event['p_GP'],event['success'],event['query'],prior_transport=True)
            if negatives:new['kf'].update(status='paused',paused_round=t,good_streak=0,local_updates=0)
            if old is not None:
                old['kf'].update(status='retracted',weight=0.,retracted_round=t)
                old['retirement_reason']='superseded by explicit revision'
            result.append(new);touched.add(hid)
            review.update(accepted=True,outcome=op,version=new['version'],inherited_negative_count=len(negatives))
        except (ValueError,TypeError,KeyError) as error:review['reason']=str(error)
        reviews.append(review)
    return result,reviews
