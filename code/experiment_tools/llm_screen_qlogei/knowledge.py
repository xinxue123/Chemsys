"""Prospective scoped yield forecasts and separate recommendation permissions."""
from copy import deepcopy
import math
import numpy as np
from scipy.special import ndtr
from reasoning_memory_v13 import calibration
from . import PARAMETERS as P


def scope_mask(pool,scope):
    mask=np.ones(len(pool.ids),dtype=bool)
    for axis,condition in scope.items():
        if pool.kinds[pool.axes.index(axis)]=='categorical':
            mask &= np.asarray([point[axis] in condition for point in pool.points])
        else:
            lo,hi=condition
            mask &= np.asarray([lo<=point[axis]<=hi for point in pool.points])
    return mask


def validate(raw,pool):
    if not isinstance(raw,dict) or set(raw)!={'hypotheses','summary'}:
        raise ValueError('Exactly hypotheses and summary required')
    notes=raw['hypotheses']
    if not isinstance(raw['summary'],str) or len(raw['summary'])>1800:
        raise ValueError('Concise public summary required')
    if not isinstance(notes,list) or len(notes)>P['maximum_notes']:
        raise ValueError('At most four hypotheses; empty is a valid abstention')
    ids=[]
    for note in notes:
        if not isinstance(note,dict) or set(note)!={'id','claim','direction','scope','yield_mean','yield_sd'}:
            raise ValueError('Each hypothesis needs id,claim,direction,scope,yield_mean,yield_sd')
        if not isinstance(note['id'],str) or not 1<=len(note['id'])<=32 or note['id'] in ids:
            raise ValueError('Distinct short hypothesis IDs required')
        ids.append(note['id'])
        if not isinstance(note['claim'],str) or not 1<=len(note['claim'])<=1200:
            raise ValueError('A concise scientific claim is required')
        if note['direction'] not in ('favour','avoid'):
            raise ValueError('direction must be favour or avoid')
        scope=note['scope']
        if not isinstance(scope,dict) or not 1<=len(scope)<=len(pool.axes) or set(scope)-set(pool.axes):
            raise ValueError('A nonempty scope on real input axes is required')
        for axis,condition in scope.items():
            kind=pool.kinds[pool.axes.index(axis)]
            if not isinstance(condition,list) or not condition:
                raise ValueError('Scope conditions must be nonempty lists')
            domain=[point[axis] for point in pool.points]
            if kind=='categorical':
                if any(not isinstance(v,str) for v in condition) or len(set(condition))!=len(condition) or set(condition)-set(domain):
                    raise ValueError('Exact supplied category strings required')
            elif (len(condition)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) for v in condition)
                  or not min(domain)<=condition[0]<condition[1]<=max(domain)):
                raise ValueError('Continuous scope must lie inside the supplied physical domain')
        if int(scope_mask(pool,scope).sum())<P['minimum_scope_candidates']:
            raise ValueError('Scope must cover at least four actual pool conditions')
        if (type(note['yield_mean']) not in (int,float) or not math.isfinite(note['yield_mean'])
            or not 0<=note['yield_mean']<=100 or type(note['yield_sd']) not in (int,float)
            or not math.isfinite(note['yield_sd']) or not P['minimum_yield_sd']<=note['yield_sd']<=P['maximum_yield_sd']):
            raise ValueError('Yield mean in [0,100] and SD in [5,50] percentage points required')
    return deepcopy(notes)


def install(raw,pool,old,query):
    drafts=validate(raw,pool)
    output=[]
    for draft in drafts:
        same=[n for n in old if n['direction']==draft['direction'] and n['scope']==draft['scope']]
        if same:
            parent=min(same,key=lambda n:n['KF']['reliability'])
            note=deepcopy(parent)
            changed=any(note[k]!=draft[k] for k in draft)
            note.update(draft)
            note['version']+=int(changed)
            note['evidence_inherited_from']=dict(id=parent['id'],version=parent['version'])
        else:
            note=dict(**draft,version=1,KF=calibration.initial(),permission='provisional',
                negative_streak=0,deployed_failures=0,feedback=[],evidence_inherited_from=None)
        note['installed_query']=query
        output.append(note)
    return output


def forecast(note,incumbent):
    return float(ndtr((note['yield_mean']/100-incumbent)/(note['yield_sd']/100)))


def feedback(notes,forecasts,observation,query):
    after=deepcopy(notes)
    events=[]
    for pre in forecasts:
        note=next(n for n in after if (n['id'],n['version'])==(pre['id'],pre['version']))
        updated,event=calibration.update(note['KF'],pre['p_LLM'],pre['p_GP'],
            bool(observation['y']>pre['incumbent_y']))
        note['KF']=updated
        note['negative_streak']=note['negative_streak']+1 if event['Brier_gain']<-1e-12 else 0
        improved=observation['y']>pre['incumbent_y']
        if pre['promoted_selection']:
            note['deployed_failures']=0 if improved else note['deployed_failures']+1
        reasons=[]
        if updated['reliability']<P['minimum_reliability']:
            reasons.append('low_relative_predictive_skill')
        if note['negative_streak']>=P['negative_skill_streak']:
            reasons.append('consecutive_negative_relative_skill')
        if note['direction']=='favour' and note['deployed_failures']>=P['deployed_nonimprovement_pause']:
            reasons.append('repeated_deployed_nonimprovement')
        before=note['permission']
        if reasons:
            note['permission']='paused'
        elif before=='provisional':
            note['permission']='active'
        # A predictive-skill increase never restores a paused recommendation.
        evidence=dict(query=query,candidate_id=observation['candidate_id'],
            actual_yield=observation['raw_value'],improved=bool(improved),
            p_LLM=pre['p_LLM'],p_GP=pre['p_GP'],promoted_selection=pre['promoted_selection'],
            permission_before=before,permission_after=note['permission'],reasons=reasons,**event)
        note['feedback'].append(evidence)
        events.append(dict(id=note['id'],version=note['version'],**evidence))
    return after,events


def refresh_event(state,query):
    if len(state['refresh_queries'])>=P['maximum_refreshes']:
        return dict(allowed=False,reasons=['refresh_budget_exhausted'])
    if query<3 or not state['notes']:
        return dict(allowed=False,reasons=['no_scored_knowledge'])
    if state['refresh_queries'] and query-state['refresh_queries'][-1]<P['refresh_minimum_gap']:
        return dict(allowed=False,reasons=['minimum_refresh_gap'])
    bad=[]
    last_edit=max([2]+state['refresh_queries'])
    for note in state['notes']:
        recent=[f for f in note['feedback'] if f['query']>=last_edit]
        if recent and note['permission']=='paused':
            bad.append(dict(id=note['id'],version=note['version'],reasons=recent[-1]['reasons']))
    return dict(allowed=bool(bad),reasons=bad or ['no_degraded_scored_knowledge'])
