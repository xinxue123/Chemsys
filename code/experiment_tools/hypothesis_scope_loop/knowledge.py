"""Explicit input scopes and distinct predictive/recommendation states."""
from copy import deepcopy
import math
import numpy as np
from scipy.special import ndtr
from chemical_hypothesis_bo.hypotheses import digest,canonical,region
from reasoning_memory_v13 import calibration
from . import PARAMETERS as P


class Scopes:
    def __init__(self,pool):self.pool=pool;self.cache={}
    def mask(self,scope):
        key=digest(canonical(scope))
        if key not in self.cache:self.cache[key]=region(self.pool,scope)
        return self.cache[key]
    def semantic_key(self,direction,scope):
        mask=self.mask(scope)
        return digest(dict(direction=direction,pool_task=self.pool.task,
            mask=np.packbits(mask).tolist(),candidate_count=len(mask)))[:24]


def initial_state(draft,key,query):
    return dict(**deepcopy(draft),semantic_key=key,version=1,installed_query=query,
        KF=calibration.initial(),permission='provisional',negative_streak=0,
        absolute_miss_streak=0,deployed_nonimprovements=0,retests_used=0,
        probation_remaining=0,feedback=[],pause_reasons=[],inherited_from=None)


def install(drafts,previous,archive,scopes,query,stagnated):
    notes=[];edits=[];live_keys={n['semantic_key'] for n in previous}
    for draft in drafts:
        key=scopes.semantic_key(draft['direction'],draft['scope'])
        parents=[n for n in previous+archive if n['semantic_key']==key]
        if parents:
            # Preserve the newest evidence-bearing state; duplicate snapshots
            # never reset KF, failure counters or the retest clock.
            parent=max(parents,key=lambda n:(len(n['feedback']),n['version'],n['installed_query']))
            note=deepcopy(parent);note.update(deepcopy(draft))
            changed=any(parent.get(k)!=draft[k] for k in draft)
            note['version']=parent['version']+int(changed)
            note['inherited_from']=dict(id=parent['id'],version=parent['version'],semantic_key=key)
            if key not in live_keys:
                note['permission']='paused';note['pause_reasons']=['previously_withdrawn_same_rule']
        else:note=initial_state(draft,key,query)
        note['installed_query']=query
        if note['permission']=='paused' and draft['review_action']=='retest':
            if stagnated and note['retests_used']<P['maximum_retests_per_semantic_rule']:
                note['permission']='probation';note['probation_remaining']=1;note['retests_used']+=1
                note['pause_reasons']=[]
                edits.append(dict(id=note['id'],action='one_query_retest_after_stagnation',semantic_key=key))
            else:edits.append(dict(id=note['id'],action='retest_not_eligible',semantic_key=key))
        notes.append(note)
    returned={n['semantic_key'] for n in notes}
    for old in previous:
        if old['semantic_key'] not in returned:edits.append(dict(id=old['id'],action='withdraw',semantic_key=old['semantic_key']))
    return notes,edits


def p_improve(note,incumbent_raw):
    if incumbent_raw is None:return None
    f=note['forecast'];return float(ndtr((f['mean_raw']-incumbent_raw)/f['sd_raw']))


def pause_for_recommendation(notes,incumbent_raw):
    after=deepcopy(notes);events=[]
    if incumbent_raw is None:return after,events
    for note in after:
        p=p_improve(note,incumbent_raw)
        if note['permission'] in ('active','provisional','probation') and note['direction']=='favour' and p<P['minimum_positive_probability']:
            before=note['permission'];note['permission']='paused'
            note['pause_reasons']=['forecast_does_not_support_positive_recommendation']
            events.append(dict(id=note['id'],before=before,after='paused',reason=note['pause_reasons'][0],
                p_improve=p,chemical_claim_declared_false=False))
    return after,events


def eligible(note,incumbent_raw):
    if note['permission'] not in ('provisional','active','probation'):return False
    p=p_improve(note,incumbent_raw)
    if p is None:return note['direction']=='favour'
    if note['direction']=='favour' and p<P['minimum_positive_probability']:return False
    if note['direction']=='avoid' and p>P['minimum_positive_probability']:return False
    return note['permission']=='probation' or note['KF']['reliability']>=P['minimum_KF_reliability']


def filter_mask(scopes,notes,available,incumbent_raw,omit_key=None):
    mask=available.copy();active=[]
    for note in notes:
        if note['semantic_key']==omit_key or not eligible(note,incumbent_raw):continue
        region_mask=scopes.mask(note['scope'])
        mask &= region_mask if note['direction']=='favour' else ~region_mask
        active.append(note)
    return mask,active


def freeze_forecasts(pool,scopes,notes,selected,query,incumbent_raw,context,causes):
    result=[]
    for note in notes:
        if not scopes.mask(note['scope'])[selected]:continue
        gp_probability=None
        if context is not None:
            mean=context['stats']['mu_y'][selected]*pool.scale
            sd=max(context['stats']['sd_predictive_y'][selected]*pool.scale,1e-12)
            gp_probability=float(ndtr((mean-incumbent_raw)/sd))
        result.append(dict(id=note['id'],version=note['version'],semantic_key=note['semantic_key'],
            direction=note['direction'],query=query,forecast=deepcopy(note['forecast']),
            incumbent_raw=incumbent_raw,p_LLM=p_improve(note,incumbent_raw),p_GP=gp_probability,
            caused_selection=bool(causes.get(note['semantic_key'],False)),
            permission_before=note['permission'],KF_before=deepcopy(note['KF']),
            probation_used=note['permission']=='probation' and eligible(note,incumbent_raw)))
    return result


def feedback(notes,forecasts,observation,query):
    result=deepcopy(notes);events=[]
    for pre in forecasts:
        note=next(n for n in result if n['semantic_key']==pre['semantic_key'])
        actual=observation['raw_value'];f=pre['forecast']
        z=(actual-f['mean_raw'])/f['sd_raw'];miss=abs(z)>P['forecast_miss_SD']
        note['absolute_miss_streak']=note['absolute_miss_streak']+1 if miss else 0
        improved=None if pre['incumbent_raw'] is None else bool(actual>pre['incumbent_raw']+1e-12)
        kf_event=None
        if pre['p_GP'] is not None:
            note['KF'],kf_event=calibration.update(note['KF'],pre['p_LLM'],pre['p_GP'],improved)
            note['negative_streak']=note['negative_streak']+1 if kf_event['Brier_gain']<-1e-12 else 0
        if pre['caused_selection'] and note['direction']=='favour' and improved is not None:
            note['deployed_nonimprovements']=0 if improved else note['deployed_nonimprovements']+1
        reasons=[]
        if note['KF']['reliability']<P['minimum_KF_reliability']:reasons.append('low_relative_predictive_skill')
        if note['negative_streak']>=P['negative_skill_streak']:reasons.append('two_negative_relative_forecasts')
        if note['absolute_miss_streak']>=P['forecast_miss_streak']:reasons.append('two_forecasts_outside_2SD')
        if note['direction']=='favour' and note['deployed_nonimprovements']>=P['deployed_nonimprovement_pause']:reasons.append('two_guided_nonimprovements')
        before=note['permission']
        if pre['probation_used']:
            note['probation_remaining']=0
            recovered=bool(improved and kf_event is not None and kf_event['Brier_gain']>0 and
                note['KF']['reliability']>=P['minimum_KF_reliability'] and not miss)
            if recovered:
                # Historical evidence stays in feedback/KF. A successful explicit
                # test starts new operational streaks; old streaks cannot make
                # this carefully gated recovery route logically unreachable.
                reasons=[];note['negative_streak']=0;note['absolute_miss_streak']=0
                note['deployed_nonimprovements']=0
                note['permission']='active';note['pause_reasons']=[]
            else:reasons.append('retest_did_not_demonstrate_positive_value')
        if reasons:
            note['permission']='paused';note['pause_reasons']=reasons
        elif before=='provisional':note['permission']='active'
        # Improving predictive calibration alone never restores paused guidance.
        event=dict(query=query,candidate_id=observation['candidate_id'],objective_raw=actual,
            forecast=deepcopy(f),residual_SD=float(z),outside_2SD=bool(miss),improved=improved,
            caused_selection=pre['caused_selection'],p_LLM=pre['p_LLM'],p_GP=pre['p_GP'],KF=kf_event,
            permission_before=before,permission_after=note['permission'],reasons=reasons,
            chemical_claim_declared_false=False)
        note['feedback'].append(event)
        events.append(dict(id=note['id'],semantic_key=note['semantic_key'],**event))
    return result,events
