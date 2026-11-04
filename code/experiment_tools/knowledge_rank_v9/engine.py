"""Rank-only knowledge decisions and atomic committed-feedback transitions."""
from copy import deepcopy
from hashlib import sha256
from math import isfinite
import numpy as np
from scipy.special import ndtr
from chem.benchmarks.knowledge_llm import digest
from . import kf,policy

PARAMETERS=dict(kf.PARAMETERS,shortlist=5,maximum_live_rules=4,maximum_edit_calls=4,
    minimum_edit_gap=3,late_edit_round=11,maximum_total_weight=.8,
    GP_modified=False,knowledge_scope='rank_only; no V8 opportunity gate')


def gp_hash(context):
    keys=dict(mu=context['stats']['mu_y'],sd=context['stats']['sd_y'],
        observation_sd=context['stats']['sd_predictive_y'],base=context['base_logei'],
        available=context['available'],tie_priority=context['tie_priority'])
    return digest({key:dict(shape=list(np.asarray(value).shape),dtype=str(np.asarray(value).dtype),
        sha256=sha256(np.ascontiguousarray(value).tobytes()).hexdigest()) for key,value in keys.items()})


def candidates(pool,context,t):
    available=np.flatnonzero(context['available']);base=context['base_logei'];tie=context['tie_priority']
    ordered=available[np.lexsort((tie[available],-base[available]))][:5]
    if len(ordered)!=5:raise ValueError('The registered interface requires5 unused candidates')
    # Display order is independent of candidate IDs and does not reveal GP rank.
    physical=sorted(map(int,ordered),key=lambda j:pool.features[j])
    order=np.random.default_rng(np.random.SeedSequence([0,t,991])).permutation(5)
    return ordered,np.asarray(physical,dtype=int)[order]


def edit_event(pool,state,shortlist,t):
    calls=state['edit_calls'];reasons=[]
    if not calls:return dict(allowed=True,reasons=['initial_knowledge'])
    if len(calls)>=4:return dict(allowed=False,reasons=['edit_budget_exhausted'])
    if t-calls[-1]<3:return dict(allowed=False,reasons=['minimum_edit_gap'])
    if t<11 and len(calls)>=3:return dict(allowed=False,reasons=['reserved_late_edit'])
    if t>=11 and not any(q>=11 for q in calls):reasons.append('reserved_late_edit')
    active=[h for h in state['hypotheses'] if kf.deployment_weight(h['kf'])>0]
    if not any(len(policy.scope_indices(pool,h,shortlist)) for h in active):reasons.append('no_active_coverage')
    for i,h in enumerate(active):
        for other in active[i+1:]:
            if (h['focus']==other['focus'] and h['effect']!=other['effect']
                and set(policy.scope_indices(pool,h,shortlist))&set(policy.scope_indices(pool,other,shortlist))):
                reasons.append('opposed_overlapping_rules')
    recent=state['decisions'][-3:]
    if len(recent)==3 and recent[-1]['incumbent_after_y']<=recent[0]['incumbent_before_y']+1e-12:reasons.append('stagnation')
    feedback=[event for d in state['decisions'][-2:] for event in d['KF_feedback']]
    if any(e['Brier_gain']<-.05 for e in feedback):reasons.append('negative_prediction')
    if any(e['reason'] in ('automatic_pause','automatic_retract') for e in feedback):reasons.append('automatic_lifecycle_event')
    return dict(allowed=bool(reasons),reasons=sorted(set(reasons)))


def borda(probabilities):
    p=np.asarray(probabilities,float);n=len(p)
    if n<2:raise ValueError('Ranking requires at least2 candidates')
    return np.asarray([(np.count_nonzero(p<x)+.5*(np.count_nonzero(p==x)-1))/(n-1) for x in p])


def rank_select(pool,context,ordered,hypotheses,probabilities,*,fixed_initial=False):
    base=np.linspace(1.,0.,len(ordered));scores=base.copy();terms=[]
    for h in hypotheses:
        if h['kf']['status']=='retracted':continue
        inside=np.isin(ordered,policy.scope_indices(pool,h,ordered))
        p=np.asarray([probabilities[h['id']][pool.ids[int(j)]] for j in ordered])
        no_effect=bool(np.all(p==p[0]))
        local=base.copy() if no_effect else np.where(inside,borda(p),base)
        weight=(.4 if h['kf']['status']=='active' else 0.) if fixed_initial else kf.deployment_weight(h['kf'])
        if not inside.any():weight=0.
        terms.append(dict(id=h['id'],version=h['version'],weight=weight,inside=inside.tolist(),
            probabilities=p.tolist(),rank_scores=local.tolist(),no_rank_information=no_effect))
    total=sum(row['weight'] for row in terms);scale=min(1.,.8/total) if total else 1.
    for row in terms:
        row['effective_weight']=row['weight']*scale
        scores+=row['effective_weight']*(np.asarray(row['rank_scores'])-base)
    best=np.flatnonzero(scores==scores.max());ties=context['tie_priority'][ordered[best]]
    selected=int(ordered[best[np.argmin(ties)]])
    return selected,dict(base_rank_scores=base.tolist(),combined_rank_scores=scores.tolist(),terms=terms)


def gp_probability(context,j,reference):
    mu=float(context['stats']['mu_y'][j]);sd=float(context['stats']['sd_predictive_y'][j])
    if not isfinite(mu) or not isfinite(sd) or sd<0:raise ValueError('Invalid unchanged GP prediction')
    return float(ndtr((mu-reference)/sd)) if sd>0 else float(mu>reference)


def plan(pool,state,context,t,mode,receipt=None,refresh=None):
    if mode not in ('bo','reasoning_bo','kf_rank'):raise ValueError('Unregistered rank arm')
    original=gp_hash(context);ordered,display=candidates(pool,context,t);bo=int(ordered[0])
    hs=deepcopy(state['hypotheses']);reviews=[];shadows=[];ranking=None;fixed=None
    if mode=='bo':selected=bo
    elif mode=='reasoning_bo':selected=pool.ix[receipt['plan']['selected_id']]
    else:
        hs=deepcopy(receipt['plan']['hypotheses']);reviews=deepcopy(receipt['plan']['reviews'])
        probabilities=receipt['plan']['probabilities']
        selected,ranking=rank_select(pool,context,ordered,hs,probabilities)
        fixed,_=rank_select(pool,context,ordered,hs,probabilities,fixed_initial=True)
        for h in hs:
            policy.validate_existing(h)
            if h['kf']['status']=='retracted' or selected not in policy.scope_indices(pool,h,[selected]):continue
            pred=dict(id=h['id'],version=h['version'],query=t,point_id=pool.ids[selected],
                semantic_sha256=h['semantic_sha256'],reference_id=h['reference_id'],reference_y=h['reference_y'],
                p_LLM=probabilities[h['id']][pool.ids[selected]],p_GP=gp_probability(context,selected,h['reference_y']),
                history_sha256=digest(state['history']),GP_sha256=original)
            ph=digest(pred);shadows.append(dict(prediction=pred,prediction_sha256=ph))
    if selected not in ordered:raise ValueError('Choice escaped the shared shortlist')
    if gp_hash(context)!=original:raise ValueError('Knowledge modified the original GP or acquisition')
    commit=dict(query=t,arm=mode,protocol_sha256=state['protocol_sha256'],history_sha256=digest(state['history']),GP_sha256=original,
        GP_sha256_after_ranking=gp_hash(context),GP_fit_key=context['stats']['fit_key'],
        selected_id=pool.ids[selected],BO_reference_id=pool.ids[bo],
        mode='base_bo' if selected==bo else 'KF_rank' if mode=='kf_rank' else 'ReasoningBO_rank',
        ordered_ids=[pool.ids[int(j)] for j in ordered],display_ids=[pool.ids[int(j)] for j in display],
        raw_qLogEI=[float(context['base_logei'][int(j)]) for j in ordered],
        opportunity_gap=float(context['base_logei'][bo]-context['base_logei'][selected]),
        V8_epsilon=context['epsilon'],outside_V8_gate=not bool(context['eligible'][selected]),
        hypotheses_at_commit=hs,reviews=reviews,predictions=shadows,ranking=ranking,
        fixed_initial_weight_selected_id=pool.ids[fixed] if fixed is not None else None,
        fixed_weight_shadow_contract='same current versions and lifecycle states; replace only active weights by .4; not a complete no-KF trajectory',
        KF_changes_selection_from_fixed_weights=bool(fixed is not None and fixed!=selected),
        refresh=refresh,LLM_context_sha256=receipt['context_sha256'] if receipt else None,
        LLM_raw=receipt['raw'] if receipt else None,
        prestate=deepcopy(state),
        prestate_sha256=digest(state))
    commit['commit_sha256']=digest(commit)
    return commit


def feedback(state,commit,observation):
    if commit['commit_sha256']!=digest({k:v for k,v in commit.items() if k!='commit_sha256'}):raise ValueError('Commit was changed')
    if (commit['query']!=len(state['decisions'])+1 or commit['prestate_sha256']!=digest(state)
        or commit['history_sha256']!=digest(state['history']) or observation['candidate_id']!=commit['selected_id']
        or observation['candidate_id'] in {r['candidate_id'] for r in state['history']}):raise ValueError('Feedback violates frozen state/point')
    if any(type(observation.get(k)) not in (int,float) or not isfinite(observation[k]) for k in ('y','raw_value')):
        raise ValueError('Experiment objectives must be finite real values')
    if commit['GP_sha256']!=commit['GP_sha256_after_ranking']:raise ValueError('GP changed during ranking')
    updated=deepcopy(state);updated['hypotheses']=deepcopy(commit['hypotheses_at_commit']);events=[]
    for sh in commit['predictions']:
        p=sh['prediction'];h=next(h for h in updated['hypotheses'] if (h['id'],h['version'])==(p['id'],p['version']))
        policy.validate_existing(h)
        if (digest(p)!=sh['prediction_sha256'] or p['semantic_sha256']!=h['semantic_sha256']
            or p['history_sha256']!=commit['history_sha256'] or p['GP_sha256']!=commit['GP_sha256']
            or p['point_id']!=observation['candidate_id'] or p['query']!=commit['query']
            or p['query']<h['created_round'] or any(e['point_id']==p['point_id'] or e['query']>=p['query'] for e in h['events'])):
            raise ValueError('Repeated or changed prospective forecast')
        ref=next(row for row in state['history'] if row['candidate_id']==p['reference_id'])
        if ref['y']!=p['reference_y'] or h['reference_y']!=p['reference_y']:raise ValueError('Fixed reference changed')
        after,event=kf.update(h['kf'],p['p_LLM'],p['p_GP'],bool(observation['y']>p['reference_y']),commit['query'])
        event.update(id=h['id'],version=h['version'],point_id=p['point_id'],reference_id=p['reference_id'],
            prediction_sha256=sh['prediction_sha256'],observed_y=observation['y'],
            deployment_weight_before=kf.deployment_weight(h['kf']),deployment_weight_after=kf.deployment_weight(after))
        event['event_id']=digest(event);h['events'].append(event);h['kf']=after;events.append(event)
    before=max(r['y'] for r in state['history']);updated['history'].append(deepcopy(observation))
    if commit['LLM_raw'] is not None:updated['insights'].append(dict(query=commit['query'],insight=commit['LLM_raw']))
    if commit['arm']=='kf_rank' and commit['refresh']['allowed']:updated['edit_calls'].append(commit['query'])
    updated['decisions'].append(dict(query=commit['query'],selected_id=commit['selected_id'],mode=commit['mode'],
        BO_reference_id=commit['BO_reference_id'],raw_value=observation['raw_value'],y=observation['y'],
        incumbent_before_y=before,incumbent_after_y=max(before,observation['y']),KF_feedback=events,
        commit_sha256=commit['commit_sha256'],KF_changes_selection_from_fixed_weights=commit['KF_changes_selection_from_fixed_weights']))
    updated['status']='COMPLETE' if len(updated['decisions'])==16 else 'RUNNING'
    state.clear();state.update(updated)
