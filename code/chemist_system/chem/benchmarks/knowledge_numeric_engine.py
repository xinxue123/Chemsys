"""V8: frozen-birth numeric hypotheses and bounded prospective decisions."""
from copy import deepcopy
from math import isfinite, log

import numpy as np
from scipy.special import logsumexp

from . import knowledge_engine as schedule
from . import knowledge_numeric_policy as policy
from . import knowledge_numeric_evidence as evidence
from . import knowledge_numeric_forecast as forecast
from .knowledge_llm import digest

PARAMETERS = dict(schedule.PARAMETERS,
    mechanism='frozen_birth_numeric_v8',minimum_remaining_scope_candidates=2,
    numeric_prior='LLM full expected contrast and uncertainty minus birth GP contrast',
    feature='equal-weight bounded anchored continuous contrasts and real-label plus indicators',
    confidence='four correlated prospective utilities, increment-limited min positive tanh',
    information='birth-model scalar-coefficient mutual information; not incremental GP truth',
    EI_ratio='analytic Gaussian expert/current ratio times native MC512 acquisition',
    EI_cap='proper density mixture to [1/4,4], same tempering for feedback',
    calibration_control='same prior uncertainty, beta remains prior; birth-null and residual feature still condition')

choose=schedule.choose
should_call=schedule.should_call


def warm():
    from .knowledge_birth_effect import warm as birth_warm
    return birth_warm()


def _key(h):return h['id'],h['version']


def _gaussian(distribution,position):
    return {name:float(distribution[name][position]) for name in ('mean','latent_variance','observation_variance')}


def coverage(pool,gp,context,hypotheses,history):
    rows=[];calculations={};eligible=np.flatnonzero(context['eligible'])
    base=context['base_logei'];stats=context['stats'];bo=int(context['bo_index'])
    for h in hypotheses:
        if h['status']=='retracted':continue
        policy.validate_hypothesis(pool,h)
        js=policy.scope_indices(pool,h,eligible)
        latent_credit=evidence.evidence_summary(h)['trusted_weight']
        row=dict(id=h['id'],version=h['version'],eligible_inside_scope=len(js),
            executable_probes=len(js),weight=latent_credit if len(js) else 0.,
            can_change_rank_at_cap=False,eligibility_state='no_budgeted_scope' if not len(js) else 'numeric_scope')
        if len(js):
            # BO may be outside scope yet informative about beta through the
            # birth joint model. Do not replace its information with zero.
            query=np.asarray(sorted(set(map(int,js))|{bo}),dtype=int)
            pred=gp.birth_store.predict(h,history,query,calibrate=gp.numeric_calibrate)
            values=forecast.vector_forecast(np.asarray(stats['mu_y'])[query],
                np.square(np.asarray(stats['sd_y'])[query]),np.square(np.asarray(stats['sd_predictive_y'])[query]),
                pred['mean'],pred['latent_variance'],pred['observation_variance'],max(r['y'] for r in history))
            positions={int(j):p for p,j in enumerate(query)}
            info=forecast.calibration_information(pred['beta_variance'],pred['residual_feature'],
                pred['null']['latent_variance'],h['birth']['noise_variance'])
            offsets=np.zeros_like(base)
            offsets[js]=np.asarray(values['offset'])[[positions[int(j)] for j in js]]
            cap_correction=np.logaddexp(log(.5),log(.5)+offsets)
            cap_correction[np.abs(offsets)<=PARAMETERS['numerical_noop_atol']]=0.
            cap_winner=choose(base+cap_correction,context)
            row['can_change_rank_at_cap']=bool(cap_winner!=bo and cap_winner in set(map(int,js))
                and abs(offsets[cap_winner])>PARAMETERS['numerical_noop_atol'])
            row['positive_effect_probes']=int(np.count_nonzero(offsets[js]>PARAMETERS['numerical_noop_atol']))
            row['negative_effect_probes']=int(np.count_nonzero(offsets[js]<-PARAMETERS['numerical_noop_atol']))
            row['eligibility_state']='numeric_rank_effect' if row['can_change_rank_at_cap'] else 'numeric_without_rank_effect'
            calculations[_key(h)]=dict(indices=js,query_indices=query,positions=positions,
                prediction=pred,values=values,information=np.asarray(info),offsets=offsets)
        rows.append(row)
    return rows,calculations


def _refresh(pool,gp,state,context,t):
    prepared=deepcopy(state)
    rows,calc=coverage(pool,gp,context,prepared['hypotheses'],prepared['history'])
    bykey={_key(row):row for row in rows};events=[]
    for h in prepared['hypotheses']:
        if h['status']=='retracted':continue
        row=bykey[_key(h)];status='active' if row['eligible_inside_scope'] else 'dormant'
        if h['status']!=status:
            events.append(dict(round=t,id=h['id'],version=h['version'],from_status=h['status'],
                to_status=status,reason=row['eligibility_state']))
        h['status']=status;h['coverage']=dict(round=t,eligible_inside_scope=row['eligible_inside_scope'])
        h['ledger']['weight']=evidence.weight(h)
    return prepared,events,rows,calc


def prepare_state(pool,gp,state,context,t):
    prepared,events,_,_=_refresh(pool,gp,state,context,t)
    return prepared,events


def _candidate_record(h,calc):
    pred=calc['prediction'];values=calc['values']
    return dict(id=h['id'],version=h['version'],query_indices=calc['query_indices'].tolist(),
        scope_indices=calc['indices'].tolist(),
        gaussians={key:{name:np.asarray(pred[key][name]).tolist()
            for name in ('mean','latent_variance','observation_variance')}
            for key in ('full','null','cut_feedback','variance_only')},
        actual={name:np.asarray(pred[name]).tolist() for name in ('mean','latent_variance','observation_variance')},
        beta_mean=float(pred['beta_mean']),beta_variance=float(pred['beta_variance']),
        residual_feature=np.asarray(pred['residual_feature']).tolist(),
        information=calc['information'].tolist(),offset=np.asarray(values['offset']).tolist(),
        tempering=np.asarray(values['tempering']).tolist(),
        log_tempering_base=np.asarray(values['log_tempering_base']).tolist(),
        log_tempering_remainder=[None if np.isneginf(x) else float(x) for x in np.asarray(values['log_tempering_remainder'])],
        log_tempering_null_encoding='null denotes negative infinity, the explicit no-op',
        zero_effect_policy=np.asarray(values['zero_effect_policy']).tolist(),
        block_receipt=deepcopy(pred['block_receipt']))


def plan(pool,gp,state,t,context,proposal=None,menu_ids=(),mode='full'):
    if mode not in ('bo','full','verification','no_calibration'):raise ValueError('Unregistered numeric arm')
    if gp.numeric_calibrate!=(mode!='no_calibration'):raise ValueError('Adapter calibration arm mismatch')
    working=deepcopy(state);reviews=[]
    if proposal is not None:
        working['hypotheses'],reviews=policy.review_and_apply(pool,working['hypotheses'],proposal,menu_ids,t,working['history'])
    for h in working['hypotheses']:
        if h['status']!='retracted' and h['birth'] is None:
            if h['created_round']!=t or digest(h['birth_prefix'])!=digest(working['history']):
                raise ValueError('A numeric expert must be born on its declared current prefix')
            h['birth']=gp.birth_store.birth(h,working['history'],context)
    working,updates,rows,calc=_refresh(pool,gp,working,context,t);reviews.extend(updates)
    hypotheses=working['hypotheses'];base=context['base_logei'];bo=int(context['bo_index'])
    combined=np.array(base,copy=True);selected=bo;selection='base_bo';deployed=[];fallback=None
    if mode in ('full','no_calibration'):
        active=[row for row in rows if row['weight']>0 and _key(row) in calc]
        total=sum(row['weight'] for row in active)
        scale=min(1.,PARAMETERS['maximum_deployment_weight']/total) if total else 1.
        deployed=[dict(id=row['id'],version=row['version'],weight=row['weight'],effective_mixture_weight=row['weight']*scale) for row in active]
        if deployed:
            weights=[row['effective_mixture_weight'] for row in deployed]
            offsets=np.stack([calc[_key(row)]['offsets'] for row in deployed])
            correction=logsumexp(np.vstack([np.full_like(base,log(1-sum(weights)))]+
                [log(w)+o for w,o in zip(weights,offsets)]),axis=0)
            correction[np.all(np.abs(offsets)<=PARAMETERS['numerical_noop_atol'],axis=0)]=0.
            combined+=correction;selected=choose(combined,context)
            if selected!=bo:
                supported=any(selected in calc[_key(row)]['indices'] and
                    abs(calc[_key(row)]['offsets'][selected])>PARAMETERS['numerical_noop_atol'] for row in deployed)
                if supported:selection='knowledge_guided'
                else:fallback='uncovered_or_no_effect_weighted_winner';selected=bo
    verification=None
    if mode!='bo' and selected==bo and t<=PARAMETERS['last_verification_round'] and state['verification_used']<PARAMETERS['maximum_optional_verifications']:
        candidates=[]
        for row in rows:
            key=_key(row)
            if row['weight']>0 or not row['can_change_rank_at_cap'] or row['eligible_inside_scope']<2 or key not in calc:continue
            h=next(h for h in hypotheses if _key(h)==key)
            if any(x<0 for x in evidence.evidence_summary(h)['effective_scores'].values()):continue
            data=calc[key];bo_info=float(data['information'][data['positions'][bo]])
            for j in data['indices']:
                pos=data['positions'][int(j)];info=float(data['information'][pos]);gap=float(base[bo]-base[j])
                if (j!=bo and gap<=min(context['epsilon'],PARAMETERS['verification_maximum_gap'])
                    and info-bo_info>=PARAMETERS['minimum_incremental_information']
                    and abs(data['offsets'][j])>PARAMETERS['numerical_noop_atol']):
                    candidates.append((info-bo_info,-gap,-context['tie_priority'][j],int(j),key,info,bo_info))
        if candidates:
            winner=max(candidates,key=lambda row:row[:3]);selected=winner[3];selection='optional_verification'
            verification=dict(id=winner[4][0],version=winner[4][1],incremental_information=winner[0],
                information=winner[5],BO_information=winner[6],information_kind='birth_beta_mutual_information',
                remaining_budgeted_scope_candidates=len(calc[winner[4]]['indices'])-1)
    shadows=[]
    for h in hypotheses:
        key=_key(h)
        if h['status']=='retracted' or key not in calc or selected not in calc[key]['indices']:continue
        data=calc[key];pred=data['prediction'];pos=data['positions'][selected]
        stats=context['stats'];ref=next(row for row in h['birth_prefix'] if row['candidate_id']==h['minus_id'])
        current=dict(mean=float(stats['mu_y'][selected]),latent_variance=float(stats['sd_y'][selected])**2,
            observation_variance=float(stats['sd_predictive_y'][selected])**2)
        frozen=forecast.freeze_forecast(hypothesis_id=h['id'],version=h['version'],point_id=pool.ids[selected],
            query_round=t,prefix_sha256=digest(state['history']),current_gp_sha256=digest(stats['fit_parameters']),
            birth_snapshot_sha256=h['birth']['snapshot_sha256'],semantic_key=h['semantic_key'],
            current=current,birth_null=_gaussian(pred['null'],pos),raw_expert=_gaussian(pred,pos),
            cut_feedback=_gaussian(pred['cut_feedback'],pos),variance_only=_gaussian(pred['variance_only'],pos),
            full_expert=_gaussian(pred['full'],pos),best=max(row['y'] for row in state['history']),
            beta_variance=float(pred['beta_variance']),residual_feature=float(pred['residual_feature'][pos]),
            birth_noise_variance=h['birth']['noise_variance'],reference_id=h['minus_id'],reference_y=ref['y'],
            actual_calibrate=gp.numeric_calibrate)
        shadows.append(dict(id=h['id'],version=h['version'],prediction=frozen,
            prediction_sha256=frozen['prediction_sha256'],
            information_nats=float(data['information'][pos]),reference_is_observed=True))
    selected_weights={_key(row):row['effective_mixture_weight'] for row in deployed}
    mixture=[dict(id=sh['id'],version=sh['version'],weight=selected_weights[_key(sh)],prediction_sha256=sh['prediction']['prediction_sha256'])
        for sh in shadows if selected_weights.get(_key(sh),0.)>0]
    if selection=='knowledge_guided' and not mixture:raise ValueError('Weighted winner lacks a frozen mixture predictor')
    return dict(query=t,history_sha256=digest(state['history']),selected_id=pool.ids[selected],BO_reference_id=pool.ids[bo],
        mode=selection,mechanism='frozen_birth_numeric_v8',base_logei=float(base[selected]),reference_logei=float(base[bo]),
        combined_logei=float(combined[selected]),opportunity_gap=float(base[bo]-base[selected]),epsilon=context['epsilon'],
        eligible_count=int(context['eligible'].sum()),coverage=rows,reviews=reviews,hypotheses_at_commit=hypotheses,
        shadows=shadows,proposal=proposal,menu_ids=list(menu_ids),verification=verification,
        GP_fit_key=context['stats']['fit_key'],GP_acquisition_meta=context['acquisition_meta'],
        deployed_versions=deployed,selected_mixture=mixture,fallback_reason=fallback,
        numeric_candidates=[_candidate_record(h,calc[_key(h)]) for h in hypotheses if _key(h) in calc])


def feedback(state,commit,observation):
    if (any(type(observation.get(name)) not in (int,float) or not isfinite(observation[name])
            for name in ('y','raw_value')) or observation.get('query_round',commit['query'])!=commit['query']):
        raise ValueError('Experiment must contain finite objectives for the committed query')
    if (commit['query']!=len(state['decisions'])+1 or commit['selected_id']!=observation['candidate_id']
        or commit['history_sha256']!=digest(state['history'])):raise ValueError('Feedback violates committed prefix/point')
    if observation['candidate_id'] in {row['candidate_id'] for row in state['history']}:raise ValueError('Repeated experiment')
    updated=deepcopy(state);updated['hypotheses']=deepcopy(commit['hypotheses_at_commit']);receipts=[];scores={}
    for sh in commit['shadows']:
        h=next(h for h in updated['hypotheses'] if _key(h)==_key(sh));pred=sh['prediction']
        if h['semantic_key']!=policy.semantic_key(h) or h['family_key']!=policy.family_key(h):
            raise ValueError('Committed hypothesis semantics were changed')
        if h['birth_prefix_sha256']!=digest(h['birth_prefix']):raise ValueError('Birth prefix was changed')
        if pred['prefix_sha256']!=commit['history_sha256']:raise ValueError('Forecast prefix was changed')
        ref=next(row for row in state['history'] if row['candidate_id']==pred['reference_id'])
        if ref['y']!=pred['reference_y']:raise ValueError('Known reference changed')
        receipt=evidence.record_feedback(h,pred,observation,expected_prediction_sha256=sh['prediction_sha256']);receipts.append(receipt)
        scores[pred['prediction_sha256']]=receipt['gp_gain']
    mixture=commit['selected_mixture'];total=sum(row['weight'] for row in mixture)
    gain=float(logsumexp([log(1-total)]+[log(row['weight'])+scores[row['prediction_sha256']] for row in mixture])) if mixture else 0.
    before=max(row['y'] for row in state['history']);updated['history'].append(observation)
    updated['verification_used']+=int(commit['mode']=='optional_verification')
    updated['decisions'].append(dict(query=commit['query'],selected_id=commit['selected_id'],BO_reference_id=commit['BO_reference_id'],
        mode=commit['mode'],evidence=receipts,raw_value=observation['raw_value'],y=observation['y'],
        incumbent_before_y=before,incumbent_after_y=max(before,observation['y']),
        deployed_mixture_log_score_gain=gain,commit_sha256=digest(commit)))
    updated['status']='COMPLETE' if len(updated['decisions'])==16 else 'RUNNING'
    state.clear();state.update(updated)
