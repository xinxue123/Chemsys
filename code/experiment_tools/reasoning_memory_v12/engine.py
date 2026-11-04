"""Prospective current-incumbent forecasts and independent category ablation."""
from copy import deepcopy
from math import isfinite
import numpy as np
from chem.benchmarks.knowledge_llm import digest
from knowledge_rank_v9 import engine as previous,kf
from . import policy,llm

PARAMETERS=dict(kf.PARAMETERS,shortlist=5,maximum_live_notes=4,maximum_memory_calls=4,maximum_reasoning_calls=16,
    minimum_edit_gap=3,late_edit_round=11,GP_modified=False,selection='unchanged_official_stable_confidence_sort_n1',
    KF_use='current_incumbent_predictive_calibration_advisory_only',positive_support_minimum_successful_gain_events=2,
    numeric_prior_transport=False,category_maximum_insertions=2,category_last_query=8,category_repeat_length=3,
    category_alternative_max_observations=1,category_replace_native_rank=5)


def category_opportunity(pool,state,ordered):
    """Input and own-prefix only; no category meaning inferred from its label."""
    recent=state['decisions'][-3:];eligible=set();details=[]
    if len(recent)<3:return [],[]
    for k,(axis,kind) in enumerate(zip(pool.axes,pool.kinds)):
        if kind!='categorical':continue
        labels=[pool.features[pool.ix[d['selected_id']]][k] for d in recent]
        if len(set(labels))!=1 or any(pool.features[int(j)][k]!=labels[0] for j in ordered):continue
        counts={label:sum(pool.features[pool.ix[r['candidate_id']]][k]==label for r in state['history']) for label in {f[k] for f in pool.features}}
        alternatives={label for label,count in counts.items() if label!=labels[0] and count<=1}
        if not alternatives:continue
        eligible.update(j for j,row in enumerate(pool.features) if row[k] in alternatives)
        details.append(dict(axis=axis,repeated_level=labels[0],alternative_observation_counts={label:counts[label] for label in sorted(alternatives)}))
    return sorted(eligible),details


def menu(pool,state,context,t,arm):
    original,_=previous.candidates(pool,context,t);ordered=original.copy()
    audit=dict(enabled=arm=='category',inserted=False,details=[],challenger_id=None,replaced_id=None)
    if arm=='category' and t<=8 and state.get('category_menu_insertions',0)<2:
        choices,details=category_opportunity(pool,state,original)
        js=np.array([j for j in choices if context['available'][j] and j not in original],dtype=int)
        if len(js):
            j=int(js[np.lexsort((context['tie_priority'][js],-context['base_logei'][js]))[0]])
            audit.update(inserted=True,details=details,challenger_id=pool.ids[j],replaced_id=pool.ids[int(ordered[-1])])
            ordered[-1]=j
    physical=sorted(map(int,ordered),key=lambda j:pool.features[j])
    display=np.asarray(physical)[np.random.default_rng(np.random.SeedSequence([0,t,991])).permutation(5)]
    return original,ordered,display,audit


def edit_event(pool,state,ordered,t):
    calls=state['edit_calls']
    if not calls:return dict(allowed=True,reasons=['initial_knowledge'])
    if len(calls)>=4:return dict(allowed=False,reasons=['edit_budget_exhausted'])
    if t-calls[-1]<3:return dict(allowed=False,reasons=['minimum_edit_gap'])
    if t<11 and len(calls)>=3:return dict(allowed=False,reasons=['reserved_late_edit'])
    reasons=[];recent=state['decisions'][-3:]
    if len(recent)==3 and recent[-1]['incumbent_after_y']<=recent[0]['incumbent_before_y']+1e-12:reasons.append('stagnation')
    feedback=[e for d in state['decisions'][-2:] for e in d['KF_feedback']]
    if any(e['Brier_gain']<-.05 for e in feedback):reasons.append('negative_prediction')
    if any(e['reason'] in ('automatic_pause','automatic_retract') for e in feedback):reasons.append('automatic_lifecycle_event')
    if any(d['incumbent_after_y']>d['incumbent_before_y']+1e-12 for d in state['decisions'][-2:]):reasons.append('new_incumbent_evidence')
    if category_opportunity(pool,state,ordered)[1]:reasons.append('category_concentration_without_contrasts')
    last=state['decisions'][-2:]
    if len(last)==2 and all(any(r['requested_use']=='support' and r['use']!='support' for r in d.get('memory_references',[])) for d in last):reasons.append('repeated_unsupported_positive_citation')
    active=[h for h in state['hypotheses'] if h['kf']['status']=='active']
    for i,h in enumerate(active):
        for other in active[i+1:]:
            if (h['focus']==other['focus'] and h['effect']!=other['effect']
                and set(policy.scope_indices(pool,h,ordered))&set(policy.scope_indices(pool,other,ordered))):reasons.append('opposed_overlapping_rules')
    # Neither elapsed late budget nor absent scope coverage is itself evidence.
    return dict(allowed=bool(reasons),reasons=sorted(set(reasons)))


def plan(root,pool,state,context,t,memory,reasoning,refresh):
    before=previous.gp_hash(context);original,ordered,display,category=menu(pool,state,context,t,state['arm'])
    hs=deepcopy(memory['plan']['hypotheses'] if memory else state['hypotheses']);result=reasoning['plan']
    selected=pool.ix[result['selected_id']];ref=policy.incumbent(pool,state['history']);predictions=[]
    if selected not in ordered or llm.official_extract(root,reasoning['raw']['insight'])!=pool.ids[selected]:raise ValueError('Official choice escaped supplied menu')
    for h in hs:
        policy.validate_existing(h)
        if h['id'] not in result['forecasts'] or result['forecasts'][h['id']]['version']!=h['version']:continue
        if h['kf']['status']=='retracted' or not len(policy.scope_indices(pool,h,[selected])):raise ValueError('Inapplicable forecast')
        p=dict(id=h['id'],version=h['version'],query=t,point_id=pool.ids[selected],semantic_sha256=h['semantic_sha256'],
            reference_id=ref['candidate_id'],reference_y=float(ref['y']),target='strict_improvement_over_current_incumbent',
            birth_reference_id=h['reference_id'],p_LLM=result['forecasts'][h['id']]['p'],
            p_GP=previous.gp_probability(context,selected,ref['y']),history_sha256=digest(state['history']),GP_sha256=before)
        predictions.append(dict(prediction=p,prediction_sha256=digest(p)))
    if before!=previous.gp_hash(context):raise ValueError('Memory changed GP arrays')
    hypotheses=reasoning['raw']['insight']['hypotheses'];priority={'high':0,'medium':1,'low':2}
    winner=min(range(len(hypotheses)),key=lambda i:priority[hypotheses[i]['confidence'].lower()])
    c=dict(query=t,arm=state['arm'],protocol_sha256=state['protocol_sha256'],history_sha256=digest(state['history']),
        GP_sha256=before,GP_sha256_after_ranking=previous.gp_hash(context),GP_fit_key=context['stats']['fit_key'],
        selected_id=pool.ids[selected],BO_reference_id=pool.ids[int(original[0])],mode='base_bo' if selected==original[0] else 'ReasoningBO_memory',
        original_ordered_ids=[pool.ids[int(j)] for j in original],ordered_ids=[pool.ids[int(j)] for j in ordered],display_ids=[pool.ids[int(j)] for j in display],
        raw_qLogEI=[float(context['base_logei'][j]) for j in ordered],winning_hypothesis_index=winner,
        opportunity_gap=float(context['base_logei'][original[0]]-context['base_logei'][selected]),category=category,
        current_incumbent=dict(candidate_id=ref['candidate_id'],y=ref['y']),hypotheses_at_commit=hs,
        reviews=memory['plan']['reviews'] if memory else [],predictions=predictions,
        memory_context_sha256=memory['context_sha256'] if memory else None,memory_raw=memory['raw'] if memory else None,
        LLM_context_sha256=reasoning['context_sha256'],LLM_raw=deepcopy(reasoning['raw']['insight']),reasoning_raw=deepcopy(reasoning['raw']),
        memory_references=result['memory_references'],forecasts=result['forecasts'],sidecar_warnings=result['warnings'],
        expected_forecasts=result['expected_forecasts'],decision_purpose=result['decision_purpose'],refresh=refresh,
        KF_changes_selection_from_fixed_weights=False,prestate=deepcopy(state),prestate_sha256=digest(state))
    c['commit_sha256']=digest(c);return c


def feedback(state,commit,observation):
    if commit['commit_sha256']!=digest({k:v for k,v in commit.items() if k!='commit_sha256'}):raise ValueError('Commit changed')
    if (commit['query']!=len(state['decisions'])+1 or commit['prestate_sha256']!=digest(state)
        or commit['history_sha256']!=digest(state['history']) or observation['candidate_id']!=commit['selected_id']
        or observation['candidate_id'] in {r['candidate_id'] for r in state['history']}):raise ValueError('Invalid or repeated feedback')
    if any(type(observation.get(k)) not in (int,float) or not isfinite(observation[k]) for k in ('y','raw_value')):raise ValueError('Finite objectives required')
    if commit['GP_sha256']!=commit['GP_sha256_after_ranking']:raise ValueError('GP mutated')
    updated=deepcopy(state);updated['hypotheses']=deepcopy(commit['hypotheses_at_commit']);events=[]
    best=max(r['y'] for r in state['history']);scored=set()
    for sh in commit['predictions']:
        p=sh['prediction'];key=(p['id'],p['version']);h=next(h for h in updated['hypotheses'] if (h['id'],h['version'])==key)
        policy.validate_existing(h)
        if (key in scored or digest(p)!=sh['prediction_sha256'] or p['semantic_sha256']!=h['semantic_sha256']
            or p['history_sha256']!=commit['history_sha256'] or p['GP_sha256']!=commit['GP_sha256']
            or p['point_id']!=observation['candidate_id'] or p['query']!=commit['query'] or p['query']<h['created_round']
            or any(e['query']>=p['query'] or e['point_id']==p['point_id'] for e in h['events'])):raise ValueError('Forecast custody violation')
        ref=next(r for r in state['history'] if r['candidate_id']==p['reference_id'])
        if ref['y']!=p['reference_y'] or p['reference_y']!=best or commit['current_incumbent']!=dict(candidate_id=ref['candidate_id'],y=ref['y']):raise ValueError('Forecast must target prequery current incumbent')
        after,e=kf.update(h['kf'],p['p_LLM'],p['p_GP'],bool(observation['y']>best),commit['query'])
        e.update(id=h['id'],version=h['version'],point_id=p['point_id'],reference_id=p['reference_id'],reference_y=p['reference_y'],
            prediction_sha256=sh['prediction_sha256'],observed_y=observation['y'],
            positive_improvement_evidence=bool(observation['y']>best and e['Brier_gain']>1e-12),
            deployment_weight_before=kf.deployment_weight(h['kf']),deployment_weight_after=kf.deployment_weight(after))
        e['event_id']=digest(e);h['events'].append(e);h['kf']=after;events.append(e);scored.add(key)
    updated['history'].append(deepcopy(observation));updated['insights'].append(dict(query=commit['query'],insight=commit['LLM_raw']))
    if commit['refresh']['allowed']:updated['edit_calls'].append(commit['query'])
    updated['category_menu_insertions']=state.get('category_menu_insertions',0)+int(commit['category']['inserted'])
    updated['category_selected_queries']=state.get('category_selected_queries',0)+int(commit['category']['challenger_id']==commit['selected_id'])
    updated['decisions'].append(dict(query=commit['query'],selected_id=commit['selected_id'],mode=commit['mode'],
        BO_reference_id=commit['BO_reference_id'],raw_value=observation['raw_value'],y=observation['y'],
        incumbent_before_y=best,incumbent_after_y=max(best,observation['y']),KF_feedback=events,
        commit_sha256=commit['commit_sha256'],KF_changes_selection_from_fixed_weights=False,
        memory_references=deepcopy(commit['memory_references']),winning_hypothesis_index=commit['winning_hypothesis_index'],
        sidecar_warnings=commit['sidecar_warnings'],decision_purpose=commit['decision_purpose']))
    updated['status']='COMPLETE' if len(updated['decisions'])==16 else 'RUNNING'
    state.clear();state.update(updated)
