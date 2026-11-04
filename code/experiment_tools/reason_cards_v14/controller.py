"""Pure prospective selection and feedback. No fits, oracle reads or LLM IO."""
from copy import deepcopy
from math import isfinite, log
import numpy as np
from cold_start_v13 import adapter as old_adapter, control as old, policy as old_policy
from knowledge_rank_v9 import engine
from reasoning_memory_v12.llm import validate_insight
from reasoning_memory_v13 import calibration

PARAMETERS=dict(maximum_memory_calls=4,minimum_edit_gap=3,late_edit_query=12,
    minimum_editor_observations=2,maximum_learning_overrides=2,maximum_review_queries=2,
    reliability_prior_count=4,maximum_logEI_loss=log(4),tentative_budget_factor=.5,
    severe_shortfall_SD=1.,continuous_local_radius=.25,categorical_local_radius=1,
    successful_review_tests=2,ordinary_local_failures=2,
    continuous_focus_change=.1,maximum_background_drift=.25)
TEST_USES={'positive_support','tentative_test','review_test'}


def initial_channel():
    return dict(calibration=calibration.initial(),facts=[],learning_queries=[],
                review_queries=[],used_review_events=[],events=[])


def credit_eligibility(pool,note,history,selected):
    if not history:
        return dict(eligible=False,reason='no_observed_reference',changed_axes=[])
    ref=old_policy.incumbent(pool,history)
    a=old_adapter.conditions(pool,ref['candidate_id']);b=old_adapter.conditions(pool,selected)
    changed=[axis for axis in pool.axes if a[axis]!=b[axis]]
    focus_change=any((a[axis]!=b[axis] if kind=='categorical' else abs(a[axis]-b[axis])>=PARAMETERS['continuous_focus_change'])
                     for axis,kind in zip(pool.axes,pool.kinds) if axis in note['focus'])
    background_ok=all((a[axis]==b[axis] if kind=='categorical' else abs(a[axis]-b[axis])<=PARAMETERS['maximum_background_drift'])
                      for axis,kind in zip(pool.axes,pool.kinds) if axis not in note['focus'])
    eligible=focus_change and background_ok
    return dict(eligible=eligible,reason='focus_change_in_checked_local_background' if eligible else 'background_changed_or_focus_unchanged',
                changed_axes=changed,reference_id=ref['candidate_id'])


def effective_forecasts(pool,legacy,notes,history):
    by_key={(n['id'],n['version']):n for n in notes};out=[]
    for f in legacy['forecasts']:
        check=credit_eligibility(pool,by_key[f['id'],f['version']],history,legacy['selected_id'])
        effective=deepcopy(f)
        if f['use'] in TEST_USES and not check['eligible']:
            effective['use']='shadow'
            effective['commit_sha256']=old.digest({k:v for k,v in effective.items() if k!='commit_sha256'})
        out.append(dict(declared=deepcopy(f),effective=effective,credit=check))
    return out


def close_conditions(pool,left,right):
    categorical=sum(left[a]!=right[a] for a,k in zip(pool.axes,pool.kinds) if k=='categorical')
    continuous=[abs(left[a]-right[a]) for a,k in zip(pool.axes,pool.kinds) if k=='continuous']
    return categorical<=PARAMETERS['categorical_local_radius'] and all(x<=PARAMETERS['continuous_local_radius'] for x in continuous)


def material_change(pool,fact,point,focus):
    return any((fact[a]!=point[a] if k=='categorical' else abs(fact[a]-point[a])>=PARAMETERS['continuous_local_radius'])
               for a,k in zip(pool.axes,pool.kinds) if a in focus)


def risk_context(pool,state,candidate,query,focus):
    physical=old_adapter.conditions(pool,candidate)
    facts=list(state['decision_channel']['facts']);known={f['fact_id'] for f in facts}
    for n in state['notes']:
        if n['recommendation']['permission'] in ('suppressed','withdrawn'):
            for fact in n['recommendation']['risk_facts']:
                if fact['fact_id'] not in known:
                    facts.append(dict(fact_id=fact['fact_id'],conditions=fact['conditions'],query=fact['query'],
                        severity='note_suppression',status='suppressed',review_successes=[],review_failures=0))
                    known.add(fact['fact_id'])
    matched=[f for f in facts if f.get('status','suppressed')!='released' and close_conditions(pool,f['conditions'],physical)]
    if not matched:
        return dict(blocked=False,matched=[],matched_facts=[],eligible_review=False,event=None)
    budget=len(state['review_queries'])+len(state['decision_channel']['review_queries'])
    stagnation=old.stagnation_event(state['history'],query) if state['history'] else None
    changed=any(material_change(pool,f['conditions'],physical,focus) for f in matched)
    event=(dict(kind='material_condition_change',event_id='card_change_'+old.digest(
        dict(facts=sorted(f['fact_id'] for f in matched),conditions=physical,focus=focus))) if changed else stagnation)
    used=state['decision_channel']['used_review_events']
    eligible=bool(event) and budget<PARAMETERS['maximum_review_queries'] and event['event_id'] not in used
    return dict(blocked=True,matched=[f['fact_id'] for f in matched],matched_facts=deepcopy(matched),eligible_review=eligible,event=event)


def validate_card(raw,ids,pool,history):
    if not isinstance(raw,dict) or set(raw)!={'claim','focus','purpose','predictions'}:
        raise ValueError('decision_card needs claim, focus, purpose, predictions')
    if type(raw['claim']) is not str or not 1<=len(raw['claim'])<=1200:
        raise ValueError('A concise decision-specific public claim is required')
    if (not isinstance(raw['focus'],list) or not 1<=len(raw['focus'])<=2 or len(set(raw['focus']))!=len(raw['focus'])
            or set(raw['focus'])-set(pool.axes) or raw['purpose'] not in ('improve','learn','review')):
        raise ValueError('Use one/two real axes and improve/learn/review purpose')
    if not isinstance(raw['predictions'],list) or len(raw['predictions'])!=5:
        raise ValueError('Exactly five independent pre-experiment candidate forecasts required')
    probabilities={}
    for row in raw['predictions']:
        if not isinstance(row,dict) or set(row)!={'candidate_id','p_improve_incumbent'}:
            raise ValueError('Each forecast needs candidate_id and p_improve_incumbent')
        cid,p=row['candidate_id'],row['p_improve_incumbent']
        if cid not in ids or cid in probabilities:
            raise ValueError('Forecast every supplied candidate exactly once')
        if history:
            if type(p) not in (int,float) or not isfinite(p) or not 0<=p<=1:
                raise ValueError('Improvement probabilities must be finite in [0,1]')
        elif p is not None:
            raise ValueError('No incumbent exists on query1: all five probabilities must be null')
        probabilities[cid]=p
    return probabilities


def gate(pool,state,notes,context,ids,proposal,card,probabilities,legacy):
    ordered,_=engine.candidates(pool,context,len(state['history'])+1)
    bo=pool.ids[int(ordered[0])];q=len(state['history'])+1
    if not state['history']:
        return dict(selected_id=proposal,accepted=True,reason='data_free_domain_proposal',
            BO_reference_id=bo,logEI_gap=None,budget=None,review=None,learning=False)
    p_gp={cid:engine.gp_probability(context,pool.ix[cid],max(h['raw_value'] for h in state['history'])/pool.scale) for cid in ids}
    c=state['decision_channel']['calibration'];n=c['updates']
    prior=PARAMETERS['reliability_prior_count']
    reliability=(prior*.4+n*c['reliability'])/(prior+n)
    mature=any(r['use']=='positive_support' and credit_eligibility(pool,
        next(n for n in notes if (n['id'],n['version'])==(r['id'],r['version'])),state['history'],proposal)['eligible']
        for r in legacy['knowledge_uses'])
    budget=PARAMETERS['maximum_logEI_loss']*reliability*(1. if mature else PARAMETERS['tentative_budget_factor'])
    gap=float(context['base_logei'][pool.ix[bo]]-context['base_logei'][pool.ix[proposal]])
    discrimination=probabilities[proposal]-probabilities[bo]
    incremental=discrimination-(p_gp[proposal]-p_gp[bo])
    risk=risk_context(pool,dict(state,notes=notes),proposal,q,card['focus'])
    result=dict(selected_id=proposal,accepted=True,reason='prospective_incremental_preference',
        BO_reference_id=bo,logEI_gap=gap,budget=budget,calibration_reliability=reliability,
        positive_note_authority=mature,LLM_preference=discrimination,incremental_preference=incremental,
        GP_probabilities=p_gp,local_risk=risk,review=None,learning=False)
    if proposal==bo:
        result['reason']='proposal_matches_GP';return result
    reason=None
    if gap>budget+1e-12:
        reason='opportunity_cost_exceeds_calibrated_budget'
    elif risk['blocked']:
        if card['purpose']=='review' and risk['eligible_review']:
            result['review']=risk['event']
        else:
            reason='local_adverse_evidence_requires_eligible_review'
    if reason is None and (discrimination<=1e-12 or incremental<=1e-12):
        if (card['purpose']=='learn' and len(state['decision_channel']['learning_queries'])<PARAMETERS['maximum_learning_overrides']
                and (q<=5 or old.stagnation_event(state['history'],q))):
            result['learning']=True;result['reason']='bounded_information_test'
        elif result['review'] is None:
            reason='no_candidate_discrimination_beyond_GP'
    if reason:
        result.update(selected_id=bo,accepted=False,reason=reason,review=None,learning=False)
    return result


def decide(root,pool,notes,state,context,ids,query,raw):
    if not isinstance(raw,dict) or set(raw)!={'insight','knowledge_uses','assessments','decision_card'}:
        raise ValueError('V14 requires insight, knowledge_uses, assessments, decision_card')
    probabilities=validate_card(raw['decision_card'],ids,pool,state['history'])
    proposal=validate_insight(root,raw['insight'],ids)
    legacy=old_adapter.commit_decision(root,pool,notes,state['history'],context,ids,query,raw,
                                      change_thresholds=old_policy.thresholds(pool))
    proposed_reviews=any(x['use']=='review_test' for x in legacy['knowledge_uses'])
    used=len(state['review_queries'])+len(state['decision_channel']['review_queries'])
    if proposed_reviews and used>=PARAMETERS['maximum_review_queries']:
        raise ValueError('Combined note/card review budget exhausted')
    deployment=gate(pool,state,notes,context,ids,proposal,raw['decision_card'],probabilities,legacy)
    actual=deployment['selected_id'];deployed=actual==proposal
    if deployment['review'] and proposed_reviews:
        raise ValueError('One paid experiment cannot consume two independent review grants')
    packet=dict(query=query,selected_id=actual,proposed_id=proposal,BO_reference_id=deployment['BO_reference_id'],
        history_sha256=old.digest(state['history']),notes_before_sha256=old.digest(notes),
        channel_before_sha256=old.digest(state['decision_channel']),GP_sha256=engine.gp_hash(context),
        notes_at_commit=legacy['notes_at_commit'] if deployed else deepcopy(notes),
        knowledge_uses=legacy['knowledge_uses'] if deployed else [],
        note_forecasts=effective_forecasts(pool,legacy,notes,state['history']) if deployed else [],
        proposed_note_packet=legacy,decision_card=deepcopy(raw['decision_card']),all_probabilities=probabilities,
        actual_p_LLM=probabilities[actual],actual_p_GP=deployment.get('GP_probabilities',{}).get(actual),
        incumbent=max((h['raw_value'] for h in state['history']),default=None),
        deployment=deployment,raw_response_sha256=old.digest(raw))
    packet['commit_sha256']=old.digest(packet)
    return packet


def feedback(pool,notes,channel,history,packet,observation):
    if (packet['commit_sha256']!=old.digest({k:v for k,v in packet.items() if k!='commit_sha256'})
        or packet['history_sha256']!=old.digest(history) or packet['notes_before_sha256']!=old.digest(notes)
        or packet['channel_before_sha256']!=old.digest(channel) or packet['selected_id']!=observation['candidate_id']):
        raise ValueError('V14 pre-experiment commitment changed')
    out=deepcopy(packet['notes_at_commit']);by_key={(n['id'],n['version']):j for j,n in enumerate(out)}
    note_events=[]
    for row in packet['note_forecasts']:
        f=row['effective'];j=by_key[f['id'],f['version']]
        out[j],event=old.feedback(out[j],f,history,observation)
        event.update(declared_use=row['declared']['use'],direction_credit=row['credit'])
        note_events.append(event)
    result=deepcopy(channel);q=packet['query'];deployment=packet['deployment'];value=observation['raw_value']
    if deployment['learning']:
        result['learning_queries'].append(q)
    if deployment['review']:
        result['review_queries'].append(q);result['used_review_events'].append(deployment['review']['event_id'])
        known={f['fact_id'] for f in result['facts']}
        for fact in deployment['local_risk']['matched_facts']:
            if fact['fact_id'] not in known:
                result['facts'].append(deepcopy(fact));known.add(fact['fact_id'])
    if not history:
        event=dict(query=q,kind='no_incumbent_no_calibration',selected_id=packet['selected_id'])
    else:
        before=max(h['raw_value'] for h in history);improved=value>before
        result['calibration'],skill=calibration.update(result['calibration'],packet['actual_p_LLM'],packet['actual_p_GP'],improved)
        gain=value-before
        observed_sd=float(np.std([h['raw_value'] for h in history])) if len(history)>1 else pool.scale
        scale=max(observed_sd,.01*pool.scale)
        deployed_override=packet['selected_id']==packet['proposed_id'] and packet['selected_id']!=packet['BO_reference_id']
        accurate_failure=not improved and packet['actual_p_LLM']<.5 and skill['Brier_gain']>1e-12
        event=dict(query=q,kind='candidate_card_feedback',selected_id=packet['selected_id'],
            proposed_id=packet['proposed_id'],deployed_override=deployed_override,improved=improved,
            actual_gain=gain,standardized_gain=gain/scale,calibration=skill,
            accurate_failure_forecast=accurate_failure,local_suppression=False,review=deployment['review'])
        for fact in result['facts']:
            if deployment['review'] and fact['fact_id'] in deployment['local_risk']['matched']:
                if improved:
                    fact['review_successes'].append(q)
                    if len(fact['review_successes'])>=PARAMETERS['successful_review_tests']:
                        fact['status']='released'
                else:
                    fact['review_failures']+=1
                    fact['status']='withdrawn' if fact['review_failures']>=2 else 'suppressed'
        if deployed_override and not improved:
            physical=old_adapter.conditions(pool,packet['selected_id'])
            failed=[e for e in result['events'] if e.get('deployed_override') and not e.get('improved',True)
                    and close_conditions(pool,old_adapter.conditions(pool,e['selected_id']),physical)]
            severe=gain/scale<=-PARAMETERS['severe_shortfall_SD']
            if severe or accurate_failure or len(failed)+1>=PARAMETERS['ordinary_local_failures']:
                fact=dict(fact_id=old.digest(dict(commit=packet['commit_sha256'],observation=observation)),query=q,
                    candidate_id=packet['selected_id'],conditions=physical,focus=packet['decision_card']['focus'],
                    observed=value,incumbent=before,severity='large_shortfall' if severe else 'accurate_failure' if accurate_failure else 'repeated_local_failures',
                    status='suppressed',review_successes=[],review_failures=0)
                result['facts'].append(fact);event['local_suppression']=True
    result['events'].append(event)
    return out,result,note_events,event
