"""Same probabilities, deterministic ranks; only the deployed weight differs."""
from copy import deepcopy
from math import isfinite
import numpy as np
from cold_start_v13 import adapter, control
from reasoning_memory_v13 import calibration
from knowledge_rank_v9 import engine

PARAMETERS = dict(fixed_weight=.4, maximum_weight=.8,
    rank_rule='(1-w)*tie-aware GP Borda + w*tie-aware candidate-forecast Borda',
    uniform_forecast_rule='unchanged native GP first point',
    calibration_channel='all selected candidate forecasts, not individual chemical claims',
    first_query='official proposal; no GP, incumbent, ranks, or calibration')


def borda(values):
    x = np.asarray(values, float)
    return np.asarray([(np.count_nonzero(x<v)+.5*(np.count_nonzero(x==v)-1))/(len(x)-1) for v in x])


def predictions(raw,pool,ids,history):
    if not isinstance(raw,dict) or set(raw)!={'insight','knowledge_uses','assessments','decision_card'}:
        raise ValueError('Require insight, knowledge_uses, assessments, decision_card')
    card = raw['decision_card']
    if not isinstance(card,dict) or set(card)!={'claim','focus','purpose','predictions'}:
        raise ValueError('Invalid candidate forecast card')
    if (type(card['claim']) is not str or not 1<=len(card['claim'])<=1200
        or not isinstance(card['focus'],list) or not 1<=len(card['focus'])<=2
        or len(set(card['focus']))!=len(card['focus']) or set(card['focus'])-set(pool.axes)
        or card['purpose'] not in ('improve','learn','review')):
        raise ValueError('Valid claim, physical focus and purpose required')
    rows=card['predictions'];out={}
    if not isinstance(rows,list) or len(rows)!=5:
        raise ValueError('Forecast all five candidates')
    for row in rows:
        if not isinstance(row,dict) or set(row)!={'candidate_id','p_improve_incumbent'}:
            raise ValueError('Candidate prediction needs ID and improvement probability')
        cid,p=row['candidate_id'],row['p_improve_incumbent']
        if cid not in ids or cid in out:
            raise ValueError('Each displayed candidate must occur exactly once')
        if history:
            if type(p) not in (int,float) or not isfinite(p) or not 0<=p<=1:
                raise ValueError('Finite probability in [0,1] required')
        elif p is not None:
            raise ValueError('No incumbent on query1; all probabilities must be null')
        out[cid]=p
    return out


def rank(pool,context,ids,probabilities,weight,proposal,history):
    original=engine.gp_hash(context)
    ordered,_=engine.candidates(pool,context,len(history)+1)
    ordered_ids=[pool.ids[int(j)] for j in ordered]
    if set(ids)!=set(ordered_ids) or not 0<=weight<=PARAMETERS['maximum_weight']:
        raise ValueError('Unchanged five-point GP menu and bounded weight required')
    if not history:
        return dict(selected_id=proposal,weight=weight,GP_reference_id=ordered_ids[0],
            reason='no_data_official_proposal',ordered_ids=ordered_ids,
            GP_rank_scores=None,LLM_rank_scores=None,combined_scores=None,
            GP_sha256=original,GP_all_tied=None,LLM_all_tied=None)
    gp=np.asarray([context['base_logei'][int(j)] for j in ordered],float)
    p=np.asarray([probabilities[cid] for cid in ordered_ids],float)
    base=borda(gp);local=borda(p)
    no_information=bool(np.all(p==p[0]))
    scores=base.copy() if no_information else (1-weight)*base+weight*local
    best=np.flatnonzero(scores==scores.max())
    selected=int(ordered[best[np.argmin(context['tie_priority'][ordered[best]])]])
    if no_information: selected=int(ordered[0])
    if engine.gp_hash(context)!=original:
        raise ValueError('Rank controller mutated the GP')
    return dict(selected_id=pool.ids[selected],weight=weight,GP_reference_id=ordered_ids[0],
        reason='uniform_forecast_use_GP' if no_information else 'program_rank_fusion',
        ordered_ids=ordered_ids,GP_rank_scores=base.tolist(),LLM_rank_scores=local.tolist(),
        combined_scores=scores.tolist(),GP_sha256=original,
        GP_all_tied=bool(np.all(gp==gp[0])),LLM_all_tied=no_information)


def plan(root,pool,notes,state,context,ids,q,raw,arm):
    probabilities=predictions(raw,pool,ids,state['history'])
    legacy=adapter.commit_decision(root,pool,notes,state['history'],context,ids,q,raw,
                                  change_thresholds={a:.25 for a,k in zip(pool.axes,pool.kinds) if k=='continuous'})
    if len(state['review_queries'])>=2 and any(u['use']=='review_test' for u in legacy['knowledge_uses']):
        raise ValueError('Trajectory review budget exhausted')
    proposal=legacy['selected_id']
    weight=PARAMETERS['fixed_weight'] if arm=='fixed' else state['ranker_KF']['reliability']
    selected=rank(pool,context,ids,probabilities,weight,proposal,state['history'])
    shadow=rank(pool,context,ids,probabilities,PARAMETERS['fixed_weight'],proposal,state['history'])
    actual=selected['selected_id'];deployed=actual==proposal
    p_gp=engine.gp_probability(context,pool.ix[actual],max(h['y'] for h in state['history'])) if state['history'] else None
    packet=dict(query=q,selected_id=actual,proposed_id=proposal,
        BO_reference_id=selected['GP_reference_id'],history_sha256=control.digest(state['history']),
        notes_before_sha256=control.digest(notes),ranker_before=deepcopy(state['ranker_KF']),
        GP_sha256=engine.gp_hash(context),probabilities=probabilities,ranking=selected,
        fixed_weight_shadow=shadow,official_proposal_packet=legacy,
        notes_at_commit=deepcopy(legacy['notes_at_commit'] if deployed else notes),
        note_forecasts=deepcopy(legacy['forecasts'] if deployed else []),
        knowledge_uses=deepcopy(legacy['knowledge_uses'] if deployed else []),
        actual_p_LLM=probabilities[actual],actual_p_GP=p_gp,
        unexecuted_proposal_has_no_feedback=not deployed)
    packet['commit_sha256']=control.digest(packet)
    return packet


def feedback(state,commit,observation):
    if (commit['commit_sha256']!=control.digest({k:v for k,v in commit.items() if k!='commit_sha256'})
        or commit['prestate_sha256']!=control.digest(state)
        or commit['query']!=len(state['history'])+1 or observation['candidate_id']!=commit['selected_id']
        or observation['candidate_id'] in {h['candidate_id'] for h in state['history']}):
        raise ValueError('Pre-outcome state or selected point changed')
    packet=commit['decision']
    if (packet['commit_sha256']!=control.digest({k:v for k,v in packet.items() if k!='commit_sha256'})
        or packet['ranker_before']!=state['ranker_KF'] or packet['selected_id']!=observation['candidate_id']
        or packet['notes_before_sha256']!=control.digest(commit['edited_notes'])
        or packet['history_sha256']!=control.digest(state['history'])):
        raise ValueError('Program forecast commitment changed')
    out=deepcopy(state);notes=deepcopy(packet['notes_at_commit']);events=[]
    for forecast in packet['note_forecasts']:
        index=next(i for i,n in enumerate(notes) if (n['id'],n['version'])==(forecast['id'],forecast['version']))
        notes[index],event=control.feedback(notes[index],forecast,state['history'],observation);events.append(event)
    out['notes']=notes
    if state['history']:
        improved=observation['y']>max(h['y'] for h in state['history'])
        out['ranker_KF'],event=calibration.update(state['ranker_KF'],packet['actual_p_LLM'],packet['actual_p_GP'],improved)
        event=dict(event,query=commit['query'],candidate_id=observation['candidate_id'],improved=improved,
                   p_LLM=packet['actual_p_LLM'],p_GP=packet['actual_p_GP'])
    else:
        event=dict(query=1,kind='no_incumbent_no_KF_update')
    if commit['memory']: out['edit_calls'].append(commit['query'])
    if any(u['use']=='review_test' for u in packet['knowledge_uses']):out['review_queries'].append(commit['query'])
    out['insights'].append(dict(query=commit['query'],insight=deepcopy(commit['reasoning']['raw']['insight']),
        proposed_id=packet['proposed_id'],executed_id=packet['selected_id']))
    before=max((h['y'] for h in state['history']),default=None)
    out['history'].append(deepcopy(observation))
    out['decisions'].append(dict(query=commit['query'],selected_id=packet['selected_id'],
        proposed_id=packet['proposed_id'],BO_reference_id=packet['BO_reference_id'],
        raw_value=observation['raw_value'],y=observation['y'],incumbent_before_y=before,
        incumbent_after_y=max(h['y'] for h in out['history']),feedback=events,
        ranker_feedback=event,knowledge_uses=packet['knowledge_uses'],
        commit_sha256=commit['commit_sha256']))
    out['status']='COMPLETE' if len(out['history'])==20 else 'RUNNING'
    return out
