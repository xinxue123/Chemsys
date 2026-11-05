"""Cached knowledge preferences within a contracting qLogEI budget."""
from copy import deepcopy
import numpy as np
from scipy.special import ndtr
from fast_hit_bo.policy import choose
from llm_event_start import policy as original
from llm_screen_qlogei import knowledge
from . import PARAMETERS as P


def authority(query,total_queries=P['total_queries']):
    if type(query) is not int or type(total_queries) is not int or total_queries<2 or not 1<=query<=total_queries:
        raise ValueError('A valid query inside a budget of at least two is required')
    return (total_queries-query)/(total_queries-1)


def decide(pool,context,notes,proposal,query,seed=0,total_queries=P['total_queries']):
    cap=authority(query,total_queries)
    if query==1:
        if context is not None:
            raise ValueError('The LLM first choice has no observation-conditioned GP')
        if proposal is None:
            raise ValueError('The first point must be explicitly proposed by the LLM')
        packet=original.decide(pool,None,notes,proposal,1,seed,'event')
        if not packet['proposal_accepted']:
            raise ValueError('The first LLM proposal must project to an executable point')
        packet.update(handoff_authority_cap=cap,effective_logei_gap=None,
            knowledge_confidence_factor=None,phase='LLM_initial',knowledge_used=True)
        return packet,{}
    if context is None:
        raise ValueError('After the first observation, the native GP context is required')
    original_arrays={k:context[k].copy() for k in ('mu','variance','logei','available','priorities')}
    available=context['available']
    base=choose(context['logei'],available,context['priorities'])
    preference=np.zeros(len(pool.ids))
    enabled=[]
    diagnostics=[]
    # The exact endpoint ignores all Note states, including candidate ties.
    if cap>0:
        for note in notes:
            r=note['KF']['reliability']
            probability=knowledge.forecast(note,context['incumbent'])
            permitted=note['permission'] in ('provisional','active') and r>=P['minimum_KF_reliability']
            if note['direction']=='favour' and probability<P['favour_probability_threshold']:
                permitted=False
            mask=knowledge.scope_mask(pool,note['scope'])
            if permitted and (mask & available).any():
                enabled.append(note)
                preference[mask]+=r if note['direction']=='favour' else -r
            diagnostics.append(dict(id=note['id'],version=note['version'],permitted=bool(permitted),
                direction=note['direction'],reliability=r,p_improve=probability,
                available_scope=int((mask & available).sum())))
    confidence=min(1.,max((n['KF']['reliability'] for n in enabled),default=0.)/P['reference_KF_reliability'])
    epsilon=P['maximum_logei_gap']*cap*confidence
    budget=available & (context['logei']>=context['logei'][base]-epsilon)
    eligible=budget.copy()
    selected=base
    mode='pure_BO_endpoint' if cap==0 else 'no_actionable_knowledge_use_BO'
    if enabled and epsilon>0:
        scores=preference[budget]
        if scores.max()!=scores.min():
            eligible &= preference==scores.max()
            selected=choose(context['logei'],eligible,context['priorities'])
            mode='contracting_knowledge_preference'
    # A fresh event proposal may be executed only inside both current limits.
    projected_id=None
    accepted=False
    rejection=None
    if cap>0 and proposal is not None:
        projected,distance=original.project(pool,proposal,notes,available,query)
        basis=[next(n for n in notes if n['id']==key) for key in proposal['basis_ids']]
        allowed_ids={n['id'] for n in enabled if n['direction']=='favour'}
        permitted=all(n['id'] in allowed_ids for n in basis)
        if projected is not None:
            projected_id=pool.ids[projected]
        if projected is None:
            rejection='no_scoped_unqueried_projection'
        elif distance>original.P['maximum_projection_distance']:
            rejection='projection_too_far'
        elif not permitted:
            rejection='basis_not_recommendable'
        elif not eligible[projected]:
            rejection='outside_current_handoff_budget_or_preference'
        else:
            selected=projected
            accepted=True
            mode='refresh_proposal_within_remaining_authority'
    elif cap==0 and proposal is not None:
        rejection='pure_BO_endpoint_has_no_LLM_authority'
    gap=float(context['logei'][base]-context['logei'][selected])
    assert gap<=epsilon+1e-12
    if cap==0:
        assert selected==base
    for key,value in original_arrays.items():
        np.testing.assert_array_equal(context[key],value)
    promoted=[]
    for note in enabled:
        if note['direction']=='favour' and knowledge.scope_mask(pool,note['scope'])[selected]:
            promoted.append(note['id'])
    forecasts=[]
    p_gp=float(ndtr((context['mu'][selected]-context['incumbent'])/np.sqrt(max(context['variance'][selected],1e-18))))
    for note in notes:
        if knowledge.scope_mask(pool,note['scope'])[selected]:
            forecasts.append(dict(id=note['id'],version=note['version'],incumbent_y=context['incumbent'],
                p_LLM=knowledge.forecast(note,context['incumbent']),p_GP=p_gp,
                yield_mean=note['yield_mean'],yield_sd=note['yield_sd'],
                promoted_selection=bool(selected!=base and note['id'] in promoted)))
    packet=dict(query=query,selected_id=pool.ids[selected],BO_reference_id=pool.ids[base],
        rule=mode,phase='pure_BO' if cap==0 else 'handoff',
        handoff_authority_cap=cap,knowledge_confidence_factor=confidence,effective_logei_gap=epsilon,
        minimum_retained_exp_qLogEI=float(np.exp(-epsilon)),actual_logei_gap=gap,
        knowledge_used=bool(selected!=base),promoting_note_ids=promoted,
        budget_candidates=int(budget.sum()),eligible_candidates=int(eligible.sum()),
        projected_id=projected_id,proposal_accepted=accepted,proposal_rejection=rejection,
        forecasts=forecasts,note_diagnostics=diagnostics,GP_mean_or_variance_changed=False,rollout_used=False)
    return packet,dict(budget=budget,eligible=eligible,knowledge_preference=preference)
