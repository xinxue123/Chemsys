"""Filter a bounded qLogEI set; select its unchanged native qLogEI maximum."""
from copy import deepcopy
import numpy as np
from scipy.special import ndtr
from fast_hit_bo.policy import choose
from . import PARAMETERS as P
from .knowledge import scope_mask,forecast


def screening(pool,context,notes,*,fixed_shadow=False):
    available=context['available']
    logei=context['logei']
    priority=context['priorities']
    baseline=choose(logei,available,priority)
    budget=available & (logei>=logei[baseline]-P['logei_budget'])
    incumbent=context['incumbent']
    p_gp=ndtr((context['mu']-incumbent)/np.sqrt(np.maximum(context['variance'],1e-18)))
    positive=np.zeros(len(pool.ids))
    negative=np.zeros(len(pool.ids))
    details=[]
    for note in notes:
        mask=scope_mask(pool,note['scope'])
        permission='provisional' if fixed_shadow else note['permission']
        reliability=.4 if fixed_shadow else note['KF']['reliability']
        enabled=permission in ('provisional','active') and reliability>=P['minimum_reliability']
        probability=forecast(note,incumbent)
        signed=reliability*(probability-p_gp)
        strength=np.zeros(len(pool.ids))
        if enabled:
            if note['direction']=='favour':
                strength[mask]=np.maximum(signed[mask],0.)
                positive=np.maximum(positive,strength)
            else:
                strength[mask]=np.maximum(-signed[mask],0.)
                negative=np.maximum(negative,strength)
        details.append(dict(id=note['id'],version=note['version'],enabled=enabled,
            reliability=reliability,permission=permission,p_LLM=probability,
            scope_candidates=int((mask & available).sum()),
            budget_overlap=int((mask & budget).sum()),mask=mask,strength=strength))
    veto=negative>=P['minimum_weighted_probability_gap']
    promoted=positive>=P['minimum_weighted_probability_gap']
    eligible=budget & promoted & ~veto
    mode='LLM_favour_scope'
    if not eligible.any():
        eligible=budget & ~veto
        mode='LLM_avoid_scope' if (budget & veto).any() else 'no_actionable_scope_use_qLogEI'
    if not eligible.any():
        eligible=available.copy()
        mode='empty_screen_fallback_full_qLogEI'
    selected=choose(logei,eligible,priority)
    return dict(selected=selected,baseline=baseline,budget=budget,eligible=eligible,
        p_GP=p_gp,positive=positive,negative=negative,promoted=promoted,veto=veto,
        details=details,mode=mode)


def decision(pool,context,notes,query):
    original={k:context[k].copy() for k in ('mu','variance','logei','available','priorities')}
    plan=screening(pool,context,notes)
    fixed=screening(pool,context,notes,fixed_shadow=True)
    j=plan['selected']
    predictions=[]
    detail=[]
    for note,row in zip(notes,plan['details']):
        if row['mask'][j]:
            predictions.append(dict(id=note['id'],version=note['version'],
                incumbent_y=context['incumbent'],p_LLM=row['p_LLM'],p_GP=float(plan['p_GP'][j]),
                yield_mean=note['yield_mean'],yield_sd=note['yield_sd'],
                promoted_selection=bool(row['enabled'] and note['direction']=='favour'
                    and row['strength'][j]>=P['minimum_weighted_probability_gap']
                    and plan['mode']=='LLM_favour_scope')))
        detail.append({k:v for k,v in row.items() if k not in ('mask','strength')})
    for k,value in original.items():
        np.testing.assert_array_equal(value,context[k])
    packet=dict(selected_id=pool.ids[j],BO_reference_id=pool.ids[plan['baseline']],
        rule='native_qLogEI_after_scoped_LLM_screen',screen_mode=plan['mode'],
        rollout_used=False,query=query,notes_before=deepcopy(notes),
        forecasts=predictions,note_diagnostics=detail,
        budget_candidates=int(plan['budget'].sum()),screened_candidates=int(plan['eligible'].sum()),
        baseline_survives=bool(plan['eligible'][plan['baseline']]),
        selected_qLogEI=float(context['logei'][j]),
        qLogEI_opportunity_loss=float(context['logei'][plan['baseline']]-context['logei'][j]),
        fixed_weight_shadow_id=pool.ids[fixed['selected']],
        pure_qlogei_shadow_id=pool.ids[plan['baseline']])
    if plan['mode']!='empty_screen_fallback_full_qLogEI':
        assert packet['qLogEI_opportunity_loss']<=P['logei_budget']+1e-12
    arrays=dict(budget=plan['budget'],eligible=plan['eligible'],p_GP=plan['p_GP'],
        positive_strength=plan['positive'],negative_strength=plan['negative'])
    return packet,arrays
