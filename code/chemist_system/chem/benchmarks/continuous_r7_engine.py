"""R7-S: passive, covariance-propagated, evidence-reversible preference BO.

Experimental mechanism only. The original GP, MC512 qLogEI, candidate pool,
time-varying opportunity gate, and pre-observation commitment are unchanged.
There is no active verification or gate exception in this R7-S stage.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import numpy as np

from . import continuous_r6_core as base_core
from . import continuous_r6_engine as r6
from . import continuous_r7_core as core


ARMS = ('B0', 'B2', 'B4', 'B2_online')
CALL_ROUNDS = (1,)  # One shared first-round hypothesis for controlled comparisons.
ONLINE_CALL_ROUNDS = r6.CALL_ROUNDS  # Separate adaptive LLM mechanism study.


def build_context(cache, state, t):
    if state['arm'] not in ARMS:
        raise ValueError('unknown R7-S arm')
    # R6 context is a read-only adapter to the identical original GP and menu.
    simulated_arm = 'A0' if state['arm'] == 'B0' else 'A4'
    context = r6.build_context(cache, dict(state, arm=simulated_arm), t)
    authorized_calls = ONLINE_CALL_ROUNDS if state['arm'] == 'B2_online' else CALL_ROUNDS
    if t not in authorized_calls:
        context = dict(context, menu=())
    return context


def public_menu(cache, state, t, context):
    return r6.public_menu(cache, dict(state, arm='A4'), t, context)


def diagnose_hypothesis(cache, context, history, hypothesis):
    """Pre-observation reachability; never loads an oracle outcome.

    Report both true current-weight action changes and counterfactual changes
    with the PREDECLARED R6 deployment cap, not a fitted optimum. An allowed
    action change is not evidence of chemical correctness or better AUC.
    """
    d = context['d']
    pool = cache.pool
    eligible = np.flatnonzero(d['eligible'])
    available = np.flatnonzero(d['available'])
    geometric = r6._scope_indices(cache, hypothesis, available)
    intersection = r6._scope_indices(cache, hypothesis, eligible)
    output = {
        'hypothesis_id': hypothesis['id'],
        'eligible_count': int(len(eligible)),
        'geometric_available': int(len(geometric)),
        'geometric_eligible': int(len(intersection)),
        'weight': float(r6._ledger_from_json(hypothesis).deployment_weight),
        'max_allowed_weight': float(hypothesis['ledger']['cap']),
        'BO_reference_id': context['gate'].bo_point_id,
        'covariance_relevant_eligible': 0,
        'actual_flip': False,
        'cap_flip': False,
        'max_abs_conditional_logEI_shift': 0.,
    }
    if not len(eligible):
        return output
    moments = r6.contrast_moments(cache, d['stats'], hypothesis, eligible)
    md, vd, cross = moments
    vz = vd + float(hypothesis['tau_y']) ** 2
    # For the decision, use the latent f(x), not observation-noisy y(x).
    latent_var = np.maximum(np.asarray(d['stats']['sd_y'])[eligible] ** 2, 1e-18)
    rho2 = np.minimum(cross ** 2 / (latent_var * vz), 1.)
    if np.any(cross ** 2 / (latent_var * vz) > 1. + 1e-8):
        raise ValueError('invalid candidate-to-contrast covariance')
    output['covariance_relevant_eligible'] = int(np.sum(rho2 > core.MIN_RHO_SQUARED))
    # Use the original exact sign-conditional skew-normal EI quadrature.
    scores = r6.guided_logei(cache, context, hypothesis, eligible, moments, history)
    for j in eligible:
        candidate_id = str(pool['ids'][j])
        base_score = float(d['base'][j])
        conditional_score = float(scores[candidate_id])
        if np.isfinite(base_score) and np.isfinite(conditional_score):
            output['max_abs_conditional_logEI_shift'] = max(
                output['max_abs_conditional_logEI_shift'], abs(conditional_score - base_score))
    candidates, gate = context['candidates'], context['gate']
    weight = output['weight']
    if weight > 0:
        actual = base_core.choose_guided_inside_gate(candidates, gate, scores, weight)
        output['actual_candidate'] = actual.point_id
        output['actual_flip'] = actual.point_id != gate.bo_point_id
    upper = base_core.choose_guided_inside_gate(
        candidates, gate, scores, output['max_allowed_weight'])
    output['cap_candidate'] = upper.point_id
    output['cap_flip'] = upper.point_id != gate.bo_point_id
    return output


def plan(cache, state, t, context, proposal=None):
    """Choose and return a durable prequery commitment, without oracle access."""
    arm = state['arm']
    if arm not in ARMS:
        raise ValueError('unknown R7-S arm')
    if proposal is not None and not context['menu']:
        raise ValueError('proposal supplied outside authorized R7 call')
    if arm == 'B0' and proposal is not None:
        raise ValueError('B0 must not receive LLM proposals')
    review = r6._review_proposal(cache, state, t, context, proposal)
    if proposal is not None:
        state['calls'].append({'round': t, 'proposal': deepcopy(proposal),
                               'review': review, 'menu_ids': [p.point_id for p in context['menu']],
                               'GP_filtered_membership': True})
    d, gate = context['d'], context['gate']
    decision = base_core.Decision(gate.bo_point_id, 'base_bo', 0.)
    active_hypothesis_id = None
    reachability = None
    if state['hypotheses'] and arm != 'B0':
        ranked = sorted(state['hypotheses'], key=lambda h:
                        (-r6._ledger_from_json(h).deployment_weight,
                         -h['created_round'], h['id']))
        hypothesis = ranked[0]
        weight = r6._ledger_from_json(hypothesis).deployment_weight
        active_hypothesis_id = hypothesis['id']
        if weight > 0:
            eligible = np.flatnonzero(d['eligible'])
            moments = r6.contrast_moments(cache, d['stats'], hypothesis, eligible)
            scores = r6.guided_logei(cache, context, hypothesis, eligible,
                                     moments, state['history'])
            decision = base_core.choose_guided_inside_gate(
                context['candidates'], gate, scores, weight)
            reachability = {'eligible_count': int(len(eligible)),
                            'geometric_eligible_count': int(len(r6._scope_indices(cache, hypothesis, eligible))),
                            'covariance_propagation': 'all_original_gate_candidates'}

    index = cache.ix[decision.point_id]
    shadows = []
    for hypothesis in state['hypotheses']:
        md, vd, covs = r6.contrast_moments(cache, d['stats'], hypothesis, [index])
        prediction = r6._shadow_for(cache, d['stats'], hypothesis, index, t,
                                    (md, vd, covs[0]))
        rho2 = core.squared_correlation(prediction.gp_var_y, vd,
                                        covs[0], hypothesis['tau_y'])
        informative = rho2 > core.MIN_RHO_SQUARED
        sign_info = core.sign_information_for_shadow(prediction) if informative else 0.
        shadows.append({'prediction': asdict(prediction),
                        'informative': informative, 'rho_squared': rho2,
                        'information_nats': sign_info,
                        'information_kind': 'binary_sign',
                        'scope_geometry_passed': bool(len(r6._scope_indices(cache, hypothesis, [index])) == 1)})

    return {
        'iteration': t, 'task': cache.pool['task'], 'arm': arm,
        'history_sha256': r6.digest(state['history']),
        'selected_id': decision.point_id, 'BO_reference_id': gate.bo_point_id,
        'selection_mode': decision.mode, 'opportunity_gap': float(decision.acquisition_gap),
        'normal_budget': float(gate.epsilon), 'normal_gate_eligible': decision.point_id in gate.eligible_point_ids,
        'used_exception': False, 'eligible_count': len(gate.eligible_point_ids),
        'menu_ids': [p.point_id for p in context['menu']],
        'review': review, 'proposal': proposal,
        'active_hypothesis_id': active_hypothesis_id,
        'reachability_at_commit': reachability,
        'prequery_shadows': shadows,
        'hypotheses_at_commit': deepcopy(state['hypotheses']),
        'calls_at_commit': deepcopy(state['calls']),
        'budgets_at_commit': {'active_verifications': 0, 'exceptions': 0},
        'GP_fit_key': d['stats']['fit_key'], 'GP_fit_seed': d['stats']['fit_seed'],
        'GP_reference_logEI': float(gate.best_logei), 'GP_selected_logEI': float(d['base'][index]),
        'GP_mean_y': float(d['stats']['mu_y'][index]),
        'GP_predictive_SD_y': float(d['stats']['sd_predictive_y'][index]),
        'GP_acquisition_meta': d['base_meta'],
        'comparison_schema_version': 'R7_covariance_passive_v1',
    }


def apply_feedback(state, commitment, observation):
    """Reuse R6's strictly prospective, reversible evidence bookkeeping."""
    return r6.apply_feedback(state, commitment, observation)
