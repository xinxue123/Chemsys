"""V7 B: actual expert-vs-GP discrimination gate and ranking for validation.

All V5 common gates, full-scope probes, learned deployment, credit and feedback
remain unchanged. Only the optional information gate and first ranking key
change to numerically resolved positive JS increment over the same-rule BO.
This is an explicit gate-and-ranking ablation, not a ranking-only variant.
"""
from copy import deepcopy
from math import log

import numpy as np
from scipy.special import logsumexp

from . import knowledge_transfer_engine as transfer
from . import knowledge_discrimination_math as discrimination_math
from .knowledge_llm import digest

PARAMETERS = dict(transfer.PARAMETERS,
    minimum_incremental_information=0.,
    verification_information='actual tempered expert versus GP model-label Jensen-Shannon information',
    discrimination_model_prior=.5,
    discrimination_numerical_tolerance=1e-12,
    discrimination_gate='delta_JS > max(numerical_tolerance, candidate_error + BO_error)')
policy, math = transfer.policy, transfer.math
choose, should_call = transfer.choose, transfer.should_call
coverage, prepare_state, _refresh = transfer.coverage, transfer.prepare_state, transfer._refresh
feedback = transfer.feedback


def warm():
    transfer.warm()
    discrimination_math.vector_discrimination(np.array([0.]), np.array([1.]),
        np.array([0.]), np.array([0.]), tau=PARAMETERS['preference_noise_sd'])


def plan(pool, gp, state, t, context, proposal=None, menu_ids=(), mode='full'):
    if mode not in ('bo', 'shadow', 'deployment', 'full', 'verification'):
        raise ValueError('Unknown rule arm')
    hypotheses, reviews = deepcopy(state['hypotheses']), []
    if proposal is not None:
        hypotheses, reviews = policy.review_and_apply(pool, hypotheses, proposal, menu_ids, t, state['history'])
    hypotheses, updates, rows, calculations = _refresh(pool, gp, hypotheses, state['history'], context, t)
    reviews.extend(updates)
    base, bo, stats = context['base_logei'], context['bo_index'], context['stats']
    selected, selection = bo, 'base_bo'
    combined = np.array(base, copy=True)
    deployed = []
    if mode in ('deployment', 'full'):
        raw = [(r, calculations[(r['id'], r['version'])]) for r in rows
               if r['weight'] > 0 and (r['id'], r['version']) in calculations]
        total = sum(r['weight'] for r, _ in raw)
        factor = min(1., PARAMETERS['maximum_deployment_weight'] / total) if total else 1.
        deployed = [(r, c, r['weight'] * factor) for r, c in raw]
        if deployed:
            offsets = np.stack([c['offsets'] for _, c, _ in deployed])
            correction = logsumexp(np.stack([np.full_like(base, log(1 - sum(w for _, _, w in deployed)))] +
                [log(w) + c['offsets'] for _, c, w in deployed]), axis=0)
            correction[np.all(offsets <= PARAMETERS['numerical_noop_atol'], axis=0)] = 0.
            combined = base + correction
            selected = choose(combined, context)
            if selected != bo:
                if correction[selected] <= PARAMETERS['numerical_noop_atol']:
                    raise ValueError('Unsupported third point cannot be a knowledge-guided winner')
                selection = 'knowledge_guided'
    verification = None
    if (mode in ('full', 'verification') and selected == bo and t <= PARAMETERS['last_verification_round']
            and state['verification_used'] < PARAMETERS['maximum_optional_verifications']):
        candidates = []
        for row in rows:
            key = row['id'], row['version']
            if row['weight'] > 0 or not row['can_change_rank_at_cap'] or key not in calculations:
                continue
            h = next(h for h in hypotheses if (h['id'], h['version']) == key)
            evidence = policy.evidence_summary(h)
            if evidence['effective_prediction_score'] < 0 or evidence['failures'] > evidence['successes']:
                continue
            c = calculations[key]
            bo_position = c['positions'].get(int(bo))
            bo_direction = float(c['values']['sign_information'][bo_position]) if bo_position is not None else 0.
            common = []
            for position, j in enumerate(c['indices']):
                gap = float(base[bo] - base[j])
                if (j != bo and gap <= min(context['epsilon'], PARAMETERS['verification_maximum_gap'])
                        and c['values']['offset'][position] > PARAMETERS['numerical_noop_atol']):
                    common.append((position, int(j), gap))
            if not common:
                continue
            # Evaluate the unchanged actual tempered density only after all
            # common gates. BO is included whenever this rule can predict it;
            # otherwise its expert is exactly its GP and its JS is zero.
            positions = ([bo_position] if bo_position is not None else []) + [k for k, _, _ in common]
            js = np.asarray([c['indices'][k] for k in positions], dtype=int)
            refs = np.asarray([c['probes'][int(j)]['reference_y'] for j in js], dtype=float)
            discrimination = discrimination_math.vector_discrimination(
                np.asarray(stats['mu_y'])[js], np.square(np.asarray(stats['sd_predictive_y'])[js]),
                refs, np.asarray(c['values']['log_tempering'])[positions],
                tau=PARAMETERS['preference_noise_sd'])
            information = np.asarray(discrimination['js'], dtype=float)
            errors = np.asarray(discrimination['error'], dtype=float)
            if (information.shape != js.shape or errors.shape != js.shape
                    or not np.all(np.isfinite(information)) or not np.all(np.isfinite(errors))
                    or np.any(information < 0.) or np.any(errors < 0.)):
                raise ValueError('Invalid model-discrimination result')
            offset = int(bo_position is not None)
            reference_information = float(information[0]) if offset else 0.
            reference_error = float(errors[0]) if offset else 0.
            for d, (position, j, gap) in enumerate(common, offset):
                info, error = float(information[d]), float(errors[d])
                increment = info - reference_information
                tolerance = max(PARAMETERS['discrimination_numerical_tolerance'], error + reference_error)
                if increment <= tolerance:
                    continue
                direction = float(c['values']['sign_information'][position])
                details = dict(incremental_information=increment, information=info,
                    reference_information=reference_information, information_error=error,
                    ref_information_error=reference_error, information_gate_tolerance=tolerance,
                    direction_information=direction, incremental_direction_information=direction-bo_direction)
                candidates.append((increment, -gap, -context['tie_priority'][j], j, key, details))
        if candidates:
            winner = max(candidates, key=lambda x: x[:3])
            selected, selection = winner[3], 'optional_verification'
            c = calculations[winner[4]]
            verification = dict(id=winner[4][0], version=winner[4][1], **winner[5],
                reference_id=c['probes'][selected]['reference_id'],
                reference_y=c['probes'][selected]['reference_y'],
                remaining_executable_probes=len(c['indices']) - 1,
                information_kind='model_label_JS_half_prior', minimum_unqueried_endpoints=1)
    shadows = []
    for h in hypotheses:
        key = h['id'], h['version']
        if h['status'] == 'retracted' or key not in calculations:
            continue
        c = calculations[key]
        position = c['positions'].get(int(selected))
        if position is None:
            continue
        ref = c['probes'][int(selected)]
        pred = math.freeze_prediction(h['id'], h['version'], pool.ids[selected], t,
            float(stats['mu_y'][selected]), float(stats['sd_y'][selected]) ** 2,
            float(stats['sd_predictive_y'][selected]) ** 2, ref['reference_id'], ref['reference_y'],
            tempering=float(c['values']['tempering'][position]), tau=PARAMETERS['preference_noise_sd'],
            log_tempering=float(c['values']['log_tempering'][position]), semantic_key=h['semantic_key'])
        shadows.append(dict(id=h['id'], version=h['version'], prediction=pred,
            information_nats=float(c['values']['sign_information'][position]),
            informative=bool(np.isfinite(c['values']['log_tempering'][position])),
            reference_is_observed=True))
    selected_weights = {(r['id'], r['version']): w for r, _, w in deployed}
    mixture = [dict(id=s['id'], version=s['version'], weight=selected_weights[(s['id'], s['version'])],
                    prediction_sha256=s['prediction']['prediction_sha256']) for s in shadows
               if selected_weights.get((s['id'], s['version']), 0.) > 0]
    if selection == 'knowledge_guided' and not mixture:
        raise ValueError('Every knowledge winner requires a deployed prospective predictor')
    return dict(query=t, history_sha256=digest(state['history']), selected_id=pool.ids[selected],
        BO_reference_id=pool.ids[bo], mode=selection, mechanism='discrimination_rule_v7',
        base_logei=float(base[selected]), reference_logei=float(base[bo]), combined_logei=float(combined[selected]),
        opportunity_gap=float(base[bo] - base[selected]), epsilon=context['epsilon'],
        eligible_count=int(context['eligible'].sum()), coverage=rows, reviews=reviews,
        hypotheses_at_commit=hypotheses, shadows=shadows, proposal=proposal, menu_ids=list(menu_ids),
        verification=verification, GP_fit_key=stats['fit_key'], GP_acquisition_meta=context['acquisition_meta'],
        deployed_versions=[dict(id=r['id'], version=r['version'], weight=r['weight'], effective_mixture_weight=w)
                           for r, _, w in deployed], selected_mixture=mixture)

