"""V6 example-point validation; all V5 prediction/deployment gates retained.

The frozen V5 prompt supplies example pairs, not independent ranked experiment
recommendations. Optional validation may use only accepted examples, including
new examples of an existing semantic rule. Rule credit and proposal provenance
are separate; neither repeated examples nor a changed point reset evidence.
"""
from copy import deepcopy
from math import log

import numpy as np
from scipy.special import logsumexp

from . import knowledge_transfer_engine as transfer
from .knowledge_llm import digest

PARAMETERS = transfer.PARAMETERS
policy, math = transfer.policy, transfer.math
warm, choose, should_call = transfer.warm, transfer.choose, transfer.should_call
coverage, prepare_state = transfer.coverage, transfer.prepare_state
_refresh = transfer._refresh


def _collect_proposals(state, proposal, reviews, hypotheses, t):
    records = deepcopy(state.get('validation_proposals', []))
    if proposal is None:
        return records
    for review in reviews:
        if (not review.get('accepted') or review.get('op') not in ('add', 'revise')
                or review.get('outcome') not in ('add', 'revise', 'duplicate_semantics_preserved')):
            continue
        action = proposal['actions'][review['action_index']]
        h = next(h for h in hypotheses if (h['id'], h['version']) == (review['id'], review['version']))
        record = dict(id=h['id'], version=h['version'], semantic_key=h['semantic_key'],
            point_id=action['plus_id'], proposed_reference_id=action['minus_id'],
            proposal_round=t, action_index=review['action_index'], action_sha256=digest(action))
        # Idempotent durable replay; repeated later calls remain distinct source
        # records but can never create an extra experimental/evidence budget.
        if record not in records:
            records.append(record)
    return records


def _executable_proposals(pool, h, calculation, records):
    selected = {}
    if h['status'] == 'retracted':
        return selected
    for record in records:
        if ((record['id'], record['version'], record['semantic_key'])
                != (h['id'], h['version'], h['semantic_key'])):
            continue
        j = pool.ix.get(record['point_id'])
        # make_probes already enforces unqueried, scope, direction, eligibility
        # and a highest-valued compatible own observed reference.
        if j is not None and j in calculation['positions']:
            selected.setdefault(int(j), record)
    return selected


def plan(pool, gp, state, t, context, proposal=None, menu_ids=(), mode='full'):
    if mode not in ('bo', 'shadow', 'deployment', 'full', 'verification'):
        raise ValueError('Unknown rule arm')
    hypotheses, reviews = deepcopy(state['hypotheses']), []
    if proposal is not None:
        hypotheses, reviews = policy.review_and_apply(pool, hypotheses, proposal, menu_ids, t, state['history'])
    validation_proposals = _collect_proposals(state, proposal, reviews, hypotheses, t)
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
            k = c['positions'].get(int(bo))
            bo_info = float(c['values']['sign_information'][k]) if k is not None else 0.
            proposed = _executable_proposals(pool, h, c, validation_proposals)
            for k, j in enumerate(c['indices']):
                if int(j) not in proposed:
                    continue
                gap = float(base[bo] - base[j])
                info = float(c['values']['sign_information'][k])
                if (j != bo and gap <= min(context['epsilon'], PARAMETERS['verification_maximum_gap'])
                        and info - bo_info >= PARAMETERS['minimum_incremental_information']
                        and c['values']['offset'][k] > PARAMETERS['numerical_noop_atol']):
                    candidates.append((info - bo_info, -gap, -context['tie_priority'][j], int(j), key, info, proposed[int(j)]))
        if candidates:
            winner = max(candidates, key=lambda x: x[:3])
            selected, selection = winner[3], 'optional_verification'
            c = calculations[winner[4]]
            verification = dict(id=winner[4][0], version=winner[4][1],
                incremental_information=winner[0], information=winner[5],
                reference_id=c['probes'][selected]['reference_id'],
                reference_y=c['probes'][selected]['reference_y'],
                remaining_executable_probes=len(c['indices']) - 1,
                information_kind='binary_soft_sign', minimum_unqueried_endpoints=1,
                proposal=deepcopy(winner[6]),
                reference_changed_from_proposal=c['probes'][selected]['reference_id'] != winner[6]['proposed_reference_id'])
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
        BO_reference_id=pool.ids[bo], mode=selection, mechanism='anchor_validation_v6',
        base_logei=float(base[selected]), reference_logei=float(base[bo]), combined_logei=float(combined[selected]),
        opportunity_gap=float(base[bo] - base[selected]), epsilon=context['epsilon'],
        eligible_count=int(context['eligible'].sum()), coverage=rows, reviews=reviews,
        hypotheses_at_commit=hypotheses, shadows=shadows, proposal=proposal, menu_ids=list(menu_ids),
        validation_proposals=validation_proposals,
        verification=verification, GP_fit_key=stats['fit_key'], GP_acquisition_meta=context['acquisition_meta'],
        deployed_versions=[dict(id=r['id'], version=r['version'], weight=r['weight'], effective_mixture_weight=w)
                           for r, _, w in deployed], selected_mixture=mixture)


def feedback(state, commit, observation):
    # The original state remains intact even when a malformed proposal ledger,
    # invalid feedback, or copying a late field fails after rule calculations.
    updated = deepcopy(state)
    transfer.feedback(updated, commit, observation)
    updated['validation_proposals'] = deepcopy(commit['validation_proposals'])
    state.clear()
    state.update(updated)
