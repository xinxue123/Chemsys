"""V4: persistent local rules, prospective new-plus probes, and dual evidence.

Only an executable unqueried plus receives a nonnegative acquisition correction.
Every such correction has a frozen, density-normalized predictor at that plus.
The known reference is never penalized to make an unsupported third point win.
"""
from copy import deepcopy
from math import log

import numpy as np
from scipy.special import logsumexp

from . import knowledge_engine as legacy
from . import knowledge_rule_policy as policy
from . import knowledge_rule_math as math
from .knowledge_llm import digest


PARAMETERS = dict(legacy.PARAMETERS, minimum_remaining_scope_candidates=1,
    rule_reliability_prior_successes=1, rule_reliability_prior_failures=1,
    score_policy='prospective tempered expert density, not fixed-event posterior odds',
    reliability_policy='min(cap, positive tanh(score/2), positive beta-mean excess); no increase after failure or nonpositive score',
    probe_reference='highest observed compatible point; physical-feature tie break',
    probe_effect='new plus only; never suppress the known reference',
    verification_information='binary soft-sign mutual information by GH64',
    EI_ratio='GH64 over truncated sign variable with analytic conditional normal EI',
    EI_cap='normalized GP/conditional density tempering, then expert mixture')


def warm():
    # This module has no JIT. The math module caches the fixed quadrature once.
    math.vector_scores(np.array([0.]), np.array([1.]), np.array([1.]),
                       np.array([0.]), 0.)


choose = legacy.choose
should_call = legacy.should_call


def coverage(pool, gp, context, hypotheses, history):
    rows, calculations = [], {}
    eligible = np.flatnonzero(context['eligible'])
    stats, base = context['stats'], context['base_logei']
    for h in hypotheses:
        if h['status'] == 'retracted':
            continue
        probes = policy.make_probes(pool, h, history, eligible)
        js = np.asarray(sorted(probes), dtype=int)
        row = dict(id=h['id'], version=h['version'], weight=policy.weight(h),
            eligible_inside_scope=len(policy.scope_indices(pool, h, eligible)),
            executable_probes=len(js), can_change_rank_at_cap=False,
            eligibility_state='no_known_comparable_reference' if not len(js) else 'executable_rule')
        if len(js):
            reference_y = np.asarray([probes[int(j)]['reference_y'] for j in js])
            values = math.vector_scores(np.asarray(stats['mu_y'])[js],
                np.square(np.asarray(stats['sd_y'])[js]),
                np.square(np.asarray(stats['sd_predictive_y'])[js]), reference_y,
                max(r['y'] for r in history), tau=PARAMETERS['preference_noise_sd'])
            offsets = np.zeros_like(base)
            offsets[js] = values['offset']
            if np.any(offsets < -1e-12):
                raise ValueError('Rule expert cannot suppress candidates')
            capped = base + np.logaddexp(log(.5), log(.5) + offsets)
            capped[np.abs(offsets) <= PARAMETERS['numerical_noop_atol']] = base[np.abs(offsets) <= PARAMETERS['numerical_noop_atol']]
            row['can_change_rank_at_cap'] = choose(capped, context) != context['bo_index']
            row['positive_effect_probes'] = int(np.count_nonzero(offsets[js] > PARAMETERS['numerical_noop_atol']))
            row['eligibility_state'] = 'executable_rank_effect' if row['can_change_rank_at_cap'] else 'executable_without_rank_effect'
            calculations[(h['id'], h['version'])] = dict(indices=js, probes=probes,
                positions={int(j): i for i, j in enumerate(js)}, values=values, offsets=offsets)
        rows.append(row)
    return rows, calculations


def _refresh(pool, gp, hypotheses, history, context, t):
    result = deepcopy(hypotheses)
    rows, calculations = coverage(pool, gp, context, result, history)
    by_key = {(r['id'], r['version']): r for r in rows}
    events = []
    for h in result:
        if h['status'] == 'retracted':
            continue
        row = by_key[(h['id'], h['version'])]
        status = 'active' if row['executable_probes'] else 'dormant'
        if h['status'] != status:
            events.append(dict(round=t, id=h['id'], version=h['version'],
                from_status=h['status'], to_status=status, reason=row['eligibility_state']))
        h['status'] = status
        h['coverage'] = dict(round=t, executable_probes=row['executable_probes'],
                             eligible_inside_scope=row['eligible_inside_scope'])
        h['ledger']['weight'] = policy.weight(h)
        row['weight'] = policy.weight(h)
    return result, events, rows, calculations


def prepare_state(pool, gp, state, context, t):
    prepared = deepcopy(state)
    hypotheses, events, _, _ = _refresh(pool, gp, prepared['hypotheses'], prepared['history'], context, t)
    prepared['hypotheses'] = hypotheses
    return prepared, events


def plan(pool, gp, state, t, context, proposal=None, menu_ids=(), mode='full'):
    if mode not in ('bo', 'shadow', 'deployment', 'full'):
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
    if (mode == 'full' and selected == bo and t <= PARAMETERS['last_verification_round']
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
            for k, j in enumerate(c['indices']):
                gap = float(base[bo] - base[j])
                info = float(c['values']['sign_information'][k])
                if (j != bo and gap <= min(context['epsilon'], PARAMETERS['verification_maximum_gap'])
                        and info - bo_info >= PARAMETERS['minimum_incremental_information']
                        and c['values']['offset'][k] > PARAMETERS['numerical_noop_atol']):
                    candidates.append((info - bo_info, -gap, -context['tie_priority'][j], int(j), key, info))
        if candidates:
            winner = max(candidates, key=lambda x: x[:3])
            selected, selection = winner[3], 'optional_verification'
            c = calculations[winner[4]]
            verification = dict(id=winner[4][0], version=winner[4][1],
                incremental_information=winner[0], information=winner[5],
                reference_id=c['probes'][selected]['reference_id'],
                reference_y=c['probes'][selected]['reference_y'],
                remaining_executable_probes=len(c['indices']) - 1,
                information_kind='binary_soft_sign', minimum_unqueried_endpoints=1)
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
        BO_reference_id=pool.ids[bo], mode=selection, mechanism='rule_probe_v4',
        base_logei=float(base[selected]), reference_logei=float(base[bo]), combined_logei=float(combined[selected]),
        opportunity_gap=float(base[bo] - base[selected]), epsilon=context['epsilon'],
        eligible_count=int(context['eligible'].sum()), coverage=rows, reviews=reviews,
        hypotheses_at_commit=hypotheses, shadows=shadows, proposal=proposal, menu_ids=list(menu_ids),
        verification=verification, GP_fit_key=stats['fit_key'], GP_acquisition_meta=context['acquisition_meta'],
        deployed_versions=[dict(id=r['id'], version=r['version'], weight=r['weight'], effective_mixture_weight=w)
                           for r, _, w in deployed], selected_mixture=mixture)


def feedback(state, commit, observation):
    if (commit['query'] != len(state['decisions']) + 1 or commit['selected_id'] != observation['candidate_id']
            or commit['history_sha256'] != digest(state['history'])):
        raise ValueError('Feedback does not match the committed round and history')
    if observation['candidate_id'] in {h['candidate_id'] for h in state['history']}:
        raise ValueError('Duplicate experiment')
    updated = deepcopy(state)
    updated['hypotheses'] = deepcopy(commit['hypotheses_at_commit'])
    receipts, scores = [], {}
    for s in commit['shadows']:
        h = next(h for h in updated['hypotheses'] if h['id'] == s['id'] and h['version'] == s['version'])
        # A reference must be in the committed own prefix, never a control result.
        reference = next(x for x in state['history'] if x['candidate_id'] == s['prediction']['reference_id'])
        if reference['y'] != s['prediction']['reference_y']:
            raise ValueError('Reference outcome was changed after prediction')
        receipt = policy.record_feedback(h, s['prediction'], observation)
        receipts.append(receipt)
        scores[s['prediction']['prediction_sha256']] = math.score_prediction(s['prediction'], observation['y'])
    mixture = commit['selected_mixture']
    total = sum(x['weight'] for x in mixture)
    deployed_gain = float(logsumexp([log(1 - total)] +
        [log(x['weight']) + scores[x['prediction_sha256']] for x in mixture])) if mixture else 0.
    before = max(h['y'] for h in state['history'])
    updated['history'].append(observation)
    updated['verification_used'] += int(commit['mode'] == 'optional_verification')
    updated['decisions'].append(dict(query=commit['query'], selected_id=commit['selected_id'],
        BO_reference_id=commit['BO_reference_id'], mode=commit['mode'], evidence=receipts,
        raw_value=observation['raw_value'], y=observation['y'],
        incumbent_before_y=before, incumbent_after_y=max(before, observation['y']),
        deployed_mixture_log_score_gain=deployed_gain, commit_sha256=digest(commit)))
    updated['status'] = 'COMPLETE' if len(updated['decisions']) == 16 else 'RUNNING'
    state.clear()
    state.update(updated)
