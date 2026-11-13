"""R6 adapter: actual finite-pool GP, qualitative hypotheses, prospective review.

No oracle is imported here.  Existing R5/R4/R2 files are never modified.
The caller must write the full returned commitment *before* revealing y.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from functools import lru_cache
from hashlib import sha256
import json
from math import log, sqrt
from random import Random
import numpy as np
from scipy.special import log_ndtr, logsumexp
from scipy.spatial import cKDTree

from . import continuous_r6_core as core


def digest(value):
    """Match the original canonical SHA256 JSON serialization."""
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                             separators=(',', ':')).encode('utf-8')).hexdigest()


ARMS = ('A0', 'A1', 'A2', 'A3', 'A4', 'A4_no_exception')
CALL_ROUNDS = (1, 3, 6, 9)
PREFERENCE_NOISE_Y = 0.06  # Both tasks use predeclared engineering-scaled y.
MAX_MENU = 24
CONTRAST_SCOPE_RADIUS = 0.25  # In normalized physical axes.
MAX_PREFERENCE_LOG_EI_SHIFT = log(4)


def _arm(state):
    arm = state['arm']
    if arm not in ARMS:
        raise ValueError('Unknown R6 arm: ' + str(arm))
    return arm


def _candidate_rows(cache, d):
    pool = cache.pool
    ids = pool['ids']
    return [core.Candidate(
        str(ids[j]), tuple(float(z) for z in pool['z'][j]), float(d['base'][j]),
        float(d['information_proxy'][j]), float(d['stats']['sd_y'][j]),
        -float(d['tie_priority'][j]))
        for j in np.flatnonzero(d['available'])]


def _meaningful_pair(a, b):
    """A local, non-identical continuous pair with <=2 major changed axes."""
    changes = np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))
    return bool(np.any(changes > 1e-8) and np.count_nonzero(changes > .25 + 1e-12) <= 2)


def _paired_menu(pool, preliminary, max_items, seed, required_id=None):
    """Ensure that a diverse menu also contains an executable *local pair*.

    A farthest-first menu alone tends to show only far-apart points, making
    nearly all LLM direction/background contracts semantically impossible.
    Membership of `pool` is entirely feature-only in blind A1 and the
    *explicitly* GP-filtered eligible subset in A2+.
    """
    if len(pool) < 2:
        return ()
    if any(_meaningful_pair(a.features, b.features)
           for i, a in enumerate(preliminary) for b in preliminary[i + 1:]):
        return preliminary
    x = np.asarray([p.features for p in pool], dtype=float)
    tree = cKDTree(x)
    k = min(len(pool), 24)
    _, near = tree.query(x, k=k)
    near = np.asarray(near).reshape(len(pool), k)
    priorities = []
    for i in range(len(pool)):
        for j in near[i, 1:]:
            j = int(j)
            if i < j and _meaningful_pair(x[i], x[j]):
                sep = float(np.linalg.norm(x[i] - x[j]))
                # Avoid near-identical pairs that cannot have a useful effect.
                priorities.append((sep < .05, abs(sep - .16), i, j))
    if not priorities:
        return ()
    _, _, a, b = min(priorities)
    required = [pool[a].point_id, pool[b].point_id]
    if required_id is not None:
        required.append(required_id)
    selected = dict((p.point_id, p) for p in preliminary)
    all_points = {p.point_id: p for p in pool}
    for point_id in required:
        selected[point_id] = core.MenuPoint(point_id, all_points[point_id].features)
    while len(selected) > max_items:
        # Keep the required physical pair and, in filtered mode, the BO point.
        unwanted = [v for key, v in selected.items() if key not in required]
        if not unwanted:
            break
        selected.pop(unwanted[-1].point_id)
    output = list(selected.values())
    Random(seed + 712).shuffle(output)
    return tuple(output)


def build_context(cache, state, t):
    """Use the existing exact GP, original MC qLogEI, and original tie rule."""
    from . import continuous_benchmark_engine as original
    _arm(state)
    shadow_state = dict(state, method='Pure_BO', knowledge={'confidence': .7, 'clauses': []})
    d = original.setup(cache, shadow_state, t)
    pool = cache.pool
    candidates = _candidate_rows(cache, d)
    gate = core.Gate(
        str(pool['ids'][d['reference']]),
        frozenset(str(pool['ids'][j]) for j in np.flatnonzero(d['eligible'])),
        float(d['base'][d['reference']]), float(d['budget']))
    arm = _arm(state)
    menu = ()
    if arm != 'A0' and t in CALL_ROUNDS and len(state['calls']) < len(CALL_ROUNDS):
        types = ['continuous'] * len(pool['axes'])
        seed = int(state['seed']) * 1000 + t + (0 if arm == 'A1' else 200)
        if arm == 'A1':
            raw = core.independent_llm_menu(candidates, types, MAX_MENU, seed)
            menu = _paired_menu(candidates, raw, MAX_MENU, seed)
        else:
            raw = core.feasible_llm_menu(candidates, gate, types, MAX_MENU, seed)
            eligible_candidates = [c for c in candidates if c.point_id in gate.eligible_point_ids]
            menu = _paired_menu(eligible_candidates, raw, MAX_MENU, seed, gate.bo_point_id)
    if len(menu) < 2:
        menu = ()
    return {'d': d, 'candidates': candidates, 'gate': gate, 'menu': menu}


def public_menu(cache, state, t, context):
    """No GP means, uncertainty, qLogEI, BO label, or future outcomes."""
    pool = cache.pool
    ix = cache.ix
    return {
        'task_description': pool['description'], 'objective_name': pool['objective_name'],
        'objective_unit': pool['objective_unit'], 'bounds': pool['bounds'],
        'round': t, 'remaining_queries': 17 - t,
        'method': 'GP-blind physical menu' if _arm(state) == 'A1' else 'GP-filtered physical menu; scores hidden',
        'own_observations': [
            {'point': pool['points'][ix[h['candidate_id']]], 'objective': float(h['raw_value'])}
            for h in state['history']],
        'candidate_menu': [
            {'candidate_id': row.point_id, 'point': pool['points'][ix[row.point_id]]}
            for row in context['menu']],
        'prior_claims': [
            {'hypothesis_id': h['id'], 'better_id': h['better_id'], 'worse_id': h['worse_id'],
             'focus_directions': h['directions'], 'scope_radius': CONTRAST_SCOPE_RADIUS}
            for h in state['hypotheses']],
    }


def _ledger_from_json(h):
    x = h['ledger']
    return core.EvidenceLedger(
        hypothesis_id=h['id'], creation_round=int(h['created_round']),
        prior_model_weight=x['prior'], max_deployment_weight=x['cap'],
        cumulative_log_score_gain=x['score'], seen_point_ids=set(x['seen']),
        event_log=[tuple(r) for r in x['events']])


def _ledger_to_json(ledger):
    return {'prior': ledger.prior_model_weight, 'cap': ledger.max_deployment_weight,
            'score': ledger.cumulative_log_score_gain,
            'seen': sorted(ledger.seen_point_ids), 'events': list(ledger.event_log),
            'deployment_weight': ledger.deployment_weight}


def _review_proposal(cache, state, t, context, proposal):
    """All comparisons must cite two real displayed, unqueried conditions."""
    if proposal is None or proposal.get('action') == 'abstain':
        return {'accepted': False, 'reason': 'abstain', 'hypothesis_id': None,
                'nominee_id': None}
    if proposal.get('action') != 'compare':
        return {'accepted': False, 'reason': 'unknown action', 'hypothesis_id': None,
                'nominee_id': None}
    directions = proposal.get('focus_directions')
    if not isinstance(directions, dict):
        return {'accepted': False, 'reason': 'invalid directions', 'hypothesis_id': None,
                'nominee_id': None}
    axes = cache.pool['axes']
    if set(directions) - set(axes) or len(directions) not in (1, 2):
        return {'accepted': False, 'reason': 'unknown focus axes', 'hypothesis_id': None,
                'nominee_id': None}
    choices = {'increase': 1, 'decrease': -1}
    if any(value not in choices for value in directions.values()):
        return {'accepted': False, 'reason': 'invalid direction value', 'hypothesis_id': None,
                'nominee_id': None}
    cmp = core.Comparison(
        str(proposal.get('better_id')), str(proposal.get('worse_id')),
        {axes.index(k): choices[v] for k, v in directions.items()})
    review = core.review_comparison(cmp, context['menu'], ['continuous'] * len(axes))
    if not review.passed:
        return {'accepted': False, 'reason': review.reason, 'hypothesis_id': None,
                'nominee_id': None}
    for prior in state['hypotheses']:
        pair = (prior['better_id'], prior['worse_id'])
        if pair == (cmp.worse_point_id, cmp.better_point_id):
            return {'accepted': False, 'reason': 'opposing existing pair', 'hypothesis_id': None,
                    'nominee_id': None}
        if pair == (cmp.better_point_id, cmp.worse_point_id):
            return {'accepted': False, 'reason': 'duplicate existing pair', 'hypothesis_id': None,
                    'nominee_id': None}
    hid = f'H{len(state["hypotheses"]) + 1:02d}'
    state['hypotheses'].append({
        'id': hid, 'created_round': t, 'better_id': cmp.better_point_id,
        'worse_id': cmp.worse_point_id, 'directions': dict(directions),
        'tau_y': PREFERENCE_NOISE_Y, 'scope_radius': CONTRAST_SCOPE_RADIUS,
        'ledger': _ledger_to_json(core.EvidenceLedger(hid, t))})
    return {'accepted': True, 'reason': review.reason,
            'hypothesis_id': hid, 'nominee_id': cmp.better_point_id}


def _scope_indices(cache, h, candidate_indices):
    """Never deploy a local statement arbitrarily across the global pool."""
    pool = cache.pool
    z = np.asarray(pool['z'], dtype=float)
    a, b = z[cache.ix[h['better_id']]], z[cache.ix[h['worse_id']]]
    focus = {pool['axes'].index(k) for k in h['directions']}
    js = np.asarray(candidate_indices, dtype=int)
    keep = np.ones(len(js), dtype=bool)
    for i in range(z.shape[1]):
        if i in focus:
            keep &= np.abs(z[js, i] - a[i]) <= h['scope_radius'] + 1e-12
        else:
            keep &= np.maximum(np.abs(z[js, i] - a[i]), np.abs(z[js, i] - b[i])) <= h['scope_radius'] + 1e-12
    return js[keep]


def contrast_moments(cache, stats, h, indices, chunk=128):
    """Read *joint latent* GP covariances without materializing an N-by-N pool matrix.

    The GP is trained on standardized y, so scale every covariance by std^2.
    Predictive variance for observed y comes from original model(noise=True).
    """
    import torch  # Lazy: pure R6-core tests need no BoTorch/Torch installation.
    js = np.asarray(indices, dtype=int)
    plus, minus = cache.ix[h['better_id']], cache.ix[h['worse_id']]
    m = stats['model']
    out = np.empty(len(js), dtype=float)
    var_d = None
    scale2 = float(stats['std']) ** 2
    with torch.no_grad():
        for start in range(0, len(js), chunk):
            part = js[start:start + chunk]
            xx = torch.cat([cache.X[part], cache.X[[plus, minus]]], dim=0)
            cov = m.posterior(xx).mvn.covariance_matrix.detach().cpu().numpy() * scale2
            n = len(part)
            out[start:start + n] = cov[:n, n] - cov[:n, n + 1]
            vd = max(0., float(cov[n, n] + cov[n + 1, n + 1] - 2 * cov[n, n + 1]))
            if var_d is not None and not np.isclose(var_d, vd, atol=1e-7, rtol=1e-6):
                raise AssertionError('GP contrast variance inconsistent across chunks')
            var_d = vd
    if var_d is None:
        # Needed when no candidate passes scope; anchor pair always exists.
        with torch.no_grad():
            cov = m.posterior(cache.X[[plus, minus]]).mvn.covariance_matrix.detach().cpu().numpy() * scale2
        var_d = max(0., float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]))
    mean_d = float(stats['mu_y'][plus] - stats['mu_y'][minus])
    return mean_d, var_d, out


@lru_cache(maxsize=1)
def _hermite():
    nodes, weights = np.polynomial.hermite.hermgauss(64)
    return nodes, np.log(weights / sqrt(np.pi))


def guided_logei(cache, context, h, indices, moments, history):
    """q=1 EI from the exact one-sign skew-normal preference posterior.

    64-node Gaussian-Hermite quadrature estimates the conditional/base EI
    ratio; applying that log-ratio to the original MC qLogEI prevents an
    unvalidated baseline change from analytic-versus-MC noise.
    """
    js = np.asarray(indices, dtype=int)
    d = context['d']
    stats = d['stats']
    mu_d, vd, covs = moments
    tau = h['tau_y']
    vz = vd + tau * tau
    base_preference_logprob = float(log_ndtr(mu_d / sqrt(vz)))
    nodes, log_weights = _hermite()
    best = max(float(row['y']) for row in history)  # Observation-only.
    result = {}
    for j, c in zip(js, covs):
        vx = max(float(stats['sd_y'][j]) ** 2, 1e-18)
        rem = max(vz - c * c / vx, 1e-15 * vz)
        y = float(stats['mu_y'][j]) + sqrt(2 * vx) * nodes
        improvement = np.maximum(y - best, 0.)
        positive = improvement > 0.
        if not positive.any():
            result[str(cache.pool['ids'][j])] = float(d['base'][j])
            continue
        common = log_weights[positive] + np.log(improvement[positive])
        conditional = (mu_d + c / vx * (y[positive] - stats['mu_y'][j])) / sqrt(rem)
        delta = float(logsumexp(common + log_ndtr(conditional))
                      - logsumexp(common) - base_preference_logprob)
        delta = float(np.clip(delta, -MAX_PREFERENCE_LOG_EI_SHIFT, MAX_PREFERENCE_LOG_EI_SHIFT))
        result[str(cache.pool['ids'][j])] = float(d['base'][j]) + delta
    return result


def _shadow_for(cache, stats, h, j, t, moments):
    md, vd, cov = moments
    return core.ShadowPrediction(
        h['id'], str(cache.pool['ids'][j]), t,
        float(stats['mu_y'][j]), float(stats['sd_predictive_y'][j]) ** 2,
        md, vd, float(cov), h['tau_y'])


def plan(cache, state, t, context, proposal=None):
    """Select an input without touching the oracle; produce JSON-safe commitment."""
    arm = _arm(state)
    d, candidates, gate = context['d'], context['candidates'], context['gate']
    if proposal is not None and not context['menu']:
        raise ValueError('Proposal supplied without an authorized menu')
    review = _review_proposal(cache, state, t, context, proposal)
    if proposal is not None:
        state['calls'].append({'round': t, 'proposal': proposal, 'review': review,
                               'menu_ids': [m.point_id for m in context['menu']],
                               'GP_filtered_membership': arm != 'A1'})
    pool = cache.pool
    by_id = {c.point_id: c for c in candidates}
    decision = core.Decision(gate.bo_point_id, 'base_bo', 0.)
    direct = None
    if arm in ('A1', 'A2', 'A3') and proposal is not None:
        direct = core.choose_llm_rerank(
            candidates, gate, context['menu'],
            [review['nominee_id']] if review['accepted'] else [],
            core.Review(review['accepted'], review['reason']))
        if direct.mode in ('direct_LLM_rerank', 'LLM_agreed_with_BO'):
            decision = direct

    # Identify one local, reviewable hypothesis. All other hypotheses still
    # accumulate their *own* prospective evidence at an informative query.
    active = sorted(state['hypotheses'],
                    key=lambda h: (-_ledger_from_json(h).deployment_weight, -h['created_round'], h['id']))
    preference_model = active[0] if active else None
    verification = None
    guided = None
    information_proxy = None
    if preference_model is not None and arm in ('A3', 'A4', 'A4_no_exception'):
        h = preference_model
        available = np.flatnonzero(d['available'])
        max_gap = gate.epsilon + (0 if arm == 'A4_no_exception' else log(2))
        candidate_js = available[d['base'][gate_index(cache, gate)] - d['base'][available] <= max_gap + 1e-12]
        candidate_js = np.unique(np.r_[candidate_js, cache.ix[gate.bo_point_id]])
        scoped = _scope_indices(cache, h, candidate_js)
        candidate_js = np.unique(np.r_[scoped, cache.ix[gate.bo_point_id]])
        md, vd, cs = contrast_moments(cache, d['stats'], h, candidate_js)
        probes = {
            str(pool['ids'][j]): core.ContrastProbe(
                str(pool['ids'][j]), float(d['stats']['sd_predictive_y'][j]) ** 2,
                vd, float(c), h['tau_y'])
            for j, c in zip(candidate_js, cs)}
        policy = core.VerificationPolicy(max_exceptions=0 if arm == 'A4_no_exception' else 1)
        # First seek a useful normal-gate probe. Spend an explicitly budgeted
        # exception only when no useful normal-gate probe exists.
        strict = core.VerificationPolicy(max_exceptions=0)
        verification = core.choose_active_verification(
            candidates, gate, probes, t, int(state['validation_queries_used']),
            int(state['exceptions_used']), strict)
        if verification is None and policy.max_exceptions:
            verification = core.choose_active_verification(
                candidates, gate, probes, t, int(state['validation_queries_used']),
                int(state['exceptions_used']), policy)
        information_proxy = {key: core.information_about_contrast(p) for key, p in probes.items()}
        if verification is not None:
            decision = verification
        elif arm in ('A4', 'A4_no_exception'):
            weight = _ledger_from_json(h).deployment_weight
            if weight > 0:
                js = _scope_indices(cache, h, np.flatnonzero(d['eligible']))
                if len(js):
                    gmom = contrast_moments(cache, d['stats'], h, js)
                    g = guided_logei(cache, context, h, js, gmom, state['history'])
                    # Outside the local scope, preserve the exact original GP score.
                    g.update({str(pool['ids'][j]): float(d['base'][j])
                              for j in np.flatnonzero(d['eligible']) if str(pool['ids'][j]) not in g})
                    guided = core.choose_guided_inside_gate(candidates, gate, g, weight)
                    decision = guided

    j = cache.ix[decision.point_id]
    shadows = []
    for h in state['hypotheses']:
        md, vd, cs = contrast_moments(cache, d['stats'], h, [j])
        pred = _shadow_for(cache, d['stats'], h, j, t, (md, vd, cs[0]))
        info = core.information_about_contrast(core.ContrastProbe(
            decision.point_id, pred.gp_var_y, vd, float(cs[0]), h['tau_y']))
        # Material prospective evidence only when the point tests a stated
        # physical scope; record skipped predictions explicitly for audit.
        in_scope = len(_scope_indices(cache, h, [j])) == 1
        shadows.append({'prediction': asdict(pred), 'information_nats': info,
                        'informative': bool(info >= .01 and in_scope)})

    if decision.mode == 'active_verification':
        state['validation_queries_used'] += 1
        if decision.used_exception:
            state['exceptions_used'] += 1
    return {
        'iteration': t, 'task': pool['task'], 'arm': arm, 'history_sha256': digest(state['history']),
        'selected_id': decision.point_id, 'BO_reference_id': gate.bo_point_id,
        'selection_mode': decision.mode, 'opportunity_gap': float(decision.acquisition_gap),
        'normal_budget': gate.epsilon, 'normal_gate_eligible': decision.point_id in gate.eligible_point_ids,
        'used_exception': decision.used_exception, 'verification_information_nats': decision.information_gain_nats,
        'eligible_count': len(gate.eligible_point_ids), 'menu_ids': [m.point_id for m in context['menu']],
        'review': review, 'proposal': proposal, 'direct_candidate': asdict(direct) if direct else None,
        'verification_candidate': asdict(verification) if verification else None,
        'guided_candidate': asdict(guided) if guided else None,
        'verification_information_proxy': information_proxy, 'prequery_shadows': shadows,
        'hypotheses_at_commit': deepcopy(state['hypotheses']),
        'calls_at_commit': deepcopy(state['calls']),
        'budgets_at_commit': {'active_verifications': state['validation_queries_used'],
                              'exceptions': state['exceptions_used']},
        'GP_fit_key': d['stats']['fit_key'], 'GP_fit_seed': d['stats']['fit_seed'],
        'GP_reference_logEI': gate.best_logei, 'GP_selected_logEI': float(d['base'][j]),
        'GP_mean_y': float(d['stats']['mu_y'][j]),
        'GP_predictive_SD_y': float(d['stats']['sd_predictive_y'][j]),
        'GP_acquisition_meta': d['base_meta'],
        'comparison_schema_version': 'R6_qualitative_v1',
    }


def gate_index(cache, gate):
    return cache.ix[gate.bo_point_id]


def apply_feedback(state, commit, observed):
    """Called only after the prequery commitment was persisted."""
    if observed['candidate_id'] != commit['selected_id']:
        raise ValueError('Outcome ID differs from committed selection')
    t = commit['iteration']
    if len(state['decisions']) + 1 != t:
        raise ValueError('Feedback applied out of order')
    receipts = []
    for row in commit['prequery_shadows']:
        pred = core.ShadowPrediction(**row['prediction'])
        h = next(h for h in state['hypotheses'] if h['id'] == pred.hypothesis_id)
        ledger = _ledger_from_json(h)
        old = ledger.deployment_weight
        delta = ledger.observe(pred, float(observed['y'])) if row['informative'] else None
        h['ledger'] = _ledger_to_json(ledger)
        receipts.append({'hypothesis_id': h['id'], 'informative': row['informative'],
                         'information_nats': row['information_nats'], 'delta_log_score': delta,
                         'weight_before': old, 'weight_after': ledger.deployment_weight})
    state['decisions'].append({
        'iteration': t, 'selected_id': commit['selected_id'],
        'BO_reference_id': commit['BO_reference_id'], 'execution_mode': commit['selection_mode'],
        'used_exception': commit['used_exception'], 'opportunity_gap': commit['opportunity_gap'],
        'raw_value': float(observed['raw_value']), 'y': float(observed['y']),
        'evidence': receipts, 'decision_commit_sha256': digest(commit)})
    state['history'].append(observed)
    state['status'] = 'COMPLETE' if len(state['decisions']) == 16 else 'RUNNING'
    return receipts
