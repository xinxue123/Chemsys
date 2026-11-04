"""Prospective soft-preference evidence and bounded selection on a shared GP."""
from copy import deepcopy
from functools import lru_cache
from math import log, sqrt
import numpy as np
from scipy.special import log_ndtr, logsumexp

from . import knowledge_policy as policy
from .knowledge_llm import digest

PARAMETERS = dict(preference_noise_sd=.06, maximum_deployment_weight=.5,
    maximum_logEI_shift=log(4), maximum_optional_verifications=1,
    verification_maximum_gap=log(1.1), last_verification_round=10,
    minimum_incremental_information=.01, minimum_remaining_scope_candidates=2,
    GH_nodes=64, numerical_noop_atol=1e-12, call_limit=4, minimum_call_gap=3, late_call_round=11)


@lru_cache(maxsize=1)
def warm():
    nodes, weights = np.polynomial.hermite.hermgauss(PARAMETERS['GH_nodes'])
    return nodes, np.log(weights / sqrt(np.pi))


def information(predvar, contrastvar, cov, tau=.06):
    vz = contrastvar + tau*tau
    rem = max(vz - cov*cov / max(predvar, 1e-18), 1e-15*vz)
    return .5*log(vz/rem)


def conditional_scores(gp, context, h, js, history):
    """One-sign EI ratio, referenced to unchanged native MC512 qLogEI."""
    stats = context['stats']
    md, vd, covs = gp.contrast(stats, h['plus_id'], h['minus_id'], js)
    vz = vd + PARAMETERS['preference_noise_sd']**2
    prior = log_ndtr(md/sqrt(vz))
    nodes, logw = warm()
    best = max(row['y'] for row in history)
    scores = np.array(context['base_logei'], copy=True)
    for j, cov in zip(js, covs):
        vx = max(float(stats['sd_y'][j])**2, 1e-18)
        draws = stats['mu_y'][j] + sqrt(2*vx)*nodes
        improvement = np.maximum(draws-best, 0.)
        positive = improvement > 0.
        if not positive.any():
            continue
        common = logw[positive] + np.log(improvement[positive])
        rem = max(vz-cov*cov/vx, 1e-15*vz)
        conditional = (md + cov/vx*(draws[positive]-stats['mu_y'][j]))/sqrt(rem)
        offset = logsumexp(common+log_ndtr(conditional))-logsumexp(common)-prior
        if abs(offset) <= PARAMETERS['numerical_noop_atol']:
            offset = 0.
        scores[j] += float(np.clip(offset, -PARAMETERS['maximum_logEI_shift'], PARAMETERS['maximum_logEI_shift']))
    return scores, (md, vd, covs)


def choose(scores, context):
    js = np.flatnonzero(context['eligible'])
    js = js[scores[js] == scores[js].max()]
    return int(js[np.argmin(context['tie_priority'][js])])


def coverage(pool, gp, context, hypotheses, history):
    """Assess every current version before ranking hypotheses by credibility."""
    rows, calculations = [], {}
    eligible = np.flatnonzero(context['eligible'])
    for h in hypotheses:
        if h['status'] == 'retracted':
            continue
        js = policy.scope_indices(pool, h, eligible)
        row = dict(id=h['id'], version=h['version'], weight=policy.weight(h),
                   eligible_inside_scope=len(js), can_change_rank_at_cap=False,
                   eligibility_state='dormant_no_coverage' if not len(js) else 'covered')
        if len(js):
            scores, moments = conditional_scores(gp, context, h, js, history)
            offset = scores-context['base_logei']
            correction = np.logaddexp(log(.5), log(.5)+offset)
            correction[np.abs(offset) <= PARAMETERS['numerical_noop_atol']] = 0.
            capped = context['base_logei'] + correction
            row['can_change_rank_at_cap'] = choose(capped, context) != context['bo_index']
            if not row['can_change_rank_at_cap']:
                row['eligibility_state'] = 'covered_without_current_rank_effect'
            calculations[(h['id'], h['version'])] = (js, scores, moments)
        rows.append(row)
    return rows, calculations


def should_call(state, t, rows):
    if len(state['calls']) >= PARAMETERS['call_limit']:
        return False
    if not state['calls']:
        return True
    if t - state['calls'][-1]['query'] < PARAMETERS['minimum_call_gap']:
        return False
    if t < PARAMETERS['late_call_round'] and len(state['calls']) >= 3:
        return False
    if t >= PARAMETERS['late_call_round'] and not any(c['query'] >= PARAMETERS['late_call_round'] for c in state['calls']):
        return True
    no_live_support = not any(r['can_change_rank_at_cap'] for r in rows)
    recent = state['decisions'][-3:]
    stalled = len(recent) == 3 and recent[-1]['incumbent_after_y'] <= recent[0]['incumbent_before_y'] + 1e-12
    negative = any(e.get('delta_log_score', 0) is not None and e.get('delta_log_score', 0) < -.1
                   for d in state['decisions'][-2:] for e in d['evidence'])
    return no_live_support or stalled or negative


def plan(pool, gp, state, t, context, proposal=None, menu_ids=(), mode='full'):
    if mode not in ('bo', 'shadow', 'deployment', 'full'):
        raise ValueError('unknown arm')
    hypotheses = deepcopy(state['hypotheses'])
    reviews = []
    if proposal is not None:
        hypotheses, reviews = policy.review_and_apply(pool, hypotheses, proposal, menu_ids, t)
    hypotheses, coverage_events = policy.refresh_coverage(pool, hypotheses, np.flatnonzero(context['eligible']), t)
    reviews.extend(coverage_events)
    rows, calculations = coverage(pool, gp, context, hypotheses, state['history'])
    base = context['base_logei']
    bo = context['bo_index']
    selected, selection = bo, 'base_bo'
    combined = np.array(base, copy=True)
    deployed = []
    if mode in ('deployment', 'full'):
        for row in rows:
            key = (row['id'], row['version'])
            if row['weight'] > 0 and row['eligible_inside_scope'] and key in calculations:
                deployed.append((row, calculations[key][1]))
        total = sum(r['weight'] for r, _ in deployed)
        if total:
            scale = min(1., .5/total)
            weights = [r['weight']*scale for r, _ in deployed]
            offsets = np.stack([scores-base for _, scores in deployed])
            correction = logsumexp(np.stack([np.full_like(base, log(1-sum(weights)))] +
                    [log(w)+offset for w, offset in zip(weights, offsets)]), axis=0)
            correction[np.all(np.abs(offsets) <= PARAMETERS['numerical_noop_atol'], axis=0)] = 0.
            combined = base + correction
            selected = choose(combined, context)
            if selected != bo:
                selection = 'knowledge_guided'
    verification = None
    if mode == 'full' and selected == bo and t <= PARAMETERS['last_verification_round'] and state['verification_used'] < PARAMETERS['maximum_optional_verifications']:
        candidates = []
        for row in rows:
            key = (row['id'], row['version'])
            if row['weight'] > 0 or not row['can_change_rank_at_cap'] or row['eligible_inside_scope'] < 2:
                continue
            h = next(h for h in hypotheses if (h['id'], h['version']) == key)
            # Do not spend a new validation on already refuted versions.
            effective = h.get('ledger', {}).get('score', 0.) + sum(min(0., e['delta_log_score'])
                for e in h.get('inherited_negative_evidence', []) if e['applicable'])
            if effective < 0:
                continue
            js, _, moments = calculations[key]
            md, vd, covs = moments
            _, _, bo_cov = gp.contrast(context['stats'], h['plus_id'], h['minus_id'], [bo])
            bo_info = information(context['stats']['sd_predictive_y'][bo]**2, vd, float(bo_cov[0]))
            for j, cov in zip(js, covs):
                gap = float(base[bo]-base[j])
                info = information(context['stats']['sd_predictive_y'][j]**2, vd, float(cov))
                if j != bo and gap <= min(context['epsilon'], PARAMETERS['verification_maximum_gap']) and info-bo_info >= .01:
                    candidates.append((info-bo_info, -gap, -context['tie_priority'][j], int(j), key, info))
        if candidates:
            best = max(candidates, key=lambda c: c[:3])
            selected = best[3]
            selection = 'optional_verification'
            verification = {'id': best[4][0], 'version': best[4][1], 'incremental_information': best[0],
                            'information': best[5], 'remaining_scoped_candidate_minimum': 1}
    shadows = []
    for h in hypotheses:
        if h['status'] == 'retracted':
            continue
        md, vd, cs = gp.contrast(context['stats'], h['plus_id'], h['minus_id'], [selected])
        pred = policy.shadow(h, pool.ids[selected], t,
            float(context['stats']['mu_y'][selected]), float(context['stats']['sd_predictive_y'][selected])**2,
            md, vd, float(cs[0]))
        info = information(context['stats']['sd_predictive_y'][selected]**2, vd, float(cs[0]))
        scoped = len(policy.scope_indices(pool, h, [selected])) == 1
        shadows.append({'id': h['id'], 'version': h['version'], 'prediction': pred,
                        'information_nats': info, 'informative': bool(scoped and info >= .01)})
    return dict(query=t, history_sha256=digest(state['history']), selected_id=pool.ids[selected],
        BO_reference_id=pool.ids[bo], mode=selection, base_logei=float(base[selected]),
        reference_logei=float(base[bo]), combined_logei=float(combined[selected]),
        opportunity_gap=float(base[bo]-base[selected]), epsilon=context['epsilon'],
        eligible_count=int(context['eligible'].sum()), coverage=rows, reviews=reviews,
        hypotheses_at_commit=hypotheses, shadows=shadows, proposal=proposal,
        menu_ids=list(menu_ids), verification=verification,
        GP_fit_key=context['stats']['fit_key'], GP_acquisition_meta=context['acquisition_meta'],
        deployed_versions=[{'id': r['id'], 'version': r['version'], 'weight': r['weight']} for r, _ in deployed])


def feedback(state, commit, observation):
    if commit['query'] != len(state['decisions']) + 1:
        raise ValueError('Feedback query is out of sequence')
    if commit['selected_id'] != observation['candidate_id'] or digest(state['history']) != commit['history_sha256']:
        raise ValueError('Feedback does not match committed prefix and experiment')
    if observation['candidate_id'] in {h['candidate_id'] for h in state['history']}:
        raise ValueError('Duplicate experiment')
    updated = deepcopy(state)
    updated['hypotheses'] = deepcopy(commit['hypotheses_at_commit'])
    receipts = []
    for row in commit['shadows']:
        h = next(h for h in updated['hypotheses'] if h['id'] == row['id'] and h['version'] == row['version'])
        receipts.append(policy.apply_evidence(h, row['prediction'], observation['y'], row['informative']))
    before = max(h['y'] for h in state['history'])
    updated['history'].append(observation)
    updated['verification_used'] += int(commit['mode'] == 'optional_verification')
    updated['decisions'].append(dict(query=commit['query'], selected_id=commit['selected_id'],
        BO_reference_id=commit['BO_reference_id'], mode=commit['mode'], evidence=receipts,
        raw_value=observation['raw_value'], y=observation['y'],
        incumbent_before_y=before, incumbent_after_y=max(before, observation['y']),
        commit_sha256=digest(commit)))
    updated['status'] = 'COMPLETE' if len(updated['decisions']) == 16 else 'RUNNING'
    state.clear()
    state.update(updated)
