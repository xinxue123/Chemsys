"""Independent read-only V5 transfer-rule audit.

Reuse V4's already independently checked mathematical/GP/input helpers, never
production policy/engine replay. V5 scope, smooth recovery, verification control,
and actual transfer use are checked separately here. No API/oracle/fit access.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from math import exp, log, pi, sqrt, tanh
from pathlib import Path
import numpy as np
from scipy.special import log_ndtr, logsumexp
import audit_knowledge_rule_loop as v4
common = v4.common
ROOT = v4.ROOT
read, digest, filehash = v4.read, v4.digest, v4.filehash
require, close, exact = v4.require, v4.close, v4.exact
ATOL = v4.ATOL
hkey, contract, scope_mask = v4.hkey, v4.contract, v4.scope_mask
compatible, probes, evidence, weight = v4.compatible, v4.probes, v4.evidence, v4.weight
math_fields, score, adaptive_ei_check = v4.math_fields, v4.score, v4.adaptive_ei_check
select, calculations, refresh = v4.select, v4.calculations, v4.refresh
saved_gp_audit, inherited_records = v4.saved_gp_audit, v4.inherited_records
applicable_negative = v4.applicable_negative
candidate_menu, should_call = v4.candidate_menu, v4.should_call


def rule_geometry_audit(pool, h):
    focus = h['focus']
    background = h['background']
    require(isinstance(background, dict) and not set(background)-set(pool['axes']) and not set(background)&set(focus), 'background must name only real nonfocus axes')
    require(1 <= len(focus) <= 2 and set(focus) <= set(pool['axes']), 'invalid focus axes')
    require(h['scope']['background_radius'] == .25, 'background radius drift')
    a, b = h['plus_features'], h['minus_features']
    expected_scope, categorical = {}, {}
    for k, (axis, kind) in enumerate(zip(pool['axes'], pool['kinds'])):
        if kind == 'continuous':
            require(0 <= a[k] <= 1 and 0 <= b[k] <= 1, 'non-normalized continuous anchor')
            if axis in focus:
                spec = focus[axis]
                require(set(spec) == {'direction', 'bounds'} and spec['direction'] in ('increase', 'decrease'), 'continuous focus contract')
                sign = 1 if spec['direction'] == 'increase' else -1
                lo, hi = spec['bounds']
                require(0 <= lo < hi <= 1 and lo <= min(a[k], b[k]) <= max(a[k], b[k]) <= hi
                        and sign*(a[k]-b[k]) > 0, 'continuous focus direction/anchor bounds')
            else:
                require(abs(a[k]-b[k]) <= .25+ATOL, 'nonfocus continuous confounding distance')
                lo, hi = max(0., a[k]-.25, b[k]-.25), min(1., a[k]+.25, b[k]+.25)
            expected_scope[axis] = dict(kind=kind, bounds=[float(lo), float(hi)])
        else:
            if axis in focus:
                spec = focus[axis]
                require(set(spec) == {'direction', 'allowed'} and spec['direction'] == 'prefer_plus', 'categorical focus schema')
                allowed = spec['allowed']
                real = {r[k] for r in pool['features']}
                require(a[k] != b[k] and a[k] in allowed and b[k] in allowed
                        and len(set(allowed)) == len(allowed) and set(allowed) <= real, 'categorical preference real labels')
                require(allowed == sorted(allowed, key=digest), 'categorical canonical set order')
                categorical[axis] = dict(plus=a[k], minus=b[k])
            else:
                require(a[k] == b[k], 'nonfocus categorical confounding')
                allowed = [a[k]]
            expected_scope[axis] = dict(kind=kind, allowed=allowed)
    for axis, spec in background.items():
        k = pool['axes'].index(axis)
        kind = pool['kinds'][k]
        if kind == 'continuous':
            require(set(spec) == {'bounds'} and len(spec['bounds']) == 2, 'declared continuous background schema')
            lo, hi = spec['bounds']
            require(all(type(v) in (int, float) and np.isfinite(v) for v in (lo, hi))
                    and 0 <= lo < hi <= 1 and lo <= min(a[k], b[k]) <= max(a[k], b[k]) <= hi,
                    'declared background bounds must contain both origin anchors')
            expected_scope[axis] = dict(kind=kind, bounds=[float(lo), float(hi)])
        else:
            require(set(spec) == {'allowed'} and isinstance(spec['allowed'], list) and spec['allowed'], 'declared categorical background schema')
            labels = spec['allowed']
            require(all(isinstance(x, str) for x in labels) and len(set(labels)) == len(labels)
                    and set(labels) <= {row[k] for row in pool['features']}
                    and a[k] in labels and b[k] in labels and labels == sorted(labels, key=digest),
                    'declared background must use canonical real labels and contain both anchors')
            expected_scope[axis] = dict(kind=kind, allowed=labels)
    exact(h['scope']['axes'], expected_scope, 'independently reconstructed explicit/default rule scope')
    exact(h['categorical_preference'], categorical, 'categorical preference orientation')



def validate_hypotheses(pool, hyps, previous, prefix, t, menu_ids):
    mapping = {hkey(h): h for h in hyps}
    require(len(mapping) == len(hyps), 'duplicate rule/version')
    old = {hkey(h): h for h in previous}
    require(old.keys() <= mapping.keys(), 'historical rule version disappeared')
    live = [h for h in hyps if h['status'] != 'retracted']
    require(len(live) <= 4 and len({h['id'] for h in live}) == len(live), 'live-rule or version count')
    for h in hyps:
        require(h['semantic_key'] == digest(contract(h)), 'rule semantic fingerprint')
        require(h['tau_y'] == .06 and set(h['scope']['axes']) == set(pool['axes']), 'rule geometry/noise schema')
        for side in ('plus', 'minus'):
            j = pool['ix'][h[side+'_id']]
            exact(h[side+'_features'], list(pool['features'][j]), 'origin anchor real physical features')
            require(scope_mask(pool, h, [j])[0], 'origin example outside rule scope')
        rule_geometry_audit(pool, h)
        if hkey(h) in old:
            prior = old[hkey(h)]
            exact(contract(h), contract(prior), 'same-version semantic rewrite')
            exact(h['background'], prior['background'], 'same-version background declaration rewrite')
            for field in ('events', 'seen', 'prediction_score', 'score', 'successes', 'failures', 'trusted_weight'):
                exact(h['ledger'][field], prior['ledger'][field], 'prequery credit rewrite: '+field)
            exact(h['inherited_negative_evidence'], prior['inherited_negative_evidence'], 'inherited negatives rewritten')
            require(prior['status'] != 'retracted' or h['status'] == 'retracted', 'old version silently reactivated')
        else:
            require(h['created_round'] == t, 'new rule creation round')
            l = h['ledger']
            require(not l['events'] and not l['seen'] and l['prediction_score'] == l['score'] == l['successes'] == l['failures'] == l['trusted_weight'] == 0, 'new version has retrospective positive credit')
            require(h['excluded_observed_ids'] == sorted(x['candidate_id'] for x in prefix), 'new rule excludes wrong old outcomes')
            require(h['minus_id'] in h['excluded_observed_ids'] and h['plus_id'] not in h['excluded_observed_ids'], 'origin new-plus/known-reference roles')
            require(h['plus_id'] in menu_ids, 'new rule plus not proposed from visible input menu')
            exact(h['inherited_negative_evidence'], inherited_records(pool, hyps, h), 'negative prediction/fact inheritance completeness')
            if h['parent'] is not None:
                parent = mapping[hkey(h['parent'])]
                require(h['id'] == parent['id'] and h['version'] == parent['version']+1
                        and parent['status'] == 'retracted' and parent['retired_round'] == t, 'revision parent/version retirement')
        seen = set()
        for row in h['inherited_negative_evidence']:
            require(row['event_id'] not in seen, 'duplicate inherited event')
            seen.add(row['event_id'])
            source = mapping[(row['source_id'], row['source_version'])]
            ev = next(e for e in source['ledger']['events'] if e['event_id'] == row['event_id'])
            require(ev['delta_log_score'] < 0 or not ev['fact_success'], 'inherited positive-only evidence')
            for field in ('point_id', 'reference_id', 'reference_y', 'query_round', 'observed_y', 'delta_log_score', 'fact_success', 'prediction'):
                exact(row[field], ev[field], 'inherited event changed: '+field)
            exact(row['source_rule'], contract(source), 'inherited source contract')
            require(row['applicable'] == applicable_negative(pool, h, row), 'inherited scope/relation applicability')
        l = h['ledger']
        close(l['score'], l['prediction_score'], 'ledger score alias')
        close(l['weight'], weight(h), 'ledger deployment weight')


def declarations_audit(c):
    """Bind each newly accepted rule's background to the prequery proposal."""
    new = [h for h in c['hypotheses_at_commit'] if h['created_round'] == c['query']]
    if not new:
        return
    require(c['proposal'] is not None, 'new transfer rule without prequery proposal')
    for h in new:
        accepted = [r for r in c['reviews'] if 'action_index' in r and r.get('accepted')
                    and r.get('outcome') in ('add', 'revise') and hkey(r) == hkey(h)]
        require(len(accepted) == 1, 'new transfer rule lacks exactly one accepted action')
        action = c['proposal']['actions'][accepted[0]['action_index']]
        require(action['plus_id'] == h['plus_id'] and action['minus_id'] == h['minus_id'], 'rule origin examples differ from prequery action')
        expected = {}
        for axis, spec in action['background'].items():
            expected[axis] = ({'bounds': [float(v) for v in spec['bounds']]} if 'bounds' in spec
                              else {'allowed': sorted(spec['allowed'], key=digest)})
        exact(h['background'], expected, 'actual background was not declared before feedback')


def outside_origin_background(pool, h, point_id):
    """Actual probe versus the original V4 default background, input-only."""
    j = pool['ix'][point_id]
    require(scope_mask(pool, h, [j])[0], 'transfer-use point outside actual declared scope')
    x, a, b = pool['features'][j], h['plus_features'], h['minus_features']
    outside = []
    for k, (axis, kind) in enumerate(zip(pool['axes'], pool['kinds'])):
        if axis in h['focus']: continue
        if kind == 'continuous':
            lo, hi = max(0., a[k]-.25, b[k]-.25), min(1., a[k]+.25, b[k]+.25)
            beyond = x[k] < lo-ATOL or x[k] > hi+ATOL
        else:
            require(a[k] == b[k], 'noncomparable origin categorical background')
            beyond = x[k] != a[k]
        if beyond:
            require(axis in h['background'], 'undeclared background expansion')
            outside.append(axis)
    return outside


def transfer_role(pool, c):
    hs = {hkey(h): h for h in c['hypotheses_at_commit']}
    weights = {hkey(r): r['weight'] for r in c['selected_mixture']}
    target = hkey(c['verification']) if c['verification'] else None
    details = []
    for sh in c['shadows']:
        outside = outside_origin_background(pool, hs[hkey(sh)], c['selected_id'])
        if outside:
            details.append(dict(id=sh['id'], version=sh['version'], outside_default_background_axes=outside,
                point_id=c['selected_id'], reference_id=sh['prediction']['reference_id'],
                actual_mixture_weight=weights.get(hkey(sh), 0.), optional_target=hkey(sh) == target))
    weighted = sum(x['actual_mixture_weight'] > 0 for x in details)
    return dict(transfer_forecasts=len(details), weighted_transfer_forecasts=weighted,
        optional_target_transfers=int(any(x['optional_target'] for x in details)),
        guided_point_has_transfer_expert=int(c['mode'] == 'knowledge_guided' and weighted > 0),
        transfer_prediction_details=details)


def recovery_limit(previous_latent_trust, formula, gain, fact_success):
    limit = max(0., tanh(gain/2)) if fact_success and gain > 0 else 0.
    ceiling = previous_latent_trust+limit
    return limit, ceiling, min(formula, ceiling)



def selection_audit(pool, prefix, c, arrays, arm, verification_used, parameters):
    base, eligible, ties = arrays['base_logei'], arrays['eligible'], arrays['tie_priority']
    bo = select(base, eligible, ties)
    rows, calc = calculations(pool, c['hypotheses_at_commit'], prefix, arrays)
    exact(c['coverage'], rows, 'independent complete rule coverage')
    deployed = []
    if arm == 'full':
        raw = [r for r in rows if r['weight'] > 0 and hkey(r) in calc]
        total = sum(r['weight'] for r in raw)
        factor = min(1., .5/total) if total else 1.
        deployed = [dict(id=r['id'], version=r['version'], weight=r['weight'],
                         effective_mixture_weight=r['weight']*factor) for r in raw]
    exact(c['deployed_versions'], deployed, 'actual mixture weights/cap')
    combined = base.copy()
    if deployed:
        ws = [x['effective_mixture_weight'] for x in deployed]
        offsets = np.stack([calc[hkey(x)]['offsets'] for x in deployed])
        correction = logsumexp(np.vstack([np.full_like(base, log(1-sum(ws)))] +
            [log(w)+offset for w, offset in zip(ws, offsets)]), axis=0)
        correction[np.all(offsets <= ATOL, axis=0)] = 0.
        combined += correction
    selected = select(combined, eligible, ties)
    mode = 'knowledge_guided' if selected != bo else 'base_bo'
    ver = None
    if arm in ('full', 'verification') and selected == bo and c['query'] <= parameters['last_verification_round'] and verification_used < parameters['maximum_optional_verifications']:
        candidates = []
        for row in rows:
            k = hkey(row)
            if row['weight'] > 0 or not row['can_change_rank_at_cap'] or k not in calc:
                continue
            h = next(h for h in c['hypotheses_at_commit'] if hkey(h) == k)
            e = evidence(h)
            if e['effective_prediction_score'] < 0 or e['failures'] > e['successes']:
                continue
            info = calc[k]['values']['sign_information']
            pos = calc[k]['positions'].get(bo)
            ref_info = float(info[pos]) if pos is not None else 0.
            for pos, j in enumerate(calc[k]['indices']):
                gap = float(base[bo]-base[j])
                if j != bo and gap <= min(c['epsilon'], parameters['verification_maximum_gap']) and info[pos]-ref_info >= parameters['minimum_incremental_information'] and calc[k]['values']['offset'][pos] > ATOL:
                    candidates.append((float(info[pos]-ref_info), -gap, -ties[j], int(j), k, float(info[pos])))
        if candidates:
            ver = max(candidates, key=lambda x: x[:3])
            selected, mode = ver[3], 'optional_verification'
    require(pool['ids'][selected] == c['selected_id'] and mode == c['mode'], 'independent full-candidate/optional argmax differs')
    close(c['combined_logei'], combined[selected], 'independent selected mixture score', atol=1e-10, rtol=1e-10)
    if mode == 'knowledge_guided':
        require(combined[selected] > base[selected]+ATOL, 'unsupported outside point selected by suppression')
    if ver:
        r = calc[ver[4]]['probes'][selected]
        expected = dict(id=ver[4][0], version=ver[4][1], incremental_information=ver[0], information=ver[5],
            reference_id=r['reference_id'], reference_y=r['reference_y'],
            remaining_executable_probes=len(calc[ver[4]]['indices'])-1,
            information_kind='binary_soft_sign', minimum_unqueried_endpoints=1)
        exact(c['verification'], expected, 'optional verification receipt/global information argmax')
    else:
        require(c['verification'] is None, 'unexpected verification receipt')
    errors = []
    for data in calc.values():
        for j in {selected, bo}:
            pos = data['positions'].get(j)
            if pos is None:
                continue
            errors.append(adaptive_ei_check(float(arrays['mu_y'][j]), float(arrays['sd_y'][j]**2),
                float(arrays['sd_predictive_y'][j]**2), data['probes'][j]['reference_y'], max(x['y'] for x in prefix),
                float(data['values']['pure_log_ei_ratio'][pos])))
    return calc, max(errors, default=0.)



def shadow_feedback(pool, h, sh, receipt, observation, prefix, arrays, data, t):
    p = sh['prediction']
    require(digest({k: v for k, v in p.items() if k != 'prediction_sha256'}) == p['prediction_sha256'], 'prediction self hash')
    require((p['hypothesis_id'], p['version']) == hkey(h) and p['semantic_key'] == h['semantic_key'], 'prediction rule/version identity')
    require(p['point_id'] == observation['candidate_id'] and p['query_round'] == t, 'prediction point/query')
    j = pool['ix'][p['point_id']]
    require(j in data['probes'], 'forecast lacks a compatible known reference')
    ref = data['probes'][j]
    require(p['reference_id'] == ref['reference_id'] and p['reference_y'] == ref['reference_y'], 'not highest compatible already-observed reference')
    pos = data['positions'][j]
    fields = data['values']
    for name, expected in [('gp_mean_y', arrays['mu_y'][j]), ('gp_var_latent_y', arrays['sd_y'][j]**2),
                           ('gp_var_y', arrays['sd_predictive_y'][j]**2), ('tempering', fields['tempering'][pos]),
                           ('event_log_probability', fields['event_log_probability'][pos])]:
        close(p[name], expected, 'frozen predictor '+name)
    lt = float(fields['log_tempering'][pos])
    require(p['log_tempering'] is None if not np.isfinite(lt) else p['log_tempering'] is not None, 'no-effect/log-tempering contract')
    if np.isfinite(lt): close(p['log_tempering'], lt, 'frozen log tempering')
    require(p['zero_effect_policy'] == (not np.isfinite(lt)) and sh['informative'] == np.isfinite(lt), 'scoreability/no-op flag')
    close(sh['information_nats'], fields['sign_information'][pos], 'binary sign information')
    mu, vy, r, tau = p['gp_mean_y'], p['gp_var_y'], p['reference_y'], p['preference_noise_sd']
    require(tau == .06, 'forecast preference noise')
    alpha = (mu-r)/sqrt(vy+tau*tau)
    lam = exp(-.5*alpha*alpha-.5*log(2*pi)-float(log_ndtr(alpha)))
    cm = mu+vy/sqrt(vy+tau*tau)*lam
    cv = max(0., vy-vy*vy/(vy+tau*tau)*np.clip(lam*(lam+alpha), 0, 1))
    temper = p['tempering']
    for name, value in [('conditional_mean_y', cm), ('conditional_var_y', cv),
                        ('knowledge_mean_y', mu+temper*(cm-mu)),
                        ('knowledge_var_y', (1-temper)*vy+temper*cv+temper*(1-temper)*(cm-mu)**2)]:
        close(p[name], value, 'frozen predictor moment '+name)
    l = h['ledger']
    require(p['point_id'] not in l['seen'] and p['point_id'] not in h['excluded_observed_ids'], 'feedback repeated/retrospective')
    require(all(e['query_round'] < t for e in l['events']), 'feedback order')
    before, before_w = evidence(h), weight(h)
    gain, success = score(p, observation['y']), bool(observation['y'] > ref['reference_y'])
    event = dict(hypothesis_id=h['id'], version=h['version'], point_id=p['point_id'], reference_id=ref['reference_id'],
        reference_y=ref['reference_y'], query_round=t, prediction=deepcopy(p), observed_y=float(observation['y']),
        fact_success=success, delta_prediction_score=gain, delta_log_score=gain)
    event['event_id'] = digest(event)
    l['prediction_score'] += gain
    l['score'] = l['prediction_score']
    l['successes'] += int(success)
    l['failures'] += int(not success)
    l['seen'] = sorted(set(l['seen']) | {p['point_id']})
    l['events'].append(event)
    formula = evidence(h)['formula_weight']
    recover = success and gain > 0
    credit_limit, recovery_ceiling, l['trusted_weight'] = recovery_limit(before['trusted_weight'], formula, gain, success)
    l['weight'] = weight(h)
    after = evidence(h)
    expected = dict(event_id=event['event_id'], hypothesis_id=h['id'], version=h['version'], point_id=p['point_id'],
        reference_id=ref['reference_id'], reference_y=ref['reference_y'], query_round=t, delta_log_score=gain,
        delta_prediction_score=gain, fact_success=success, weight_before=before_w, weight_after=l['weight'],
        trusted_weight_before=before['trusted_weight'], trusted_weight_after=after['trusted_weight'],
        score_after=l['prediction_score'], effective_prediction_score=after['effective_prediction_score'],
        successes=after['successes'], failures=after['failures'], reliability_cap=after['reliability_cap'],
        formula_weight=after['formula_weight'], recovery_credit_limit=credit_limit, recovery_weight_ceiling=recovery_ceiling,
        weight_transition='increment_limited_positive_fact_and_forecast' if recover
        else 'nonincreasing_after_failure_or_nonpositive_forecast')
    exact(receipt, expected, 'independent dual-ledger feedback/weight receipt')
    require(recover or after['trusted_weight'] <= before['trusted_weight'], 'failure/nonpositive prediction increased trust')
    require(after['trusted_weight'] <= before['trusted_weight']+credit_limit+ATOL, 'trust recovery exceeds current positive credit')
    return gain



def prompt_audit(folder, protocol, pool, call, prefix, c, prepared, rows):
    context = read(folder/'requests'/call['context_sha256']/'context.json')
    require(digest(context) == call['context_sha256'], 'LLM context hash')
    require(context['model'] == protocol['model'] and context['reasoning_effort'] == protocol['reasoning_effort'], 'LLM identity/effort')
    require(digest(context['system']) == protocol['prompt_sha256'], 'LLM frozen system prompt')
    public = context['public']
    require(set(public) == {'task', 'objective_unit', 'query', 'remaining_queries', 'domain_context', 'menu_policy', 'axes',
        'own_observations', 'candidate_menu', 'knowledge_ledger', 'actionable_coverage', 'feedback_note'}, 'LLM field whitelist')
    common.domain_context_audit(public['domain_context'], pool)
    require(public['task'] == pool['task'] and public['query'] == c['query'] and public['remaining_queries'] == 17-c['query'], 'LLM task/round')
    require(public['objective_unit'] == pool['unit'], 'LLM engineering units')
    exact(public['axes'], [dict(name=a, type=k) for a, k in zip(pool['axes'], pool['kinds'])], 'LLM true axes/categories')
    def point(key):
        j = pool['ix'][key]
        return dict(id=key, point=pool['points'][j], normalized_continuous={
            a: float(pool['features'][j][k]) for k, (a, kind) in enumerate(zip(pool['axes'], pool['kinds'])) if kind == 'continuous'})
    exact(public['candidate_menu'], [point(k) for k in c['menu_ids']], 'LLM input-only candidate menu')
    exact(public['own_observations'], [dict(point(h['candidate_id']), objective=h['raw_value']) for h in prefix], 'LLM own exact prefix only')
    ledger = []
    for h in prepared:
        row = {k: deepcopy(h[k]) for k in ('id', 'version', 'parent', 'focus', 'background', 'scope', 'categorical_preference', 'status', 'reason') if k in h}
        row['origin_example'] = dict(plus_id=h['plus_id'], minus_id=h['minus_id'])
        row['evidence'] = evidence(h)
        row['trusted_weight'] = evidence(h)['trusted_weight']
        row['tested_pairs'] = [dict(plus_id=e['point_id'], reference_id=e['reference_id'], query_round=e['query_round'],
            plus_objective=e['observed_y']*pool['scale'], reference_objective=e['reference_y']*pool['scale'],
            fact_success=e['fact_success'], delta_prediction_logscore=e['delta_log_score']) for e in h['ledger']['events']]
        row['inherited_counterevidence'] = [dict(event_id=e['event_id'], plus_id=e['point_id'], reference_id=e['reference_id'],
            plus_objective=e['observed_y']*pool['scale'], reference_objective=e['reference_y']*pool['scale'],
            fact_success=e['fact_success'], delta_prediction_logscore=e['delta_log_score'], applicable=e['applicable'])
            for e in h['inherited_negative_evidence']]
        ledger.append(row)
    exact(public['knowledge_ledger'], ledger, 'LLM ledger must exclude all GP/shadow fields and include only past feedback')
    safe = ('id', 'version', 'weight', 'eligible_inside_scope', 'executable_probes', 'remaining_probe_count', 'can_change_rank_at_cap', 'eligibility_state')
    exact(public['actionable_coverage'], [{k: r[k] for k in safe if k in r} for r in rows], 'LLM current executable coverage')
    receipt = read(folder/'requests'/call['context_sha256']/'validated.json')
    exact(call['receipt'], receipt, 'request cache receipt')
    response = read(folder/'requests'/call['context_sha256']/f"response_{receipt['attempt']:02d}.json")
    require(digest(response) == receipt['response_sha256'], 'LLM raw response hash')
    import json
    exact(json.loads(response['output_text']), receipt['proposal'], 'validated response text')
    exact(c['proposal'], receipt['proposal'], 'committed proposal changed')



def audit_run(folder, path, protocol, pool, manifest, include_incomplete=False, joint=False):
    state = read(path)
    params = protocol['parameters']
    trial, tie_seed = common.trial_seeds(protocol)
    require(state['protocol_sha256'] == protocol['protocol_sha256'], 'state protocol')
    require(state['condition'] == 'agent_initial_terra' and state['mode'] in protocol['supported_modes'], 'unsupported V5 arm/condition')
    history, decisions = state['history'], state['decisions']
    require(state['seed'] == trial and len(history) == 4+len(decisions), 'trial/budget/history')
    require(len(decisions) == 16 or include_incomplete and 0 < len(decisions) < 16 and state['status'] != 'COMPLETE', 'query budget')
    require([r['candidate_id'] for r in history[:4]] == manifest['initial_ids'], 'initial design changed')
    common.initial_commit_audit(path, state, protocol, pool, manifest)
    require(len({r['candidate_id'] for r in history}) == len(history), 'duplicate query')
    for row in history:
        require(row['candidate_id'] in pool['ix'] and np.isfinite(row['y']), 'invalid observed point')
        close(row['raw_value'], row['y']*pool['scale'], 'observed engineering unit')
    previous, calls, modes, forecasts = [], [], Counter(), 0
    max_quad_error, roles = 0., []
    seen_predictions = set()
    for t, decision in enumerate(decisions, 1):
        prefix, observation = history[:t+3], history[t+3]
        cp = path.parent/'rounds'/f'{t:02d}_commit.json'
        c = read(cp)
        require(c['query'] == decision['query'] == t and c['mechanism'] == 'transfer_rule_v5', 'commit sequence/mechanism')
        require(c['history_sha256'] == digest(prefix), 'commit observed prefix hash')
        require(c['protocol_sha256'] == protocol['protocol_sha256'] and digest(c) == decision['commit_sha256'], 'commit content hash')
        require(c['selected_id'] == decision['selected_id'] == observation['candidate_id'], 'selected outcome identity')
        require(c['mode'] == decision['mode'] and c['BO_reference_id'] == decision['BO_reference_id'], 'selected mode/reference')
        close(decision['y'], observation['y'], 'decision y')
        close(decision['raw_value'], observation['raw_value'], 'decision objective')
        close(decision['incumbent_before_y'], max(r['y'] for r in prefix), 'incumbent before')
        close(decision['incumbent_after_y'], max(r['y'] for r in history[:t+4]), 'incumbent after')
        with np.load(cp.with_name(f'{t:02d}_GP.npz'), allow_pickle=False) as saved:
            arrays = {k: saved[k].copy() for k in saved.files}
        require(set(arrays) == {'mu_y', 'sd_y', 'sd_predictive_y', 'base_logei', 'available', 'eligible', 'tie_priority'}, 'GP artifact fields')
        n = len(pool['ids'])
        require(all(x.shape == (n,) and np.isfinite(x).all() for x in arrays.values()), 'invalid stored GP fields')
        require(np.all(arrays['sd_y'] >= 0) and np.all(arrays['sd_predictive_y'] >= arrays['sd_y']), 'stored variance contract')
        available = ~np.isin(pool['ids'], [r['candidate_id'] for r in prefix])
        order = sorted(range(n), key=lambda j: pool['features'][j])
        ties = np.empty(n, np.int64)
        ties[order] = np.random.default_rng(np.random.SeedSequence([tie_seed, t])).permutation(n)
        require(np.array_equal(ties, arrays['tie_priority']), 'physical tie RNG/order')
        require(np.array_equal(available, arrays['available']), 'available mask')
        base = arrays['base_logei']
        bo = select(base, available, ties)
        epsilon = log(4)*(16-t)/15
        eligible = available & (base[bo]-base <= epsilon)
        require(np.array_equal(eligible, arrays['eligible']), 'qLogEI opportunity set')
        selected = pool['ix'][c['selected_id']]
        require(c['BO_reference_id'] == pool['ids'][bo] and eligible[selected], 'BO argmax or selected budget')
        require(c['eligible_count'] == int(eligible.sum()), 'eligible count')
        for field, value in [('epsilon', epsilon), ('base_logei', base[selected]), ('reference_logei', base[bo]), ('opportunity_gap', base[bo]-base[selected])]:
            close(c[field], value, 'commit '+field)
        common.native_seed_audit(pool, c)
        require(c['GP_fit_key'] == c['GP_acquisition_meta']['fit_key'], 'GP acquisition fit key')
        fit = read(cp.with_name(f'{t:02d}_fit.json'))
        require(fit and all(np.isfinite(np.asarray(x)).all() for x in fit.values()), 'nonfinite fit receipt')
        if joint: saved_gp_audit(folder, pool, prefix, c, arrays, fit)
        prepared, events, pre_rows = refresh(pool, previous, prefix, arrays, t)
        exact(c['pre_call_coverage_events'], events, 'pre-call coverage transition')
        want_call = state['mode'] != 'bo' and should_call(calls, decisions[:t-1], pre_rows, t, params)
        expected_menu = candidate_menu(pool, prefix, arrays, t) if want_call else []
        exact(c['menu_ids'], expected_menu, 'semantic candidate menu')
        require(c['calls_at_commit'][:len(calls)] == calls and len(c['calls_at_commit']) == len(calls)+int(want_call and bool(expected_menu)), 'LLM event schedule/call prefix')
        if len(c['calls_at_commit']) > len(calls):
            call = c['calls_at_commit'][-1]
            require(call['query'] == t, 'LLM call current round')
            prompt_audit(folder, protocol, pool, call, prefix, c, prepared, pre_rows)
        else:
            require(c['proposal'] is None, 'proposal without a scheduled call')
        calls = c['calls_at_commit']
        hyps = deepcopy(c['hypotheses_at_commit'])
        validate_hypotheses(pool, hyps, prepared, prefix, t, c['menu_ids'])
        declarations_audit(c)
        for h in hyps:
            if h['status'] == 'retracted': continue
            ps = probes(pool, h, prefix, eligible)
            require((h['status'] == 'active') == bool(ps), 'rule dormancy/executable probes')
        calc, err = selection_audit(pool, prefix, c, arrays, state['mode'], modes['optional_verification'], params)
        max_quad_error = max(max_quad_error, err)
        expected_shadows = [hkey(h) for h in hyps if hkey(h) in calc and selected in calc[hkey(h)]['probes']]
        require([hkey(s) for s in c['shadows']] == expected_shadows, 'missing/extraneous new-plus forecasts')
        require(len(c['shadows']) == len(decision['evidence']), 'forecast/receipt count')
        weights = {hkey(x): x['effective_mixture_weight'] for x in c['deployed_versions']}
        expected_mixture = [dict(id=s['id'], version=s['version'], weight=weights[hkey(s)], prediction_sha256=s['prediction']['prediction_sha256'])
                            for s in c['shadows'] if weights.get(hkey(s), 0) > 0]
        exact(c['selected_mixture'], expected_mixture, 'selected actual deployed mixture')
        scores = {}
        for sh, receipt in zip(c['shadows'], decision['evidence']):
            pred_sha = sh['prediction']['prediction_sha256']
            require(pred_sha not in seen_predictions, 'prediction reused')
            seen_predictions.add(pred_sha)
            h = next(h for h in hyps if hkey(h) == hkey(sh))
            scores[pred_sha] = shadow_feedback(pool, h, sh, receipt, observation, prefix, arrays, calc[hkey(sh)], t)
            forecasts += 1
        mixture = c['selected_mixture']
        total = sum(r['weight'] for r in mixture)
        actual_gain = float(logsumexp([log(1-total)]+[log(r['weight'])+scores[r['prediction_sha256']] for r in mixture])) if mixture else 0.
        close(decision['deployed_mixture_log_score_gain'], actual_gain, 'actual deployed mixture logscore')
        require(actual_gain >= -log(2)-ATOL, 'half-GP mixture predictive logloss bound')
        modes[c['mode']] += 1
        roles.append(dict(query=t, mode=c['mode'], selected_id=c['selected_id'], BO_reference_id=c['BO_reference_id'],
            deployed_experts=len(c['selected_mixture']), forecasts=len(c['shadows']), mixture_gain=actual_gain,
            positive_prediction_failed_fact=sum(e['delta_log_score'] > 0 and not e['fact_success'] for e in decision['evidence']),
            **transfer_role(pool, c)))
        previous = hyps
    exact(state['hypotheses'], previous, 'state final rule ledgers differ from independent replay')
    exact(state['calls'], calls, 'state final call history')
    require(state['verification_used'] == modes['optional_verification'] <= params['maximum_optional_verifications'], 'optional query budget')
    require(state.get('cache_counters', {}).get('operator_compiles_in_loop', 0) == 0, 'operators compiled during experiment loop')
    raw = np.array([h['raw_value'] for h in history])
    curve = np.maximum.accumulate(raw)
    return dict(task=state['task'], group=pool['group'], mode=state['mode'], condition=state['condition'], status=state['status'],
        technical_checks='PASS_LISTED_CHECKS', queries=len(decisions), forecasts=forecasts, selection_modes=dict(modes),
        initial_history_sha256=digest(history[:4]), SearchAUC=float(curve[4:].mean()), final_best=float(curve[-1]),
        first_best_total=int(np.flatnonzero(raw == curve[-1])[0])+1,
        argmax_scope='FULL_CANDIDATE_RULE_AND_OPTIONAL_SELECTION', saved_GP_and_MC512_recomputed=bool(joint),
        max_selected_BO_quad_logratio_error=max_quad_error, roles=roles,
        transfer_usage={k:sum(row[k] for row in roles) for k in ('transfer_forecasts','weighted_transfer_forecasts',
                       'optional_target_transfers','guided_point_has_transfer_expert')},
        run=path.relative_to(ROOT).as_posix())


def paired_learning_audit(rows):
    """Compare predeclared arms; no per-task policy selection or oracle data."""
    by_key = {(r['task'], r['mode']): r for r in rows if r['status'] == 'COMPLETE'}
    pairs = []
    for task in sorted({r['task'] for r in rows}):
        full, control = by_key.get((task, 'full')), by_key.get((task, 'verification'))
        if full is None or control is None: continue
        require(full['initial_history_sha256'] == control['initial_history_sha256'], 'full/control initial observations differ')
        fp, vp = ROOT/full['run'], ROOT/control['run']
        fs, vs = read(fp), read(vp)
        first_point = first_mode = None
        shared = 0
        quotas = None
        for t in range(1, 17):
            fc, vc = [read(path.parent/'rounds'/f'{t:02d}_commit.json') for path in (fp, vp)]
            if first_mode is None and fc['mode'] != vc['mode']: first_mode = t
            if first_point is None and fc['selected_id'] != vc['selected_id']:
                first_point = t
                quotas = dict(full=sum(d['mode'] == 'optional_verification' for d in fs['decisions'][:t-1]),
                              verification=sum(d['mode'] == 'optional_verification' for d in vs['decisions'][:t-1]))
            if fc['history_sha256'] == vc['history_sha256']:
                shared += 1
                for field in ('hypotheses_at_commit', 'coverage', 'calls_at_commit', 'proposal', 'menu_ids',
                              'pre_call_coverage_events', 'GP_fit_key', 'GP_acquisition_meta'):
                    exact(fc[field], vc[field], 'same-prefix full/verification state or prompt differs: '+field)
            require(not vc['deployed_versions'] and not vc['selected_mixture'] and vc['mode'] != 'knowledge_guided',
                    'verification control accidentally applied positive-weight acquisition')
        pairs.append(dict(task=task, shared_prefix_rounds=shared, same_prefix_state_and_prompt='PASS',
            first_selection_mode_difference=first_mode, first_actual_point_difference=first_point,
            optional_quota_used_before_first_point_difference=quotas,
            full_minus_verification_SearchAUC=full['SearchAUC']-control['SearchAUC'],
            full_minus_verification_final_best=full['final_best']-control['final_best']))
    return pairs



def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', default='v5')
    parser.add_argument('--joint', action='store_true', help='Also reconstruct saved GP and MC512; never fit')
    parser.add_argument('--allow-source-drift', action='store_true')
    parser.add_argument('--include-incomplete', action='store_true')
    parser.add_argument('--task', action='append')
    parser.add_argument('--run', action='append')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    require(args.variant == 'v5', 'This independent auditor supports only the registered V5 mechanism')
    folder = ROOT/'experiments/runs/knowledge_loop_20261101'/args.variant
    p = read(folder/'protocol.json')
    require(p['protocol_sha256'] == digest({k: v for k, v in p.items() if k != 'protocol_sha256'}), 'protocol self hash')
    require(p['variant'] == 'v5' and p['mechanism'] == 'transfer_rule_v5' and p['initial'] == 4 and p['queries'] == 16, 'V5 protocol identity/budget')
    require(p['supported_modes'] == ['bo', 'full', 'verification'], 'unregistered V5 development arms')
    common.trial_seeds(p)
    require(p['seed'] == 0, 'V5 seed1 was already inspected; no new confirmation implied')
    drift = [name for name, expected in p['source_sha256'].items() if not (ROOT/name).exists() or filehash(ROOT/name) != expected]
    require(not drift or args.allow_source_drift, 'Frozen source changed; historical internals require --allow-source-drift')
    goal_path = p.get('goal_config_path', 'code/resources/knowledge_loop_goal_20261101_majority.json')
    require(filehash(ROOT/goal_path) == p['goal_config_sha256'], 'goal-config custody')
    require(filehash(ROOT/p['design_document_path']) == p['design_document_sha256'], 'frozen V5 design receipt')
    uninspected_input_receipts = []
    for item in p['inputs']:
        for path, expected in item['files_sha256'].items():
            if path.endswith('/pool.json'):
                require(filehash(ROOT/path) == expected, 'input pool content SHA: '+path)
            else:
                # The inherited manifest also records full oracle/advice
                # files. Their protocol hashes are retained without opening
                # them: this audit is restricted to candidate-only pools and
                # already observed histories.
                uninspected_input_receipts.append(dict(path=path, declared_sha256=expected))
    source = common.source_order_audit(ROOT)
    require(source['initial_commit_before_initial_oracle'], 'initial commitment must precede initial outcome read')
    manifests = {x['task']: x for x in p['inputs']}
    pools = {task: dict(common.public_pool(ROOT, item), task=task) for task, item in manifests.items()}
    rows, failures, pending = [], [], []
    for path in sorted((folder/'runs').glob('*/state.json')):
        state = read(path)
        if args.run and path.parent.name not in args.run: continue
        if args.task and state['task'] not in args.task: continue
        if state['status'] != 'COMPLETE':
            pending.append(dict(run=path.parent.name, status=state['status'], queries=len(state['decisions'])))
            if not args.include_incomplete or not state['decisions']: continue
        try:
            rows.append(audit_run(folder, path, p, pools[state['task']], manifests[state['task']], args.include_incomplete, args.joint))
            print(json.dumps(dict(audited=path.parent.name, queries=rows[-1]['queries'], result='PASS_LISTED_CHECKS'), ensure_ascii=False), flush=True)
        except Exception as error:
            failures.append(dict(run=path.parent.name, error_type=type(error).__name__, error=str(error)))
            print(json.dumps(dict(audited=path.parent.name, result='FAIL', error=str(error)), ensure_ascii=False), flush=True)
    baselines = {r['task']: r for r in rows if r['mode'] == 'bo' and r['status'] == 'COMPLETE'}
    for row in rows:
        if row['mode'] == 'bo' or row['status'] != 'COMPLETE' or row['task'] not in baselines: continue
        bo = baselines[row['task']]
        require(row['initial_history_sha256'] == bo['initial_history_sha256'], 'matched baseline initial outcomes differ')
        row['AUC_delta_vs_matched_BO'] = row['SearchAUC']-bo['SearchAUC']
        row['final_delta_vs_matched_BO'] = row['final_best']-bo['final_best']
    comparisons = []
    for arm in sorted({r['mode'] for r in rows if r['mode'] != 'bo'}):
        same = [r for r in rows if r['mode'] == arm and 'AUC_delta_vs_matched_BO' in r]
        comparisons.append(dict(mode=arm, matched_tasks=len(same),
            strict_SearchAUC_wins=sum(r['AUC_delta_vs_matched_BO'] > 1e-9 for r in same),
            SearchAUC_ties=sum(abs(r['AUC_delta_vs_matched_BO']) <= 1e-9 for r in same),
            SearchAUC_losses=sum(r['AUC_delta_vs_matched_BO'] < -1e-9 for r in same)))
    report = dict(created_at=datetime.now(timezone.utc).isoformat(), variant=args.variant,
        auditor_sha256=filehash(Path(__file__)), protocol_sha256=p['protocol_sha256'], source_drift=drift,
        audit_helper_sha256={str(Path(v4.__file__).relative_to(ROOT)):filehash(Path(v4.__file__)),
                            str(Path(common.__file__).relative_to(ROOT)):filehash(Path(common.__file__))},
        technical_result='PASS_LISTED_CHECKS' if rows and not failures else 'FAIL_OR_NO_AUDITED_RUNS',
        performance_goal_achieved=None, performance_interpretation='Development comparisons only; technical PASS is not scientific superiority.',
        read_only=True, fits=0, LLM_calls=0, unqueried_targets_read=0,
        source_order=source, joint=bool(args.joint), runs=rows, failures=failures, pending=pending,
        uninspected_input_receipts=uninspected_input_receipts,
        totals=dict(runs=len(rows), rounds=sum(r['queries'] for r in rows), forecasts=sum(r['forecasts'] for r in rows)),
        comparisons=comparisons,
        learning_ablation=paired_learning_audit(rows),
        limitations=['Commit hashes prove internal causal ordering; they are not an external timestamp attestation.',
            'Source/schema inspection is not an OS-level I/O trace.',
            'Only candidate-only pool inputs were rehashed; oracle/advice custody remains the frozen protocol receipt and was not reopened.',
            'GP is reconstructed from saved fitted parameters, never independently optimized.',
            'GH64 is approximate; selected and executable BO forecasts have a separate adaptive F-integral check.',
            'Native MC512 qLogEI times conditional-normal-EI ratio remains the preregistered acquisition approximation.',
            'Transfer-use flags require an actually selected plus outside its origin default background; declarations alone are not counted.',
            'Full versus verification is the total trajectory effect of enabling learned acquisition, including later quota/call paths, not a single-round causal mean-correction effect.',
            *([] if args.joint else ['Stored GP/MC arrays were not reconstructed; full rule argmax uses the saved arrays.'])])
    output = args.output or ROOT/'experiments/reports/knowledge_loop_20261101/v5_independent_audit.json'
    output = output.resolve()
    approved = (ROOT/'experiments/reports/knowledge_loop_20261101').resolve()
    require(output.is_relative_to(approved), 'Audit output must stay in dedicated reports directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(result=report['technical_result'], totals=report['totals'], report=str(output)), ensure_ascii=False), flush=True)
    return int(bool(failures) or not rows)



if __name__ == '__main__':
    raise SystemExit(main())
