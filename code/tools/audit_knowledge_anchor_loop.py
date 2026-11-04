"""Independent read-only V6 accepted-example validation audit.

No production plan, policy, LLM, oracle or fitting is called. V5's independent
mathematics, mixed geometry, prompt, credit and GP reconstruction are reused.
Accepted proposal provenance is reconstructed from actions, including duplicate
semantic rules with new examples. V6 changes only optional candidate membership.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from math import log
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

import audit_knowledge_transfer_loop as v5

v4, common, ROOT = v5.v4, v5.common, v5.ROOT
read, digest, filehash = v5.read, v5.digest, v5.filehash
require, close, exact, ATOL = v5.require, v5.close, v5.exact, v5.ATOL
hkey, contract, scope_mask = v5.hkey, v5.contract, v5.scope_mask
probes, evidence, weight = v5.probes, v5.evidence, v5.weight
select, calculations, refresh = v5.select, v5.calculations, v5.refresh
adaptive_ei_check, saved_gp_audit = v5.adaptive_ei_check, v5.saved_gp_audit
candidate_menu, should_call = v5.candidate_menu, v5.should_call
validate_hypotheses, declarations_audit = v5.validate_hypotheses, v5.declarations_audit
shadow_feedback, prompt_audit, transfer_role = v5.shadow_feedback, v5.prompt_audit, v5.transfer_role


def action_contract(pool, action, menu_ids, prefix):
    """Independent real-input contract construction, including duplicate actions."""
    require(isinstance(action, dict) and set(action) == {
        'op', 'id', 'plus_id', 'minus_id', 'focus', 'background', 'reason'}, 'accepted action schema')
    require(action['op'] in ('add', 'revise'), 'example requires add/revise')
    require(isinstance(action['reason'], str) and 1 <= len(action['reason'].strip()) <= 1000,
            'accepted action reason')
    known = {x['candidate_id'] for x in prefix}
    plus, minus = action['plus_id'], action['minus_id']
    require(isinstance(plus, str) and isinstance(minus, str) and plus != minus
            and plus in menu_ids and plus in pool['ix'] and plus not in known
            and minus in known and minus in pool['ix'], 'example must be displayed new plus versus own known reference')
    a, b = [list(pool['features'][pool['ix'][x]]) for x in (plus, minus)]
    require(a != b, 'physically identical example pair')
    focus, background = action['focus'], action['background']
    require(isinstance(focus, dict) and 1 <= len(focus) <= 2 and not set(focus)-set(pool['axes']),
            'accepted example focus axes')
    require(isinstance(background, dict) and not set(background)-set(pool['axes'])
            and not set(background)&set(focus), 'accepted example background axes')

    def bounds(spec, k, fields):
        require(isinstance(spec, dict) and set(spec) == fields, 'declared bounds schema')
        values = spec['bounds']
        require(isinstance(values, (list, tuple)) and len(values) == 2
                and all(type(v) in (int, float) and np.isfinite(v) for v in values), 'finite declared bounds')
        lo, hi = map(float, values)
        require(0 <= lo < hi <= 1 and lo <= min(a[k], b[k]) <= max(a[k], b[k]) <= hi,
                'declared scope must contain both examples')
        return [lo, hi]

    def labels(spec, k, fields):
        require(isinstance(spec, dict) and set(spec) == fields, 'declared labels schema')
        allowed = spec['allowed']
        require(isinstance(allowed, list) and allowed and all(isinstance(v, str) for v in allowed)
                and len(set(allowed)) == len(allowed)
                and set(allowed) <= {row[k] for row in pool['features']}
                and a[k] in allowed and b[k] in allowed, 'true declared labels containing both examples')
        return sorted(allowed, key=digest)

    normalized, scopes, categories, declared = {}, {}, {}, {}
    for k, (axis, kind) in enumerate(zip(pool['axes'], pool['kinds'])):
        require(kind in ('continuous', 'categorical'), 'unsupported input kind')
        if kind == 'continuous':
            require(all(type(v) in (int, float) and np.isfinite(v) and 0 <= v <= 1 for v in (a[k], b[k])),
                    'normalized continuous examples')
            if axis in focus:
                spec = focus[axis]
                interval = bounds(spec, k, {'direction', 'bounds'})
                require(spec['direction'] in ('increase', 'decrease'), 'continuous direction')
                sign = 1 if spec['direction'] == 'increase' else -1
                require(sign*(a[k]-b[k]) > 0, 'example contradicts direction')
                normalized[axis] = dict(direction=spec['direction'], bounds=interval)
            else:
                require(abs(a[k]-b[k]) <= .25+ATOL, 'background comparison exceeds radius')
                interval = [max(0., a[k]-.25, b[k]-.25), min(1., a[k]+.25, b[k]+.25)]
            scopes[axis] = dict(kind=kind, bounds=interval)
        else:
            require(isinstance(a[k], str) and isinstance(b[k], str), 'categories must be real string labels')
            if axis in focus:
                spec = focus[axis]
                if spec == 'prefer_plus':
                    spec = dict(direction='prefer_plus', allowed=[a[k], b[k]])
                allowed = labels(spec, k, {'direction', 'allowed'})
                require(spec['direction'] == 'prefer_plus' and a[k] != b[k], 'categorical example orientation')
                normalized[axis] = dict(direction='prefer_plus', allowed=allowed)
                categories[axis] = dict(plus=a[k], minus=b[k])
            else:
                require(a[k] == b[k], 'background category must be equal within pair')
                allowed = [a[k]]
            scopes[axis] = dict(kind=kind, allowed=allowed)
        if axis in background:
            declared[axis] = ({'bounds': bounds(background[axis], k, {'bounds'})} if kind == 'continuous'
                              else {'allowed': labels(background[axis], k, {'allowed'})})
            scopes[axis] = dict(kind=kind, **declared[axis])
    return dict(plus_id=plus, minus_id=minus, plus_features=a, minus_features=b,
                focus=normalized, scope=dict(axes=scopes, background_radius=.25),
                categorical_preference=categories, background=declared)


def proposal_ledger_audit(pool, prefix, c, previous_records):
    """Rebuild append-only example provenance without modifying rule evidence."""
    require(len({digest(x) for x in previous_records}) == len(previous_records), 'duplicated prior proposal record')
    records = deepcopy(previous_records)
    proposal = c['proposal']
    reviews = [r for r in c['reviews'] if 'action_index' in r]
    if proposal is None:
        require(not reviews, 'action review without an LLM proposal')
    else:
        require(isinstance(proposal, dict) and set(proposal) == {'actions', 'summary'}
                and isinstance(proposal['summary'], str) and 1 <= len(proposal['summary'].strip()) <= 1600
                and isinstance(proposal['actions'], list) and len(proposal['actions']) <= 3, 'proposal top-level schema')
        actions = proposal['actions']
        require([r['action_index'] for r in reviews] == list(range(len(actions))), 'missing/duplicate/out-of-order action reviews')
        menu = c['menu_ids']
        require(len(set(menu)) == len(menu) and set(menu) <= set(pool['ids'])
                and not set(menu)&{r['candidate_id'] for r in prefix}, 'displayed menu availability')
        hyps = {hkey(h): h for h in c['hypotheses_at_commit']}
        touched = set()
        for review, action in zip(reviews, actions):
            require(review['round'] == c['query'] and type(review['accepted']) is bool, 'action review round/acceptance')
            if not review['accepted']:
                continue
            require(isinstance(action, dict) and set(action) == {
                'op', 'id', 'plus_id', 'minus_id', 'focus', 'background', 'reason'}
                and isinstance(action['reason'], str) and 1 <= len(action['reason'].strip()) <= 1000,
                'accepted action complete schema/reason')
            require(review['op'] == action['op'], 'accepted operation changed')
            op = action['op']
            require(op in ('add', 'revise', 'retain', 'retract'), 'accepted unknown operation')
            if op == 'add':
                require(action['id'] is None, 'accepted add names existing ID')
            else:
                require(isinstance(action['id'], str) and action['id'] not in touched
                        and any(h['id'] == action['id'] for h in hyps.values()), 'accepted action identity/touched rule')
                touched.add(action['id'])
            require(hkey(review) in hyps, 'accepted review points to missing version')
            h = hyps[hkey(review)]
            if op in ('retain', 'retract'):
                require(review['outcome'] == op and action['plus_id'] is None and action['minus_id'] is None
                        and action['focus'] == {} and action['background'] == {}, 'retain/retract cannot supply examples')
                continue
            require(review['outcome'] in ('add', 'revise', 'duplicate_semantics_preserved'), 'accepted example outcome')
            built = action_contract(pool, action, menu, prefix)
            exact(contract(built), contract(h), 'accepted example contract differs from assigned rule semantics')
            require(digest(contract(built)) == h['semantic_key'], 'accepted example semantic hash')
            if review['outcome'] in ('add', 'revise'):
                require(review['outcome'] == op and h['created_round'] == c['query'], 'new example version creation')
                exact(built, {k: h[k] for k in built}, 'new rule differs from accepted full geometry')
                touched.add(h['id'])
            record = dict(id=h['id'], version=h['version'], semantic_key=h['semantic_key'],
                point_id=action['plus_id'], proposed_reference_id=action['minus_id'],
                proposal_round=c['query'], action_index=review['action_index'], action_sha256=digest(action))
            if record not in records:
                records.append(record)
    exact(c['validation_proposals'], records, 'accepted proposal ledger missing, rewritten, reordered or fabricated')
    return records


def origin_identity_audit(hypotheses, previous):
    """A new proposal source never silently changes an existing rule origin."""
    prior = {hkey(h): h for h in previous}
    for h in hypotheses:
        if hkey(h) not in prior:
            continue
        old = prior[hkey(h)]
        for field in ('plus_id', 'minus_id', 'plus_features', 'minus_features', 'created_round'):
            exact(h[field], old[field], 'same-version origin metadata rewritten: '+field)


def executable_examples(pool, h, data, records):
    """First provenance record per real, currently executable accepted plus."""
    result = {}
    if h['status'] == 'retracted':
        return result
    for record in records:
        if (record['id'], record['version'], record['semantic_key']) != (*hkey(h), h['semantic_key']):
            continue
        j = pool['ix'].get(record['point_id'])
        if j is not None and j in data['positions']:
            result.setdefault(int(j), record)
    return result


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
    if arm == 'full' and selected == bo and c['query'] <= parameters['last_verification_round'] and verification_used < parameters['maximum_optional_verifications']:
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
            proposed = executable_examples(pool, h, calc[k], c['validation_proposals'])
            for pos, j in enumerate(calc[k]['indices']):
                if int(j) not in proposed:
                    continue
                gap = float(base[bo]-base[j])
                if j != bo and gap <= min(c['epsilon'], parameters['verification_maximum_gap']) and info[pos]-ref_info >= parameters['minimum_incremental_information'] and calc[k]['values']['offset'][pos] > ATOL:
                    candidates.append((float(info[pos]-ref_info), -gap, -ties[j], int(j), k, float(info[pos]), proposed[int(j)]))
        if candidates:
            ver = max(candidates, key=lambda x: x[:3])
            selected, mode = ver[3], 'optional_verification'
    require(pool['ids'][selected] == c['selected_id'] and mode == c['mode'], 'independent full-scope weighted / accepted-example optional argmax differs')
    close(c['combined_logei'], combined[selected], 'independent selected mixture score', atol=1e-10, rtol=1e-10)
    if mode == 'knowledge_guided':
        require(combined[selected] > base[selected]+ATOL, 'unsupported outside point selected by suppression')
    if ver:
        r = calc[ver[4]]['probes'][selected]
        expected = dict(id=ver[4][0], version=ver[4][1], incremental_information=ver[0], information=ver[5],
            reference_id=r['reference_id'], reference_y=r['reference_y'],
            remaining_executable_probes=len(calc[ver[4]]['indices'])-1,
            information_kind='binary_soft_sign', minimum_unqueried_endpoints=1,
            proposal=ver[6], reference_changed_from_proposal=r['reference_id'] != ver[6]['proposed_reference_id'])
        exact(c['verification'], expected, 'accepted-example verification provenance/reference/information argmax')
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



def audit_run(folder, path, protocol, pool, manifest, include_incomplete=False, joint=False):
    state = read(path)
    params = protocol['parameters']
    trial, tie_seed = common.trial_seeds(protocol)
    require(state['protocol_sha256'] == protocol['protocol_sha256'], 'state protocol')
    require(state['condition'] == 'agent_initial_terra' and state['mode'] in protocol['supported_modes'], 'unsupported V6 arm/condition')
    history, decisions = state['history'], state['decisions']
    require(state['seed'] == trial and len(history) == 4+len(decisions), 'trial/budget/history')
    require(len(decisions) == 16 or include_incomplete and 0 < len(decisions) < 16 and state['status'] != 'COMPLETE', 'query budget')
    require([r['candidate_id'] for r in history[:4]] == manifest['initial_ids'], 'initial design changed')
    common.initial_commit_audit(path, state, protocol, pool, manifest)
    require(len({r['candidate_id'] for r in history}) == len(history), 'duplicate query')
    for row in history:
        require(row['candidate_id'] in pool['ix'] and np.isfinite(row['y']), 'invalid observed point')
        close(row['raw_value'], row['y']*pool['scale'], 'observed engineering unit')
    previous, previous_proposals, calls, modes, forecasts = [], [], [], Counter(), 0
    max_quad_error, roles = 0., []
    seen_predictions = set()
    for t, decision in enumerate(decisions, 1):
        prefix, observation = history[:t+3], history[t+3]
        cp = path.parent/'rounds'/f'{t:02d}_commit.json'
        c = read(cp)
        require(c['query'] == decision['query'] == t and c['mechanism'] == 'anchor_validation_v6', 'commit sequence/mechanism')
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
        origin_identity_audit(hyps, prepared)
        declarations_audit(c)
        current_proposals = proposal_ledger_audit(pool, prefix, c, previous_proposals)
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
            accepted_examples=len(current_proposals), verification_proposal=c['verification']['proposal'] if c['verification'] else None,
            verification_reference_changed=c['verification']['reference_changed_from_proposal'] if c['verification'] else None,
            positive_prediction_failed_fact=sum(e['delta_log_score'] > 0 and not e['fact_success'] for e in decision['evidence']),
            **transfer_role(pool, c)))
        previous = hyps
        previous_proposals = current_proposals
    exact(state['hypotheses'], previous, 'state final rule ledgers differ from independent replay')
    exact(state['calls'], calls, 'state final call history')
    exact(state.get('validation_proposals', []), previous_proposals, 'state final accepted proposal ledger')
    require(state['verification_used'] == modes['optional_verification'] <= params['maximum_optional_verifications'], 'optional query budget')
    require(state.get('cache_counters', {}).get('operator_compiles_in_loop', 0) == 0, 'operators compiled during experiment loop')
    raw = np.array([h['raw_value'] for h in history])
    curve = np.maximum.accumulate(raw)
    return dict(task=state['task'], group=pool['group'], mode=state['mode'], condition=state['condition'], status=state['status'],
        technical_checks='PASS_LISTED_CHECKS', queries=len(decisions), forecasts=forecasts, selection_modes=dict(modes),
        initial_history_sha256=digest(history[:4]), SearchAUC=float(curve[4:].mean()), final_best=float(curve[-1]),
        first_best_total=int(np.flatnonzero(raw == curve[-1])[0])+1,
        argmax_scope='FULL_SCOPE_WEIGHTED_AND_ACCEPTED_EXAMPLE_OPTIONAL_SELECTION',
        accepted_proposal_records=len(previous_proposals), saved_GP_and_MC512_recomputed=bool(joint),
        max_selected_BO_quad_logratio_error=max_quad_error, roles=roles,
        transfer_usage={k:sum(row[k] for row in roles) for k in ('transfer_forecasts','weighted_transfer_forecasts',
                       'optional_target_transfers','guided_point_has_transfer_expert')},
        run=path.relative_to(ROOT).as_posix())


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', default='v6')
    parser.add_argument('--joint', action='store_true', help='Also reconstruct saved GP and MC512; never fit')
    parser.add_argument('--allow-source-drift', action='store_true')
    parser.add_argument('--include-incomplete', action='store_true')
    parser.add_argument('--task', action='append')
    parser.add_argument('--run', action='append')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    require(args.variant == 'v6', 'This independent auditor supports only the registered V6 mechanism')
    folder = ROOT/'experiments/runs/knowledge_loop_20261101'/args.variant
    p = read(folder/'protocol.json')
    require(p['protocol_sha256'] == digest({k: v for k, v in p.items() if k != 'protocol_sha256'}), 'protocol self hash')
    require(p['variant'] == 'v6' and p['mechanism'] == 'anchor_validation_v6' and p['initial'] == 4 and p['queries'] == 16, 'V6 protocol identity/budget')
    require(p['supported_modes'] == ['bo', 'full'], 'unregistered V6 development arms')
    common.trial_seeds(p)
    require(p['seed'] == 0, 'V6 is seed0 development only; no new initialization confirmation implied')
    drift = [name for name, expected in p['source_sha256'].items() if not (ROOT/name).exists() or filehash(ROOT/name) != expected]
    require(not drift or args.allow_source_drift, 'Frozen source changed; historical internals require --allow-source-drift')
    goal_path = p.get('goal_config_path', 'code/resources/knowledge_loop_goal_20261101_majority.json')
    require(filehash(ROOT/goal_path) == p['goal_config_sha256'], 'goal-config custody')
    require(filehash(ROOT/p['design_document_path']) == p['design_document_sha256'], 'frozen V6 design receipt')
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
        audit_helper_sha256={str(Path(v5.__file__).relative_to(ROOT)):filehash(Path(v5.__file__)),
                            str(Path(v4.__file__).relative_to(ROOT)):filehash(Path(v4.__file__)),
                            str(Path(common.__file__).relative_to(ROOT)):filehash(Path(common.__file__))},
        technical_result='PASS_LISTED_CHECKS' if rows and not failures else 'FAIL_OR_NO_AUDITED_RUNS',
        performance_goal_achieved=None, performance_interpretation='Development comparisons only; technical PASS is not scientific superiority.',
        read_only=True, fits=0, LLM_calls=0, unqueried_targets_read=0,
        source_order=source, joint=bool(args.joint), runs=rows, failures=failures, pending=pending,
        uninspected_input_receipts=uninspected_input_receipts,
        totals=dict(runs=len(rows), rounds=sum(r['queries'] for r in rows), forecasts=sum(r['forecasts'] for r in rows)),
        comparisons=comparisons,
        limitations=['Commit hashes prove internal causal ordering; they are not an external timestamp attestation.',
            'Source/schema inspection is not an OS-level I/O trace.',
            'Only candidate-only pool inputs were rehashed; oracle/advice custody remains the frozen protocol receipt and was not reopened.',
            'GP is reconstructed from saved fitted parameters, never independently optimized.',
            'GH64 is approximate; selected and executable BO forecasts have a separate adaptive F-integral check.',
            'Native MC512 qLogEI times conditional-normal-EI ratio remains the preregistered acquisition approximation.',
            'Transfer-use flags require an actually selected plus outside its origin default background; declarations alone are not counted.',
            'Accepted examples are not LLM ranked recommendations: optional choice retains GP information ranking and highest compatible known reference.',
            *([] if args.joint else ['Stored GP/MC arrays were not reconstructed; full rule argmax uses the saved arrays.'])])
    output = args.output or ROOT/'experiments/reports/knowledge_loop_20261101/v6_independent_audit.json'
    output = output.resolve()
    approved = (ROOT/'experiments/reports/knowledge_loop_20261101').resolve()
    require(output.is_relative_to(approved), 'Audit output must stay in dedicated reports directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(result=report['technical_result'], totals=report['totals'], report=str(output)), ensure_ascii=False), flush=True)
    return int(bool(failures) or not rows)



if __name__ == '__main__':
    raise SystemExit(main())
