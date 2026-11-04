"""Independent V7 proposal-interface audit, retaining V5 decision mathematics.

Reconstructs all geometry hints, own-reference ordering, narrow-cost flags and
prefix-time quota flags without importing the production helper or wrapper.
Audit scope respects the predeclared BO/full then conditional verification stages.
No API, oracle or GP fitting; optional joint checks rebuild only saved fits.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from itertools import combinations
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
saved_gp_audit = v5.saved_gp_audit
candidate_menu, should_call = v5.candidate_menu, v5.should_call
validate_hypotheses, declarations_audit = v5.validate_hypotheses, v5.declarations_audit
selection_audit, shadow_feedback, transfer_role = v5.selection_audit, v5.shadow_feedback, v5.transfer_role


def expected_proposal_support(pool, prefix, menu_ids, arrays, t, verification_used, epsilon):
    """Independently enumerate legal focus choices, then encode R/A compactly."""
    require(type(t) is int and 1 <= t <= 16 and type(verification_used) is int
            and 0 <= verification_used <= 1, 'support prefix time/quota')
    known = {r['candidate_id']: r for r in prefix}
    require(len(known) == len(prefix) and all(k in pool['ix'] for k in known), 'support own history IDs')
    require(all(np.isfinite(float(r['y'])) for r in prefix), 'support known values must be finite')
    require(len(set(menu_ids)) == len(menu_ids) and not set(menu_ids)&set(known)
            and set(menu_ids) <= set(pool['ids']), 'support displayed unqueried IDs')
    axes, kinds = pool['axes'], pool['kinds']
    for key in set(menu_ids)|set(known):
        features = pool['features'][pool['ix'][key]]
        require(len(features) == len(axes), 'support feature dimensions')
        for value, kind in zip(features, kinds):
            require((isinstance(value, str) if kind == 'categorical' else
                     type(value) in (int, float) and np.isfinite(value) and 0 <= value <= 1),
                    'support true categorical labels/normalized continuous values')
    base, eligible, ties = arrays['base_logei'], arrays['eligible'], arrays['tie_priority']
    available = np.asarray([k not in known for k in pool['ids']], dtype=bool)
    bo = select(base, available, ties)
    references = sorted(prefix, key=lambda r:(-float(r['y']), tuple(pool['features'][pool['ix'][r['candidate_id']]])))
    results = []
    for key in menu_ids:
        j = pool['ix'][key]
        x = pool['features'][j]
        legal = []
        for ref in references:
            r = pool['features'][pool['ix'][ref['candidate_id']]]
            changed = [k for k in range(len(axes)) if x[k] != r[k]]
            allowed = []
            for size in (1, 2):
                for focus in combinations(changed, size):
                    if all(k in focus or (x[k] == r[k] if kinds[k] == 'categorical'
                                          else abs(float(x[k])-float(r[k])) <= .25+1e-12)
                           for k in range(len(axes))):
                        allowed.append(focus)
            if not allowed:
                continue
            required = [k for k in changed if kinds[k] == 'categorical' or abs(float(x[k])-float(r[k])) > .25+1e-12]
            additional = [k for k in changed if k not in required]
            # Check that the compressed form is exactly the enumerated sets.
            decoded = [c for size in (1, 2) for c in combinations(changed, size) if set(required) <= set(c)]
            require(allowed == decoded, 'internal focus compression equivalence')
            legal.append(dict(reference_id=ref['candidate_id'],
                required_focus_axes=[axes[k] for k in required],
                additional_changed_axes=[axes[k] for k in additional]))
        results.append(dict(point_id=key, references=legal[:4], compatible_reference_count=len(legal),
            references_truncated=len(legal)>4,
            passes_narrow_cost_gate=bool(eligible[j] and float(base[bo]-base[j]) <= min(float(epsilon), log(1.1)))))
    return dict(schema='known_reference_cost_flags_v1',
        optional_window_open=bool(t <= 10 and verification_used < 1),
        maximum_references_per_candidate=4,
        reference_order='Own observed objective descending, then physical features; never candidate ID order.',
        focus_rule='Choose one or two changed axes, including all required_focus_axes; any other focus must come from additional_changed_axes.',
        geometry_note='Nonfocus categories must be equal; nonfocus normalized continuous differences may not exceed .25 plus 1e-12 numerical tolerance. Geometry alone does not establish a chemical effect.',
        cost_flag_note='Only the current narrow acquisition-cost gate, including the base-BO point if displayed. No guarantee of scope, incremental information, rank effect or rule validity.',
        candidate_support=results)


def support_payload_audit(actual, pool, prefix, menu_ids, arrays, t, verification_used, epsilon):
    expected = expected_proposal_support(pool, prefix, menu_ids, arrays, t, verification_used, epsilon)
    exact(actual, expected, 'independent proposal support geometry/order/truncation/cost/window mismatch')
    return expected


def stage_scope(rows, expected_tasks, *, filtered=False):
    """Distinguish an allowed stage2 stop from unexamined/unfinished runs."""
    tasks = set(expected_tasks)
    complete = {(r['task'], r['mode']):r for r in rows if r['status'] == 'COMPLETE'}
    missing_stage1 = sorted((task, arm) for task in tasks for arm in ('bo', 'full') if (task, arm) not in complete)
    missing_verification = sorted(task for task in tasks if (task, 'verification') not in complete)
    result = dict(scope='filtered' if filtered else 'all_registered_tasks',
        registered_tasks=len(tasks), complete_stage1_trajectories=2*len(tasks)-len(missing_stage1),
        complete_verification_trajectories=len(tasks)-len(missing_verification),
        missing_stage1_trajectories=missing_stage1, stage2_trigger=None, strict_full_wins=None,
        actual_weighted_choices=None, stage2_missing_tasks=[], stage2_policy_violation=False)
    if filtered or missing_stage1:
        result['stage_status'] = 'FILTERED_NO_STAGE_CONCLUSION' if filtered else 'STAGE1_NOT_FULLY_AUDITED'
        return result
    wins = sum(complete[task, 'full']['SearchAUC'] > complete[task, 'bo']['SearchAUC']+1e-9 for task in tasks)
    weighted = sum(complete[task, 'full']['selection_modes'].get('knowledge_guided', 0) for task in tasks)
    trigger = wins >= 5 and weighted > 0
    result.update(stage2_trigger=trigger, strict_full_wins=wins, actual_weighted_choices=weighted)
    if trigger:
        result['stage_status'] = 'STAGE2_COMPLETE' if not missing_verification else 'STAGE2_REQUIRED_NOT_FULLY_AUDITED'
        result['stage2_missing_tasks'] = missing_verification
    else:
        started = any(r['mode'] == 'verification' for r in rows)
        result['stage_status'] = 'UNEXPECTED_STAGE2_AFTER_FAILED_GATE' if started else 'PREDECLARED_STOP_AFTER_STAGE1'
        result['stage2_policy_violation'] = started
        result['stage2_not_run_reason'] = 'Full did not attain >=5 strict AUC wins and an actual weighted choice; verification is not pending.'
    return result


def paired_learning_audit(rows):
    """Equal own outcomes alone no longer imply equal public quota metadata."""
    by_key = {(r['task'],r['mode']):r for r in rows if r['status']=='COMPLETE'}
    pairs = []
    for task in sorted({r['task'] for r in rows}):
        full, control = by_key.get((task,'full')), by_key.get((task,'verification'))
        if full is None or control is None:
            continue
        require(full['initial_history_sha256']==control['initial_history_sha256'], 'full/control initial observations differ')
        fp,vp = ROOT/full['run'],ROOT/control['run']
        fs,vs = read(fp),read(vp)
        first_point=first_mode=first_quota=None
        shared_history=shared_complete_state=0
        for t in range(1,17):
            fc,vc = [read(path.parent/'rounds'/f'{t:02d}_commit.json') for path in (fp,vp)]
            fq,vq = [sum(d['mode']=='optional_verification' for d in s['decisions'][:t-1]) for s in (fs,vs)]
            if first_quota is None and fq != vq: first_quota=t
            if first_mode is None and fc['mode']!=vc['mode']: first_mode=t
            if first_point is None and fc['selected_id']!=vc['selected_id']: first_point=t
            if fc['history_sha256']==vc['history_sha256']:
                shared_history += 1
                def prior_calls(c): return [r for r in c['calls_at_commit'] if r['query']<t]
                # Later feedback/rules may already differ after a quota-induced
                # prompt difference. Require parity only for the full public
                # prequery state, not just identical point/y histories.
                previous_f = fs['decisions'][:t-1]
                previous_v = vs['decisions'][:t-1]
                evidence_equal = [x['evidence'] for x in previous_f] == [x['evidence'] for x in previous_v]
                if fq==vq and prior_calls(fc)==prior_calls(vc) and evidence_equal:
                    shared_complete_state += 1
                    for field in ('hypotheses_at_commit','coverage','calls_at_commit','proposal','menu_ids',
                                  'pre_call_coverage_events','GP_fit_key','GP_acquisition_meta'):
                        exact(fc[field],vc[field], 'same full public-prefix state differs: '+field)
            require(not vc['deployed_versions'] and not vc['selected_mixture'] and vc['mode']!='knowledge_guided',
                    'verification arm accidentally applied learned acquisition')
        pairs.append(dict(task=task, shared_observed_history_rounds=shared_history,
            shared_complete_public_state_rounds=shared_complete_state, complete_state_parity='PASS_LISTED_CHECKS',
            first_selection_mode_difference=first_mode,first_actual_point_difference=first_point,
            first_prefix_quota_difference=first_quota,
            full_minus_verification_SearchAUC=full['SearchAUC']-control['SearchAUC'],
            full_minus_verification_final_best=full['final_best']-control['final_best']))
    return pairs


def prompt_audit(folder, protocol, pool, call, prefix, c, prepared, rows, arrays, verification_used):
    context = read(folder/'requests'/call['context_sha256']/'context.json')
    require(digest(context) == call['context_sha256'], 'LLM context hash')
    require(context['model'] == protocol['model'] and context['reasoning_effort'] == protocol['reasoning_effort'], 'LLM identity/effort')
    require(digest(context['system']) == protocol['prompt_sha256'], 'LLM frozen system prompt')
    public = context['public']
    require(set(public) == {'task', 'objective_unit', 'query', 'remaining_queries', 'domain_context', 'menu_policy', 'axes',
        'own_observations', 'candidate_menu', 'knowledge_ledger', 'actionable_coverage', 'feedback_note', 'proposal_support'}, 'LLM field whitelist')
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
    support_payload_audit(public['proposal_support'], pool, prefix, c['menu_ids'], arrays,
                          c['query'], verification_used, c['epsilon'])
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
    require(state['condition'] == 'agent_initial_terra' and state['mode'] in protocol['supported_modes'], 'unsupported V7 arm/condition')
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
            prompt_audit(folder, protocol, pool, call, prefix, c, prepared, pre_rows, arrays, modes['optional_verification'])
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


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', default='v7')
    parser.add_argument('--joint', action='store_true', help='Also reconstruct saved GP and MC512; never fit')
    parser.add_argument('--allow-source-drift', action='store_true')
    parser.add_argument('--include-incomplete', action='store_true')
    parser.add_argument('--task', action='append')
    parser.add_argument('--run', action='append')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    require(args.variant == 'v7', 'This independent auditor supports only the registered V7 interface')
    folder = ROOT/'experiments/runs/knowledge_loop_20261101'/args.variant
    p = read(folder/'protocol.json')
    require(p['protocol_sha256'] == digest({k: v for k, v in p.items() if k != 'protocol_sha256'}), 'protocol self hash')
    require(p['variant'] == 'v7' and p['mechanism'] == 'transfer_rule_v5' and p['initial'] == 4 and p['queries'] == 16, 'V7 protocol identity/budget')
    require(p['supported_modes'] == ['bo', 'full', 'verification'], 'unregistered V7 staged development arms')
    require(p['proposal_interface'] == 'known_reference_cost_flags_v1' and len(p['inputs']) == 8, 'V7 support contract/eight-task protocol')
    common.trial_seeds(p)
    require(p['seed'] == 0, 'V7 seed0 development only; no independent confirmation implied')
    drift = [name for name, expected in p['source_sha256'].items() if not (ROOT/name).exists() or filehash(ROOT/name) != expected]
    require(not drift or args.allow_source_drift, 'Frozen source changed; historical internals require --allow-source-drift')
    goal_path = p.get('goal_config_path', 'code/resources/knowledge_loop_goal_20261101_majority.json')
    require(filehash(ROOT/goal_path) == p['goal_config_sha256'], 'goal-config custody')
    require(filehash(ROOT/p['design_document_path']) == p['design_document_sha256'], 'frozen V7 design receipt')
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
            pending.append(dict(run=path.parent.name, task=state['task'], mode=state['mode'], status=state['status'], queries=len(state['decisions'])))
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
    stages = stage_scope(rows+pending, manifests, filtered=bool(args.task or args.run))
    if stages['stage2_policy_violation']:
        failures.append(dict(run='stage_protocol', error_type='StageGateViolation', error=stages['stage_status']))
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
        learning_ablation=paired_learning_audit(rows), stage_scope=stages,
        limitations=['Commit hashes prove internal causal ordering; they are not an external timestamp attestation.',
            'Source/schema inspection is not an OS-level I/O trace.',
            'Only candidate-only pool inputs were rehashed; oracle/advice custody remains the frozen protocol receipt and was not reopened.',
            'GP is reconstructed from saved fitted parameters, never independently optimized.',
            'GH64 is approximate; selected and executable BO forecasts have a separate adaptive F-integral check.',
            'Native MC512 qLogEI times conditional-normal-EI ratio remains the preregistered acquisition approximation.',
            'Transfer-use flags require an actually selected plus outside its origin default background; declarations alone are not counted.',
            'Full versus verification is the total trajectory effect of learned acquisition, including downstream quota-dependent proposal-support differences.',
            'Stage2 verification not run after a failed complete stage1 gate is a predeclared stop, not pending work.',
            *([] if args.joint else ['Stored GP/MC arrays were not reconstructed; full rule argmax uses the saved arrays.'])])
    output = args.output or ROOT/'experiments/reports/knowledge_loop_20261101/v7_independent_audit.json'
    output = output.resolve()
    approved = (ROOT/'experiments/reports/knowledge_loop_20261101').resolve()
    require(output.is_relative_to(approved), 'Audit output must stay in dedicated reports directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(result=report['technical_result'], totals=report['totals'], report=str(output)), ensure_ascii=False), flush=True)
    return int(bool(failures) or not rows)



if __name__ == '__main__':
    raise SystemExit(main())
