"""Independent ledger/policy audit and task-clustered descriptive analysis.

The auditor does not call select_policy, compile_membership or score_campaign.
It reconstructs predicates, decisions and fixed-budget metrics from raw files.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from functools import lru_cache
from hashlib import sha256
import importlib.metadata
import inspect
from itertools import combinations
import json
from pathlib import Path
import time
import warnings

import numpy as np

from .replay import write_json_atomic


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                             ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def independent_membership(task, batch):
    specs = {p['name']: p for p in task['parameters']}
    result = []
    for h in batch['hypotheses']:
        if h['kind'] == 'interval':
            col = np.array([r[h['parameter']] for r in task['candidates']], dtype=float)
            spec = specs[h['parameter']]
            lo, hi = spec['bounds']
            check(spec['unit'] == h['unit'] and lo <= h['lower'] <= h['upper'] <= hi, 'invalid interval')
            result.append((col >= h['lower']) & (col <= h['upper']))
        elif h['kind'] == 'allowed_values':
            spec = specs[h['parameter']]
            check(spec['unit'] == h['unit'] and set(h['values']) <= set(spec['levels']), 'invalid categories')
            result.append(np.array([r[h['parameter']] in h['values'] for r in task['candidates']]))
        else:
            check(h['kind'] == 'linear_le' and h['unit'] == 'normalized', 'unsupported predicate')
            lhs = np.zeros(len(task['candidate_ids']))
            for name, coefficient in sorted(h['coefficients'].items()):
                lo, hi = specs[name]['bounds']
                col = np.array([r[name] for r in task['candidates']], dtype=float)
                lhs += coefficient * (col - lo) / (hi - lo)
            result.append(lhs <= h['rhs'])
    return np.array(result, dtype=bool).reshape(len(result), len(task['candidate_ids']))


def independent_choice(method, scores, h, budget, ids, hids, available, rng,
                       round_number, escape_rounds, suggestion):
    def best(mask):
        positions = np.flatnonzero(mask)
        return min(positions, key=lambda i: (-float(scores[i]), ids[i])) if len(positions) else None
    base = best(available)
    near = available & (scores[base] - scores <= budget)
    groups = []
    for size in range(len(hids) + 1):
        for group in combinations(range(len(hids)), size):
            mask = available.copy()
            for j in group:
                mask &= h[j]
            groups.append((tuple(sorted(hids[j] for j in group)), mask))
    full = available & (h.all(axis=0) if len(h) else True)
    retained = tuple(sorted(hids))
    if method == 'B0':
        return base, (), near, groups
    if method == 'B2' and round_number in escape_rounds:
        return int(rng.choice(np.flatnonzero(available))), (), near, groups
    if method in {'B1', 'B2'}:
        return best(full), retained, near, groups
    if method == 'B3':
        index = ids.index(suggestion) if suggestion in ids else None
        return (index if index is not None and near[index] else base), (), near, groups
    if method == 'A1':
        kept = [j for j in range(len(h)) if (h[j] & near).any()]
        mask = available.copy()
        for j in kept:
            mask &= h[j]
        return best(mask), tuple(sorted(hids[j] for j in kept)), near, groups
    if method == 'A2':
        return (best(full), retained, near, groups) if (full & near).any() else (base, (), near, groups)
    valid = [(names, mask) for names, mask in groups if (mask & near).any()] if method == 'B4' else [
        (names, mask) for names, mask in groups if mask.any()]
    names, mask = min(valid, key=lambda pair: (-len(pair[0]), pair[0]))
    return best(mask), names, near, groups


def independent_curve(ledger, initial, additional):
    values = [row['value'] for row in ledger]
    check(initial <= len(values) <= initial + additional, 'observation count outside budget')
    best, running = [], -float('inf')
    for value in values:
        running = max(running, value)
        best.append(running)
    best += [running] * (initial + additional - len(best))
    return best, sum(best[initial:]) / additional


@lru_cache(maxsize=512)
def _prediction(path):
    with np.load(path, allow_pickle=False) as saved:
        return saved['mu'], saved['sigma'], json.loads(str(saved['metadata']))


def _direct_gp_check(task, history, settings, seed, mu, sigma):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern
    from threadpoolctl import threadpool_limits
    columns = []
    for p in task['parameters']:
        if p['bounds'] is not None:
            lo, hi = p['bounds']
            columns.append([(r[p['name']] - lo) / (hi - lo) for r in task['candidates']])
        else:
            for level in p['levels']:
                columns.append([float(r[p['name']] == level) for r in task['candidates']])
    x = np.array(columns).T
    positions = {key: i for i, key in enumerate(task['candidate_ids'])}
    ordered = sorted(history, key=lambda row: row['candidate_id'])
    gp = GaussianProcessRegressor(
        kernel=ConstantKernel(1., (1e-3, 1e3)) * Matern(np.ones(x.shape[1]), (1e-2, 1e2), nu=2.5),
        alpha=settings['gp_alpha'], normalize_y=False, n_restarts_optimizer=settings['gp_restarts'],
        random_state=settings['base_seed'] + seed)
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gp.fit(x[[positions[r['candidate_id']] for r in ordered]], [r['value'] for r in ordered])
        expected_mu, expected_sigma = gp.predict(x, return_std=True)
    error = max(float(np.max(np.abs(expected_mu - mu))), float(np.max(np.abs(expected_sigma - sigma))))
    check(error <= 1e-10, f'independent sklearn GP mismatch {error}')
    return error


def audit_suite(directory, *, verify_gp=True):
    from ..bo.llm import GPRModel
    from .evaluation import engine_fingerprint
    root, started = Path(directory).resolve(), time.perf_counter()
    manifest = read_json(root / 'manifest.json')
    definition = manifest['definition']
    assistant = definition.get('track') == 'actual_interactive_assistant'
    regimes = ['actual_assistant'] if assistant else definition['prior_regimes']
    methods = definition['methods']
    settings = definition['settings']
    initial, additional = settings['initial_queries'], settings['additional_queries']
    totals, rows, artifact_hashes, gp_checks = Counter(), [], {}, []
    identities = set()
    expected_engine = engine_fingerprint()
    source = inspect.getsource(GPRModel)
    software = {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'scikit-learn')}
    for td in manifest['tasks']:
        for kind in ('bundle', 'oracle'):
            check(sha256((root / td[kind]).read_bytes()).hexdigest() == td[kind + '_sha256'], 'input hash changed')
        bundle = read_json(root / td['bundle'])
        task, ids = bundle['task'], bundle['task']['candidate_ids']
        check(digest(task) == td['task_sha256'], 'public task changed')
        parameters = {p['name']: p for p in task['parameters']}
        for candidate in task['candidates']:
            check(set(candidate) == set(parameters), 'incomplete full parameter vector')
            for name, spec in parameters.items():
                check((spec['bounds'][0] <= candidate[name] <= spec['bounds'][1]) if spec['bounds'] else
                      candidate[name] in spec['levels'], 'hard domain violation')
        with np.load(root / td['oracle'], allow_pickle=False) as saved:
            values = saved['values']
        true_y = dict(zip(ids, map(float, values)))
        oracle_fingerprint = digest({'values': true_y, 'sigmas': None})
        positions = {key: i for i, key in enumerate(ids)}
        membership_cache = {}
        checked_gp = False
        for seed in definition['paired_seeds']:
            paired_initial = bundle['initial_ids'][str(seed)]
            for regime in regimes:
                for method in methods:
                    folder = root / 'trajectories' / td['task_id'] / f's{seed}'
                    folder = folder / method if assistant else folder / regime / method
                    state_path, result_path = folder / 'state.json', folder / 'result.json'
                    state, saved_result = read_json(state_path), read_json(result_path)
                    header, ledger = state['header'], state['oracle_ledger']
                    identity = (td['task_id'], seed, regime, method)
                    check(identity not in identities, 'duplicate campaign')
                    identities.add(identity)
                    check(header['task_id'] == td['task_id'] and header['method'] == method and header['seed'] == seed,
                          'campaign identity differs from manifest')
                    check(header['settings'] == settings and header['task_sha256'] == td['task_sha256']
                          and header['oracle_sha256'] == oracle_fingerprint, 'campaign inputs changed')
                    check(header['engine_sha256'] == expected_engine and digest(header) == state['header_sha256'],
                          'decision engine or header changed')
                    check(header['initial_ids'] == paired_initial and [r['candidate_id'] for r in ledger[:initial]] ==
                          paired_initial, 'unpaired initialization')
                    check(len(ledger) == initial + len(state['decisions']), 'ledger/decision length differs')
                    check(len({r['candidate_id'] for r in ledger}) == len(ledger), 'repeat query')
                    for number, row in enumerate(ledger, 1):
                        check(row['candidate_id'] in true_y and row['query'] == number
                              and row['value'] == true_y[row['candidate_id']], 'oracle ledger corrupted')
                    check(state['status'] in {'COMPLETE', 'STOPPED'}, 'nonterminal or erroneous campaign')
                    check(state['operator_compiles_in_loop'] == 0, 'repeated runtime compilation')
                    batches = {digest({'version': 1, 'hypotheses': []}): {'version': 1, 'hypotheses': []}}
                    active = {'version': 1, 'hypotheses': []}
                    if header['fixed_batch'] is not None and method != 'B0':
                        active = header['fixed_batch']
                        batches[digest(active)] = active
                    events_by_round = {}
                    llm_events = state['hypothesis_state']['events']
                    check(len(llm_events) <= settings['max_llm_attempts'], 'LLM budget exceeded')
                    previous_times = set()
                    for attempt, event in enumerate(llm_events, 1):
                        t = event['completed_additional']
                        check(event['attempt'] == attempt and t in settings['refresh_after'] and t not in previous_times,
                              'invalid model call schedule')
                        previous_times.add(t)
                        observed = ledger[:initial + t]
                        request = read_json(root / 'requests' / event['context_sha256'] / 'request.json')
                        context = request['context']
                        check(digest(context) == event['context_sha256'] and context['previous_batch'] == active,
                              'request or hypothesis lineage changed')
                        check(context['expected_version'] == attempt and context['task'] == task,
                              'model task or version changed')
                        expected_obs = [{'candidate_id': r['candidate_id'], 'params': task['candidates'][positions[r['candidate_id']]],
                                         'value': r['value'], 'sigma': None} for r in observed]
                        check(context['observations'] == expected_obs, 'model received outcomes beyond its history')
                        response = read_json(root / 'requests' / event['context_sha256'] / 'response.json')
                        check(response['response'] == event['raw_response'], 'raw assistant answer changed')
                        if event['status'] == 'accepted':
                            active = event['raw_response']['batch']
                            known_obs = {r['candidate_id'] for r in observed}
                            evidence = {r['id'] for r in task['evidence']}
                            for h in active['hypotheses']:
                                check(set(h['observation_ids']) <= known_obs and set(h['evidence_ids']) <= evidence,
                                      'accepted hypothesis cites unknown evidence')
                            batches[digest(active)] = active
                        else:
                            totals['failed_llm_logical_events'] += 1
                        events_by_round[t] = event
                    check(active == state['hypothesis_state']['payload'], 'final hypothesis payload differs')
                    rng, queue = np.random.default_rng(settings['base_seed'] + seed), []
                    decision_batch = header['fixed_batch'] if header['fixed_batch'] is not None and method != 'B0' else {
                        'version': 1, 'hypotheses': []}
                    all_events = list(state['decisions'])
                    if state['status'] == 'STOPPED':
                        check('failed_decision' in state, 'missing stopped decision receipt')
                        all_events.append(state['failed_decision'])
                    cost = Counter()
                    chosen_gaps = []
                    joint_conflicts, changed, retention_n, retention_d, violations = 0, 0, 0, 0, 0
                    for t, event in enumerate(all_events):
                        history = ledger[:initial + t]
                        observed_ids = [r['candidate_id'] for r in history]
                        check(event['round'] == t + 1 and event['observed_ids'] == observed_ids, 'decision history changed')
                        if t in events_by_round and events_by_round[t]['status'] == 'accepted':
                            queue = list(events_by_round[t]['raw_response']['candidate_ids'])
                            decision_batch = events_by_round[t]['raw_response']['batch']
                        check(digest(decision_batch) == event['hypothesis_batch_sha256'],
                              'decision uses a hypothesis before its accepted update')
                        batch = batches[event['hypothesis_batch_sha256']]
                        check(batch['version'] == event['hypothesis_version'], 'decision hypothesis version changed')
                        if digest(batch) not in membership_cache:
                            membership_cache[digest(batch)] = independent_membership(task, batch)
                        h = membership_cache[digest(batch)]
                        hids = [row['id'] for row in batch['hypotheses']]
                        check(hids == event['hypothesis_ids'], 'hypothesis IDs changed')
                        key_definition = {'task_sha256': td['task_sha256'], 'history': sorted(history, key=lambda r: r['candidate_id']),
                                          'restarts': settings['gp_restarts'], 'alpha': settings['gp_alpha'],
                                          'random_state': settings['base_seed'] + seed, 'normalize_y': False,
                                          'gp_source': source, 'software': software}
                        key_definition['history'] = [{k: row[k] for k in ('candidate_id', 'value', 'sigma') if k in row}
                                                     for row in key_definition['history']]
                        check(digest(key_definition) == event['prediction_key'], 'GP cache includes wrong observation prefix')
                        mu, sigma, metadata = _prediction(str(root / 'predictions' / (event['prediction_key'] + '.npz')))
                        check(metadata['key'] == event['prediction_key'] and metadata['observations'] == len(history),
                              'GP cache metadata differs')
                        check(mu.shape == (len(ids),) and sigma.shape == mu.shape and np.isfinite(mu).all()
                              and np.isfinite(sigma).all() and (sigma >= 0).all(), 'invalid prediction arrays')
                        if verify_gp and not checked_gp:
                            error = _direct_gp_check(task, history, settings, seed, mu, sigma)
                            gp_checks.append({'task_id': td['task_id'], 'seed': seed, 'round': t + 1,
                                              'max_absolute_error': error})
                            checked_gp = True
                        scores = mu + settings['kappa'] * sigma
                        available = np.array([key not in observed_ids for key in ids])
                        suggestion = event['suggested_candidate_id']
                        if method == 'B3':
                            if assistant:
                                while queue and queue[0] in observed_ids:
                                    queue.pop(0)
                                expected_suggestion = queue.pop(0) if queue else None
                            else:
                                expected_suggestion = next((k for k in bundle['priors'][regime]['priority_ids']
                                                            if k not in observed_ids), None)
                            check(suggestion == expected_suggestion, 'B3 suggestions not from its budgeted queue')
                        budget = settings['budget_end'] + (settings['budget_start'] - settings['budget_end']) * (
                            additional - 1 - t) / (additional - 1) if additional > 1 else settings['budget_end']
                        index, retained, near, groups = independent_choice(
                            method, scores, h, budget, ids, hids, available, rng, t + 1,
                            settings['escape_rounds'], suggestion)
                        selection = event['selection']
                        check(selection['budget'] == budget and tuple(selection['retained_ids']) == retained,
                              'subset or budget differs from independent reconstruction')
                        if index is None:
                            check(selection['status'] == 'STOPPED' and t == len(state['decisions']),
                                  'invalid stop on a nonempty conjunction')
                            totals['independently_verified_stops'] += 1
                            continue
                        check(selection['candidate_index'] == index and selection['candidate_id'] == ids[index]
                              and ledger[initial + t]['candidate_id'] == ids[index], 'policy decision differs')
                        base = min(np.flatnonzero(available), key=lambda i: (-float(scores[i]), ids[i]))
                        check(event['ucb_reference_candidate_id'] == ids[base], 'incorrect full-pool UCB reference')
                        loss = float(scores[base] - scores[index])
                        check(selection['opportunity_loss'] == loss and selection['budget_violation'] == (loss > budget),
                              'opportunity loss differs')
                        check(tuple(selection['revoked_ids']) == tuple(sorted(set(hids) - set(retained))), 'incorrect revocation')
                        joint = bool(len(h) and all((r & near).any() for r in h) and not (h.all(axis=0) & near).any())
                        check(event['joint_only_conflict'] == joint, 'incorrect joint-only conflict flag')
                        joint_conflicts += joint
                        changed += index != base
                        if index != base:
                            chosen_gaps.append(true_y[ids[index]] - true_y[ids[base]])
                        if method not in {'B0', 'B3'}:
                            retention_n += len(retained)
                            retention_d += len(hids)
                        violations += loss > budget
                        if method == 'B4':
                            check(loss <= budget, 'B4 opportunity budget violated')
                            detail = selection['guard_detail']
                            checks = {tuple(r['retained_ids']): r for r in detail['subset_checks']}
                            check(len(checks) == 2 ** len(hids), 'incomplete exact subset receipt')
                            for names, mask in groups:
                                entry = checks[names]
                                check(entry['feasible_count'] == int(mask.sum()) and
                                      entry['eligible_count'] == int((mask & near).sum()), 'subset receipt counts differ')
                            conflicts = [names for names, mask in groups if names and not (mask & near).any()]
                            minimal = sorted(names for names in conflicts if not any(set(parent) < set(names)
                                                                                   for parent in conflicts))
                            check([tuple(x) for x in detail['minimal_conflict_sets']] == minimal,
                                  'inclusion-minimal conflict diagnosis differs')
                            totals['independent_b4_exhaustive_decisions'] += 1
                        totals['independently_verified_decisions'] += 1
                        for name in ('gp_fit_seconds_actual', 'gp_predict_seconds_actual', 'guard_seconds'):
                            cost[name] += event[name]
                    curve, auc = independent_curve(ledger, initial, additional)
                    optimum = max(true_y.values())
                    check(np.allclose(curve, saved_result['best_curve'], rtol=0, atol=1e-12)
                          and abs(auc - saved_result['search_auc']) <= 1e-12
                          and abs(optimum - curve[-1] - saved_result['final_regret']) <= 1e-12,
                          'saved optimization metrics differ from ledger reconstruction')
                    row = {'task_id': td['task_id'], 'family': td['family'], 'seed': seed, 'regime': regime,
                           'method': method, 'status': state['status'], 'actual_queries': len(ledger),
                           'best_curve': curve, 'search_auc': auc, 'final_best': curve[-1],
                           'final_regret': optimum - curve[-1], 'best_at_12': curve[min(11, len(curve) - 1)],
                           'llm_attempts': len(llm_events), 'budget_violations': int(violations),
                           'changed_decisions': int(changed), 'changed_true_gain': chosen_gaps,
                           'joint_only_conflicts': int(joint_conflicts), 'retained_count': retention_n,
                           'possible_retained_count': retention_d, 'timing': dict(cost),
                           'active_seconds': state['timing']['active_seconds']}
                    rows.append(row)
                    totals['campaigns'] += 1
                    totals['oracle_queries'] += len(ledger)
                    artifact_hashes[str(state_path.relative_to(root))] = sha256(state_path.read_bytes()).hexdigest()
                    artifact_hashes[str(result_path.relative_to(root))] = sha256(result_path.read_bytes()).hexdigest()
    check(totals['campaigns'] == manifest['planned_campaigns'], 'planned campaigns are missing')
    check(len(list((root / 'trajectories').glob('**/state.json'))) == totals['campaigns'], 'unplanned trajectories exist')
    output = {'schema_version': 1, 'status': 'PASS', 'suite': root.name,
              'manifest_sha256': sha256((root / 'manifest.json').read_bytes()).hexdigest(),
              'auditor_sha256': sha256(Path(__file__).read_bytes()).hexdigest(), 'engine_sha256': expected_engine,
              'counters': dict(totals), 'independent_gp_checks': gp_checks,
              'artifact_hashes': artifact_hashes, 'rows': rows,
              'audit_seconds': time.perf_counter() - started,
              'scope': 'ledger, provenance, hard domain, all policy decisions, B4 exhaustive subsets and full-budget metrics'}
    write_json_atomic(root / 'independent_audit.json', output)
    return output


def summarize_audit(audit):
    rows = audit['rows']
    tables = []
    for regime in sorted({r['regime'] for r in rows}):
        for method in sorted({r['method'] for r in rows}):
            selected = [r for r in rows if r['regime'] == regime and r['method'] == method]
            tables.append({'regime': regime, 'method': method, 'campaigns': len(selected),
                           'stopped': sum(r['status'] == 'STOPPED' for r in selected),
                           **{key: float(np.mean([r[key] for r in selected])) for key in (
                               'search_auc', 'final_best', 'final_regret', 'actual_queries', 'changed_decisions')},
                           'budget_violations': sum(r['budget_violations'] for r in selected),
                           'joint_only_conflicts': sum(r['joint_only_conflicts'] for r in selected),
                           'changed_true_gain_mean': float(np.mean([v for r in selected for v in r['changed_true_gain']]))
                           if any(r['changed_true_gain'] for r in selected) else None})
    by_id = {(r['task_id'], r['seed'], r['regime'], r['method']): r for r in rows}
    comparisons = []
    regimes = sorted({r['regime'] for r in rows})
    for regime in [*regimes, 'all_regimes']:
        current_regimes = regimes if regime == 'all_regimes' else [regime]
        for other in ('B0', 'B1', 'B2', 'B3', 'A1', 'A2', 'A3'):
            if other not in {r['method'] for r in rows}:
                continue
            paired = []
            for task in sorted({r['task_id'] for r in rows}):
                selected = [r for r in rows if r['task_id'] == task and r['method'] == 'B4'
                            and r['regime'] in current_regimes]
                paired.append({'task_id': task, 'family': selected[0]['family'],
                               **{key: float(np.mean([r[key] - by_id[(task, r['seed'], r['regime'], other)][key]
                                                     for r in selected])) for key in ('search_auc', 'final_regret')}})
            clusters = []
            for family in sorted({p['family'] for p in paired}):
                same = [r for r in paired if r['family'] == family]
                clusters.append({'family': family, **{key: float(np.mean([r[key] for r in same]))
                                                      for key in ('search_auc', 'final_regret')}})
            result = {'regime': regime, 'comparison': 'B4_minus_' + other, 'tasks': paired,
                      'clusters': clusters, 'independent_clusters': len(clusters),
                      'delta_search_auc': float(np.mean([r['search_auc'] for r in paired])),
                      'delta_final_regret': float(np.mean([r['final_regret'] for r in paired]))}
            if len(clusters) >= 5:
                rng = np.random.default_rng(20261026)
                indices = rng.integers(0, len(clusters), size=(10000, len(clusters)))
                result['exploratory_family_bootstrap_95pct'] = {
                    key: np.quantile(np.array([r[key] for r in clusters])[indices].mean(axis=1), [.025, .975]).tolist()
                    for key in ('search_auc', 'final_regret')}
            comparisons.append(result)
    recovery = []
    for regime in [r for r in regimes if r not in {'relevant', 'actual_assistant'}]:
        for method in sorted({r['method'] for r in rows}):
            selected = [r for r in rows if r['regime'] == regime and r['method'] == method]
            differences = [r['final_regret'] - by_id[(r['task_id'], r['seed'], regime, 'B0')]['final_regret']
                           for r in selected]
            recovery.append({'regime': regime, 'method': method, 'within_margin': sum(d <= .02 for d in differences),
                             'denominator': len(selected), 'margin': .02,
                             'mean_regret_difference_from_B0': float(np.mean(differences))})
    return {'suite': audit['suite'], 'audit_status': audit['status'], 'counters': audit['counters'],
            'tables': tables, 'paired_comparisons': comparisons, 'optimization_recovery': recovery,
            'statistics': {'unit': 'seed-paired task means; mathematical uncertainty clustered over five families',
                           'seed_count_is_not_independent_system_count': True,
                           'bootstrap_scope': 'exploratory percentile intervals only; no confirmatory significance claim',
                           'reaction_intervals': 'not estimated from only three test cases',
                           'correct_prior_noninferiority_margin': .02}}
