"""Deterministic mechanism checks and material-input audit, not BO results."""
from __future__ import annotations

from hashlib import sha256
from itertools import combinations
from pathlib import Path
import platform
from dataclasses import asdict, dataclass
from collections import Counter
from copy import deepcopy
import importlib.metadata
import inspect
import json
import time
from uuid import uuid4
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from .domain_guard import METHODS, select_candidate, select_policy
from .hypothesis import (FeatureEncoder, HypothesisManager, LLMResponsePending,
                         canonical_hash)
from .replay import audit_cealge_dataset, replace_file_with_retry, write_json_atomic
from .replay import (InteractiveAssistantProvider, MATH_REGIMES, ReplayOracle, SynthesisTask,
                     make_reaction_stress_priors, make_stress_priors, mathematical_task)


def _reference(scores, h, budget, ids, hids, hard, queried):
    """Independent Python set enumeration for validating the NumPy policy."""
    available = [i for i in range(len(ids)) if hard[i] and not queried[i]]
    best = max(float(scores[i]) for i in available)
    eligible = {i for i in available if best - float(scores[i]) <= budget}
    valid = []
    for size in range(len(hids) + 1):
        for group in combinations(range(len(hids)), size):
            possible = [i for i in eligible if all(h[j, i] for j in group)]
            if possible:
                kept = tuple(sorted(hids[j] for j in group))
                chosen = min(possible, key=lambda i: (-float(scores[i]), ids[i]))
                valid.append((kept, chosen))
    return min(valid, key=lambda pair: (-len(pair[0]), pair[0]))


def _fixtures() -> list[dict]:
    # These are acquisition snapshots. They are deliberately not described
    # as the proposal's twenty mathematical BO benchmark problems.
    rows = [
        ("no_hypotheses", [1, .9, .2], [], 0),
        ("all_advice_compatible", [1, .9, .2], [[1, 1, 0], [1, 0, 1]], .05),
        ("one_wrong_prior", [1, .8, .2], [[0, 1, 1]], .05),
        ("joint_quality_conflict", [1, .99, .2], [[1, 0, 1], [0, 1, 1]], .02),
        ("joint_empty_conflict", [1, .99, .2], [[1, 0, 0], [0, 1, 0]], .02),
        ("revoke_two", [1, .99, .98], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], .03),
        ("all_four_wrong", [1, .9, .2], [[0, 0, 0]] * 4, .05),
        ("four_correct", [1, .9, .2], [[1, 0, 0]] * 4, .05),
        ("zero_budget_unique_max", [1, .999, .2], [[0, 1, 1]], 0),
        ("zero_budget_tied_max", [1, 1, .2], [[0, 1, 0]], 0),
        ("inclusive_budget_boundary", [1, .875, .2], [[0, 1, 1]], .125),
        ("just_outside_budget", [1, .874, .2], [[0, 1, 1]], .125),
        ("negative_acquisition", [-1, -1.02, -2], [[0, 1, 0]], .03),
        ("large_budget", [1, .3, .2], [[0, 1, 0]], 1),
        ("three_way_only_conflict", [1, .99, .98], [[1, 1, 0], [0, 1, 1], [1, 0, 1]], .03),
        ("two_independent_bad_priors", [1, .9, .2], [[0, 1, 0], [0, 0, 1], [1, 1, 1]], .05),
        ("hard_domain_excludes_max", [2, 1, .99], [[0, 1, 1]], .02),
        ("queried_max_is_removed", [2, 1, .99], [[0, 1, 1]], .02),
        ("candidate_tie_uses_id", [1, 1, .2], [[1, 1, 1]], .05),
        ("subset_tie_uses_ids", [1, .99, .2], [[0, 1, 0], [1, 0, 0]], .02),
    ]
    for name, scores, membership, budget in rows:
        n = len(scores)
        h = np.asarray(membership, dtype=bool).reshape((-1, n))
        hard, queried = np.ones(n, dtype=bool), np.zeros(n, dtype=bool)
        if name == "hard_domain_excludes_max":
            hard[0] = False
        if name == "queried_max_is_removed":
            queried[0] = True
        result = {"name": name, "scores": scores, "h": h, "budget": budget,
                  "ids": ["c2", "c1", "c3"], "hids": [f"H{j + 1}" for j in range(len(h))],
                  "hard": hard, "queried": queried}
        yield result


def _check(case: dict) -> dict:
    scores, h = case["scores"], case["h"]
    actual = select_candidate(scores, h, case["budget"], candidate_ids=case["ids"],
                              hypothesis_ids=case["hids"], hard_mask=case["hard"], queried_mask=case["queried"])
    expected_kept, expected_index = _reference(scores, h, case["budget"], case["ids"], case["hids"],
                                               case["hard"], case["queried"])
    if actual.retained_ids != expected_kept or actual.candidate_index != expected_index:
        raise AssertionError(f"exhaustive-reference mismatch: {case['name']}")
    if not 0 <= actual.opportunity_loss <= case["budget"]:
        raise AssertionError(f"budget violation: {case['name']}")
    if not case["hard"][actual.candidate_index] or case["queried"][actual.candidate_index]:
        raise AssertionError(f"hard-domain or repeat-query violation: {case['name']}")
    if any(not h[case["hids"].index(key), actual.candidate_index] for key in actual.retained_ids):
        raise AssertionError(f"selected candidate violates retained advice: {case['name']}")
    return {"case": case["name"], "status": "PASS", **actual.to_dict()}


def run_mechanism_suite(random_states: int = 250, seed: int = 20261026) -> dict:
    if type(random_states) is not int or random_states < 0:
        raise ValueError("random_states must be a nonnegative integer")
    fixture_results = [_check(case) for case in _fixtures()]
    rng = np.random.default_rng(seed)
    subset_count = sum(len(row["subset_checks"]) for row in fixture_results)
    for j in range(random_states):
        n, m = int(rng.integers(1, 65)), int(rng.integers(0, 5))
        hard, queried = rng.random(n) > .15, rng.random(n) < .25
        hard[0], queried[0] = True, False
        case = {"name": f"random_{j}", "scores": rng.normal(size=n), "h": rng.random((m, n)) < .6,
                "budget": float(rng.uniform(0, .3)), "ids": [f"c{i:04d}" for i in range(n)],
                "hids": [f"H{i + 1}" for i in range(m)], "hard": hard, "queried": queried}
        row = _check(case)
        subset_count += len(row["subset_checks"])
    source_dir = Path(__file__).parent
    return {
        "schema_version": 1, "status": "PASS", "scope": "finite-pool acquisition mechanism only",
        "fixture_count": len(fixture_results), "random_state_count": random_states,
        "total_decisions_checked": len(fixture_results) + random_states,
        "total_subsets_checked": subset_count, "seed": seed,
        "hard_constraint_violations": 0, "opportunity_budget_violations": 0,
        "minimum_revocation_or_tie_break_mismatches": 0,
        "python": platform.python_version(), "numpy": np.__version__,
        "source_sha256": {p.name: sha256(p.read_bytes()).hexdigest() for p in sorted(source_dir.glob("*.py"))},
        "does_not_establish": ["twenty mathematical BO benchmarks completed", "GPR or LLM integration qualified",
                               "chemical optimization gains", "real-world synthesis success"],
        "fixtures": fixture_results,
    }


def run_data_audit(dataset_dir: str | Path) -> dict:
    source = Path(dataset_dir)
    paths = sorted(source.glob("*rietveld*.json"))
    if not paths:
        raise ValueError(f"no Rietveld datasets in {source}")
    return {"schema_version": 1, "scope": "read-only numeric input audit",
            "datasets": [audit_cealge_dataset(path) for path in paths]}


@dataclass(frozen=True)
class CampaignSettings:
    initial_queries: int = 4
    additional_queries: int = 16
    kappa: float = 2.0
    budget_start: float = .05
    budget_end: float = 0.0
    gp_restarts: int = 5
    gp_alpha: float = 1e-6
    base_seed: int = 20261026
    max_llm_attempts: int = 4
    refresh_after: tuple[int, ...] = (0, 4, 8, 12)
    escape_rounds: tuple[int, ...] = (4, 8, 12, 16)

    def __post_init__(self):
        if (self.initial_queries < 2 or self.additional_queries < 1
                or not 0 <= self.budget_end <= self.budget_start
                or not np.isfinite([self.kappa, self.budget_start, self.budget_end, self.gp_alpha]).all()
                or self.kappa < 0 or self.gp_alpha <= 0 or self.gp_restarts < 0):
            raise ValueError("invalid fixed campaign configuration")

    def budget(self, zero_based_round):
        if not 0 <= zero_based_round < self.additional_queries:
            raise ValueError("round is outside the fixed query budget")
        if self.additional_queries == 1:
            return self.budget_end
        return self.budget_end + (self.budget_start - self.budget_end) * (
            self.additional_queries - 1 - zero_based_round) / (self.additional_queries - 1)

    @classmethod
    def from_protocol(cls, protocol):
        shared, start = protocol['shared_design'], protocol['development_starting_values']
        return cls(initial_queries=shared['initial_queries'], additional_queries=shared['additional_queries'],
                   kappa=start['kappa'], budget_start=start['budget_schedule']['first_additional_query'],
                   budget_end=start['budget_schedule']['last_additional_query'],
                   gp_restarts=start['gp']['n_restarts_optimizer'],
                   gp_alpha=start['gp']['deterministic_oracle_alpha'],
                   base_seed=shared['pool_generation_seed_base'],
                   max_llm_attempts=protocol['llm']['max_attempts_per_task_method_seed'],
                   refresh_after=tuple(protocol['llm']['call_after_additional_queries']),
                   escape_rounds=tuple(protocol['b2']['escape_additional_query_numbers']))


def engine_fingerprint():
    """Hash decision semantics; adding an unrelated report does not reset runs."""
    from ..bo.llm import GPRModel
    functions = (GPRModel, CampaignSettings, FeatureEncoder, HypothesisManager, ReplayOracle, SynthesisTask,
                 InteractiveAssistantProvider,
                 select_candidate, select_policy, _reference, _predict_snapshot, run_campaign)
    return canonical_hash({
        'sources': {item.__name__: inspect.getsource(item) for item in functions},
        'software': {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'scikit-learn')},
    })


def _predict_snapshot(task, encoded, ledger, settings, seed, cache_dir, memory_cache):
    """One fit/batch prediction per unique history; reuse exact shared prefixes."""
    from ..bo.llm import GPRModel
    indices = {key: i for i, key in enumerate(task.candidate_ids)}
    ordered = sorted(ledger, key=lambda row: row['candidate_id'])
    definition = {'task_sha256': task.fingerprint,
                  'history': [{k: row[k] for k in ('candidate_id', 'value', 'sigma') if k in row} for row in ordered],
                  'restarts': settings.gp_restarts, 'alpha': settings.gp_alpha,
                  'random_state': settings.base_seed + seed, 'normalize_y': False,
                  'gp_source': inspect.getsource(GPRModel),
                  'software': {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'scikit-learn')}}
    key = canonical_hash(definition)
    path = Path(cache_dir) / f'{key}.npz'
    if key in memory_cache:
        return memory_cache[key], True
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            snapshot = {'mu': saved['mu'], 'sigma': saved['sigma'],
                        'meta': json.loads(str(saved['metadata']))}
        if (snapshot['meta']['key'] != key or snapshot['mu'].shape != (len(encoded),)
                or snapshot['sigma'].shape != (len(encoded),)
                or not np.isfinite(snapshot['mu']).all() or not np.isfinite(snapshot['sigma']).all()
                or (snapshot['sigma'] < 0).any()):
            raise ValueError('cached GP prediction is invalid')
        memory_cache[key] = snapshot
        return snapshot, True
    gp = GPRModel({name: (0, 1) for name in encoded[0]}, normalize_y=False,
                  n_restarts_optimizer=settings.gp_restarts,
                  random_state=settings.base_seed + seed, alpha=settings.gp_alpha)
    train = [encoded[indices[row['candidate_id']]] for row in ordered]
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        started = time.perf_counter()
        uncertainties = [row.get('sigma') for row in ordered]
        if any(s is None for s in uncertainties) and any(s is not None for s in uncertainties):
            raise ValueError('observations mix incompatible uncertainty protocols')
        gp.fit(train, [row['value'] for row in ordered],
               y_sigma=None if uncertainties[0] is None else uncertainties)
        fitted = time.perf_counter()
        mu, sigma = gp.predict_many(encoded)
        predicted = time.perf_counter()
    metadata = {'key': key, 'observations': len(ordered), 'fit_seconds': fitted - started,
                'predict_seconds': predicted - fitted, 'kernel': str(gp._gpr.kernel_),
                'warning_counts': dict(Counter(w.category.__name__ for w in captured)),
                'warning_messages': sorted({str(w.message) for w in captured}),
                'dtype': str(mu.dtype), 'prediction_scope': 'fixed_full_pool_with_queried_mask'}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{key}.{uuid4().hex}.npz')
    np.savez_compressed(temporary, mu=mu, sigma=sigma,
                        metadata=np.array(json.dumps(metadata, allow_nan=False)))
    temporary.replace(path)
    snapshot = {'mu': mu, 'sigma': sigma, 'meta': metadata}
    memory_cache[key] = snapshot
    return snapshot, False


def run_campaign(task, oracle, method, seed, settings, initial_ids, *, output_dir,
                 cache_dir, fixed_batch=None, priority_ids=(), provider=None,
                 memory_cache=None, encoded=None, max_new_steps=None):
    """Sequential optimization with resumable observations and model events.

    Only this orchestration layer queries the oracle; select_policy receives
    scores and membership only. No pool optimum enters model contexts.
    """
    if method not in METHODS:
        raise ValueError('unknown method')
    if type(seed) is not int or seed < 0 or (max_new_steps is not None and
                                           (type(max_new_steps) is not int or max_new_steps < 0)):
        raise ValueError('seed and optional checkpoint step limit must be nonnegative integers')
    initial_ids = tuple(initial_ids)
    if len(initial_ids) != settings.initial_queries or len(set(initial_ids)) != len(initial_ids):
        raise ValueError('initial IDs must be distinct and match the frozen budget')
    if any(key not in task.candidate_ids for key in initial_ids):
        raise ValueError('initial point is outside the candidate pool')
    if fixed_batch is not None and provider is not None:
        raise ValueError('fixed hypotheses and adaptive model proposals are separate experiment tracks')
    output = Path(output_dir)
    checkpoint = output / 'state.json'
    header = {'schema_version': 1, 'task_id': task.task_id, 'task_sha256': task.fingerprint,
              'oracle_sha256': oracle.fingerprint,
              'method': method, 'seed': seed, 'settings': asdict(settings),
              'initial_ids': list(initial_ids), 'fixed_batch': fixed_batch,
              'priority_sha256': canonical_hash(list(priority_ids)),
              'provider_metadata': getattr(provider, 'model_metadata', None),
              'engine_sha256': engine_fingerprint()}
    header_hash = canonical_hash(header)
    manager = HypothesisManager(task.schema, task.candidates, task.candidate_ids,
                                [e['id'] for e in task.evidence],
                                max_attempts=settings.max_llm_attempts, refresh_after=settings.refresh_after)
    rng = np.random.default_rng(settings.base_seed + seed)
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text(encoding='utf-8'))
        if state['header_sha256'] != header_hash or canonical_hash(state['header']) != header_hash:
            raise ValueError('checkpoint task, settings, provider, or decision code changed; refusing restart')
        oracle.restore(state['oracle_ledger'])
        manager.restore(state['hypothesis_state'], observed_ids={r['candidate_id'] for r in oracle.ledger})
        rng.bit_generator.state = state['rng_state']
        if state['status'] in {'COMPLETE', 'STOPPED', 'MODEL_ERROR', 'INVARIANT_FAILURE'}:
            return state
    else:
        if oracle.ledger:
            raise ValueError('a fresh campaign requires an empty oracle ledger')
        for key in initial_ids:
            oracle.query(key)
        if fixed_batch is not None and method != 'B0':
            manager.install_fixed(fixed_batch, observed_ids=initial_ids)
        state = {'header': header, 'header_sha256': header_hash, 'status': 'RUNNING',
                 'decisions': [], 'oracle_ledger': [], 'timing': {'active_seconds': 0.0},
                 'operator_compiles_in_loop': 0}
    memory_cache = {} if memory_cache is None else memory_cache
    encoded = FeatureEncoder(task.schema).encode(task.candidates) if encoded is None else encoded
    indices = {key: i for i, key in enumerate(task.candidate_ids)}
    started, completed_here = time.perf_counter(), 0

    def save():
        state['oracle_ledger'] = list(oracle.ledger)
        state['hypothesis_state'] = manager.state_dict()
        state['rng_state'] = rng.bit_generator.state
        state['timing']['active_seconds'] += time.perf_counter() - save.last_time
        save.last_time = time.perf_counter()
        write_json_atomic(checkpoint, state)
    save.last_time = started
    state['status'] = 'RUNNING'
    state.pop('pending_request', None)
    save()
    while len(state['decisions']) < settings.additional_queries:
        t = len(state['decisions'])
        if max_new_steps is not None and completed_here >= max_new_steps:
            state['status'] = 'CHECKPOINTED'
            break
        queried_ids = {row['candidate_id'] for row in oracle.ledger}
        queried = np.array([key in queried_ids for key in task.candidate_ids], dtype=bool)
        if queried.all():
            state.update(status='STOPPED', stop_reason='hard_feasible_pool_exhausted')
            break
        if method != 'B0' and provider is not None:
            context = {'task': task.to_dict(), 'observations': [
                {'candidate_id': row['candidate_id'], 'params': dict(task.candidates[indices[row['candidate_id']]]),
                 'value': row['value'], 'sigma': row.get('sigma')} for row in oracle.ledger],
                'completed_additional_queries': t,
                'objective_unit': 'predefined_normalized_objective',
                'remaining_additional_queries': settings.additional_queries - t,
                'protocol_prompt': getattr(provider, 'prompt_text', None)}
            try:
                previous_attempts = manager.attempts
                manager.refresh(t, context, provider)
                if manager.attempts != previous_attempts:
                    save()
            except LLMResponsePending as pending:
                state.update(status='WAITING_LLM', pending_request=pending.request_path)
                break
        try:
            snapshot, cache_hit = _predict_snapshot(task, encoded, oracle.ledger, settings,
                                                   seed, cache_dir, memory_cache)
        except Exception as exc:
            state.update(status='MODEL_ERROR', error=f'{type(exc).__name__}: {exc}')
            break
        scores = snapshot['mu'] + settings.kappa * snapshot['sigma']
        budget = settings.budget(t)
        suggestion = None
        if method == 'B3':
            if provider is not None:
                suggestion = manager.pop_suggestion(queried_ids)
            else:
                suggestion = next((key for key in priority_ids if key not in queried_ids), None)
        selected_at = time.perf_counter()
        choice = select_policy(method, scores, manager.membership, budget,
                               candidate_ids=task.candidate_ids, hypothesis_ids=manager.hypotheses,
                               queried_mask=queried, round_number=t + 1, rng=rng,
                               suggested_candidate_id=suggestion, escape_rounds=settings.escape_rounds)
        event = {'round': t + 1, 'observed_ids': [row['candidate_id'] for row in oracle.ledger],
                 'ucb_reference_candidate_id': min((i for i in range(len(scores)) if not queried[i]),
                                                   key=lambda i: (-float(scores[i]), task.candidate_ids[i])),
                 'hypothesis_ids': manager.hypotheses, 'hypothesis_batch_sha256': canonical_hash(manager.state_dict()['payload']),
                 'hypothesis_version': manager.state_dict()['payload']['version'],
                 'prediction_key': snapshot['meta']['key'], 'gp_cache_hit': cache_hit,
                 'gp_fit_seconds_equivalent': snapshot['meta']['fit_seconds'],
                 'gp_predict_seconds_equivalent': snapshot['meta']['predict_seconds'],
                 'gp_fit_seconds_actual': 0.0 if cache_hit else snapshot['meta']['fit_seconds'],
                 'gp_predict_seconds_actual': 0.0 if cache_hit else snapshot['meta']['predict_seconds'],
                 'warning_counts': snapshot['meta']['warning_counts'], 'kappa': settings.kappa,
                 'suggested_candidate_id': suggestion, 'selection': choice,
                 'guard_seconds': time.perf_counter() - selected_at}
        event['ucb_reference_candidate_id'] = task.candidate_ids[event['ucb_reference_candidate_id']]
        eligible = ~queried & (float(scores[~queried].max()) - scores <= budget)
        h = manager.membership
        event['joint_only_conflict'] = bool(len(h) and all((row & eligible).any() for row in h)
                                             and not (h.all(axis=0) & eligible).any())
        if choice['status'] == 'STOPPED':
            state.update(status='STOPPED', stop_reason=choice['reason'], failed_decision=event)
            break
        index = choice['candidate_index']
        if queried[index] or choice['candidate_id'] != task.candidate_ids[index]:
            state.update(status='INVARIANT_FAILURE', error='repeat query or candidate identity mismatch', failed_decision=event)
            break
        if method == 'B4':
            kept, expected = _reference(scores, manager.membership, budget, task.candidate_ids,
                                         manager.hypotheses, np.ones(len(scores), dtype=bool), queried)
            if choice['budget_violation'] or expected != index or kept != tuple(choice['retained_ids']):
                state.update(status='INVARIANT_FAILURE', error='joint decision failed independent exhaustive check', failed_decision=event)
                break
        value = oracle.query(choice['candidate_id'])
        event['observed_value'] = value
        state['decisions'].append(event)
        completed_here += 1
        save()
    if len(state['decisions']) == settings.additional_queries:
        state['status'] = 'COMPLETE'
    save()
    return state


def score_campaign(state, candidate_ids, values):
    """Post-hoc evaluator. Full outcomes are never supplied to the policy."""
    if state['status'] not in {'COMPLETE', 'STOPPED', 'MODEL_ERROR', 'INVARIANT_FAILURE'}:
        raise ValueError('pending or checkpointed trajectories are not final experiment results')
    values = np.asarray(values, dtype=float)
    if values.shape != (len(candidate_ids),) or not np.isfinite(values).all():
        raise ValueError('evaluation outcomes must align with the frozen pool')
    by_id = dict(zip(candidate_ids, map(float, values)))
    n_init = state['header']['settings']['initial_queries']
    extra = state['header']['settings']['additional_queries']
    observed = [row['value'] for row in state['oracle_ledger']]
    if len(observed) < n_init:
        raise ValueError('missing frozen initialization observations')
    best = np.maximum.accumulate(observed).tolist()
    best.extend([best[-1]] * (n_init + extra - len(best)))
    optimum = float(values.max())
    decisions = state['decisions']
    return {'task_id': state['header']['task_id'], 'method': state['header']['method'],
            'engine_sha256': state['header']['engine_sha256'],
            'task_sha256': state['header']['task_sha256'],
            'oracle_sha256': state['header']['oracle_sha256'],
            'settings_sha256': canonical_hash(state['header']['settings']),
            'seed': state['header']['seed'], 'status': state['status'],
            'actual_queries': len(observed), 'planned_queries': n_init + extra,
            'best_curve': best, 'final_best': best[-1], 'pool_optimum': optimum,
            'regret_curve': [optimum - value for value in best], 'final_regret': optimum - best[-1],
            'search_auc': float(np.mean(best[n_init:n_init + extra])),
            'initial_best': best[n_init - 1], 'llm_attempts': len(state['hypothesis_state']['events']),
            'budget_violations': sum(event['selection']['budget_violation'] for event in decisions),
            'repeat_queries': len(observed) - len({r['candidate_id'] for r in state['oracle_ledger']}),
            'b4_exhaustive_mismatches': int(state['status'] == 'INVARIANT_FAILURE'),
            'mean_revoked': float(np.mean([len(event['selection']['revoked_ids']) for event in decisions
                                          if event['selection']['hypothesis_subset_decision']]))
                            if any(event['selection']['hypothesis_subset_decision'] for event in decisions) else None,
            'hard_constraint_violations': sum(event['selection']['candidate_id'] not in by_id for event in decisions),
            'knowledge_changed_selection': sum(event['selection']['candidate_id'] != event['ucb_reference_candidate_id'] for event in decisions),
            'joint_only_conflict_rounds': sum(event['joint_only_conflict'] for event in decisions),
            'true_one_step_opportunity_gap': [
                by_id[event['ucb_reference_candidate_id']] - event['observed_value'] for event in decisions],
            'timing': state['timing'], 'operator_compiles_in_loop': state['operator_compiles_in_loop']}


def prepare_math_suite(directory, protocol, *, split='development'):
    """Generate immutable inputs without running the optimization policies."""
    if split not in {'development', 'test'}:
        raise ValueError('unknown mathematical split')
    root = Path(directory).resolve()
    manifest_path = root / 'manifest.json'
    settings = CampaignSettings.from_protocol(protocol)
    seeds = [17] if split == 'development' else list(protocol['shared_design']['paired_seeds'])
    instances = [4] if split == 'development' else list(protocol['math']['instance_r'])
    definition = {'split': split, 'settings': asdict(settings), 'paired_seeds': seeds,
                  'instances': instances, 'families': [1, 2, 3, 4, 5],
                  'prior_regimes': list(MATH_REGIMES), 'methods': list(METHODS),
                  'pool_size': 512, 'pool_seed_base': protocol['shared_design']['pool_generation_seed_base'],
                  'generators_sha256': canonical_hash({'task': inspect.getsource(mathematical_task),
                                                       'priors': inspect.getsource(make_stress_priors)})}
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding='utf-8'))
        if canonical_hash(saved['definition']) != canonical_hash(definition):
            raise ValueError('existing suite definition changed; refusing to replace its inputs')
        _verify_suite_inputs(root, saved)
        return saved
    root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(root / 'protocol_snapshot.json', protocol)
    tasks = []
    for family in range(1, 6):
        for instance in instances:
            pool_seed = settings.base_seed + family * 100 + instance + (10000 if split == 'development' else 0)
            task, values = mathematical_task(family, instance, pool_seed=pool_seed)
            priors = make_stress_priors(task, values)
            initial = {str(seed): [task.candidate_ids[i] for i in np.random.default_rng(seed).choice(
                len(task.candidate_ids), settings.initial_queries, replace=False)] for seed in seeds}
            folder = root / 'inputs' / task.task_id
            folder.mkdir(parents=True, exist_ok=True)
            oracle_path = folder / 'oracle.npz'
            np.savez_compressed(oracle_path, values=values)
            bundle_path = folder / 'bundle.json'
            bundle = {'task': task.to_dict(), 'priors': priors, 'initial_ids': initial,
                      'pool_seed': pool_seed, 'oracle_sha256': sha256(oracle_path.read_bytes()).hexdigest()}
            write_json_atomic(bundle_path, bundle)
            tasks.append({'task_id': task.task_id, 'family': f'F{family}', 'instance': instance,
                          'pool_seed': pool_seed, 'task_sha256': task.fingerprint,
                          'bundle': bundle_path.relative_to(root).as_posix(),
                          'bundle_sha256': sha256(bundle_path.read_bytes()).hexdigest(),
                          'oracle': oracle_path.relative_to(root).as_posix(),
                          'oracle_sha256': bundle['oracle_sha256']})
    manifest = {'schema_version': 1, 'definition': definition, 'tasks': tasks,
                'planned_groups': len(tasks) * len(seeds),
                'planned_campaigns': len(tasks) * len(seeds) * len(METHODS) * len(MATH_REGIMES),
                'scope': 'fixed oracle-aware artificial priors, not natural LLM proposals'}
    write_json_atomic(manifest_path, manifest)
    return manifest


def _verify_suite_inputs(root, manifest):
    for task in manifest['tasks']:
        for key in ('bundle', 'oracle'):
            path = (Path(root) / task[key]).resolve()
            if not path.is_relative_to(Path(root).resolve()):
                raise ValueError('suite artifact escaped its declared root')
            if sha256(path.read_bytes()).hexdigest() != task[f'{key}_sha256']:
                raise ValueError(f'frozen {key} changed: {task["task_id"]}')


def _load_settings(value):
    cfg = dict(value)
    cfg['refresh_after'] = tuple(cfg['refresh_after'])
    cfg['escape_rounds'] = tuple(cfg['escape_rounds'])
    return CampaignSettings(**cfg)


def _run_math_group(arguments):
    from threadpoolctl import threadpool_limits
    # Import the numerical libraries before limiting all of their BLAS pools.
    import sklearn.gaussian_process
    import scipy.linalg
    root_name, task_definition, seed, settings_dict = arguments
    root, settings = Path(root_name), _load_settings(settings_dict)
    bundle_path = root / task_definition['bundle']
    oracle_path = root / task_definition['oracle']
    if (sha256(bundle_path.read_bytes()).hexdigest() != task_definition['bundle_sha256']
            or sha256(oracle_path.read_bytes()).hexdigest() != task_definition['oracle_sha256']):
        raise ValueError('worker input differs from the frozen manifest')
    bundle = json.loads(bundle_path.read_text(encoding='utf-8'))
    task = SynthesisTask.from_dict(bundle['task'])
    with np.load(oracle_path, allow_pickle=False) as saved:
        values = saved['values']
    encoded = FeatureEncoder(task.schema).encode(task.candidates)
    cache, results = {}, []
    with threadpool_limits(limits=1):
        for regime in MATH_REGIMES:
            prior = bundle['priors'][regime]
            for method in METHODS:
                folder = root / 'trajectories' / task.task_id / f's{seed}' / regime / method
                state = run_campaign(task, ReplayOracle(task.candidate_ids, values), method, seed,
                                     settings, bundle['initial_ids'][str(seed)], output_dir=folder,
                                     cache_dir=root / 'predictions', fixed_batch=prior['batch'],
                                     priority_ids=prior['priority_ids'], memory_cache=cache, encoded=encoded)
                result = score_campaign(state, task.candidate_ids, values)
                result.update(regime=regime, family=task_definition['family'],
                              trajectory=folder.relative_to(root).as_posix())
                write_json_atomic(folder / 'result.json', result)
                results.append(result)
    output = {'task_id': task.task_id, 'seed': seed, 'campaign_count': len(results), 'results': results}
    write_json_atomic(root / 'groups' / f'{task.task_id}_s{seed}.json', output)
    return {'task_id': task.task_id, 'seed': seed, 'campaign_count': len(results)}


def suite_status(directory):
    root = Path(directory).resolve()
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    rows = []
    for path in sorted((root / 'groups').glob('*.json')):
        rows.extend(json.loads(path.read_text(encoding='utf-8'))['results'])
    identities = [(row['task_id'], row['seed'], row['regime'], row['method']) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError('duplicate campaign identities in suite results')
    expected_ids = {(task['task_id'], seed, regime, method) for task in manifest['tasks']
                    for seed in manifest['definition']['paired_seeds']
                    for regime in MATH_REGIMES for method in METHODS}
    if set(identities) - expected_ids:
        raise ValueError('results contain campaigns outside the manifest')
    engine = engine_fingerprint()
    if any(row['engine_sha256'] != engine for row in rows):
        raise ValueError('saved results were produced by a different decision engine')
    expected = manifest['planned_campaigns']
    return {'schema_version': 1, 'suite': str(root), 'split': manifest['definition']['split'],
            'expected_campaigns': expected, 'reported_campaigns': len(rows),
            'complete_campaigns': sum(row['status'] == 'COMPLETE' for row in rows),
            'stopped_campaigns': sum(row['status'] == 'STOPPED' for row in rows),
            'numerical_or_invariant_errors': sum(row['status'] not in {'COMPLETE', 'STOPPED'} for row in rows),
            'hard_constraint_violations': sum(row['hard_constraint_violations'] for row in rows),
            'repeat_queries': sum(row['repeat_queries'] for row in rows),
            'b4_budget_violations': sum(row['budget_violations'] for row in rows if row['method'] == 'B4'),
            'b4_exhaustive_mismatches': sum(row['b4_exhaustive_mismatches'] for row in rows if row['method'] == 'B4'),
            'all_planned_reported': len(rows) == expected,
            'infrastructure_error_records': len(list((root / 'group_errors').glob('*.json'))),
            'engine_sha256': engine}


def run_math_suite(directory, *, workers=4, max_groups=None):
    if type(workers) is not int or workers < 1:
        raise ValueError('worker count must be positive')
    if max_groups is not None and (type(max_groups) is not int or max_groups < 0):
        raise ValueError('optional group limit must be a nonnegative integer')
    root = Path(directory).resolve()
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    _verify_suite_inputs(root, manifest)
    if manifest['definition']['split'] == 'test':
        freeze_path = root / 'freeze_receipt.json'
        if not freeze_path.exists():
            raise ValueError('formal test requires an explicit pre-test freeze after development qualification')
        freeze = json.loads(freeze_path.read_text(encoding='utf-8'))
        if (freeze['manifest_sha256'] != sha256(manifest_path.read_bytes()).hexdigest()
                or freeze['engine_sha256'] != engine_fingerprint()):
            raise ValueError('formal inputs or decision engine changed after freezing')
    jobs = []
    for task in manifest['tasks']:
        for seed in manifest['definition']['paired_seeds']:
            path = root / 'groups' / f'{task["task_id"]}_s{seed}.json'
            if not path.exists():
                jobs.append((str(root), task, seed, manifest['definition']['settings']))
    if max_groups is not None:
        jobs = jobs[:max_groups]
    write_json_atomic(root / 'run_status.json', {**suite_status(root), 'scheduled_groups': len(jobs), 'workers': workers})
    if workers == 1:
        for job in jobs:
            message = _run_math_group(job)
            print(json.dumps(message), flush=True)
            write_json_atomic(root / 'run_status.json', suite_status(root))
    elif jobs:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run_math_group, job): (job[1]['task_id'], job[2]) for job in jobs}
            for future in as_completed(futures):
                try:
                    print(json.dumps(future.result()), flush=True)
                except Exception as exc:
                    task_id, seed = futures[future]
                    write_json_atomic(root / 'group_errors' / f'{task_id}_s{seed}.json',
                                      {'task_id': task_id, 'seed': seed, 'error': f'{type(exc).__name__}: {exc}'})
                    print(json.dumps({'task_id': task_id, 'seed': seed, 'status': 'GROUP_ERROR',
                                      'error': f'{type(exc).__name__}: {exc}'}), flush=True)
                write_json_atomic(root / 'run_status.json', suite_status(root))
    status = suite_status(root)
    write_json_atomic(root / 'run_status.json', status)
    return status


def freeze_math_test(test_directory, development_directory):
    test, dev = Path(test_directory).resolve(), Path(development_directory).resolve()
    dev_status = suite_status(dev)
    required_zero = ('numerical_or_invariant_errors', 'hard_constraint_violations', 'repeat_queries',
                     'b4_budget_violations', 'b4_exhaustive_mismatches')
    if not dev_status['all_planned_reported'] or any(dev_status[key] for key in required_zero):
        raise ValueError('development suite is incomplete or failed implementation qualification')
    manifest = json.loads((test / 'manifest.json').read_text(encoding='utf-8'))
    development = json.loads((dev / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['definition']['split'] != 'test' or development['definition']['split'] != 'development':
        raise ValueError('development and test identities are incorrect')
    if canonical_hash(manifest['definition']['settings']) != canonical_hash(development['definition']['settings']):
        raise ValueError('test settings differ from the qualified development settings')
    if set(row['task_id'] for row in manifest['tasks']) & set(row['task_id'] for row in development['tasks']):
        raise ValueError('development and test task identities overlap')
    _verify_suite_inputs(test, manifest)
    receipt = {'schema_version': 1, 'manifest_sha256': sha256((test / 'manifest.json').read_bytes()).hexdigest(),
               'development_manifest_sha256': sha256((dev / 'manifest.json').read_bytes()).hexdigest(),
               'development_qualification': dev_status, 'engine_sha256': engine_fingerprint(),
               'settings_sha256': canonical_hash(manifest['definition']['settings']),
               'schedule_values': [CampaignSettings.from_protocol(json.loads((test / 'protocol_snapshot.json').read_text(
                   encoding='utf-8'))).budget(i) for i in range(manifest['definition']['settings']['additional_queries'])],
               'expected_campaigns': manifest['planned_campaigns'],
               'scientific_effects_assumed': False}
    target = test / 'freeze_receipt.json'
    if target.exists() and json.loads(target.read_text(encoding='utf-8')) != receipt:
        raise ValueError('a different test freeze already exists; refusing replacement')
    write_json_atomic(target, receipt)
    return receipt


def prepare_summit_suite(directory, protocol, cache_directory, *, split='development'):
    if split not in {'development', 'test'}:
        raise ValueError('unknown reaction split')
    root, cache = Path(directory).resolve(), Path(cache_directory).resolve()
    settings = CampaignSettings.from_protocol(protocol)
    cases = [1] if split == 'development' else list(protocol['summit']['test_cases'])
    seeds = [17, 18, 19, 20, 21] if split == 'development' else list(protocol['shared_design']['paired_seeds'])
    definition = {'benchmark': 'Summit_Reizman_Suzuki_0.8.9', 'split': split,
                  'settings': asdict(settings), 'paired_seeds': seeds, 'cases': cases,
                  'prior_regimes': list(MATH_REGIMES), 'methods': list(METHODS),
                  'pool_size': 512, 'stress_source_sha256': canonical_hash(inspect.getsource(make_reaction_stress_priors))}
    manifest_path = root / 'manifest.json'
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding='utf-8'))
        if canonical_hash(saved['definition']) != canonical_hash(definition):
            raise ValueError('existing reaction suite definition changed')
        _verify_suite_inputs(root, saved)
        return saved
    root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(root / 'protocol_snapshot.json', protocol)
    tasks = []
    for case in cases:
        source = cache / f'case_{case}'
        receipt = json.loads((source / 'cache_receipt.json').read_text(encoding='utf-8'))
        if (receipt['optimizer_steps'] or receipt['backward_calls']
                or not receipt['checkpoint_and_scaler_equality']
                or receipt['independent_forward_max_abs_error'] > receipt['parity_tolerance_percentage_units']
                or receipt['public_api_max_abs_error'] > receipt['parity_tolerance_percentage_units']):
            raise ValueError('pretrained simulator did not pass frozen inference qualification')
        if sha256((source / 'oracle.npz').read_bytes()).hexdigest() != receipt['oracle_sha256']:
            raise ValueError('qualified emulator cache changed')
        public = json.loads((source / 'task.json').read_text(encoding='utf-8'))
        if SynthesisTask.from_dict(public).fingerprint != receipt['task_sha256']:
            raise ValueError('qualified reaction task changed')
        public['evidence'].append({'id': 'synthetic_prior_constructor',
                                   'text': 'Artificial oracle-aware prior controls, not naturally generated LLM hypotheses.'})
        task = SynthesisTask.from_dict(public)
        with np.load(source / 'oracle.npz', allow_pickle=False) as saved:
            values = saved['values']
        priors = make_reaction_stress_priors(task, values)
        folder = root / 'inputs' / task.task_id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'oracle.npz').write_bytes((source / 'oracle.npz').read_bytes())
        initial = {str(seed): [task.candidate_ids[i] for i in np.random.default_rng(seed).choice(
            len(task.candidate_ids), settings.initial_queries, replace=False)] for seed in seeds}
        bundle = {'task': task.to_dict(), 'priors': priors, 'initial_ids': initial,
                  'oracle_sha256': receipt['oracle_sha256'], 'cache_qualification': receipt}
        write_json_atomic(folder / 'bundle.json', bundle)
        tasks.append({'task_id': task.task_id, 'family': f'reaction_case_{case}', 'instance': case,
                      'task_sha256': task.fingerprint, 'bundle': (folder / 'bundle.json').relative_to(root).as_posix(),
                      'bundle_sha256': sha256((folder / 'bundle.json').read_bytes()).hexdigest(),
                      'oracle': (folder / 'oracle.npz').relative_to(root).as_posix(),
                      'oracle_sha256': receipt['oracle_sha256']})
    manifest = {'schema_version': 1, 'definition': definition, 'tasks': tasks,
                'planned_groups': len(tasks) * len(seeds),
                'planned_campaigns': len(tasks) * len(seeds) * len(MATH_REGIMES) * len(METHODS),
                'scope': 'pretrained reaction emulator, fixed artificial priors, no simulator training'}
    write_json_atomic(manifest_path, manifest)
    return manifest
