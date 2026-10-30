"""Independent R6 SnAr/HPLC runs: same four observations, then 16 own queries.

Unlike archived R5, R6 NEVER copies R2/R3/R4/R5 decision prefixes. Results
are isolated in new directories and existing protocol hashes remain unchanged.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import time
import traceback

import numpy as np

from . import continuous_r6_engine as r6

ROOT = Path('data/runs/continuous_r6_20261030')
OUT = Path('research/continuous_r6_20261030')
TASKS = ('SnAr', 'HPLC')


def _data():
    from . import continuous_benchmark_data as data
    return data


def _runtime():
    data = _data()
    from . import continuous_benchmark_engine as base
    from . import deepseek_suzuki_client as api
    from . import continuous_r6_llm as llm
    return data, base, api, llm


def _paths():
    from . import continuous_r6_core as core
    from . import continuous_r6_llm as llm
    from . import continuous_benchmark_engine as base
    from . import continuous_benchmark_data as data
    runtime = [
        'chem/benchmarks/official_suzuki_adapter.py',
        'chem/benchmarks/deepseek_suzuki_client.py',
        'chem/benchmarks/reaction_transfer_llm.py',
        'chem/benchmarks/suzuki_domain.py',
        'chem/benchmarks/r2_region_trust_engine.py',
        'chem/benchmarks/regional_ablation_suzuki_engine.py',
        'chem/benchmarks/stable_information_engine.py',
        'chem/decision/domain_guard.py',
        'third_party/suzuki_official/lgbo/prior.py',
        'third_party/suzuki_official/lgbo/decide.py',
    ]
    return ([Path(__file__), Path(core.__file__), Path(r6.__file__), Path(llm.__file__),
             Path(base.__file__), Path(data.__file__), Path('continuous_r6.py')]
            + [Path(p) for p in runtime])


def qualify():
    """Non-mutating inspection of source pools and the common four-start IDs."""
    data = _data()
    checks = []
    for task in TASKS:
        path = data.ROOT / task / 'pool.json'
        oracle = data.ROOT / task / 'oracle.npz'
        if not path.exists() or not oracle.exists():
            checks.append({'task': task, 'status': 'MISSING_DATA',
                           'pool': str(path), 'oracle': str(oracle)})
            continue
        pool = data.read(path)
        assert (pool['all_continuous'] and len(set(pool['ids'])) == len(pool['ids'])
                and len(pool['initial_ids']) == 4)
        with np.load(oracle) as f:
            assert f['values'].shape == (len(pool['ids']),) and np.isfinite(f['values']).all()
        checks.append({'task': task, 'status': 'PASS', 'pool_size': len(pool['ids']),
                       'axes': pool['axes'], 'common_four_ids': pool['initial_ids'],
                       'input_sha256': data.filehash(path), 'oracle_sha256': data.filehash(oracle)})
    return {'status': 'PASS' if all(c['status'] == 'PASS' for c in checks) else 'MISSING_DATA',
            'tasks': checks}


def prepare():
    data, _, api, llm = _runtime()
    if (OUT / 'protocol.json').exists():
        return check()
    inspection = qualify()
    if inspection['status'] != 'PASS':
        raise FileNotFoundError('R6 requires original SnAr/HPLC pool.json and oracle.npz; run original data preparation first')
    sources = {p.as_posix(): data.filehash(p) for p in _paths()}
    protocol = {
        'name': 'Continuous_R6_qualitative_preference_prequential_evidence',
        'tasks': list(TASKS), 'arms': list(r6.ARMS), 'GP': 'unchanged continuous_benchmark_engine',
        'initial_queries': 4, 'additional_queries': 16, 'per_task_budget': 20,
        'initialization': 'same deterministic input-only 4 points per seed and arm; seed0 matches pool.initial_ids',
        'LLM': {'model': api.MODEL, 'max_calls_per_trajectory': len(r6.CALL_ROUNDS),
                'call_rounds': list(r6.CALL_ROUNDS), 'qualitative_prompt_sha256': r6.digest(llm.SYSTEM)},
        'preference_noise_y': r6.PREFERENCE_NOISE_Y, 'scope_radius': r6.CONTRAST_SCOPE_RADIUS,
        'max_menu': r6.MAX_MENU, 'max_logEI_preference_offset': r6.MAX_PREFERENCE_LOG_EI_SHIFT,
        'verification': {'max_queries': 2, 'last_round': 10, 'max_exceptions': 1,
                         'extra_logEI_gap': float(np.log(2)), 'max_impact':'no regret guarantee',
                         'within_gate_first': True},
        'A4_selection': 'active verification first, then positive-evidence preference posterior; NEVER direct unverified LLM rerank',
        'A2_A3_selection': 'immediate qualitative LLM rerank allowed within original qLogEI gate',
        'data_sha256': {c['task']: {'pool': c['input_sha256'], 'oracle': c['oracle_sha256']}
                        for c in inspection['tasks']},
        'sources_sha256': sources, 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'scope': 'mechanism study, no superiority claim; each task and seed is analyzed separately',
    }
    protocol['protocol_sha256'] = r6.digest(protocol)
    data.write(OUT / 'protocol.json', protocol)
    return protocol


def check():
    data, _, _, _ = _runtime()
    p = data.read(OUT / 'protocol.json')
    check_hash = deepcopy(p)
    assert r6.digest({k: v for k, v in check_hash.items() if k != 'protocol_sha256'}) == p['protocol_sha256']
    assert p['sources_sha256'] == {path.as_posix(): data.filehash(path) for path in _paths()}, 'R6 source changed after preparation'
    for task, hashes in p['data_sha256'].items():
        assert data.filehash(data.ROOT / task / 'pool.json') == hashes['pool']
        assert data.filehash(data.ROOT / task / 'oracle.npz') == hashes['oracle']
    return p


def common_initial_ids(pool, seed):
    if seed < 0:
        raise ValueError('seed must be nonnegative')
    indices = np.random.default_rng(seed).choice(len(pool['ids']), 4, replace=False)
    ids = [pool['ids'][int(j)] for j in indices]
    if seed == 0 and ids != pool['initial_ids']:
        raise AssertionError('seed0 initialization differs from original input-only four')
    return ids


def initial(task, arm, seed, protocol):
    data, _, _, _ = _runtime()
    if arm not in r6.ARMS or task not in TASKS:
        raise ValueError('unknown arm/task')
    pool = data.read(data.ROOT / task / 'pool.json')
    four = common_initial_ids(pool, seed)
    return {
        'task': task, 'arm': arm, 'seed': seed,
        'protocol_sha256': protocol['protocol_sha256'],
        'initial_ids': four, 'history': [data.reveal(task, key) for key in four],
        'hypotheses': [], 'calls': [], 'validation_queries_used': 0,
        'exceptions_used': 0, 'decisions': [], 'status': 'READY',
        'active_seconds': 0.,
    }


def _run_dir(task, arm, seed):
    return ROOT / 'runs' / f'{task}_{arm}_seed{seed:03d}'


def step(state, state_file, cache, client, protocol):
    data, _, _, llm = _runtime()
    t = len(state['decisions']) + 1
    folder = state_file.parent / 'rounds'
    folder.mkdir(parents=True, exist_ok=True)
    commit_path = folder / f'{t:02d}_committed.json'
    if commit_path.exists():
        commit = data.read(commit_path)
        assert commit['iteration'] == t and commit['protocol_sha256'] == protocol['protocol_sha256']
        assert commit['history_sha256'] == r6.digest(state['history'])
        # Recover crash after commitment but before outcome write, WITHOUT
        # re-calling the LLM or re-selecting an experiment.
        state['hypotheses'] = deepcopy(commit['hypotheses_at_commit'])
        state['calls'] = deepcopy(commit['calls_at_commit'])
        state['validation_queries_used'] = commit['budgets_at_commit']['active_verifications']
        state['exceptions_used'] = commit['budgets_at_commit']['exceptions']
    else:
        context = r6.build_context(cache, state, t)
        proposal = llm_receipt = None
        if context['menu']:
            if client is None:
                raise RuntimeError('LLM required for this arm; use A0 for no-LLM baseline')
            proposal, llm_receipt = llm.request(
                client, cache, state, t, context, protocol['protocol_sha256'], ROOT)
        commit = r6.plan(cache, state, t, context, proposal)
        commit['protocol_sha256'] = protocol['protocol_sha256']
        commit['LLM_receipt'] = llm_receipt
        data.write(commit_path, commit)  # Pre-observation durable commitment.
    observed = data.reveal(state['task'], commit['selected_id'])
    receipts = r6.apply_feedback(state, commit, observed)
    state['last_commit_sha256'] = data.filehash(commit_path)
    data.write(state_file, state)
    cache.release()
    return {'task': state['task'], 'arm': state['arm'], 'seed': state['seed'],
            'round': t, 'mode': commit['selection_mode'], 'selected': commit['selected_id'],
            'LLM_calls': len(state['calls']), 'active_verifications': state['validation_queries_used'],
            'exception_queries': state['exceptions_used'],
            'evidence_updates': sum(1 for e in receipts if e['informative'])}


def run_one(task, arm, seed=0, config='../config.txt'):
    data, base, api, _ = _runtime()
    protocol = check()
    directory = _run_dir(task, arm, seed)
    state_file = directory / 'state.json'
    if state_file.exists():
        state = data.read(state_file)
    else:
        state = initial(task, arm, seed, protocol)
        data.write(state_file, state)
    if state['status'] == 'COMPLETE':
        return {'task': task, 'arm': arm, 'seed': seed, 'status': 'COMPLETE', 'queries': 16}
    if state['protocol_sha256'] != protocol['protocol_sha256']:
        raise ValueError('R6 protocol mismatch; will not resume on changed source')
    base.warm()
    pool = data.read(data.ROOT / task / 'pool.json')
    class R6Cache(base.Cache):
        def __init__(self, full_pool):
            super().__init__(full_pool)
            # Do not create/write new fit files in the historical run folder.
            self.directory = ROOT / 'cache' / full_pool['task']
            self.directory.mkdir(parents=True, exist_ok=True)

    cache = R6Cache(pool)
    # Same GP recipe, independent R6 cache; all arms are independently fit.
    client = api.APIClient(api.load_config(config)) if arm != 'A0' else None
    start = time.perf_counter()
    try:
        while len(state['decisions']) < 16:
            record = step(state, state_file, cache, client, protocol)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    except Exception as exc:
        state['status'] = 'INTERRUPTED'
        state['error'] = api.safe_error(exc, client.config) if client else {
            'kind': type(exc).__name__, 'message': str(exc)[:500]}
        data.write(directory / 'error.json', {'error': state['error'],
                                             'traceback': traceback.format_exc()[-2500:]})
    finally:
        if client is not None:
            client.close()
        state['active_seconds'] = state.get('active_seconds', 0.) + time.perf_counter() - start
        state['cache_counters'] = cache.counters
        data.write(state_file, state)
    return {'task': task, 'arm': arm, 'seed': seed, 'status': state['status'],
            'queries': len(state['decisions']), 'error': state.get('error')}


def status():
    data = _data()
    rows = []
    for path in sorted((ROOT / 'runs').glob('*/state.json')):
        s = data.read(path)
        row = {'task': s['task'], 'arm': s['arm'], 'seed': s['seed'],
               'status': s['status'], 'added_queries': len(s['decisions']),
               'LLM_calls': len(s['calls']), 'active_verifications': s['validation_queries_used'],
               'exceptions': s['exceptions_used']}
        if s['status'] == 'COMPLETE':
            objectives = np.asarray([h['raw_value'] for h in s['history']], dtype=float)
            curve = np.maximum.accumulate(objectives)
            row.update(SearchAUC=float(np.mean(curve[4:])), final_best=float(curve[-1]),
                       changed_vs_BO=sum(e['selected_id'] != e['BO_reference_id'] for e in s['decisions']),
                       positive_evidence=sum(sum(x['delta_log_score'] > 0 for x in e['evidence']
                                                 if x['delta_log_score'] is not None) for e in s['decisions']))
        rows.append(row)
    return {'runs': rows, 'completed': sum(r['status'] == 'COMPLETE' for r in rows),
            'total_added_queries': sum(r['added_queries'] for r in rows)}


def audit():
    """Inspect committed traces without executing experiments or refitting GP."""
    data = _data()
    rows = []
    for state_file in sorted((ROOT / 'runs').glob('*/state.json')):
        state = data.read(state_file)
        history = state['history']
        assert len(history) == 4 + len(state['decisions'])
        assert len({h['candidate_id'] for h in history}) == len(history), state_file
        exceptions = validations = 0
        for t, decision in enumerate(state['decisions'], 1):
            cp = state_file.parent / 'rounds' / f'{t:02d}_committed.json'
            commit = data.read(cp)
            assert commit['iteration'] == t and commit['selected_id'] == decision['selected_id']
            assert commit['selected_id'] == history[t + 3]['candidate_id']
            assert commit['history_sha256'] == r6.digest(history[:t + 3])
            assert commit['arm'] == state['arm']
            assert commit['selection_mode'] == decision['execution_mode']
            assert 'observed_y' not in commit and 'raw_value' not in commit
            if commit['used_exception']:
                assert not commit['normal_gate_eligible']
                assert commit['selection_mode'] == 'active_verification'
                assert commit['opportunity_gap'] <= commit['normal_budget'] + np.log(2) + 1e-10
                exceptions += 1
            else:
                assert commit['normal_gate_eligible']
            if commit['selection_mode'] == 'active_verification':
                assert t <= 10
                validations += 1
            if state['arm'] in ('A4', 'A4_no_exception'):
                assert commit['selection_mode'] not in ('direct_LLM_rerank', 'LLM_agreed_with_BO')
            assert all(x['prediction']['point_id'] == commit['selected_id']
                       and x['prediction']['query_round'] == t for x in commit['prequery_shadows'])
        assert validations <= 2 and exceptions <= (0 if state['arm'] == 'A4_no_exception' else 1)
        assert validations == state['validation_queries_used']
        assert exceptions == state['exceptions_used']
        rows.append({'task': state['task'], 'arm': state['arm'], 'seed': state['seed'],
                     'audited_queries': len(state['decisions']), 'LLM_calls': len(state['calls']),
                     'active_verifications': validations, 'exceptions': exceptions, 'status': 'PASS'})
    return {'status': 'PASS', 'traces': rows, 'audited_runs': len(rows)}


def report():
    """Separate task-unit and seed-matched AUC; no cross-task unit mixing."""
    rows = status()['runs']
    by_key = {(r['task'], r['seed'], r['arm']): r for r in rows if r['status'] == 'COMPLETE'}
    records = []
    for (task, seed, arm), row in sorted(by_key.items()):
        if arm == 'A0' or (task, seed, 'A0') not in by_key:
            continue
        bo = by_key[task, seed, 'A0']
        records.append({'task': task, 'seed': seed, 'arm': arm,
                        'AUC': row['SearchAUC'], 'A0_AUC': bo['SearchAUC'],
                        'paired_AUC_gain': row['SearchAUC'] - bo['SearchAUC'],
                        'relative_AUC_gain_percent': 100. * (row['SearchAUC'] / bo['SearchAUC'] - 1.)
                        if bo['SearchAUC'] else None,
                        'final_best': row['final_best'], 'changed_vs_BO': row['changed_vs_BO'],
                        'LLM_calls': row['LLM_calls'], 'active_verifications': row['active_verifications'],
                        'exceptions': row['exceptions']})
    aggregate = []
    for task in TASKS:
        for arm in r6.ARMS[1:]:
            gains = np.array([r['paired_AUC_gain'] for r in records
                              if r['task'] == task and r['arm'] == arm], dtype=float)
            if not len(gains):
                continue
            row = {'task': task, 'arm': arm, 'paired_seeds': len(gains),
                   'mean_paired_AUC_gain': float(gains.mean())}
            if len(gains) >= 6:
                rng = np.random.default_rng(0)
                resampled = np.mean(rng.choice(gains, size=(2000, len(gains)), replace=True), axis=1)
                row['bootstrap_95_percent_interval'] = [float(x) for x in np.quantile(resampled, [.025, .975])]
            aggregate.append(row)
    return {'per_seed_paired': records, 'per_task_aggregate': aggregate,
            'caution': 'No independent statistical confirmation from seed0 or from two unlike task units.'}
