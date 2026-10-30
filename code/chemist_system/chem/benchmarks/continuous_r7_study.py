"""Frozen R7-S SnAr/HPLC finite-pool study, independent of all R5/R6 runs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import time
import traceback

import numpy as np

from . import continuous_r6_engine as r6
from . import continuous_r6_study as archived
from . import continuous_r7_core as core
from . import continuous_r7_engine as r7


ROOT = Path('data/runs/continuous_r7_20261030')
OUT = Path('research/continuous_r7_20261030')
TASKS = ('SnAr', 'HPLC')


def _data():
    from . import continuous_benchmark_data as data
    return data


def _runtime():
    from . import continuous_benchmark_engine as base
    from . import deepseek_suzuki_client as api
    from . import continuous_r6_llm as llm
    return _data(), base, api, llm


def _paths():
    from . import continuous_r7_diagnostics as diagnostics
    return sorted(set(archived._paths() + [
        Path(__file__), Path(core.__file__), Path(r7.__file__),
        Path(diagnostics.__file__), Path('continuous_r7.py')]))


def qualify():
    return archived.qualify()


def prepare():
    data, _, api, llm = _runtime()
    if (OUT / 'protocol.json').exists():
        return check()
    inspected = qualify()
    if inspected['status'] != 'PASS':
        raise FileNotFoundError('R7-S requires original SnAr/HPLC pool.json and oracle.npz')
    protocol = {
        'name': 'R7_S_covariance_propagation_passive_evidence',
        'arms': list(r7.ARMS), 'tasks': list(TASKS),
        'initial_queries': 4, 'additional_queries': 16,
        'GP': 'frozen original SingleTaskGP plus MC512 qLogEI',
        'LLM': {'model': api.MODEL, 'prompt_sha256': r6.digest(llm.SYSTEM),
                'fixed_hypothesis_call_rounds': list(r7.CALL_ROUNDS),
                'adaptive_hypothesis_call_rounds': list(r7.ONLINE_CALL_ROUNDS),
                'shared_first_round_request': True,
                'future_response_reuse': False,
                'online_arm': 'B2_online'},
        'knowledge': {'representation': 'one precommitted anchored pair-sign event',
                      'propagation': 'full GP posterior covariance to existing gate',
                      'not_global_monotonicity': True,
                      'preference_noise_y': r6.PREFERENCE_NOISE_Y,
                      'logEI_shift_cap': r6.MAX_PREFERENCE_LOG_EI_SHIFT,
                      'max_weight': .5,
                      'min_rho_squared_numerical': core.MIN_RHO_SQUARED,
                      'active_verifications': 0, 'gate_exceptions': 0},
        'data_sha256': {row['task']: {'pool': row['input_sha256'], 'oracle': row['oracle_sha256']}
                        for row in inspected['tasks']},
        'sources_sha256': {path.as_posix(): data.filehash(path) for path in _paths()},
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'no_performance_claim': True,
    }
    protocol['protocol_sha256'] = r6.digest(protocol)
    data.write(OUT / 'protocol.json', protocol)
    return protocol


def check():
    data = _data()
    protocol = data.read(OUT / 'protocol.json')
    assert r6.digest({k: v for k, v in protocol.items() if k != 'protocol_sha256'}) == protocol['protocol_sha256']
    assert {path.as_posix(): data.filehash(path) for path in _paths()} == protocol['sources_sha256'], 'R7 source has changed after prepare'
    for task, hashes in protocol['data_sha256'].items():
        assert data.filehash(data.ROOT / task / 'pool.json') == hashes['pool']
        assert data.filehash(data.ROOT / task / 'oracle.npz') == hashes['oracle']
    return protocol


def _run_dir(task, arm, seed):
    return ROOT / 'runs' / f'{task}_{arm}_seed{seed:03d}'


def initial(task, arm, seed, protocol):
    if arm not in r7.ARMS or task not in TASKS:
        raise ValueError('unknown arm/task')
    data = _data()
    pool = data.read(data.ROOT / task / 'pool.json')
    four = archived.common_initial_ids(pool, seed)
    return {'task': task, 'arm': arm, 'seed': seed,
            'protocol_sha256': protocol['protocol_sha256'],
            'initial_ids': four, 'history': [data.reveal(task, key) for key in four],
            'hypotheses': [], 'calls': [], 'validation_queries_used': 0,
            'exceptions_used': 0, 'decisions': [], 'status': 'READY',
            'active_seconds': 0.}


def step(state, state_file, cache, client, protocol):
    """Commit first, reveal second; resume exactly from saved commitments."""
    data, _, _, llm = _runtime()
    t = len(state['decisions']) + 1
    directory = state_file.parent / 'rounds'
    directory.mkdir(parents=True, exist_ok=True)
    commit_file = directory / f'{t:02d}_committed.json'
    if commit_file.exists():
        commit = data.read(commit_file)
        assert commit['iteration'] == t
        assert commit['protocol_sha256'] == protocol['protocol_sha256']
        assert commit['history_sha256'] == r6.digest(state['history'])
        state['hypotheses'] = deepcopy(commit['hypotheses_at_commit'])
        state['calls'] = deepcopy(commit['calls_at_commit'])
    else:
        context = r7.build_context(cache, state, t)
        proposal = receipt = original = None
        if context['menu']:
            if client is None:
                raise RuntimeError('LLM is required for B2/B4; B0 needs no LLM')
            # Normalizing arm to R6 A4 at the SAME first-round prefix makes
            # B2 and B4 reuse one immutable request/response, not two draws.
            original, receipt = llm.request(client, cache, dict(state, arm='A4'),
                                            t, context, protocol['protocol_sha256'], ROOT)
            proposal = core.flip_directions(original) if state['arm'] == 'B4' else deepcopy(original)
        commit = r7.plan(cache, state, t, context, proposal)
        commit['protocol_sha256'] = protocol['protocol_sha256']
        commit['LLM_receipt'] = receipt
        if original is not None:
            commit['LLM_original_proposal'] = original
        data.write(commit_file, commit)
    observed = data.reveal(state['task'], commit['selected_id'])
    receipts = r7.apply_feedback(state, commit, observed)
    state['last_commit_sha256'] = data.filehash(commit_file)
    data.write(state_file, state)
    cache.release()
    return {'task': state['task'], 'arm': state['arm'], 'seed': state['seed'],
            'round': t, 'mode': commit['selection_mode'], 'selected': commit['selected_id'],
            'knowledge_guided': commit['selection_mode'] == 'knowledge_guided',
            'passive_evidence_updates': sum(v['informative'] for v in receipts)}


def run_one(task, arm, seed=0, config='../config.txt'):
    data, base, api, _ = _runtime()
    protocol = check()
    path = _run_dir(task, arm, seed)
    state_file = path / 'state.json'
    state = data.read(state_file) if state_file.exists() else initial(task, arm, seed, protocol)
    if state['protocol_sha256'] != protocol['protocol_sha256']:
        raise ValueError('R7 protocol mismatch; refusing continuation')
    if state['status'] == 'COMPLETE':
        return {'task': task, 'arm': arm, 'seed': seed, 'status': 'COMPLETE', 'queries': 16}
    if not state_file.exists():
        data.write(state_file, state)
    base.warm()
    pool = data.read(data.ROOT / task / 'pool.json')

    class Cache(base.Cache):
        def __init__(self, full_pool):
            super().__init__(full_pool)
            self.directory = ROOT / 'cache' / full_pool['task']
            self.directory.mkdir(parents=True, exist_ok=True)

    cache = Cache(pool)
    client = api.APIClient(api.load_config(config)) if arm != 'B0' else None
    start = time.perf_counter()
    try:
        while len(state['decisions']) < 16:
            print(json.dumps(step(state, state_file, cache, client, protocol), ensure_ascii=False), flush=True)
    except Exception as exc:
        state['status'] = 'INTERRUPTED'
        state['error'] = api.safe_error(exc, client.config) if client else {
            'kind': type(exc).__name__, 'message': str(exc)[:500]}
        data.write(path / 'error.json', {'error': state['error'],
                                         'traceback': traceback.format_exc()[-2500:]})
    finally:
        if client is not None:
            client.close()
        state['active_seconds'] += time.perf_counter() - start
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
               'LLM_calls': len(s['calls']), 'knowledge_guided': sum(
                   x['execution_mode'] == 'knowledge_guided' for x in s['decisions']),
               'passive_evidence_updates': sum(sum(e['informative'] for e in d['evidence'])
                                               for d in s['decisions'])}
        if s['status'] == 'COMPLETE':
            values = np.asarray([x['raw_value'] for x in s['history']], dtype=float)
            curve = np.maximum.accumulate(values)
            row.update(SearchAUC=float(np.mean(curve[4:])), final_best=float(curve[-1]),
                       first_best_total_round=int(np.argmax(curve == curve[-1]) + 1))
        rows.append(row)
    return {'runs': rows, 'completed': sum(r['status'] == 'COMPLETE' for r in rows)}


def audit():
    """No GP fits, LLM calls, or oracle queries: file-only audit."""
    data = _data()
    rows = []
    for state_file in sorted((ROOT / 'runs').glob('*/state.json')):
        state = data.read(state_file)
        history = state['history']
        assert len(history) == 4 + len(state['decisions'])
        assert len({h['candidate_id'] for h in history}) == len(history)
        maximum_calls = len(r7.ONLINE_CALL_ROUNDS) if state['arm'] == 'B2_online' else len(r7.CALL_ROUNDS)
        assert len(state['calls']) <= maximum_calls
        allowed_call_rounds = r7.ONLINE_CALL_ROUNDS if state['arm'] == 'B2_online' else r7.CALL_ROUNDS
        assert all(c['round'] in allowed_call_rounds for c in state['calls'])
        for t, decision in enumerate(state['decisions'], 1):
            commit = data.read(state_file.parent / 'rounds' / f'{t:02d}_committed.json')
            assert commit['history_sha256'] == r6.digest(history[:t + 3])
            assert commit['selected_id'] == decision['selected_id'] == history[t + 3]['candidate_id']
            assert commit['normal_gate_eligible'] and not commit['used_exception']
            assert commit['selection_mode'] in ('base_bo', 'knowledge_guided')
            assert commit['protocol_sha256'] == state['protocol_sha256']
            assert 'observed_y' not in commit and 'raw_value' not in commit
            assert all(s['prediction']['point_id'] == commit['selected_id'] and
                       s['prediction']['query_round'] == t for s in commit['prequery_shadows'])
            if state['arm'] == 'B0':
                assert commit['selection_mode'] == 'base_bo'
                assert not commit['prequery_shadows']
            if state['arm'] == 'B4' and 'LLM_original_proposal' in commit:
                assert commit['proposal'] == core.flip_directions(commit['LLM_original_proposal'])
        rows.append({'task': state['task'], 'arm': state['arm'], 'seed': state['seed'],
                     'audited_queries': len(state['decisions']), 'status': 'PASS'})
    return {'status': 'PASS', 'audited_runs': len(rows), 'traces': rows}


def report():
    rows = status()['runs']
    by_key = {(r['task'], r['seed'], r['arm']): r for r in rows if r['status'] == 'COMPLETE'}
    paired = []
    for (task, seed, arm), row in sorted(by_key.items()):
        if arm == 'B0' or (task, seed, 'B0') not in by_key:
            continue
        bo = by_key[task, seed, 'B0']
        paired.append({'task': task, 'seed': seed, 'arm': arm,
                       'SearchAUC': row['SearchAUC'], 'B0_SearchAUC': bo['SearchAUC'],
                       'paired_AUC_gain': row['SearchAUC'] - bo['SearchAUC'],
                       'relative_gain_percent': 100. * (row['SearchAUC'] / bo['SearchAUC'] - 1.)
                       if bo['SearchAUC'] else None,
                       'knowledge_guided': row['knowledge_guided']})
    return {'per_seed_paired': paired,
            'caution': 'Report SnAr and HPLC separately; seed0 is diagnostic, not confirmatory.'}


def diagnose():
    from . import continuous_r7_diagnostics as diag
    return diag.run()
