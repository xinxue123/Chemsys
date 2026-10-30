"""Fresh, own-feedback 4x4x3 repair ladder, separate from frozen G3 results."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import time

import httpx
import numpy as np

from . import grounded_regional_suzuki_engine as e
from . import regional_ablation_suzuki_study as previous
from . import deepseek_suzuki_client as api
from . import deepseek_small_suzuki_study as transport
from .suzuki_domain import CATALYSTS, digest

ROOT = Path('data/runs/suzuki_grounded_regional')
OUT = Path('research/suzuki_grounded_regional')
METHODS = e.METHODS
REGIMES = previous.REGIMES
PARAMETERS = deepcopy(previous.PARAMETERS)
API_WORKERS = 2
read, write, filehash = previous.read, previous.write, previous.filehash


def hashes():
    paths = [Path(__file__), Path(e.__file__), Path(previous.__file__), Path(e.original.__file__),
        Path(e.original.common.__file__), Path(e.original.common.mixed.__file__),
        Path(e.original.previous.__file__), Path(api.__file__), Path(transport.__file__),
        Path('chem/decision/domain_guard.py'), Path('chem/benchmarks/suzuki_grounded_regional.py'),
        Path('tests/test_suzuki_grounded_regional.py')]
    return {p.as_posix(): filehash(p) for p in paths}


def cache_for(case, pool, qualification=False, category_map=e.original.common.mixed.CATEGORY_MAP):
    group = f'c{case}_s0'
    directory = ROOT / ('qualification_cache' if qualification else 'cache') / group
    if tuple(category_map) != e.original.common.mixed.CATEGORY_MAP:
        directory = directory / 'permuted'
    sources = [root / folder / group
        for root in (previous.ROOT, previous.prior.ROOT, previous.prior.previous.ROOT)
        for folder in ('cache', 'qualification_cache', 'audit_cache')]
    sources.append(Path('research/suzuki_regional_module_diagnostic/cache') / f'c{case}')
    if not qualification:
        sources.insert(0, ROOT / 'qualification_cache' / group)
    return e.GroundedRegionalCache(pool, directory, case, category_map=category_map, sources=sources)


def worker_init():
    previous.worker_init()


def protected():
    inventory = previous.OUT / 'release_inventory.json'
    files = {row['path']: row['sha256'] for row in read(inventory)['files']}
    files[inventory.as_posix()] = filehash(inventory)
    for f in ('docs/suzuki_regional_module_diagnostic.md',
              'research/suzuki_regional_module_diagnostic/results.json',
              'research/suzuki_regional_module_diagnostic/validation_receipt.json'):
        files[f] = filehash(f)
    return files


def preflight(config_path='../config.txt'):
    client = api.APIClient(api.load_config(config_path))
    try:
        response = client.http.get('/user/balance')
        response.raise_for_status()
        data = response.json()
        result = {'status_code': response.status_code, 'is_available': data.get('is_available'),
            'balances': [{k: row.get(k) for k in ('currency', 'total_balance')}
                         for row in data.get('balance_infos', [])],
            'workers': API_WORKERS, 'threads_each': 1, 'maximum_logical_refreshes': 144}
        write(OUT / 'resource_preflight.json', result)
        return result
    finally:
        client.close()


def qualification_case(case):
    pool = previous.inputs(case)
    history = previous.prior.initial(case, pool)
    cache = cache_for(case, pool, True)
    permuted = cache_for(case, pool, True, [19, -3, 101, 7, 0, 12, -9, 4])
    rows = []
    for regime in REGIMES:
        knowledge = previous.initial_knowledge(case, regime)
        state = {'method': METHODS[0], 'knowledge': knowledge, 'initial_knowledge': deepcopy(knowledge),
                 'history': history, 'banks': {}}
        old = e.original.setup(cache, {**state, 'method': previous.METHODS[3]}, pool, 1)
        new = e.setup(cache, state, pool, 1)
        assert old['selected'] == new['selected'] and np.array_equal(old['score'], new['score'])
        assert np.array_equal(old['raw_shift_std'], new['raw_shift_std']) and new['amplitude'] == 1.
        relabeled = e.setup(permuted, {**state, 'method': METHODS[2]}, pool, 1)
        control = e.setup(cache, {**state, 'method': METHODS[2]}, pool, 1)
        error = float(np.max(np.abs(relabeled['score'] - control['score'])))
        assert relabeled['selected'] == control['selected'] and error < 1e-9
        rows.append({'case': case, 'regime': regime, 'R1_initial_original_scores_exact': True,
                     'R3_label_permutation_max_error': error})
    return {'rows': rows, 'operator_compiles_in_loop': sum(previous.prior.previous.JIT.values())}


def qualify():
    with ProcessPoolExecutor(max_workers=4, initializer=worker_init) as executor:
        allrows = list(executor.map(qualification_case, (1, 2, 3, 4)))
    result = {'status': 'PASS', 'rows': [row for group in allrows for row in group['rows']],
              'operator_compiles_in_loop': sum(g['operator_compiles_in_loop'] for g in allrows)}
    assert result['operator_compiles_in_loop'] == 0
    write(OUT / 'qualification.json', result)
    return result


def prepare(config_path='../config.txt'):
    if (OUT / 'protocol.json').exists():
        return check_protocol()
    assert read(OUT / 'qualification.json')['status'] == 'PASS'
    old = previous.check_protocol()
    cfg = api.load_config(config_path)
    manifest = [{'case': c, 'seed': 0, 'regime': r, 'method': m,
                 'run_id': f'c{c}_s0_{r}_{m}'}
                for c in (1, 2, 3, 4) for r in REGIMES for m in METHODS]
    for i, spec in enumerate(sorted(manifest, key=lambda x: digest({'salt': 'grounded-regional-v1', 'id': x['run_id']}))):
        spec['task'] = f'T{i:02d}'
    p = {'study': 'Grounded_regional_G3_repairs_cumulative_v1',
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'scientific_sha256': hashes(), 'manifest_sha256': digest(manifest),
        'qualification_sha256': filehash(OUT / 'qualification.json'),
        'parent_protocol_sha256': old['protocol_sha256'], 'cases': [1, 2, 3, 4],
        'primary_cases': [2, 3, 4], 'seeds': [0], 'regimes': REGIMES, 'methods': METHODS,
        'planned_new_trajectories': 48, 'frozen_G3_comparators': 16,
        'initial_queries': 4, 'additional_queries': 16, 'new_added_queries': 768,
        'maximum_new_LLM_calls': 144, 'candidate_count': 4692,
        'parameters': e.PARAMETERS, 'request_parameters': PARAMETERS, 'API': cfg.public(),
        'initial_knowledge_sha256': old['initial_knowledge_sha256'],
        'shared': 'Unchanged MixedSingleTaskGP, nominal catalyst kernel, fixed candidate pool, initialization, seed, GP fit/MC seeds, qLogEI MC512, r4 decaying gate, original regional field and KF Joseph updates. No outcome noise added.',
        'ladder': {
            METHODS[0]: 'Original G3 plus named normalized/physical features and checked typed factual citations; original B4 subset and clipped KF-mean deployment.',
            METHODS[1]: 'R1 plus zero deployed regional field whenever current B4 diagnosis revokes any clause or conflict_kind is not none; select exact base qLogEI argmax. Original knowledge remains; original retained-region bank receives an unscaled shadow residual update.',
            METHODS[2]: 'R2 plus alpha=clip(mean_a-sqrt(Paa_forecast),0,1) for every actual geometry differing from the complete initial region. Initial geometry remains clip(mean_a,0,1); no cross-geometry evidence transfer. Applies throughout that geometry lifetime, not only its first round; conflict overrides alpha to zero.'},
        'evidence': 'Action support contains index, fixed named axis, normalized_value, physical_value, yield_percent; all match own observed displayed values within1e-8, no duplicate support indices. Catalyst normalized=null and physical=exact nominal name. Reject mismatched axes/numbers or future evidence; up to3 received plans, first valid. This does not verify semantic causal claims in free text.',
        'LLM_input': 'Only own observed prefix with named features, saved prequery predictions, current GP top3 per catalyst, current knowledge, KF and B4 diagnosis. No case/condition/method truth labels, unqueried yields, other-arm feedback or old responses. Each arm has fresh own-context requests.',
        'events': old['refresh'], 'retries': old['retries'],
        'region_banks': 'Same original geometry+confidence keys and elapsed Q process noise. Shadow learning uses unscaled field even during neutral fallback or conservative deployment. Empty region: no bank update.',
        'system_prompt_sha256': sha256(e.SYSTEM_PROMPT.encode('utf8')).hexdigest(),
        'input_sha256': old['input_sha256'], 'protected_files': protected(),
        'resources': {'workers': API_WORKERS, 'threads_each': 1, 'device': 'cpu',
            'warmup': 'Mixed GP kernels, regional grid and Cholesky before counted loops',
            'cache': 'Only exact same history/recipe scores and fitted models reused; no runtime operator compilation'},
        'evaluation': 'All48 full feedback trajectories, old G3 frozen comparator. Case-wise four-condition tables, primary cases2-4, AUC=mean incumbent after queries1..16, final best, first own-best position. Paired cumulative increments conditional on insertion order, not independent module causal effects. No post-result tuning or cherry-picked paths.',
        'limits': 'One seed and one new LLM realization per arm. API alias not immutable snapshot. Artificial three knowledge regimes retain original outcome-aware stress priors. Named factual grounding is not chemical knowledge correctness.'}
    p['protocol_sha256'] = digest(p)
    write(OUT / 'protocol.json', p)
    write(ROOT / 'protocol.json', p)
    write(ROOT / 'manifest.json', manifest)
    write(OUT / 'start_receipt.json', {'states_before_start': len(list((ROOT / 'runs').glob('*/state.json'))),
            'old_results_sha256': filehash(previous.OUT / 'results.json')})
    return p


def check_protocol():
    p = read(OUT / 'protocol.json')
    raw = deepcopy(p)
    key = raw.pop('protocol_sha256')
    assert digest(raw) == key and p['scientific_sha256'] == hashes()
    assert digest(read(ROOT / 'manifest.json')) == p['manifest_sha256']
    assert filehash(OUT / 'qualification.json') == p['qualification_sha256']
    for f, expected in {**p['input_sha256'], **p['protected_files']}.items():
        assert filehash(f) == expected, f
    return p


def state_for(spec, protocol, pool):
    path = ROOT / 'runs' / spec['run_id'] / 'state.json'
    if path.exists():
        state = read(path)
        assert state['protocol_sha256'] == protocol['protocol_sha256']
        return state, path
    knowledge = previous.initial_knowledge(spec['case'], spec['regime'])
    state = {**spec, 'protocol_sha256': protocol['protocol_sha256'], 'status': 'RUNNING',
        'history': previous.prior.initial(spec['case'], pool), 'knowledge': knowledge,
        'initial_knowledge': deepcopy(knowledge), 'slots': {h['id']: h['coordinate'] for h in knowledge['clauses']},
        'banks': {}, 'decisions': [], 'calls': [], 'active_seconds': 0., 'operator_compiles_in_loop': 0}
    write(path, state)
    return state, path


def public_request(state, pool, decision, reasons):
    result = previous.public_request(state, pool, decision, reasons)
    for row, history in zip(result['observations'], state['history']):
        row.pop('z')
        row.pop('cat')
        row.update(e.named_features(pool, pool['ids'].index(history['candidate_id'])))
    for row in result['GP_top3_per_catalyst']:
        row.pop('z')
        row.pop('cat')
        row.update(e.named_features(pool, pool['ids'].index(row['id'])))
    result['current_knowledge'] = {'confidence': state['knowledge']['confidence'],
        'clauses': [{**{k: v for k, v in h.items() if k != 'coordinate'}, 'axis': e.VARIABLES[h['coordinate']]}
                    for h in state['knowledge']['clauses']]}
    result['deployment_rule_result'] = decision['deployment']
    result['units'] = {'residence_time': 'seconds', 'temperature': 'Celsius', 'catalyst_loading': 'mol_percent'}
    return result


def get_refresh(client, context, state, pool, folder):
    if (folder / 'validated.json').exists():
        saved = read(folder / 'validated.json')
        if saved['context_sha256'] != context['context_sha256']:
            raise ValueError('cached LLM context mismatch')
        checked = e.validate_refresh({'actions': saved['plan']['actions'], 'summary': saved['plan']['summary']},
            state['knowledge'], state['slots'], context['public']['observations'], pool)
        assert checked == saved['plan']
        return saved
    base = [{'role': 'system', 'content': e.SYSTEM_PROMPT},
        {'role': 'user', 'content': json.dumps(context['public'], ensure_ascii=False, separators=(',', ':'))}]
    messages = base
    received = 0
    for attempt in range(1, 9):
        directory = folder / f'attempt{attempt:02d}'
        body = dict(PARAMETERS, messages=messages)
        if (directory / 'request.json').exists():
            assert read(directory / 'request.json') == body
        else:
            write(directory / 'request.json', body)
        if (directory / 'transport_error.json').exists():
            if not read(directory / 'transport_error.json')['retryable']:
                raise RuntimeError('nonretryable API failure')
            continue
        if (directory / 'response.json').exists():
            answer = read(directory / 'response.json')
        else:
            try:
                answer = transport.complete(client, messages)
                write(directory / 'response.json', answer)
            except Exception as error:
                status = error.response.status_code if isinstance(error, httpx.HTTPStatusError) else None
                retryable = isinstance(error, (httpx.TimeoutException, httpx.TransportError)) or status in (429, 500, 502, 503, 504)
                write(directory / 'transport_error.json', {'error': api.safe_error(error, client.config), 'retryable': retryable})
                if not retryable:
                    raise RuntimeError('nonretryable API failure') from None
                if attempt < 8:
                    time.sleep(min(5 * 2 ** (attempt - 1), 30))
                continue
        received += 1
        try:
            if answer['model'] != api.MODEL or len(answer['choices']) != 1 or answer['choices'][0]['finish_reason'] != 'stop':
                raise ValueError('model mismatch or incomplete response')
            plan = e.validate_refresh(json.loads(answer['choices'][0]['message']['content']), state['knowledge'],
                state['slots'], context['public']['observations'], pool)
        except (ValueError, TypeError, KeyError, IndexError) as error:
            message = type(error).__name__ + ': ' + str(error)[:240]
            write(directory / 'validation_error.json', {'error': message})
            if received >= 3:
                raise RuntimeError('received schema limit exhausted') from None
            content = answer['choices'][0]['message'].get('content') if answer.get('choices') else '{}'
            messages = base + [{'role': 'assistant', 'content': content or '{}'},
                {'role': 'user', 'content': 'Factual/schema validation failed: ' + message + '. Correct only JSON from displayed named facts; no experiment occurred. Return exactly actions,summary.'}]
            continue
        valid = {'plan': plan, 'accepted_attempt': attempt, 'received_responses': received,
            'response_id': answer['id'], 'response_model': answer['model'],
            'response_sha256': filehash(directory / 'response.json'), 'context_sha256': context['context_sha256']}
        write(folder / 'validated.json', valid)
        return valid
    raise RuntimeError('HTTP attempt limit exhausted')


def step(state, path, pool, cache, values, client):
    t = len(state['decisions']) + 1
    started = time.perf_counter()
    before_jit = sum(previous.prior.previous.JIT.values())
    decision = e.setup(cache, state, pool, t)
    pre_refresh = {'diagnosis': decision['diagnosis'].to_dict(), 'deployment': decision['deployment'],
                   'selected_id': pool['ids'][decision['selected']]}
    if not any(c['iteration'] == t for c in state['calls']):
        reasons = e.original.event_reasons(state, decision, t)
        if reasons:
            public = public_request(state, pool, decision, reasons)
            context = {'public': public, 'history': deepcopy(state['history']), 'protocol_sha256': state['protocol_sha256']}
            context['context_sha256'] = digest(context)
            folder = ROOT / 'requests' / context['context_sha256']
            write(folder / 'context.json', context)
            refresh = {'iteration': t, 'reasons': reasons, 'context_sha256': context['context_sha256'],
                       'knowledge_before': deepcopy(state['knowledge']), 'diagnosis_before': decision['diagnosis'].to_dict()}
            try:
                valid = get_refresh(client, context, state, pool, folder)
                state['knowledge'] = valid['plan']['knowledge']
                refresh.update(status='ACCEPTED', **valid)
            except RuntimeError as error:
                refresh.update(status='FAILED', error=str(error))
            state['calls'].append(refresh)
            write(path, state)
            decision = e.setup(cache, state, pool, t)
    j = decision['selected']
    _, value = cache.model(state['history'])
    scale = float(value['std'])
    raw_y = float(decision['raw_shift_std'][j] * scale)
    shift_y = float(decision['shift_std'][j] * scale)
    mu = float(decision['stats']['mu_y'][j])
    sd = float(decision['stats']['sd_predictive_y'][j])
    # Only now reveal the simulated outcome. values never enters policy/LLM.
    y = float(values[j])
    after = None
    if decision['bank'] is not None:
        after, measurement = e.original.update_bank(decision['bank'], decision['prior'], t, raw_y, mu, sd, y)
        state['banks'][decision['bank_key']] = after
    else:
        variance = sd * sd + .0009
        measurement = {'prequery_mu_y': mu, 'prequery_predictive_sd_y': sd, 'raw_region_shift_y': raw_y,
            'observed_y': y, 'residual': y - mu, 'innovation': y - mu,
            'innovation_variance': variance, 'measurement_variance': variance,
            'innovation_z': float((y - mu) / np.sqrt(variance))}
    predictions = {'GP': mu, 'regional_unscaled': mu + raw_y, 'deployed': mu + shift_y}
    if decision['prior'] is not None:
        predictions['KF_joint'] = mu + decision['prior']['mean'][0] + decision['prior']['mean'][1] * raw_y
    measurement.update(predictions=predictions, squared_errors={k: float((y - p) ** 2) for k, p in predictions.items()})
    event = {'iteration': t, 'selected_id': pool['ids'][j], 'reference_id': pool['ids'][decision['reference']],
        'base_acquisition': decision['base_meta'], 'field_acquisition': decision['field_meta'],
        'guided_acquisition': decision['guided_meta'], 'budget_logei': decision['budget'],
        'base_logei_loss': float(decision['base'][decision['reference']] - decision['base'][j]),
        'eligible_candidates': int(decision['eligible'].sum()), 'retained_ids': decision['retained'],
        'diagnosis': decision['diagnosis'].to_dict(), 'knowledge': deepcopy(state['knowledge']),
        'bank_key': decision['bank_key'], 'bank_before': decision['bank'], 'bank_prior': decision['prior'],
        'bank_after': after, 'amplitude': decision['amplitude'], 'deployment': decision['deployment'],
        'pre_refresh': pre_refresh, 'raw_shift_y': raw_y, 'deployed_shift_y': shift_y, 'measurement': measurement}
    assert event['base_logei_loss'] <= event['budget_logei']
    state['decisions'].append(event)
    state['history'].append({'candidate_id': pool['ids'][j], 'y': y})
    state['active_seconds'] += time.perf_counter() - started
    state['operator_compiles_in_loop'] += sum(previous.prior.previous.JIT.values()) - before_jit
    state['status'] = 'COMPLETE' if t == 16 else 'RUNNING'
    write(path, state)
    print(json.dumps({'task': state['task'], 'round': t, 'refreshes': len(state['calls']),
                      'fallback': decision['deployment']['neutral_fallback']}, separators=(',', ':')), flush=True)


def run_case(case, config_path='../config.txt'):
    protocol = check_protocol()
    pool = previous.inputs(case)
    cache = cache_for(case, pool)
    with np.load(previous.prior.original.ROOT / 'inputs' / f'case{case}' / 'oracle.npz') as f:
        values = f['values'].copy()
    states = [state_for(v, protocol, pool) for v in read(ROOT / 'manifest.json') if v['case'] == case]
    client = api.APIClient(api.load_config(config_path))
    try:
        for t in range(1, 17):
            for state, path in states:
                if state['status'] in ('COMPLETE', 'FAILED') or len(state['history']) != t + 3:
                    continue
                try:
                    step(state, path, pool, cache, values, client)
                except Exception as error:
                    state.update(status='FAILED', error={'kind': type(error).__name__, 'message': str(error)[:300]})
                    write(path, state)
    finally:
        client.close()
    result = {'case': case, 'planned': len(states), 'complete': sum(s['status'] == 'COMPLETE' for s, _ in states),
        'failed': [s['run_id'] for s, _ in states if s['status'] == 'FAILED'], 'cache': cache.counters,
        'operator_compiles_in_loop': sum(s['operator_compiles_in_loop'] for s, _ in states)}
    write(ROOT / 'runtime' / f'c{case}.json', result)
    return result


def run(config_path='../config.txt'):
    check_protocol()
    with ProcessPoolExecutor(max_workers=API_WORKERS, initializer=worker_init) as executor:
        futures = [executor.submit(run_case, c, config_path) for c in (1, 2, 3, 4)]
        for future in as_completed(futures):
            print(json.dumps(future.result(), separators=(',', ':')), flush=True)
    return status()


def status():
    counts, failures, steps, calls = {}, [], 0, 0
    for path in (ROOT / 'runs').glob('*/state.json'):
        row = read(path)
        counts[row['status']] = counts.get(row['status'], 0) + 1
        steps += len(row['decisions'])
        calls += len(row['calls'])
        if row['status'] == 'FAILED':
            failures.append({'run': row['run_id'], 'error': row.get('error')})
    return {'runs': counts, 'additional_queries': steps, 'planned': 48, 'LLM_refreshes': calls, 'failures': failures}
