"""Budgeted experiments using responses actually authored by this chat's assistant.

This module never generates a chemical hypothesis. A missing answer checkpoints
the campaign; submit_answers only validates and stores assistant-authored data.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from hashlib import sha256
import inspect
import json
from pathlib import Path

import numpy as np

from .evaluation import (CampaignSettings, _load_settings, _verify_suite_inputs,
                         engine_fingerprint, run_campaign, score_campaign)
from .hypothesis import FeatureEncoder, canonical_hash
from .replay import (InteractiveAssistantProvider, ReplayOracle, SynthesisTask,
                     write_json_atomic)

LLM_METHODS = ('B0', 'B1', 'B2', 'B3', 'B4')
TERMINAL = {'COMPLETE', 'STOPPED', 'MODEL_ERROR', 'INVARIANT_FAILURE'}


def assistant_implementation_hash():
    return canonical_hash({name: inspect.getsource(globals()[name]) for name in (
        'prepare_assistant_suite', 'request_view', '_run_group',
        'run_assistant_suite', 'submit_answers', 'score_campaign')})


def prepare_assistant_suite(directory, protocol, cache_directory, metadata_path,
                            prompt_path, *, split='development'):
    if split not in {'development', 'test'}:
        raise ValueError('unknown assistant experiment split')
    root, cache = Path(directory).resolve(), Path(cache_directory).resolve()
    metadata = json.loads(Path(metadata_path).read_text(encoding='utf-8'))
    prompt = Path(prompt_path).read_text(encoding='utf-8')
    settings = CampaignSettings.from_protocol(protocol)
    cases = [1] if split == 'development' else list(protocol['summit']['test_cases'])
    seeds = [17] if split == 'development' else list(protocol['shared_design']['paired_seeds'])
    definition = {'split': split, 'track': 'actual_interactive_assistant',
                  'cases': cases, 'paired_seeds': seeds, 'methods': list(LLM_METHODS),
                  'settings': asdict(settings), 'model_metadata': metadata,
                  'prompt_sha256': sha256(prompt.encode('utf-8')).hexdigest(),
                  'implementation_sha256': assistant_implementation_hash(),
                  'presentation': 'request_view: observed history plus 16 public unqueried candidates; full request archived',
                  'context_isolation': False, 'backend_weight_snapshot': None,
                  'input_output_token_usage': None}
    manifest_path = root / 'manifest.json'
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding='utf-8'))
        if canonical_hash(saved['definition']) != canonical_hash(definition):
            raise ValueError('frozen assistant definition differs')
        _verify_suite_inputs(root, saved)
        return saved
    root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(root / 'protocol_snapshot.json', protocol)
    write_json_atomic(root / 'model_metadata.json', metadata)
    (root / 'prompt.txt').write_bytes(prompt.encode('utf-8'))
    tasks = []
    for case in cases:
        source = cache / f'case_{case}'
        receipt = json.loads((source / 'cache_receipt.json').read_text(encoding='utf-8'))
        if (receipt['optimizer_steps'] or receipt['backward_calls']
                or not receipt['checkpoint_and_scaler_equality']
                or receipt['independent_forward_max_abs_error'] > receipt['parity_tolerance_percentage_units']
                or receipt['public_api_max_abs_error'] > receipt['parity_tolerance_percentage_units']):
            raise ValueError('unqualified simulator')
        oracle_bytes = (source / 'oracle.npz').read_bytes()
        if sha256(oracle_bytes).hexdigest() != receipt['oracle_sha256']:
            raise ValueError('simulator cache changed')
        task = SynthesisTask.from_dict(json.loads((source / 'task.json').read_text(encoding='utf-8')))
        if task.fingerprint != receipt['task_sha256']:
            raise ValueError('simulator public task changed')
        if any('synthetic' in e['id'] for e in task.evidence):
            raise ValueError('oracle-aware artificial priors must not enter the assistant track')
        folder = root / 'inputs' / task.task_id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'oracle.npz').write_bytes(oracle_bytes)
        initial = {str(seed): [task.candidate_ids[i] for i in np.random.default_rng(seed).choice(
            len(task.candidate_ids), settings.initial_queries, replace=False)] for seed in seeds}
        bundle = {'task': task.to_dict(), 'initial_ids': initial, 'cache_qualification': receipt}
        write_json_atomic(folder / 'bundle.json', bundle)
        tasks.append({'task_id': task.task_id, 'family': f'reaction_case_{case}', 'instance': case,
                      'task_sha256': task.fingerprint,
                      'bundle': (folder / 'bundle.json').relative_to(root).as_posix(),
                      'bundle_sha256': sha256((folder / 'bundle.json').read_bytes()).hexdigest(),
                      'oracle': (folder / 'oracle.npz').relative_to(root).as_posix(),
                      'oracle_sha256': receipt['oracle_sha256']})
    manifest = {'schema_version': 1, 'definition': definition, 'tasks': tasks,
                'planned_campaigns': len(tasks) * len(seeds) * len(LLM_METHODS),
                'maximum_logical_attempts': len(tasks) * len(seeds) * 4 * settings.max_llm_attempts,
                'scope': 'interactive assistant study; shared chat context and unverifiable backend snapshot limit inference'}
    write_json_atomic(manifest_path, manifest)
    return manifest


def request_view(request):
    """Deterministic compact presentation, without hidden outcomes or GP ranks."""
    ctx = request['context']
    task = ctx['task']
    observed = {row['candidate_id'] for row in ctx['observations']}
    round_index = ctx['completed_additional_queries'] // 4
    available = []
    # Two points per category. Order depends only on public pool and round.
    for start in range(0, len(task['candidate_ids']), 64):
        count = 0
        for offset in range(64):
            index = start + (round_index * 11 + offset * 29) % 64
            if index >= len(task['candidate_ids']):
                continue
            key = task['candidate_ids'][index]
            if key not in observed:
                available.append({'id': key, **task['candidates'][index]})
                count += 1
                if count == 2:
                    break
    return {'context_sha256': request['context_sha256'], 'task_id': task['task_id'],
            'expected_version': ctx['expected_version'], 'parameters': task['parameters'],
            'observations': ctx['observations'], 'previous_batch': ctx['previous_batch'],
            'candidate_options': available, 'evidence_ids': [row['id'] for row in task['evidence']],
            'presentation_excludes': ['unqueried_yields', 'pool_optimum', 'GP_acquisition_scores']}


def _run_group(arguments):
    from threadpoolctl import threadpool_limits
    import sklearn.gaussian_process
    import scipy.linalg
    root_name, task_def, seed, definition = arguments
    root = Path(root_name)
    _verify_suite_inputs(root, {'tasks': [task_def]})
    bundle = json.loads((root / task_def['bundle']).read_text(encoding='utf-8'))
    task = SynthesisTask.from_dict(bundle['task'])
    with np.load(root / task_def['oracle'], allow_pickle=False) as saved:
        values = saved['values']
    settings = _load_settings(definition['settings'])
    provider = InteractiveAssistantProvider(root / 'requests', definition['model_metadata'],
                                           prompt_text=(root / 'prompt.txt').read_text(encoding='utf-8'))
    memory, results, statuses = {}, [], []
    encoded = FeatureEncoder(task.schema).encode(task.candidates)
    with threadpool_limits(limits=1):
        for method in LLM_METHODS:
            folder = root / 'trajectories' / task.task_id / f's{seed}' / method
            state = run_campaign(task, ReplayOracle(task.candidate_ids, values), method, seed,
                                 settings, bundle['initial_ids'][str(seed)], output_dir=folder,
                                 cache_dir=root / 'predictions', provider=None if method == 'B0' else provider,
                                 memory_cache=memory, encoded=encoded)
            statuses.append({'method': method, 'status': state['status']})
            if state['status'] in TERMINAL:
                result = score_campaign(state, task.candidate_ids, values)
                result.update(regime='actual_assistant', family=task_def['family'],
                              trajectory=folder.relative_to(root).as_posix())
                write_json_atomic(folder / 'result.json', result)
                results.append(result)
    write_json_atomic(root / 'groups' / f'{task.task_id}_s{seed}.json',
                      {'task_id': task.task_id, 'seed': seed, 'results': results, 'statuses': statuses})
    return {'task_id': task.task_id, 'seed': seed, 'statuses': statuses}


def assistant_status(directory):
    root = Path(directory).resolve()
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    rows, pending = [], set()
    statuses, identities, attempts = {}, set(), 0
    for task in manifest['tasks']:
        for seed in manifest['definition']['paired_seeds']:
            for method in LLM_METHODS:
                path = root / 'trajectories' / task['task_id'] / f's{seed}' / method / 'state.json'
                if not path.exists():
                    statuses['NOT_STARTED'] = statuses.get('NOT_STARTED', 0) + 1
                    continue
                state = json.loads(path.read_text(encoding='utf-8'))
                identity = (state['header']['task_id'], state['header']['seed'], state['header']['method'])
                if identity != (task['task_id'], seed, method) or identity in identities:
                    raise ValueError('assistant campaign identity mismatch')
                identities.add(identity)
                status = state['status']
                statuses[status] = statuses.get(status, 0) + 1
                attempts += len(state['hypothesis_state']['events'])
                if status == 'WAITING_LLM':
                    pending.add(state['pending_request'])
                if status in TERMINAL:
                    rows.append(json.loads(path.with_name('result.json').read_text(encoding='utf-8')))
    return {'schema_version': 1, 'expected_campaigns': manifest['planned_campaigns'],
            'reported_campaigns': len(rows), 'statuses': statuses,
            'all_planned_reported': len(rows) == manifest['planned_campaigns'],
            'logical_llm_attempts': attempts, 'maximum_logical_attempts': manifest['maximum_logical_attempts'],
            'unique_responses': len(list((root / 'requests').glob('*/response.json'))),
            'pending_requests': sorted(pending),
            'numerical_or_invariant_errors': sum(r['status'] not in {'COMPLETE', 'STOPPED'} for r in rows),
            'hard_constraint_violations': sum(r['hard_constraint_violations'] for r in rows),
            'repeat_queries': sum(r['repeat_queries'] for r in rows),
            'b4_budget_violations': sum(r['budget_violations'] for r in rows if r['method'] == 'B4'),
            'b4_exhaustive_mismatches': sum(r['b4_exhaustive_mismatches'] for r in rows if r['method'] == 'B4'),
            'engine_sha256': engine_fingerprint()}


def freeze_assistant_test(test_directory, development_directory):
    test, dev = Path(test_directory).resolve(), Path(development_directory).resolve()
    dev_status = assistant_status(dev)
    if not dev_status['all_planned_reported'] or any(dev_status[k] for k in (
            'numerical_or_invariant_errors', 'hard_constraint_violations', 'repeat_queries',
            'b4_budget_violations', 'b4_exhaustive_mismatches')):
        raise ValueError('assistant development is incomplete or invalid')
    definition = json.loads((test / 'manifest.json').read_text(encoding='utf-8'))['definition']
    old = json.loads((dev / 'manifest.json').read_text(encoding='utf-8'))['definition']
    for key in ('settings', 'model_metadata', 'prompt_sha256', 'implementation_sha256', 'presentation'):
        if canonical_hash(definition[key]) != canonical_hash(old[key]):
            raise ValueError('assistant test changed the qualified development configuration')
    if set(definition['cases']) & set(old['cases']):
        raise ValueError('development and test cases overlap')
    receipt = {'manifest_sha256': sha256((test / 'manifest.json').read_bytes()).hexdigest(),
               'development_manifest_sha256': sha256((dev / 'manifest.json').read_bytes()).hexdigest(),
               'engine_sha256': engine_fingerprint(), 'implementation_sha256': assistant_implementation_hash(),
               'development_status': dev_status, 'backend_snapshot_unverifiable': True,
               'shared_chat_context': True, 'independent_replication_claimed': False}
    path = test / 'freeze_receipt.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != receipt:
        raise ValueError('test freeze already exists and differs')
    write_json_atomic(path, receipt)
    return receipt


def run_assistant_suite(directory, *, workers=4):
    root = Path(directory).resolve()
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    definition = manifest['definition']
    if assistant_implementation_hash() != definition['implementation_sha256']:
        raise ValueError('assistant orchestration changed')
    if sha256((root / 'prompt.txt').read_bytes()).hexdigest() != definition['prompt_sha256']:
        raise ValueError('assistant prompt changed')
    _verify_suite_inputs(root, manifest)
    if definition['split'] == 'test':
        freeze = json.loads((root / 'freeze_receipt.json').read_text(encoding='utf-8'))
        if (freeze['manifest_sha256'] != sha256(manifest_path.read_bytes()).hexdigest()
                or freeze['engine_sha256'] != engine_fingerprint()
                or freeze['implementation_sha256'] != assistant_implementation_hash()):
            raise ValueError('frozen assistant test changed')
    jobs = [(str(root), task, seed, definition) for task in manifest['tasks']
            for seed in definition['paired_seeds']]
    errors = []
    if workers == 1:
        for job in jobs:
            _run_group(job)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_group, job): job for job in jobs}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    print(json.dumps(result), flush=True)
                except Exception as exc:
                    job = futures[future]
                    errors.append({'task_id': job[1]['task_id'], 'seed': job[2],
                                   'error': f'{type(exc).__name__}: {exc}'})
    if errors:
        write_json_atomic(root / 'infrastructure_errors.json', errors)
        raise RuntimeError(f'{len(errors)} assistant groups failed; resume without deleting ledgers')
    status = assistant_status(root)
    write_json_atomic(root / 'run_status.json', status)
    return status


def submit_answers(directory, answers_path):
    """Store literal model answers, without repairing malformed hypotheses.

    Each answer is {context_sha256,batch,candidate_ids}; invalid batches are
    still returned to HypothesisManager and consume their logical attempt.
    """
    root = Path(directory).resolve()
    answers = json.loads(Path(answers_path).read_text(encoding='utf-8'))
    if not isinstance(answers, list):
        raise ValueError('answers must be a literal list authored by the assistant')
    submitted = []
    for answer in answers:
        if set(answer) != {'context_sha256', 'batch', 'candidate_ids'}:
            raise ValueError('unexpected answer envelope')
        key = answer['context_sha256']
        if len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
            raise ValueError('invalid context hash')
        folder = root / 'requests' / key
        request = json.loads((folder / 'request.json').read_text(encoding='utf-8'))
        if canonical_hash(request['context']) != key:
            raise ValueError('request hash changed')
        metadata = request['model_metadata']
        response = {'context_sha256': key, 'response': {
            'batch': answer['batch'], 'candidate_ids': answer['candidate_ids'],
            'provider': {'kind': 'interactive_codex_assistant', 'model_id': metadata['model_id'],
                         'conversation_id': metadata['conversation_id'],
                         'configured_reasoning_effort': metadata['configured_reasoning_effort'],
                         'input_tokens': None, 'output_tokens': None,
                         'authoring': 'literal model answer in current chat; no scripted hypothesis generator'}}}
        target = folder / 'response.json'
        if target.exists() and json.loads(target.read_text(encoding='utf-8')) != response:
            raise ValueError('cannot overwrite an earlier assistant response')
        write_json_atomic(target, response)
        write_json_atomic(folder / 'presentation.json', request_view(request))
        submitted.append(key)
    return {'submitted': len(submitted), 'context_sha256': submitted,
            'answers_file_sha256': sha256(Path(answers_path).read_bytes()).hexdigest()}
