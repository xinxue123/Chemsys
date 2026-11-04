"""Run the unmodified V13 loop through a process-local input/I/O adapter."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager, ExitStack
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
import json
import os
import shutil

import numpy as np
from filelock import FileLock
from state_split_v13_experiment import study as frozen, transport, policy, prompts
from paper_v12 import study as paper
from paper_v12_subagent import study as children
from v12_initialization_seeds import study as earlier

VARIANT = 'v13_seed123_20261102'
DESIGN = 'docs/v13_initialization_seeds_protocol_20261102.txt'
SHORT = dict(frozen.SHORT)
TASKS = tuple(SHORT)
SEEDS = (1, 2, 3)
JOBS = tuple((task, seed) for task in TASKS for seed in SEEDS)
NATIVE_LOAD = frozen.gpmod.load_task
NATIVE_GP = frozen.gpmod.GPAdapter
FROZEN_CHECK = frozen.check
FROZEN_BASE = transport.base
old = frozen.old


def directory(root):
    return (Path(root).resolve() / 'experiments/runs' / VARIANT).resolve()


def tag(task, seed):
    if (task, seed) not in JOBS or type(seed) is not int:
        raise ValueError('Unregistered task/seed')
    return f'm{seed}_{SHORT[task]}'


def base(root, task, seed):
    tag(task, seed)
    return directory(root) / f's{seed}' / SHORT[task]


def run_path(root, task, seed):
    return base(root, task, seed) / 'runs' / tag(task, seed)


def initialization(pool, seed):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError('Unregistered initialization seed')
    order = sorted(range(len(pool.ids)), key=lambda j: pool.features[j])
    if len(set(pool.features)) != len(pool.ids):
        raise ValueError('Physical inputs must be unique')
    positions = np.random.default_rng(np.random.SeedSequence([seed, 0])).choice(len(order), 4, replace=False)
    return tuple(pool.ids[order[int(i)]] for i in positions)


def registrations(root):
    root = Path(root).resolve()
    rows = []
    for task in TASKS:
        pool = NATIVE_LOAD(root, task, seed=0)
        for seed in SEEDS:
            ids = initialization(pool, seed)
            rows.append(dict(task=task, initialization_seed=seed, initial_ids=list(ids),
                initial_features=[list(pool.features[pool.ix[k]]) for k in ids],
                input_path=pool.source_path.relative_to(root).as_posix(), input_sha256=old.sha(pool.source_path),
                candidate_count=len(pool.ids), axes=list(pool.axes), kinds=list(pool.kinds), unit=pool.unit))
    return rows


def controls(root):
    root = Path(root).resolve()
    rows = []
    for task, seed in JOBS:
        bo = earlier.run_path(root, task, seed, 'bo') if task in paper.FOCUS else paper.run_path(root, task, seed, 'bo')
        paths = dict(bo=bo / 'state.json',
            v12_subagent=children.run_path(root, task, seed, 'memory') / 'state.json',
            reasoning_bo=children.run_path(root, task, seed, 'reasoning_bo') / 'state.json')
        row = dict(task=task, initialization_seed=seed, arms={})
        for arm, path in paths.items():
            state = old.read(path)
            if state['status'] != 'COMPLETE' or len(state['history']) != 20 or len(state['decisions']) != 16:
                raise ValueError('Incomplete frozen control')
            row['arms'][arm] = dict(path=path.relative_to(root).as_posix(), sha256=old.sha(path))
        rows.append(row)
    return rows


def sources(root):
    root = Path(root).resolve()
    result = frozen.sources(root)
    paths = list((root / 'code/experiment_tools/v13_initialization_seeds').glob('*.py'))
    paths += [root / 'code/experiment_tools/run_v13_initialization_seeds.py', root / DESIGN]
    result.update({p.relative_to(root).as_posix(): old.sha(p) for p in paths})
    return result


def protect(root, control_rows):
    root = Path(root).resolve()
    paths = [FROZEN_BASE(root) / 'protocol.json']
    paths += [FROZEN_BASE(root) / 'runs' / short / 'state.json' for short in SHORT.values()]
    for row in control_rows:
        for arm, value in row['arms'].items():
            path = root / value['path']
            paths += [path, path.parent / 'rounds/01_GP.npz']
            if arm == 'bo':
                paths += list((path.parent / 'rounds').glob('*_GP.npz'))
                paths += list((path.parent / 'rounds').glob('*_commit.json'))
    return {p.relative_to(root).as_posix(): old.sha(p) for p in paths}


def check(root, verify_controls=False):
    root = Path(root).resolve()
    p = transport.read(directory(root) / 'protocol.json')
    f = FROZEN_CHECK(root, verify_controls=verify_controls)
    if p['protocol_sha256'] != transport.digest({k: v for k, v in p.items() if k != 'protocol_sha256'}):
        raise ValueError('Multi-seed protocol changed')
    if p['sources_sha256'] != sources(root) or p['frozen_V13_protocol_sha256'] != f['protocol_sha256']:
        raise ValueError('Frozen implementation changed')
    if p['parameters'] != policy.PARAMETERS or p['jobs'] != [list(j) for j in JOBS]:
        raise ValueError('Policy or scope changed')
    if p['initializations'] != registrations(root) or (p['initial'], p['queries']) != (4, 16):
        raise ValueError('Registered initializations or budget changed')
    if verify_controls:
        for name, digest in p['protected_sha256'].items():
            if old.sha(root / name) != digest:
                raise ValueError('Historical evidence changed: ' + name)
    return p


def prepare(root):
    root = Path(root).resolve()
    d = directory(root)
    if (d / 'protocol.json').exists():
        return check(root, True)
    if any(d.glob('s[123]')):
        raise ValueError('Register before new trajectories')
    frozen_protocol = FROZEN_CHECK(root, True)
    refs = controls(root)
    p = dict(variant=VARIANT, created_utc=datetime.now(timezone.utc).isoformat(),
        jobs=[list(j) for j in JOBS], tasks=list(TASKS), initialization_seeds=list(SEEDS),
        seed=0, trial_seed=0, numerical_seed=0, tie_seed=0, menu_seed=0, initial=4, queries=16,
        parameters=deepcopy(policy.PARAMETERS), model=frozen.MODEL, effort=frozen.EFFORT,
        backend=frozen.BACKEND, fork_turns='none', inputs=deepcopy(frozen_protocol['inputs']),
        initializations=registrations(root), controls=refs, protected_sha256=protect(root, refs),
        frozen_V13_protocol_sha256=frozen_protocol['protocol_sha256'], sources_sha256=sources(root),
        prompt_sha256=deepcopy(frozen_protocol['prompt_sha256']),
        maximum_new_queries=384, maximum_logical_LLM_calls=480,
        seed_semantics='Only initialization changes; native numerical/tie/display seed remains0',
        response_cache='fresh independent packet and agent for every context and schema repair',
        new_outcomes_accessed_at_registration=False, new_LLM_calls_at_registration=0)
    p['initial_registry_sha256'] = transport.digest(p['initializations'])
    p['protocol_sha256'] = transport.digest(p)
    transport.write_once(d / 'protocol.json', p)
    snapshot = root / 'runtime/knowledge_loop/snapshots' / VARIANT
    for name, digest in p['sources_sha256'].items():
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target)
        if old.sha(target) != digest:
            raise ValueError('Source snapshot mismatch')
    transport.write_once(snapshot / 'protocol.json', p)
    return check(root, True)


def pool_for(root, task, seed, p):
    pool = NATIVE_LOAD(root, task, seed=0)
    row = next(r for r in p['initializations'] if r['task'] == task and r['initialization_seed'] == seed)
    ids = initialization(pool, seed)
    if list(ids) != row['initial_ids'] or old.sha(pool.source_path) != row['input_sha256']:
        raise ValueError('Input-only registration mismatch')
    return replace(pool, initial_ids=ids)


def envelope(p, task, seed):
    result = deepcopy(p)
    row = next(r for r in p['controls'] if r['task'] == task and r['initialization_seed'] == seed)
    result['controls'] = {task: row['arms']}
    return result


@contextmanager
def redirected(root, task, seed, p):
    root = Path(root).resolve()
    pool = pool_for(root, task, seed, p)
    b = base(root, task, seed)
    env = envelope(p, task, seed)

    def loader(r, name, *, seed=0):
        if Path(r).resolve() != root or name != task or seed != 0:
            raise ValueError('Unexpected native loader call')
        return pool

    def gp(obj, path, seed=0):
        if obj is not pool or seed != 0 or Path(path) != b / 'cache' / task:
            raise ValueError('Unexpected native GP call')
        return NATIVE_GP(pool, directory(root) / 'cache' / task, seed=0)

    with ExitStack() as stack:
        stack.enter_context(patch.object(frozen, 'check', return_value=deepcopy(env)))
        stack.enter_context(patch.object(frozen, 'SHORT', {task: tag(task, seed)}))
        stack.enter_context(patch.object(frozen, 'copy_cache', return_value=0))
        stack.enter_context(patch.object(frozen.gpmod, 'load_task', side_effect=loader))
        stack.enter_context(patch.object(frozen.gpmod, 'GPAdapter', side_effect=gp))
        stack.enter_context(patch.object(transport, 'base', return_value=b))
        stack.enter_context(patch.object(transport, 'VARIANT', VARIANT + '_' + tag(task, seed)))
        yield pool, env


def copy_cache(root):
    root = Path(root).resolve()
    source = children.directory(root) / 'cache'
    target = directory(root) / 'cache'
    target.mkdir(parents=True, exist_ok=True)
    count = 0
    with FileLock(str(target / '.copy.lock'), timeout=900):
        for path in source.rglob('*'):
            if not path.is_file() or path.suffix not in ('.npz', '.pt', '.json'):
                continue
            dest = target / path.relative_to(source)
            if dest.exists():
                if old.sha(dest) != old.sha(path):
                    raise ValueError('Native numerical cache conflict')
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
                count += 1
    return count


def preflight_one(root, task, seed):
    root = Path(root).resolve()
    os.chdir(root / 'experiments/chemist_system')
    p = check(root)
    pool = pool_for(root, task, seed, p)
    env = envelope(p, task, seed)
    folder = base(root, task, seed) / 'preflight'
    old.initial_commit(folder, pool, p, 'state_split', 'agent_initial_terra', create=True)
    history = [frozen.gpmod.read_observation(root, pool, k) for k in pool.initial_ids]
    baseline_path = root / env['controls'][task]['bo']['path']
    state = old.read(baseline_path)
    if history != state['history'][:4]:
        raise ValueError('Matched BO initialization differs')
    gp = NATIVE_GP(pool, directory(root) / 'cache' / task, seed=0)
    ctx = gp.context(history, 1)
    with np.load(baseline_path.parent / 'rounds/01_GP.npz', allow_pickle=False) as z:
        for k in ('mu_y', 'sd_y', 'sd_predictive_y'):
            np.testing.assert_array_equal(ctx['stats'][k], z[k])
        for k in ('base_logei', 'available', 'tie_priority'):
            np.testing.assert_array_equal(ctx[k], z[k])
    # Verify the frozen pure-BO arm actually used its archived native acquisition.
    for q, observation in enumerate(state['history'][4:], 1):
        with np.load(baseline_path.parent / 'rounds' / f'{q:02d}_GP.npz', allow_pickle=False) as z:
            available = ~np.isin(pool.ids, [r['candidate_id'] for r in state['history'][:q+3]])
            np.testing.assert_array_equal(z['available'], available)
            order, _ = frozen.native.candidates(pool, {k: z[k] for k in ('available', 'base_logei', 'tie_priority')}, q)
            if observation['candidate_id'] != pool.ids[int(order[0])]:
                raise ValueError('Archived BO is not the matched GP argmax')
    receipt = dict(task=task, initialization_seed=seed, status='PASS_EXACT',
        initial_ids=list(pool.initial_ids), initial_history=history,
        GP_sha256=frozen.native.gp_hash(ctx), baseline_queries_audited=16,
        counters=gp.counters, new_queries=0, LLM_calls=0)
    gp.release()
    transport.write_once(folder / 'receipt.json', receipt)
    return receipt


def preflight(root, workers=3):
    p = check(root, True)
    copied = copy_cache(root)
    with ProcessPoolExecutor(max_workers=min(workers, 3)) as executor:
        rows = list(executor.map(preflight_one, [str(root)] * len(JOBS),
                                [x[0] for x in JOBS], [x[1] for x in JOBS]))
    result = dict(status='PASS', protocol_sha256=p['protocol_sha256'], rows=rows,
                  numerical_cache_files_copied=copied, new_queries=0, LLM_calls=0)
    old.write(directory(root) / 'preflight.json', result)
    return result


def run_one(root, task, seed):
    root = Path(root).resolve()
    p = check(root)
    b = base(root, task, seed)
    pf = transport.read(b / 'preflight/receipt.json')
    if pf['status'] != 'PASS_EXACT':
        raise ValueError('Matched numerical preflight required')
    launch = dict(task=task, initialization_seed=seed, native_seed=0,
        protocol_sha256=p['protocol_sha256'], initial_ids=pf['initial_ids'],
        backend=frozen.BACKEND, response_files_copied=0, LLM_cache_initially_empty=True)
    with FileLock(str(b / '.pair.lock'), timeout=1):
        if not (b / 'launch.json').exists() and any((b / k).exists() for k in ('runs', 'requests')):
            raise ValueError('A new pair requires fresh response namespaces')
        transport.write_once(b / 'launch.json', launch)
        with redirected(root, task, seed, p):
            result = dict(frozen.run_one(root, task), initialization_seed=seed)
        check(root)
        transport.write_once(b / 'completion.json', result)
    print('FINISH ' + json.dumps(result), flush=True)
    return result


def batch(root, workers=3):
    check(root, True)
    errors = []
    with ProcessPoolExecutor(max_workers=min(workers, 3)) as executor:
        fs = {executor.submit(run_one, str(root), *j): j for j in JOBS}
        for future in as_completed(fs):
            try:
                future.result()
            except Exception as exc:
                row = dict(job=fs[future], type=type(exc).__name__, message=str(exc))
                errors.append(row)
                print('JOB_ERROR ' + json.dumps(row), flush=True)
    check(root, True)
    old.write(directory(root) / 'batch_receipt.json', dict(status='INCOMPLETE' if errors else 'COMPLETE', errors=errors))
    if errors:
        raise RuntimeError('Preserve prefixes and inspect errors: ' + json.dumps(errors))


def pending(root):
    rows = []
    for task, seed in JOBS:
        with patch.object(transport, 'base', return_value=base(root, task, seed)):
            rows.extend(dict(row, task=task, seed=seed) for row in transport.pending(root))
    return rows


def status(root):
    rows = []
    for task, seed in JOBS:
        path = run_path(root, task, seed) / 'state.json'
        s = transport.read(path) if path.exists() else {}
        rows.append(dict(task=SHORT[task], seed=seed, status=s.get('status', 'NOT_STARTED'), queries=len(s.get('decisions', []))))
    return dict(complete=sum(r['status'] == 'COMPLETE' for r in rows), queries=sum(r['queries'] for r in rows), rows=rows)
