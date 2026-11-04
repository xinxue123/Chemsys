"""Process-local random-start and I/O adapter; scientific loop stays frozen."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager, ExitStack
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
import json
import os
import shutil

import numpy as np
from cold_start_v13 import study as frozen, transport, validation, policy

VARIANT = 'cold_s123_20261102'
DESIGN = 'docs/cold_start_multiseed_20261102.txt'
SEEDS = (1, 2, 3)
SHORT = dict(frozen.SHORT)
ARMS = tuple(frozen.ARMS)
JOBS = tuple((task, seed, arm) for task in SHORT for seed in SEEDS for arm in ARMS)
NATIVE_CHECK = frozen.check
NATIVE_COLD_GP = frozen.ColdGP
NATIVE_INITIAL_STATE = frozen.initial_state
NATIVE_PRIORITIES = frozen.native_gp.semantic_priorities
NATIVE_POOL = frozen.pool_for
NATIVE_DIRECTORY = frozen.directory
SCHEMA_MESSAGE = ('Schema-only clarification for this Reasoning BO request: the top-level output is exactly '
    'comment, keywords, hypotheses. The V13-only knowledge_uses and assessments clauses do not apply '
    'to this bare official Insight. Follow the original scientific selection and rationale rules; '
    'no other instruction changes.')


def directory(root):
    return (Path(root).resolve() / 'experiments/runs' / VARIANT).resolve()


def base(root, seed):
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError('Only registered new starting seeds1,2,3')
    return directory(root) / f's{seed}'


def tag(task, seed, arm):
    if (task, seed, arm) not in JOBS:
        raise ValueError('Unregistered task/seed/method')
    return f'ms{seed}_{SHORT[task]}_' + {'bo':'bo','v13':'v13','reasoning_bo':'rbo'}[arm]


def run_path(root, task, seed, arm):
    return base(root, seed) / 'runs' / tag(task, seed, arm)


def priorities(start_seed):
    if type(start_seed) is not int or start_seed not in (0, *SEEDS):
        raise ValueError('Unsupported starting seed')

    def draw(pool, query, seed=0):
        if seed != 0:
            raise ValueError('Native GP numerical/tie seed must remain0')
        return NATIVE_PRIORITIES(pool, query, start_seed if query == 1 else 0)
    return draw


def registrations(root):
    rows = []
    for task in SHORT:
        pool = NATIVE_POOL(root, task)
        for seed in SEEDS:
            ties = NATIVE_PRIORITIES(pool, 1, seed)
            context = dict(available=np.ones(len(pool.ids), bool),
                           base_logei=np.zeros(len(pool.ids)), tie_priority=ties)
            ordered, display = frozen.native.candidates(pool, context, 1)
            rows.append(dict(task=task, starting_seed=seed, initial_ids=[],
                first_prior_ids=[pool.ids[int(j)] for j in ordered],
                first_display_ids=[pool.ids[int(j)] for j in display],
                first_display_features=[list(pool.features[int(j)]) for j in display],
                BO_first_id=pool.ids[int(ordered[0])], candidate_count=len(pool.ids),
                sorted_features_sha256=transport.digest(sorted(pool.features)),
                input_path=pool.source_path.relative_to(Path(root)).as_posix()))
    return rows


def sources(root):
    root = Path(root).resolve()
    result = frozen.sources(root)
    paths = list((root / 'code/experiment_tools/cold_start_v13_seeds').glob('*.py'))
    paths += [root / 'code/experiment_tools/run_cold_start_multiseed.py', root / DESIGN]
    result.update({p.relative_to(root).as_posix():transport.file_sha(p) for p in paths})
    return result


def protect_seed0(root):
    root = Path(root).resolve()
    old = NATIVE_DIRECTORY(root)
    paths = [old / 'protocol.json', old / 'preflight.json', old / 'batch_receipt.json']
    for name in ('runs', 'requests', 'preflight'):
        paths.extend(p for p in (old / name).rglob('*') if p.is_file() and p.suffix != '.lock')
    for folder in (root / 'experiments/reports/cold0_20261102',
                   root / 'experiments/agent_io/cold0_20261102'):
        paths.extend(p for p in folder.rglob('*') if p.is_file())
    return {p.relative_to(root).as_posix():transport.file_sha(p) for p in sorted(set(paths))}


def check(root, historical=False):
    root = Path(root).resolve()
    p = transport.read(directory(root) / 'protocol.json')
    old = NATIVE_CHECK(root, historical)
    if p['protocol_sha256'] != transport.digest({k:v for k,v in p.items() if k!='protocol_sha256'}):
        raise ValueError('Multi-seed registration changed')
    if p['sources_sha256'] != sources(root) or p['frozen_cold_protocol_sha256'] != old['protocol_sha256']:
        raise ValueError('Frozen source changed')
    if (p['initial'],p['queries'],p['starting_seeds'],p['numerical_seed'],p['post_first_tie_seed'],p['display_seed']) != (0,20,list(SEEDS),0,0,0):
        raise ValueError('Random streams or budget changed')
    if p['parameters'] != policy.PARAMETERS or p['jobs'] != [list(j) for j in JOBS]:
        raise ValueError('Scientific settings or scope changed')
    if p['first_menu_registry_sha256'] != transport.digest(p['first_menus']):
        raise ValueError('Starting menu registry changed')
    if historical:
        if p['first_menus'] != registrations(root):
            raise ValueError('Input-only starting menus changed')
        for name, digest in p['protected_seed0_sha256'].items():
            if transport.file_sha(root / name) != digest:
                raise ValueError('Seed0 evidence changed: '+name)
    return p


def prepare(root):
    root = Path(root).resolve()
    d = directory(root)
    if (d / 'protocol.json').exists():
        return check(root, True)
    if any(d.glob('s[123]')):
        raise ValueError('Register before any new outcomes or responses')
    prior = NATIVE_CHECK(root, True)
    audit = transport.read(root / 'experiments/reports/cold0_20261102/audit.json')
    if (audit['status'],audit['trajectories'],audit['queries_replayed']) != ('PASS_COMPLETE',24,480):
        raise ValueError('Complete frozen seed0 required')
    menus = registrations(root)
    p = dict(variant=VARIANT,created_utc=datetime.now(timezone.utc).isoformat(),
        jobs=[list(j) for j in JOBS],tasks=list(SHORT),starting_seeds=list(SEEDS),
        initial=0,queries=20,numerical_seed=0,post_first_tie_seed=0,display_seed=0,
        parameters=deepcopy(policy.PARAMETERS),model=frozen.MODEL,effort=frozen.EFFORT,
        backend=frozen.BACKEND,fork_turns='none',maximum_new_queries=1440,
        maximum_logical_LLM_calls=1056,first_menus=menus,
        first_menu_registry_sha256=transport.digest(menus),
        frozen_cold_protocol_sha256=prior['protocol_sha256'],
        protected_seed0_sha256=protect_seed0(root),sources_sha256=sources(root),
        seed_semantics='Only data-free query1 prior/menu stream changes; GP, MC512 qLogEI, post-first exact ties and display remain seed0',
        response_semantics='One fresh gpt-6.1-terra medium child per request; no cache reuse across any seed, arm or request',
        endpoint='All20 post-query cumulative maxima enter SearchAUC; zeros and nonimprovements count',
        aggregation='Per-task paired seed0,1,2,3; mean and sample SD; per-seed paired differences and win/tie/loss counts; never average raw units across tasks',
        inference_limit='Four starting seeds with one LLM trajectory per arm/seed; joint start/response variability, not an isolated estimate of either variance',
        schema_adapter='Same separate format-only first-round Reasoning BO clarification as seed0; original bytes and schema failures retained',
        allowed_adapter_changes=['starting_prior_seed','seeded_IO_namespace','explicit_seed_state_metadata','shared_exact_prefix_numeric_cache'],
        no_oracle_values_read_at_registration=True,no_LLM_calls_at_registration=True)
    p['protocol_sha256'] = transport.digest(p)
    transport.write_once(d / 'protocol.json', p)
    snapshot = root / 'runtime/knowledge_loop/snapshots' / VARIANT
    for name, digest in p['sources_sha256'].items():
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target)
        if transport.file_sha(target) != digest:
            raise ValueError('Source snapshot mismatch')
    transport.write_once(snapshot / 'protocol.json', p)
    return check(root, True)


@contextmanager
def redirected(root, seed, protocol):
    root = Path(root).resolve()
    b = base(root, seed)
    env = deepcopy(protocol)
    env.update(seed=seed,starting_seed=seed)

    def initial(task, arm, p, commitment):
        value = NATIVE_INITIAL_STATE(task, arm, p, commitment)
        value.update(seed=seed,starting_seed=seed,numerical_seed=0)
        return value

    def gp(pool, path, seed=0):
        if seed != 0 or Path(path) != b / 'cache' / pool.task:
            raise ValueError('Unexpected GP stream or cache request')
        return NATIVE_COLD_GP(pool, directory(root) / 'cache' / pool.task, seed=0)

    with ExitStack() as stack:
        stack.enter_context(patch.object(frozen, 'check', return_value=env))
        stack.enter_context(patch.object(frozen, 'directory', return_value=b))
        stack.enter_context(patch.object(frozen, 'tag', side_effect=lambda task,arm:tag(task,seed,arm)))
        stack.enter_context(patch.object(frozen, 'run_path', side_effect=lambda r,task,arm:run_path(r,task,seed,arm)))
        stack.enter_context(patch.object(frozen, 'initial_state', side_effect=initial))
        stack.enter_context(patch.object(frozen, 'ColdGP', side_effect=gp))
        stack.enter_context(patch.object(frozen.native_gp, 'semantic_priorities', side_effect=priorities(seed)))
        stack.enter_context(patch.object(frozen, 'VARIANT', VARIANT+f'/s{seed}'))
        stack.enter_context(patch.object(transport, 'base', return_value=b))
        stack.enter_context(patch.object(transport, 'VARIANT', VARIANT+f'_s{seed}'))
        yield env


def _preflight_task(root, task):
    root = Path(root).resolve()
    os.chdir(root / 'experiments/chemist_system')
    pool = NATIVE_POOL(root,task)
    gp = NATIVE_COLD_GP(pool,directory(root) / 'preflight_cache' / task,seed=0)
    zero = gp.context([],1)
    old_path = NATIVE_DIRECTORY(root) / 'runs' / (SHORT[task]+'_bo') / 'rounds/01_GP.npz'
    with np.load(old_path,allow_pickle=False) as z:
        for key in ('base_logei','available','tie_priority'):
            np.testing.assert_array_equal(zero[key],z[key])
        for key in ('mu_y','sd_y','sd_predictive_y'):
            np.testing.assert_array_equal(zero['stats'][key],z[key])
    menus = []
    for seed in SEEDS:
        with patch.object(frozen.native_gp,'semantic_priorities',side_effect=priorities(seed)):
            first = gp.context([],1)
        np.testing.assert_array_equal(first['tie_priority'],NATIVE_PRIORITIES(pool,1,seed))
        assert not first['fitted_GP'] and np.isnan(first['stats']['mu_y']).all()
        ordered,display = frozen.native.candidates(pool,first,1)
        assert int(ordered[0]) == first['bo_index']
        menus.append(dict(seed=seed,first_menu=[pool.ids[int(j)] for j in display]))
    indices = np.argsort(zero['tie_priority'])[:4]
    history = [dict(candidate_id=pool.ids[int(j)],y=y,raw_value=y*pool.scale)
               for j,y in zip(indices,(.2,.4,.6,.3))]
    proofs = []
    for count in (1,2,4):
        reference = gp.context(history[:count],count+1)
        expected = frozen.native.gp_hash(reference)
        for seed in SEEDS:
            with patch.object(frozen.native_gp,'semantic_priorities',side_effect=priorities(seed)):
                candidate = gp.context(history[:count],count+1)
            if frozen.native.gp_hash(candidate) != expected:
                raise ValueError('Post-first GP or acquisition changed')
            assert candidate['stats']['fit_parameters'] == reference['stats']['fit_parameters']
            gp.release()
        proofs.append(dict(synthetic_observations=count,all_seeds_exact_GP_and_acquisition=True))
    assert gp.counters['operator_compiles_in_loop'] == 0
    gp.release()
    result = dict(task=task,status='PASS',seed0_first_round_exact=True,menus=menus,
                  proofs=proofs,oracle_values_read=0,LLM_calls=0,counters=gp.counters)
    transport.write_once(directory(root) / 'preflight' / (SHORT[task]+'.json'),result)
    return result


def preflight(root, workers=3):
    check(root,True)
    with ProcessPoolExecutor(max_workers=min(workers,3)) as executor:
        rows = list(executor.map(_preflight_task,[str(root)]*len(SHORT),SHORT))
    result = dict(status='PASS',tasks=len(rows),starting_menus=24,rows=rows,
                  oracle_values_read=0,LLM_calls=0,synthetic_cache_separate=True)
    transport.write_once(directory(root) / 'preflight.json',result)
    return dict(status='PASS',tasks=len(rows),starting_menus=24,oracle_values_read=0,LLM_calls=0)


def run_one(root, task, seed, arm):
    p = check(root)
    if transport.read(directory(root) / 'preflight.json')['status'] != 'PASS':
        raise ValueError('Numerical preflight required')
    with redirected(root,seed,p):
        result = dict(frozen.run_one(root,task,arm),starting_seed=seed)
    check(root)
    return result


def batch(root, workers=3):
    check(root,True)
    errors = []
    with ProcessPoolExecutor(max_workers=min(workers,3)) as executor:
        futures = {executor.submit(run_one,str(root),*job):job for job in JOBS}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                row = dict(job=futures[future],type=type(exc).__name__,message=str(exc))
                errors.append(row)
                print('JOB_ERROR '+json.dumps(row),flush=True)
    check(root,True)
    transport.write_once(directory(root) / 'batch_receipt.json',dict(status='INCOMPLETE' if errors else 'COMPLETE',errors=errors))
    if errors:
        raise RuntimeError('Failed prefixes retained: '+json.dumps(errors))


def pending(root):
    rows = []
    for seed in SEEDS:
        with patch.object(transport,'base',return_value=base(root,seed)):
            rows.extend(dict(r,seed=seed) for r in transport.pending(root))
    return rows


def finish(root, agents):
    rows = {'/root/'+r['invocation']['task_name']:r for r in pending(root)}
    for agent in agents:
        row = rows[agent]
        with patch.object(transport,'base',return_value=base(root,row['seed'])):
            transport.record_spawn(root,row['request'],dict(task_name=agent))
            transport.seal(root,row['request'])


def record_schema(root, agent):
    rows = [r for r in pending(root) if '/root/'+r['invocation']['task_name']==agent]
    if len(rows) != 1:
        raise ValueError('One pending actual invocation required')
    path = Path(rows[0]['request'])
    req = transport.verify_request(path)
    if req['query'] != 1 or not req['task'].endswith('_rbo') or req['attempt'] != 1:
        raise ValueError('Only first-round Reasoning BO schema clarification')
    transport.write_once(path.parent / 'format_clarification.json',dict(agent_id=agent,
        message=SCHEMA_MESSAGE,kind='schema_clarification_before_first_response',
        request_sha256=req['request_sha256'],response_bytes_edited_by_parent=False))


def status(root):
    rows = []
    for task,seed,arm in JOBS:
        path = run_path(root,task,seed,arm) / 'state.json'
        state = transport.read(path) if path.exists() else {}
        rows.append(dict(task=SHORT[task],seed=seed,arm=arm,status=state.get('status','NOT_STARTED'),
                         queries=len(state.get('history',[]))))
    return dict(complete=sum(r['status']=='COMPLETE' for r in rows),queries=sum(r['queries'] for r in rows),rows=rows)
