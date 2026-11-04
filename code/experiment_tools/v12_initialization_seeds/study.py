"""Initialization/I/O adapter; the original V12 and pure-BO loops stay intact."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from contextlib import contextmanager,ExitStack
from copy import deepcopy
from dataclasses import replace
from datetime import datetime,timezone
from pathlib import Path
from unittest.mock import patch
import json,shutil,subprocess
import numpy as np
from filelock import FileLock
from chem.benchmarks.knowledge_llm import digest,write
from v12_llm_repeats import study as prior

v12=prior.v12
v9=v12.v9
old=v12.old
NATIVE_LOAD_TASK=v12.gpmod.load_task
VARIANT='v12_seed123'
TASKS=tuple(v12.TASKS)
SEEDS=(1,2,3)
ARMS=('bo','memory')
PAIRS=tuple((task,seed) for task in TASKS for seed in SEEDS)
DESIGN='docs/v12_initialization_seeds_protocol_20261101.txt'


def directory(root):return Path(root)/'experiments/runs/knowledge_loop_20261101'/VARIANT


def pair_base(root,task,initialization_seed):
    if initialization_seed not in SEEDS or type(initialization_seed) is not int or not task or any(c in task for c in '/\\:'):
        raise ValueError('Invalid task/initialization path')
    return directory(root)/f's{initialization_seed}'/task


def run_path(root,task,initialization_seed,arm):
    if arm not in ARMS:raise ValueError('Only frozen BO and V12 memory are registered')
    return pair_base(root,task,initialization_seed)/'runs'/f'{task}_{arm}'


def draw_initial_ids(pool,initialization_seed):
    if type(initialization_seed) is not int or initialization_seed not in SEEDS:raise ValueError('Unregistered initialization seed')
    if len(set(pool.features))!=len(pool.ids):raise ValueError('Physical features must uniquely order candidates')
    order=sorted(range(len(pool.ids)),key=lambda j:pool.features[j])
    positions=np.random.default_rng(np.random.SeedSequence([initialization_seed,0])).choice(len(order),4,replace=False)
    return tuple(pool.ids[order[int(j)]] for j in positions),list(map(int,positions))


def input_registration(root):
    root=Path(root).resolve();rows=[]
    for task in TASKS:
        pool=NATIVE_LOAD_TASK(root,task,seed=0)
        for seed in SEEDS:
            ids,positions=draw_initial_ids(pool,seed)
            if seed in (1,2) and ids!=NATIVE_LOAD_TASK(root,task,seed=seed).initial_ids:
                raise ValueError('Existing registered seed1/2 initialization differs from common input-only draw')
            row=dict(task=task,initialization_seed=seed,input_pool=pool.source_path.relative_to(root).as_posix(),
                input_pool_sha256=old.sha(pool.source_path),candidate_count=len(pool.ids),axes=list(pool.axes),kinds=list(pool.kinds),
                sorted_features_sha256=digest([list(x) for x in sorted(pool.features)]),drawn_sorted_positions=positions,
                initial_ids=list(ids),initial_points=[pool.points[pool.ix[k]] for k in ids],
                initial_features=[list(pool.features[pool.ix[k]]) for k in ids],previously_used_initialization=seed in (1,2))
            row['registration_sha256']=digest(row);rows.append(row)
    return rows


def registered_pool(root,task,seed,protocol):
    pool=NATIVE_LOAD_TASK(root,task,seed=0)
    row=next(r for r in protocol['initializations'] if r['task']==task and r['initialization_seed']==seed)
    ids,positions=draw_initial_ids(pool,seed)
    if (list(ids)!=row['initial_ids'] or positions!=row['drawn_sorted_positions'] or old.sha(pool.source_path)!=row['input_pool_sha256']
            or [list(pool.features[pool.ix[k]]) for k in ids]!=row['initial_features']):raise ValueError('Registered inputs changed')
    return replace(pool,initial_ids=ids)


def sources(root):
    root=Path(root);files=prior.sources(root)
    paths=list((root/'code/experiment_tools/v12_initialization_seeds').rglob('*.py'))+[root/'code/experiment_tools/run_v12_initialization_seeds.py']
    files.update({p.relative_to(root).as_posix():old.sha(p) for p in paths});return files


def protected(root):
    root=Path(root);frozen=prior.check(root);files=deepcopy(frozen['protected_controls_sha256'])
    paths=[p for p in prior.directory(root).rglob('*') if p.is_file() and p.suffix!='.lock']
    paths += [p for p in (root/'experiments/reports/knowledge_loop_20261101').glob('v12_repeats_*') if p.is_file()]
    paths += [root/'docs/v12_llm_repeats_protocol_20261101.txt',root/'docs/v12_llm_repeats_results_20261101.txt']
    for seed in (1,2):
        paths += [root/getattr(v12.gpmod,f'SEED{seed}_REGISTRY_PATH'),root/getattr(v12.gpmod,f'SEED{seed}_RULE_PATH')]
    files.update({p.relative_to(root).as_posix():old.sha(p) for p in paths});return files


def prepare(root):
    root=Path(root).resolve();base=directory(root)
    if (base/'protocol.json').exists():return check(root)
    if any((base/f's{s}').exists() for s in SEEDS) or (base/'preflight').exists():raise ValueError('Register before observing new initial data or running models')
    frozen=v12.check(root);prior.check(root);rows=input_registration(root)
    p=dict(variant=VARIANT,created_utc=datetime.now(timezone.utc).isoformat(),tasks=list(TASKS),initialization_seeds=list(SEEDS),
        paired_runs=[list(x) for x in PAIRS],runs=[[task,arm] for task in TASKS for arm in ARMS],arms=list(ARMS),
        seed=0,trial_seed=0,tie_seed=0,numerical_seed=0,menu_seed=0,initial=4,queries=16,
        model=frozen['model'],reasoning_effort=frozen['reasoning_effort'],parameters=deepcopy(frozen['parameters']),
        prompt_sha256=deepcopy(frozen['prompt_sha256']),inputs=deepcopy(frozen['inputs']),initializations=rows,
        initial_registry_sha256=digest(rows),initialization_rule='physical-feature lexicographic order; PCG64 SeedSequence([initialization_seed,0]); choice(N,4,replace=False), retain order',
        frozen_V12_protocol_sha256=frozen['protocol_sha256'],frozen_V12_core_commit=frozen['code_milestone'],
        frozen_repeats_protocol_sha256=prior.check(root)['protocol_sha256'],
        source_sha256=sources(root),protected_controls_sha256=protected(root),design_path=DESIGN,design_sha256=old.sha(root/DESIGN),
        code_milestone=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        seed_semantics='initialization_seed selects four input points; native state.seed remains0, GP numerical/tie/display streams remain frozen0; task/initialization_seed recorded in registration, paths, launches and summaries',
        execution='unmodified V12 run_one(memory) and V9 run_one(bo); only input pool initial_ids, protocol envelope and I/O paths redirected',
        cache_policy='one fresh LLM namespace per task/initialization_seed, none copied; shared new native GP cache with prefix locks',
        maximum_new_queries=288,maximum_logical_LLM_calls=180,maximum_schema_attempts_per_context=3,
        initial_outcomes_accessed=False,LLM_calls_at_registration=0,GP_fits_at_registration=0,
        inference_boundary='Three paired initializations, one fresh V12 LLM trajectory each; seed1/2 used in prior development; no pooling seed0 or significant/global stability claim',
        no_performance_based_rerolls=True)
    p['protocol_sha256']=digest(p);write(base/'protocol.json',p)
    snapshot=root/'runtime/knowledge_loop/snapshots'/VARIANT
    for name,sha in p['source_sha256'].items():
        dest=snapshot/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,dest)
        if old.sha(dest)!=sha:raise ValueError('Snapshot drift')
    (snapshot/DESIGN).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/DESIGN,snapshot/DESIGN)
    shutil.copy2(base/'protocol.json',snapshot/'protocol.json');return check(root)


def check(root,verify_controls=False):
    root=Path(root);p=old.read(directory(root)/'protocol.json');frozen=v12.check(root);prior.check(root)
    if p['protocol_sha256']!=digest({k:v for k,v in p.items() if k!='protocol_sha256'}):raise ValueError('Seed protocol drift')
    if p['source_sha256']!=sources(root) or p['design_sha256']!=old.sha(root/p['design_path']):raise ValueError('Frozen implementation drift')
    if p['frozen_V12_protocol_sha256']!=frozen['protocol_sha256'] or p['parameters']!=frozen['parameters'] or p['prompt_sha256']!=frozen['prompt_sha256']:raise ValueError('V12 policy changed')
    if p['initializations']!=input_registration(root) or p['initial_registry_sha256']!=digest(p['initializations']):raise ValueError('Initialization registry drift')
    if p['paired_runs']!=[list(x) for x in PAIRS] or (p['initial'],p['queries'])!=(4,16):raise ValueError('Registered scope changed')
    if any(p[k]!=0 for k in ('seed','trial_seed','tie_seed','numerical_seed','menu_seed')):raise ValueError('Native frozen random streams changed')
    if verify_controls:
        for name,sha in p['protected_controls_sha256'].items():
            if old.sha(root/name)!=sha:raise ValueError('Historical artifact drift: '+name)
    return p


@contextmanager
def redirected(root,task,seed,protocol,*,redirect_gp=True):
    """Process-local binding only; the loop, GP class and decision functions remain native."""
    root=Path(root);base=pair_base(root,task,seed);pool=registered_pool(root,task,seed,protocol)
    native_adapter=v12.gpmod.GPAdapter
    def loader(r,name,*,seed=0):
        if Path(r).resolve()!=root.resolve() or name!=task or seed!=0:raise ValueError('Unexpected frozen loader request')
        return pool
    def adapter(p,path,seed=0):
        if p is not pool or Path(path)!=base/'cache'/task or seed!=0:raise ValueError('Unexpected native GP request')
        return native_adapter(p,directory(root)/'cache'/task,seed=0)
    with ExitStack() as stack:
        for module in (v12,v9):
            stack.enter_context(patch.object(module,'directory',return_value=base))
            stack.enter_context(patch.object(module,'check',return_value=deepcopy(protocol)))
        stack.enter_context(patch.object(v12.gpmod,'load_task',side_effect=loader))
        if redirect_gp:stack.enter_context(patch.object(v12.gpmod,'GPAdapter',side_effect=adapter))
        yield pool


def preflight_one(root,task,seed):
    root=Path(root);p=check(root);base=directory(root)/'preflight'/task/f'seed_{seed}'
    with redirected(root,task,seed,p) as pool:
        for arm in ARMS:old.initial_commit(base/arm,pool,p,arm,'agent_initial_terra',create=True)
        history=[v12.gpmod.read_observation(root,pool,key) for key in pool.initial_ids]
        gp=v12.gpmod.GPAdapter(pool,pair_base(root,task,seed)/'cache'/task,seed=0)
        ctx=gp.context(history,1);gp.release();other=gp.context(history,1)
        for key in ('mu_y','sd_y','sd_predictive_y'):np.testing.assert_array_equal(ctx['stats'][key],other['stats'][key])
        np.testing.assert_array_equal(ctx['base_logei'],other['base_logei'])
        pre=dict(task=task,arm='memory',history=history,hypotheses=[],decisions=[],insights=[],edit_calls=[],category_menu_insertions=0)
        original,ordered,display,cat=v12.engine.menu(pool,pre,ctx,1,'memory')
        bo_order,bo_display=v9.engine.candidates(pool,ctx,1)
        np.testing.assert_array_equal(original,bo_order);np.testing.assert_array_equal(display,bo_display)
        np.savez_compressed(base/'01_GP.npz',mu_y=ctx['stats']['mu_y'],sd_y=ctx['stats']['sd_y'],sd_predictive_y=ctx['stats']['sd_predictive_y'],
            base_logei=ctx['base_logei'],available=ctx['available'],tie_priority=ctx['tie_priority'])
        result=dict(task=task,initialization_seed=seed,status='PASS_SHARED_INITIAL_GP_AND_MENU',initial_ids=list(pool.initial_ids),
            initial_history=history,GP_sha256=v12.engine.previous.gp_hash(ctx),ordered_ids=[pool.ids[int(j)] for j in original],
            counters=gp.counters,LLM_calls=0,post_initial_queries=0)
        gp.release();write(base/'receipt.json',result);return result


def preflight(root,workers=9):
    p=check(root,verify_controls=True)
    with ProcessPoolExecutor(max_workers=min(workers,9)) as ex:
        rows=list(ex.map(preflight_one,[str(root)]*9,[x[0] for x in PAIRS],[x[1] for x in PAIRS]))
    result=dict(status='PASS',protocol_sha256=p['protocol_sha256'],rows=rows,LLM_calls=0,new_queries=0,
        boundary='Only registered four initial outcomes revealed after durable commitments; shared native GP and first-menu equality; full run first-array equality audited afterward')
    write(Path(root)/'experiments/reports/knowledge_loop_20261101/v12_seeds_preflight.json',result);return result


def run_pair(root,task,seed):
    root=Path(root).resolve();p=check(root)
    if [task,seed] not in p['paired_runs']:raise ValueError('Unregistered initialization pair')
    pf=old.read(directory(root)/'preflight'/task/f'seed_{seed}/receipt.json')
    if pf['status']!='PASS_SHARED_INITIAL_GP_AND_MENU':raise ValueError('Matched preflight required')
    base=pair_base(root,task,seed);base.mkdir(parents=True,exist_ok=True)
    with FileLock(str(base/'pair.lock'),timeout=900):
        launch=base/'launch.json'
        if launch.exists():
            if old.read(launch)['protocol_sha256']!=p['protocol_sha256']:raise ValueError('Launch protocol changed')
        else:
            if (base/'requests').exists() or (base/'runs').exists():raise ValueError('Fresh pair must have empty response/trajectory namespaces')
            write(launch,dict(task=task,initialization_seed=seed,protocol_sha256=p['protocol_sha256'],started_utc=datetime.now(timezone.utc).isoformat(),
                LLM_cache_initially_empty=True,original_response_files_copied=0,public_seed_tag=False,initial_ids=pf['initial_ids']))
        results=[]
        with redirected(root,task,seed,p):
            for arm,runner in [('bo',v9.run_one),('memory',v12.run_one)]:
                row=dict(runner(root,task,arm),initialization_seed=seed)
                results.append(row);print('FINISH '+json.dumps(row,ensure_ascii=False),flush=True)
        check(root)
        write(base/'completion.json',dict(results=results,protocol_sha256=p['protocol_sha256'],finished_utc=datetime.now(timezone.utc).isoformat()))
        return results


def batch(root,workers=9):
    p=check(root,verify_controls=True);errors=[]
    with ProcessPoolExecutor(max_workers=min(workers,9)) as ex:
        futures={ex.submit(run_pair,str(root),task,seed):(task,seed) for task,seed in PAIRS}
        for future,key in futures.items():
            try:future.result()
            except Exception as error:
                errors.append(dict(task=key[0],initialization_seed=key[1],kind=type(error).__name__))
                print('PAIR_ERROR '+json.dumps(errors[-1]),flush=True)
    if errors:raise RuntimeError('Incomplete pairs; preserve prefixes and resume same namespaces: '+json.dumps(errors))


def status(root):
    rows=[]
    for task,seed in PAIRS:
        for arm in ARMS:
            path=run_path(root,task,seed,arm)/'state.json'
            if path.exists():row=(v9 if arm=='bo' else v12).summary(old.read(path))
            else:row=dict(task=task,arm=arm,status='NOT_STARTED',queries=0)
            rows.append(dict(row,initialization_seed=seed))
    return rows
