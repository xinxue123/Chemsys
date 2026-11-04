"""Only redirect V12 I/O; execute its original unmodified run_one and policy."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
from unittest.mock import patch
import json,shutil,subprocess
import numpy as np
from filelock import FileLock
from chem.benchmarks.knowledge_llm import digest,write
from reasoning_memory_v12 import study as v12

old=v12.old
VARIANT='v12_llm_repeats'
TASKS=tuple(v12.TASKS)
REPLICATES=('r01','r02','r03')
RUNS=tuple((task,rep) for task in TASKS for rep in REPLICATES)
DESIGN='docs/v12_llm_repeats_protocol_20261101.txt'


def directory(root):return Path(root)/'experiments/runs/knowledge_loop_20261101'/VARIANT


def replicate_base(root,task,rep):
    if rep not in REPLICATES or not task or any(c in task for c in '/\\:'):raise ValueError('Invalid replication path')
    return directory(root)/'replicates'/task/rep


def run_path(root,task,rep):return replicate_base(root,task,rep)/'runs'/f'{task}_memory'


def sources(root):
    root=Path(root);files=v12.sources(root)
    paths=list((root/'code/experiment_tools/v12_llm_repeats').rglob('*.py'))+[root/'code/experiment_tools/run_v12_llm_repeats.py']
    files.update({p.relative_to(root).as_posix():old.sha(p) for p in paths});return files


def protected(root):
    root=Path(root);files=v12.protected(root)
    # The original V12 cache is also frozen: all new writes use a copied cache.
    paths=[p for p in v12.directory(root).rglob('*') if p.is_file() and p.suffix!='.lock']
    paths += [p for p in (root/'experiments/reports/knowledge_loop_20261101').glob('v12_*') if p.is_file() and not p.name.startswith('v12_repeats_')]
    paths += [root/'docs/reasoning_memory_v12_protocol_20261101.txt',root/'docs/reasoning_memory_v12_results_20261101.txt']
    files.update({p.relative_to(root).as_posix():old.sha(p) for p in paths});return files


def copy_gp_cache(root):
    count=0;size=0
    for task in TASKS:
        for source in (v12.directory(root)/'cache'/task).rglob('*'):
            if not source.is_file() or source.suffix not in ('.pt','.npz','.json'):continue
            dest=directory(root)/'cache'/source.relative_to(v12.directory(root)/'cache')
            if not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest);count+=1;size+=source.stat().st_size
            elif old.sha(dest)!=old.sha(source):raise ValueError('Conflicting inherited GP cache')
    return dict(copied_files=count,copied_bytes=size,LLM_responses_copied=0)


def preflight_one(root,task):
    root=Path(root);pool=v12.gpmod.load_task(root,task,seed=0)
    original=old.read(v12.directory(root)/'runs'/f'{task}_memory/rounds/01_commit.json')
    gp=v12.gpmod.GPAdapter(pool,directory(root)/'cache'/task,seed=0);ctx=gp.context(original['prestate']['history'],1)
    with np.load(v12.v9.directory(root)/'runs'/f'{task}_bo/rounds/01_GP.npz',allow_pickle=False) as z:
        for key in ('mu_y','sd_y','sd_predictive_y'):np.testing.assert_array_equal(z[key],ctx['stats'][key])
        np.testing.assert_array_equal(z['base_logei'],ctx['base_logei'])
    orig,_,display,_=v12.engine.menu(pool,original['prestate'],ctx,1,'memory');ids=[pool.ids[int(j)] for j in display]
    event=v12.engine.edit_event(pool,original['prestate'],orig,1)
    public=v12.llm.memory_public(pool,original['prestate'],ids,1,event)
    context=dict(system=v12.llm.MEMORY_SYSTEM,public=public,model=v12.api.MODEL,reasoning_effort=v12.api.REASONING_EFFORT)
    if digest(context)!=original['memory_context_sha256']:raise ValueError('Frozen initial prompt changed')
    result=dict(task=task,status='PASS_EXACT_GP_AND_INITIAL_PROMPT',initial_prompt_sha256=digest(context),
        initial_ids=list(pool.initial_ids),GP_sha256=v12.engine.previous.gp_hash(ctx),counters=gp.counters)
    gp.release();return result


def preflight(root,workers=3):
    v12.check(root);copied=copy_gp_cache(root)
    with ProcessPoolExecutor(max_workers=min(workers,3)) as ex:rows=list(ex.map(preflight_one,[str(root)]*3,TASKS))
    value=dict(status='PASS',rows=rows,cache_copy=copied,LLM_calls=0,new_queries=0)
    write(Path(root)/'experiments/reports/knowledge_loop_20261101/v12_repeats_preflight.json',value);return value


def prepare(root):
    root=Path(root).resolve();base=directory(root)
    if (base/'protocol.json').exists():return check(root)
    if (base/'replicates').exists():raise ValueError('Register before creating any replica')
    frozen=v12.check(root);pf=old.read(root/'experiments/reports/knowledge_loop_20261101/v12_repeats_preflight.json')
    if pf['status']!='PASS' or {r['task'] for r in pf['rows']}!=set(TASKS):raise ValueError('Three matched preflights required')
    p=dict(variant=VARIANT,created_utc=datetime.now(timezone.utc).isoformat(),tasks=list(TASKS),replicates=list(REPLICATES),
        runs=[[task,'memory'] for task in TASKS],replication_runs=[list(r) for r in RUNS],arms=['memory'],
        seed=0,trial_seed=0,tie_seed=0,numerical_seed=0,initial=4,queries=16,model=frozen['model'],reasoning_effort=frozen['reasoning_effort'],
        frozen_V12_protocol_sha256=frozen['protocol_sha256'],frozen_V12_core_commit=frozen['code_milestone'],
        parameters=deepcopy(frozen['parameters']),prompt_sha256=deepcopy(frozen['prompt_sha256']),inputs=deepcopy(frozen['inputs']),
        source_sha256=sources(root),protected_controls_sha256=protected(root),design_path=DESIGN,design_sha256=old.sha(root/DESIGN),
        code_milestone=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        response_cache_policy='isolated per task and replicate; fresh initial requests; resume only within its own namespace',
        GP_cache_policy='shared new copied native cache with native prefix locks; original caches read-only',
        policy_execution='unmodified reasoning_memory_v12.study.run_one; only directory, protocol envelope and GP cache location redirected',
        original_V12_in_new_sample=False,LLM_parameters_changed=False,replicate_ID_in_prompt=False,
        maximum_new_queries=144,maximum_logical_LLM_calls=180,maximum_schema_attempts_per_context=3,
        metrics=['per-run SearchAUC/final_best/first_best_query and deltas versus frozen matched BO',
            'three-repeat sample mean, sample SD, min/max and observed win/tie/loss counts',
            'per-round distinct selected points, pairwise disagreement, first divergence and final-set Jaccard',
            'identical-context response/choice differences and same-history GP/menu consistency'],
        inference_boundary='Conditional LLM-service variability at fixed seed0, initialization and model alias. n=3 per task; no multi-initialization or new-task generalization, no noninferiority/significance declaration.',
        no_performance_based_rerolls=True)
    p['protocol_sha256']=digest(p);write(base/'protocol.json',p)
    snapshot=root/'runtime/knowledge_loop/snapshots'/VARIANT
    for name,sha in p['source_sha256'].items():
        dest=snapshot/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,dest)
        if old.sha(dest)!=sha:raise ValueError('Snapshot drift')
    (snapshot/DESIGN).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/DESIGN,snapshot/DESIGN)
    shutil.copy2(base/'protocol.json',snapshot/'protocol.json');return check(root)


def check(root):
    root=Path(root);p=old.read(directory(root)/'protocol.json');frozen=v12.check(root)
    if p['protocol_sha256']!=digest({k:v for k,v in p.items() if k!='protocol_sha256'}):raise ValueError('Replication protocol drift')
    if p['source_sha256']!=sources(root) or p['design_sha256']!=old.sha(root/p['design_path']):raise ValueError('Frozen implementation drift')
    if p['frozen_V12_protocol_sha256']!=frozen['protocol_sha256'] or p['parameters']!=frozen['parameters'] or p['prompt_sha256']!=frozen['prompt_sha256']:raise ValueError('V12 policy changed')
    if p['replication_runs']!=[list(r) for r in RUNS] or p['initial']!=4 or p['queries']!=16 or p['seed']!=0:raise ValueError('Replication scope drift')
    return p


@contextmanager
def redirected_v12(root,task,rep,protocol,*,redirect_gp=True):
    """Per-process I/O adapter, with no new public prompt or selection code."""
    root=Path(root);base=replicate_base(root,task,rep);native_adapter=v12.gpmod.GPAdapter
    def adapter(pool,path,seed=0):
        if Path(path)!=base/'cache'/task or pool.task!=task or seed!=0:raise ValueError('Unexpected GP I/O request')
        return native_adapter(pool,directory(root)/'cache'/task,seed=seed)
    with patch.object(v12,'directory',return_value=base),patch.object(v12,'check',return_value=deepcopy(protocol)):
        if redirect_gp:
            with patch.object(v12.gpmod,'GPAdapter',side_effect=adapter):yield
        else:yield


def run_one(root,task,rep):
    root=Path(root).resolve();p=check(root)
    if [task,rep] not in p['replication_runs']:raise ValueError('Unregistered replicate')
    base=replicate_base(root,task,rep);base.mkdir(parents=True,exist_ok=True)
    with FileLock(str(base/'replica.lock'),timeout=900):
        marker=base/'launch.json'
        if marker.exists():
            if old.read(marker)['protocol_sha256']!=p['protocol_sha256']:raise ValueError('Replica launch protocol changed')
        else:
            if (base/'requests').exists() or (base/'runs').exists():raise ValueError('Fresh replica must start with empty response and trajectory storage')
            write(marker,dict(task=task,replicate=rep,protocol_sha256=p['protocol_sha256'],
                started_utc=datetime.now(timezone.utc).isoformat(),LLM_cache_initially_empty=True,
                original_V12_response_files_copied=0,public_replicate_tag=False))
        with redirected_v12(root,task,rep,p):result=v12.run_one(root,task,'memory')
        check(root)
        result=dict(result,replicate=rep)
        write(base/'completion.json',dict(result=result,protocol_sha256=p['protocol_sha256'],finished_utc=datetime.now(timezone.utc).isoformat()))
        return result


def batch(root,workers=9):
    p=check(root)
    with ProcessPoolExecutor(max_workers=min(workers,9)) as ex:
        futures={ex.submit(run_one,str(root),task,rep):(task,rep) for task,rep in p['replication_runs']}
        for future in as_completed(futures):print('FINISH '+json.dumps(future.result(),ensure_ascii=False),flush=True)


def status(root):
    results=[]
    for task,rep in RUNS:
        path=run_path(root,task,rep)/'state.json'
        if path.exists():results.append(dict(v12.summary(old.read(path)),replicate=rep))
        else:results.append(dict(task=task,replicate=rep,status='NOT_STARTED',queries=0))
    return results
