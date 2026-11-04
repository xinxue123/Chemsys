"""External namespace and transport adapter; original scientific loops are untouched."""
from contextlib import contextmanager, ExitStack, redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from unittest.mock import patch
import json
import shutil

from filelock import FileLock
from paper_v12 import study as original

from . import BACKEND, CONTEXT_POLICY, MODEL, REASONING_EFFORT
from . import transport

JOBS = tuple(j for j in original.JOBS if j[2] != 'bo') + tuple(
    (t, s, 'memory') for t in original.FOCUS for s in (0, 1, 2, 3))
SHORT = original.SHORT
v12, v9, old = original.v12, original.v9, original.old


def directory(root):
    return (Path(root).resolve() / 'experiments/runs/paper_v12_subagent').resolve()


def base(root, task, seed, arm):
    if type(seed) is not int or (task, seed, arm) not in JOBS:
        raise ValueError('Unregistered subagent job')
    path = directory(root) / f's{seed}' / SHORT[task] / arm
    if not path.resolve().is_relative_to(directory(root)):
        raise ValueError('Subagent job escapes namespace')
    return path


def run_path(root, task, seed, arm):
    return base(root, task, seed, arm) / 'runs' / f'{task}_{original.native_arm(arm)}'


def sources(root):
    root = Path(root).resolve()
    return {p.relative_to(root).as_posix(): transport.file_sha(p)
            for p in sorted((root / 'code/experiment_tools/paper_v12_subagent').rglob('*.py'))}


def check(root, verify_controls=False):
    root = Path(root).resolve()
    p = transport.read(directory(root) / 'protocol.json')
    frozen = original.check(root, verify_controls=verify_controls)
    if p.get('protocol_sha256') != transport.digest({k: v for k, v in p.items() if k != 'protocol_sha256'}):
        raise ValueError('Subagent protocol digest changed')
    if (p.get('variant') != 'paper_v12_subagent' or p.get('backend') != BACKEND
            or p.get('original_protocol_sha256') != frozen['protocol_sha256']
            or p.get('model') != MODEL or p.get('reasoning_effort') != REASONING_EFFORT
            or p.get('context_policy') != CONTEXT_POLICY or p.get('fork_turns') != 'none'
            or p.get('jobs') != [list(j) for j in JOBS]):
        raise ValueError('Registered transport or 68-job matrix changed')
    if p.get('adapter_source_sha256') != sources(root):
        raise ValueError('Registered external adapter source changed')
    design = (root / p['design_path']).resolve()
    if not design.is_relative_to(root) or transport.file_sha(design) != p['design_sha256']:
        raise ValueError('Registered subagent design changed')
    instructions = (root / p['decision_instructions_path']).resolve()
    if (not instructions.is_relative_to(root)
            or transport.file_sha(instructions) != p['decision_instructions_sha256']
            or p.get('public_exposure_policy') != 'opaque_exact_system_public_packet'):
        raise ValueError('Registered decision instructions or exposure policy changed')
    for key in ('initializations', 'inputs', 'initial', 'queries', 'seed', 'trial_seed',
                'tie_seed', 'numerical_seed', 'menu_seed', 'parameters', 'prompt_sha256'):
        if p.get(key) != frozen.get(key):
            raise ValueError('Frozen scientific setting changed: ' + key)
    if verify_controls:
        for name, expected in p.get('protected_controls_sha256', {}).items():
            if transport.file_sha(root / name) != expected:
                raise ValueError('Historical artifact drift: ' + name)
    return p


def pool_for(root, task, seed, p):
    return original.pool_for(root, task, seed, p)


def _cached_v9_extract(root, raw):
    # Same official function/n=1 and returned list as V9; compile once per process.
    with redirect_stdout(StringIO()):
        return v12.llm.compiled_extractor(str(Path(root).resolve()))(None, json.dumps(raw), n=1)


@contextmanager
def redirected(root, task, seed, arm, p, client):
    root = Path(root).resolve()
    b = base(root, task, seed, arm)
    pool = pool_for(root, task, seed, p)
    native_gp = v12.gpmod.GPAdapter

    def loader(r, name, *, seed=0):
        if Path(r).resolve() != root or name != task or seed != 0:
            raise ValueError('Unexpected frozen input loader')
        return pool

    def adapter(obj, path, seed=0):
        if obj is not pool or seed != 0:
            raise ValueError('Unexpected frozen GP input')
        return native_gp(obj, directory(root) / 'cache' / task, seed=0)

    with ExitStack() as stack:
        for module in (v12, v9):
            stack.enter_context(patch.object(module, 'directory', return_value=b))
            stack.enter_context(patch.object(module, 'check', return_value=deepcopy(p)))
        stack.enter_context(patch.object(v12.gpmod, 'load_task', side_effect=loader))
        stack.enter_context(patch.object(v12.gpmod, 'GPAdapter', side_effect=adapter))
        stack.enter_context(patch.object(v12.api, 'load_config', return_value=None))
        stack.enter_context(patch.object(v12.api, 'APIClient', return_value=client))
        stack.enter_context(patch.object(v12.llm, 'request', side_effect=transport.request))
        stack.enter_context(patch.object(v9.llm, 'request', side_effect=transport.request))
        stack.enter_context(patch.object(v9.llm, 'official_extract', side_effect=_cached_v9_extract))
        if arm == 'no_kf':
            stack.enter_context(patch.object(v12.engine.kf, 'update', side_effect=original.no_kf_update))
        if arm == 'no_refresh':
            stack.enter_context(patch.object(v12.engine, 'edit_event', side_effect=original.no_refresh_event))
        yield pool


def copy_cache(root):
    """Copy bytes only; never evaluate or fit a GP or copy LLM response files."""
    root = Path(root).resolve()
    target_base = directory(root) / 'cache'
    target_base.mkdir(parents=True, exist_ok=True)
    count = 0
    with FileLock(str(target_base / '.copy.lock'), timeout=900):
        sources_ = (original.directory(root) / 'cache', original.seeds.directory(root) / 'cache',
                    v12.directory(root) / 'cache', v12.v11.directory(root) / 'cache', v9.directory(root) / 'cache')
        for source_base in sources_:
            for source in source_base.rglob('*'):
                if not source.is_file() or source.suffix not in ('.json', '.npz', '.pt'):
                    continue
                target = target_base / source.relative_to(source_base)
                if target.exists():
                    if transport.file_sha(target) != transport.file_sha(source):
                        raise ValueError('Native GP prefix cache conflict')
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
                    count += 1
    return count


def run_one(root, task, seed, arm, *, wait_seconds=-1):
    root = Path(root).resolve()
    p = check(root)
    if [task, seed, arm] not in p['jobs']:
        raise ValueError('Job absent from registered protocol')
    b = base(root, task, seed, arm)
    pf_path = original.directory(root) / 'preflight' / f's{seed}' / SHORT[task] / 'receipt.json'
    if transport.read(pf_path)['status'] != 'PASS':
        raise ValueError('Original frozen initial GP/menu preflight must pass')
    b.mkdir(parents=True, exist_ok=True)
    with FileLock(str(b / 'run.lock'), timeout=1):
        launch = b / 'launch.json'
        if launch.exists():
            if transport.read(launch)['protocol_sha256'] != p['protocol_sha256']:
                raise ValueError('Resumed launch protocol changed')
        else:
            if any((b / name).exists() for name in ('runs', 'requests', 'queue')):
                raise ValueError('Fresh trajectory namespace required before launch')
            transport.write_once(launch, dict(task=task, initialization_seed=seed, arm=arm,
                native_arm=original.native_arm(arm), protocol_sha256=p['protocol_sha256'], backend=BACKEND,
                context_policy=CONTEXT_POLICY, started_utc=datetime.now(timezone.utc).isoformat(),
                LLM_cache_initially_empty=True, response_files_copied=0,
                preflight_receipt_sha256=transport.file_sha(pf_path), torch_threads=1))
        if (b / 'completion.json').exists():
            return transport.read(b / 'completion.json')['result']
        # Warm the inherited native operators and exact official extractor once.
        v12.gpmod.warm()
        v12.llm.compiled_extractor(str(root))
        client = transport.FileQueueClient(b, (task, seed, arm), wait_seconds=wait_seconds)
        with redirected(root, task, seed, arm, p, client):
            runner = v9.run_one if arm == 'reasoning_bo' else v12.run_one
            result = dict(runner(root, task, original.native_arm(arm)), initialization_seed=seed,
                          study_arm=arm, backend=BACKEND, usage_available=False)
        check(root)
        transport.write_once(b / 'completion.json', dict(result=result, backend=BACKEND,
            protocol_sha256=p['protocol_sha256'], finished_utc=datetime.now(timezone.utc).isoformat()))
        return result


def pending(root, job=None):
    """List queue metadata only, optionally restricted to one worker's trajectory."""
    jobs = [tuple(job)] if job is not None else JOBS
    rows = []
    for task, seed, arm in jobs:
        b = base(root, task, seed, arm)
        for path in sorted((b / 'queue').glob('*/attempt*/request.json')):
            if not (path.parent / 'provenance.json').exists():
                value = transport.request_record(path)
                rows.append(dict(job=[task, seed, arm], request=str(path),
                                 request_sha256=value['request_sha256'], attempt=value['attempt'],
                                 assigned=(path.parent / 'assignment.json').exists()))
    return rows


def status(root):
    rows = []
    for task, seed, arm in JOBS:
        b = base(root, task, seed, arm)
        sf = run_path(root, task, seed, arm) / 'state.json'
        state = transport.read(sf) if sf.exists() else {}
        rows.append(dict(job=[task, seed, arm], status=state.get('status', 'NOT_STARTED'),
                         queries=len(state.get('decisions', [])), backend=BACKEND,
                         pending=len(pending(root, (task, seed, arm)))))
    return rows

