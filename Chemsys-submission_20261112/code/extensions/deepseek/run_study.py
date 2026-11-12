"""Isolated model extension of the original Table-4 implementation.

Only relocation, request backend and model-specific effort differ. All selection,
GP, hypothesis, KF, permission, refresh and validation code comes from extrnal.
"""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / 'evidence/deepseek'
SOURCE = ROOT / 'code'
import sys
sys.dont_write_bytecode = True
sys.path[:0] = [str(SOURCE / 'chemist_system'), str(SOURCE / 'experiment_tools'),
                str(ROOT / 'runtime/tools/rdkit_runtime')]

# Override the relocated root before downstream modules copy this constant.
from kernel_contrast_qlogei import study as old
old.ROOT = ROOT
from chemical_hypothesis_bo import background
background.ROOT = ROOT
from hypothesis_scope_loop import study as frozen
from hypothesis_scope_multiseed import study as multi
from hypothesis_scope_v2 import study as v2, heldout as h, heldout_named as hn, heldout_lgbo as hl, cpa, shared_start as shared, transport
from grounded_first_start import proposal
from grounded_baselines import study as gb
import deepseek_transport as backend
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from copy import deepcopy
import importlib.metadata
import json
from unittest.mock import patch

read, write = frozen.read, frozen.write
STARTS = {(r['task'], r['pair']): r for r in read(HERE / 'frozen/shared_starts.json')}
QUALIFICATION = HERE / 'qualification_restored'


def configure():
    """One warm-up per persistent worker, outside the experimental loop."""
    for module in (old, frozen, multi, v2, h, hn, cpa, gb):
        if hasattr(module, 'ROOT'):
            module.ROOT = ROOT
    h.INPUT = HERE / 'frozen/heldout/input'
    cpa.INPUT = HERE / 'frozen/cpa/input'
    transport.BASE = backend.BASE
    transport.request = backend.request
    shared.NATIVE_REQUEST = backend.request
    cpa.NATIVE_REQUEST = backend.request
    # Backend tokenization/reasoning differs; these are resource guardrails,
    # never the controller's unchanged four-call limit or 20-query budget.
    shared.MAXIMUM_TOKENS, shared.MAXIMUM_ATTEMPTS = 48_000_000, 6000
    cpa.MAXIMUM_TOKENS, cpa.MAXIMUM_ATTEMPTS = 16_000_000, 1000
    shared.OUT = HERE / 'new/shared_start'
    cpa.BASE = HERE / 'new/cpa'
    hn.NAMED = HERE / 'new/heldout_named'
    shared.point_of = lambda t: STARTS['aryl', t]['candidate_id']
    cpa.point_of = lambda t: STARTS['cpa', t]['candidate_id']
    frozen.initialize()


def task_pool(short):
    return h.pool_for() if short == 'aryl' else cpa.pool_for()


def run_bo(short, pair):
    """Qualify against the exact archived shared-start BO, with a fresh fit cache."""
    configure()
    task = h.TASK if short == 'aryl' else cpa.TASK
    pool = task_pool(short)
    frozen.pool_for = lambda t: pool
    first = dict(candidate_id=STARTS[short, pair]['candidate_id'], rationale='fixed by design', assumptions=[])
    first['conditions'] = deepcopy(pool.points[pool.ix[first['candidate_id']]])
    frozen.locked_first = lambda t: first
    multi.BASE = QUALIFICATION
    multi.folder_for = lambda t, s, a: QUALIFICATION / 'runs' / short / f's{s}'
    p = dict(protocol_sha256='deepseek_table4_relocation_only', initial_priors={task: dict(first_input=first)})
    multi.run_control(task, pair, 'first', p)
    new = read(multi.folder_for(task, pair, 'first') / 'state.json')
    saved = read(HERE / 'frozen' / STARTS[short, pair]['baseline_state'])
    same = [a == b for a, b in zip(new['history'], saved['history'])]
    result = dict(task=short, pair=pair, exact_points=sum(same), of=20,
                  status='PASS' if all(same) and len(same) == 20 else 'FAIL',
                  first_mismatch=next((i+1 for i, ok in enumerate(same) if not ok), None))
    write(QUALIFICATION / 'results' / short / f's{pair}.json', result, replace=True)
    return result


def run_draw(short, pair):
    """Fresh model's zero-data proposal and hypotheses; never access an outcome."""
    configure()
    if short == 'cpa':
        cpa.configure()
        return cpa.run_draw(pair)
    h.background = hn.card
    proposal.background = hn.card
    pool = h.pool_for()
    space = f'aryl_initial/s{pair}'
    draw_path = HERE / 'new/heldout_named/draws' / f's{pair}.json'
    if draw_path.exists():
        locked = read(draw_path)
    else:
        result = backend.request(space, 'propose', proposal.PROPOSE_SYSTEM, proposal.public(pool),
                                 lambda raw: proposal.validate_proposal(raw, pool))
        candidate, choice = result['plan']['exact_candidate_id'], result['plan']['choice']
        if candidate is None:
            result2 = backend.request(space, 'confirm', proposal.CONFIRM_SYSTEM, proposal.confirmation(pool, result),
                                      lambda raw: proposal.validate_confirmation(raw, pool, result['plan']['neighbors']))
            candidate, choice = result2['plan']['choice']['candidate_id'], result2['plan']['choice']
        locked = dict(candidate_id=candidate, conditions=deepcopy(pool.points[pool.ix[candidate]]),
                      rationale=choice['rationale'], assumptions=deepcopy(choice['key_assumptions']))
        write(draw_path, locked)
    target = hn.NAMED / 'runs/h1' / f's{pair}' / h.SHORT
    if (target / 'rounds/01_commit.json').exists():
        return dict(task=short, pair=pair, stage='draw', status='COMPLETE')
    control = dict(protocol_sha256='deepseek_table4', initial_priors={h.TASK: dict(first_input=locked)})
    def stop(*a, **kw):
        raise cpa.StopAfterCommit
    with ExitStack() as stack:
        for name, value in (('check', lambda *a, **kw: control), ('BASE', hn.NAMED),
                            ('locked_first', lambda t: locked), ('folder_for', lambda t: target),
                            ('apply_outcome', stop)):
            stack.enter_context(patch.object(frozen, name, value))
        stack.enter_context(patch.object(frozen.transport, 'request',
                                         lambda name, stage, system, public, validator: backend.request(f'aryl_initial/h1/s{pair}', stage, system, public, validator)))
        stack.enter_context(patch.object(frozen, 'pool_for', h.pool_for))
        try:
            frozen.run_one(h.TASK)
        except cpa.StopAfterCommit:
            pass
    assert (target / 'rounds/01_commit.json').exists() and not (target / 'state.json').exists()
    return dict(task=short, pair=pair, stage='draw', status='COMPLETE')


def run_controller(short, pair):
    configure()
    if short == 'cpa':
        cpa.configure()
        return cpa.run_controller(pair)
    shared.controller_worker()
    return shared.run_controller(pair)


def run_comparator(pair, arm):
    configure()
    shared.baseline_worker()
    return shared.run_baseline(pair, arm)


def environment():
    import platform
    return dict(python=sys.version, executable=sys.executable, OS=platform.platform(),
                packages={p: importlib.metadata.version(p) for p in
                          ('torch', 'botorch', 'gpytorch', 'numpy', 'scipy', 'pandas', 'filelock')},
                independent_processes=True, BLAS_threads_per_process=1, GP_device='CPU')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('environment', 'qualify', 'draws', 'controller', 'baselines'))
    p.add_argument('--tasks', nargs='+', default=['aryl', 'cpa'], choices=('aryl', 'cpa'))
    p.add_argument('--pairs', nargs='+', type=int, default=list(range(24)))
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--arms', nargs='+', default=['lgbo', 'rbo'], choices=('lgbo', 'rbo'))
    a = p.parse_args()
    if a.action == 'environment':
        configure()
        record = environment()
        write(HERE / 'environment.json', record, replace=True)
        print(json.dumps(record)); return
    if a.action == 'qualify':
        os.environ['CHEM_DEEPSEEK_OFFLINE'] = '1'
        jobs = [(run_bo, t, s) for t in a.tasks for s in a.pairs]
    else:
        assert not os.environ.get('CHEM_DEEPSEEK_OFFLINE')
        gate = read(HERE / 'matched_local_gate.json')
        assert gate['status'] == 'PASS_48_MATCHED_LOCAL_CONTROLS', 'No API before matched BO controls are audited'
        if a.action == 'draws':
            jobs = [(run_draw, t, s) for t in a.tasks for s in a.pairs if t == 'cpa' or s < 8]
        elif a.action == 'controller':
            jobs = [(run_controller, t, s) for t in a.tasks for s in a.pairs]
        else:
            jobs = [(run_comparator, s, arm) for arm in a.arms for s in a.pairs]
    with ProcessPoolExecutor(max_workers=a.workers) as executor:
        futures = [executor.submit(fn, *args) for fn, *args in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
            except Exception:
                for f in futures:
                    f.cancel()
                raise
            print(json.dumps(dict(completed=i, total=len(jobs), result=result)), flush=True)
    if a.action == 'qualify':
        results = [read(q) for q in (QUALIFICATION / 'results').glob('**/s*.json')]
        exact = sum(r['status'] == 'PASS' for r in results)
        gate = dict(status='PASS_EXACT_48_TRAJECTORIES' if len(results) == exact == 48 else 'QUALIFICATION_INCOMPLETE_OR_FAILED',
                    trajectories=len(results), exact=exact, rows=results)
        write(QUALIFICATION / 'gate.json', gate, replace=True)
        print(json.dumps(gate), flush=True)


if __name__ == '__main__':
    main()
