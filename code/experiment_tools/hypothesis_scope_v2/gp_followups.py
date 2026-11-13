"""GP-only follow-ups registered on 6 November 2026. No model calls are made.

Test F runs plain BO from first points fixed by design. Test S runs arm r0: pure BO's
first point plus sealed pre-data hypotheses, with no later refresh. GP, acquisition,
pools and budget are the registered ones. Oracle values choose the Test F design only,
and each run still observes its own outcomes in order.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import argparse
import json
import numpy as np
from grounded_first_start import proposal
from hypothesis_scope_loop import study as frozen, knowledge as k
from hypothesis_scope_multiseed import study as multi
from . import heldout as h, study as v2

BASE0 = v2.BASE
G = BASE0/'gp_followup'
NATIVE_POOL = frozen.pool_for
SHORT = frozen.SHORT
ARYL, DOYLE, PERERA = h.TASK, 'Doyle_Buchwald_Hartwig', 'Perera_Suzuki'
LLM_POINTS = ('a00102', 'a00493', 'a00499')
SEED = 20261106
read, write = frozen.read, frozen.write


def pool_of(task):
    return h.pool_for() if task == ARYL else NATIVE_POOL(task)


def values_of(pool):
    with np.load(pool.source_path.with_name('oracle.npz'), allow_pickle=False) as oracle:
        return np.asarray(oracle['raw_values'] if 'raw_values' in oracle else oracle['values']*pool.scale, dtype=float)


def level_means(pool, y, axis):
    levels = sorted({p[axis] for p in pool.points})
    return {v: float(np.mean([a for p, a in zip(pool.points, y) if p[axis] == v])) for v in levels}


def split(pool, y, failed, axis, best):
    inside = [j for j in np.flatnonzero(failed) if pool.points[j][axis] == best]
    outside = [j for j in np.flatnonzero(failed) if pool.points[j][axis] != best]
    return inside, outside


def design():
    path = G/'design.json'
    if path.exists():
        return read(path)
    rng = np.random.default_rng(SEED)
    plan = {}
    for task, count in ((ARYL, 30), (DOYLE, 20), (PERERA, 20)):
        pool = pool_of(task)
        y = values_of(pool)
        failed = y <= (2. if task == ARYL else np.quantile(y, .2))
        if task == ARYL:
            axis = 'Solvent_SMILES'
        else:
            options = []
            for a in pool.axes:
                means = level_means(pool, y, a)
                best = max(means, key=means.get)
                inside, outside = split(pool, y, failed, a, best)
                if len(inside) >= count and len(outside) >= count:
                    others = [v for level, v in means.items() if level != best]
                    options.append((means[best] - float(np.mean(others)), a))
            axis = max(options)[1]
        means = level_means(pool, y, axis)
        best = max(means, key=means.get)
        inside, outside = split(pool, y, failed, axis, best)
        pick = lambda js: sorted(pool.ids[j] for j in rng.choice(js, count, replace=False))
        plan[task] = dict(short=SHORT[task], axis=axis, best_level=best, level_means=means, failed_count=int(failed.sum()),
                          failed_inside=len(inside), failed_outside=len(outside), inside=pick(inside), outside=pick(outside), streams=[0, 1])
    plan[ARYL]['llm_points'] = list(LLM_POINTS)
    plan[ARYL]['llm_streams'] = list(range(8))
    write(path, plan)
    return plan


def configure():
    frozen.initialize()
    proposal.background = h.background
    frozen.pool_for = pool_of
    multi.BASE = G
    v2.BASE = G  # One fit cache for every follow-up, so identical prefixes reuse identical fits.
    v2.folder_for = lambda t, s, a: G/a/SHORT[t]/f's{s}'
    v2.EVIDENCE_FIRST_ARMS = v2.EVIDENCE_FIRST_ARMS + ('r0', 'p0', 'p0g')  # No sealed query-4 response exists for these arms.
    v2.prefix_from_first = lambda t, s, a: sealed_prefix(t, s, a) if a == 'r0' else llm_prefix(t, s, a)


def run_fixed_first(task, first_id, stream):
    """Plain BO after a first point fixed by design, through the registered control loop."""
    pool = pool_of(task)
    locked = dict(candidate_id=first_id, conditions=deepcopy(pool.points[pool.ix[first_id]]), rationale='fixed by design', assumptions=[])
    frozen.locked_first = lambda t: locked
    multi.folder_for = lambda t, s, a: G/'first_point'/SHORT[t]/first_id/f's{s}'
    control = dict(protocol_sha256='gp_followup_first_point', initial_priors={task: dict(first_input=locked)})
    return dict(multi.run_control(task, stream, 'first', control), first=first_id)


def run_extra_bo(stream):
    """Pure BO on held-out streams 8-23."""
    multi.folder_for = lambda t, s, a: G/'bo_extra'/SHORT[t]/f's{s}'
    return multi.run_control(ARYL, stream, 'bo', dict(protocol_sha256='gp_followup_bo', initial_priors={}))


def sources(task, seed):
    """Sealed query-1 commit with the hypotheses, and pure BO's first decision for this stream."""
    if task == ARYL:
        hypotheses = read(h.BASE/'runs/h1'/f's{seed % 8}'/h.SHORT/'rounds/01_commit.json')
        folder = h.BASE/'runs/bo'/f's{seed}'/h.SHORT if seed < 8 else G/'bo_extra'/h.SHORT/f's{seed}'
        return hypotheses, read(folder/'rounds/01_commit.json')['decision']
    hypotheses = read(v2.source_folder(task, seed, 'hyp')/'rounds/01_commit.json')
    folder = BASE0/'reproduction/runs'/f's{seed}'/(SHORT[task] + '_bo')
    return hypotheses, read(folder/'rounds/01_commit.json')['decision']


def sealed_prefix(task, seed, arm):
    """Commit pure BO's first point under the sealed pre-data hypotheses, then observe it."""
    pool = frozen.pool_for(task)
    scopes = k.Scopes(pool)
    folder = v2.folder_for(task, seed, arm)
    source, start = sources(task, seed)
    j = pool.ix[start['selected_id']]
    state = v2.header(task, seed, arm)
    state.update(history=[], decisions=[], notes=[], archive=[], calls=[], permission_events=[])
    work = deepcopy(source['working_knowledge'])
    decision = dict(deepcopy(source['decision']), selected_id=pool.ids[j], conditions=deepcopy(pool.points[j]),
                    LLM_preferred=None, LLM_projection=None, reason='random_first_with_blinded_hypotheses',
                    locked_forecasts=k.freeze_forecasts(pool, scopes, work['notes'], j, 1, None, None, {}))
    commit = dict(query=1, prestate_sha256=frozen.digest(state), decision=decision, working_knowledge=work)
    commit['commit_sha256'] = frozen.digest(commit)
    write(folder/'rounds/01_commit.json', commit)
    frozen.apply_outcome(pool, folder, state, commit)
    return state


def llm_prefix(task, seed, arm):
    """Commit the LLM's first point under the sealed pre-data hypotheses, then observe it."""
    pool = frozen.pool_for(task)
    scopes = k.Scopes(pool)
    folder = v2.folder_for(task, seed, arm)
    source, _ = sources(task, seed)
    j = pool.ix[source['decision']['selected_id']]
    state = v2.header(task, seed, arm)
    state.update(history=[], decisions=[], notes=[], archive=[], calls=[], permission_events=[])
    work = deepcopy(source['working_knowledge'])
    decision = dict(deepcopy(source['decision']), LLM_preferred=None, LLM_projection=None,
                    locked_forecasts=k.freeze_forecasts(pool, scopes, work['notes'], j, 1, None, None, {}))
    commit = dict(query=1, prestate_sha256=frozen.digest(state), decision=decision, working_knowledge=work)
    commit['commit_sha256'] = frozen.digest(commit)
    write(folder/'rounds/01_commit.json', commit)
    frozen.apply_outcome(pool, folder, state, commit)
    return state


def run_r0(task, seed):
    return v2.run_arm(task, seed, 'r0')


NATIVE_GATE = (k.eligible, k.pause_for_recommendation)


def run_gate(task, seed, arm):
    """Arm p0 keeps the permission gate. Arm p0g lets every installed hypothesis constrain."""
    k.eligible, k.pause_for_recommendation = NATIVE_GATE if arm == 'p0' else (lambda note, inc: True, lambda notes, inc: (deepcopy(notes), []))
    return v2.run_arm(task, seed, arm)


def jobs(plan):
    first = []
    for task, row in plan.items():
        for point in row['inside'] + row['outside']:
            first += [(run_fixed_first, task, point, s) for s in row['streams']]
    first += [(run_fixed_first, ARYL, point, s) for point in LLM_POINTS for s in range(8)]
    extra = [(run_extra_bo, s) for s in range(8, 24)]
    scopes = [(run_r0, t, s) for t in frozen.native.TASKS for s in range(4)] + [(run_r0, ARYL, s) for s in range(8)]
    late = [(run_r0, ARYL, s) for s in range(8, 24)]
    return extra + scopes + first, late


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('design', 'time', 'run', 'gate'))
    parser.add_argument('--workers', type=int, default=10)
    args = parser.parse_args()
    G.mkdir(parents=True, exist_ok=True)
    frozen.pool_for = pool_of
    plan = design()
    if args.action == 'design':
        for task, row in plan.items():
            print(json.dumps({key: row[key] for key in ('short', 'axis', 'failed_count', 'failed_inside', 'failed_outside')}))
            print('  best', row['best_level'][:50], '| means', {a[:24]: round(b, 1) for a, b in row['level_means'].items()})
        return
    first, late = jobs(plan)
    if args.action == 'gate':
        first = [(run_gate, t, s, a) for a in ('p0', 'p0g') for t in frozen.native.TASKS for s in range(4)]
        first += [(run_gate, ARYL, s, a) for a in ('p0', 'p0g') for s in range(8)]
        late = []
    if args.action == 'time':
        first, late = [(run_fixed_first, ARYL, plan[ARYL]['inside'][0], 0), (run_fixed_first, DOYLE, plan[DOYLE]['inside'][0], 0)], []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=configure) as executor:
        for group in (first, late):
            futures = [executor.submit(*job) for job in group]
            for done, future in enumerate(as_completed(futures), 1):
                print(json.dumps(dict(done=done, of=len(futures), **future.result())), flush=True)
