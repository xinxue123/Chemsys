"""Exploratory held-out follow-up: scopes from pure BO's own random first points.

Each stream keeps pure BO's first point and tie stream. It reuses that stream's sealed
pre-data hypotheses from h1, which were locked before any observation. Later refreshes
run live through the unchanged controller, and revision X is not used. Pure BO and r1
therefore differ only in the scopes.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import argparse
import json
from grounded_first_start import proposal
from hypothesis_scope_loop import study as frozen, knowledge as k
from . import heldout as h, study as v2

ARM = 'r1'


def random_prefix(task, seed, arm):
    """Commit pure BO's first point under the sealed pre-data hypotheses, then observe it."""
    pool = frozen.pool_for(task)
    scopes = k.Scopes(pool)
    folder = v2.folder_for(task, seed, arm)
    source = frozen.read(h.BASE/'runs/h1'/f's{seed}'/h.SHORT/'rounds/01_commit.json')
    start = frozen.read(h.BASE/'runs/bo'/f's{seed}'/h.SHORT/'rounds/01_commit.json')['decision']
    j = pool.ix[start['selected_id']]
    state = v2.header(task, seed, arm)
    state.update(history=[], decisions=[], notes=[], archive=[], calls=[], permission_events=[])
    work = deepcopy(source['working_knowledge'])
    decision = dict(deepcopy(source['decision']), selected_id=pool.ids[j], conditions=deepcopy(pool.points[j]),
                    LLM_preferred=None, LLM_projection=None, reason='random_first_with_blinded_hypotheses',
                    locked_forecasts=k.freeze_forecasts(pool, scopes, work['notes'], j, 1, None, None, {}))
    commit = dict(query=1, prestate_sha256=frozen.digest(state), decision=decision, working_knowledge=work)
    commit['commit_sha256'] = frozen.digest(commit)
    frozen.write(folder/'rounds/01_commit.json', commit)
    frozen.apply_outcome(pool, folder, state, commit)
    return state


def configure():
    """Point the registered run loop at this arm, in each worker process."""
    frozen.initialize()
    frozen.pool_for = h.pool_for
    frozen.locked_first = lambda task: None  # Read only before the first observation, and r1 starts after it.
    proposal.background = h.background
    v2.BASE = h.BASE  # Shares pure BO's fit cache, so identical prefixes reuse identical fits.
    v2.folder_for = lambda t, s, a: h.BASE/'runs'/a/f's{s}'/h.SHORT
    v2.LIVE_ARMS = v2.LIVE_ARMS + (ARM,)
    v2.EVIDENCE_FIRST_ARMS = v2.EVIDENCE_FIRST_ARMS + (ARM,)  # No sealed query-4 response exists for this arm.
    v2.prefix_from_first = random_prefix


def run_stream(seed):
    return v2.run_arm(h.TASK, seed, ARM)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='*', type=int, default=list(range(8)))
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=min(8, args.workers), initializer=configure) as executor:
        for future in as_completed([executor.submit(run_stream, s) for s in args.seeds]):
            print(json.dumps(future.result()), flush=True)
