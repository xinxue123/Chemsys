"""Exploratory GP-only test: the LLM's first points on all 24 pure BO streams.

Each run is plain BO after one fixed first point, through the registered control loop. Only the
first point differs from pure BO on the same stream. No model calls are made. Runs and the fit
cache go in a new folder, so the saved first_point records stay as they are. Two runs repeat saved
ones from an empty cache, as a gate on the recipe.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import argparse
import json
from hypothesis_scope_loop import study as frozen
from hypothesis_scope_multiseed import study as multi
from . import gp_followups as gf, study as v2

OUT = gf.BASE0/'first_point_streams'
PROTOCOL = 'first_point_streams_20261107'
GATE = (('a00102', 0), ('a00493', 1))
POINTS = ('a00210', 'a00102', 'a00499', 'a00493')


def configure():
    gf.configure()
    multi.BASE = OUT
    v2.BASE = OUT


def folder(point, stream, group=None):
    return OUT/(group or gf.SHORT[gf.ARYL])/point/f's{stream}'


def run(point, stream, group=None):
    pool = gf.pool_of(gf.ARYL)
    locked = dict(candidate_id=point, conditions=deepcopy(pool.points[pool.ix[point]]), rationale='fixed by design', assumptions=[])
    frozen.locked_first = lambda t: locked
    multi.folder_for = lambda t, s, a: folder(point, s, group)
    control = dict(protocol_sha256=PROTOCOL, initial_priors={gf.ARYL: dict(first_input=locked)})
    return dict(multi.run_control(gf.ARYL, stream, 'first', control), first=point)


def picks(path):
    return [r['candidate_id'] for r in frozen.read(path/'state.json')['history']]


def bo_folder(stream):
    short = gf.SHORT[gf.ARYL]
    return gf.BASE0/'heldout/runs/bo'/f's{stream}'/short if stream < 8 else gf.G/'bo_extra'/short/f's{stream}'


def identity_jobs():
    return [(picks(bo_folder(s))[0], s, 'identity') for s in range(24)]


def crossover_jobs():
    return [(picks(bo_folder(s))[0], (s + k) % 24, 'crossover') for s in range(24) for k in range(1, 6)]


def check_identity():
    same = [picks(bo_folder(s)) == picks(folder(picks(bo_folder(s))[0], s, 'identity')) for s in range(24)]
    print(f'identity: {sum(same)} of 24 repeat pure BO; differ on {[s for s, ok in enumerate(same) if not ok]}', flush=True)


def check_gate():
    for point, stream in GATE:
        saved = picks(gf.G/'first_point'/gf.SHORT[gf.ARYL]/point/f's{stream}')
        assert picks(folder(point, stream)) == saved, f'{point} s{stream} does not repeat the saved run'
    print('gate passed: both runs repeat the saved picks', flush=True)


def jobs():
    return [(p, s) for p in POINTS for s in range(8, 24)] + [('a00210', s) for s in range(8)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('gate', 'run', 'identity', 'crossover'))
    parser.add_argument('--workers', type=int, default=10)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    frozen.pool_for = gf.pool_of
    with ProcessPoolExecutor(max_workers=args.workers, initializer=configure) as executor:
        group = dict(gate=GATE, run=jobs(), identity=identity_jobs(), crossover=crossover_jobs())[args.action]
        if args.action == 'run':
            check_gate()
        futures = [executor.submit(run, *job) for job in group]
        for done, future in enumerate(as_completed(futures), 1):
            print(json.dumps(dict(done=done, of=len(futures), **future.result())), flush=True)
    if args.action == 'gate':
        check_gate()
    if args.action == 'identity':
        check_identity()
