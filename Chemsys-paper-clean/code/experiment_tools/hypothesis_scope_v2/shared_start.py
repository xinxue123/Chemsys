"""Exploratory comparison of all methods from shared first points on Direct Arylation.

Pair t runs on stream t from pure BO's first point of stream (t - 1) mod 24, so every method in a
pair shares its first point and tie stream. Plain BO is the saved crossover run. The LLM arms use the
named card and the sealed pre-data hypotheses of stream t mod 8. Rules are in the registration.

Neighbour search patches shared decision code, so each phase runs in its own worker processes.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
import argparse
import json
from grounded_baselines import study as gb
from grounded_first_start import proposal
from hypothesis_scope_loop import study as frozen, knowledge as k
from . import contrast_search as c, first_point_streams as ps, gp_followups as f, heldout as h, heldout_lgbo as hl, heldout_named as hn, study as v2, transport

OUT = v2.BASE/'shared_start'
CONTROLLER, LOCAL, BASELINES = 'r1', 'n6m1', ('lgbo', 'rbo')
PROTOCOL = 'shared_start_20261107'
STREAMS = tuple(range(24))
MAXIMUM_ATTEMPTS, MAXIMUM_TOKENS = 1100, 16_000_000
NATIVE_REQUEST = transport.request


def spent():
    results = list((transport.BASE/'requests/shared_start').glob('**/api_result.json'))
    return len(results), sum(frozen.read(p)['usage'].get('total_tokens', 0) for p in results)


def request(namespace, stage, system, public, validator):
    name = 'shared_start/' + namespace.replace('heldout_', '', 1)
    if not (transport.BASE/'requests'/name/stage/'validated.json').exists():
        attempts, tokens = spent()
        if attempts >= MAXIMUM_ATTEMPTS or tokens >= MAXIMUM_TOKENS:
            raise RuntimeError(f'Budget stop: {attempts} attempts, {tokens} tokens')
    return NATIVE_REQUEST(name, stage, system, public, validator)


def point_of(stream):
    """The pair's shared first point: pure BO's first point of the previous stream."""
    return ps.picks(ps.bo_folder((stream - 1) % 24))[0]


def sealed_commit(stream):
    return frozen.read(hn.NAMED/'runs/h1'/f's{stream % 8}'/h.SHORT/'rounds/01_commit.json')


def sealed_prior(stream):
    saved = h.read(h.ROOT/Path(sealed_commit(stream)['working_knowledge']['calls'][0]['request']).parent.parent/'validated.json')
    keys = ('id', 'claim', 'direction', 'scope', 'forecast', 'falsification')
    return [{key: x[key] for key in keys} for x in saved['raw']['hypotheses']]


def shared_prefix(task, seed, arm):
    """Commit the shared first point under the sealed pre-data hypotheses, then observe it."""
    pool = frozen.pool_for(task)
    scopes = k.Scopes(pool)
    folder = v2.folder_for(task, seed, arm)
    source = sealed_commit(seed)
    j = pool.ix[point_of(seed)]
    state = v2.header(task, seed, arm)
    state.update(history=[], decisions=[], notes=[], archive=[], calls=[], permission_events=[])
    work = deepcopy(source['working_knowledge'])
    decision = dict(deepcopy(source['decision']), selected_id=pool.ids[j], conditions=deepcopy(pool.points[j]), LLM_preferred=None, LLM_projection=None,
                    reason='shared_first_with_sealed_hypotheses', locked_forecasts=k.freeze_forecasts(pool, scopes, work['notes'], j, 1, None, None, {}))
    commit = dict(query=1, prestate_sha256=frozen.digest(state), decision=decision, working_knowledge=work)
    commit['commit_sha256'] = frozen.digest(commit)
    frozen.write(folder/'rounds/01_commit.json', commit)
    frozen.apply_outcome(pool, folder, state, commit)
    return state


def point_folders():
    v2.BASE = OUT
    v2.folder_for = lambda t, s, a: OUT/'runs'/a/f's{s}'/h.SHORT
    v2.EVIDENCE_FIRST_ARMS = v2.EVIDENCE_FIRST_ARMS + (CONTROLLER, LOCAL)
    v2.prefix_from_first = shared_prefix


def controller_worker():
    h.initialize()
    frozen.pool_for = h.pool_for
    frozen.locked_first = lambda task: None  # Read only before the first observation, and these arms start after it.
    proposal.background = hn.card
    transport.request = request
    v2.LIVE_ARMS = v2.LIVE_ARMS + (CONTROLLER,)
    point_folders()


def local_worker():
    c.configure()
    frozen.locked_first = lambda task: None
    proposal.background = hn.card
    point_folders()


def baseline_worker():
    h.BASE, h.background, transport.request = OUT, hn.card, request
    hl.configure()
    gb.path_for = lambda t, s, a: OUT/'runs'/a/f's{s}'/h.SHORT


def run_controller(stream):
    return v2.run_arm(h.TASK, stream, CONTROLLER)


def run_local(stream):
    return c.run_local(f.ARYL, stream, LOCAL)


def run_baseline(stream, arm):
    pool = h.pool_for()
    point = point_of(stream)
    first = dict(candidate_id=point, conditions=deepcopy(pool.points[pool.ix[point]]), rationale='fixed by design', assumptions=[])
    p = dict(protocol_sha256=PROTOCOL, shared_first={h.TASK: first}, shared_blinded_priors={h.TASK: sealed_prior(stream)})
    gb.check = lambda protected=False: p
    gb.JOBS = ((h.TASK, stream, arm),)
    return gb.run_one(h.TASK, stream, arm)


def phase(worker, jobs, workers):
    with ProcessPoolExecutor(max_workers=workers, initializer=worker) as executor:
        futures = [executor.submit(*job) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            try:
                print(json.dumps(dict(done=done, of=len(futures), **future.result())), flush=True)
            except Exception:
                for other in futures:
                    other.cancel()
                raise


def check_pilot(stream):
    """The pilot gates: card, shared first point, and the BO reference at query 2."""
    point = point_of(stream)
    saved = ps.picks(ps.folder(point, stream, 'crossover'))
    out = dict(stream=stream, point=point, plain_bo_q2=saved[1])
    for arm in (CONTROLLER, LOCAL) + BASELINES:
        folder = OUT/'runs'/arm/f's{stream}'/h.SHORT
        state = frozen.read(folder/'state.json')
        d = frozen.read(folder/'rounds/02_commit.json')['decision']
        out[arm] = dict(status=state['status'], first_matches=state['history'][0]['candidate_id'] == point, seed=state.get('seed'),
                        q2_reference_matches=d.get('unrestricted_BO_id', d.get('BO_reference_id')) == saved[1], calls=len(state['calls']))
    requests = list((transport.BASE/'requests/shared_start').glob('**/attempt1/request.json'))
    out['requests'] = len(requests)
    out['requests_with_named_card'] = sum('1-bromo-2-fluorobenzene' in p.read_text(encoding='utf-8') for p in requests)
    out['attempts'], out['tokens'] = spent()
    print(json.dumps(out, indent=1), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('run', 'check', 'spent'))
    parser.add_argument('--streams', nargs='*', type=int, default=list(STREAMS))
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    frozen.pool_for = h.pool_for
    if args.action == 'check':
        return check_pilot(args.streams[0])
    if args.action == 'spent':
        return print(json.dumps(spent()))
    OUT.mkdir(parents=True, exist_ok=True)
    phase(local_worker, [(run_local, s) for s in args.streams], args.workers)
    phase(controller_worker, [(run_controller, s) for s in args.streams], args.workers)
    phase(baseline_worker, [(run_baseline, s, a) for a in BASELINES for s in args.streams], args.workers)
    print(json.dumps(dict(status='COMPLETE', spent=spent())), flush=True)
