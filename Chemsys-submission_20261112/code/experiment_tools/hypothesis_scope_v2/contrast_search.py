"""Contrast-neighbourhood search, registered on 6 November 2026. No model calls are made.

From a fixed query on, qLogEI chooses among neighbours of the incumbent: candidates that differ on
at most m axes under the audit's matching rule. GP, acquisition, pools, budget and first points are
the registered ones. Arms: <variant>, h_<variant> (adds the first-point hedge) and b_<variant>
(pure BO's first point with no LLM input).
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from functools import lru_cache
import argparse
import json
import numpy as np
from hypothesis_scope_loop import study as frozen, knowledge as k, policy
from . import gp_followups as f, mechanism as m, study as v2

VARIANTS = dict(n6m1=(6, 1, False), n6m2=(6, 2, False), n11m1=(11, 1, False), n11m2=(11, 2, False), a6m1=(6, 1, True))
CLOCK = dict(query=None, state=None, spec=None)
ARMS = tuple(VARIANTS) + tuple(p + v for p in ('h_', 'b_') for v in VARIANTS)
NATIVE_DECIDE, NATIVE_SELECT = m.NATIVE_DECIDE, m.NATIVE_SELECT


def spec_of(arm):
    start, limit, alternate = VARIANTS[arm.split('_')[-1]]
    return dict(start=start, limit=limit, alternate=alternate, hedge=arm.startswith('h_'))


@lru_cache(maxsize=None)
def features(task):
    pool = frozen.pool_for(task)
    columns = []
    for name, kind in zip(pool.axes, pool.kinds):
        values = [p[name] for p in pool.points]
        if kind == 'categorical':
            columns.append((np.unique(values, return_inverse=True)[1], None))
        else:
            x = np.asarray(values, dtype=float)
            columns.append((x, m.CALIPER*(x.max() - x.min())))
    return columns


def neighbours(task, j, limit):
    """Candidates that differ from candidate j on at most `limit` axes."""
    differ = 0
    for x, tolerance in features(task):
        differ = differ + ((x != x[j]) if tolerance is None else (np.abs(x - x[j]) > tolerance + 1e-12))
    return differ <= limit


def best_in(context, region):
    js = np.flatnonzero(region)
    return int(js[np.lexsort((context['tie_priority'][js], -context['base_logei'][js]))[0]])


def hedge_point(pool, state):
    """Pure BO's seeded first point, if the first outcome fell over 1 SD below its locked forecast."""
    first = state['decisions'][0]
    locked = first['decision'].get('locked_forecasts', [])
    if not locked:
        return None
    forecast = locked[0]['forecast']
    if first['raw_value'] >= forecast['mean_raw'] - forecast['sd_raw']:
        return None
    return pool.ix[f.sources(pool.task, state['seed'])[1]['selected_id']]


def select_local(context, scopes, notes, incumbent):
    choice = NATIVE_SELECT(context, scopes, notes, incumbent)
    state, q, spec = CLOCK['state'], CLOCK['query'], CLOCK['spec']
    pool = scopes.pool
    if spec['hedge'] and q == 2:
        j = hedge_point(pool, state)
        if j is not None and context['available'][j]:
            return dict(choice, index=int(j), reason='failed_first_hedge', fallback=False)
    if q >= spec['start'] and (not spec['alternate'] or q % 2 == 0):
        best = max(state['history'], key=lambda r: r['raw_value'])
        region = choice['mask'] & context['available'] & neighbours(pool.task, pool.ix[best['candidate_id']], spec['limit'])
        if region.any():
            return dict(choice, index=best_in(context, region), reason='incumbent_contrast_scope', fallback=False)
    return choice


def local_decide(task, seed, arm, pool, scopes, gp, state, q, sealed4, folder):
    CLOCK.update(query=q, state=state, spec=spec_of(arm))
    return NATIVE_DECIDE(task, seed, arm, pool, scopes, gp, state, q, sealed4, folder)


def blank_prefix(task, seed, arm):
    """Pure BO's first point, with no LLM hypotheses at all."""
    pool = frozen.pool_for(task)
    scopes = k.Scopes(pool)
    folder = v2.folder_for(task, seed, arm)
    source, start = f.sources(task, seed)
    j = pool.ix[start['selected_id']]
    state = v2.header(task, seed, arm)
    state.update(history=[], decisions=[], notes=[], archive=[], calls=[], permission_events=[])
    work = dict(notes=[], archive=[], calls=deepcopy(source['working_knowledge']['calls']), permission_events=[])
    decision = dict(deepcopy(source['decision']), selected_id=pool.ids[j], conditions=deepcopy(pool.points[j]), LLM_preferred=None, LLM_projection=None,
                    reason='pure_BO_first_no_LLM', locked_forecasts=k.freeze_forecasts(pool, scopes, [], j, 1, None, None, {}))
    commit = dict(query=1, prestate_sha256=frozen.digest(state), decision=decision, working_knowledge=work)
    commit['commit_sha256'] = frozen.digest(commit)
    frozen.write(folder/'rounds/01_commit.json', commit)
    frozen.apply_outcome(pool, folder, state, commit)
    return state


ROUND2 = dict(r2a=dict(tests=2, translate=False), r2b=dict(tests=2, translate=True), r2c=dict(tests=1, translate=False))
ARMS2 = tuple(ROUND2) + tuple('c_' + v for v in ROUND2)
START, LIMIT, EXIT_AFTER = 6, 1, 3


def round2_of(arm):
    return ROUND2[arm.split('_')[-1]]


def attach_contrast(pool, note, claim):
    """A favour note keeps its scope and gains its first coded contrast, for tests and the contrast exit."""
    if not claim['contrasts']:
        return note
    spec = claim['contrasts'][0]
    within = spec['within'] if 'within' in spec else {a: s for a, s in claim['scope'].items() if a != spec['axis']}
    note['contrast'] = dict(axis=spec['axis'], high=m.side_scope(pool, within, spec['axis'], spec['high']), low=m.side_scope(pool, within, spec['axis'], spec['low']),
                            pairs=spec.get('pairs'), evidence=[], paused=False)
    return note


def round2_prefix(task, seed, arm):
    """Commit the first point under the sealed hypotheses. The card rule applies to c_ arms."""
    pool = frozen.pool_for(task)
    scopes = k.Scopes(pool)
    folder = v2.folder_for(task, seed, arm)
    source, start = f.sources(task, seed)
    random_start = arm.startswith('c_') and task == f.ARYL  # Only this card leaves the substrates unspecified.
    j = pool.ix[(start if random_start else source['decision'])['selected_id']]
    state = v2.header(task, seed, arm)
    state.update(history=[], decisions=[], notes=[], archive=[], calls=[], permission_events=[])
    work = deepcopy(source['working_knowledge'])
    claims = m.coding()
    short = frozen.SHORT[task]
    key_for = lambda note: f"{short}|{seed % 8 if task == f.ARYL else 'all'}|{note['id']}"
    if round2_of(arm)['translate']:
        work['notes'] = m.faithful_notes(pool, scopes, work['notes'], claims, key_for)
    else:
        work['notes'] = [attach_contrast(pool, n, claims[key_for(n)]) for n in work['notes']]
    decision = dict(deepcopy(source['decision']), selected_id=pool.ids[j], conditions=deepcopy(pool.points[j]), LLM_preferred=None, LLM_projection=None,
                    reason='card_rule_random_first' if random_start else source['decision'].get('reason'),
                    locked_forecasts=k.freeze_forecasts(pool, scopes, work['notes'], j, 1, None, None, {}))
    commit = dict(query=1, prestate_sha256=frozen.digest(state), decision=decision, working_knowledge=work)
    commit['commit_sha256'] = frozen.digest(commit)
    frozen.write(folder/'rounds/01_commit.json', commit)
    frozen.apply_outcome(pool, folder, state, commit)
    return state


def select_round2(context, scopes, notes, incumbent):
    choice = NATIVE_SELECT(context, scopes, notes, incumbent)
    state, q, spec = CLOCK['state'], CLOCK['query'], CLOCK['spec']
    pool = scopes.pool
    used = sum(r['decision'].get('reason') == 'matched_contrast_test' for r in state['decisions'])
    if used < spec['tests']:
        found = m.test_candidate(pool, scopes, notes, state, context)
        if found:
            return dict(choice, index=found[0], reason='matched_contrast_test', causes={found[1]['semantic_key']: True}, fallback=False)
    if q >= START:
        best = max(state['history'], key=lambda r: r['raw_value'])
        region = choice['mask'] & context['available'] & neighbours(pool.task, pool.ix[best['candidate_id']], LIMIT)
        if region.any():
            return dict(choice, index=best_in(context, region), reason='incumbent_contrast_scope', fallback=False)
    return choice


def round2_decide(task, seed, arm, pool, scopes, gp, state, q, sealed4, folder):
    CLOCK.update(query=q, state=state, spec=round2_of(arm))
    m.update_contrasts(pool, scopes, state)
    return NATIVE_DECIDE(task, seed, arm, pool, scopes, gp, state, q, sealed4, folder)


def run_round2(task, seed, arm):
    v2.decide, policy.select = round2_decide, select_round2
    if round2_of(arm)['translate']:
        k.pause_for_recommendation = lambda notes, inc: (deepcopy(notes), [])
        k.eligible = lambda note, inc: CLOCK['query'] <= 1 + EXIT_AFTER and not note.get('contrast', {}).get('paused', False)
    else:
        k.eligible, k.pause_for_recommendation = f.NATIVE_GATE
    return v2.run_arm(task, seed, arm)


def configure():
    f.configure()
    v2.EVIDENCE_FIRST_ARMS = v2.EVIDENCE_FIRST_ARMS + ARMS + ARMS2
    v2.prefix_from_first = lambda t, s, a: round2_prefix(t, s, a) if a in ARMS2 else blank_prefix(t, s, a) if a.startswith('b_') else f.llm_prefix(t, s, a)


def run_local(task, seed, arm):
    v2.decide, policy.select = local_decide, select_local
    k.eligible, k.pause_for_recommendation = f.NATIVE_GATE
    return v2.run_arm(task, seed, arm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('pilot', 'select', 'arms', 'round2', 'extend'))
    parser.add_argument('--arms', nargs='*', default=[])
    parser.add_argument('--data', choices=('dev', 'held', 'both'), default='both')
    parser.add_argument('--workers', type=int, default=10)
    args = parser.parse_args()
    frozen.initialize()
    frozen.pool_for = f.pool_of
    units = m.units()
    if args.action == 'pilot':
        jobs = [(frozen.native.TASKS[0], 0, a) for a in VARIANTS]
    elif args.action == 'select':
        jobs = [(t, s, a) for a in VARIANTS for t, s in units if t != f.ARYL]
    else:
        keep = {'dev': lambda t: t != f.ARYL, 'held': lambda t: t == f.ARYL, 'both': lambda t: True}[args.data]
        jobs = [(t, s, a) for a in args.arms for t, s in units if keep(t)]
    if args.action == 'extend':
        jobs = [(f.ARYL, s, 'c_r2a') for s in range(8, 24)]
    runner = run_round2 if args.action in ('round2', 'extend') else run_local
    with ProcessPoolExecutor(max_workers=args.workers, initializer=configure) as executor:
        futures = [executor.submit(runner, *job) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            print(json.dumps(dict(done=done, of=len(futures), **future.result())), flush=True)
