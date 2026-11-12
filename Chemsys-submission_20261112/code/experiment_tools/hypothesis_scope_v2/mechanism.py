"""GP-only mechanism tests registered on 6 November 2026. No model calls are made.

Test C replaces each sealed pre-data scope with a same-sized placebo, with and without the
permission gate. Test T keeps the real scopes and replaces the gate with a blind timer. GP,
acquisition, pools and budget are the registered ones. Placebo regions never read pool values.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from functools import lru_cache
import argparse
import hashlib
import json
import numpy as np
from chem.benchmarks.knowledge_domain_context import task_context
from hypothesis_scope_loop import study as frozen, knowledge as k, policy
from . import gp_followups as f, study as v2

G = f.G
ARYL = f.ARYL
SEED = 20261106
PLACEBO = ('p0s', 'p0sg')
TIMERS = dict(p0t1=1, p0t3=3, p0t5=5, p0t10=10)
ARMS = PLACEBO + tuple(TIMERS)
FAITHFUL = ('q0g', 'q0')
MAXIMUM_TESTS = 2
CALIPER = .10
CODING_SHA256 = 'ef7c389e7873b7b81378a7ecda6676f759c4c6fdd9fd75b5ec7d444456949398'
DESIGN = G/'placebo_design.json'
CLOCK = dict(query=None, state=None, tests=False)
NATIVE_DECIDE = v2.decide
NATIVE_SELECT = policy.select
read, write = frozen.read, frozen.write


def units():
    return [(t, s) for t in frozen.native.TASKS for s in range(4)] + [(ARYL, s) for s in range(8)]


def axis_limits(pool, axis):
    if pool.task == ARYL:
        raise ValueError('Direct Arylation has no continuous axis')
    limits = next(v['declared_input_bounds'] for v in task_context(pool.task)['inputs'] if v['name'] == axis)
    return float(limits[0]), float(limits[1])


def placebo_scope(pool, scopes, scope, rng):
    """Same axes and same size as the sealed scope, placed at random. Pool values are never read."""
    real = scopes.mask(scope)
    share = float(real.mean())
    for _ in range(100000):
        new = {}
        for axis, rule in scope.items():
            if 'allowed' in rule:
                levels = sorted({p[axis] for p in pool.points})
                picks = rng.choice(len(levels), len(rule['allowed']), replace=False)
                new[axis] = dict(allowed=sorted(levels[i] for i in picks))
            else:
                low, high = axis_limits(pool, axis)
                width = min(float(rule['bounds'][1]) - float(rule['bounds'][0]), high - low)
                start = float(rng.uniform(low, high - width))
                new[axis] = dict(bounds=[start, start + width])
        mask = scopes.mask(new)
        if mask.any() and not np.array_equal(mask, real) and share/2 <= float(mask.mean()) <= share*2:
            return new
    raise RuntimeError(f'No placebo found for {pool.task}: {scope}')


def design():
    if DESIGN.exists():
        return read(DESIGN)
    plan = {}
    for index, (task, seed) in enumerate(units()):
        pool = f.pool_of(task)
        scopes = k.Scopes(pool)
        rng = np.random.default_rng([SEED, index])
        source, _ = f.sources(task, seed)
        rows = []
        for note in source['working_knowledge']['notes']:
            scope = placebo_scope(pool, scopes, note['scope'], rng)
            rows.append(dict(id=note['id'], direction=note['direction'], real_share=float(scopes.mask(note['scope']).mean()),
                             placebo_share=float(scopes.mask(scope).mean()), real=note['scope'], placebo=scope))
        plan[f'{task}|{seed}'] = rows
    write(DESIGN, plan)
    return plan


def placebo_prefix(task, seed, arm):
    """f.llm_prefix with each sealed scope replaced by its placebo."""
    pool = frozen.pool_for(task)
    scopes = k.Scopes(pool)
    folder = v2.folder_for(task, seed, arm)
    source, _ = f.sources(task, seed)
    j = pool.ix[source['decision']['selected_id']]
    state = v2.header(task, seed, arm)
    state.update(history=[], decisions=[], notes=[], archive=[], calls=[], permission_events=[])
    work = deepcopy(source['working_knowledge'])
    rows = {r['id']: r for r in read(DESIGN)[f'{task}|{seed}']}
    for note in work['notes']:
        note['scope'] = deepcopy(rows[note['id']]['placebo'])
        note['semantic_key'] = scopes.semantic_key(note['direction'], note['scope'])
    decision = dict(deepcopy(source['decision']), LLM_preferred=None, LLM_projection=None,
                    locked_forecasts=k.freeze_forecasts(pool, scopes, work['notes'], j, 1, None, None, {}))
    commit = dict(query=1, prestate_sha256=frozen.digest(state), decision=decision, working_knowledge=work)
    commit['commit_sha256'] = frozen.digest(commit)
    write(folder/'rounds/01_commit.json', commit)
    frozen.apply_outcome(pool, folder, state, commit)
    return state


def side_scope(pool, conditions, axis, values):
    """A contrast side as a scope: the fixed conditions plus one side of the active axis."""
    scope = deepcopy(conditions)
    if any(isinstance(v, str) for v in values):
        scope[axis] = dict(allowed=list(values))
    else:
        low, high = axis_limits(pool, axis)
        scope[axis] = dict(bounds=[low if values[0] is None else float(values[0]), high if values[1] is None else float(values[1])])
    return scope


def faithful_notes(pool, scopes, notes, claims, key_for):
    """Claim-faithful translation: each contrast removes only its claimed-lower side."""
    out = []
    for note in notes:
        claim = claims[key_for(note)]
        if not claim['contrasts'] and note['direction'] == 'avoid':
            out.append(deepcopy(note))
        for i, spec in enumerate(claim['contrasts'], 1):
            within = spec['within'] if 'within' in spec else {a: s for a, s in claim['scope'].items() if a != spec['axis']}
            low = side_scope(pool, within, spec['axis'], spec['low'])
            new = deepcopy(note)
            new.update(id=f"{note['id']}_c{i}", direction='avoid', scope=low, semantic_key=scopes.semantic_key('avoid', low),
                       claimed_scope=deepcopy(note['scope']), claimed_direction=note['direction'],
                       contrast=dict(axis=spec['axis'], high=side_scope(pool, within, spec['axis'], spec['high']), low=low, pairs=spec.get('pairs'), evidence=[], paused=False))
            out.append(new)
    return out


def faithful_prefix(task, seed, arm):
    """f.llm_prefix with sealed pre-data hypotheses in claim-faithful form."""
    pool = frozen.pool_for(task)
    scopes = k.Scopes(pool)
    folder = v2.folder_for(task, seed, arm)
    source, _ = f.sources(task, seed)
    j = pool.ix[source['decision']['selected_id']]
    state = v2.header(task, seed, arm)
    state.update(history=[], decisions=[], notes=[], archive=[], calls=[], permission_events=[])
    work = deepcopy(source['working_knowledge'])
    claims = coding()
    short = frozen.SHORT[task]
    key_for = lambda note: f"{short}|{seed if task == ARYL else 'all'}|{note['id']}"
    work['notes'] = faithful_notes(pool, scopes, work['notes'], claims, key_for)
    decision = dict(deepcopy(source['decision']), LLM_preferred=None, LLM_projection=None,
                    locked_forecasts=k.freeze_forecasts(pool, scopes, work['notes'], j, 1, None, None, {}))
    commit = dict(query=1, prestate_sha256=frozen.digest(state), decision=decision, working_knowledge=work)
    commit['commit_sha256'] = frozen.digest(commit)
    write(folder/'rounds/01_commit.json', commit)
    frozen.apply_outcome(pool, folder, state, commit)
    return state


def coding():
    text = (G/'claim_coding.json').read_text(encoding='utf-8')
    assert hashlib.sha256(text.encode('utf-8')).hexdigest() == CODING_SHA256, 'Claim coding changed after it was frozen'
    return {c['key']: c for c in json.loads(text)['hypotheses']}


def is_match(pool, a, b, axis, pairs):
    """The audit's matching rule: exact on categorical and grid axes, 10% of range on other continuous axes."""
    one, two = pool.points[a], pool.points[b]
    if pairs and [one[axis], two[axis]] not in pairs and [two[axis], one[axis]] not in pairs:
        return False
    for name, kind in zip(pool.axes, pool.kinds):
        if name == axis:
            continue
        if kind == 'categorical' or len(LEVELS(pool, name)) <= 12:
            if one[name] != two[name]:
                return False
        elif abs(one[name] - two[name]) > CALIPER*WIDTH(pool, name):
            return False
    return True


@lru_cache(maxsize=None)
def axis_values(task, name):
    return sorted({p[name] for p in frozen.pool_for(task).points})


def LEVELS(pool, name):
    return axis_values(pool.task, name)


def WIDTH(pool, name):
    values = axis_values(pool.task, name)
    return values[-1] - values[0]


def sides(scopes, note):
    return scopes.mask(note['contrast']['high']), scopes.mask(note['contrast']['low'])


def observed_pairs(pool, scopes, note, order):
    """Disjoint matched pairs among measured points, in query order, as (high, low)."""
    high, low = sides(scopes, note)
    used, out = set(), []
    for a in order:
        if not high[a] or a in used:
            continue
        for b in order:
            if low[b] and b not in used and is_match(pool, a, b, note['contrast']['axis'], note['contrast']['pairs']):
                used |= {a, b}
                out.append((a, b))
                break
    return out, used


def update_contrasts(pool, scopes, state):
    """One contrary matched contrast pauses a rule. Labels still need two independent contrasts."""
    order = [pool.ix[r['candidate_id']] for r in state['history']]
    values = {pool.ix[r['candidate_id']]: r['raw_value'] for r in state['history']}
    query = {j: i + 1 for i, j in enumerate(order)}
    for note in state['notes']:
        record = note.get('contrast')
        if not record:
            continue
        pairs, _ = observed_pairs(pool, scopes, note, order)
        # A pair counts only if a member was measured at or after installation, so hindsight earns no credit.
        record['evidence'] = [dict(high=pool.ids[a], low=pool.ids[b], difference=values[a] - values[b]) for a, b in pairs
                              if max(query[a], query[b]) >= note['installed_query']]
        if not record['paused'] and any(e['difference'] < -1e-9 for e in record['evidence']):
            record['paused'] = True
            note['permission'] = 'paused'
            note['pause_reasons'] = ['contrary_matched_contrast']


def test_candidate(pool, scopes, notes, state, context):
    """The next matched partner of a measured point, chosen by qLogEI among matched candidates."""
    order = [pool.ix[r['candidate_id']] for r in state['history']]
    for note in notes:
        record = note.get('contrast')
        if not record or record['paused']:
            continue
        high, low = sides(scopes, note)
        _, used = observed_pairs(pool, scopes, note, order)
        for point in order:
            if point in used or not (high[point] or low[point]):
                continue
            other = low if high[point] else high
            js = np.array([j for j in np.flatnonzero(other & context['available']) if is_match(pool, point, int(j), record['axis'], record['pairs'])], dtype=int)
            if len(js):
                return int(js[np.lexsort((context['tie_priority'][js], -context['base_logei'][js]))[0]]), note
    return None


def select_with_test(context, scopes, notes, incumbent):
    choice = NATIVE_SELECT(context, scopes, notes, incumbent)
    state = CLOCK['state']
    used = sum(r['decision'].get('reason') == 'matched_contrast_test' for r in state['decisions'])
    if CLOCK['tests'] and used < MAXIMUM_TESTS:
        found = test_candidate(scopes.pool, scopes, notes, state, context)
        if found:
            j, note = found
            choice = dict(choice, index=j, reason='matched_contrast_test', causes={note['semantic_key']: True}, fallback=False)
    return choice


def faithful_decide(task, seed, arm, pool, scopes, gp, state, q, sealed4, folder):
    CLOCK.update(query=q, state=state, tests=arm in ('q0', 'q1'))
    if arm in ('q0', 'q1'):
        update_contrasts(pool, scopes, state)
    return NATIVE_DECIDE(task, seed, arm, pool, scopes, gp, state, q, sealed4, folder)


def run_faithful(task, seed, arm):
    """q0g always removes the claimed-lower side. q0 adds matched tests and the contrast exit."""
    v2.decide = faithful_decide
    policy.select = select_with_test
    k.pause_for_recommendation = lambda notes, inc: (deepcopy(notes), [])
    k.eligible = (lambda note, inc: True) if arm == 'q0g' else (lambda note, inc: not note.get('contrast', {}).get('paused', False))
    return v2.run_arm(task, seed, arm)


def configure():
    f.configure()
    v2.EVIDENCE_FIRST_ARMS = v2.EVIDENCE_FIRST_ARMS + ARMS + FAITHFUL
    v2.prefix_from_first = lambda t, s, a: placebo_prefix(t, s, a) if a in PLACEBO else faithful_prefix(t, s, a) if a in FAITHFUL else f.llm_prefix(t, s, a)


def timed_decide(task, seed, arm, pool, scopes, gp, state, q, sealed4, folder):
    CLOCK['query'] = q
    return NATIVE_DECIDE(task, seed, arm, pool, scopes, gp, state, q, sealed4, folder)


def run_placebo(task, seed, arm):
    """p0s keeps the permission gate. p0sg lets every installed placebo constrain."""
    v2.decide = NATIVE_DECIDE
    policy.select = NATIVE_SELECT
    k.eligible, k.pause_for_recommendation = f.NATIVE_GATE if arm == 'p0s' else (lambda note, inc: True, lambda notes, inc: (deepcopy(notes), []))
    return v2.run_arm(task, seed, arm)


def run_timer(task, seed, arm):
    """Every installed hypothesis constrains queries 2 to K+1, then stops. No forecast rule applies."""
    limit = 1 + TIMERS[arm]
    v2.decide = timed_decide
    policy.select = NATIVE_SELECT
    k.eligible, k.pause_for_recommendation = (lambda note, inc: CLOCK['query'] <= limit), (lambda notes, inc: (deepcopy(notes), []))
    return v2.run_arm(task, seed, arm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('design', 'time', 'run', 'pilot', 'faithful'))
    parser.add_argument('--workers', type=int, default=10)
    args = parser.parse_args()
    G.mkdir(parents=True, exist_ok=True)
    frozen.initialize()
    f.proposal.background = f.h.background
    frozen.pool_for = f.pool_of
    plan = design()
    if args.action == 'design':
        shares = [(r['real_share'], r['placebo_share']) for rows in plan.values() for r in rows]
        print(json.dumps(dict(units=len(plan), scopes=len(shares), mean_real_share=float(np.mean([a for a, _ in shares])), mean_placebo_share=float(np.mean([b for _, b in shares])))))
        return
    jobs = [(run_placebo, t, s, a) for a in PLACEBO for t, s in units()] + [(run_timer, t, s, a) for a in TIMERS for t, s in units()]
    if args.action == 'time':
        jobs = [(run_placebo, frozen.native.TASKS[0], 0, 'p0s'), (run_timer, frozen.native.TASKS[0], 0, 'p0t3')]
    if args.action == 'pilot':
        jobs = [(run_faithful, frozen.native.TASKS[0], 0, a) for a in FAITHFUL] + [(run_faithful, ARYL, 0, a) for a in FAITHFUL]
    if args.action == 'faithful':
        jobs = [(run_faithful, t, s, a) for a in FAITHFUL for t, s in units()]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=configure) as executor:
        futures = [executor.submit(*job) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            print(json.dumps(dict(done=done, of=len(futures), **future.result())), flush=True)
