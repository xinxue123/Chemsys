"""Registered screen of new tasks: does a well-placed LLM start beat BO?

Protocol: docs/new_task_screen_protocol_DRAFT_20261107.md. This module holds step 0 only:
building pools and cards for the eligible tasks. It reads outcomes solely to write the oracle,
and prints row counts, coverage and levels, never outcome statistics, so predictions can be
fixed before any score is seen. Paid steps are added only after the user's go-ahead.

Frozen modules supply the prompts and the card format, and are not edited.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

import numpy as np
from chem.benchmarks import knowledge_gp as native
from chem.benchmarks.knowledge_domain_context import task_context
from chemical_hypothesis_bo import background as original
from grounded_first_start import proposal
from hypothesis_scope_loop import study as frozen
from . import study as v2, transport

ROOT = frozen.ROOT
RAW = ROOT/'data/source_tables/new_task_screen'
BASE = v2.BASE/'screen'
read, write, sha = frozen.read, frozen.write, frozen.sha
FIXED = ['solvent volume, temperature, loadings, concentration, time and ancillary fixed protocol details are not specified; do not vary or invent them']

TASKS = {
    'dfl': dict(
        task='DFL_Deoxyfluorination', source='currybo/data-analysis/deoxyfluorination/deoxyfluorination_original.csv',
        axes={'Alcohol': 'substrate_smiles', 'Sulfonyl_fluoride': 'fluoride_smiles', 'Base': 'base_smiles'}, response='yield', fixed={},
        reaction='Deoxyfluorination of an alcohol with a sulfonyl fluoride reagent and an organic base, giving the alkyl fluoride',
        meaning={'Alcohol': 'Alcohol substrate identity.', 'Sulfonyl_fluoride': 'Sulfonyl fluoride reagent identity.', 'Base': 'Organic base identity.'},
        objective=('yield', 'Reported yield of the alkyl fluoride product, in percent. Higher is better.'),
        substrates='The alcohol substrate is an optimization variable chosen from the supplied list.'),
    'bor': dict(
        task='BOR_Ni_Borylation', source='currybo/data-analysis/borylation/borylation_cleaned.csv',
        axes={'Electrophile': 'Electrophile', 'Ligand': 'ligand', 'Solvent': 'solvent'}, response='yield', fixed={},
        reaction='Nickel-catalysed borylation of an aryl electrophile, giving the aryl boron product',
        meaning={'Electrophile': 'Aryl electrophile substrate identity.', 'Ligand': 'Ligand identity for the nickel catalyst.', 'Solvent': 'Alcohol solvent identity.'},
        objective=('yield', 'Reported yield of the borylation product, in percent. Higher is better.'),
        substrates='The aryl electrophile is an optimization variable chosen from the supplied list.'),
    'chx': dict(
        task='CHX_Pd_Heterocoupling', source='currybo/data-analysis/Cernak/Cernak_original.csv',
        axes={'Nucleophile': 'Nucleophile', 'Precatalyst': 'Catalyst', 'Base': 'Base'}, response='Conversion',
        fixed={'Electrophile': 'BrC1=CC=CN=C1'},
        reaction='Palladium-catalysed cross-coupling of 3-bromopyridine with a nucleophile, using a palladium precatalyst and an organic base',
        meaning={'Nucleophile': 'Nucleophile coupling partner identity.', 'Precatalyst': 'Palladium precatalyst identity.', 'Base': 'Organic base identity.'},
        objective=('conversion', 'Reported conversion to the cross-coupling product, in percent. Higher is better.'),
        substrates='The nucleophile is an optimization variable chosen from the supplied list; the aryl bromide is fixed.'),
}


def rows(rel):
    with open(RAW/rel, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def borylation_matches_originals(table):
    """The cleaned file must carry the original measurements per solvent, as multisets. It stores them divided by 100."""
    for solvent, name in (('CCO', 'etoh'), ('CO', 'meoh')):
        cleaned = sorted(float(r['yield']) for r in table if r['solvent'] == solvent)
        source = sorted(float(r['yield'])/100 for r in rows(f'currybo/data-analysis/borylation/{name}_borylation_original.csv'))
        assert len(cleaned) == len(source) and all(abs(a - b) < 1e-9 for a, b in zip(cleaned, source)), f'borylation cleaned values differ from {name} original'


def prepare(short):
    spec = TASKS[short]
    folder = BASE/short/'input'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'pool.json').exists():
        return read(folder/'receipt.json')
    table = rows(spec['source'])
    if short == 'bor':
        borylation_matches_originals(table)
    if short == 'chx':
        assert {r['Electrophile'] for r in table} == {spec['fixed']['Electrophile']}
    points = [{a: r[c].strip() for a, c in spec['axes'].items()} for r in table]
    keys = Counter(tuple(p.values()) for p in points)
    assert max(keys.values()) == 1, 'duplicate input row'
    levels = {a: sorted({p[a] for p in points}) for a in spec['axes']}
    assert len(points) == int(np.prod([len(v) for v in levels.values()])), 'grid not full'
    raw = np.asarray([float(r[spec['response']]) for r in table])
    assert np.isfinite(raw).all()
    fraction = bool(raw.max() <= 2.0)  # Fixed rule: a source on a 0-1 scale is converted to percent. The value is not printed.
    raw = raw*100 if fraction else raw
    top = float(raw.max())
    write(folder/'pool.json', dict(task_id=spec['task'], reaction_class=spec['reaction'], axes=list(spec['axes']),
                                   ids=[f'{short[0]}{j:05d}' for j in range(len(points))], points=points, levels=levels,
                                   fixed_substrates=spec['fixed'], objective_unit='%', objective_scale=top, source_member=spec['source'],
                                   source_sha256=sha(RAW/spec['source'])))
    np.savez_compressed(folder/'oracle.npz', values=raw/top, raw_values=raw)
    receipt = dict(task=spec['task'], rows=len(points), levels={a: len(v) for a, v in levels.items()}, grid_coverage=100.0,
                   source_converted_from_fraction=fraction, pool_sha256=sha(folder/'pool.json'), oracle_sha256=sha(folder/'oracle.npz'),
                   source_sha256=sha(RAW/spec['source']), outcome_access='oracle written by code; no outcome statistic printed before predictions')
    write(folder/'receipt.json', receipt)
    return receipt


def pool_for(short):
    path = BASE/short/'input/pool.json'
    p = read(path)
    pool = native.Pool(p['task_id'], 'P1', tuple(p['ids']), tuple(p['axes']), ('categorical',)*len(p['axes']), tuple(p['points']),
                       tuple(tuple(x[a] for a in p['axes']) for x in p['points']), tuple(p['ids'][:4]),
                       p['objective_unit'], float(p['objective_scale']), p, ROOT, path)
    pool.initial_ids = ()
    return pool


def card_for(short):
    """Neutral card in the CPA format: names the reaction class, gives levels by exact structure, names no source or result."""
    spec = TASKS[short]
    policy = task_context('Doyle_Buchwald_Hartwig')

    def card(pool):
        inputs = []
        for axis in spec['axes']:
            row = dict(name=axis, kind='categorical', meaning=spec['meaning'][axis], levels=list(pool.raw_pool['levels'][axis]))
            row['chemical_identities'] = {v: dict(name=original.COMMON_NAMES.get(v, 'name not assigned; exact molecular graph supplied'),
                                                  SMILES=v, graph=original.molecular_graph(v)) for v in row['levels']}
            inputs.append(row)
        fixed = {k: dict(SMILES=v, graph=original.molecular_graph(v)) for k, v in spec['fixed'].items()}
        name, meaning = spec['objective']
        return dict(group='P1', reaction_or_process=spec['reaction'],
                    description='Choose one value for each input from the supplied measured combinations.',
                    objective=dict(name=name, unit='%', direction='maximize', meaning=meaning),
                    substrate_scope=dict(substrates_are_optimization_variables=True, identity_status='substrate identities supplied as exact structures',
                                         statement=spec['substrates'], fixed_substrates=fixed),
                    inputs=inputs, candidate_policy=policy['candidate_policy'], knowledge_policy=policy['knowledge_policy'],
                    candidate_count=len(pool.ids),
                    identity_evidence='Curated experimental definitions and supplied input structures; no benchmark performance conclusions.',
                    unknown_fixed_details=FIXED,
                    reasoning_requirements=['State a conditional, falsifiable chemical or process relationship using meaningful identities and physical units.',
                                            'Separate general mechanistic expectations from associations supported by current own observations.',
                                            'A source molecular identity is not a source performance claim. Do not recall published benchmark results.',
                                            'Candidate IDs identify transport records only. Use supplied full physical conditions to reason.'])
    return card


def show_card(short):
    """Write the card and the first prompt without any model call, and report their size."""
    pool = pool_for(short)
    proposal.background = card_for(short)
    public = proposal.public(pool)
    write(BASE/short/'card.json', public['background'], replace=True)
    text = json.dumps(public, ensure_ascii=False)
    write(BASE/short/'first_prompt_public.json', public, replace=True)
    return dict(task=short, candidates=len(pool.ids), levels={a: len(pool.raw_pool['levels'][a]) for a in pool.raw_pool['axes']},
                first_prompt_characters=len(text), examples=len(public['input_only_examples']),
                named=sum(not v['name'].startswith('name not assigned') for x in public['background']['inputs'] for v in x['chemical_identities'].values()),
                card_sha256=hashlib.sha256(json.dumps(public['background'], sort_keys=True).encode()).hexdigest())


STREAMS = tuple(range(24))
MAXIMUM_ATTEMPTS, MAXIMUM_TOKENS = 250, 6_000_000


def spent():
    results = list((transport.BASE/'requests/screen').glob('**/api_result.json'))
    return len(results), sum(read(p)['usage'].get('total_tokens', 0) for p in results)


NATIVE_REQUEST = transport.request
STEP4_ATTEMPTS, STEP4_TOKENS = 400, 10_000_000


def spent4():
    results = [p for arm in ('h0', 'r1') for p in (transport.BASE/'requests/screen'/arm).glob('**/api_result.json')]
    return len(results), sum(read(p)['usage'].get('total_tokens', 0) for p in results)


def guarded(namespace, stage, system, public, validator):
    if not (transport.BASE/'requests'/namespace/stage/'validated.json').exists():
        attempts, tokens = spent()
        step4 = namespace.startswith(('screen/h0/', 'screen/r1/'))
        if step4:
            a4, t4 = spent4()
            if a4 >= STEP4_ATTEMPTS or t4 >= STEP4_TOKENS:
                raise RuntimeError(f'Step 4 budget stop: {a4} attempts, {t4} tokens')
        elif attempts >= MAXIMUM_ATTEMPTS or tokens >= MAXIMUM_TOKENS:
            raise RuntimeError(f'Budget stop: {attempts} attempts, {tokens} tokens')
    return NATIVE_REQUEST(namespace, stage, system, public, validator)


WORKER_TASK = None


def setup(short):
    """Process initializer: one task per process pool, so cards cannot mix."""
    global WORKER_TASK
    frozen.initialize()
    proposal.background = card_for(short)
    WORKER_TASK = short


def first_draw(job):
    """Step 1: one independent zero-data pick through the unchanged CPA first-point prompts. No outcome is read."""
    short, stream = job
    path = BASE/short/'draws'/f's{stream}.json'
    if path.exists():
        return dict(task=short, stream=stream, first=read(path)['candidate_id'], status='CACHED')
    pool = pool_for(short)
    assert proposal.background.__qualname__.startswith('card_for') and WORKER_TASK == short, 'worker set up for another task'
    space = f'screen/first/{short}/s{stream}'
    first = guarded(space, 'propose', proposal.PROPOSE_SYSTEM, proposal.public(pool), lambda raw: proposal.validate_proposal(raw, pool))
    candidate, choice = first['plan']['exact_candidate_id'], first['plan']['choice']
    if candidate is None:
        confirmed = guarded(space, 'confirm', proposal.CONFIRM_SYSTEM, proposal.confirmation(pool, first),
                            lambda raw: proposal.validate_confirmation(raw, pool, first['plan']['neighbors']))
        candidate, choice = confirmed['plan']['choice']['candidate_id'], confirmed['plan']['choice']
    write(path, dict(candidate_id=candidate, conditions=deepcopy(pool.points[pool.ix[candidate]]), rationale=choice['rationale'],
                     assumptions=deepcopy(choice['key_assumptions'])))
    return dict(task=short, stream=stream, first=candidate, status='COMPLETE')


def check_pilot(short, stream):
    requests = list((transport.BASE/f'requests/screen/first/{short}/s{stream}').glob('**/request.json'))
    marker = TASKS[short]['reaction'][:40]
    print(json.dumps(dict(task=short, stream=stream, draw_written=(BASE/short/'draws'/f's{stream}.json').exists(), requests=len(requests),
                          requests_with_card=sum(marker in p.read_text(encoding='utf-8') for p in requests),
                          oracle_or_source_named=sum(any(w in p.read_text(encoding='utf-8').lower() for w in ('currybo', 'cernak', 'nielsen', 'oracle', 'raw_values'))
                                                     for p in requests),
                          spent=spent())), flush=True)


def predict():
    """Step 2: mean mid-rank pool percentile of each task's 24 picks, and the registered rule. Written before any BO run."""
    out = BASE/'predictions.json'
    assert not out.exists(), 'predictions are written once'
    assert not any((BASE/s/'runs').exists() for s in TASKS), 'a BO run exists: predictions must come first'
    tasks = {}
    for short in TASKS:
        pool = pool_for(short)
        values = np.load(BASE/short/'input/oracle.npz')['values']
        picks = [read(BASE/short/'draws'/f's{t}.json')['candidate_id'] for t in STREAMS]
        pct = [100*(np.mean(values < values[pool.ix[p]]) + 0.5*np.mean(values == values[pool.ix[p]])) for p in picks]
        mean = float(np.mean(pct))
        rule = 'gain' if mean >= 90 else 'no gain' if mean < 80 else 'no prediction'
        tasks[short] = dict(mean_pick_percentile=round(mean, 2), prediction=rule, draws=len(picks),
                            distinct_picks=dict(Counter(picks).most_common()), pick_percentiles={p: round(x, 2) for p, x in zip(picks, pct)})
    record = dict(rule='mean mid-rank pick percentile >= 90: gain; < 80: no gain; otherwise no prediction',
                  registration='docs/hypothesis_scope_v2_registration_20261105.md, New-task screen', tasks=tasks, spent=spent())
    write(out, record)
    return dict(predictions_sha256=sha(out), **{s: {k: v for k, v in t.items() if k != 'pick_percentiles'} for s, t in tasks.items()})


PROTOCOL = 'new_task_screen_20261107'


def setup_bo(short):
    """Process initializer for step 3: the registered BO recipe on one task, with its own fit cache."""
    from hypothesis_scope_multiseed import study as multi
    global WORKER_TASK
    frozen.initialize()
    task = TASKS[short]['task']
    frozen.SHORT[task] = short
    frozen.pool_for = lambda t: pool_for(short)
    multi.BASE = v2.BASE = BASE/short  # Pool sits in input/, cache in BASE/short/cache.
    WORKER_TASK = short


def run_folder(short, group, stream):
    return BASE/short/'runs'/group/f's{stream}'


def history(short, group, stream):
    return [r['candidate_id'] for r in read(run_folder(short, group, stream)/'state.json')['history']]


def run_bo(job):
    """Step 3: BO from a fixed first point ('llm', 'identity') or from its own random first point ('bo'), on tie stream t."""
    from hypothesis_scope_multiseed import study as multi
    short, group, stream, point = job
    assert WORKER_TASK == short
    task = TASKS[short]['task']
    target = run_folder(short, group, stream)
    if (target/'state.json').exists() and read(target/'state.json')['status'] == 'COMPLETE':
        return dict(task=short, group=group, stream=stream, status='CACHED')
    multi.folder_for = lambda t, s, a: target
    if group == 'bo':
        multi.run_control(task, stream, 'bo', dict(protocol_sha256=PROTOCOL, initial_priors={}))
    else:
        pool = pool_for(short)
        locked = dict(candidate_id=point, conditions=deepcopy(pool.points[pool.ix[point]]), rationale='fixed by design', assumptions=[])
        frozen.locked_first = lambda t: locked
        multi.run_control(task, stream, 'first', dict(protocol_sha256=PROTOCOL, initial_priors={task: dict(first_input=locked)}))
    return dict(task=short, group=group, stream=stream, status='COMPLETE')


def bo_phase(short, jobs, workers):
    with ProcessPoolExecutor(max_workers=workers, initializer=setup_bo, initargs=(short,)) as executor:
        futures = [executor.submit(run_bo, j) for j in jobs]
        for future in as_completed(futures):
            future.result()


def step3(tasks, workers):
    assert (BASE/'predictions.json').exists(), 'predictions must be written before any BO run'
    for short in tasks:
        bo_phase(short, [(short, 'bo', t, None) for t in STREAMS], workers)
        bo_phase(short, [(short, 'identity', t, history(short, 'bo', t)[0]) for t in STREAMS], workers)
        same = [history(short, 'bo', t) == history(short, 'identity', t) for t in STREAMS]
        print(json.dumps(dict(task=short, identity_repeats=sum(same), of=len(same))), flush=True)
        assert all(same), 'identity check failed: fixed-first BO does not repeat pure BO'
        bo_phase(short, [(short, 'llm', t, read(BASE/short/'draws'/f's{t}.json')['candidate_id']) for t in STREAMS], workers)
        print(json.dumps(dict(task=short, step=3, status='COMPLETE')), flush=True)


class StopAfterCommit(Exception):
    """Raised once the query-1 commit, which holds the pre-data hypotheses, is written."""


def setup_controller(short):
    """Process initializer for step 4: the BO setup, the task card, and budget-guarded transport."""
    setup_bo(short)
    proposal.background = card_for(short)
    transport.request = lambda ns, stage, system, public, validator: guarded(ns if ns.startswith('screen/') else f'screen/{ns.split("/")[0]}/{short}/{ns.split("/", 1)[1]}',
                                                                              stage, system, public, validator)


def hypotheses(short, stream):
    """Arm h0: the frozen controller from draw t's pick, stopped after its query-1 commit. No outcome is read."""
    from unittest.mock import patch
    from contextlib import ExitStack
    task = TASKS[short]['task']
    locked = read(BASE/short/'draws'/f's{stream}.json')
    target = run_folder(short, 'h0', stream)
    if (target/'rounds/01_commit.json').exists():
        return
    control = dict(protocol_sha256=PROTOCOL, initial_priors={task: dict(first_input=locked)})

    def stop(pool, folder, state, commit):
        raise StopAfterCommit

    with ExitStack() as stack:
        for name, value in (('check', lambda *a, **kw: control), ('BASE', BASE/short), ('locked_first', lambda t: locked),
                            ('folder_for', lambda t: target), ('apply_outcome', stop)):
            stack.enter_context(patch.object(frozen, name, value))
        stack.enter_context(patch.object(frozen.transport, 'request',
                                         lambda s, stage, system, public, validator: guarded(f'screen/h0/{short}/s{stream}', stage, system, public, validator)))
        try:
            frozen.run_one(task)
        except StopAfterCommit:
            pass
    assert (target/'rounds/01_commit.json').exists() and not (target/'state.json').exists()


def llm_prefix(short):
    """Commit draw t's own pick under draw t's pre-data hypotheses, then observe it (CPA's shared_prefix, with the LLM pick as the point)."""
    from hypothesis_scope_loop import knowledge as k

    def prefix(task, stream, arm):
        pool = pool_for(short)
        scopes = k.Scopes(pool)
        source = read(run_folder(short, 'h0', stream)/'rounds/01_commit.json')
        j = pool.ix[read(BASE/short/'draws'/f's{stream}.json')['candidate_id']]
        assert source['decision']['selected_id'] == pool.ids[j]
        state = v2.header(task, stream, arm)
        state.update(history=[], decisions=[], notes=[], archive=[], calls=[], permission_events=[])
        work = deepcopy(source['working_knowledge'])
        decision = dict(deepcopy(source['decision']), selected_id=pool.ids[j], conditions=deepcopy(pool.points[j]), LLM_preferred=None, LLM_projection=None,
                        reason='llm_first_with_draw_hypotheses', locked_forecasts=k.freeze_forecasts(pool, scopes, work['notes'], j, 1, None, None, {}))
        commit = dict(query=1, prestate_sha256=frozen.digest(state), decision=decision, working_knowledge=work)
        commit['commit_sha256'] = frozen.digest(commit)
        target = v2.folder_for(task, stream, arm)
        write(target/'rounds/01_commit.json', commit)
        frozen.apply_outcome(pool, target, state, commit)
        return state
    return prefix


def run_controller(job):
    """Arm r1: the unchanged controller with live refreshes, from the LLM pick, on tie stream t."""
    short, stream = job
    assert WORKER_TASK == short
    hypotheses(short, stream)
    task = TASKS[short]['task']
    frozen.locked_first = lambda t: None  # Read only before the first observation, and this arm starts after it.
    v2.folder_for = lambda t, s, a: run_folder(short, a, s)
    if 'r1' not in v2.EVIDENCE_FIRST_ARMS:
        v2.EVIDENCE_FIRST_ARMS = v2.EVIDENCE_FIRST_ARMS + ('r1',)
        v2.LIVE_ARMS = v2.LIVE_ARMS + ('r1',)
    v2.prefix_from_first = llm_prefix(short)
    v2.run_arm(task, stream, 'r1')
    return dict(task=short, stream=stream, status='COMPLETE')


def check_controller_pilot(short, stream):
    state = read(run_folder(short, 'r1', stream)/'state.json')
    d2 = read(run_folder(short, 'r1', stream)/'rounds/02_commit.json')['decision']
    requests = [p for arm in ('h0', 'r1') for p in (transport.BASE/f'requests/screen/{arm}/{short}').glob('**/request.json')]
    marker = TASKS[short]['reaction'][:40]
    print(json.dumps(dict(task=short, stream=stream, status=state['status'],
                          first_matches=state['history'][0]['candidate_id'] == read(BASE/short/'draws'/f's{stream}.json')['candidate_id'],
                          q2_reference_matches=d2['unrestricted_BO_id'] == history(short, 'llm', stream)[1],
                          calls=len(state['calls']), requests=len(requests), requests_with_card=sum(marker in p.read_text(encoding='utf-8') for p in requests),
                          step4_spent=spent4()), indent=1), flush=True)


def step4(tasks, streams, workers):
    for short in tasks:
        with ProcessPoolExecutor(max_workers=workers, initializer=setup_controller, initargs=(short,)) as executor:
            futures = [executor.submit(run_controller, (short, t)) for t in streams]
            for done, future in enumerate(as_completed(futures), 1):
                try:
                    print(json.dumps(dict(done=done, of=len(futures), **future.result())), flush=True)
                except Exception:
                    for other in futures:
                        other.cancel()
                    raise
    print(json.dumps(dict(status='COMPLETE', step=4, step4_spent=spent4())), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('pool', 'card', 'draws', 'pilot', 'spent', 'predict', 'bo', 'controller', 'controller_pilot'))
    parser.add_argument('--tasks', nargs='*', default=list(TASKS))
    parser.add_argument('--streams', nargs='*', type=int, default=list(STREAMS))
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if args.action == 'spent':
        return print(json.dumps(spent()))
    if args.action == 'pilot':
        return check_pilot(args.tasks[0], args.streams[0])
    frozen.initialize()
    if args.action == 'predict':
        return print(json.dumps(predict(), indent=1))
    if args.action == 'bo':
        return step3(args.tasks, args.workers)
    if args.action == 'controller':
        return step4(args.tasks, args.streams, args.workers)
    if args.action == 'controller_pilot':
        return check_controller_pilot(args.tasks[0], args.streams[0])
    if args.action in ('pool', 'card'):
        for short in args.tasks:
            print(json.dumps(prepare(short) if args.action == 'pool' else show_card(short)), flush=True)
        return
    for short in args.tasks:
        (BASE/short/'draws').mkdir(parents=True, exist_ok=True)
    for short in args.tasks:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=setup, initargs=(short,)) as executor:
            futures = [executor.submit(first_draw, (short, t)) for t in args.streams]
            for done, future in enumerate(as_completed(futures), 1):
                try:
                    print(json.dumps(dict(done=done, of=len(futures), **future.result())), flush=True)
                except Exception:
                    for other in futures:
                        other.cancel()
                    raise
    print(json.dumps(dict(status='COMPLETE', step='draws', spent=spent())), flush=True)
