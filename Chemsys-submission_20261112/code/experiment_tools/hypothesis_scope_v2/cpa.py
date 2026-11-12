"""Held-out test on CPA: plain BO against the controller from 24 shared starts.

CPA is the one dataset in the bundled archive that this research program has not used. Its source
column is labelled yield but holds enantioselectivity as ddG (kcal/mol, 298 K); the maximum equals
RT ln 199, which is 99% ee. Pair t runs on tie stream t from pure BO's first point of stream
(t - 1) mod 24. The controller in pair t uses pair t's own pre-data hypotheses, written after the
LLM's own first pick and before any outcome. Rules are in the registration, under "CPA held-out test".

Frozen modules supply the GP, acquisition, selection, knowledge state, prompts and transport, and
are not edited. This module adds only the pool adapter, the card, folders and a budget stop.

Free steps (no model calls): card, pool, bo, identity, plain.
Paid steps (need the user's go-ahead): draws, controller.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from copy import deepcopy
from unittest.mock import patch
import argparse
import csv
import io
import json
import zipfile
import numpy as np
from chem.benchmarks import knowledge_gp as native
from chem.benchmarks.knowledge_domain_context import task_context
from chemical_hypothesis_bo import background as original
from grounded_first_start import proposal
from hypothesis_scope_loop import study as frozen, knowledge as k
from hypothesis_scope_multiseed import study as multi
from . import study as v2, transport

TASK, SHORT = 'CPA_Thiol_Imine', 'cpa'
AXES = ('Catalyst', 'Imine', 'Thiol')
ROOT = frozen.ROOT
BASE = v2.BASE/'cpa'
INPUT = BASE/'input'  # The GP adapter refuses a cache inside the folder that holds the pool.
ARCHIVE = ROOT/'code/chemist_system/third_party/reasoning_bo_official/data/chemistry_data.zip'
PROTOCOL = 'cpa_heldout_20261107'
PAIRS = tuple(range(24))
CONTROLLER, DRAW = 'r1', 'h0'
MAXIMUM_ATTEMPTS, MAXIMUM_TOKENS = 250, 4_000_000
NATIVE_REQUEST = transport.request
read, write, sha = frozen.read, frozen.write, frozen.sha
frozen.SHORT[TASK] = SHORT


class StopAfterCommit(Exception):
    """Raised once the query-1 commit, which holds the pre-data hypotheses, is written."""


def prepare():
    path = INPUT/'pool.json'
    if path.exists():
        return read(INPUT/'receipt.json')
    with zipfile.ZipFile(ARCHIVE) as handle:
        data = handle.read('CPA.csv')
    seen, points, values = set(), [], []
    for row in csv.DictReader(io.StringIO(data.decode('utf-8-sig'))):
        point = {a: str(row[a]) for a in AXES}
        key = tuple(point[a] for a in AXES)
        assert key not in seen, 'duplicate input row'
        seen.add(key)
        points.append(point)
        values.append(float(row['yield']))
    raw = np.asarray(values)
    top = float(raw.max())
    assert len(points) == 1075 and abs(top - 3.134624572) < 1e-9
    write(path, dict(task_id=TASK, reaction_class='Chiral phosphoric acid catalysed thiol addition to N-acyl imines', axes=list(AXES),
                     ids=[f'c{j:05d}' for j in range(len(points))], points=points, levels={a: sorted({p[a] for p in points}) for a in AXES},
                     fixed_substrates={}, objective_unit='kcal/mol', objective_scale=top, source_member='CPA.csv', source_archive_sha256=sha(ARCHIVE)))
    np.savez_compressed(INPUT/'oracle.npz', values=raw/top, raw_values=raw)
    receipt = dict(rows=len(points), unique_inputs=len(seen), pool_sha256=sha(path), oracle_sha256=sha(INPUT/'oracle.npz'),
                   source_archive_sha256=sha(ARCHIVE), maximum_raw=top, tied_at_maximum=int((raw == top).sum()), negative=int((raw < 0).sum()),
                   outcome_access='adapter conversion and evaluation only; decision agents see queried outcomes only')
    write(INPUT/'receipt.json', receipt)
    return receipt


def pool_for(task=TASK):
    path = INPUT/'pool.json'
    p = read(path)
    pool = native.Pool(TASK, 'P1', tuple(p['ids']), tuple(p['axes']), ('categorical',)*len(p['axes']), tuple(p['points']),
                       tuple(tuple(x[a] for a in p['axes']) for x in p['points']), tuple(p['ids'][:4]),
                       p['objective_unit'], float(p['objective_scale']), p, ROOT, path)
    pool.initial_ids = ()
    return pool


def card(pool):
    """Neutral card in the grounded format; it does not name the benchmark, source or any result."""
    policy = task_context('Doyle_Buchwald_Hartwig')
    meaning = dict(Catalyst='Chiral phosphoric acid catalyst identity.', Imine='N-acyl imine electrophile identity.', Thiol='Thiol nucleophile identity.')
    inputs = []
    for axis in AXES:
        row = dict(name=axis, kind='categorical', meaning=meaning[axis], levels=list(pool.raw_pool['levels'][axis]))
        row['chemical_identities'] = {v: dict(name=original.COMMON_NAMES.get(v, 'name not assigned; exact molecular graph supplied'),
                                              SMILES=v, graph=original.molecular_graph(v)) for v in row['levels']}
        inputs.append(row)
    return dict(group='P1', reaction_or_process='Enantioselective addition of a thiol to an N-acyl imine, catalysed by a chiral phosphoric acid',
                description='Choose one catalyst, one imine and one thiol from the supplied measured combinations.',
                objective=dict(name='enantioselectivity_ddG', unit='kcal/mol', direction='maximize',
                               meaning=('Enantioselectivity expressed as the free-energy difference ddG between the two enantiomeric pathways, '
                                        'in kcal/mol at 298 K. Higher is better. A negative measured value means the opposite enantiomer is favoured. '
                                        'Forecast means must be zero or positive in this interface, so use 0 for no or reversed selectivity.')),
                substrate_scope=dict(substrates_are_optimization_variables=True, identity_status='imine and thiol identities supplied as exact structures',
                                     statement='The imine and thiol are optimization variables chosen from the supplied lists.'),
                inputs=inputs, candidate_policy=policy['candidate_policy'], knowledge_policy=policy['knowledge_policy'],
                candidate_count=len(pool.ids),
                identity_evidence='Curated experimental definitions and supplied input structures; no benchmark performance conclusions.',
                unknown_fixed_details=['solvent, temperature, loadings, time and ancillary fixed protocol details are not specified; do not vary or invent them'],
                reasoning_requirements=['State a conditional, falsifiable chemical or process relationship using meaningful identities and physical units.',
                                        'Separate general mechanistic expectations from associations supported by current own observations.',
                                        'A source molecular identity is not a source performance claim. Do not recall published benchmark results.',
                                        'Candidate IDs identify transport records only. Use supplied full physical conditions to reason.'])


def spent():
    results = list((transport.BASE/'requests/cpa').glob('**/api_result.json'))
    return len(results), sum(read(p)['usage'].get('total_tokens', 0) for p in results)


def request(namespace, stage, system, public, validator):
    name = namespace if namespace.startswith('cpa/') else 'cpa/' + namespace
    if not (transport.BASE/'requests'/name/stage/'validated.json').exists():
        attempts, tokens = spent()
        if attempts >= MAXIMUM_ATTEMPTS or tokens >= MAXIMUM_TOKENS:
            raise RuntimeError(f'Budget stop: {attempts} attempts, {tokens} tokens')
    return NATIVE_REQUEST(name, stage, system, public, validator)


def configure():
    """Worker setup shared by every step."""
    frozen.initialize()
    frozen.pool_for = pool_for
    proposal.background = card
    transport.request = request
    multi.BASE = v2.BASE = BASE  # One fit cache, so identical prefixes reuse identical fits.


def folder(group, point, stream):
    return BASE/group/point/f's{stream}'


def picks(path):
    return [r['candidate_id'] for r in read(path/'state.json')['history']]


def bo_folder(stream):
    return BASE/'runs/bo'/f's{stream}'/SHORT


def point_of(pair):
    """The pair's shared first point: pure BO's first point of the previous stream."""
    return picks(bo_folder((pair - 1) % 24))[0]


def run_bo(stream):
    multi.folder_for = lambda t, s, a: bo_folder(s)
    return multi.run_control(TASK, stream, 'bo', dict(protocol_sha256=PROTOCOL, initial_priors={}))


def run_fixed(point, stream, group):
    pool = pool_for()
    locked = dict(candidate_id=point, conditions=deepcopy(pool.points[pool.ix[point]]), rationale='fixed by design', assumptions=[])
    frozen.locked_first = lambda t: locked
    multi.folder_for = lambda t, s, a: folder(group, point, s)
    control = dict(protocol_sha256=PROTOCOL, initial_priors={TASK: dict(first_input=locked)})
    return dict(multi.run_control(TASK, stream, 'first', control), first=point, group=group)


def first_draw(pair, pool):
    """An independent zero-data pick, through the unchanged prompts."""
    path = BASE/'draws'/f's{pair}.json'
    if path.exists():
        return read(path)
    space = f'cpa/first/s{pair}_{SHORT}'
    first = transport.request(space, 'propose', proposal.PROPOSE_SYSTEM, proposal.public(pool), lambda raw: proposal.validate_proposal(raw, pool))
    candidate, choice = first['plan']['exact_candidate_id'], first['plan']['choice']
    if candidate is None:
        confirmed = transport.request(space, 'confirm', proposal.CONFIRM_SYSTEM, proposal.confirmation(pool, first),
                                      lambda raw: proposal.validate_confirmation(raw, pool, first['plan']['neighbors']))
        candidate, choice = confirmed['plan']['choice']['candidate_id'], confirmed['plan']['choice']
    locked = dict(candidate_id=candidate, conditions=deepcopy(pool.points[pool.ix[candidate]]), rationale=choice['rationale'],
                  assumptions=deepcopy(choice['key_assumptions']))
    write(path, locked)
    return locked


def run_draw(pair):
    """The LLM's first pick and its pre-data hypotheses. Stops after the query-1 commit, so no outcome is read."""
    pool = pool_for()
    locked = first_draw(pair, pool)
    target = BASE/'runs'/DRAW/f's{pair}'/SHORT
    if (target/'rounds/01_commit.json').exists():
        return dict(pair=pair, step='draw', first=locked['candidate_id'], status='COMPLETE')
    control = dict(protocol_sha256=PROTOCOL, initial_priors={TASK: dict(first_input=locked)})

    def stop(pool, folder, state, commit):
        raise StopAfterCommit

    with ExitStack() as stack:
        for name, value in (('check', lambda *a, **kw: control), ('BASE', BASE), ('locked_first', lambda t: locked),
                            ('folder_for', lambda t: target), ('apply_outcome', stop)):
            stack.enter_context(patch.object(frozen, name, value))
        stack.enter_context(patch.object(frozen.transport, 'request',
                                         lambda short, stage, system, public, validator: request(f'cpa/{DRAW}/s{pair}_{SHORT}', stage, system, public, validator)))
        try:
            frozen.run_one(TASK)
        except StopAfterCommit:
            pass
    assert (target/'rounds/01_commit.json').exists() and not (target/'state.json').exists()
    return dict(pair=pair, step='draw', first=locked['candidate_id'], status='COMPLETE')


def shared_prefix(task, pair, arm):
    """Commit the shared first point under pair t's pre-data hypotheses, then observe it."""
    pool = pool_for()
    scopes = k.Scopes(pool)
    source = read(BASE/'runs'/DRAW/f's{pair}'/SHORT/'rounds/01_commit.json')
    j = pool.ix[point_of(pair)]
    state = v2.header(task, pair, arm)
    state.update(history=[], decisions=[], notes=[], archive=[], calls=[], permission_events=[])
    work = deepcopy(source['working_knowledge'])
    decision = dict(deepcopy(source['decision']), selected_id=pool.ids[j], conditions=deepcopy(pool.points[j]), LLM_preferred=None, LLM_projection=None,
                    reason='shared_first_with_pair_hypotheses', locked_forecasts=k.freeze_forecasts(pool, scopes, work['notes'], j, 1, None, None, {}))
    commit = dict(query=1, prestate_sha256=frozen.digest(state), decision=decision, working_knowledge=work)
    commit['commit_sha256'] = frozen.digest(commit)
    target = v2.folder_for(task, pair, arm)
    write(target/'rounds/01_commit.json', commit)
    frozen.apply_outcome(pool, target, state, commit)
    return state


def run_controller(pair):
    frozen.locked_first = lambda task: None  # Read only before the first observation, and this arm starts after it.
    v2.folder_for = lambda t, s, a: BASE/'runs'/a/f's{s}'/SHORT
    v2.EVIDENCE_FIRST_ARMS = v2.EVIDENCE_FIRST_ARMS + (CONTROLLER,)
    v2.LIVE_ARMS = v2.LIVE_ARMS + (CONTROLLER,)
    v2.prefix_from_first = shared_prefix
    return v2.run_arm(TASK, pair, CONTROLLER)


def show_card():
    """Write the card and first prompt without any model call, and report its size."""
    pool = pool_for()
    proposal.background = card
    public = proposal.public(pool)
    write(BASE/'card.json', public['background'], replace=True)
    text = json.dumps(public, ensure_ascii=False)
    print(json.dumps(dict(candidates=len(pool.ids), levels={a: len(pool.raw_pool['levels'][a]) for a in AXES}, first_prompt_characters=len(text),
                          examples=len(public['input_only_examples']), unnamed=sum(v['name'].startswith('name not assigned') for x in public['background']['inputs']
                                                                                 for v in x['chemical_identities'].values()))), flush=True)


def check_identity():
    same = [picks(bo_folder(s)) == picks(folder('identity', picks(bo_folder(s))[0], s)) for s in PAIRS]
    print(json.dumps(dict(identity_repeats=sum(same), of=len(same), differ=[s for s, ok in enumerate(same) if not ok])), flush=True)
    assert all(same), 'identity check failed: stop before any model call'


def check_pilot(pair):
    point = point_of(pair)
    plain = picks(folder('plain', point, pair))
    state = read(BASE/'runs'/CONTROLLER/f's{pair}'/SHORT/'state.json')
    d = read(BASE/'runs'/CONTROLLER/f's{pair}'/SHORT/'rounds/02_commit.json')['decision']
    requests = list((transport.BASE/'requests/cpa').glob('**/attempt1/request.json'))
    attempts, tokens = spent()
    print(json.dumps(dict(pair=pair, point=point, status=state['status'], first_matches=state['history'][0]['candidate_id'] == point,
                          q2_reference_matches=d['unrestricted_BO_id'] == plain[1], calls=len(state['calls']), requests=len(requests),
                          requests_with_card=sum('chiral phosphoric acid' in p.read_text(encoding='utf-8') for p in requests),
                          attempts=attempts, tokens=tokens), indent=1), flush=True)


def phase(jobs, workers):
    with ProcessPoolExecutor(max_workers=workers, initializer=configure) as executor:
        futures = [executor.submit(*job) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            try:
                print(json.dumps(dict(done=done, of=len(futures), **future.result())), flush=True)
            except Exception:
                for other in futures:
                    other.cancel()
                raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('pool', 'card', 'bo', 'identity', 'plain', 'draws', 'controller', 'pilot', 'spent'))
    parser.add_argument('--pairs', nargs='*', type=int, default=list(PAIRS))
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    BASE.mkdir(parents=True, exist_ok=True)
    INPUT.mkdir(parents=True, exist_ok=True)
    if args.action == 'pool':
        return print(json.dumps(prepare(), indent=1))
    if args.action == 'card':
        return show_card()
    if args.action == 'spent':
        return print(json.dumps(spent()))
    if args.action == 'pilot':
        return check_pilot(args.pairs[0])
    if args.action == 'bo':
        phase([(run_bo, s) for s in args.pairs], args.workers)
    elif args.action == 'identity':
        phase([(run_fixed, picks(bo_folder(s))[0], s, 'identity') for s in args.pairs], args.workers)
        check_identity()
    elif args.action == 'plain':
        phase([(run_fixed, point_of(t), t, 'plain') for t in args.pairs], args.workers)
    elif args.action == 'draws':
        phase([(run_draw, t) for t in args.pairs], args.workers)
    else:
        assert all((BASE/'runs'/DRAW/f's{t}'/SHORT/'rounds/01_commit.json').exists() for t in args.pairs), 'run draws first'
        phase([(run_controller, t) for t in args.pairs], args.workers)
    print(json.dumps(dict(status='COMPLETE', step=args.action, spent=spent())), flush=True)
