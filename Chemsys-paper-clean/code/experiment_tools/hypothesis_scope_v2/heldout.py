"""Held-out confirmation on Direct Arylation, unseen by this controller family.

Each stream draws its own independent LLM first point. Arms sharing a stream share
that point: pure BO keeps its random start. The pool comes from the vendored public
table by the rule used in v13_late_validation. Decision agents see only a neutral
card and their own queried outcomes. Oracle values are read after each commit.
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
from hypothesis_scope_loop import study as frozen
from hypothesis_scope_multiseed import study as multi
from . import study as v2, transport

TASK, SHORT = 'Direct_Arylation', 'aryl'
AXES = ('Base_SMILES', 'Ligand_SMILES', 'Solvent_SMILES', 'Concentration', 'Temp_C')
ROOT = frozen.ROOT
BASE = v2.BASE/'heldout'
INPUT = BASE/'input'  # The GP adapter refuses a cache inside the folder that holds the pool.
read, write, sha = frozen.read, frozen.write, frozen.sha
frozen.SHORT[TASK] = SHORT


def prepare():
    path = INPUT/'pool.json'
    if path.exists():
        return
    archive = ROOT/'code/chemist_system/third_party/reasoning_bo_official/data/chemistry_data.zip'
    with zipfile.ZipFile(archive) as handle:
        data = handle.read('direct_arylation.csv')
    seen, points, values = set(), [], []
    for row in csv.DictReader(io.StringIO(data.decode('utf-8-sig'))):
        point = {a: str(row[a]) for a in AXES}
        key = tuple(point[a] for a in AXES)
        if key not in seen:
            seen.add(key)
            points.append(point)
            values.append(float(row['yield']))
    write(path, dict(task_id=TASK, reaction_class='Direct arylation', axes=list(AXES), ids=[f'a{j:05d}' for j in range(len(points))],
                     points=points, levels={a: sorted({p[a] for p in points}) for a in AXES}, fixed_substrates={},
                     objective_unit='reported reaction yield percent', source_member='direct_arylation.csv',
                     source_archive_sha256=sha(archive)))
    np.savez_compressed(INPUT/'oracle.npz', values=np.asarray(values)/100., raw_values=np.asarray(values))


def pool_for(task=TASK):
    path = INPUT/'pool.json'
    p = read(path)
    pool = native.Pool(TASK, 'P1', tuple(p['ids']), tuple(p['axes']), ('categorical',)*5, tuple(p['points']),
                       tuple(tuple(x[a] for a in p['axes']) for x in p['points']), tuple(p['ids'][:4]),
                       p['objective_unit'], 100., p, ROOT, path)
    pool.initial_ids = ()
    return pool


def background(pool):
    """Neutral card in the grounded format; it does not name the benchmark or any result."""
    policy = task_context('Doyle_Buchwald_Hartwig')
    meaning = dict(Base_SMILES='Base and counterion identity.', Ligand_SMILES='Phosphine ligand identity.',
                   Solvent_SMILES='Solvent identity.', Concentration='Reactant concentration in mol/L, as discrete settings.',
                   Temp_C='Reaction temperature in degC, as discrete settings.')
    inputs = []
    for axis in AXES:
        row = dict(name=axis, kind='categorical', meaning=meaning[axis], levels=list(pool.raw_pool['levels'][axis]))
        if axis.endswith('SMILES'):
            row['chemical_identities'] = {v: dict(name=original.COMMON_NAMES.get(v, 'name not assigned; exact molecular graph supplied'),
                                                  SMILES=v, graph=original.molecular_graph(v)) for v in row['levels']}
        inputs.append(row)
    return dict(group='P1', reaction_or_process='Palladium-catalysed direct C-H arylation',
                description='Choose base, phosphine ligand, solvent, concentration and temperature from the supplied measured combinations.',
                objective=dict(name='reaction_yield', unit='percent', direction='maximize', meaning='Reported reaction yield percentage.'),
                substrate_scope=dict(substrates_are_optimization_variables=False,
                                     identity_status='fixed coupling partners are not specified in the supplied public domain',
                                     statement='Substrates are fixed and unspecified. Do not invent them.'),
                inputs=inputs, candidate_policy=policy['candidate_policy'], knowledge_policy=policy['knowledge_policy'],
                candidate_count=len(pool.ids),
                identity_evidence='Curated experimental definitions and supplied input structures; no benchmark performance conclusions.',
                unknown_fixed_details=['substrates, precursor loading, time and ancillary fixed protocol details are not specified; do not vary or invent them'],
                reasoning_requirements=['State a conditional, falsifiable chemical or process relationship using meaningful identities and physical units.',
                                        'Separate general mechanistic expectations from associations supported by current own observations.',
                                        'A source molecular identity is not a source performance claim. Do not recall published benchmark results.',
                                        'Candidate IDs identify transport records only. Use supplied full physical conditions to reason.'])


def first_draw(seed, pool):
    """An independent zero-data pick for this stream, through the unchanged prompts."""
    space = f'heldout_first/s{seed}_{SHORT}'
    first = transport.request(space, 'propose', proposal.PROPOSE_SYSTEM, proposal.public(pool),
                              lambda raw: proposal.validate_proposal(raw, pool))
    candidate = first['plan']['exact_candidate_id']
    choice = first['plan']['choice']
    if candidate is None:
        confirmed = transport.request(space, 'confirm', proposal.CONFIRM_SYSTEM, proposal.confirmation(pool, first),
                                      lambda raw: proposal.validate_confirmation(raw, pool, first['plan']['neighbors']))
        candidate, choice = confirmed['plan']['choice']['candidate_id'], confirmed['plan']['choice']
    return dict(candidate_id=candidate, conditions=deepcopy(pool.points[pool.ix[candidate]]), rationale=choice['rationale'],
                assumptions=deepcopy(choice['key_assumptions']))


def run_stream(seed, arms):
    frozen.pool_for = pool_for
    proposal.background = background
    pool = pool_for()
    locked = first_draw(seed, pool)
    frozen.locked_first = lambda task: locked
    control = dict(protocol_sha256='heldout_' + v2.VARIANT, initial_priors={TASK: dict(first_input=locked)})
    multi.BASE = BASE
    v2.BASE = BASE  # All arms share one fit cache, so identical prefixes reuse identical fits.
    multi.folder_for = lambda t, s, a: BASE/'runs'/a/f's{s}'/SHORT
    v2.folder_for = lambda t, s, a: BASE/'runs'/a/f's{s}'/SHORT
    v2.source_folder = lambda t, s, kind: BASE/'runs'/('h1' if kind == 'hyp' else 'first')/f's{s}'/SHORT
    for arm in ('bo', 'first'):
        multi.run_control(TASK, seed, arm, control)
    gp = lambda p, path, seed=0: multi.SeededGP(p, BASE/'cache'/SHORT, tie_seed=stream)
    stream = seed
    request = lambda short, stage, system, public, validator: transport.request(f'heldout_h1/s{stream}_{SHORT}', stage, system, public, validator)
    with ExitStack() as stack:
        for name, value in (('check', lambda *a, **kw: control), ('BASE', BASE), ('ColdGP', gp),
                            ('folder_for', lambda t: BASE/'runs'/'h1'/f's{stream}'/SHORT)):
            stack.enter_context(patch.object(frozen, name, value))
        stack.enter_context(patch.object(frozen.transport, 'request', request))
        frozen.run_one(TASK)
    for arm in arms:
        v2.run_arm(TASK, seed, arm)
    return dict(seed=seed, status='COMPLETE', first=locked['candidate_id'])


def initialize():
    frozen.initialize()


def report(seeds, arms):
    top = float(np.load(INPUT/'oracle.npz')['raw_values'].max())
    rows = []
    for arm in ('bo', 'first', 'h1') + tuple(arms):
        for seed in seeds:
            path = BASE/'runs'/arm/f's{seed}'/SHORT/'state.json'
            if path.exists() and read(path)['status'] == 'COMPLETE':
                best = np.maximum.accumulate([r['raw_value'] for r in read(path)['history']])
                rows.append(dict(arm=arm, seed=seed, first_id=read(path)['history'][0]['candidate_id'], first_value=read(path)['history'][0]['raw_value'],
                                 SearchAUC=float(best.mean()), final_best=float(best[-1]), auc_pct_of_max=100*float(best.mean())/top))
    with open(BASE/'per_seed_results.csv', 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arms', nargs='*', default=[])
    parser.add_argument('--seeds', nargs='*', type=int, default=list(range(8)))
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    BASE.mkdir(parents=True, exist_ok=True)
    prepare()
    with ProcessPoolExecutor(max_workers=min(8, args.workers), initializer=initialize) as executor:
        futures = [executor.submit(run_stream, s, args.arms) for s in args.seeds]
        for future in as_completed(futures):
            print(json.dumps(future.result()), flush=True)
    for row in report(args.seeds, args.arms):
        print(json.dumps(row), flush=True)
