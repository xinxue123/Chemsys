"""Direct Arylation rerun with the fixed substrates named on the task card.

The original held-out card hid the substrates, unlike the development cards. This study changes
only that card and redoes every arm that reads it, with new API calls and new folders. The pool,
GP, acquisition, budget, streams, prompts and model are the registered ones. Frozen held-out
records are read by nothing here and stay as they are.

Substrates and fixed conditions are from Shields et al., Nature 590, 89-96 (2021), Fig. 4a.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import argparse
import json
from chemical_hypothesis_bo import background as original
from grounded_baselines import study as gb
from grounded_first_start import proposal
from hypothesis_scope_multiseed import study as multi
from . import VARIANT, contrast_search as c, gp_followups as f, heldout as h, heldout_lgbo as hl, study as v2, transport

NAMED = v2.BASE/'heldout_named'
HIDDEN_CARD = h.background
NATIVE_REQUEST = transport.request
REACTANTS = (dict(role='azole C-H coupling partner', name='1-methyl-1H-imidazole-4-carbonitrile', SMILES='Cn1cnc(C#N)c1'),
             dict(role='aryl bromide', name='1-bromo-2-fluorobenzene', SMILES='Fc1ccccc1Br'))
BASELINES = ('lgbo', 'rbo')
LOCAL_ARM = 'n6m1'


def card(pool):
    """The hidden card plus the reactants and fixed conditions, in the development format."""
    public = HIDDEN_CARD(pool)
    public['fixed_reactants'] = [dict(row, structure_status='transcribed from published reaction scheme', graph=original.molecular_graph(row['SMILES'])) for row in REACTANTS]
    public['substrate_scope'] = dict(public['substrate_scope'], identity_status='fixed reactant names and structures supplied',
                                     statement='The two listed reactants are fixed; do not change substrate identity.')
    public['fixed_conditions'] = dict(palladium_source='[PdCl(allyl)]2', palladium_source_loading_mol_percent=2.25, ligand_loading_mol_percent=5.,
                                      base_equivalents=3., note='These are source experimental definitions; only the five registered variables may change.')
    public['unknown_fixed_details'] = ['reaction time and ancillary fixed protocol details are not specified; do not vary or invent them']
    return public


def request(namespace, stage, system, public, validator):
    return NATIVE_REQUEST(namespace.replace('heldout_', 'named_', 1), stage, system, public, validator)


def install():
    """Point the held-out study at the new card, folders and request records."""
    h.BASE, h.background, transport.request = NAMED, card, request


def core_worker():
    h.initialize()
    install()


def baseline_worker():
    install()
    hl.configure()
    gb.path_for = lambda t, s, a: NAMED/'runs'/a/f's{s}'/h.SHORT


def local_worker():
    install()
    c.configure()
    multi.BASE = v2.BASE = NAMED
    v2.folder_for = lambda t, s, a: NAMED/'runs'/a/f's{s}'/h.SHORT


def run_baseline(stream, arm):
    p = dict(protocol_sha256='heldout_named_' + VARIANT, shared_first={h.TASK: hl.sealed_first(stream)}, shared_blinded_priors={h.TASK: hl.sealed_prior(stream)})
    gb.check = lambda protected=False: p
    gb.JOBS = ((h.TASK, stream, arm),)
    return gb.run_one(h.TASK, stream, arm)


def run_local(stream):
    return c.run_local(f.ARYL, stream, LOCAL_ARM)


def show():
    """Write the card and its first prompt without any API call."""
    h.initialize()
    install()
    proposal.background = h.background
    pool = h.pool_for()
    NAMED.mkdir(parents=True, exist_ok=True)
    public = proposal.public(pool)
    h.write(NAMED/'named_card.json', public['background'], replace=True)
    hidden = HIDDEN_CARD(pool)
    print(json.dumps(dict(added=sorted(set(public['background']) - set(hidden)),
                          changed=sorted(a for a in hidden if hidden[a] != public['background'][a]),
                          reactants=public['background']['fixed_reactants'][0]['name'], candidates=len(pool.ids))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('card', 'core', 'baselines', 'local'))
    parser.add_argument('--seeds', nargs='*', type=int, default=list(range(8)))
    parser.add_argument('--arms', nargs='*', choices=BASELINES, default=list(BASELINES))
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.action == 'card':
        show()
        return
    worker, jobs = dict(core=(core_worker, [(h.run_stream, s, []) for s in args.seeds]),
                        baselines=(baseline_worker, [(run_baseline, s, a) for a in args.arms for s in args.seeds]),
                        local=(local_worker, [(run_local, s) for s in args.seeds]))[args.action]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=worker) as executor:
        for future in as_completed([executor.submit(*job) for job in jobs]):
            print(json.dumps(future.result()), flush=True)
