"""Held-out LGBO comparator: the development loop, unchanged, on Direct Arylation.

Each stream shares the LLM first point and sealed pre-data hypotheses of the same stream.
Only the task inputs, paths and API transport point at the held-out study, so the decision
code is the frozen grounded-baselines loop.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import json
from grounded_baselines import compat, policy, study as gb
from grounded_first_start import proposal
from hypothesis_scope_loop import study as frozen
from . import heldout as h, transport, VARIANT

ARM = 'lgbo'


def sealed_prior(stream):
    """The stream's pre-data hypotheses, in the form the development loop uses."""
    commit = h.read(h.BASE/'runs/h1'/f's{stream}'/h.SHORT/'rounds/01_commit.json')
    saved = h.read(h.ROOT/Path(commit['working_knowledge']['calls'][0]['request']).parent.parent/'validated.json')
    keys = ('id', 'claim', 'direction', 'scope', 'forecast', 'falsification')
    return [{k: x[k] for k in keys} for x in saved['raw']['hypotheses']]


def sealed_first(stream):
    return h.read(h.BASE/'runs/first'/f's{stream}'/h.SHORT/'state.json')['decisions'][0]['decision']['first_source']


def configure():
    gb.initialize()
    policy.scores = compat.scores  # The audited transport for categorical pools, as in development.
    frozen.pool_for = h.pool_for
    proposal.background = h.background
    gb.BASE = h.BASE
    gb.path_for = lambda t, s, a: h.BASE/'runs'/ARM/f's{s}'/h.SHORT
    gb.transport.request = lambda name, stage, system, public, validator: transport.request(f'heldout_lgbo/{name}', stage, system, public, validator)


def run_stream(stream):
    p = dict(protocol_sha256='heldout_lgbo_' + VARIANT, shared_first={h.TASK: sealed_first(stream)}, shared_blinded_priors={h.TASK: sealed_prior(stream)})
    gb.check = lambda protected=False: p
    gb.JOBS = ((h.TASK, stream, ARM),)
    return gb.run_one(h.TASK, stream, ARM)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='*', type=int, default=list(range(8)))
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=min(8, args.workers), initializer=configure) as executor:
        for future in as_completed([executor.submit(run_stream, s) for s in args.seeds]):
            print(json.dumps(future.result()), flush=True)
