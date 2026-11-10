"""Arm q1: the claim-faithful controller with live refreshes, registered on 6 November 2026.

It starts from each stream's first point and sealed pre-data hypotheses in claim-faithful form.
Refresh triggers and call limits are the frozen ones. Only the prompt text changes: it adds a
contrast field and describes how contrasts act. No other arm is rerun.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import argparse
import json
from grounded_first_start import proposal
from hypothesis_scope_loop import study as frozen, knowledge as k, policy, prompts
from . import gp_followups as f, mechanism as m, study as v2

ARM = 'q1'
NATIVE_SYSTEM, NATIVE_PUBLIC, NATIVE_VALIDATE, NATIVE_INSTALL = prompts.SYSTEM, prompts.public, prompts.validate, k.install
NATIVE_BACKGROUND = proposal.background

SCHEMA_OLD = '"supporting_observation_ids":[observed IDs],"review_action":"new" or "keep" or "revise" or "retest"}]}.'
SCHEMA_NEW = ('"supporting_observation_ids":[observed IDs],"review_action":"new" or "keep" or "revise" or "retest",\n'
              '"contrast":{"axis":axis name,"higher":{"allowed":[...]} or {"bounds":[lo,hi]},"lower":same form,\n'
              '"conditions":{axis:same form}} or null}]}.')
SCOPE_OLD = '''All favour scopes are INTERSECTED;
all avoid scopes are SUBTRACTED. Choose compatible, useful, not needlessly narrow scopes;
their combined unmeasured region must contain at least 4 inputs.'''
SCOPE_NEW = '''A scope states where the claim and its forecast
apply. Favour scopes intersected, minus avoid scopes, must leave at least 4 unmeasured inputs,
and the preferred experiment must lie in that combined region.'''
RULES = '''AFTER THE FIRST: the unchanged native GP maximizes native qLogEI over the whole pool, minus
removed regions. GP predictions and training targets are never edited. Your preferred prototype
is a scientific proposal, NOT guaranteed to be executed. There is NO global qLogEI-gap gate.
Constraints come from contrasts. Give a hypothesis a contrast whenever it compares two sides of
one input axis: axis is the input that changes, higher is the side you claim gives the higher
objective, lower is the side you claim gives the lower objective, and conditions are the other
inputs held within stated values ({} if the claim holds for all of them). The software removes
ONLY the lower side under those conditions. It never restricts search to the higher side, so
state conditions only as far as the claim really applies. If a claim compares nothing, use
contrast=null: a favour hypothesis with contrast=null does not constrain search, and an avoid
hypothesis with contrast=null subtracts its scope. At most two queries per run are matched
tests: when a measured point inside a contrast's conditions has no matched partner on the other
side, the next query is that partner, with other inputs equal, or within 10% of their range for
continuous inputs. One contrary matched contrast pauses that rule. A pause does not establish
that the chemical claim is false. Calling a claim retained or refuted needs two independent
matched contrasts, and yield alone cannot identify an elementary step.

ON REFRESH: you receive ALL own experiments, locked predictions, matched contrast results,
permissions and event reasons. Retain, revise or withdraw claims from this evidence and propose
a new preferred point. Return only hypotheses you want to keep; omissions withdraw old claims.
You may abstain with hypotheses=[] and preferred_experiment=null. Renaming or rewording the same
finite-input region inherits its evidence. A rule paused by a contrary matched contrast does not
resume. New regions are new provisional claims, not validated discoveries. Supporting IDs must
refer only to provided own observations. Current GP rankings/scores are deliberately withheld
so your chemical hypotheses and preferred proposal are independent of them.'''


def build_system():
    assert SCHEMA_OLD in NATIVE_SYSTEM and SCOPE_OLD in NATIVE_SYSTEM
    text = NATIVE_SYSTEM.replace(SCHEMA_OLD, SCHEMA_NEW).replace(SCOPE_OLD, SCOPE_NEW)
    return text[:text.index('AFTER THE FIRST:')] + RULES


def translate(pool, scopes, draft, contrast):
    """A contrast becomes an avoid region on its claimed-lower side."""
    out = dict(deepcopy(draft), claimed_scope=deepcopy(draft['scope']), claimed_direction=draft['direction'])
    if contrast is None:
        out['inert'] = draft['direction'] == 'favour'
        return out
    if not isinstance(contrast, dict) or set(contrast) != {'axis', 'higher', 'lower', 'conditions'}:
        raise ValueError('A contrast needs axis, higher, lower and conditions')
    axis, conditions = contrast['axis'], contrast['conditions']
    if axis not in pool.axes or not isinstance(conditions, dict) or axis in conditions:
        raise ValueError('The contrast axis must be a real input and must not appear in conditions')
    low, high = dict(deepcopy(conditions), **{axis: contrast['lower']}), dict(deepcopy(conditions), **{axis: contrast['higher']})
    low_mask, high_mask = scopes.mask(low), scopes.mask(high)
    if not low_mask.any() or not high_mask.any() or (low_mask & high_mask).any() or low_mask.all():
        raise ValueError('Contrast sides must be non-empty and disjoint, and must leave some of the pool')
    out.update(direction='avoid', scope=low, contrast=dict(axis=axis, high=high, low=low, pairs=None, evidence=[], paused=False))
    return out


def validate(raw, pool, state, locked_first):
    if not isinstance(raw, dict) or not isinstance(raw.get('hypotheses'), list):
        raise ValueError('Expected summary, preferred_experiment, hypotheses')
    core, contrasts = deepcopy(raw), []
    for note in core['hypotheses']:
        if not isinstance(note, dict) or 'contrast' not in note:
            raise ValueError('Each hypothesis needs a contrast field, which may be null')
        contrasts.append(note.pop('contrast'))
    plan = NATIVE_VALIDATE(core, pool, state, locked_first)
    scopes = k.Scopes(pool)
    plan['drafts'] = [translate(pool, scopes, d, c) for d, c in zip(plan['drafts'], contrasts)]
    return plan


def display(note):
    """How a note reads to the LLM: as it was stated, not as it was translated."""
    out = dict(deepcopy(note), scope=deepcopy(note.get('claimed_scope', note['scope'])), direction=note.get('claimed_direction', note['direction']))
    record = out.pop('contrast', None)
    out['contrast'] = None
    if record:
        axis = record['axis']
        out['contrast'] = dict(axis=axis, higher=record['high'][axis], lower=record['low'][axis], conditions={a: s for a, s in record['low'].items() if a != axis})
    return out


def public(pool, state, event, locked_first):
    out = NATIVE_PUBLIC(pool, state, event, locked_first)
    out['current_hypotheses'] = [display(n) for n in state['notes']]
    out['withdrawn_or_revised_history'] = [display(n) for n in state['archive']]
    out['matched_contrast_ledger'] = [dict(id=n['id'], axis=n['contrast']['axis'], paused=n['contrast']['paused'], evidence=deepcopy(n['contrast']['evidence']))
                                      for n in state['notes'] if n.get('contrast')]
    out['scope_combination'] = 'Each eligible contrast removes its claimed-lower side. An avoid hypothesis with no contrast subtracts its scope. Favour scopes do not restrict search.'
    return out


def install(drafts, previous, archive, scopes, query, stagnated):
    """A rule paused by a contrary contrast stays paused, even if it returns under a new name."""
    notes, edits = NATIVE_INSTALL(drafts, previous, archive, scopes, query, stagnated)
    paused = {n['semantic_key'] for n in archive + previous if n.get('contrast', {}).get('paused')}
    for note in notes:
        if note.get('contrast') and note['semantic_key'] in paused:
            note['contrast']['paused'] = True
    return notes, edits


def eligible(note, incumbent):
    record = note.get('contrast')
    return not record['paused'] if record else not note.get('inert', False)


def background(pool):
    return f.h.background(pool) if pool.task == f.ARYL else NATIVE_BACKGROUND(pool)


def configure():
    m.configure()
    frozen.locked_first = lambda task: None  # Read only before the first observation, and q1 starts after it.
    proposal.background = background
    v2.LIVE_ARMS = v2.LIVE_ARMS + (ARM,)
    v2.EVIDENCE_FIRST_ARMS = v2.EVIDENCE_FIRST_ARMS + (ARM,)
    native = v2.prefix_from_first
    v2.prefix_from_first = lambda t, s, a: m.faithful_prefix(t, s, a) if a == ARM else native(t, s, a)
    prompts.SYSTEM, prompts.public, prompts.validate = build_system(), public, validate
    k.install, k.eligible = install, eligible
    k.pause_for_recommendation = lambda notes, inc: (deepcopy(notes), [])
    v2.decide, policy.select = m.faithful_decide, m.select_with_test


def run_stream(task, seed):
    return v2.run_arm(task, seed, ARM)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prompt', 'pilot', 'run'))
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.action == 'prompt':
        print(build_system())
        return
    jobs = [(t, s) for t, s in m.units() if s == 0] if args.action == 'pilot' else m.units()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=configure) as executor:
        futures = [executor.submit(run_stream, t, s) for t, s in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            print(json.dumps(dict(done=done, of=len(futures), **future.result())), flush=True)
