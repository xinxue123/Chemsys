"""V5 public declared-transfer prompts/cache; V4 menus and fixed Terra API."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from filelock import FileLock

from . import knowledge_llm as legacy_menu
from . import knowledge_openai_client as api
from . import knowledge_transfer_policy as policy
from .knowledge_domain_context import task_context

digest, write = legacy_menu.digest, legacy_menu.write

SYSTEM = '''You assist a sequential chemistry/process experiment. General chemical and
process-mechanism knowledge may inform uncertain LOCAL RULES. Use the supplied
domain, input conditions, own observations and public feedback. Only supplied own
observations are measured experiments. Do not recall benchmark target values,
invent measurements, or rank physical quality by candidate IDs. Do not invent
identities for opaque catalysts/processes. Yield, STY and detector peak area differ.

Each rule has a frozen physical scope and one or two focus axes. You give a real
example pair: plus_id MUST be an unqueried ID in candidate_menu; minus_id MUST be
an already measured ID in own_observations, never another unqueried candidate.
Choose a useful high-value physically comparable observed reference. Software
will instantiate the SAME rule each round on other unqueried in-scope plus points,
using the highest measured compatible reference, with physical-feature tie breaks.
The example pair is not a global causal theorem. Explicitly state the local
generalization and interactions/confounding you intend. Changing probe points
alone does not erase or reset this rule's evidence; changing its scope or meaning
requires an explicit revision with a new version.

Output exactly one JSON object with actions and summary. actions has zero to three
objects, each with exactly op,id,plus_id,minus_id,focus,background,reason. At most four rules
can remain live, including dormant rules. op is add,retain,revise,retract; add uses
id:null and other actions cite an existing rule id. For add/revise, focus names
one or two domain axes. Continuous focus is {"direction":"increase" or "decrease",
"bounds":[lo,hi]} in supplied normalized 0..1 coordinates, containing both example
points, with plus moving in the declared direction. Categorical focus is
{"direction":"prefer_plus","allowed":[real category labels]} containing both
example labels. Its actual preference is the plus example label over the minus
example label; other allowed labels gain no invented ordinal meaning. A category
can also use the shorthand "prefer_plus".

background is an object for NONFOCUS axes only. Explicitly declare a transferable
background scope before observing this plus: continuous {"bounds":[lo,hi]} in
normalized 0..1 coordinates, categorical {"allowed":[real category labels from
the supplied domain]}. Each declared scope must include BOTH example anchors.
Undeclared axes retain the local default: continuous intersection of the two
example-centered .25 neighborhoods, categorical the single shared label. You may
declare a broader scope when chemically justified, but state the plausible
interactions and why transfer across these backgrounds might fail. A single
successful pair does NOT verify cross-background generalization. More allowed
focus categories do not invent extra preferred labels. The same rule may be
tested in multiple declared backgrounds with its own new evidence.

Every individual probe remains physically comparable: nonfocus continuous
values differ by at most .25 and categorical background labels MUST be identical
within that probe. A broad rule scope does not permit borrowing an observed
reference from an incomparable background. Changing actual scope requires an
explicit new version with NO inherited positive credit. Never silently expand
scope after results merely to preserve old credit. background:{} uses defaults.

For retain/retract use plus_id:null,minus_id:null,focus:{},background:{}. Prefix each reason with
mechanistic_prior, observed_association or both; say which known facts support it
and what could reverse it. Mechanistic priors are not measurements. Keep reasons
under 600 characters and summary under 1000. Public rationales only. Empty actions
are a valid abstention, not evidence the chemistry hypothesis was disproved.

Feedback has TWO separate ledgers. Prediction score is the actually frozen
tempered forecast's log score relative to GP; it measures incremental prediction,
not whether the rule is chemically true. Pair fact_success means the NEW plus
objective was strictly greater than its FIXED already-observed reference. A
positive prediction score can accompany a failed comparison. Both ledgers bound
trust; they are not independent likelihoods to multiply. A failed comparison or
nonpositive prediction increment cannot increase trusted influence. A later new
point with positive fact and positive prediction increment may restore it, but
the increase is capped by tanh(new_prediction_increment/2); a tiny new gain cannot
release all historical credit. Dormancy preserves latent trust, not deployment.
Applicable negative scores and failed pair facts survive revisions once each;
new versions inherit no positive credit. Previously observed points cannot earn
new-version forward evidence. Retain, revise or retract using both ledgers and
known pair facts. Empty current coverage means dormant, not false. Do not invent
precise yields or posterior statistics; software computes all predictions.'''


def make_menu(pool, context, t, seed=0, maximum=24, state=None, history=None):
    """Original GP menu plus <=4 eligible neighbors of top own references.

    No target lookup: only already observed y ranks the references. Category
    distances are equality mismatches; candidate tie order uses existing physical
    priorities. The final display is shuffled, never presented as a GP ranking.
    """
    if history is None:
        history = state['history'] if state is not None else []
    known = {r['candidate_id'] for r in history}
    eligible = np.array([j for j in np.flatnonzero(context['eligible']) if pool.ids[int(j)] not in known], dtype=int)
    if not len(eligible):
        return []
    if maximum < 1:
        raise ValueError('Menu size must be positive')
    if len(eligible) == 1:
        return [pool.ids[int(eligible[0])]]
    base = legacy_menu.make_menu(pool, context, t, seed=seed, maximum=maximum)
    base = [pool.ix[key] for key in base if key not in known and pool.ix[key] in eligible]
    features = np.asarray(pool.features, dtype=object)
    priority = context['tie_priority']
    refs = sorted(history, key=lambda r: (-float(r['y']), tuple(features[pool.ix[r['candidate_id']]])))[:4]
    protected = [int(context['bo_index'])] if int(context['bo_index']) in eligible else []
    for ref in refs:
        anchor = features[pool.ix[ref['candidate_id']]]
        distance = np.zeros(len(eligible), dtype=float)
        necessary_focus = np.zeros(len(eligible), dtype=int)
        any_difference = np.zeros(len(eligible), dtype=bool)
        for k, kind in enumerate(pool.kinds):
            if kind == 'categorical':
                delta = features[eligible, k] != anchor[k]
                distance += delta
                necessary_focus += delta
                any_difference |= delta
            else:
                delta = np.abs(np.asarray(features[eligible, k], dtype=float) - float(anchor[k]))
                distance += delta * delta
                necessary_focus += delta > policy.BACKGROUND_RADIUS + 1e-12
                any_difference |= delta > 0
        choices = [i for i, j in enumerate(eligible) if necessary_focus[i] <= 2 and any_difference[i] and int(j) not in protected]
        if choices and len(protected) < maximum:
            i = min(choices, key=lambda i: (distance[i], priority[int(eligible[i])]))
            protected.append(int(eligible[i]))
    chosen = list(protected)
    chosen.extend(int(j) for j in base if int(j) not in chosen)
    chosen = chosen[:maximum]
    np.random.default_rng(713_000 + seed * 100 + t).shuffle(chosen)
    return [pool.ids[j] for j in chosen]


def public_payload(pool, state, t, menu_ids, coverage):
    domain = task_context(pool.task, pool)
    domain.pop('source_references', None)
    known = {r['candidate_id'] for r in state['history']}
    if set(menu_ids) & known:
        raise ValueError('Candidate menu may contain only new plus candidates')
    def point(key):
        j = pool.ix[key]
        return dict(id=key, point=deepcopy(pool.points[j]), normalized_continuous={
            a: float(pool.features[j][k]) for k, (a, kind) in enumerate(zip(pool.axes, pool.kinds)) if kind == 'continuous'})
    ledger = []
    for h in state['hypotheses']:
        row = {k: deepcopy(h[k]) for k in ('id', 'version', 'parent', 'focus', 'background', 'scope',
                    'categorical_preference', 'status', 'reason') if k in h}
        row['origin_example'] = {'plus_id': h['plus_id'], 'minus_id': h['minus_id']}
        row['evidence'] = policy.evidence_summary(h)
        # Latent learned credit is identical in deployment/verification arms;
        # neither the public prompt nor this field reveals the arm's mode.
        row['trusted_weight'] = policy.evidence_summary(h)['trusted_weight']
        row['tested_pairs'] = [dict(plus_id=e['point_id'], reference_id=e['reference_id'],
            query_round=e['query_round'], plus_objective=e['observed_y'] * pool.scale,
            reference_objective=e['reference_y'] * pool.scale, fact_success=e['fact_success'],
            delta_prediction_logscore=e['delta_log_score']) for e in h['ledger']['events']]
        row['inherited_counterevidence'] = [dict(event_id=e['event_id'], plus_id=e['point_id'],
            reference_id=e['reference_id'], plus_objective=e['observed_y'] * pool.scale,
            reference_objective=e['reference_y'] * pool.scale, fact_success=e['fact_success'],
            delta_prediction_logscore=e['delta_log_score'], applicable=e['applicable'])
            for e in h.get('inherited_negative_evidence', [])]
        ledger.append(row)
    safe_coverage = ('id', 'version', 'weight', 'eligible_inside_scope', 'executable_probes',
                     'remaining_probe_count', 'can_change_rank_at_cap', 'eligibility_state')
    return dict(task=pool.task, objective_unit=pool.unit, query=t, remaining_queries=17 - t,
        domain_context=domain,
        menu_policy='GP-filtered unqueried plus menu with up to four physical neighbors of top own observed references; no GP scores or BO labels',
        axes=[dict(name=a, type=k) for a, k in zip(pool.axes, pool.kinds)],
        own_observations=[dict(point(r['candidate_id']), objective=r['raw_value']) for r in state['history']],
        candidate_menu=[point(key) for key in menu_ids], knowledge_ledger=ledger,
        actionable_coverage=[{k: deepcopy(row[k]) for k in safe_coverage if k in row} for row in coverage],
        feedback_note='Observed pair objectives use the task objective unit; prediction log scores are dimensionless. Only own already queried observations are available. Background declarations are pre-feedback transfer hypotheses; actual scope is frozen and every probe still needs a comparable observed reference. A successful pair does not validate other backgrounds. Trust restoration is limited to new positive predictive credit.')


def validate(raw):
    if not isinstance(raw, dict) or set(raw) != {'actions', 'summary'}:
        raise ValueError('expected actions and summary')
    if not isinstance(raw['summary'], str) or not 1 <= len(raw['summary'].strip()) <= 1000:
        raise ValueError('invalid summary')
    if not isinstance(raw['actions'], list) or len(raw['actions']) > 3:
        raise ValueError('at most three actions')
    for a in raw['actions']:
        if not isinstance(a, dict) or set(a) != {'op', 'id', 'plus_id', 'minus_id', 'focus', 'background', 'reason'}:
            raise ValueError('invalid action keys')
        if a['op'] not in ('add', 'retain', 'revise', 'retract'):
            raise ValueError('invalid operation')
        if not isinstance(a['focus'], dict) or not isinstance(a['background'], dict):
            raise ValueError('focus and background must be objects')
        if a['op'] in ('retain', 'retract') and (a['plus_id'] is not None or a['minus_id'] is not None
                or a['focus'] != {} or a['background'] != {}):
            raise ValueError('retain/retract cannot change rule fields')
        if not isinstance(a['reason'], str) or not 1 <= len(a['reason']) <= 600:
            raise ValueError('invalid public reason')
    return deepcopy(raw)


def request(client, payload, folder):
    """Same fixed Terra/medium API and safe response cache; no model fallback."""
    context = dict(public=payload, system=SYSTEM, model=api.MODEL, reasoning_effort=api.REASONING_EFFORT)
    key = digest(context)
    directory = Path(folder) / key
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / '.lock'), timeout=300):
        saved = directory / 'validated.json'
        if saved.exists():
            result = json.loads(saved.read_text(encoding='utf-8'))
            if result['context_sha256'] != key:
                raise ValueError('Request cache context mismatch')
            validate(result['proposal'])
            return result
        write(directory / 'context.json', context)
        correction = None
        for attempt in range(1, 3):
            user = payload if correction is None else dict(same_experiment_context=payload, schema_correction=correction)
            path = directory / f'response_{attempt:02d}.json'
            if path.exists():
                response = json.loads(path.read_text(encoding='utf-8'))
            else:
                response = client.complete(SYSTEM, user, json_output=True)
                write(path, response)
            try:
                proposal = validate(json.loads(response['output_text']))
            except (ValueError, TypeError):
                correction = 'Return the exact required action schema. No experiment has occurred.'
                continue
            result = dict(context_sha256=key, proposal=proposal, attempt=attempt,
                          response_sha256=digest(response), resolved_model=response['resolved_model'])
            write(saved, result)
            return result
    raise ValueError('two response schema attempts exhausted')


__all__ = ['SYSTEM', 'make_menu', 'public_payload', 'validate', 'request', 'digest', 'write']
