"""Finite-input, public-evidence prompts shared by knowledge-loop ablations."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
from filelock import FileLock

from . import knowledge_openai_client as api
from .knowledge_domain_context import task_context

SYSTEM = '''You assist a sequential chemistry/process experiment. General chemical and
process-mechanism knowledge may inform uncertain hypotheses, using the supplied
domain context, input conditions, observations and public feedback. Only supplied
own observations are experimental measurements. Do not recall benchmark target
values, invent measurements or use candidate identifiers as physical ranks. Do not
invent molecular or process identities that the domain marks as unknown or opaque.
Objective
units may be yield, STY or detector peak area. These are not interchangeable.

Maintain at most four uncertain, locally testable pairwise preferences. A claim
is the soft statement that one displayed real experiment is better than another,
within an explicitly declared context. It is not a causal or global monotonicity
theorem. Candidate membership is GP-filtered, but no GP scores or reference labels
are supplied. Only unqueried displayed candidates can be new comparison anchors.

Output exactly one JSON object with keys actions and summary. actions has zero to
three objects, each with exactly op,id,plus_id,minus_id,focus,reason.
op is add,retain,revise,retract. add uses id:null; other operations cite an existing
hypothesis id. For add/revise, plus_id and minus_id are distinct displayed IDs.
focus names one or two domain axes. For a continuous axis, use
{"direction":"increase" or "decrease","bounds":[lo,hi]} with bounds in the
supplied normalized 0..1 coordinates, including both comparison points. This is
your stated deployment scope, fixed BEFORE future observations. For a categorical
axis use {"direction":"prefer_plus","allowed":[real category labels]}, including
both compared labels. No numeric distance or direction exists between categories.
On each continuous focus axis the declared direction must match plus minus minus.
Nonfocus continuous axes may move by at most .25 of their range; nonfocus categories
must be identical. Be specific about interaction/confounding when two axes change.
For retain/retract use plus_id:null,minus_id:null,focus:{}.
Prefix each reason with mechanistic_prior, observed_association or both to identify
its evidence basis. A mechanism prior is not an observed result; an association
must refer only to supplied own observations. Keep each reason under 600 characters
and summary under 1000 characters. Public
rationales only; no private reasoning. An empty actions array is a valid abstention.

Feedback contains pre-observation predictive evidence against GP. Negative scores
are counterevidence, not proof of chemical falsehood. Empty actionable coverage
means dormant, not false. Revise a claim only with a substantive changed statement;
renaming the same claim does not reset evidence. Retractions remain in the ledger.
New versions start without new positive evidence; applicable old negative evidence
may constrain their trust. Prefer claims that still have multiple executable
candidates and could usefully distinguish their outcomes. Do not invent a precise
yield or posterior: software computes predictions from your qualitative claim.'''


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             allow_nan=False, separators=(',', ':')).encode('utf-8')).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def make_menu(pool, context, t, seed=0, maximum=24):
    """GP-filtered anchors plus physically close contrasts; no ID-order tie break."""
    eligible = np.flatnonzero(context['eligible'])
    if len(eligible) < 2:
        return []
    priority = context['tie_priority']
    scores = context['base_logei']
    ranked = sorted(eligible, key=lambda j: (-scores[j], priority[j]))
    chosen = [int(context['bo_index'])]
    features = pool.features

    def distance(i, j):
        return sum(float(features[i][k] != features[j][k]) if kind == 'categorical'
                   else (float(features[i][k]) - float(features[j][k]))**2
                   for k, kind in enumerate(pool.kinds))

    # Include a few competitive geometrically distinct anchors; then pair them.
    anchors = ranked[:min(6, len(ranked))]
    for i in anchors:
        if int(i) not in chosen:
            chosen.append(int(i))
        others = sorted((int(j) for j in eligible if j != i and int(j) not in chosen),
                        key=lambda j: (distance(i, j), priority[j]))
        for j in others[:2]:
            if len(chosen) < maximum:
                chosen.append(j)
    while len(chosen) < min(maximum, len(eligible)):
        rest = [int(j) for j in eligible if int(j) not in chosen]
        j = max(rest, key=lambda j: (min(distance(j, i) for i in chosen), -priority[j]))
        chosen.append(j)
    np.random.default_rng(713_000 + seed * 100 + t).shuffle(chosen)
    return [pool.ids[j] for j in chosen]


def public_payload(pool, state, t, menu_ids, coverage):
    domain = task_context(pool.task, pool)
    domain.pop('source_references', None)
    ledger = []
    for h in state['hypotheses']:
        row = {k: deepcopy(h[k]) for k in ('id', 'version', 'parent', 'plus_id', 'minus_id', 'focus', 'scope', 'status') if k in h}
        row['score'] = h['ledger']['score']
        row['weight'] = h['ledger'].get('weight', 0.)
        row['events'] = [{k: deepcopy(event[k]) for k in ('point_id', 'query_round', 'observed_y', 'informative', 'delta_log_score') if k in event}
                         for event in h['ledger']['events']]
        row['event_units'] = 'observed_y = raw objective divided by fixed engineering scale ' + str(pool.scale)
        row['inherited_counterevidence_count'] = len(h.get('inherited_negative_evidence', []))
        ledger.append(row)
    return {
        'task': pool.task, 'objective_unit': pool.unit, 'query': t, 'remaining_queries': 17 - t,
        'domain_context': domain,
        'menu_policy': 'GP-filtered executable set; no GP means, scores, uncertainty or BO reference labels',
        'axes': [{'name': axis, 'type': kind} for axis, kind in zip(pool.axes, pool.kinds)],
        'own_observations': [{'point': pool.points[pool.ix[h['candidate_id']]],
                              'objective': h['raw_value']} for h in state['history']],
        'candidate_menu': [{'id': key, 'point': pool.points[pool.ix[key]],
                            'normalized_continuous': {a: pool.features[pool.ix[key]][k]
                                 for k, (a, kind) in enumerate(zip(pool.axes, pool.kinds)) if kind == 'continuous'}}
                           for key in menu_ids],
        'knowledge_ledger': ledger,
        'actionable_coverage': deepcopy(coverage),
        'feedback_note': 'All events are frozen prequery forecasts scored only after their selected outcome. No future target is available.'}


def validate(raw):
    if not isinstance(raw, dict) or set(raw) != {'actions', 'summary'}:
        raise ValueError('expected actions and summary')
    if not isinstance(raw['summary'], str) or not 1 <= len(raw['summary'].strip()) <= 1000:
        raise ValueError('invalid summary')
    if not isinstance(raw['actions'], list) or len(raw['actions']) > 3:
        raise ValueError('at most three actions')
    for a in raw['actions']:
        if not isinstance(a, dict) or set(a) != {'op', 'id', 'plus_id', 'minus_id', 'focus', 'reason'}:
            raise ValueError('invalid action keys')
        if a['op'] not in ('add', 'retain', 'revise', 'retract'):
            raise ValueError('invalid operation')
        if not isinstance(a['reason'], str) or not 1 <= len(a['reason']) <= 600:
            raise ValueError('invalid public reason')
    return deepcopy(raw)


def request(client, payload, folder):
    """Cache identical public requests across arms; max two schema attempts."""
    context = {'public': payload, 'system': SYSTEM, 'model': api.MODEL,
               'reasoning_effort': api.REASONING_EFFORT}
    key = digest(context)
    directory = Path(folder) / key
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / '.lock'), timeout=300):
        saved = directory / 'validated.json'
        if saved.exists():
            result = json.loads(saved.read_text(encoding='utf-8'))
            assert result['context_sha256'] == key
            validate(result['proposal'])
            return result
        write(directory / 'context.json', context)
        correction = None
        for attempt in range(1, 3):
            user = payload if correction is None else {'same_experiment_context': payload, 'schema_correction': correction}
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
            result = {'context_sha256': key, 'proposal': proposal, 'attempt': attempt,
                      'response_sha256': digest(response), 'resolved_model': response['resolved_model']}
            write(saved, result)
            return result
    raise ValueError('two response schema attempts exhausted')
