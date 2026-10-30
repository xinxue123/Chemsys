"""R6 qualitative comparison requests.  Never request a numerical yield forecast."""

from copy import deepcopy
from pathlib import Path

from . import continuous_r6_engine as engine


SYSTEM = '''You assist in one sequential chemistry/process optimization task.
Only the listed real unqueried experiments can be compared. All named axes
are CONTINUOUS PHYSICAL inputs; the objective and units are given.
Do not invent measurements, quote outcomes from datasets, use future results,
or equate ID ordering with score ranking. You have only your own observations.

Your task is to express ONE UNCERTAIN, LOCALLY TESTABLE qualitative pairwise
preference, not to predict a numerical objective or assign a GP posterior.
Return one EXACT JSON object with keys:
{"action":"compare","better_id":"shown ID","worse_id":"shown ID",
 "focus_directions":{"one_axis":"increase"},"reason":"brief, falsifiable rationale"}
Or abstain:
{"action":"abstain","better_id":null,"worse_id":null,
 "focus_directions":{},"reason":"brief uncertainty explanation"}

The better and worse points must both occur in the supplied candidate_menu.
Name one or two focus axes. Direction is how the ACTUAL better point changes
relative to the worse point: increase or decrease. Other axes may differ by
at most 0.25 of their full physical range; choose near-matched comparisons.
Your statement is observational, not a proof of a causal isolated-axis effect.
It is fine to abstain if no credible close comparison exists. Plain public
rationale, not private reasoning traces. Limit reason to 1000 characters.

If method says GP-filtered, candidates are already eligible under a hidden GP
acquisition constraint; menu membership itself reflects GP information. You
do NOT see means, uncertainties, score values, reference labels or ordering.
If method says GP-blind, candidate membership depends only on input geometry.
No choice is executed without a software direction and finite-pool review.
Return JSON only.'''


def validate(raw, menu):
    required = {'action', 'better_id', 'worse_id', 'focus_directions', 'reason'}
    if not isinstance(raw, dict) or set(raw) != required:
        raise ValueError('exact qualitative comparison schema required')
    if raw['action'] not in ('compare', 'abstain'):
        raise ValueError('invalid action')
    reason = raw['reason']
    if not isinstance(reason, str) or not 1 <= len(reason.strip()) <= 1000:
        raise ValueError('concise reason required')
    if raw['action'] == 'abstain':
        if raw['better_id'] is not None or raw['worse_id'] is not None or raw['focus_directions'] != {}:
            raise ValueError('invalid abstention fields')
        return deepcopy(raw)
    ids = {row['candidate_id'] for row in menu}
    if (not isinstance(raw['better_id'], str) or not isinstance(raw['worse_id'], str)
            or raw['better_id'] == raw['worse_id'] or raw['better_id'] not in ids
            or raw['worse_id'] not in ids):
        raise ValueError('both distinct IDs must be in the displayed menu')
    directions = raw['focus_directions']
    if (not isinstance(directions, dict) or len(directions) not in (1, 2)
            or any(not isinstance(k, str) or v not in ('increase', 'decrease')
                   for k, v in directions.items())):
        raise ValueError('one or two named signed focus axes required')
    # The engine separately checks physical direction, bounded background,
    # duplicate/opposite hypotheses and acquisition authorization.
    return deepcopy(raw)


def request(client, cache, state, t, context, protocol_sha256, root):
    """Uses the repository's existing DeepSeek transport and retry audit."""
    from filelock import FileLock
    from . import continuous_benchmark_data as data
    from . import reaction_transfer_llm as transport

    payload = engine.public_menu(cache, state, t, context)
    request_context = {
        'public': payload, 'observed_prefix': deepcopy(state['history']),
        'task': cache.pool['task'], 'phase': 'R6_qualitative_pairwise',
        'arm': state['arm'], 'protocol_sha256': protocol_sha256,
        'system_sha256': engine.digest(SYSTEM),
    }
    request_context['context_sha256'] = engine.digest(request_context)
    key = request_context['context_sha256']
    folder = Path(root) / 'requests' / key
    folder.mkdir(parents=True, exist_ok=True)
    with FileLock(str(folder / '.request.lock'), timeout=900):
        if (folder / 'context.json').exists():
            assert data.read(folder / 'context.json') == request_context
        else:
            data.write(folder / 'context.json', request_context)
        response = transport.get(client, request_context, folder, SYSTEM,
                                 lambda raw: validate(raw, payload['candidate_menu']))
    return response['plan'], {
        'context_sha256': key, 'response_sha256': response['response_sha256'],
        'accepted_attempt': response['accepted_attempt'],
        'received_responses': response['received_responses'],
    }
