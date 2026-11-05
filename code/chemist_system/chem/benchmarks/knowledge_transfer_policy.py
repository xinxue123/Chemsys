"""V5 declared background transfer, observed-reference probes, smooth trust.

Rule semantics exclude candidate IDs and continuous example-anchor locations.
Only frozen focus/scope and categorical preferred/reference labels define them.
Probe re-anchoring never mutates a rule or resets its ledger. Negative prediction
scores and failed observed comparisons survive applicable revisions once each.
Declared backgrounds widen the rule scope, never the per-probe comparison.
An unmodified V4 implementation remains available in knowledge_rule_policy.py.
"""
from copy import deepcopy
from math import isfinite, tanh
import numpy as np

from . import knowledge_policy as geometry


MAX_ACTIONS = 3
MAX_LIVE_CLAIMS = 4
BACKGROUND_RADIUS = .25
MAX_WEIGHT = .5
PREFERENCE_NOISE_SD = .06
_digest = geometry._digest
_get = geometry._get
_features = geometry._features
_kinds = geometry._kinds


def initial_ledger():
    return dict(prediction_score=0., score=0., successes=0, failures=0,
                seen=[], events=[], trusted_weight=0., weight=0., cap=MAX_WEIGHT)


def _rule_contract(h):
    return {k: deepcopy(h[k]) for k in ('focus', 'scope', 'categorical_preference')}


def _semantic(h):
    return _digest(_rule_contract(h))


def _history(pool, history):
    result = {}
    for row in history:
        key, y = row['candidate_id'], float(row['y'])
        if key not in _get(pool, 'ix') or key in result or not isfinite(y):
            raise ValueError('History must contain unique known observed IDs and finite y')
        result[key] = y
    return result


def _build_contract(pool, action, menu_ids, observed):
    plus, minus = action.get('plus_id'), action.get('minus_id')
    if plus not in menu_ids or plus in observed:
        raise ValueError('plus_id must be a displayed unqueried candidate')
    if minus not in observed:
        raise ValueError('minus_id must be an own already observed candidate')
    # Reuse only the frozen mixed-domain geometry validator, not its ledger or
    # legacy requirement that both anchors are unqueried.
    contract = geometry._build_contract(pool, action, set(menu_ids) | set(observed))
    categories = {}
    axes, kinds = list(_get(pool, 'axes')), _kinds(pool)
    for axis in contract['focus']:
        if kinds[axis] == 'categorical':
            j = axes.index(axis)
            a, b = contract['plus_features'][j], contract['minus_features'][j]
            if not isinstance(a, str) or not isinstance(b, str):
                raise ValueError('Categorical preferences require true string labels')
            categories[axis] = dict(plus=a, minus=b)
    contract['categorical_preference'] = categories
    background = action.get('background')
    if not isinstance(background, dict) or set(background) - set(axes):
        raise ValueError('Background must name real nonfocus axes')
    if set(background) & set(contract['focus']):
        raise ValueError('Background cannot replace a focus axis')
    declared = {}
    for axis, spec in background.items():
        j = axes.index(axis)
        a, b = contract['plus_features'][j], contract['minus_features'][j]
        if kinds[axis] == 'continuous':
            if not isinstance(spec, dict) or set(spec) != {'bounds'}:
                raise ValueError('Continuous background requires normalized bounds')
            bounds = spec['bounds']
            if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2
                    or not all(geometry._finite_number(v) for v in bounds)
                    or not 0 <= bounds[0] < bounds[1] <= 1
                    or min(a, b) < bounds[0] or max(a, b) > bounds[1]):
                raise ValueError('Background bounds must contain both anchors within [0,1]')
            declared[axis] = {'bounds': list(map(float, bounds))}
        else:
            if not isinstance(spec, dict) or set(spec) != {'allowed'}:
                raise ValueError('Categorical background requires real allowed labels')
            allowed = spec['allowed']
            real = {_features(pool, key)[j] for key in _get(pool, 'ids')}
            if (not isinstance(allowed, list) or not allowed
                    or not all(isinstance(v, str) for v in allowed)
                    or not isinstance(a, str) or not isinstance(b, str)
                    or len(set(allowed)) != len(allowed) or set(allowed) - real
                    or a not in allowed or b not in allowed):
                raise ValueError('Background allowed must contain both real string labels without duplicates')
            declared[axis] = {'allowed': sorted(allowed, key=_digest)}
        contract['scope']['axes'][axis] = dict(kind=kinds[axis], **deepcopy(declared[axis]))
    # The actual scope is authoritative and participates in semantic identity.
    # The declaration records which backgrounds were expressly specified; an
    # explicit declaration equal to the default does not create new semantics.
    contract['background'] = declared
    return contract


def _indices(pool, indices):
    js = np.asarray(indices, dtype=int)
    if js.ndim != 1 or np.any(js < 0) or np.any(js >= len(_get(pool, 'ids'))):
        raise ValueError('Invalid candidate index vector')
    if len(np.unique(js)) != len(js):
        raise ValueError('Duplicate candidate indices')
    return js


def _scope_mask(h, axes, rows):
    mask = np.ones(len(rows), dtype=bool)
    for j, axis in enumerate(axes):
        rule = h['scope']['axes'][axis]
        if rule['kind'] == 'continuous':
            column = np.asarray(rows[:, j], dtype=float)
            lo, hi = rule['bounds']
            mask &= (column >= lo - 1e-12) & (column <= hi + 1e-12)
        else:
            mask &= np.isin(rows[:, j], rule['allowed'])
    return mask


def scope_indices(pool, h, indices):
    js = _indices(pool, indices)
    if not len(js):
        return js
    rows = np.asarray(_get(pool, 'features'), dtype=object)[js]
    return js[_scope_mask(h, _get(pool, 'axes'), rows)]


def _compatible_mask(h, axes, kinds, candidates, reference):
    mask = np.ones(len(candidates), dtype=bool)
    for j, axis in enumerate(axes):
        if axis in h['focus']:
            if kinds[axis] == 'continuous':
                sign = 1 if h['focus'][axis]['direction'] == 'increase' else -1
                mask &= sign * (np.asarray(candidates[:, j], dtype=float) - float(reference[j])) > 0
            else:
                pref = h['categorical_preference'][axis]
                if reference[j] != pref['minus']:
                    return np.zeros(len(candidates), dtype=bool)
                mask &= candidates[:, j] == pref['plus']
        elif kinds[axis] == 'continuous':
            mask &= np.abs(np.asarray(candidates[:, j], dtype=float) - float(reference[j])) <= BACKGROUND_RADIUS + 1e-12
        else:
            mask &= candidates[:, j] == reference[j]
    return mask


def make_probes(pool, h, history, indices):
    """Vectorized new-plus -> best physically compatible own observed reference.

    Reference ties use physical features, not IDs. Only references are sorted;
    each gets a vector mask, and already assigned candidate rows are skipped.
    Returning a different reference does not modify h or its evidence ledger.
    """
    if h['semantic_key'] != _semantic(h):
        raise ValueError('Rule semantics changed without a version')
    js = _indices(pool, indices)
    observed = _history(pool, history)
    if h['status'] == 'retracted' or not len(js) or not observed:
        return {}
    axes, kinds, ids, ix = list(_get(pool, 'axes')), _kinds(pool), _get(pool, 'ids'), _get(pool, 'ix')
    features = np.asarray(_get(pool, 'features'), dtype=object)
    unqueried = np.array([ids[int(j)] not in observed for j in js], dtype=bool)
    js = js[unqueried & _scope_mask(h, axes, features[js])]
    if not len(js):
        return {}
    references = sorted(observed, key=lambda key: (-observed[key], tuple(features[ix[key]])))
    ref_ix = np.asarray([ix[key] for key in references], dtype=int)
    refs_in_scope = _scope_mask(h, axes, features[ref_ix])
    remaining = np.ones(len(js), dtype=bool)
    candidates = features[js]
    result = {}
    for key, in_scope in zip(references, refs_in_scope):
        if not in_scope:
            continue
        positions = np.flatnonzero(remaining)
        if not len(positions):
            break
        ref = features[ix[key]]
        chosen = positions[_compatible_mask(h, axes, kinds, candidates[positions], ref)]
        ref_features = [x.item() if isinstance(x, np.generic) else x for x in ref]
        for position in chosen:
            j = int(js[position])
            result[j] = dict(reference_id=key, reference_y=float(observed[key]),
                             reference_features=list(ref_features), point_id=ids[j])
        remaining[chosen] = False
    return result


def evidence_summary(h):
    ledger = h['ledger']
    inherited = [r for r in h.get('inherited_negative_evidence', []) if r['applicable']]
    # Distinct source experts may have evaluated the same physical comparison.
    # Keep all source events in the audit, but never count one observation as
    # multiple failures or add its overlapping negative expert scores together.
    groups = {}
    for source, rows in (('local', ledger['events']), ('inherited', inherited)):
        for row in rows:
            key = (row['query_round'], row['point_id'], row['reference_id'])
            groups.setdefault(key, {'local': [], 'inherited': []})[source].append(row)
    local_score = effective_score = 0.
    local_successes = local_failures = successes = failures = 0
    for group in groups.values():
        own, old = group['local'], group['inherited']
        own_scores = [float(row['delta_log_score']) for row in own]
        negative_scores = [float(row['delta_log_score']) for row in old if row['delta_log_score'] < 0]
        own_failure = any(not row['fact_success'] for row in own)
        failure = own_failure or any(not row['fact_success'] for row in old)
        local_score += min(own_scores, default=0.)
        effective_score += min(own_scores + negative_scores, default=0.)
        local_failures += int(own_failure)
        local_successes += int(bool(own) and not own_failure)
        failures += int(failure)
        successes += int(bool(own) and not failure)
    reliability = max(0., (successes - failures) / (successes + failures + 2))
    formula = min(MAX_WEIGHT, max(0., tanh(effective_score / 2)), reliability)
    trusted = min(formula, max(0., float(ledger['trusted_weight'])))
    return dict(effective_prediction_score=effective_score, successes=successes, failures=failures,
                trusted_weight=trusted, formula_weight=formula, reliability_cap=reliability,
                local_prediction_score=local_score,
                local_successes=local_successes, local_failures=local_failures,
                inherited_failure_count=failures - local_failures,
                inherited_prediction_penalty=effective_score - local_score)


def weight(h):
    return evidence_summary(h)['trusted_weight'] if h['status'] == 'active' else 0.


def _relation(h):
    return {'directions': {a: v['direction'] for a, v in h['focus'].items()},
            'categorical_preference': h['categorical_preference']}


def _applicable(pool, new, record):
    if _digest(_relation(new)) != _digest(_relation(record['source_rule'])):
        return False
    ids = [record['point_id'], record['reference_id']]
    if any(key not in _get(pool, 'ix') for key in ids):
        return False
    js = [_get(pool, 'ix')[key] for key in ids]
    if len(scope_indices(pool, new, js)) != 2:
        return False
    rows = np.asarray([_features(pool, key) for key in ids], dtype=object)
    return bool(_compatible_mask(new, _get(pool, 'axes'), _kinds(pool), rows[:1], rows[1])[0])


def _inherit_negative(pool, sources, new):
    records = {}
    for old in sources:
        for row in old.get('inherited_negative_evidence', []):
            records[row['event_id']] = deepcopy(row)
        for event in old['ledger']['events']:
            if event['delta_log_score'] < 0 or not event['fact_success']:
                records[event['event_id']] = dict(event_id=event['event_id'],
                    source_id=old['id'], source_version=old['version'], source_rule=_rule_contract(old),
                    point_id=event['point_id'], reference_id=event['reference_id'],
                    reference_y=event['reference_y'], observed_y=event['observed_y'],
                    query_round=event['query_round'], delta_log_score=event['delta_log_score'],
                    fact_success=event['fact_success'], prediction=deepcopy(event['prediction']))
    result = []
    for row in records.values():
        row['applicable'] = _applicable(pool, new, row)
        row['interpretation'] = 'conservative inherited constraint; not a new likelihood or positive fact'
        result.append(row)
    return sorted(result, key=lambda row: row['event_id'])


def review_and_apply(pool, hypotheses, proposal, menu_ids, t, history):
    """Review <=3 actions without mutating inputs; preserve old versions/audits."""
    if type(t) is not int or t < 1:
        raise ValueError('Creation round must be positive')
    if (not isinstance(proposal, dict) or set(proposal) != {'actions', 'summary'} or
            not isinstance(proposal['summary'], str) or not 1 <= len(proposal['summary'].strip()) <= 1600 or
            not isinstance(proposal['actions'], list) or len(proposal['actions']) > MAX_ACTIONS):
        raise ValueError('Proposal requires summary and at most three actions')
    observed = _history(pool, history)
    menu_ids = list(menu_ids)
    if (len(set(menu_ids)) != len(menu_ids) or set(menu_ids) - set(_get(pool, 'ids')) or
            set(menu_ids) & set(observed)):
        raise ValueError('Menu must contain unique unqueried real IDs')
    result, events, touched = deepcopy(hypotheses), [], set()
    for position, action in enumerate(proposal['actions']):
        event = dict(round=t, action_index=position, accepted=False)
        try:
            if not isinstance(action, dict) or set(action) != {'op', 'id', 'plus_id', 'minus_id', 'focus', 'background', 'reason'}:
                raise ValueError('Action fields must be op,id,plus_id,minus_id,focus,background,reason')
            op, hid = action['op'], action['id']
            event.update(op=op, id=hid)
            if op not in ('add', 'retain', 'revise', 'retract'):
                raise ValueError('Unknown operation')
            if not isinstance(action['reason'], str) or not 1 <= len(action['reason'].strip()) <= 1000:
                raise ValueError('A concise reason is required')
            matches = [h for h in result if h['id'] == hid]
            parent = max(matches, key=lambda h: h['version']) if matches else None
            if op == 'add' and hid is not None:
                raise ValueError('Add requires id=null')
            if op != 'add' and (not isinstance(hid, str) or parent is None):
                raise ValueError('Operation must name an existing rule')
            if hid is not None and hid in touched:
                raise ValueError('At most one action per existing rule')
            if parent and any(e['query_round'] >= t for e in parent['ledger']['events']):
                raise ValueError('Operation must precede this round feedback')
            if op in ('retain', 'retract'):
                if (action['plus_id'] is not None or action['minus_id'] is not None
                        or action['focus'] != {} or action['background'] != {}):
                    raise ValueError('Retain/retract cannot alter a rule contract')
                if parent['status'] == 'retracted':
                    raise ValueError('A retracted rule requires an explicit revision')
                if op == 'retract':
                    parent.update(status='retracted', retired_round=t, retirement_reason=action['reason'])
                    parent['ledger']['weight'] = 0.
                event.update(accepted=True, outcome=op, version=parent['version'])
                touched.add(hid)
            else:
                contract = _build_contract(pool, action, set(menu_ids), observed)
                semantic = _semantic(contract)
                equal = [h for h in result if h['semantic_key'] == semantic]
                existing = max(equal, key=lambda h: (h['created_round'], h['version'])) if equal else None
                if existing is not None and not (op == 'revise' and existing['status'] == 'retracted'):
                    event.update(accepted=True, outcome='duplicate_semantics_preserved',
                                 id=existing['id'], version=existing['version'], status=existing['status'])
                    if hid is not None:
                        touched.add(hid)
                else:
                    live = sum(h['status'] != 'retracted' for h in result)
                    if live - int(parent is not None and parent['status'] != 'retracted') >= MAX_LIVE_CLAIMS:
                        raise ValueError('At most four live rules, including dormant rules')
                    if parent is None:
                        number = 1
                        used = {h['id'] for h in result}
                        while f'H{number:03d}' in used:
                            number += 1
                        hid = f'H{number:03d}'
                    new = dict(contract, id=hid, version=parent['version'] + 1 if parent else 1,
                        parent={'id': parent['id'], 'version': parent['version']} if parent else None,
                        created_round=t, status='active', reason=action['reason'], semantic_key=semantic,
                        tau_y=PREFERENCE_NOISE_SD, ledger=initial_ledger(),
                        excluded_observed_ids=sorted(observed), inherited_negative_evidence=[])
                    # An applicable negative event cannot be laundered by a new
                    # ID or a smaller semantic revision. Distinct relations do
                    # not inherit an applicable penalty; positive credit never moves.
                    new['inherited_negative_evidence'] = _inherit_negative(pool, result, new)
                    if parent:
                        parent.update(status='retracted', retired_round=t,
                                      retirement_reason='superseded by explicit revision')
                        parent['ledger']['weight'] = 0.
                    result.append(new)
                    touched.add(hid)
                    event.update(accepted=True, outcome=op, id=hid, version=new['version'],
                                 inherited_negative_count=len(new['inherited_negative_evidence']))
        except (ValueError, TypeError, KeyError) as error:
            event['reason'] = str(error)
        events.append(event)
    return result, events


def _score_prediction(prediction, y):
    from .knowledge_rule_math import score_prediction
    return score_prediction(prediction, y)


def record_feedback(h, pred, observation):
    """Score the frozen actually tempered predictor, then update the two ledgers.

    The constructor/engine must obtain references only through make_probes and
    persist pred before revealing the selected observation. This function never
    reads an oracle or reconstructs an untempered alternative prediction.
    """
    if h['status'] == 'retracted' or h['semantic_key'] != _semantic(h):
        raise ValueError('Feedback requires a live, unchanged rule version')
    if (pred['hypothesis_id'] != h['id'] or pred['version'] != h['version'] or
            pred['semantic_key'] != h['semantic_key']):
        raise ValueError('Prediction belongs to another rule/version')
    expected = _digest({k: v for k, v in pred.items() if k != 'prediction_sha256'})
    if pred.get('prediction_sha256') != expected:
        raise ValueError('Frozen prediction changed')
    t, point, ref = pred['query_round'], pred['point_id'], pred['reference_id']
    ledger = h['ledger']
    if (type(t) is not int or t < h['created_round'] or
            any(e['query_round'] >= t for e in ledger['events']) or
            point in ledger['seen'] or point in h.get('excluded_observed_ids', [])):
        raise ValueError('Repeated or retrospective feedback')
    if (observation['candidate_id'] != point or point == ref or
            ('query_round' in observation and observation['query_round'] != t)):
        raise ValueError('Feedback does not match the committed new plus')
    y, reference_y = float(observation['y']), float(pred['reference_y'])
    if not isfinite(y) or not isfinite(reference_y):
        raise ValueError('Feedback and fixed reference must be finite')
    before = evidence_summary(h)
    deployed_before = weight(h)
    gain = float(_score_prediction(pred, y))
    if not isfinite(gain):
        raise ValueError('Prediction score must be finite')
    success = bool(y > reference_y)
    event = dict(hypothesis_id=h['id'], version=h['version'], point_id=point,
                 reference_id=ref, reference_y=reference_y, query_round=t,
                 prediction=deepcopy(pred), observed_y=y, fact_success=success,
                 delta_prediction_score=gain, delta_log_score=gain)
    event['event_id'] = _digest(event)
    ledger['prediction_score'] += gain
    ledger['score'] = ledger['prediction_score']  # compatibility alias, same score
    ledger['successes'] += int(success)
    ledger['failures'] += int(not success)
    ledger['seen'] = sorted(set(ledger['seen']) | {point})
    ledger['events'].append(event)
    formula = evidence_summary(h)['formula_weight']
    # Release only the new positive predictive credit, not all old accumulated
    # score. Zero/tiny increments cannot trigger a discontinuous trust jump.
    recover = success and gain > 0
    credit_limit = max(0., tanh(gain / 2)) if recover else 0.
    recovery_ceiling = before['trusted_weight'] + credit_limit
    ledger['trusted_weight'] = min(formula, recovery_ceiling)
    ledger['weight'] = weight(h)
    after = evidence_summary(h)
    return dict(event_id=event['event_id'], hypothesis_id=h['id'], version=h['version'],
                point_id=point, reference_id=ref, reference_y=reference_y, query_round=t,
                delta_log_score=gain, delta_prediction_score=gain, fact_success=success,
                weight_before=deployed_before, weight_after=ledger['weight'],
                trusted_weight_before=before['trusted_weight'], trusted_weight_after=after['trusted_weight'],
                score_after=ledger['prediction_score'], effective_prediction_score=after['effective_prediction_score'],
                successes=after['successes'], failures=after['failures'],
                reliability_cap=after['reliability_cap'], formula_weight=after['formula_weight'],
                recovery_credit_limit=credit_limit, recovery_weight_ceiling=recovery_ceiling,
                weight_transition='increment_limited_positive_fact_and_forecast' if recover else 'nonincreasing_after_failure_or_nonpositive_forecast')


__all__ = ['initial_ledger', 'review_and_apply', 'make_probes', 'scope_indices',
           'record_feedback', 'weight', 'evidence_summary', 'MAX_ACTIONS',
           'MAX_LIVE_CLAIMS', 'BACKGROUND_RADIUS', 'MAX_WEIGHT', 'PREFERENCE_NOISE_SD']
