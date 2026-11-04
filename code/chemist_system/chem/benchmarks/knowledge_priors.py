"""Input-only P0 artificial-region-prior adapter, staged for the next protocol.

This is NOT the historical full-region GP-mean injection: each original clause
is instantiated as one real, unqueried inside/outside candidate comparison.
The GP is conditioned on that single inequality only. Explicit [0,1] focus
bounds permit conditional GP propagation; they do not assert global monotonicity.
No oracle, target column, BO score, or GP posterior is read here.
"""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from chem.benchmarks import knowledge_policy as policy

CONDITIONS = ('correct', 'wrong_category', 'mixed')
PAIR_PRIORITY_SEED = 20261101


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def _hash(value):
    return sha256(_canonical(value).encode('utf-8')).hexdigest()


def _native(value):
    return value.item() if isinstance(value, np.generic) else value


def _row(pool, index):
    return [_native(value) for value in pool.features[index]]


class PriorInstantiationError(ValueError):
    """The receipt explicitly identifies every clause that could not instantiate."""
    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = deepcopy(receipt)


def _decode_category(pool, coordinate, interval):
    raw = np.asarray(pool.raw_pool['z'], dtype=float)
    if raw.ndim != 2 or raw.shape != (len(pool.ids), len(pool.axes)):
        raise ValueError('Original normalized input matrix is not aligned with the pool')
    mapping, labels_to_codes = {}, {}
    for j, row in enumerate(raw):
        code, label = float(row[coordinate]), _row(pool, j)[coordinate]
        token = _canonical(label)
        if not np.isfinite(code):
            raise ValueError('Categorical encoding is nonfinite')
        if code in mapping and _canonical(mapping[code]) != token:
            raise ValueError('One category encoding maps to multiple real labels')
        if token in labels_to_codes and labels_to_codes[token] != code:
            raise ValueError('One real category label maps to multiple original encodings')
        mapping[code], labels_to_codes[token] = label, code
    lo, hi = interval
    allowed = [label for code, label in mapping.items() if lo <= code <= hi]
    if not allowed or len(allowed) == len(mapping):
        raise ValueError('Decoded categorical prior must be a nonempty proper subset')
    allowed = sorted(allowed, key=_canonical)
    return allowed, [{'original_code': code, 'real_label': mapping[code]}
                     for code in sorted(mapping)]


def _pair_priority(pool, plus, minus, coordinate):
    # A fixed independent pseudorandom priority over semantic physical pairs;
    # neither candidate IDs nor original categorical numeric encodings enter.
    return _hash({'seed': PAIR_PRIORITY_SEED, 'axis': pool.axes[coordinate],
                  'plus': _row(pool, plus), 'minus': _row(pool, minus)})


def _select_pair(pool, coordinate, inside, available):
    plus_candidates = np.flatnonzero(inside & available)
    minus_candidates = np.flatnonzero(~inside & available)
    if not len(plus_candidates) or not len(minus_candidates):
        raise ValueError('No unqueried inside/outside candidate pair remains')
    continuous = [i for i, kind in enumerate(pool.kinds) if kind == 'continuous' and i != coordinate]
    categorical = [i for i, kind in enumerate(pool.kinds) if kind == 'categorical' and i != coordinate]
    numeric = {i: np.asarray([float(_row(pool, j)[i]) for j in range(len(pool.ids))])
               for i in continuous + ([coordinate] if pool.kinds[coordinate] == 'continuous' else [])}
    nominal = {i: np.asarray([_canonical(_row(pool, j)[i]) for j in range(len(pool.ids))], dtype=object)
               for i in categorical}
    best, admissible_pairs = None, 0
    for plus in plus_candidates:
        valid = np.ones(len(minus_candidates), dtype=bool)
        distance = np.zeros(len(minus_candidates), dtype=float)
        max_delta = np.zeros(len(minus_candidates), dtype=float)
        for i in categorical:
            valid &= nominal[i][minus_candidates] == nominal[i][plus]
        for i in continuous:
            delta = np.abs(numeric[i][minus_candidates] - numeric[i][plus])
            valid &= delta <= .25 + 1e-12
            distance += delta * delta
            max_delta = np.maximum(max_delta, delta)
        eligible = np.flatnonzero(valid)
        admissible_pairs += len(eligible)
        if not len(eligible):
            continue
        # Match background first. Category identity never contributes a metric.
        minimum = float(distance[eligible].min())
        eligible = eligible[distance[eligible] == minimum]
        if pool.kinds[coordinate] == 'continuous':
            gaps = np.abs(numeric[coordinate][minus_candidates[eligible]] - numeric[coordinate][plus])
            nearest = float(gaps.min())
            eligible = eligible[gaps == nearest]
        else:
            nearest = 0.  # Not an ordinal/category-code distance.
        if best is not None and (minimum, nearest) > best[:2]:
            continue
        for local in eligible:
            minus = int(minus_candidates[int(local)])
            priority = _pair_priority(pool, int(plus), minus, coordinate)
            candidate = (minimum, nearest, priority, int(plus), minus, float(max_delta[int(local)]))
            if best is None or candidate[:3] < best[:3]:
                best = candidate
    if best is None:
        raise ValueError('No pair satisfies identical categorical and bounded continuous background')
    return best[3], best[4], {
        'background_squared_distance': best[0],
        'continuous_focus_distance': best[1] if pool.kinds[coordinate] == 'continuous' else None,
        'background_max_absolute_change': best[5], 'physical_pair_priority': best[2],
        'admissible_pair_count': int(admissible_pairs),
        'unqueried_inside_count': len(plus_candidates), 'unqueried_outside_count': len(minus_candidates)}


def build_priors(pool, condition, observed_ids, t=1):
    """Instantiate all four P0 clauses or raise with a diagnostic receipt.

    Only this function reads a file: source_path.with_name('advice.json').
    Condition labels stay in the LOCAL receipt; hypotheses contain only neutral
    'synthetic_pressure' provenance and input-defined pair/scope statements.
    """
    if getattr(pool, 'group', None) != 'P0' or condition not in CONDITIONS:
        raise ValueError('P0 artificial priors support correct/wrong_category/mixed only')
    if type(t) is not int or t < 1:
        raise ValueError('Prior creation round must be positive')
    observed_ids = list(observed_ids)
    if len(set(observed_ids)) != len(observed_ids) or set(observed_ids) - set(pool.ids):
        raise ValueError('Observed IDs must be distinct real pool IDs')
    advice_path = Path(pool.source_path).with_name('advice.json')
    advice_bytes = advice_path.read_bytes()
    advice = json.loads(advice_bytes.decode('utf-8-sig'))
    prior = advice['0'][condition]
    receipt = {'schema': 'P0_region_to_single_pair_v1', 'task': pool.task,
        'condition': condition, 'source_seed': '0', 'source_advice_path': str(advice_path),
        'source_advice_sha256': sha256(advice_bytes).hexdigest(),
        'selected_prior_sha256': _hash(prior), 'source_confidence': prior.get('confidence'),
        'initial_weight': 0., 'initial_evidence_score': 0., 'observed_ids_excluded': observed_ids,
        'priority': {'kind': 'fixed_seed_semantic_pair_SHA256', 'seed': PAIR_PRIORITY_SEED,
                     'uses_candidate_ID': False, 'uses_category_code_distance': False,
                     'uses_objectives_or_BO_scores': False},
        'difference_from_original': 'One real pair per clause, not simultaneous full-region mean injection; declared direction applies to that pair only.',
        'clauses': [], 'uninstantiable_clauses': [], 'review_events': []}
    clauses = prior.get('clauses')
    if not isinstance(clauses, list) or len(clauses) != 4:
        receipt['uninstantiable_clauses'].append({'reason': 'Expected exactly four original clauses'})
        raise PriorInstantiationError('The pressure prior does not contain four clauses', receipt)
    available = np.asarray([key not in set(observed_ids) for key in pool.ids], dtype=bool)
    actions = []
    for clause in clauses:
        record = {'source_clause': deepcopy(clause)}
        try:
            coordinate, interval = clause['coordinate'], clause['interval']
            if (type(coordinate) is not int or not 0 <= coordinate < len(pool.axes)
                    or not isinstance(interval, list) or len(interval) != 2
                    or not all(type(v) in (int, float) and np.isfinite(v) for v in interval)
                    or not 0 <= interval[0] < interval[1] <= 1):
                raise ValueError('Invalid original clause coordinate or normalized interval')
            axis, kind = pool.axes[coordinate], pool.kinds[coordinate]
            record.update(axis=axis, kind=kind)
            if kind == 'categorical':
                allowed, decoding = _decode_category(pool, coordinate, interval)
                tokens = {_canonical(label) for label in allowed}
                inside = np.asarray([_canonical(_row(pool, j)[coordinate]) in tokens for j in range(len(pool.ids))])
                record.update(decoded_allowed_labels=allowed, original_code_to_real_label=decoding)
            elif kind == 'continuous':
                values = np.asarray([float(_row(pool, j)[coordinate]) for j in range(len(pool.ids))])
                inside = (values >= interval[0]) & (values <= interval[1])
            else:
                raise ValueError('Unsupported feature kind')
            plus, minus, geometry = _select_pair(pool, coordinate, inside, available)
            a, b = _row(pool, plus), _row(pool, minus)
            if kind == 'continuous':
                if a[coordinate] == b[coordinate]:
                    raise ValueError('Inside/outside pair has no continuous focus difference')
                direction = 'increase' if a[coordinate] > b[coordinate] else 'decrease'
                focus = {axis: {'direction': direction, 'bounds': [0., 1.]}}
            else:
                focus = {axis: {'direction': 'prefer_plus', 'allowed': [a[coordinate], b[coordinate]]}}
            reason = (f'Input-defined prior: this real candidate inside the preferred {axis} region '
                      'is hypothesized to exceed the paired outside candidate under closely matched background. '
                      'This is one uncertain comparison, not a global monotonicity assertion.')
            action = {'op': 'add', 'id': None, 'plus_id': pool.ids[plus], 'minus_id': pool.ids[minus],
                      'focus': focus, 'reason': reason}
            actions.append(action)
            record.update(plus_id=pool.ids[plus], minus_id=pool.ids[minus],
                          plus_features=a, minus_features=b, focus=focus, geometry=geometry,
                          original_region_membership={'plus': True, 'minus': False})
            receipt['clauses'].append(record)
        except (ValueError, TypeError, KeyError) as error:
            receipt['uninstantiable_clauses'].append(dict(record, reason=str(error)))
    if receipt['uninstantiable_clauses']:
        raise PriorInstantiationError('At least one pressure clause cannot be instantiated; no partial prior returned', receipt)
    hypotheses, menu = [], [key for key, valid in zip(pool.ids, available) if valid]
    for start in (0, 3):
        hypotheses, events = policy.review_and_apply(pool, hypotheses,
            {'actions': actions[start:start + 3], 'summary': 'Instantiate the declared input-region priors as uncertain real comparisons.'},
            menu, t)
        receipt['review_events'].extend(events)
    if len(hypotheses) != 4 or not all(event['accepted'] and event['outcome'] == 'add' for event in receipt['review_events']):
        receipt['uninstantiable_clauses'].append({'reason': 'Semantic review did not create all four distinct prior claims'})
        raise PriorInstantiationError('All four pressure clauses must survive semantic review', receipt)
    for h, record in zip(hypotheses, receipt['clauses']):
        h['origin'] = 'synthetic_pressure'
        h['origin_clause_id'] = record['source_clause']['id']
        h['excluded_observed_ids'] = sorted(observed_ids)
        record.update(hypothesis_id=h['id'], version=h['version'], semantic_key=h['semantic_key'])
        assert h['ledger']['score'] == 0. and policy.weight(h) == 0.
    receipt['status'] = 'PASS'
    receipt['created_hypotheses'] = 4
    return hypotheses, receipt
