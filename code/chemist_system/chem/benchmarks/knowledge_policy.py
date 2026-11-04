"""Versioned, prospective knowledge evidence for mixed finite candidate pools.

A claim conditions the GP on ONE real pairwise preference. Its declared scope
limits where that conditional model may act; it is not a global monotonic law.
Continuous focus bounds are mandatory and normalized to [0, 1]. Categorical
features are compared by real label equality, never by ordinal distance.

Inherited negative scores are explicitly conservative trust penalties, not new
version likelihood observations. New versions always start with score=0.
The caller must persist each shadow before observing its selected experiment.
"""
from copy import deepcopy
from hashlib import sha256
import json
from math import exp, isfinite, log, sqrt

import numpy as np
from scipy.special import log_ndtr

MAX_ACTIONS = 3
MAX_LIVE_CLAIMS = 4
BACKGROUND_RADIUS = .25
PREFERENCE_NOISE_SD = .06
MAX_WEIGHT = .5


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             allow_nan=False, separators=(',', ':')).encode('utf-8')).hexdigest()


def _get(pool, name):
    return pool[name] if isinstance(pool, dict) else getattr(pool, name)


def _kinds(pool):
    kinds = _get(pool, 'kinds')
    return dict(kinds) if isinstance(kinds, dict) else dict(zip(_get(pool, 'axes'), kinds))


def _features(pool, point_id):
    row = _get(pool, 'features')[_get(pool, 'ix')[point_id]]
    return [v.item() if isinstance(v, np.generic) else v for v in row]


def _same(a, b):
    return _digest(a) == _digest(b)


def _finite_number(value):
    return type(value) in (int, float) and isfinite(value)


def _contract(h):
    return {k: h[k] for k in ('plus_id', 'minus_id', 'plus_features', 'minus_features', 'focus', 'scope')}


def _semantic(h):
    # Candidate IDs are trace identifiers, not part of the scientific meaning.
    return _digest({k: h[k] for k in ('plus_features', 'minus_features', 'focus', 'scope')})


def _build_contract(pool, action, menu_ids):
    plus, minus = action.get('plus_id'), action.get('minus_id')
    if not isinstance(plus, str) or not isinstance(minus, str) or plus == minus:
        raise ValueError('Two distinct real candidate IDs are required')
    if plus not in menu_ids or minus not in menu_ids:
        raise ValueError('Both candidates must occur in the displayed unqueried menu')
    a, b = _features(pool, plus), _features(pool, minus)
    if _same(a, b):
        raise ValueError('The compared candidates have identical physical features')
    axes, kinds = _get(pool, 'axes'), _kinds(pool)
    focus = action.get('focus')
    if not isinstance(focus, dict) or not 1 <= len(focus) <= 2 or set(focus) - set(axes):
        raise ValueError('One or two named focus axes are required')
    normalized, scopes = {}, {}
    for j, axis in enumerate(axes):
        kind = kinds[axis]
        if kind not in ('continuous', 'categorical'):
            raise ValueError('Unsupported feature kind')
        if kind == 'continuous':
            if not all(_finite_number(v) and 0 <= v <= 1 for v in (a[j], b[j])):
                raise ValueError('Continuous features must be normalized to [0,1]')
            if axis in focus:
                spec = focus[axis]
                if not isinstance(spec, dict) or set(spec) != {'direction', 'bounds'}:
                    raise ValueError('Continuous focus requires explicit direction and normalized bounds')
                direction, bounds = spec['direction'], spec['bounds']
                if direction not in ('increase', 'decrease'):
                    raise ValueError('Continuous direction must be increase or decrease')
                sign = 1 if direction == 'increase' else -1
                if sign * (a[j] - b[j]) <= 0:
                    raise ValueError('Continuous direction contradicts the real candidate pair')
                if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2
                        or not all(_finite_number(v) for v in bounds)
                        or not 0 <= bounds[0] < bounds[1] <= 1
                        or min(a[j], b[j]) < bounds[0] or max(a[j], b[j]) > bounds[1]):
                    raise ValueError('Focus bounds must contain both anchors within [0,1]')
                lo, hi = map(float, bounds)
                normalized[axis] = {'direction': direction, 'bounds': [lo, hi]}
            else:
                if abs(a[j] - b[j]) > BACKGROUND_RADIUS + 1e-12:
                    raise ValueError('Background continuous axis changes by more than 0.25')
                lo = max(0., a[j] - BACKGROUND_RADIUS, b[j] - BACKGROUND_RADIUS)
                hi = min(1., a[j] + BACKGROUND_RADIUS, b[j] + BACKGROUND_RADIUS)
            scopes[axis] = {'kind': kind, 'bounds': [float(lo), float(hi)]}
        else:
            if axis in focus:
                spec = focus[axis]
                if spec == 'prefer_plus':
                    spec = {'direction': 'prefer_plus', 'allowed': [a[j], b[j]]}
                if (not isinstance(spec, dict) or set(spec) != {'direction', 'allowed'}
                        or spec['direction'] != 'prefer_plus' or _same(a[j], b[j])):
                    raise ValueError('Categorical focus requires a real retype and prefer_plus')
                allowed = spec['allowed']
                if not isinstance(allowed, list) or not allowed:
                    raise ValueError('Categorical allowed must be a nonempty real-label list')
                real = {_digest(_features(pool, p)[j]) for p in _get(pool, 'ids')}
                tokens = [_digest(v) for v in allowed]
                if (len(set(tokens)) != len(tokens) or set(tokens) - real
                        or _digest(a[j]) not in tokens or _digest(b[j]) not in tokens):
                    raise ValueError('Categorical allowed must contain both real anchors and no invented label')
                allowed = sorted(deepcopy(allowed), key=_digest)
                normalized[axis] = {'direction': 'prefer_plus', 'allowed': allowed}
            else:
                if not _same(a[j], b[j]):
                    raise ValueError('A background categorical axis must remain identical')
                allowed = [deepcopy(a[j])]
            scopes[axis] = {'kind': kind, 'allowed': allowed}
    return {'plus_id': plus, 'minus_id': minus, 'plus_features': a, 'minus_features': b,
            'focus': normalized, 'scope': {'axes': scopes, 'background_radius': BACKGROUND_RADIUS}}


def scope_indices(pool, h, indices):
    """Return input indices lying in the frozen explicit mixed-feature scope."""
    indices = np.asarray(indices, dtype=int)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(_get(pool, 'ids'))):
        raise ValueError('Candidate indices must be a valid one-dimensional index list')
    if not len(indices):
        return indices
    axes = _get(pool, 'axes')
    keep = []
    for index in indices:
        row = _features(pool, _get(pool, 'ids')[int(index)])
        accept = True
        for j, axis in enumerate(axes):
            spec = h['scope']['axes'][axis]
            if spec['kind'] == 'continuous':
                lo, hi = spec['bounds']
                accept &= lo - 1e-12 <= float(row[j]) <= hi + 1e-12
            else:
                accept &= any(_same(row[j], label) for label in spec['allowed'])
        if accept:
            keep.append(int(index))
    return np.asarray(keep, dtype=int)


def weight(h):
    """Bounded evidence weight, with separately identified inherited penalties."""
    if h['status'] != 'active':
        return 0.
    ledger = h['ledger']
    penalty = sum(min(0., float(row['delta_log_score']))
                  for row in h.get('inherited_negative_evidence', []) if row['applicable'])
    effective = float(ledger['score']) + penalty
    if effective <= 0 or not ledger['events']:
        return 0.
    # Stable excess over a 0.5 starting model weight; not chemistry-truth probability.
    excess = 2. / (1. + exp(-effective)) - 1.
    return min(float(ledger.get('cap', MAX_WEIGHT)), max(0., excess))


def _applicable_negative(pool, new, row):
    old = row['source_contract']
    if set(old['focus']) != set(new['focus']):
        return False
    axes = _get(pool, 'axes')
    for axis, spec in new['focus'].items():
        if old['focus'][axis]['direction'] != spec['direction']:
            return False
        if spec['direction'] == 'prefer_plus':
            j = axes.index(axis)
            if (not _same(old['plus_features'][j], new['plus_features'][j])
                    or not _same(old['minus_features'][j], new['minus_features'][j])):
                return False
    j = _get(pool, 'ix').get(row['point_id'])
    return j is not None and bool(len(scope_indices(pool, new, [j])))


def _inherit_negative(pool, parent, new):
    records = {row['event_id']: deepcopy(row) for row in parent.get('inherited_negative_evidence', [])}
    for event in parent['ledger']['events']:
        if event['delta_log_score'] is not None and event['delta_log_score'] < 0:
            records[event['event_id']] = {
                'event_id': event['event_id'], 'source_id': parent['id'], 'source_version': parent['version'],
                'source_contract': _contract(parent), 'point_id': event['point_id'],
                'query_round': event['query_round'], 'prediction': deepcopy(event['prediction']),
                'observed_y': event['observed_y'], 'delta_log_score': event['delta_log_score']}
    result = []
    for record in records.values():
        record['applicable'] = _applicable_negative(pool, new, record)
        record['interpretation'] = 'conservative trust penalty, not new-version log-likelihood'
        result.append(record)
    return sorted(result, key=lambda row: row['event_id'])


def review_and_apply(pool, hypotheses, proposal, menu_ids, t):
    """Review <=3 actions; return new hypotheses and events without changing inputs.

    menu_ids must already be the caller's displayed unqueried menu. Each rejected
    action leaves the existing hypothesis untouched. Historical versions remain.
    """
    if type(t) is not int or t < 1:
        raise ValueError('Creation round must be a positive integer')
    if (not isinstance(proposal, dict) or set(proposal) != {'actions', 'summary'}
            or not isinstance(proposal['summary'], str) or not 1 <= len(proposal['summary'].strip()) <= 1600
            or not isinstance(proposal['actions'], list) or len(proposal['actions']) > MAX_ACTIONS):
        raise ValueError('Proposal requires a concise summary and at most three actions')
    menu_ids = list(menu_ids)
    if len(set(menu_ids)) != len(menu_ids) or set(menu_ids) - set(_get(pool, 'ids')):
        raise ValueError('Menu contains duplicate or unknown IDs')
    result, events, touched = deepcopy(hypotheses), [], set()
    for position, action in enumerate(proposal['actions']):
        event = {'round': t, 'action_index': position, 'accepted': False}
        try:
            if not isinstance(action, dict) or set(action) != {'op', 'id', 'plus_id', 'minus_id', 'focus', 'reason'}:
                raise ValueError('Action fields must be op,id,plus_id,minus_id,focus,reason')
            op, hid = action['op'], action['id']
            event.update(op=op, id=hid)
            if op not in ('add', 'retain', 'revise', 'retract'):
                raise ValueError('Unknown knowledge operation')
            if not isinstance(action['reason'], str) or not 1 <= len(action['reason'].strip()) <= 1000:
                raise ValueError('A concise action reason is required')
            if op == 'add' and hid is not None:
                raise ValueError('Add requires id=null')
            matches = [h for h in result if h['id'] == hid]
            parent = max(matches, key=lambda h: h['version']) if matches else None
            if op != 'add' and (not isinstance(hid, str) or parent is None):
                raise ValueError('Operation must name an existing knowledge ID')
            if hid is not None and hid in touched:
                raise ValueError('At most one operation per existing ID in a request')
            if parent and any(event['query_round'] >= t for event in parent['ledger']['events']):
                raise ValueError('Knowledge operations must precede the current round outcome')
            if op in ('retain', 'retract'):
                if action['plus_id'] is not None or action['minus_id'] is not None or action['focus'] != {}:
                    raise ValueError('Retain/retract cannot alter candidate or scope fields')
                if parent['status'] == 'retracted':
                    raise ValueError('A retracted version requires an explicit new revision')
                if op == 'retract':
                    parent['status'] = 'retracted'
                    parent['retired_round'] = t
                    parent['retirement_reason'] = action['reason']
                    parent['ledger']['weight'] = 0.
                event.update(accepted=True, outcome=op, version=parent['version'])
                touched.add(hid)
            else:
                contract = _build_contract(pool, action, set(menu_ids))
                semantic_key = _semantic(contract)
                same = [h for h in result if h['semantic_key'] == semantic_key]
                existing = max(same, key=lambda h: (h['created_round'], h['version'])) if same else None
                if existing is not None and not (op == 'revise' and existing['status'] == 'retracted'):
                    # A wording change, duplicate add, or reverting to an old
                    # semantic version cannot manufacture a fresh trust ledger.
                    event.update(accepted=True, outcome='duplicate_semantics_preserved',
                                 id=existing['id'], version=existing['version'], status=existing['status'])
                    if hid is not None:
                        touched.add(hid)
                else:
                    if op == 'add':
                        for old in result:
                            same_pair = (_same(old['plus_features'], contract['plus_features']) and
                                         _same(old['minus_features'], contract['minus_features']))
                            reversed_pair = (_same(old['plus_features'], contract['minus_features']) and
                                             _same(old['minus_features'], contract['plus_features']))
                            if same_pair or reversed_pair:
                                raise ValueError('An existing real candidate pair requires revision of its original ID')
                    live = sum(h['status'] != 'retracted' for h in result)
                    if live - int(parent is not None and parent['status'] != 'retracted') >= MAX_LIVE_CLAIMS:
                        raise ValueError('At most four live claims, including dormant claims')
                    if parent is None:
                        used = {h['id'] for h in result}
                        number = 1
                        while f'H{number:03d}' in used:
                            number += 1
                        hid = f'H{number:03d}'
                    new = dict(contract, id=hid, version=parent['version'] + 1 if parent else 1,
                        parent={'id': parent['id'], 'version': parent['version']} if parent else None,
                        created_round=t, status='active', reason=action['reason'], semantic_key=semantic_key,
                        tau_y=PREFERENCE_NOISE_SD,
                        ledger={'score': 0., 'seen': [], 'events': [], 'weight': 0., 'cap': MAX_WEIGHT},
                        inherited_negative_evidence=[], excluded_observed_ids=[])
                    if parent:
                        new['inherited_negative_evidence'] = _inherit_negative(pool, parent, new)
                        if existing is not None and existing is not parent:
                            # Returning to old semantics remains an explicit new
                            # version; its former negative evidence cannot vanish.
                            inherited = {row['event_id']: row for row in new['inherited_negative_evidence']}
                            inherited.update({row['event_id']: row for row in _inherit_negative(pool, existing, new)})
                            new['inherited_negative_evidence'] = sorted(inherited.values(), key=lambda row: row['event_id'])
                        new['excluded_observed_ids'] = sorted(set(parent['ledger']['seen']) |
                                                              set(parent.get('excluded_observed_ids', [])) |
                                                              (set(existing['ledger']['seen']) if existing else set()))
                        parent['status'] = 'retracted'
                        parent['retired_round'] = t
                        parent['retirement_reason'] = 'superseded by explicit revision'
                        parent['ledger']['weight'] = 0.
                    result.append(new)
                    touched.add(hid)
                    event.update(accepted=True, outcome=op, id=hid, version=new['version'],
                                 inherited_negative_count=len(new['inherited_negative_evidence']))
        except (ValueError, TypeError, KeyError) as error:
            event['reason'] = str(error)
        events.append(event)
    return result, events


def refresh_coverage(pool, hypotheses, eligible_indices, t):
    """Reversible dormancy is candidate unavailability, not chemical refutation."""
    result, events = deepcopy(hypotheses), []
    for h in result:
        if h['status'] == 'retracted':
            continue
        support = len(scope_indices(pool, h, eligible_indices))
        status = 'active' if support else 'dormant'
        if status != h['status']:
            events.append({'round': t, 'id': h['id'], 'version': h['version'],
                           'from_status': h['status'], 'to_status': status,
                           'reason': 'eligible_scope_overlap' if support else 'no_eligible_scope_overlap'})
        h['status'] = status
        h['coverage'] = {'round': t, 'eligible_count': support}
        h['ledger']['weight'] = weight(h)
    return result, events


def _moments(prediction):
    values = [prediction[k] for k in ('gp_mean_y', 'gp_var_y', 'contrast_mean', 'contrast_var',
                                      'cov_y_contrast', 'preference_noise_sd')]
    if not all(_finite_number(v) for v in values):
        raise ValueError('Prediction moments must be finite real numbers')
    _, vy, _, vd, cov, tau = values
    if vy <= 0 or vd < 0 or tau <= 0:
        raise ValueError('Prediction variances/noise are invalid')
    vz = vd + tau * tau
    rem = vz - cov * cov / vy
    if rem < -1e-8 * vz:
        raise ValueError('Prediction covariance is inconsistent with its variances')
    return vz, max(rem, 1e-15 * vz)


def shadow(h, point_id, t, mu, predvar, contrastmean, var, cov):
    """Freeze one public prediction BEFORE its experiment outcome is revealed."""
    if h['status'] == 'retracted' or type(t) is not int or t < h['created_round']:
        raise ValueError('Prediction must belong to a live version at or after creation')
    if not isinstance(point_id, str) or not point_id:
        raise ValueError('A real selected point ID is required')
    if h['semantic_key'] != _semantic(h):
        raise ValueError('Hypothesis contract changed without a new version')
    prediction = {'hypothesis_id': h['id'], 'version': h['version'], 'point_id': point_id,
        'query_round': t, 'semantic_key': h['semantic_key'], 'gp_mean_y': float(mu),
        'gp_var_y': float(predvar), 'contrast_mean': float(contrastmean),
        'contrast_var': float(var), 'cov_y_contrast': float(cov), 'preference_noise_sd': float(h['tau_y'])}
    vz, _ = _moments(prediction)
    alpha = float(contrastmean) / sqrt(vz)
    lam = exp(-.5*alpha*alpha - .5*log(2*np.pi) - float(log_ndtr(alpha)))
    prediction['knowledge_mean_y'] = float(mu) + float(cov)/sqrt(vz)*lam
    prediction['knowledge_var_y'] = max(0., float(predvar) - float(cov)**2/vz*lam*(lam+alpha))
    prediction['distribution'] = 'one-sign conditional Gaussian; skew-normal, not moment-matched Gaussian'
    prediction['prediction_sha256'] = _digest(prediction)
    return prediction


def apply_evidence(h, prediction, y, informative):
    """Consume frozen evidence once; mutate only this version's ledger."""
    if type(informative) is not bool or not _finite_number(y):
        raise ValueError('Feedback requires a finite outcome and boolean informativeness')
    if h['status'] == 'retracted':
        raise ValueError('Cannot apply new evidence to a retracted version')
    if (prediction['hypothesis_id'] != h['id'] or prediction['version'] != h['version']
            or prediction['semantic_key'] != h['semantic_key'] or _semantic(h) != h['semantic_key']):
        raise ValueError('Prediction belongs to another knowledge version or contract')
    expected_hash = _digest({k: v for k, v in prediction.items() if k != 'prediction_sha256'})
    if prediction.get('prediction_sha256') != expected_hash:
        raise ValueError('Frozen prediction was modified')
    t, point = prediction['query_round'], prediction['point_id']
    ledger = h['ledger']
    if (type(t) is not int or t < h['created_round']
            or any(event['query_round'] >= t for event in ledger['events'])):
        raise ValueError('Evidence is retrospective or out of temporal order')
    if point in ledger['seen'] or point in h.get('excluded_observed_ids', []):
        raise ValueError('An already observed point cannot become new evidence')
    vz, rem = _moments(prediction)
    before = weight(h)
    gain = None
    if informative:
        alpha = prediction['contrast_mean'] / sqrt(vz)
        beta = (prediction['contrast_mean'] + prediction['cov_y_contrast'] / prediction['gp_var_y'] *
                (y - prediction['gp_mean_y'])) / sqrt(rem)
        gain = float(log_ndtr(beta) - log_ndtr(alpha))
    event = {'hypothesis_id': h['id'], 'version': h['version'], 'point_id': point, 'query_round': t,
             'prediction': deepcopy(prediction), 'observed_y': float(y), 'informative': informative,
             'delta_log_score': gain}
    event['event_id'] = _digest(event)
    ledger['seen'] = sorted(set(ledger['seen']) | {point})
    ledger['events'].append(event)
    if gain is not None:
        ledger['score'] += gain
    ledger['weight'] = weight(h)
    return {'event_id': event['event_id'], 'hypothesis_id': h['id'], 'version': h['version'],
            'point_id': point, 'query_round': t, 'informative': informative,
            'delta_log_score': gain, 'weight_before': before, 'weight_after': ledger['weight'],
            'score_after': ledger['score'],
            'inherited_negative_penalty': sum(min(0., row['delta_log_score'])
                for row in h.get('inherited_negative_evidence', []) if row['applicable'])}
