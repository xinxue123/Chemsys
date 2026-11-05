"""Development-only offline shadow mechanisms; no I/O, GP fitting or selection.

The field mapping was NOT registered before the archived experiments. These
functions preserve the order of archived information, but their output is NOT
prospective experimental evidence, a new trajectory, or a new SearchAUC.

Public API: field_from_commit -> initial_state -> prepare_prediction ->
score_and_update. Every function returns new data; caller inputs are untouched.
No outcome is accepted by prepare_prediction. Only score_and_update sees the
new outcome, and scores the frozen forecast BEFORE updating either calibrator.
"""
from copy import deepcopy
from hashlib import sha256
from math import exp, isfinite, log, pi, sqrt
import json

import numpy as np
from scipy.special import log_ndtr


LABEL = 'development_only_offline_mechanism_check_not_prospective_experiment'
MAPPING = 'frozen_scope_gaussian_anchor_difference_v0'


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             allow_nan=False, separators=(',', ':')).encode('utf-8')).hexdigest()


def _number(value, name):
    if isinstance(value, (bool, str)) or not isfinite(float(value)):
        raise ValueError(f'{name} must be a finite number')
    return float(value)


def _sealed(value, key):
    if value.get(key) != digest({k: v for k, v in value.items() if k != key}):
        raise ValueError(f'{key} mismatch; frozen object changed')


def _hypothesis(commit, hypothesis_id, version):
    rows = [h for h in commit['hypotheses_at_commit']
            if h['id'] == hypothesis_id and h['version'] == version]
    if len(rows) != 1:
        raise ValueError('Exactly one requested version must exist in this commit')
    h = rows[0]
    if h['status'] == 'retracted' or h['created_round'] > commit['query']:
        raise ValueError('Retracted or future hypothesis cannot enter a forecast')
    return h


def field_from_commit(commit, hypothesis_id, version, axes):
    """Freeze one deterministic field using ONLY this commit's real anchors/scope.

    axes must be the public pool feature order, never sorted category encodings.
    The RBF length on a continuous axis is half its frozen scope width; category
    kernels use equality only. Difference is scaled by its value at plus and
    clipped to [-1,1]. This extension of a pair preference is a NEW assumption.
    """
    h = _hypothesis(commit, hypothesis_id, version)
    axes = list(axes)
    scope = deepcopy(h['scope']['axes'])
    if len(set(axes)) != len(axes) or set(axes) != set(scope):
        raise ValueError('axes must exactly match the frozen scope')
    if len(h['plus_features']) != len(axes) or len(h['minus_features']) != len(axes):
        raise ValueError('Anchor features do not match the public axes')
    for rule in scope.values():
        if rule['kind'] == 'continuous':
            lo, hi = (_number(x, 'scope bound') for x in rule['bounds'])
            if not 0 <= lo < hi <= 1:
                raise ValueError('Continuous bounds must be nonempty normalized bounds')
        elif rule['kind'] == 'categorical':
            if not rule['allowed'] or not all(isinstance(x, str) for x in rule['allowed']):
                raise ValueError('Categorical scope requires true string labels')
        else:
            raise ValueError('Unknown scope kind')
    field = dict(assessment=LABEL, mapping=MAPPING, hypothesis_id=h['id'],
                 version=h['version'], semantic_key=h['semantic_key'],
                 created_round=h['created_round'], axes=axes, scope=scope,
                 plus_id=h['plus_id'], minus_id=h['minus_id'],
                 plus=dict(zip(axes, deepcopy(h['plus_features']))),
                 minus=dict(zip(axes, deepcopy(h['minus_features']))),
                 excluded_observed_ids=sorted(h.get('excluded_observed_ids', [])))
    if not _inside(field, field['plus']) or not _inside(field, field['minus']):
        raise ValueError('Both original anchors must be inside their original scope')
    if _kernel(field, field['plus'], field['minus']) >= 1 - 1e-14:
        raise ValueError('Indistinguishable anchors do not define a field')
    field['field_sha256'] = digest(field)
    return field


def _inside(field, point):
    if set(point) != set(field['axes']):
        raise ValueError('Point must contain exactly the declared physical axes')
    inside = True
    for axis, rule in field['scope'].items():
        value = point[axis]
        if rule['kind'] == 'categorical':
            if not isinstance(value, str):
                raise ValueError('Category values must be true string labels')
            inside &= value in rule['allowed']
        else:
            x = _number(value, axis)
            if not 0 <= x <= 1:
                raise ValueError('Continuous features must be normalized to [0,1]')
            inside &= rule['bounds'][0] <= x <= rule['bounds'][1]
    return bool(inside)


def _kernel(field, a, b):
    squared = 0.
    for axis, rule in field['scope'].items():
        if rule['kind'] == 'categorical':
            if a[axis] != b[axis]:
                return 0.
        else:
            scale = (rule['bounds'][1] - rule['bounds'][0]) / 2
            squared += ((a[axis] - b[axis]) / scale) ** 2
    return exp(-.5 * squared)


def field_value(field, point):
    _sealed(field, 'field_sha256')
    if not _inside(field, point):
        return 0.
    denominator = 1 - _kernel(field, field['plus'], field['minus'])
    value = (_kernel(field, point, field['plus']) -
             _kernel(field, point, field['minus'])) / denominator
    return float(np.clip(value, -1., 1.))


def gp_conditioned_value(field, point, training_points, k_noise, k_x):
    """r(x)-k_xX (K_XX+noise)^-1 r(X), masked by ORIGINAL scope.

    k_noise/k_x must be matching pre-query PRIOR covariance blocks, in common
    units, NOT posterior marginal variances. The scope mask is explicit: this
    is a heuristic local residualized field, not the exact global GP mean model.
    No matrix is inferred, fitted, or repaired. A missing block is unavailable.
    """
    raw = field_value(field, point)
    rv = np.array([field_value(field, p) for p in training_points], dtype=float)
    matrix, cross = np.asarray(k_noise, dtype=float), np.asarray(k_x, dtype=float)
    n = len(training_points)
    if (n == 0 or matrix.shape != (n, n) or cross.shape != (n,) or
            not np.isfinite(matrix).all() or not np.isfinite(cross).all() or
            not np.allclose(matrix, matrix.T, atol=1e-12, rtol=1e-10)):
        raise ValueError('Invalid matching prior covariance blocks')
    try:
        chol = np.linalg.cholesky(matrix)
        solution = np.linalg.solve(chol.T, np.linalg.solve(chol, rv))
    except np.linalg.LinAlgError as exc:
        raise ValueError('Observation covariance must be positive definite') from exc
    value = raw - float(cross @ solution)
    # Do not clip the in-scope residualized field: its magnitude need not be <=1.
    return value if _inside(field, point) else 0.


def initial_state(field, prior_variance):
    """Fresh version means a=0; no older positive scores/coefficients are copied."""
    _sealed(field, 'field_sha256')
    variance = _number(prior_variance, 'prior_variance')
    if variance <= 0:
        raise ValueError('prior_variance must be positive and specified by the caller')
    return dict(assessment=LABEL, field_sha256=field['field_sha256'],
                last_query=field['created_round'] - 1, seen=[], events=[],
                calibrators={name: dict(mean=0., variance=variance)
                             for name in ('raw_field', 'gp_conditioned_field')})


def _pair_moments(pred):
    mu, vy, md, vd, cov, tau = (_number(pred[k], k) for k in (
        'gp_mean_y', 'gp_var_y', 'contrast_mean', 'contrast_var',
        'cov_y_contrast', 'preference_noise_sd'))
    if vy <= 0 or vd < 0 or tau <= 0:
        raise ValueError('Invalid GP/pair variances')
    vz = vd + tau * tau
    rem = vz - cov * cov / vy
    if rem < -1e-8 * vz:
        raise ValueError('GP pair covariance is inconsistent')
    alpha = md / sqrt(vz)
    lam = exp(-.5 * alpha * alpha - .5 * log(2 * pi) - float(log_ndtr(alpha)))
    return mu, vy, md, cov, alpha, max(rem, 1e-15 * vz), mu + cov / sqrt(vz) * lam


def prepare_prediction(commit, field, state, selected_features, prefix, conditioning=None):
    """Freeze all shadow forecasts; accepts no new outcome.

    prefix is the exact already-observed history serialized by the commit.
    Optional conditioning keys: history_sha256, gp_fit_key, training_ids,
    training_points, k_noise, k_x. Data must come from that prefix's saved GP
    prior; missing blocks leave this arm explicitly unavailable (no fallback).
    """
    _sealed(field, 'field_sha256')
    current = field_from_commit(commit, field['hypothesis_id'], field['version'], field['axes'])
    if current['field_sha256'] != field['field_sha256'] or state['field_sha256'] != field['field_sha256']:
        raise ValueError('Version field changed or wrong calibration state')
    t, selected = commit['query'], commit['selected_id']
    ids = [r['candidate_id'] for r in prefix]
    if (digest(prefix) != commit['history_sha256'] or len(prefix) != t + 3 or
            len(set(ids)) != len(ids) or selected in ids):
        raise ValueError('Forecast requires the exact unmodified pre-query prefix')
    past = {r['candidate_id']: r['y'] for r in prefix}
    if any(e['point_id'] not in past or past[e['point_id']] != e['observed_y'] or
           e['query_round'] >= t for e in state['events']):
        raise ValueError('Calibration ledger disagrees with the committed observed prefix')
    if (t != state['last_query'] + 1 or selected in state['seen'] or
            selected in field['excluded_observed_ids']):
        raise ValueError('Duplicate, retrospective, or skipped-round prediction')
    if not state['events'] and t != field['created_round']:
        raise ValueError('Replay must start at this version creation, not skip early evidence')
    rows = [s for s in commit['shadows'] if s['id'] == field['hypothesis_id'] and
            s['version'] == field['version']]
    if len(rows) != 1:
        raise ValueError('One original pair shadow is required; do not synthesize GP predictions')
    original = rows[0]
    pred = deepcopy(original['prediction'])
    _sealed(pred, 'prediction_sha256')
    if (pred['point_id'] != selected or pred['query_round'] != t or
            pred['hypothesis_id'] != field['hypothesis_id'] or pred['version'] != field['version'] or
            pred['semantic_key'] != field['semantic_key']):
        raise ValueError('Original pair forecast does not match selected point/version')
    mu, var, _, _, _, _, pair_mean = _pair_moments(pred)
    inside = _inside(field, selected_features)
    basis = {'raw_field': field_value(field, selected_features), 'gp_conditioned_field': None}
    conditioning_audit = None
    if conditioning is not None:
        if (conditioning['history_sha256'] != commit['history_sha256'] or
                conditioning['gp_fit_key'] != commit['GP_fit_key'] or
                conditioning['training_ids'] != ids or
                len(conditioning['training_points']) != len(ids)):
            raise ValueError('Conditioning matrix belongs to a different GP/prefix')
        basis['gp_conditioned_field'] = gp_conditioned_value(field, selected_features,
            conditioning['training_points'], conditioning['k_noise'], conditioning['k_x'])
        conditioning_audit = dict(history_sha256=conditioning['history_sha256'],
            gp_fit_key=conditioning['gp_fit_key'], blocks_sha256=digest(dict(
                training_ids=conditioning['training_ids'], training_points=conditioning['training_points'],
                k_noise=np.asarray(conditioning['k_noise'], dtype=float).tolist(),
                k_x=np.asarray(conditioning['k_x'], dtype=float).tolist())),
            scope_mask='original_scope_reapplied_explicit_local_heuristic')
    arms = {'pair': dict(available=True, mean=pair_mean if inside else mu,
                        original_pair_informative=bool(original['informative']))}
    for name, r in basis.items():
        if r is None:
            arms[name] = dict(available=False, reason='matching_prior_covariance_blocks_not_supplied')
            continue
        a = state['calibrators'][name]
        a_mean = _number(a['mean'], 'calibration mean')
        a_variance = _number(a['variance'], 'calibration variance')
        if a_variance < 0:
            raise ValueError('Calibration variance must be nonnegative')
        arms[name] = dict(available=True, basis=_number(r, 'field basis'),
                         mean=_number(mu + a_mean * r, 'field mean'),
                         variance=_number(var + r * r * a_variance, 'field variance'))
    snapshot = dict(assessment=LABEL, field_sha256=field['field_sha256'],
                    state_sha256=digest(state), hypothesis_id=field['hypothesis_id'],
                    version=field['version'], point_id=selected, query_round=t,
                    history_sha256=commit['history_sha256'], gp_fit_key=commit['GP_fit_key'],
                    in_frozen_scope=inside, gp_mean=mu, gp_variance=var,
                    pair_prediction=pred, arms=arms, conditioning_audit=conditioning_audit)
    snapshot['snapshot_sha256'] = digest(snapshot)
    return snapshot


def _normal_logpdf(y, mean, variance):
    return -.5 * (log(2 * pi * variance) + (y - mean) ** 2 / variance)


def score_and_update(state, snapshot, outcome):
    """Score fixed forecasts, then KF-update; returns (new_state, receipt).

    outcome has candidate_id, query_round, y. This does not change experiment
    history, policy weights, or selected points. Reusing a snapshot with the
    returned state is rejected. Versions have independent zero-mean starts.
    """
    _sealed(snapshot, 'snapshot_sha256')
    if snapshot['state_sha256'] != digest(state) or snapshot['field_sha256'] != state['field_sha256']:
        raise ValueError('Snapshot is stale, reused, or from another state/version')
    if (outcome['candidate_id'] != snapshot['point_id'] or
            outcome['query_round'] != snapshot['query_round'] or
            snapshot['point_id'] in state['seen'] or snapshot['query_round'] != state['last_query'] + 1):
        raise ValueError('Outcome must match the new committed experiment once')
    y = _number(outcome['y'], 'observed y')
    mu, var = snapshot['gp_mean'], snapshot['gp_variance']
    base_squared_error = (y - mu) ** 2
    base_logpdf = _normal_logpdf(y, mu, var)
    metrics = {}
    for name, arm in snapshot['arms'].items():
        if not arm['available']:
            metrics[name] = deepcopy(arm)
            continue
        mean_gain = base_squared_error - (y - arm['mean']) ** 2
        if name == 'pair':
            _, _, md, cov, alpha, rem, _ = _pair_moments(snapshot['pair_prediction'])
            beta = (md + cov / var * (y - mu)) / sqrt(rem)
            log_gain = float(log_ndtr(beta) - log_ndtr(alpha)) if snapshot['in_frozen_scope'] else 0.
        else:
            log_gain = _normal_logpdf(y, arm['mean'], arm['variance']) - base_logpdf
        metrics[name] = dict(available=True, mean_squared_error_gain=mean_gain,
                            same_variance_mean_logscore_gain=mean_gain / (2 * var),
                            predictive_logscore_gain=log_gain,
                            scored_before_calibration=True)
    new = deepcopy(state)
    updates = {}
    for name in ('raw_field', 'gp_conditioned_field'):
        arm = snapshot['arms'][name]
        if not arm['available']:
            updates[name] = dict(updated=False, reason=arm['reason'])
            continue
        a, r = new['calibrators'][name], arm['basis']
        before = deepcopy(a)
        gain = a['variance'] * r / (var + r * r * a['variance'])
        a['mean'] += gain * ((y - mu) - a['mean'] * r)
        a['variance'] *= 1 - gain * r
        updates[name] = dict(updated=bool(r != 0), gain=gain, before=before, after=deepcopy(a))
    receipt = dict(assessment=LABEL, hypothesis_id=snapshot['hypothesis_id'],
                   version=snapshot['version'], point_id=snapshot['point_id'],
                   query_round=snapshot['query_round'], observed_y=y,
                   snapshot_sha256=snapshot['snapshot_sha256'],
                   in_frozen_scope=snapshot['in_frozen_scope'],
                   original_pair_informative=snapshot['arms']['pair']['original_pair_informative'],
                   metrics=metrics, calibration_updates=updates)
    receipt['event_id'] = digest(receipt)
    new['seen'].append(snapshot['point_id'])
    new['last_query'] = snapshot['query_round']
    new['events'].append(receipt)
    return new, receipt
