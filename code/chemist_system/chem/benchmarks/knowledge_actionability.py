"""Pure geometry hints for an existing unqueried LLM candidate menu.

These hints are not chemistry recommendations or complete validation gates.
They use only input features and the caller's own already observed history.
No GP, outcome lookup, files, model calls, or rule/credit state is consulted.

For each listed reference, choose one or two focus axes, include every required
axis, and choose only from required + additional_changed_axes. The compressed
description represents ALL legal focus sets for that pair. It supplies no
direction or categorical preference. Bounds and chemical claims remain for the
LLM to state and for the unchanged policy validator to check.
"""
from itertools import combinations
from math import isfinite
from numbers import Real
from collections.abc import Mapping

BACKGROUND_RADIUS = .25
COMPARISON_TOLERANCE = 1e-12


def _get(pool, field):
    return pool[field] if isinstance(pool, Mapping) else getattr(pool, field)


def _schema(pool):
    axes = tuple(_get(pool, 'axes'))
    kinds = _get(pool, 'kinds')
    kinds = tuple(kinds[a] for a in axes) if isinstance(kinds, Mapping) else tuple(kinds)
    if (not axes or len(set(axes)) != len(axes) or not all(isinstance(a, str) for a in axes)
            or len(kinds) != len(axes) or any(k not in ('continuous', 'categorical') for k in kinds)):
        raise ValueError('Expected unique named axes with continuous/categorical kinds')
    return axes, kinds


def _features(pool, point_id, kinds):
    index = _get(pool, 'ix').get(point_id)
    if index is None or _get(pool, 'ids')[index] != point_id:
        raise ValueError('Unknown point ID')
    row = tuple(_get(pool, 'features')[index])
    if len(row) != len(kinds):
        raise ValueError('Point feature dimension differs from axes')
    normalized = []
    for value, kind in zip(row, kinds):
        if kind == 'continuous':
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)) or not 0 <= value <= 1:
                raise ValueError('Continuous input features must be finite normalized numbers')
            normalized.append(float(value))
        else:
            if not isinstance(value, str):
                raise ValueError('Categorical input features must use their real string labels')
            normalized.append(value)
    return tuple(normalized)


def _pair(axes, kinds, plus, reference):
    required, additional = [], []
    for axis, kind, x, r in zip(axes, kinds, plus, reference):
        if x == r:
            continue
        if kind == 'categorical' or abs(x-r) > BACKGROUND_RADIUS + COMPARISON_TOLERANCE:
            required.append(axis)
        else:
            additional.append(axis)
    if len(required) > 2 or not (required or additional):
        return None
    return dict(required_focus_axes=required, additional_changed_axes=additional)


def legal_focus_combinations(pool, plus_id, reference_id):
    """Expand all legal one/two-axis sets for tests/diagnostics, not prompts."""
    axes, kinds = _schema(pool)
    pair = _pair(axes, kinds, _features(pool, plus_id, kinds), _features(pool, reference_id, kinds))
    if pair is None:
        return []
    required = set(pair['required_focus_axes'])
    changed = required | set(pair['additional_changed_axes'])
    ordered = [axis for axis in axes if axis in changed]
    return [list(c) for size in (1, 2) for c in combinations(ordered, size) if required <= set(c)]


def candidate_support(pool, history, menu_ids, maximum_references=4):
    """Return compact pair support, preserving the existing menu order.

    All own observed references are scanned. Per candidate, retain at most the
    highest-valued ``maximum_references`` geometrically legal references, tied
    by physical features. Candidate IDs never determine rank. This is NOT a
    preselection of the globally highest four observations: legal references
    outside that global set may be displayed. Counts expose any truncation.

    The wrapper must describe this limit and the focus-combination rule once.
    Optional window/cost flags belong to its explicit context, not this helper.
    """
    if type(maximum_references) is not int or maximum_references < 1:
        raise ValueError('maximum_references must be a positive integer')
    axes, kinds = _schema(pool)
    known, references = set(), []
    for row in history:
        key, value = row['candidate_id'], row['y']
        if key in known or isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)):
            raise ValueError('Own history must contain unique point IDs and finite observed values')
        feature = _features(pool, key, kinds)
        known.add(key)
        references.append((key, float(value), feature))
    references.sort(key=lambda item: (-item[1], item[2]))
    menu_ids = list(menu_ids)
    if len(set(menu_ids)) != len(menu_ids) or set(menu_ids) & known:
        raise ValueError('Menu must contain unique, unqueried candidate IDs')
    result = []
    for point_id in menu_ids:
        feature = _features(pool, point_id, kinds)
        legal = []
        for reference_id, _, reference in references:
            pair = _pair(axes, kinds, feature, reference)
            if pair is not None:
                legal.append(dict(reference_id=reference_id, **pair))
        result.append(dict(point_id=point_id, references=legal[:maximum_references],
            compatible_reference_count=len(legal), references_truncated=len(legal) > maximum_references))
    return result


__all__ = ['BACKGROUND_RADIUS', 'COMPARISON_TOLERANCE', 'candidate_support', 'legal_focus_combinations']
