"""Predeclared repairs to the frozen regional G3 policy; no oracle access.

R1: named inputs and typed, checked factual citations.
R2: also neutralize deployment for a B4 conflict, keeping shadow calibration.
R3: also subtract one amplitude SD for geometries other than the initial one.
"""
from copy import deepcopy
import math

import numpy as np

from . import regional_ablation_suzuki_engine as original
from .suzuki_domain import CATALYSTS, LOW, HIGH

METHODS = ('R1_grounded', 'R2_conflict_fallback', 'R3_conservative_region')
PARAMETERS = {**deepcopy(original.PARAMETERS), 'new_region_sd_penalty': 1.,
              'evidence_numeric_tolerance': 1e-8, 'normalized_decimals': 6,
              'physical_decimals': 3, 'yield_percent_decimals': 3}
VARIABLES = original.VARIABLES


class GroundedRegionalCache(original.RegionalAblationCache):
    # Cache transport only: original _evaluate does not ensure its own file.
    # Scores and model recipes remain the frozen original recipes.
    def _evaluate(self, history, tilt, value, fit, key, amplitude, spec):
        self.ensure(key + '.npz')
        return super()._evaluate(history, tilt, value, fit, key, amplitude, spec)


def named_features(pool, index):
    z = np.asarray(pool['z'][index][:3], dtype=float)
    return {'normalized': {name: round(float(value), 6) for name, value in zip(VARIABLES[:3], z)},
            'physical': {name: round(float(value), 3) for name, value in zip(VARIABLES[:3], LOW + z * (HIGH - LOW))},
            'catalyst': pool['points'][index]['catalyst'],
            'catalyst_index': CATALYSTS.index(pool['points'][index]['catalyst'])}


def evidence_fact(observation, axis):
    """The complete typed factual citation for ONE observed row and ONE axis."""
    nominal = axis == 'catalyst'
    return {'index': observation['index'], 'axis': axis,
            'normalized_value': None if nominal else observation['normalized'][axis],
            'physical_value': observation['catalyst'] if nominal else observation['physical'][axis],
            'yield_percent': observation['yield_percent']}


def _number_equal(actual, expected):
    return (type(actual) in (int, float) and math.isfinite(actual)
            and abs(actual - expected) <= PARAMETERS['evidence_numeric_tolerance'])


def validate_refresh(data, knowledge, slots, observations, pool):
    """Check named-axis facts, then apply the unchanged original region schema.

    Checking factual citations does not certify causal or chemical conclusions
    in free-text reason/summary. Those are explicitly outside this validator.
    """
    if not isinstance(data, dict) or set(data) != {'actions', 'summary'}:
        raise ValueError('expected actions,summary')
    if not isinstance(data['actions'], list) or len(data['actions']) > 4:
        raise ValueError('at most4 actions')
    by_index = {row['index']: row for row in observations}
    converted = deepcopy(data)
    facts = []
    for row, target in zip(data['actions'], converted['actions']):
        if not isinstance(row, dict) or set(row) != {'id', 'operation', 'bounds', 'categories', 'reason', 'support'}:
            raise ValueError('invalid action fields')
        if row['id'] not in slots:
            raise ValueError('unknown fixed slot')
        support = row['support']
        if not isinstance(support, list) or not support:
            raise ValueError('support requires typed observed facts')
        checked = []
        axis = VARIABLES[slots[row['id']]]
        for fact in support:
            if not isinstance(fact, dict) or set(fact) != {'index', 'axis', 'normalized_value', 'physical_value', 'yield_percent'}:
                raise ValueError('invalid typed support fields')
            if type(fact['index']) != int or fact['index'] not in by_index or fact['index'] in checked:
                raise ValueError('unobserved or duplicate support index')
            if fact['axis'] != axis:
                raise ValueError('support axis must match the fixed slot axis')
            expected = evidence_fact(by_index[fact['index']], axis)
            if axis == 'catalyst':
                if fact['normalized_value'] is not None or fact['physical_value'] != expected['physical_value']:
                    raise ValueError('nominal catalyst citation mismatch')
            else:
                for key in ('normalized_value', 'physical_value'):
                    if not _number_equal(fact[key], expected[key]):
                        raise ValueError('observed ' + axis + ' ' + key + ' mismatch')
            if not _number_equal(fact['yield_percent'], expected['yield_percent']):
                raise ValueError('observed yield citation mismatch')
            checked.append(fact['index'])
            facts.append({'slot': row['id'], **deepcopy(fact)})
        target['support'] = checked
    result = original.validate_refresh(converted, knowledge, slots, len(observations), pool)
    result['actions'] = deepcopy(data['actions'])
    result['checked_facts'] = facts
    return result


def deployment(method, knowledge, retained, initial_key, bank, prior, diagnosis):
    key = original.region_key(knowledge, retained)
    amplitude = prior['amplitude'] if prior is not None else 1.
    raw = amplitude
    new_geometry = key is not None and key != initial_key
    sd = float(np.sqrt(prior['covariance'][1][1])) if prior is not None else 0.
    discount = method == METHODS[2] and new_geometry
    if discount:
        amplitude = float(np.clip(prior['mean'][1] - PARAMETERS['new_region_sd_penalty'] * sd, 0., 1.))
    conservative_amplitude = amplitude
    conflict = bool(diagnosis.revoked_ids) or diagnosis.conflict_kind != 'none'
    fallback = method in METHODS[1:] and conflict
    if fallback:
        amplitude = 0.
    return amplitude, {'KF_mean_amplitude_clipped': raw, 'amplitude_SD': sd,
        'geometry_differs_from_initial': new_geometry,
        'geometry_feedback_count': bank['updates'] if bank is not None else 0,
        'conservative_penalty_applied': discount, 'amplitude_before_conflict_rule': conservative_amplitude,
        'B4_conflict': conflict, 'neutral_fallback': fallback,
        'reason': 'B4 conflict: base qLogEI' if fallback else
                  'new-region one-SD amplitude penalty' if discount else 'original G3 deployment'}


def setup(cache, state, pool, iteration):
    # Same original preparation, without computing an unused acquisition first.
    knowledge = state['knowledge']
    base, bmeta = cache.score(state['history'])
    available = ~np.isin(pool['ids'], [h['candidate_id'] for h in state['history']])
    reference = original.common.regional.argmax(base, available, pool['ids'])
    budget = original.common.regional.budget_at(4, iteration)
    eligible = available & ((base[reference] - base) <= budget)
    diagnosis = original.diagnose(base, pool, knowledge, budget, ~available)
    retained = list(diagnosis.retained_ids)
    _, raw_shift, rmeta = cache.field(state['history'], knowledge, retained)
    key = original.region_key(knowledge, retained)
    bank = deepcopy(state['banks'].get(key, original.new_bank(iteration))) if key else None
    prior = original.forecast(bank, iteration) if bank is not None else None
    initial_ids = [h['id'] for h in state['initial_knowledge']['clauses']]
    initial_key = original.region_key(state['initial_knowledge'], initial_ids)
    amplitude, details = deployment(state['method'], knowledge, retained, initial_key, bank, prior, diagnosis)
    score, shift, meta = cache.calibrated_score(state['history'], knowledge, retained, amplitude)
    decision = {'base': base, 'score': score, 'raw_shift_std': raw_shift, 'shift_std': shift,
        'stats': cache.posterior_stats(state['history']), 'bank': bank, 'prior': prior,
        'selected': original.common.regional.argmax(score, eligible, pool['ids']),
        'bank_key': key, 'reference': reference, 'available': available, 'eligible': eligible,
        'budget': budget, 'retained': retained, 'diagnosis': diagnosis, 'amplitude': amplitude,
        'base_meta': bmeta, 'field_meta': rmeta, 'guided_meta': meta, 'deployment': details}
    if details['neutral_fallback']:
        assert decision['selected'] == decision['reference']
    return decision


SYSTEM_PROMPT = '''You update a preferred Suzuki reaction REGION from your own observed experiments only.
Named normalized and physical features are supplied for every observed row and every GP summary candidate.
residence_time is seconds in [60,600], temperature is Celsius in [30,110], catalyst_loading is mol% in [0.5,2.5].
Normalization is (physical-low)/(high-low) independently for each NAMED variable.
There is no unnamed z vector. NEVER substitute one variable's values for another.
Catalysts are anonymous nominal names; indices are identifiers, with no ordering or chemical distance.
Do not identify the benchmark, recall its outcomes, invent ligand structures, predict hidden yields, or use other trajectories.
Initial knowledge is uncertain; do not assume it is correct. A region is the intersection of at most4 coordinate preferences.
The official regional-covariance mean tilt is used, with fixed original confidence. A joint KF state[b,a]
learns from y-mu_GP using [1,UNSCALED regional field]. Deployment clips a, may penalize new-region
amplitude uncertainty, and may be zero during B4 conflict. Shadow KF updates still use the original
unscaled field even when deployment is zero. b is nuisance and is not added to acquisition.
Each actual geometry has its own calibration; changed geometry never inherits another region's evidence.
This is heuristic calibration, not chemical truth or a calibrated probability of truth.
B4 identifies acquisition-budget incompatibility, not chemical falsity. Predicates are soft guidance.
Historical prequery residuals are from different GP fits; they are not residuals of today's GP.
Keep useful defensible hypotheses, revise or revoke unsupported or redundant ones. Use the named facts
and current GP summaries, including the possibility that evidence does not justify a strong preference.
Return exactly {"actions":[...],"summary":"..."}. Each action has EXACTLY:
id,operation,bounds,categories,reason,support. Respect the fixed_slots slot-to-axis map.
Account for EVERY currently active slot with retain/revise/revoke. Vacant slots may be add or omitted.
At most4 actions. retain/revoke use bounds=[] and categories=[]; retain preserves exact existing geometry.
add/revise numeric bounds are normalized [lo,hi] in[0,1], <=4 decimals, width0.1..0.9, categories=[].
For catalysts use bounds=[] and1..7 unique indices0..7 as a nominal set. Do not edit confidence.
reason is a concise rationale <=1200 chars, summary <=1600 chars; neither is a detailed reasoning trace.
support is a NONEMPTY list of typed observed facts. Every fact has EXACTLY:
{"index":<observed 1-based index>,"axis":<the fixed slot's named axis>,
 "normalized_value":<copy that named observed normalized value>,
 "physical_value":<copy that named observed physical value>,"yield_percent":<copy observed yield>}
For axis catalyst use normalized_value=null and physical_value=<exact catalyst name>.
Copy the DISPLAYED values exactly; do not recompute from rounded values. No duplicate indices within an action.
Every support fact is checked against the observed table before any update. Incorrect facts trigger schema repair.
Only cite the axis assigned to this action. Do not claim a temperature contrast when only residence_time differs.
Factual validation does not establish causality; avoid causal certainty from confounded observations.
You may revoke all active knowledge if there is no defensible remaining preference.
No experiment occurs during repair. Output only the JSON object.'''
