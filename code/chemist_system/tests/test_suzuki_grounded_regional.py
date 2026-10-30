"""Behavioral checks for factual grounding, neutral fallback, and deployment."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from chem.benchmarks import grounded_regional_suzuki_engine as e
from chem.benchmarks import grounded_regional_suzuki_study as s

torch.set_num_threads(1)


@pytest.fixture
def pool():
    return {'ids': [f'p{i:05d}' for i in range(8)],
        'z': [[.1 + i * .1, .25 + i * .04, .8 - i * .08, (i + .5) / 8] for i in range(8)],
        'points': [{'catalyst': catalyst} for catalyst in e.CATALYSTS]}


@pytest.fixture
def knowledge():
    return {'confidence': .7, 'clauses': [
        {'id': 'H1', 'coordinate': 0, 'bounds': [.2, .8], 'categories': []},
        {'id': 'H2', 'coordinate': 3, 'bounds': [], 'categories': [0, 3]}]}


def observations(pool):
    return [{'index': i + 1, **e.named_features(pool, i), 'yield_percent': 10. + i}
            for i in range(4)]


def retain_plan(pool, knowledge):
    obs = observations(pool)
    return {'actions': [{'id': h['id'], 'operation': 'retain', 'bounds': [], 'categories': [],
            'reason': 'Keep an uncertain preference supported by the cited named observation.',
            'support': [e.evidence_fact(obs[0], e.VARIABLES[h['coordinate']])]}
            for h in knowledge['clauses']], 'summary': 'Retain as an uncertain hypothesis.'}


def test_named_values_map_time_temperature_loading_without_an_ordinal_catalyst(pool):
    feature = e.named_features(pool, 0)
    assert feature['normalized'] == dict(residence_time=.1, temperature=.25, catalyst_loading=.8)
    assert feature['physical'] == dict(residence_time=114., temperature=50., catalyst_loading=2.1)
    assert feature['catalyst'] == 'P1-L1' and 'z' not in feature


def test_checked_retention_preserves_exact_geometry_confidence_and_facts(pool, knowledge):
    plan = retain_plan(pool, knowledge)
    result = e.validate_refresh(plan, knowledge, {'H1': 0, 'H2': 3}, observations(pool), pool)
    assert result['knowledge'] == knowledge and len(result['checked_facts']) == 2
    assert result['actions'] == plan['actions']


@pytest.mark.parametrize('field,value', [('axis', 'temperature'), ('normalized_value', .25),
    ('physical_value', 50.), ('yield_percent', 99.), ('index', 5), ('index', True),
    ('normalized_value', float('nan')), ('normalized_value', True)])
def test_wrong_axis_future_or_incorrect_numbers_are_rejected(pool, knowledge, field, value):
    plan = retain_plan(pool, knowledge)
    plan['actions'][0]['support'][0][field] = value
    with pytest.raises(ValueError):
        e.validate_refresh(plan, knowledge, {'H1': 0, 'H2': 3}, observations(pool), pool)


def test_nominal_support_rejects_numeric_substitution(pool, knowledge):
    plan = retain_plan(pool, knowledge)
    plan['actions'][1]['support'][0]['physical_value'] = 0
    with pytest.raises(ValueError, match='nominal'):
        e.validate_refresh(plan, knowledge, {'H1': 0, 'H2': 3}, observations(pool), pool)


def test_duplicate_support_and_confidence_injection_rejected(pool, knowledge):
    plan = retain_plan(pool, knowledge)
    plan['actions'][0]['support'] *= 2
    with pytest.raises(ValueError, match='duplicate'):
        e.validate_refresh(plan, knowledge, {'H1': 0, 'H2': 3}, observations(pool), pool)
    plan = retain_plan(pool, knowledge)
    plan['confidence'] = .99
    with pytest.raises(ValueError):
        e.validate_refresh(plan, knowledge, {'H1': 0, 'H2': 3}, observations(pool), pool)


def test_revise_keeps_locked_axis_and_can_revoke_other_knowledge(pool, knowledge):
    plan = retain_plan(pool, knowledge)
    plan['actions'][0].update(operation='revise', bounds=[.3, .9])
    plan['actions'][1]['operation'] = 'revoke'
    checked = e.validate_refresh(plan, knowledge, {'H1': 0, 'H2': 3}, observations(pool), pool)
    assert checked['knowledge']['clauses'] == [{'id': 'H1', 'coordinate': 0, 'bounds': [.3, .9], 'categories': []}]


def test_R1_is_original_deployment_and_R2_conflict_is_exactly_neutral(knowledge):
    ids = ['H1', 'H2']
    key = e.original.region_key(knowledge, ids)
    bank = e.original.new_bank(1)
    prior = e.original.forecast(bank, 1)
    conflict = SimpleNamespace(revoked_ids=('H2',), conflict_kind='quality_exclusion')
    assert e.deployment(e.METHODS[0], knowledge, ids, key, bank, prior, conflict)[0] == 1.
    alpha, detail = e.deployment(e.METHODS[1], knowledge, ids, key, bank, prior, conflict)
    assert alpha == 0. and detail['neutral_fallback']
    assert bank['mean'] == [0., 1.]  # no destruction of the shadow bank


def test_R3_initial_region_exempt_new_geometry_penalized_and_bank_is_not_transferred(knowledge):
    ids = ['H1', 'H2']
    initial = e.original.region_key(knowledge, ids)
    clear = SimpleNamespace(revoked_ids=(), conflict_kind='none')
    bank = e.original.new_bank(4)
    prior = e.original.forecast(bank, 4)
    alpha, detail = e.deployment(e.METHODS[2], knowledge, ids, initial, bank, prior, clear)
    assert alpha == 1. and not detail['conservative_penalty_applied']
    changed = deepcopy(knowledge)
    changed['clauses'][0]['bounds'] = [.3, .9]
    alpha, detail = e.deployment(e.METHODS[2], changed, ids, initial, bank, prior, clear)
    assert alpha == pytest.approx(1. - np.sqrt(.2525))
    assert detail['conservative_penalty_applied'] and detail['geometry_feedback_count'] == 0


def test_conservative_rule_uses_unclipped_mean_and_forecast_SD_and_conflict_has_priority(knowledge):
    bank = e.original.new_bank(5)
    prior = {'amplitude': 1., 'mean': [0., 1.2], 'covariance': [[.01, 0.], [0., .04]]}
    clear = SimpleNamespace(revoked_ids=(), conflict_kind='none')
    assert e.deployment(e.METHODS[2], knowledge, ['H1'], 'different', bank, prior, clear)[0] == pytest.approx(1.)
    prior['mean'][1] = .1
    assert e.deployment(e.METHODS[2], knowledge, ['H1'], 'different', bank, prior, clear)[0] == 0.
    prior['mean'][1] = 1.2
    conflict = SimpleNamespace(revoked_ids=('H2',), conflict_kind='quality_exclusion')
    assert e.deployment(e.METHODS[2], knowledge, ['H1'], 'different', bank, prior, conflict)[0] == 0.


def test_all_three_decision_mechanisms_share_same_base_GP_and_original_field(pool, knowledge, tmp_path):
    cache = e.GroundedRegionalCache(pool, tmp_path / 'cache', 1)
    history = [{'candidate_id': pool['ids'][i], 'y': y} for i, y in enumerate((.1, .3, .15, .5))]
    state = {'method': e.METHODS[0], 'history': history, 'knowledge': knowledge,
             'initial_knowledge': deepcopy(knowledge), 'banks': {}}
    original = e.original.setup(cache, {**state, 'method': e.original.METHODS[3]}, pool, 1)
    r1 = e.setup(cache, state, pool, 1)
    r2 = e.setup(cache, {**state, 'method': e.METHODS[1]}, pool, 1)
    r3 = e.setup(cache, {**state, 'method': e.METHODS[2]}, pool, 1)
    assert np.array_equal(original['score'], r1['score']) and original['selected'] == r1['selected']
    for d in (r2, r3):
        assert np.array_equal(d['base'], r1['base']) and np.array_equal(d['eligible'], r1['eligible'])
        assert np.array_equal(d['raw_shift_std'], r1['raw_shift_std'])
        if d['deployment']['neutral_fallback']:
            assert d['selected'] == d['reference'] and np.array_equal(d['score'], d['base'])
        else:
            assert d['amplitude'] == 1.


def test_LLM_public_payload_contains_named_observed_features_and_no_bare_z(pool, knowledge, tmp_path):
    history = [{'candidate_id': pool['ids'][i], 'y': y} for i, y in enumerate((.1, .3, .15, .5))]
    state = {'method': e.METHODS[0], 'history': history, 'knowledge': knowledge,
             'initial_knowledge': deepcopy(knowledge), 'banks': {}, 'decisions': [], 'calls': [],
             'slots': {'H1': 0, 'H2': 3}, 'task': 'blind'}
    cache = e.GroundedRegionalCache(pool, tmp_path / 'cache', 1)
    d = e.setup(cache, state, pool, 1)
    public = s.public_request(state, pool, d, ['stagnation'])
    for row in public['observations'] + public['GP_top3_per_catalyst']:
        assert 'z' not in row and set(row['normalized']) == set(e.VARIABLES[:3])
    assert 'case' not in public and 'regime' not in public and 'method' not in public
    assert 'yield_percent' not in public['GP_top3_per_catalyst'][0]


def test_cached_refresh_rejects_wrong_context_without_API_or_outcome_access(pool, tmp_path):
    folder = tmp_path / 'request'
    s.write(folder / 'validated.json', {'context_sha256': 'other'})
    with pytest.raises(ValueError, match='context mismatch'):
        s.get_refresh(None, {'context_sha256': 'actual'}, {}, pool, folder)


def test_inference_and_initial_budget_settings_are_unchanged():
    assert s.PARAMETERS == s.previous.PARAMETERS
    assert e.PARAMETERS['budget_ratio'] == 4 and e.PARAMETERS['new_region_sd_penalty'] == 1.
    assert s.METHODS == ('R1_grounded', 'R2_conflict_fallback', 'R3_conservative_region')
