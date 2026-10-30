from copy import deepcopy
import numpy as np
from chem.benchmarks import r2_tuning_policy_v3_shuffle as policy
from chem.benchmarks import r2_tuning_shuffle_engine as ties
from chem.decision.domain_guard import select_candidate


def example(monkeypatch):
    monkeypatch.setattr(policy.base, 'settings', lambda s: {'high_yield_threshold_percent': 95., 'counterexample_shortlist_size': 5})
    pool = {'axes': ['x', 'y'], 'levels': {'x': ['a', 'b'], 'y': [str(i) for i in range(10)]},
            'points': [{'x': a, 'y': str(i)} for a in ['a', 'b'] for i in range(10)],
            'ids': [f'p{i:02d}' for i in range(20)]}
    state = {'history': [{'candidate_id': 'p00', 'y': .6}, {'candidate_id': 'p11', 'y': .1}],
             'knowledge': {'confidence': .7, 'clauses': [{'id': 'H1', 'axis': 'x', 'allowed': ['a']}, {'id': 'H2', 'axis': 'y', 'allowed': ['0']}]},
             'validations': [], 'decisions': [{}, {}, {}]}
    d = {'base': np.zeros(20), 'reference': 0, 'eligible': np.ones(20, dtype=bool),
         'stats': {'mu_y': np.ones(20)*.3, 'sd_y': np.ones(20)*.2}}
    return pool, state, d


def test_random_order_is_reproducible_round_specific_and_not_ID_sorted():
    score = np.zeros(60)
    got = ties.ordered(score, np.ones(60, dtype=bool), 4)
    np.testing.assert_array_equal(got, ties.ordered(score, np.ones(60, dtype=bool), 4))
    assert not np.array_equal(got, np.arange(60))
    assert not np.array_equal(got, ties.ordered(score, np.ones(60, dtype=bool), 6))


def test_distinct_float_scores_never_cross_even_near_ties():
    scores = np.array([2., 2., 2.+1e-14, -1., 3., 2.])
    order = ties.ordered(scores, np.ones(6, dtype=bool), 4)
    assert order[0] == 4 and order[1] == 2 and order[-1] == 3
    assert np.all(np.diff(scores[order]) <= 0)


def test_local_rng_does_not_consume_GP_or_global_numpy_state():
    before = np.random.get_state()
    ties.priority(3955, 4)
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])


def test_competition_ranks_share_ties_and_exclude_observed():
    scores = np.array([3., 3., 2., 2., 1.])
    np.testing.assert_array_equal(ties.competition_ranks(scores, np.ones(5, dtype=bool)), [1, 1, 3, 3, 5])
    mask = np.array([False, False, True, True, True])
    np.testing.assert_array_equal(ties.competition_ranks(scores, mask)[mask], [1, 1, 3])


def test_renaming_ID_strings_preserves_physical_shortlist(monkeypatch):
    pool, state, d = example(monkeypatch)
    state['validations'] = [{'changed_axis': 'x'}]
    first = policy.build_candidates(pool, state, d)
    renamed = deepcopy(pool)
    renamed['ids'] = [f'z{100-i:03d}' for i in range(20)]
    state2 = deepcopy(state)
    state2['history'][0]['candidate_id'] = renamed['ids'][0]
    state2['history'][1]['candidate_id'] = renamed['ids'][11]
    second = policy.build_candidates(renamed, state2, d)
    assert [c['point'] for c in first['candidates']] == [c['point'] for c in second['candidates']]
    assert len(first['candidates']) == 5
    assert len({c['base_rank'] for c in first['candidates']}) == 1
    assert [c['candidate_id'] for c in first['candidates']] != ['p02', 'p03', 'p04', 'p05', 'p06']


def test_first_test_retains_full_pool_and_high_yield_gate_still_skips(monkeypatch):
    pool, state, d = example(monkeypatch)
    old = policy.base.build_candidates(pool, state, d)
    got = policy.build_candidates(pool, state, d)
    assert {c['candidate_id'] for c in got['candidates']} == {c['candidate_id'] for c in old['candidates']}
    assert len(got['candidates']) > 5
    state['history'][0]['y'] = .95
    assert not policy.build_candidates(pool, state, d)['candidates']


def test_fallback_reference_and_choice_share_priority_B4_subsets_unchanged(monkeypatch):
    pool, state, d = example(monkeypatch)
    available = np.ones(20, dtype=bool)
    membership = np.ones((2, 20), dtype=bool)
    dg = select_candidate(d['base'], membership, 0., candidate_ids=pool['ids'], hypothesis_ids=['H1', 'H2'])
    d.update(score=d['base'].copy(), available=available, selected=0, diagnosis=dg, fallback=True)
    before = deepcopy(d)
    monkeypatch.setattr(ties.fixed, 'setup', lambda *args: deepcopy(before))
    monkeypatch.setattr(ties.original.nominal, 'membership', lambda *args: membership)
    cache = type('Cache', (), {'pool': pool})()
    got = ties.setup(cache, state, 4)
    assert got['selected'] == got['reference']
    assert got['diagnosis'].retained_ids == dg.retained_ids
    assert got['diagnosis'].subset_checks == dg.subset_checks
    assert got['diagnosis'].candidate_index == got['selected']
    np.testing.assert_array_equal(got['base'], before['base'])
    np.testing.assert_array_equal(got['score'], before['score'])
    np.testing.assert_array_equal(got['eligible'], before['eligible'])
