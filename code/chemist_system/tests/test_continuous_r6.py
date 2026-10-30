"""Run with: python3 -m unittest discover -s r6_decision -v"""

import math
import unittest
from unittest.mock import patch
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from hashlib import sha256
from contextlib import nullcontext
from types import SimpleNamespace
import sys

import numpy as np

from chem.benchmarks.continuous_r6_core import (
    Candidate,
    Comparison,
    ContrastProbe,
    EvidenceLedger,
    ShadowPrediction,
    VerificationPolicy,
    choose_active_verification,
    choose_guided_inside_gate,
    choose_llm_rerank,
    conditional_preference_moments,
    feasible_llm_menu,
    independent_llm_menu,
    information_about_contrast,
    prospective_log_score_gain,
    qlogei_gate,
    review_comparison,
)


def candidate(point_id, x, score, priority=0.0):
    return Candidate(point_id, tuple(x), score, 0.0, 0.0, priority)


class GateAndMenuTests(unittest.TestCase):
    def setUp(self):
        self.pool = [
            candidate("bo", (0.1, "A"), 0.0, 0.1),
            candidate("near", (0.9, "B"), -0.2, 0.2),
            candidate("also", (0.4, "C"), -0.5, 0.3),
            candidate("outside", (0.6, "D"), -3.0, 0.4),
        ]
        self.kinds = ("continuous", "categorical")

    def test_gate_tightens_to_exact_top_by_round16(self):
        first = qlogei_gate(self.pool, 1)
        self.assertAlmostEqual(first.epsilon, math.log(4))
        self.assertEqual(first.eligible_point_ids, {"bo", "near", "also"})
        last = qlogei_gate(self.pool, 16)
        self.assertEqual(last.epsilon, 0.0)
        self.assertEqual(last.eligible_point_ids, {"bo"})

    def test_all_minus_infinity_has_a_defined_fallback(self):
        pool = [candidate("a", (0.2,), -math.inf, 0.1), candidate("b", (0.8,), -math.inf, 0.9)]
        result = qlogei_gate(pool, 16)
        self.assertEqual(result.bo_point_id, "b")
        self.assertEqual(result.eligible_point_ids, {"a", "b"})

    def test_menu_only_contains_eligible_points_and_bo(self):
        gate = qlogei_gate(self.pool, 5)
        menu = feasible_llm_menu(self.pool, gate, self.kinds, max_items=2, seed=8)
        ids = {p.point_id for p in menu}
        self.assertIn("bo", ids)
        self.assertNotIn("outside", ids)
        self.assertEqual(len(ids), 2)
        self.assertEqual(set(menu[0].__dataclass_fields__), {"point_id", "features"})

    def test_blind_menu_does_not_depend_on_gp_scores(self):
        menu_1 = independent_llm_menu(self.pool, self.kinds, max_items=3, seed=4)
        changed = [candidate(c.point_id, c.features, -1e6 + i, c.random_tie)
                   for i, c in enumerate(self.pool)]
        menu_2 = independent_llm_menu(changed, self.kinds, max_items=3, seed=4)
        self.assertEqual(menu_1, menu_2)

    def test_comparison_direction_and_background_are_enforced(self):
        menu = feasible_llm_menu(self.pool, qlogei_gate(self.pool, 5), self.kinds, seed=4)
        valid = Comparison("near", "bo", {0: +1, 1: None})
        self.assertTrue(review_comparison(valid, menu, self.kinds).passed)
        reversed_direction = Comparison("near", "bo", {0: -1, 1: None})
        self.assertFalse(review_comparison(reversed_direction, menu, self.kinds).passed)
        changed_background = Comparison("near", "bo", {0: +1})
        self.assertFalse(review_comparison(changed_background, menu, self.kinds).passed)

    def test_continuous_background_more_than_025_rejected(self):
        pool = [candidate("a", (0.1, 0.1), 0), candidate("b", (0.2, 0.9), -.1)]
        menu = feasible_llm_menu(pool, qlogei_gate(pool, 1), ("continuous", "continuous"))
        self.assertFalse(review_comparison(Comparison("b", "a", {0: +1}), menu,
                                           ("continuous", "continuous")).passed)

    def test_feasible_llm_reranking_is_distinct_from_evidence_guidance(self):
        gate = qlogei_gate(self.pool, 5)
        menu = feasible_llm_menu(self.pool, gate, self.kinds, seed=4)
        review = review_comparison(Comparison("near", "bo", {0: +1, 1: None}), menu, self.kinds)
        choice = choose_llm_rerank(self.pool, gate, menu, ("near", "bo"), review)
        self.assertEqual(choice.point_id, "near")
        self.assertEqual(choice.mode, "direct_LLM_rerank")

    def test_blind_llm_rerank_outside_gate_is_rejected(self):
        gate = qlogei_gate(self.pool, 5)
        menu = independent_llm_menu(self.pool, self.kinds, max_items=4, seed=7)
        review = review_comparison(Comparison("outside", "bo", {0: +1, 1: None}), menu, self.kinds)
        self.assertTrue(review.passed)
        choice = choose_llm_rerank(self.pool, gate, menu, ("outside", "bo"), review)
        self.assertEqual(choice.point_id, "bo")
        self.assertEqual(choice.mode, "GP_rejected_blind_proposal")


class R6RepositoryAdapterTests(unittest.TestCase):
    """Exercise the real integration logic with deterministic fake GP moments.

    No BoTorch, LLM API, hidden oracle, or historical data are required.
    """

    @staticmethod
    def fixture(arm='A2'):
        from chem.benchmarks import continuous_r6_core as core

        class Cache:
            pool = {
                'task': 'SnAr', 'axes': ['x', 'z'],
                'ids': ['a', 'b', 'c', 'seen'],
                'z': [[0.2, 0.3], [0.5, 0.3], [0.8, 0.3], [0.1, 0.9]],
                'points': [{'x': .2, 'z': .3}, {'x': .5, 'z': .3},
                           {'x': .8, 'z': .3}, {'x': .1, 'z': .9}],
                'description': 'fake reactor', 'objective_name': 'sty', 'objective_unit': 'unit',
                'bounds': {'x': [0, 1], 'z': [0, 1]},
            }
            ix = {'a': 0, 'b': 1, 'c': 2, 'seen': 3}

        cache = Cache()
        candidates = [core.Candidate('a', (.2, .3), 0., .1, 1., .1),
                      core.Candidate('b', (.5, .3), -.3, .1, 1., .2),
                      core.Candidate('c', (.8, .3), -3., .1, 1., .3)]
        d = {'base': np.array([0., -.3, -3.]), 'eligible': np.array([True, True, False]),
             'available': np.array([True, True, True]), 'budget': 1.0,
             'reference': 0, 'base_meta': {'paths': 512},
             'stats': {'mu_y': np.zeros(3), 'sd_y': np.ones(3),
                       'sd_predictive_y': np.ones(3), 'fit_key': 'test_gp', 'fit_seed': 2}}
        gate = core.Gate('a', frozenset(('a', 'b')), 0., 1.)
        menu = [core.MenuPoint('a', (.2, .3)), core.MenuPoint('b', (.5, .3))]
        context = {'d': d, 'gate': gate, 'candidates': candidates, 'menu': menu}
        state = {'arm': arm, 'task': 'SnAr', 'seed': 0, 'history': [
            {'candidate_id': 'seen', 'y': 0., 'raw_value': 0.}],
            'hypotheses': [], 'calls': [], 'validation_queries_used': 0,
            'exceptions_used': 0, 'decisions': []}
        proposal = {'action': 'compare', 'better_id': 'b', 'worse_id': 'a',
                    'focus_directions': {'x': 'increase'}, 'reason': 'hypothesis'}
        return cache, context, state, proposal

    def test_gp_filtered_public_payload_hides_gp_scores(self):
        from chem.benchmarks import continuous_r6_engine as e
        cache, ctx, state, _ = self.fixture()
        public = e.public_menu(cache, state, 1, ctx)
        self.assertEqual(len(public['candidate_menu']), 2)
        self.assertIn('GP-filtered', public['method'])
        for forbidden in ('qLogEI', 'GP_mean', 'GP_sd', 'GP_reference', 'expected_improvement'):
            self.assertNotIn(forbidden, str(public))

    def test_filtered_menu_repairs_missing_nearby_pairs(self):
        from chem.benchmarks import continuous_r6_engine as e
        from chem.benchmarks import continuous_r6_core as core
        candidates = [core.Candidate(str(i), (float(i) / 30., .1), -.1, 0., 1., 0.)
                      for i in range(30)]
        # Deliberately present a small diverse subset without any close pair.
        preliminary = [core.MenuPoint('0', candidates[0].features),
                       core.MenuPoint('29', candidates[29].features)]
        repaired = e._paired_menu(candidates, preliminary, 5, 123, required_id='0')
        self.assertIn('0', {p.point_id for p in repaired})
        self.assertTrue(any(e._meaningful_pair(a.features, b.features)
                            for i, a in enumerate(repaired) for b in repaired[i + 1:]))
        self.assertLessEqual(len(repaired), 5)

    def test_menu_abstains_if_no_locally_comparable_pair(self):
        from chem.benchmarks import continuous_r6_engine as e
        from chem.benchmarks import continuous_r6_core as core
        pool = [core.Candidate('a', (0., 0., 0., 0.), 0., 0., 1., 0.),
                core.Candidate('b', (1., 1., 1., 1.), -.1, 0., 1., 0.)]
        menu = tuple(core.MenuPoint(p.point_id, p.features) for p in pool)
        self.assertEqual(e._paired_menu(pool, menu, 4, 1), ())

    def test_direct_rerank_is_executable_and_forward_scored(self):
        from chem.benchmarks import continuous_r6_engine as e
        cache, ctx, state, proposal = self.fixture()
        with patch.object(e, 'contrast_moments', return_value=(0., .4, np.array([.2]))):
            commitment = e.plan(cache, state, 1, ctx, proposal)
        self.assertEqual(commitment['selected_id'], 'b')
        self.assertEqual(commitment['selection_mode'], 'direct_LLM_rerank')
        self.assertEqual(commitment['GP_fit_key'], 'test_gp')
        self.assertEqual(len(commitment['prequery_shadows']), 1)
        self.assertEqual(len(state['history']), 1)  # No outcome read in plan.
        self.assertEqual(len(state['hypotheses']), 1)
        receipts = e.apply_feedback(state, commitment, {'candidate_id': 'b', 'y': 2., 'raw_value': 2.})
        self.assertEqual(len(state['history']), 2)
        self.assertTrue(receipts[0]['informative'])
        self.assertGreater(receipts[0]['weight_after'], 0.)
        self.assertEqual(commitment['hypotheses_at_commit'][0]['ledger']['score'], 0.)

    def test_invalid_reversed_direction_abstains_without_learning(self):
        from chem.benchmarks import continuous_r6_engine as e
        cache, ctx, state, proposal = self.fixture()
        proposal['focus_directions'] = {'x': 'decrease'}
        commitment = e.plan(cache, state, 1, ctx, proposal)
        self.assertEqual(commitment['selection_mode'], 'base_bo')
        self.assertFalse(commitment['review']['accepted'])
        self.assertEqual(commitment['prequery_shadows'], [])
        self.assertEqual(state['hypotheses'], [])

    def test_A4_does_not_directly_deploy_unverified_llm(self):
        from chem.benchmarks import continuous_r6_engine as e
        cache, ctx, state, proposal = self.fixture('A4')
        state['validation_queries_used'] = 2  # Active-validation budget used.
        with patch.object(e, 'contrast_moments', side_effect=lambda c, s, h, js: (0., .4, np.full(len(js), .2))):
            commitment = e.plan(cache, state, 1, ctx, proposal)
        self.assertEqual(commitment['selected_id'], 'a')
        self.assertEqual(commitment['selection_mode'], 'base_bo')
        self.assertEqual(commitment['review']['accepted'], True)

    def test_preference_EI_zero_covariance_reverts_to_original(self):
        from chem.benchmarks import continuous_r6_engine as e
        cache, ctx, state, proposal = self.fixture()
        h = {'tau_y': .06}
        scores = e.guided_logei(cache, ctx, h, [0, 1], (0., .4, np.zeros(2)),
                                [{'y': -1.}])
        self.assertAlmostEqual(scores['a'], 0., places=9)
        self.assertAlmostEqual(scores['b'], -.3, places=9)

    def test_contrast_bridge_uses_joint_covariance_and_outcome_scale(self):
        from chem.benchmarks import continuous_r6_engine as e

        class T:
            def __init__(self, x):
                self.x = np.asarray(x, dtype=float)

            def __getitem__(self, index):
                return T(self.x[index])

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.x

        def cov(z):
            sq = (z[:, None, :] - z[None, :, :]) ** 2
            return np.exp(-sq.sum(-1) / .3)

        class Model:
            def posterior(self, points):
                covariance = cov(points.x)
                return SimpleNamespace(mvn=SimpleNamespace(covariance_matrix=T(covariance)))

        fake_torch = SimpleNamespace(no_grad=nullcontext,
                                     cat=lambda parts, dim=0: T(np.concatenate([p.x for p in parts], axis=dim)))
        cache, _, _, _ = self.fixture()
        cache.X = T(cache.pool['z'])
        stats = {'model': Model(), 'std': 2., 'mu_y': np.array([.3, .1, .4, 0.])}
        h = {'better_id': 'b', 'worse_id': 'a'}
        with patch.dict(sys.modules, {'torch': fake_torch}):
            md, vd, cross = e.contrast_moments(cache, stats, h, [0, 1, 2], chunk=2)
        k = cov(np.asarray(cache.pool['z']))
        self.assertAlmostEqual(md, -.2)
        self.assertAlmostEqual(vd, 4. * (2. - 2. * k[1, 0]))
        np.testing.assert_allclose(cross, 4. * (k[[0, 1, 2], 1] - k[[0, 1, 2], 0]))

    def test_llm_schema_is_strict_about_real_menu(self):
        from chem.benchmarks import continuous_r6_llm as llm
        cache, ctx, _, proposal = self.fixture()
        menu = [{'candidate_id': p.point_id} for p in ctx['menu']]
        self.assertEqual(llm.validate(proposal, menu), proposal)
        invalid = dict(proposal, better_id='out-of-pool')
        with self.assertRaises(ValueError):
            llm.validate(invalid, menu)

    def test_feedback_rejects_uncommitted_or_wrong_outcome(self):
        from chem.benchmarks import continuous_r6_engine as e
        cache, ctx, state, _ = self.fixture()
        commit = e.plan(cache, state, 1, ctx)
        with self.assertRaises(ValueError):
            e.apply_feedback(state, commit, {'candidate_id': 'b', 'y': 0., 'raw_value': 0.})
        self.assertEqual(len(state['history']), 1)


class R6CommitOrderTests(unittest.TestCase):
    def test_durable_prequery_commit_then_oracle_and_resume(self):
        from chem.benchmarks import continuous_r6_study as study
        from chem.benchmarks import continuous_r6_engine as engine

        cache, ctx, state, _ = R6RepositoryAdapterTests.fixture('A0')
        state.update(task='SnAr', protocol_sha256='test-protocol', status='READY')
        ctx['menu'] = ()  # The A0 arm never requests an LLM call.

        class Data:
            oracle_reveals = 0

            @staticmethod
            def write(path, value):
                path = Path(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value), encoding='utf-8')

            @staticmethod
            def read(path):
                return json.loads(Path(path).read_text(encoding='utf-8'))

            @staticmethod
            def filehash(path):
                return sha256(Path(path).read_bytes()).hexdigest()

            def reveal(self, task, cid):
                assert task == 'SnAr' and cid == 'a'
                assert committed_path.exists(), 'Oracle must not be read before commit!'
                self.oracle_reveals += 1
                return {'candidate_id': cid, 'y': .8, 'raw_value': 8.}

        cache.release = lambda: None
        data = Data()
        with TemporaryDirectory() as temp:
            state_file = Path(temp) / 'run' / 'state.json'
            committed_path = state_file.parent / 'rounds' / '01_committed.json'
            original = json.loads(json.dumps(state))
            with patch.object(study, '_runtime', return_value=(data, None, None, None)), \
                 patch.object(engine, 'build_context', return_value=ctx):
                result = study.step(state, state_file, cache, None, {'protocol_sha256': 'test-protocol'})
                self.assertEqual(result['selected'], 'a')
                self.assertEqual(data.oracle_reveals, 1)
                self.assertEqual(len(state['history']), 2)

                # A process that died after the commit must not compute a new
                # choice or call the LLM when resuming that same round.
                with patch.object(engine, 'build_context', side_effect=AssertionError('replanned')):
                    replay = study.step(original, state_file, cache, None, {'protocol_sha256': 'test-protocol'})
                self.assertEqual(replay['selected'], 'a')
                self.assertEqual(original['history'], state['history'])
                self.assertEqual(data.oracle_reveals, 2)


class PreferenceMathTests(unittest.TestCase):
    def prediction(self, point="x", cov=0.5, round_index=2):
        return ShadowPrediction("h1", point, round_index, 0.0, 1.0, 0.0, 1.0, cov, 0.5)

    def test_independent_point_has_zero_increment(self):
        for value in (-100.0, 0.0, 100.0):
            self.assertAlmostEqual(prospective_log_score_gain(self.prediction(cov=0), value), 0)

    def test_directional_prospective_gain(self):
        self.assertGreater(prospective_log_score_gain(self.prediction(), 2), 0)
        self.assertLess(prospective_log_score_gain(self.prediction(), -2), 0)

    def test_moments_match_half_normal_formula(self):
        mean, var = conditional_preference_moments(0.0, 1.0, 0.5, 0.0, 1.0, 0.5)
        expected_shift = 0.5 / math.sqrt(1.25) * math.sqrt(2 / math.pi)
        expected_var = 1.0 - 0.25 / 1.25 * (2 / math.pi)
        self.assertAlmostEqual(mean, expected_shift)
        self.assertAlmostEqual(var, expected_var)

    def test_negative_prior_contrast_is_numerically_stable(self):
        mean, var = conditional_preference_moments(0, 1, 0.5, -35, 1.0, 0.5)
        self.assertTrue(math.isfinite(mean))
        self.assertGreaterEqual(var, 0)

    def test_information_must_increase_with_correlation(self):
        lo = information_about_contrast(ContrastProbe("x", 1.0, 1.0, 0.1, 0.5))
        hi = information_about_contrast(ContrastProbe("x", 1.0, 1.0, 0.7, 0.5))
        self.assertGreater(hi, lo)
        with self.assertRaises(ValueError):
            information_about_contrast(ContrastProbe("x", 1, 1, 3, 0.5))

    def test_evidence_is_prospective_distinct_and_reversible(self):
        evidence = EvidenceLedger("h1", creation_round=2)
        self.assertEqual(evidence.deployment_weight, 0.0)
        evidence.observe(self.prediction("p1"), 2.0)
        self.assertGreater(evidence.deployment_weight, 0.0)
        with self.assertRaises(ValueError):
            evidence.observe(self.prediction("p1"), 2.0)
        evidence.observe(self.prediction("p2", round_index=3), -10.0)
        self.assertEqual(evidence.deployment_weight, 0.0)
        with self.assertRaises(ValueError):
            evidence.observe(self.prediction("old", round_index=1), 3.0)


class QuerySelectionTests(unittest.TestCase):
    def setUp(self):
        self.pool = [
            candidate("bo", (0.2,), 0.0, .1),
            candidate("test_in", (0.4,), -0.2, .2),
            candidate("test_out", (0.6,), -1.7, .3),
            candidate("far", (0.9,), -20, .4),
        ]
        self.probes = {
            "bo": ContrastProbe("bo", 1.0, 1.0, .05, .5),
            "test_in": ContrastProbe("test_in", 1.0, 1.0, .7, .5),
            "test_out": ContrastProbe("test_out", 1.0, 1.0, .9, .5),
            "far": ContrastProbe("far", 1.0, 1.0, .99, .5),
        }

    def test_early_test_in_original_gate(self):
        gate = qlogei_gate(self.pool, 5)
        result = choose_active_verification(self.pool, gate, self.probes, 5, 0, 0,
                                            VerificationPolicy(max_extra_qlogei_gap=0))
        self.assertEqual(result.point_id, "test_in")
        self.assertFalse(result.used_exception)

    def test_exception_is_explicit_and_capped(self):
        gate = qlogei_gate(self.pool, 5)
        # At round 5, logEI gap 1.7 > epsilon; allow an extra 1.0 margin.
        policy = VerificationPolicy(max_extra_qlogei_gap=1.0)
        result = choose_active_verification(self.pool, gate, self.probes, 5, 0, 0, policy)
        self.assertEqual(result.point_id, "test_out")
        self.assertTrue(result.used_exception)
        blocked = choose_active_verification(self.pool, gate, self.probes, 5, 0, 1, policy)
        self.assertEqual(blocked.point_id, "test_in")
        self.assertFalse(blocked.used_exception)

    def test_verification_is_not_issued_late_or_after_quota(self):
        gate = qlogei_gate(self.pool, 11)
        self.assertIsNone(choose_active_verification(self.pool, gate, self.probes, 11, 0, 0))
        gate = qlogei_gate(self.pool, 5)
        self.assertIsNone(choose_active_verification(self.pool, gate, self.probes, 5, 2, 0))

    def test_guidance_changes_rank_only_inside_original_gate(self):
        gate = qlogei_gate(self.pool, 5)
        model_log_ei = {"bo": -5.0, "test_in": 1.0, "test_out": 100.0}
        base = choose_guided_inside_gate(self.pool, gate, model_log_ei, 0)
        self.assertEqual(base.point_id, "bo")
        choice = choose_guided_inside_gate(self.pool, gate, model_log_ei, .5)
        self.assertEqual(choice.point_id, "test_in")
        self.assertEqual(choice.mode, "knowledge_guided")
        self.assertLessEqual(choice.acquisition_gap, gate.epsilon)

    def test_missing_guided_score_rejected(self):
        with self.assertRaises(ValueError):
            choose_guided_inside_gate(self.pool, qlogei_gate(self.pool, 5), {}, .2)


if __name__ == "__main__":
    unittest.main()
