"""Shared-foundation regressions that protect the scientific comparisons."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

import numpy as np

from chem.bo.llm import GPRModel, LLMDrivenBO, LLMModel
from chem.decision.hypothesis import (FeatureEncoder, HypothesisManager,
                                     LLMResponsePending, ParameterSpec, canonical_hash)
from chem.decision.replay import (InteractiveAssistantProvider, ReplayOracle,
                                 SynthesisTask, load_cealge_observations)


class GPFoundationTests(unittest.TestCase):
    def model(self, **kwargs):
        return GPRModel({'x': (0, 1), 'y': (0, 1)}, normalize_y=False,
                        n_restarts_optimizer=0, random_state=7, **kwargs)

    def test_domain_is_copied_and_missing_or_extra_coordinates_rejected(self):
        source = {'x': (0, 1)}
        gp = GPRModel(source, n_restarts_optimizer=0)
        source['x'] = (0, 2)
        self.assertEqual(gp.param_space['x'], (0, 1))
        with self.assertRaises(TypeError):
            gp.param_space['new'] = (0, 1)
        for invalid in ({}, {'x': .2, 'z': .3}, {'x': 1.1}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                gp.fit([invalid, invalid], [0, 1])

    def test_batch_prediction_matches_individual_predictions(self):
        gp = self.model()
        rows = [{'x': 0, 'y': .1}, {'x': 1, 'y': .9}, {'x': .4, 'y': .5}]
        gp.fit(rows, [0, 1, .3])
        mu, sigma = gp.predict_many(rows)
        for i, row in enumerate(rows):
            np.testing.assert_allclose(gp.predict(row), [mu[i], sigma[i]], rtol=1e-9, atol=1e-9)

    def test_noise_uses_objective_units_and_does_not_duplicate_observations(self):
        gp = self.model()
        gp.fit([{'x': 0, 'y': .1}, {'x': 1, 'y': .9}], [0, 1], y_sigma=[.1, .2])
        np.testing.assert_allclose(gp._gpr.alpha, np.square([.1, .2]) + 1e-6)
        gp.add_observation({'x': .4, 'y': .5}, .3, y_sigma=.15)
        self.assertEqual(len(gp._X_buf), 3)
        self.assertEqual(len(gp._gpr.X_train_), 3)

    def test_first_gpr_transition_includes_current_observation_exactly_once(self):
        factory = lambda space: GPRModel(space, normalize_y=False, n_restarts_optimizer=0)
        with patch('chem.bo.llm.GPRModel', side_effect=factory):
            bo = LLMDrivenBO('numerical regression', full_param_space={'x': (0, 1)},
                              max_llm_calls=0, gpr_threshold=2)
            bo.add_observation(0, x=0)
            bo.add_observation(1, x=1)
            self.assertEqual(len(bo._gpr_model._X_buf), 2)
            bo.add_observation(.3, x=.4)
            self.assertEqual(len(bo._gpr_model._X_buf), 3)

    def test_refresh_budget_fixed_domain_and_full_vector_survive_activity_changes(self):
        calls = []
        def reasoner(description, observations):
            calls.append(len(observations))
            if len(observations) == 2:
                raise RuntimeError('injected provider failure')
            return LLMModel(lambda x: x['x'], lambda x: .1,
                            {'x': (.2, .8)}, ['x'], 'provisional restriction')
        bo = LLMDrivenBO('test', reasoner, full_param_space={'x': (0, 1), 'y': (0, 1)},
                          refresh_observation_counts=(0, 1, 2, 3, 4), max_llm_calls=4)
        for n in range(4):
            bo.add_observation(.1 * n, x=.3, y=.5)
        self.assertEqual(calls, [0, 1, 2, 3])
        self.assertEqual(bo._param_space['x'], (0, 1))
        self.assertEqual(bo.llm_events[2]['status'], 'failed_retaining_previous')
        self.assertEqual(set(bo.suggest().params), {'x', 'y'})
        self.assertEqual(len(bo.llm_events), 4)

    def test_unit_interval_continuous_coordinates_are_not_rounded_to_binary(self):
        model = LLMModel(lambda x: -(x['fraction'] - .4) ** 2, lambda x: .1,
                         {'fraction': (0, 1)}, ['fraction'])
        bo = LLMDrivenBO('test', lambda *_: model)
        bo.add_observation(0, fraction=.2)
        suggestion = bo.suggest('UCB')
        self.assertGreater(suggestion.params['fraction'], 0)
        self.assertLess(suggestion.params['fraction'], 1)


class HypothesisLedgerTests(unittest.TestCase):
    def setUp(self):
        self.schema = {'x': ParameterSpec('x', 'fraction', (0, 1))}
        self.candidates = [{'x': 0}, {'x': .5}, {'x': 1}]
        self.context = {'observations': [{'candidate_id': 'a', 'value': .1}]}

    def response(self, version, **changes):
        row = {'id': 'H1', 'kind': 'interval', 'parameter': 'x', 'unit': 'fraction',
               'lower': 0, 'upper': .7, 'evidence_ids': ['source1'], 'observation_ids': ['a'],
               'rationale': 'provisional test fixture'}
        row.update(changes)
        return {'batch': {'version': version, 'hypotheses': [row]}, 'candidate_ids': ['b', 'c'],
                'provider': {'kind': 'test_fixture'}}

    def manager(self):
        return HypothesisManager(self.schema, self.candidates, ['a', 'b', 'c'], ['source1'])

    def test_invalid_revision_consumes_budget_and_keeps_last_valid_version(self):
        manager = self.manager()
        original = self.response(1)
        manager.refresh(0, self.context, lambda _: original)
        matrix = manager.membership
        original['batch']['hypotheses'][0]['upper'] = .1
        manager.refresh(4, self.context, lambda _: self.response(2, observation_ids=['future']))
        self.assertEqual(manager.attempts, 2)
        self.assertIs(manager.membership, matrix)
        self.assertEqual(manager.events[0]['raw_response']['batch']['hypotheses'][0]['upper'], .7)
        self.assertEqual(manager.events[1]['status'], 'failed_retaining_previous')
        self.assertFalse(manager.refresh(4, self.context, lambda _: self.response(3)))

    def test_pending_request_does_not_invent_a_model_response(self):
        manager = self.manager()
        def pending(_):
            raise LLMResponsePending('request.json')
        with self.assertRaises(LLMResponsePending):
            manager.refresh(0, self.context, pending)
        self.assertEqual(manager.attempts, 0)
        self.assertEqual(manager.hypotheses, ())

    def test_restore_retains_versions_queue_and_call_budget(self):
        manager = self.manager()
        manager.refresh(0, self.context, lambda _: self.response(1))
        self.assertEqual(manager.pop_suggestion({'a'}), 'b')
        restored = self.manager()
        restored.restore(manager.state_dict(), observed_ids={'a'})
        self.assertEqual(restored.attempts, 1)
        self.assertEqual(restored.pop_suggestion({'a'}), 'c')
        np.testing.assert_array_equal(manager.membership, restored.membership)

    def test_fixed_one_hot_encoding_retains_absent_categories(self):
        encoder = FeatureEncoder({'x': self.schema['x'], 'c': ParameterSpec('c', 'category', levels=('A', 'B', 'C'))})
        encoded = encoder.encode([{'x': .3, 'c': 'A'}])[0]
        self.assertEqual(encoded, {'x': .3, 'c::A': 1, 'c::B': 0, 'c::C': 0})

    def test_interactive_provider_requires_matching_context_and_model(self):
        metadata = {'provider': 'interactive_codex_assistant', 'model_id': 'unit_test_fixture',
                    'conversation_id': 'not_a_real_llm_run'}
        scratch = Path(__file__).resolve().parents[1] / 'data/runs/chapter5/test_scratch'
        scratch.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=scratch) as temporary:
            provider = InteractiveAssistantProvider(temporary, metadata)
            with self.assertRaises(LLMResponsePending) as pending:
                provider(self.context)
            request = Path(pending.exception.request_path)
            response = self.response(1)
            response['provider'] = {'kind': 'interactive_codex_assistant', 'model_id': 'unit_test_fixture',
                                    'conversation_id': 'not_a_real_llm_run'}
            path = request.with_name('response.json')
            path.write_text(json.dumps({'context_sha256': canonical_hash(self.context), 'response': response}), encoding='utf-8')
            self.assertEqual(provider(self.context), response)
            wrong = {'context_sha256': 'wrong', 'response': response}
            path.write_text(json.dumps(wrong), encoding='utf-8')
            with self.assertRaises(ValueError):
                provider(self.context)


class DataAndAuditFoundationTests(unittest.TestCase):
    def test_successful_execution_does_not_imply_chemical_approval(self):
        from chem.loop import parse_audit_result
        self.assertFalse(parse_audit_result({'status': 'SUCCESS'})['passed'])
        self.assertFalse(parse_audit_result({'status': 'APPROVED'})['passed'])
        self.assertFalse(parse_audit_result({'status': 'SUCCESS', 'review_status': 'APPROVED',
                                            'audit_flags': ['REDOX_RISK_CRITICAL'],
                                            'stage_results': [{'redox_status': 'FAIL'}]})['passed'])
        result = parse_audit_result({'status': 'SUCCESS', 'review_status': 'APPROVED',
                                     'critical_failures': [{'flag': 'REDOX_RISK_CRITICAL'}]})
        self.assertFalse(result['passed'])
        self.assertEqual(result['review_status'], 'REJECTED')
        self.assertTrue(parse_audit_result({'status': 'APPROVED', 'checks': [{'status': 'PASS'}]})['passed'])

    def test_unknown_quantification_is_not_a_numeric_zero_or_fallback(self):
        from chem.xrd import AutoXRDYieldEstimator
        from chem.xrd.gsas2.dynamic import DynamicGSAS2RietveldEstimator
        from chem.xrd.gsas2.batch import result_to_row
        est = DynamicGSAS2RietveldEstimator('ceal2ge2')
        bad = est._report_to_yield_result({'phase_fractions_wt': {'CeAl2Ge2': 8}, 'rwp': 30, 'gof': 6})
        self.assertIsNone(bad.yield_pct)
        self.assertFalse(bad.phase_peaks_used['use_for_bo'])
        self.assertIn('unknown', str(bad))
        good = est._report_to_yield_result({'phase_fractions_wt': {'CeAl2Ge2': 0}, 'rwp': 5, 'gof': 1})
        self.assertEqual(good.yield_pct, 0)
        self.assertTrue(good.phase_peaks_used['use_for_bo'])
        class Primary:
            def estimate(self, *_, **__):
                return bad
        class ForbiddenFallback:
            def estimate(self, *_, **__):
                raise AssertionError('must not fabricate a replacement yield')
        self.assertIs(AutoXRDYieldEstimator('ceal2ge2', Primary(), ForbiddenFallback()).estimate('unused'), bad)
        self.assertFalse(result_to_row('CeAlGe_test', Path('unused'), bad, 0)['bo_training_eligible'])

    def test_cli_and_legacy_benchmark_share_the_same_quality_filtered_pool(self):
        from chem import paths
        from chem.cli import _observations_from_dataset
        from chem.bo.rietveld import _load_pool
        source = paths.ce_al_ge_dataset('arcmelt_dataset_rietveld.json')
        rows = load_cealge_observations(source)
        self.assertEqual(len(rows), 12)
        self.assertFalse(any('DSC' in row['candidate_id'] for row in rows))
        self.assertEqual(_observations_from_dataset(str(source)), rows)
        x, y = _load_pool()
        np.testing.assert_array_equal(y, [row['yield'] for row in rows])
        np.testing.assert_allclose(x, [list(row['params'].values()) for row in rows])

    def test_restoring_replay_marks_old_queries_without_repeating_them(self):
        original = ReplayOracle(['a', 'b'], [.2, .8])
        original.query('a')
        resumed = ReplayOracle(['a', 'b'], [.2, .8])
        resumed.restore(original.ledger)
        with self.assertRaises(ValueError):
            resumed.query('a')
        self.assertEqual(resumed.query('b'), .8)
        self.assertEqual(len(resumed.ledger), 2)

    def test_task_pool_is_immutable_and_roundtrips_without_objective_values(self):
        rows = [{'x': 0}, {'x': 1}]
        task = SynthesisTask('unit_test', 'not_applicable_math', 'numerical_fixture',
                             (ParameterSpec('x', 'fraction', (0, 1)),), ('a', 'b'), tuple(rows),
                             ({'id': 'source1', 'text': 'test fixture'},))
        fingerprint = task.fingerprint
        rows[0]['x'] = .9
        self.assertEqual(task.fingerprint, fingerprint)
        with self.assertRaises(TypeError):
            task.candidates[0]['x'] = .9
        restored = SynthesisTask.from_dict(json.loads(json.dumps(task.to_dict())))
        self.assertEqual(restored.fingerprint, fingerprint)
        self.assertNotIn('values', task.to_dict())


if __name__ == '__main__':
    unittest.main()
