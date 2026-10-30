"""R7-S mechanism and safety checks, independent of private GP/oracle files."""

from __future__ import annotations

from contextlib import nullcontext
from hashlib import sha256
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from chem.benchmarks import continuous_r6_core as r6core
from chem.benchmarks import continuous_r6_engine as r6
from chem.benchmarks import continuous_r7_core as core
from chem.benchmarks import continuous_r7_engine as r7


def fixture(arm='B2', positive=False):
    class Cache:
        pool = {
            'task': 'SnAr', 'axes': ['x', 'z'],
            'ids': ['bo', 'near', 'far', 'seen'],
            'z': [[.1, .3], [.25, .3], [.9, .3], [.1, .9]],
            'points': [{'x': .1, 'z': .3}, {'x': .25, 'z': .3},
                       {'x': .9, 'z': .3}, {'x': .1, 'z': .9}],
            'description': 'fake reactor', 'objective_name': 'STY',
            'objective_unit': 'unit', 'bounds': {'x': [0, 1], 'z': [0, 1]},
        }
        ix = {'bo': 0, 'near': 1, 'far': 2, 'seen': 3}

    candidates = [r6core.Candidate('bo', (.1, .3), 0., .1, 1., .1),
                  r6core.Candidate('near', (.25, .3), -.2, .1, 1., .2),
                  r6core.Candidate('far', (.9, .3), -.3, .1, 1., .3)]
    d = {'base': np.array([0., -.2, -.3, -math.inf]),
         'eligible': np.array([True, True, True, False]),
         'available': np.array([True, True, True, False]),
         'budget': 1., 'reference': 0, 'base_meta': {'paths': 512},
         'stats': {'mu_y': np.zeros(4), 'sd_y': np.ones(4),
                   'sd_predictive_y': np.ones(4), 'fit_key': 'fake', 'fit_seed': 0}}
    context = {'d': d, 'gate': r6core.Gate('bo', frozenset(('bo', 'near', 'far')), 0., 1.),
               'candidates': candidates,
               'menu': [r6core.MenuPoint('bo', (.1, .3)),
                        r6core.MenuPoint('near', (.25, .3))]}
    state = {'arm': arm, 'task': 'SnAr', 'seed': 0,
             'history': [{'candidate_id': 'seen', 'y': 0., 'raw_value': 0.}],
             'hypotheses': [], 'calls': [], 'validation_queries_used': 0,
             'exceptions_used': 0, 'decisions': [], 'status': 'READY'}
    if positive:
        ledger = r6core.EvidenceLedger('H01', 1)
        ledger.cumulative_log_score_gain = 2.
        ledger.seen_point_ids.add('previous')
        state['hypotheses'] = [{'id': 'H01', 'created_round': 1,
                                'better_id': 'near', 'worse_id': 'bo',
                                'directions': {'x': 'increase'},
                                'tau_y': .06, 'scope_radius': .25,
                                'ledger': r6._ledger_to_json(ledger)}]
    proposal = {'action': 'compare', 'better_id': 'near', 'worse_id': 'bo',
                'focus_directions': {'x': 'increase'}, 'reason': 'possible physical ordering'}
    return Cache(), context, state, proposal


def moments(_cache, _stats, _h, indices):
    return 0., .4, np.full(len(indices), .2)


class BinaryInformationTests(unittest.TestCase):
    def test_zero_covariance_returns_zero(self):
        self.assertEqual(core.squared_correlation(1., 1., 0., .2), 0.)
        self.assertEqual(core.sign_information_nats(0., 1., 0., 1., 0., .2), 0.)

    def test_binary_information_is_nonnegative_and_bounded(self):
        result = core.sign_information_nats(0., 1., 0., 1., .8, .2)
        self.assertGreater(result, .01)
        self.assertLess(result, math.log(2))

    def test_binary_information_decreases_for_already_certain_sign(self):
        unsure = core.sign_information_nats(0., 1., 0., 1., .7, .2)
        nearly_certain = core.sign_information_nats(0., 1., 8., 1., .7, .2)
        self.assertLess(nearly_certain, unsure / 100)

    def test_correlation_rejects_invalid_joint_covariance(self):
        with self.assertRaises(ValueError):
            core.squared_correlation(1., .1, 2., .1)

    def test_sign_flip_is_semantically_consistent_and_abstain_safe(self):
        cache, context, _, original = fixture()
        flipped = core.flip_directions(original)
        self.assertEqual(flipped['better_id'], original['worse_id'])
        self.assertEqual(flipped['focus_directions']['x'], 'decrease')
        comparison = r6core.Comparison(flipped['better_id'], flipped['worse_id'], {0: -1})
        self.assertTrue(r6core.review_comparison(comparison, context['menu'],
                                                 ['continuous', 'continuous']).passed)
        self.assertEqual(core.flip_directions({'action': 'abstain'}), {'action': 'abstain'})


class PassiveCovarianceEngineTests(unittest.TestCase):
    def test_base_arm_preserves_frozen_bo(self):
        cache, context, state, _ = fixture('B0')
        chosen = r7.plan(cache, state, 1, context)
        self.assertEqual(chosen['selected_id'], 'bo')
        self.assertEqual(chosen['selection_mode'], 'base_bo')
        self.assertEqual(chosen['prequery_shadows'], [])
        self.assertEqual(chosen['GP_reference_logEI'], 0.)

    def test_unverified_hypothesis_cannot_change_first_decision(self):
        cache, context, state, proposal = fixture()
        with patch.object(r6, 'contrast_moments', side_effect=moments):
            chosen = r7.plan(cache, state, 1, context, proposal)
        self.assertEqual(chosen['selected_id'], 'bo')
        self.assertEqual(chosen['selection_mode'], 'base_bo')
        self.assertEqual(len(state['hypotheses']), 1)
        self.assertEqual(chosen['hypotheses_at_commit'][0]['ledger']['score'], 0.)
        self.assertTrue(chosen['prequery_shadows'][0]['informative'])

    def test_passive_shadows_are_precommitted_and_reversible(self):
        cache, context, state, proposal = fixture()
        with patch.object(r6, 'contrast_moments', side_effect=moments):
            chosen = r7.plan(cache, state, 1, context, proposal)
        self.assertEqual(len(state['history']), 1)
        # Outcome is first supplied after the immutable commitment.
        receipts = r7.apply_feedback(state, chosen,
                                     {'candidate_id': 'bo', 'y': -1., 'raw_value': -10.})
        self.assertEqual(len(state['history']), 2)
        self.assertLess(receipts[0]['delta_log_score'], 0.)
        self.assertEqual(receipts[0]['weight_after'], 0.)

    def test_covariance_can_guide_outside_r6_geometric_box(self):
        cache, context, state, _ = fixture(positive=True)
        self.assertEqual(len(r6._scope_indices(cache, state['hypotheses'][0], [2])), 0)
        with patch.object(r6, 'contrast_moments', side_effect=moments), \
             patch.object(r6, 'guided_logei', return_value={'bo': 0., 'near': -.2, 'far': .9}):
            chosen = r7.plan(cache, state, 2, context)
        self.assertEqual(chosen['selection_mode'], 'knowledge_guided')
        self.assertEqual(chosen['selected_id'], 'far')
        self.assertTrue(chosen['normal_gate_eligible'])
        self.assertFalse(chosen['used_exception'])
        self.assertEqual(chosen['reachability_at_commit']['geometric_eligible_count'], 2)

    def test_zero_covariance_recovers_exact_original_rank(self):
        cache, context, state, _ = fixture(positive=True)
        with patch.object(r6, 'contrast_moments', side_effect=lambda c, s, h, js:
                          (0., .4, np.zeros(len(js)))), \
             patch.object(r6, 'guided_logei', wraps=r6.guided_logei) as real:
            chosen = r7.plan(cache, state, 2, context)
        self.assertEqual(chosen['selected_id'], 'bo')
        self.assertTrue(real.called)
        self.assertFalse(chosen['prequery_shadows'][0]['informative'])

    def test_no_active_verification_even_with_evidence(self):
        cache, context, state, _ = fixture(positive=True)
        with patch.object(r6, 'contrast_moments', side_effect=moments), \
             patch.object(r6, 'guided_logei', return_value={'bo': 0., 'near': -.2, 'far': .8}):
            chosen = r7.plan(cache, state, 2, context)
        self.assertFalse(chosen['used_exception'])
        self.assertEqual(chosen['budgets_at_commit']['active_verifications'], 0)
        self.assertEqual(state['validation_queries_used'], 0)

    def test_diagnostics_disentangle_scope_and_rank_flip(self):
        cache, context, state, _ = fixture(positive=True)
        with patch.object(r6, 'contrast_moments', side_effect=moments), \
             patch.object(r6, 'guided_logei', return_value={'bo': 0., 'near': -.2, 'far': .9}):
            report = r7.diagnose_hypothesis(cache, context, state['history'], state['hypotheses'][0])
        self.assertLess(report['geometric_eligible'], report['eligible_count'])
        self.assertEqual(report['covariance_relevant_eligible'], 3)
        self.assertTrue(report['actual_flip'])
        self.assertTrue(report['cap_flip'])

    def test_rejects_out_of_menu_llm_after_first_round(self):
        cache, context, state, proposal = fixture()
        context['menu'] = ()
        with self.assertRaises(ValueError):
            r7.plan(cache, state, 3, context, proposal)


class R7CommitOrderingTests(unittest.TestCase):
    def test_commit_before_reveal_and_crash_resume(self):
        from chem.benchmarks import continuous_r7_study as study

        cache, context, state, _ = fixture('B0')
        state['protocol_sha256'] = 'unit_protocol'
        cache.release = lambda: None
        context['menu'] = ()

        class Data:
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

            @staticmethod
            def reveal(task, point_id):
                self.assertTrue(commit_file.exists())
                return {'candidate_id': point_id, 'y': .5, 'raw_value': 5.}

        with TemporaryDirectory() as directory:
            state_file = Path(directory) / 'task' / 'state.json'
            commit_file = state_file.parent / 'rounds' / '01_committed.json'
            initial = json.loads(json.dumps(state))
            with patch.object(study, '_runtime', return_value=(Data(), None, None, None)), \
                 patch.object(r7, 'build_context', return_value=context):
                record = study.step(state, state_file, cache, None, {'protocol_sha256': 'unit_protocol'})
                self.assertEqual(record['selected'], 'bo')
                self.assertTrue(commit_file.exists())
                with patch.object(r7, 'build_context', side_effect=AssertionError('replanned')):
                    study.step(initial, state_file, cache, None, {'protocol_sha256': 'unit_protocol'})
                self.assertEqual(initial['history'], state['history'])


if __name__ == '__main__':
    unittest.main()
