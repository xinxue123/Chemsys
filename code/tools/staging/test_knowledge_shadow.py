"""Offline synthetic tests only; no LLM, file, GP fitting, or oracle access."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import unittest

import numpy as np

import knowledge_shadow as s


AXES = ['temperature', 'catalyst']


def fixture(t=1, selected='new1', version=1, created=1, previous=None):
    prefix = [{'candidate_id': f'init{i}', 'y': 0.} for i in range(4)]
    for i in range(1, t):
        prefix.append({'candidate_id': f'new{i}', 'y': 1.})
    h = dict(id='H1', version=version, semantic_key=f'semantic-{version}',
             created_round=created, status='active', plus_id='plus', minus_id='minus',
             plus_features=[.8, 'P1-L7'], minus_features=[.2, 'P1-L7'],
             scope={'axes': {'temperature': {'kind': 'continuous', 'bounds': [0., 1.]},
                             'catalyst': {'kind': 'categorical', 'allowed': ['P1-L7']}}},
             excluded_observed_ids=[])
    pred = dict(hypothesis_id='H1', version=version, point_id=selected, query_round=t,
                semantic_key=h['semantic_key'], gp_mean_y=0., gp_var_y=1.,
                contrast_mean=0., contrast_var=1., cov_y_contrast=.5,
                preference_noise_sd=.1)
    pred['prediction_sha256'] = s.digest(pred)
    c = dict(query=t, selected_id=selected, history_sha256=s.digest(prefix),
             GP_fit_key=f'fit-{t}', hypotheses_at_commit=[h],
             shadows=[dict(id='H1', version=version, prediction=pred, informative=True)])
    return c, prefix


def setup(t=1, **kwargs):
    c, prefix = fixture(t=t, **kwargs)
    field = s.field_from_commit(c, 'H1', kwargs.get('version', 1), AXES)
    state = s.initial_state(field, 1.)
    return c, prefix, field, state


class ShadowTests(unittest.TestCase):
    def test_field_anchors_scope_and_category_equality(self):
        _, _, field, _ = setup()
        self.assertAlmostEqual(s.field_value(field, field['plus']), 1.)
        self.assertAlmostEqual(s.field_value(field, field['minus']), -1.)
        self.assertEqual(s.field_value(field, {'temperature': .8, 'catalyst': 'other'}), 0.)
        with self.assertRaises(ValueError):
            s.field_value(field, {'temperature': .8, 'catalyst': 7})

    def test_categorical_labels_have_no_ordinal_distance(self):
        c, _ = fixture()
        h = c['hypotheses_at_commit'][0]
        h['minus_features'] = [.8, 'P1-L2']
        h['scope']['axes']['catalyst']['allowed'] = ['P1-L7', 'P1-L2', 'third']
        f = s.field_from_commit(c, 'H1', 1, AXES)
        self.assertEqual(s.field_value(f, dict(temperature=.8, catalyst='P1-L7')), 1.)
        self.assertEqual(s.field_value(f, dict(temperature=.8, catalyst='P1-L2')), -1.)
        self.assertEqual(s.field_value(f, dict(temperature=.8, catalyst='third')), 0.)
        h['plus_features'][1] = 'ZZZ'
        h['minus_features'][1] = 'AAA'
        h['scope']['axes']['catalyst']['allowed'] = ['ZZZ', 'AAA', 'third']
        renamed = s.field_from_commit(c, 'H1', 1, AXES)
        self.assertEqual(s.field_value(renamed, dict(temperature=.8, catalyst='ZZZ')), 1.)

    def test_only_current_live_version_is_allowed(self):
        c, _ = fixture()
        with self.assertRaises(ValueError):
            s.field_from_commit(c, 'future', 1, AXES)
        c['hypotheses_at_commit'][0]['created_round'] = 2
        with self.assertRaises(ValueError):
            s.field_from_commit(c, 'H1', 1, AXES)
        c['hypotheses_at_commit'][0]['created_round'] = 1
        c['hypotheses_at_commit'][0]['status'] = 'retracted'
        with self.assertRaises(ValueError):
            s.field_from_commit(c, 'H1', 1, AXES)

    def test_score_is_before_update_and_distinct_from_variance_score(self):
        c, prefix, field, state = setup()
        snap = s.prepare_prediction(c, field, state, field['plus'], prefix)
        next_state, receipt = s.score_and_update(state, snap,
            dict(candidate_id='new1', query_round=1, y=3.))
        metrics = receipt['metrics']['raw_field']
        self.assertEqual(metrics['same_variance_mean_logscore_gain'], 0.)
        self.assertGreater(metrics['predictive_logscore_gain'], 0.)
        self.assertEqual(metrics['mean_squared_error_gain'], 0.)
        self.assertEqual(snap['arms']['raw_field']['mean'], 0.)
        self.assertEqual(state['calibrators']['raw_field']['mean'], 0.)
        self.assertAlmostEqual(next_state['calibrators']['raw_field']['mean'], 1.5)

    def test_new_point_gets_previous_calibration_not_its_own_feedback(self):
        c, prefix, field, state = setup()
        snap = s.prepare_prediction(c, field, state, field['plus'], prefix)
        state2, _ = s.score_and_update(state, snap, dict(candidate_id='new1', query_round=1, y=1.))
        c2, prefix2 = fixture(t=2, selected='new2')
        snap2 = s.prepare_prediction(c2, field, state2, field['plus'], prefix2)
        self.assertAlmostEqual(snap2['arms']['raw_field']['mean'], .5)
        _, receipt2 = s.score_and_update(state2, snap2, dict(candidate_id='new2', query_round=2, y=1.))
        self.assertAlmostEqual(receipt2['metrics']['raw_field']['same_variance_mean_logscore_gain'], .375)

    def test_duplicate_outcome_and_mutated_snapshot_are_rejected(self):
        c, prefix, field, state = setup()
        snap = s.prepare_prediction(c, field, state, field['plus'], prefix)
        outcome = dict(candidate_id='new1', query_round=1, y=1.)
        new, _ = s.score_and_update(state, snap, outcome)
        with self.assertRaises(ValueError):
            s.score_and_update(new, snap, outcome)
        bad = deepcopy(snap)
        bad['arms']['raw_field']['mean'] = 1.
        with self.assertRaises(ValueError):
            s.score_and_update(state, bad, outcome)

    def test_wrong_outcome_and_prefix_are_rejected(self):
        c, prefix, field, state = setup()
        with self.assertRaises(ValueError):
            s.prepare_prediction(c, field, state, field['plus'], prefix + [dict(candidate_id='future', y=1.)])
        snap = s.prepare_prediction(c, field, state, field['plus'], prefix)
        with self.assertRaises(ValueError):
            s.score_and_update(state, snap, dict(candidate_id='other', query_round=1, y=1.))

    def test_past_calibration_outcomes_must_match_next_commit_prefix(self):
        c, prefix, field, state = setup()
        snap = s.prepare_prediction(c, field, state, field['plus'], prefix)
        wrong, _ = s.score_and_update(state, snap, dict(candidate_id='new1', query_round=1, y=99.))
        c2, prefix2 = fixture(t=2, selected='new2')
        with self.assertRaises(ValueError):
            s.prepare_prediction(c2, field, wrong, field['plus'], prefix2)

    def test_cannot_skip_initial_evidence_or_expand_frozen_scope(self):
        _, _, field, state = setup()
        c2, prefix2 = fixture(t=2, selected='new2')
        with self.assertRaises(ValueError):
            s.prepare_prediction(c2, field, state, field['plus'], prefix2)
        c, prefix = fixture()
        c['hypotheses_at_commit'][0]['scope']['axes']['catalyst']['allowed'].append('new-label')
        with self.assertRaises(ValueError):
            s.prepare_prediction(c, field, state, field['plus'], prefix)

    def test_cannot_skip_a_later_round_after_calibration_started(self):
        c, prefix, field, state = setup()
        snap = s.prepare_prediction(c, field, state, field['plus'], prefix)
        state2, _ = s.score_and_update(state, snap, dict(candidate_id='new1', query_round=1, y=1.))
        c3, prefix3 = fixture(t=3, selected='new3')
        with self.assertRaises(ValueError):
            s.prepare_prediction(c3, field, state2, field['plus'], prefix3)

    def test_pair_version_cannot_be_substituted_under_same_semantics(self):
        c, prefix, field, state = setup()
        p = c['shadows'][0]['prediction']
        p['version'] = 99
        p['prediction_sha256'] = s.digest({k: v for k, v in p.items() if k != 'prediction_sha256'})
        with self.assertRaises(ValueError):
            s.prepare_prediction(c, field, state, field['plus'], prefix)

    def test_new_version_does_not_inherit_positive_calibration(self):
        c, prefix, field, state = setup()
        snap = s.prepare_prediction(c, field, state, field['plus'], prefix)
        old, _ = s.score_and_update(state, snap, dict(candidate_id='new1', query_round=1, y=1.))
        c2, prefix2, field2, state2 = setup(t=2, selected='new2', version=2, created=2)
        with self.assertRaises(ValueError):
            s.prepare_prediction(c2, field2, old, field2['plus'], prefix2)
        snap2 = s.prepare_prediction(c2, field2, state2, field2['plus'], prefix2)
        self.assertEqual(snap2['arms']['raw_field']['mean'], 0.)

    def test_conditioned_field_cancels_fully_absorbed_basis(self):
        _, _, field, _ = setup()
        points = [field['plus'], field['minus']]
        value = s.gp_conditioned_value(field, field['plus'], points, np.eye(2), [1., 0.])
        self.assertAlmostEqual(value, 0.)
        far = s.gp_conditioned_value(field, field['plus'], points, np.eye(2), [0., 0.])
        self.assertAlmostEqual(far, 1.)

    def test_conditioning_preserves_original_scope_even_when_projection_is_nonzero(self):
        _, _, field, _ = setup()
        outside = dict(temperature=.8, catalyst='outside')
        value = s.gp_conditioned_value(field, outside, [field['plus']], [[1.]], [.5])
        self.assertEqual(value, 0.)
        with self.assertRaises(ValueError):
            s.gp_conditioned_value(field, field['plus'], [field['plus']], [[-1.]], [.5])

    def test_missing_conditioning_is_unavailable_not_fabricated(self):
        c, prefix, field, state = setup()
        snap = s.prepare_prediction(c, field, state, field['plus'], prefix)
        self.assertFalse(snap['arms']['gp_conditioned_field']['available'])
        new, receipt = s.score_and_update(state, snap, dict(candidate_id='new1', query_round=1, y=1.))
        self.assertFalse(receipt['metrics']['gp_conditioned_field']['available'])
        self.assertEqual(new['calibrators']['gp_conditioned_field'], state['calibrators']['gp_conditioned_field'])

    def test_conditioning_blocks_must_match_prefix_and_fit(self):
        c, prefix, field, state = setup()
        blocks = dict(history_sha256=c['history_sha256'], gp_fit_key='wrong',
                      training_ids=[r['candidate_id'] for r in prefix],
                      training_points=[field['minus']] * 4, k_noise=np.eye(4), k_x=[0.] * 4)
        with self.assertRaises(ValueError):
            s.prepare_prediction(c, field, state, field['plus'], prefix, blocks)
        blocks['gp_fit_key'] = c['GP_fit_key']
        snap = s.prepare_prediction(c, field, state, field['plus'], prefix, blocks)
        self.assertEqual(snap['arms']['gp_conditioned_field']['basis'], 1.)

    def test_pair_scores_sign_and_scope_are_not_normal_approximations(self):
        c, prefix, field, state = setup()
        snap = s.prepare_prediction(c, field, state, field['plus'], prefix)
        _, positive = s.score_and_update(state, snap, dict(candidate_id='new1', query_round=1, y=1.))
        _, negative = s.score_and_update(state, snap, dict(candidate_id='new1', query_round=1, y=-1.))
        self.assertGreater(positive['metrics']['pair']['predictive_logscore_gain'], 0.)
        self.assertLess(negative['metrics']['pair']['predictive_logscore_gain'], 0.)
        self.assertNotEqual(positive['metrics']['pair']['predictive_logscore_gain'],
                            positive['metrics']['pair']['same_variance_mean_logscore_gain'])
        snap_out = s.prepare_prediction(c, field, state, dict(temperature=.8, catalyst='outside'), prefix)
        _, out = s.score_and_update(state, snap_out, dict(candidate_id='new1', query_round=1, y=1.))
        self.assertEqual(out['metrics']['pair']['predictive_logscore_gain'], 0.)
        self.assertEqual(out['metrics']['raw_field']['predictive_logscore_gain'], 0.)

    def test_pure_functions_do_not_read_files_or_mutate_inputs(self):
        c, prefix, field, state = setup()
        before = deepcopy((c, prefix, field, state))
        with patch('builtins.open', side_effect=AssertionError('No file reads')):
            with patch.object(Path, 'read_text', side_effect=AssertionError('No file reads')):
                snap = s.prepare_prediction(c, field, state, field['plus'], prefix)
                _, receipt = s.score_and_update(state, snap, dict(candidate_id='new1', query_round=1, y=1.))
        self.assertEqual((c, prefix, field, state), before)
        self.assertEqual(receipt['assessment'], s.LABEL)
        self.assertNotIn('SearchAUC', receipt)


if __name__ == '__main__':
    unittest.main()
