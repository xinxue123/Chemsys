"""Mechanism tests for versioned mixed-domain prospective knowledge."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np

from chem.benchmarks import knowledge_policy as p


def pool():
    # Deliberately nonordered labels: none of their spellings imply distances.
    rows = [[.8, 'Pd-A', .50], [.4, 'Pd-A', .55], [.6, 'Pd-A', .52],
            [.6, 'Pd-Z', .52], [.6, 'Pd-Q', .52], [.1, 'Pd-A', .52],
            [.7, 'Pd-A', .50], [.5, 'Pd-A', .52], [.65, 'Pd-A', .51]]
    ids = ['plus', 'minus', 'probe', 'cat_minus', 'cat_third', 'outside', 'later', 'other', 'last']
    return SimpleNamespace(ids=ids, ix={key: i for i, key in enumerate(ids)},
                           axes=['temperature', 'catalyst', 'time'],
                           kinds=['continuous', 'categorical', 'continuous'],
                           features=np.asarray(rows, dtype=object), points=[])


def action(op='add', hid=None, plus='plus', minus='minus', bounds=(.3, .9), direction='increase'):
    return {'op': op, 'id': hid, 'plus_id': plus, 'minus_id': minus,
            'focus': {'temperature': {'direction': direction, 'bounds': list(bounds)}},
            'reason': 'A prospective local pairwise preference.'}


def proposal(*actions):
    return {'actions': list(actions), 'summary': 'Update the locally testable claims.'}


def initial():
    data = pool()
    hs, events = p.review_and_apply(data, [], proposal(action()), data.ids, 1)
    assert events[0]['accepted']
    return data, hs


def evidence(h, point, t, y):
    pred = p.shadow(h, point, t, mu=0., predvar=1., contrastmean=0., var=1., cov=.75)
    return p.apply_evidence(h, pred, y, informative=True)


class KnowledgePolicyTests(unittest.TestCase):
    def test_explicit_scope_and_background_category(self):
        data, hs = initial()
        self.assertEqual(p.scope_indices(data, hs[0], np.arange(len(data.ids))).tolist(), [0, 1, 2, 6, 7, 8])
        self.assertEqual(hs[0]['ledger']['score'], 0.)
        self.assertEqual(p.weight(hs[0]), 0.)

    def test_continuous_bounds_required_and_direction_checked(self):
        data = pool()
        bad = [action(direction='decrease'), action(bounds=(.45, .9)), action(bounds=(-.1, .9))]
        shorthand = action()
        shorthand['focus'] = {'temperature': 'increase'}
        bad.append(shorthand)
        for value in bad:
            with self.subTest(value=value['focus']):
                hs, events = p.review_and_apply(data, [], proposal(value), data.ids, 1)
                self.assertEqual(hs, [])
                self.assertFalse(events[0]['accepted'])

    def test_categorical_semantics_and_real_labels(self):
        data = pool()
        value = action(plus='probe', minus='cat_minus')
        value['focus'] = {'catalyst': 'prefer_plus'}
        hs, events = p.review_and_apply(data, [], proposal(value), data.ids, 1)
        self.assertTrue(events[0]['accepted'])
        self.assertIn(3, p.scope_indices(data, hs[0], [2, 3, 4]))
        self.assertNotIn(4, p.scope_indices(data, hs[0], [2, 3, 4]))
        for focus in [{'catalyst': 'increase'},
                      {'catalyst': {'direction': 'prefer_plus', 'allowed': ['Pd-A', 'invented']}},
                      {'temperature': {'direction': 'increase', 'bounds': [0., 1.]}}]:
            invalid = deepcopy(value)
            invalid['focus'] = focus
            hs, events = p.review_and_apply(data, [], proposal(invalid), data.ids, 1)
            self.assertFalse(events[0]['accepted'])

    def test_negative_rephrasing_and_duplicate_add_cannot_reset(self):
        data, hs = initial()
        evidence(hs[0], 'probe', 1, -2.)
        original = deepcopy(hs[0]['ledger'])
        rewritten = action()
        rewritten['reason'] = 'Different words, exactly the same scientific comparison.'
        for value in [rewritten, dict(rewritten, op='revise', id=hs[0]['id'])]:
            new, events = p.review_and_apply(data, hs, proposal(value), data.ids, 2)
            self.assertEqual(len(new), 1)
            self.assertEqual(new[0]['ledger'], original)
            self.assertEqual(events[0]['outcome'], 'duplicate_semantics_preserved')
            self.assertEqual(p.weight(new[0]), 0.)

    def test_revision_starts_without_positive_evidence(self):
        data, hs = initial()
        evidence(hs[0], 'probe', 1, 2.)
        self.assertGreater(p.weight(hs[0]), 0)
        changed = action(op='revise', hid=hs[0]['id'], bounds=(.2, 1.))
        updated, events = p.review_and_apply(data, hs, proposal(changed), data.ids, 2)
        self.assertTrue(events[0]['accepted'])
        old, new = updated
        self.assertEqual(old['status'], 'retracted')
        self.assertEqual(new['version'], 2)
        self.assertEqual(new['parent'], {'id': old['id'], 'version': 1})
        self.assertEqual(new['ledger']['score'], 0.)
        self.assertEqual(new['ledger']['events'], [])
        self.assertEqual(p.weight(new), 0.)
        with self.assertRaises(ValueError):
            evidence(new, 'probe', 2, 2.)  # Previously seen point is not fresh evidence.
        with self.assertRaises(ValueError):
            p.shadow(new, 'later', 1, 0., 1., 0., 1., .75)

    def test_scope_change_or_reversal_cannot_bypass_lineage_using_add(self):
        data, hs = initial()
        evidence(hs[0], 'probe', 1, -2.)
        for changed in [action(bounds=(.2, 1.)),
                        action(plus='minus', minus='plus', direction='decrease')]:
            updated, events = p.review_and_apply(data, hs, proposal(changed), data.ids, 2)
            self.assertFalse(events[0]['accepted'])
            self.assertEqual(updated, hs)

    def test_negative_inheritance_is_conditional_and_unique_across_versions(self):
        data, hs = initial()
        evidence(hs[0], 'probe', 1, -1.)
        changed = action(op='revise', hid=hs[0]['id'], bounds=(.2, 1.))
        hs, _ = p.review_and_apply(data, hs, proposal(changed), data.ids, 2)
        self.assertEqual(len(hs[-1]['inherited_negative_evidence']), 1)
        self.assertTrue(hs[-1]['inherited_negative_evidence'][0]['applicable'])
        receipt = evidence(hs[-1], 'later', 2, .2)
        self.assertLess(receipt['inherited_negative_penalty'], 0.)
        self.assertEqual(receipt['weight_after'], 0.)
        changed = action(op='revise', hid=hs[-1]['id'], bounds=(.1, 1.))
        hs, _ = p.review_and_apply(data, hs, proposal(changed), data.ids, 3)
        self.assertEqual(len(hs[-1]['inherited_negative_evidence']), 1)
        self.assertEqual(hs[-1]['ledger']['score'], 0.)

    def test_opposite_revision_keeps_negative_audit_but_no_flipped_positive(self):
        data, hs = initial()
        evidence(hs[0], 'probe', 1, -2.)
        value = action(op='revise', hid=hs[0]['id'], plus='minus', minus='plus', direction='decrease')
        hs, events = p.review_and_apply(data, hs, proposal(value), data.ids, 2)
        self.assertTrue(events[0]['accepted'])
        self.assertEqual(hs[-1]['version'], 2)
        self.assertEqual(len(hs[-1]['inherited_negative_evidence']), 1)
        self.assertFalse(hs[-1]['inherited_negative_evidence'][0]['applicable'])
        self.assertEqual(p.weight(hs[-1]), 0.)
        self.assertEqual(hs[-1]['ledger']['score'], 0.)

    def test_return_to_old_semantics_requires_new_version_and_keeps_penalty(self):
        data, hs = initial()
        evidence(hs[0], 'probe', 1, -2.)
        opposite = action(op='revise', hid=hs[0]['id'], plus='minus', minus='plus', direction='decrease')
        hs, _ = p.review_and_apply(data, hs, proposal(opposite), data.ids, 2)
        original = action(op='revise', hid=hs[0]['id'])
        hs, events = p.review_and_apply(data, hs, proposal(original), data.ids, 3)
        self.assertTrue(events[0]['accepted'])
        self.assertEqual(hs[-1]['version'], 3)
        self.assertTrue(hs[-1]['inherited_negative_evidence'][0]['applicable'])
        self.assertEqual(len(hs[-1]['inherited_negative_evidence']), 1)

    def test_signed_score_and_retraction_by_negative_feedback(self):
        _, hs = initial()
        h = hs[0]
        first = evidence(h, 'probe', 1, 2.)
        self.assertGreater(first['delta_log_score'], 0.)
        self.assertGreater(first['weight_after'], 0.)
        second = evidence(h, 'later', 2, -5.)
        self.assertLess(second['delta_log_score'], 0.)
        self.assertEqual(second['weight_after'], 0.)
        self.assertLess(h['ledger']['score'], 0.)

    def test_zero_covariance_has_no_increment_and_uninformative_cannot_replay(self):
        _, hs = initial()
        h = hs[0]
        pred = p.shadow(h, 'probe', 1, 0., 1., 0., 1., 0.)
        receipt = p.apply_evidence(h, pred, 9., True)
        self.assertEqual(receipt['delta_log_score'], 0.)
        pred = p.shadow(h, 'later', 2, 0., 1., 0., 1., .75)
        receipt = p.apply_evidence(h, pred, 1., False)
        self.assertIsNone(receipt['delta_log_score'])
        with self.assertRaises(ValueError):
            p.apply_evidence(h, pred, 1., True)

    def test_shadow_tampering_and_retroactive_rewrite_are_rejected(self):
        data, hs = initial()
        pred = p.shadow(hs[0], 'probe', 1, 0., 1., 0., 1., .75)
        pred['gp_mean_y'] = 100.
        with self.assertRaises(ValueError):
            p.apply_evidence(hs[0], pred, 1., True)
        evidence(hs[0], 'probe', 1, -2.)
        updated, events = p.review_and_apply(data, hs, proposal(action(op='revise', hid=hs[0]['id'], bounds=(.2, 1.))), data.ids, 1)
        self.assertFalse(events[0]['accepted'])
        self.assertEqual(updated, hs)

    def test_dormancy_preserves_evidence_and_recovers_without_reset(self):
        data, hs = initial()
        evidence(hs[0], 'probe', 1, 2.)
        before = deepcopy(hs[0]['ledger'])
        hs, events = p.refresh_coverage(data, hs, [5], 2)
        self.assertEqual(hs[0]['status'], 'dormant')
        self.assertEqual(p.weight(hs[0]), 0.)
        self.assertEqual(hs[0]['ledger']['score'], before['score'])
        hs, events = p.refresh_coverage(data, hs, [6], 3)
        self.assertEqual(hs[0]['status'], 'active')
        self.assertEqual(p.weight(hs[0]), before['weight'])
        self.assertEqual(hs[0]['ledger']['events'], before['events'])

    def test_retract_retains_audit_and_does_not_mutate_input(self):
        data, hs = initial()
        evidence(hs[0], 'probe', 1, -2.)
        original = deepcopy(hs)
        value = {'op': 'retract', 'id': hs[0]['id'], 'plus_id': None, 'minus_id': None,
                 'focus': {}, 'reason': 'Withdraw after its prospective counterexample.'}
        new, events = p.review_and_apply(data, hs, proposal(value), [], 2)
        self.assertEqual(hs, original)
        self.assertEqual(new[0]['status'], 'retracted')
        self.assertEqual(new[0]['ledger']['events'], original[0]['ledger']['events'])
        with self.assertRaises(ValueError):
            evidence(new[0], 'later', 2, 3.)

    def test_menu_contract_and_action_limits(self):
        data = pool()
        hs, events = p.review_and_apply(data, [], proposal(action()), ['plus'], 1)
        self.assertEqual(hs, [])
        self.assertFalse(events[0]['accepted'])
        with self.assertRaises(ValueError):
            p.review_and_apply(data, [], proposal(*[action()] * 4), data.ids, 1)
        hs = []
        for t, plus in enumerate(['plus', 'later', 'probe', 'other'], 1):
            hs, events = p.review_and_apply(data, hs, proposal(action(plus=plus)), data.ids, t)
            self.assertTrue(events[0]['accepted'])
        hs, events = p.review_and_apply(data, hs, proposal(action(plus='last')), data.ids, 5)
        self.assertFalse(events[0]['accepted'])
        self.assertEqual(len(hs), 4)


if __name__ == '__main__':
    unittest.main()
