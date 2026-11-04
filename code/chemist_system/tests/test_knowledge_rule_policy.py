"""Rule/probe semantics and two-ledger safeguards; no oracle/network/GP fits."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np

from chem.benchmarks import knowledge_rule_policy as p
from chem.benchmarks import knowledge_rule_math as math_policy
from chem.benchmarks import knowledge_rule_llm as llm


def pool():
    rows = [[.8, 'Pd-A', .50], [.4, 'Pd-A', .55], [.6, 'Pd-A', .52],
            [.6, 'Pd-Z', .52], [.6, 'Pd-Q', .52], [.1, 'Pd-A', .52],
            [.7, 'Pd-A', .50], [.5, 'Pd-A', .52], [.65, 'Pd-A', .51],
            [.6, 'Pd-Z', .51], [.3, 'Pd-A', .55]]
    ids = ['plus', 'minus', 'probe', 'cat_minus', 'cat_third', 'outside',
           'later', 'other', 'last', 'cat_later', 'tie_reference']
    axes = ['temperature', 'catalyst', 'time']
    return SimpleNamespace(ids=ids, ix={key: i for i, key in enumerate(ids)},
        axes=axes, kinds=['continuous', 'categorical', 'continuous'],
        features=np.asarray(rows, dtype=object),
        points=[dict(zip(axes, row)) for row in rows], task='synthetic', scale=100., unit='yield percent')


def action(op='add', hid=None, plus='plus', minus='minus', bounds=(.3, .9)):
    return dict(op=op, id=hid, plus_id=plus, minus_id=minus,
        focus={'temperature': dict(direction='increase', bounds=list(bounds))}, reason='mechanistic_prior: local testable rule')


def proposal(*actions):
    return dict(actions=list(actions), summary='Test local rules prospectively.')


def history(*items):
    return [dict(candidate_id=key, y=y, raw_value=y * 100) for key, y in items]


def initial():
    data, past = pool(), history(('minus', .5))
    menu = [i for i in data.ids if i != 'minus']
    hs, events = p.review_and_apply(data, [], proposal(action()), menu, 1, past)
    assert events[0]['accepted']
    return data, past, hs


def prediction(h, point, t, reference='minus', reference_y=.5, mu=0., tempering=.5):
    return math_policy.freeze_prediction(h['id'], h['version'], point, t, mu, 1., 1.1,
        reference, reference_y, tempering=tempering, semantic_key=h['semantic_key'])


def feedback(h, point, t, y, gain):
    pred = prediction(h, point, t)
    with patch.object(p, '_score_prediction', return_value=gain):
        return p.record_feedback(h, pred, dict(candidate_id=point, y=y))


class RulePolicyTests(unittest.TestCase):
    def test_plus_new_minus_own_observed_and_geometry(self):
        data, past, hs = initial()
        self.assertEqual(p.weight(hs[0]), 0.)
        menu = [i for i in data.ids if i != 'minus']
        for invalid in [action(plus='minus'), action(minus='probe'), action(bounds=(.5, 1.))]:
            out, events = p.review_and_apply(data, [], proposal(invalid), menu, 1, past)
            self.assertFalse(events[0]['accepted'])
            self.assertFalse(out)
        with self.assertRaises(ValueError):
            p.review_and_apply(data, [], proposal(action()), data.ids, 1, past)

    def test_continuous_example_location_is_not_rule_identity(self):
        data, past, hs = initial()
        feedback(hs[0], 'probe', 1, .8, .4)
        original = deepcopy(hs[0]['ledger'])
        past += history(('probe', .8))
        menu = [i for i in data.ids if i not in {r['candidate_id'] for r in past}]
        rewritten = action(plus='later')
        out, events = p.review_and_apply(data, hs, proposal(rewritten), menu, 2, past)
        self.assertEqual(events[0]['outcome'], 'duplicate_semantics_preserved')
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]['ledger'], original)

    def test_probe_reanchors_highest_compatible_reference_without_ledger_reset(self):
        data, past, hs = initial()
        h = hs[0];before = deepcopy(h)
        past += history(('other', .7), ('later', .9))
        probes = p.make_probes(data, h, past, np.arange(len(data.ids)))
        self.assertEqual(probes[data.ix['plus']]['reference_id'], 'later')
        self.assertEqual(probes[data.ix['probe']]['reference_id'], 'other')
        self.assertNotIn(data.ix['outside'], probes)
        self.assertNotIn(data.ix['minus'], probes)
        self.assertEqual(h, before)

    def test_reference_tie_uses_physical_features_not_ids_or_history_order(self):
        data, _, hs = initial()
        past = history(('minus', .7), ('tie_reference', .7))
        a = p.make_probes(data, hs[0], past, [data.ix['plus']])
        b = p.make_probes(data, hs[0], list(reversed(past)), [data.ix['plus']])
        self.assertEqual(a[data.ix['plus']]['reference_id'], 'tie_reference')
        self.assertEqual(a, b)

    def test_categorical_rule_uses_labels_not_allowed_set_order(self):
        data = pool();past = history(('cat_minus', .5))
        a = action(plus='probe', minus='cat_minus')
        a['focus'] = {'catalyst': {'direction': 'prefer_plus', 'allowed': ['Pd-Q', 'Pd-Z', 'Pd-A']}}
        menu = [i for i in data.ids if i != 'cat_minus']
        hs, events = p.review_and_apply(data, [], proposal(a), menu, 1, past)
        self.assertTrue(events[0]['accepted'])
        probes = p.make_probes(data, hs[0], past, np.arange(len(data.ids)))
        self.assertIn(data.ix['probe'], probes)
        self.assertNotIn(data.ix['cat_third'], probes)
        self.assertNotIn(data.ix['cat_later'], probes)
        self.assertEqual(hs[0]['categorical_preference'], {'catalyst': {'plus': 'Pd-A', 'minus': 'Pd-Z'}})

    def test_actual_tempered_score_is_used_and_noop_does_not_gain_credit(self):
        _, _, hs = initial();h = hs[0]
        pred = prediction(h, 'probe', 1, tempering=.2)
        expected = math_policy.score_prediction(pred, .8)
        receipt = p.record_feedback(h, pred, dict(candidate_id='probe', y=.8))
        self.assertAlmostEqual(receipt['delta_log_score'], expected)
        self.assertTrue(receipt['fact_success'])
        self.assertGreater(p.weight(h), 0.)
        before = p.evidence_summary(h)['trusted_weight']
        pred2 = prediction(h, 'other', 2, tempering=0.)
        receipt2 = p.record_feedback(h, pred2, dict(candidate_id='other', y=1.))
        self.assertEqual(receipt2['delta_log_score'], 0.)
        self.assertLessEqual(receipt2['trusted_weight_after'], before)

    def test_failed_fact_with_larger_positive_score_cannot_increase_trust(self):
        _, _, hs = initial();h = hs[0]
        for t, point in enumerate(['probe', 'other', 'last'], 1):
            feedback(h, point, t, .8, .05)
        before = p.evidence_summary(h)['trusted_weight']
        receipt = feedback(h, 'later', 4, .4, 1.)
        self.assertFalse(receipt['fact_success'])
        self.assertGreater(receipt['formula_weight'], before)
        self.assertEqual(receipt['trusted_weight_after'], before)
        self.assertGreater(receipt['score_after'], .15)

    def test_negative_prediction_even_successful_fact_cannot_increase_trust(self):
        _, _, hs = initial();h = hs[0]
        first = feedback(h, 'probe', 1, .8, 2.)
        second = feedback(h, 'other', 2, .8, -.1)
        self.assertTrue(second['fact_success'])
        self.assertGreater(second['formula_weight'], first['trusted_weight_after'])
        self.assertEqual(second['trusted_weight_after'], first['trusted_weight_after'])
        third = feedback(h, 'last', 3, .8, .01)
        self.assertGreater(third['trusted_weight_after'], second['trusted_weight_after'])

    def test_real_positive_forecast_can_coexist_with_failed_pair(self):
        _, _, hs = initial();h = hs[0]
        pred = prediction(h, 'probe', 1, reference_y=.8, mu=-1.)
        receipt = p.record_feedback(h, pred, dict(candidate_id='probe', y=.79))
        self.assertGreater(receipt['delta_log_score'], 0.)
        self.assertFalse(receipt['fact_success'])
        self.assertEqual(p.weight(h), 0.)

    def test_dormancy_preserves_potential_trust_and_allows_shadow_recovery(self):
        _, _, hs = initial();h = hs[0]
        feedback(h, 'probe', 1, .8, 2.)
        before = p.evidence_summary(h)['trusted_weight']
        h['status'] = 'dormant'
        self.assertEqual(p.weight(h), 0.)
        self.assertEqual(p.evidence_summary(h)['trusted_weight'], before)
        receipt = feedback(h, 'other', 2, .8, .1)
        self.assertEqual(receipt['weight_before'], 0.)
        self.assertEqual(receipt['weight_after'], 0.)
        self.assertGreater(receipt['trusted_weight_after'], before)
        h['status'] = 'active'
        self.assertGreater(p.weight(h), before)

    def test_revision_inherits_failure_once_but_no_positive_credit(self):
        data, past, hs = initial();h = hs[0]
        failed = feedback(h, 'probe', 1, .4, .3)
        past += history(('probe', .4))
        menu = [i for i in data.ids if i not in {r['candidate_id'] for r in past}]
        out, events = p.review_and_apply(data, hs, proposal(action('revise', h['id'], bounds=(.2, .95))), menu, 2, past)
        self.assertTrue(events[0]['accepted'])
        revised = out[-1]
        info = p.evidence_summary(revised)
        self.assertEqual(info['effective_prediction_score'], 0.)
        self.assertEqual(info['successes'], 0)
        self.assertEqual(info['failures'], 1)
        self.assertEqual(info['trusted_weight'], 0.)
        self.assertEqual(revised['inherited_negative_evidence'][0]['event_id'], failed['event_id'])
        out2, _ = p.review_and_apply(data, out, proposal(action('revise', h['id'], bounds=(.1, 1.))), menu, 3, past)
        self.assertEqual(len(out2[-1]['inherited_negative_evidence']), 1)
        self.assertEqual(p.evidence_summary(out2[-1])['failures'], 1)

    def test_inheritance_requires_both_endpoints_inside_new_scope(self):
        data, past, hs = initial();h = hs[0]
        feedback(h, 'probe', 1, .8, -.4)
        past += history(('probe', .8), ('other', .5))
        menu = [i for i in data.ids if i not in {r['candidate_id'] for r in past}]
        changed = action('revise', h['id'], minus='other', bounds=(.45, .9))
        out, events = p.review_and_apply(data, hs, proposal(changed), menu, 2, past)
        self.assertTrue(events[0]['accepted'])
        self.assertFalse(out[-1]['inherited_negative_evidence'][0]['applicable'])
        self.assertEqual(p.evidence_summary(out[-1])['effective_prediction_score'], 0.)

    def test_overlapping_rules_same_probe_inherit_one_failure_and_worst_score(self):
        data, past, hs = initial()
        menu = [i for i in data.ids if i != 'minus']
        hs, events = p.review_and_apply(data, hs, proposal(action(bounds=(.2, .95))), menu, 1, past)
        self.assertTrue(events[0]['accepted'])
        first = feedback(hs[0], 'probe', 1, .4, -.2)
        second = feedback(hs[1], 'probe', 1, .4, -.6)
        self.assertNotEqual(first['event_id'], second['event_id'])
        past += history(('probe', .4))
        menu.remove('probe')
        out, events = p.review_and_apply(data, hs, proposal(action(bounds=(.1, 1.))), menu, 2, past)
        self.assertTrue(events[0]['accepted'])
        child = out[-1]
        self.assertEqual(len(child['inherited_negative_evidence']), 2)
        self.assertTrue(all(row['applicable'] for row in child['inherited_negative_evidence']))
        info = p.evidence_summary(child)
        self.assertEqual(info['failures'], 1)
        self.assertEqual(info['inherited_failure_count'], 1)
        self.assertAlmostEqual(info['effective_prediction_score'], -.6)
        self.assertAlmostEqual(info['inherited_prediction_penalty'], -.6)
        # Defensive aggregation also holds if a restored ledger contains a
        # duplicate source record; valid record_feedback already forbids this.
        own = deepcopy(hs[0])
        own['ledger']['events'].append(deepcopy(own['ledger']['events'][0]))
        own_info = p.evidence_summary(own)
        self.assertEqual(own_info['failures'], 1)
        self.assertAlmostEqual(own_info['effective_prediction_score'], -.2)

    def test_positive_evidence_never_reappears_as_new_version_credit(self):
        data, past, hs = initial();h = hs[0]
        feedback(h, 'probe', 1, .8, 1.)
        past += history(('probe', .8))
        menu = [i for i in data.ids if i not in {r['candidate_id'] for r in past}]
        out, _ = p.review_and_apply(data, hs, proposal(action('revise', h['id'], bounds=(.2, 1.))), menu, 2, past)
        new = out[-1]
        self.assertEqual(new['ledger']['events'], [])
        self.assertEqual(p.evidence_summary(new)['successes'], 0)
        self.assertEqual(p.weight(new), 0.)
        with self.assertRaises(ValueError):
            p.record_feedback(new, prediction(new, 'probe', 2), dict(candidate_id='probe', y=.8))

    def test_duplicate_or_tampered_feedback_is_rejected(self):
        _, _, hs = initial();h = hs[0]
        pred = prediction(h, 'probe', 1)
        p.record_feedback(h, pred, dict(candidate_id='probe', y=.8))
        with self.assertRaises(ValueError):
            p.record_feedback(h, pred, dict(candidate_id='probe', y=.8))
        other = prediction(h, 'other', 2)
        other['reference_y'] = -.9
        with self.assertRaises(ValueError):
            p.record_feedback(h, other, dict(candidate_id='other', y=.8))

    def test_action_limit_and_live_limit(self):
        data, past, _ = initial();menu = [i for i in data.ids if i != 'minus']
        with self.assertRaises(ValueError):
            p.review_and_apply(data, [], proposal(*[action()] * 4), menu, 1, past)
        hs = []
        for t, lo in enumerate([.3, .2, .1, 0.], 1):
            hs, events = p.review_and_apply(data, hs, proposal(action(bounds=(lo, .9))), menu, t, past)
            self.assertTrue(events[0]['accepted'])
        out, events = p.review_and_apply(data, hs, proposal(action(bounds=(0., 1.))), menu, 5, past)
        self.assertFalse(events[0]['accepted'])
        self.assertEqual(len(out), 4)

    def test_single_remaining_plus_can_be_displayed(self):
        data, past, _ = initial()
        eligible = np.zeros(len(data.ids), dtype=bool);eligible[data.ix['plus']] = True
        context = dict(eligible=eligible, bo_index=data.ix['plus'], tie_priority=np.arange(len(data.ids)), base_logei=np.zeros(len(data.ids)))
        self.assertEqual(llm.make_menu(data, context, 1, history=past), ['plus'])

    def test_public_payload_discloses_ids_facts_not_private_gp_or_future_targets(self):
        data, past, hs = initial()
        feedback(hs[0], 'probe', 1, .8, .2)
        past += history(('probe', .8))
        state = dict(history=past, hypotheses=hs, oracle_targets='FORBIDDEN_SENTINEL')
        coverage = [dict(id=hs[0]['id'], version=1, executable_probes=2, gp_mean='FORBIDDEN_SENTINEL')]
        with patch.object(llm, 'task_context', return_value={'task': 'synthetic', 'source_references': ['private/path']}):
            payload = llm.public_payload(data, state, 2, ['plus'], coverage)
        self.assertEqual(payload['own_observations'][0]['id'], 'minus')
        self.assertIn('normalized_continuous', payload['own_observations'][0])
        self.assertEqual(payload['knowledge_ledger'][0]['tested_pairs'][0]['reference_id'], 'minus')
        self.assertEqual(payload['knowledge_ledger'][0]['tested_pairs'][0]['plus_objective'], 80.)
        self.assertNotIn('FORBIDDEN_SENTINEL', str(payload))
        self.assertNotIn('private/path', str(payload))

    def test_menu_stays_eligible_and_excludes_observed_references(self):
        data, past, _ = initial()
        eligible = np.ones(len(data.ids), dtype=bool);eligible[data.ix['minus']] = False
        context = dict(eligible=eligible, bo_index=data.ix['plus'],
                       tie_priority=np.arange(len(data.ids)), base_logei=np.zeros(len(data.ids)))
        menu = llm.make_menu(data, context, 1, maximum=5, history=past)
        self.assertLessEqual(len(menu), 5)
        self.assertEqual(len(menu), len(set(menu)))
        self.assertIn('plus', menu)
        self.assertNotIn('minus', menu)
        self.assertTrue(all(eligible[data.ix[i]] for i in menu))


if __name__ == '__main__':
    unittest.main()
