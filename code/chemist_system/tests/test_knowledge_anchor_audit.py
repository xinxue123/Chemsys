"""Independent V6 auditor tamper checks; no production policy/engine or I/O."""
from copy import deepcopy
from math import log
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
import audit_knowledge_anchor_loop as audit


def fixture():
    ids = ['r0', 'r1', 'base', 'p', 'q']
    pool = dict(ids=ids, ix={key: i for i, key in enumerate(ids)},
                axes=['temperature', 'solvent'], kinds=['continuous', 'categorical'],
                features=[[.2, 'S'], [.3, 'S'], [.1, 'S'], [.6, 'S'], [.7, 'S']])
    past = [dict(candidate_id='r0', y=.4), dict(candidate_id='r1', y=.8)]
    action = dict(op='add', id=None, plus_id='p', minus_id='r0',
                  focus={'temperature': {'direction': 'increase', 'bounds': [0., 1.]}},
                  background={}, reason='Predeclared example.')
    h = audit.action_contract(pool, action, ['base', 'p', 'q'], past)
    h.update(id='H001', version=1, created_round=1, status='active',
             semantic_key=audit.digest(audit.contract(h)), inherited_negative_evidence=[],
             ledger=dict(events=[], seen=[], prediction_score=0., score=0., successes=0,
                         failures=0, trusted_weight=0., weight=0., cap=.5))
    c = dict(query=1, proposal=dict(actions=[action], summary='A prospective proposal.'),
             menu_ids=['base', 'p', 'q'], hypotheses_at_commit=[h],
             reviews=[dict(round=1, action_index=0, accepted=True, op='add',
                           outcome='add', id='H001', version=1)])
    record = dict(id='H001', version=1, semantic_key=h['semantic_key'], point_id='p',
                  proposed_reference_id='r0', proposal_round=1, action_index=0,
                  action_sha256=audit.digest(action))
    c['validation_proposals'] = [record]
    return pool, past, c


def selection_fixture():
    pool, past, c = fixture()
    h = c['hypotheses_at_commit'][0]
    row = dict(id=h['id'], version=1, weight=0., can_change_rank_at_cap=True)
    ref = dict(reference_id='r1', reference_y=.8)
    data = dict(indices=np.array([3, 4]), positions={3: 0, 4: 1},
                probes={3: ref, 4: ref}, values={'sign_information': np.array([.2, .4]),
                'offset': np.array([1., 1.]), 'pure_log_ei_ratio': np.array([1., 1.])},
                offsets=np.array([0., 0., 0., 1., 1.]))
    arrays = dict(base_logei=np.array([-1., -1., 0., -.02, -.01]),
                  eligible=np.array([False, False, True, True, True]),
                  tie_priority=np.arange(5), mu_y=np.zeros(5),
                  sd_y=np.ones(5), sd_predictive_y=np.ones(5))
    c.update(coverage=[row], epsilon=log(4), selected_id='p', mode='optional_verification',
             deployed_versions=[], combined_logei=-.02,
             verification=dict(id='H001', version=1, incremental_information=.2, information=.2,
                 reference_id='r1', reference_y=.8, remaining_executable_probes=1,
                 information_kind='binary_soft_sign', minimum_unqueried_endpoints=1,
                 proposal=deepcopy(c['validation_proposals'][0]), reference_changed_from_proposal=True))
    params = dict(last_verification_round=10, maximum_optional_verifications=1,
                  verification_maximum_gap=log(1.1), minimum_incremental_information=.01)
    return pool, past, c, arrays, ([row], {('H001', 1): data}), params


class AnchorAuditTests(unittest.TestCase):
    def test_accepted_duplicate_new_example_preserves_origin_and_old_sources(self):
        pool, past, first = fixture()
        prior = audit.proposal_ledger_audit(pool, past, first, [])
        c = deepcopy(first)
        c['query'] = 2
        a = c['proposal']['actions'][0]
        a['plus_id'] = 'q'
        c['reviews'][0].update(round=2, outcome='duplicate_semantics_preserved')
        record = dict(prior[0], point_id='q', proposal_round=2, action_sha256=audit.digest(a))
        c['validation_proposals'].append(record)
        before = deepcopy(c['hypotheses_at_commit'])
        self.assertEqual(audit.proposal_ledger_audit(pool, past, c, prior), prior+[record])
        self.assertEqual(c['hypotheses_at_commit'], before)
        self.assertEqual(c['hypotheses_at_commit'][0]['plus_id'], 'p')
        for fault in ('drop_previous', 'change_old_origin', 'invent_future_record'):
            bad = deepcopy(c)
            if fault == 'drop_previous': bad['validation_proposals'].pop(0)
            elif fault == 'change_old_origin': bad['validation_proposals'][0]['point_id'] = 'q'
            else: bad['validation_proposals'][-1]['proposal_round'] = 3
            with self.subTest(fault=fault), self.assertRaises(AssertionError):
                audit.proposal_ledger_audit(pool, past, bad, prior)

    def test_duplicate_cannot_launder_changed_scope_or_wrong_category(self):
        for fault in ('scope', 'background', 'category_label'):
            pool, past, c = fixture()
            c['reviews'][0]['outcome'] = 'duplicate_semantics_preserved'
            action = c['proposal']['actions'][0]
            if fault == 'scope': action['focus']['temperature']['bounds'] = [0., .9]
            elif fault == 'background': action['background'] = {'temperature': {'bounds': [0., 1.]}}
            else: action['background'] = {'solvent': {'allowed': ['S', 'invented']}}
            c['validation_proposals'][0]['action_sha256'] = audit.digest(action)
            with self.subTest(fault=fault), self.assertRaises(AssertionError):
                audit.proposal_ledger_audit(pool, past, c, [])

    def test_displayed_new_plus_and_own_known_reference_are_mandatory(self):
        for fault in ('not_displayed', 'already_observed', 'unknown_reference'):
            pool, past, c = fixture()
            action = c['proposal']['actions'][0]
            if fault == 'not_displayed': c['menu_ids'].remove('p')
            elif fault == 'already_observed': past.append(dict(candidate_id='p', y=.5))
            else: action['minus_id'] = 'q'
            c['validation_proposals'][0].update(point_id=action['plus_id'],
                proposed_reference_id=action['minus_id'], action_sha256=audit.digest(action))
            with self.subTest(fault=fault), self.assertRaises(AssertionError):
                audit.proposal_ledger_audit(pool, past, c, [])

    def test_review_and_ledger_must_agree_and_rejected_example_is_not_added(self):
        for fault in ('missing_review', 'duplicate_review', 'wrong_version', 'wrong_sha', 'rejected'):
            pool, past, c = fixture()
            if fault == 'missing_review': c['reviews'] = []
            elif fault == 'duplicate_review': c['reviews'].append(deepcopy(c['reviews'][0]))
            elif fault == 'wrong_version': c['reviews'][0]['version'] = 2
            elif fault == 'wrong_sha': c['validation_proposals'][0]['action_sha256'] = 'tampered'
            else: c['reviews'][0]['accepted'] = False
            with self.subTest(fault=fault), self.assertRaises(AssertionError):
                audit.proposal_ledger_audit(pool, past, c, [])

    def test_unchanged_call_preserves_list_and_retracted_cannot_execute(self):
        pool, past, c = fixture()
        prior = audit.proposal_ledger_audit(pool, past, c, [])
        c.update(query=2, proposal=None, reviews=[])
        self.assertEqual(audit.proposal_ledger_audit(pool, past, c, prior), prior)
        h = c['hypotheses_at_commit'][0]
        h['status'] = 'retracted'
        self.assertEqual(audit.executable_examples(pool, h, {'positions': {3: 0}}, prior), {})

    def test_new_example_must_not_rewrite_existing_rule_origin(self):
        _, _, c = fixture()
        old = c['hypotheses_at_commit']
        new = deepcopy(old)
        audit.origin_identity_audit(new, old)
        new[0]['plus_id'] = 'q'
        with self.assertRaises(AssertionError):
            audit.origin_identity_audit(new, old)

    def check_selection(self, fault=None):
        pool, past, c, arrays, calc, params = selection_fixture()
        if fault == 'nonexample': c.update(selected_id='q', combined_logei=-.01)
        elif fault == 'ref_flag': c['verification']['reference_changed_from_proposal'] = False
        elif fault == 'reference': c['verification'].update(reference_id='r0', reference_y=.4)
        elif fault == 'provenance': c['verification']['proposal']['action_index'] = 1
        elif fault == 'gap': arrays['base_logei'][3] = -.11
        elif fault == 'MI': calc[1][('H001', 1)]['values']['sign_information'][0] = .009
        elif fault == 'no_effect': calc[1][('H001', 1)]['values']['offset'][0] = 0.
        elif fault == 'late': c['query'] = 11
        used = int(fault == 'quota')
        with patch.object(audit, 'calculations', return_value=calc), patch.object(audit, 'adaptive_ei_check', return_value=0.):
            return audit.selection_audit(pool, past, c, arrays, 'full', used, params)

    def test_valid_example_beats_only_unproposed_larger_MI_scope_point(self):
        self.check_selection()

    def test_nonexample_or_reference_provenance_tampering_is_detected(self):
        for fault in ('nonexample', 'ref_flag', 'reference', 'provenance'):
            with self.subTest(fault=fault), self.assertRaises(AssertionError):
                self.check_selection(fault)

    def test_existing_gap_MI_effect_cutoff_and_quota_cannot_be_bypassed(self):
        for fault in ('gap', 'MI', 'no_effect', 'late', 'quota'):
            with self.subTest(fault=fault), self.assertRaises(AssertionError):
                self.check_selection(fault)


if __name__ == '__main__':
    unittest.main()
