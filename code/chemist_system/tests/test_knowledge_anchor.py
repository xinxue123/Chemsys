"""V6 example-point validation; synthetic arrays only, no API/oracle/fitting."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chem.benchmarks import knowledge_anchor_engine as anchor
from chem.benchmarks import knowledge_transfer_engine as transfer
from chem.benchmarks import knowledge_transfer_policy as policy
from test_knowledge_rule_engine import fixture


def cold_fixture():
    pool, state, context = fixture()
    state['hypotheses'][0]['ledger'] = policy.initial_ledger()
    return pool, state, context


def action(point='plus1', **changes):
    row = dict(op='add', id=None, plus_id=point, minus_id='r1',
        focus={'temperature': {'direction': 'increase', 'bounds': [0., 1.]}},
        background={}, reason='mechanistic_prior: a local temperature example.')
    row.update(changes)
    return row


def proposal(*rows):
    return dict(actions=list(rows), summary='Examples for prospective validation.')


def plan(pool, state, context, *rows, t=2, mode='full'):
    return anchor.plan(pool, None, state, t, context,
        proposal=proposal(*rows) if rows else None,
        menu_ids=['base', 'plus1', 'plus2', 'plus3'], mode=mode)


class AnchorValidationTests(unittest.TestCase):
    def test_legacy_state_does_not_invent_an_unrecorded_proposal(self):
        pool, state, context = cold_fixture()
        self.assertEqual(transfer.plan(pool, None, state, 2, context)['mode'], 'optional_verification')
        result = plan(pool, state, context)
        self.assertEqual(result['mode'], 'base_bo')
        self.assertEqual(result['selected_id'], 'base')
        self.assertEqual(result['validation_proposals'], [])
        self.assertNotIn('validation_proposals', state)

    def test_accepted_example_not_other_scope_winner_gets_verified(self):
        pool, state, context = cold_fixture()
        before = deepcopy(state)
        old = transfer.plan(pool, None, state, 2, context)
        result = plan(pool, state, context, action())
        self.assertEqual(old['selected_id'], 'plus3')
        self.assertEqual(result['mode'], 'optional_verification')
        self.assertEqual(result['selected_id'], 'plus1')
        self.assertEqual(result['mechanism'], 'anchor_validation_v6')
        self.assertEqual(result['verification']['proposal']['point_id'], 'plus1')
        self.assertEqual(result['verification']['proposal']['proposed_reference_id'], 'r1')
        # The old highest-compatible rule remains authoritative, not the easy
        # lower-valued reference nominated by the LLM example.
        self.assertEqual(result['verification']['reference_id'], 'past_probe')
        self.assertTrue(result['verification']['reference_changed_from_proposal'])
        self.assertEqual(result['shadows'][0]['prediction']['reference_y'], .75)
        self.assertEqual(state, before)

    def test_new_example_of_duplicate_semantics_preserves_old_rule_credit(self):
        pool, state, context = fixture()
        original = deepcopy(state['hypotheses'][0])
        result = plan(pool, state, context, action('plus2'))
        self.assertEqual(result['reviews'][0]['outcome'], 'duplicate_semantics_preserved')
        self.assertEqual(len(result['hypotheses_at_commit']), 1)
        self.assertEqual(result['hypotheses_at_commit'][0]['ledger'], original['ledger'])
        self.assertEqual(result['hypotheses_at_commit'][0]['plus_id'], 'plus1')
        record = result['validation_proposals'][0]
        self.assertEqual(record['point_id'], 'plus2')
        self.assertEqual((record['id'], record['version'], record['semantic_key']),
            (original['id'], original['version'], original['semantic_key']))
        self.assertEqual(record['action_sha256'], anchor.digest(action('plus2')))

    def test_fresh_rule_add_records_the_accepted_action_identity(self):
        pool, state, context = cold_fixture()
        state['hypotheses'] = []
        result = plan(pool, state, context, action('plus2'))
        self.assertEqual(result['reviews'][0]['outcome'], 'add')
        record = result['validation_proposals'][0]
        self.assertEqual(record['proposal_round'], 2)
        self.assertEqual(record['action_index'], 0)
        self.assertEqual(result['selected_id'], 'plus2')

    def test_rejected_action_does_not_become_an_experiment_proposal(self):
        pool, state, context = cold_fixture()
        result = plan(pool, state, context, action('past_probe'))
        self.assertFalse(result['reviews'][0]['accepted'])
        self.assertEqual(result['validation_proposals'], [])
        self.assertEqual(result['selected_id'], 'base')

    def test_max_information_gap_and_tie_order_remain_within_proposals(self):
        pool, state, context = cold_fixture()
        result = plan(pool, state, context, action('plus1'), action('plus2'))
        # Equal synthetic MI; the existing smaller-gap tie break selects plus2.
        # plus3 is more competitive but was never proposed and is excluded.
        self.assertEqual(result['selected_id'], 'plus2')
        self.assertEqual(result['verification']['proposal']['action_index'], 1)
        self.assertEqual(len(result['validation_proposals']), 2)

    def test_old_version_wrong_semantics_retracted_or_queried_points_do_not_substitute(self):
        for invalid in ('version', 'semantic', 'retracted', 'queried'):
            with self.subTest(invalid=invalid):
                pool, state, context = cold_fixture()
                commit = plan(pool, state, context, action())
                state['validation_proposals'] = deepcopy(commit['validation_proposals'])
                if invalid == 'version':
                    state['validation_proposals'][0]['version'] += 1
                elif invalid == 'semantic':
                    state['validation_proposals'][0]['semantic_key'] = 'other_contract'
                elif invalid == 'retracted':
                    state['hypotheses'][0]['status'] = 'retracted'
                else:
                    state['history'].append(dict(candidate_id='plus1', y=.7, raw_value=.7))
                    context['eligible'][pool.ix['plus1']] = False
                    context['available'][pool.ix['plus1']] = False
                result = plan(pool, state, context)
                self.assertEqual(result['mode'], 'base_bo')
                self.assertEqual(result['selected_id'], 'base')

    def test_retracted_duplicate_does_not_revive_a_rule(self):
        pool, state, context = cold_fixture()
        state['hypotheses'][0]['status'] = 'retracted'
        result = plan(pool, state, context, action('plus2'))
        self.assertEqual(result['reviews'][0]['outcome'], 'duplicate_semantics_preserved')
        self.assertEqual(len(result['validation_proposals']), 1)
        self.assertEqual(result['hypotheses_at_commit'][0]['status'], 'retracted')
        self.assertEqual(result['mode'], 'base_bo')

    def test_weighted_deployment_is_unchanged_and_not_restricted_to_examples(self):
        pool, state, context = fixture()
        old = transfer.plan(pool, None, state, 2, context)
        result = plan(pool, state, context)
        self.assertEqual(result['mode'], 'knowledge_guided')
        for field in ('selected_id', 'combined_logei', 'deployed_versions', 'selected_mixture',
                      'shadows', 'hypotheses_at_commit', 'coverage'):
            self.assertEqual(result[field], old[field], field)
        self.assertEqual(result['validation_proposals'], [])

    def test_original_gap_cutoff_quota_and_zero_effect_gates_still_apply(self):
        for gate in ('gap', 'cutoff', 'quota', 'zero_latent_effect'):
            with self.subTest(gate=gate):
                pool, state, context = cold_fixture()
                t = 2
                if gate == 'gap':
                    context['base_logei'][pool.ix['plus1']] = -.2
                elif gate == 'cutoff':
                    t = 11
                elif gate == 'quota':
                    state['verification_used'] = 1
                else:
                    context['stats']['sd_y'][:] = 0.
                result = plan(pool, state, context, action(), t=t)
                self.assertEqual(result['mode'], 'base_bo')

    def test_feedback_records_provenance_once_without_extra_credit_or_quota(self):
        pool, state, context = cold_fixture()
        result = plan(pool, state, context, action(), action())
        self.assertEqual(len(result['validation_proposals']), 2)
        self.assertEqual(result['verification']['proposal']['action_index'], 0)
        observation = dict(candidate_id='plus1', y=.9, raw_value=.9)
        anchor.feedback(state, result, observation)
        self.assertEqual(state['verification_used'], 1)
        self.assertEqual(len(state['hypotheses'][0]['ledger']['events']), 1)
        self.assertEqual(state['validation_proposals'], result['validation_proposals'])
        snapshot = deepcopy(state)
        with self.assertRaises(ValueError):
            anchor.feedback(state, result, observation)
        self.assertEqual(state, snapshot)
        result['validation_proposals'][0]['point_id'] = 'mutated_external_commit'
        self.assertEqual(state, snapshot)

    def test_late_missing_proposal_ledger_and_prediction_failure_are_atomic(self):
        for fault in ('ledger_missing', 'prediction_reference'):
            with self.subTest(fault=fault):
                pool, state, context = cold_fixture()
                result = plan(pool, state, context, action())
                snapshot = deepcopy(state)
                if fault == 'ledger_missing':
                    del result['validation_proposals']
                    error = KeyError
                else:
                    result['shadows'][0]['prediction']['reference_y'] += .01
                    error = ValueError
                with self.assertRaises(error):
                    anchor.feedback(state, result, dict(candidate_id='plus1', y=.9, raw_value=.9))
                self.assertEqual(state, snapshot)


if __name__ == '__main__':
    unittest.main()
