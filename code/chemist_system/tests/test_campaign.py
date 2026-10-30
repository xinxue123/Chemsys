"""Policy distinctions, resumable BO, and pre-test freeze qualification."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from chem.decision.domain_guard import METHODS, select_policy
from chem.decision.evaluation import (CampaignSettings, freeze_math_test, prepare_math_suite,
                                      run_campaign, run_math_suite, score_campaign)
from chem.decision.hypothesis import LLMResponsePending, ParameterSpec
from chem.decision.replay import (MATH_REGIMES, ReplayOracle, SynthesisTask,
                                 make_reaction_stress_priors, make_stress_priors,
                                 mathematical_task, summit_task, write_json_atomic)


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.scores = np.array([1, .99, .4])
        self.h = np.array([[0, 1, 1], [1, 0, 1]], dtype=bool)
        self.kwargs = {'candidate_ids': ('a', 'b', 'c'), 'hypothesis_ids': ('H1', 'H2')}

    def choose(self, method, **kwargs):
        return select_policy(method, self.scores, self.h, .05, **self.kwargs, **kwargs)

    def test_joint_minimum_revision_and_revoke_all_have_distinct_results(self):
        joint, all_revoked = self.choose('B4'), self.choose('A2')
        self.assertEqual(joint['candidate_id'], 'b')
        self.assertEqual(joint['retained_ids'], ('H1',))
        self.assertEqual(all_revoked['candidate_id'], 'a')
        self.assertEqual(all_revoked['retained_ids'], ())

    def test_single_candidate_validation_is_not_revoke_all(self):
        accepted = self.choose('B3', suggested_candidate_id='b')
        rejected = self.choose('B3', suggested_candidate_id='c')
        self.assertEqual(accepted['candidate_id'], 'b')
        self.assertEqual(rejected['candidate_id'], 'a')
        self.assertFalse(accepted['hypothesis_subset_decision'])
        self.assertNotEqual(accepted['reason'], self.choose('A2')['reason'])

    def test_single_checks_and_nonempty_checks_can_violate_the_budget(self):
        for method in ('B1', 'A1', 'A3'):
            result = self.choose(method)
            self.assertEqual(result['candidate_id'], 'c')
            self.assertTrue(result['budget_violation'])
        self.assertFalse(self.choose('B4')['budget_violation'])

    def test_empty_conjunction_stops_direct_trust_and_does_not_fallback(self):
        h = np.array([[1, 0, 0], [0, 1, 0]], dtype=bool)
        result = select_policy('B1', self.scores, h, .05, **self.kwargs)
        self.assertEqual(result['status'], 'STOPPED')
        self.assertIsNone(result['candidate_index'])

    def test_fixed_escape_is_reproducible_and_only_on_scheduled_rounds(self):
        first = self.choose('B2', round_number=4, rng=np.random.default_rng(3))
        second = self.choose('B2', round_number=4, rng=np.random.default_rng(3))
        self.assertEqual(first, second)
        self.assertEqual(first['reason'], 'scheduled_uniform_escape')
        self.assertEqual(self.choose('B2', round_number=3)['candidate_id'], 'c')

    def test_every_policy_respects_hard_and_queried_masks(self):
        for method in METHODS:
            result = self.choose(method, hard_mask=np.array([False, True, True]),
                                  queried_mask=np.array([False, False, True]),
                                  round_number=4, rng=np.random.default_rng(4), suggested_candidate_id='a')
            if result['status'] == 'SELECTED':
                self.assertEqual(result['candidate_id'], 'b')


class CampaignTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / 'data/runs/chapter5/test_scratch'
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = TemporaryDirectory(dir=scratch)
        self.root = Path(self.temporary.name)
        self.ids = tuple(f'c{i:02d}' for i in range(12))
        self.rows = tuple({'x': float(x)} for x in np.linspace(0, 1, 12))
        self.values = np.exp(-8 * (np.linspace(0, 1, 12) - .65) ** 2)
        self.task = SynthesisTask('unit_test_campaign', 'not_applicable_math', 'numerical_fixture',
                                  (ParameterSpec('x', 'fraction', (0, 1)),), self.ids, self.rows,
                                  ({'id': 'source', 'text': 'unit test evidence'},))
        self.settings = CampaignSettings(initial_queries=2, additional_queries=3, gp_restarts=0,
                                         refresh_after=(0, 1, 2))
        self.batch = {'version': 1, 'hypotheses': [self.hypothesis()]}

    def tearDown(self):
        self.temporary.cleanup()

    def hypothesis(self, **overrides):
        result = {'id': 'H1', 'kind': 'interval', 'parameter': 'x', 'unit': 'fraction',
                  'lower': .2, 'upper': .8, 'evidence_ids': ['source'], 'observation_ids': [],
                  'rationale': 'Explicitly a unit-test fixture.'}
        result.update(overrides)
        return result

    def run_one(self, name, *, oracle=None, method='B4', settings=None, **kwargs):
        return run_campaign(self.task, oracle or ReplayOracle(self.ids, self.values), method, 0,
                            settings or self.settings, self.ids[:2], output_dir=self.root / name,
                            cache_dir=self.root / 'predictions', **kwargs)

    def test_uninterrupted_and_resumed_trajectories_match_without_requery(self):
        uninterrupted = self.run_one('full', fixed_batch=self.batch)
        partial = self.run_one('resume', fixed_batch=self.batch, max_new_steps=1)
        self.assertEqual(partial['status'], 'CHECKPOINTED')
        self.assertEqual(len(partial['oracle_ledger']), 3)
        resumed = self.run_one('resume', fixed_batch=self.batch)
        self.assertEqual(resumed['status'], 'COMPLETE')
        self.assertEqual(resumed['oracle_ledger'], uninterrupted['oracle_ledger'])
        self.assertEqual(len({r['candidate_id'] for r in resumed['oracle_ledger']}), 5)
        self.assertTrue(resumed['decisions'][1]['gp_cache_hit'])
        result = score_campaign(resumed, self.ids, self.values)
        self.assertEqual(result['budget_violations'], 0)
        self.assertEqual(result['repeat_queries'], 0)

    def test_checkpoint_refuses_changed_settings_and_unseen_oracle_values(self):
        self.run_one('resume', max_new_steps=1)
        with self.assertRaises(ValueError):
            self.run_one('resume', settings=replace(self.settings, kappa=3))
        changed = self.values.copy(); changed[-1] += .01
        with self.assertRaises(ValueError):
            self.run_one('resume', oracle=ReplayOracle(self.ids, changed))

    def test_stopped_campaign_carries_incumbent_to_the_full_budget(self):
        empty = {'version': 1, 'hypotheses': [self.hypothesis(lower=.01, upper=.02)]}
        state = self.run_one('stopped', method='B1', fixed_batch=empty)
        self.assertEqual(state['status'], 'STOPPED')
        result = score_campaign(state, self.ids, self.values)
        self.assertEqual(result['actual_queries'], 2)
        self.assertEqual(len(result['best_curve']), 5)
        self.assertEqual(result['search_auc'], result['initial_best'])

    def test_numerical_failure_is_recorded_and_not_dropped(self):
        def failed_snapshot(*args, **kwargs):
            raise np.linalg.LinAlgError('injected numerical failure')
        with patch('chem.decision.evaluation._predict_snapshot', new=failed_snapshot):
            state = self.run_one('failure', method='B0')
        result = score_campaign(state, self.ids, self.values)
        self.assertEqual(result['status'], 'MODEL_ERROR')
        self.assertEqual(result['actual_queries'], 2)
        self.assertEqual(len(result['best_curve']), 5)

    def test_pending_assistant_resumes_the_same_attempt_and_retains_history(self):
        requests = []
        def provider(context):
            requests.append(context)
            if context['expected_version'] == 2 and not provider.ready:
                raise LLMResponsePending('fixture_pending_request.json')
            return {'batch': {'version': context['expected_version'], 'hypotheses': [self.hypothesis()]},
                    'candidate_ids': ['c05', 'c06', 'c07', 'c08'], 'provider': {'kind': 'unit_test_fixture'}}
        provider.ready = False
        provider.model_metadata = {'kind': 'unit_test_fixture'}
        provider.prompt_text = 'unit-test protocol prompt'
        pending = self.run_one('assistant', provider=provider)
        self.assertEqual(pending['status'], 'WAITING_LLM')
        self.assertEqual(len(pending['oracle_ledger']), 3)
        self.assertEqual(len(pending['hypothesis_state']['events']), 1)
        with self.assertRaises(ValueError):
            score_campaign(pending, self.ids, self.values)
        provider.ready = True
        completed = self.run_one('assistant', provider=provider)
        self.assertEqual(completed['status'], 'COMPLETE')
        self.assertEqual(len(completed['oracle_ledger']), 5)
        self.assertEqual(len(completed['hypothesis_state']['events']), 3)
        self.assertEqual([r['candidate_id'] for r in requests[1]['observations']],
                         [r['candidate_id'] for r in requests[2]['observations']])
        self.assertNotIn('pool_optimum', json.dumps(requests))
        self.assertNotIn('oracle_values', json.dumps(requests))


class MathematicalInputTests(unittest.TestCase):
    def test_reaction_unit_conversion_does_not_mutate_other_regimes(self):
        task = summit_task(1, pool_seed=20262026, evidence=[{'id': 'synthetic_prior_constructor',
                                                          'text': 'Artificial test fixture.'}])
        values = np.array([np.exp(-((row['temperature'] - 75) / 30) ** 2) for row in task.candidates])
        priors = make_reaction_stress_priors(task, values)
        from chem.decision.hypothesis import compile_membership, parse_hypotheses
        for definition in priors.values():
            hypotheses = parse_hypotheses(definition['batch'], task.schema,
                                          evidence_ids={'synthetic_prior_constructor'}, observed_ids=set())
            self.assertEqual(compile_membership(hypotheses, task.candidates, task.schema).shape[1], 512)
            for row in definition['batch']['hypotheses']:
                if row['kind'] == 'interval':
                    self.assertEqual(row['unit'], task.schema[row['parameter']].unit)

    def test_atomic_write_retries_a_transient_windows_lock(self):
        root = Path(__file__).resolve().parents[1] / 'data/runs/chapter5/test_scratch'
        root.mkdir(parents=True, exist_ok=True)
        original_replace = Path.replace
        attempts = []
        def replace_after_lock(source, target):
            attempts.append(str(source))
            if len(attempts) == 1:
                raise PermissionError('injected transient file lock')
            return original_replace(source, target)
        with TemporaryDirectory(dir=root) as temporary, patch.object(Path, 'replace', replace_after_lock):
            path = Path(temporary) / 'state.json'
            write_json_atomic(path, {'query': 1})
            self.assertEqual(json.loads(path.read_text(encoding='utf-8')), {'query': 1})
        self.assertEqual(len(attempts), 2)

    def test_explicit_functions_and_stress_batches_have_expected_properties(self):
        for family in range(1, 6):
            task, values = mathematical_task(family, 4, pool_seed=20271026 + family)
            self.assertEqual(len(values), 512)
            self.assertTrue(np.all((values >= 0) & (values <= 1)))
            priors = make_stress_priors(task, values)
            self.assertEqual(tuple(priors), MATH_REGIMES)
            correct = priors['relevant']['construction']
            self.assertEqual(correct['reference_high_count_joint'], 52)
            joint = priors['joint_only_harmful']['construction']
            self.assertEqual(joint['reference_high_count_joint'], 0)
            self.assertGreater(joint['full_pool_joint_count'], 0)
            self.assertTrue(all(v > 0 for v in joint['reference_high_counts_per_constraint']))

    def test_test_inputs_cannot_run_before_development_freeze(self):
        root = Path(__file__).resolve().parents[1]
        protocol = json.loads((root / 'research/chapter5/protocol.json').read_text(encoding='utf-8'))
        scratch = root / 'data/runs/chapter5/test_scratch'
        scratch.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=scratch) as temporary:
            folder = Path(temporary)
            test = prepare_math_suite(folder / 'test', protocol, split='test')
            dev = prepare_math_suite(folder / 'dev', protocol, split='development')
            self.assertEqual(test['planned_campaigns'], 3200)
            self.assertEqual(dev['planned_campaigns'], 160)
            self.assertFalse({r['task_id'] for r in test['tasks']} & {r['task_id'] for r in dev['tasks']})
            with self.assertRaises(ValueError):
                run_math_suite(folder / 'test', workers=1)
            with self.assertRaises(ValueError):
                freeze_math_test(folder / 'test', folder / 'dev')
            path = folder / 'test' / test['tasks'][0]['bundle']
            path.write_text('{}', encoding='utf-8')
            with self.assertRaises(ValueError):
                prepare_math_suite(folder / 'test', protocol, split='test')


if __name__ == '__main__':
    unittest.main()
