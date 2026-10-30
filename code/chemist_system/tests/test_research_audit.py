"""Adversarial evidence checks: altered outcomes, subset receipts and answers."""
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from chem.decision.assistant_suite import submit_answers
from chem.decision.evaluation import CampaignSettings, run_campaign, score_campaign
from chem.decision.hypothesis import LLMResponsePending, ParameterSpec
from chem.decision.replay import InteractiveAssistantProvider, ReplayOracle, SynthesisTask, write_json_atomic
from chem.decision.reporting import audit_suite, digest


class ResearchEvidenceTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / 'data/runs/chapter5/test_scratch'
        scratch.mkdir(parents=True, exist_ok=True)
        self.tmp = TemporaryDirectory(dir=scratch)
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def make_suite(self):
        task = SynthesisTask('audit_fixture', 'numerical', 'fixture',
                             (ParameterSpec('x', 'fraction', (0., 1.)),),
                             tuple(f'c{i:02}' for i in range(12)),
                             tuple({'x': float(x)} for x in np.linspace(0., 1., 12)),
                             ({'id': 'fixture', 'text': 'fixture only'},))
        settings = CampaignSettings(initial_queries=2, additional_queries=3, gp_restarts=0)
        values = np.exp(-4 * (np.linspace(0., 1., 12) - .65) ** 2)
        batch = {'version': 1, 'hypotheses': [
            {'id': 'H1', 'kind': 'interval', 'parameter': 'x', 'unit': 'fraction', 'lower': 0., 'upper': 1.,
             'evidence_ids': ['fixture'], 'observation_ids': [], 'rationale': 'fixture domain constraint'}]}
        initial = list(task.candidate_ids[3:5])
        inputs = self.root / 'inputs' / task.task_id
        inputs.mkdir(parents=True)
        np.savez_compressed(inputs / 'oracle.npz', values=values)
        bundle = {'task': task.to_dict(), 'initial_ids': {'0': initial},
                  'priors': {'relevant': {'batch': batch, 'priority_ids': list(task.candidate_ids)}}}
        write_json_atomic(inputs / 'bundle.json', bundle)
        td = {'task_id': task.task_id, 'task_sha256': task.fingerprint, 'family': 'fixture',
              'bundle': f'inputs/{task.task_id}/bundle.json', 'oracle': f'inputs/{task.task_id}/oracle.npz',
              'bundle_sha256': sha256((inputs / 'bundle.json').read_bytes()).hexdigest(),
              'oracle_sha256': sha256((inputs / 'oracle.npz').read_bytes()).hexdigest()}
        write_json_atomic(self.root / 'manifest.json', {
            'definition': {'settings': asdict(settings), 'paired_seeds': [0],
                           'prior_regimes': ['relevant'], 'methods': ['B4']},
            'tasks': [td], 'planned_campaigns': 1})
        folder = self.root / 'trajectories' / task.task_id / 's0/relevant/B4'
        state = run_campaign(task, ReplayOracle(task.candidate_ids, values), 'B4', 0, settings, initial,
                             output_dir=folder, cache_dir=self.root / 'predictions', fixed_batch=batch)
        write_json_atomic(folder / 'result.json', score_campaign(state, task.candidate_ids, values))
        return folder, state

    def test_oracle_tampering_is_rejected_even_if_the_header_is_intact(self):
        folder, state = self.make_suite()
        self.assertEqual(audit_suite(self.root)['status'], 'PASS')
        state['oracle_ledger'][-1]['value'] += .01
        write_json_atomic(folder / 'state.json', state)
        with self.assertRaisesRegex(AssertionError, 'oracle ledger corrupted'):
            audit_suite(self.root, verify_gp=False)

    def test_revocation_receipt_is_independently_recomputed(self):
        folder, state = self.make_suite()
        self.assertEqual(state['decisions'][0]['selection']['retained_ids'], ('H1',))
        state['decisions'][0]['selection']['retained_ids'] = []
        write_json_atomic(folder / 'state.json', state)
        with self.assertRaisesRegex(AssertionError, 'subset or budget'):
            audit_suite(self.root, verify_gp=False)

    def test_prior_assistant_answer_cannot_be_overwritten(self):
        metadata = {'provider': 'interactive_codex_assistant', 'model_id': 'unit-test-only',
                    'conversation_id': 'unit-test-only', 'configured_reasoning_effort': 'unit-test-only'}
        provider = InteractiveAssistantProvider(self.root / 'requests', metadata)
        context = {'expected_version': 1, 'observations': [], 'previous_batch': {'version': 1, 'hypotheses': []},
                   'task': {'task_id': 'unit-test-only', 'parameters': [], 'candidate_ids': [],
                            'candidates': [], 'evidence': []}, 'completed_additional_queries': 0}
        with self.assertRaises(LLMResponsePending):
            provider(context)
        path = self.root / 'answers.json'
        answer = {'context_sha256': digest(context), 'batch': {'version': 1, 'hypotheses': []}, 'candidate_ids': []}
        write_json_atomic(path, [answer])
        submit_answers(self.root, path)
        answer['batch']['version'] = 2
        write_json_atomic(path, [answer])
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            submit_answers(self.root, path)


if __name__ == '__main__':
    unittest.main()
