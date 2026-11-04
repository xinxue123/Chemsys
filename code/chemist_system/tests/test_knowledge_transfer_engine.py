"""V5 complete-loop and no-weighted-acquisition control contracts."""
from copy import deepcopy
from contextlib import ExitStack, redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from chem.benchmarks import knowledge_transfer_engine as engine
from chem.benchmarks import knowledge_transfer_llm as llm
from chem.benchmarks import knowledge_transfer_policy as policy
from chem.benchmarks import knowledge_study as study
from test_knowledge_rule_engine import fixture


class TransferEngineTests(unittest.TestCase):
    def test_control_keeps_learning_but_never_uses_acquisition_credit(self):
        pool, state, context = fixture()
        before = deepcopy(state)
        full = engine.plan(pool, None, state, 2, context, mode='full')
        control = engine.plan(pool, None, state, 2, context, mode='verification')
        self.assertEqual(state, before)
        self.assertEqual(full['mode'], 'knowledge_guided')
        self.assertEqual(control['mode'], 'base_bo')
        self.assertEqual(control['selected_id'], 'base')
        self.assertEqual(control['deployed_versions'], [])
        self.assertGreater(policy.evidence_summary(control['hypotheses_at_commit'][0])['trusted_weight'], 0.)

    def test_cold_start_is_identical_until_learning_can_change_choice(self):
        pool, state, context = fixture()
        state['hypotheses'][0]['ledger'] = policy.initial_ledger()
        context['eligible'][:] = False
        context['eligible'][[4, 5]] = True
        full = engine.plan(pool, None, state, 2, context, mode='full')
        control = engine.plan(pool, None, state, 2, context, mode='verification')
        self.assertEqual(full['mode'], 'optional_verification')
        for field in ('mode', 'selected_id', 'shadows', 'verification', 'hypotheses_at_commit'):
            self.assertEqual(full[field], control[field])
        engine.feedback(state, control, dict(candidate_id=control['selected_id'], y=.9, raw_value=.9))
        self.assertEqual(state['verification_used'], 1)
        self.assertGreater(policy.evidence_summary(state['hypotheses'][0])['trusted_weight'], 0.)
        self.assertEqual(state['decisions'][-1]['deployed_mixture_log_score_gain'], 0.)

    def test_no_prediction_effect_does_not_become_a_routing_bonus(self):
        pool, state, context = fixture()
        context['stats']['sd_y'][:] = 0.
        commit = engine.plan(pool, None, state, 2, context, mode='full')
        self.assertEqual(commit['mode'], 'base_bo')
        self.assertEqual(commit['selected_id'], 'base')
        self.assertEqual(commit['combined_logei'], commit['reference_logei'])

    def test_verification_runner_commit_crash_resume(self):
        from test_knowledge_study import make_pool, observation, FakeGP
        pool = make_pool()
        protocol = dict(seed=0, trial_seed=0, tie_seed=0, numerical_seed=0, menu_seed=0,
            protocol_sha256='synthetic_transfer_runner', supported_modes=['bo', 'full', 'verification'],
            supported_conditions=['agent_initial_terra'])
        calls, queried = [], []
        crash = {'armed': True}
        class Client:
            def close(self):
                pass
        workspace = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory(prefix='transfer_runner_', dir=workspace/'runtime') as temp, ExitStack() as stack:
            root = Path(temp)
            pool.source_path = root/'synthetic_pool.json'
            pool.source_path.write_text('{"synthetic":true}', encoding='utf-8')
            directory = study.paths(root, 'v5')/'runs'/'FakeTask_verification_agent_initial_terra'
            def oracle(root_arg, actual_pool, key):
                if key not in pool.initial_ids:
                    saved = study.read(directory/'state.json')
                    query = len(saved['decisions']) + 1
                    commit = study.read(directory/'rounds'/f'{query:02d}_commit.json')
                    self.assertEqual(commit['selected_id'], key)
                    if crash['armed']:
                        crash['armed'] = False
                        raise RuntimeError('synthetic crash after durable commitment')
                    queried.append(key)
                return observation(pool, key)
            def request(client, payload, folder):
                calls.append(payload['query'])
                return dict(proposal={'actions': [], 'summary': 'No supported transfer rule.'},
                    context_sha256=llm.digest(payload), attempt=1, resolved_model='gpt-6.1-terra')
            stack.enter_context(patch.object(study, 'check', return_value=protocol))
            stack.enter_context(patch.object(study.gpmod, 'load_task', return_value=pool))
            stack.enter_context(patch.object(study.gpmod, 'GPAdapter', FakeGP))
            stack.enter_context(patch.object(study.gpmod, 'read_observation', side_effect=oracle))
            stack.enter_context(patch.object(study.api, 'load_config', return_value=object()))
            stack.enter_context(patch.object(study.api, 'APIClient', return_value=Client()))
            stack.enter_context(patch.object(llm, 'public_payload', side_effect=lambda p,s,t,m,r: {'query': t}))
            stack.enter_context(patch.object(llm, 'request', side_effect=request))
            with redirect_stdout(StringIO()):
                with self.assertRaises(RuntimeError):
                    study.run_one(root, 'v5', pool.task, 'verification')
                first = study.read(directory/'rounds/01_commit.json')
                self.assertEqual(study.read(directory/'state.json')['status'], 'INTERRUPTED')
                result = study.run_one(root, 'v5', pool.task, 'verification')
            self.assertEqual(result['status'], 'COMPLETE')
            self.assertEqual(calls.count(1), 1)
            self.assertLessEqual(len(calls), 4)
            self.assertEqual(len(set(queried)), 16)
            self.assertEqual(first, study.read(directory/'rounds/01_commit.json'))
            final = study.read(directory/'state.json')
            self.assertEqual(len(final['history']), 20)
            self.assertEqual(final['decisions'][0]['commit_sha256'], llm.digest(first))


if __name__ == '__main__':
    unittest.main()
