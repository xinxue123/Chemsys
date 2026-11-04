"""Causal rule/probe integration tests with synthetic inputs and no oracle/API."""
from copy import deepcopy
from types import SimpleNamespace
from contextlib import ExitStack, redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from chem.benchmarks import knowledge_rule_engine as engine
from chem.benchmarks import knowledge_rule_math as math
from chem.benchmarks import knowledge_rule_policy as policy
from chem.benchmarks import knowledge_rule_llm as rule_llm
from chem.benchmarks import knowledge_study as study


def fixture():
    ids = ('r0', 'r1', 'r2', 'r3', 'base', 'plus1', 'plus2', 'plus3', 'past_probe')
    features = ((.1, 'A'), (.2, 'A'), (.3, 'B'), (.4, 'B'), (.5, 'B'),
                (.6, 'A'), (.7, 'A'), (.8, 'A'), (.4, 'A'))
    pool = SimpleNamespace(task='Synthetic', ids=ids, ix={key:i for i,key in enumerate(ids)},
        axes=('temperature', 'catalyst'), kinds=('continuous', 'categorical'), features=features,
        points=tuple(dict(temperature=x, catalyst=c) for x,c in features), scale=1., unit='synthetic')
    history = [dict(candidate_id=key, y=y, raw_value=y) for key,y in zip(ids[:4], [.45, .60, .30, .50])]
    proposal = dict(actions=[dict(op='add', id=None, plus_id='plus1', minus_id='r1',
        focus={'temperature': {'direction':'increase', 'bounds':[0.,1.]}},
        reason='mechanistic_prior: locally higher temperature may improve the objective.')],
        summary='Test a prospective local direction against an observed reference.')
    hs, reviews = policy.review_and_apply(pool, [], proposal, list(ids[4:]), 1, history)
    assert reviews[0]['accepted'], reviews
    h = hs[0]
    pred = math.freeze_prediction(h['id'], h['version'], 'past_probe', 1,
        .5, .09, .10, 'r1', .6, tempering=1., semantic_key=h['semantic_key'])
    outcome = dict(candidate_id='past_probe', y=.75, raw_value=.75)
    policy.record_feedback(h, pred, outcome)
    assert policy.weight(h) > 0
    history.append(outcome)
    state = dict(task=pool.task, hypotheses=hs, history=history, calls=[], verification_used=0,
        decisions=[dict(query=1, evidence=[], incumbent_before_y=.6, incumbent_after_y=.75)])
    n = len(ids)
    base = np.full(n, -10.)
    base[4:8] = [0., -.04, -.03, -.02]
    available = np.asarray([key not in {x['candidate_id'] for x in history} for key in ids])
    context = dict(stats=dict(mu_y=np.full(n, .64), sd_y=np.full(n, .3),
        sd_predictive_y=np.full(n, np.sqrt(.10)), fit_key='synthetic'),
        base_logei=base, bo_index=4, available=available, eligible=available.copy(),
        tie_priority=np.arange(n), epsilon=1., acquisition_meta={})
    return pool, state, context


class RuleEngineTests(unittest.TestCase):
    def test_supported_new_plus_changes_choice_and_bad_feedback_cannot_raise_credit(self):
        pool, state, context = fixture()
        before = deepcopy(state)
        commit = engine.plan(pool, None, state, 2, context, mode='deployment')
        self.assertEqual(state, before)
        self.assertEqual(commit['mode'], 'knowledge_guided')
        self.assertIn(commit['selected_id'], ('plus1', 'plus2', 'plus3'))
        self.assertGreater(commit['combined_logei'], commit['base_logei'])
        self.assertTrue(commit['selected_mixture'])
        self.assertEqual(len(commit['shadows']), 1)
        pred = commit['shadows'][0]['prediction']
        self.assertEqual(pred['reference_id'], 'past_probe')
        self.assertEqual(pred['reference_y'], .75)
        self.assertEqual(pred['point_id'], commit['selected_id'])
        engine.feedback(state, commit, dict(candidate_id=commit['selected_id'], y=.2, raw_value=.2))
        receipt = state['decisions'][-1]['evidence'][0]
        self.assertFalse(receipt['fact_success'])
        self.assertLessEqual(receipt['weight_after'], receipt['weight_before'])
        self.assertLess(state['decisions'][-1]['deployed_mixture_log_score_gain'], 0.)

    def test_no_supported_candidate_cannot_make_third_point_win(self):
        pool, state, context = fixture()
        context['eligible'][:] = False
        context['eligible'][4] = True
        prepared, events = engine.prepare_state(pool, None, state, context, 2)
        self.assertEqual(prepared['hypotheses'][0]['status'], 'dormant')
        self.assertGreater(policy.evidence_summary(prepared['hypotheses'][0])['trusted_weight'], 0.)
        commit = engine.plan(pool, None, prepared, 2, context, mode='full')
        self.assertEqual(commit['selected_id'], 'base')
        self.assertEqual(commit['mode'], 'base_bo')
        self.assertEqual(commit['shadows'], [])
        self.assertEqual(commit['selected_mixture'], [])
        self.assertTrue(events)

    def test_zero_latent_effect_is_exact_bo_and_no_fake_mixture(self):
        pool, state, context = fixture()
        context['stats']['sd_y'][:] = 0.
        commit = engine.plan(pool, None, state, 2, context, mode='full')
        self.assertEqual(commit['selected_id'], 'base')
        self.assertEqual(commit['mode'], 'base_bo')
        self.assertEqual(commit['combined_logei'], commit['reference_logei'])
        self.assertEqual(commit['selected_mixture'], [])

    def test_reference_tampering_or_wrong_round_is_atomic(self):
        pool, state, context = fixture()
        commit = engine.plan(pool, None, state, 2, context, mode='deployment')
        snapshot = deepcopy(state)
        commit['shadows'][0]['prediction']['reference_y'] += .01
        with self.assertRaises(ValueError):
            engine.feedback(state, commit, dict(candidate_id=commit['selected_id'], y=.7, raw_value=.7))
        self.assertEqual(snapshot, state)
        commit = engine.plan(pool, None, state, 2, context, mode='deployment')
        commit['query'] = 3
        with self.assertRaises(ValueError):
            engine.feedback(state, commit, dict(candidate_id=commit['selected_id'], y=.7, raw_value=.7))
        self.assertEqual(snapshot, state)

    def test_single_new_endpoint_can_be_a_valid_cold_start_probe(self):
        pool, state, context = fixture()
        # New rule has no credit; only one unqueried plus remains, but its minus is known.
        h = state['hypotheses'][0]
        h['ledger'] = policy.initial_ledger()
        context['eligible'][:] = False
        context['eligible'][[4, 5]] = True
        commit = engine.plan(pool, None, state, 2, context, mode='full')
        self.assertEqual(commit['mode'], 'optional_verification')
        self.assertEqual(commit['selected_id'], 'plus1')
        self.assertEqual(commit['verification']['remaining_executable_probes'], 0)
        self.assertEqual(commit['verification']['reference_id'], 'past_probe')
        self.assertEqual(len(commit['shadows']), 1)

    def test_rule_runner_commit_crash_resume_preserves_all16_queries(self):
        from test_knowledge_study import make_pool, observation, FakeGP
        pool = make_pool()
        protocol = dict(seed=0, trial_seed=0, tie_seed=0, numerical_seed=0, menu_seed=0,
            protocol_sha256='synthetic_rule_runner', supported_modes=['bo','full'],
            supported_conditions=['agent_initial_terra'])
        calls, queried = [], []
        crash = {'armed': True}
        class Client:
            def close(self):
                pass
        workspace = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory(prefix='rule_runner_', dir=workspace/'runtime') as temp, ExitStack() as stack:
            root = Path(temp)
            pool.source_path = root/'synthetic_pool.json'
            pool.source_path.write_text('{"synthetic":true}', encoding='utf-8')
            directory = study.paths(root,'v4')/'runs'/'FakeTask_full_agent_initial_terra'
            def oracle(root_arg, actual_pool, key):
                if key not in pool.initial_ids:
                    state = study.read(directory/'state.json')
                    query = len(state['decisions']) + 1
                    commit = study.read(directory/'rounds'/f'{query:02d}_commit.json')
                    self.assertEqual(commit['selected_id'], key)
                    if crash['armed']:
                        crash['armed'] = False
                        raise RuntimeError('synthetic crash after durable rule commitment')
                    queried.append(key)
                return observation(pool,key)
            def request(client, payload, folder):
                calls.append(payload['query'])
                return dict(proposal={'actions':[], 'summary':'No supported rule.'},
                    context_sha256=rule_llm.digest(payload), attempt=1, resolved_model='gpt-6.1-terra')
            stack.enter_context(patch.object(study,'check',return_value=protocol))
            stack.enter_context(patch.object(study.gpmod,'load_task',return_value=pool))
            stack.enter_context(patch.object(study.gpmod,'GPAdapter',FakeGP))
            stack.enter_context(patch.object(study.gpmod,'read_observation',side_effect=oracle))
            stack.enter_context(patch.object(study.api,'load_config',return_value=object()))
            stack.enter_context(patch.object(study.api,'APIClient',return_value=Client()))
            stack.enter_context(patch.object(rule_llm,'public_payload',side_effect=lambda p,s,t,m,r: {'query':t}))
            stack.enter_context(patch.object(rule_llm,'request',side_effect=request))
            with redirect_stdout(StringIO()):
                with self.assertRaises(RuntimeError):
                    study.run_one(root,'v4',pool.task,'full')
                first = study.read(directory/'rounds/01_commit.json')
                self.assertEqual(study.read(directory/'state.json')['status'],'INTERRUPTED')
                result = study.run_one(root,'v4',pool.task,'full')
            self.assertEqual(result['status'],'COMPLETE')
            self.assertEqual(calls.count(1),1)
            self.assertLessEqual(len(calls),4)
            self.assertEqual(len(set(queried)),16)
            self.assertEqual(first,study.read(directory/'rounds/01_commit.json'))
            final = study.read(directory/'state.json')
            self.assertEqual(len(final['history']),20)
            self.assertEqual(final['decisions'][0]['commit_sha256'],rule_llm.digest(first))


if __name__ == '__main__':
    unittest.main()
