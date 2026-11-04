"""V5 declared-background transfer and smooth prospective trust; offline only."""
from copy import deepcopy
import json
from math import tanh
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np

from chem.benchmarks import knowledge_transfer_policy as p
from chem.benchmarks import knowledge_transfer_llm as llm
from chem.benchmarks import knowledge_rule_policy as old
from chem.benchmarks import knowledge_rule_llm as old_llm
from chem.benchmarks import knowledge_rule_math as maths


def dataset(categorical=True):
    if categorical:
        axes = ['ligand', 'solvent', 'base']
        kinds = ['categorical'] * 3
        rows = [['L+', 'S1', 'B'], ['L-', 'S1', 'B'],
                ['L+', 'S2', 'B'], ['L-', 'S2', 'B'],
                ['L+', 'S3', 'B'], ['L-', 'S3', 'B'], ['Lx', 'S2', 'B']]
    else:
        axes = ['temperature', 'time', 'catalyst']
        kinds = ['continuous', 'continuous', 'categorical']
        rows = [[.8, .2, 'Pd'], [.4, .2, 'Pd'], [.8, .8, 'Pd'],
                [.4, .8, 'Pd'], [.8, .6, 'Pd'], [.4, .6, 'Pd'], [.8, .95, 'Pd']]
    ids = ['a_plus', 'a_ref', 'b_plus', 'b_ref', 'c_plus', 'c_ref', 'extra']
    return SimpleNamespace(axes=axes, kinds=kinds, ids=ids,
        ix={key: i for i, key in enumerate(ids)}, features=np.asarray(rows, dtype=object),
        points=[dict(zip(axes, row)) for row in rows], task='synthetic', scale=100., unit='yield percent')


def history(*pairs):
    return [dict(candidate_id=key, y=y, raw_value=100 * y) for key, y in pairs]


def action(categorical=True, background=None, **overrides):
    focus = {'ligand': {'direction': 'prefer_plus', 'allowed': ['L+', 'L-']}} if categorical else {
        'temperature': {'direction': 'increase', 'bounds': [.3, .9]}}
    a = dict(op='add', id=None, plus_id='a_plus', minus_id='a_ref', focus=focus,
             background=background or {}, reason='mechanistic_prior: transferable with uncertain interactions')
    a.update(overrides)
    return a


def proposal(*actions):
    return dict(actions=list(actions), summary='Predeclare transfer, then test it prospectively.')


def create(categorical=True, background=None):
    pool = dataset(categorical)
    past = history(('a_ref', .5), ('b_ref', .7))
    menu = [key for key in pool.ids if key not in {'a_ref', 'b_ref'}]
    hs, events = p.review_and_apply(pool, [], proposal(action(categorical, background)), menu, 1, past)
    assert events[0]['accepted'], events
    return pool, past, hs


def feedback(h, point, t, y, gain, reference='a_ref', reference_y=.5):
    pred = maths.freeze_prediction(h['id'], h['version'], point, t, 0., 1., 1.1,
        reference, reference_y, tempering=.5, semantic_key=h['semantic_key'])
    with patch.object(p, '_score_prediction', return_value=gain):
        return p.record_feedback(h, pred, dict(candidate_id=point, y=y))


class TransferPolicyTests(unittest.TestCase):
    def test_p1_two_backgrounds_reuse_rule_after_first_feedback(self):
        pool, past, hs = create(background={'solvent': {'allowed': ['S1', 'S2']}})
        h = hs[0]
        probes = p.make_probes(pool, h, past, np.arange(len(pool.ids)))
        self.assertEqual(probes[pool.ix['a_plus']]['reference_id'], 'a_ref')
        self.assertEqual(probes[pool.ix['b_plus']]['reference_id'], 'b_ref')
        first = feedback(h, 'a_plus', 1, .8, .4)
        past += history(('a_plus', .8))
        before = deepcopy(h['ledger'])
        probes = p.make_probes(pool, h, past, np.arange(len(pool.ids)))
        self.assertNotIn(pool.ix['a_plus'], probes)
        self.assertEqual(probes[pool.ix['b_plus']]['reference_id'], 'b_ref')
        self.assertEqual(h['ledger'], before)
        self.assertGreater(first['trusted_weight_after'], 0.)
        self.assertEqual(h['version'], 1)

    def test_expanded_background_needs_own_same_background_reference(self):
        pool, _, hs = create(background={'solvent': {'allowed': ['S1', 'S2', 'S3']}})
        past = history(('a_ref', .95), ('b_ref', .2))
        probes = p.make_probes(pool, hs[0], past, np.arange(len(pool.ids)))
        self.assertEqual(probes[pool.ix['b_plus']]['reference_id'], 'b_ref')
        self.assertEqual(probes[pool.ix['b_plus']]['reference_y'], .2)
        self.assertNotIn(pool.ix['c_plus'], probes)
        self.assertNotIn(pool.ix['extra'], probes)

    def test_two_tested_backgrounds_leave_third_probe_with_same_rule_version(self):
        pool, past, hs = create(background={'solvent': {'allowed': ['S1', 'S2', 'S3']}})
        past += history(('c_ref', .6))
        h = hs[0]
        feedback(h, 'a_plus', 1, .8, .4)
        feedback(h, 'b_plus', 2, .9, .3, 'b_ref', .7)
        past += history(('a_plus', .8), ('b_plus', .9))
        probes = p.make_probes(pool, h, past, np.arange(len(pool.ids)))
        self.assertEqual(set(probes), {pool.ix['c_plus']})
        self.assertEqual(probes[pool.ix['c_plus']]['reference_id'], 'c_ref')
        self.assertEqual(p.evidence_summary(h)['successes'], 2)
        self.assertEqual(h['version'], 1)

    def test_undeclared_background_preserves_v4_singleton(self):
        pool, past, hs = create()
        probes = p.make_probes(pool, hs[0], past, np.arange(len(pool.ids)))
        self.assertEqual(set(probes), {pool.ix['a_plus']})
        legacy_action = action(); legacy_action.pop('background')
        legacy_hs, _ = old.review_and_apply(pool, [], proposal(legacy_action), ['a_plus'], 1, past)
        self.assertEqual(hs[0]['scope'], legacy_hs[0]['scope'])
        self.assertEqual(hs[0]['semantic_key'], legacy_hs[0]['semantic_key'])

    def test_explicit_continuous_background_keeps_probe_distance_limit(self):
        pool, _, hs = create(False, {'time': {'bounds': [0., 1.]}})
        past = history(('a_ref', .95), ('b_ref', .2))
        probes = p.make_probes(pool, hs[0], past, np.arange(len(pool.ids)))
        self.assertEqual(probes[pool.ix['b_plus']]['reference_id'], 'b_ref')
        self.assertNotEqual(probes[pool.ix['b_plus']]['reference_id'], 'a_ref')
        # b_ref at .8 can compare to .6, but the high .2 reference cannot.
        self.assertEqual(probes[pool.ix['c_plus']]['reference_id'], 'b_ref')
        narrow = create(False)[2][0]
        self.assertNotIn(pool.ix['b_plus'], p.make_probes(pool, narrow, past, np.arange(len(pool.ids))))

    def test_background_does_not_relax_original_seed_pair_comparability(self):
        pool = dataset();past = history(('b_ref', .7))
        a = action(background={'solvent': {'allowed': ['S1', 'S2']}}, minus_id='b_ref')
        out, ev = p.review_and_apply(pool, [], proposal(a), ['a_plus'], 1, past)
        self.assertFalse(ev[0]['accepted']);self.assertFalse(out)
        pool = dataset(False)
        a = action(False, {'time': {'bounds': [0., 1.]}}, minus_id='b_ref')
        out, ev = p.review_and_apply(pool, [], proposal(a), ['a_plus'], 1, past)
        self.assertFalse(ev[0]['accepted']);self.assertFalse(out)

    def test_invalid_background_rejected_not_silently_ignored(self):
        for categorical, invalid in [
            (True, {'missing': {'allowed': ['S1']}}),
            (True, {'ligand': {'allowed': ['L+', 'L-']}}),
            (True, {'solvent': {'allowed': ['S2']}}),
            (True, {'solvent': {'allowed': ['S1', 'invented']}}),
            (True, {'solvent': {'allowed': ['S1', 'S1']}}),
            (True, {'solvent': {'allowed': [0, 1]}}),
            (True, {'solvent': {'bounds': [0., 1.]}}),
            (False, {'time': {'bounds': [.3, .9]}}),
            (False, {'time': {'bounds': [0., float('inf')]}}),
            (False, {'time': {'bounds': [float('nan'), 1.]}}),
            (False, {'time': {'bounds': [False, 1.]}}),
            (False, {'time': {'bounds': [0., 1.1]}}),
            (False, {'time': {'bounds': [1., 0.]}}),
            (False, {'time': {'bounds': [0., 1.], 'extra': True}}),
        ]:
            with self.subTest(invalid=invalid):
                pool = dataset(categorical);past = history(('a_ref', .5))
                out, events = p.review_and_apply(pool, [], proposal(action(categorical, invalid)), ['a_plus'], 1, past)
                self.assertFalse(events[0]['accepted']);self.assertFalse(out)

    def test_revision_scope_new_version_no_positive_credit_negative_inheritance(self):
        pool, past, hs = create(background={'solvent': {'allowed': ['S1', 'S2']}})
        h = hs[0]
        negative = feedback(h, 'a_plus', 1, .4, -.3)
        feedback(h, 'b_plus', 2, .9, 1., 'b_ref', .7)
        past += history(('a_plus', .4), ('b_plus', .9), ('c_ref', .6))
        revised = action(background={'solvent': {'allowed': ['S1', 'S2', 'S3']}},
            op='revise', id=h['id'], plus_id='c_plus', minus_id='c_ref')
        out, ev = p.review_and_apply(pool, hs, proposal(revised), ['c_plus'], 3, past)
        self.assertTrue(ev[0]['accepted'])
        new = out[-1]
        self.assertEqual(new['version'], 2)
        self.assertNotEqual(new['semantic_key'], h['semantic_key'])
        self.assertEqual(out[0]['status'], 'retracted')
        self.assertEqual(new['ledger']['events'], [])
        self.assertEqual(p.evidence_summary(new)['successes'], 0)
        self.assertEqual(p.evidence_summary(new)['failures'], 1)
        self.assertEqual(p.evidence_summary(new)['effective_prediction_score'], -.3)
        self.assertEqual(new['inherited_negative_evidence'][0]['event_id'], negative['event_id'])
        self.assertEqual(p.weight(new), 0.)
        with self.assertRaises(ValueError):
            feedback(new, 'a_plus', 3, .4, 1.)

    def test_revision_excludes_old_background_without_losing_audit(self):
        pool, past, hs = create(background={'solvent': {'allowed': ['S1', 'S2']}})
        feedback(hs[0], 'a_plus', 1, .4, -.3)
        past += history(('a_plus', .4))
        revised = action(background={'solvent': {'allowed': ['S2']}}, op='revise',
            id=hs[0]['id'], plus_id='b_plus', minus_id='b_ref')
        out, ev = p.review_and_apply(pool, hs, proposal(revised), ['b_plus'], 2, past)
        self.assertTrue(ev[0]['accepted'])
        self.assertFalse(out[-1]['inherited_negative_evidence'][0]['applicable'])
        self.assertEqual(p.evidence_summary(out[-1])['failures'], 0)

    def test_equivalent_background_order_does_not_reset_semantics(self):
        pool, past, hs = create(background={'solvent': {'allowed': ['S2', 'S1']}})
        feedback(hs[0], 'a_plus', 1, .8, .4)
        past += history(('a_plus', .8))
        rewritten = action(background={'solvent': {'allowed': ['S1', 'S2']}}, plus_id='b_plus', minus_id='b_ref')
        out, ev = p.review_and_apply(pool, hs, proposal(rewritten), ['b_plus'], 2, past)
        self.assertEqual(ev[0]['outcome'], 'duplicate_semantics_preserved')
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]['ledger'], hs[0]['ledger'])

    def test_tiny_positive_gain_recovers_only_tiny_credit(self):
        h = create()[2][0]
        first = feedback(h, 'a_plus', 1, .8, .2)
        self.assertAlmostEqual(first['trusted_weight_after'], tanh(.1))
        failed = feedback(h, 'b_plus', 2, .4, 2.)
        self.assertEqual(failed['trusted_weight_after'], 0.)
        tiny = feedback(h, 'c_plus', 3, .8, 1e-8)
        self.assertGreater(tiny['formula_weight'], .1)
        self.assertAlmostEqual(tiny['recovery_credit_limit'], 5e-9)
        self.assertLessEqual(tiny['trusted_weight_after'], 5e-9)
        self.assertEqual(tiny['score_after'], 2.20000001)

    def test_dormancy_keeps_potential_not_deployment_weight(self):
        h = create()[2][0]
        feedback(h, 'a_plus', 1, .8, .2)
        before = p.evidence_summary(h)['trusted_weight']
        h['status'] = 'dormant'
        receipt = feedback(h, 'b_plus', 2, .8, .01)
        self.assertEqual(receipt['weight_before'], 0.)
        self.assertEqual(receipt['weight_after'], 0.)
        self.assertAlmostEqual(receipt['recovery_weight_ceiling'], before + tanh(.005))
        self.assertGreater(receipt['trusted_weight_after'], before)
        h['status'] = 'active'
        self.assertEqual(p.weight(h), receipt['trusted_weight_after'])

    def test_zero_or_negative_gain_and_false_fact_cannot_increase_trust(self):
        for gain, y in [(0., .8), (-.1, .8), (1., .4)]:
            h = create()[2][0]
            first = feedback(h, 'a_plus', 1, .8, .2)
            receipt = feedback(h, 'b_plus', 2, y, gain)
            self.assertEqual(receipt['recovery_credit_limit'], 0.)
            self.assertLessEqual(receipt['trusted_weight_after'], first['trusted_weight_after'])


class TransferLLMTests(unittest.TestCase):
    def test_seven_fields_and_retain_contract(self):
        raw = proposal(action(background={'solvent': {'allowed': ['S1', 'S2']}}))
        self.assertEqual(llm.validate(raw), raw)
        bad = deepcopy(raw);bad['actions'][0].pop('background')
        with self.assertRaises(ValueError):llm.validate(bad)
        a = action(op='retain', id='H001', plus_id=None, minus_id=None, focus={})
        self.assertEqual(llm.validate(proposal(a)), proposal(a))
        a['background'] = {'solvent': {'allowed': ['S1']}}
        with self.assertRaises(ValueError):llm.validate(proposal(a))

    def test_public_payload_contract_no_gp_and_identical_menu(self):
        pool, past, hs = create(background={'solvent': {'allowed': ['S1', 'S2']}})
        state = dict(hypotheses=hs, history=past, oracle='FORBIDDEN')
        with patch.object(llm, 'task_context', return_value={'categories': ['S1', 'S2'], 'source_references': ['PRIVATE']}):
            payload = llm.public_payload(pool, state, 1, ['a_plus', 'b_plus'], [{'id':'H001', 'gp_mean':'FORBIDDEN'}])
        self.assertEqual(payload['knowledge_ledger'][0]['background'], hs[0]['background'])
        self.assertEqual(payload['knowledge_ledger'][0]['scope'], hs[0]['scope'])
        self.assertNotIn('FORBIDDEN', str(payload));self.assertNotIn('PRIVATE', str(payload))
        eligible = np.array([key not in {'a_ref', 'b_ref'} for key in pool.ids])
        context = dict(eligible=eligible, bo_index=0, tie_priority=np.arange(len(pool.ids)), base_logei=np.zeros(len(pool.ids)))
        self.assertEqual(llm.make_menu(pool, context, 2, history=past), old_llm.make_menu(pool, context, 2, history=past))

    def test_cache_uses_new_prompt_fixed_model_and_no_second_call(self):
        raw = proposal(action())
        calls = []
        class Fake:
            def complete(self, system, user, **kwargs):
                calls.append((system, user, kwargs))
                return dict(output_text=json.dumps(raw), resolved_model=llm.api.MODEL)
        payload = {'task': 'synthetic'}
        with TemporaryDirectory(dir=Path(__file__).parent) as folder:
            first = llm.request(Fake(), payload, folder)
            second = llm.request(Fake(), payload, folder)
            self.assertEqual(first, second);self.assertEqual(len(calls), 1)
            context = json.loads((Path(folder) / first['context_sha256'] / 'context.json').read_text(encoding='utf-8'))
            self.assertEqual(context['model'], 'gpt-6.1-terra')
            self.assertEqual(context['reasoning_effort'], 'medium')
            self.assertEqual(calls[0][2], {'json_output': True})
            self.assertNotEqual(llm.SYSTEM, old_llm.SYSTEM)
            self.assertEqual(context['system'], llm.SYSTEM)

    def test_public_prefix_is_arm_independent_and_exposes_latent_trust(self):
        pool, past, hs = create(background={'solvent': {'allowed': ['S1', 'S2']}})
        feedback(hs[0], 'a_plus', 1, .8, .4)
        hs[0]['status'] = 'dormant'
        past += history(('a_plus', .8))
        state = dict(hypotheses=hs, history=past, mode='full')
        other = dict(state, mode='verification')
        with patch.object(llm, 'task_context', return_value={'task':'synthetic'}):
            full = llm.public_payload(pool, state, 2, ['b_plus'], [])
            control = llm.public_payload(pool, other, 2, ['b_plus'], [])
        self.assertEqual(full, control)
        self.assertGreater(full['knowledge_ledger'][0]['trusted_weight'], 0.)
        self.assertNotIn('deployment_weight', full['knowledge_ledger'][0])
        self.assertNotIn('mode', full)

    def test_schema_retry_is_same_context_no_experiment_or_model_fallback(self):
        raw = proposal(action());invalid = deepcopy(raw);invalid['actions'][0].pop('background')
        calls = []
        class Fake:
            def complete(self, system, user, **kwargs):
                calls.append(user)
                return dict(output_text=json.dumps(invalid if len(calls)==1 else raw), resolved_model=llm.api.MODEL)
        with TemporaryDirectory(dir=Path(__file__).parent) as folder:
            result = llm.request(Fake(), {'task': 'synthetic'}, folder)
        self.assertEqual(result['attempt'], 2)
        self.assertEqual(calls[1]['same_experiment_context'], calls[0])
        self.assertIn('No experiment has occurred', calls[1]['schema_correction'])


if __name__ == '__main__':
    unittest.main()
