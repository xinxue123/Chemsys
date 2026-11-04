from pathlib import Path
from unittest.mock import patch
import json
import unittest

from chem.benchmarks.knowledge_domain_context import TASKS, task_context, audit_notes
from chem.benchmarks.knowledge_gp import load_task
from chem.benchmarks.knowledge_llm import public_payload, SYSTEM, validate


ROOT = Path(__file__).resolve().parents[3]


class DomainContextTests(unittest.TestCase):
    def test_all_tasks_are_pure_and_have_units(self):
        with patch('builtins.open', side_effect=AssertionError('No file reads')):
            with patch.object(Path, 'read_text', side_effect=AssertionError('No file reads')):
                contexts = [task_context(t) for t in TASKS]
        self.assertEqual(len(contexts), 8)
        for c in contexts:
            self.assertEqual(c['objective']['direction'], 'maximize')
            self.assertTrue(c['source_references'])
            for axis in c['inputs']:
                if axis['kind'] == 'continuous':
                    self.assertTrue(axis['unit'])

    def test_opaque_p0_identity_is_explicit(self):
        c = task_context('Suzuki_Case1')
        catalyst = next(x for x in c['inputs'] if x['name'] == 'catalyst')
        self.assertEqual(catalyst['representation'], 'opaque label')
        self.assertEqual(len(catalyst['levels']), 8)
        self.assertFalse(c['substrate_scope']['substrates_are_optimization_variables'])
        self.assertIn('not specified', c['substrate_scope']['identity_status'])

    def test_variable_substrates_and_non_yield_objectives(self):
        for task in ('Perera_Suzuki', 'Doyle_Buchwald_Hartwig'):
            c = task_context(task)
            self.assertTrue(c['substrate_scope']['substrates_are_optimization_variables'])
            self.assertTrue(all(x['representation'] == 'SMILES' for x in c['inputs']))
        self.assertEqual(task_context('SnAr')['objective']['name'], 'space_time_yield')
        self.assertEqual(task_context('HPLC')['objective']['name'], 'peak_area')
        hplc = {x['name']: x for x in task_context('HPLC')['inputs']}
        self.assertEqual(hplc['push_speed']['unit'], 'Hz')
        self.assertEqual(hplc['wait_time']['declared_input_bounds'], [.5, 10.])
        self.assertEqual(hplc['sample_flow']['unit'], 'mL/min')
        self.assertEqual(len(audit_notes()), 3)

    def test_all_continuous_units_and_declared_ranges(self):
        expected = {
            'Reizman_Suzuki_Case_1': {
                't_res': ('s', [60., 600.]),
                'temperature': ('degC', [30., 110.]),
                'catalyst_loading': ('mol%', [.5, 2.5]),
            },
            'SnAr': {
                'tau': ('min', [.5, 2.]),
                'equiv_pldn': ('equiv', [1., 5.]),
                'conc_dfnb': ('mol/L', [.1, .5]),
                'temperature': ('degC', [30., 120.]),
            },
            'HPLC': {
                'sample_loop': ('mL', [0., .08]),
                'additional_volume': ('mL', [0., .06]),
                'tubing_volume': ('mL', [.1, .9]),
                'sample_flow': ('mL/min', [.5, 2.5]),
                'push_speed': ('Hz', [80., 150.]),
                'wait_time': ('s', [.5, 10.]),
            },
        }
        for task, axes in expected.items():
            actual = {x['name']: (x['unit'], x['declared_input_bounds'])
                      for x in task_context(task)['inputs'] if x['kind'] == 'continuous'}
            self.assertEqual(actual, axes, task)

    def test_real_eight_task_payload_metadata_has_no_private_or_outcome_fields(self):
        if not (ROOT / 'experiments/datasets').exists():
            self.skipTest('Local input archive unavailable')
        forbidden_keys = {'source_references', 'source_path', 'root', 'oracle', 'targets',
                          'history', 'best', 'knowledge_regime', 'private_path'}

        def assert_no_private_keys(value):
            if isinstance(value, dict):
                self.assertFalse(forbidden_keys.intersection(value))
                for item in value.values():
                    assert_no_private_keys(item)
            elif isinstance(value, list):
                for item in value:
                    assert_no_private_keys(item)

        with patch('numpy.load', side_effect=AssertionError('No oracle reads')):
            for task in TASKS:
                pool = load_task(ROOT, task)
                # Poison non-input fields without reading any real target/history.
                for field in ('oracle', 'targets', 'history', 'best', 'knowledge_regime',
                              'private_path', 'source_references', 'description'):
                    pool.raw_pool[field] = 'UNSEEN_PRIVATE_SENTINEL'
                payload = public_payload(pool, {'hypotheses': [], 'history': []}, 1, [], {})
                metadata = payload['domain_context']
                assert_no_private_keys(metadata)
                serialized = json.dumps(metadata)
                for forbidden in ('UNSEEN_PRIVATE_SENTINEL', str(ROOT), 'C:\\Users\\',
                                  'experiments/', 'runtime/environments/', 'code/'):
                    self.assertNotIn(forbidden, serialized, task)
                self.assertEqual(metadata['task'], task)
                self.assertEqual(metadata['candidate_count'], len(pool.ids))
                self.assertEqual([x['name'] for x in metadata['inputs']], list(pool.axes))
                self.assertEqual(payload['own_observations'], [])

    def test_prompt_allows_mechanism_priors_without_changing_action_schema(self):
        self.assertIn('General chemical and', SYSTEM)
        for evidence_kind in ('mechanistic_prior', 'observed_association', 'both'):
            self.assertIn(evidence_kind, SYSTEM)
        # Evidence prefixes are prompt guidance; existing JSON actions remain valid.
        proposal = {'actions': [{'op': 'retract', 'id': 'h1', 'plus_id': None,
                    'minus_id': None, 'focus': {}, 'reason': 'Insufficient evidence.'}],
                    'summary': 'Abstain from deployment.'}
        self.assertEqual(validate(proposal), proposal)

    def test_arbitrary_target_and_prior_fields_are_not_forwarded(self):
        p = dict(task='SnAr', ids=['only_id'], axes=['tau', 'equiv_pldn', 'conc_dfnb', 'temperature'],
                 oracle='UNSEEN_SENTINEL', targets=[12345], history='UNSEEN_SENTINEL',
                 description='UNSEEN_SENTINEL', knowledge_regime='UNSEEN_SENTINEL', best='UNSEEN_SENTINEL')
        text = json.dumps(task_context('SnAr', p))
        self.assertNotIn('UNSEEN_SENTINEL', text)
        self.assertNotIn('12345', text)
        self.assertNotIn('only_id', text)
        self.assertEqual(task_context('SnAr', p)['candidate_count'], 1)

    def test_real_public_pools_and_source_paths(self):
        if not (ROOT / 'experiments/datasets').exists():
            self.skipTest('Local input archive unavailable')
        for task in TASKS:
            if task.startswith('Reizman'):
                rel = f'experiments/runs/suzuki_guarded/inputs/case{task[-1]}/pool.json'
            elif task in ('SnAr', 'HPLC'):
                rel = f'experiments/datasets/continuous_20261029/{task}/pool.json'
            else:
                rel = f'experiments/datasets/reasoning_bo/{task}/pool.json'
            p = json.loads((ROOT / rel).read_text(encoding='utf-8'))
            c = task_context(task, p)
            self.assertEqual(c['candidate_count'], len(p['ids']))
            for source in c['source_references']:
                self.assertTrue((ROOT / source.split('::')[0]).is_file(), source)
            if task in ('Perera_Suzuki', 'Doyle_Buchwald_Hartwig'):
                self.assertTrue(all(x['levels'] == p['levels'][x['name']] for x in c['inputs']))

    def test_bad_task_or_misaligned_axes_rejected(self):
        with self.assertRaises(ValueError):
            task_context('unknown')
        with self.assertRaises(ValueError):
            task_context('SnAr', dict(task='HPLC', ids=[]))
        with self.assertRaises(ValueError):
            task_context('SnAr', dict(task='SnAr', ids=[], axes=['fake']))


if __name__ == '__main__':
    unittest.main()
