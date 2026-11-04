from pathlib import Path
from unittest.mock import patch
import json
import unittest

from knowledge_domain_context import TASKS, task_context, audit_notes


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
