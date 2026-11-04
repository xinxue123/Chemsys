"""Staged P0 prior tests: input-only real fixtures, never read an oracle."""
from copy import deepcopy
from pathlib import Path
import json
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'code/chemist_system'))
from chem.benchmarks import knowledge_gp as gp
from chem.benchmarks import knowledge_policy as policy
from chem.benchmarks import knowledge_priors as prior


def semantic_pairs(hypotheses):
    return [(h['plus_features'], h['minus_features'], h['focus']) for h in hypotheses]


class KnowledgePriorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pools = {case: gp.load_task(ROOT, f'Reizman_Suzuki_Case_{case}') for case in range(1, 5)}
        cls.results = {}
        original_read_bytes = Path.read_bytes
        allowed = {pool.source_path.with_name('advice.json').resolve() for pool in cls.pools.values()}
        def read_only_advice(path):
            if path.resolve() not in allowed:
                raise AssertionError('Adapter attempted a non-advice file read')
            return original_read_bytes(path)
        with patch.object(np, 'load', side_effect=AssertionError('Oracle/array loading prohibited')), \
                patch.object(Path, 'read_bytes', read_only_advice), \
                patch.object(Path, 'read_text', side_effect=AssertionError('Additional text loading prohibited')), \
                patch.object(gp, 'read_observation', side_effect=AssertionError('Observation reads prohibited')):
            for case, pool in cls.pools.items():
                for condition in prior.CONDITIONS:
                    cls.results[case, condition] = prior.build_priors(pool, condition, pool.initial_ids)

    def test_all_twelve_case_conditions_preserve_four_clauses(self):
        self.assertEqual(len(self.results), 12)
        for (case, condition), (hypotheses, receipt) in self.results.items():
            with self.subTest(case=case, condition=condition):
                self.assertEqual(len(hypotheses), 4)
                self.assertEqual(len(receipt['clauses']), 4)
                self.assertEqual(receipt['created_hypotheses'], 4)
                self.assertEqual(receipt['uninstantiable_clauses'], [])
                self.assertEqual(receipt['status'], 'PASS')
                self.assertEqual(len(receipt['source_advice_sha256']), 64)
                self.assertEqual(len(receipt['selected_prior_sha256']), 64)
                self.assertEqual({h['origin_clause_id'] for h in hypotheses}, {'H1', 'H2', 'H3', 'H4'})
                for h in hypotheses:
                    self.assertEqual(h['origin'], 'synthetic_pressure')
                    self.assertEqual(h['ledger']['score'], 0.)
                    self.assertEqual(h['ledger']['events'], [])
                    self.assertEqual(policy.weight(h), 0.)

    def test_actual_membership_geometry_and_unqueried_pairs(self):
        for (case, condition), (hypotheses, receipt) in self.results.items():
            pool = self.pools[case]
            for h, record in zip(hypotheses, receipt['clauses']):
                coordinate = record['source_clause']['coordinate']
                lo, hi = record['source_clause']['interval']
                a, b = h['plus_features'], h['minus_features']
                self.assertNotIn(h['plus_id'], pool.initial_ids)
                self.assertNotIn(h['minus_id'], pool.initial_ids)
                if pool.kinds[coordinate] == 'continuous':
                    self.assertTrue(lo <= a[coordinate] <= hi)
                    self.assertFalse(lo <= b[coordinate] <= hi)
                    self.assertEqual(h['focus'][pool.axes[coordinate]]['bounds'], [0., 1.])
                else:
                    allowed = record['decoded_allowed_labels']
                    self.assertIn(a[coordinate], allowed)
                    self.assertNotIn(b[coordinate], allowed)
                    self.assertIsNone(record['geometry']['continuous_focus_distance'])
                for j, kind in enumerate(pool.kinds):
                    if j == coordinate:
                        continue
                    if kind == 'categorical':
                        self.assertEqual(a[j], b[j])
                    else:
                        self.assertLessEqual(abs(a[j] - b[j]), .25 + 1e-12)

    def test_wrong_category_decodes_to_different_real_labels(self):
        for case in self.pools:
            records = {}
            for condition in ('correct', 'wrong_category'):
                receipt = self.results[case, condition][1]
                records[condition] = next(r for r in receipt['clauses'] if r['kind'] == 'categorical')
            correct = set(records['correct']['decoded_allowed_labels'])
            wrong = set(records['wrong_category']['decoded_allowed_labels'])
            self.assertTrue(correct)
            self.assertTrue(wrong)
            self.assertTrue(correct.isdisjoint(wrong), (case, correct, wrong))
            self.assertTrue(all(isinstance(label, str) for label in correct | wrong))

    def test_condition_labels_do_not_enter_hypotheses(self):
        for hypotheses, receipt in self.results.values():
            public = json.dumps(hypotheses)
            for label in ('correct', 'wrong_category', 'mixed'):
                self.assertNotIn(label, public)
            self.assertNotIn('source_advice_path', public)
            self.assertIn(receipt['condition'], prior.CONDITIONS)

    def test_candidate_renaming_and_input_permutation_do_not_change_pairs(self):
        pool = deepcopy(self.pools[1])
        baseline = self.results[1, 'correct'][0]
        old_observed = set(pool.initial_ids)
        permutation = np.random.default_rng(114).permutation(len(pool.ids))
        renamed = tuple(f'opaque_{j:06d}' for j in range(len(pool.ids)))
        pool.initial_ids = tuple(renamed[new] for new, old in enumerate(permutation) if pool.ids[int(old)] in old_observed)
        pool.features = tuple(pool.features[int(j)] for j in permutation)
        pool.points = tuple(pool.points[int(j)] for j in permutation)
        pool.raw_pool['z'] = [pool.raw_pool['z'][int(j)] for j in permutation]
        pool.ids = renamed
        pool.ix = {key: j for j, key in enumerate(renamed)}
        hypotheses, unused = prior.build_priors(pool, 'correct', pool.initial_ids)
        self.assertEqual(semantic_pairs(hypotheses), semantic_pairs(baseline))

    def test_category_code_permutation_preserving_membership_has_no_metric_effect(self):
        pool = deepcopy(self.pools[1])
        baseline, receipt = self.results[1, 'correct']
        category_record = next(r for r in receipt['clauses'] if r['kind'] == 'categorical')
        coordinate = category_record['source_clause']['coordinate']
        lo, hi = category_record['source_clause']['interval']
        codes = sorted({row[coordinate] for row in pool.raw_pool['z']})
        inside = [code for code in codes if lo <= code <= hi]
        outside = [code for code in codes if code not in inside]
        rng = np.random.default_rng(51)
        replacement = dict(zip(inside, rng.permutation(inside).tolist()))
        replacement.update(zip(outside, rng.permutation(outside).tolist()))
        for row in pool.raw_pool['z']:
            row[coordinate] = replacement[row[coordinate]]
        hypotheses, changed = prior.build_priors(pool, 'correct', pool.initial_ids)
        self.assertEqual(semantic_pairs(hypotheses), semantic_pairs(baseline))
        self.assertFalse(changed['priority']['uses_category_code_distance'])

    def test_uninstantiable_clauses_raise_instead_of_silently_disappearing(self):
        pool = self.pools[1]
        with self.assertRaises(prior.PriorInstantiationError) as caught:
            prior.build_priors(pool, 'correct', pool.ids)
        receipt = caught.exception.receipt
        self.assertEqual(len(receipt['uninstantiable_clauses']), 4)
        self.assertNotEqual(receipt.get('status'), 'PASS')


if __name__ == '__main__':
    unittest.main()
