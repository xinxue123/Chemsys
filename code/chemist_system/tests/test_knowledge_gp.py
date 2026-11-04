"""Input isolation and optional real-GP checks using only each task's initial4."""
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import json
import os
import tempfile
import unittest

import numpy as np
from chem.benchmarks import knowledge_gp as k

ROOT = Path(__file__).resolve().parents[3]
HAVE_DATA = (ROOT / 'experiments/datasets/continuous_20261029/SnAr/pool.json').is_file()


@unittest.skipUnless(HAVE_DATA, 'Local frozen benchmark inputs are not bundled with GitHub code')
class PublicPoolTests(unittest.TestCase):
    def test_eight_public_pools_never_read_an_oracle(self):
        with patch.object(k.np, 'load', side_effect=AssertionError('Oracle was accessed')):
            pools = [k.load_task(ROOT, name) for name in k.TASKS]
        self.assertEqual([len(p.ids) for p in pools], [4692] * 4 + [3696, 3955, 4096, 1007])
        for p in pools:
            self.assertFalse(hasattr(p, 'targets'))
            for i, kind in enumerate(p.kinds):
                if kind == 'categorical':
                    self.assertTrue(all(isinstance(row[i], str) for row in p.features))
        self.assertEqual(pools[0].task, k.load_task(ROOT, 'Suzuki_Case1').task)

    def test_ties_do_not_depend_on_ids_or_input_row_order(self):
        p = k.load_task(ROOT, 'Perera_Suzuki')
        original = k.semantic_priorities(p, 1)
        renamed = replace(p, ids=tuple('renamed_' + x for x in p.ids),
                          initial_ids=tuple('renamed_' + x for x in p.initial_ids))
        np.testing.assert_array_equal(original, k.semantic_priorities(renamed, 1))
        order = np.random.default_rng(19).permutation(len(p.ids))
        shuffled = replace(p, ids=tuple(p.ids[i] for i in order),
            points=tuple(p.points[i] for i in order), features=tuple(p.features[i] for i in order))
        np.testing.assert_array_equal(original[order], k.semantic_priorities(shuffled, 1))
        self.assertFalse(np.array_equal(original, k.semantic_priorities(p, 2)))

    def test_old_cache_writes_and_undeclared_seed_are_rejected(self):
        p = k.load_task(ROOT, 'HPLC')
        with self.assertRaises(ValueError):
            k.GPAdapter(p, ROOT / 'experiments/runs/continuous_20261029/cache/HPLC')
        with self.assertRaises(ValueError):
            k.GPAdapter(p, ROOT / 'runtime/unused_cache', seed=1)

    def test_full_prefix_lock_is_shared_for_reordered_history(self):
        from filelock import FileLock as NativeLock
        p = k.load_task(ROOT, 'HPLC')
        history = [dict(candidate_id=key, y=float(i) / 10) for i, key in enumerate(p.initial_ids)]
        with tempfile.TemporaryDirectory(prefix='knowledge_lock_', dir=ROOT / 'runtime') as temp:
            adapter = k.GPAdapter.__new__(k.GPAdapter)
            adapter.pool, adapter.directory = p, Path(temp)
            seen = []
            def make_lock(path, timeout):
                lock = NativeLock(path, timeout=timeout)
                seen.append(lock)
                return lock
            def inside(prefix, t):
                self.assertTrue(seen[-1].is_locked)
                return len(prefix)
            adapter._context = inside
            with patch('filelock.FileLock', side_effect=make_lock):
                self.assertEqual(adapter.context(history, 1), 4)
                self.assertEqual(adapter.context(list(reversed(history)), 1), 4)
            self.assertEqual(seen[0].lock_file, seen[1].lock_file)
            self.assertFalse(any(lock.is_locked for lock in seen))


@unittest.skipUnless(HAVE_DATA and os.environ.get('CHEM_RUN_GP_TESTS') == '1',
                     'Set CHEM_RUN_GP_TESTS=1 for the eight-task initial4 numerical audit')
class RealInitialGPTests(unittest.TestCase):
    def test_all_eight_initial_gp_contexts_and_joint_contrasts(self):
        import torch
        receipts = []
        k.warm()
        jit = dict(k.COMPILE_COUNTS)
        with tempfile.TemporaryDirectory(prefix='knowledge_gp_', dir=ROOT / 'runtime') as temp:
            for task in k.TASKS:
                with self.subTest(task=task):
                    pool = k.load_task(ROOT, task)
                    # Only the four predetermined initial experiments are revealed.
                    history = [k.read_observation(ROOT, pool, point) for point in pool.initial_ids]
                    adapter = k.GPAdapter(pool, Path(temp) / task)
                    with patch.object(k, 'read_observation', side_effect=AssertionError('New experiment')):
                        context = adapter.context(history, 1)
                    self.assertEqual(int(context['available'].sum()), len(pool.ids) - 4)
                    self.assertTrue(context['eligible'][context['bo_index']])
                    stats = context['stats']
                    for name in ('mu_y', 'sd_y', 'sd_predictive_y'):
                        self.assertEqual(stats[name].shape, (len(pool.ids),))
                        self.assertTrue(np.isfinite(stats[name]).all())
                    self.assertTrue(np.isfinite(context['base_logei']).all())
                    self.assertTrue((stats['sd_predictive_y'] >= stats['sd_y']).all())
                    indices = np.flatnonzero(context['available'])[:8]
                    plus, minus = pool.ids[indices[0]], pool.ids[indices[1]]
                    md, vd, covs = adapter.contrast(stats, plus, minus, indices)
                    m, x = stats['model'], adapter.X[indices]
                    train = m.train_inputs[0]
                    with torch.no_grad():
                        covariance = m.covar_module(train).to_dense()
                        covariance += torch.eye(len(train), dtype=torch.double) * m.likelihood.noise.reshape(())
                        cross = m.covar_module(x, train).to_dense()
                        direct = m.covar_module(x).to_dense() - cross @ torch.linalg.solve(covariance, cross.T)
                        direct *= m.outcome_transform.stdvs.reshape(()).square() * stats['std'] ** 2
                        means = m.mean_module(x) + cross @ torch.linalg.solve(covariance, m.train_targets - m.mean_module(train))
                        means = ((means * m.outcome_transform.stdvs.reshape(()) + m.outcome_transform.means.reshape(()))
                                 * stats['std'] + stats['mean'])
                    direct = direct.numpy()
                    np.testing.assert_allclose(covs, direct[:, 0] - direct[:, 1], rtol=1e-7, atol=1e-10)
                    self.assertAlmostEqual(vd, direct[0, 0] + direct[1, 1] - 2 * direct[0, 1], places=9)
                    np.testing.assert_allclose(stats['mu_y'][indices], means.numpy(), atol=1e-10)
                    self.assertAlmostEqual(md, float(means[0] - means[1]), places=9)
                    if pool.group == 'P0':
                        case = int(task[-1])
                        old_state = json.loads((ROOT / f'experiments/runs/suzuki_mixed_kalman/base/c{case}_s0/state.json').read_text(encoding='utf-8'))
                        key = old_state['decisions'][0]['base_score_key']
                        reference = ROOT / f'experiments/runs/suzuki_mixed_kalman/cache/c{case}_s0/{key}.npz'
                        with np.load(reference) as archive:
                            expected = archive['scores'].copy()
                    else:
                        folder = (ROOT / f'experiments/runs/hte_pointwise_seed0/runs/{task}_s0_Pure_BO/rounds'
                                  if pool.group == 'P1' else ROOT / f'experiments/runs/continuous_20261029/runs/{task}_Pure_BO/rounds')
                        with np.load(folder / '01_candidates.npz') as archive:
                            expected = archive['base_qLogEI'].copy()
                            for field in ('mu_y', 'sd_y'):
                                np.testing.assert_allclose(stats[field], archive[field], rtol=1e-9, atol=1e-10)
                    error = float(np.max(abs(expected - context['base_logei'])))
                    np.testing.assert_allclose(context['base_logei'], expected, rtol=1e-8, atol=1e-9)
                    fit_count = adapter.counters['fits']
                    # Repeated access is cache-backed and consumes no RNG oracle.
                    np.testing.assert_array_equal(context['base_logei'], adapter.context(history, 1)['base_logei'])
                    self.assertEqual(fit_count, adapter.counters['fits'])
                    receipts.append(dict(task=task, initial_observations=4,
                        full_candidate_qLogEI_max_error=error, fit_count=fit_count,
                        BO_reference=pool.ids[context['bo_index']], counters=adapter.counters))
                    adapter.release()
        self.assertEqual(k.COMPILE_COUNTS, jit)
        print('INITIAL_GP_AUDIT ' + json.dumps(receipts, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    unittest.main()
