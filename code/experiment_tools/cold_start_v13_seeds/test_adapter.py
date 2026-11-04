"""Meaningful extension checks: starting randomness, restoration and pairing."""
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

from . import study
from .results import aggregate


class AdapterTests(unittest.TestCase):
    def test_first_prior_is_feature_based_and_later_stream_unchanged(self):
        pool = SimpleNamespace(ids=tuple('p'+str(i) for i in range(30)),
                               features=tuple((float(i),) for i in range(30)))
        firsts = []
        for seed in study.SEEDS:
            rng = study.priorities(seed)
            np.testing.assert_array_equal(rng(pool,1),study.NATIVE_PRIORITIES(pool,1,seed))
            np.testing.assert_array_equal(rng(pool,8),study.NATIVE_PRIORITIES(pool,8,0))
            firsts.append(tuple(np.argsort(rng(pool,1))[:5]))
        self.assertEqual(len(set(firsts)),3)
        reverse = SimpleNamespace(ids=tuple(reversed(pool.ids)),features=tuple(reversed(pool.features)))
        for seed in study.SEEDS:
            a = {p:rank for p,rank in zip(pool.features,study.priorities(seed)(pool,1))}
            b = {p:rank for p,rank in zip(reverse.features,study.priorities(seed)(reverse,1))}
            self.assertEqual(a,b)
        with self.assertRaises(ValueError):
            study.priorities(1)(pool,2,seed=1)

    def test_scoped_adapter_restores_frozen_functions(self):
        root = Path(__file__).resolve().parents[3]
        original = (study.frozen.check,study.frozen.directory,study.transport.base,
                    study.frozen.native_gp.semantic_priorities,study.frozen.initial_state)
        p = dict(protocol_sha256='synthetic')
        for seed in study.SEEDS:
            with study.redirected(root,seed,p):
                self.assertEqual(study.transport.base(root),study.base(root,seed))
                state = study.frozen.initial_state(next(iter(study.SHORT)),'v13',p,dict(commit_sha256='zero'))
                self.assertEqual((state['seed'],state['starting_seed'],state['numerical_seed']),(seed,seed,0))
                self.assertEqual(state['history'],[])
            self.assertEqual(original,(study.frozen.check,study.frozen.directory,study.transport.base,
                study.frozen.native_gp.semantic_priorities,study.frozen.initial_state))

    def test_pairing_and_sample_sd_are_per_task(self):
        rows = []
        for seed in range(4):
            for arm,delta in (('bo',0),('v13',seed-1),('reasoning_bo',1)):
                rows.append(dict(task='c1',seed=seed,method=arm,SearchAUC=10+seed+delta,
                                 final_best=20+delta,first_final_best_query=5))
        paired,summary = aggregate(rows)
        v = next(r for r in summary if r['method']=='v13')
        self.assertEqual((v['wins_vs_BO'],v['ties_vs_BO'],v['losses_vs_BO']),(2,1,1))
        self.assertAlmostEqual(v['paired_delta_sample_sd'],np.std([-1,0,1,2],ddof=1))
        self.assertEqual(len(paired),4)
        self.assertEqual(len(summary),3)
        with self.assertRaises(ValueError):
            aggregate(rows+[rows[0]])


if __name__=='__main__':
    unittest.main()
