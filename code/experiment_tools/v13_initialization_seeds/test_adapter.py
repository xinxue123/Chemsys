from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest

from v13_initialization_seeds import study
from state_split_v13_experiment import study as frozen, transport

ROOT = Path(__file__).resolve().parents[3]


class Adapter(unittest.TestCase):
    def test_initialization_ignores_identifier_order(self):
        features = tuple((float(i), 'a' if i % 2 else 'b') for i in range(30))
        a = SimpleNamespace(ids=tuple(f'a{i}' for i in range(30)), features=features)
        b = SimpleNamespace(ids=tuple(f'z{29-i}' for i in range(30)), features=features[::-1])
        for seed in study.SEEDS:
            fa = [a.features[a.ids.index(k)] for k in study.initialization(a, seed)]
            fb = [b.features[b.ids.index(k)] for k in study.initialization(b, seed)]
            self.assertEqual(fa, fb)
        with self.assertRaises(ValueError):
            study.initialization(a, True)

    def test_all_controls_have_the_registered_initial_ids(self):
        registrations = study.registrations(ROOT)
        controls = study.controls(ROOT)
        self.assertEqual(len(registrations), 24)
        for row, ref in zip(registrations, controls):
            self.assertEqual((row['task'], row['initialization_seed']), (ref['task'], ref['initialization_seed']))
            for arm in ref['arms'].values():
                state = study.old.read(ROOT / arm['path'])
                self.assertEqual(row['initial_ids'], [r['candidate_id'] for r in state['history'][:4]])

    def test_adapter_only_redirects_inputs_paths_and_restores_bindings(self):
        task, seed = study.JOBS[0]
        p = dict(initializations=study.registrations(ROOT), controls=study.controls(ROOT))
        loader, gp, check, base, loop = frozen.gpmod.load_task, frozen.gpmod.GPAdapter, frozen.check, transport.base, frozen.run_one
        sentinel = object()
        with patch.object(study, 'NATIVE_GP', return_value=sentinel) as spy:
            with study.redirected(ROOT, task, seed, p) as (pool, envelope):
                self.assertIs(frozen.run_one, loop)
                self.assertIs(frozen.gpmod.load_task(ROOT, task, seed=0), pool)
                self.assertEqual(transport.base(ROOT), study.base(ROOT, task, seed))
                self.assertEqual(frozen.SHORT[task], 'm1_c1')
                self.assertEqual(frozen.check(ROOT), envelope)
                self.assertIs(frozen.gpmod.GPAdapter(pool, study.base(ROOT, task, seed) / 'cache' / task, seed=0), sentinel)
                spy.assert_called_once_with(pool, study.directory(ROOT) / 'cache' / task, seed=0)
                with self.assertRaises(ValueError):
                    frozen.gpmod.load_task(ROOT, task, seed=1)
        self.assertIs(frozen.gpmod.load_task, loader)
        self.assertIs(frozen.gpmod.GPAdapter, gp)
        self.assertIs(frozen.check, check)
        self.assertIs(transport.base, base)
        self.assertEqual(len({study.base(ROOT, *j) for j in study.JOBS}), 24)
        self.assertEqual(len({study.tag(*j) for j in study.JOBS}), 24)


if __name__ == '__main__':
    unittest.main()
