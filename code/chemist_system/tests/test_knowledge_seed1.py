"""Input-only staged adapter tests: never reveal a real held-out target."""
import ast
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SNAPSHOT = ROOT / 'runtime/knowledge_loop/snapshots/v3'
from chem.benchmarks import knowledge_gp as gp
from chem.benchmarks import knowledge_study as study

HAVE_DATA = (SNAPSHOT/'code/chemist_system/chem/benchmarks/knowledge_gp.py').is_file()
if HAVE_DATA:
    spec = importlib.util.spec_from_file_location('chem.benchmarks._frozen_v3_gp',
        SNAPSHOT/'code/chemist_system/chem/benchmarks/knowledge_gp.py')
    original_gp = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = original_gp
    spec.loader.exec_module(original_gp)


class StopAfterSyntheticInitial(Exception):
    pass


@unittest.skipUnless(HAVE_DATA, 'Local frozen snapshot and benchmark inputs are not bundled with code')
class Seed1InputTests(unittest.TestCase):
    def setUp(self):
        self.oracle_guard = patch('numpy.load', side_effect=AssertionError('No real oracle arrays'))
        self.oracle_guard.start()
        self.addCleanup(self.oracle_guard.stop)

    def test_all_eight_seed0_initials_features_and_ties_remain_exact(self):
        for task in gp.TASKS:
            old, new = original_gp.load_task(ROOT, task), gp.load_task(ROOT, task, seed=0)
            self.assertEqual(old.initial_ids, new.initial_ids)
            self.assertEqual(old.raw_pool, new.raw_pool)
            self.assertEqual(old.features, new.features)
            for t in (1, 8, 16):
                np.testing.assert_array_equal(original_gp.semantic_priorities(old, t),
                                              gp.semantic_priorities(new, t, seed=0))

    def test_all_eight_seed1_match_registration_and_input_only_rule(self):
        registry, _ = gp.seed1_registration(ROOT)
        with patch.object(gp, 'read_observation', side_effect=AssertionError('No target read')):
            for row in registry['tasks']:
                old = gp.load_task(ROOT, row['task'])
                pool = gp.load_task(ROOT, row['task'], seed=1)
                self.assertEqual(pool.raw_pool, old.raw_pool)
                self.assertEqual(pool.initial_ids, tuple(row['initial_ids']))
                order = sorted(range(len(pool.ids)), key=lambda j: pool.features[j])
                positions = np.random.default_rng(np.random.SeedSequence([1, 0])).choice(len(order), 4, replace=False)
                indices = [order[int(j)] for j in positions]
                self.assertEqual(pool.initial_ids, tuple(pool.ids[j] for j in indices))
                features = [list(pool.features[j]) for j in indices]
                self.assertEqual(gp.input_digest(features), row['initial_features_sha256'])
                np.testing.assert_array_equal(gp.semantic_priorities(pool, 1, seed=1),
                                              original_gp.semantic_priorities(old, 1, seed=1))

    def test_unregistered_seed_and_modified_registry_or_features_rejected(self):
        for seed in (-1, 3, True, 1.0):
            with self.assertRaises(ValueError):
                gp.load_task(ROOT, 'HPLC', seed=seed)
        pool = gp.load_task(ROOT, 'HPLC', seed=1)
        rows = list(pool.features)
        j = pool.ix[pool.initial_ids[0]]
        rows[j] = tuple(v + .001 for v in rows[j])
        with self.assertRaises(ValueError):
            gp.registered_initial_ids(replace(pool, features=tuple(rows)))
        original_read = Path.read_text
        def changed(path, *args, **kwargs):
            text = original_read(path, *args, **kwargs)
            if path.name == 'v3_seed1_input_registration.json':
                value = json.loads(text)
                value['tasks'][0]['initial_ids'].reverse()
                value['registration_sha256'] = gp.input_digest({k:v for k,v in value.items() if k != 'registration_sha256'})
                return json.dumps(value)
            return text
        with patch.object(Path, 'read_text', changed), self.assertRaises(ValueError):
            gp.load_task(ROOT, 'HPLC', seed=1)

    def test_p0_native_seed_is_zero_for_both_trials_without_fitting(self):
        calls = []
        def constructor(raw, directory, case, seed, sources):
            calls.append((case, seed, sources))
            return SimpleNamespace(z=np.empty((len(raw['ids']), 4)))
        module = SimpleNamespace(MixedKnowledgeCache=constructor)
        with tempfile.TemporaryDirectory(prefix='seed1_native_mock_', dir=ROOT/'runtime') as temp:
            with patch.object(gp, 'warm'), patch.object(gp, '_legacy_import', return_value=module):
                for trial in (0, 1):
                    pool = gp.load_task(ROOT, 'Reizman_Suzuki_Case_1', seed=trial)
                    cache = gp.GPAdapter(pool, Path(temp)/str(trial), seed=trial)
                    self.assertEqual(cache.seed, trial)
                old_pool = gp.load_task(ROOT, 'Reizman_Suzuki_Case_1')
                with self.assertRaises(ValueError):
                    gp.GPAdapter(old_pool, Path(temp)/'wrong', seed=1)
        self.assertEqual(calls, [(1, 0, ()), (1, 0, ())])

    def test_native_numerical_functions_and_menu_seed_not_changed(self):
        old = ast.parse((SNAPSHOT/'code/chemist_system/chem/benchmarks/knowledge_gp.py').read_text(encoding='utf-8'))
        new = ast.parse((ROOT/'code/chemist_system/chem/benchmarks/knowledge_gp.py').read_text(encoding='utf-8'))
        def nodes(tree):
            found = {x.name: x for x in tree.body if isinstance(x, ast.FunctionDef)}
            adapter = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'GPAdapter')
            found.update({x.name: x for x in adapter.body if isinstance(x, ast.FunctionDef)})
            return found
        before, after = nodes(old), nodes(new)
        for name in ('warm', 'semantic_priorities', 'context', '_context', 'contrast', 'release', '_legacy_import'):
            self.assertEqual(ast.dump(before[name]), ast.dump(after[name]), name)
        tree = ast.parse((ROOT/'code/chemist_system/chem/benchmarks/knowledge_study.py').read_text(encoding='utf-8'))
        menus = [x for x in ast.walk(tree) if isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute) and x.func.attr == 'make_menu']
        self.assertGreaterEqual(len(menus), 1)
        for call in menus:
            self.assertEqual(next(k.value.value for k in call.keywords if k.arg == 'seed'), 0)

    def test_seed1_protocol_preview_is_input_only_and_not_written(self):
        # Mock writes: this is not a freeze of the final executable protocol.
        with tempfile.TemporaryDirectory(prefix='seed1_protocol_preview_', dir=ROOT/'runtime') as temp:
            with patch.object(study, 'paths', return_value=Path(temp)), patch.object(study, 'source_hashes', return_value={'test_only': 'not_a_real_freeze'}), patch.object(study.llm, 'write') as save:
                p = study.prepare(ROOT, 'v3_seed1_confirm', seed=1)
                self.assertEqual((p['seed'], p['trial_seed'], p['tie_seed'], p['numerical_seed'], p['menu_seed']), (1, 1, 1, 0, 0))
                self.assertEqual(p['initial_registry_sha256'], gp.SEED1_REGISTRY_SHA256)
                self.assertEqual(p['supported_modes'], ['bo', 'full'])
                self.assertEqual(len(p['inputs']), 8)
                self.assertTrue(all(len(x['initial_ids']) == 4 and 'historical_BO' not in x for x in p['inputs']))
                self.assertTrue(all(all(k.endswith('/pool.json') for k in x['files_sha256']) for x in p['inputs']))
                self.assertFalse((Path(temp)/'protocol.json').exists())
                save.assert_called_once()
        with self.assertRaises(ValueError):
            study.prepare(ROOT, 'v3', seed=1)
        with self.assertRaises(ValueError):
            study._trial_seed({'seed': 1, 'trial_seed': 1}, requested=0)

    def test_initial_commit_is_fsynced_before_synthetic_initial_results(self):
        pool = gp.load_task(ROOT, 'HPLC', seed=1)
        protocol = dict(seed=1, trial_seed=1, tie_seed=1, numerical_seed=0, menu_seed=0,
            supported_modes=['bo', 'full'], supported_conditions=['agent_initial_terra'],
            initial_registry_sha256=gp.SEED1_REGISTRY_SHA256, protocol_sha256='synthetic_test_only')
        with tempfile.TemporaryDirectory(prefix='seed1_order_test_', dir=ROOT/'runtime') as temp:
            directory = Path(temp)/'runs'/f'{pool.task}_bo_agent_initial_terra'
            seen = []
            real_fsync = study.os.fsync
            with patch.object(study, 'check', return_value=protocol), patch.object(study, 'paths', return_value=Path(temp)), patch.object(study.os, 'fsync', wraps=real_fsync) as sync:
                def synthetic_observation(root, actual_pool, key):
                    self.assertTrue((directory/'initial_commit.json').exists())
                    self.assertGreater(sync.call_count, 0)
                    commit = study.read(directory/'initial_commit.json')
                    self.assertEqual(commit['initial_ids'], list(pool.initial_ids))
                    self.assertFalse({'y', 'raw_value', 'history', 'targets'}.intersection(commit))
                    seen.append(key)
                    return dict(candidate_id=key, y=0., raw_value=0.)
                with patch.object(gp, 'read_observation', side_effect=synthetic_observation), patch.object(gp, 'GPAdapter', side_effect=StopAfterSyntheticInitial):
                    with self.assertRaises(StopAfterSyntheticInitial):
                        study.run_one(ROOT, 'v3_seed1_confirm', pool.task, mode='bo')
                self.assertEqual(seen, list(pool.initial_ids))
                state = study.read(directory/'state.json')
                self.assertEqual(state['seed'], 1)
                self.assertEqual(state['initial_commit_sha256'], study.read(directory/'initial_commit.json')['commit_sha256'])
                with self.assertRaises(ValueError):
                    study.initial_commit(directory, pool, {**protocol, 'protocol_sha256': 'changed'}, 'bo', 'agent_initial_terra', create=False)

    def test_real_archive_mapping_allows_input_only_prepare(self):
        directory = study.paths(ROOT, 'v3_seed1_confirm')
        self.assertNotEqual(directory, directory.resolve())
        self.assertTrue(directory.resolve().is_relative_to(ROOT))
        real_exists = Path.exists
        def preview_exists(path):
            return False if path == directory.resolve()/'protocol.json' else real_exists(path)
        with patch.object(Path, 'exists', preview_exists), patch.object(study, 'source_hashes', return_value={'test_only': 'preview'}), patch.object(study.llm, 'write') as save:
            protocol = study.prepare(ROOT, 'v3_seed1_confirm', seed=1)
            self.assertEqual(protocol['seed'], 1)
            self.assertEqual(save.call_args.args[0], directory.resolve()/'protocol.json')

    def test_external_path_escape_rejected_before_outcomes(self):
        with tempfile.TemporaryDirectory(prefix='seed1_escape_') as temp:
            with patch.object(study, 'paths', return_value=Path(temp)), self.assertRaises(ValueError):
                study.prepare(ROOT, 'v3_seed1_confirm', seed=1)

    def test_historical_snapshot_unchanged(self):
        protocol = json.loads((SNAPSHOT/'protocol.json').read_text(encoding='utf-8'))
        for rel, expected in protocol['source_sha256'].items():
            self.assertEqual(sha256((SNAPSHOT/rel).read_bytes()).hexdigest(), expected)


if __name__ == '__main__':
    unittest.main()
