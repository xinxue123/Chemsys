"""Staged seed2 input-only / synthetic resume tests; no real target, fit or API."""
import ast
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT/'code/chemist_system'))
from chem.benchmarks import knowledge_gp as original_gp
from chem.benchmarks import knowledge_study as original_study


def staged(name):
    spec = importlib.util.spec_from_file_location('chem.benchmarks._staged_seed2_' + name, HERE/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gp, study = staged('knowledge_gp'), staged('knowledge_study')
study.gpmod = gp


class SyntheticCrash(Exception):
    pass


class Seed2InputTests(unittest.TestCase):
    def setUp(self):
        self.guard = patch('numpy.load', side_effect=AssertionError('Real arrays/targets forbidden'))
        self.guard.start();self.addCleanup(self.guard.stop)

    def test_seed0_seed1_initials_features_and_ties_unchanged(self):
        for task in gp.TASKS:
            for trial in (0, 1):
                before = original_gp.load_task(ROOT, task, seed=trial)
                after = gp.load_task(ROOT, task, seed=trial)
                self.assertEqual(before.initial_ids, after.initial_ids)
                self.assertEqual(before.features, after.features)
                self.assertEqual(before.raw_pool, after.raw_pool)
                for t in (1, 8, 16):
                    np.testing.assert_array_equal(original_gp.semantic_priorities(before, t, seed=trial),
                                                  gp.semantic_priorities(after, t, seed=trial))

    def test_seed2_real_input_only_matches_immutable_registration(self):
        if gp.SEED2_REGISTRY_SHA256.startswith('PENDING'):
            self.skipTest('Real seed2 input registration has not yet been finalized')
        registry, _ = gp.seed2_registration(ROOT)
        with patch.object(gp, 'read_observation', side_effect=AssertionError('No target reads')):
            for row in registry['tasks']:
                pool = gp.load_task(ROOT, row['task'], seed=2)
                self.assertEqual(pool.initial_ids, tuple(row['initial_ids']))
                order = sorted(range(len(pool.ids)), key=lambda j: pool.features[j])
                positions = np.random.default_rng(np.random.SeedSequence([2, 0])).choice(len(order), 4, replace=False)
                self.assertEqual(pool.initial_ids, tuple(pool.ids[order[int(j)]] for j in positions))
                priorities = np.empty(len(pool.ids), int)
                priorities[order] = np.random.default_rng(np.random.SeedSequence([2, 1])).permutation(len(order))
                np.testing.assert_array_equal(gp.semantic_priorities(pool, 1, seed=2), priorities)

    def test_unregistered_seed_rejected_and_seed2_not_silently_seed1(self):
        for seed in (-1, 3, True, 2.0):
            with self.assertRaises(ValueError):gp.validate_trial_seed(seed)
        self.assertEqual(gp.validate_trial_seed(2), 2)
        with patch.object(gp, 'SEED2_REGISTRY_SHA256', 'PENDING'), self.assertRaises(ValueError):
            gp.seed2_registration(ROOT)

    def test_malformed_registry_and_wrong_draw_rejected(self):
        if gp.SEED2_REGISTRY_SHA256.startswith('PENDING'):
            self.skipTest('Real seed2 input registration has not yet been finalized')
        pool = gp.load_task(ROOT, 'HPLC', seed=2)
        registry, rule = gp.seed2_registration(ROOT)
        changed = deepcopy(registry)
        row = next(r for r in changed['tasks'] if r['task'] == 'HPLC')
        row['initial_ids'].reverse();row['initial_points'].reverse();row['initial_features'].reverse()
        row['initial_features_sha256'] = gp.input_digest(row['initial_features'])
        with patch.object(gp, 'seed2_registration', return_value=(changed, rule)), self.assertRaises(ValueError):
            gp.registered_initial_ids(pool, seed=2)
        read = Path.read_text
        def corrupt(path, *args, **kwargs):
            text = read(path, *args, **kwargs)
            if path.name == 'v5_seed2_input_registration.json':
                value = json.loads(text);value['menu_seed'] = 2
                value['registration_sha256'] = gp.input_digest({k:v for k,v in value.items() if k!='registration_sha256'})
                return json.dumps(value)
            return text
        with patch.object(Path, 'read_text', corrupt), self.assertRaises(ValueError):gp.seed2_registration(ROOT)

    def test_native_numerical_recipe_identical_and_menu_seed_zero(self):
        def nodes(path):
            tree = ast.parse(path.read_text(encoding='utf-8'))
            found = {n.name:n for n in tree.body if isinstance(n, ast.FunctionDef)}
            adapter = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name=='GPAdapter')
            found.update({n.name:n for n in adapter.body if isinstance(n, ast.FunctionDef)})
            return found
        before, after = nodes(ROOT/'code/chemist_system/chem/benchmarks/knowledge_gp.py'), nodes(HERE/'knowledge_gp.py')
        for name in ('warm', 'semantic_priorities', 'context', '_context', 'contrast', 'release', '_legacy_import', 'read_observation'):
            self.assertEqual(ast.dump(before[name]), ast.dump(after[name]), name)
        # Constructor adaptation ends before import torch; the entire numerical
        # constructor body, including native P0 seed=0, is bytecode-AST identical.
        def numerical_body(fn):
            start = next(i for i,n in enumerate(fn.body) if isinstance(n, ast.Import) and n.names[0].name=='torch')
            return [ast.dump(n) for n in fn.body[start:]]
        self.assertEqual(numerical_body(before['__init__']), numerical_body(after['__init__']))
        tree = ast.parse((HERE/'knowledge_study.py').read_text(encoding='utf-8'))
        menus = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr=='make_menu']
        self.assertEqual(len(menus), 2)
        self.assertTrue(all(next(k.value.value for k in n.keywords if k.arg=='seed')==0 for n in menus))

    def test_dispatch_reserved_variant_and_seed_streams(self):
        self.assertIs(study.components('v5_seed2_confirm')[0], original_study.components('v5')[0])
        self.assertIs(study.components('v5_seed2_confirm')[1], original_study.components('v5')[1])
        for variant, seed in [('v5', 2), ('v5_seed2_confirm', 0), ('v3_seed1_confirm', 2)]:
            with self.assertRaises(ValueError):study.prepare(ROOT, variant, seed=seed)
        with self.assertRaises(ValueError):study._trial_seed(dict(seed=2,trial_seed=2,tie_seed=2,numerical_seed=2,menu_seed=0))
        with self.assertRaises(ValueError):study._trial_seed(dict(seed=2,trial_seed=2), requested=1)

    def test_protocol_preview_does_not_write_or_read_targets(self):
        if gp.SEED2_REGISTRY_SHA256.startswith('PENDING'):
            self.skipTest('Real seed2 input registration has not yet been finalized')
        with tempfile.TemporaryDirectory(dir=ROOT/'runtime') as temp:
            with patch.object(study, 'paths', return_value=Path(temp)), patch.object(study, 'source_hashes', return_value={'synthetic_preview':'not_frozen'}), patch.object(study.llm, 'write') as save:
                p = study.prepare(ROOT, 'v5_seed2_confirm', seed=2)
        self.assertEqual((p['seed'],p['tie_seed'],p['numerical_seed'],p['menu_seed']), (2,2,0,0))
        self.assertEqual(p['supported_modes'], ['bo','full'])
        self.assertEqual(p['parameters'], study.components('v5')[0].PARAMETERS)
        self.assertEqual(p['prompt_sha256'], study.llm.digest(study.components('v5')[1].SYSTEM))
        self.assertEqual(len(p['registered_core_source_sha256']), 9)
        self.assertEqual(len(p['inputs']), 8)
        self.assertTrue(all(all(k.endswith('/pool.json') for k in row['files_sha256']) for row in p['inputs']))
        save.assert_called_once()

    def test_p0_native_constructor_seed_remains_zero_for_seed2(self):
        pool = original_gp.load_task(ROOT, 'Reizman_Suzuki_Case_1', seed=0)
        calls = []
        def constructor(raw, directory, case, seed, sources):
            calls.append(seed)
            return SimpleNamespace(z=np.empty((len(raw['ids']),4)))
        with tempfile.TemporaryDirectory(dir=ROOT/'runtime') as temp:
            with patch.object(gp, 'registered_initial_ids', return_value=pool.initial_ids), patch.object(gp, 'warm'), patch.object(gp, '_legacy_import', return_value=SimpleNamespace(MixedKnowledgeCache=constructor)):
                result = gp.GPAdapter(pool, Path(temp)/'mockcache', seed=2)
        self.assertEqual(result.seed,2);self.assertEqual(calls,[0])

    def test_staged_source_base_hashes_unchanged(self):
        receipt = json.loads((HERE/'base_receipt.json').read_text(encoding='utf-8'))
        for rel, item in receipt['files'].items():
            self.assertEqual(sha256((ROOT/rel).read_bytes()).hexdigest(), item['sha256'])


class SyntheticResumeTests(unittest.TestCase):
    def test_commit_before_synthetic_target_and_crash_resume_no_rechoose(self):
        with tempfile.TemporaryDirectory(prefix='seed2_synthetic_',dir=ROOT/'runtime') as temp:
            temp = Path(temp)
            ids = tuple(f'synthetic_{j}' for j in range(24))
            features = tuple((j/23.,) for j in range(24))
            source = temp/'pool.json';source.write_text(json.dumps({'ids':ids,'features':features}),encoding='utf-8')
            pool = gp.Pool('HPLC','P2',ids,('x',),('continuous',),tuple({'x':x[0]} for x in features),
                features,ids[:4],'synthetic',100.,{},ROOT,source)
            protocol = dict(seed=2,trial_seed=2,tie_seed=2,numerical_seed=0,menu_seed=0,
                supported_modes=['bo','full'],supported_conditions=['agent_initial_terra'],
                protocol_sha256='synthetic_only',initial_registry_sha256='synthetic_registration')
            directory = temp/'runs/HPLC_bo_agent_initial_terra'
            context_calls, successful_reads = [], []
            crash = [True]
            class FakeGP:
                def __init__(self, pool, cache_directory, seed):
                    self.counters={'fits':0};assert seed==2
                def context(self, history, t):
                    context_calls.append(t)
                    available=np.array([key not in {r['candidate_id'] for r in history} for key in ids])
                    base=-np.arange(len(ids),dtype=float)
                    bo=int(np.flatnonzero(available)[0])
                    stats=dict(mu_y=np.zeros(len(ids)),sd_y=np.ones(len(ids))*.1,
                        sd_predictive_y=np.ones(len(ids))*.2,fit_key='synthetic',fit_parameters={})
                    return dict(available=available,eligible=available,base_logei=base,bo_index=bo,
                        tie_priority=np.arange(len(ids)),epsilon=1.,stats=stats,acquisition_meta={})
                def release(self):pass
            real_fsync = study.os.fsync
            with patch.object(study,'check',return_value=protocol), patch.object(study,'paths',return_value=temp), patch.object(gp,'load_task',return_value=pool), patch.object(gp,'GPAdapter',FakeGP), patch.object(study.api,'APIClient',side_effect=AssertionError('No API')), patch.object(study.os,'fsync',wraps=real_fsync) as sync:
                def synthetic(root, actual_pool, key):
                    self.assertTrue((directory/'initial_commit.json').exists())
                    self.assertGreater(sync.call_count,0)
                    init=study.read(directory/'initial_commit.json')
                    self.assertFalse({'y','raw_value','history','targets'}&set(init))
                    if key not in pool.initial_ids:
                        cp=directory/'rounds'/f'{len(successful_reads)-3:02d}_commit.json'
                        self.assertEqual(study.read(cp)['selected_id'],key)
                        if crash[0]:crash[0]=False;raise SyntheticCrash()
                    successful_reads.append(key)
                    return dict(candidate_id=key,y=pool.ix[key]/23.,raw_value=100*pool.ix[key]/23.)
                with patch.object(gp,'read_observation',side_effect=synthetic):
                    with self.assertRaises(SyntheticCrash):study.run_one(ROOT,'v5_seed2_confirm','HPLC',mode='bo')
                    self.assertEqual(context_calls,[1])
                    statefile=directory/'state.json';saved=study.read(statefile)
                    self.assertEqual(saved['status'],'INTERRUPTED')
                    commit_bytes=(directory/'rounds/01_commit.json').read_bytes()
                    # A missing state marker must not bypass seed2 initialization checks.
                    missing=deepcopy(saved);missing.pop('initial_commit_sha256')
                    study.llm.write(statefile,missing)
                    with self.assertRaises(ValueError):study.run_one(ROOT,'v5_seed2_confirm','HPLC',mode='bo')
                    study.llm.write(statefile,saved)
                    initial=(directory/'initial_commit.json').read_bytes()
                    (directory/'initial_commit.json').unlink()
                    with self.assertRaises(ValueError):study.run_one(ROOT,'v5_seed2_confirm','HPLC',mode='bo')
                    (directory/'initial_commit.json').write_bytes(initial)
                    result=study.run_one(ROOT,'v5_seed2_confirm','HPLC',mode='bo')
                self.assertEqual(result['status'],'COMPLETE')
                self.assertEqual(context_calls,list(range(1,17)))
                self.assertEqual(len(successful_reads),20)
                self.assertEqual(len(set(successful_reads)),20)
                self.assertEqual((directory/'rounds/01_commit.json').read_bytes(),commit_bytes)


if __name__ == '__main__':unittest.main()
