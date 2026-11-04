"""Input-only/synthetic checks for the independent seed0/seed1 artifact auditor."""
from copy import deepcopy
import importlib.util
import json
from math import log
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('independent_knowledge_audit', HERE / 'audit_knowledge_loop.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, allow_nan=False), encoding='utf-8')


class AuditSeedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = audit.read(ROOT / 'experiments/runs/knowledge_loop_20261101/v3/protocol.json')
        cls.registry = audit.read(ROOT / audit.SEED1_REGISTRY_PATH)
        cls.rule = audit.read(ROOT / audit.SEED1_RULE_PATH)
        cls.pools = {i['task']: dict(audit.public_pool(ROOT, i), task=i['task']) for i in cls.base['inputs']}

    def confirmation_protocol(self):
        p = deepcopy(self.base)
        p.update(variant='v3_seed1_confirm', seed=1, trial_seed=1, tie_seed=1, numerical_seed=0, menu_seed=0,
                 initial_registry_path=audit.SEED1_REGISTRY_PATH, initial_registry_sha256=audit.SEED1_REGISTRY_SHA256,
                 initial_registry_file_sha256=audit.filehash(ROOT / audit.SEED1_REGISTRY_PATH),
                 initial_rule_path=audit.SEED1_RULE_PATH, initial_rule_sha256=audit.SEED1_RULE_SHA256,
                 numerical_seed_policy=self.rule['native_gp_numerical_seed_policy'],
                 supported_modes=['bo', 'full'], supported_conditions=['agent_initial_terra'])
        p['inputs'] = []
        for row in self.registry['tasks']:
            pool = self.pools[row['task']]
            p['inputs'].append(dict(task=row['task'], group=pool['group'], candidate_count=len(pool['ids']),
                initial_ids=row['initial_ids'], initial_features_sha256=row['initial_features_sha256'],
                unit=pool['unit'], knowledge_conditions=['agent_initial_terra'],
                files_sha256={row['input_pool']: row['input_pool_sha256']}))
        p['protocol_sha256'] = audit.digest({k: v for k, v in p.items() if k != 'protocol_sha256'})
        return p

    def test_legacy_seed0_defaults_and_registered_streams(self):
        self.assertEqual(audit.trial_seeds(self.base), (0, 0))
        self.assertIsNone(audit.seed1_registration_audit(ROOT, self.base, self.pools))
        p = self.confirmation_protocol()
        self.assertEqual(audit.trial_seeds(p), (1, 1))
        for changes in ({'seed': True}, {'trial_seed': 2}, {'tie_seed': 0}, {'numerical_seed': 1},
                        {'menu_seed': 1}, {'variant': 'v3'}, {'trial_seed': 1.0}):
            with self.subTest(changes=changes), self.assertRaises(AssertionError):
                audit.trial_seeds({**p, **changes})

    def test_eight_registered_designs_are_verified_without_oracle_arrays(self):
        with patch.object(np, 'load', side_effect=AssertionError('No real outcome arrays may be read')):
            result = audit.seed1_registration_audit(ROOT, self.confirmation_protocol(), self.pools)
        self.assertEqual(result['initial_designs'], 8)
        self.assertFalse(result['targets_inspected'])

    def test_registration_or_initial_draw_cannot_be_relabelled(self):
        p = self.confirmation_protocol()
        changed = deepcopy(p)
        changed['inputs'][0]['initial_ids'].reverse()
        with self.assertRaisesRegex(AssertionError, 'initial design'):
            audit.seed1_registration_audit(ROOT, changed, self.pools)
        changed = deepcopy(p)
        changed['source_sha256']['code/chemist_system/chem/benchmarks/knowledge_engine.py'] = 'modified'
        with self.assertRaisesRegex(AssertionError, 'mechanism/prompt'):
            audit.seed1_registration_audit(ROOT, changed, self.pools)
        changed = deepcopy(p)
        changed['initial_registry_sha256'] = 'newly_recomputed_registration'
        with self.assertRaisesRegex(AssertionError, 'frozen pre-target'):
            audit.seed1_registration_audit(ROOT, changed, self.pools)
        changed_pools = deepcopy(self.pools)
        task = p['inputs'][0]['task']
        index = changed_pools[task]['ix'][p['inputs'][0]['initial_ids'][0]]
        features = list(changed_pools[task]['features'])
        features[index] = (features[index][0] + .0001, *features[index][1:])
        changed_pools[task]['features'] = features
        with self.assertRaises(AssertionError):
            audit.seed1_registration_audit(ROOT, p, changed_pools)

    def test_archived_native_numerical_seeds_stay_zero_across_all_tasks(self):
        for task, pool in self.pools.items():
            path = ROOT / 'experiments/runs/knowledge_loop_20261101/v3/runs' / (task + '_bo_agent_initial_terra') / 'rounds/01_commit.json'
            commit = audit.read(path)
            audit.native_seed_audit(pool, commit, commit['GP_acquisition_meta']['fit_seed'])
            changed = deepcopy(commit)
            changed['GP_acquisition_meta']['fit_seed'] += 100
            with self.subTest(task=task), self.assertRaisesRegex(AssertionError, 'numerical GP/MC seed'):
                audit.native_seed_audit(pool, changed)

    def synthetic_run(self, directory, seed):
        p = deepcopy(self.base)
        p.update(variant='v3_seed1_confirm' if seed else 'v3', seed=seed, trial_seed=seed,
                 tie_seed=seed, numerical_seed=0, menu_seed=0, supported_modes=['bo', 'full'],
                 supported_conditions=['agent_initial_terra'], protocol_sha256='synthetic_no_real_trial')
        if seed:
            p['initial_registry_sha256'] = audit.SEED1_REGISTRY_SHA256
        ids = [f'input_{j}' for j in range(20)]
        features = [(j / 19.,) for j in range(20)]
        pool = dict(task='Synthetic', group='P2', ids=ids, ix={v: i for i, v in enumerate(ids)},
                    axes=['x'], kinds=['continuous'], points=[dict(x=x[0]) for x in features],
                    features=features, scale=1., unit='synthetic unit', raw={})
        manifest = dict(task=pool['task'], group='P2', candidate_count=20, initial_ids=ids[:4],
                        files_sha256={'synthetic/pool.json': 'synthetic_input_sha'})
        prefix = [dict(candidate_id=i, y=0., raw_value=0.) for i in ids[:4]]
        ties = np.random.default_rng(np.random.SeedSequence([seed, 1])).permutation(20)
        available = np.array([False] * 4 + [True] * 16)
        candidates = np.flatnonzero(available)
        chosen = int(candidates[np.argmin(ties[candidates])])
        selected = ids[chosen]
        cp = dict(query=1, history_sha256=audit.digest(prefix), selected_id=selected, BO_reference_id=selected,
            mode='base_bo', base_logei=0., reference_logei=0., combined_logei=0., opportunity_gap=0.,
            epsilon=log(4), eligible_count=16, coverage=[], reviews=[], hypotheses_at_commit=[], shadows=[],
            proposal=None, menu_ids=[], verification=None, GP_fit_key='synthetic_fit',
            GP_acquisition_meta={'fit_key': 'synthetic_fit'}, deployed_versions=[],
            protocol_sha256=p['protocol_sha256'], calls_at_commit=[], pre_call_coverage_events=[])
        decision = dict(query=1, selected_id=selected, BO_reference_id=selected, mode='base_bo', evidence=[],
                        raw_value=0., y=0., incumbent_before_y=0., incumbent_after_y=0., commit_sha256=audit.digest(cp))
        state = dict(task='Synthetic', mode='bo', condition='agent_initial_terra', seed=seed,
                     history=prefix + [dict(candidate_id=selected, y=0., raw_value=0.)], hypotheses=[], calls=[],
                     decisions=[decision], verification_used=0, status='RUNNING', protocol_sha256=p['protocol_sha256'],
                     cache_counters={'operator_compiles_in_loop': 0})
        path = directory / 'state.json'
        if seed:
            commit = dict(task='Synthetic', mode='bo', condition='agent_initial_terra', seed=seed,
                protocol_sha256=p['protocol_sha256'], initial_registry_sha256=p['initial_registry_sha256'],
                initial_ids=ids[:4], initial_points=pool['points'][:4], initial_features=[list(x) for x in features[:4]],
                input_pool_sha256=manifest['files_sha256']['synthetic/pool.json'])
            commit['commit_sha256'] = audit.digest(commit)
            state['initial_commit_sha256'] = commit['commit_sha256']
            write(directory / 'initial_commit.json', commit)
        write(path, state)
        write(directory / 'rounds/01_commit.json', cp)
        write(directory / 'rounds/01_fit.json', {'synthetic_constant': 0.})
        np.savez_compressed(directory / 'rounds/01_GP.npz', mu_y=np.zeros(20), sd_y=np.ones(20) * .1,
                            sd_predictive_y=np.ones(20) * .2, base_logei=np.zeros(20),
                            available=available, eligible=available, tie_priority=ties)
        return p, pool, manifest, state, path

    def test_synthetic_seed0_and_seed1_exact_ties_use_registered_stream(self):
        with tempfile.TemporaryDirectory(prefix='knowledge_audit_seed_', dir=ROOT / 'runtime') as temp:
            for seed in (0, 1):
                with self.subTest(seed=seed):
                    p, pool, manifest, state, path = self.synthetic_run(Path(temp) / str(seed), seed)
                    row = audit.audit_run(Path(temp), path, p, pool, manifest, include_incomplete=True)
                    self.assertEqual((row['trial_seed'], row['tie_seed'], row['numerical_seed']), (seed, seed, 0))
                    self.assertEqual(row['technical_checks'], 'PASS')
            with np.load(path.parent / 'rounds/01_GP.npz') as saved:
                arrays = {k: saved[k] for k in saved.files}
            arrays['tie_priority'] = np.random.default_rng(np.random.SeedSequence([0, 1])).permutation(20)
            np.savez_compressed(path.parent / 'rounds/01_GP.npz', **arrays)
            with self.assertRaisesRegex(AssertionError, 'tie priority'):
                audit.audit_run(Path(temp), path, p, pool, manifest, include_incomplete=True)

    def test_initial_commit_rejects_extra_target_fields_even_with_new_hash(self):
        with tempfile.TemporaryDirectory(prefix='knowledge_audit_commit_', dir=ROOT / 'runtime') as temp:
            p, pool, manifest, state, path = self.synthetic_run(Path(temp), 1)
            audit.initial_commit_audit(path, state, p, pool, manifest)
            commit = audit.read(path.parent / 'initial_commit.json')
            commit['y'] = [0.] * 4
            commit['commit_sha256'] = audit.digest({k: v for k, v in commit.items() if k != 'commit_sha256'})
            state['initial_commit_sha256'] = commit['commit_sha256']
            write(path.parent / 'initial_commit.json', commit)
            with self.assertRaisesRegex(AssertionError, 'non-input data'):
                audit.initial_commit_audit(path, state, p, pool, manifest)

    def test_initial_outcome_order_and_seed_cannot_be_changed_on_resume(self):
        with tempfile.TemporaryDirectory(prefix='knowledge_audit_resume_', dir=ROOT / 'runtime') as temp:
            p, pool, manifest, state, path = self.synthetic_run(Path(temp), 1)
            state['history'][:4] = reversed(state['history'][:4])
            with self.assertRaisesRegex(AssertionError, 'committed points'):
                audit.initial_commit_audit(path, state, p, pool, manifest)
            state['seed'] = 0
            write(path, state)
            with self.assertRaisesRegex(AssertionError, 'seed/budget'):
                audit.audit_run(Path(temp), path, p, pool, manifest, include_incomplete=True)


if __name__ == '__main__':
    unittest.main()
