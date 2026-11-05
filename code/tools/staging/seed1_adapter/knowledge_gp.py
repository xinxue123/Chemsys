"""Input-only task adapters and the archived GP recipes for knowledge-loop studies.

The only oracle reader is ``read_observation``. GP/acquisition implementations
are inherited unchanged. Exact acquisition ties use a new, shared feature-keyed
PCG64 ordering, so historical BO trajectories remain references, not matched
controls. P0/P1 fitting retains its historical candidate-ID sort for numerical
recipe compatibility; IDs are never GP covariates or tie priorities.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from unittest.mock import patch
import json
import re

import numpy as np


TASKS = tuple(f'Reizman_Suzuki_Case_{i}' for i in range(1, 5)) + (
    'Perera_Suzuki', 'Doyle_Buchwald_Hartwig', 'SnAr', 'HPLC')
P1_INITIAL = {
    'Perera_Suzuki': ('p03480', 'p00190', 'p01263', 'p02125'),
    'Doyle_Buchwald_Hartwig': ('p03885', 'p01465', 'p00497', 'p02103'),
}
_WARMED = False
COMPILE_COUNTS = {}
SEED1_REGISTRY_PATH = 'experiments/reports/knowledge_loop_20261101/v3_seed1_input_registration.json'
SEED1_REGISTRY_SHA256 = '97235716a5a0bb9157b0c1f4da723af4145534a18438f8f3872513cdfc736937'
SEED1_RULE_PATH = 'experiments/reports/knowledge_loop_20261101/v3_seed1_rule_registration.json'
SEED1_RULE_SHA256 = '6897a3d08ad9842a2ad36487fd2c667c4037ec5a407e5873a9a17380b668b59b'


def input_digest(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                            separators=(',', ':')).encode('utf-8')).hexdigest()


def validate_trial_seed(seed):
    if type(seed) is not int or seed not in (0, 1):
        raise ValueError('Only archived seed0 and preregistered seed1 are supported')
    return seed


def seed1_registration(root):
    """Read the two immutable input-only registrations; never sample new points."""
    root = Path(root).resolve()
    values = []
    for rel, expected in ((SEED1_REGISTRY_PATH, SEED1_REGISTRY_SHA256),
                          (SEED1_RULE_PATH, SEED1_RULE_SHA256)):
        path = (root / rel).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Input registration escapes workspace')
        value = json.loads(path.read_text(encoding='utf-8'))
        actual = input_digest({k:v for k,v in value.items() if k != 'registration_sha256'})
        if value.get('registration_sha256') != expected or actual != expected:
            raise ValueError('Registered input/rule content changed')
        if value['initial_y_read'] is not False or value['oracle_access'] is not False:
            raise ValueError('Registration must precede target access')
        values.append(value)
    registry, rule = values
    if registry['rule_registration_sha256'] != SEED1_RULE_SHA256 or rule['tasks'] != list(TASKS):
        raise ValueError('Wrong registered rule or task set')
    if [r['task'] for r in registry['tasks']] != list(TASKS):
        raise ValueError('Wrong registered initial task set')
    return registry, rule


def registered_initial_ids(pool):
    """Validate registered input identities/features without reading any outcome."""
    registry, _ = seed1_registration(pool.root)
    row = next(r for r in registry['tasks'] if r['task'] == pool.task)
    if (pool.source_path.relative_to(pool.root).as_posix() != row['input_pool'] or
            sha256(pool.source_path.read_bytes()).hexdigest() != row['input_pool_sha256']):
        raise ValueError('Registered candidate input file changed')
    if (len(pool.ids) != row['candidate_count'] or list(pool.axes) != row['axes'] or
            list(pool.kinds) != row['kinds']):
        raise ValueError('Registered input schema changed')
    ids = tuple(row['initial_ids'])
    if len(ids) != 4 or len(set(ids)) != 4 or not set(ids) <= pool.ix.keys():
        raise ValueError('Invalid registered initial IDs')
    features = [list(pool.features[pool.ix[key]]) for key in ids]
    points = [pool.points[pool.ix[key]] for key in ids]
    if (features != row['initial_features'] or points != row['initial_points'] or
            input_digest(features) != row['initial_features_sha256'] or
            input_digest([list(x) for x in sorted(pool.features)]) != row['sorted_pool_features_sha256']):
        raise ValueError('Registered initial physical features changed')
    return ids


@dataclass
class Pool:
    task: str
    group: str
    ids: tuple[str, ...]
    axes: tuple[str, ...]
    kinds: tuple[str, ...]
    points: tuple[dict, ...]
    features: tuple[tuple, ...]
    initial_ids: tuple[str, ...]
    unit: str
    scale: float
    raw_pool: dict
    root: Path
    source_path: Path
    ix: dict[str, int] = field(init=False)

    def __post_init__(self):
        self.ix = {k: i for i, k in enumerate(self.ids)}
        if len(self.ix) != len(self.ids) or len(self.features) != len(self.ids):
            raise ValueError('Candidate IDs and feature rows must be unique and aligned')
        if len(self.initial_ids) != 4 or len(set(self.initial_ids)) != 4 or not set(self.initial_ids) <= self.ix.keys():
            raise ValueError('Expected the four archived input-only initial candidates')
        if any(len(row) != len(self.axes) for row in self.features):
            raise ValueError('Feature axes are misaligned')


def load_task(root, task, *, seed=0):
    """Read only the public pool; neither oracle nor archived results are read."""
    validate_trial_seed(seed)
    root = Path(root).resolve()
    match = re.fullmatch(r'(?:Reizman_Suzuki_Case_|Reizman_Suzuki_Case|Suzuki_Case|case)([1-4])', task)
    if match:
        case = int(match.group(1))
        task = f'Reizman_Suzuki_Case_{case}'
        path = root / f'experiments/runs/suzuki_guarded/inputs/case{case}/pool.json'
        p = json.loads(path.read_text(encoding='utf-8'))
        axes = ('t_res', 'temperature', 'catalyst_loading', 'catalyst')
        features = tuple(tuple(float(x) for x in z[:3]) + (point['catalyst'],)
                         for z, point in zip(p['z'], p['points']))
        kinds = ('continuous', 'continuous', 'continuous', 'categorical')
        initial = tuple(p['initial_ids']['0'])
        group, unit, scale = 'P0', 'simulated yield percent', 100.
    elif task in P1_INITIAL:
        path = root / f'experiments/datasets/reasoning_bo/{task}/pool.json'
        p = json.loads(path.read_text(encoding='utf-8'))
        axes = tuple(p['axes'])
        kinds = ('categorical',) * len(axes)
        features = tuple(tuple(point[a] for a in axes) for point in p['points'])
        initial = P1_INITIAL[task]
        group, unit, scale = 'P1', p['objective_unit'], 100.
    elif task in ('SnAr', 'HPLC'):
        path = root / f'experiments/datasets/continuous_20261029/{task}/pool.json'
        p = json.loads(path.read_text(encoding='utf-8'))
        axes = tuple(p['axes'])
        kinds = ('continuous',) * len(axes)
        features = tuple(tuple(float(x) for x in row) for row in p['z'])
        initial = tuple(p['initial_ids'])
        group, unit, scale = 'P2', p['objective_unit'], float(p['objective_scale'])
    else:
        raise ValueError(f'Unknown task: {task}')
    pool = Pool(task, group, tuple(p['ids']), axes, kinds, tuple(p['points']),
                features, initial, unit, scale, p, root, path)
    return replace(pool, initial_ids=registered_initial_ids(pool)) if seed == 1 else pool


def read_observation(root, pool, candidate_id):
    """Oracle boundary: caller must durably commit the experiment first."""
    if Path(root).resolve() != pool.root:
        raise ValueError('Task root mismatch')
    j = pool.ix[candidate_id]
    with np.load(pool.source_path.with_name('oracle.npz'), allow_pickle=False) as oracle:
        y = float(oracle['values'][j])
        raw = float(oracle['raw_values'][j]) if 'raw_values' in oracle else y * pool.scale
        result = dict(candidate_id=candidate_id, y=y, raw_value=raw)
        if 'e_factor' in oracle:
            result['e_factor'] = float(oracle['e_factor'][j])
    if not np.isfinite(y) or not np.isfinite(raw):
        raise ValueError('Nonfinite observed objective')
    return result


def _legacy_import(name, root):
    """Find relocated provenance without changing any archived module or file."""
    adapter = import_module('.official_suzuki_adapter', __package__)
    project = adapter.PROJECT
    if not (project / 'research/suzuki_official/sources.json').is_file():
        project = Path(root) / 'experiments/chemist_system'
    # modules() checks vendor source bytes; VENDOR still names canonical code.
    with patch.object(adapter, 'PROJECT', project):
        return import_module('.' + name, __package__)


def warm():
    """Warm mixed/continuous kernels and MC512 once per persistent process."""
    global _WARMED
    if _WARMED:
        return
    import torch
    from botorch.models import SingleTaskGP
    from botorch.models.gp_regression_mixed import MixedSingleTaskGP
    from botorch.acquisition.logei import qLogExpectedImprovement
    from botorch.sampling.normal import SobolQMCNormalSampler
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype=torch.double)
    y = torch.tensor([[.1], [.4], [.6], [.3]], dtype=torch.double)
    with torch.no_grad():
        for model in (SingleTaskGP(x, y), MixedSingleTaskGP(x, y, cat_dims=[1]),
                      MixedSingleTaskGP(x, y, cat_dims=[0, 1])):
            model.eval()
            acq = qLogExpectedImprovement(model, best_f=.6,
                sampler=SobolQMCNormalSampler(torch.Size([512]), seed=0))
            acq(x[:2].unsqueeze(1))
    for name in ('script', 'trace', 'trace_module'):
        native = getattr(torch.jit, name)
        def counted(*args, _native=native, _name=name, **kwargs):
            COMPILE_COUNTS[_name] += 1
            return _native(*args, **kwargs)
        COMPILE_COUNTS[name] = 0
        setattr(torch.jit, name, counted)
    _WARMED = True


def semantic_priorities(pool, t, seed=0):
    """Map an independent RNG stream through physical features, never IDs."""
    order = sorted(range(len(pool.ids)), key=lambda j: pool.features[j])
    if len(set(pool.features)) != len(pool.features):
        raise ValueError('Duplicate physical points need an explicit replicate policy')
    result = np.empty(len(order), dtype=np.int64)
    result[order] = np.random.default_rng(np.random.SeedSequence([seed, t])).permutation(len(order))
    return result


class GPAdapter:
    def __init__(self, pool, cache_dir, seed=0):
        validate_trial_seed(seed)
        if seed == 1 and pool.initial_ids != registered_initial_ids(pool):
            raise ValueError('Seed1 GP requires the preregistered initial pool')
        import torch
        self.pool, self.seed = pool, seed
        self.directory = Path(cache_dir).resolve()
        if self.directory == pool.root or self.directory.is_relative_to(pool.source_path.parent):
            raise ValueError('Cache must be outside the source input archive')
        old_names = ('suzuki_guarded', 'suzuki_mixed_kalman', 'suzuki_mixed_comparison',
                     'suzuki_stable_20261029', 'hte_pointwise_seed0', 'stable_r2_20261029',
                     'continuous_20261029', 'continuous_r5_20261030')
        for name in old_names:
            old = (pool.root / 'experiments/runs' / name).resolve()
            if self.directory.is_relative_to(old):
                raise ValueError('Do not write new fits into historical runs')
        self.directory.mkdir(parents=True, exist_ok=True)
        warm()
        if pool.group == 'P0':
            module = _legacy_import('mixed_kalman_suzuki_engine', pool.root)
            self.cache = module.MixedKnowledgeCache(pool.raw_pool, self.directory,
                                                   int(pool.task[-1]), seed=0, sources=())
            self.X = self.cache.z
        elif pool.group == 'P1':
            module = _legacy_import('pointwise_hte_engine', pool.root)
            self.cache = module.Cache(pool.raw_pool, self.directory)
            self.X = self.cache.X
        else:
            module = _legacy_import('continuous_benchmark_engine', pool.root)
            # Its native __init__ creates an archival cache directory. Initialize
            # identical fields at the new path, then inherit native model/score.
            cache = module.Cache.__new__(module.Cache)
            cache.pool, cache.ids = pool.raw_pool, pool.raw_pool['ids']
            cache.ix = dict(pool.ix)
            cache.X = torch.tensor(pool.raw_pool['z'], dtype=torch.double)
            cache.directory = self.directory
            cache.memory, cache.fields = {}, {}
            cache.recipe = module.digest({'engine': module.data.filehash(module.__file__),
                'pool': module.digest(pool.raw_pool), 'botorch': '0.14.0', 'parameters': module.PARAMETERS})
            cache.counters = dict(fits=0, fit_cache_hits=0, posteriors=0, acquisitions=0)
            self.cache, self.X = cache, cache.X

    @property
    def counters(self):
        return dict(self.cache.counters, operator_compiles_in_loop=sum(COMPILE_COUNTS.values()))

    def context(self, history, t):
        if not 1 <= t <= 16 or len(history) != t + 3:
            raise ValueError('Expected four initial observations and sequential query rounds1..16')
        if len({h['candidate_id'] for h in history}) != len(history):
            raise ValueError('Repeated observed candidate')
        if any(h['candidate_id'] not in self.pool.ix or not np.isfinite(float(h['y'])) for h in history):
            raise ValueError('Invalid observed prefix')
        # P0/P1 native cache keys were written with exactly ID/y history rows.
        prefix = [dict(candidate_id=h['candidate_id'], y=float(h['y'])) for h in history]
        from filelock import FileLock
        # Model fitting is order-invariant within the inherited sort policy.
        # Lock all score/model/posterior writes together for that observed set.
        lock_key = sha256(json.dumps(sorted(prefix, key=lambda h: h['candidate_id']),
            sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()
        with FileLock(str(self.directory / (lock_key + '.prefix.lock')), timeout=900):
            return self._context(prefix, t)

    def _context(self, prefix, t):
        if self.pool.group == 'P0':
            base, meta = self.cache.score(prefix)
            fit, value = self.cache.model(prefix)
            stats = dict(self.cache.posterior_stats(prefix), **value)
            stats['fit_key'] = fit
        elif self.pool.group == 'P1':
            base, _, meta = self.cache.score(prefix)
            fit, value = self.cache.model(prefix)
            stats = dict(self.cache.stats(prefix), **value, fit_key=fit)
            stats['sd_predictive_y'] = stats['sd_observation_y']
        else:
            base, meta = self.cache.score(prefix)
            stats = dict(self.cache.model(prefix))
        stats['mean'], stats['std'] = float(stats['mean']), float(stats['std'])
        stats['fit_key'] = str(stats['fit_key'])
        model = stats['model']
        stats['fit_parameters'] = {name: parameter.detach().cpu().tolist()
            for name, parameter in model.named_parameters()}
        stats['observation_noise_internal'] = float(model.likelihood.noise.detach().reshape(()))
        base = np.asarray(base, dtype=float)
        available = ~np.isin(self.pool.ids, [h['candidate_id'] for h in prefix])
        ties = semantic_priorities(self.pool, t, self.seed)
        js = np.flatnonzero(available)
        js = js[base[js] == base[js].max()]
        bo = int(js[np.argmin(ties[js])])
        epsilon = float(np.log(4) * (16 - t) / 15)
        eligible = available & (base[bo] - base <= epsilon)
        return dict(stats=stats, base_logei=base, available=available, eligible=eligible,
                    bo_index=bo, epsilon=epsilon, tie_priority=ties, acquisition_meta=meta,
                    counters=self.counters,
                    tie_rule='exact_qLogEI_then_PCG64_mapped_through_sorted_physical_features',
                    fit_order='archived_ID_sort' if self.pool.group in ('P0', 'P1') else 'physical_feature_sort')

    def contrast(self, stats, plus_id, minus_id, indices):
        import torch
        js = np.asarray(indices, dtype=int)
        if js.ndim != 1 or np.any(js < 0) or np.any(js >= len(self.pool.ids)):
            raise ValueError('Invalid contrast candidate indices')
        plus, minus = self.pool.ix[plus_id], self.pool.ix[minus_id]
        model, scale2 = stats['model'], float(stats['std']) ** 2
        out, vd = np.empty(len(js)), None
        with torch.no_grad():
            for start in range(0, max(1, len(js)), 128):
                part = js[start:start + 128]
                x = torch.cat((self.X[part], self.X[[plus, minus]]), dim=0)
                cov = model.posterior(x).mvn.covariance_matrix.detach().cpu().numpy() * scale2
                n = len(part)
                out[start:start + n] = cov[:n, n] - cov[:n, n + 1]
                current = max(0., float(cov[n, n] + cov[n + 1, n + 1] - 2 * cov[n, n + 1]))
                if vd is not None and not np.isclose(vd, current, rtol=1e-6, atol=1e-8):
                    raise AssertionError('Contrast covariance changed with chunk')
                vd = current
        return float(stats['mu_y'][plus] - stats['mu_y'][minus]), vd, out

    def release(self):
        if self.pool.group == 'P0':
            self.cache.models.clear()
            self.cache.scores.clear()
            self.cache.posterior_memory.clear()
        else:
            self.cache.release()
