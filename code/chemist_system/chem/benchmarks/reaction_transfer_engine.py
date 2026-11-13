"""Finite nominal-domain transport of the frozen regional/KF/B4 mechanisms.

All varying input axes in these two empirical tasks are categorical. Their
integer codes are identifiers only; MixedSingleTaskGP uses its all-category
Scale(CategoricalKernel) branch. No outcome table is accepted by this module.
"""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import math
import random

import numpy as np
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood

from chem.decision.domain_guard import select_candidate
from . import regional_ablation_suzuki_engine as old
from .suzuki_domain import digest

METHODS = ('Pure_BO', 'LGBO_region', 'Gate_only', 'G3', 'R2', 'LLINBO_Justify')
REGIMES = ('correct', 'wrong_category', 'mixed', 'assistant_initial')
PARAMETERS = {**deepcopy(old.PARAMETERS), 'confidence': .7, 'initial_queries': 4,
              'additional_queries': 16, 'MC_paths': 512, 'covariance_jitter': 1e-6}
new_bank, forecast, update_bank = old.new_bank, old.forecast, old.update_bank


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def labels(pool, permutation=None):
    mapping = {axis: dict(zip(pool['levels'][axis], range(len(pool['levels'][axis])))) for axis in pool['axes']}
    if permutation is not None:
        for axis in pool['axes']:
            values = permutation[axis]
            if len(values) != len(mapping[axis]) or len(set(values)) != len(values):
                raise ValueError('category codes require a bijection')
            mapping[axis] = dict(zip(pool['levels'][axis], values))
    return mapping


def encode(pool, permutation=None):
    mapping = labels(pool, permutation)
    return np.asarray([[mapping[axis][p[axis]] for axis in pool['axes']] for p in pool['points']], dtype=float)


def membership(pool, knowledge):
    return np.asarray([[p[h['axis']] in h['allowed'] for p in pool['points']] for h in knowledge['clauses']],
                      dtype=bool).reshape(len(knowledge['clauses']), len(pool['ids']))


def region(pool, knowledge, retained):
    if not retained:
        return None
    allowed = {axis: list(pool['levels'][axis]) for axis in pool['axes']}
    for h in knowledge['clauses']:
        if h['id'] in retained:
            allowed[h['axis']] = sorted(set(allowed[h['axis']]) & set(h['allowed']))
    if any(not values for values in allowed.values()):
        raise ValueError('empty category preference')
    return {'allowed': allowed, 'confidence': knowledge['confidence']}


class NominalFiniteRegion(old.common.native.LinearExponentialRegionalMeanTiltPlugAndPlay):
    """Only the domain measure changes: exact uniform finite-pool quadrature.

    All official fit_lambda_by_delta / prepare_cache / posterior bodies are
    inherited verbatim. No sigmoid or interval is applied to a category code.
    """
    def __init__(self, Xg, weights):
        self.Xg = Xg
        self.d = Xg.shape[1]
        self.dtype, self.device = Xg.dtype, Xg.device
        self.bounds = None
        self.smooth = None
        self._a = weights.to(Xg)
        self._lam = None
        self._region_lb = self._region_ub = None
        assert len(self._a) == len(Xg) and torch.isclose(self._a.sum(), torch.tensor(1., dtype=Xg.dtype))


class JitterModel(Model):
    def __init__(self, base):
        super().__init__()
        self.base = base
    @property
    def num_outputs(self):
        return self.base.num_outputs
    @property
    def batch_shape(self):
        return self.base.batch_shape
    def posterior(self, X, observation_noise=False, **kwargs):
        p = self.base.posterior(X, observation_noise=observation_noise, **kwargs)
        return GPyTorchPosterior(MultivariateNormal(p.mvn.mean, p.mvn.lazy_covariance_matrix.add_jitter(1e-6)))


class Cache:
    def __init__(self, pool, directory, permutation=None, sources=()):
        self.pool, self.ids = pool, tuple(pool['ids'])
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.mapping = labels(pool, permutation)
        self.X = torch.tensor(encode(pool, permutation), dtype=torch.double)
        self.sources = [Path(p) for p in sources]
        self.task_seed = int(digest({'task': pool['task_id']})[:7], 16) % 1000000
        self.recipe = digest({'engine': sha256(Path(__file__).read_bytes()).hexdigest(),
            'pool': digest(pool), 'mapping': self.mapping, 'GP': 'botorch0.14.MixedSingleTaskGP',
            'cat_dims': list(range(len(pool['axes']))), 'grid': 'exact_uniform_measured_pool', 'parameters': PARAMETERS})
        self.models, self.statistics, self.fields, self.tilts, self.acquisitions = {}, {}, {}, {}, {}
        self.counters = dict(fits=0, model_loads=0, acquisition_evaluations=0, field_evaluations=0, copied=0)

    def ensure(self, name):
        path = self.directory / name
        if path.exists():
            return
        for folder in self.sources:
            source = folder / name
            if source.exists():
                import shutil
                temporary = path.with_suffix(path.suffix + '.copytmp')
                shutil.copyfile(source, temporary)
                temporary.replace(path)
                self.counters['copied'] += 1
                return

    def model(self, history):
        ordered = sorted(history, key=lambda h: h['candidate_id'])
        key = digest({'history': ordered, 'recipe': self.recipe})
        if key in self.models:
            return key, self.models[key]
        ix = {key: i for i, key in enumerate(self.ids)}
        X = self.X[[ix[h['candidate_id']] for h in ordered]]
        y = torch.tensor([[h['y']] for h in ordered], dtype=torch.double)
        mean, std = y.mean(0, keepdim=True), y.std(0, unbiased=False, keepdim=True).clamp_min(1e-12)
        ystd = (y-mean)/std
        fit_seed = 28102026 + self.task_seed*100 + len(history)-3
        seed_all(fit_seed)
        model = MixedSingleTaskGP(X, ystd, cat_dims=list(range(X.shape[1])))
        path = self.directory / (key+'.pt')
        self.ensure(path.name)
        if path.exists():
            saved = torch.load(path, map_location='cpu', weights_only=False)
            assert saved['key'] == key and saved['recipe'] == self.recipe
            assert saved['mapping'] == self.mapping and torch.equal(saved['train_X'], X)
            model.load_state_dict(saved['state'])
            self.counters['model_loads'] += 1
        else:
            fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
            temporary = path.with_suffix('.tmp')
            torch.save({'key': key, 'recipe': self.recipe, 'state': model.state_dict(),
                'train_X': X, 'train_candidate_ids': [h['candidate_id'] for h in ordered],
                'fit_seed': fit_seed, 'mapping': self.mapping, 'gp_class': 'MixedSingleTaskGP',
                'cat_dims': list(range(X.shape[1])), 'kernel': 'Scale(CategoricalKernel(ARD Hamming))'}, temporary)
            temporary.replace(path)
            self.counters['fits'] += 1
        model.eval()
        value = dict(model=model, mean=float(mean), std=float(std), best_std=float(ystd.max()), fit_seed=fit_seed)
        self.models[key] = value
        return key, value

    def stats(self, history):
        fit, v = self.model(history)
        if fit not in self.statistics:
            rows = {k: [] for k in ('mu_std', 'sd_std', 'sd_observation_std')}
            with torch.no_grad():
                for X in self.X.split(256):
                    latent = v['model'].posterior(X.unsqueeze(1))
                    obs = v['model'].posterior(X.unsqueeze(1), observation_noise=True)
                    rows['mu_std'].append(latent.mean.reshape(-1))
                    rows['sd_std'].append(latent.variance.reshape(-1).clamp_min(1e-18).sqrt())
                    rows['sd_observation_std'].append(obs.variance.reshape(-1).clamp_min(1e-18).sqrt())
            rows = {k: torch.cat(values).numpy() for k, values in rows.items()}
            rows.update(mu_y=rows['mu_std']*v['std']+v['mean'], sd_y=rows['sd_std']*v['std'],
                        sd_observation_y=rows['sd_observation_std']*v['std'])
            self.statistics[fit] = rows
        return self.statistics[fit]

    def tilt(self, history, spec):
        fit, value = self.model(history)
        key = fit, digest(spec)
        if key not in self.tilts:
            mask = np.asarray([all(p[a] in spec['allowed'][a] for a in self.pool['axes']) for p in self.pool['points']])
            if not mask.any():
                raise ValueError('no measured point witnesses this categorical region')
            weights = torch.tensor(mask/mask.sum(), dtype=torch.double)
            tilt = NominalFiniteRegion(self.X, weights)
            delta = old.common.decide.confidence_to_delta(spec['confidence'], scale=3.)
            tilt.fit_lambda_by_delta(value['model'], delta)
            tilt.prepare_cache(value['model'])
            self.tilts[key] = tilt
        return self.tilts[key], value, fit

    def field(self, history, spec):
        if spec is None:
            return np.zeros(len(self.ids)), {'key': None, 'region': None}
        tilt, value, fit = self.tilt(history, spec)
        key = digest({'field': spec, 'fit': fit, 'recipe': self.recipe})
        if key in self.fields:
            return self.fields[key]
        path = self.directory / (key+'.npz')
        self.ensure(path.name)
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                result = data['shift_std'].copy(), json.loads(str(data['metadata']))
        else:
            with torch.no_grad():
                shift = torch.cat([(tilt.posterior(value['model'], X.unsqueeze(1)).mean-
                    value['model'].posterior(X.unsqueeze(1)).mean).reshape(-1) for X in self.X.split(256)]).numpy()
            metadata = {'key': key, 'fit_key': fit, 'region': spec, 'lambda': float(tilt._lam),
                        'grid_size': len(self.ids), 'quadrature': 'exact uniform measured candidate pool'}
            temporary = path.with_suffix('.tmp')
            with temporary.open('wb') as target:
                np.savez_compressed(target, shift_std=shift, metadata=json.dumps(metadata))
            temporary.replace(path)
            result = shift, metadata
            self.counters['field_evaluations'] += 1
        self.fields[key] = result
        return result

    def score(self, history, spec=None, amplitude=0.):
        fit, value = self.model(history)
        if spec is None or amplitude == 0.:
            spec, amplitude = None, 0.
        mc = 38102026 + self.task_seed*100 + len(history)-3
        key = digest({'fit': fit, 'region': spec, 'amplitude': amplitude, 'mc_seed': mc, 'recipe': self.recipe})
        if key in self.acquisitions:
            return self.acquisitions[key]
        path = self.directory / (key+'.npz')
        self.ensure(path.name)
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                result = data['scores'].copy(), data['shift_std'].copy(), json.loads(str(data['metadata']))
        else:
            seed_all(mc)
            if spec is None:
                effective = JitterModel(value['model'])
            else:
                tilt, _, _ = self.tilt(history, spec)
                effective = old.ScaledRegionModel(value['model'], tilt, amplitude)
            acq = qLogExpectedImprovement(effective, best_f=value['best_std'],
                sampler=SobolQMCNormalSampler(torch.Size([512]), seed=mc))
            with torch.no_grad():
                scores = torch.cat([acq(X.unsqueeze(1)).reshape(-1) for X in self.X.split(256)]).numpy()
            raw, _ = self.field(history, spec)
            shift = raw*amplitude
            metadata = {'key': key, 'fit_key': fit, 'recipe': self.recipe, 'region': spec,
                'amplitude': float(amplitude), 'fit_seed': value['fit_seed'], 'mc_seed': mc,
                'MC_paths': 512, 'covariance_jitter': 1e-6}
            assert np.isfinite(scores).all()
            temporary = path.with_suffix('.tmp')
            with temporary.open('wb') as target:
                np.savez_compressed(target, scores=scores, shift_std=shift, metadata=json.dumps(metadata))
            temporary.replace(path)
            result = scores, shift, metadata
            self.counters['acquisition_evaluations'] += 1
        self.acquisitions[key] = result
        return result


def argmax(score, mask, ids):
    return min(np.flatnonzero(mask), key=lambda i: (-float(score[i]), ids[i]))


def setup(cache, state, iteration):
    pool, knowledge, history = cache.pool, state['knowledge'], state['history']
    base, _, bmeta = cache.score(history)
    available = ~np.isin(pool['ids'], [r['candidate_id'] for r in history])
    reference = argmax(base, available, pool['ids'])
    budget = float(math.log(4)*(16-iteration)/15)
    eligible = available & ((base[reference]-base) <= budget)
    diagnostic_knowledge = knowledge if state['method'] in ('G3', 'R2') else {'clauses': []}
    diagnosis = select_candidate(base, membership(pool, diagnostic_knowledge), budget,
        candidate_ids=pool['ids'], hypothesis_ids=[h['id'] for h in diagnostic_knowledge['clauses']], queried_mask=~available)
    if state['method'] in ('Pure_BO', 'LLINBO_Justify'):
        retained = []
    elif state['method'] in ('G3', 'R2'):
        retained = list(diagnosis.retained_ids)
    else:
        retained = sorted(h['id'] for h in knowledge['clauses'])
    spec = region(pool, knowledge, retained)
    raw, fmeta = cache.field(history, spec)
    key = digest(spec) if spec is not None else None
    bank = prior = None
    amplitude = 1.
    if state['method'] in ('G3', 'R2') and key:
        bank = deepcopy(state['banks'].get(key, new_bank(iteration)))
        prior = forecast(bank, iteration)
        amplitude = prior['amplitude']
    fallback = state['method'] == 'R2' and (bool(diagnosis.revoked_ids) or diagnosis.conflict_kind != 'none')
    if fallback:
        amplitude = 0.
    scores, shift, metadata = cache.score(history, spec, amplitude)
    allowed = available if state['method'] == 'LGBO_region' else eligible
    selected = argmax(scores, allowed, pool['ids'])
    assert not fallback or selected == reference
    return {'base': base, 'score': scores, 'available': available, 'eligible': eligible, 'allowed': allowed,
        'reference': reference, 'selected': selected, 'budget': budget, 'diagnosis': diagnosis,
        'retained': retained, 'region': spec, 'raw_shift_std': raw, 'shift_std': shift,
        'amplitude': amplitude, 'fallback': fallback, 'bank_key': key, 'bank': bank, 'prior': prior,
        'stats': cache.stats(history), 'base_meta': bmeta, 'field_meta': fmeta, 'guided_meta': metadata}


def event_reasons(state, decision, iteration):
    return old.event_reasons(state, decision, iteration)


def project_proposal(point, pool, history):
    if not isinstance(point, dict) or set(point) != set(pool['axes']):
        raise ValueError('proposal must contain exactly the named nominal axes')
    if any(not isinstance(point[a], str) or point[a] not in pool['levels'][a] for a in pool['axes']):
        raise ValueError('unknown nominal proposal level')
    queried = {h['candidate_id'] for h in history}
    indices = [i for i, key in enumerate(pool['ids']) if key not in queried]
    chosen = min(indices, key=lambda i: (sum(pool['points'][i][a] != point[a] for a in pool['axes']), pool['ids'][i]))
    distance = sum(pool['points'][chosen][a] != point[a] for a in pool['axes'])
    return {'candidate_id': pool['ids'][chosen], 'point': pool['points'][chosen],
            'raw_point': point, 'Hamming_distance': distance}
