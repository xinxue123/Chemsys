"""A shared nondegenerate prior and posterior-only fantasy updates."""
from pathlib import Path
import hashlib
import json

import numpy as np
import torch
from filelock import FileLock
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior

from chem.benchmarks import knowledge_gp
from chem.benchmarks.mixed_kalman_suzuki_engine import encode_mixed
from . import PARAMETERS as P


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def make_model(x, y):
    model = MixedSingleTaskGP(
        train_X=x, train_Y=y, cat_dims=[3],
        train_Yvar=torch.full_like(y, P['noise_variance']), outcome_transform=None)
    model.mean_module.initialize(constant=P['prior_mean'])
    outer_sum, outer_product = model.covar_module.kernels
    for kernel in (outer_sum, outer_product):
        kernel.register_constraint('raw_outputscale', GreaterThan(P['minimum_outer_scale']))
        kernel.initialize(outputscale=P['prior_outer_scale'])
        kernel.register_prior(
            'outputscale_prior',
            GammaPrior(P['outer_scale_prior_concentration'], P['outer_scale_prior_rate']),
            'outputscale')
    inner_category = outer_sum.base_kernel.kernels[1]
    inner_category.initialize(outputscale=P['prior_inner_scale'])
    for kernel in (outer_sum.base_kernel.kernels[0], outer_product.base_kernel.kernels[0]):
        kernel.initialize(lengthscale=P['prior_continuous_lengthscale'])
    for kernel in (inner_category.base_kernel, outer_product.base_kernel.kernels[1]):
        kernel.initialize(lengthscale=P['prior_categorical_lengthscale'])
    return model


class SharedGP:
    def __init__(self, pool, directory):
        knowledge_gp.warm()
        self.pool = pool
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.X = torch.tensor(encode_mixed(pool.raw_pool), dtype=torch.double)
        self.counters = dict(fits=0, loaded_fits=0, acquisitions=0)

    def context(self, history, query):
        if len(history) != query-1 or not 2 <= query <= P['total_queries']:
            raise ValueError('One initial point and sequential queries are required')
        if len({o['candidate_id'] for o in history}) != len(history):
            raise ValueError('Repeated observation')
        ordered = sorted(history, key=lambda o: self.pool.features[self.pool.ix[o['candidate_id']]])
        prefix = [dict(candidate_id=o['candidate_id'], y=float(o['y'])) for o in ordered]
        key = digest(dict(history=prefix, parameters=P, source=hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest()))
        x = self.X[[self.pool.ix[o['candidate_id']] for o in ordered]]
        y = torch.tensor([[o['y']] for o in ordered], dtype=torch.double)
        seed = 20261104 + len(history)
        with FileLock(str(self.directory/(key+'.lock')), timeout=900):
            torch.manual_seed(seed)
            model = make_model(x, y)
            checkpoint = self.directory/(key+'.pt')
            if checkpoint.exists():
                saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
                if saved['key'] != key:
                    raise ValueError('Checkpoint changed')
                model.load_state_dict(saved['state'])
                self.counters['loaded_fits'] += 1
            else:
                if len(history) >= P['fit_after_observations']:
                    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
                    self.counters['fits'] += 1
                saved = dict(key=key, state=model.state_dict(), train_X=x.tolist(),
                             train_Y=y.tolist(), hyperparameters_fitted=len(history)>=4)
                temporary = checkpoint.with_suffix('.tmp')
                torch.save(saved, temporary)
                temporary.replace(checkpoint)
            model.eval()
            score_file = self.directory/(key+'.npz')
            if score_file.exists():
                with np.load(score_file, allow_pickle=False) as data:
                    logei = data['logei'].copy()
                    mu = data['mu'].copy()
                    variance = data['variance'].copy()
            else:
                sampler = SobolQMCNormalSampler(torch.Size([P['qlogei_samples']]), seed=38112000+query)
                acquisition = qLogExpectedImprovement(model, best_f=float(y.max()), sampler=sampler)
                with torch.no_grad():
                    logei = torch.cat([acquisition(part.unsqueeze(1)).reshape(-1)
                                       for part in self.X.split(256)]).numpy()
                    posterior = model.posterior(self.X)
                    mu = posterior.mean.reshape(-1).numpy()
                    variance = posterior.variance.reshape(-1).numpy()
                temporary = score_file.with_suffix('.tmp')
                with temporary.open('wb') as f:
                    np.savez_compressed(f, logei=logei, mu=mu, variance=variance)
                temporary.replace(score_file)
                self.counters['acquisitions'] += 1
        available = ~np.isin(self.pool.ids, [o['candidate_id'] for o in history])
        priorities = knowledge_gp.semantic_priorities(self.pool, query, 0)
        return dict(key=key, model=model, mu=mu, variance=variance, logei=logei,
                    available=available, priorities=priorities, incumbent=float(y.max()),
                    hyperparameters_fitted=len(history)>=4,
                    checkpoint=checkpoint, score_file=score_file)

    def conditional_cross_covariance(self, context, outer):
        model = context['model']
        train = model.train_inputs[0]
        with torch.no_grad():
            k_train = model.covar_module(train).to_dense()
            noise = torch.full((len(train),), P['noise_variance'], dtype=torch.double)
            factor = torch.linalg.cholesky(k_train + torch.diag(noise))
            k_all_train = model.covar_module(self.X, train).to_dense()
            k_train_outer = model.covar_module(train, self.X[outer]).to_dense()
            prior = model.covar_module(self.X, self.X[outer]).to_dense()
            covariance = prior-k_all_train@torch.cholesky_solve(k_train_outer, factor)
        return covariance
