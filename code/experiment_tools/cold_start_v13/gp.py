"""Native GP after the first paid observation; explicit no-data first round."""
from hashlib import sha256
import json

import numpy as np
from filelock import FileLock
from chem.benchmarks import knowledge_gp as native
from . import QUERIES


class ColdGP(native.GPAdapter):
    def context(self, history, query):
        if not 1 <= query <= QUERIES or len(history) != query - 1:
            raise ValueError('Cold-start history must contain exactly query-1 paid observations')
        if len({h['candidate_id'] for h in history}) != len(history):
            raise ValueError('Repeated paid observation')
        if any(h['candidate_id'] not in self.pool.ix or not np.isfinite(h['y']) for h in history):
            raise ValueError('Invalid own observation')
        if not history:
            n = len(self.pool.ids)
            ties = native.semantic_priorities(self.pool, query, 0)
            # No fitted predictions or EI exist without data/incumbent. Zeros
            # below are neutral sorting sentinels, explicitly NOT log-EI.
            return dict(stats=dict(mu_y=np.full(n, np.nan), sd_y=np.full(n, np.nan),
                sd_predictive_y=np.full(n, np.nan), fit_key='NO_DATA_NO_GP',
                fit_parameters={}, model=None, mean=None, std=None),
                base_logei=np.zeros(n), available=np.ones(n, bool),
                eligible=np.ones(n, bool), bo_index=int(np.argmin(ties)), epsilon=None,
                tie_priority=ties, acquisition_meta=dict(kind='uniform_prior_order',
                    qLogEI_defined=False, fitted_GP=False, observed_count=0),
                counters=self.counters, tie_rule='PCG64_over_sorted_physical_features',
                fit_order='none', stage='no_data', fitted_GP=False)
        prefix = [dict(candidate_id=h['candidate_id'], y=float(h['y'])) for h in history]
        lock = sha256(json.dumps(sorted(prefix, key=lambda h:h['candidate_id']),
            sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        with FileLock(str(self.directory / (lock + '.prefix.lock')), timeout=900):
            context = self._context(prefix, query)
        constant = len({h['y'] for h in prefix}) == 1
        if constant:
            self._singleton_units(context)
        context['epsilon'] = float(np.log(4) * (QUERIES - query) / (QUERIES - 1))
        context['eligible'] = context['available'] & (
            context['base_logei'][context['bo_index']] - context['base_logei'] <= context['epsilon'])
        context.update(stage='singleton_GP' if len(prefix) == 1 else 'constant_response_GP' if constant else 'native_GP', fitted_GP=True)
        context['acquisition_meta'] = dict(context['acquisition_meta'],
            qLogEI_defined=True, fitted_GP=True, observed_count=len(prefix))
        return context

    def _singleton_units(self, context):
        """A singleton or constant prefix cannot estimate a nonzero output scale.

        All native training targets are identically zero for these prefixes. Keep
        that same fitted model and its SAME qLogEI scores. Express its posterior
        using scale=1 normalized objective unit instead of an arbitrary1e-12
        empirical-scale floor. This shared, preregistered convention affects only
        reported uncertainty and GP-relative forecast calibration.
        """
        import torch
        stats = context['stats']
        model, center = stats['model'], stats['mean']
        mu, sd, obs = [], [], []
        with torch.no_grad():
            for x in self.X.split(256):
                p = model.posterior(x.unsqueeze(1))
                n = model.posterior(x.unsqueeze(1), observation_noise=True)
                mu.append(p.mean.reshape(-1))
                sd.append(p.variance.reshape(-1).clamp_min(1e-18).sqrt())
                obs.append(n.variance.reshape(-1).clamp_min(1e-18).sqrt())
        previous_scale = stats['std']
        stats.update(mu_y=torch.cat(mu).numpy() + center,
            sd_y=torch.cat(sd).numpy(), sd_predictive_y=torch.cat(obs).numpy(), std=1.)
        stats['cold_scale_convention'] = dict(native_empirical_floor=previous_scale,
            normalized_objective_scale=1., reason='singleton_or_constant_observed_values',
            native_model_and_acquisition_unchanged=True)
