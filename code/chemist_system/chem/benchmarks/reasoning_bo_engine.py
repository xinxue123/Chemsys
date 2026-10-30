"""Common categorical GP / qLogNEI and simultaneous batch regional decisions.

This module accepts observed prefixes and a feature-only domain. It has no
oracle access. q=3 batches are committed before any of their outcomes arrive.
"""
from copy import deepcopy
from hashlib import sha256
from itertools import combinations
from pathlib import Path
import json
import math
import time
import warnings

import numpy as np
import torch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.optim.optimize import optimize_acqf_discrete
from botorch.sampling.normal import SobolQMCNormalSampler
from filelock import FileLock

from chem.decision.domain_guard import select_candidate
from . import reaction_transfer_engine as nominal
from .suzuki_domain import digest

METHODS = ('ReasoningBO', 'BO_shared', 'Gate_only', 'New_R2', 'LGBO_region', 'BO_q3_shared', 'BO_q3_Sobol')
PARAMETERS = {**deepcopy(nominal.PARAMETERS), 'initial_queries': 3, 'batches': 30,
    'batch_size': 3, 'shortlist_size': 5, 'MC_paths': 128, 'quadrature_cap': 512,
    'maximum_active_clauses': 4, 'last_refresh_round': 28, 'stagnation_rounds': 3,
    'device': 'cpu', 'torch_threads_per_process': 1, 'operator_compiles_in_loop': 0}


class BatchCache(nominal.Cache):
    def __init__(self, pool, directory, permutation=None, sources=()):
        super().__init__(pool, directory, permutation, sources)
        self.recipe = digest({'batch_source': sha256(Path(__file__).read_bytes()).hexdigest(),
            'nominal_source': sha256(Path(nominal.__file__).read_bytes()).hexdigest(),
            'pool': digest(pool), 'mapping': self.mapping, 'parameters': PARAMETERS,
            'model': 'MixedSingleTaskGP(all categorical)', 'acquisition': 'qLogNoisyExpectedImprovement'})
        self.shortlists = {}

    def model(self, history):
        key=digest({'history':sorted(history,key=lambda h:h['candidate_id']),'recipe':self.recipe})
        with FileLock(str(self.directory/(key+'.fit.lock')),timeout=300):
            return super().model(history)

    def tilt(self, history, spec):
        fit, value = self.model(history)
        key = fit, digest(spec)
        if key not in self.tilts:
            mask = np.array([all(p[a] in spec['allowed'][a] for a in self.pool['axes']) for p in self.pool['points']])
            indices = np.flatnonzero(mask)
            if not len(indices):
                raise ValueError('empty measured regional intersection')
            # Zero weights outside the nominal region can be omitted exactly.
            # Above512, fixed outcome-independent quadrature bounds matrix size.
            if len(indices) > PARAMETERS['quadrature_cap']:
                rng = np.random.default_rng(int(digest(spec)[:12],16))
                indices = np.sort(rng.choice(indices, PARAMETERS['quadrature_cap'], replace=False))
            Xg = self.X[indices]
            tilt = nominal.NominalFiniteRegion(Xg, torch.full((len(indices),), 1/len(indices), dtype=torch.double))
            delta = nominal.old.common.decide.confidence_to_delta(spec['confidence'], scale=3.)
            tilt.fit_lambda_by_delta(value['model'], delta)
            tilt.prepare_cache(value['model'])
            tilt.grid_indices = indices.tolist()
            self.tilts[key] = tilt
        return self.tilts[key], value, fit

    def acquisition(self, history, spec=None, amplitude=0.):
        fit, value = self.model(history)
        seed = 38102026+self.task_seed*100+len(history)-2
        if spec is None or amplitude == 0:
            model = nominal.JitterModel(value['model'])
        else:
            tilt, _, _ = self.tilt(history, spec)
            model = nominal.old.ScaledRegionModel(value['model'], tilt, amplitude)
        # Root caching is disabled in BOTH arms: custom tilted posterior has no
        # GPyTorchModel interface, and identical treatment is the common recipe.
        nominal.seed_all(seed)  # also fixes MC baseline pruning, not only draws below
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*prune.*')
            acq = qLogNoisyExpectedImprovement(model, X_baseline=value['model'].train_inputs[0],
                sampler=SobolQMCNormalSampler(torch.Size([PARAMETERS['MC_paths']]), seed=seed),
                prune_baseline=True, cache_root=False)
        return acq, value, {'fit_key': fit, 'mc_seed': seed, 'MC_paths': PARAMETERS['MC_paths'],
            'pruned_baseline_size': len(acq.X_baseline), 'acquisition': 'qLogNEI', 'cache_root': False}

    def shortlist(self, history, q=5):
        fit, _ = self.model(history)
        lock=digest({'fit':fit,'q':q})
        with FileLock(str(self.directory/(lock+'.shortlist.lock')),timeout=300):
            return self._shortlist(history,q)

    def _shortlist(self, history, q):
        fit, _ = self.model(history)
        if (fit,q) in self.shortlists:
            return self.shortlists[fit,q]
        key = digest({'fit': fit, 'recipe': self.recipe, 'shortlist': q})
        path = self.directory/(key+'.json')
        if path.exists():
            saved = json.loads(path.read_text(encoding='utf8'))
            assert saved['fit_key'] == fit
        else:
            available = np.array([key not in {h['candidate_id'] for h in history} for key in self.ids])
            indices = np.flatnonzero(available)
            acq, value, meta = self.acquisition(history)
            started = time.perf_counter()
            with torch.no_grad():
                X, joint = optimize_acqf_discrete(acq, q=q, choices=self.X[indices],
                    max_batch_size=64, unique=True)
            selected = []
            for x in X:
                matches = torch.all(self.X == x, dim=-1).nonzero().reshape(-1)
                assert len(matches) == 1
                selected.append(int(matches[0]))
            saved = {'key': key, **meta, 'indices': selected, 'ids': [self.ids[i] for i in selected],
                'q5_greedy_logNEI_values': joint.reshape(-1).tolist(), 'seconds': time.perf_counter()-started}
            nominal.old.common.regional.seed_all(meta['mc_seed'])
            from .reaction_transfer_data import write
            write(path, saved)
        self.shortlists[fit,q] = saved
        return saved

    def batch_scores(self, history, batches, spec=None, amplitude=0.):
        fit, _ = self.model(history)
        key = digest({'fit': fit, 'batches': batches, 'spec': spec, 'amplitude': amplitude, 'recipe': self.recipe})
        with FileLock(str(self.directory/(key+'.scores.lock')),timeout=300):
            return self._batch_scores(history,batches,spec,amplitude,key)

    def _batch_scores(self,history,batches,spec,amplitude,key):
        path = self.directory/(key+'.npz')
        if path.exists():
            with np.load(path, allow_pickle=False) as d:
                return d['scores'].copy(), json.loads(str(d['metadata']))
        acq, _, meta = self.acquisition(history, spec, amplitude)
        with torch.no_grad():
            scores = acq(self.X[torch.tensor(batches, dtype=torch.long)]).reshape(-1).numpy()
        assert np.isfinite(scores).all()
        meta.update(key=key, region=spec, amplitude=float(amplitude))
        temporary = path.with_suffix('.tmp')
        with temporary.open('wb') as handle:
            np.savez_compressed(handle, scores=scores, metadata=json.dumps(meta))
        temporary.replace(path)
        return scores, meta

    def release(self):
        self.models.clear(); self.statistics.clear(); self.fields.clear()
        self.tilts.clear(); self.acquisitions.clear(); self.shortlists.clear()


def batch_diagnosis(base, batches, pool, knowledge, budget):
    """Original exact B4 over (batch, common witness) pairs, at most4 clauses.

    All retained clauses must hold at ONE common point in an eligible batch.
    Three different points satisfying disjoint clauses do not fake conjunction.
    """
    h = nominal.membership(pool, knowledge)
    expanded = np.asarray(batches, dtype=int).reshape(-1)
    diag = select_candidate(np.repeat(base, len(batches[0])), h[:,expanded], budget,
        candidate_ids=[f'b{i:02d}_w{j}' for i in range(len(batches)) for j in range(len(batches[0]))],
        hypothesis_ids=[r['id'] for r in knowledge['clauses']])
    return diag


def setup(cache, state, iteration):
    history, knowledge, pool = state['history'], state['knowledge'], cache.pool
    native = state['method'] in ('BO_q3_shared','BO_q3_Sobol')
    shortlist = cache.shortlist(history,q=3 if native else 5)
    batches = [sorted(shortlist['indices'])] if native else [list(v) for v in combinations(sorted(shortlist['indices']),3)]
    batch_ids = ['+'.join(pool['ids'][j] for j in batch) for batch in batches]
    base, bmeta = cache.batch_scores(history,batches)
    reference = min(range(len(batches)),key=lambda j:(-float(base[j]),batch_ids[j]))
    budget = float(math.log(4)*(30-iteration)/29)
    eligible = base[reference]-base <= budget
    diagnostic = knowledge if state['method'] == 'New_R2' else {'clauses': []}
    diag = batch_diagnosis(base,batches,pool,diagnostic,budget)
    if state['method'] == 'New_R2':
        retained = list(diag.retained_ids)
    elif state['method'] in ('Gate_only','LGBO_region'):
        retained = sorted(r['id'] for r in knowledge['clauses'])
    else:
        retained = []
    spec = nominal.region(pool,knowledge,retained)
    key = digest(spec) if spec else None
    bank = prior = None
    amplitude = 1.
    if state['method'] == 'New_R2' and key:
        bank = deepcopy(state['banks'].get(key,nominal.new_bank(iteration)))
        prior = nominal.forecast(bank,iteration)
        amplitude = prior['amplitude']
    fallback = state['method'] == 'New_R2' and (bool(diag.revoked_ids) or diag.conflict_kind != 'none')
    if fallback: amplitude = 0.
    guided, gmeta = cache.batch_scores(history,batches,spec,amplitude)
    allowed = np.ones(len(batches),dtype=bool) if state['method'] == 'LGBO_region' else eligible
    selected = min(np.flatnonzero(allowed),key=lambda j:(-float(guided[j]),batch_ids[j]))
    if state['method'] in ('BO_shared','BO_q3_shared','BO_q3_Sobol','ReasoningBO'):
        assert selected == reference
    assert not fallback or selected == reference
    return {'shortlist': shortlist, 'batches': batches, 'batch_ids': batch_ids,
        'base': base, 'score': guided, 'eligible': eligible, 'allowed': allowed,
        'selected': int(selected), 'reference': reference, 'budget': budget,
        'diagnosis': diag, 'retained': retained, 'region': spec, 'amplitude': amplitude,
        'fallback': fallback, 'bank_key': key, 'bank': bank, 'prior': prior,
        'base_meta': bmeta, 'guided_meta': gmeta}


def measurement_inputs(cache, history, decision, indices):
    _, value = cache.model(history)
    with torch.no_grad():
        p = value['model'].posterior(cache.X[indices], observation_noise=True)
        means = p.mean.reshape(-1).numpy()*value['std']+value['mean']
        covariance = p.mvn.covariance_matrix.numpy()*value['std']**2
    raw = np.zeros(len(indices))
    field_meta = {'region': None}
    if decision['region']:
        tilt, _, fit = cache.tilt(history,decision['region'])
        with torch.no_grad():
            xx=cache.X[indices]
            raw = (tilt.posterior(value['model'],xx).mean-value['model'].posterior(xx).mean).reshape(-1).numpy()*value['std']
        field_meta = {'fit_key': fit, 'lambda': float(tilt._lam), 'grid_indices': tilt.grid_indices,
            'quadrature_size': len(tilt.grid_indices), 'region': decision['region']}
    return {'prequery_mu_y': means.tolist(), 'prequery_observation_covariance': covariance.tolist(),
        'raw_region_shift_y': raw.tolist(), 'field_meta': field_meta}


def joint_kf(bank, prior, iteration, measurement, y):
    raw = np.array(measurement['raw_region_shift_y'])
    H = np.column_stack([np.ones(len(raw)),raw])
    P = np.array(prior['covariance']); theta=np.array(prior['mean'])
    R = np.array(measurement['prequery_observation_covariance'])+np.eye(len(raw))*.0009
    residual = np.array(y)-np.array(measurement['prequery_mu_y'])
    innovation = residual-H@theta
    S = H@P@H.T+R
    K = np.linalg.solve(S,H@P).T
    after = theta+K@innovation
    A = np.eye(2)-K@H
    covariance = A@P@A.T+K@R@K.T
    covariance=(covariance+covariance.T)/2
    updated={'mean':after.tolist(),'covariance':covariance.tolist(),
        'last_round':iteration,'updates':bank['updates']+1}
    receipt={**deepcopy(measurement),'observed_y':list(y),'feature':H.tolist(),
        'residual':residual.tolist(),'measurement_covariance':R.tolist(),'innovation':innovation.tolist(),
        'innovation_covariance':S.tolist(),'innovation_z':float(np.sqrt(innovation@np.linalg.solve(S,innovation)/len(raw))),
        'gain':K.tolist(),'posterior':updated}
    return updated,receipt


def event_reasons(state, decision, iteration):
    if state['method'] != 'New_R2' or iteration>28 or len(state['calls'])>=3:
        return []
    last = state['calls'][-1]['iteration'] if state['calls'] else 1
    if iteration-last<2: return []
    reasons=[]
    if decision['fallback']: reasons.append('B4_conflict')
    ds=state['decisions']
    if len(ds)>=3 and max(d['incumbent_gain'] for d in ds[-3:])<.005:
        reasons.append('stagnation_3_batches')
    if len(ds)>=2 and all(d.get('measurement',{}).get('innovation_z',0)>2.5 for d in ds[-2:]):
        reasons.append('bad_joint_innovation_2_batches')
    return reasons
