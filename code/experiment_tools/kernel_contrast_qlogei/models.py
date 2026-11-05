"""Same-data alternative starts; never mutate the primary model parameters."""
from pathlib import Path
import hashlib
import json
import random
import warnings
import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def parameters(model):
    return {k:p.detach().cpu().tolist() for k,p in model.named_parameters()}


def state_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def factory(pool, x, y):
    return SingleTaskGP(x, y) if pool.group == 'P2' else MixedSingleTaskGP(
        x, y, cat_dims=[3] if pool.group == 'P0' else list(range(x.shape[1])))


def data(gp, history):
    pool = gp.pool
    ordered = sorted(history, key=lambda r:tuple(gp.X[pool.ix[r['candidate_id']]].tolist())
        if pool.group == 'P2' else r['candidate_id'])
    x = gp.X[[pool.ix[r['candidate_id']] for r in ordered]]
    raw = torch.tensor([[r['y']] for r in ordered], dtype=torch.double)
    mean, std = float(raw.mean()), float(raw.std(unbiased=False).clamp_min(1e-12))
    return x, (raw-mean)/std, mean, std, ordered


def correlations(model, pool):
    base = np.array([0. if k == 'categorical' else .5 for k in pool.kinds])
    probes = [base]
    for axis, kind in enumerate(pool.kinds):
        p = base.copy(); p[axis] += 1 if kind == 'categorical' else .2; probes.append(p)
    with torch.no_grad():
        k = model.covar_module(torch.tensor(np.asarray(probes), dtype=torch.double)).to_dense().numpy()
    return k[0, 1:]/np.sqrt(k[0, 0]*np.diag(k)[1:])


def marginal_likelihood(model):
    model.train(); model.likelihood.train()
    with torch.no_grad():
        result = float(ExactMarginalLogLikelihood(model.likelihood, model)(
            model(model.train_inputs[0]), model.train_targets))
    model.eval(); model.likelihood.eval()
    return result


def predictions(model, gp, mean, std):
    mu, sd, obs = [], [], []
    with torch.no_grad():
        for xx in gp.X.split(256):
            p = model.posterior(xx.unsqueeze(1))
            q = model.posterior(xx.unsqueeze(1), observation_noise=True)
            mu.append(p.mean.reshape(-1)); sd.append(p.variance.reshape(-1).clamp_min(1e-18).sqrt())
            obs.append(q.variance.reshape(-1).clamp_min(1e-18).sqrt())
    return dict(mu=torch.cat(mu).numpy()*std+mean, sd=torch.cat(sd).numpy()*std,
        observation_sd=torch.cat(obs).numpy()*std)


def describe(model, gp, mean, std, source, parent, checkpoint, fit_warnings):
    pars = parameters(model)
    item = dict(id=digest(dict(parameters=pars, mean=mean, std=std))[:20],
        source=source, parent=parent, parameters=pars, MLL=marginal_likelihood(model),
        rho=correlations(model, gp.pool).tolist(), checkpoint=str(checkpoint),
        optimizer_warnings=fit_warnings)
    item.update(predictions(model, gp, mean, std))
    return item


def alternatives(gp, history, context, parents, directory):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    x, y, mean, std, ordered = data(gp, history)
    native = context['stats']['model']; initial_hash = state_hash(native)
    np.testing.assert_allclose(native.train_inputs[0].detach().numpy(), x.numpy(), rtol=0, atol=0)
    seed = int(context['stats']['fit_seed'])
    native_item = describe(native, gp, mean, std, 'native', None,
        gp.directory/(context['stats']['fit_key']+'.pt'), [])
    np.testing.assert_allclose(native_item['mu'], context['stats']['mu_y'], rtol=0, atol=1e-12)
    np.testing.assert_allclose(native_item['sd'], context['stats']['sd_y'], rtol=0, atol=1e-12)
    items, starts = [native_item], set()
    for parent in parents:
        start_key = digest(parent['parameters'])
        if start_key in starts:
            continue
        starts.add(start_key)
        recipe = dict(history=[dict(candidate_id=r['candidate_id'], y=r['y']) for r in ordered],
            pool_sha256=hashlib.sha256(gp.pool.source_path.read_bytes()).hexdigest(),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            fit_seed=seed, start_parameters=parent['parameters'], torch=torch.__version__)
        key = digest(recipe); path = directory/(key+'.pt')
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        model = factory(gp.pool, x, y)
        if path.exists():
            saved = torch.load(path, map_location='cpu', weights_only=False)
            if saved['key'] != key:
                raise ValueError('Alternative cache recipe changed')
            model.load_state_dict(saved['state']); fit_warnings = saved['warnings']
        else:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    parameter.copy_(torch.tensor(parent['parameters'][name], dtype=parameter.dtype))
            with warnings.catch_warnings(record=True) as logs:
                warnings.simplefilter('always')
                fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
            fit_warnings = [str(w.message) for w in logs]
            saved = dict(key=key, recipe=recipe, state=model.state_dict(), warnings=fit_warnings,
                training_ids=[r['candidate_id'] for r in ordered], training_y=[r['y'] for r in ordered])
            temp = path.with_suffix('.tmp'); torch.save(saved, temp); temp.replace(path)
        model.eval(); model.likelihood.eval()
        # Check internal targets/scaling against the primary native model.
        np.testing.assert_allclose(model.train_targets.detach().numpy(), native.train_targets.detach().numpy(), rtol=0, atol=1e-12)
        item = describe(model, gp, mean, std, 'from_previous_'+parent['role'], parent['id'], path, fit_warnings)
        items.append(item)
    if state_hash(native) != initial_hash:
        raise AssertionError('Native model state was changed by alternative fitting')
    return items
