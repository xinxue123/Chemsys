"""The Reasoning BO paper's BO engine on the held-out tasks, from the existing shared starts.

The paper's code runs Ax 0.5 with a plain SingleTaskGP and qLogNoisyExpectedImprovement. Ax is not
installed here, so its defaults are emulated in BoTorch 0.14: text factors one-hot, numeric choices
ordered and scaled to [0, 1], BoTorch's default SingleTaskGP (RBF ARD, Standardize, inferred noise),
and qLogNEI with the observed points as baseline and 512 Sobol samples, scored over every unqueried
candidate. Ax's outcome winsorization and other internals are not reproduced.

Only the engine differs from our plain BO: same pools, shared starts, tie streams, one point per
query and 20 queries. Rules are in docs/hypothesis_scope_v2_registration_20261105.md, under
"Reasoning BO paper's BO engine on the held-out tasks". No model calls.

    C:\\ChemEnv310\\Scripts\\python.exe -m hypothesis_scope_v2.paper_bo run [--pairs 0 1 ...] [--tasks aryl cpa]
"""
import argparse
import csv
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[3]/'experiments/runs/hypothesis_scope_v2_20261105'
OUT = BASE/'paper_bo'
ROOT = Path(__file__).resolve().parents[3]
TASKS = dict(aryl=dict(input=BASE/'heldout/input', numeric=('Concentration', 'Temp_C')),
             cpa=dict(input=BASE/'cpa/input', numeric=()))
QUERIES, MC = 20, 512
RECIPE = dict(engine='paper_bo_v1', gp='SingleTaskGP default (RBF ARD, Standardize, Gaussian noise)', acquisition='qLogNoisyExpectedImprovement',
              mc_samples=MC, encoding='one-hot text factors; ordered numeric choices scaled to [0, 1]', candidates='every unqueried pool candidate',
              batch=1, queries=QUERIES, ties='feature-keyed PCG64 stream [stream, query], as in our BO')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()


def load(task):
    spec = TASKS[task]
    pool = json.loads((spec['input']/'pool.json').read_text(encoding='utf-8'))
    oracle = np.load(spec['input']/'oracle.npz')
    columns = []
    for axis in pool['axes']:
        levels = pool['levels'][axis]
        values = [p[axis] for p in pool['points']]
        if axis in spec['numeric']:
            ordered = sorted(levels, key=float)
            columns.append(np.array([[ordered.index(v)/(len(ordered) - 1)] for v in values]))
        else:
            columns.append(np.array([[float(v == level) for level in levels] for v in values]))
    features = [tuple(p[a] for a in pool['axes']) for p in pool['points']]
    assert len(set(features)) == len(features)
    return dict(ids=pool['ids'], X=np.hstack(columns), y=oracle['values'], raw=oracle['raw_values'], features=features, axes=pool['axes'])


def priorities(features, query, stream):
    """Identical to knowledge_gp.semantic_priorities: an RNG stream mapped through sorted physical features."""
    order = sorted(range(len(features)), key=lambda j: features[j])
    result = np.empty(len(order), dtype=np.int64)
    result[order] = np.random.default_rng(np.random.SeedSequence([stream, query])).permutation(len(order))
    return result


def starts(task):
    if task == 'aryl':
        rows = list(csv.DictReader(open(ROOT/'results/hypothesis_scope_v2_shared_start_20261107/pairs.csv', encoding='utf-8')))
        return {int(r['pair']): r['first_point'] for r in rows}
    rows = list(csv.DictReader(open(ROOT/'results/hypothesis_scope_v2_cpa_20261107/pairs.csv', encoding='utf-8')))
    return {int(r['pair']): r['start'] for r in rows}


def scores(X, y, observed, seed):
    import torch
    from botorch.acquisition.logei import qLogNoisyExpectedImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.sampling.normal import SobolQMCNormalSampler
    from gpytorch.mlls import ExactMarginalLogLikelihood
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xt = torch.tensor(X, dtype=torch.double)
    train_X, train_Y = Xt[observed], torch.tensor(y[observed], dtype=torch.double).unsqueeze(-1)
    model = SingleTaskGP(train_X, train_Y)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    model.eval()
    acq = qLogNoisyExpectedImprovement(model, X_baseline=train_X, sampler=SobolQMCNormalSampler(torch.Size([MC]), seed=seed), prune_baseline=True)
    with torch.no_grad():
        values = torch.cat([acq(chunk.unsqueeze(1)).reshape(-1) for chunk in Xt.split(256)]).numpy()
    parameters = {n: p.detach().reshape(-1).tolist() for n, p in model.named_parameters()}
    return values, parameters


def run(task, pair):
    import torch
    torch.set_num_threads(1)
    folder = OUT/task/f's{pair}'
    path = folder/'state.json'
    if path.exists():
        state = json.loads(path.read_text(encoding='utf-8'))
        if state['status'] == 'COMPLETE':
            return task, pair, 'already complete'
    data = load(task)
    ix = {c: j for j, c in enumerate(data['ids'])}
    start = starts(task)[pair]
    state = dict(task=task, pair=pair, stream=pair, start=start, recipe=RECIPE, recipe_sha256=digest(RECIPE), history=[], status='RUNNING')
    task_seed = int(digest(task)[:6], 16) % 100000
    for q in range(1, QUERIES + 1):
        observed = [ix[h['candidate_id']] for h in state['history']]
        if q == 1:
            j, meta = ix[start], dict(reason='shared start')
        else:
            seed = 7112026 + task_seed*100 + q
            values, parameters = scores(data['X'], data['y'], observed, seed)
            available = np.ones(len(values), dtype=bool)
            available[observed] = False
            js = np.flatnonzero(available)
            best = values[js].max()
            js = js[values[js] == best]
            ties = priorities(data['features'], q, pair)
            j = int(js[np.argmin(ties[js])])
            meta = dict(reason='paper-style qLogNEI', log_nei=float(best), exact_ties=int(len(js)), seed=seed, gp_parameters=parameters)
        assert j not in observed
        state['history'].append(dict(query=q, candidate_id=data['ids'][j], y=float(data['y'][j]), raw_value=float(data['raw'][j]), **meta))
    state['status'] = 'COMPLETE'
    folder.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=1) + '\n', encoding='utf-8')
    temporary.replace(path)
    return task, pair, 'complete'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('run',))
    parser.add_argument('--tasks', nargs='+', default=list(TASKS))
    parser.add_argument('--pairs', nargs='+', type=int, default=list(range(24)))
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = parser.parse_args()
    jobs = [(t, p) for t in a.tasks for p in a.pairs]
    with ProcessPoolExecutor(min(a.workers, len(jobs))) as pool:
        futures = [pool.submit(run, t, p) for t, p in jobs]
        for f in as_completed(futures):
            print(*f.result(), flush=True)


if __name__ == '__main__':
    main()
