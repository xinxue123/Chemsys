"""Registered matched eight-observation trajectories."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os

import numpy as np

from chem.benchmarks import knowledge_gp
from . import PARAMETERS as P, VARIANT
from .model import SharedGP, digest
from .policy import decision

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'experiments/runs'/VARIANT
REPORT = ROOT/'experiments/reports'/VARIANT
DESIGN = ROOT/'docs'/f'{VARIANT}.txt'
GP = None
POOL = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        if read(path) != value:
            raise ValueError('Immutable evidence differs: '+str(path))
        return
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def prepare():
    if (BASE/'protocol.json').exists():
        return check()
    pool = knowledge_gp.load_task(ROOT, 'Reizman_Suzuki_Case_1', seed=0)
    # Use the compact exported start registry in the portable workspace.
    with (ROOT/'results/suzuki_initial_yield_20261104/metrics.csv').open(
            encoding='utf-8-sig', newline='') as source:
        starts = [dict(seed=int(r['seed']), arm=r['arm'], initial_id=r['initial_id'])
                  for r in csv.DictReader(source) if r['case']=='1' and r['arm']!='mixed']
    jobs = [dict(id=f"s{j['seed']}_{j['arm']}_{arm}", seed=j['seed'],
                 initial_group=j['arm'], initial_id=j['initial_id'], arm=arm)
            for j in starts for arm in ('qlogei', 'rollout')]
    assert len(jobs)==20
    sources = {}
    for directory in ('code/chemist_system/chem', 'code/experiment_tools'):
        for dependency in (ROOT/directory).rglob('*.py'):
            sources[dependency.relative_to(ROOT).as_posix()] = sha(dependency)
    start_registry = ROOT/'results/suzuki_initial_yield_20261104/metrics.csv'
    sources[start_registry.relative_to(ROOT).as_posix()] = sha(start_registry)
    for path in Path(__file__).parent.glob('*.py'):
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
    for path in (DESIGN, ROOT/'code/experiment_tools/run_fast_hit_bo.py'):
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
    protocol = dict(variant=VARIANT, registered_utc=datetime.now(timezone.utc).isoformat(),
                    jobs=jobs, parameters=P, initial=1, total_queries=8,
                    sources_sha256=sources,
                    inputs_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in
                                   (pool.source_path, pool.source_path.with_name('oracle.npz'))},
                    design=DESIGN.read_text(encoding='utf-8-sig'), new_observations_at_registration=0)
    protocol['protocol_sha256'] = digest(protocol)
    write(BASE/'protocol.json', protocol)
    # This endpoint record is evaluator-only and is never read by the worker.
    with np.load(pool.source_path.with_name('oracle.npz'), allow_pickle=False) as z:
        maximum = float(z['values'].max())*pool.scale
    write(BASE/'evaluation_endpoint.json', dict(maximum=maximum, threshold95=.95*maximum,
                                               budget=8, no_hit_comparison_penalty=9))
    return check()


def check():
    p = read(BASE/'protocol.json')
    if p['protocol_sha256'] != digest({k:v for k,v in p.items() if k!='protocol_sha256'}):
        raise ValueError('Protocol changed')
    for group in ('sources_sha256','inputs_sha256'):
        for path,h in p[group].items():
            if sha(ROOT/path) != h:
                raise ValueError('Source/input changed: '+path)
    return p


def initialize():
    global GP, POOL
    check()
    (ROOT/'experiments/chemist_system').mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT/'experiments/chemist_system')
    POOL = knowledge_gp.load_task(ROOT, 'Reizman_Suzuki_Case_1', seed=0)
    GP = SharedGP(POOL, BASE/'cache')


def run_job(job):
    folder = BASE/'runs'/job['id']
    sf = folder/'state.json'
    state = read(sf) if sf.exists() else dict(job=job, history=[], decisions=[], status='RUNNING')
    if state['status']=='COMPLETE':
        return dict(job=job['id'], queries=8, status='COMPLETE')
    while len(state['history'])<P['total_queries']:
        q = len(state['history'])+1
        cp = folder/'rounds'/f'{q:02}_commit.json'
        if cp.exists():
            commit = read(cp)
            if commit['prestate_sha256'] != digest(state):
                raise ValueError('State differs before observation')
        else:
            if q==1:
                packet = dict(selected_id=job['initial_id'], rule='controlled_shared_initial_point')
                selected_gp = None
                key = None
            else:
                context = GP.context(state['history'], q)
                packet = decision(GP, context, q, job['arm'])
                j = POOL.ix[packet['selected_id']]
                key = context['key']
                selected_gp = dict(mean=float(context['mu'][j])*100,
                                   sd=float(context['variance'][j]**.5)*100,
                                   qLogEI=float(context['logei'][j]),
                                   hyperparameters_fitted=context['hyperparameters_fitted'])
                arrays = folder/'rounds'/f'{q:02}_GP.npz'
                arrays.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(arrays, mu=context['mu'], variance=context['variance'],
                                    logei=context['logei'], available=context['available'],
                                    priorities=context['priorities'])
                packet.update(GP_arrays_sha256=sha(arrays),
                              checkpoint=context['checkpoint'].relative_to(ROOT).as_posix(),
                              checkpoint_sha256=sha(context['checkpoint']))
            commit = dict(query=q, prestate_sha256=digest(state), history_sha256=digest(state['history']),
                          protocol_sha256=read(BASE/'protocol.json')['protocol_sha256'],
                          decision=packet, selected_GP=selected_gp, fit_key=key)
            commit['commit_sha256'] = digest(commit)
            write(cp, commit)
        cid = commit['decision']['selected_id']
        if cid in {o['candidate_id'] for o in state['history']}:
            raise ValueError('Repeated experiment')
        obs = knowledge_gp.read_observation(ROOT, POOL, cid)
        state['history'].append(obs)
        state['decisions'].append(commit)
        state['status'] = 'COMPLETE' if q==8 else 'RUNNING'
        write(sf, state, replace=True)
        if sum(knowledge_gp.COMPILE_COUNTS.values()):
            raise ValueError('Operator compilation inside loop')
        write(folder/'runtime.json', dict(operator_compiles_in_loop=0, **GP.counters), replace=True)
        print(json.dumps(dict(job=job['id'], query=q, yield_percent=obs['raw_value'],
                              best=max(o['raw_value'] for o in state['history']))), flush=True)
    return dict(job=job['id'], queries=8, status='COMPLETE')


def preflight():
    p = check()
    initialize()
    initial = next(j['initial_id'] for j in p['jobs']
                   if j['seed']==4 and j['initial_group']=='middle')
    history = [knowledge_gp.read_observation(ROOT, POOL, initial)]
    context = GP.context(history[:1], 2)
    assert context['variance'].max()-context['variance'].min()>1e-6
    assert not context['hyperparameters_fitted']
    from .policy import choose
    import torch
    outer = np.flatnonzero(context['available'])[:3].tolist()
    cross = GP.conditional_cross_covariance(context, outer)
    checks = []
    for col,j in enumerate(outer):
        mean = context['mu'][j]
        y = mean + .2*np.sqrt(context['variance'][j])
        indices = np.flatnonzero(context['available'])[6:11]
        updated = context['model'].condition_on_observations(
            GP.X[j:j+1], torch.tensor([[y]],dtype=torch.double),
            noise=torch.full((1,1),P['noise_variance'],dtype=torch.double))
        with torch.no_grad():
            expected = updated.posterior(GP.X[indices])
        denominator = context['variance'][j]+P['noise_variance']
        mu = context['mu'][indices]+cross[indices,col].numpy()*(y-mean)/denominator
        var = context['variance'][indices]-cross[indices,col].numpy()**2/denominator
        np.testing.assert_allclose(mu,expected.mean.reshape(-1).numpy(),rtol=0,atol=1e-9)
        np.testing.assert_allclose(var,expected.variance.reshape(-1).numpy(),rtol=0,atol=1e-9)
        checks.append(dict(outer=POOL.ids[j],conditional_posterior_matches=True))
    fallback = decision(GP, context, 8, 'rollout')
    assert fallback['selected_id']==POOL.ids[choose(context['logei'],context['available'],context['priorities'])]
    # Evaluate an actual two-step decision without consuming an outcome.
    trial = decision(GP,context,2,'rollout')
    assert trial['selected_id'] in trial['outer_ids'] and len(trial['outer_ids'])<=48
    assert sum(knowledge_gp.COMPILE_COUNTS.values())==0
    result = dict(status='PASS', protocol_sha256=p['protocol_sha256'],
        singleton_variance_range=[float(context['variance'].min()),float(context['variance'].max())],
        singleton_acquisition_range=[float(context['logei'].min()),float(context['logei'].max())],
        checks=checks,last_step_qLogEI_exact=True,actual_rollout_value_finite=True,
        new_experiments=0,operator_compiles_in_loop=0)
    write(BASE/'preflight.json', result)
    return result


def batch(workers):
    p = check()
    if read(BASE/'preflight.json')['status']!='PASS':
        raise ValueError('Preflight required')
    results, errors = [], []
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize) as executor:
        futures = {executor.submit(run_job,j):j for j in p['jobs']}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as error:
                errors.append(dict(job=futures[future]['id'],type=type(error).__name__,error=str(error)))
                print(json.dumps(errors[-1]),flush=True)
    check()
    result = dict(status='COMPLETE' if not errors else 'FAILED',results=results,errors=errors)
    write(BASE/'batch_receipt.json',result,replace=True)
    if errors:
        raise RuntimeError(errors)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('prepare','preflight','run'))
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    result=prepare() if args.action=='prepare' else preflight() if args.action=='preflight' else batch(args.workers)
    if args.action=='prepare':
        result=dict(status='REGISTERED',jobs=len(result['jobs']),protocol_sha256=result['protocol_sha256'])
    print(json.dumps(result))


if __name__=='__main__':
    main()
