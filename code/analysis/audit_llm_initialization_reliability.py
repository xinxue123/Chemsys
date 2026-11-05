"""Fresh native reconstruction, with no production GP, selector or cache reads."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import argparse
import csv
import hashlib
import json
import warnings
import numpy as np
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior

ROOT=Path(__file__).resolve().parents[2]
ORIGINAL=ROOT/'experiments/runs/llm_event_start_qlogei_20261104'
OUT=ROOT/'experiments/reports/llm_initialization_audit_20261104'
BASE=ROOT/'experiments/runs/llm_initialization_audit_20261104'
POOL_PATH=ROOT/'experiments/runs/suzuki_guarded/inputs/case1/pool.json'
X=FEATURES=IDS=IX=PARAMS=None
COMPILES={}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def prepare():
    path=BASE/'registration.json'
    if path.exists():
        return check()
    protocol=read(ORIGINAL/'protocol.json')
    groups={}
    protected={}
    for job in protocol['jobs']:
        folder=ORIGINAL/'runs'/job['id']
        state=read(folder/'state.json')
        key=state['history'][0]['candidate_id']
        groups.setdefault(key,[]).append(job['id'])
        for source in folder.rglob('*'):
            if source.is_file():
                protected[source.relative_to(ROOT).as_posix()]=sha(source)
    for source in (ORIGINAL/'protocol.json',POOL_PATH,POOL_PATH.with_name('oracle.npz')):
        protected[source.relative_to(ROOT).as_posix()]=sha(source)
    source=Path(__file__).resolve()
    p=dict(audit='fresh_native_GP_and_qLogEI_reconstruction',
        purpose='Audit stored outcomes, not a new optimization comparison or tuning run',
        jobs=[dict(initial_id=key,source_jobs=values,reference_job=values[0]) for key,values in groups.items()],
        GP_parameters=protocol['GP_parameters'],protected_sha256=protected,
        audit_source_sha256=sha(source),initial_point_groups=len(groups),
        old_GP_or_acquisition_cache_reads=0,new_LLM_calls=0,new_oracle_experiments=0)
    save(path,p)
    return check()


def check():
    p=read(BASE/'registration.json')
    assert p['audit_source_sha256']==sha(__file__)
    for name,h in p['protected_sha256'].items():
        assert sha(ROOT/name)==h,name
    return p


def native_model(x,y):
    p=PARAMS
    model=MixedSingleTaskGP(x,y,cat_dims=[3],train_Yvar=torch.full_like(y,p['noise_variance']),outcome_transform=None)
    model.mean_module.initialize(constant=p['prior_mean'])
    additive,interaction=model.covar_module.kernels
    for kernel in (additive,interaction):
        kernel.register_constraint('raw_outputscale',GreaterThan(p['minimum_outer_scale']))
        kernel.initialize(outputscale=p['prior_outer_scale'])
        kernel.register_prior('outputscale_prior',GammaPrior(p['outer_scale_prior_concentration'],p['outer_scale_prior_rate']),'outputscale')
    nested_category=additive.base_kernel.kernels[1]
    nested_category.initialize(outputscale=p['prior_inner_scale'])
    for kernel in (additive.base_kernel.kernels[0],interaction.base_kernel.kernels[0]):
        kernel.initialize(lengthscale=p['prior_continuous_lengthscale'])
    for kernel in (nested_category.base_kernel,interaction.base_kernel.kernels[1]):
        kernel.initialize(lengthscale=p['prior_categorical_lengthscale'])
    return model


def initialize():
    global X,FEATURES,IDS,IX,PARAMS
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    PARAMS=read(BASE/'registration.json')['GP_parameters']
    pool=read(POOL_PATH)
    IDS=pool['ids']
    IX={cid:j for j,cid in enumerate(IDS)}
    categories=sorted({p['catalyst'] for p in pool['points']})
    ordinal={name:j for j,name in enumerate(categories)}
    continuous=(np.array([[p[a] for a in ('t_res','temperature','catalyst_loading')] for p in pool['points']])-[60.,30.,.5])/[540.,80.,2.]
    np.testing.assert_array_equal(continuous,np.array(pool['z'])[:,:3])
    FEATURES=[tuple(z)+(p['catalyst'],) for z,p in zip(continuous,pool['points'])]
    X=torch.tensor(np.column_stack([continuous.clip(1e-6,1-1e-6),
        [ordinal[p['catalyst']] for p in pool['points']]]),dtype=torch.double)
    # All native kernels/acquisition transforms load and warm before measurement.
    model=native_model(X[:4],torch.tensor([[.1],[.4],[.6],[.3]],dtype=torch.double))
    model.eval()
    acq=qLogExpectedImprovement(model,best_f=.6,sampler=SobolQMCNormalSampler(torch.Size([512]),seed=0))
    with torch.no_grad():
        acq(X[:2].unsqueeze(1))
    for name in ('script','trace','trace_module'):
        original=getattr(torch.jit,name)
        def counter(*args,_name=name,_original=original,**kwargs):
            COMPILES[_name]+=1
            return _original(*args,**kwargs)
        COMPILES[name]=0
        setattr(torch.jit,name,counter)


def priorities(query):
    order=sorted(range(len(IDS)),key=lambda j:FEATURES[j])
    result=np.empty(len(IDS),dtype=np.int64)
    result[order]=np.random.default_rng(np.random.SeedSequence([0,query])).permutation(len(IDS))
    return result


def job_audit(job):
    state=read(ORIGINAL/'runs'/job['reference_job']/'state.json')
    rows=[]
    for q in range(2,9):
        prefix=state['history'][:q-1]
        ordered=sorted(prefix,key=lambda h:FEATURES[IX[h['candidate_id']]])
        train_X=X[[IX[h['candidate_id']] for h in ordered]]
        train_Y=torch.tensor([[h['y']] for h in ordered],dtype=torch.double)
        torch.manual_seed(20261104+len(prefix))
        model=native_model(train_X,train_Y)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            if len(prefix)>=PARAMS['fit_after_observations']:
                fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood,model))
            model.eval()
            acq=qLogExpectedImprovement(model,best_f=float(train_Y.max()),
                sampler=SobolQMCNormalSampler(torch.Size([512]),seed=38112000+q))
            with torch.no_grad():
                scores=torch.cat([acq(part.unsqueeze(1)) for part in X.split(256)]).numpy()
                posterior=model.posterior(X)
                mu=posterior.mean.reshape(-1).numpy()
                variance=posterior.variance.reshape(-1).numpy()
        available=~np.isin(IDS,[h['candidate_id'] for h in prefix])
        priority=priorities(q)
        best=float(scores[available].max())
        leaders=np.flatnonzero(available & (scores==best))
        selected=int(leaders[np.argmin(priority[leaders])])
        recorded=IX[state['history'][q-1]['candidate_id']]
        with np.load(ORIGINAL/'runs'/job['reference_job']/'rounds'/f'{q:02}_GP.npz',allow_pickle=False) as old:
            errors={k:float(np.max(np.abs(actual-old[k]))) for k,actual in
                    (('mu',mu),('variance',variance),('logei',scores))}
            assert np.array_equal(priority,old['priorities'])
            assert np.array_equal(available,old['available'])
        folder=BASE/'fresh_scores'/job['initial_id']
        folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/f'{q:02}.npz',mu=mu,variance=variance,logei=scores,priorities=priority,available=available)
        row=dict(initial_id=job['initial_id'],reference_job=job['reference_job'],query=q,
            independently_selected=IDS[selected],recorded_selected=IDS[recorded],strict_choice_match=selected==recorded,
            recorded_choice_logei_regret=best-float(scores[recorded]),
            mean_max_abs_error=errors['mu'],variance_max_abs_error=errors['variance'],logei_max_abs_error=errors['logei'],
            exact_maximum_count=len(leaders),numerical_maximum_count=int((available & (best-scores<=1e-10)).sum()),
            hyperparameters_fitted=len(prefix)>=4,warnings=[str(w.message) for w in captured],
            operator_compiles_in_loop=sum(COMPILES.values()))
        rows.append(row)
    save(BASE/'jobs'/f"{job['initial_id']}.json",dict(job=job,rows=rows))
    print(json.dumps(dict(initial=job['initial_id'],strict_matches=sum(r['strict_choice_match'] for r in rows),
        prefixes=len(rows),max_regret=max(r['recorded_choice_logei_regret'] for r in rows))),flush=True)
    return rows


def csv_save(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def run(workers):
    p=check()
    rows=[]
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize) as executor:
        fs=[executor.submit(job_audit,j) for j in p['jobs']]
        for f in as_completed(fs):
            rows.extend(f.result())
    check()
    csv_save(OUT/'fresh_prefix_checks.csv',[dict(r,warnings=json.dumps(r['warnings'])) for r in rows])
    summary=dict(status='COMPLETE',unique_initial_points=len(p['jobs']),prefixes=len(rows),
        strict_choice_matches=sum(r['strict_choice_match'] for r in rows),
        maximum_recorded_choice_regret=max(r['recorded_choice_logei_regret'] for r in rows),
        maximum_mean_error=max(r['mean_max_abs_error'] for r in rows),
        maximum_variance_error=max(r['variance_max_abs_error'] for r in rows),
        maximum_logei_error=max(r['logei_max_abs_error'] for r in rows),
        prefixes_with_optimizer_warnings=sum(bool(r['warnings']) for r in rows),
        operator_compiles_in_loop=max(r['operator_compiles_in_loop'] for r in rows),
        old_model_or_score_cache_reads=0,new_GP_fits=sum(r['hyperparameters_fitted'] for r in rows),
        new_LLM_calls=0,new_oracle_experiments=0,all_original_files_unchanged=True)
    save(OUT/'fresh_reconstruction.json',summary)
    print(json.dumps(summary))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('prepare','run'))
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    if args.action=='prepare':
        p=prepare()
        print(json.dumps(dict(status='REGISTERED',initial_point_groups=len(p['jobs']))))
    else:
        run(args.workers)


if __name__=='__main__':
    main()
