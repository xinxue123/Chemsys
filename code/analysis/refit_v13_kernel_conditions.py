"""Isolated, cached, process-parallel GP refits of already observed prefixes."""
import os
for variable in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[variable]='1'

from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import inspect
import io
import json
import random
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
import botorch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/analysis'))
import profile_v13_kernel_conditions as design
origin=design.origin
OUT=design.OUT
RUNS=ROOT/'experiments/runs/v13_kernel_conditions_20261104'


def digest(value):return origin.profile.helpers.digest(design.plain(value))
def factory(task,x,y):
    pool=origin.POOLS[task]
    return SingleTaskGP(x,y) if pool.group=='P2' else MixedSingleTaskGP(x,y,cat_dims=[3] if pool.group=='P0' else list(range(x.shape[1])))


def ordered_data(task,history):
    pool=origin.POOLS[task]
    ordered=sorted(history,key=lambda h:tuple(origin.X[task][pool.ix[h['candidate_id']]].numpy())) if pool.group=='P2' else sorted(history,key=lambda h:h['candidate_id'])
    x=origin.X[task][[pool.ix[h['candidate_id']] for h in ordered]]
    y=torch.tensor([[h['y']] for h in ordered],dtype=torch.double)
    center=float(y.mean());std=float(y.std(unbiased=False).clamp_min(1e-12))
    return x,(y-center)/std,center,std,ordered


def fit_model(task,history,seed,start_parameters=None):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    x,y,center,std,ordered=ordered_data(task,history)
    model=factory(task,x,y)
    if start_parameters is not None:
        with torch.no_grad():
            for name,param in model.named_parameters():param.copy_(torch.as_tensor(start_parameters[name],dtype=param.dtype))
    start=time.perf_counter()
    with warnings.catch_warnings(record=True) as logs:
        warnings.simplefilter('always')
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood,model))
    elapsed=time.perf_counter()-start
    # A and W share the same standardized objective; compare their likelihoods
    # at their solutions rather than equating a changed kernel with information.
    model.train()
    with torch.no_grad():
        prior=model(x);ll=float(model.likelihood(prior).log_prob(model.train_targets))
        mll=float(ExactMarginalLogLikelihood(model.likelihood,model)(prior,model.train_targets))
    model.eval();model.likelihood.eval()
    return model,dict(mean=center,std=std,fit_seconds=elapsed,fit_seed=seed,
        training_ids=[h['candidate_id'] for h in ordered],training_y=[h['y'] for h in ordered],
        marginal_loglikelihood_per_observation=mll,data_loglikelihood=ll,
        optimizer_warnings=[str(w.message) for w in logs])


def worker_initialize():
    torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    for task,full in origin.profile.FULL.items():
        if task=='aryl':continue
        pool=origin.profile.knowledge_gp.load_task(ROOT,full,seed=0);origin.POOLS[task]=pool
        with np.load(pool.source_path.with_name('oracle.npz'),allow_pickle=False) as z:
            origin.TRUTH[task]=z['raw_values'].copy() if 'raw_values' in z else z['values'].copy()*pool.scale
    origin.MOD['mixed']=origin.profile.knowledge_gp._legacy_import('mixed_kalman_suzuki_engine',ROOT)
    origin.MOD['nominal']=origin.profile.knowledge_gp._legacy_import('reaction_transfer_engine',ROOT)
    for task,pool in origin.POOLS.items():
        z=origin.MOD['mixed'].encode_mixed(pool.raw_pool) if pool.group=='P0' else origin.MOD['nominal'].encode(pool.raw_pool) if pool.group=='P1' else pool.raw_pool['z']
        origin.X[task]=torch.tensor(z,dtype=torch.double)
        # Initialize the native kernels once in each persistent worker. There is
        # no JIT/torch.compile call in this workflow.
        toy=factory(task,origin.X[task][:2],torch.tensor([[0.],[1.]],dtype=torch.double))
        design.CONSTRAINTS[task]={name:toy.constraint_for_parameter_name(name) for name,_ in toy.named_parameters()}
        with torch.no_grad():toy.covar_module(origin.X[task][:2]).to_dense()


def cached_fit(case_id,variant,task,history,seed,start_parameters=None):
    recipe=dict(task=task,history=sorted(history,key=lambda h:h['candidate_id']),seed=seed,
        start_parameters=start_parameters,torch=torch.__version__,botorch=botorch.__version__,
        factory_sha256=hashlib.sha256(inspect.getsource(factory).encode()).hexdigest(),
        fit_sha256=hashlib.sha256(inspect.getsource(fit_model).encode()).hexdigest(),
        ordering_sha256=hashlib.sha256(inspect.getsource(ordered_data).encode()).hexdigest())
    key=digest(recipe);folder=RUNS/case_id;folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'{variant}_{key[:16]}.pt'
    if path.exists():
        saved=torch.load(path,map_location='cpu',weights_only=False);assert saved['recipe_sha256']==key
        x,y,_,_,_=ordered_data(task,history);model=factory(task,x,y)
        model.load_state_dict(saved['state']);model.eval();model.likelihood.eval()
        return model,saved['metadata'],True,path
    model,meta=fit_model(task,history,int(seed),start_parameters)
    saved=dict(recipe_sha256=key,recipe=recipe,state=model.state_dict(),metadata=meta)
    temp=path.with_suffix(f'.{os.getpid()}.tmp');torch.save(saved,temp);temp.replace(path)
    return model,meta,False,path


def evaluate(model,meta,r,after,fixed_norm):
    pool=origin.POOLS[r.task];truth=origin.TRUTH[r.task];span=r.pool_range
    mu=[];sd=[]
    with torch.no_grad():
        for xx in origin.X[r.task].split(256):
            p=model.posterior(xx.unsqueeze(1));mu.append(p.mean.reshape(-1));sd.append(p.variance.reshape(-1).clamp_min(1e-18).sqrt())
    mu=torch.cat(mu).numpy()*meta['std']+meta['mean'];sd=torch.cat(sd).numpy()*meta['std']
    # All variants face exactly the same acquisition incumbent, normalization,
    # draws, mask and tie priorities. Synthetic feedback never changes these.
    scores=origin.logei((mu-fixed_norm['mean'])/fixed_norm['std'],sd/fixed_norm['std'],fixed_norm['best'],fixed_norm['seed'])
    mask=after['available'];high=mask&(truth>=.9*truth.max());low=mask&(truth<.5*truth.max())
    targets=np.flatnonzero(mask&(truth==truth.max()))
    target=min(targets,key=lambda j:(-scores[j],after['tie_priority'][j])) if len(targets) else None
    rawmu=mu*pool.scale;rawsd=sd*pool.scale
    pars={name:value.detach().numpy().tolist() for name,value in model.named_parameters()}
    shape=design.kernel_shape(r.task,pars)
    result=dict(rho=shape['rho'].tolist(),ell1=shape['ell1'],ell2=shape['ell2'],
        kernel_diag=shape['kernel_diag'],noise=shape['noise'],background=shape['background'],parameters=pars,
        high_MAE=float(np.abs(rawmu[high]-truth[high]).mean()/span) if high.any() else None,
        high_mean=float(rawmu[high].mean()/span) if high.any() else None,
        high_coverage95=float((np.abs(rawmu[high]-truth[high])<=1.96*rawsd[high]).mean()) if high.any() else None,
        discrimination=float((rawmu[high].mean()-rawmu[low].mean())/span) if high.any() and low.any() else None,
        global_rank=origin.rank(scores,mask,after['tie_priority'],target)['rank'] if target is not None else None,
        global_target_id=pool.ids[target] if target is not None else None,
        mean_SD=float(rawsd[mask].mean()/span))
    return result,mu,sd,scores


def run_case(job):
    started=time.perf_counter();r=pd.Series(job['meta']);case=job['case_id'];n=int(job['measured_total'])
    history=job['history'];assert len(history)==n and n<20
    origin.STATES[r.path_id]=dict(history=history)
    before=origin.frame(r,n);after=origin.frame(r,n+1)
    assert digest(before['parameters'])==job['old_parameters_sha256']
    assert digest(after['parameters'])==job['new_parameters_sha256']
    nb=origin.normalization(r,n-1);na=origin.normalization(r,n)
    before_mu,before_sd=origin.physical_stats(before,nb);after_mu,after_sd=origin.physical_stats(after,na)
    pool=origin.POOLS[r.task];j=pool.ix[history[-1]['candidate_id']]
    seed_before=torch.load(origin.checkpoint(r,before),map_location='cpu',weights_only=False)['fit_seed']
    seed_after=torch.load(origin.checkpoint(r,after),map_location='cpu',weights_only=False)['fit_seed']
    expected=[dict(h) for h in history];expected[-1]=dict(expected[-1],y=float(before_mu[j]),raw_value=float(before_mu[j]*pool.scale))
    variants=[('A_actual_native',history,seed_after,None),('E_GP_expected',expected,seed_after,None),
              ('S_old_data_next_seed',history[:-1],seed_after,None),('W_actual_warm_start',history,seed_after,before['parameters'])]
    rows=[];results={};cache_hits=0;new_fits=0
    for variant,hist,seed,start in variants:
        row=dict(case_id=case,path_id=r.path_id,task=r.task,method=r.method,regime=r.regime,
            seed=int(r.seed),measured_total=n,new_candidate=history[-1]['candidate_id'],variant=variant)
        try:
            model,meta,hit,path=cached_fit(case,variant,r.task,hist,int(seed),start)
            cache_hits+=int(hit);new_fits+=int(not hit)
            ev,mu,sd,scores=evaluate(model,meta,r,after,na)
            row.update(status='PASS',cache_hit=hit,checkpoint=path.relative_to(ROOT).as_posix(),
                fit_seed=int(seed),fit_seconds=meta['fit_seconds'],MLL=meta['marginal_loglikelihood_per_observation'],
                data_loglikelihood=meta['data_loglikelihood'],warnings=meta['optimizer_warnings'],
                empirical_mean=meta['mean'],empirical_SD=meta['std'],**ev)
            if variant=='A_actual_native':
                row.update(archived_mu_max_abs_error=float(np.max(np.abs(mu-after_mu))),
                    archived_SD_max_abs_error=float(np.max(np.abs(sd-after_sd))),
                    archived_qLogEI_max_abs_error=float(np.max(np.abs(scores-after['base_logei']))),
                    archived_rho_max_abs_error=float(np.max(np.abs(np.array(ev['rho'])-design.kernel_shape(r.task,after['parameters'])['rho']))))
            results[variant]=row
        except Exception as e:
            row.update(status='FAILED',error=repr(e));results[variant]=row
        rows.append(row)
    # If changing just the seed alters the old-data shape, rerun the original
    # old-data seed to determine whether a seed attribution is justified.
    oldshape=design.kernel_shape(r.task,before['parameters'])['rho']
    s=results['S_old_data_next_seed']
    if s['status']=='PASS' and np.max(np.abs(np.array(s['rho'])-oldshape))>.25:
        try:
            model,meta,hit,path=cached_fit(case,'O_old_data_original_seed',r.task,history[:-1],int(seed_before))
            cache_hits+=int(hit);new_fits+=int(not hit)
            ev,_,_,_=evaluate(model,meta,r,after,na)
            rows.append(dict(case_id=case,path_id=r.path_id,task=r.task,method=r.method,regime=r.regime,
                seed=int(r.seed),measured_total=n,new_candidate=history[-1]['candidate_id'],variant='O_old_data_original_seed',
                status='PASS',cache_hit=hit,checkpoint=path.relative_to(ROOT).as_posix(),fit_seed=int(seed_before),
                fit_seconds=meta['fit_seconds'],MLL=meta['marginal_loglikelihood_per_observation'],**ev))
        except Exception as e:rows.append(dict(case_id=case,variant='O_old_data_original_seed',status='FAILED',error=repr(e)))
    receipt=dict(case_id=case,new_fits=new_fits,cache_hits=cache_hits,elapsed_seconds=time.perf_counter()-started,
        observed_result=history[-1]['raw_value'],GP_expected_result=float(before_mu[j]*pool.scale),
        before_rho=oldshape.tolist(),archived_after_rho=design.kernel_shape(r.task,after['parameters'])['rho'].tolist())
    folder=RUNS/case;folder.mkdir(parents=True,exist_ok=True)
    design.write(folder/'diagnostics.json',dict(job=job,receipt=receipt,variants=rows))
    return receipt,rows


def select_pilot(jobs):
    registry=pd.read_csv(OUT/'controlled_case_registry.csv',encoding='utf-8-sig')
    c2=registry[(registry.task=='c2')&(registry.method=='v13')&(registry.regime=='warm')]
    fixed=c2[c2.selection=='fixed_Case2_admission_example']
    # Include every documented entry example, even when it was already the
    # largest-change case and therefore has the general selection label.
    d=pd.read_csv(origin.SOURCE/'conditional_path_profiles.csv',encoding='utf-8-sig')
    ev=pd.read_csv(origin.OUT/'all_first_exposure_events.csv',encoding='utf-8-sig')
    ids=[]
    for _,r in d[(d.primary)&(d.task=='c2')&(d.method=='v13')&(d.regime=='warm')].iterrows():
        total=int(ev[(ev.path_id==r.path_id)&(ev.target_kind=='pool_global_max')].iloc[0].first_exposure_total)-1
        ids.append(registry[(registry.path_id==r.path_id)&(registry.measured_total==total)].iloc[0].case_id)
    for task in ('doyle','hplc'):ids.append(next(j['case_id'] for j in jobs if j['meta']['task']==task))
    return [j for j in jobs if j['case_id'] in set(ids)]


def main():
    jobs=json.loads((OUT/'controlled_jobs.json').read_text(encoding='utf-8'))
    pilot='--pilot' in sys.argv;jobs=select_pilot(jobs) if pilot else jobs
    workers=int(sys.argv[sys.argv.index('--workers')+1]) if '--workers' in sys.argv else 8
    RUNS.mkdir(parents=True,exist_ok=True)
    source_files=[Path(__file__),Path(design.__file__),Path(origin.__file__),ROOT/'docs/v13_kernel_conditions_20261104.txt',OUT/'controlled_jobs.json']
    hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    design.write(RUNS/('pilot_protocol.json' if pilot else 'protocol.json'),dict(created_UTC=datetime.now(timezone.utc).isoformat(),
        scope='controlled prefix refits, no experimental queries',jobs=len(jobs),workers=workers,
        variants=['A_actual_native','E_GP_expected','S_old_data_next_seed','W_actual_warm_start'],
        original_experiments_unchanged=True,source_hashes=hashes))
    allrows=[];receipts=[];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers,initializer=worker_initialize) as executor:
        futures={executor.submit(run_case,j):j['case_id'] for j in jobs}
        for future in as_completed(futures):
            receipt,rows=future.result();receipts.append(receipt);allrows.extend(rows)
            if len(receipts)%5==0 or len(receipts)==len(jobs):
                print(json.dumps(dict(completed=len(receipts),total=len(jobs),elapsed_seconds=round(time.perf_counter()-start,1),
                    new_fits=sum(r['new_fits'] for r in receipts),failed_variants=sum(r['status']!='PASS' for r in allrows))),flush=True)
    label='pilot_' if pilot else ''
    records=[]
    for row in allrows:
        records.append({k:json.dumps(design.plain(v),allow_nan=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})
    design.save(label+'controlled_refit_variants.csv',records);design.save(label+'case_receipts.csv',receipts)
    for path,h in hashes.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h,path
    audit=dict(status='PASS_CONTROLLED_REFITS' if all(r['status']=='PASS' for r in allrows) else 'PARTIAL_CONTROLLED_REFITS',
        cases=len(receipts),variant_rows=len(allrows),new_GP_fits=sum(r['new_fits'] for r in receipts),
        cache_hits=sum(r['cache_hits'] for r in receipts),failed_variants=sum(r['status']!='PASS' for r in allrows),
        elapsed_seconds=time.perf_counter()-start,workers=workers,new_experiments=0,new_LLM_calls=0)
    design.write(OUT/(label+'refit_audit.json'),audit);print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
