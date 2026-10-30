"""Replay the original four-point GP fit, recording optimizer iterations.

The historical cache kept final parameters, not an optimizer trace. This is an
explicitly labeled replay with identical inputs, seed and original fitting path.
No old files are modified, no oracle/API calls or BO trajectory reruns occur.
"""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
from hashlib import sha256
from pathlib import Path
import json

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.optim.fit import fit_gpytorch_mll_scipy
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

from chem.benchmarks import guarded_suzuki_engine as common

ROOT=Path('data/runs/suzuki_kalman_knowledge');OUT=Path('research/suzuki_case2_gp_fit_trace')


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def digest(p):return sha256(Path(p).read_bytes()).hexdigest()


def snapshot(model,mll,label,iteration=0,attempt=0):
    model.train();x=model.train_inputs[0];target=model.train_targets
    objective=-mll(model(x),target)
    params=list(model.named_parameters());derivatives=torch.autograd.grad(objective,[p for n,p in params])
    gradients={name:g.detach().reshape(-1).tolist() for (name,p),g in zip(params,derivatives)}
    with torch.no_grad():
        ls=model.covar_module.lengthscale.reshape(-1);noise=model.likelihood.noise.item();c=model.mean_module.constant.item()
        K=model.covar_module(x).to_dense();C=K+noise*torch.eye(len(target),dtype=torch.double);r=target-c
        alpha=torch.linalg.solve(C,r);quad=float(.5*r@alpha);logdet=float(.5*torch.linalg.slogdet(C)[1]);constant=float(len(target)/2*np.log(2*np.pi))
        priors={name:float(prior.log_prob(closure(module)).sum()) for name,module,prior,closure,setter in model.named_priors()}
        manual=(quad+logdet+constant-sum(priors.values()))/len(target)
        assert abs(manual-objective.item())<1e-12
        u=torch.linalg.solve(C,torch.ones_like(target));profile_c=float(u@target/u.sum())
        # Independent closed-form gradients, including the registered priors.
        B=torch.outer(alpha,alpha)-torch.linalg.inv(C)
        dl=[]
        for j in range(4):
            dC=K*(x[:,j,None]-x[None,:,j]).square()/ls[j]**3
            likelihood_grad=float(.5*(B*dC.T).sum())
            prior=next(prior for name,module,prior,closure,setter in model.named_priors() if 'lengthscale_prior' in name)
            prior_grad=float(-(1+(torch.log(ls[j])-prior.loc)/prior.scale**2)/ls[j])
            dl.append(-(likelihood_grad+prior_grad)/4)
        dn_likelihood=float(.5*torch.trace(B))
        nprior=next(prior for name,module,prior,closure,setter in model.named_priors() if 'noise_prior' in name)
        dn_prior=float(-(1+(np.log(noise)-float(nprior.loc))/float(nprior.scale)**2)/noise)
        dc=-float(alpha.sum())/4
        assert np.allclose(dl,gradients['covar_module.raw_lengthscale'],rtol=0,atol=1e-10)
        assert abs(dc-gradients['mean_module.raw_constant'][0])<1e-10
        assert abs(-(dn_likelihood+dn_prior)/4-gradients['likelihood.noise_covar.raw_noise'][0])<1e-10, {'closed':-(dn_likelihood+dn_prior)/4,'autograd':gradients['likelihood.noise_covar.raw_noise'][0],'prior':dn_prior,'likelihood':dn_likelihood,'noise':noise}
        return {'label':label,'attempt':attempt,'iteration':iteration,'lengthscales':ls.tolist(),'constant_internal':c,'noise_variance_internal':noise,
            'quadratic_data_term':quad,'half_logdet_term':logdet,'gaussian_constant':constant,'negative_log_likelihood':quad+logdet+constant,
            'log_priors':priors,'negative_log_posterior_per_observation':manual,'gradients_of_minimized_objective':gradients,
            'profile_optimal_c_for_fixed_covariance':profile_c,'profile_c_weights':(u/u.sum()).tolist(),
            'noise_log_likelihood_gradient':dn_likelihood,'noise_log_prior_gradient':dn_prior,
            'K':K.tolist(),'C':C.tolist(),'alpha':alpha.tolist()}


def main():
    torch.set_num_threads(1);common.modules('LGBO');__import__('boo');__import__('decide');__import__('prior_monte_carlo')
    torch.linalg.cholesky(torch.eye(4,dtype=torch.double))
    state_path=ROOT/'runs/c2_s0/state.json';pool_path=Path('data/runs/suzuki_guarded/inputs/case2/pool.json')
    state=read(state_path);pool=read(pool_path);event=state['decisions'][0];saved_path=ROOT/'cache/c2_s0'/(event['fit_key']+'.pt')
    source_hashes={p.as_posix():digest(p) for p in (state_path,pool_path,saved_path)}
    saved=torch.load(saved_path,map_location='cpu',weights_only=False);history=sorted(state['history'][:4],key=lambda h:h['candidate_id']);lookup={k:i for i,k in enumerate(pool['ids'])}
    x=torch.tensor(np.array(pool['z']).clip(1e-6,1-1e-6)[[lookup[h['candidate_id']] for h in history]],dtype=torch.double)
    y=torch.tensor([[h['y']] for h in history],dtype=torch.double);mean=y.mean(0,keepdim=True);std=y.std(0,unbiased=False,keepdim=True);ys=(y-mean)/std
    common.seed_all(saved['fit_seed']);model=SingleTaskGP(x,ys);mll=ExactMarginalLogLikelihood(model.likelihood,model)
    priors=[{'name':name,'loc':prior.loc.tolist(),'scale':prior.scale.tolist()} for name,module,prior,closure,setter in model.named_priors()]
    records=[];attempts=[]
    def optimizer(mll,closure=None,**kwargs):
        attempt=len(attempts)+1;attempts.append({'attempt':attempt})
        records.append(snapshot(model,mll,'initial' if attempt==1 else 'retry_initial',attempt=attempt))
        def callback(parameters,result):
            records.append(snapshot(model,mll,'accepted_iteration',int(result.step),attempt))
        result=fit_gpytorch_mll_scipy(mll,closure=closure,callback=callback,**kwargs)
        attempts[-1].update(status=str(result.status),steps=result.step,final_loss=result.fval,message=result.message)
        return result
    fit_gpytorch_mll(mll,optimizer=optimizer)
    final=snapshot(model,mll,'final',iteration=attempts[-1]['steps'],attempt=len(attempts));records.append(final)
    differences={}
    for name,value in model.state_dict().items():
        if not torch.is_floating_point(value):continue
        reference=saved['state'][name]
        if torch.equal(value,reference):differences[name]=0.;continue
        finite=torch.isfinite(value)
        assert torch.equal(finite,torch.isfinite(reference)) and torch.equal(value[~finite],reference[~finite])
        differences[name]=float((value[finite]-reference[finite]).abs().max()) if finite.any() else 0.
    maximum=max(differences.values());assert maximum<1e-10,maximum
    for p,expected in source_hashes.items():assert digest(p)==expected
    result={'scope':'Case2 seed0, original first4 observations; optimizer replay, not a historical trace or a new BO experiment.',
        'fit_seed':saved['fit_seed'],'inputs':source_hashes,'input_X':x.tolist(),'initial_y_percent':(y*100).reshape(-1).tolist(),
        'training_targets_internal':model.train_targets.tolist(),'external_y_mean_percent':mean.item()*100,'external_y_pop_sd_pp':std.item()*100,
        'effective_output_scale_pp':std.item()*model.outcome_transform.stdvs.item()*100,'priors':priors,
        'bounds':{'lengthscales_min':.025,'noise_variance_min':.0001,'constant':'unbounded'},
        'minimized_objective':'(0.5*r.T*C^-1*r +0.5*logdet(C) +n/2*log(2*pi) -sum(log_prior))/n; r=y_internal-c; n=4',
        'optimizer':'Original fit_gpytorch_mll dispatcher with fit_gpytorch_mll_scipy default L-BFGS-B; callback only records accepted iterations.',
        'attempts':attempts,'records':records,'replay_max_parameter_difference_from_frozen_model':maximum,
        'per_parameter_differences':differences,'source_hashes_unchanged':True,'new_LLM_calls':0,'new_oracle_queries':0,
        'new_GP_fits_for_diagnostic_replay':1,'code_sha256':digest(Path(__file__))}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'fit_trace.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    summary={'initial':records[0],'final':records[-1],'attempts':attempts,'replay_max_parameter_difference_from_frozen_model':maximum,
             'iterations':[{k:r[k] for k in ('iteration','lengthscales','constant_internal','noise_variance_internal','negative_log_posterior_per_observation')} for r in records if r['label']=='accepted_iteration']}
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
