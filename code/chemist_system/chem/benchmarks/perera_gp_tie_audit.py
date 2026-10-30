"""Read-only campaign audit: categorical-GP collapse versus acquisition errors.

Scratch copies of frozen models are loaded; no original trace/cache is written.
Diagnostic refits consume only a six-observation prefix, never an oracle table.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import json
import math
from pathlib import Path
import warnings
import numpy as np
import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.acquisition.logei import qLogExpectedImprovement, _log_improvement
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from botorch.optim.fit import fit_gpytorch_mll_scipy
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import LogNormalPrior

from . import pointwise_hte_engine as engine
from . import pointwise_hte_study as common
from . import reasoning_bo_data as data
from . import r2_tuning_study_v3_shuffle as protected

OUT = Path('research/perera_mixed_gp_tie_audit_20261029')
SCRATCH = Path('data/runs/perera_gp_tie_audit_20261029')
ROOT = Path('data/runs/hte_r2_tuning_20261028')
VARIANTS = ('focused_countercheck_v3', 'focused_countercheck_v3_shuffle_s0')


def manual(model, X):
    """Independent NumPy exact-GP posterior and categorical covariance."""
    train = model.train_inputs[0].detach().numpy()
    xx = X.detach().numpy()
    y = model.train_targets.detach().numpy()
    ls = model.covar_module.base_kernel.lengthscale.detach().numpy().reshape(-1)
    scale = model.covar_module.outputscale.item()
    noise = model.likelihood.noise.item()
    c = model.mean_module.constant.item()
    logcorr = -np.mean((xx[:, None, :] != train[None, :, :])/ls, axis=-1)
    k = scale*np.exp(-np.mean((train[:, None, :] != train[None, :, :])/ls, axis=-1))
    cross = scale*np.exp(logcorr)
    A = k + noise*np.eye(len(train))
    chol = np.linalg.cholesky(A)
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y-c))
    mu = c + cross@alpha
    solve = np.linalg.solve(chol, cross.T)
    variance = np.maximum(scale - np.sum(solve**2, axis=0), 0.)
    om = model.outcome_transform.means.item()
    os = model.outcome_transform.stdvs.item()
    with torch.no_grad():
        native_k = model.covar_module(model.train_inputs[0]).to_dense().numpy()
    residual = y-c
    data_logml = -.5*(residual@alpha + 2*np.log(np.diag(chol)).sum() + len(y)*math.log(2*math.pi))
    return {'mu': mu*os+om, 'var': variance*os**2, 'logcorr': logcorr,
            'kernel_error': float(np.max(np.abs(k-native_k))),
            'condition': float(np.linalg.cond(A)), 'data_logml': float(data_logml),
            'prior_mu': c*os+om, 'prior_sd': math.sqrt(scale)*os,
            'internal_K': k, 'internal_cross': cross, 'alpha': alpha, 'chol': chol}


def stats(model, X):
    with torch.no_grad():
        ps = [model.posterior(x.unsqueeze(1)) for x in X.split(256)]
        mu = torch.cat([p.mean.reshape(-1) for p in ps]).numpy()
        sd = torch.cat([p.variance.reshape(-1).sqrt() for p in ps]).numpy()
    return mu, sd


def tensor_scores(acq, X, chunk=256):
    with torch.no_grad():
        return torch.cat([acq(x.unsqueeze(1)).reshape(-1) for x in X.split(chunk)]).numpy()


def aq_audit(cache, hist, v, saved, mask, meta):
    model, X = v['model'], cache.X
    effective = engine.nominal.JitterModel(model)
    mc = meta['mc_seed']
    acq = qLogExpectedImprovement(effective, best_f=v['best_std'],
        sampler=SobolQMCNormalSampler(torch.Size([512]), seed=mc))
    scores = tensor_scores(acq, X)
    scores128 = tensor_scores(acq, X, 128)
    st = cache.stats(hist)
    z = acq.sampler.base_samples.reshape(512, -1)
    assert z.shape[1] == 1
    with torch.no_grad():
        samples = torch.as_tensor(st['mu_std'])[None, :] + z[:, :1]*torch.as_tensor(np.sqrt(st['sd_std']**2+1e-6))[None, :]
        log_imp = _log_improvement(samples, acq.best_f, tau=acq.tau_relu, fat=True)
        hand_mc = (torch.logsumexp(log_imp, dim=0)-math.log(512)).numpy()
    analytic = LogExpectedImprovement(effective, best_f=torch.tensor(v['best_std'], dtype=torch.double))
    exact = tensor_scores(analytic, X)
    acq_double = qLogExpectedImprovement(effective, best_f=torch.tensor(v['best_std'], dtype=torch.double),
        sampler=SobolQMCNormalSampler(torch.Size([512]), seed=mc))
    double_scores = tensor_scores(acq_double, X)
    result = {'cached_vs_recomputed_max_error': float(np.max(np.abs(scores-saved))),
        'chunk256_vs128_max_error': float(np.max(np.abs(scores-scores128))),
        'manual_marginal_MC_max_error': float(np.max(np.abs(scores-hand_mc))),
        'MC512_vs_analytic_max_error': float(np.max(np.abs(double_scores[mask]-exact[mask]))),
        'MC512_vs_analytic_median_error': float(np.median(np.abs(double_scores[mask]-exact[mask]))),
        'MC_top_ties': int(np.sum(mask & (scores==scores[mask].max()))),
        'analytic_top_ties': int(np.sum(mask & (exact==exact[mask].max()))),
        'MC_unique': len(np.unique(scores[mask])), 'analytic_unique': len(np.unique(exact[mask])),
        'best_f_dtype': str(acq.best_f.dtype), 'best_f_double': v['best_std'],
        'best_f_stored': acq.best_f.item(),
        'float32_bestf_vs_double_score_max_error': float(np.max(np.abs(scores-double_scores))),
        'standardized_incumbent_maps_back_to_observed_error': abs(v['best_std']*v['std']+v['mean']-max(h['y'] for h in hist))}
    assert result['cached_vs_recomputed_max_error'] < 1e-9
    assert result['chunk256_vs128_max_error'] < 1e-9
    assert result['manual_marginal_MC_max_error'] < 1e-9
    return result


def objective(model):
    model.train();model.likelihood.train();model.zero_grad()
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    loss = -mll(model(*model.train_inputs), model.train_targets)
    loss.backward()
    gradients = {name: float(p.grad.norm()) for name,p in model.named_parameters() if p.grad is not None}
    model.eval();model.likelihood.eval()
    return {'negative_MLL_with_native_priors_per_datum': loss.item(), 'raw_parameter_gradient_norms': gradients}


def model_summary(model, X, mask):
    mu, sd = stats(model, X);p=manual(model,X)
    aq = LogExpectedImprovement(engine.nominal.JitterModel(model),
        best_f=torch.tensor(float(model.outcome_transform.untransform(model.train_targets[:, None])[0].max()), dtype=torch.double))
    scores=tensor_scores(aq,X)
    return {'lengthscales': model.covar_module.base_kernel.lengthscale.detach().reshape(-1).tolist(),
        'outputscale': model.covar_module.outputscale.item(), 'noise_internal': model.likelihood.noise.item(),
        'data_log_marginal_likelihood_internal': p['data_logml'],
        'unique_mu': len(np.unique(mu[mask])), 'unique_sd': len(np.unique(sd[mask])),
        'analytic_top_ties': int(np.sum(mask & (scores==scores[mask].max()))),
        'uncorrelated_fraction': float(np.mean(p['logcorr'][mask].max(axis=1)<math.log(1e-12)))}


def probes(task, pool, state, cache):
    """Fitting diagnostics only, all use the identical first six observations."""
    hist=state['history'][:6];_,v=cache.model(hist);original=v['model']
    x=original.train_inputs[0];ys=original.outcome_transform.untransform(original.train_targets[:,None])[0].detach()
    mask=~np.isin(pool['ids'],[h['candidate_id'] for h in hist])
    out={'task':task,'query':3,'observation_count':6,'only_observed_prefix':deepcopy(hist),
        'cached':model_summary(original,cache.X,mask),'cached_objective':objective(original),
        'diagnostic_start_rule':'three fixed log-spaced lengthscale starts0.05,0.5,5; no candidate truths or outcome-based choice',
        'refits':[]}
    for start in (.05,.5,5.):
        model=MixedSingleTaskGP(x,ys,cat_dims=list(range(x.shape[1])))
        model.covar_module.base_kernel.lengthscale=start
        mll=ExactMarginalLogLikelihood(model.likelihood,model)
        with warnings.catch_warnings(record=True) as ws:
            result=fit_gpytorch_mll_scipy(mll,options={'maxiter':200})
        model.eval()
        row={'initial_lengthscale':start,'status':str(result.status),'message':str(result.message),
            'steps':result.step,'warnings':[str(w.message) for w in ws],
            **model_summary(model,cache.X,mask),**objective(model)}
        out['refits'].append(row)
    model=MixedSingleTaskGP(x,ys,cat_dims=list(range(x.shape[1])))
    model.covar_module.base_kernel.register_prior('audit_lengthscale_prior',LogNormalPrior(math.log(.2),.75),'lengthscale')
    model.covar_module.base_kernel.lengthscale=.2
    mll=ExactMarginalLogLikelihood(model.likelihood,model)
    result=fit_gpytorch_mll_scipy(mll,options={'maxiter':200});model.eval()
    out['MAP_probe']={'prior':'each ARD lengthscale LogNormal(log0.2,0.75), single fixed start0.2; no oracle tuning',
        'status':str(result.status),'message':str(result.message),**model_summary(model,cache.X,mask),**objective(model)}
    data.write(OUT/(task+'_q3_fitting_probes.json'),out)
    return out


def scan(task):
    common.warm();pool=data.load_pool(task);rows=[]
    for variant in VARIANTS:
        folder=ROOT/variant/'runs'/(task+'_s0_'+variant);state=data.read(folder/'state.json')
        cache=engine.Cache(pool,SCRATCH/'cache'/task)
        cache.sources=[ROOT/v/'cache'/task for v in VARIANTS]+[protected.trust.ROOT/'cache'/task,common.ROOT/'cache'/task]
        for t in range(1,17):
            hist=state['history'][:t+3];fit,v=cache.model(hist);model=v['model']
            target=folder/'rounds'/f'{t:02d}_committed.json';commit=data.read(target)
            assert fit==commit['decision']['base_meta']['fit_key']
            with np.load(target.with_name(f'{t:02d}_candidates.npz')) as z:
                arrays={k:z[k].copy() for k in z.files}
            mask=arrays['available'];base=arrays['base_qLogEI'];guided=arrays['guided_qLogEI']
            st=cache.stats(hist);p=manual(model,cache.X)
            err_mu=float(np.max(np.abs(p['mu']-st['mu_std'])))
            err_var=float(np.max(np.abs(p['var']-st['sd_std']**2)))
            assert err_mu<1e-9 and err_var<1e-9 and p['kernel_error']<1e-12
            assert np.max(np.abs(st['mu_y']-arrays['mu_y']))<1e-10
            assert np.max(np.abs(st['sd_y']-arrays['sd_y']))<1e-10
            # Arbitrary bijections on category codes must leave covariance unchanged.
            xx=cache.X.detach().numpy();train=model.train_inputs[0].detach().numpy()
            ls=model.covar_module.base_kernel.lengthscale.detach().reshape(-1).numpy()
            relabeled=-np.mean(((31.-7.*xx)[:,None,:]!=(31.-7.*train)[None,:,:])/ls,axis=-1)
            assert np.array_equal(relabeled,p['logcorr'])
            signature=(xx[:,None,:]==train[None,:,:]).reshape(len(xx),-1)
            signatures=len(np.unique(np.packbits(signature[mask],axis=1),axis=0))
            prior_mu=p['prior_mu']*v['std']+v['mean'];prior_sd=p['prior_sd']*v['std']
            row={'task':task,'variant':variant,'query':t,'n':len(hist),'fit_key':fit,
                'commit_sha256':data.filehash(target),'lengthscales':dict(zip(pool['axes'],ls.tolist())),
                'outputscale':model.covar_module.outputscale.item(),'noise_internal':model.likelihood.noise.item(),
                'constant_internal':model.mean_module.constant.item(),'external_mean_y':v['mean'],'external_sd_y':v['std'],
                'outcome_mean':model.outcome_transform.means.item(),'outcome_sd':model.outcome_transform.stdvs.item(),
                'priors':[name for name,*_ in model.named_priors()],
                'available':int(mask.sum()),'unique_mu':len(np.unique(st['mu_y'][mask])),
                'unique_sd':len(np.unique(st['sd_y'][mask])),'unique_base':len(np.unique(base[mask])),
                'unique_base_rounded10':len(np.unique(np.round(base[mask],10))),
                'unique_guided':len(np.unique(guided[mask])),
                'base_top_ties':int(np.sum(mask & (base==base[mask].max()))),
                'guided_top_ties':int(np.sum(mask & (guided==guided[mask].max()))),
                'train_equality_signature_classes':signatures,
                'all_crosscov_exact_zero':int(np.sum(mask & (p['internal_cross']==0).all(axis=1))),
                'uncorrelated_fraction':float(np.mean(p['logcorr'][mask].max(axis=1)<math.log(1e-12))),
                'prior_mu_percent':prior_mu*100,'prior_latent_SD_pp':prior_sd*100,
                'exact_prior_mean_candidates':int(np.sum(mask & (st['mu_y']==prior_mu))),
                'manual_posterior_mu_max_error_std':err_mu,'manual_posterior_var_max_error_std':err_var,
                'manual_kernel_max_error':p['kernel_error'],'train_Knoise_condition':p['condition'],
                'selected_id':state['decisions'][t-1]['selected_id'],
                'selected_observed_percent':state['decisions'][t-1]['y']*100,
                'selected_mode':state['decisions'][t-1]['execution_mode'],
                'base_meta':commit['decision']['base_meta']}
            row['collapse_guard']=(row['base_top_ties']/row['available']>=.9 and row['uncorrelated_fraction']>=.9)
            if t in (1,3,6,16):row['acquisition_audit']=aq_audit(cache,hist,v,base,mask,row['base_meta'])
            rows.append(row);cache.release()
        assert cache.counters['fits']==0
        if variant==VARIANTS[0]:probe=probes(task,pool,state,cache)
        assert cache.counters['fits']==0
    data.write(OUT/(task+'_rounds.json'),rows)
    return {'task':task,'rounds':len(rows),'guard_queries':{v:[r['query'] for r in rows if r['variant']==v and r['collapse_guard']] for v in VARIANTS},
            'all_posterior_checks':'PASS','probes':probe}


def main():
    kept=protected.preservation();OUT.mkdir(parents=True,exist_ok=True)
    data.write(OUT/'audit_plan.json',{'scope':'restore64 frozen models; independent categorical kernel/posterior checks;16 acquisition checks; six diagnostic unregularized restarts and two MAP probes',
        'MC_checks':'cached vs fresh, chunk invariance, manual marginal MC, analytic q1 EI, best-f units/dtype',
        'no_new_campaigns':True,'no_oracle_loaded':True,'no_LLM_calls':True,'CPU_workers':2,'threads_per_worker':1,
        'operator_compiles_in_loop':0,'protected_releases':kept})
    with ProcessPoolExecutor(max_workers=2,initializer=common.warm) as ex:
        results=[]
        for f in as_completed([ex.submit(scan,task) for task in data.SPECS]):
            result=f.result();results.append(result);print(json.dumps({k:v for k,v in result.items() if k!='probes'}),flush=True)
    rows=[r for task in data.SPECS for r in data.read(OUT/(task+'_rounds.json'))]
    result={'status':'PASS','rounds':len(rows),'acquisition_checks':sum('acquisition_audit' in r for r in rows),
        'results':results,'max_manual_mu_error':max(r['manual_posterior_mu_max_error_std'] for r in rows),
        'max_manual_variance_error':max(r['manual_posterior_var_max_error_std'] for r in rows),
        'protected_before':kept,'protected_after':protected.preservation()}
    assert result['protected_before']==result['protected_after']
    data.write(OUT/'results.json',result)
    print(json.dumps({'status':'PASS','rounds':len(rows),'acquisition_checks':result['acquisition_checks']},indent=2))


if __name__=='__main__':main()
