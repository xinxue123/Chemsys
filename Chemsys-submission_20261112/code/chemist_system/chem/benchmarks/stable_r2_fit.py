"""Health-triggered restart of the same nominal Mixed GP, in isolated caches."""
from copy import deepcopy
from hashlib import sha256
import math
from pathlib import Path
import random
import warnings
import numpy as np
import torch
from filelock import FileLock
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim.fit import fit_gpytorch_mll_scipy
from gpytorch.mlls import ExactMarginalLogLikelihood
from . import pointwise_hte_engine as base
from . import reasoning_bo_data as data
from .suzuki_domain import digest

PARAMETERS={'top_tie_fraction':.9,'uncorrelated_fraction':.9,'max_train_correlation':1e-12,
    'starts':[.05,5.],'maxiter':200,'total_MLL_improvement':.001}


def native_loss(model):
    model.train();model.likelihood.train()
    with torch.no_grad():loss=float(-ExactMarginalLogLikelihood(model.likelihood,model)(model(*model.train_inputs),model.train_targets))
    model.eval();model.likelihood.eval()
    return loss


def assess(model,value,X,available,mc):
    with torch.no_grad():
        acq=qLogExpectedImprovement(base.nominal.JitterModel(model),best_f=value['best_std'],
            sampler=SobolQMCNormalSampler(torch.Size([512]),seed=mc))
        scores=torch.cat([acq(xx.unsqueeze(1)).reshape(-1) for xx in X.split(256)]).numpy()
    xx=X.numpy();train=model.train_inputs[0].numpy();ls=model.covar_module.base_kernel.lengthscale.detach().reshape(-1).numpy()
    logcorr=-np.mean((xx[:,None,:]!=train[None,:,:])/ls,axis=-1)
    ties=int(np.sum(available & (scores==scores[available].max())))
    uncorrelated=float(np.mean(logcorr[available].max(axis=1)<math.log(PARAMETERS['max_train_correlation'])))
    return {'available':int(available.sum()),'top_ties':ties,'top_tie_fraction':ties/int(available.sum()),
        'uncorrelated_fraction':uncorrelated,'trigger':ties/int(available.sum())>=PARAMETERS['top_tie_fraction'] and uncorrelated>=PARAMETERS['uncorrelated_fraction']}


class Cache(base.Cache):
    def __init__(self,pool,directory,fit_mode='robust'):
        super().__init__(pool,directory)
        assert fit_mode=='robust'
        self.robust_recipe=digest({'source':sha256(Path(__file__).read_bytes()).hexdigest(),'params':PARAMETERS,'base_recipe':self.recipe})
        self.recovered={};self.last_health=None

    def model(self,history):
        basekey,value=super().model(history)
        key=digest({'base':basekey,'robust':self.robust_recipe})
        if key in self.recovered:
            chosen,v,record=self.recovered[key];self.last_health=record;return chosen,v
        path=self.directory/(key+'.health.json')
        with FileLock(str(path.with_suffix('.health.lock')),timeout=300):
            if path.exists():
                record=data.read(path);assert record['base_key']==basekey and record['robust_recipe']==self.robust_recipe
            else:
                npstate=np.random.get_state();tstate=torch.random.get_rng_state();pystate=random.getstate()
                try:
                    available=~np.isin(self.ids,[h['candidate_id'] for h in history])
                    mc=38102026+self.task_seed*100+len(history)-3
                    health=assess(value['model'],value,self.X,available,mc)
                    loss=native_loss(value['model']);best=loss;selected=None;attempts=[]
                    if health['trigger']:
                        X=value['model'].train_inputs[0]
                        Y=value['model'].outcome_transform.untransform(value['model'].train_targets[:,None])[0].detach()
                        for start in PARAMETERS['starts']:
                            base.nominal.seed_all(value['fit_seed'])
                            candidate=MixedSingleTaskGP(X,Y,cat_dims=list(range(X.shape[1])))
                            candidate.covar_module.base_kernel.lengthscale=start
                            with warnings.catch_warnings(record=True) as ws:
                                res=fit_gpytorch_mll_scipy(ExactMarginalLogLikelihood(candidate.likelihood,candidate),options={'maxiter':PARAMETERS['maxiter']})
                            candidate.eval();current=native_loss(candidate)
                            attempts.append({'start':start,'loss_per_datum':current,'status':str(res.status),'warnings':[str(w.message) for w in ws]})
                            if math.isfinite(current) and current<best and 'SUCCESS' in str(res.status):
                                best=current;selected=deepcopy(candidate.state_dict())
                    gain=(loss-best)*len(history)
                    used=selected is not None and gain>PARAMETERS['total_MLL_improvement']
                    record={'base_key':basekey,'robust_recipe':self.robust_recipe,'health':health,'base_loss':loss,
                        'selected_loss':best if used else loss,'total_MLL_gain':gain if used else 0.,'used_robust':used,
                        'attempts':attempts,'selected_key':key if used else basekey,'observation_count':len(history)}
                    if used:
                        temporary=self.directory/(key+'.robust.tmp');torch.save({'state':selected,'record':record},temporary)
                        temporary.replace(self.directory/(key+'.robust.pt'))
                    data.write(path,record)
                finally:
                    np.random.set_state(npstate);torch.random.set_rng_state(tstate);random.setstate(pystate)
            if record['used_robust']:
                m=value['model'];Y=m.outcome_transform.untransform(m.train_targets[:,None])[0].detach()
                npstate=np.random.get_state();tstate=torch.random.get_rng_state();pystate=random.getstate()
                selected=MixedSingleTaskGP(m.train_inputs[0],Y,cat_dims=list(range(self.X.shape[1])))
                np.random.set_state(npstate);torch.random.set_rng_state(tstate);random.setstate(pystate)
                saved=torch.load(self.directory/(key+'.robust.pt'),map_location='cpu',weights_only=False)
                assert saved['record']==record;selected.load_state_dict(saved['state']);selected.eval()
                chosen=key;v={**value,'model':selected}
            else:chosen=basekey;v=value
            self.recovered[key]=(chosen,v,record);self.last_health=record
            return chosen,v

    def release(self):
        super().release();self.recovered.clear()
