"""A feature-equality additive/interaction GP activated after measured collapse."""
from hashlib import sha256
from pathlib import Path
import math
import numpy as np
import torch
from filelock import FileLock
from gpytorch.kernels import Kernel,ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from botorch.optim.fit import fit_gpytorch_mll_scipy
from . import stable_r2_fit as native
from . import reasoning_bo_data as data
from .suzuki_domain import digest

PARAMETERS={'additive_weight':.75,'interaction_weight':.25,'interaction_lengthscale':.2,'latch_after_collapse':True}


class AdditiveInteractionKernel(Kernel):
    def forward(self,x1,x2,diag=False,last_dim_is_batch=False,**kwargs):
        if last_dim_is_batch:raise ValueError('unsupported batch interpretation')
        matched=(x1.unsqueeze(-2)==x2.unsqueeze(-3)).to(x1)
        average=matched.mean(-1)
        result=.75*average+.25*torch.exp(-(1.-average)/.2)
        return torch.diagonal(result,dim1=-2,dim2=-1) if diag else result


def make_model(base_model):
    X=base_model.train_inputs[0]
    Y=base_model.outcome_transform.untransform(base_model.train_targets[:,None])[0].detach()
    model=MixedSingleTaskGP(X,Y,cat_dims=list(range(X.shape[-1])))
    model.covar_module=ScaleKernel(AdditiveInteractionKernel()).to(X)
    return model


class Cache(native.Cache):
    def __init__(self,pool,directory,fit_mode='additive'):
        super().__init__(pool,directory)
        self.additive_recipe=digest({'source':sha256(Path(__file__).read_bytes()).hexdigest(),'params':PARAMETERS,'base':self.robust_recipe})
        self.latched=False;self.additive_values={}

    def model(self,history):
        basekey,basevalue=super().model(history);before=self.last_health
        if before['health']['trigger']:self.latched=True
        if not self.latched:return basekey,basevalue
        key=digest({'fit':basekey,'additive':self.additive_recipe})
        if key in self.additive_values:
            v,record=self.additive_values[key];self.last_health=record;return key,v
        path=self.directory/(key+'.additive.pt')
        with FileLock(str(path.with_suffix('.additive.lock')),timeout=300):
            model=make_model(basevalue['model'])
            if path.exists():
                saved=torch.load(path,map_location='cpu',weights_only=False)
                assert saved['key']==key and saved['recipe']==self.additive_recipe
                model.load_state_dict(saved['state']);record=saved['record']
            else:
                native.base.nominal.seed_all(basevalue['fit_seed'])
                res=fit_gpytorch_mll_scipy(ExactMarginalLogLikelihood(model.likelihood,model),options={'maxiter':200})
                loss=native.native_loss(model)
                if not math.isfinite(loss) or 'SUCCESS' not in str(res.status):raise ValueError('additive fit did not converge to a finite objective')
                record={'health':before['health'],'native_record':before,'used_robust':True,'model_kind':'additive_interaction',
                    'native_loss_per_datum':before['selected_loss'],'selected_loss':loss,
                    'total_MLL_gain':(before['selected_loss']-loss)*len(history),
                    'selection_reason':'predeclared structural regularization latched after collapse; not selected by oracle or marginal-likelihood comparison',
                    'latched':True,'status':str(res.status),'parameters':PARAMETERS,'selected_key':key}
                tmp=path.with_suffix('.tmp');torch.save({'key':key,'recipe':self.additive_recipe,'record':record,'state':model.state_dict()},tmp);tmp.replace(path)
                data.write(path.with_suffix('.json'),record)
            model.eval();v={**basevalue,'model':model};self.additive_values[key]=(v,record);self.last_health=record
            return key,v

    def release(self):
        super().release();self.additive_values.clear()
