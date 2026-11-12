"""Molecular similarity on nominal inputs; no category-code metric."""
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
from . import stable_structure_features as features
from . import reasoning_bo_data as data
from .suzuki_domain import digest

PARAMETERS={'chemical_weight':.75,'interaction_weight':.25,'interaction_lengthscale':.2,
    'min_lengthscale_max':1e-4,'lengthscale_ratio_min':1e6,'require_zero_isolated_pairs':True,'latch':True}


class MolecularCategoricalKernel(Kernel):
    def __init__(self,matrices,**kwargs):
        super().__init__(**kwargs);self.dimensions=len(matrices)
        for j,m in enumerate(matrices):self.register_buffer(f'similarity_{j}',torch.as_tensor(m,dtype=torch.double))

    def forward(self,x1,x2,diag=False,last_dim_is_batch=False,**kwargs):
        if last_dim_is_batch:raise ValueError('unsupported batch interpretation')
        chemical=0.
        for j in range(self.dimensions):
            matrix=getattr(self,f'similarity_{j}');i=x1[...,j].long();k=x2[...,j].long()
            chemical=chemical+matrix[i.unsqueeze(-1),k.unsqueeze(-2)]/self.dimensions
        matched=(x1.unsqueeze(-2)==x2.unsqueeze(-3)).to(x1).mean(-1)
        result=.75*chemical+.25*torch.exp(-(1.-matched)/.2)
        return torch.diagonal(result,dim1=-2,dim2=-1) if diag else result


def diagnose(model,axes):
    x=model.train_inputs[0].detach().numpy();ls=model.covar_module.base_kernel.lengthscale.detach().reshape(-1).numpy()
    contrasts={a:sum([axes[j] for j in range(len(axes)) if x[u,j]!=x[v,j]]==[a] for u in range(len(x)) for v in range(u)) for a in axes}
    critical=axes[int(np.argmin(ls))]
    return {'lengthscales':dict(zip(axes,ls.tolist())),'isolated_pairs':contrasts,'critical_axis':critical,
        'trigger':bool(ls.min()<=1e-4 and ls.max()/ls.min()>=1e6 and contrasts[critical]==0)}


class Cache(native.Cache):
    def __init__(self,pool,directory,fit_mode='structure'):
        super().__init__(pool,directory)
        path=features.ROOT/(pool['task_id']+'.json');catalog=data.read(path)
        q=dict(catalog);h=q.pop('feature_sha256');assert digest(q)==h
        assert catalog['parameters']==features.PARAMETERS
        for a in pool['axes']:assert [r['input'] for r in catalog['axes'][a]['levels']]==pool['levels'][a]
        self.matrices=[catalog['axes'][a]['Tanimoto'] for a in pool['axes']]
        self.structure_recipe=digest({'source':sha256(Path(__file__).read_bytes()).hexdigest(),'params':PARAMETERS,
            'features':data.filehash(path),'native':self.robust_recipe})
        self.latched=False;self.structure_values={}

    def make_model(self,base_model):
        x=base_model.train_inputs[0];y=base_model.outcome_transform.untransform(base_model.train_targets[:,None])[0].detach()
        model=MixedSingleTaskGP(x,y,cat_dims=list(range(x.shape[-1])))
        model.covar_module=ScaleKernel(MolecularCategoricalKernel(self.matrices)).to(x)
        return model

    def model(self,history):
        basekey,basevalue=super().model(history);before=self.last_health
        identification=diagnose(basevalue['model'],self.pool['axes'])
        if identification['trigger']:self.latched=True
        if not self.latched:return basekey,basevalue
        key=digest({'fit':basekey,'structure':self.structure_recipe})
        if key in self.structure_values:
            v,record=self.structure_values[key];self.last_health=record;return key,v
        path=self.directory/(key+'.structure.pt')
        with FileLock(str(path.with_suffix('.structure.lock')),timeout=300):
            model=self.make_model(basevalue['model'])
            if path.exists():
                saved=torch.load(path,map_location='cpu',weights_only=False)
                assert saved['key']==key and saved['recipe']==self.structure_recipe
                model.load_state_dict(saved['state']);record=saved['record']
            else:
                native.base.nominal.seed_all(basevalue['fit_seed'])
                res=fit_gpytorch_mll_scipy(ExactMarginalLogLikelihood(model.likelihood,model),options={'maxiter':200})
                loss=native.native_loss(model)
                if not math.isfinite(loss) or 'SUCCESS' not in str(res.status):raise ValueError('structure GP fit failed to converge')
                record={'health':before['health'],'native_record':before,'identification':identification,'used_robust':True,
                    'model_kind':'molecular_categorical','native_loss_per_datum':before['selected_loss'],'selected_loss':loss,
                    'total_MLL_gain':(before['selected_loss']-loss)*len(history),'latched':True,'status':str(res.status),
                    'selection_reason':'predeclared molecular structural prior after an unidentifiable extreme axis; not oracle/model-score selection',
                    'parameters':PARAMETERS,'selected_key':key}
                tmp=path.with_suffix('.tmp');torch.save({'key':key,'recipe':self.structure_recipe,'record':record,'state':model.state_dict()},tmp);tmp.replace(path)
                data.write(path.with_suffix('.json'),record)
            model.eval();v={**basevalue,'model':model};self.structure_values[key]=(v,record);self.last_health=record
            return key,v

    def release(self):
        super().release();self.structure_values.clear()
