"""Nominal categorical Mixed GP adapter for the frozen KF/event/B4 policy.

The GP categorical column is never normalized or treated as a metric. Legacy
pool z encodes labels for the shared knowledge parser only (set membership).
"""
from hashlib import sha256
from pathlib import Path
import json

import numpy as np
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood

from . import kalman_knowledge_suzuki_engine as policy
from .suzuki_domain import CATALYSTS,digest

CATEGORY_MAP=tuple(range(8))


def encode_mixed(pool,category_map=CATEGORY_MAP):
    if len(category_map)!=8 or len(set(category_map))!=8:raise ValueError('category labels require a bijection')
    continuous=np.asarray(pool['z'],dtype=float)[:,:3].clip(1e-6,1-1e-6)
    labels=np.array([category_map[CATALYSTS.index(point['catalyst'])] for point in pool['points']],dtype=float)
    return np.column_stack([continuous,labels])


class MixedResidualMeanModel(Model):
    """Same residual mean/cap as before; nominal labels replace bin arithmetic."""
    def __init__(self,base,hypotheses,weights,retained,y_std,category_map=CATEGORY_MAP):
        super().__init__();self.base=base;self.hypotheses=hypotheses;self.weights=weights
        self.retained=set(retained);self.y_std=float(y_std);self.category_map=tuple(category_map)
    @property
    def num_outputs(self):return self.base.num_outputs
    @property
    def batch_shape(self):return self.base.batch_shape
    def posterior(self,X,observation_noise=False,**kwargs):
        p=self.base.posterior(X,observation_noise=observation_noise,**kwargs)
        latent=p if not observation_noise else self.base.posterior(X,observation_noise=False,**kwargs)
        change=torch.zeros_like(p.mvn.mean)
        for h in self.hypotheses:
            if h['id'] not in self.retained:continue
            if h['variable']=='catalyst':
                mask=torch.zeros_like(X[...,3],dtype=torch.bool)
                for k in h['categories']:mask|=X[...,3]==self.category_map[k]
            else:
                j=policy.VARIABLES.index(h['variable']);lo,hi=h['bounds'];mask=(X[...,j]>=lo)&(X[...,j]<=hi)
            change+=self.weights[h['id']]*(h['effect_pp']/100/self.y_std)*(mask.to(X)-h['pool_fraction'])
        cap=policy.PARAMETERS['mean_cap_sd']*latent.variance.squeeze(-1).clamp_min(1e-18).sqrt()
        return GPyTorchPosterior(MultivariateNormal(p.mvn.mean+change.clamp(min=-cap,max=cap),p.mvn.lazy_covariance_matrix.add_jitter(1e-6)))


class MixedKnowledgeCache:
    """Each exact prefix is fit once. Only Mixed-GP caches may be copied."""
    def __init__(self,pool,directory,case,seed=0,category_map=CATEGORY_MAP,sources=()):
        self.pool=pool;self.ids=tuple(pool['ids']);self.case=case;self.seed=seed;self.category_map=tuple(category_map)
        self.z=torch.tensor(encode_mixed(pool,category_map),dtype=torch.double)
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True);self.sources=[Path(p) for p in sources]
        self.engine_sha256=sha256(Path(__file__).read_bytes()).hexdigest()
        self.recipe=digest({'engine':self.engine_sha256,'GP':'botorch0.14.MixedSingleTaskGP','cat_dims':[3],
                            'category_map':list(self.category_map),'input_X_sha256':sha256(self.z.numpy().tobytes()).hexdigest()})
        self.extension_recipe=self.recipe;self.models={};self.scores={};self.posterior_memory={}
        self.counters={'fits':0,'fit_cache_hits':0,'acquisition_evaluations':0,'score_cache_hits':0,'posterior_evaluations':0,'copied':0,'knowledge_acquisitions':0}
    def ensure(self,name):
        target=self.directory/name
        if target.exists():return
        for folder in self.sources:
            source=folder/name
            if source.exists():
                import shutil
                temporary=target.with_suffix(target.suffix+'.copytmp');shutil.copyfile(source,temporary);temporary.replace(target)
                self.counters['copied']+=1;return
    def model(self,history):
        ordered=sorted(history,key=lambda h:h['candidate_id']);key=digest({'history':ordered,'case':self.case,'seed':self.seed,'recipe':self.recipe})
        if key in self.models:self.counters['fit_cache_hits']+=1;return key,self.models[key]
        lookup={k:i for i,k in enumerate(self.ids)};x=self.z[[lookup[h['candidate_id']] for h in ordered]]
        y=torch.tensor([[h['y']] for h in ordered],dtype=torch.double);mean=y.mean(0,keepdim=True);std=y.std(0,unbiased=False,keepdim=True).clamp_min(1e-12);ystd=(y-mean)/std
        fit_seed=28102026+self.case*10000+self.seed*100+len(history)-3;policy.common.seed_all(fit_seed)
        model=MixedSingleTaskGP(train_X=x,train_Y=ystd,cat_dims=[3])
        path=self.directory/(key+'.pt');self.ensure(path.name)
        if path.exists():
            saved=torch.load(path,map_location='cpu',weights_only=False)
            assert saved['key']==key and saved['gp_class']=='MixedSingleTaskGP' and saved['category_map']==list(self.category_map)
            model.load_state_dict(saved['state']);self.counters['fit_cache_hits']+=1
        else:
            fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood,model));self.counters['fits']+=1
            saved={'key':key,'state':model.state_dict(),'fit_seed':fit_seed,'gp_class':'MixedSingleTaskGP','cat_dims':[3],
                   'category_map':list(self.category_map),'train_X':x.tolist(),'train_candidate_ids':[h['candidate_id'] for h in ordered],
                   'recipe':self.recipe,'kernel':'Scale(RBF_cont1+Scale(Categorical1))+Scale(RBF_cont2*Categorical2)'}
            temporary=path.with_suffix('.tmp');torch.save(saved,temporary);temporary.replace(path)
        model.eval();value={'model':model,'best_std':float(ystd.max()),'mean':mean,'std':std,'fit_seed':fit_seed}
        self.models[key]=value;return key,value
    def score(self,history):return self.knowledge_score(history,[],{},())
    def knowledge_score(self,history,hypotheses,weights,retained):
        fit,value=self.model(history);mc=38102026+self.case*10000+self.seed*100+len(history)-3
        knowledge={'hypotheses':[{**policy.canonical(h),'pool_fraction':h['pool_fraction']} for h in hypotheses],
                   'weights':weights,'retained':sorted(retained)}
        key=digest({'fit':fit,'knowledge':knowledge,'mc_seed':mc,'recipe':self.recipe});path=self.directory/(key+'.npz')
        if key in self.scores:self.counters['score_cache_hits']+=1;return self.scores[key]
        self.ensure(path.name)
        if path.exists():
            with np.load(path,allow_pickle=False) as data:result=(data['scores'].copy(),json.loads(str(data['metadata'])))
            assert result[1]['key']==key;self.counters['score_cache_hits']+=1
        else:
            policy.common.seed_all(mc)
            model=MixedResidualMeanModel(value['model'],hypotheses,weights,retained,value['std'],self.category_map).eval()
            acq=qLogExpectedImprovement(model=model,best_f=value['best_std'],sampler=SobolQMCNormalSampler(torch.Size([512]),seed=mc))
            with torch.no_grad():scores=torch.cat([acq(x.unsqueeze(1)).reshape(-1) for x in self.z.split(256)]).numpy()
            assert len(scores)==len(self.ids) and np.isfinite(scores).all()
            meta={'key':key,'fit_key':fit,'mc_seed':mc,'mc_paths':512,'fit_seed':value['fit_seed'],'recipe':self.recipe,
                  'gp_class':'MixedSingleTaskGP','cat_dims':[3],'category_map':list(self.category_map),'knowledge':knowledge,'covariance_jitter':1e-6}
            temporary=path.with_suffix('.tmp')
            with temporary.open('wb') as f:np.savez_compressed(f,scores=scores,metadata=json.dumps(meta))
            temporary.replace(path);result=(scores,meta);self.counters['acquisition_evaluations']+=1
            if retained:self.counters['knowledge_acquisitions']+=1
        self.scores[key]=result;return result
    # This routine only calls the adapter's model()/ensure() and scales its
    # posterior. Keeping it verbatim avoids changing uncertainty semantics.
    posterior_stats=policy.UncertaintyCache.posterior_stats
