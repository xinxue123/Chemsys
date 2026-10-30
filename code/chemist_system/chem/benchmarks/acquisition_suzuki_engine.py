"""Controlled q=1 acquisition comparison; the frozen GP and native tilt are reused."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch
from botorch.acquisition.analytic import UpperConfidenceBound, LogExpectedImprovement

from . import guarded_suzuki_engine as common
from .uncertain_suzuki_engine import UncertaintyCache
from .suzuki_domain import digest

KAPPAS=(.5,1.,2.)


def beta_at(mode,iteration,kappa=None):
    if not 1<=iteration<=16:raise ValueError('iteration must be 1..16')
    if mode=='official':return float(2*np.log(iteration*4*np.pi**2/.1*6))
    if mode=='fixed' and kappa is not None and kappa>0:return float(kappa**2)
    raise ValueError('invalid UCB configuration')


def equal_capacity_mask(scores,available,count,ids):
    """Rank gate for offline diagnostics only; this is not native LLINBO psi."""
    indices=np.flatnonzero(available)
    if not 0<=count<=len(indices):raise ValueError('capacity outside available pool')
    chosen=sorted(indices,key=lambda i:(-float(scores[i]),ids[i]))[:count]
    result=np.zeros(len(scores),dtype=bool);result[chosen]=True
    return result


class AcquisitionCache(UncertaintyCache):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.acquisition_recipe=sha256(Path(__file__).read_bytes()).hexdigest()
        self.components_memory={}
        self.counters.update(component_evaluations=0,component_cache_hits=0)

    def effective_model(self,history,box=None,confidence=.7):
        key,value=self.model(history)
        mc=38102026+self.case*10000+self.seed*100+len(history)-3
        common.seed_all(mc)
        tilt=self.boo.LinearExponentialRegionalMeanTiltPlugAndPlay(
            bounds=torch.tensor([[0.]*4,[1.]*4],dtype=torch.double),grid_size=512,
            smooth=.08,dtype=torch.double,device=torch.device('cpu'))
        tilt.set_box_region(*(box if box else ([0.]*4,[1.]*4)))
        delta=self.decide.confidence_to_delta(confidence,scale=3.) if box else 0.
        tilt.fit_lambda_by_delta(value['model'],delta,observation_noise=False)
        tilt.prepare_cache(value['model'])
        return self.boo.TiltedModel(value['model'],tilt).eval(),value,{
            'fit_key':key,'regional_seed':mc,'box':box,'confidence':confidence if box else None,
            'delta':float(delta),'lambda':float(tilt._lam),'common_covariance_jitter':1e-6}

    def components(self,history,box=None,confidence=.7,analytic_ei=False):
        fit,_=self.model(history)
        key=digest({'fit':fit,'box':box,'confidence':confidence if box else None,
                    'analytic_ei':analytic_ei,'recipe':self.acquisition_recipe})
        if key in self.components_memory:return self.components_memory[key]
        path=self.directory/(key+'.npz');self.ensure(path.name)
        if path.exists():
            with np.load(path,allow_pickle=False) as data:
                result={k:data[k].copy() for k in data.files if k!='metadata'}
                result['metadata']=json.loads(str(data['metadata']))
            assert result['metadata']['key']==key
            self.counters['component_cache_hits']+=1
        else:
            effective,value,meta=self.effective_model(history,box,confidence)
            means=[];sds=[];logs=[]
            acq=LogExpectedImprovement(effective,best_f=value['best_std']) if analytic_ei else None
            with torch.no_grad():
                for x in self.z.split(256):
                    p=effective.posterior(x.unsqueeze(1),observation_noise=False)
                    means.append(p.mean.reshape(-1));sds.append(p.variance.reshape(-1).sqrt())
                    if acq is not None:logs.append(acq(x.unsqueeze(1)).reshape(-1))
            result={'mu':torch.cat(means).numpy(),'sd':torch.cat(sds).numpy()}
            if acq is not None:result['analytic_logei']=torch.cat(logs).numpy()
            meta.update(key=key,recipe=self.acquisition_recipe,best_std=value['best_std'])
            result['metadata']=meta
            assert all(np.isfinite(v).all() for k,v in result.items() if k!='metadata')
            with path.with_suffix('.tmp').open('wb') as h:
                np.savez_compressed(h,**{k:v for k,v in result.items() if k!='metadata'},metadata=json.dumps(meta))
            path.with_suffix('.tmp').replace(path);self.counters['component_evaluations']+=1
        self.components_memory[key]=result;return result

    def ucb(self,history,beta,box=None,confidence=.7):
        value=self.components(history,box,confidence)
        scores=value['mu']+np.sqrt(beta)*value['sd']
        return scores,dict(value['metadata'],beta=float(beta),kappa=float(np.sqrt(beta)))
