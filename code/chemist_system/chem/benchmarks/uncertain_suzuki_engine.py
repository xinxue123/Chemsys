"""Explicit uncertainty mechanisms on the frozen common GP/qLogEI base."""
from hashlib import sha256
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.distributions import MultivariateNormal

from chem.decision.domain_guard import select_candidate
from . import guarded_suzuki_engine as common
from .suzuki_domain import digest

METHODS=('sigma_cap','credible_B4','boundary_probe','residual_B4')
CONTROLS=('sigma_probe','sigma_BO')
PARAMETERS={'sigma_cap':(.5,1.),'credible_B4':(1.,2.),'boundary_probe':(3,4),'residual_B4':(1.,2.)}
FALLBACK={'sigma_cap':1.,'credible_B4':2.,'boundary_probe':3,'residual_B4':2.}
PROBES={3:(4,8,12),4:(3,6,9,12)}


def residual_scale(events):
    """Prefix-only variance inflation heuristic, not conformal calibration."""
    if len(events)<3:return 1.
    return float(np.clip(np.sqrt(np.mean([min(float(e['predictive_z'])**2,100.) for e in events])),1.,3.))


def select(method,parameter,base,mu,sd,membership,budget,queried,ids,hids,guided,iteration,scale=1.):
    """No unseen outcomes or knowledge-quality labels enter the policy."""
    available=~np.asarray(queried,dtype=bool)
    reference=common.argmax(base,available,ids);eligible=available&((base[reference]-base)<=budget)
    retained=();detail=None;probe=False;fallback=False;potential=available.copy()
    if method=='sigma_cap':
        selected=common.argmax(guided('cap',parameter),eligible,ids);retained=tuple(sorted(hids))
    elif method in ('credible_B4','residual_B4'):
        k=float(parameter)*(scale if method=='residual_B4' else 1.)
        lower=mu-k*sd;upper=mu+k*sd;champion=float(lower[available].max())
        potential=upper>=champion
        # Each retained predicate also requires an optimistic witness. The
        # empty subset always keeps the base fallback available.
        effective=membership&potential[None,:]
        detail=select_candidate(base,effective,budget,candidate_ids=ids,hypothesis_ids=hids,queried_mask=queried)
        selected=detail.candidate_index;retained=detail.retained_ids
    elif method in ('boundary_probe','sigma_probe','sigma_BO'):
        probe=iteration in PROBES[int(parameter)]
        if probe:
            mask=eligible.copy()
            if method=='boundary_probe':
                mask&=(membership.sum(0)==len(hids)-1)
                if not mask.any():mask=eligible.copy();fallback=True
            selected=common.argmax(sd,mask,ids)
        else:selected=reference if method=='sigma_BO' else common.argmax(guided('full',None),eligible,ids)
        retained=() if method=='sigma_BO' else tuple(sorted(hids))
    else:raise ValueError(method)
    loss=float(base[reference]-base[selected]);assert loss<=budget and not queried[selected]
    return int(selected),{'reference_id':ids[reference],'selected_id':ids[selected],
        'base_logei_loss':loss,'budget_logei':float(budget),'retained_ids':list(retained),
        'revoked_ids':sorted(set(hids)-set(retained)),'probe_round':probe,'boundary_probe_fallback':fallback,
        'optimistic_candidates':int((potential&available).sum()),'optimistic_eligible_candidates':int((potential&eligible).sum()),
        'residual_scale':float(scale),'minimum_revocation_receipt':detail.to_dict() if detail else None}


class SigmaCappedModel(Model):
    """Preserve signed native mean changes and cap their absolute size."""
    def __init__(self,base,guided,kappa):
        super().__init__();self.base=base;self.guided=guided;self.kappa=float(kappa)
    @property
    def num_outputs(self):return self.base.num_outputs
    @property
    def batch_shape(self):return self.base.batch_shape
    def posterior(self,X,observation_noise=False,**kwargs):
        p0=self.base.posterior(X,observation_noise=observation_noise,**kwargs)
        p1=self.guided.posterior(X,observation_noise=observation_noise,**kwargs)
        change=p1.mvn.mean-p0.mvn.mean
        # Epistemic sd, independent of an observation-noise request.
        latent=p0 if not observation_noise else self.base.posterior(X,observation_noise=False,**kwargs)
        cap=self.kappa*latent.variance.squeeze(-1).clamp_min(1e-18).sqrt()
        mean=p0.mvn.mean+change.sign()*torch.minimum(change.abs(),cap)
        return GPyTorchPosterior(MultivariateNormal(mean,p1.mvn.lazy_covariance_matrix))


class UncertaintyCache(common.ScoreCache):
    """Read-through copies preserve every earlier study and reuse exact fits."""
    def __init__(self,*args,sources=(),**kwargs):
        super().__init__(*args,**kwargs);self.sources=[Path(p) for p in sources]
        self.posterior_memory={};self.extension_recipe=sha256(Path(__file__).read_bytes()).hexdigest()
        self.counters.update(copied=0,posterior_evaluations=0,capped_acquisition_evaluations=0)
    def ensure(self,name):
        target=self.directory/name
        if target.exists():return
        for folder in self.sources:
            source=folder/name
            if source.exists():
                temporary=target.with_suffix(target.suffix+'.copytmp')
                shutil.copyfile(source,temporary);temporary.replace(target)
                self.counters['copied']+=1;return
    def model(self,history):
        ordered=sorted(history,key=lambda r:r['candidate_id'])
        key=digest({'history':ordered,'case':self.case,'seed':self.seed,'recipe':self.recipe})
        self.ensure(key+'.pt');return super().model(history)
    def score(self,history,box=None,confidence=.7):
        fit,_=self.model(history);mc=38102026+self.case*10000+self.seed*100+len(history)-3
        key=digest({'fit':fit,'box':box,'confidence':confidence if box else None,'mc_seed':mc})
        self.ensure(key+'.npz');return super().score(history,box,confidence)
    def posterior_stats(self,history):
        fit,value=self.model(history);key=digest({'fit':fit,'posterior':'epistemic_and_predictive','recipe':self.extension_recipe})
        if key in self.posterior_memory:return self.posterior_memory[key]
        path=self.directory/(key+'.npz');self.ensure(path.name)
        if path.exists():
            with np.load(path,allow_pickle=False) as d:result={k:d[k].copy() for k in d.files}
        else:
            means=[];latent=[];predictive=[]
            with torch.no_grad():
                for x in self.z.split(512):
                    p=value['model'].posterior(x.unsqueeze(1),observation_noise=False)
                    pn=value['model'].posterior(x.unsqueeze(1),observation_noise=True)
                    means.append(p.mean.reshape(-1));latent.append(p.variance.reshape(-1));predictive.append(pn.variance.reshape(-1))
            mu=torch.cat(means).numpy();sd=torch.cat(latent).clamp_min(1e-18).sqrt().numpy()
            sp=torch.cat(predictive).clamp_min(1e-18).sqrt().numpy();ys=float(value['std']);ym=float(value['mean'])
            result={'mu_std':mu,'sd_std':sd,'sd_predictive_std':sp,'mu_y':mu*ys+ym,'sd_y':sd*ys,'sd_predictive_y':sp*ys,
                    'key':np.array(key),'fit_key':np.array(fit)}
            with path.with_suffix('.tmp').open('wb') as h:np.savez_compressed(h,**result)
            path.with_suffix('.tmp').replace(path);self.counters['posterior_evaluations']+=1
        self.posterior_memory[key]=result;return result
    def capped_score(self,history,box,confidence,kappa):
        fit,value=self.model(history);mc=38102026+self.case*10000+self.seed*100+len(history)-3
        key=digest({'fit':fit,'box':box,'confidence':confidence,'kappa':kappa,'mc_seed':mc,'recipe':self.extension_recipe})
        if key in self.scores:return self.scores[key]
        path=self.directory/(key+'.npz');self.ensure(path.name)
        if path.exists():
            with np.load(path,allow_pickle=False) as d:result=(d['scores'].copy(),json.loads(str(d['metadata'])))
        else:
            common.seed_all(mc)
            tilt=self.boo.LinearExponentialRegionalMeanTiltPlugAndPlay(bounds=torch.tensor([[0.]*4,[1.]*4],dtype=torch.double),grid_size=512,smooth=.08,dtype=torch.double)
            tilt.set_box_region(*box);delta=self.decide.confidence_to_delta(confidence,scale=3.)
            tilt.fit_lambda_by_delta(value['model'],delta,observation_noise=False);tilt.prepare_cache(value['model'])
            native=self.boo.TiltedModel(value['model'],tilt).eval();effective=SigmaCappedModel(value['model'],native,kappa).eval()
            acq=qLogExpectedImprovement(model=effective,best_f=value['best_std'],sampler=SobolQMCNormalSampler(sample_shape=torch.Size([512]),seed=mc))
            with torch.no_grad():scores=torch.cat([acq(x.unsqueeze(1)).reshape(-1) for x in self.z.split(256)]).numpy()
            assert np.isfinite(scores).all()
            meta={'key':key,'fit_key':fit,'mc_seed':mc,'kappa':float(kappa),'recipe':self.extension_recipe,'covariance_jitter':1e-6}
            with path.with_suffix('.tmp').open('wb') as h:np.savez_compressed(h,scores=scores,metadata=json.dumps(meta))
            path.with_suffix('.tmp').replace(path);result=(scores,meta);self.counters['capped_acquisition_evaluations']+=1
        self.scores[key]=result;return result
