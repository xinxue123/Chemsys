"""Matched regional LGBO and B4 prior-admission experiment.

The official repositories and older studies remain untouched. This is a new
finite-pool regional-component study, not an official-paper reproduction.
"""
from __future__ import annotations
from hashlib import sha256
import importlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler

from chem.decision.domain_guard import select_candidate
from chem.decision.replay import write_json_atomic
from .official_suzuki_adapter import modules,check_sources
from .suzuki_domain import digest

METHODS=('B0','LGBO_region','B3_gate','LGBO_gate','B4_base','B4_LGBO','A2_LGBO')
GATED=METHODS[2:]
REGIMES=('correct','wrong_category','mixed','assistant_initial')


def seed_all(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value)


def budget_at(ratio,iteration):
    if ratio<1 or not 1<=iteration<=16:raise ValueError('invalid budget schedule')
    return float(np.log(ratio)*(16-iteration)/15)


def clause_membership(z,advice):
    x=np.asarray(z,dtype=float)
    rows=[]
    for h in advice['clauses']:
        j=h['coordinate'];lo,hi=h['interval']
        if not 0<=j<4 or not 0<=lo<=hi<=1:raise ValueError('invalid region clause')
        rows.append((x[:,j]>=lo)&(x[:,j]<=hi))
    return np.array(rows,dtype=bool).reshape(len(rows),len(x))


def region_for(advice,retained):
    lo=np.zeros(4);hi=np.ones(4)
    for clause in advice['clauses']:
        if clause['id'] in retained:
            j=clause['coordinate'];lo[j]=max(lo[j],clause['interval'][0]);hi[j]=min(hi[j],clause['interval'][1])
    if np.any(lo>hi):raise ValueError('retained box is empty')
    return lo.tolist(),hi.tolist()


def argmax(scores,mask,ids):
    indices=np.flatnonzero(mask)
    if not len(indices):raise ValueError('no available candidate')
    return min(indices,key=lambda i:(-float(scores[i]),ids[i]))


def choose_policy(method,base,advice,membership,budget,queried,ids,guided):
    """No objective values, truth labels or regime names enter this function.

    B4_LGBO admits a largest compatible subset as a *soft prior*. Its query is
    budget-feasible, but need not satisfy the retained preferred-region box.
    B4_base retains the original hard-clause execution semantics separately.
    """
    if method not in METHODS:raise ValueError(method)
    base=np.asarray(base,dtype=float);available=~np.asarray(queried,dtype=bool)
    if not np.isfinite(base).all():raise ValueError('nonfinite base acquisition')
    reference=argmax(base,available,ids);loss=base[reference]-base
    eligible=available&(loss<=budget)
    hids=tuple(c['id'] for c in advice['clauses'])
    full=tuple(sorted(hids));retained=();diagnosis=None;score=base
    all_feasible=eligible&(membership.all(axis=0) if len(hids) else True)
    if method=='B0':selected=reference
    elif method in ('LGBO_region','B3_gate','LGBO_gate'):
        score=guided(full);retained=full
        if method=='LGBO_gate':selected=argmax(score,eligible,ids)
        else:
            proposed=argmax(score,available,ids)
            selected=proposed if method=='LGBO_region' or eligible[proposed] else reference
    elif method=='A2_LGBO':
        retained=full if all_feasible.any() else ()
        score=guided(retained) if retained else base
        selected=argmax(score,eligible,ids)
    else:
        diagnosis=select_candidate(base,membership,budget,candidate_ids=ids,hypothesis_ids=hids,queried_mask=queried)
        retained=diagnosis.retained_ids
        if method=='B4_base':selected=diagnosis.candidate_index
        else:
            score=guided(retained) if retained else base
            selected=argmax(score,eligible,ids)
    witness=eligible.copy()
    for i,key in enumerate(hids):
        if key in retained:witness&=membership[i]
    satisfaction=tuple(sorted(hids[j] for j in range(len(hids)) if membership[j,selected]))
    return int(selected),{'reference_id':ids[reference],'selected_id':ids[selected],
        'base_logei_max':float(base[reference]),'base_logei_selected':float(base[selected]),
        'base_logei_loss':float(loss[selected]),'budget_logei':float(budget),
        'budget_violation':bool(loss[selected]>budget),'eligible_candidates':int(eligible.sum()),
        'full_prior_admissible':bool(all_feasible.any()),'retained_ids':list(retained),
        'revoked_ids':sorted(set(hids)-set(retained)),'satisfied_ids':list(satisfaction),
        'retained_box_witness_count':int(witness.sum()),
        'selected_satisfies_retained_box':set(retained)<=set(satisfaction),
        'selected_guided_logei':float(score[selected]),
        'minimal_revocation_receipt':diagnosis.to_dict() if diagnosis else None,
        'prior_semantics':'soft_prior_admission' if method in ('B4_LGBO','A2_LGBO') else 'method_specific'}


class ScoreCache:
    """Fit each distinct history once; cache scores independently of policy.

    All acquisition calculations use the official TiltedModel (including its
    1e-6 covariance jitter). Lambda0 supplies the common unguided reference.
    Every prior uses the same MC seed, model, pool and sample count.
    """
    def __init__(self,z,ids,directory,case,seed):
        modules('LGBO')
        self.boo=importlib.import_module('boo')
        self.decide=importlib.import_module('decide')
        self.native=importlib.import_module('prior_monte_carlo')
        self.z=torch.tensor(np.asarray(z).clip(1e-6,1-1e-6),dtype=torch.double)
        self.ids=tuple(ids);self.case=case;self.seed=seed
        self.recipe=sha256(Path(__file__).read_bytes()).hexdigest()
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.models={};self.scores={};self.counters={'fits':0,'fit_cache_hits':0,'acquisition_evaluations':0,'score_cache_hits':0}

    def model(self,history):
        ordered=sorted(history,key=lambda r:r['candidate_id'])
        key=digest({'history':ordered,'case':self.case,'seed':self.seed,'recipe':self.recipe})
        if key in self.models:self.counters['fit_cache_hits']+=1;return key,self.models[key]
        lookup={v:i for i,v in enumerate(self.ids)}
        x=self.z[[lookup[r['candidate_id']] for r in ordered]]
        y=torch.tensor([[r['y']] for r in ordered],dtype=torch.double)
        path=self.directory/(key+'.pt')
        fit_seed=28102026+self.case*10000+self.seed*100+len(history)-3
        seed_all(fit_seed)
        if path.exists():
            saved=torch.load(path,map_location='cpu',weights_only=False)
            assert saved['key']==key
            from botorch.models import SingleTaskGP
            ystd,mean,std=self.native.BaseBOSampler._standardize_with_stats(y)
            model=SingleTaskGP(x,ystd);model.load_state_dict(saved['state']);model.eval()
            value={'model':model,'best_std':float(ystd.max()),'mean':mean,'std':std,'fit_seed':fit_seed}
            self.counters['fit_cache_hits']+=1
        else:
            sampler=self.native.WeightedQLogEI(bounds=torch.tensor([[0.]*4,[1.]*4],dtype=torch.double),
                X_init=x,Y_init=y,dtype=torch.double,device=torch.device('cpu'))
            value={'model':sampler.model,'best_std':float(sampler._current_best_std()),
                'mean':sampler._y_mean,'std':sampler._y_std,'fit_seed':fit_seed}
            saved={'key':key,'state':sampler.model.state_dict(),'fit_seed':fit_seed,
                'gp_class':type(sampler.model).__module__+'.'+type(sampler.model).__name__,
                'kernel':type(sampler.model.covar_module).__name__}
            temp=path.with_suffix('.tmp');torch.save(saved,temp);temp.replace(path)
            self.counters['fits']+=1
        self.models[key]=value
        return key,value

    def score(self,history,box=None,confidence=.7):
        key,value=self.model(history)
        mc_seed=38102026+self.case*10000+self.seed*100+len(history)-3
        score_key=digest({'fit':key,'box':box,'confidence':confidence if box else None,'mc_seed':mc_seed})
        if score_key in self.scores:self.counters['score_cache_hits']+=1;return self.scores[score_key]
        path=self.directory/(score_key+'.npz')
        if path.exists():
            with np.load(path,allow_pickle=False) as data:
                result=(data['scores'].copy(),json.loads(str(data['metadata'])))
            assert result[1]['key']==score_key
            self.counters['score_cache_hits']+=1
        else:
            seed_all(mc_seed)
            bounds=torch.tensor([[0.]*4,[1.]*4],dtype=torch.double)
            tilt=self.boo.LinearExponentialRegionalMeanTiltPlugAndPlay(bounds=bounds,grid_size=512,
                smooth=.08,dtype=torch.double,device=torch.device('cpu'))
            lb,ub=box if box else ([0.]*4,[1.]*4)
            tilt.set_box_region(lb,ub)
            delta=self.decide.confidence_to_delta(confidence,scale=3.) if box else 0.
            tilt.fit_lambda_by_delta(value['model'],delta,observation_noise=False)
            tilt.prepare_cache(value['model'])
            effective=self.boo.TiltedModel(value['model'],tilt).eval()
            acquisition=qLogExpectedImprovement(model=effective,best_f=value['best_std'],
                sampler=SobolQMCNormalSampler(sample_shape=torch.Size([512]),seed=mc_seed))
            with torch.no_grad():
                scores=torch.cat([acquisition(batch.unsqueeze(1)).reshape(-1)
                                  for batch in self.z.split(256)]).numpy()
            if not np.isfinite(scores).all():raise ValueError('nonfinite qLogEI')
            meta={'key':score_key,'fit_key':key,'mc_seed':mc_seed,'mc_paths':512,'regional_grid':512,
                'smooth':.08,'delta':float(delta),'lambda':float(tilt._lam),
                'common_covariance_jitter':1e-6,'candidates':len(scores),'fit_seed':value['fit_seed']}
            temp=path.with_suffix('.tmp')
            with temp.open('wb') as handle:np.savez_compressed(handle,scores=scores,metadata=json.dumps(meta))
            temp.replace(path);result=(scores,meta);self.counters['acquisition_evaluations']+=1
        self.scores[score_key]=result
        return result
