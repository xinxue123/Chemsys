"""Suzuki-Case-style4+16 replay over full nominal Perera/Doyle tables.

Only observed prefixes and features enter this module. Legacy sources and
previous experimental evidence remain frozen. No acquisition shortlist limits
LGBO, Gate, B4 or R2. ReasoningBO alone receives its explicitly adapted top5.
"""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import math

import numpy as np
import torch
from filelock import FileLock
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal

from . import reaction_transfer_engine as nominal
from . import guarded_suzuki_engine as policies
from .suzuki_domain import digest

METHODS=('Pure_BO','LGBO_region','Gate_only','B4_hard','B4_soft','New_R2','LLINBO_Justify','ReasoningBO_point')
PARAMETERS={**deepcopy(nominal.PARAMETERS),'seed':0,'initial_queries':4,'additional_queries':16,
    'MC_paths':512,'quadrature_cap':512,'maximum_active_clauses':4,
    'unit_fix':'cached internal regional covariance multiplied by scalar outcome stdvs squared',
    'candidate_scope':'entire measured pool; only ReasoningBO uses top5 base-q1 candidates',
    'operator_compiles_in_loop':0}


class CorrectedNominalTilt(nominal.NominalFiniteRegion):
    """Keep native regional formula; reconcile its cached/output covariance units."""
    def posterior(self,base_model,X,observation_noise=False,**kwargs):
        p=super().posterior(base_model,X,observation_noise=observation_noise,**kwargs)
        # Native uncached branch already uses transformed-back joint posterior.
        if getattr(self,'_cache',None) is None:return p
        outcome=getattr(base_model,'outcome_transform',None)
        if outcome is None:return p
        if not hasattr(outcome,'stdvs') or outcome.stdvs.numel()!=1:
            raise ValueError('regional cache unit transport requires a scalar Standardize outcome transform')
        scale2=outcome.stdvs.square().reshape(())
        b=base_model.posterior(X,observation_noise=observation_noise,**kwargs)
        corrected=b.mvn.mean+scale2*(p.mvn.mean-b.mvn.mean)
        return GPyTorchPosterior(MultivariateNormal(corrected,p.mvn.lazy_covariance_matrix))


class Cache(nominal.Cache):
    def __init__(self,pool,directory,permutation=None):
        super().__init__(pool,directory,permutation)
        self.recipe=digest({'engine_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
            'legacy_cache_sha256':sha256(Path(nominal.__file__).read_bytes()).hexdigest(),
            'pool':digest(pool),'mapping':self.mapping,'parameters':PARAMETERS})

    def model(self,history):
        key=digest({'history':sorted(history,key=lambda h:h['candidate_id']),'recipe':self.recipe})
        with FileLock(str(self.directory/(key+'.fit.lock')),timeout=300):return super().model(history)

    def tilt(self,history,spec):
        fit,value=self.model(history);key=fit,digest(spec)
        if key not in self.tilts:
            mask=np.array([all(p[a] in spec['allowed'][a] for a in self.pool['axes']) for p in self.pool['points']])
            indices=np.flatnonzero(mask)
            if not len(indices):raise ValueError('region has no measured feature-domain witness')
            if len(indices)>PARAMETERS['quadrature_cap']:
                indices=np.sort(np.random.default_rng(int(digest(spec)[:12],16)).choice(indices,512,replace=False))
            tilt=CorrectedNominalTilt(self.X[indices],torch.full((len(indices),),1/len(indices),dtype=torch.double))
            delta=nominal.old.common.decide.confidence_to_delta(spec['confidence'],scale=3.)
            tilt.fit_lambda_by_delta(value['model'],delta);tilt.prepare_cache(value['model'])
            tilt.grid_indices=indices.tolist();self.tilts[key]=tilt
        return self.tilts[key],value,fit

    def field(self,history,spec):
        if spec is None:return np.zeros(len(self.ids)),{'region':None,'key':None}
        fit,v=self.model(history);key=digest({'field':spec,'fit':fit,'recipe':self.recipe})
        if key in self.fields:return self.fields[key]
        tilt,_,_=self.tilt(history,spec)
        with torch.no_grad():
            raw=torch.cat([(tilt.posterior(v['model'],xx.unsqueeze(1)).mean-
                v['model'].posterior(xx.unsqueeze(1)).mean).reshape(-1) for xx in self.X.split(256)]).numpy()
        meta={'key':key,'fit_key':fit,'region':spec,'lambda':float(tilt._lam),
            'grid_indices':tilt.grid_indices,'quadrature_size':len(tilt.grid_indices),
            'outcome_covariance_unit_multiplier':float(v['model'].outcome_transform.stdvs.square())}
        self.fields[key]=(raw,meta)
        return raw,meta

    def score(self,history,spec=None,amplitude=0.):
        if spec is None or amplitude==0.:spec,amplitude=None,0.
        fit,_=self.model(history)
        lock=digest({'fit':fit,'spec':spec,'amplitude':amplitude})
        with FileLock(str(self.directory/(lock+'.score.lock')),timeout=300):
            return super().score(history,spec,amplitude)

    def release(self):
        self.models.clear();self.statistics.clear();self.fields.clear();self.tilts.clear();self.acquisitions.clear()


def setup(cache,state,iteration):
    history,knowledge,pool=state['history'],state['knowledge'],cache.pool
    base,_,bmeta=cache.score(history)
    available=~np.isin(pool['ids'],[h['candidate_id'] for h in history])
    reference=nominal.argmax(base,available,pool['ids'])
    budget=float(np.log(4)*(16-iteration)/15)
    eligible=available&(base[reference]-base<=budget)
    h=nominal.membership(pool,knowledge)
    dg=nominal.select_candidate(base,h,budget,candidate_ids=pool['ids'],
        hypothesis_ids=[r['id'] for r in knowledge['clauses']],queried_mask=~available)
    bank=prior=None;key=None;fallback=False;amplitude=1.
    method=state['method']
    if method=='New_R2':
        retained=list(dg.retained_ids);spec=nominal.region(pool,knowledge,retained)
        key=digest(spec) if spec else None
        if key:
            bank=deepcopy(state['banks'].get(key,nominal.new_bank(iteration)));prior=nominal.forecast(bank,iteration)
            amplitude=prior['amplitude']
        fallback=bool(dg.revoked_ids) or dg.conflict_kind!='none'
        if fallback:amplitude=0.
        scores,shift,gmeta=cache.score(history,spec,amplitude)
        selected=nominal.argmax(scores,eligible,pool['ids']);receipt=None
    else:
        mapping={'Pure_BO':'B0','LGBO_region':'LGBO_region','Gate_only':'LGBO_gate',
            'B4_hard':'B4_base','B4_soft':'B4_LGBO','LLINBO_Justify':'B0','ReasoningBO_point':'B0'}
        def guided(kept):
            pref=nominal.region(pool,knowledge,kept)
            return cache.score(history,pref,1.)[0]
        selected,receipt=policies.choose_policy(mapping[method],base,knowledge,h,budget,~available,pool['ids'],guided)
        retained=list(receipt['retained_ids']);spec=nominal.region(pool,knowledge,retained)
        if method=='B4_hard':
            scores,shift,gmeta=base,np.zeros(len(base)),bmeta
            spec=None
        else:scores,shift,gmeta=cache.score(history,spec,1.)
    raw,fmeta=cache.field(history,spec)
    assert not fallback or selected==reference
    return {'base':base,'score':scores,'available':available,'eligible':eligible,
        'reference':int(reference),'selected':int(selected),'budget':budget,'diagnosis':dg,
        'retained':retained,'region':spec,'raw_shift_std':raw,'shift_std':shift,'amplitude':float(amplitude),
        'fallback':fallback,'bank_key':key,'bank':bank,'prior':prior,'stats':cache.stats(history),
        'base_meta':bmeta,'field_meta':fmeta,'guided_meta':gmeta,'policy_receipt':receipt}


def event_reasons(state,d,t):
    if state['method']!='New_R2':return []
    # Same scalar4+16 schedule as the original Suzuki G3/R2, with None-safe input.
    view=[{**x,'measurement':x.get('measurement') or {'innovation_z':0.}} for x in state['decisions']]
    return nominal.old.previous.event_reasons(t,state['history'],view,state['knowledge']['clauses'],
        d['diagnosis'],[1]+[c['iteration'] for c in state['calls']])


def shortlist(cache,state,d):
    return sorted(np.flatnonzero(d['available']),key=lambda j:(-float(d['base'][j]),cache.ids[j]))[:5]
