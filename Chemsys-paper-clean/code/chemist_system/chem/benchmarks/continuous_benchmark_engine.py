"""All-continuous transport of the frozen region / KF / B4 / trust policy.

All methods share a prefix-keyed SingleTaskGP (continuous limit of Mixed GP),
q=1 MC512 qLogEI, corrected official LGBO covariance units and tie rule.
No objective provider is imported or called in model or selection functions.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import importlib
import random
import warnings
import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal
from filelock import FileLock
from chem.decision.domain_guard import select_candidate
from .official_suzuki_adapter import modules
from .suzuki_domain import digest
from . import continuous_benchmark_data as data
from . import r2_region_trust_engine as trust
from .regional_ablation_suzuki_engine import new_bank,forecast,update_bank
from .stable_information_engine import information_proxy,priority

modules('LGBO')
native=importlib.import_module('prior')
decide=importlib.import_module('decide')
METHODS=('Pure_BO','LGBO_region','Gate_only','B4_hard','B4_soft','LLINBO_Justify','ReasoningBO','New_Agent')
JIT={}
PARAMETERS={'initial_queries':4,'additional_queries':16,'seed':0,'tie_seed':0,'MC_paths':512,
    'grid_size':512,'region_smooth':.08,'confidence':.7,'budget_ratio':4,
    'GP':'SingleTaskGP; all input axes continuous; no fabricated category',
    'GP_fit':'same marginal-likelihood routine/default priors as existing adapter; outer prefix standardization',
    'normal_policy':'same corrected LGBO / scalar KF / B4 conflict fallback / persistent region trust',
    'validation_rounds':[2,4,6],'maximum_refreshes':3,'minimum_refresh_gap':2,'last_refresh_round':14,
    'units':'y = raw objective / fixed engineering scale; KF noise floor 0.03 in these units',
    'high_objective_stop':'disabled: no meaningful universal 95% yield threshold for STY or detector area',
    'molecular_recovery':'not applicable: all variables continuous',
    'joint_tests':'not applicable: no categorical axes',
    'ties':'exact acquisition equality only: isolated-contrast proxy, latent SD, independent PCG64 priority'}

def seed_all(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)

def warm():
    if JIT:return True
    torch.set_num_threads(1)
    # Compile each JIT/sampler path once before any counted trajectory.
    from botorch.acquisition import utils as au
    from botorch.utils import safe_math as sm
    from botorch.acquisition import logei as le
    x=torch.tensor([[-1.,0.,1.]],dtype=torch.double)
    for n in ('log_softplus','logmeanexp'):
        if hasattr(sm,n):
            try:getattr(sm,n)(x,dim=-1) if n=='logmeanexp' else getattr(sm,n)(x)
            except TypeError:pass
    model=SingleTaskGP(torch.tensor([[0.,0.],[1.,1.]],dtype=torch.double),torch.tensor([[0.],[1.]],dtype=torch.double)).eval()
    with torch.no_grad():
        qLogExpectedImprovement(model,best_f=1.,sampler=SobolQMCNormalSampler(torch.Size([512]),seed=0))(torch.ones(2,1,2,dtype=torch.double)*.5)
    # Count any JIT compilation only AFTER warm-up; re-instantiation does not compile.
    for name in ('script','trace','trace_module'):
        original=getattr(torch.jit,name)
        def counted(*args,_original=original,_name=name,**kwargs):
            JIT[_name]+=1
            return _original(*args,**kwargs)
        JIT[name]=0;setattr(torch.jit,name,counted)
    return True

class CorrectedTilt(native.LinearExponentialRegionalMeanTiltPlugAndPlay):
    def posterior(self,base_model,X,observation_noise=False,**kwargs):
        p=super().posterior(base_model,X,observation_noise=observation_noise,**kwargs)
        outcome=getattr(base_model,'outcome_transform',None)
        if getattr(self,'_cache',None) is None or outcome is None:return p
        b=base_model.posterior(X,observation_noise=observation_noise,**kwargs)
        return GPyTorchPosterior(MultivariateNormal(b.mvn.mean+outcome.stdvs.square().reshape(())*(p.mvn.mean-b.mvn.mean),p.mvn.lazy_covariance_matrix))

class Marginals(Model):
    def __init__(self,mu,variance):super().__init__();self.mu=mu;self.var=variance
    @property
    def num_outputs(self):return 1
    @property
    def batch_shape(self):return torch.Size()
    def posterior(self,X,**unused):
        # The last input column is an exact lookup index, NOT a GP covariate.
        ix=X[...,0].long();mean=self.mu[ix];variance=self.var[ix]
        return GPyTorchPosterior(MultivariateNormal(mean,torch.diag_embed(variance)))

def acquisition(mu,sd,shift,best,seed):
    model=Marginals(torch.as_tensor(mu+shift,dtype=torch.double),torch.as_tensor(sd**2+1e-6,dtype=torch.double))
    a=qLogExpectedImprovement(model,best_f=best,sampler=SobolQMCNormalSampler(torch.Size([512]),seed=seed))
    ix=torch.arange(len(mu),dtype=torch.double).reshape(-1,1,1)
    with torch.no_grad():out=torch.cat([a(x).reshape(-1) for x in ix.split(256)]).numpy()
    assert np.isfinite(out).all();return out

def membership(pool,knowledge):
    z=np.asarray(pool['z']);rows=[]
    for h in knowledge['clauses']:
        j=pool['axes'].index(h['axis']);lo,hi=h['bounds']
        rows.append((z[:,j]>=lo)&(z[:,j]<=hi))
    return np.asarray(rows,dtype=bool).reshape(len(rows),len(z))

def region(pool,knowledge,retained):
    if not retained:return None
    bounds={a:[0.,1.] for a in pool['axes']}
    for h in knowledge['clauses']:
        if h['id'] in retained:bounds[h['axis']]=list(h['bounds'])
    return {'allowed':bounds,'confidence':knowledge['confidence']}

class Cache:
    def __init__(self,pool):
        self.pool=pool;self.ids=pool['ids'];self.ix={k:i for i,k in enumerate(self.ids)}
        self.X=torch.tensor(pool['z'],dtype=torch.double);self.directory=Path('data/runs/continuous_20261029/cache')/pool['task']
        self.directory.mkdir(parents=True,exist_ok=True);self.memory={};self.fields={}
        self.recipe=digest({'engine':data.filehash(__file__),'pool':digest(pool),'botorch':'0.14.0','parameters':PARAMETERS})
        self.counters={'fits':0,'fit_cache_hits':0,'posteriors':0,'acquisitions':0}
    def model(self,history):
        # Semantic physical order makes fitting independent of ID spelling.
        ordered=sorted([{'candidate_id':h['candidate_id'],'y':float(h['y'])} for h in history],key=lambda h:tuple(self.pool['z'][self.ix[h['candidate_id']]]))
        key=digest({'history':ordered,'recipe':self.recipe})
        if key in self.memory:return self.memory[key]
        x=self.X[[self.ix[h['candidate_id']] for h in ordered]]
        y=torch.tensor([[h['y']] for h in ordered],dtype=torch.double)
        mean=y.mean();std=y.std(unbiased=False).clamp_min(1e-12);ys=(y-mean)/std
        seed=28102026+(1 if self.pool['task']=='SnAr' else 2)*10000+len(history)-3
        seed_all(seed);m=SingleTaskGP(x,ys)
        path=self.directory/(key+'.pt')
        with FileLock(str(path)+'.lock',timeout=300):
            if path.exists():
                saved=torch.load(path,map_location='cpu',weights_only=False);assert saved['key']==key
                m.load_state_dict(saved['state']);self.counters['fit_cache_hits']+=1
            else:
                fit_gpytorch_mll(ExactMarginalLogLikelihood(m.likelihood,m));self.counters['fits']+=1
                tmp=path.with_suffix('.tmp');torch.save({'key':key,'state':m.state_dict(),'fit_seed':seed,'history':ordered},tmp);tmp.replace(path)
        m.eval();mu=[];sd=[];pred=[]
        with torch.no_grad():
            for xx in self.X.split(256):
                p=m.posterior(xx.unsqueeze(1));n=m.posterior(xx.unsqueeze(1),observation_noise=True)
                mu.append(p.mean.reshape(-1));sd.append(p.variance.reshape(-1).clamp_min(1e-18).sqrt());pred.append(n.variance.reshape(-1).clamp_min(1e-18).sqrt())
        mu=torch.cat(mu).numpy();sd=torch.cat(sd).numpy();pred=torch.cat(pred).numpy()
        v={'model':m,'mean':float(mean),'std':float(std),'best_std':float(ys.max()),'fit_key':key,'fit_seed':seed,
           'mu_std':mu,'sd_std':sd,'mu_y':mu*float(std)+float(mean),'sd_y':sd*float(std),'sd_predictive_y':pred*float(std)}
        self.memory[key]=v;self.counters['posteriors']+=1;return v
    def score(self,history,shift=None):
        v=self.model(history);zero=np.zeros(len(self.ids));shift=zero if shift is None else shift
        seed=38102026+(1 if self.pool['task']=='SnAr' else 2)*10000+len(history)-3
        out=acquisition(v['mu_std'],v['sd_std'],shift,v['best_std'],seed);self.counters['acquisitions']+=1
        return out,{'fit_key':v['fit_key'],'fit_seed':v['fit_seed'],'mc_seed':seed,'paths':512,'marginal_variance_jitter':1e-6}
    def field(self,history,spec):
        if spec is None:return np.zeros(len(self.ids)),{'region':None}
        v=self.model(history);key=digest({'fit':v['fit_key'],'region':spec})
        if key in self.fields:return self.fields[key]
        dim=self.X.shape[1];tilt=CorrectedTilt(torch.tensor([[0.]*dim,[1.]*dim],dtype=torch.double),grid_size=512,smooth=.08)
        lo=[spec['allowed'][a][0] for a in self.pool['axes']];hi=[spec['allowed'][a][1] for a in self.pool['axes']]
        tilt.set_box_region(lo,hi);tilt.fit_lambda_by_delta(v['model'],decide.confidence_to_delta(spec['confidence'],scale=3.));tilt.prepare_cache(v['model'])
        with torch.no_grad():raw=torch.cat([(tilt.posterior(v['model'],x.unsqueeze(1)).mean-v['model'].posterior(x.unsqueeze(1)).mean).reshape(-1) for x in self.X.split(256)]).numpy()
        meta={'region':spec,'lambda':float(tilt._lam),'outcome_covariance_multiplier':float(v['model'].outcome_transform.stdvs.square()),'quadrature_size':512}
        self.fields[key]=(raw,meta);return raw,meta
    def release(self):self.memory.clear();self.fields.clear()

def setup(cache,state,t):
    pool=cache.pool;history=state['history'];knowledge=state['knowledge'];base,bmeta=cache.score(history);v=cache.model(history)
    available=~np.isin(pool['ids'],[h['candidate_id'] for h in history]);budget=float(np.log(4)*(16-t)/15)
    proxy,pairs,unseen=information_proxy(cache.X.numpy(),[cache.ix[h['candidate_id']] for h in history]);ranks=priority(len(base),t,0)
    def choose(scores,mask):
        js=np.flatnonzero(mask);assert len(js)
        js=js[scores[js]==scores[js].max()];js=js[proxy[js]==proxy[js].max()]
        js=js[v['sd_y'][js]==v['sd_y'][js].max()]
        return int(js[np.argmin(ranks[js])])
    ref=choose(base,available);eligible=available&(base[ref]-base<=budget)
    mem=membership(pool,knowledge);dg=select_candidate(base,mem,budget,candidate_ids=pool['ids'],hypothesis_ids=[h['id'] for h in knowledge['clauses']],queried_mask=~available)
    kept=list(dg.retained_ids);positions=[i for i,h in enumerate(knowledge['clauses']) if h['id'] in kept]
    feasible=eligible&(mem[positions].all(0) if positions else True)
    witness=choose(base,feasible);dg=replace(dg,candidate_index=witness,candidate_id=pool['ids'][witness])
    method=state['method'];full=[h['id'] for h in knowledge['clauses']]
    retained=kept if method in ('New_Agent','B4_hard','B4_soft') else full
    spec=region(pool,knowledge,retained) if method not in ('Pure_BO','LLINBO_Justify','ReasoningBO','B4_hard') else None
    raw,fmeta=cache.field(history,spec);bank=prior=key=gate=None;alpha=1. if spec else 0.;fallback=False
    d={'region':spec,'stats':v,'selected':ref}
    if method=='New_Agent':
        key=trust.geometry_key(spec)
        if key:bank=deepcopy(state['banks'].get(key,new_bank(t)));prior=forecast(bank,t);alpha=prior['amplitude']
        fallback=bool(dg.revoked_ids) or dg.conflict_kind!='none'
        if fallback:alpha=0.
        d.update(amplitude=alpha,fallback=fallback);gate=trust.permission(state,d,t);alpha=gate['amplitude_after_trust'];fallback|=gate['trust_blocked']
    scores,gmeta=cache.score(history,alpha*raw) if alpha else (base,bmeta)
    selected=choose(scores,available if method=='LGBO_region' else feasible if method=='B4_hard' else eligible)
    assert not fallback or selected==ref
    d.update(base=base,score=scores,raw_shift_std=raw,shift_std=alpha*raw,base_meta=bmeta,guided_meta=gmeta,field_meta=fmeta,
        reference=ref,selected=selected,normal_proposal=selected,budget=budget,available=available,eligible=eligible,retained=retained,
        diagnosis=dg,bank=bank,prior=prior,bank_key=key,amplitude=alpha,fallback=fallback,trust=gate,
        information_proxy=proxy,tie_priority=ranks,isolated_pairs=pairs,unseen_values=unseen)
    return d

def event_reasons(state,d,t):
    if state['method']!='New_Agent' or len(state['calls'])>=3 or t>14:return []
    if t-([1]+[c['iteration'] for c in state['calls']])[-1]<2:return []
    reasons=[]
    history=state['history']
    if len(state['decisions'])>=3 and max(h['y'] for h in history)-max(h['y'] for h in history[:-3])<.005:reasons.append('stagnation_3')
    if len(state['decisions'])>=2 and all(abs(e['measurement']['innovation_z'])>2.5 for e in state['decisions'][-2:]):reasons.append('bad_innovations_2')
    if d['diagnosis'].revoked_ids or d['diagnosis'].conflict_kind!='none':reasons.append('B4_conflict')
    if not state['knowledge']['clauses']:reasons.append('empty_knowledge')
    recent=next((v for v in state['validations'] if v['query']==t-1),None)
    if recent and recent['observed_y']>=recent['incumbent_before_y']+.005:reasons.append('successful_controlled_counterexample')
    return reasons

def validation_candidates(cache,state,d):
    pool=cache.pool;hist=state['history'];anchor=max(hist,key=lambda h:h['y']);j0=cache.ix[anchor['candidate_id']]
    z=cache.X.numpy();diff=z!=z[j0];mem=membership(pool,state['knowledge']);rows=[]
    for j in np.flatnonzero(d['available']&(diff.sum(1)==1)):
        axis=int(np.flatnonzero(diff[j])[0]);viol=[h['id'] for k,h in enumerate(state['knowledge']['clauses']) if h['axis']==pool['axes'][axis] and not mem[k,j]]
        if viol:rows.append({'candidate_id':pool['ids'][j],'point':pool['points'][j],'changed_axis':pool['axes'][axis],
            'challenged_clause_ids':viol,'GP_mean':float(d['stats']['mu_y'][j])*pool['objective_scale'],
            'GP_latent_SD':float(d['stats']['sd_y'][j])*pool['objective_scale'],'base_qLogEI':float(d['base'][j]),'normal_gate_eligible':bool(d['eligible'][j])})
    focus=None
    if state['validations'] and rows:
        groups={}
        for row in rows:
            a=pool['axes'].index(row['changed_axis']);j=cache.ix[row['candidate_id']]
            if z[j,a] not in set(z[[cache.ix[h['candidate_id']] for h in hist],a]):groups.setdefault(a,[]).append(row)
        if groups:
            a=min(groups,key=lambda a:(-len(groups[a]),-max(r['GP_latent_SD'] for r in groups[a]),pool['axes'][a]));rows=groups[a];focus=pool['axes'][a]
    ranks=d['tie_priority'];rows.sort(key=lambda r:ranks[cache.ix[r['candidate_id']]])
    return {'anchor_candidate_id':anchor['candidate_id'],'anchor_observation_index':hist.index(anchor)+1,'anchor_point':pool['points'][j0],
            'anchor_objective':anchor['y']*pool['objective_scale'],'candidates':rows,'coverage_focus':focus,
            'skip_reason':None if rows else 'no unqueried exact one-axis neighbor outside current preferences'}
