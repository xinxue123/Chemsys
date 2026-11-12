"""Prequential residual knowledge filtering on the frozen Suzuki GP base.

No unobserved objective values enter this module. The filter is a working
linear-Gaussian model, not a probability that a chemical statement is true.
"""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.distributions import MultivariateNormal

from chem.decision.domain_guard import select_candidate
from . import guarded_suzuki_engine as common
from .uncertain_suzuki_engine import UncertaintyCache
from .suzuki_domain import digest

SLOTS=('H1','H2','H3','H4')
VARIABLES=('residence_time','temperature','catalyst_loading','catalyst')
PARAMETERS={'coefficient_prior_mean':1.,'coefficient_prior_variance':.25,
    'coefficient_process_variance':.0025,'offset_prior_variance':.0025,
    'offset_process_variance':.000025,'measurement_variance_floor':.0009,
    'trust_sd_multiplier':1.,'minimum_trust':.05,'mean_cap_sd':.5,
    'budget_ratio':4.,'stagnation_rounds':3,'stagnation_gain_y':.005,
    'innovation_threshold':2.5,'bad_innovation_rounds':2,
    'min_call_gap':2,'last_refresh_round':14,'max_calls':4}


def membership(z,hypotheses):
    z=np.asarray(z,dtype=float);rows=[]
    for h in hypotheses:
        if h['variable']=='catalyst':m=np.isin(np.floor(z[:,3]*8).astype(int).clip(0,7),h['categories'])
        else:
            j=VARIABLES.index(h['variable']);lo,hi=h['bounds'];m=(z[:,j]>=lo)&(z[:,j]<=hi)
        rows.append(m)
    return np.asarray(rows,dtype=bool).reshape(len(rows),len(z))


def canonical(h):return {k:h[k] for k in ('id','variable','bounds','categories','effect_pp')}


def validate_plan(data,previous,history_count,z):
    """Strict, executable predicate schema; every old hypothesis is accounted for."""
    if not isinstance(data,dict) or set(data)!={'hypotheses','revocations','summary'}:raise ValueError('expected hypotheses, revocations, summary')
    hs=data['hypotheses'];rev=data['revocations'];old={h['id']:h for h in previous}
    if not isinstance(hs,list) or len(hs)>4 or not isinstance(rev,list):raise ValueError('at most four hypotheses')
    if not isinstance(data['summary'],str) or not 1<=len(data['summary'])<=2000:raise ValueError('invalid summary')
    seen=set();result=[];revoked=[]
    def evidence(row):
        if not isinstance(row['reason'],str) or not 1<=len(row['reason'])<=1500:raise ValueError('invalid reason')
        support=row['support']
        if not isinstance(support,list) or not support or len(set(support))!=len(support) or any(type(i)!=int or not 1<=i<=history_count for i in support):raise ValueError('support must reference observed one-based indices')
        if row['id'] not in SLOTS or row['id'] in seen:raise ValueError('duplicate or invalid hypothesis id')
        seen.add(row['id'])
    for row in hs:
        if not isinstance(row,dict) or set(row)!={'id','operation','variable','bounds','categories','effect_pp','reason','support'}:raise ValueError('invalid hypothesis fields')
        evidence(row);h=deepcopy(row);op=h['operation'];key=h['id']
        if op not in ('add','retain','revise') or (op=='add')!=(key not in old):raise ValueError('add for vacant id; retain or revise for active id')
        if h['variable'] not in VARIABLES:raise ValueError('invalid variable')
        if type(h['effect_pp']) not in (int,float) or not np.isfinite(h['effect_pp']) or not 1<=h['effect_pp']<=20:raise ValueError('effect_pp must be 1..20')
        h['effect_pp']=float(h['effect_pp'])
        if h['variable']=='catalyst':
            cats=h['categories']
            if h['bounds']!=[] or not isinstance(cats,list) or not 1<=len(cats)<=7 or any(type(k)!=int or k not in range(8) for k in cats) or len(set(cats))!=len(cats):raise ValueError('catalyst needs 1..7 unique categories 0..7, bounds=[]')
            h['categories']=sorted(cats)
        else:
            b=h['bounds']
            if h['categories']!=[] or not isinstance(b,list) or len(b)!=2 or any(type(v) not in (int,float) or not np.isfinite(v) or abs(v-round(v,4))>1e-10 for v in b):raise ValueError('numeric bounds need two numbers with at most four decimal places, categories=[]')
            if not 0<=b[0]<b[1]<=1 or not .1-1e-12<=b[1]-b[0]<=.9+1e-12:raise ValueError('normalized numeric interval width must be .1..9')
            h['bounds']=[float(v) for v in b]
        if op=='retain' and canonical(h)!=canonical(old[key]):raise ValueError('retain must preserve predicate and effect exactly')
        if op=='revise' and canonical(h)==canonical(old[key]):raise ValueError('unchanged predicate/effect requires retain')
        m=membership(z,[h])[0]
        if not m.any() or m.all():raise ValueError('predicate must partition the fixed pool')
        result.append(h)
    for row in rev:
        if not isinstance(row,dict) or set(row)!={'id','reason','support'}:raise ValueError('invalid revocation fields')
        evidence(row)
        if row['id'] not in old:raise ValueError('only active ids may be revoked')
        revoked.append(deepcopy(row))
    if not set(old)<=seen:raise ValueError('every active id must be retained, revised or revoked')
    return {'hypotheses':sorted(result,key=lambda h:h['id']),'revocations':sorted(revoked,key=lambda h:h['id']),'summary':data['summary']}


def initial_filter():
    return {'theta':[0.]*5,'covariance':np.diag([PARAMETERS['offset_prior_variance'],0,0,0,0]).tolist(),
            'null_mean':0.,'null_variance':PARAMETERS['offset_prior_variance'],'versions':{k:0 for k in SLOTS},'updates':0}


def apply_plan(state,previous,plan,z):
    state=deepcopy(state);theta=np.asarray(state['theta']);cov=np.asarray(state['covariance']);active={h['id'] for h in plan['hypotheses']}
    for k,key in enumerate(SLOTS,1):
        h=next((h for h in plan['hypotheses'] if h['id']==key),None)
        if h is None or h['operation']!='retain':
            theta[k]=PARAMETERS['coefficient_prior_mean'] if h else 0.;cov[k,:]=0.;cov[:,k]=0.
            cov[k,k]=PARAMETERS['coefficient_prior_variance'] if h else 0.
            if h:state['versions'][key]+=1
    hypotheses=[]
    for h in plan['hypotheses']:
        row=deepcopy(h);row['version']=state['versions'][row['id']];row['pool_fraction']=float(membership(z,[row])[0].mean());hypotheses.append(row)
    state['theta']=theta.tolist();state['covariance']=cov.tolist()
    return state,hypotheses


def forecast(state,hypotheses):
    theta=np.asarray(state['theta'],dtype=float);cov=np.asarray(state['covariance'],dtype=float).copy()
    q=np.zeros(5);q[0]=PARAMETERS['offset_process_variance']
    for h in hypotheses:q[SLOTS.index(h['id'])+1]=PARAMETERS['coefficient_process_variance']
    cov+=np.diag(q)
    weights={h['id']:float(np.clip(theta[SLOTS.index(h['id'])+1]-PARAMETERS['trust_sd_multiplier']*np.sqrt(cov[SLOTS.index(h['id'])+1,SLOTS.index(h['id'])+1]),0,1)) for h in hypotheses}
    return {'theta':theta.tolist(),'covariance':cov.tolist(),'weights':weights,
            'null_mean':state['null_mean'],'null_variance':state['null_variance']+PARAMETERS['offset_process_variance']}


def features(z,hypotheses):
    m=membership(z,hypotheses);phi=np.zeros((len(z),5));phi[:,0]=1.
    for row,h in zip(m,hypotheses):phi[:,SLOTS.index(h['id'])+1]=(h['effect_pp']/100)*(row.astype(float)-h['pool_fraction'])
    return phi


def update_filter(state,prior,feature,mu_y,predictive_sd_y,observed_y):
    """One new outcome, paired with the saved prediction from BEFORE querying."""
    h=np.asarray(feature);theta=np.asarray(prior['theta']);p=np.asarray(prior['covariance'])
    residual=float(observed_y-mu_y);r=float(predictive_sd_y**2+PARAMETERS['measurement_variance_floor'])
    innovation=residual-float(h@theta);s=float(h@p@h+r);gain=p@h/s
    updated=theta+gain*innovation;a=np.eye(5)-np.outer(gain,h)
    covariance=a@p@a.T+r*np.outer(gain,gain);covariance=(covariance+covariance.T)/2
    nv=prior['null_variance'];ng=nv/(nv+r);nm=prior['null_mean']+ng*(residual-prior['null_mean'])
    result=deepcopy(state);result.update(theta=updated.tolist(),covariance=covariance.tolist(),null_mean=float(nm),null_variance=float((1-ng)**2*nv+ng**2*r),updates=state['updates']+1)
    preds={'GP':float(mu_y),'GP_offset_null':float(mu_y+prior['null_mean']),'GP_joint_KF':float(mu_y+h@theta)}
    receipt={'prequery_mu_y':float(mu_y),'prequery_predictive_sd_y':float(predictive_sd_y),'observed_y':float(observed_y),'residual':residual,
             'measurement_variance':r,'feature':h.tolist(),'innovation':float(innovation),'innovation_variance':s,'innovation_z':float(innovation/np.sqrt(s)),
             'gain':gain.tolist(),'predictions':preds,'squared_errors':{k:float((observed_y-v)**2) for k,v in preds.items()}}
    return result,receipt


def correction(z,hypotheses,weights,sd_y,retained=None):
    retain=set(weights) if retained is None else set(retained);phi=features(z,hypotheses);change=np.zeros(len(z))
    for h in hypotheses:
        if h['id'] in retain:change+=weights[h['id']]*phi[:,SLOTS.index(h['id'])+1]
    cap=PARAMETERS['mean_cap_sd']*np.asarray(sd_y);return np.clip(change,-cap,cap)


def diagnose(base,z,hypotheses,weights,budget,queried,ids):
    active=[h for h in hypotheses if weights.get(h['id'],0)>PARAMETERS['minimum_trust']]
    return select_candidate(base,membership(z,active),budget,candidate_ids=ids,hypothesis_ids=[h['id'] for h in active],queried_mask=queried)


def event_reasons(iteration,history,events,hypotheses,diagnosis,call_rounds):
    if not call_rounds:return ['initial']
    if len(call_rounds)>=PARAMETERS['max_calls'] or iteration-call_rounds[-1]<PARAMETERS['min_call_gap'] or iteration>PARAMETERS['last_refresh_round']:return []
    reasons=[];n=PARAMETERS['stagnation_rounds']
    if len(events)>=n and max(r['y'] for r in history)-max(r['y'] for r in history[:-n])<PARAMETERS['stagnation_gain_y']:reasons.append('stagnation')
    n=PARAMETERS['bad_innovation_rounds']
    if len(events)>=n and all(abs(e['measurement']['innovation_z'])>PARAMETERS['innovation_threshold'] for e in events[-n:]):reasons.append('consecutive_predictive_errors')
    if diagnosis.conflict_kind!='none':reasons.append('B4_'+diagnosis.conflict_kind)
    if not hypotheses:reasons.append('no_active_knowledge')
    return reasons


class ResidualMeanModel(Model):
    """Cap the knowledge mean in physical yield units; preserve base covariance."""
    def __init__(self,base,hypotheses,weights,retained,y_std):
        super().__init__();self.base=base;self.hypotheses=hypotheses;self.weights=weights;self.retained=set(retained);self.y_std=float(y_std)
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
                cat=(X[...,3]*8).floor().long().clamp(0,7);mask=torch.zeros_like(cat,dtype=torch.bool)
                for k in h['categories']:mask|=cat==k
            else:
                j=VARIABLES.index(h['variable']);lo,hi=h['bounds'];mask=(X[...,j]>=lo)&(X[...,j]<=hi)
            change+=self.weights[h['id']]*(h['effect_pp']/100/self.y_std)*(mask.to(X)-h['pool_fraction'])
        cap=PARAMETERS['mean_cap_sd']*latent.variance.squeeze(-1).clamp_min(1e-18).sqrt()
        mean=p.mvn.mean+change.clamp(min=-cap,max=cap)
        return GPyTorchPosterior(MultivariateNormal(mean,p.mvn.lazy_covariance_matrix.add_jitter(1e-6)))


class KnowledgeCache(UncertaintyCache):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.knowledge_recipe=sha256(Path(__file__).read_bytes()).hexdigest();self.counters['knowledge_acquisitions']=0
    def knowledge_score(self,history,hypotheses,weights,retained):
        fit,value=self.model(history);mc=38102026+self.case*10000+self.seed*100+len(history)-3
        spec={'hypotheses':[{**canonical(h),'pool_fraction':h['pool_fraction']} for h in hypotheses], 'weights':weights,'retained':sorted(retained)}
        key=digest({'fit':fit,'knowledge':spec,'mc_seed':mc,'recipe':self.knowledge_recipe});path=self.directory/(key+'.npz')
        if key in self.scores:return self.scores[key]
        if path.exists():
            with np.load(path,allow_pickle=False) as d:result=(d['scores'].copy(),json.loads(str(d['metadata'])))
        else:
            common.seed_all(mc);model=ResidualMeanModel(value['model'],hypotheses,weights,retained,value['std']).eval()
            acq=qLogExpectedImprovement(model=model,best_f=value['best_std'],sampler=SobolQMCNormalSampler(sample_shape=torch.Size([512]),seed=mc))
            with torch.no_grad():scores=torch.cat([acq(x.unsqueeze(1)).reshape(-1) for x in self.z.split(256)]).numpy()
            if not np.isfinite(scores).all():raise ValueError('nonfinite corrected acquisition')
            meta={'key':key,'fit_key':fit,'mc_seed':mc,'mc_paths':512,'recipe':self.knowledge_recipe,'knowledge':spec,'covariance_jitter':1e-6,'mean_cap_sd':PARAMETERS['mean_cap_sd']}
            with path.with_suffix('.tmp').open('wb') as handle:np.savez_compressed(handle,scores=scores,metadata=json.dumps(meta))
            path.with_suffix('.tmp').replace(path);result=(scores,meta);self.counters['knowledge_acquisitions']+=1
        self.scores[key]=result;return result


def choose(cache,history,z,ids,hypotheses,prior,iteration,knowledge_enabled=True):
    base,bmeta=cache.score(history);queried=np.array([key in {h['candidate_id'] for h in history} for key in ids]);available=~queried
    budget=common.budget_at(PARAMETERS['budget_ratio'],iteration);reference=common.argmax(base,available,ids);eligible=available&((base[reference]-base)<=budget)
    weights=prior['weights'] if knowledge_enabled else {h['id']:0. for h in hypotheses}
    diag=diagnose(base,z,hypotheses,weights,budget,queried,ids)
    if diag.retained_ids:guided,gmeta=cache.knowledge_score(history,hypotheses,weights,diag.retained_ids)
    else:guided,gmeta=base,bmeta
    selected=common.argmax(guided,eligible,ids) if diag.retained_ids else reference
    stats=cache.posterior_stats(history);delta=correction(z,hypotheses,weights,stats['sd_y'],diag.retained_ids)
    assert not queried[selected] and float(base[reference]-base[selected])<=budget
    receipt={'selected_id':ids[selected],'reference_id':ids[reference],'budget_logei':budget,'base_logei_loss':float(base[reference]-base[selected]),
        'base_logei_max':float(base[reference]),'base_logei_selected':float(base[selected]),'guided_logei_selected':float(guided[selected]),
        'eligible_candidates':int(eligible.sum()),'diagnosis':diag.to_dict(),'base_score_key':bmeta['key'],'guided_score_key':gmeta['key'],
        'posterior_key':str(stats['key']),'fit_key':bmeta['fit_key'],'trust':weights,'knowledge_enabled':bool(knowledge_enabled),
        'mean_change_y':float(delta[selected]),'selected_gp_mu_y':float(stats['mu_y'][selected]),'selected_gp_sd_y':float(stats['sd_y'][selected]),
        'selected_predictive_sd_y':float(stats['sd_predictive_y'][selected]),'selected_satisfies_retained':all(membership(z,[h])[0,selected] for h in hypotheses if h['id'] in diag.retained_ids),
        'semantics':'B4 soft prior admission; maximize corrected qLogEI over the entire common budget set'}
    return int(selected),receipt
