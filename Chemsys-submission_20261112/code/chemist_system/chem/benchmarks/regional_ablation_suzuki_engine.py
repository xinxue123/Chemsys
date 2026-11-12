"""Matched regional-mean ablation: gate -> KF -> B4 -> event refresh.

The four arms share the exact regional covariance tilt, not indicator-feature
mean additions. KF calibrates one scalar regional amplitude (initially one).
"""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json

import numpy as np
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.distributions import MultivariateNormal

from . import mixed_comparison_suzuki_engine as common
from . import kalman_knowledge_suzuki_engine as previous
from .suzuki_domain import CATALYSTS,digest

METHODS=('G0_gate','G1_gate_KF','G2_gate_KF_B4','G3_gate_KF_B4_event')
PARAMETERS={'coefficient_initial_mean':1.,'coefficient_initial_variance':.25,
            'offset_initial_variance':.0025,'coefficient_process_variance':.0025,
            'offset_process_variance':.000025,'measurement_variance_floor':.0009,
            'deployment_clip':[0.,1.],'budget_ratio':4,'maximum_refreshes':3,
            'minimum_refresh_gap':2,'last_refresh_round':14,
            'stagnation_rounds':3,'stagnation_gain_y':.005,
            'bad_innovation_rounds':2,'innovation_threshold':2.5}
VARIABLES=('residence_time','temperature','catalyst_loading','catalyst')


def from_advice(advice):
    clauses=[]
    for h in advice['clauses']:
        j=h['coordinate'];lo,hi=h['interval']
        clauses.append({'id':h['id'],'coordinate':j,'bounds':[lo,hi] if j<3 else [],
                        'categories':[k for k in range(8) if lo<=(k+.5)/8<=hi] if j==3 else []})
    return {'confidence':advice['confidence'],'clauses':clauses}


def membership(pool,knowledge):
    z=np.asarray(pool['z']);cats=np.array([CATALYSTS.index(point['catalyst']) for point in pool['points']]);rows=[]
    for h in knowledge['clauses']:
        if h['coordinate']==3:mask=np.isin(cats,h['categories'])
        else:
            lo,hi=h['bounds'];mask=(z[:,h['coordinate']]>=lo)&(z[:,h['coordinate']]<=hi)
        rows.append(mask)
    return np.asarray(rows,dtype=bool).reshape(len(rows),len(z))


def region(knowledge,retained):
    bounds=[[0.,1.] for _ in range(3)];categories=list(range(8))
    for h in knowledge['clauses']:
        if h['id'] not in retained:continue
        if h['coordinate']==3:categories=sorted(set(categories)&set(h['categories']))
        else:
            j=h['coordinate'];bounds[j]=[max(bounds[j][0],h['bounds'][0]),min(bounds[j][1],h['bounds'][1])]
    if not categories or any(lo>=hi for lo,hi in bounds):raise ValueError('empty retained region')
    return {'bounds':bounds,'categories':categories,'confidence':knowledge['confidence']}


def region_key(knowledge,retained):return digest(region(knowledge,retained)) if retained else None


class SetRegionalTilt(common.NominalRegionalTilt):
    """Same numeric quadrature; explicit nominal set supports nonadjacent cats."""
    def __init__(self,categories,category_map=common.mixed.CATEGORY_MAP):
        self.allowed_set=tuple(sorted(categories));super().__init__(category_map)
    def _build_a(self):
        lo,hi=self._region_lb,self._region_ub;self.allowed_categories=self.allowed_set
        xyz=self.Xg[:,:3]
        raw=(torch.sigmoid((xyz-lo[:3])/(self.smooth+1e-12))*
             torch.sigmoid((hi[:3]-xyz)/(self.smooth+1e-12))).prod(-1)
        mask=torch.zeros(len(xyz),dtype=torch.bool)
        for k in self.allowed_set:mask|=self.canonical_category==k
        raw=raw*mask.to(raw);self._a=raw/raw.sum().clamp_min(1e-12)


class ScaledRegionModel(Model):
    def __init__(self,base,tilt,amplitude):
        super().__init__();self.base=base;self.tilt=tilt;self.amplitude=float(amplitude)
    @property
    def num_outputs(self):return self.base.num_outputs
    @property
    def batch_shape(self):return self.base.batch_shape
    def posterior(self,X,observation_noise=False,**kwargs):
        q=self.tilt.posterior(self.base,X,observation_noise=observation_noise,**kwargs)
        if self.amplitude==1.:return q
        b=self.base.posterior(X,observation_noise=observation_noise,**kwargs)
        mean=b.mvn.mean+self.amplitude*(q.mvn.mean-b.mvn.mean)
        return GPyTorchPosterior(MultivariateNormal(mean,q.mvn.lazy_covariance_matrix))


class RegionalAblationCache(common.MixedRegionalCache):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.ablation_recipe=digest({'source':sha256(Path(__file__).read_bytes()).hexdigest(),'mixed':self.recipe,'grid':common.GRID_RECIPE})
        self.tilts={};self.counters.update(scaled_acquisitions=0,set_acquisitions=0)

    def make_tilt(self,history,spec):
        fit,value=self.model(history);key=(fit,digest(spec))
        if key not in self.tilts:
            tilt=SetRegionalTilt(spec['categories'],self.category_map)
            tilt.set_box_region([b[0] for b in spec['bounds']]+[0.],[b[1] for b in spec['bounds']]+[1.])
            tilt.fit_lambda_by_delta(value['model'],common.decide.confidence_to_delta(spec['confidence'],scale=3.))
            tilt.prepare_cache(value['model']);self.tilts[key]=tilt
        return self.tilts[key],value,fit

    def field(self,history,knowledge,retained):
        if not retained:
            score,meta=self.score(history);return score,np.zeros(len(self.ids)),meta
        spec=region(knowledge,retained);cats=spec['categories']
        if cats==list(range(cats[0],cats[-1]+1)):
            box=([b[0] for b in spec['bounds']]+[cats[0]/8],
                 [b[1] for b in spec['bounds']]+[(cats[-1]+1)/8])
            score,meta=super().score(history,box,spec['confidence'])
            with np.load(self.directory/(meta['key']+'.npz'),allow_pickle=False) as d:shift=d['shift_std'].copy()
            return score,shift,meta
        tilt,value,fit=self.make_tilt(history,spec)
        key=digest({'fit':fit,'spec':spec,'recipe':self.ablation_recipe,'amplitude':1.})
        return self._evaluate(history,tilt,value,fit,key,1.,spec)

    def _evaluate(self,history,tilt,value,fit,key,amplitude,spec):
        path=self.directory/(key+'.npz');mc=38102026+self.case*10000+self.seed*100+len(history)-3
        if path.exists():
            with np.load(path,allow_pickle=False) as d:return d['scores'].copy(),d['shift_std'].copy(),json.loads(str(d['metadata']))
        common.regional.seed_all(mc)
        model=ScaledRegionModel(value['model'],tilt,amplitude)
        acq=qLogExpectedImprovement(model=model,best_f=value['best_std'],sampler=SobolQMCNormalSampler(torch.Size([512]),seed=mc))
        scores=[];shifts=[]
        with torch.no_grad():
            for x in self.z.split(256):
                xx=x.unsqueeze(1);scores.append(acq(xx).reshape(-1));shifts.append((model.posterior(xx).mean-value['model'].posterior(xx).mean).reshape(-1))
        score=torch.cat(scores).numpy();shift=torch.cat(shifts).numpy();assert np.isfinite(score).all()
        meta={'key':key,'fit_key':fit,'mc_seed':mc,'mc_paths':512,'covariance_jitter':1e-6,'region':spec,
              'amplitude':float(amplitude),'recipe':self.ablation_recipe,'lambda':float(tilt._lam)}
        with path.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,scores=score,shift_std=shift,metadata=json.dumps(meta))
        path.with_suffix('.tmp').replace(path)
        self.counters['scaled_acquisitions' if amplitude!=1 else 'set_acquisitions']+=1
        return score,shift,meta

    def calibrated_score(self,history,knowledge,retained,amplitude):
        if not retained or amplitude==0.:
            score,meta=self.score(history);return score,np.zeros(len(self.ids)),meta
        if amplitude==1.:return self.field(history,knowledge,retained)
        spec=region(knowledge,retained);tilt,value,fit=self.make_tilt(history,spec)
        key=digest({'fit':fit,'spec':spec,'recipe':self.ablation_recipe,'amplitude':float(amplitude)})
        return self._evaluate(history,tilt,value,fit,key,float(amplitude),spec)


def new_bank(iteration):
    return {'mean':[0.,1.],'covariance':np.diag([.0025,.25]).tolist(),'last_round':iteration-1,'updates':0}


def forecast(bank,iteration):
    elapsed=iteration-bank['last_round'];assert elapsed>=1
    cov=np.asarray(bank['covariance'])+np.diag([.000025,.0025])*elapsed
    return {'mean':list(bank['mean']),'covariance':cov.tolist(),'elapsed_rounds':elapsed,
            'amplitude':float(np.clip(bank['mean'][1],0,1))}


def update_bank(bank,prior,iteration,raw_shift_y,mu_y,predictive_sd_y,y):
    feature=np.array([1.,raw_shift_y]);theta=np.asarray(prior['mean']);cov=np.asarray(prior['covariance'])
    variance=float(predictive_sd_y**2+.0009);residual=float(y-mu_y)
    innovation=float(residual-feature@theta);innovation_variance=float(feature@cov@feature+variance)
    gain=cov@feature/innovation_variance;after=theta+gain*innovation
    a=np.eye(2)-np.outer(gain,feature);pc=a@cov@a.T+variance*np.outer(gain,gain);pc=(pc+pc.T)/2
    result={'mean':after.tolist(),'covariance':pc.tolist(),'last_round':iteration,'updates':bank['updates']+1}
    m={'feature':feature.tolist(),'prequery_mu_y':float(mu_y),'prequery_predictive_sd_y':float(predictive_sd_y),
       'raw_region_shift_y':float(raw_shift_y),'observed_y':float(y),'residual':residual,
       'measurement_variance':variance,'innovation':innovation,'innovation_variance':innovation_variance,
       'innovation_z':float(innovation/np.sqrt(innovation_variance)),'gain':gain.tolist()}
    return result,m


def diagnose(base,pool,knowledge,budget,queried):
    return previous.select_candidate(base,membership(pool,knowledge),budget,candidate_ids=pool['ids'],
                                     hypothesis_ids=[h['id'] for h in knowledge['clauses']],queried_mask=queried)


def setup(cache,state,pool,iteration,knowledge=None):
    knowledge=state['knowledge'] if knowledge is None else knowledge
    base,bmeta=cache.score(state['history']);available=~np.isin(pool['ids'],[h['candidate_id'] for h in state['history']])
    ref=common.regional.argmax(base,available,pool['ids']);budget=common.regional.budget_at(4,iteration)
    eligible=available&((base[ref]-base)<=budget)
    dg=diagnose(base,pool,knowledge,budget,~available)
    retained=list(dg.retained_ids) if state['method'] in METHODS[2:] else sorted(h['id'] for h in knowledge['clauses'])
    _,raw_shift,rmeta=cache.field(state['history'],knowledge,retained)
    key=region_key(knowledge,retained);bank=None;prior=None;alpha=1.
    if state['method']!=METHODS[0] and key:
        bank=deepcopy(state['banks'].get(key,new_bank(iteration)));prior=forecast(bank,iteration);alpha=prior['amplitude']
    score,shift,meta=cache.calibrated_score(state['history'],knowledge,retained,alpha)
    j=common.regional.argmax(score,eligible,pool['ids']);stats=cache.posterior_stats(state['history'])
    return {'base':base,'score':score,'raw_shift_std':raw_shift,'shift_std':shift,'stats':stats,'bank':bank,'prior':prior,
            'selected':j,'bank_key':key,'reference':ref,'available':available,'eligible':eligible,'budget':budget,
            'retained':retained,'diagnosis':dg,'amplitude':alpha,'base_meta':bmeta,'field_meta':rmeta,'guided_meta':meta}


def event_reasons(state,decision,t):
    # Initial knowledge is loaded, not regenerated. The virtual round1 call
    # preserves the previous cooldown / maximum3 additional-refresh schedule.
    return previous.event_reasons(t,state['history'],state['decisions'],state['knowledge']['clauses'],
                                  decision['diagnosis'],[1]+[c['iteration'] for c in state['calls']])


def validate_refresh(data,knowledge,slots,history_count,pool):
    if not isinstance(data,dict) or set(data)!={'actions','summary'}:raise ValueError('expected actions,summary')
    if not isinstance(data['summary'],str) or not 1<=len(data['summary'])<=1600:raise ValueError('invalid summary')
    if not isinstance(data['actions'],list) or len(data['actions'])>4:raise ValueError('at most4 actions')
    old={h['id']:h for h in knowledge['clauses']};seen=set();new={};actions=[]
    for row in data['actions']:
        if not isinstance(row,dict) or set(row)!={'id','operation','bounds','categories','reason','support'}:raise ValueError('invalid action fields')
        key=row['id'];op=row['operation']
        if key not in slots or key in seen:raise ValueError('unknown/duplicate slot')
        seen.add(key)
        if not isinstance(row['reason'],str) or not 1<=len(row['reason'])<=1200:raise ValueError('invalid reason')
        support=row['support']
        if not isinstance(support,list) or not support or any(type(i)!=int or not 1<=i<=history_count for i in support):raise ValueError('unobserved support')
        if op in ('retain','revoke'):
            if key not in old or row['bounds']!=[] or row['categories']!=[]:raise ValueError('retain/revoke require active slot and empty bounds/categories')
            if op=='retain':new[key]=deepcopy(old[key])
        elif op in ('add','revise'):
            if (op=='add')!=(key not in old):raise ValueError('add vacant, revise active')
            j=slots[key];h={'id':key,'coordinate':j,'bounds':[],'categories':[]}
            if j==3:
                cats=row['categories']
                if row['bounds']!=[] or not isinstance(cats,list) or not 1<=len(cats)<=7 or len(set(cats))!=len(cats) or any(type(k)!=int or k not in range(8) for k in cats):raise ValueError('invalid nominal catalyst set')
                h['categories']=sorted(cats)
            else:
                b=row['bounds']
                if row['categories']!=[] or not isinstance(b,list) or len(b)!=2 or any(type(x) not in (int,float) or not np.isfinite(x) or abs(x-round(x,4))>1e-10 for x in b):raise ValueError('invalid numeric bounds')
                if not 0<=b[0]<b[1]<=1 or not .1-1e-12<=b[1]-b[0]<=.9+1e-12:raise ValueError('interval width .1..9')
                h['bounds']=list(map(float,b))
            m=membership(pool,{'clauses':[h]})[0]
            if not m.any() or m.all():raise ValueError('predicate must partition pool')
            if op=='revise' and h==old[key]:raise ValueError('identical geometry must retain')
            new[key]=h
        else:raise ValueError('unknown operation')
        actions.append(deepcopy(row))
    if not set(old)<=seen:raise ValueError('account for every active clause')
    return {'actions':actions,'summary':data['summary'],
            'knowledge':{'confidence':knowledge['confidence'],'clauses':[new[k] for k in sorted(new)]}}


SYSTEM_PROMPT='''You update a preferred Suzuki reaction REGION for an optimizer.
Use only this request's own observed experiments, saved prequery predictions,
and current GP summaries. Do not identify a benchmark or recall its outcomes.
Catalyst names are anonymous nominal labels; do not invent ligand structures.
All arms started with the exact supplied initial regional knowledge. It is an
uncertain hypothesis, not certified chemistry. A region is the intersection of
up to four coordinate preferences. The same analytic regional-covariance mean
tilt is used throughout. Its fixed confidence is NOT editable; a joint scalar
Kalman offset/amplitude calibration learns only from new prequery residuals.
Its clipped posterior mean amplitude is heuristic, not a truth probability.
B4 tests compatibility with the acquisition budget, not chemical correctness.
Revise or revoke knowledge that is contradicted, already captured by the CURRENT
GP, or too broad to support. Prequery residuals arose under different GP fits;
do not treat old GP underprediction as a stable residual of today's GP.
Return exactly {"actions":[...],"summary":"..."}. Each action has exactly
id,operation,bounds,categories,reason,support. Use the supplied fixed slot-to-axis
map; never change a slot's axis. Account for EVERY currently active slot with
retain,revise,revoke. A vacant slot may be added, or omitted. At most4 actions.
retain/revoke MUST set bounds=[] and categories=[]; retain preserves the exact
existing geometry automatically (do not retype bounds). add/revise for a numeric
axis uses two normalized bounds in[0,1], <=4 decimals, width0.1..0.9, categories=[].
For a catalyst axis use bounds=[] and1..7 unique indices0..7 as a nominal SET.
reason is a concise rationale (<=1200 chars); support is a nonempty list of
1-based observed experiment indices. Summary <=1600 chars. No other fields.
No point proposals, hidden-yield predictions, new effect sizes or confidence
values. You may revoke all active knowledge if it adds no defensible information.
No experiment occurs during schema repair. Output only the JSON object.'''
