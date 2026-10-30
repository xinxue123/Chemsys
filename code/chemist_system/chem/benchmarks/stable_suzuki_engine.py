"""Stage11 policy transport to the original mixed Reizman candidate pools.

No generated experiments, molecular identities, nominal distances or oracle
values enter this module. The all-categorical molecular rescue is inapplicable.
"""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import numpy as np
import torch
from gpytorch.distributions import MultivariateNormal
from botorch.posteriors.gpytorch import GPyTorchPosterior

from . import regional_ablation_suzuki_engine as original
from . import grounded_regional_suzuki_engine as grounded
from . import r2_region_trust_engine as trust
from .stable_information_engine import priority, information_proxy, competition_ranks
from .suzuki_domain import digest

PARAMETERS={**original.PARAMETERS, 'tie_seed':0, 'validation_rounds':[2,4,6],
    'validation_high_yield_threshold':.95, 'successful_counterexample_gain_y':.005,
    'regional_covariance_units':'cached internal covariance times outcome_transform.stdvs squared',
    'molecular_recovery':'inapplicable: anonymous catalysts and three continuous axes',
    'joint_tests':'inapplicable: only one categorical axis, no molecular context catalog'}


class CorrectedSetTilt(original.SetRegionalTilt):
    """Same correction as Stage11: cached and joint-posterior covariance units."""
    def posterior(self,base_model,X,observation_noise=False,**kwargs):
        p=super().posterior(base_model,X,observation_noise=observation_noise,**kwargs)
        if getattr(self,'_cache',None) is None:return p
        outcome=getattr(base_model,'outcome_transform',None)
        if outcome is None:return p
        assert outcome.stdvs.numel()==1
        b=base_model.posterior(X,observation_noise=observation_noise,**kwargs)
        mean=b.mvn.mean+outcome.stdvs.square().reshape(())*(p.mvn.mean-b.mvn.mean)
        return GPyTorchPosterior(MultivariateNormal(mean,p.mvn.lazy_covariance_matrix))


class Cache(original.RegionalAblationCache):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.ablation_recipe=digest({'source':sha256(Path(__file__).read_bytes()).hexdigest(),
            'mixed':self.recipe,'grid':original.common.GRID_RECIPE,'parameters':PARAMETERS})

    def make_tilt(self,history,spec):
        fit,v=self.model(history);key=(fit,digest(spec))
        if key not in self.tilts:
            tilt=CorrectedSetTilt(spec['categories'],self.category_map)
            tilt.set_box_region([b[0] for b in spec['bounds']]+[0.], [b[1] for b in spec['bounds']]+[1.])
            tilt.fit_lambda_by_delta(v['model'],original.common.decide.confidence_to_delta(spec['confidence'],scale=3.))
            tilt.prepare_cache(v['model']);self.tilts[key]=tilt
        return self.tilts[key],v,fit

    def field(self,history,knowledge,retained):
        if not retained:
            score,meta=self.score(history);return score,np.zeros(len(self.ids)),meta
        spec=original.region(knowledge,retained);tilt,v,fit=self.make_tilt(history,spec)
        key=digest({'fit':fit,'spec':spec,'recipe':self.ablation_recipe,'amplitude':1.})
        return self._evaluate(history,tilt,v,fit,key,1.,spec)

    def release(self):
        self.models.clear();self.scores.clear();self.posterior_memory.clear();self.tilts.clear()


def canonical_region(knowledge,retained):
    if not retained:return None
    spec=original.region(knowledge,retained)
    # The existing trust function uses sets only. Bounds here are a canonical
    # geometric pair, not a list of categorical continuous levels.
    return {'allowed':{**{a:list(b) for a,b in zip(original.VARIABLES[:3],spec['bounds'])},
                       'catalyst':list(spec['categories'])}}


def setup(cache,state,pool,t):
    history=state['history'];knowledge=state['knowledge']
    base,bmeta=cache.score(history);stats=cache.posterior_stats(history)
    available=~np.isin(pool['ids'],[h['candidate_id'] for h in history])
    ix={k:i for i,k in enumerate(pool['ids'])}
    proxy,old_pairs,unseen=information_proxy(cache.z.numpy(),[ix[h['candidate_id']] for h in history])
    ranks=priority(len(base),t,state.get('tie_seed',0))
    def choose(scores,mask):
        js=np.flatnonzero(mask);js=js[scores[js]==np.max(scores[js])]
        js=js[proxy[js]==np.max(proxy[js])]
        sd=stats['sd_y'][js];js=js[sd==sd.max()]
        return int(js[np.argmin(ranks[js])])
    ref=choose(base,available);budget=original.common.regional.budget_at(4,t)
    eligible=available&(base[ref]-base<=budget)
    dg=original.diagnose(base,pool,knowledge,budget,~available);retained=list(dg.retained_ids)
    mem=original.membership(pool,knowledge)
    positions=[i for i,c in enumerate(knowledge['clauses']) if c['id'] in retained]
    feasible=eligible&(mem[positions].all(0) if positions else True)
    witness=choose(base,feasible)
    dg=replace(dg,candidate_index=witness,candidate_id=pool['ids'][witness])
    _,raw,fmeta=cache.field(history,knowledge,retained)
    spec=canonical_region(knowledge,retained)
    key=trust.geometry_key(spec);bank=deepcopy(state['banks'].get(key,original.new_bank(t))) if key else None
    prior=original.forecast(bank,t) if bank is not None else None
    conflict=bool(dg.revoked_ids) or dg.conflict_kind!='none'
    alpha=prior['amplitude'] if prior is not None else 1.
    d={'base':base,'reference':ref,'budget':budget,'available':available,'eligible':eligible,
       'stats':{**stats,'information_proxy':proxy},'bank_key':key,'bank':bank,'prior':prior,
       'raw_shift_std':raw,'field_meta':fmeta,'base_meta':bmeta,'diagnosis':dg,
       'retained':retained,'region':spec,'amplitude':0. if conflict else alpha,'fallback':conflict}
    gate=trust.permission(state,d,t);d['trust']=gate;d['amplitude']=gate['amplitude_after_trust']
    d['fallback']=bool(conflict or gate['trust_blocked'])
    score,shift,gmeta=cache.calibrated_score(history,knowledge,retained,d['amplitude'])
    j=choose(score,eligible);assert not d['fallback'] or j==ref
    d.update(score=score,shift_std=shift,guided_meta=gmeta,selected=j)
    d['tie_order']={'seed':state.get('tie_seed',0),'equality':'exact_float','priority_sha256':digest(ranks.tolist()),
        'base_max_ties':int(np.sum(available&(base==base[ref]))),
        'guided_max_ties':int(np.sum(eligible&(score==score[j]))),
        'information_proxy_at_reference':float(proxy[ref]),'information_proxy_at_proposal':float(proxy[j]),
        'isolated_pairs':dict(zip(original.VARIABLES,old_pairs.tolist())),
        'unseen_values':dict(zip(original.VARIABLES,unseen.tolist())),
        'rule':'exact score ties: one-axis contrast information proxy, latent SD, independent PCG64 priority; never candidate ID'}
    _,v=cache.model(history)
    lengths={n:p.detach().reshape(-1).tolist() for n,p in v['model'].named_parameters() if 'raw_lengthscale' in n}
    d['GP_health']={'model':'MixedSingleTaskGP','lengthscale_raw_parameters':lengths,
        'categorical_dimensions':[3], 'molecular_recovery_applied':False,
        'reason':'Three continuous axes and anonymous catalyst labels; Stage11 molecular fallback requires real structures.'}
    return d


def event_reasons(state,d,t):
    reasons=original.event_reasons(state,d,t)
    # Same maximum, cooldown and last-call boundary as ordinary refreshes.
    last=[1]+[c['iteration'] for c in state['calls']]
    if len(state['calls'])>=3 or t-last[-1]<2 or t>14:return reasons
    recent=next((c for c in state['validations'] if c['query']==t-1),None)
    if recent and recent['tested_yield_percent']>=recent['incumbent_before_percent']+.5:
        reasons=list(dict.fromkeys(reasons+['successful_controlled_counterexample']))
    return reasons


feedback=trust.feedback
