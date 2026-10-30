"""Nominal Mixed-GP adapters; existing regional/B4/Justify rules stay upstream.

Only the regional quadrature domain needs a mixed-domain transport: 64 numeric
Sobol locations crossed with eight catalyst names, not a continuous category.
The vendor mean-tilt formula, confidence calibration and covariance are reused.
"""
from hashlib import sha256
import importlib
import json
from pathlib import Path

import numpy as np
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler

from . import guarded_suzuki_engine as regional
from . import llinbo_guarded_engine as justify
from . import mixed_kalman_suzuki_engine as mixed
from .official_suzuki_adapter import modules
from .suzuki_domain import CATALYSTS, digest, validate_point

modules('LGBO')
native = importlib.import_module('prior')
decide = importlib.import_module('decide')

GRID_RECIPE = {'total':512, 'numeric_sobol':64, 'catalysts':8,
               'continuous_smooth':.08, 'categorical_weight':'exact set membership'}


class NominalRegionalTilt(native.LinearExponentialRegionalMeanTiltPlugAndPlay):
    """Keep official analytic tilt; adapt grid and weights to named categories."""
    def __init__(self, category_map=mixed.CATEGORY_MAP):
        if len(category_map)!=8 or len(set(category_map))!=8:
            raise ValueError('nominal labels require a bijection')
        super().__init__(torch.tensor([[0.]*4,[1.]*4],dtype=torch.double),
                         grid_size=512,smooth=.08,dtype=torch.double)
        self.category_map=tuple(category_map)
        xyz=torch.quasirandom.SobolEngine(3).draw(64).double().clamp(1e-6,1-1e-6)
        self.canonical_category=torch.arange(8).repeat(64)
        labels=torch.tensor(self.category_map,dtype=torch.double).repeat(64)
        self.Xg=torch.column_stack([xyz.repeat_interleave(8,dim=0),labels])

    def _build_a(self):
        # The legacy interval is decoded ONCE into a set of catalyst names.
        # It is never a range/distance on the GP's arbitrary category labels.
        lo,hi=self._region_lb,self._region_ub
        self.allowed_categories=tuple(k for k in range(8)
                                      if float(lo[3])<=(k+.5)/8<=float(hi[3]))
        if not self.allowed_categories:raise ValueError('empty category set')
        xyz=self.Xg[:,:3]
        raw=(torch.sigmoid((xyz-lo[:3])/(self.smooth+1e-12))*
             torch.sigmoid((hi[:3]-xyz)/(self.smooth+1e-12))).prod(-1)
        mask=torch.zeros(512,dtype=torch.bool)
        for k in self.allowed_categories:mask|=self.canonical_category==k
        raw=raw*mask.to(raw)
        self._a=raw/raw.sum().clamp_min(1e-12)


class MixedRegionalCache(mixed.MixedKnowledgeCache):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.regional_recipe=digest({'engine':sha256(Path(__file__).read_bytes()).hexdigest(),
                                    'mixed_recipe':self.recipe,'grid':GRID_RECIPE})
        self.counters['regional_acquisitions']=0

    def score(self,history,box=None,confidence=.7):
        if box is None:return super().score(history)
        fit,value=self.model(history)
        mc=38102026+self.case*10000+self.seed*100+len(history)-3
        key=digest({'fit':fit,'box':box,'confidence':confidence,'mc_seed':mc,
                    'regional_recipe':self.regional_recipe})
        if key in self.scores:
            self.counters['score_cache_hits']+=1;return self.scores[key]
        path=self.directory/(key+'.npz');self.ensure(path.name)
        if path.exists():
            with np.load(path,allow_pickle=False) as data:
                result=data['scores'].copy(),json.loads(str(data['metadata']))
            assert result[1]['key']==key
            self.counters['score_cache_hits']+=1
        else:
            regional.seed_all(mc)
            tilt=NominalRegionalTilt(self.category_map);tilt.set_box_region(*box)
            delta=decide.confidence_to_delta(confidence,scale=3.)
            tilt.fit_lambda_by_delta(value['model'],delta,observation_noise=False)
            tilt.prepare_cache(value['model'])
            effective=native.TiltedModel(value['model'],tilt).eval()
            acquisition=qLogExpectedImprovement(model=effective,best_f=value['best_std'],
                sampler=SobolQMCNormalSampler(torch.Size([512]),seed=mc))
            with torch.no_grad():
                scores=torch.cat([acquisition(x.unsqueeze(1)).reshape(-1)
                                  for x in self.z.split(256)]).numpy()
                shifts=torch.cat([(effective.posterior(x.unsqueeze(1)).mean-
                    value['model'].posterior(x.unsqueeze(1)).mean).reshape(-1)
                    for x in self.z.split(256)]).numpy()
            assert len(scores)==len(self.ids) and np.isfinite(scores).all()
            meta={'key':key,'fit_key':fit,'fit_seed':value['fit_seed'],'mc_seed':mc,
                  'mc_paths':512,'gp_class':'MixedSingleTaskGP','cat_dims':[3],
                  'category_map':list(self.category_map),'covariance_jitter':1e-6,
                  'regional_recipe':self.regional_recipe,'grid':GRID_RECIPE,
                  'box':box,'confidence':confidence,'delta':float(delta),
                  'lambda':float(tilt._lam),'category_names':[CATALYSTS[k] for k in tilt.allowed_categories]}
            with path.with_suffix('.tmp').open('wb') as handle:
                np.savez_compressed(handle,scores=scores,shift_std=shifts,
                                    grid=tilt.Xg.numpy(),weights=tilt._a.numpy(),metadata=json.dumps(meta))
            path.with_suffix('.tmp').replace(path)
            result=scores,meta;self.counters['regional_acquisitions']+=1
        self.scores[key]=result;return result


POINT_SYSTEM='''You propose one next Suzuki experiment from ONLY the observations
in this request. Do not identify the benchmark, recall its outcomes, invent
ligand structures for anonymous catalysts, or transfer other trajectories.
The original LLINBO prompt is included verbatim. Its fourth vector coordinate
is only an I/O label code, with no distance or chemical order. You may question
the supplied uncertain preferences using your own observations. No BO scores
or unseen yields are available to you. Return one JSON object with exactly
point,reason,support. point has catalyst (one of the supplied names), t_res
(60..600 seconds), temperature (30..110 C), catalyst_loading (0.5..2.5 mol%).
reason is a concise experimental rationale, not private reasoning. support is
a nonempty list of this request's 1-based observed indices. The physical point
is projected to the nearest unused same-catalyst fixed-pool point using numeric
coordinates only, before the unchanged Justify acquisition-budget review.'''


def parse_point(answer,history):
    if answer.get('model')!='deepseek-flash':raise ValueError('unexpected model')
    choices=answer.get('choices',[])
    if len(choices)!=1 or choices[0]['finish_reason']!='stop':
        raise ValueError('expected one complete JSON response')
    raw=json.loads(choices[0]['message']['content'])
    if not isinstance(raw,dict) or set(raw)!={'point','reason','support'}:
        raise ValueError('expected exactly point,reason,support')
    point=validate_point(raw['point'])
    if not isinstance(raw['reason'],str) or not raw['reason'].strip():raise ValueError('empty reason')
    support=raw['support']
    if not isinstance(support,list) or not support or any(type(i)!=int or not 1<=i<=len(history) for i in support):
        raise ValueError('support must reference observed indices')
    return dict(point=point,reason=raw['reason'],support=support)


def public_point_request(history,prompt,t,pool,advice,task):
    ix={key:i for i,key in enumerate(pool['ids'])}
    return {'task':task,'iteration':t,'remaining_queries':17-t,
            'original_LLINBO_prompt':prompt,'domain_catalysts':list(CATALYSTS),
            'observations':[{'index':i,'point':pool['points'][ix[h['candidate_id']]],
                             'yield_fraction':h['y']} for i,h in enumerate(history,1)],
            'uncertain_preferences':json.loads(justify.description(advice).split('A supplied preference is uncertain and may be questioned using this trajectory observations: ')[1].split('. Use only')[0])}
