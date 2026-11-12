"""Official finite-region mean tilt and stable Insight selection adapters."""
import ast
from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
import math
import re
from unittest.mock import patch
import numpy as np
import torch
from chemical_hypothesis_bo.experts import logei
from chemical_hypothesis_bo.hypotheses import digest
from grounded_first_start import proposal
from hypothesis_scope_loop.knowledge import Scopes
from knowledge_rank_v9 import llm as official
from . import PARAMETERS as P

_EXTRACTOR=None

def warm(root):
    """Compile the official extractor once, before any experimental round."""
    global _EXTRACTOR
    if _EXTRACTOR is not None:return
    path=root/'code/chemist_system/third_party/reasoning_bo_official/src/bo/reasoner/base.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BaseReasoner')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_extract_candidates_from_insight')
    namespace={'json':json,'re':re}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),namespace)
    _EXTRACTOR=namespace[method.name]

def extract(root,raw):
    assert _EXTRACTOR is not None,'Prepare official extractor before the round loop'
    with redirect_stdout(io.StringIO()):return _EXTRACTOR(None,json.dumps(raw),n=1)

def validate_rbo(root,raw,ids):
    with patch.object(official,'official_extract',extract):
        return official.validate_rbo(root,raw,ids)

def validate_lgbo(raw,pool,history,query,seed):
    if not isinstance(raw,dict) or set(raw)!={'summary','mode','confidence','conditions','scope','rationale'}:
        raise ValueError('Use exactly summary,mode,confidence,conditions,scope,rationale')
    if any(not isinstance(raw[k],str) or not raw[k] or len(raw[k])>1800 for k in ('summary','rationale')):
        raise ValueError('Provide concise summary and scientific rationale')
    if raw['mode'] not in ('point','region','none'):raise ValueError('mode must be point,region or none')
    confidence=raw['confidence']
    if type(confidence) not in (float,int) or not math.isfinite(confidence) or not 0<confidence<1:
        raise ValueError('Finite confidence strictly between0 and1 required')
    available=~np.isin(pool.ids,[h['candidate_id'] for h in history])
    if raw['mode']=='none':
        if raw['conditions'] is not None or raw['scope']!={}:raise ValueError('Abstention requires null conditions and empty scope')
        return dict(mode='none',confidence=float(confidence),indices=[])
    if raw['mode']=='region':
        if raw['conditions'] is not None or not isinstance(raw['scope'],dict) or not raw['scope']:
            raise ValueError('A region requires nonempty scope and null conditions')
        # Scopes validates physical bounds and exact categorical identities.
        mask=Scopes(pool).mask(raw['scope'])
        if not (mask&available).any():raise ValueError('Region must contain an unqueried executable input')
        return dict(mode='region',confidence=float(confidence),indices=np.flatnonzero(mask).tolist(),scope=deepcopy(raw['scope']))
    if raw['scope']!={}:raise ValueError('Point mode requires empty scope')
    valid,distance=proposal.distances(pool,raw['conditions'])
    js=np.flatnonzero(valid&available)
    if not len(js):raise ValueError('No unqueried executable point with identical categories')
    from chem.benchmarks.knowledge_gp import semantic_priorities
    tie=semantic_priorities(pool,query,seed)
    j=int(js[np.lexsort((tie[js],distance[js]))[0]])
    return dict(mode='point',confidence=float(confidence),indices=[j],scope={},
        projected_id=pool.ids[j],input_distance=float(distance[j]))

def quadrature(pool,plan):
    js=sorted(plan['indices'],key=lambda j:pool.features[j])
    if len(js)>P['quadrature_cap']:
        key=digest(dict(mode=plan['mode'],features=[pool.features[j] for j in js]))
        positions=np.random.default_rng(int(key[:12],16)).choice(len(js),P['quadrature_cap'],replace=False)
        js=[js[int(i)] for i in sorted(positions)]
    return js

def scores(gp,context,plan):
    stats=context['stats'];base=context['base_logei']
    if plan['mode']=='none':
        return base.copy(),np.zeros(len(base)),dict(mode='none',native_replay_error=0.,quadrature_ids=[])
    from chem.benchmarks import knowledge_gp as native
    nominal=native._legacy_import('pointwise_hte_engine',gp.pool.root)
    CorrectedNominalTilt=nominal.CorrectedNominalTilt
    old=nominal.nominal.old
    js=quadrature(gp.pool,plan)
    tilt=CorrectedNominalTilt(gp.X[js],torch.full((len(js),),1./len(js),dtype=torch.double))
    delta=old.common.decide.confidence_to_delta(plan['confidence'],scale=P['guidance_scale'])
    model=stats['model']
    tilt.fit_lambda_by_delta(model,delta)
    tilt.prepare_cache(model)
    shifts=[]
    with torch.no_grad():
        for xx in gp.X.split(256):
            x=xx.unsqueeze(1)
            shifts.append((tilt.posterior(model,x).mean-model.posterior(x).mean).reshape(-1))
    shift=torch.cat(shifts).numpy()
    mu=(stats['mu_y']-stats['mean'])/stats['std'];sd=stats['sd_y']/stats['std']
    seed=context['acquisition_meta']['mc_seed'];best=stats['best_std']
    reproduced=logei(mu,sd,best,seed)
    error=float(np.max(np.abs(reproduced-base)))
    if error>2e-8:raise ValueError('Native acquisition reconstruction failed '+str(error))
    score=logei(mu+shift,sd,best,seed) if np.any(shift) else base.copy()
    return score,shift*stats['std'],dict(mode=plan['mode'],confidence=plan['confidence'],
        delta=float(delta),lambda_value=float(tilt._lam),quadrature_ids=[gp.pool.ids[j] for j in js],
        quadrature_size=len(js),native_replay_error=error,
        covariance_unchanged=True,output_covariance_units_corrected=True)

def select(context,score):
    js=np.flatnonzero(context['available']);js=js[score[js]==score[js].max()]
    return int(js[np.argmin(context['tie_priority'][js])])
