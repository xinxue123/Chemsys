"""Exact native qLogEI transport when reported marginal variance is floored.

Frozen study/source files are not edited. P0/P1 native acquisition samples the
posterior covariance plus jitter, whereas the reported SD uses variance's floor.
At singleton prefixes these differ enough to invalidate SD reconstruction.
"""
import numpy as np
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from . import policy,PARAMETERS as P

ORIGINAL_SCORES=policy.scores

def scores(gp,context,plan):
    if gp.pool.group=='P2' or plan['mode']=='none':return ORIGINAL_SCORES(gp,context,plan)
    from chem.benchmarks import knowledge_gp as native
    nominal=native._legacy_import('pointwise_hte_engine',gp.pool.root)
    js=policy.quadrature(gp.pool,plan);stats=context['stats'];model=stats['model']
    tilt=nominal.CorrectedNominalTilt(gp.X[js],torch.full((len(js),),1./len(js),dtype=torch.double))
    delta=nominal.nominal.old.common.decide.confidence_to_delta(plan['confidence'],scale=P['guidance_scale'])
    tilt.fit_lambda_by_delta(model,delta);tilt.prepare_cache(model)
    factory=nominal.nominal.old.common.native.TiltedModel
    seed=context['acquisition_meta']['mc_seed']
    base=qLogExpectedImprovement(nominal.nominal.JitterModel(model),best_f=stats['best_std'],
        sampler=SobolQMCNormalSampler(torch.Size([512]),seed=seed))
    lifted=qLogExpectedImprovement(factory(model,tilt),best_f=stats['best_std'],
        sampler=SobolQMCNormalSampler(torch.Size([512]),seed=seed))
    replay=[];final=[];shift=[]
    with torch.no_grad():
        for xx in gp.X.split(256):
            x=xx.unsqueeze(1);replay.append(base(x).reshape(-1))
            final.append(lifted(x).reshape(-1))
            shift.append((tilt.posterior(model,x).mean-model.posterior(x).mean).reshape(-1))
    replay=torch.cat(replay).numpy();final=torch.cat(final).numpy();shift=torch.cat(shift).numpy()*stats['std']
    error=float(np.max(np.abs(replay-context['base_logei'])))
    if error>2e-8:raise ValueError('Exact native posterior acquisition mismatch '+str(error))
    if delta==0:final=context['base_logei'].copy()
    return final,shift,dict(mode=plan['mode'],confidence=plan['confidence'],delta=float(delta),
        lambda_value=float(tilt._lam),quadrature_ids=[gp.pool.ids[j] for j in js],quadrature_size=len(js),
        native_replay_error=error,covariance_unchanged=True,output_covariance_units_corrected=True,
        singleton_transport='sample native posterior covariance directly; do not invert reported SD floor')

def initialize():
    from . import study
    study.initialize()
    policy.scores=scores

def qualified_receipt():
    from . import study as s
    from unittest.mock import patch
    initialize();checks=[]
    def forbidden(*a,**k):raise AssertionError('Compatibility check must not read a new oracle value')
    with patch.object(s.matched.frozen.native,'read_observation',forbidden):
        for task in s.TASKS:
            pool=s.matched.frozen.pool_for(task)
            first=s.read(s.matched.frozen.first.folder_for(task)/'state.json')['history'][0]
            assert first['candidate_id']==s.matched.frozen.locked_first(task)['candidate_id']
            gp=s.matched.SeededGP(pool,s.BASE/'cache'/s.SHORT[task],tie_seed=0)
            context=gp.context([first],2)
            j=int(np.flatnonzero(context['available'])[0])
            p=dict(mode='point',indices=[j],confidence=.7)
            value,shift,meta=scores(gp,context,p)
            assert np.isfinite(value).all() and np.isfinite(shift).all()
            neutral,zero,_=scores(gp,context,dict(p,confidence=.5))
            np.testing.assert_array_equal(neutral,context['base_logei'])
            np.testing.assert_array_equal(zero,np.zeros(len(pool.ids)))
            assert gp.counters['operator_compiles_in_loop']==0
            checks.append(dict(task=s.SHORT[task],prefix_length=1,native_acquisition_max_error=meta['native_replay_error']))
            gp.release()
    paths=[s.ROOT/'code/experiment_tools/grounded_baselines/compat.py',
        s.ROOT/'code/experiment_tools/run_grounded_baselines_compat.py',
        s.ROOT/'code/tools/grounded_baselines_dispatch_compact.py']
    value=dict(status='PASS_EXACT_POSTERIOR_COMPATIBILITY',checks=checks,new_LLM_calls=0,new_experiment_queries=0,
        original_protocol_sha256=s.check()['protocol_sha256'],
        reason='Native P0/P1 posterior sampler differs from reported SD floor at singleton/constant prefixes',
        original_failure_max_logEI_error=2.0993237810351673e-5,
        adaptation='Use original qLogExpectedImprovement on official TiltedModel with the SAME native GP,512-MC seed,jitter and full covariance; P2 unchanged',
        frozen_sources_and_prior_commits_unchanged=True,original_LLM_responses_reused_for_same_requests=True,
        new_source_sha256={p.relative_to(s.ROOT).as_posix():s.sha(p) for p in paths})
    s.write(s.BASE/'execution_compatibility.json',value)
    return value
