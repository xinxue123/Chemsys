"""Synthetic numerical and interface qualification before registration."""
from copy import deepcopy
from unittest.mock import patch
import numpy as np
import torch
from . import study as s,policy

def preflight():
    s.initialize()
    checks=[]
    def forbidden(*args,**kwargs):raise AssertionError('No oracle reads in preflight')
    tasks=(s.TASKS[0],s.TASKS[4],s.TASKS[6])
    with patch.object(s.matched.frozen.native,'read_observation',forbidden):
        for task in tasks:
            pool=s.matched.frozen.pool_for(task)
            ordered=sorted(range(len(pool.ids)),key=lambda j:pool.features[j])
            ix=[ordered[i] for i in np.linspace(0,len(ordered)-1,4,dtype=int)]
            history=[dict(candidate_id=pool.ids[j],y=y) for j,y in zip(ix,(.11,.29,.18,.37))]
            gp=s.matched.SeededGP(pool,s.BASE/'preflight_cache'/s.SHORT[task],tie_seed=0)
            context=gp.context(history,5)
            before={name:value.detach().clone() for name,value in context['stats']['model'].state_dict().items()}
            plain,shift,meta=policy.scores(gp,context,dict(mode='none'))
            np.testing.assert_array_equal(plain,context['base_logei'])
            unobserved=np.flatnonzero(context['available'])
            js=sorted(map(int,unobserved[:5]),key=lambda j:pool.features[j])
            plan=dict(mode='region',indices=js,confidence=.7)
            scores,shift,meta=policy.scores(gp,context,plan)
            assert np.isfinite(scores).all() and np.isfinite(shift).all()
            assert meta['native_replay_error']<2e-8
            from chem.benchmarks.pointwise_hte_engine import CorrectedNominalTilt
            tilt=CorrectedNominalTilt(gp.X[js],torch.full((len(js),),1/len(js),dtype=torch.double))
            model=context['stats']['model'];tilt.fit_lambda_by_delta(model,.75)
            xx=gp.X[unobserved[:4]].unsqueeze(1)
            uncached=tilt.posterior(model,xx)
            tilt.prepare_cache(model);cached=tilt.posterior(model,xx)
            mean_error=float((uncached.mean-cached.mean).abs().max())
            cov_error=float((uncached.mvn.covariance_matrix-cached.mvn.covariance_matrix).abs().max())
            assert mean_error<1e-7 and cov_error<1e-10,(task,mean_error,cov_error)
            neutral,neutral_shift,_=policy.scores(gp,context,dict(plan,confidence=.5))
            np.testing.assert_array_equal(neutral,context['base_logei'])
            np.testing.assert_array_equal(neutral_shift,np.zeros(len(pool.ids)))
            for name,value in model.state_dict().items():assert torch.equal(value,before[name]),name
            for seed in s.SEEDS:
                tied=s.matched.retie(context,pool,5,seed)
                np.testing.assert_array_equal(tied['base_logei'],context['base_logei'])
                j=policy.select(tied,tied['base_logei'])
                assert j==tied['bo_index']
            raw=dict(summary='synthetic interface check',mode='region',confidence=.7,conditions=None,
                     scope={pool.axes[0]:{'allowed':[pool.points[unobserved[0]][pool.axes[0]]]}} if pool.kinds[0]=='categorical' else
                           {pool.axes[0]:{'bounds':[min(p[pool.axes[0]] for p in pool.points),max(p[pool.axes[0]] for p in pool.points)]}},
                     rationale='Synthetic schema only; no scientific claim.')
            validated=policy.validate_lgbo(raw,pool,history,5,0);assert len(validated['indices'])>0
            bad=deepcopy(raw);bad['scope']={'nonexistent_axis':{'bounds':[0,1]}}
            try:policy.validate_lgbo(bad,pool,history,5,0)
            except ValueError:pass
            else:raise AssertionError('Invalid physical scope accepted')
            checks.append(dict(task=s.SHORT[task],native_replay_error=meta['native_replay_error'],
                official_cached_mean_error=mean_error,official_cached_covariance_error=cov_error))
            assert gp.counters['operator_compiles_in_loop']==0
            gp.release()
        ids=['alpha','beta','gamma','delta','epsilon']
        raw=dict(comment='synthetic schema',keywords='synthetic',hypotheses=[
            dict(strategy='one',rationale='synthetic',confidence='medium',parameter_sets=[{'candidate_id':'alpha'}]),
            dict(strategy='two',rationale='synthetic',confidence='high',parameter_sets=[{'candidate_id':'beta'}]),
            dict(strategy='three',rationale='synthetic',confidence='high',parameter_sets=[{'candidate_id':'gamma'}])])
        assert policy.validate_rbo(s.ROOT,raw,ids)['selected_id']=='beta'
    value=dict(status='PASS_GROUNDED_BASELINES_PREFLIGHT',checks=checks,
               official_stable_confidence_order=True,new_oracle_reads=0,new_LLM_calls=0,
               zero_tilt_exact_native_equivalence=True,base_GP_parameters_unchanged=True,
               physical_schema_validated=True,all_query_tie_streams_verified=True,
               official_extractor_compiled_before_rounds=True)
    s.write(s.BASE/'preflight.json',value,replace=True)
    return value
