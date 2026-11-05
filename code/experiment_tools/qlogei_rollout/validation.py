"""Independent full-pool native qLogEI checks, before observing new outcomes."""
import numpy as np
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler

from chem.benchmarks import knowledge_gp
from fast_hit_bo import PARAMETERS as P
from fast_hit_bo.policy import choose
from .acquisition import maximum_qlogei,base_samples,marginal_scores
from .policy import decision


def qualify(gp,history):
    checks=[]
    with torch.no_grad():
        for q in (2,5,7):
            c=gp.context(history[:q-1],q)
            mu=torch.from_numpy(c['mu'])
            var=torch.from_numpy(c['variance'])
            incumbent=torch.tensor([c['incumbent']],dtype=torch.float32).double()
            normals=base_samples(38112000+q)
            values=torch.cat([marginal_scores(mu[k:k+256],var[k:k+256].sqrt(),
                incumbent.expand(len(mu[k:k+256])),normals) for k in range(0,len(mu),256)])
            np.testing.assert_allclose(values.numpy(),c['logei'],rtol=0,atol=1e-8)
            best,winners,counts=maximum_qlogei(mu[None,:],var,incumbent,
                torch.from_numpy(c['available']),torch.from_numpy(c['priorities']),38112000+q)
            native_max=c['logei'][c['available']].max()
            assert abs(float(best[0])-native_max)<1e-8
            assert c['logei'][winners[0]]>=native_max-1e-10
            checks.append(dict(type='current_posterior',query=q,
                max_abs_logei_error=float(np.abs(values.numpy()-c['logei']).max()),
                maximum_error=abs(float(best[0])-native_max),exact_evaluations=counts))

        for q in (2,5):
            c=gp.context(history[:q-1],q)
            j=choose(c['logei'],c['available'],c['priorities'])
            cross=gp.conditional_cross_covariance(c,[j])[:,0]
            y=c['mu'][j]+np.sqrt(c['variance'][j])*torch.tensor([-1.3,.2,1.4],dtype=torch.double)
            denominator=c['variance'][j]+P['noise_variance']
            mu=torch.from_numpy(c['mu'])[None,:]+(y-c['mu'][j])[:,None]*cross[None,:]/denominator
            var=torch.from_numpy(c['variance'])-cross.square()/denominator
            incumbent=y.clamp_min(c['incumbent']).float().double()
            valid=torch.from_numpy(c['available'].copy())
            valid[j]=False
            priority=torch.from_numpy(knowledge_gp.semantic_priorities(gp.pool,q+1,0))
            best,winners,counts=maximum_qlogei(mu,var,incumbent,valid,priority,38112000+q+1)
            c['model'].posterior(gp.X[:1])
            errors=[]
            for row in range(len(y)):
                native=c['model'].condition_on_observations(gp.X[j:j+1],y[row].reshape(1,1),
                    noise=torch.full((1,1),P['noise_variance'],dtype=torch.double))
                acq=qLogExpectedImprovement(native,best_f=float(incumbent[row]),
                    sampler=SobolQMCNormalSampler(torch.Size([512]),seed=38112000+q+1))
                full=torch.cat([acq(x.unsqueeze(1)) for x in gp.X.split(128)])
                full[~valid]=-torch.inf
                value=float(full.max())
                error=abs(value-float(best[row]))
                assert error<1e-8,(q,row,error)
                assert float(full[winners[row]])>=value-1e-10
                errors.append(error)
            checks.append(dict(type='native_conditioned_GP_full_pool',query=q,
                fantasies=len(y),candidates=int(valid.sum()),maximum_errors=errors,
                exact_evaluations=counts))

        rng=torch.Generator().manual_seed(7192)
        mean=torch.randn(18,300,generator=rng,dtype=torch.double)*.5
        variance=10**(-10+10*torch.rand(18,300,generator=rng,dtype=torch.double))
        incumbent=torch.linspace(-.5,2.,18,dtype=torch.double)
        valid=torch.ones(300,dtype=torch.bool)
        priority=torch.randperm(300,generator=rng)
        best,winners,counts=maximum_qlogei(mean,variance,incumbent,valid,priority,5932)
        normals=base_samples(5932)
        errors=[]
        for row in range(len(mean)):
            full=marginal_scores(mean[row],variance[row].clamp_min(1e-10).sqrt(),
                incumbent[row].expand(300),normals)
            error=abs(float(full.max())-float(best[row]))
            assert error<1e-10
            errors.append(error)
        checks.append(dict(type='synthetic_mean_variance_incumbent_stress',
            maxima=18,candidates=300,max_abs_error=max(errors),exact_evaluations=counts))

        c=gp.context(history[:1],2)
        terminal=decision(gp,c,8)
        assert terminal['selected_id']==gp.pool.ids[choose(c['logei'],c['available'],c['priorities'])]
        trial=decision(gp,c,2)
        assert trial['selected_id'] in trial['outer_ids'] and len(trial['outer_ids'])<=48
        for current,future,value in zip(trial['first_logei'],trial['future_logei'],trial['outer_values']):
            assert abs(np.exp(current)+np.exp(future).mean()-value)<1e-12
    return dict(status='PASS',checks=checks,last_step_qLogEI_exact=True,
        native_qLogEI_used_at_both_stages=True,analytic_EI_used=False,
        log_values_exponentiated_before_addition=True,trial_selected_id=trial['selected_id'],
        future_tie_priorities_use_next_query=True,new_GP_fits=gp.counters['fits'])
