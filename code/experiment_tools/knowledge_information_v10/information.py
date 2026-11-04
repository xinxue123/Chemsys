"""Read-only original-GP covariance columns in the native y coordinate system."""
import numpy as np
from knowledge_design_v10.decision_information import expected_best_value_gain


def evaluate(gp,context,indices):
    import torch
    indices=np.asarray(sorted(set(map(int,indices))),dtype=int)
    stats=context['stats'];model=stats['model'];n=len(gp.pool.ids)
    cov=np.empty((n,len(indices)));mean_error=variance_error=0.
    scale=float(stats['std']);offset=float(stats['mean']);X=gp.X
    with torch.no_grad():
        for start in range(0,n,128):
            end=min(start+128,n);z=torch.cat((X[start:end],X[indices]),dim=0)
            posterior=model.posterior(z);matrix=posterior.mvn.covariance_matrix.detach().cpu().numpy()*scale**2
            count=end-start;mu=posterior.mean.detach().cpu().numpy().reshape(-1)*scale+offset
            cov[start:end]=matrix[:count,count:]
            mean_error=max(mean_error,float(np.max(abs(mu[:count]-stats['mu_y'][start:end]))))
            variance_error=max(variance_error,float(np.max(abs(np.diag(matrix)[:count]-np.asarray(stats['sd_y'][start:end])**2))))
    # This check catches outcome-transform/standardization or latent/noisy mixing.
    np.testing.assert_allclose(mean_error,0.,rtol=0,atol=2e-7)
    np.testing.assert_allclose(variance_error,0.,rtol=0,atol=2e-7)
    observation_variance=np.asarray(stats['sd_predictive_y'])[indices]**2
    latent_variance=np.asarray(stats['sd_y'])[indices]**2
    if np.any(observation_variance<latent_variance-1e-9):raise ValueError('Observation noise variance cannot be negative')
    values={int(j):expected_best_value_gain(stats['mu_y'],cov[:,k],observation_variance[k]) for k,j in enumerate(indices)}
    return values,dict(query_indices=indices,cross_covariance=cov,observation_variance=observation_variance,
        mu_y=np.asarray(stats['mu_y']).copy(),max_mean_error=np.asarray(mean_error),max_variance_error=np.asarray(variance_error))
