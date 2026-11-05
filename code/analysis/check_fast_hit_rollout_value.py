"""Check rollout utilities against direct batched BoTorch conditioning."""
from pathlib import Path
import json
import sys
import numpy as np
import torch
from scipy.special import ndtr

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from fast_hit_bo import study, PARAMETERS as P
from fast_hit_bo.model import SharedGP

def ei(mu,var,best):
    sd=np.sqrt(np.maximum(var,1e-18))
    z=(mu-best)/sd
    return np.maximum((mu-best)*ndtr(z)+sd*np.exp(-z*z/2)/np.sqrt(2*np.pi),0.)

def main():
    study.check()
    pool=study.knowledge_gp.load_task(ROOT,'Reizman_Suzuki_Case_1',seed=0)
    gp=SharedGP(pool,study.BASE/'cache')
    state=study.read(study.BASE/'runs/s4_middle_rollout/state.json')
    checks=[]
    for q in (2,5):
        c=state['decisions'][q-1]
        d=c['decision']
        context=gp.context(state['history'][:q-1],q)
        cid=d['selected_id']; j=pool.ix[cid]
        torch.manual_seed(0)
        uniforms=torch.quasirandom.SobolEngine(1,scramble=True,seed=d['fantasy_seed']).draw(P['fantasy_samples']).double().reshape(-1).clamp(1e-12,1-1e-12)
        normals=torch.distributions.Normal(0.,1.).icdf(uniforms)
        y=context['mu'][j]+np.sqrt(context['variance'][j])*normals
        x=gp.X[j].reshape(1,1,-1).expand(P['fantasy_samples'],1,-1)
        with torch.no_grad():
            context['model'].posterior(gp.X[:1])
            model=context['model'].condition_on_observations(
                x,y.reshape(-1,1,1),noise=torch.full((P['fantasy_samples'],1,1),P['noise_variance'],dtype=torch.double))
            posterior=model.posterior(gp.X.unsqueeze(0))
            mu=posterior.mean.squeeze(-1).numpy()
            var=posterior.variance.squeeze(-1).numpy()
        incumbent=max(o['y'] for o in state['history'][:q-1])
        next_best=np.maximum(y.numpy(),incumbent)
        values=ei(mu,var,next_best[:,None])
        valid=context['available'].copy();valid[j]=False
        values[:,~valid]=-np.inf
        independent=float(ei(np.array(context['mu'][j]),np.array(context['variance'][j]),incumbent)+values.max(1).mean())
        recorded=d['outer_values'][d['outer_ids'].index(cid)]
        assert np.isclose(independent,recorded,rtol=0,atol=1e-9),(q,independent,recorded)
        checks.append(dict(query=q,selected_id=cid,recorded=recorded,direct_conditioning=independent,
                           absolute_error=abs(independent-recorded),fantasies=P['fantasy_samples'],future_pool_size=int(valid.sum())))
    assert gp.counters['fits']==0
    result=dict(status='PASS',checks=checks,new_GP_fits=0,new_experiments=0)
    (study.REPORT/'rollout_value_check.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))

if __name__=='__main__':
    main()
