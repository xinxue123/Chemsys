"""Direct native BoTorch conditioning/MC acquisition check of saved decisions."""
from pathlib import Path
import json
import sys
import numpy as np
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from qlogei_rollout import study
from fast_hit_bo.model import SharedGP
from fast_hit_bo import PARAMETERS as P


def main():
    study.check()
    pool=study.knowledge_gp.load_task(ROOT,'Reizman_Suzuki_Case_1',seed=0)
    gp=SharedGP(pool,study.BASE/'cache')
    state=study.read(study.BASE/'runs/s4_middle_rollout_qlogei/state.json')
    checks=[]
    for q in (2,5):
        d=state['decisions'][q-1]['decision']
        c=gp.context(state['history'][:q-1],q)
        j=pool.ix[d['selected_id']]
        uniforms=torch.quasirandom.SobolEngine(1,scramble=True,seed=d['fantasy_seed']).draw(32).double().flatten().clamp(1e-12,1-1e-12)
        y=c['mu'][j]+np.sqrt(c['variance'][j])*torch.distributions.Normal(0.,1.).icdf(uniforms)
        x=gp.X[j].reshape(1,1,-1).expand(32,1,-1)
        with torch.no_grad():
            c['model'].posterior(gp.X[:1])
            native=c['model'].condition_on_observations(x,y.reshape(32,1,1),
                noise=torch.full((32,1,1),P['noise_variance'],dtype=torch.double))
            acq=qLogExpectedImprovement(native,best_f=y.clamp_min(c['incumbent']).float(),
                sampler=SobolQMCNormalSampler(torch.Size([512]),seed=d['future_seed']))
            values=torch.cat([acq(xs[:,None,None,:]).T for xs in gp.X.split(128)],dim=1)
            valid=c['available'].copy()
            valid[j]=False
            values[:,~valid]=-torch.inf
            best=values.max(dim=1).values
        outer_index=d['outer_ids'].index(d['selected_id'])
        recorded=np.asarray(d['future_logei'][outer_index])
        error=float(np.max(np.abs(best.numpy()-recorded)))
        assert error<1e-8,(q,error)
        ids=d['future_ids'][outer_index]
        actual=values[np.arange(32),[pool.ix[cid] for cid in ids]].numpy()
        assert np.max(np.abs(best.numpy()-actual))<1e-10
        utility=float(np.exp(c['logei'][j])+best.exp().mean())
        saved=d['outer_values'][outer_index]
        assert abs(utility-saved)<1e-10
        checks.append(dict(query=q,selected_id=d['selected_id'],fantasies=32,
            full_future_candidates=int(valid.sum()),native_MC_samples=512,
            max_future_logei_error=error,recorded_rollout_value=saved,
            direct_native_rollout_value=utility,absolute_utility_error=abs(utility-saved)))
        print(json.dumps(checks[-1]),flush=True)
    assert gp.counters['fits']==0
    study.check()
    result=dict(status='PASS',checks=checks,new_GP_fits=0,new_experiments=0,
        analytic_EI_used=False,all_fantasies_checked=True)
    study.REPORT.mkdir(parents=True,exist_ok=True)
    (study.REPORT/'rollout_value_check.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
