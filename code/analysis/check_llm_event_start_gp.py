"""Recalculate native qLogEI from saved fitted models without new fitting."""
from pathlib import Path
import json
import sys
import numpy as np
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from llm_event_start import study
from fast_hit_bo.model import SharedGP


def main():
    study.check(True)
    pool=study.knowledge_gp.load_task(ROOT,'Reizman_Suzuki_Case_1',seed=0)
    gp=SharedGP(pool,study.BASE/'cache')
    rows=[]
    for job,q in (('bo_s0',2),('bo_s0',5),('bo_s2',8),('event_s1',7),
                  ('event_s3',4),('event_s8',8),('llm_start_bo_s1',7)):
        s=study.read(study.BASE/'runs'/job/'state.json')
        context=gp.context(s['history'][:q-1],q)
        array=study.BASE/'runs'/job/'rounds'/f'{q:02}_GP.npz'
        with np.load(array,allow_pickle=False) as z:
            expected=z['logei'].copy()
            mean=z['mu'].copy()
            variance=z['variance'].copy()
        native=qLogExpectedImprovement(context['model'],best_f=context['incumbent'],
            sampler=SobolQMCNormalSampler(torch.Size([512]),seed=38112000+q))
        with torch.no_grad():
            actual=torch.cat([native(x.unsqueeze(1)) for x in gp.X.split(256)]).numpy()
            posterior=context['model'].posterior(gp.X)
        error=float(np.max(np.abs(actual-expected)))
        assert error<1e-8
        np.testing.assert_allclose(mean,posterior.mean.reshape(-1).numpy(),rtol=0,atol=1e-10)
        np.testing.assert_allclose(variance,posterior.variance.reshape(-1).numpy(),rtol=0,atol=1e-10)
        rows.append(dict(job=job,query=q,full_pool_candidates=len(pool.ids),max_abs_logei_error=error))
    assert gp.counters['fits']==0
    result=dict(status='PASS',checks=rows,new_fits=0,new_experiments=0)
    study.write(study.REPORT/'native_GP_check.json',result,replace=True)
    print(json.dumps(result))


if __name__=='__main__':
    main()
