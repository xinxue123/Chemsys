"""Resolve large qLogEI/analytic differences without changing campaigns."""
import math
from pathlib import Path
import numpy as np
import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from . import perera_gp_tie_audit as audit


def main():
    audit.common.warm();results=[]
    for task in audit.data.SPECS:
        pool=audit.data.load_pool(task)
        cache=audit.engine.Cache(pool,audit.SCRATCH/'cache'/task)
        for row in audit.data.read(audit.OUT/(task+'_rounds.json')):
            if 'acquisition_audit' not in row:continue
            folder=audit.ROOT/row['variant']/'runs'/(task+'_s0_'+row['variant'])
            state=audit.data.read(folder/'state.json');hist=state['history'][:row['query']+3]
            _,v=cache.model(hist);st=cache.stats(hist);m=audit.engine.nominal.JitterModel(v['model'])
            bf=torch.tensor(v['best_std'],dtype=torch.double);mc=row['base_meta']['mc_seed']
            q=qLogExpectedImprovement(m,best_f=bf,sampler=SobolQMCNormalSampler(torch.Size([512]),seed=mc))
            scores=audit.tensor_scores(q,cache.X)
            logei=audit.tensor_scores(LogExpectedImprovement(m,best_f=bf),cache.X)
            with np.load(folder/'rounds'/f"{row['query']:02d}_candidates.npz") as a:mask=a['available'].copy()
            budget=math.log(4)*(16-row['query'])/15
            allowed_mc=mask&(scores[mask].max()-scores<=budget)
            allowed_exact=mask&(logei[mask].max()-logei<=budget)
            top_mc=mask&(scores==scores[mask].max());top_exact=mask&(logei==logei[mask].max())
            z=(v['best_std']-st['mu_std'])/np.sqrt(st['sd_std']**2+1e-6)
            errors=np.abs(scores-logei)
            reduced=qLogExpectedImprovement(m,best_f=bf,tau_relu=1e-9,
                sampler=SobolQMCNormalSampler(torch.Size([512]),seed=mc))
            finer=audit.tensor_scores(reduced,cache.X)
            result={'task':task,'variant':row['variant'],'query':row['query'],
                'same_top_set':bool(np.array_equal(top_mc,top_exact)),
                'eligible_MC512':int(allowed_mc.sum()),'eligible_analytic':int(allowed_exact.sum()),
                'eligible_disagreements':int(np.sum(allowed_mc!=allowed_exact)),
                'max_log_error_within_MC_gate':float(errors[allowed_mc].max()),
                'median_log_error_within_MC_gate':float(np.median(errors[allowed_mc])),
                'max_log_error_outside_MC_gate':float(errors[mask&~allowed_mc].max()) if (mask&~allowed_mc).any() else 0.,
                'max_z_observed_best_over_prediction':float(z[mask].max()),
                'median_z_observed_best_over_prediction':float(np.median(z[mask])),
                'median_analytic_logEI':float(np.median(logei[mask])),
                'median_qLogEI':float(np.median(scores[mask])),
                'tau1e_9_median_abs_error':float(np.median(np.abs(finer[mask]-logei[mask]))),
                'tau1e_6_median_abs_error':float(np.median(errors[mask]))}
            results.append(result);cache.release()
        assert cache.counters['fits']==0
    audit.data.write(audit.OUT/'acquisition_tail_checks.json',{
        'status':'PASS_COMPUTED','scope':'Numerical approximation sensitivity; no change to frozen512-MC acquisition or campaigns',
        'checks':results,'same_top_sets':sum(r['same_top_set'] for r in results),'total':len(results),
        'gate_disagreement_checks':sum(r['eligible_disagreements']>0 for r in results)})
    print(__import__('json').dumps(results,indent=2))


if __name__=='__main__':main()
