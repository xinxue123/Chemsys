"""Post-hoc gate/MC diagnostic: frozen own prefixes, no oracle or API calls.

Does not modify any published study, cache, parameter or optimization trajectory.
Run from chemist_system with python -m research.suzuki_kalman_gate_diagnostic.diagnose_case2.
"""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
from hashlib import sha256
from pathlib import Path
import json

import numpy as np
import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.distributions import MultivariateNormal
from scipy.special import ndtr

ROOT=Path('data/runs/suzuki_kalman_knowledge')
OUT=Path('research/suzuki_kalman_gate_diagnostic')


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def filehash(path):return sha256(Path(path).read_bytes()).hexdigest()


class FrozenMarginals(Model):
    num_outputs=1;batch_shape=torch.Size([])
    def __init__(self):super().__init__();self.mean=None;self.variance=None
    def posterior(self,X,**kwargs):
        assert X.shape[-2]==1 and X.shape[0]==len(self.mean)
        return GPyTorchPosterior(MultivariateNormal(self.mean[:,None],torch.diag_embed(self.variance[:,None])))


def evaluate(mu,variance,best,seed,samples=None):
    model=FrozenMarginals()
    sampler=SobolQMCNormalSampler(torch.Size([samples]),seed=seed) if samples else None
    acq=qLogExpectedImprovement(model,best,sampler=sampler) if samples else LogExpectedImprovement(model,best)
    output=[]
    with torch.no_grad():
        for start in range(0,len(mu),256):
            model.mean=torch.tensor(mu[start:start+256],dtype=torch.double)
            model.variance=torch.tensor(variance[start:start+256],dtype=torch.double)
            output.append(acq(torch.zeros((len(model.mean),1,4),dtype=torch.double)).reshape(-1))
    return torch.cat(output).numpy(),sampler


def ranked(scores,available,ids):return sorted(available,key=lambda j:(-float(scores[j]),ids[j]))


def main():
    torch.set_num_threads(1);torch.linalg.cholesky(torch.eye(4,dtype=torch.double))
    state_path=ROOT/'runs/c2_s0/state.json';pool_path=Path('data/runs/suzuki_guarded/inputs/case2/pool.json')
    state=read(state_path);ids=read(pool_path)['ids'];records=[];input_hashes={str(p):filehash(p) for p in (state_path,pool_path)}
    for t,event in enumerate(state['decisions'],1):
        score_path=ROOT/'cache/c2_s0'/(event['base_score_key']+'.npz')
        with np.load(score_path,allow_pickle=False) as data:base=data['scores'].copy();meta=json.loads(str(data['metadata']))
        prefix=state['history'][:t+3];seen={h['candidate_id'] for h in prefix};av=np.array([j for j,k in enumerate(ids) if k not in seen]);rank=ranked(base,av,ids)
        best=float(base[rank[0]]);gap2=float(best-base[rank[1]]);eps=float(event['budget_logei'])
        record={'iteration':t,'actual_epsilon':eps,'base_gap2':gap2,'base_gap3':float(best-base[rank[2]]),'actual_eligible':int(np.sum(best-base[av]<=eps)),
                'fixed_epsilon_log2_count':int(np.sum(best-base[av]<=np.log(2))),
                'fixed_epsilon_log4_count':int(np.sum(best-base[av]<=np.log(4))),
                'fixed_epsilon_log10_count':int(np.sum(best-base[av]<=np.log(10))),
                'base_top2':[{'id':ids[j],'score':float(base[j])} for j in rank[:2]]}
        if t>=10:
            moment_path=ROOT/'cache/c2_s0'/(event['posterior_key']+'.npz')
            with np.load(moment_path,allow_pickle=False) as data:moments={k:data[k].copy() for k in data.files}
            y=np.array([h['y'] for h in prefix]);ym=y.mean();ys=y.std(ddof=0);best_std=float((y.max()-ym)/ys)
            # The same explicit output standardization and1e-6 covariance jitter
            # as the published common base. No refit or uncertainty substitution.
            mu=moments['mu_std'];variance=moments['sd_std']**2+1e-6;sd=np.sqrt(variance)
            analytic,_=evaluate(mu,variance,best_std,meta['mc_seed']);arank=ranked(analytic,av,ids)
            record.update(analytic_logEI_gap2=float(analytic[arank[0]]-analytic[arank[1]]),
                          analytic_same_epsilon_count=int(np.sum(analytic[arank[0]]-analytic[av]<=eps)),
                          analytic_log4_count=int(np.sum(analytic[arank[0]]-analytic[av]<=np.log(4))))
            record['acquisition_comparison']={}
            for samples in (512,4096,16384):
                scores,sampler=evaluate(mu,variance,best_std,meta['mc_seed'],samples);srank=ranked(scores,av,ids)
                zmax=float(sampler.base_samples.max());chosen=srank[0];athreshold=float((best_std-mu[arank[0]])/sd[arank[0]])
                values={'top_id':ids[chosen],'gap2':float(scores[chosen]-scores[srank[1]]),
                        'same_epsilon_count':int(np.sum(scores[chosen]-scores[av]<=eps)),
                        'log4_count':int(np.sum(scores[chosen]-scores[av]<=np.log(4))),
                        'matches_analytic_argmax':bool(chosen==arank[0]),
                        'normal_sample_max':zmax,
                        'positive_samples_for_analytic_argmax':int((sampler.base_samples.numpy()>athreshold).sum()),
                        'analytic_probability_improvement':float(ndtr(-athreshold))}
                if samples==512:
                    values['maximum_difference_from_saved_score']=float(np.max(np.abs(scores-base)));assert values['maximum_difference_from_saved_score']<1e-8
                record['acquisition_comparison'][str(samples)]=values
            relevant=list(dict.fromkeys([*rank[:2],*arank[:2]]))
            record['candidate_comparison']=[{'id':ids[j],'mu_percent':float(mu[j]*ys*100+ym*100),'sd_with_common_jitter_pp':float(sd[j]*ys*100),
                  'standard_normal_improvement_threshold':float((best_std-mu[j])/sd[j]),
                  'saved_MC_logEI_std':float(base[j]),'analytic_logEI_std':float(analytic[j])} for j in relevant]
            input_hashes[str(moment_path)]=filehash(moment_path)
        input_hashes[str(score_path)]=filehash(score_path);records.append(record)
    result={'scope':'Case2 seed0 frozen actual prefixes; post-hoc acquisition diagnostics only, no trajectory reruns or new outcomes.',
            'diagnostic_sha256':filehash(Path(__file__)),'protected_release_inventory_sha256':filehash(Path('research/suzuki_kalman_knowledge/release_inventory.json')),
            'new_oracle_queries':0,'new_LLM_calls':0,'new_GP_fits':0,'same_latent_posterior_and_covariance_jitter':True,
            'threshold_count_analysis':'Reapply alternate fixed thresholds to each existing512-MC score vector; these are not independently run BO curves.',
            'analytic_analysis':'q=1 Gaussian analytic LogExpectedImprovement, identical moments,best_f,output scaling and1e-6 jitter. Compare against512,4096,16384MC paths with the original per-prefix seed. Not an algorithm change to the frozen experiment.',
            'inputs':input_hashes,'records':records,
            'limits':['One case and one realized prefix sequence. Larger candidate sets do not prove different choices or better yields.',
                      'Late expected improvements may be extremely small; log ratios can magnify tiny absolute values.',
                      'Analytic EI is a diagnostic forq=1 Gaussian, not a direct replacement for general batch or non-Gaussian acquisitions.']}
    for path,expected in input_hashes.items():assert filehash(path)==expected
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'case2_diagnostic.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps({'late_rounds':[{k:v for k,v in record.items() if k in ('iteration','actual_epsilon','base_gap2','actual_eligible','fixed_epsilon_log4_count','analytic_logEI_gap2','analytic_same_epsilon_count','acquisition_comparison','candidate_comparison')} for record in records if record['iteration']>=10]},ensure_ascii=False))


if __name__=='__main__':main()
