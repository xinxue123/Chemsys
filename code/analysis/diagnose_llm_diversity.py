"""Profile archived designs and fitted kernels; no new scientific queries."""
from collections import Counter
from pathlib import Path
import csv
import hashlib
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'experiments/reports/llm_start_crossstudy_diagnosis_20261104'
SOURCES = {}


def tracked(path):
    path=Path(path)
    SOURCES[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read(path):
    return json.loads(tracked(path).read_text(encoding='utf-8'))


def main():
    torch.set_num_threads(1)
    sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
    from fast_hit_bo.model import make_model
    tracked(ROOT/'code/experiment_tools/fast_hit_bo/model.py')
    tracked(ROOT/'code/experiment_tools/fast_hit_bo/__init__.py')
    tracked(Path(__file__))
    pool=read(ROOT/'experiments/runs/suzuki_guarded/inputs/case1/pool.json')
    ix={cid:i for i,cid in enumerate(pool['ids'])}
    normalized=np.asarray(pool['z'])[:,:3]
    with np.load(tracked(ROOT/'experiments/runs/suzuki_guarded/inputs/case1/oracle.npz')) as z:
        truth=z['values'].copy()*100
    optimum=int(truth.argmax())
    profiles=[]
    for method,variant in [('bo','llm_event_start_qlogei_20261104'),
            ('llm_start_bo','llm_event_start_qlogei_20261104'),('handoff','llm_bo_handoff_20261104')]:
        for seed in range(10):
            folder=ROOT/'experiments/runs'/variant/'runs'/f'{method}_s{seed}'
            state=read(folder/'state.json')
            for n in (4,7):
                js=[ix[h['candidate_id']] for h in state['history'][:n]]
                for h,j in zip(state['history'][:n],js):
                    assert abs(h['raw_value']-truth[j])<1e-8
                x=normalized[js]
                categories=[pool['points'][j]['catalyst'] for j in js]
                counts=Counter(categories)
                pairs=[(a,b) for a in range(n) for b in range(a+1,n)]
                distance=np.array([np.linalg.norm(x[a]-x[b]) for a,b in pairs])
                matched=int(sum(categories[a]!=categories[b] and d<1e-8
                                for (a,b),d in zip(pairs,distance)))
                commit=read(folder/'rounds'/f'{n+1:02}_commit.json')
                checkpoint=torch.load(tracked(ROOT/commit['decision']['checkpoint']),
                                      map_location='cpu',weights_only=False)
                model=make_model(torch.tensor(checkpoint['train_X'],dtype=torch.double),
                                 torch.tensor(checkpoint['train_Y'],dtype=torch.double))
                model.load_state_dict(checkpoint['state'])
                additive,interaction=model.covar_module.kernels
                v1,v2=float(additive.outputscale),float(interaction.outputscale)
                category=additive.base_kernel.kernels[1]
                vc=float(category.outputscale)
                lc1=float(category.base_kernel.lengthscale.item())
                lc2=float(interaction.base_kernel.kernels[1].lengthscale.item())
                # Kernel correlation for identical continuous inputs and two
                # distinct nominal labels, including BOTH covariance components.
                correlation=(v1*(1+vc*np.exp(-1/lc1))+v2*np.exp(-1/lc2))/(v1*(1+vc)+v2)
                with np.load(tracked(folder/'rounds'/f'{n+1:02}_GP.npz')) as z:
                    mu,sd=z['mu']*100,np.sqrt(z['variance'])*100
                    available=z['available']
                    rmse=float(np.sqrt(np.mean((mu[available]-truth[available])**2)))
                    coverage=float(np.mean(np.abs(mu[available]-truth[available])<=1.96*sd[available]))
                profiles.append(dict(method=method,seed=seed,n=n,categories=len(counts),
                    continuous_distance=float(distance.mean()),matched_cat_pairs=matched,
                    catalyst_covariance_correlation=float(correlation),pool_RMSE_pp=rmse,
                    latent_interval95_coverage=coverage,optimum_mu=float(mu[optimum]),
                    optimum_sd=float(sd[optimum]),
                    optimum_outside95=bool(abs(mu[optimum]-truth[optimum])>1.96*sd[optimum]),
                    cat_counts=dict(counts),conditions=[pool['points'][j] for j in js],
                    yields=[h['raw_value'] for h in state['history'][:n]]))
    summary=[]
    for n in (4,7):
        for method in ('bo','llm_start_bo','handoff'):
            rs=[r for r in profiles if r['n']==n and r['method']==method]
            summary.append(dict(method=method,observations=n,runs=len(rs),
                **{key:float(np.mean([r[key] for r in rs])) for key in (
                    'categories','continuous_distance','matched_cat_pairs',
                    'pool_RMSE_pp','latent_interval95_coverage')},
                category_correlation_gt99=sum(r['catalyst_covariance_correlation']>.99 for r in rs),
                optimum_outside95=sum(r['optimum_outside95'] for r in rs)))
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'diversity_and_fit_profiles.json').write_text(json.dumps(profiles,indent=2)+'\n',encoding='utf-8')
    with (OUT/'diversity_and_fit_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]),lineterminator='\n')
        w.writeheader();w.writerows(summary)
    for path,digest in SOURCES.items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    (OUT/'diversity_source_hashes.json').write_text(json.dumps(SOURCES,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='PASS',new_GP_fits=0,new_acquisition_calculations=0,
        new_LLM_calls=0,new_sequential_experiments=0,archived_models_loaded=len(profiles),
        unchanged_inputs=len(SOURCES),summaries=summary),indent=2))


if __name__=='__main__':
    main()
