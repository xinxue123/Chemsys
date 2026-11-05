"""One-step parameter restoration in frozen GPs; no parameter optimization."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import io
import json
import sys

import numpy as np
import pandas as pd
import torch
from botorch.models.kernels.categorical import CategoricalKernel

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/analysis'))
import trace_v13_highpoint_origin as origin
OUT=origin.OUT


def group_parameters(model):
    result={}
    for name,param in model.named_parameters():
        if name.startswith('likelihood.'):group='observation_noise'
        elif name.startswith('mean_module.'):group='background_mean'
        else:
            module=model.get_submodule(name.rsplit('.',1)[0])
            if isinstance(module,CategoricalKernel) or isinstance(getattr(module,'base_kernel',None),CategoricalKernel):
                group='categorical_kernel'
            elif name.endswith('raw_lengthscale'):group='continuous_lengthscales'
            else:group='kernel_amplitudes'
        result.setdefault(group,[]).append(name)
    return result


def posterior_scores(model,r,norm):
    mu=[];sd=[]
    with torch.no_grad():
        for xx in origin.X[r.task].split(256):
            p=model.posterior(xx.unsqueeze(1));mu.append(p.mean.reshape(-1));sd.append(p.variance.reshape(-1).clamp_min(1e-18).sqrt())
    return origin.logei(torch.cat(mu).numpy(),torch.cat(sd).numpy(),norm['best'],norm['seed'])


def evaluate(spec,d):
    path_id,total,samples=spec;r=d.loc[path_id];after=origin.frame(r,total);before=origin.frame(r,total-1)
    result=[]
    try:
        model,norm,_,_,_=origin.reconstruct(r,after)
        actual=posterior_scores(model,r,norm)
        np.testing.assert_allclose(actual,after['base_logei'],rtol=1e-9,atol=2e-8)
        params=dict(model.named_parameters());original={key:value.detach().clone() for key,value in params.items()}
        groups=group_parameters(model);variants={'actual_refitted_parameters':[],'all_preceding_parameters':list(params),
                                                **{'restore_'+key:values for key,values in groups.items()}}
        for variant,replace in variants.items():
            with torch.no_grad():
                for name,param in params.items():param.copy_(original[name])
                for name in replace:param=params[name];param.copy_(torch.as_tensor(before['parameters'][name],dtype=param.dtype))
            model.prediction_strategy=None
            values=actual if not replace else posterior_scores(model,r,norm)
            for _,sample in samples.iterrows():
                ix=int(sample.target_index);rank=origin.rank(values,after['available'],after['tie_priority'],ix)
                result.append(dict(path_id=path_id,total=total,task=r.task,regime=r.regime,method=r.method,
                    seed=r.seed,target_kind=sample.target_kind,target_id=origin.POOLS[r.task].ids[ix],
                    variant=variant,rank=rank['rank'],strict_rank=rank['strict_rank'],ties=rank['ties'],
                    remains_in_top5=rank['rank']<=5,status='PASS',
                    interpretation='actual observed dataset and after-step output normalization; only named parameter group restored'))
    except (ValueError,AssertionError,RuntimeError,FileNotFoundError,KeyError) as e:
        for _,sample in samples.iterrows():
            result.append(dict(path_id=path_id,total=total,task=r.task,method=r.method,target_kind=sample.target_kind,
                               status='UNRESOLVED',error=str(e)[:1200]))
    return result


def main():
    origin.tracked(__file__);d=origin.initialize().set_index('path_id',drop=False)
    data=pd.read_csv(io.BytesIO(origin.tracked(OUT/'one_step_numerical_mechanisms.csv')),encoding='utf-8-sig')
    need=data[(data.status=='PASS')&(~data.initial)&(data.fixed_kernel_observation_update_rank>5)]
    jobs=[(path_id,int(total),b) for (path_id,total),b in need.groupby(['path_id','total'])]
    if '--check' in sys.argv:jobs=jobs[:1]
    rows=[]
    with ThreadPoolExecutor(max_workers=8) as executor:
        for value in executor.map(lambda spec:evaluate(spec,d),jobs):rows.extend(value)
    frame=pd.DataFrame(rows);origin.save('refit_parameter_channel_restorations.csv',frame)
    for path,h in origin.READS.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h,path
    receipt=dict(status='PASS_PARAMETER_RESTORATION' if (frame.status=='PASS').all() else 'PARTIAL_PARAMETER_RESTORATION',
        distinct_transitions=len(jobs),target_admission_events=len(need),diagnostic_rows=len(rows),
        unresolved_rows=int((frame.status!='PASS').sum()),new_GP_fits=0,new_LLM_calls=0,new_experiments=0,
        interpretation='conditional computational necessity; overlapping parameter effects, not additive causal shares')
    (OUT/'parameter_audit.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    (OUT/'parameter_source_hashes.json').write_text(json.dumps(origin.READS,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2))
    if 'rank' in frame:
        print(frame.groupby(['target_kind','variant']).remains_in_top5.agg(['count','sum']).to_string())


if __name__=='__main__':main()
