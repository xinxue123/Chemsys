"""Secondary post-hoc ablation: zero lift with the recorded branch schedule.

The actual stored assistant replies are replayed as a fixed schedule, not
claimed to be new on-policy replies. Only the regional lambda is disabled.
PointBumpPrior is left untouched, including its equal-path-weight behavior.
"""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
from concurrent.futures import ProcessPoolExecutor,as_completed
from contextlib import redirect_stdout,redirect_stderr
from copy import deepcopy
from hashlib import sha256
import io
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from scripts.diagnose_suzuki_comparison import ROOT,OUT,CASES,SEEDS,read,write,digest,filehash,native_state,worker_native


def prepare():
    from chem.benchmarks.official_suzuki_adapter import modules
    modules('LGBO')
    from prior_monte_carlo import PointBumpPrior
    class Paths:
        def __call__(self,x):return torch.arange(7,dtype=torch.double).reshape(7,1,1,1).expand(7,1,x.shape[-2],1)
    examples=[]
    for point,sigma in [([.1,.2,.3,.4],.05),([.9,.8,.7,.6],2.)]:
        weights=PointBumpPrior(torch.tensor(point,dtype=torch.double),sigma=sigma).compute_norm_probs(Paths(),raw_samples=64)
        assert float(weights.max()-weights.min())==0.
        examples.append({'point':point,'sigma':sigma,'weights':weights.flatten().tolist(),
                         'between_path_spread':float(weights.max()-weights.min())})
    p={'study':'posthoc_lgbo_region_lift_ablation','parent_protocol_sha256':read(OUT/'protocol.json')['protocol_sha256'],
       'cases':list(CASES),'seeds':list(SEEDS),'runs':15,'queries_per_run':20,
       'additional_LLM_calls':0,'official_source_files_changed':False,'driver_sha256':filehash(__file__),
       'intervention':'Delegate to original fit_lambda_by_delta with delta=0.0, assert lambda0, leave every other source function unchanged.',
       'branch_schedule':'Original recorded active preferences at each iteration; exact original plan modes retained. Not new LLM responses on the changed history.',
       'interpretation':'Conditional effect of regional mean lift under the fixed observed branch/preference schedule; not an independent blind evaluation.',
       'point_prior_algebra':'position_score.sum().expand(P) is constant over posterior paths; normalization leaves equal weights. Kept unchanged.',
       'point_prior_uniform_weight_checks':examples}
    p['protocol_sha256']=digest(p)
    path=OUT/'regional_lift_protocol.json'
    if path.exists():assert read(path)==p
    else:write(path,p)
    return p


def one(job):
    from chem.benchmarks import official_suzuki_adapter as a
    from chem.benchmarks import official_suzuki_runner as r
    from chem.benchmarks.suzuki_domain import FrozenSuzukiOracle
    p=read(OUT/'regional_lift_protocol.json');case,seed=job
    folder=ROOT/'controls'/f'c{case}_s{seed}_LGBO_zero_regional_lift'
    if (folder/'result.json').exists():return read(folder/'result.json')
    source=native_state(case,seed,'LGBO');history=deepcopy(source['history'][:4])
    core,prompt=a.modules('LGBO');boo=__import__('boo')
    original=boo.LinearExponentialRegionalMeanTiltPlugAndPlay.fit_lambda_by_delta
    oracle=FrozenSuzukiOracle('data/runs/suzuki_baselines/oracles',case)
    calls=[];plans=[];log=io.StringIO();jit0=sum(r.JIT_CALLS.values())
    def zero(self,base_model,delta,observation_noise=False,**kwargs):
        result=original(self,base_model,0.0,observation_noise=observation_noise,**kwargs)
        assert self._lam==0.
        calls.append({'requested_delta':float(delta),'applied_delta':0.,'lambda':self._lam})
        return result
    with redirect_stdout(log),redirect_stderr(log),patch.object(boo.LinearExponentialRegionalMeanTiltPlugAndPlay,'fit_lambda_by_delta',zero):
        for t in range(1,17):
            reply=source['responses'][t-1]['upstream_reply']
            z,detail=a.execute_lgbo(history,reply,20261027+case*10000+seed*100+t)
            assert detail['plan']['mode']==source['decisions'][t-1]['plan']['mode']
            point=a.decode_z(z);value=float(oracle.predict([point])[0])
            history.append({'query':len(history)+1,'point':point,'z':z,'y':value,'initial':False});plans.append(detail)
    assert len(calls)==sum(x['plan']['mode'].startswith('region') for x in plans)
    curve=np.maximum.accumulate([h['y'] for h in history]).tolist()
    result={'recipe':'LGBO_zero_regional_lift','case':case,'seed':seed,'status':'COMPLETE','actual_queries':len(history),
        'initial_best':curve[3],'final_best':curve[-1],'search_auc':float(np.mean(curve[4:])),
        'best_curve':curve,'history':history,'plans':plans,'regional_fit_calls':calls,
        'diagnostic_protocol_sha256':p['protocol_sha256'],'LLM_calls':0,
        'operator_compiles_in_loop':sum(r.JIT_CALLS.values())-jit0}
    write(folder/'result.json',result)
    if log.getvalue():(folder/'native.log').write_text(log.getvalue(),encoding='utf8')
    return result


def main():
    prepare()
    with ProcessPoolExecutor(max_workers=4,initializer=worker_native) as pool:
        for f in as_completed([pool.submit(one,(c,s)) for c in CASES for s in SEEDS]):
            x=f.result();print(json.dumps({k:x[k] for k in ('case','seed','search_auc','status')}),flush=True)


if __name__=='__main__':main()
