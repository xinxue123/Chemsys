"""Complete response x initialization controls without rewriting four-arm results."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import hashlib
import json
import sys
import time

import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/analysis'))
import refit_v13_kernel_conditions as refits
origin=refits.origin;design=refits.design;OUT=design.OUT


def run(job):
    r=pd.Series(job['meta']);n=int(job['measured_total']);hist=job['history'];case=job['case_id']
    origin.STATES[r.path_id]=dict(history=hist)
    before=origin.frame(r,n);after=origin.frame(r,n+1)
    assert refits.digest(before['parameters'])==job['old_parameters_sha256']
    oldnorm=origin.normalization(r,n-1);newnorm=origin.normalization(r,n)
    oldmu,_=origin.physical_stats(before,oldnorm);pool=origin.POOLS[r.task]
    j=pool.ix[hist[-1]['candidate_id']];expected=[dict(h) for h in hist]
    expected[-1]=dict(expected[-1],y=float(oldmu[j]),raw_value=float(oldmu[j]*pool.scale))
    seed=torch.load(origin.checkpoint(r,after),map_location='cpu',weights_only=False)['fit_seed']
    variant='EW_GP_expected_warm_start'
    model,meta,hit,path=refits.cached_fit(case,variant,r.task,expected,int(seed),before['parameters'])
    ev,_,_,_=refits.evaluate(model,meta,r,after,newnorm)
    result=dict(case_id=case,path_id=r.path_id,task=r.task,regime=r.regime,method=r.method,seed=int(r.seed),
        measured_total=n,variant=variant,status='PASS',cache_hit=hit,fit_seconds=meta['fit_seconds'],
        fit_seed=int(seed),MLL=meta['marginal_loglikelihood_per_observation'],
        data_loglikelihood=meta['data_loglikelihood'],checkpoint=path.relative_to(ROOT).as_posix(),**ev)
    design.write(refits.RUNS/case/'expected_warm_diagnostics.json',result)
    return result


def main():
    jobs=json.loads((OUT/'controlled_jobs.json').read_text(encoding='utf-8'))
    sources=[Path(__file__),Path(refits.__file__),Path(design.__file__),OUT/'controlled_jobs.json']
    hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    design.write(refits.RUNS/'factorial_extension_protocol.json',dict(cases=len(jobs),source_hashes=hashes,
        variant='expected feedback, preceding parameters as initialization',
        purpose='check response information under both initialization choices; no new trajectories'))
    rows=[];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=12,initializer=refits.worker_initialize) as executor:
        futures=[executor.submit(run,j) for j in jobs]
        for f in as_completed(futures):rows.append(f.result())
    output=[{k:json.dumps(design.plain(v),allow_nan=False) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows]
    design.save('expected_warm_refits.csv',output)
    for path,h in hashes.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h,path
    design.write(OUT/'factorial_audit.json',dict(status='PASS_TWO_BY_TWO_RESPONSE_INITIALIZATION_DIAGNOSTIC',
        cases=len(rows),new_GP_fits=sum(not r['cache_hit'] for r in rows),elapsed_seconds=time.perf_counter()-start,
        new_experiments=0,new_LLM_calls=0,source_hashes=hashes))
    print(json.dumps(dict(cases=len(rows),new_GP_fits=sum(not r['cache_hit'] for r in rows),elapsed_seconds=time.perf_counter()-start),indent=2))


if __name__=='__main__':main()
