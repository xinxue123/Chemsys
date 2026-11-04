"""Small synthetic resource probe; no chemistry outcome or optimization test."""
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/chemist_system'))
from chem.benchmarks.knowledge_birth_effect import BirthEffect


def main():
    rng=np.random.default_rng(20261101);count=5000;n=16
    coordinates=rng.normal(size=(count,3))
    mean=.1*coordinates[:,1];feature=np.tanh(coordinates[:,0])
    ids=tuple(f'synthetic_{i}' for i in range(count))
    z=np.arange(4,4+n);q=np.arange(4+n,count)
    def covariance(a,b):
        return .04*np.exp(-np.sum((a[:,None,:]-b[None,:,:])**2,axis=-1)/2)
    zz=covariance(coordinates[z],coordinates[z])
    xz=covariance(coordinates[q],coordinates[z])
    model=BirthEffect(ids,ids[:4],mean,np.full(count,.04),feature,
                      .15,.1**2,.03,'synthetic_resource_probe')
    observed=mean[z]+.3*feature[z]
    args=([ids[i] for i in z],observed,zz,[ids[i] for i in q],xz)
    start=perf_counter();first=model.predict(*args);cold=perf_counter()-start
    start=perf_counter();second=model.predict(*args);cached=perf_counter()-start
    np.testing.assert_array_equal(first['expert_mean'],second['expert_mean'])
    assert np.all(first['expert_latent_variance']>=0)
    cache=model.cache_info()
    assert cache['covariance_registry_bytes']==count*n*8
    assert cache['cholesky']==1 and cache['prediction_cache_hits']==1
    result=dict(status='SYNTHETIC_RESOURCE_PROBE_ONLY',seed=20261101,points=count,
        queried_since_birth=n,query_points=len(q),covariance_shapes=[list(zz.shape),list(xz.shape)],
        cold_seconds=cold,cached_seconds=cached,cache=cache,
        diagnostics=first['diagnostics'],new_optimization_runs=0,GP_fits=0,LLM_calls=0,
        limitation='One synthetic scalar-effect model; not chemistry performance or an end-to-end multi-worker timing.')
    out=ROOT/'experiments/reports/knowledge_loop_20261101/birth_component_resource_probe.json'
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
