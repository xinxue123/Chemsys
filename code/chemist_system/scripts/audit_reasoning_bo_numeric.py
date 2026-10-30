"""Fixed-prefix independent GP algebra and categorical coding audit for M2."""
from pathlib import Path
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from botorch.models.kernels.categorical import CategoricalKernel
from chem.benchmarks import reasoning_bo_data as data
from chem.benchmarks import reasoning_bo_engine as engine
from chem.benchmarks import reasoning_bo_study as study


@torch.no_grad()
def check_prefix(pool,history,receipt,folder):
    cache=engine.BatchCache(pool,folder)
    fit,v=cache.model(history);model=v['model'];Xtr=model.train_inputs[0]
    assert isinstance(model.covar_module.base_kernel,CategoricalKernel)
    lookup={key:i for i,key in enumerate(pool['ids'])}
    indices=sorted(set(np.linspace(0,len(pool['ids'])-1,32,dtype=int).tolist()+
        receipt['decision']['shortlist']['indices']+[lookup[k] for k in receipt['selected_ids']]))
    xx=cache.X[indices];n=len(Xtr)
    K=model.covar_module(Xtr).to_dense()+torch.eye(n,dtype=torch.double)*model.likelihood.noise.reshape(-1)[0]
    cross=model.covar_module(xx,Xtr).to_dense()
    alpha=torch.linalg.solve(K,model.train_targets-model.mean_module(Xtr))
    internal_mean=model.mean_module(xx)+cross@alpha
    internal_var=model.covar_module(xx).to_dense().diagonal()-(cross*torch.linalg.solve(K,cross.T).T).sum(-1)
    transform=model.outcome_transform
    scale=float(transform.stdvs.reshape(-1)[0]);offset=float(transform.means.reshape(-1)[0])
    manual_mean=(internal_mean*scale+offset)*v['std']+v['mean']
    manual_var=internal_var*scale**2*v['std']**2
    posterior=model.posterior(xx)
    mean=posterior.mean.reshape(-1)*v['std']+v['mean'];var=posterior.variance.reshape(-1)*v['std']**2
    mean_err=float((manual_mean-mean).abs().max());var_err=float((manual_var-var).abs().max())
    assert mean_err<1e-8 and var_err<1e-8
    # Arbitrarily spaced category IDs, same fitted parameters, exact kernel.
    rng=np.random.default_rng(28102026)
    xt_new=Xtr.clone();xx_new=xx.clone()
    for j,axis in enumerate(pool['axes']):
        codes=100+31*rng.permutation(len(pool['levels'][axis]))
        for old,new in enumerate(codes):
            xt_new[Xtr[:,j]==old,j]=int(new);xx_new[xx[:,j]==old,j]=int(new)
    assert torch.equal(model.covar_module(Xtr).to_dense(),model.covar_module(xt_new).to_dense())
    assert torch.equal(cross,model.covar_module(xx_new,xt_new).to_dense())
    d=receipt['decision'];batches=d['batches']
    base_acquisition,_,meta=cache.acquisition(history)
    base=base_acquisition(cache.X[torch.tensor(batches,dtype=torch.long)]).reshape(-1).numpy()
    score_err=float(np.max(np.abs(base-np.array(d['base']))));assert score_err<1e-10
    field_err=0.;cov_err=0.;guided_err=0.
    if d['region']:
        tilt,_,_=cache.tilt(history,d['region'])
        grid=tilt.Xg;a=tilt._a
        ktg=model.covar_module(Xtr,grid).to_dense()
        manual_field=tilt._lam*(model.covar_module(xx,grid).to_dense()@a-cross@torch.linalg.solve(K,ktg@a))
        native=tilt.posterior(model,xx)
        field_err=float(((native.mean.reshape(-1)-posterior.mean.reshape(-1))-manual_field).abs().max())
        assert field_err<1e-9
        # Compare common-jitter base and deployed knowledge covariances.
        guided=engine.nominal.old.ScaledRegionModel(model,tilt,d['amplitude']).posterior(xx)
        basep=engine.nominal.JitterModel(model).posterior(xx)
        cov_err=float((guided.mvn.covariance_matrix-basep.mvn.covariance_matrix).abs().max())
        assert cov_err<1e-10
        m=receipt['prequery_measurement'];selected=[indices.index(lookup[k]) for k in receipt['selected_ids']]
        recorded=np.array(m['raw_region_shift_y']);actual=(manual_field[selected]*v['std']).numpy()
        assert np.max(np.abs(recorded-actual))<1e-9
        guided_acquisition,_,_=cache.acquisition(history,d['region'],d['amplitude'])
        scores=guided_acquisition(cache.X[torch.tensor(batches,dtype=torch.long)]).reshape(-1).numpy()
        guided_err=float(np.max(np.abs(scores-np.array(d['score']))));assert guided_err<1e-10
    return {'fit_key':fit,'observed':len(history),'test_points':len(indices),
        'GP_mean_max_error_yield_fraction':mean_err,'GP_variance_max_error':var_err,
        'base_qLogNEI_max_error':score_err,'native_cached_field_max_error_std':field_err,
        'guided_qLogNEI_max_error':guided_err,'knowledge_covariance_change_beyond_common_jitter':cov_err,
        'arbitrary_category_bijection_kernel_equal':True}


def main():
    torch.set_num_threads(1);p=study.check_protocol();rows=[];missing=[]
    methods=('ReasoningBO','BO_shared','Gate_only','New_R2','LGBO_region')
    for task in data.SPECS:
        pool=data.load_pool(task)
        for method in methods:
            run_id=f'{task}_s00_{method}';folder=study.ROOT/'runs'/run_id
            if not (folder/'state.json').exists():missing.extend((run_id,t) for t in (1,10,30));continue
            state=data.read(folder/'state.json')
            for iteration in (1,10,30):
                if len(state['decisions'])<iteration:missing.append((run_id,iteration));continue
                receipt=data.read(folder/'batches'/f'{iteration:02d}_committed.json')
                result=check_prefix(pool,state['history'][:3*iteration],receipt,
                    study.ROOT/'cache'/f'{task}_s00')
                rows.append({'run_id':run_id,'iteration':iteration,**result})
    result={'status':'PASS_COMPLETE_FIXED_PREFIXES' if not missing else 'PASS_AVAILABLE_FIXED_PREFIXES',
        'protocol_sha256':p['protocol_sha256'],'auditor_sha256':data.filehash(__file__),
        'selection':'Before/early/mid/late fixed rounds1,10,30; seed0; five main methods; both tasks. No outcome-based selection.',
        'expected_prefixes':30,'checked_prefixes':len(rows),'missing':missing,'rows':rows}
    data.write(study.OUT/'numeric_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
