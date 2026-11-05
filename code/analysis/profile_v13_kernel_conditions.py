"""All low-start GP transitions, exact native parameter transforms, and refit cases."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import io
import json
import math
import sys

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/analysis'))
import trace_v13_highpoint_origin as origin

OUT=ROOT/'experiments/reports/v13_kernel_conditions_20261104'
CONSTRAINTS={}


def plain(value):
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    if isinstance(value,np.ndarray):return plain(value.tolist())
    if isinstance(value,np.generic):return plain(value.item())
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def write(path,value):path.write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n',encoding='utf-8')
def save(name,data):
    (data if isinstance(data,pd.DataFrame) else pd.DataFrame(data)).to_csv(
        OUT/name,index=False,encoding='utf-8-sig',lineterminator='\n')


def decode(task,params):
    out={}
    for name,value in params.items():
        t=torch.as_tensor(value,dtype=torch.double);constraint=CONSTRAINTS[task].get(name)
        out[name]=t.numpy() if constraint is None else constraint.transform(t).detach().numpy()
    return out


def kernel_shape(task,params):
    pool=origin.POOLS[task];v=decode(task,params)
    def scalar(name):return float(np.asarray(v[name]).reshape(()))
    def vec(name):return np.asarray(v[name]).reshape(-1)
    if pool.group=='P0':
        root='covar_module.kernels.'
        a=scalar(root+'0.raw_outputscale');b=scalar(root+'1.raw_outputscale')
        c=scalar(root+'0.base_kernel.kernels.1.raw_outputscale')
        l1=vec(root+'0.base_kernel.kernels.0.raw_lengthscale');l2=vec(root+'1.base_kernel.kernels.0.raw_lengthscale')
        lc1=scalar(root+'0.base_kernel.kernels.1.base_kernel.raw_lengthscale')
        lc2=scalar(root+'1.base_kernel.kernels.1.raw_lengthscale')
        diag=a*(1+c)+b
        rho=[(a*(math.exp(-.5*(.2/l1[i])**2)+c)+b*math.exp(-.5*(.2/l2[i])**2))/diag for i in range(3)]
        rho.append((a*(1+c*math.exp(-1/lc1))+b*math.exp(-1/lc2))/diag)
        ell1=list(l1)+[lc1];ell2=list(l2)+[lc2]
    elif pool.group=='P1':
        ell1=vec('covar_module.base_kernel.raw_lengthscale').tolist();ell2=[None]*len(ell1)
        rho=np.exp(-1/(len(ell1)*np.array(ell1))).tolist();diag=scalar('covar_module.raw_outputscale')
    else:
        ell1=vec('covar_module.raw_lengthscale').tolist();ell2=[None]*len(ell1)
        rho=np.exp(-.5*(.2/np.array(ell1))**2).tolist();diag=1.
    return dict(rho=np.array(rho),ell1=ell1,ell2=ell2,kernel_diag=diag,
        noise=scalar('likelihood.noise_covar.raw_noise'),background=scalar('mean_module.raw_constant'))


def verify_decoders(data):
    checks=[];specs=[]
    for task,pool in origin.POOLS.items():
        candidates=data[(data.primary)&(data.task==task)&(data.start_fraction_max<.8)]
        if candidates.empty:continue
        r=candidates.iloc[0];f=origin.frame(r,5);m,norm,_,_,_=origin.reconstruct(r,f)
        CONSTRAINTS[task]={name:m.constraint_for_parameter_name(name) for name,_ in m.named_parameters()}
        for name,constraint in CONSTRAINTS[task].items():
            specs.append(dict(task=task,parameter=name,type=type(constraint).__name__ if constraint else 'identity',
                enforced=bool(constraint.enforced) if constraint else False,
                lower=constraint.lower_bound.tolist() if constraint else None))
        anchor=np.array([0. if kind=='categorical' else .5 for kind in pool.kinds])
        probes=[anchor.copy()]
        for i,kind in enumerate(pool.kinds):
            p=anchor.copy();p[i]+=1 if kind=='categorical' else .2;probes.append(p)
        with torch.no_grad():K=m.covar_module(torch.tensor(np.array(probes),dtype=torch.double)).to_dense().numpy()
        native=K[0,1:]/np.sqrt(K[0,0]*np.diag(K)[1:]);decoded=kernel_shape(task,f['parameters'])
        np.testing.assert_allclose(native,decoded['rho'],rtol=1e-10,atol=1e-10)
        checks.append(dict(task=task,native=native.tolist(),decoded=decoded['rho'].tolist(),status='PASS'))
    write(OUT/'native_decoder_checks.json',checks);write(OUT/'parameter_constraints.json',specs)


def novelty_features(pool,history,new):
    j=pool.ix[new['candidate_id']];x=origin.X[pool.task if pool.task in origin.X else next(k for k,v in origin.POOLS.items() if v.task==pool.task)][j].numpy()
    key=next(k for k,v in origin.POOLS.items() if v.task==pool.task)
    prior=np.array([origin.X[key][pool.ix[h['candidate_id']]].numpy() for h in history])
    changed=np.column_stack([np.abs(prior[:,i]-x[i])>1e-8 if kind=='continuous' else prior[:,i]!=x[i]
                              for i,kind in enumerate(pool.kinds)])
    incumbent=int(np.argmax([h['raw_value'] for h in history]));axis=[]
    for i,(name,kind) in enumerate(zip(pool.axes,pool.kinds)):
        matches=np.flatnonzero(changed[:,i]&(changed.sum(axis=1)==1));anchor=None
        if len(matches):
            anchor=int(min(matches,key=lambda z:abs(prior[z,i]-x[i]))) if kind=='continuous' else int(matches[-1])
        axis.append(dict(axis=name,kind=kind,changed_from_incumbent=bool(changed[incumbent,i]),
            novel_nominal_level=bool(kind=='categorical' and not np.any(prior[:,i]==x[i])),
            outside_observed_range=bool(kind=='continuous' and (x[i]<prior[:,i].min() or x[i]>prior[:,i].max())),
            boundary=bool(kind=='continuous' and (x[i]<1e-5 or x[i]>1-1e-5)),
            isolated_contrast=anchor is not None,
            contrast_anchor=history[anchor]['candidate_id'] if anchor is not None else None,
            contrast_response_abs=abs(new['raw_value']-history[anchor]['raw_value']) if anchor is not None else None,
            contrast_coordinate_distance=abs(x[i]-prior[anchor,i]) if anchor is not None and kind=='continuous' else None))
    return axis


def path_profile(r):
    pool=origin.POOLS[r.task];truth=origin.TRUTH[r.task];span=float(np.ptp(truth));history=origin.STATES[r.path_id]['history']
    frames={t:origin.frame(r,t) for t in range(5,21)}
    shapes={t:kernel_shape(r.task,f['parameters']) for t,f in frames.items()}
    first90=next((i for i,h in enumerate(history,1) if h['raw_value']>=.9*truth.max()),21)
    edges=[];axes=[]
    for t in range(5,20):
        before,after=frames[t],frames[t+1];a,b=shapes[t],shapes[t+1];new=history[t-1]
        delta=b['rho']-a['rho'];axis=novelty_features(pool,history[:t-1],new)
        common=after['available'];high=common&(truth>=.9*truth.max())
        mu0=before['mu_y']*pool.scale;mu1=after['mu_y']*pool.scale
        sd0=before['sd_y']*pool.scale;sd1=after['sd_y']*pool.scale
        j=pool.ix[new['candidate_id']];den=before['sd_predictive_y'][j]*pool.scale
        z=(new['raw_value']-mu0[j])/den if den>1e-12 else None
        shape_key=int(np.argmax(np.abs(delta)))
        prior_y=np.array([h['y'] for h in history[:t-1]]);new_y=np.array([h['y'] for h in history[:t]])
        row=dict(path_id=r.path_id,task=r.task,method=r.method,backend=r.backend,regime=r.regime,seed=int(r.seed),
            measured_total=t,candidate_id=new['candidate_id'],raw_value=new['raw_value'],
            conditions=json.dumps(pool.points[j]),pre_discovery=t<first90,first90_total=first90 if first90<=20 else None,
            innovation_z=z,predicted_before=float(mu0[j]),predicted_observation_SD=float(den),
            kernel_response=float(np.max(np.abs(delta))),largest_change_axis=pool.axes[shape_key],
            changed_axes_vs_incumbent=sum(v['changed_from_incumbent'] for v in axis),
            new_nominal_levels=sum(v['novel_nominal_level'] for v in axis),
            isolated_axes=sum(v['isolated_contrast'] for v in axis),
            nominal_isolated_axes=sum(v['isolated_contrast'] and v['kind']=='categorical' for v in axis),
            on_continuous_boundary=any(v['boundary'] for v in axis),
            old_empirical_SD=float(prior_y.std(ddof=0)*pool.scale),new_empirical_SD=float(new_y.std(ddof=0)*pool.scale),
            old_kernel_diag=a['kernel_diag'],new_kernel_diag=b['kernel_diag'],
            old_noise=a['noise'],new_noise=b['noise'],
            high_MAE_before=float(np.abs(mu0[high]-truth[high]).mean()/span) if high.any() else None,
            high_MAE_after=float(np.abs(mu1[high]-truth[high]).mean()/span) if high.any() else None,
            old_parameters_sha256=origin.profile.helpers.digest(before['parameters']),
            new_parameters_sha256=origin.profile.helpers.digest(after['parameters']))
        row['high_MAE_improvement']=row['high_MAE_before']-row['high_MAE_after'] if high.any() else None
        edges.append(row)
        for i,v in enumerate(axis):
            axes.append(dict(path_id=r.path_id,task=r.task,method=r.method,backend=r.backend,regime=r.regime,
                seed=int(r.seed),measured_total=t,candidate_id=new['candidate_id'],**v,
                rho_before=float(a['rho'][i]),rho_after=float(b['rho'][i]),delta_rho=float(delta[i]),
                ell1_before=a['ell1'][i],ell1_after=b['ell1'][i],ell2_before=a['ell2'][i],ell2_after=b['ell2'][i],
                contrast_response_fraction= v['contrast_response_abs']/span if v['contrast_response_abs'] is not None else None,
                innovation_z=z,pre_discovery=t<first90,high_MAE_improvement=row['high_MAE_improvement']))
    return edges,axes


def select_cases(data,edges):
    chosen=[]
    for path_id,b in edges[edges.pre_discovery].groupby('path_id'):
        row=b.sort_values(['kernel_response','measured_total'],ascending=[False,True]).iloc[0]
        chosen.append(dict(path_id=path_id,measured_total=int(row.measured_total),selection='largest_pre_discovery_kernel_response'))
    exposure=pd.read_csv(io.BytesIO(origin.tracked(origin.OUT/'all_first_exposure_events.csv')),encoding='utf-8-sig')
    for _,r in data[(data.task=='c2')&(data.method=='v13')&(data.regime=='warm')].iterrows():
        e=exposure[(exposure.path_id==r.path_id)&(exposure.target_kind=='pool_global_max')].iloc[0]
        t=int(e.first_exposure_total)-1
        if not any(c['path_id']==r.path_id and c['measured_total']==t for c in chosen):
            chosen.append(dict(path_id=r.path_id,measured_total=t,selection='fixed_Case2_admission_example'))
    registry=[];jobs={}
    metadata=['path_id','source_path','task','regime','initial','method','seed','backend','landmark','start_fraction_max','pool_max','pool_range','primary','family']
    indexed=data.set_index('path_id',drop=False)
    for case in chosen:
        r=indexed.loc[case['path_id']];t=case['measured_total'];hist=origin.STATES[r.path_id]['history'][:t]
        edge=edges[(edges.path_id==r.path_id)&(edges.measured_total==t)].iloc[0]
        key=origin.profile.helpers.digest(dict(task=r.task,history_before=sorted(hist[:-1],key=lambda h:h['candidate_id']),
             new=hist[-1],before_parameters=edge.old_parameters_sha256,after_parameters=edge.new_parameters_sha256))
        case_id='kernel_'+key[:16]
        registry.append(dict(**case,case_id=case_id,task=r.task,regime=r.regime,method=r.method,seed=int(r.seed),
            response=float(edge.kernel_response),largest_change_axis=edge.largest_change_axis,
            candidate_id=hist[-1]['candidate_id'],conditions=edge.conditions))
        if case_id not in jobs:
            jobs[case_id]=dict(case_id=case_id,meta=plain({k:r[k] for k in metadata}),measured_total=t,history=hist,
                old_parameters_sha256=edge.old_parameters_sha256,new_parameters_sha256=edge.new_parameters_sha256)
    return registry,list(jobs.values())


def main():
    OUT.mkdir(parents=True,exist_ok=True);origin.tracked(__file__)
    origin.tracked(ROOT/'docs/v13_kernel_conditions_20261104.txt')
    all_data=origin.initialize();data=all_data[all_data.primary&(all_data.start_fraction_max<.8)].copy()
    assert len(data)==165
    verify_decoders(data)
    edges=[];axes=[]
    with ThreadPoolExecutor(max_workers=8) as executor:
        for ee,aa in executor.map(path_profile,[r for _,r in data.iterrows()]):edges.extend(ee);axes.extend(aa)
    edgeframe=pd.DataFrame(edges);assert len(edgeframe)==2475
    save('all2475_kernel_transitions.csv',edgeframe);save('all_axis_condition_responses.csv',axes)
    registry,jobs=select_cases(data,edgeframe);save('controlled_case_registry.csv',registry)
    write(OUT/'controlled_jobs.json',jobs)
    for path,h in origin.READS.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h,path
    write(OUT/'profile_source_hashes.json',origin.READS)
    write(OUT/'profile_audit.json',dict(status='PASS_KERNEL_CONDITION_PROFILES',paths=len(data),
        transitions=len(edges),axis_rows=len(axes),selected_path_observations=len(registry),unique_refit_cases=len(jobs),
        controlled_refits_planned=len(jobs)*4,new_GP_fits=0,new_experiments=0,new_LLM_calls=0))
    print(json.dumps(dict(paths=len(data),transitions=len(edges),axis_rows=len(axes),
        selected_path_observations=len(registry),unique_cases=len(jobs),refits_planned=len(jobs)*4),indent=2))


if __name__=='__main__':main()
