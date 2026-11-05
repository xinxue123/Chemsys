"""Actual catalyst contrast before/after a weakly identified kernel fit."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/analysis'))
import refit_v13_kernel_conditions as refits
origin=refits.origin;design=refits.design;OUT=design.OUT


def restored(path,task,history):
    raw=origin.tracked(ROOT/path);saved=torch.load(__import__('io').BytesIO(raw),map_location='cpu',weights_only=False)
    x,y,_,_,_=refits.ordered_data(task,history);m=refits.factory(task,x,y)
    m.load_state_dict(saved['state']);m.eval();m.likelihood.eval()
    return m,saved['metadata']


def main():
    origin.tracked(__file__);refits.worker_initialize()
    jobs=json.loads((OUT/'controlled_jobs.json').read_text(encoding='utf-8'))
    job=next(j for j in jobs if j['meta']['path_id']=='path0028' and j['measured_total']==14)
    r=pd.Series(job['meta']);pool=origin.POOLS[r.task]
    state=origin.read(ROOT/r.source_path);history=state['history'];origin.STATES[r.path_id]=dict(history=history)
    v=pd.read_csv(OUT/'controlled_refit_variants.csv',encoding='utf-8-sig')
    before=origin.frame(r,15);norm=origin.normalization(r,14)
    models={};oldparams={};records=[];pred={}
    for variant in ('A_actual_native','W_actual_warm_start'):
        row=v[(v.case_id==job['case_id'])&(v.variant==variant)].iloc[0]
        m,meta=restored(row.checkpoint,r.task,history[:14]);models[variant]=m
        oldparams[variant]={name:p.detach().numpy().tolist() for name,p in m.named_parameters()}
        shape=design.kernel_shape(r.task,oldparams[variant])
        records.append(dict(stage='before_category_control',variant=variant,observed=14,category_correlation=shape['rho'][-1],MLL=row.MLL))
        menu=before['commit']['display_ids'];xx=origin.X[r.task][[pool.ix[c] for c in menu]]
        with torch.no_grad():
            p=m.posterior(xx.unsqueeze(1));mu=p.mean.reshape(-1).numpy()*meta['std']+meta['mean']
            sd=p.variance.reshape(-1).sqrt().numpy()*meta['std']
        pred[variant]=(mu*pool.scale,sd*pool.scale)
    menu=before['commit']['display_ids'];inc=pool.points[pool.ix[history[13]['candidate_id']]];candidates=[]
    for i,c in enumerate(menu):
        point=pool.points[pool.ix[c]];changed=[a for a in pool.axes if point[a]!=inc[a]]
        a,sa=pred['A_actual_native'][0][i],pred['A_actual_native'][1][i]
        w,sw=pred['W_actual_warm_start'][0][i],pred['W_actual_warm_start'][1][i]
        candidates.append(dict(candidate_id=c,conditions=json.dumps(point),
            native_rank=origin.rank(before['base_logei'],before['available'],before['tie_priority'],pool.ix[c])['rank'],
            mean_A=a,SD_A=sa,mean_W=w,SD_W=sw,
            standardized_model_disagreement=abs(a-w)/np.sqrt(sa*sa+sw*sw),
            category_only_control=changed==['catalyst'],measured_next=c==history[14]['candidate_id'],
            true_yield_for_audit=origin.TRUTH[r.task][pool.ix[c]]))
    frame=pd.DataFrame(candidates)
    frame['disagreement_rank_average']=frame.standardized_model_disagreement.rank(ascending=False,method='average')
    design.save('case2_before_control_candidate_disagreement.csv',frame)
    after=origin.frame(r,16);norm_after=origin.normalization(r,15)
    native,native_norm,_,_,_=origin.reconstruct(r,after)
    shape=design.kernel_shape(r.task,after['parameters'])
    native.train()
    with torch.no_grad():
        mll=float(ExactMarginalLogLikelihood(native.likelihood,native)(native(native.train_inputs[0]),native.train_targets))
    native.eval()
    records.append(dict(stage='after_category_control',variant='native_archived',observed=15,
                        category_correlation=shape['rho'][-1],MLL=mll))
    seed=torch.load(origin.checkpoint(r,after),map_location='cpu',weights_only=False)['fit_seed'];newfits=0
    for variant,start in oldparams.items():
        model,meta,hit,path=refits.cached_fit('actual_category_control_case2_seed0',variant,r.task,history[:15],int(seed),start)
        newfits+=int(not hit);shape=design.kernel_shape(r.task,{name:p.detach().numpy().tolist() for name,p in model.named_parameters()})
        records.append(dict(stage='after_category_control',variant='start_from_'+variant,observed=15,
            category_correlation=shape['rho'][-1],MLL=meta['marginal_loglikelihood_per_observation'],
            checkpoint=path.relative_to(ROOT).as_posix()))
    design.save('case2_actual_category_control_kernel_fits.csv',records)
    for path,h in origin.READS.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h,path
    design.write(OUT/'contrast_source_hashes.json',origin.READS)
    design.write(OUT/'contrast_audit.json',dict(status='PASS_ACTUAL_CATEGORY_CONTRAST_DIAGNOSTIC',
        real_observation_ids=[history[13]['candidate_id'],history[14]['candidate_id']],
        observed_yields=[history[13]['raw_value'],history[14]['raw_value']],new_GP_fits=newfits,
        new_experiments=0,new_LLM_calls=0,interpretation='same two recorded models before and after one actual matched catalyst comparison'))
    print(frame.round(5).to_string(index=False));print(pd.DataFrame(records).to_string(index=False))


if __name__=='__main__':main()
