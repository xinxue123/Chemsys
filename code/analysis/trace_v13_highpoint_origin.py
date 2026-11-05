"""Trace candidate exposure and replay frozen one-observation GP updates."""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
import hashlib
import io
import json
import math
import sys

import numpy as np
import pandas as pd
import torch
from botorch.acquisition.logei import _log_improvement
from botorch.models import SingleTaskGP
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from botorch.utils.sampling import draw_sobol_normal_samples

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/analysis'),str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
import profile_v13_all_paths as profile
from knowledge_rank_v9 import engine

SOURCE=ROOT/'experiments/reports/v13_conditional_paths_20261104'
OUT=ROOT/'experiments/reports/v13_highpoint_origin_20261104'
READS={};POOLS={};TRUTH={};STATES={};X={};MOD={}


def tracked(path):
    p=Path(path).resolve();b=p.read_bytes();READS[p.relative_to(ROOT).as_posix()]=hashlib.sha256(b).hexdigest();return b
def read(path):return json.loads(tracked(path).decode('utf-8-sig'))
def save(name,data):
    (data if isinstance(data,pd.DataFrame) else pd.DataFrame(data)).to_csv(
        OUT/name,index=False,encoding='utf-8-sig',lineterminator='\n')


def frame(r,total):
    folder,q,reused=profile.asset_folder(r.to_dict(),total)
    cp=folder/'rounds'/f'{q:02}_commit.json';c=read(cp)
    with np.load(io.BytesIO(tracked(folder/'rounds'/f'{q:02}_GP.npz')),allow_pickle=False) as z:
        a={k:z[k].copy() for k in ('mu_y','sd_y','sd_predictive_y','base_logei','available','tie_priority')}
    context=dict(stats={k:a[k] for k in ('mu_y','sd_y','sd_predictive_y')},
                 **{k:a[k] for k in ('base_logei','available','tie_priority')})
    if 'GP_sha256' in c:assert engine.gp_hash(context)==c['GP_sha256']
    fit=folder/'rounds'/f'{q:02}_fit.json'
    params=read(fit) if fit.exists() else {};params=params.get('parameters',params)
    return dict(**a,commit=c,parameters=params,folder=folder,q=q,total=total,n=total-1)


def rank(scores,available,ties,index):
    if not available[index]:return dict(rank=None,strict_rank=None,ties=None,midrank=None)
    eq=available&(scores==scores[index]);above=int(np.count_nonzero(available&(scores>scores[index])))
    return dict(rank=above+int(np.count_nonzero(eq&(ties<ties[index])))+1,
        strict_rank=above+1,ties=int(eq.sum()),midrank=(above+(eq.sum()+1)/2)/int(available.sum()))


def target_row(r,f,kind,targets):
    pool=POOLS[r.task];available=f['available'];valid=[j for j in targets if available[j]]
    menu=f['commit'].get('display_ids',f['commit'].get('ordered_ids',f['commit'].get('menu_ids',[])))
    selected=f['commit']['selected_id'];t=f['total']
    base=dict(path_id=r.path_id,family=r.family,primary=bool(r.primary),regime=r.regime,task=r.task,
        seed=int(r.seed),method=r.method,backend=r.backend,start_tier=r.start_tier,target_kind=kind,
        total_observations=t,observed=t-1,qLogEI_defined=t>1,target_available=bool(valid))
    if not valid:return base
    scores=f['base_logei'];ties=f['tie_priority']
    j=min(valid,key=lambda ix:(-scores[ix],ties[ix]));rnk=rank(scores,available,ties,j)
    mu=f['mu_y'];sd=f['sd_y'];n=int(available.sum())
    mr=(np.count_nonzero(available&(mu>mu[j]))+(np.count_nonzero(available&(mu==mu[j]))+1)/2)/n if t>1 else None
    sr=(np.count_nonzero(available&(sd>sd[j]))+(np.count_nonzero(available&(sd==sd[j]))+1)/2)/n if t>1 else None
    if t==1:route='no_data_uniform_prior'
    elif rnk['strict_rank']==1 and rnk['ties']>5:route='acquisition_plateau'
    elif mr<=.1 and sr<=.1:route='high_mean_and_uncertainty'
    elif mr<=.1:route='high_predicted_mean'
    elif sr<=.1:route='high_uncertainty'
    else:route='joint_mean_uncertainty'
    cat=profile.helpers.category_parameters(f['parameters'],'perera' if pool.group=='P1' else r.task) if t>1 else {}
    base.update(target_id=pool.ids[j],native_rank=rnk['rank'],strict_rank=rnk['strict_rank'],
        acquisition_tie_size=rnk['ties'],mean_rank_fraction=mr,SD_rank_fraction=sr,
        predicted_mean=float(mu[j]*pool.scale) if t>1 else None,
        predicted_SD=float(sd[j]*pool.scale) if t>1 else None,logEI=float(scores[j]) if t>1 else None,
        actual_menu_entry=any(pool.ids[ix] in menu for ix in valid),
        target_selected=selected in {pool.ids[ix] for ix in targets},
        GP_reference=selected==f['commit'].get('BO_reference_id'),
        target_matches_GP_reference=f['commit'].get('BO_reference_id') in {pool.ids[ix] for ix in targets},
        entry_mode=route,category_correlation=cat.get('category_correlation'))
    return base


def trace_path(r):
    s=STATES[r.path_id];pool=POOLS[r.task];truth=TRUTH[r.task];k=int(r.landmark)
    globals_=np.flatnonzero(truth==truth.max()).tolist()
    hit=next((i for i,h in enumerate(s['history'],1) if i>k and h['raw_value']>=.9*float(truth.max())),None)
    targets=[('pool_global_max',globals_)]
    if hit:targets.append(('first90_hit',[pool.ix[s['history'][hit-1]['candidate_id']]]))
    first=1 if r.regime=='cold' else 5 if r.regime=='warm' else 11
    frames={};rows=[]
    for t in range(first,21):
        f=frame(r,t);frames[t]=f
        assert f['commit']['selected_id']==s['history'][t-1]['candidate_id']
        for kind,js in targets:rows.append(target_row(r,f,kind,js))
    events=[];specs=[]
    for kind,js in targets:
        own=[a for a in rows if a['target_kind']==kind];exposed=[a for a in own if a.get('actual_menu_entry')]
        first_entry=exposed[0] if exposed else None;chosen=next((a for a in own if a.get('target_selected')),None)
        event=dict(path_id=r.path_id,family=r.family,primary=bool(r.primary),task=r.task,seed=int(r.seed),
            method=r.method,backend=r.backend,regime=r.regime,start_tier=r.start_tier,target_kind=kind,
            first_exposure_total=first_entry['total_observations'] if first_entry else None,
            selection_total=chosen['total_observations'] if chosen else None,
            never_exposed=not bool(exposed),observed_in_prefix=any(h['candidate_id'] in {pool.ids[j] for j in js} for h in s['history'][:k]),
            entry_origin='never_exposed' if not exposed else 'no_data_uniform_prior' if first_entry['total_observations']==1 else
                'already_exposed_by_initial_model' if first_entry['total_observations']==first else 'new_exposure_after_feedback',
            exposed_before_selection=sum(a.get('actual_menu_entry',False) for a in own if chosen and a['total_observations']<chosen['total_observations']),
            entry_mode=first_entry.get('entry_mode') if first_entry else None)
        if first_entry:
            t=first_entry['total_observations'];j=pool.ix[first_entry['target_id']]
            event.update(target_id=first_entry['target_id'],entry_native_rank=first_entry['native_rank'],
                entry_strict_rank=first_entry['strict_rank'],entry_tie_size=first_entry['acquisition_tie_size'])
            if t>first:
                prior=frames[t-1];now=frames[t];h=s['history'][t-2]
                oldr=rank(prior['base_logei'],prior['available'],prior['tie_priority'],j)
                x=pool.ix[h['candidate_id']]
                event.update(preceding_observation_id=h['candidate_id'],preceding_observation_value=h['raw_value'],
                    preceding_value_fraction_max=h['raw_value']/float(truth.max()),
                    preceding_was_new_record=h['raw_value']>max(z['raw_value'] for z in s['history'][:t-2]) if t>2 else True,
                    prior_target_rank=oldr['rank'],prior_target_strict_rank=oldr['strict_rank'],prior_target_ties=oldr['ties'],
                    target_mean_change=float((now['mu_y'][j]-prior['mu_y'][j])*pool.scale) if t>2 else None,
                    target_SD_change=float((now['sd_y'][j]-prior['sd_y'][j])*pool.scale) if t>2 else None,
                    preceding_prediction_residual=float(h['raw_value']-prior['mu_y'][x]*pool.scale) if t>2 else None)
            # Only the primary lower-start population receives numerical one-step
            # reconstructions; high starts/extensions remain completely traced.
            if r.primary and r.start_fraction_max<.8 and t>1:
                specs.append(dict(path_id=r.path_id,target_kind=kind,total=t,target_index=j,initial=t==first))
        events.append(event)
    return rows,events,specs


@lru_cache(maxsize=256)
def normal_draw(seed):
    return draw_sobol_normal_samples(d=1,n=512,seed=int(seed),dtype=torch.double).reshape(512,1)


def logei(mu,sd,best,seed):
    mu=torch.as_tensor(mu,dtype=torch.double);sd=torch.as_tensor(sd,dtype=torch.double)
    z=normal_draw(int(seed));result=[]
    with torch.no_grad():
        for start in range(0,len(mu),256):
            Y=(mu[start:start+256]+torch.sqrt(sd[start:start+256].square()+1e-6)*z).unsqueeze(-1)
            # Native qLogExpectedImprovement registers Python best_f with
            # torch.as_tensor(best_f), before promoting it to posterior dtype.
            li=_log_improvement(Y,torch.as_tensor(best),tau=1e-6,fat=True).squeeze(-1)
            result.append((torch.logsumexp(li,dim=0)-math.log(512)).numpy())
    return np.concatenate(result)


def normalization(r,n):
    pool=POOLS[r.task];hist=STATES[r.path_id]['history'][:n]
    ordered=sorted(hist,key=lambda h:tuple(X[r.task][pool.ix[h['candidate_id']]].numpy())) if pool.group=='P2' else sorted(hist,key=lambda h:h['candidate_id'])
    y=torch.tensor([[h['y']] for h in ordered],dtype=torch.double)
    center=float(y.mean());std=float(y.std(unbiased=False).clamp_min(1e-12))
    report_std=1. if r.regime in ('cold','continuation10+10') and len({h['y'] for h in hist})==1 else std
    if pool.group=='P0':seed=38102026+int(r.task[-1])*10000+n-3
    elif pool.group=='P1':seed=38102026+(int(MOD['nominal'].digest({'task':pool.raw_pool['task_id']})[:7],16)%1000000)*100+n-3
    else:seed=38102026+(1 if r.task=='snar' else 2)*10000+n-3
    return dict(mean=center,std=std,report_std=report_std,best=float(((y-center)/std).max()),seed=seed,ordered=ordered,y=y)


def physical_stats(f,norm):
    m=(f['mu_y']-norm['mean'])/norm['report_std'];s=f['sd_y']/norm['report_std']
    return m*norm['std']+norm['mean'],s*norm['std']


def replay(r,f):
    norm=normalization(r,f['n']);m=(f['mu_y']-norm['mean'])/norm['report_std'];s=f['sd_y']/norm['report_std']
    values=logei(m,s,norm['best'],norm['seed'])
    err=float(np.max(np.abs(values-f['base_logei'])))
    if not np.allclose(values,f['base_logei'],rtol=1e-9,atol=2e-8):raise ValueError(('ACQ_REPLAY',r.path_id,f['total'],err))
    return norm,err


def checkpoint(r,f):
    key=f['commit']['GP_fit_key'];task=POOLS[r.task].task
    for base in f['folder'].parents:
        if base==ROOT.parent:break
        candidate=base/'cache'/task/(key+'.pt')
        if candidate.exists():return candidate
    raise FileNotFoundError((r.path_id,key))


def reconstruct(r,f):
    pool=POOLS[r.task];norm=normalization(r,f['n']);saved=torch.load(io.BytesIO(tracked(checkpoint(r,f))),map_location='cpu',weights_only=False)
    trainids=[h['candidate_id'] for h in norm['ordered']];trainX=X[r.task][[pool.ix[c] for c in trainids]]
    if 'train_candidate_ids' in saved:assert trainids==saved['train_candidate_ids']
    if 'train_X' in saved:np.testing.assert_array_equal(trainX.numpy(),np.asarray(saved['train_X']))
    trainY=(norm['y']-norm['mean'])/norm['std']
    m=SingleTaskGP(trainX,trainY) if pool.group=='P2' else MixedSingleTaskGP(trainX,trainY,cat_dims=[3] if pool.group=='P0' else list(range(trainX.shape[1])))
    m.load_state_dict(saved['state']);m.eval();m.likelihood.eval()
    # Compare selected public feature rows, without choosing them by true value.
    ix=np.unique(np.linspace(0,len(pool.ids)-1,min(32,len(pool.ids)),dtype=int))
    with torch.no_grad():
        pred=m.posterior(X[r.task][ix].unsqueeze(1))
        mu=pred.mean.reshape(-1).numpy()*norm['std']+norm['mean']
        sd=pred.variance.reshape(-1).clamp_min(1e-18).sqrt().numpy()*norm['std']
    oldmu,oldsd=physical_stats(f,norm)
    np.testing.assert_allclose(mu,oldmu[ix],rtol=1e-6,atol=2e-7)
    np.testing.assert_allclose(sd,oldsd[ix],rtol=1e-6,atol=2e-7)
    return m,norm,trainX,oldmu,oldsd


def numerical_event(spec,r):
    t=spec['total'];j=spec['target_index'];pool=POOLS[r.task];after=frame(r,t)
    result=dict(**spec,task=r.task,regime=r.regime,method=r.method,seed=r.seed)
    try:
        norm_after,error_after=replay(r,after)
        m,norm,trainX,oldmu,oldsd=reconstruct(r,after if spec['initial'] else frame(r,t-1))
        with torch.no_grad():
            K=m.covar_module(trainX).to_dense().numpy();noise=float(m.likelihood.noise.reshape(()))
            A=K+noise*np.eye(len(K));c=float(m.mean_module.constant.reshape(()))
            outmean=float(m.outcome_transform.means.reshape(()));outstd=float(m.outcome_transform.stdvs.reshape(()))
            cross=m.covar_module(X[r.task][[j]],trainX).to_dense().numpy().reshape(-1)
            alpha=np.linalg.solve(A,m.train_targets.detach().numpy().reshape(-1)-c)
            contributions=cross*alpha*outstd*norm['std']*pool.scale
            background=(norm['mean']+norm['std']*(outmean+outstd*c))*pool.scale
        expected=background+contributions.sum()
        np.testing.assert_allclose(expected,oldmu[j]*pool.scale,rtol=1e-6,atol=2e-5)
        result.update(status='PASS',acquisition_replay_error_after=error_after,
            mean_background=background,mean_contributions=contributions.tolist(),
            contributing_observation_ids=[h['candidate_id'] for h in norm['ordered']],
            strongest_absolute_contributor=norm['ordered'][int(np.argmax(np.abs(contributions)))]['candidate_id'])
        if spec['initial']:return result
        before=frame(r,t-1);_,error_before=replay(r,before)
        new=STATES[r.path_id]['history'][t-2];ix=pool.ix[new['candidate_id']]
        old_n=before['n'];assert len(trainX)==old_n
        with torch.no_grad():
            ktrain=m.covar_module(trainX,X[r.task][[ix]]).to_dense().numpy().reshape(-1)
            sol=np.linalg.solve(A,ktrain);parts=[]
            for xx in X[r.task].split(512):
                ka=m.covar_module(xx,trainX).to_dense().numpy()
                kb=m.covar_module(xx,X[r.task][[ix]]).to_dense().numpy().reshape(-1)
                parts.append(kb-ka@sol)
        multiplier=(outstd*norm['std'])**2;cov=np.concatenate(parts)*multiplier
        variance=oldsd**2;denom=variance[ix]+noise*multiplier
        gain=cov/denom;innovation=float(new['y']-oldmu[ix])
        fixed_mu=oldmu+gain*innovation;fixed_sd=np.sqrt(np.maximum(variance-cov*cov/denom,1e-30))
        assert np.all(fixed_sd<=oldsd+1e-8)
        available=after['available'];ties_old=before['tie_priority'];ties_new=after['tie_priority']
        stages=[('before',before['base_logei'],before['available'],ties_old),
            ('depletion_only',before['base_logei'],available,ties_old),
            ('new_tie_order',before['base_logei'],available,ties_new)]
        def score(mu,sd,best_raw,normalizer,seed):
            return logei((mu-normalizer['mean'])/normalizer['std'],sd/normalizer['std'],
                         (best_raw-normalizer['mean'])/normalizer['std'],seed)
        best_before=max(h['y'] for h in STATES[r.path_id]['history'][:old_n]);best_after=max(best_before,float(new['y']))
        stages += [('new_incumbent',score(oldmu,oldsd,best_after,norm,norm['seed']),available,ties_new),
            ('new_MC_draw',score(oldmu,oldsd,best_after,norm,norm_after['seed']),available,ties_new),
            ('new_acquisition_scale',score(oldmu,oldsd,best_after,norm_after,norm_after['seed']),available,ties_new),
            ('fixed_kernel_observation_update',score(fixed_mu,fixed_sd,best_after,norm_after,norm_after['seed']),available,ties_new),
            ('actual_refitted_GP',after['base_logei'],available,ties_new)]
        for name,scores,mask,ties in stages:
            rr=rank(scores,mask,ties,j)
            result[name+'_rank']=rr['rank'];result[name+'_strict_rank']=rr['strict_rank'];result[name+'_ties']=rr['ties']
        adm=[name for name,scores,mask,ties in stages if rank(scores,mask,ties,j)['rank']<=5]
        result.update(first_sufficient_stage=adm[0] if adm else 'not_in_native_top5',
            acquisition_replay_error_before=error_before,conditioning_coefficient=float(gain[j]),
            measurement_innovation=innovation*pool.scale,
            target_mean_before=oldmu[j]*pool.scale,target_SD_before=oldsd[j]*pool.scale,
            target_mean_fixed_update=fixed_mu[j]*pool.scale,target_SD_fixed_update=fixed_sd[j]*pool.scale,
            target_mean_after=after['mu_y'][j]*pool.scale,target_SD_after=after['sd_y'][j]*pool.scale,
            mean_change_from_conditioning=float(gain[j]*innovation*pool.scale),
            mean_change_refit_rescale=float((after['mu_y'][j]-fixed_mu[j])*pool.scale),
            innovation_is_below80=float(new['raw_value']/TRUTH[r.task].max())<.8)
        return result
    except (ValueError,AssertionError,RuntimeError,FileNotFoundError) as e:
        result.update(status='UNRESOLVED_REPLAY',error=str(e)[:1200]);return result


def initialize():
    torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    OUT.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(io.BytesIO(tracked(SOURCE/'conditional_path_profiles.csv')),encoding='utf-8-sig')
    for task,full in profile.FULL.items():
        pool=profile.late.pool_for(full) if task=='aryl' else profile.knowledge_gp.load_task(ROOT,full,seed=0)
        POOLS[task]=pool;tracked(pool.source_path)
        with np.load(io.BytesIO(tracked(pool.source_path.with_name('oracle.npz'))),allow_pickle=False) as z:
            TRUTH[task]=z['raw_values'].copy() if 'raw_values' in z else z['values'].copy()*pool.scale
    MOD['mixed']=profile.knowledge_gp._legacy_import('mixed_kalman_suzuki_engine',ROOT)
    MOD['nominal']=profile.knowledge_gp._legacy_import('reaction_transfer_engine',ROOT)
    for task,pool in POOLS.items():
        z=MOD['mixed'].encode_mixed(pool.raw_pool) if pool.group=='P0' else MOD['nominal'].encode(pool.raw_pool) if pool.group=='P1' else pool.raw_pool['z']
        X[task]=torch.tensor(z,dtype=torch.double)
    for _,r in d.iterrows():
        raw=tracked(ROOT/r.source_path);assert hashlib.sha256(raw).hexdigest()==r.source_sha256
        STATES[r.path_id]=json.loads(raw.decode('utf-8-sig'))
    profile.STATES=STATES
    return d


def main():
    tracked(__file__);tracked(ROOT/'docs/v13_highpoint_origin_20261104.txt')
    d=initialize()
    if '--check' in sys.argv:
        rows=[]
        for task in ('c1','doyle','hplc'):
            r=d[(d.primary)&(d.method=='v13')&(d.regime=='warm')&(d.task==task)].iloc[0]
            f=frame(r,5);_,err=replay(r,f);reconstruct(r,f)
            rows.append(dict(task=task,acquisition_error=err,posterior_recovered=True))
        print(json.dumps(rows,indent=2));return
    traces=[];events=[];specs=[]
    with ThreadPoolExecutor(max_workers=8) as executor:
        for rr,ee,ss in executor.map(trace_path,[r for _,r in d.iterrows()]):
            traces.extend(rr);events.extend(ee);specs.extend(ss)
    save('all_candidate_origin_traces.csv',traces);save('all_first_exposure_events.csv',events)
    print('ARCHIVE_TRACE_COMPLETE',len(traces),'target rows;',len(specs),'numerical events',flush=True)
    indexed=d.set_index('path_id',drop=False)
    with ThreadPoolExecutor(max_workers=8) as executor:
        results=list(executor.map(lambda spec:numerical_event(spec,indexed.loc[spec['path_id']]),specs))
    for row in results:
        for k,v in row.items():
            if isinstance(v,list):row[k]=json.dumps(v)
    save('one_step_numerical_mechanisms.csv',results)
    for path,h in READS.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h,path
    audit=dict(status='PASS_ARCHIVE_TRACE',paths=len(d),trace_rows=len(traces),target_events=len(events),
        numerical_events=len(results),numerical_replay_passed=sum(r['status']=='PASS' for r in results),
        numerical_replay_unresolved=sum(r['status']!='PASS' for r in results),
        source_files_checked=len(READS),new_GP_fits=0,new_LLM_calls=0,new_experiments=0,
        diagnostic_only='frozen-kernel one-observation updates and acquisition decompositions; no new trajectories')
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    (OUT/'source_hashes.json').write_text(json.dumps(READS,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
