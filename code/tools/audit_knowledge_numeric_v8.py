"""Independent, read-only V8 audit. Never import or rerun its decision engine.

This first entrypoint explicitly distinguishes checked numerical/selection
contracts from pending birth-kernel, prompt and version-transition audits.
Missing checks cannot be reported as PASS. Only stored own outcomes are read.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from hashlib import sha256
from datetime import datetime,timezone
import json
from math import exp,log,sqrt,tanh,fsum,copysign
from pathlib import Path
import sys

import numpy as np
from scipy.special import logsumexp
import audit_knowledge_transfer_loop as old

ROOT=old.ROOT
sys.path.insert(0,str(ROOT/'code/chemist_system'))
from chem.benchmarks import knowledge_numeric_forecast as forecast_validation

TASKS=tuple(f'Reizman_Suzuki_Case_{i}' for i in range(1,5))+('Perera_Suzuki','Doyle_Buchwald_Hartwig','SnAr','HPLC')
GAINS=('gp_gain','null_gain','mean_gain','null_mean_gain')
ATOL=1e-12
REQUIRED_CHECKS=('registration_and_pool','initial_unique_budget','native_GP_and_MC',
    'birth_kernel_and_covariance_custody','birth_conditioning','candidate_forecasts',
    'full_candidate_selection','four_account_feedback','version_and_inheritance',
    'LLM_public_prefix_and_schedule','final_state_replay')
read,digest,filehash=old.read,old.digest,old.filehash


def require(condition,message):
    if not condition:raise AssertionError(message)


def close(actual,expected,message,atol=1e-9,rtol=1e-8):
    np.testing.assert_allclose(actual,expected,atol=atol,rtol=rtol,err_msg=message)


def manifest_guard(protocol,paths,arms=('bo','full')):
    require(protocol['protocol_sha256']==digest({k:v for k,v in protocol.items() if k!='protocol_sha256'}),'protocol hash')
    require(tuple(protocol['tasks'])==TASKS,'registered eight-task coverage')
    require(len(protocol['inputs'])==8 and {x['task'] for x in protocol['inputs']}==set(TASKS),'complete pool manifest')
    require(protocol['initial']==4 and protocol['queries']==16 and protocol['seed']==0,'frozen seed/budget')
    actual=[p.parent.name for p in paths]
    expected={f'{task}_{arm}_agent_initial_terra' for task in TASKS for arm in arms}
    require(len(actual)==len(set(actual)),'duplicate run directories')
    require(expected<=set(actual),'missing BO/full stage1 trajectory')


def source_custody(protocol,folder):
    code=ROOT/'code/chemist_system'
    files=list((code/'chem').rglob('*.py'))+[code/'knowledge_loop.py']
    actual={p.relative_to(ROOT).as_posix():filehash(p) for p in sorted(files)}
    require(actual==protocol['source_sha256'],'entire frozen source inventory/SHA')
    snapshot=ROOT/'runtime/knowledge_loop/snapshots'/folder.name
    require(read(snapshot/'protocol.json')==protocol,'independent frozen protocol copy')
    for name,expected in actual.items():require(filehash(snapshot/name)==expected,'frozen source snapshot SHA')
    for pathkey,hashkey in [('goal_config_path','goal_config_sha256'),('design_document_path','design_document_sha256')]:
        if hashkey in protocol:require(filehash(ROOT/protocol[pathkey])==protocol[hashkey],'frozen design/goal SHA')
    require(protocol['seed']==0 and protocol.get('numerical_seed',0)==0 and protocol.get('tie_seed',0)==0 and protocol.get('menu_seed',0)==0,'registered independent random streams')
    return dict(source_files=len(actual),source_inventory_sha256=digest(actual),snapshot=str(snapshot))


def status_from_checks(checks,failures=()):
    if failures or any(v=='FAIL' for v in checks.values()):return 'FAIL'
    return 'PASS_LISTED_CHECKS' if all(checks.get(k)=='PASS' for k in REQUIRED_CHECKS) else 'PENDING_UNAUDITED_CONTRACTS'


def array_receipt(value):
    x=np.ascontiguousarray(value)
    return dict(shape=list(x.shape),dtype=str(x.dtype),sha256=sha256(x.tobytes()).hexdigest())


def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


class BirthAudit:
    """Independent immutable birth model reconstruction; no fitting/writes."""
    def __init__(self,folder,run,pool,protocol):
        self.folder,self.run,self.pool,self.protocol=folder,run,pool,protocol
        self.cache={};self.block_count=0

    def snapshot(self,h,prefix):
        import torch
        from botorch.models import SingleTaskGP
        from botorch.models.gp_regression_mixed import MixedSingleTaskGP
        meta=h['birth'];key=meta['snapshot_sha256'];pool=self.pool
        require(key==digest({k:v for k,v in meta.items() if k!='snapshot_sha256'}),'birth metadata SHA')
        contract={k:h[k] for k in ('id','version','created_round','focus','scope','plus_id','minus_id','numeric_prior','feature_vector')}
        require(meta['contract_sha256']==digest(contract),'birth immutable rule contract')
        born=meta['birth_prefix'];require(len(born)==h['created_round']+3 and digest(prefix[:len(born)])==digest(born),'exact chronological birth prefix')
        require(h['birth_prefix_sha256']==meta['birth_prefix_sha256']==digest(born) and h['birth_prefix']==born,'birth prefix binding')
        require(meta['task']==pool['task'] and meta['additional_GP_fits']==0,'birth task/no extra fit')
        descriptor_keys=('schema','task','contract_sha256','birth_prefix','birth_prefix_sha256','fit_key','fit_cache_sha256','fit_cache_directory','pool_sha256','input_X','source_sha256')
        require(meta['snapshot_id']==digest({k:meta[k] for k in descriptor_keys}),'birth snapshot descriptor SHA')
        for name,expected in meta['source_sha256'].items():require(filehash(name)==expected,'birth source SHA')
        poolpath=next(ROOT/k for m in self.protocol['inputs'] if m['task']==pool['task'] for k in m['files_sha256'] if k.endswith('/pool.json'))
        require(meta['pool_sha256']==filehash(poolpath),'birth candidate pool SHA')
        sf=self.run/'births'/meta['snapshot_id'][:24]
        require(read(sf/'metadata.json')==meta,'durable metadata equals committed snapshot')
        require(filehash(sf/'vectors.npz')==meta['vector_file_sha256'],'birth vector artifact SHA')
        path=self.folder/'cache'/pool['task']/(meta['fit_key']+'.pt')
        require(Path(meta['fit_cache_directory']).resolve()==path.parent.resolve() and filehash(path)==meta['fit_cache_sha256'],'birth cache pinned location/SHA')
        if key in self.cache:return self.cache[key]
        a=load_npz(sf/'vectors.npz');require({k:array_receipt(v) for k,v in a.items()}==meta['vectors'],'birth per-array receipts')
        saved=torch.load(path,map_location='cpu',weights_only=True)
        require(saved['key']==meta['fit_key'],'birth cache key')
        require(digest({k:array_receipt(v.detach().cpu().numpy()) for k,v in sorted(saved['state'].items())})==meta['fit_state_sha256'],'birth parameter state SHA')
        raw=pool['raw']
        if pool['group']=='P0':
            labels=['P1-L1','P2-L1','P1-L2','P1-L3','P1-L4','P1-L5','P1-L6','P1-L7']
            features=np.column_stack((np.asarray(raw['z'])[:,:3].clip(1e-6,1-1e-6),[saved['category_map'][labels.index(p['catalyst'])] for p in raw['points']]))
            cats=[3]
        elif pool['group']=='P1':
            features=np.asarray([[saved['mapping'][axis][p[axis]] for axis in pool['axes']] for p in raw['points']],float);cats=list(range(len(pool['axes'])))
        else:features=np.asarray(raw['z'],float);cats=None
        require(array_receipt(features)==meta['input_X'],'birth immutable input encoding')
        X=torch.tensor(features,dtype=torch.double)
        ordered=sorted(born,key=(lambda r:pool['features'][pool['ix'][r['candidate_id']]]) if pool['group']=='P2' else (lambda r:r['candidate_id']))
        train=X[[pool['ix'][r['candidate_id']] for r in ordered]];y=torch.tensor([[r['y']] for r in ordered],dtype=torch.double)
        om,os=y.mean(),y.std(unbiased=False).clamp_min(1e-12)
        if 'train_candidate_ids' in saved:
            require(saved['train_candidate_ids']==[r['candidate_id'] for r in ordered] and torch.equal(torch.as_tensor(saved['train_X'],dtype=torch.double),train),'birth cached training order/encoding')
        else:require(saved['history']==[dict(candidate_id=r['candidate_id'],y=r['y']) for r in ordered],'birth cached training observations')
        model=SingleTaskGP(train,(y-om)/os) if cats is None else MixedSingleTaskGP(train,(y-om)/os,cat_dims=cats)
        model.load_state_dict(saved['state']);model.eval()
        with torch.no_grad():
            noise=float(model.likelihood.noise.reshape(()));s=float(model.outcome_transform.stdvs.reshape(())*os);s2=s*s
            K=model.covar_module(train).to_dense()+noise*torch.eye(len(train),dtype=torch.double)
            cross=torch.cat([model.covar_module(part,train).to_dense() for part in X.split(256)])
            inverse=torch.linalg.solve(K,cross.T)
            mean=(model.mean_module(X)+cross@torch.linalg.solve(K,model.train_targets-model.mean_module(train)))*s+float(model.outcome_transform.means.reshape(())*os+om)
            prior=torch.cat([model.covar_module(part,diag=True) for part in X.split(256)])*s2
            proj=(cross*inverse.T).sum(-1)*s2;var=prior-proj
        close(a['mean'],mean.numpy(),'independent birth mean',atol=2e-10)
        close(a['latent_variance'],var.numpy(),'independent birth latent variance',atol=2e-10)
        close(a['training_observation_covariance'],K.numpy(),'independent birth train matrix',atol=2e-12)
        close(a['prior_diagonal'],prior.numpy(),'birth prior diagonal');close(a['projection_diagonal'],proj.numpy(),'birth projection diagonal')
        close(meta['noise_variance'],noise*s2,'birth noise common units',atol=1e-13)
        require(meta['noise_variance']==meta['normalization']['noise_variance_y'],'birth noise aliases')
        require(a['feature'].tolist()==h['feature_vector'],'immutable feature vector snapshot')
        ref=next(r for r in born if r['candidate_id']==h['minus_id']);plus=pool['ix'][h['plus_id']]
        require(h['plus_id'] not in {r['candidate_id'] for r in born} and ref['y']==h['reference_y']==meta['reference_y'],'birth new plus and fixed observed reference')
        close(meta['prior_mean'],h['numeric_prior']['delta_mean_raw']/pool['scale']-(a['mean'][plus]-ref['y']),'birth prior mean raw units')
        close(meta['prior_variance'],(h['numeric_prior']['delta_sd_raw']/pool['scale'])**2,'birth prior variance raw units')
        with np.load(self.run/'rounds'/f"{h['created_round']:02d}_GP.npz",allow_pickle=False) as birthgp:
            close(birthgp['mu_y'],a['mean'],'birth/current equality only at creation',atol=2e-10)
            close(birthgp['sd_y']**2,a['latent_variance'],'birth/current creation variance',atol=2e-10)
        require(read(self.run/'rounds'/f"{h['created_round']:02d}_commit.json")['GP_fit_key']==meta['fit_key'],'birth fit creation round')
        self.cache[key]=dict(meta=meta,arrays=a,model=model,X=X,cross=cross.numpy(),inverse=inverse.numpy(),scale2=s2,folder=sf,columns={})
        return self.cache[key]

    def verify(self,h,prefix,record,calibrate):
        import torch
        s=self.snapshot(h,prefix);meta,a=s['meta'],s['arrays'];pool=self.pool
        post=prefix[len(meta['birth_prefix']):];zi=[pool['ix'][r['candidate_id']] for r in post];qi=record['query_indices']
        require(not set(qi)&{pool['ix'][r['candidate_id']] for r in prefix},'birth forecasts must precede observations')
        receipt=record['block_receipt'];require(receipt['receipt_sha256']==digest({k:v for k,v in receipt.items() if k!='receipt_sha256'}),'covariance block receipt SHA')
        require(receipt['snapshot_sha256']==meta['snapshot_sha256'] and receipt['history_sha256']==digest(prefix),'block snapshot/prefix binding')
        path=Path(receipt['file']);require(path.resolve().parent==s['folder'].resolve(),'block path inside own birth')
        require(filehash(path)==receipt['file_sha256'] and read(path.with_suffix('.json'))==receipt,'durable covariance block')
        block=load_npz(path)
        require(block['post_indices'].tolist()==zi and block['query_indices'].tolist()==qi,'covariance index identities/order')
        require(receipt['query_ids']==[pool['ids'][i] for i in qi] and receipt['post_birth_ids']==[pool['ids'][i] for i in zi],'covariance physical point binding')
        for name in ('cov_zz','cov_xz'):require(array_receipt(block[name])==receipt[name],'covariance array receipt')
        require(len(receipt['columns'])==len(zi),'complete postbirth covariance columns')
        for index,cr in zip(zi,receipt['columns']):
            cp=s['folder']/f'column_{index:06d}.npz';column=load_npz(cp)['column']
            require(read(cp.with_suffix('.json'))==cr and filehash(cp)==cr['file_sha256'] and array_receipt(column)==cr['array'],'column durable receipt')
            require(cr['snapshot_sha256']==meta['snapshot_sha256'] and cr['index']==index and cr['point_id']==pool['ids'][index],'column index binding')
            if index not in s['columns']:
                with torch.no_grad():raw=torch.cat([s['model'].covar_module(part,s['X'][index:index+1]).to_dense().reshape(-1) for part in s['X'].split(256)]).numpy()*s['scale2']
                projected=s['cross']@s['inverse'][:,index]*s['scale2']
                # Direct solve versus Cholesky can differ by a few ulps; this is
                # a verification tolerance, never a covariance repair.
                close(column,raw-projected,'independent full birth covariance column',atol=2e-10,rtol=1e-8)
                require(column[index]==a['latent_variance'][index],'exact frozen column diagonal')
                s['columns'][index]=column
        zz=np.column_stack([s['columns'][j][zi] for j in zi]) if zi else np.empty((0,0))
        xz=np.column_stack([s['columns'][j][qi] for j in zi]) if zi else np.empty((len(qi),0))
        require(np.array_equal(zz,block['cov_zz']) and np.array_equal(xz,block['cov_xz']),'column/block covariance equality')
        rebuilt=independent_birth_condition(a['mean'],a['latent_variance'],a['feature'],meta['prior_mean'],meta['prior_variance'],meta['noise_variance'],zi,qi,[r['y'] for r in post],zz,xz,calibrate)
        for name in ('full','null','cut_feedback','variance_only'):
            for field,value in rebuilt[name].items():close(record['gaussians'][name][field],value,'independent birth '+name+'/'+field,atol=2e-9)
        for field,value in rebuilt['actual'].items():close(record['actual'][field],value,'actual calibrated/cut Gaussian',atol=2e-9)
        for field in ('beta_mean','beta_variance','residual_feature'):close(record[field],rebuilt[field],'independent '+field,atol=2e-9)
        self.block_count+=1

    def pre_call_coverage(self,hypotheses,prefix,arrays,calibrate):
        """Restore coverage even for versions retired by the forthcoming call."""
        rows=[];bo=old.select(arrays['base_logei'],arrays['eligible'],arrays['tie_priority'])
        eligible=np.flatnonzero(arrays['eligible']);n=len(self.pool['ids'])
        for h in hypotheses:
            if h['status']=='retracted':continue
            js=eligible[old.scope_mask(self.pool,h,eligible)];weight=evidence_summary(h)['trusted_weight'] if len(js) else 0.
            row=dict(id=h['id'],version=h['version'],eligible_inside_scope=len(js),executable_probes=len(js),weight=weight,
                can_change_rank_at_cap=False,eligibility_state='no_budgeted_scope' if not len(js) else 'numeric_scope')
            if len(js):
                s=self.snapshot(h,prefix);meta,a=s['meta'],s['arrays'];query=sorted(set(map(int,js))|{bo})
                block_id=digest(dict(snapshot=meta['snapshot_sha256'],history=prefix,indices=query))
                path=s['folder']/('block_'+block_id[:24]+'.npz');block=load_npz(path);receipt=read(path.with_suffix('.json'))
                post=prefix[len(meta['birth_prefix']):];zi=[self.pool['ix'][r['candidate_id']] for r in post]
                pred=independent_birth_condition(a['mean'],a['latent_variance'],a['feature'],meta['prior_mean'],meta['prior_variance'],meta['noise_variance'],zi,query,[r['y'] for r in post],block['cov_zz'],block['cov_xz'],calibrate)
                values=independent_tempering(arrays['mu_y'][query],arrays['sd_y'][query]**2,pred['actual']['mean'],pred['actual']['latent_variance'],max(r['y'] for r in prefix))
                record=dict(query_indices=query,scope_indices=js.tolist(),gaussians={name:pred[name] for name in ('full','null','cut_feedback','variance_only')},actual=pred['actual'],block_receipt=receipt,
                    **{k:pred[k] for k in ('beta_mean','beta_variance','residual_feature')})
                self.verify(h,prefix,record,calibrate)
                positions={j:p for p,j in enumerate(query)};offsets=np.zeros(n);offsets[js]=values['offset'][[positions[int(j)] for j in js]]
                cap=np.logaddexp(log(.5),log(.5)+offsets);cap[np.abs(offsets)<=ATOL]=0.
                winner=old.select(arrays['base_logei']+cap,arrays['eligible'],arrays['tie_priority'])
                row['can_change_rank_at_cap']=bool(winner!=bo and winner in set(js) and abs(offsets[winner])>ATOL)
                row['positive_effect_probes']=int(np.count_nonzero(offsets[js]>ATOL));row['negative_effect_probes']=int(np.count_nonzero(offsets[js]<-ATOL))
                row['eligibility_state']='numeric_rank_effect' if row['can_change_rank_at_cap'] else 'numeric_without_rank_effect'
            rows.append(row)
        return rows


def independent_tempering(gmean,gvar,emean,evar,best):
    gm,gv,em,ev,b=np.broadcast_arrays(*map(lambda x:np.asarray(x,float),(gmean,gvar,emean,evar,best)))
    require(np.isfinite(gm).all() and np.isfinite(em).all() and np.isfinite(gv).all() and np.isfinite(ev).all(),'finite Gaussian EI inputs')
    require(np.all(gv>=0) and np.all(ev>=0),'nonnegative latent variance')
    base=old.v4.normal_log_ei(gm,gv,b);expert=old.v4.normal_log_ei(em,ev,b)
    with np.errstate(invalid='ignore'):raw=np.where(gv==0,0.,expert-base)
    require(not np.isnan(raw).any() and not np.isposinf(raw).any(),'representable Gaussian EI ratio')
    noop=(gv==0)|(np.abs(raw)<=ATOL);lb=np.zeros(gm.shape);lr=np.zeros(gm.shape)
    above=(raw>log(4))&~noop;below=(raw<log(.25))&~noop
    lb[above]=-raw[above];lr[above]=log(3)-np.log1p(-np.exp(-raw[above]))
    lr[below]=log(.75)-np.log1p(-np.exp(raw[below]))
    lr=np.where(noop,-np.inf,lr);lt=lb+lr
    return dict(offset=np.where(noop,0.,np.clip(raw,log(.25),log(4))),log_tempering=lt,
        log_tempering_base=lb,log_tempering_remainder=lr,tempering=np.exp(lt),raw_log_ratio=raw,zero_effect_policy=noop)


def independent_four_scores(pred,y):
    """Same frozen lambda, independent Gaussian-density and moment algebra."""
    models=pred['models'];g=models['current'];f=pred['forecast']
    lt=-np.inf if f['log_tempering'] is None else f['log_tempering']
    lb=float(f.get('log_tempering_base',0.));lr=f.get('log_tempering_remainder',f['log_tempering']);lr=-np.inf if lr is None else float(lr)
    lam=exp(lt);other=1-lam
    def gaussian_gain(d):
        v=d['observation_variance'];vg=g['observation_variance'];require(v>0 and vg>0,'positive observation variance')
        dm=d['mean']-g['mean']
        if v==vg:return dm*(2*y-d['mean']-g['mean'])/(2*v)
        return .5*(log(vg/v)+(y-g['mean'])**2/vg-(y-d['mean'])**2/v)
    def gain(d):
        if lt==-np.inf:return 0.
        return float(np.logaddexp(log(other) if other else -np.inf,fsum([lb,lr,gaussian_gain(d)])))
    gains={name:gain(model) for name,model in models.items()}
    def shifted_mean(d):
        delta=d['mean']-g['mean']
        return g['mean'] if delta==0 or lt==-np.inf else g['mean']+copysign(exp(fsum([lb,lr,log(abs(delta))])),delta)
    em=shifted_mean(models['raw_expert']);n=models['birth_null'];nm=shifted_mean(n)
    nv=other*g['observation_variance']+lam*n['observation_variance']+lam*other*(g['mean']-n['mean'])**2
    mean=lambda target,reference,var:(target-reference)*(2*y-target-reference)/(2*var)
    return dict(gp_gain=gains['raw_expert'],null_gain=gains['raw_expert']-gains['birth_null'],
        mean_gain=mean(em,g['mean'],g['observation_variance']),null_mean_gain=mean(em,nm,nv),
        birth_null_gp_gain=gains['birth_null'],cut_feedback_gp_gain=gains['cut_feedback'],
        variance_only_gp_gain=gains['variance_only'],full_expert_gp_gain=gains['full_expert'])


def independent_birth_condition(mean,variance,feature,m0,P0,noise,zi,qi,y,zz,xz,calibrate=True):
    """Direct joint rank-one Gaussian conditioning, independent of BirthEffect."""
    mean,variance,feature=map(lambda x:np.asarray(x,float),(mean,variance,feature))
    zi,qi=np.asarray(zi,int),np.asarray(qi,int);y=np.asarray(y,float)
    zz,xz=np.asarray(zz,float),np.asarray(xz,float);n=len(zi)
    require(n<=16 and zz.shape==(n,n) and xz.shape==(len(qi),n),'birth covariance block dimensions')
    require(np.array_equal(zz,zz.T) and np.array_equal(np.diag(zz),variance[zi]),'exact birth diagonal/symmetry')
    if n:
        A=zz+noise*np.eye(n);residual=y-mean[zi]
        solve_y=np.linalg.solve(A,residual);solve_phi=np.linalg.solve(A,feature[zi])
        nm=mean[qi]+xz@solve_y;nv=variance[qi]-np.sum(xz*np.linalg.solve(A,xz.T).T,axis=1)
        r=feature[qi]-xz@solve_phi
        postP=P0/(1+P0*(feature[zi]@solve_phi));postm=m0/(1+P0*(feature[zi]@solve_phi))+postP*(feature[zi]@solve_y)
        joint=zz+P0*np.outer(feature[zi],feature[zi])+noise*np.eye(n)
        cross=xz+P0*np.outer(feature[qi],feature[zi])
        fm=mean[qi]+m0*feature[qi]+cross@np.linalg.solve(joint,y-mean[zi]-m0*feature[zi])
        fv=variance[qi]+P0*feature[qi]**2-np.sum(cross*np.linalg.solve(joint,cross.T).T,axis=1)
    else:
        nm=mean[qi];nv=variance[qi];r=feature[qi];postP,postm=P0,m0
        fm=nm+r*m0;fv=nv+r*r*P0
    tolerance=256*np.finfo(float).eps*np.maximum(np.abs(variance[qi]),1e-300)
    require(np.all(nv>=-tolerance) and np.all(fv>=-tolerance),'negative independent birth variance')
    nv=np.maximum(nv,0.);fv=np.maximum(fv,0.)
    def model(mu,var):return dict(mean=mu,latent_variance=var,observation_variance=var+noise)
    full=model(fm,fv);null=model(nm,nv);cut=model(nm+r*m0,nv+r*r*P0);actual=full if calibrate else cut
    return dict(full=full,null=null,cut_feedback=cut,variance_only=model(nm,actual['latent_variance']),actual=actual,
        beta_mean=postm if calibrate else m0,beta_variance=postP if calibrate else P0,residual_feature=r)


def evidence_summary(h):
    groups={}
    for kind,events in (('local',h['ledger']['events']),('inherited',[e for e in h.get('inherited_negative_evidence',[]) if e.get('applicable')])):
        for event in events:groups.setdefault((event['query_round'],event['point_id']),dict(local=[],inherited=[]))[kind].append(event)
    local={key:0. for key in GAINS};total=local.copy()
    for group in groups.values():
        for key in GAINS:
            own=[float(e[key]) for e in group['local']];negative=[min(0.,float(e[key])) for e in group['inherited']]
            local[key]+=min(own,default=0.);total[key]+=min(own+negative,default=0.)
    formula=min([.5]+[max(0.,tanh(total[k]/2)) for k in GAINS])
    return dict(local_scores=local,effective_scores=total,formula_weight=formula,
        trusted_weight=min(formula,max(0.,h['ledger']['trusted_weight'])))


def recovery_weight(before,formula,gains):
    increment=min(max(0.,tanh(gains[k]/2)) for k in GAINS) if all(gains[k]>0 for k in GAINS) else 0.
    return min(formula,before+increment),increment


def independent_selection(c,arrays,rows,calc,arm,used,params):
    base=arrays['base_logei'];eligible=arrays['eligible'];ties=arrays['tie_priority'];bo=old.select(base,eligible,ties)
    deployed=[];combined=base.copy();fallback=None
    if arm in ('full','no_calibration'):
        active=[r for r in rows if r['weight']>0 and old.hkey(r) in calc]
        total=sum(r['weight'] for r in active);scale=min(1.,params['maximum_deployment_weight']/total) if total else 1.
        deployed=[dict(id=r['id'],version=r['version'],weight=r['weight'],effective_mixture_weight=r['weight']*scale) for r in active]
    if deployed:
        weights=[r['effective_mixture_weight'] for r in deployed];offsets=np.stack([calc[old.hkey(r)]['offsets'] for r in deployed])
        correction=logsumexp(np.vstack([np.full_like(base,log(1-sum(weights)))]+[log(w)+d for w,d in zip(weights,offsets)]),axis=0)
        correction[np.all(np.abs(offsets)<=ATOL,axis=0)]=0.;combined+=correction
    selected=old.select(combined,eligible,ties);mode='base_bo';verification=None
    if selected!=bo:
        covered=any(selected in calc[old.hkey(r)]['indices'] and abs(calc[old.hkey(r)]['offsets'][selected])>ATOL for r in deployed)
        if covered:mode='knowledge_guided'
        else:selected=bo;fallback='uncovered_or_no_effect_weighted_winner'
    if arm!='bo' and selected==bo and c['query']<=params['last_verification_round'] and used<params['maximum_optional_verifications']:
        candidates=[]
        for row in rows:
            key=old.hkey(row)
            if row['weight']>0 or not row['can_change_rank_at_cap'] or row['eligible_inside_scope']<2 or key not in calc:continue
            h=next(h for h in c['hypotheses_at_commit'] if old.hkey(h)==key)
            if any(x<0 for x in evidence_summary(h)['effective_scores'].values()):continue
            data=calc[key];binfo=data['information'][data['positions'][bo]]
            for j in data['indices']:
                info=data['information'][data['positions'][int(j)]];gap=base[bo]-base[j]
                if j!=bo and gap<=min(c['epsilon'],params['verification_maximum_gap']) and info-binfo>=params['minimum_incremental_information'] and abs(data['offsets'][j])>ATOL:
                    candidates.append((float(info-binfo),-float(gap),-int(ties[j]),int(j),key,float(info),float(binfo)))
        if candidates:
            win=max(candidates,key=lambda x:x[:3]);selected=win[3];mode='optional_verification'
            verification=dict(id=win[4][0],version=win[4][1],incremental_information=win[0],information=win[5],BO_information=win[6],
                information_kind='birth_beta_mutual_information',remaining_budgeted_scope_candidates=len(calc[win[4]]['indices'])-1)
    return dict(selected=selected,mode=mode,deployed=deployed,combined=combined,verification=verification,fallback=fallback)


def stage_gate(rows):
    full={(r['task'],r['mode']):r for r in rows if r['status']=='COMPLETE'}
    missing=[(t,a) for t in TASKS for a in ('bo','full') if (t,a) not in full]
    if missing:return dict(status='INCOMPLETE_STAGE1',missing=missing,trigger_next_stage=None)
    wins=sum(full[t,'full']['SearchAUC']>full[t,'bo']['SearchAUC']+1e-9 for t in TASKS)
    weighted=sum(full[t,'full']['selection_modes'].get('knowledge_guided',0) for t in TASKS)
    return dict(status='DEVELOPMENT_PERFORMANCE_GATE_ONLY',strict_SearchAUC_wins=wins,weighted_choices=weighted,
        trigger_next_stage=bool(wins>=5 and weighted>=1),technical_PASS_required_separately=True)


def candidate_records(c,arrays,pool,prefix):
    """Check all disclosed candidate means/EI/MI and independently rank them."""
    n=len(pool['ids']);bo=pool['ix'][c['BO_reference_id']];records={old.hkey(r):r for r in c['numeric_candidates']}
    require(len(records)==len(c['numeric_candidates']),'duplicate candidate diagnostics')
    rows=[];calc={};eligible=np.flatnonzero(arrays['eligible'])
    for h in c['hypotheses_at_commit']:
        if h['status']=='retracted':continue
        js=eligible[old.scope_mask(pool,h,eligible)];key=old.hkey(h)
        trusted=evidence_summary(h)['trusted_weight'];weight=trusted if len(js) else 0.
        row=dict(id=h['id'],version=h['version'],eligible_inside_scope=len(js),executable_probes=len(js),weight=weight,
            can_change_rank_at_cap=False,eligibility_state='no_budgeted_scope' if not len(js) else 'numeric_scope')
        require((h['status']=='active')==bool(len(js)),'scope/active state')
        if len(js):
            require(key in records,'missing full candidate diagnostics')
            record=records[key];query=np.array(sorted(set(map(int,js))|{bo}),int)
            require(record['query_indices']==query.tolist() and record['scope_indices']==js.tolist(),'complete scope plus BO candidate set')
            actual={name:np.asarray(record['actual'][name],float) for name in ('mean','latent_variance','observation_variance')}
            require(all(x.shape==query.shape and np.isfinite(x).all() for x in actual.values()),'candidate Gaussian shape/finite')
            values=independent_tempering(arrays['mu_y'][query],arrays['sd_y'][query]**2,actual['mean'],actual['latent_variance'],max(r['y'] for r in prefix))
            close(record['offset'],values['offset'],'independent signed EI offset',atol=2e-8)
            close(record['tempering'],values['tempering'],'independent density tempering',atol=2e-8)
            for field in ('log_tempering_base','log_tempering_remainder'):
                recorded=np.asarray([(-np.inf if x is None else x) for x in record[field]])
                close(recorded,values[field],'independent split '+field,atol=2e-8)
            require(record['zero_effect_policy']==values['zero_effect_policy'].tolist(),'exact numerical-noop policy')
            nullvar=np.asarray(record['gaussians']['null']['latent_variance']);r=np.asarray(record['residual_feature'])
            info=.5*np.log1p(record['beta_variance']*r*r/(nullvar+h['birth']['noise_variance']))
            close(record['information'],info,'birth coefficient mutual information')
            positions={int(j):p for p,j in enumerate(query)};offsets=np.zeros(n);offsets[js]=values['offset'][[positions[int(j)] for j in js]]
            cap=np.logaddexp(log(.5),log(.5)+offsets);cap[np.abs(offsets)<=ATOL]=0.
            winner=old.select(arrays['base_logei']+cap,arrays['eligible'],arrays['tie_priority'])
            row['can_change_rank_at_cap']=bool(winner!=bo and winner in set(js) and abs(offsets[winner])>ATOL)
            row['positive_effect_probes']=int(np.count_nonzero(offsets[js]>ATOL));row['negative_effect_probes']=int(np.count_nonzero(offsets[js]<-ATOL))
            row['eligibility_state']='numeric_rank_effect' if row['can_change_rank_at_cap'] else 'numeric_without_rank_effect'
            calc[key]=dict(indices=js,positions=positions,offsets=offsets,information=info,record=record)
        else:require(key not in records,'diagnostics for unexecutable version')
        rows.append(row)
    require(set(records)==set(calc),'extraneous numeric candidate diagnostics')
    old.exact(rows,c['coverage'],'independent numeric scope coverage')
    return rows,calc


def structural_run(folder,path,protocol,pool,manifest,joint):
    state=read(path);arm=state['mode'];checks={k:'PENDING' for k in REQUIRED_CHECKS}
    require(path.parent.name==f"{state['task']}_{arm}_agent_initial_terra",'run directory/state task identity')
    require(state['status']=='COMPLETE' and len(state['history'])==20 and len(state['decisions'])==16,'complete4+16 run')
    require(state['protocol_sha256']==protocol['protocol_sha256'] and state['seed']==0,'run protocol/seed')
    require(state['condition']=='agent_initial_terra' and arm in protocol['supported_modes'],'registered mode/condition')
    history=state['history'];require(len({x['candidate_id'] for x in history})==20,'20 unique queried candidates')
    for r in history:
        require(r['candidate_id'] in pool['ix'] and np.isfinite(r['y']) and np.isfinite(r['raw_value']),'valid own observed result')
        close(r['raw_value'],r['y']*pool['scale'],'actual objective units')
    require([r['candidate_id'] for r in history[:4]]==manifest['initial_ids'],'initial IDs')
    old.common.initial_commit_audit(path,state,protocol,pool,manifest)
    checks['initial_unique_budget']='PASS';checks['registration_and_pool']='PASS'
    modes=Counter();forecast_count=0;used=0;call_prefix=[];prior_hypotheses=[]
    births=BirthAudit(folder,path.parent,pool,protocol);policy_reports=[]
    for t,d in enumerate(state['decisions'],1):
        cp=path.parent/'rounds'/f'{t:02d}_commit.json';c=read(cp);prefix=history[:t+3];obs=history[t+3]
        require(c['history_sha256']==digest(prefix) and digest(c)==d['commit_sha256'],'committed prefix/hash')
        require(c['query']==d['query']==t and c['selected_id']==d['selected_id']==obs['candidate_id'],'decision chronology')
        require(c['mode']==d['mode'] and c['BO_reference_id']==d['BO_reference_id'],'decision mode/reference aliases')
        require(c['protocol_sha256']==protocol['protocol_sha256'] and c['mechanism']=='frozen_birth_numeric_v8','commit protocol/mechanism')
        with np.load(cp.with_name(f'{t:02d}_GP.npz'),allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        n=len(pool['ids']);require(all(x.shape==(n,) and np.isfinite(x).all() for x in arrays.values()),'finite GP fields')
        available=~np.isin(pool['ids'],[r['candidate_id'] for r in prefix]);require(np.array_equal(available,arrays['available']),'available prefix mask')
        order=sorted(range(n),key=lambda j:pool['features'][j]);ties=np.empty(n,np.int64);ties[order]=np.random.default_rng(np.random.SeedSequence([0,t])).permutation(n)
        require(np.array_equal(ties,arrays['tie_priority']),'physical tie stream')
        bo=old.select(arrays['base_logei'],available,ties);epsilon=log(4)*(16-t)/15
        require(np.array_equal(arrays['eligible'],available&(arrays['base_logei'][bo]-arrays['base_logei']<=epsilon)),'qLogEI opportunity budget')
        require(c['BO_reference_id']==pool['ids'][bo] and arrays['eligible'][pool['ix'][c['selected_id']]],'BO reference/budgeted selection')
        close(c['epsilon'],epsilon,'epsilon');old.common.native_seed_audit(pool,c)
        require(c['eligible_count']==int(arrays['eligible'].sum()),'eligible budget count')
        close(c['base_logei'],arrays['base_logei'][pool['ix'][c['selected_id']]],'native selected base score')
        close(c['reference_logei'],arrays['base_logei'][bo],'native BO score')
        close(c['opportunity_gap'],arrays['base_logei'][bo]-arrays['base_logei'][pool['ix'][c['selected_id']]],'selected opportunity gap')
        fit=read(cp.with_name(f'{t:02d}_fit.json'))
        if joint:old.saved_gp_audit(folder,pool,prefix,c,arrays,fit)
        rows,calc=candidate_records(c,arrays,pool,prefix)
        for h in c['hypotheses_at_commit']:
            if old.hkey(h) in calc:births.verify(h,prefix,calc[old.hkey(h)]['record'],arm!='no_calibration')
        try:
            from knowledge_numeric_audit_policy import audit_numeric_policy
        except ImportError:audit_numeric_policy=None
        if audit_numeric_policy is not None:
            prior_coverage=births.pre_call_coverage(prior_hypotheses,prefix,arrays,arm!='no_calibration')
            policy_reports.append(audit_numeric_policy(pool,prefix,c,prior_hypotheses,call_prefix,state['decisions'][:t-1],folder/'requests',protocol,arrays,prior_coverage=prior_coverage,arm=arm))
        elif arm=='bo':require(not c['hypotheses_at_commit'] and not c['calls_at_commit'] and c['proposal'] is None and not c['reviews'],'BO has no LLM/rule state')
        choice=independent_selection(c,arrays,rows,calc,arm,used,protocol['parameters'])
        require(pool['ids'][choice['selected']]==c['selected_id'] and choice['mode']==c['mode'],'independent full-candidate argmax/mode')
        old.exact(choice['deployed'],c['deployed_versions'],'actual mixture weights');old.exact(choice['verification'],c['verification'],'optional chosen details')
        require(choice['fallback']==c['fallback_reason'],'uncovered weighted winner fallback')
        close(c['combined_logei'],choice['combined'][choice['selected']],'combined score')
        selected=pool['ix'][c['selected_id']]
        expected_shadow_keys=[old.hkey(h) for h in c['hypotheses_at_commit'] if old.hkey(h) in calc and selected in calc[old.hkey(h)]['indices']]
        require([old.hkey(sh) for sh in c['shadows']]==expected_shadow_keys,'all and only scoped selected forecasts')
        require(len(c['shadows'])==len(d['evidence']),'forecast/feedback pairing')
        updated=deepcopy(c['hypotheses_at_commit']);score_by_sha={}
        for sh,receipt in zip(c['shadows'],d['evidence']):
            pred=sh['prediction'];require(pred['prefix_sha256']==c['history_sha256'],'forecast prefix')
            h=next(h for h in updated if old.hkey(h)==old.hkey(sh));data=calc[old.hkey(h)];record=data['record'];pos=data['positions'][selected]
            require(pred['hypothesis_id']==h['id'] and pred['version']==h['version'] and pred['point_id']==obs['candidate_id'] and pred['query_round']==t,'forecast identity/version/chronology')
            require(pred['semantic_key']==h['semantic_key'] and pred['birth_snapshot_sha256']==h['birth']['snapshot_sha256'] and pred['current_gp_sha256']==digest(fit),'forecast frozen dependencies')
            require(pred['actual_calibrate']==(arm!='no_calibration'),'actual calibration arm')
            require(pred['reference_id']==h['minus_id'] and pred['reference_y']==h['reference_y'],'fixed rule reference')
            require(pred['best']==max(r['y'] for r in prefix),'prospective incumbent')
            models=dict(current=dict(mean=arrays['mu_y'][selected],latent_variance=arrays['sd_y'][selected]**2,observation_variance=arrays['sd_predictive_y'][selected]**2))
            for output,source in [('birth_null','null'),('cut_feedback','cut_feedback'),('variance_only','variance_only'),('full_expert','full')]:models[output]={k:v[pos] for k,v in record['gaussians'][source].items()}
            models['raw_expert']={k:v[pos] for k,v in record['actual'].items()}
            for name,gaussian in models.items():
                for field,value in gaussian.items():close(pred['models'][name][field],value,'selected frozen Gaussian '+name+'/'+field,atol=1e-12)
            close(pred['beta_variance'],record['beta_variance'],'selected coefficient variance');close(pred['residual_feature'],record['residual_feature'][pos],'selected residual feature')
            close(pred['information'],record['information'][pos],'selected calibration information');close(sh['information_nats'],pred['information'],'shadow information alias')
            require(sh['reference_is_observed'] is True and h['minus_id'] in {r['candidate_id'] for r in prefix},'own reference observed before experiment')
            validated=forecast_validation.score_forecast(pred,obs['y'],expected_prediction_sha256=sh['prediction_sha256'])
            independent=independent_four_scores(pred,obs['y'])
            for key,value in independent.items():close(validated[key],value,'independent '+key,atol=2e-8)
            for key in GAINS:close(receipt[key],independent[key],'feedback '+key,atol=2e-8)
            before=evidence_summary(h);ledger=h['ledger'];require(obs['candidate_id'] not in ledger['seen'] and obs['candidate_id'] not in {r['candidate_id'] for r in h['birth_prefix']},'new unique prospective feedback')
            require(all(e['query_round']<t for e in ledger['events']),'evidence strictly precedes experiment')
            event=dict(hypothesis_id=h['id'],version=h['version'],point_id=obs['candidate_id'],reference_id=h['minus_id'],reference_y=h['reference_y'],query_round=t,
                prediction=deepcopy(pred),observed_y=obs['y'],comparison_success=bool(obs['y']>h['reference_y']),**{key:validated[key] for key in GAINS},forecast_scores=validated)
            event['event_id']=digest(event);require(receipt['event_id']==event['event_id'],'feedback immutable event hash')
            ledger['events'].append(event);ledger['seen']=sorted(set(ledger['seen'])|{obs['candidate_id']})
            after=evidence_summary(h);target,increment=recovery_weight(before['trusted_weight'],after['formula_weight'],independent)
            close(receipt['trusted_weight_before'],before['trusted_weight'],'prior cumulative credit');close(receipt['formula_weight'],after['formula_weight'],'four-account cumulative formula')
            for key,value in after['effective_scores'].items():close(receipt['effective_scores'][key],value,'inherited/local score '+key)
            close(receipt['trusted_weight_after'],target,'increment-limited trust');close(receipt['recovery_credit_limit'],increment,'recovery increment')
            close(receipt['weight_before'],before['trusted_weight'] if h['status']=='active' else 0.,'effective prior weight')
            close(receipt['weight_after'],target if h['status']=='active' else 0.,'effective posterior weight')
            require(receipt['comparison_is_credit_gate'] is False and receipt['comparison_success']==event['comparison_success'],'factual comparison is not a credit gate')
            require(receipt['transition']==('increment_limited_four_positive_utilities' if all(independent[k]>0 for k in GAINS) else 'nonincreasing_after_failed_utility'),'credit transition type')
            # Reproduce serialized credit using the independently checked exact
            # density score, so later source hashes are not changed by audit ulps.
            exact_target,_=recovery_weight(before['trusted_weight'],after['formula_weight'],validated)
            ledger['trusted_weight']=exact_target;ledger['weight']=exact_target if h['status']=='active' else 0.
            score_by_sha[pred['prediction_sha256']]=validated['gp_gain']
            forecast_count+=1
        weights={old.hkey(r):r['effective_mixture_weight'] for r in choice['deployed']}
        expected_mixture=[dict(id=sh['id'],version=sh['version'],weight=weights[old.hkey(sh)],prediction_sha256=sh['prediction_sha256']) for sh in c['shadows'] if weights.get(old.hkey(sh),0)>0]
        old.exact(c['selected_mixture'],expected_mixture,'actual scoped selected mixture')
        total=sum(r['weight'] for r in expected_mixture)
        mixture_gain=float(logsumexp([log(1-total)]+[log(r['weight'])+score_by_sha[r['prediction_sha256']] for r in expected_mixture])) if expected_mixture else 0.
        close(d['deployed_mixture_log_score_gain'],mixture_gain,'actual mixture proper score')
        close(d['incumbent_before_y'],max(r['y'] for r in prefix),'incumbent before');close(d['incumbent_after_y'],max(r['y'] for r in prefix+[obs]),'incumbent after')
        require(d['raw_value']==obs['raw_value'] and d['y']==obs['y'],'feedback outcome aliases')
        prior_hypotheses=updated
        require(c['calls_at_commit'][:len(call_prefix)]==call_prefix,'call prefix immutability');call_prefix=c['calls_at_commit']
        modes[c['mode']]+=1;used+=int(c['mode']=='optional_verification')
    require(state['calls']==call_prefix and state['verification_used']==used<=protocol['parameters']['maximum_optional_verifications'],'final calls/quota')
    old.exact(state['hypotheses'],prior_hypotheses,'final cumulative state replay')
    checks.update(native_GP_and_MC='PASS' if joint else 'PENDING',candidate_forecasts='PASS',full_candidate_selection='PASS',
        birth_kernel_and_covariance_custody='PASS',birth_conditioning='PASS',four_account_feedback='PASS',final_state_replay='PASS')
    if arm=='bo' or len(policy_reports)==16 and all(not r.get('pending') for r in policy_reports):
        checks.update(version_and_inheritance='PASS',LLM_public_prefix_and_schedule='PASS')
    raw=np.asarray([r['raw_value'] for r in history]);curve=np.maximum.accumulate(raw)
    return dict(task=state['task'],mode=arm,status=state['status'],checks=checks,technical_result=status_from_checks(checks),
        forecasts=forecast_count,birth_snapshots=len(births.cache),birth_blocks=births.block_count,policy_reports=policy_reports,
        selection_modes=dict(modes),SearchAUC=float(curve[4:].mean()),final_best=float(curve[-1]),
        initial_history_sha256=digest(history[:4]),history_sha256=digest(history),run=str(path.relative_to(ROOT)))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--variant',default='v8');parser.add_argument('--skip-gp',action='store_true');parser.add_argument('--bo-only',action='store_true')
    args=parser.parse_args();folder=ROOT/'experiments/runs/knowledge_loop_20261101'/args.variant
    out=ROOT/'experiments/reports/knowledge_loop_20261101'/f'{args.variant}_{"bo_interface_audit" if args.bo_only else "joint_independent_audit"}.json'
    rows=[];failures=[];custody=[]
    if not (folder/'protocol.json').is_file():
        result=dict(technical_result='NOT_RUN_PROTOCOL_ABSENT',runs=[],failures=[],pending=list(REQUIRED_CHECKS),fits=0,LLM_calls=0,oracle_access=False)
        out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(result['technical_result']);return
    protocol=read(folder/'protocol.json');paths=sorted((folder/'runs').glob('*_bo_agent_initial_terra/state.json' if args.bo_only else '*_agent_initial_terra/state.json'))
    try:
        manifest_guard(protocol,paths,('bo',) if args.bo_only else ('bo','full'))
        custody.append(source_custody(protocol,folder))
    except Exception as e:failures.append(dict(stage='complete_manifest',error=str(e)))
    manifests={m['task']:m for m in protocol['inputs']}
    for task,m in manifests.items():
        rel=next(k for k in m['files_sha256'] if k.endswith('/pool.json'));actual=filehash(ROOT/rel)
        require(actual==m['files_sha256'][rel],'frozen candidate pool hash')
        custody.append(dict(task=task,path=rel,actual_sha256=actual))
    for path in paths:
        try:
            state=read(path);manifest=manifests[state['task']];pool=old.common.public_pool(ROOT,manifest);pool['task']=state['task']
            row=structural_run(folder,path,protocol,pool,manifest,not args.skip_gp)
            baseline=read(ROOT/'experiments/runs/knowledge_loop_20261101/v7/runs'/f"{state['task']}_bo_agent_initial_terra/state.json")
            require(digest(state['history'][:4])==digest(baseline['history'][:4]),'matched historical BO initialization')
            if state['mode']=='bo':require(digest(state['history'])==digest(baseline['history']),'all20 matched BO points/outcomes')
            rows.append(row)
        except Exception as e:failures.append(dict(run=str(path.relative_to(ROOT)),error_type=type(e).__name__,error=str(e)))
    pending=sorted({key for row in rows for key,value in row['checks'].items() if value!='PASS'})
    result=dict(created_at=datetime.now(timezone.utc).isoformat(),auditor_sha256=filehash(__file__),protocol_sha256=protocol['protocol_sha256'],
        technical_result='FAIL' if failures else 'PENDING_UNAUDITED_CONTRACTS' if pending or len(rows)<(8 if args.bo_only else 16) else 'PASS_BO_INTERFACE_ONLY' if args.bo_only else 'PASS_LISTED_CHECKS',
        stage_gate=stage_gate(rows),runs=rows,custody=custody,failures=failures,pending=pending,
        totals=dict(runs=len(rows),rounds=16*len(rows),forecasts=sum(r['forecasts'] for r in rows)),
        fits=0,LLM_calls=0,oracle_access=False,unqueried_targets_read=0,
        limitations=['No pending item is counted as PASS; BO-only interface cannot qualify Agent behavior or performance.',
            'Native GP is reconstructed from saved weights without refitting. No new trajectory or performance claim.',
            'Commit hashes/source order establish prospective internal causality, not an OS-level wall-clock or filesystem I/O trace.',
            'Candidate pool contains inputs only; outcome checks use run observations, not independent oracle re-evaluation.',
            'Development performance is separate from numerical qualification and independent scientific confirmation.'])
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(technical_result=result['technical_result'],runs=len(rows),failures=len(failures),pending=pending),ensure_ascii=False))
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
