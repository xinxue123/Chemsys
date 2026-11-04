"""Frozen cached-GP transport for BirthEffect; no optimizer or oracle access.

The cache fit, its input encoding, outcome transforms and observation noise are
fixed at rule birth. Posterior covariance is evaluated with N-by-training and
N-by-post-birth blocks, never an N-by-N candidate matrix. Snapshot/column/block
files live outside the GP cache. Only roundoff-scale diagonal/symmetry changes
are allowed, and every such change is recorded; an invalid kernel is rejected.
"""
from copy import deepcopy
from hashlib import sha256
import inspect
import json
from pathlib import Path

import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.models.gp_regression_mixed import MixedSingleTaskGP

from . import knowledge_gp
from . import knowledge_birth_effect

_EPS=np.finfo(float).eps


def digest(value):
    return knowledge_gp.input_digest(value)


def filehash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def array_receipt(value):
    x=np.ascontiguousarray(value)
    return dict(shape=list(x.shape),dtype=str(x.dtype),sha256=sha256(x.tobytes()).hexdigest())


def _atomic_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def _atomic_npz(path,**values):
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as stream:np.savez_compressed(stream,**values)
    temporary.replace(path)


def _state_hash(state):
    return digest({k:array_receipt(v.detach().cpu().numpy()) for k,v in sorted(state.items())})


def _contract(h):
    return {k:deepcopy(h[k]) for k in ('id','version','created_round','focus','scope','plus_id','minus_id','numeric_prior','feature_vector')}


def _checked_history(pool,history):
    h=deepcopy(list(history))
    if not 4<=len(h)<=20 or len({x['candidate_id'] for x in h})!=len(h):
        raise ValueError('Expected4..20 distinct chronological observations')
    if any(x['candidate_id'] not in pool.ix or not np.isfinite(float(x['y'])) for x in h):
        raise ValueError('Unknown/nonfinite observed prefix')
    digest(h)
    return h


def _roundoff_replace(raw,target,scale,label):
    raw,target,scale=np.broadcast_arrays(np.asarray(raw,float),np.asarray(target,float),np.asarray(scale,float))
    difference=np.abs(raw-target);tolerance=128*_EPS*np.maximum(np.abs(scale),np.finfo(float).tiny)
    if np.any(~np.isfinite(raw)) or np.any(~np.isfinite(target)) or np.any(difference>tolerance):
        raise ValueError(label+' exceeds roundoff-only covariance normalization')
    return target.copy(),dict(kind=label,changed_entries=int(np.count_nonzero(difference)),
        maximum_change=float(difference.max()) if difference.size else 0.,
        maximum_tolerance=float(tolerance.max()) if tolerance.size else 0.,
        rule='128*float64_eps*sum_of_absolute_prior_and_projection_terms')


class BirthGPStore:
    """Persist and restore exactly one frozen GP per immutable rule version.

    birth returns a JSON-serializable copy, protected by snapshot_sha256. The
    caller stores it at h['birth']; h['birth_prefix'], if supplied, must match.
    predict returns full/null/cut_feedback/variance_only Gaussian dictionaries,
    each with mean/latent_variance/observation_variance arrays. calibrate=False
    selects the cut-feedback distribution without hiding full diagnostics.
    """
    def __init__(self,pool,gp_adapter,directory):
        self.pool,self.gp=pool,gp_adapter
        self.directory=Path(directory).resolve()
        self.cache_directory=Path(gp_adapter.directory).resolve()
        if gp_adapter.pool is not pool:raise ValueError('Pool/GP adapter identity mismatch')
        if (self.directory==self.cache_directory or self.directory.is_relative_to(self.cache_directory)
                or self.directory==pool.root or self.directory.is_relative_to(pool.source_path.parent.resolve())):
            raise ValueError('Birth files must be separate from GP cache and source inputs')
        self.directory.mkdir(parents=True,exist_ok=True)
        self._memory={}

    def _source_hashes(self):
        paths={Path(__file__).resolve(),Path(knowledge_gp.__file__).resolve(),Path(knowledge_birth_effect.__file__).resolve()}
        for cls in type(self.gp.cache).__mro__:
            try:p=inspect.getsourcefile(cls)
            except TypeError:p=None
            if p and Path(p).is_file():paths.add(Path(p).resolve())
        return {str(p):filehash(p) for p in sorted(paths)}

    def _restore_model(self,history,key,expected_file_sha=None):
        """Load constructor+state only: no cache.model, fitting or acquisition."""
        clean=[dict(candidate_id=x['candidate_id'],y=float(x['y'])) for x in history]
        ordered=sorted(clean,key=(lambda x:self.pool.features[self.pool.ix[x['candidate_id']]]) if self.pool.group=='P2' else (lambda x:x['candidate_id']))
        payload=dict(history=ordered,recipe=self.gp.cache.recipe)
        if self.pool.group=='P0':payload.update(case=self.gp.cache.case,seed=self.gp.cache.seed)
        if digest(payload)!=key:raise ValueError('Birth fit key does not bind this exact encoded prefix')
        path=self.cache_directory/(key+'.pt')
        if not path.is_file():raise FileNotFoundError('Existing birth GP fit required; fitting is forbidden: '+str(path))
        if expected_file_sha and filehash(path)!=expected_file_sha:raise ValueError('Birth cache file changed')
        saved=torch.load(path,map_location='cpu',weights_only=True)
        if saved['key']!=key:raise ValueError('Cache fit key mismatch')
        train=self.gp.X[[self.pool.ix[x['candidate_id']] for x in ordered]].detach().cpu().double()
        y=torch.tensor([[x['y']] for x in ordered],dtype=torch.double)
        mean=y.mean();std=y.std(unbiased=False).clamp_min(1e-12)
        if 'train_candidate_ids' in saved:
            if saved['train_candidate_ids']!=[x['candidate_id'] for x in ordered] or not torch.equal(torch.as_tensor(saved['train_X'],dtype=torch.double),train):
                raise ValueError('Cached training identities/encoding changed')
        elif saved.get('history')!=ordered:raise ValueError('Cached training observations changed')
        ys=(y-mean)/std
        model=SingleTaskGP(train,ys) if self.pool.group=='P2' else MixedSingleTaskGP(train,ys,cat_dims=[3] if self.pool.group=='P0' else list(range(len(self.pool.axes))))
        model.load_state_dict(saved['state']);model.eval()
        return model,train,float(mean),float(std),path,saved

    def _basis(self,model,train,mean,std):
        X=self.gp.X.detach().cpu().double()
        with torch.no_grad():
            noise=float(model.likelihood.noise.reshape(()))
            inner_mean=float(model.outcome_transform.means.reshape(()));inner_std=float(model.outcome_transform.stdvs.reshape(()))
            factor_scale=inner_std*std;scale2=factor_scale**2
            kernel=model.covar_module(train).to_dense()
            raw=kernel.numpy();symmetric,receipt=_roundoff_replace(raw,(raw+raw.T)*.5,np.abs(raw)+np.abs(raw.T),'training_kernel_symmetry')
            matrix=torch.from_numpy(symmetric)+noise*torch.eye(len(train),dtype=torch.double)
            L=torch.linalg.cholesky(matrix)
            cross=torch.cat([model.covar_module(part,train).to_dense() for part in X.split(256)])
            W=torch.linalg.solve_triangular(L,cross.T,upper=False)
            alpha=torch.cholesky_solve((model.train_targets-model.mean_module(train))[:,None],L).reshape(-1)
            mu=(model.mean_module(X)+cross@alpha)*factor_scale+inner_mean*std+mean
            # Some categorical kernel implementations form a dense block even
            # for diag=True. Bound that work to 256, never the whole pool.
            prior=torch.cat([model.covar_module(part,diag=True) for part in X.split(256)])
            reduction=(W*W).sum(0)
            variance=(prior-reduction)*scale2
            tolerance=128*_EPS*(prior.abs()+reduction)*scale2
            if torch.any(variance < -tolerance):raise ValueError('Birth GP has an invalid negative latent variance')
            negative=variance<0
            diagonal_receipt=dict(kind='birth_latent_diagonal_negative_roundoff',changed_entries=int(negative.sum()),maximum_change=float((-variance[negative]).max()) if negative.any() else 0.)
            variance=torch.where(negative,0.,variance)
        return dict(mean=mu.numpy(),latent_variance=variance.numpy(),whitened_cross=W.numpy(),
            train_X=train.numpy(),training_observation_covariance=matrix.numpy(),prior_diagonal=prior.numpy()*scale2,
            projection_diagonal=reduction.numpy()*scale2),dict(outer_mean=mean,outer_std=std,inner_mean=inner_mean,
            inner_std=inner_std,kernel_to_y_scale=factor_scale,noise_variance_y=noise*scale2,
            observation_noise_internal=noise,roundoff_normalizations=[receipt,diagonal_receipt])

    def birth(self,h,history,context):
        history=_checked_history(self.pool,history)
        if h['created_round']!=len(history)-3:raise ValueError('Birth round/prefix mismatch')
        if 'birth_prefix' in h and digest(h['birth_prefix'])!=digest(history):raise ValueError('h.birth_prefix mismatch')
        known={x['candidate_id']:x for x in history}
        if h['plus_id'] in known or h['plus_id'] not in self.pool.ix or h['minus_id'] not in known:raise ValueError('Birth needs new plus and known own reference')
        feature=np.asarray(h['feature_vector'],float)
        if feature.shape!=(len(self.pool.ids),) or not np.isfinite(feature).all():raise ValueError('Invalid frozen feature vector')
        if feature[self.pool.ix[h['plus_id']]]!=1 or feature[self.pool.ix[h['minus_id']]]!=0:raise ValueError('Feature must normalize plus=1/reference=0')
        prior=h['numeric_prior'];delta=float(prior['delta_mean_raw']);sd=float(prior['delta_sd_raw'])
        if not np.isfinite(delta) or not np.isfinite(sd) or sd<0:raise ValueError('Invalid numeric rule prior')
        contract=_contract(h);key=context['stats']['fit_key'];before=self.gp.counters['fits']
        model,train,mean,std,path,saved=self._restore_model(history,key)
        stats=context['stats']
        if 'model' in stats and _state_hash(stats['model'].state_dict())!=_state_hash(saved['state']):raise ValueError('Current context is not the pinned birth fit')
        for name,value in (('mean',mean),('std',std)):
            if name in stats and not np.isclose(float(stats[name]),value,rtol=1e-14,atol=1e-15):raise ValueError('Birth normalization mismatch')
        arrays,normalization=self._basis(model,train,mean,std);arrays['feature']=feature
        if 'mu_y' in stats and not np.allclose(arrays['mean'],stats['mu_y'],rtol=1e-9,atol=1e-10):raise ValueError('Birth mean disagrees with committed context')
        plus=self.pool.ix[h['plus_id']]
        m0=delta/self.pool.scale-(arrays['mean'][plus]-known[h['minus_id']]['y']);P0=(sd/self.pool.scale)**2
        descriptor=dict(schema='frozen_birth_gp_v1',task=self.pool.task,contract_sha256=digest(contract),
            birth_prefix=history,birth_prefix_sha256=digest(history),fit_key=key,fit_cache_sha256=filehash(path),
            fit_cache_directory=str(self.cache_directory),pool_sha256=filehash(self.pool.source_path),
            input_X=array_receipt(self.gp.X.detach().cpu().numpy()),source_sha256=self._source_hashes())
        # Short path tokens avoid legacy Windows path limits; full digests are
        # always checked inside metadata/receipts, so a collision fails closed.
        snapshot_id=digest(descriptor);folder=self.directory/snapshot_id[:24];folder.mkdir(exist_ok=True)
        if (folder/'metadata.json').exists():
            meta=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
            if meta['snapshot_id']!=snapshot_id:raise ValueError('Snapshot path-token collision')
            self._load(h,meta)
            return deepcopy(meta)
        vector_path=folder/'vectors.npz';_atomic_npz(vector_path,**arrays)
        sample=list(dict.fromkeys([self.pool.ix[x['candidate_id']] for x in history]+[plus]))
        with torch.no_grad():
            native=model.posterior(self.gp.X[sample]);native_mean=native.mean.reshape(-1).numpy()*std+mean
            native_cov=native.mvn.covariance_matrix.numpy()*std**2
        rawcov=model.covar_module(self.gp.X[sample]).to_dense().detach().numpy()*normalization['kernel_to_y_scale']**2-arrays['whitened_cross'][:,sample].T@arrays['whitened_cross'][:,sample]*normalization['kernel_to_y_scale']**2
        native_errors=dict(candidate_ids=[self.pool.ids[j] for j in sample],mean_max_abs=float(np.max(np.abs(native_mean-arrays['mean'][sample]))),covariance_max_abs=float(np.max(np.abs(native_cov-rawcov))))
        if native_errors['mean_max_abs']>1e-9 or native_errors['covariance_max_abs']>1e-9:raise ValueError('Birth direct/native posterior check failed')
        meta=dict(**descriptor,snapshot_id=snapshot_id,normalization=normalization,prior_mean=m0,prior_variance=P0,
            noise_variance=normalization['noise_variance_y'],
            numeric_prior_raw=deepcopy(prior),reference_y=known[h['minus_id']]['y'],scale=self.pool.scale,
            vector_file='vectors.npz',vector_file_sha256=filehash(vector_path),vectors={k:array_receipt(v) for k,v in arrays.items()},
            native_posterior_check=native_errors,fit_state_sha256=_state_hash(saved['state']),additional_GP_fits=self.gp.counters['fits']-before)
        if meta['additional_GP_fits']!=0:raise RuntimeError('Birth must never fit a GP')
        meta['snapshot_sha256']=digest(meta);_atomic_json(folder/'metadata.json',meta)
        self._load(h,meta)
        return deepcopy(meta)

    def _load(self,h,metadata=None):
        meta=deepcopy(metadata if metadata is not None else h.get('birth'))
        if not isinstance(meta,dict):raise ValueError('h.birth metadata is required')
        if meta['snapshot_sha256']!=digest({k:v for k,v in meta.items() if k!='snapshot_sha256'}):raise ValueError('Birth metadata hash changed')
        if meta['contract_sha256']!=digest(_contract(h)):raise ValueError('Frozen rule contract changed')
        if 'birth_prefix' in h and digest(h['birth_prefix'])!=meta['birth_prefix_sha256']:raise ValueError('Frozen birth prefix changed')
        if meta['source_sha256']!=self._source_hashes() or meta['pool_sha256']!=filehash(self.pool.source_path):raise ValueError('Birth source/input provenance changed')
        if meta['fit_cache_directory']!=str(self.cache_directory) or meta['input_X']!=array_receipt(self.gp.X.detach().cpu().numpy()):raise ValueError('Birth GP encoding/cache changed')
        folder=self.directory/meta['snapshot_id'][:24];on_disk=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
        if digest(on_disk)!=digest(meta):raise ValueError('Caller metadata differs from durable birth')
        if filehash(folder/'vectors.npz')!=meta['vector_file_sha256']:raise ValueError('Birth vector file changed')
        fit_path=self.cache_directory/(meta['fit_key']+'.pt')
        if filehash(fit_path)!=meta['fit_cache_sha256']:raise ValueError('Birth fit file changed')
        key=meta['snapshot_sha256']
        if key in self._memory:return self._memory[key]
        model,train,mean,std,_,saved=self._restore_model(meta['birth_prefix'],meta['fit_key'],meta['fit_cache_sha256'])
        with np.load(folder/'vectors.npz',allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        if {k:array_receipt(v) for k,v in arrays.items()}!=meta['vectors']:raise ValueError('Birth array receipts differ')
        effect=knowledge_birth_effect.BirthEffect(self.pool.ids,[x['candidate_id'] for x in meta['birth_prefix']],arrays['mean'],arrays['latent_variance'],arrays['feature'],meta['prior_mean'],meta['prior_variance'],meta['normalization']['noise_variance_y'],key)
        result=dict(meta=meta,folder=folder,model=model,arrays=arrays,effect=effect,columns={})
        self._memory[key]=result
        return result

    def _column(self,s,index,earlier):
        path=s['folder']/f'column_{index:06d}.npz';receipt_path=path.with_suffix('.json')
        if path.exists():
            receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
            if (receipt['snapshot_sha256']!=s['meta']['snapshot_sha256'] or receipt['file_sha256']!=filehash(path)
                    or receipt['point_id']!=self.pool.ids[index] or receipt['index']!=index):raise ValueError('Frozen covariance column changed')
            with np.load(path,allow_pickle=False) as z:column=z['column'].copy()
            if receipt['array']!=array_receipt(column):raise ValueError('Covariance column array hash differs')
        else:
            scale2=s['meta']['normalization']['kernel_to_y_scale']**2;W=s['arrays']['whitened_cross'];X=self.gp.X
            with torch.no_grad():raw=torch.cat([s['model'].covar_module(part,X[index:index+1]).to_dense().reshape(-1) for part in X.split(256)]).numpy()*scale2
            projection=W.T@W[:,index]*scale2;column=raw-projection;normalizations=[]
            diagonal,receipt=_roundoff_replace(column[index],s['arrays']['latent_variance'][index],abs(raw[index])+abs(projection[index]),'column_frozen_diagonal')
            column[index]=float(diagonal);normalizations.append(receipt)
            for prior_index in earlier:
                target=s['columns'][prior_index][index]
                symmetric,receipt=_roundoff_replace(column[prior_index],target,abs(raw[prior_index])+abs(projection[prior_index]),'column_frozen_symmetry')
                column[prior_index]=float(symmetric);normalizations.append(receipt)
            _atomic_npz(path,column=column)
            receipt=dict(snapshot_sha256=s['meta']['snapshot_sha256'],point_id=self.pool.ids[index],index=index,
                array=array_receipt(column),file_sha256=filehash(path),normalizations=normalizations)
            _atomic_json(receipt_path,receipt)
        s['columns'][index]=column
        return receipt

    def predict(self,h,history,indices,calibrate=True):
        history=_checked_history(self.pool,history);s=self._load(h);meta=s['meta'];birth=meta['birth_prefix']
        if len(history)<len(birth) or digest(history[:len(birth)])!=meta['birth_prefix_sha256']:raise ValueError('History no longer extends the exact birth prefix')
        post=history[len(birth):]
        if len(post)>16:raise ValueError('At most16 post-birth observations')
        js=np.asarray(indices)
        if js.ndim!=1 or (js.size and (not np.issubdtype(js.dtype,np.integer) or js.min()<0 or js.max()>=len(self.pool.ids))) or len(set(map(int,js)))!=len(js):raise ValueError('Invalid query indices')
        js=js.astype(int);known={x['candidate_id'] for x in history}
        if any(self.pool.ids[j] in known for j in js):raise ValueError('Queries must be genuinely unobserved')
        progress_path=s['folder']/'progress.json'
        if progress_path.exists():
            previous=json.loads(progress_path.read_text(encoding='utf-8'))
            if previous['snapshot_sha256']!=meta['snapshot_sha256'] or previous['history_sha256']!=digest(previous['history']):raise ValueError('Progress receipt changed')
            if len(history)<len(previous['history']) or digest(history[:len(previous['history'])])!=previous['history_sha256']:raise ValueError('Accepted chronological feedback was removed/reordered/changed')
        zi=[self.pool.ix[x['candidate_id']] for x in post];receipts=[]
        for k,j in enumerate(zi):receipts.append(self._column(s,j,zi[:k]))
        zz=np.column_stack([s['columns'][j][zi] for j in zi]) if zi else np.empty((0,0))
        xz=np.column_stack([s['columns'][j][js] for j in zi]) if zi else np.empty((len(js),0))
        before=self.gp.counters['fits']
        result=s['effect'].predict([x['candidate_id'] for x in post],[x['y'] for x in post],zz,[self.pool.ids[j] for j in js],xz)
        noise=meta['normalization']['noise_variance_y'];r=result['residual_feature'];nullvar=result['null_latent_variance']
        def gaussian(mean,var):return dict(mean=np.array(mean,copy=True),latent_variance=np.array(var,copy=True),observation_variance=np.asarray(var)+noise)
        full=gaussian(result['expert_mean'],result['expert_latent_variance']);null=gaussian(result['null_mean'],nullvar)
        cut=gaussian(result['null_mean']+r*meta['prior_mean'],nullvar+r*r*meta['prior_variance'])
        selected=full if calibrate else cut
        variance=gaussian(result['null_mean'],selected['latent_variance'])
        block_id=digest(dict(snapshot=meta['snapshot_sha256'],history=history,indices=js.tolist()))
        block_path=s['folder']/('block_'+block_id[:24]+'.npz')
        if not block_path.exists():_atomic_npz(block_path,cov_zz=zz,cov_xz=xz,query_indices=js,post_indices=np.asarray(zi,int))
        else:
            with np.load(block_path,allow_pickle=False) as z:
                if (not np.array_equal(z['cov_zz'],zz) or not np.array_equal(z['cov_xz'],xz)
                        or not np.array_equal(z['query_indices'],js) or not np.array_equal(z['post_indices'],zi)):
                    raise ValueError('Previously saved covariance block changed')
        receipt=dict(snapshot_sha256=meta['snapshot_sha256'],history_sha256=digest(history),post_birth_ids=[x['candidate_id'] for x in post],
            query_ids=[self.pool.ids[j] for j in js],file=str(block_path),file_sha256=filehash(block_path),
            cov_zz=array_receipt(zz),cov_xz=array_receipt(xz),columns=receipts,diagnostics=result['diagnostics'])
        receipt['receipt_sha256']=digest(receipt);_atomic_json(block_path.with_suffix('.json'),receipt)
        _atomic_json(progress_path,dict(snapshot_sha256=meta['snapshot_sha256'],history=history,history_sha256=digest(history),last_block_receipt_sha256=receipt['receipt_sha256']))
        if self.gp.counters['fits']!=before:raise RuntimeError('Birth prediction must never fit a GP')
        return dict(**selected,full=full,null=null,cut_feedback=cut,variance_only=variance,
            beta_mean=result['beta_mean'] if calibrate else meta['prior_mean'],
            beta_variance=result['beta_variance'] if calibrate else meta['prior_variance'],
            full_beta_mean=result['beta_mean'],full_beta_variance=result['beta_variance'],residual_feature=r,
            query_ids=result['query_ids'],snapshot_sha256=meta['snapshot_sha256'],snapshot_id=meta['snapshot_id'],
            calibrate=bool(calibrate),block_receipt=receipt,diagnostics=dict(result['diagnostics'],additional_GP_fits=0,
            numerical_source='fixed_birth_cached_GP',effect_cache=s['effect'].cache_info()))
