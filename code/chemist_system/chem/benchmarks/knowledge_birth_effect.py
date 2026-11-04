"""Frozen-birth one-feature Gaussian calibration; an unselected math module.

The base GP is conditioned on the birth observations OUTSIDE this module. Its
mean, marginal latent variances, observation noise and feature are then frozen.
Only unique post-birth observations condition beta~Normal(m0,P0). No GP, file,
network, optimizer, acquisition, or target lookup is used here.

Given A=C_b[Z,Z]+noise*I, this computes the exact Gaussian hierarchy:
 P=(P0^-1+phi_Z.T A^-1 phi_Z)^-1,
 m=P*(P0^-1*m0+phi_Z.T A^-1(y_Z-mu_bZ)),
 r_x=phi_x-C_b[x,Z] A^-1 phi_Z,
 expert_mean=null_mean+r_x*m, expert_var=null_var+r_x^2*P.
P0=0 uses the proper degenerate-prior limit, including a nonzero fixed m0.

No N-by-N matrix is constructed: snapshot vectors are O(N), covariance blocks
and their consistency registry are O(N*n), Cholesky is n-by-n with n<=16.
Each query's (n+1)-by-(n+1) LATENT joint covariance is checked before accepting
feedback, in batches of at most 256. Observation noise cannot repair an invalid
latent covariance. Eigensolver acceptance tolerance is 128*eps*||block||_inf;
this is a floating-point consistency check, not a covariance repair. No latent
matrix is modified, no positive eigenvalue is discarded and no jitter is added.
Full-prefix calls may repeat for cached predictions; IDs within a prefix cannot
repeat, birth IDs cannot enter feedback, and accepted feedback cannot be removed
or altered later. Previously seen covariance entries must remain bitwise equal.

The snapshot_id is provenance metadata, NOT proof that supplied covariance
blocks came from a particular GP fit. The caller must enforce frozen-fit/hash
custody. This module detects inconsistency of already seen entries, not every
incorrect block supplied for the first time. The feature is a joint-model input:
phi(x)=0 alone need not imply r_x=0 after other points are observed. Restricting
deployment to a scope is a separate policy, not an arbitrary truncation of this
Gaussian model.
"""
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite, sqrt
from threading import Lock, RLock
from types import MappingProxyType

import numpy as np
from scipy.linalg import solve_triangular

MAX_POST_BIRTH_OBSERVATIONS = 16
LATENT_PSD_BATCH_SIZE = 256
_EPS = np.finfo(float).eps
_WARM_LOCK = Lock()
_WARMED = False


def _vector(value, length, name):
    if np.iscomplexobj(value):
        raise ValueError(name+' must be real')
    result=np.asarray(value,dtype=float)
    if result.shape!=(length,) or not np.all(np.isfinite(result)):
        raise ValueError(name+f' must be a finite vector of length {length}')
    return np.ascontiguousarray(result)


def _matrix(value, shape, name):
    if np.iscomplexobj(value):
        raise ValueError(name+' must be real')
    result=np.asarray(value,dtype=float)
    if result.shape!=shape or not np.all(np.isfinite(result)):
        raise ValueError(name+f' must be a finite matrix with shape {shape}')
    return np.ascontiguousarray(result)


def _ids(values, name):
    result=tuple(values)
    if any(not isinstance(x,str) or not x for x in result) or len(set(result))!=len(result):
        raise ValueError(name+' must contain unique nonempty string IDs')
    return result


def _readonly(value):
    array=np.ascontiguousarray(value,dtype=float)
    # A bytes-backed array cannot be made writeable via setflags.
    return np.frombuffer(array.tobytes(),dtype=float).reshape(array.shape)


def _digest_parts(ids, *arrays):
    digest=sha256()
    for key in ids:
        data=key.encode('utf-8');digest.update(len(data).to_bytes(8,'big'));digest.update(data)
    for value in arrays:
        array=np.ascontiguousarray(value,dtype=float)
        digest.update(str(array.shape).encode('ascii'));digest.update(array.tobytes())
    return digest.hexdigest()


def _copy_result(result):
    return {key:value.copy() if isinstance(value,np.ndarray) else deepcopy(value)
            for key,value in result.items()}


def _latent_query_psd(zz,xz,variance):
    """Check query/observed latent blocks, independent of observation noise.

    Batched eigvalsh avoids a pseudoinverse/nullspace threshold that could
    discard a legitimate small positive latent eigenvalue. At most 256 small
    blocks are materialized; no covariance between query points is requested.
    Caller provenance remains responsible for the unprovided global covariance.
    """
    n=len(zz);count=len(variance)
    minimum=None;maximum_tolerance=0.
    for start in range(0,count,LATENT_PSD_BATCH_SIZE):
        end=min(count,start+LATENT_PSD_BATCH_SIZE)
        blocks=np.empty((end-start,n+1,n+1),dtype=float)
        blocks[:,:n,:n]=zz
        blocks[:,:n,n]=xz[start:end]
        blocks[:,n,:n]=xz[start:end]
        blocks[:,n,n]=variance[start:end]
        eigen_min=np.linalg.eigvalsh(blocks)[:,0]
        scale=np.max(np.sum(np.abs(blocks),axis=2),axis=1)
        tolerance=128*_EPS*np.maximum(scale,np.finfo(float).tiny)
        if not np.all(np.isfinite(eigen_min)) or not np.all(np.isfinite(tolerance)):
            raise ArithmeticError('Unrepresentable latent joint-covariance PSD check')
        invalid=eigen_min < -tolerance
        if np.any(invalid):
            j=int(np.flatnonzero(invalid)[0])
            raise ValueError('Query latent joint covariance is not positive semidefinite '
                             f'at query index {start+j}: min_eigenvalue={eigen_min[j]:.17g}, '
                             f'tolerance={tolerance[j]:.17g}; observation noise cannot repair it')
        smallest=float(np.min(eigen_min))
        minimum=smallest if minimum is None else min(minimum,smallest)
        maximum_tolerance=max(maximum_tolerance,float(np.max(tolerance)))
    return dict(latent_joint_psd_queries=count,minimum_query_latent_joint_eigenvalue=minimum,
                maximum_latent_joint_psd_tolerance=maximum_tolerance,
                latent_joint_psd_tolerance_rule='128*float64_eps*block_infinity_norm',
                latent_joint_psd_batch_size=LATENT_PSD_BATCH_SIZE,latent_covariance_repairs=0)


@dataclass(frozen=True)
class _Snapshot:
    point_ids: tuple
    birth_ids: frozenset
    index: object
    mean: np.ndarray
    latent_variance: np.ndarray
    feature: np.ndarray
    prior_mean: float
    prior_variance: float
    noise_variance: float
    snapshot_id: str
    vectors_sha256: str


def warm():
    """Once-per-process LAPACK warmup; no JIT or repeated operator compilation."""
    global _WARMED
    with _WARM_LOCK:
        if not _WARMED:
            factor=np.linalg.cholesky(np.array([[1.,.1],[.1,1.]]))
            solve_triangular(factor,np.ones(2),lower=True,check_finite=False)
            _WARMED=True
    return dict(warmed=True,operator_compiles=0)


class BirthEffect:
    """Immutable birth vectors with auditable, monotone post-birth conditioning.

    ``predict`` accepts the COMPLETE own post-birth prefix, not a list of new
    events to add again. Repeating that prefix does not duplicate evidence.
    Noise is a fixed scalar variance; covariance blocks are latent covariance.
    Returned variances distinguish latent function from a future observation.
    """
    __slots__=('_snapshot','_observations','_columns','_fits','_predictions','_stats','_lock')

    def __init__(self, point_ids, birth_ids, mean, latent_variance, feature,
                 prior_mean, prior_variance, noise_variance, snapshot_id):
        points=_ids(point_ids,'point_ids');birth=_ids(birth_ids,'birth_ids')
        if not points or not set(birth).issubset(points):
            raise ValueError('Birth IDs must belong to the nonempty snapshot point set')
        if not isinstance(snapshot_id,str) or not snapshot_id:
            raise ValueError('A nonempty frozen-GP snapshot_id is required')
        scalars=tuple(float(x) for x in (prior_mean,prior_variance,noise_variance))
        if not all(isfinite(x) for x in scalars) or min(scalars[1:])<0:
            raise ValueError('Prior mean must be finite and prior/noise variances nonnegative')
        n=len(points)
        mu=_vector(mean,n,'mean');var=_vector(latent_variance,n,'latent_variance')
        phi=_vector(feature,n,'feature')
        if np.any(var<0):raise ValueError('Latent variances cannot be negative')
        self._snapshot=_Snapshot(points,frozenset(birth),MappingProxyType({x:j for j,x in enumerate(points)}),
            _readonly(mu),_readonly(var),_readonly(phi),*scalars,snapshot_id,
            _digest_parts(points,mu,var,phi,np.array(scalars)))
        self._observations={}
        # At most 16 dense N-vectors; NaN is an unobserved-entry sentinel.
        self._columns={}
        self._fits=OrderedDict();self._predictions=OrderedDict()
        self._stats=dict(cholesky=0,fit_cache_hits=0,prediction_cache_hits=0,predictions=0)
        self._lock=RLock()
        warm()

    @property
    def snapshot_id(self):return self._snapshot.snapshot_id

    @property
    def vectors_sha256(self):return self._snapshot.vectors_sha256

    def cache_info(self):
        with self._lock:
            return dict(self._stats,fit_entries=len(self._fits),prediction_entries=len(self._predictions),
                post_birth_observations=len(self._observations),covariance_columns=len(self._columns),
                covariance_registry_bytes=sum(x.nbytes for x in self._columns.values()),
                operator_compiles=0,warmed=_WARMED)

    def predict(self,new_ids,observed_y,cov_zz,query_ids,cov_xz):
        """Condition and predict using fixed-birth covariance blocks.

        cov_zz: (n,n), cov_xz: (N_query,n), n<=16. Query IDs must be unique,
        unobserved and outside the birth set. Every accepted observation and
        previously exposed covariance entry must agree with earlier calls.
        No jitter, covariance refit or feature recoding is performed.
        """
        with self._lock:
            return self._predict(new_ids,observed_y,cov_zz,query_ids,cov_xz)

    def _predict(self,new_ids,observed_y,cov_zz,query_ids,cov_xz):
        s=self._snapshot
        ids=_ids(new_ids,'new_ids');queries=_ids(query_ids,'query_ids');n=len(ids)
        if n>MAX_POST_BIRTH_OBSERVATIONS:
            raise ValueError('At most 16 unique post-birth observations are supported')
        if (set(ids)|set(queries))-set(s.index):raise ValueError('Unknown snapshot point ID')
        if set(ids)&s.birth_ids:raise ValueError('Birth observations cannot calibrate a post-birth effect')
        if set(queries)&(set(ids)|s.birth_ids):raise ValueError('Queries must be unobserved, nonbirth points')
        if set(self._observations)-set(ids):raise ValueError('Previously accepted feedback cannot be removed')
        y=_vector(observed_y,n,'observed_y')
        zz=_matrix(cov_zz,(n,n),'cov_zz');xz=_matrix(cov_xz,(len(queries),n),'cov_xz')
        zi=np.asarray([s.index[key] for key in ids],dtype=int)
        xi=np.asarray([s.index[key] for key in queries],dtype=int)
        if not np.array_equal(zz,zz.T):raise ValueError('cov_zz must be symmetric without hidden averaging')
        if not np.array_equal(np.diag(zz),s.latent_variance[zi]):
            raise ValueError('cov_zz diagonal differs from the frozen birth marginal variance')
        for key,value in zip(ids,y):
            if key in self._observations and self._observations[key]!=value:
                raise ValueError('An accepted post-birth observation was changed')
        indices=np.concatenate((zi,xi));columns=np.vstack((zz,xz))
        for j,key in enumerate(ids):
            old=self._columns.get(key)
            if old is not None:
                known=~np.isnan(old[indices])
                if not np.array_equal(old[indices][known],columns[:,j][known]):
                    raise ValueError('Previously seen birth covariance entries were changed')
        fit_key=_digest_parts(ids,y,zz)
        prediction_key=_digest_parts((fit_key,)+queries,xz)
        if prediction_key in self._predictions:
            self._stats['prediction_cache_hits']+=1
            self._predictions.move_to_end(prediction_key)
            return _copy_result(self._predictions[prediction_key])
        # Validate the latent model itself, before adding noise to A. A large
        # observation variance must not hide an impossible cross covariance.
        latent_diagnostics=_latent_query_psd(zz,xz,s.latent_variance[xi])
        fit=self._fits.get(fit_key)
        new_fit=fit is None
        if fit is not None:
            self._stats['fit_cache_hits']+=1
        else:
            if n:
                eigen_min=float(np.linalg.eigvalsh(zz)[0])
                tolerance=128*_EPS*max(float(np.linalg.norm(zz,ord=np.inf)),np.finfo(float).tiny)
                if eigen_min < -tolerance:raise ValueError('Birth covariance block is not positive semidefinite')
                a=zz+s.noise_variance*np.eye(n)
                try:factor=np.linalg.cholesky(a)
                except np.linalg.LinAlgError as error:
                    raise ValueError('Observation covariance is not positive definite; no jitter is added') from error
                residual=y-s.mean[zi]
                wd=solve_triangular(factor,residual,lower=True,check_finite=False)
                wp=solve_triangular(factor,s.feature[zi],lower=True,check_finite=False)
                q=float(wp@wp);u=float(wp@wd)
                denom=1+s.prior_variance*q
                if not all(isfinite(x) for x in (q,u,denom)) or denom<=0:
                    raise ArithmeticError('Unrepresentable scalar posterior precision')
                variance=s.prior_variance/denom
                mean=s.prior_mean/denom+variance*u
                if not isfinite(mean) or not isfinite(variance) or not 0<=variance<=s.prior_variance:
                    raise ArithmeticError('Nonfinite or invalid coefficient posterior')
                fit=dict(factor=_readonly(factor),wd=_readonly(wd),wp=_readonly(wp),
                         beta_mean=mean,beta_variance=variance,minimum_birth_eigenvalue=eigen_min)
            else:
                fit=dict(factor=np.empty((0,0)),wd=np.empty(0),wp=np.empty(0),
                         beta_mean=s.prior_mean,beta_variance=s.prior_variance,minimum_birth_eigenvalue=None)
        if n:
            wx=solve_triangular(fit['factor'],xz.T,lower=True,check_finite=False)
            null_mean=s.mean[xi]+wx.T@fit['wd']
            reduction=np.einsum('ij,ij->j',wx,wx)
            raw_variance=s.latent_variance[xi]-reduction
            residual_feature=s.feature[xi]-wx.T@fit['wp']
        else:
            null_mean=s.mean[xi].copy();raw_variance=s.latent_variance[xi].copy()
            residual_feature=s.feature[xi].copy();reduction=np.zeros(len(queries))
        tolerance=128*_EPS*np.maximum(np.abs(s.latent_variance[xi])+reduction,np.finfo(float).tiny)
        if not np.all(np.isfinite(raw_variance)) or np.any(raw_variance < -tolerance):
            raise ValueError('Query covariance has a negative Schur-complement variance')
        # Only explicitly counted roundoff-scale negatives are projected to 0.
        repaired=raw_variance<0
        null_var=np.where(repaired,0.,raw_variance)
        beta_mean,beta_var=fit['beta_mean'],fit['beta_variance']
        expert_mean=null_mean+residual_feature*beta_mean
        expert_var=null_var+np.square(residual_feature*sqrt(beta_var))
        null_observation_var=null_var+s.noise_variance
        expert_observation_var=expert_var+s.noise_variance
        if not all(np.all(np.isfinite(value)) for value in
                   (null_mean,null_var,residual_feature,expert_mean,expert_var,null_observation_var,expert_observation_var)):
            raise ArithmeticError('Nonfinite Gaussian posterior')
        result=dict(beta_mean=float(beta_mean),beta_variance=float(beta_var),
            null_mean=_readonly(null_mean),null_latent_variance=_readonly(null_var),
            residual_feature=_readonly(residual_feature),expert_mean=_readonly(expert_mean),
            expert_latent_variance=_readonly(expert_var),
            null_observation_variance=_readonly(null_observation_var),
            expert_observation_variance=_readonly(expert_observation_var),
            query_ids=queries,new_ids=ids,birth_snapshot_id=s.snapshot_id,
            vectors_sha256=s.vectors_sha256,prefix_sha256=fit_key,
            diagnostics=dict(post_birth_observations=n,jitter_added=0.,
                **latent_diagnostics,
                minimum_birth_eigenvalue=fit['minimum_birth_eigenvalue'],
                variance_roundoff_corrections=int(np.count_nonzero(repaired)),
                maximum_variance_roundoff_correction=float(np.max(-raw_variance[repaired])) if np.any(repaired) else 0.,
                snapshot_authenticity='caller must pin original GP fit/hash; only seen-block consistency is checked'))
        # Commit consistency state only after every mathematical check succeeds.
        for j,key in enumerate(ids):
            if key not in self._columns:self._columns[key]=np.full(len(s.point_ids),np.nan)
            self._columns[key][indices]=columns[:,j]
            self._observations[key]=float(y[j])
        if new_fit:
            self._fits[fit_key]=fit
            self._stats['cholesky']+=int(n>0)
            if len(self._fits)>17:self._fits.popitem(last=False)
        self._predictions[prediction_key]=result
        if len(self._predictions)>4:self._predictions.popitem(last=False)
        self._stats['predictions']+=1
        return _copy_result(result)
