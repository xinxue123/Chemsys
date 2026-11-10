"""Bounded covariance-contrast preference fields and native-compatible logEI."""
from functools import lru_cache
import math
from pathlib import Path
import numpy as np
import torch
from filelock import FileLock
from botorch.acquisition.logei import _log_improvement
from botorch.utils.sampling import draw_sobol_normal_samples
from scipy.special import logsumexp
from . import PARAMETERS as P
from . import hypotheses


@lru_cache(maxsize=256)
def draws(seed):
    return draw_sobol_normal_samples(d=1,n=512,seed=int(seed),dtype=torch.double).reshape(512,1)


def logei(mu,sd,best,seed):
    mu=torch.as_tensor(mu,dtype=torch.double);sd=torch.as_tensor(sd,dtype=torch.double)
    z=draws(int(seed));out=[]
    with torch.no_grad():
        for start in range(0,len(mu),256):
            samples=(mu[start:start+256]+torch.sqrt(sd[start:start+256].square()+1e-6)*z).unsqueeze(-1)
            li=_log_improvement(samples,torch.as_tensor(float(best)),tau=1e-6,fat=True).squeeze(-1)
            out.append((torch.logsumexp(li,dim=0)-math.log(512)).numpy())
    return np.concatenate(out)


def quadrature(pool,note):
    plus,minus=hypotheses.masks(pool,note);selected=[];coefficients=[]
    seed=int(hypotheses.structural_key(note)[:12],16)
    for side,sign in ((plus,1.),(minus,-1.)):
        order=sorted(map(int,np.flatnonzero(side)),key=lambda j:pool.features[j])
        if len(order)>P['quadrature_per_side']:
            positions=np.random.default_rng(seed).choice(len(order),P['quadrature_per_side'],replace=False)
            order=[order[int(i)] for i in sorted(positions)]
        selected+=order;coefficients += [sign/len(order)]*len(order)
    return selected,np.asarray(coefficients)


def _preference_fields(gp,context,notes):
    n=len(gp.pool.ids);result={};metadata=[]
    if not notes:return result,metadata
    selections=[quadrature(gp.pool,note) for note in notes]
    union=sorted(set(j for indices,_ in selections for j in indices),key=lambda j:gp.pool.features[j])
    position={j:i for i,j in enumerate(union)};A=np.zeros((len(union),len(notes)))
    for h,(indices,weights) in enumerate(selections):
        for j,w in zip(indices,weights):A[position[j],h]+=w
    cross=np.empty((n,len(notes)));variance=None
    model=context['stats']['model'];outer_scale=context['stats']['std']
    with torch.no_grad():
        for start in range(0,n,128):
            xx=gp.X[start:start+128];joint=torch.cat([xx,gp.X[union]])
            covariance=model.posterior(joint).mvn.covariance_matrix.detach().numpy();m=len(xx)
            cross[start:start+m]=covariance[:m,m:]@A
            if variance is None:variance=np.einsum('ah,ab,bh->h',A,covariance[m:,m:],A)
    sd=context['stats']['sd_y'];bound=P['preference_shift_SD']*sd
    for k,note in enumerate(notes):
        v=max(0.,float(variance[k]));valid=v>1e-12
        field=P['preference_shift_SD']*cross[:,k]/math.sqrt(v)*outer_scale if valid else np.zeros(n)
        violation=float(np.max(np.maximum(np.abs(field)-bound,0.)))
        if violation>1e-7*max(1.,float(sd.max())):raise ValueError('Contrast shift violates its native uncertainty bound')
        field=np.clip(field,-bound,bound)
        result[note['id']]=field
        metadata.append(dict(id=note['id'],contrast_variance_native=v,active=bool(valid),
            quadrature_ids=[gp.pool.ids[j] for j in selections[k][0]],
            quadrature_coefficients=selections[k][1].tolist(),maximum_bound_error=violation,
            max_abs_shift_y=float(np.abs(field).max())))
    return result,metadata


def preference_fields(gp,context,notes):
    """Reuse identical prefix/structure calculations across the paired arms."""
    if not notes:return {},[]
    from kernel_contrast_qlogei.study import sha,read,write
    from kernel_contrast_revision.models import state_hash
    key=hypotheses.digest(dict(task=gp.pool.task,fit=context['stats']['fit_key'],
        native_state=state_hash(context['stats']['model']),outer_scale=context['stats']['std'],
        structures=[hypotheses.structural_key(h) for h in notes],parameters=P,
        source_sha256=sha(__file__)))
    folder=gp.directory/'preference_fields';folder.mkdir(parents=True,exist_ok=True)
    array_path=folder/(key+'.npz');meta_path=folder/(key+'.json')
    with FileLock(str(folder/(key+'.lock')),timeout=900):
        if meta_path.exists():
            meta=read(meta_path);assert meta['array_sha256']==sha(array_path)
            with np.load(array_path,allow_pickle=False) as saved:fields={h['id']:saved[h['id']] for h in notes}
            return fields,meta['metadata']
        fields,metadata=_preference_fields(gp,context,notes)
        temporary=array_path.with_suffix('.npz.tmp')
        with temporary.open('wb') as f:np.savez_compressed(f,**fields)
        temporary.replace(array_path)
        write(meta_path,dict(key=key,array_sha256=sha(array_path),metadata=metadata))
        return fields,metadata


def evaluate(gp,context,notes,log_weights):
    stats=context['stats'];mean,std=stats['mean'],stats['std']
    mu=(stats['mu_y']-mean)/std;sd=stats['sd_y']/std
    seed=context['acquisition_meta']['mc_seed'];best=stats['best_std']
    replay=logei(mu,sd,best,seed);error=float(np.max(np.abs(replay-context['base_logei'])))
    if error>2e-8:raise ValueError('Native logEI reconstruction mismatch '+str(error))
    fields,metadata=preference_fields(gp,context,notes)
    ids=['base']+[h['id'] for h in notes]
    scores=[context['base_logei']];mus=[stats['mu_y']]
    for note in notes:
        shift=fields[note['id']];mus.append(stats['mu_y']+shift)
        scores.append(logei(mu+shift/std,sd,best,seed))
    logw=np.asarray([log_weights[i] for i in ids]);logw-=logsumexp(logw)
    matrix=np.stack(scores)
    mixture=logsumexp(matrix+logw[:,None],axis=0)
    if not notes or all(np.max(np.abs(f))==0 for f in fields.values()):mixture=context['base_logei'].copy()
    return dict(ids=ids,mu=np.stack(mus),scores=matrix,mixture=mixture,fields=fields,
        log_weights=dict(zip(ids,logw.tolist())),field_metadata=metadata,native_replay_error=error)
