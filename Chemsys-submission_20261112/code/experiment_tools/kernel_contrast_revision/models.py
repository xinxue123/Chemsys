"""Deterministic, same-family kernel-sensitivity sidecars; no extra fitting."""
from pathlib import Path
import copy
import hashlib
import json
import math
import numpy as np
import torch
from filelock import FileLock
from kernel_contrast_qlogei import models as original
from . import PARAMETERS as P

digest,state_hash = original.digest,original.state_hash
parameters = original.parameters


def set_axis_correlation(model,pool,axis,target):
    changed = []
    for name,module in model.covar_module.named_modules():
        length = getattr(module,'lengthscale',None)
        if length is None:
            continue
        active = getattr(module,'active_dims',None)
        dimensions = list(range(len(pool.axes))) if active is None else list(map(int,active))
        if axis not in dimensions:
            continue
        index = dimensions.index(axis)
        if pool.kinds[axis] == 'categorical':
            value = -1./(len(dimensions)*math.log(target))
        else:
            value = P['correlation_probe']/math.sqrt(-2.*math.log(target))
        # Transform ONLY the focal entry. Round-tripping the entire public
        # lengthscale vector can turn untouched, saturated lower-bound entries
        # into -inf raw values (notably in the archived Perera category kernel).
        raw_value = module.raw_lengthscale_constraint.inverse_transform(
            torch.as_tensor(value,dtype=module.raw_lengthscale.dtype,device=module.raw_lengthscale.device))
        with torch.no_grad():
            module.raw_lengthscale[...,index].copy_(raw_value)
        if not torch.isfinite(module.raw_lengthscale).all():
            raise ValueError('Nonfinite raw focal lengthscale')
        changed.append(dict(module=name,local_axis=index,lengthscale=value))
    if not changed:
        raise ValueError('No native lengthscale component found for axis '+str(axis))
    return changed


def library(gp,history,context,directory):
    directory = Path(directory); directory.mkdir(parents=True,exist_ok=True)
    native = context['stats']['model']; before = state_hash(native)
    x,y,mean,std,ordered = original.data(gp,history)
    recipe = dict(native_state_sha256=before,
        prefix=[dict(candidate_id=h['candidate_id'],y=h['y']) for h in ordered],
        input_sha256=hashlib.sha256(gp.pool.source_path.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),parameters=P,
        torch=torch.__version__)
    key = digest(recipe); folder = directory/key
    with FileLock(str(directory/(key+'.lock')),timeout=900):
        if (folder/'receipt.json').exists():
            receipt = json.loads((folder/'receipt.json').read_text(encoding='utf-8'))
            if receipt['recipe_sha256'] != key:
                raise ValueError('Kernel hypothesis library changed')
            with np.load(folder/'predictions.npz',allow_pickle=False) as arrays:
                fits = receipt['models']
                for i,item in enumerate(fits):
                    item.update(mu=arrays['mu'][i],sd=arrays['sd'][i],observation_sd=arrays['observation_sd'][i])
        else:
            folder.mkdir(parents=True,exist_ok=True); fits = []
            specifications = [('native',None,None)]
            for axis in range(len(gp.pool.axes)):
                specifications += [(f'axis_{axis}_localized',axis,P['correlation_targets'][0]),
                    (f'axis_{axis}_transfer',axis,P['correlation_targets'][1])]
            for expert,axis,target in specifications:
                model = original.factory(gp.pool,x,y)
                model.load_state_dict(copy.deepcopy(native.state_dict()))
                changes = set_axis_correlation(model,gp.pool,axis,target) if axis is not None else []
                model.eval(); model.likelihood.eval()
                np.testing.assert_allclose(model.train_targets.detach().numpy(),native.train_targets.detach().numpy(),rtol=0,atol=1e-12)
                checkpoint = folder/(expert+'.pt')
                torch.save(dict(recipe_sha256=key,expert=expert,state=model.state_dict(),
                    native_state_sha256=before,focus_axis=axis,component_target=target,
                    training_ids=[h['candidate_id'] for h in ordered],training_y=[h['y'] for h in ordered]),checkpoint)
                item = original.describe(model,gp,mean,std,'controlled_kernel_sensitivity',None,checkpoint,[])
                item.update(id=expert,focus_axes=[] if axis is None else [axis],
                    component_target=target,changed_components=changes,native_state_sha256=before,
                    forecast_sha256=digest(dict(parameters=item['parameters'],mean=mean,std=std)))
                fits.append(item)
            temp = folder/'predictions.partial'
            with temp.open('wb') as f:
                np.savez_compressed(f,**{k:np.stack([m[k] for m in fits]) for k in ('mu','sd','observation_sd')})
            temp.replace(folder/'predictions.npz')
            metadata = [{k:v for k,v in m.items() if k not in ('mu','sd','observation_sd')} for m in fits]
            (folder/'receipt.json').write_text(json.dumps(dict(recipe_sha256=key,recipe=recipe,models=metadata),indent=2)+'\n',encoding='utf-8')
    np.testing.assert_allclose(fits[0]['mu'],context['stats']['mu_y'],rtol=0,atol=1e-12)
    np.testing.assert_allclose(fits[0]['sd'],context['stats']['sd_y'],rtol=0,atol=1e-12)
    assert state_hash(native) == before
    return fits
