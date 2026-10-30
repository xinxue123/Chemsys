from pathlib import Path
import json
import sys
import numpy as np
from chem.benchmarks import mixed_comparison_suzuki_report as r

r.s.worker_init()
for case in (1,2,3,4):
    pool=r.s.inputs(case);ix={v:i for i,v in enumerate(pool['ids'])}
    values=np.load(r.s.original.ROOT/'inputs'/f'case{case}'/'oracle.npz')['values']
    cache=r.e.MixedRegionalCache(pool,r.s.ROOT/'audit_cache'/f'c{case}_s0',case)
    files=sorted(cache.directory.glob('*.npz'),key=lambda p:p.stat().st_mtime,reverse=True) if '--recent' in sys.argv else sorted(cache.directory.glob('*.npz'))
    for path in files:
        with np.load(path,allow_pickle=False) as data:
            if 'metadata' not in data:continue
            meta=json.loads(str(data['metadata']))
            stored={k:data[k].copy() for k in data.files if k!='metadata'}
        import torch
        saved=torch.load(cache.directory/(meta['fit_key']+'.pt'),map_location='cpu',weights_only=False)
        prefix=[{'candidate_id':key,'y':float(values[ix[key]])} for key in saved['train_candidate_ids']]
        _,value=cache.model(prefix);stats=cache.posterior_stats(prefix)
        shift=r.independent_region_shift(value['model'],meta,cache.z)[0] if 'regional_recipe' in meta else np.zeros(len(cache.ids))
        replay=r.kalgebra.marginal_scores(stats['mu_std'],stats['sd_std'],shift,value['best_std'],meta['mc_seed'],cache.z)
        err=float(np.max(np.abs(replay-stored['scores'])))
        if err<=1e-8:continue
        actual_shift=stored.get('shift_std',np.zeros(len(cache.ids)))
        exact=r.kalgebra.marginal_scores(stats['mu_std'],stats['sd_std'],actual_shift,value['best_std'],meta['mc_seed'],cache.z)
        j=int(np.argmax(np.abs(replay-stored['scores'])))
        numeric={'case':case,'prefix_length':len(prefix),'key':meta['key'],
                 'candidate_id':pool['ids'][j],'max_score_error':err,
                 'max_shift_error':float(np.max(np.abs(shift-actual_shift))),
                 'score_error_with_saved_shift':float(np.max(np.abs(exact-stored['scores']))),
                 'at_max_error':{'stored_logEI':float(stored['scores'][j]),'independent_logEI':float(replay[j]),
                                 'stored_shift':float(actual_shift[j]),'independent_shift':float(shift[j]),
                                 'base_mean_std':float(stats['mu_std'][j]),'base_sd_std':float(stats['sd_std'][j])}}
        print(json.dumps(numeric),flush=True);r.s.write(r.s.OUT/f'replay_diagnostic_case{case}.json',numeric)
        break
