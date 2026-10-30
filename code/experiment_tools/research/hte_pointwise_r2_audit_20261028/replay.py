"""Same-history one-step policy switches. Load frozen GPs; never fit or query."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from copy import deepcopy
import json
import sys
import time

import numpy as np
import torch

BASE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(BASE))
from chem.benchmarks import pointwise_hte_engine as e
from chem.benchmarks import pointwise_hte_study as study

OUT=Path(__file__).resolve().parent
ROOT=BASE/'data/runs/hte_pointwise_seed0'


def read(p):return json.loads(p.read_text(encoding='utf8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')


class ReplayCache(e.Cache):
    def __init__(self,pool):
        super().__init__(pool,OUT/'cache'/pool['task_id'])
        self.sources=[ROOT/'cache'/pool['task_id']]
    def ensure(self,name):
        super().ensure(name)
        if name.endswith('.pt') and not (self.directory/name).exists():
            raise RuntimeError('Missing frozen checkpoint; this audit forbids fitting')


def run_task(task):
    study.warm();pool=study.data.load_pool(task);ids=pool['ids'];lookup={k:i for i,k in enumerate(ids)}
    s=read(ROOT/'runs'/f'{task}_s0_New_R2'/'state.json')
    cache=ReplayCache(pool);rows=[]
    with np.load(BASE/'data/benchmarks/reasoning_bo'/task/'oracle.npz') as a:y=a['values'].copy()
    for d in s['decisions']:
        t=d['iteration'];prefix=s['history'][:t+3];folder=ROOT/'runs'/s['run_id']/'rounds'
        c=read(folder/f'{t:02d}_committed.json')['decision']
        with np.load(folder/f'{t:02d}_candidates.npz') as a:
            base=a['base_qLogEI'].copy();eligible=a['eligible'].copy();available=a['available'].copy()
            raw_record=a['raw_shift_std'].copy();mu_record=a['mu_y'].copy();sd_record=a['sd_y'].copy()
        _,v=cache.model(prefix)
        stats=cache.stats(prefix)
        np.testing.assert_allclose(mu_record,stats['mu_y'],atol=1e-10,rtol=0)
        np.testing.assert_allclose(sd_record,stats['sd_y'],atol=1e-10,rtol=0)
        key=c['base_meta']['fit_key'];checkpoint=torch.load(cache.directory/(key+'.pt'),map_location='cpu',weights_only=False)
        assert checkpoint['train_candidate_ids']==sorted(h['candidate_id'] for h in prefix)
        before_alpha=c['prior']['amplitude'] if c['prior'] else 1.
        full=e.nominal.region(pool,d['knowledge'],[h['id'] for h in d['knowledge']['clauses']])
        retained=c['region']
        variants={'base':(None,0.),'full_region_alpha1':(full,1.),'B4_region_alpha1':(retained,1.),
            'B4_region_KF_before_fallback':(retained,before_alpha),
            'B4_fallback_alpha1':(retained,0. if d['fallback'] else 1.),
            'actual_R2':(retained,d['amplitude'])}
        out={}
        for name,(spec,alpha) in variants.items():
            scores,shift,_=cache.score(prefix,spec,alpha)
            j=e.nominal.argmax(scores,eligible,ids)
            out[name]={'selected':ids[j],'yield_percent':float(y[j]*100),'base_loss':float(base[c['reference']]-base[j]),
                'shift_pp':float(shift[j]*v['std']*100),'guided_qLogEI':float(scores[j])}
        assert out['actual_R2']['selected']==d['selected_id']
        # Same-history pre-refresh versus post-refresh with alpha held at1.
        refresh=next((x for x in s['calls'] if x['iteration']==t),None)
        if refresh:
            previous=s['initial_knowledge']
            for event in s['calls']:
                if event['iteration']>=t:break
                previous=event['validated']['plan']['knowledge']
            previous_full=e.nominal.region(pool,previous,[h['id'] for h in previous['clauses']])
            sc,sh,_=cache.score(prefix,previous_full,1.);j=e.nominal.argmax(sc,eligible,ids)
            out['pre_refresh_full_region_alpha1']={'selected':ids[j],'yield_percent':float(y[j]*100),
                'base_loss':float(base[c['reference']]-base[j]),'shift_pp':float(sh[j]*v['std']*100),'guided_qLogEI':float(sc[j])}
            # Recreate the old actual R2 choice using the saved before-refresh decision.
            pre=refresh['before'];sc,sh,_=cache.score(prefix,pre['region'],pre['amplitude']);j=e.nominal.argmax(sc,eligible,ids)
            assert j==pre['selected']
            out['pre_refresh_actual_R2']={'selected':ids[j],'yield_percent':float(y[j]*100),
                'base_loss':float(base[c['reference']]-base[j]),'shift_pp':float(sh[j]*v['std']*100),'guided_qLogEI':float(sc[j])}
        # A posterior covariance-to-mean shift cannot change uncertainty; independently check it.
        if retained is not None and t in (1,4,8,10,16):
            tilt,_,_=cache.tilt(prefix,retained);jj=np.array([lookup[d['selected_id']],c['reference']])
            with torch.no_grad():
                joint=v['model'].posterior(torch.cat([cache.X[jj],tilt.Xg])).mvn.covariance_matrix
                expected=tilt._lam*(joint[:2,2:]@tilt._a)
            np.testing.assert_allclose(expected.numpy(),raw_record[jj],atol=1e-10,rtol=0)
        rows.append({'query':t,'actual_y_percent':d['y']*100,'alpha_before_fallback':before_alpha,
            'variants':out,'incumbent_before_percent':d['incumbent_before']*100})
        cache.release()
    assert cache.counters['fits']==0
    result={'task':task,'rows':rows,'fits':0,'model_loads':cache.counters['model_loads'],
        'scope':'Variants reuse the ACTUAL R2 observation prefix. No switch is propagated into later history. Full-region no-KF variants use CURRENT refreshed knowledge, not a new Gate_only trajectory.'}
    save(OUT/(task+'_replay.json'),result);return {'task':task,'rounds':len(rows),'fits':0}


if __name__=='__main__':
    import os
    os.chdir(BASE)
    with ProcessPoolExecutor(max_workers=2) as ex:
        for result in ex.map(run_task,('Perera_Suzuki','Doyle_Buchwald_Hartwig')):print(json.dumps(result),flush=True)
