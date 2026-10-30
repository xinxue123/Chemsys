"""Audit recorded counterfactual outcomes and selected native score controls."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import numpy as np
from chem.benchmarks import mixed_comparison_suzuki_study as s
from chem.benchmarks import mixed_comparison_suzuki_engine as e
from chem.benchmarks import kalman_knowledge_suzuki_engine as p

OUT=Path('research/suzuki_mixed_gate_diagnostic')
manifest=s.read(OUT/'input_manifest.json')
for f,h in manifest.items():assert s.filehash(f)==h,f
counts=0;native=[]
s.worker_init()
for case in (2,4):
    d=s.read(OUT/f'case{case}.json');pool=s.inputs(case);ids=pool['ids'];ix={key:i for i,key in enumerate(ids)}
    y=np.load(s.original.ROOT/'inputs'/f'case{case}'/'oracle.npz')['values']
    states={'new':s.read(s.previous.ROOT/'runs'/f'c{case}_s0'/'state.json'),
            'gate':s.read(s.ROOT/'runs'/f'c{case}_s0_assistant_initial_LGBO_gate'/'state.json')}
    actual={(r['trajectory'],r['round']):r for r in d['trajectory']}
    for r in d['counterfactuals']:
        state=states[r['trajectory']];prefix=state['history'][:r['round']+3];inc=max(v['y'] for v in prefix)
        j=ix[r['candidate_id']];assert r['candidate_id'] not in {v['candidate_id'] for v in prefix}
        assert abs(r['yield_pp']-y[j]*100)<1e-10
        assert abs(r['immediate_incumbent_pp']-max(inc,y[j])*100)<1e-10
        ar=actual[(r['trajectory'],r['round'])]
        assert abs(r['actual_minus_variant_incumbent_pp']-(ar['incumbent_after_pp']-max(inc,y[j])*100))<1e-10
        if r['variant']=='actual':assert r['candidate_id']==ar['candidate_id']
        counts+=1
    assert abs(sum(d['round_incumbent_gaps_pp'])/16-d['gate_minus_new_AUC_pp'])<1e-12
    cache=e.MixedRegionalCache(pool,OUT/'validation_cache'/f'case{case}',case,sources=[OUT/'cache'/f'c{case}'])
    samples=[(8,'no_LLM_refresh_this_round'),(9,'no_B4_hold_trust')] if case==2 else [
        (3,'frozen_trust_hold_B4'),(4,'frozen_trust_hold_B4'),(8,'no_LLM_refresh_this_round'),(15,'no_B4_hold_trust')]
    state=states['new']
    for t,name in samples:
        r=next(x for x in d['counterfactuals'] if x['trajectory']=='new' and x['round']==t and x['variant']==name)
        ev=state['decisions'][t-1];history=state['history'][:t+3];hs=ev['hypotheses'];w=ev['trust']
        if name=='frozen_trust_hold_B4':w={h['id']:float(1-np.sqrt(.2525)) for h in hs}
        if name=='no_LLM_refresh_this_round':
            hs=ev['hypotheses_before_call'];w=p.forecast(ev['filter_before_call'],hs)['weights']
        base,_=cache.score(history)
        score=cache.knowledge_score(history,hs,w,r['retained_ids'])[0] if r['retained_ids'] else base
        av=~np.isin(ids,[x['candidate_id'] for x in history]);budget=p.common.budget_at(4,t)
        eligible=av&(base[av].max()-base<=budget)
        j=p.common.argmax(score,eligible,ids)
        assert ids[j]==r['candidate_id'],(case,t,name,ids[j],r['candidate_id'])
        native.append({'case':case,'round':t,'variant':name,'candidate_id':ids[j]})
    assert cache.counters['fits']==0
for f,h in manifest.items():assert s.filehash(f)==h,f
receipt={'status':'PASS','outcomes_and_immediate_effects_checked':counts,'native_GP_wrapper_controls':native,
         'all_source_hashes_unchanged':True,'new_GP_fits':0,'LLM_calls':0}
s.write(OUT/'verification_details.json',receipt);print(receipt)
