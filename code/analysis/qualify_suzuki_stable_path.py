"""Qualify saved Case 1 transitions under the requested stable-branch rule."""
from pathlib import Path
import csv
import hashlib
import json
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from chem.benchmarks import knowledge_gp
from cold_start_v13.gp import ColdGP

ORIGINAL=ROOT/'experiments/runs/suzuki_initial_yield_20261104'
BASE=ROOT/'experiments/runs/suzuki_case1_stable_path_20261104'
OUT=ROOT/'experiments/reports/suzuki_case1_recursive_paths_20261104'
MARGIN=1e-10
OFFSETS=tuple(range(5))
read=lambda path:json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import torch
    from botorch.acquisition.logei import qLogExpectedImprovement
    from botorch.sampling.normal import SobolQMCNormalSampler
    from chem.benchmarks.mixed_kalman_suzuki_engine import MixedResidualMeanModel
    pool=knowledge_gp.load_task(ROOT,'Reizman_Suzuki_Case_1',seed=0)
    route=ORIGINAL/'runs/c1_s4_middle'
    state=read(route/'state.json')
    BASE.mkdir(parents=True,exist_ok=True)
    OUT.mkdir(parents=True,exist_ok=True)
    registration=dict(status='REGISTERED',source_sha256=sha(__file__),
        design_sha256=sha(ROOT/'docs/suzuki_case1_stable_path_20261104.txt'),
        conditional_on_saved_GP=True,minimum_unique_margin=MARGIN,sampler_offsets=list(OFFSETS),
        route_state_sha256=sha(route/'state.json'),new_experiments=0,new_GP_fits=0)
    reg=BASE/'registration.json'
    if reg.exists():
        assert read(reg)==registration
    else:
        reg.write_text(json.dumps(registration,indent=2)+'\n',encoding='utf-8')
    gp=ColdGP(pool,BASE/'cache',seed=0)
    gp.cache.sources=[ORIGINAL/'cache/case1']
    rows=[]
    for query in range(2,7):
        prefix=[dict(candidate_id=o['candidate_id'],y=o['y']) for o in state['history'][:query-1]]
        expected=state['history'][query-1]['candidate_id']
        fit,value=gp.cache.model(prefix)
        native,meta=gp.cache.score(prefix)
        saved=route/'rounds'/f'{query:02}_GP.npz'
        with np.load(saved,allow_pickle=False) as z:
            np.testing.assert_array_equal(native,z['base_logei'])
        available=~np.isin(pool.ids,[h['candidate_id'] for h in prefix])
        model=MixedResidualMeanModel(value['model'],[],{},(),value['std'],gp.cache.category_map).eval()
        for offset in OFFSETS:
            acq=qLogExpectedImprovement(model=model,best_f=value['best_std'],
                sampler=SobolQMCNormalSampler(torch.Size([512]),seed=meta['mc_seed']+offset))
            with torch.no_grad():
                scores=torch.cat([acq(x.unsqueeze(1)).reshape(-1) for x in gp.cache.z.split(256)]).numpy()
            if offset==0:
                np.testing.assert_array_equal(scores,native)
            js=np.flatnonzero(available)
            j=pool.ix[expected]
            others=js[js!=j]
            margin=float(scores[j]-scores[others].max())
            ties=int(np.count_nonzero(available&(scores==scores[js].max())))
            rows.append(dict(query=query,sampler_offset=offset,sampler_seed=meta['mc_seed']+offset,
                intended_id=expected,intended_yield=state['history'][query-1]['raw_value'],
                rank=1+int(np.count_nonzero(available&(scores>scores[j]))),
                top_score_ties=ties,margin=margin,qualified=margin>MARGIN,
                saved_fit_key=fit))
        gp.release()
    assert gp.counters['fits']==0 and gp.counters['operator_compiles_in_loop']==0
    with (OUT/'recorded_route_stability.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        w.writeheader();w.writerows(rows)
    transitions=[]
    for q in range(2,7):
        rs=[r for r in rows if r['query']==q]
        transitions.append(dict(query=q,intended_id=rs[0]['intended_id'],intended_yield=rs[0]['intended_yield'],
            passes=sum(r['qualified'] for r in rs),evaluations=5,
            minimum_margin=min(r['margin'] for r in rs),all_qualified=all(r['qualified'] for r in rs)))
    result=dict(status='PASS',criterion=registration,transitions=transitions,
        new_GP_fits=gp.counters['fits'],new_experiments=0,operator_compiles_in_loop=0)
    (OUT/'recorded_route_stability.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':
    main()
