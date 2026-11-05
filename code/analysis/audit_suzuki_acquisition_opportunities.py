"""Audit exact and near-tied acquisition sets using saved BO arrays only."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
NAME='suzuki_initial_yield_20261104'
BASE=ROOT/'experiments/runs'/NAME
OUT=ROOT/'experiments/reports'/NAME/'optimum_backtrace'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
p=read(BASE/'protocol.json')
rows=[]
for job in p['jobs']:
    if job['arm']=='mixed':
        continue
    folder=BASE/'runs'/job['id']
    for q in range(2,21):
        path=folder/'rounds'/f'{q:02}_GP.npz'
        commit=read(folder/'rounds'/f'{q:02}_commit.json')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==commit['GP_arrays_sha256']
        with np.load(path,allow_pickle=False) as z:
            value=z['base_logei'][z['available']]
            gap=value.max()-value
        rows.append(dict(case=job['case'],arm=job['arm'],seed=job['seed'],query=q,
            available=len(value),exact_top_ties=int((gap==0).sum()),
            diagnostic_1e12_top_set=int((gap<=1e-12).sum()),
            diagnostic_5percent_top_set=int((gap<=np.log(1.05)).sum()),
            acquisition_spread=float(gap.max())))
with (OUT/'acquisition_opportunities.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
    w.writeheader()
    w.writerows(rows)
assert len(rows)==760
assert all(r['diagnostic_1e12_top_set']==4691 for r in rows if r['query']==2)
summary=[]
for case in range(1,5):
    later=[r for r in rows if r['case']==case and r['query']>=3]
    first=[r for r in rows if r['case']==case and r['query']==2]
    summary.append(dict(case=case,subsequent_decisions=len(later),
        exact_tie_opportunities=sum(r['exact_top_ties']>1 for r in later),
        diagnostic_5percent_opportunities=sum(r['diagnostic_5percent_top_set']>1 for r in later),
        q2_spread_max=max(r['acquisition_spread'] for r in first)))
(OUT/'acquisition_opportunities.json').write_text(json.dumps(dict(status='PASS',rows=760,
    diagnostic_thresholds_only=True,new_GP_fits=0,new_experiment_queries=0,summary=summary),indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary))
