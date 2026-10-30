"""Audit existing evidence only. No fits, optimization, API calls or repairs."""
from pathlib import Path
from hashlib import sha256
from collections import Counter
import json
import math
import numpy as np
from scipy.stats import t

BASE=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
RUN=BASE/'data/runs/reasoning_bo_comparison'
RESULT=BASE/'research/reasoning_bo_comparison'


def read(path):return json.loads(path.read_text(encoding='utf8'))
def hashed(path):return sha256(path.read_bytes()).hexdigest()


def main():
    states=[read(p) for p in (RUN/'runs').glob('*/state.json')]
    result={'scope':'Read existing140 trajectories and source rows; no new optimization/LLM/repair. Existing-prefix windows are not reruns with shorter stopping budgets.',
        'planned_and_complete':len(states),'batches':sum(len(s['decisions']) for s in states),'datasets':{},'pairs':[],
        'historical_errors':[s['run_id'] for s in states if s.get('error')],
        'regional_scale_probe':{'run':'Perera_Suzuki_s00_New_R2','batch':1,'n':3,
            'outcome_variance_scale':1.4999999999999998,
            'recorded_shift_pp':[.0009072753679020901,3.6071728592415733,.0009072753679020901],
            'direct_posterior_covariance_shift_pp':[.0013609130518531562,5.4107592888623595,.0013609130518531562],
            'ratio':2/3,'method':'Separate read-only existing-checkpoint joint posterior probe; see audit.md. Not a counterfactual trajectory.'}}
    for task in ('Perera_Suzuki','Doyle_Buchwald_Hartwig'):
        values=np.load(BASE/'data/benchmarks/reasoning_bo'/task/'oracle.npz')['values']*100
        data={'N':len(values),'quantiles_percent':np.percentile(values,[0,25,50,75,90,95,99,100]).tolist(),
            'quantile_positions':[0,25,50,75,90,95,99,100],'thresholds':{},'methods':{}}
        for threshold in (90,95,98,99):
            N=len(values);H=int((values>=threshold).sum())
            data['thresholds'][threshold]={'count':H,'fraction':H/N,
                'uniform_draw93_at_least_one_probability':1-math.prod((N-H-i)/(N-i) for i in range(93))}
        selected=[s for s in states if s['task']==task]
        for method in sorted({s['method'] for s in selected}):
            runs=sorted([s for s in selected if s['method']==method],key=lambda s:s['seed'])
            curve=np.array([[max(h['y'] for h in s['history'][:3])*100]+[d['incumbent_after']*100 for d in s['decisions']] for s in runs])
            decisions=[d for s in runs for d in s['decisions']]
            data['methods'][method]={'n':len(runs),'initial_best_mean':float(curve[:,0].mean()),
                'prefix_SearchAUC':{n:float(curve[:,1:n+1].mean()) for n in (1,3,5,9,10,15,30)},
                'best_at_batch':{n:float(curve[:,n].mean()) for n in (0,1,3,5,9,10,15,30)},
                'final_SD':float(curve[:,-1].std(ddof=1)),
                'eligible_counts':dict(Counter(d['eligible_batches'] for d in decisions)),
                'nonbase_choices':sum(d['selected_batch_index']!=d['base_best_batch_index'] for d in decisions),
                'fallbacks':sum(d['fallback'] for d in decisions)}
        for left,right in (('ReasoningBO','BO_q3_Sobol'),('ReasoningBO','BO_q3_shared'),
                ('ReasoningBO','BO_shared'),('New_R2','Gate_only'),('LGBO_region','Gate_only')):
            a=sorted([s for s in selected if s['method']==left],key=lambda s:s['seed'])
            b=sorted([s for s in selected if s['method']==right],key=lambda s:s['seed'])
            diff=np.array([np.mean([d['incumbent_after']*100 for d in x['decisions']])-
                np.mean([d['incumbent_after']*100 for d in y['decisions']]) for x,y in zip(a,b)])
            hw=t.ppf(.975,9)*diff.std(ddof=1)/np.sqrt(10)
            same_init=all(x['history'][:3]==y['history'][:3] for x,y in zip(a,b))
            result['pairs'].append({'task':task,'left':left,'right':right,'common_initial':same_init,
                'mean_delta':float(diff.mean()),'SD_delta':float(diff.std(ddof=1)),
                'exploratory_unadjusted_t95_interval':[float(diff.mean()-hw),float(diff.mean()+hw)] if same_init else None,
                'same_whole_trajectories':sum(all(set(dx['selected_ids'])==set(dy['selected_ids'])
                    for dx,dy in zip(x['decisions'],y['decisions'])) for x,y in zip(a,b)),
                'interval_limit':'Only common-init pairs; exploratory10-replicate t interval, no multiplicity adjustment or power claim.'})
        result['datasets'][task]=data
    # All previously sealed source/evidence files and GP caches must stay intact.
    receipt=read(RESULT/'release_receipt.json');inventory=read(RESULT/'release_inventory.json')
    assert hashed(RESULT/'release_inventory.json')==receipt['inventory_sha256']
    for row in inventory['files']:assert hashed(BASE/row['path'])==row['sha256'],row['path']
    folder=RUN/'cache'
    rows=[{'path':p.relative_to(folder).as_posix(),'bytes':p.stat().st_size,'sha256':hashed(p)}
        for p in sorted(folder.rglob('*')) if p.is_file() and p.suffix not in ('.lock','.tmp')]
    actual={'files':len(rows),'bytes':sum(r['bytes'] for r in rows),
        'sha256':sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode('utf8')).hexdigest()}
    assert actual==receipt['cache_tree']
    result['preservation']={'status':'PASS','sealed_files_verified':len(inventory['files']),
        'GP_cache_files_verified':len(rows),'original_sources_results_and_caches_modified':False}
    (OUT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'complete':len(states),'batches':result['batches'],'preservation':result['preservation']},indent=2))


if __name__=='__main__':main()
