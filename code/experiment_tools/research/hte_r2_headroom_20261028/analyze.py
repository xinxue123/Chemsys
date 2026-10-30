"""Frozen corrected-R2 headroom: feature-only pools and same-history switches."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path
from hashlib import sha256
import json
import os
import sys
import numpy as np
import torch

BASE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(BASE))
from chem.benchmarks import pointwise_hte_engine as engine
from chem.benchmarks import pointwise_hte_study as parent

OUT=Path(__file__).resolve().parent
ROOT=BASE/'data/runs/hte_r2_region_trust_seed0_v2'
OLD=BASE/'data/runs/hte_pointwise_seed0'
TASKS=('Perera_Suzuki','Doyle_Buchwald_Hartwig')


def read(p):return json.loads(p.read_text(encoding='utf8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def hashed(p):return sha256(p.read_bytes()).hexdigest()
def order(v,mask,ids):return sorted(np.flatnonzero(mask),key=lambda j:(-float(v[j]),ids[j]))


class AuditCache(engine.Cache):
    def __init__(self,pool):
        super().__init__(pool,OUT/'cache'/pool['task_id'])
        self.sources=[ROOT/'cache'/pool['task_id'],OLD/'cache'/pool['task_id']]
    def ensure(self,name):
        super().ensure(name)
        if name.endswith('.pt') and not (self.directory/name).exists():raise RuntimeError('Audit forbids fitting')


def diverse(X,base,mask,ids,k=5,start=()):
    """No yields. Nominal Hamming distance, then base acquisition, then ID."""
    candidates=np.flatnonzero(mask);chosen=list(start)
    if not chosen:chosen=[order(base,mask,ids)[0]]
    while len(chosen)<min(k,len(candidates)):
        remaining=[j for j in candidates if j not in chosen]
        distance=np.min(np.mean(X[remaining,None,:]!=X[np.array(chosen)][None,:,:],axis=2),axis=1)
        selected=min(range(len(remaining)),key=lambda i:(-float(distance[i]),-float(base[remaining[i]]),ids[remaining[i]]))
        chosen.append(remaining[selected])
    return chosen


def run_task(task):
    parent.warm();pool=parent.data.load_pool(task);ids=pool['ids'];lookup={k:i for i,k in enumerate(ids)}
    state=read(ROOT/'runs'/f'{task}_s0_R2_region_trust'/'state.json');cache=AuditCache(pool)
    with np.load(BASE/'data/benchmarks/reasoning_bo'/task/'oracle.npz') as f:y=f['values'].copy()*100
    rows=[]
    for event in state['decisions']:
        t=event['iteration'];prefix=state['history'][:t+3];folder=ROOT/'runs'/state['run_id']/'rounds'
        commit=read(folder/f'{t:02d}_committed.json');c=commit['decision'];table=folder/f'{t:02d}_candidates.npz'
        assert commit['candidate_table_sha256']==hashed(table)
        with np.load(table) as f:a={k:f[k].copy() for k in f.files}
        base=a['base_qLogEI'];score=a['guided_qLogEI'];available=a['available'];eligible=a['eligible']
        ranking=order(base,available,ids);rank={j:i+1 for i,j in enumerate(ranking)}
        eligible_order=order(base,eligible,ids);j=lookup[event['selected_id']];ref=ranking[0]
        actual_std=float(np.std([h['y'] for h in prefix],ddof=0));mu=a['mu_y']*100;sd=a['sd_observation_y']*100
        raw=a['raw_shift_std']*actual_std*100;deployed=a['deployed_shift_std']*actual_std*100
        probe_alpha=c['prior']['amplitude'] if c['prior'] else 0.;err=float(y[j]-mu[j])
        gain_actual=err**2-(err-deployed[j])**2;gain_shadow=err**2-(err-probe_alpha*raw[j])**2
        z_deployed=(err-deployed[j])/np.sqrt(sd[j]**2+9.)
        row={'query':t,'selected_id':ids[j],'yield_percent':float(y[j]),'reference_id':ids[ref],
            'reference_y_percent':float(y[ref]),'selected_base_rank':rank[j],
            'selected_vs_reference_pp':float(y[j]-y[ref]),'incumbent_before_percent':event['incumbent_before']*100,
            'incumbent_after_percent':event['incumbent_after']*100,'amplitude':event['amplitude'],
            'trust_blocked':event['trust']['trust_blocked'],'trust_recovered':event['trust']['recovered'],
            'geometry':event['trust']['geometry_key'],'GP_mean_percent':float(mu[j]),'GP_observation_SD_pp':float(sd[j]),
            'knowledge_shift_pp':float(deployed[j]),'shadow_shift_pp':float(probe_alpha*raw[j]),
            'deployment_error_z':float(z_deployed),'KF_innovation_z':event['measurement']['innovation_z'] if event['measurement'] else None,
            'actual_predictive_error_gain_pp_squared':float(gain_actual),'shadow_predictive_error_gain_pp_squared':float(gain_shadow),
            'budget':event['budget'],'eligible_count':int(eligible.sum()),
            'eligible_oracle_best_percent':float(y[eligible].max()),
            'eligible_count_above95':int(((y>=95)&eligible).sum()),
            'near_best_count_gap1e_minus6':int(((base[ref]-base<=1e-6)&available).sum()),
            'variants':{},'candidate_pool_coverage':{},'clause_switches':{}}
        def record(index):
            return {'id':ids[index],'yield_percent':float(y[index]),'base_rank':rank[index],
                'base_loss':float(base[ref]-base[index]),
                'immediate_incumbent_gain_pp':float(max(y[index],row['incumbent_before_percent'])-row['incumbent_before_percent'])}
        row['variants']['actual']=record(j);row['variants']['base']=record(ref)
        pools={'top5_unrestricted':ranking[:5],'top5_budget':eligible_order[:5],
            'diverse5_budget':diverse(cache.X.numpy(),base,eligible,ids),
            'top3_diverse2_budget':diverse(cache.X.numpy(),base,eligible,ids,start=eligible_order[:3])}
        for name,indices in pools.items():
            row['candidate_pool_coverage'][name]={'ids':[ids[z] for z in indices],
                'actual_yields_percent':[float(y[z]) for z in indices],
                'oracle_best_percent':float(y[indices].max()),'contains95':bool((y[indices]>=95).any()),
                'mean_percent':float(y[indices].mean())}
        for k in (5,10):
            mask=np.zeros(len(ids),dtype=bool);mask[eligible_order[:k]]=True
            ix=engine.nominal.argmax(score,mask,ids);row['variants'][f'guided_top{k}_budget']=record(ix)
        row['variants']['rank_veto5']=record(ref if rank[j]>5 else j)
        alarm=(len(rows)>=2 and all(r['amplitude']>0 and r['actual_predictive_error_gain_pp_squared']<0
            and abs(r['deployment_error_z'])>2.5 for r in rows[-2:]))
        row['deployment_alarm_from_previous2']=bool(alarm)
        row['variants']['deployed_error_alarm']=record(ref if alarm else j)
        _,value=cache.model(prefix)
        for name,alpha in [('half_amplitude',event['amplitude']*.5),
            ('one_SD_discount',min(event['amplitude'],max(0.,c['prior']['mean'][1]-np.sqrt(c['prior']['covariance'][1][1]))) if c['prior'] else 0.)]:
            alt=base if alpha==0 else cache.score(prefix,c['region'],alpha)[0]
            ix=engine.nominal.argmax(alt,eligible,ids);row['variants'][name]={**record(ix),'amplitude':float(alpha)}
        if task.startswith('Doyle') and t in (2,3,4):
            keys=c['retained'];clauses=event['knowledge']['clauses']
            removals=[(h['id'],) for h in clauses]+[('H2','H3')]
            for removed in removals:
                keep=[h for h in keys if h not in removed];spec=engine.nominal.region(pool,event['knowledge'],keep)
                alt=cache.score(prefix,spec,event['amplitude'])[0]
                ix=engine.nominal.argmax(alt,eligible,ids)
                row['clause_switches']['remove_'+'_'.join(removed)]=record(ix)
        if row['trust_recovered']:
            ev=event['trust']['entry_after']['recovery_evidence']
            row['recovery_evidence']={'query':ev['query'],'squared_error_gain_pp_squared':ev['squared_error_gain_y_squared']*10000,
                'positive':ev['positive_evidence']}
        rows.append(row);cache.release()
    assert cache.counters['fits']==0
    result={'task':task,'rows':rows,'SearchAUC':float(np.mean([r['incumbent_after_percent'] for r in rows])),
        'final_best_percent':max(h['y'] for h in state['history'])*100,'global_oracle_max_percent':float(y.max()),
        'mean_queried_yield_percent':float(np.mean([r['yield_percent'] for r in rows])),
        'GP_fits':0,'new_experimental_queries':0,'new_API_calls':0,
        'scope':'Actual corrected-R2 prefixes. Alternative chosen values and pool coverage are POST-HOC, not new optimization trajectories. Diversity uses features/base only; oracle scores are evaluation only.'}
    save(OUT/(task+'.json'),result)
    return {'task':task,'rounds':len(rows),'GP_fits':0}


def preserved():
    evidence=[]
    for folder in ('hte_pointwise_seed0','hte_r2_region_trust_seed0'):
        path=BASE/'research'/folder;inv=read(path/'release_inventory.json');receipt=read(path/'release_receipt.json')
        assert hashed(path/'release_inventory.json')==receipt['inventory_sha256']
        for row in inv['files']:assert hashed(BASE/row['path'])==row['sha256'],row['path']
        evidence.append({'folder':folder,'files':len(inv['files']),'status':'UNCHANGED'})
    return evidence


def main():
    before=preserved()
    with ProcessPoolExecutor(max_workers=2) as ex:
        for result in ex.map(run_task,TASKS):print(json.dumps(result),flush=True)
    after=preserved();assert before==after
    save(OUT/'receipt.json',{'status':'PASS','preservation':after,'new_GP_fits':0,'new_queries':0,'new_API_calls':0,
        'scope':'32 same-prefix diagnostics only; no method or scientific result modified.'})


if __name__=='__main__':
    os.chdir(BASE);main()
