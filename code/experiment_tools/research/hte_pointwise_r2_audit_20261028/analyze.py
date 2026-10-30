"""Read sealed traces; no new experimental trajectory, GP fit or API request."""
from pathlib import Path
from hashlib import sha256
from collections import Counter
import json
import numpy as np

OUT=Path(__file__).resolve().parent
BASE=OUT.parents[1]
ROOT=BASE/'data/runs/hte_pointwise_seed0'
OLD=BASE/'research/hte_pointwise_seed0'
TASKS=('Perera_Suzuki','Doyle_Buchwald_Hartwig')


def read(p):return json.loads(p.read_text(encoding='utf8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def hashed(p):return sha256(p.read_bytes()).hexdigest()
def argmax(v,mask,ids):return min(np.flatnonzero(mask),key=lambda j:(-float(v[j]),ids[j]))
def order(v,mask,ids):return sorted(np.flatnonzero(mask),key=lambda j:(-float(v[j]),ids[j]))


def point(pool,j):
    return {a:f'{a}_{pool["levels"][a].index(pool["points"][j][a])+1:02d}' for a in pool['axes']}


def member(pool,knowledge):
    return np.array([all(p[c['axis']] in c['allowed'] for c in knowledge['clauses']) for p in pool['points']])


def quality(pool,y,knowledge):
    mask=member(pool,knowledge)
    return {'clauses':knowledge['clauses'],'count':int(mask.sum()),'best_percent':float(y[mask].max()*100),
        'mean_percent':float(y[mask].mean()*100),'number_above90':int((y[mask]>=.90).sum()),
        'global_best_in_region':bool(mask[np.argmax(y)])}


def main():
    inv=read(OLD/'release_inventory.json');rc=read(OLD/'release_receipt.json')
    assert hashed(OLD/'release_inventory.json')==rc['inventory_sha256']
    for row in inv['files']:assert hashed(BASE/row['path'])==row['sha256'],row['path']
    before={r['path']:r['sha256'] for r in inv['files']}
    result={'scope':'Single-seed sealed4+16 traces. All oracle-based alternatives are AFTER-THE-FACT ONE-STEP comparisons at the same history, not new trajectories or causal module effects.',
        'old_protocol_sha256':rc['protocol_sha256'],'datasets':{},'checks':{'sealed_files_verified':len(inv['files'])}}
    all_states=[read(ROOT/'runs'/spec['run_id']/'state.json') for spec in read(ROOT/'manifest.json')]
    for task in TASKS:
        pool=read(BASE/'data/benchmarks/reasoning_bo'/task/'pool.json');ids=pool['ids'];lookup={k:i for i,k in enumerate(ids)}
        with np.load(BASE/'data/benchmarks/reasoning_bo'/task/'oracle.npz') as z:y=z['values'].copy()
        ss=[s for s in all_states if s['task']==task];ready=read(ROOT/'initialization'/task/'ready.json')
        ds={'global_best_percent':float(y.max()*100),'global_best_point':point(pool,int(np.argmax(y))),
            'initial_knowledge':quality(pool,y,ready['knowledge']),'initial_context':ready['request_context_sha256'],
            'methods':{},'initial_clause_quality':[]}
        for clause in ready['knowledge']['clauses']:
            ds['initial_clause_quality'].append({'id':clause['id'],'axis':clause['axis'],
                'allowed_ids':[f'{clause["axis"]}_{pool["levels"][clause["axis"]].index(v)+1:02d}' for v in clause['allowed']],
                **quality(pool,y,{'clauses':[clause]})})
        for s in ss:
            method=s['method'];curve=np.maximum.accumulate([h['y']*100 for h in s['history']])
            ms={'SearchAUC':float(curve[4:].mean()),'final_best_percent':float(curve[-1]),
                'first_final_best_query':int(np.flatnonzero(curve>=curve[-1]-1e-10)[0])+1-4,'rounds':[],'refreshes':[]}
            if method not in ('New_R2','ReasoningBO_point','Pure_BO','Gate_only','LGBO_region'):
                ds['methods'][method]=ms;continue
            for d in s['decisions']:
                t=d['iteration'];hist=s['history'][:t+3];folder=ROOT/'runs'/s['run_id']/'rounds'
                c=read(folder/f'{t:02d}_committed.json')['decision']
                with np.load(folder/f'{t:02d}_candidates.npz') as z:a={k:z[k].copy() for k in z.files}
                ix=lookup[d['selected_id']];ref=lookup[d['reference_id']];base_order=order(a['base_qLogEI'],a['available'],ids)
                rank=base_order.index(ix)+1;top=base_order[:5];std=float(np.std([h['y'] for h in hist],ddof=0))
                mask=member(pool,d['knowledge']);eligible=a['eligible'];m=d['measurement'];post=d.get('calibration_after')
                r={'query':t,'selected':ids[ix],'reference':ids[ref],'selected_point':point(pool,ix),'reference_point':point(pool,ref),
                    'yield_percent':float(y[ix]*100),'reference_yield_percent':float(y[ref]*100),
                    'one_step_delta_vs_reference_pp':float((y[ix]-y[ref])*100),
                    'incumbent_percent':d['incumbent_after']*100,'gain_pp':d['incumbent_gain']*100,
                    'base_rank':rank,'loss':d['base_opportunity_loss'],'budget':d['budget'],'amplitude':d['amplitude'],
                    'fallback':d['fallback'],'conflict':d['diagnosis']['conflict_kind'],'revoked':d['diagnosis']['revoked_ids'],
                    'retained':d['retained'],'eligible_count':int(eligible.sum()),
                    'available_count':int(a['available'].sum()),'full_knowledge_witness_in_budget':int((mask&eligible).sum()),
                    'base_mu_percent':float(a['mu_y'][ix]*100),'latent_sd_pp':float(a['sd_y'][ix]*100),
                    'observation_sd_pp':float(a['sd_observation_y'][ix]*100),
                    'raw_shift_pp':float(a['raw_shift_std'][ix]*std*100),
                    'deployed_shift_pp':float(a['deployed_shift_std'][ix]*std*100),
                    'deployed_mean_percent':float((a['mu_y'][ix]+a['deployed_shift_std'][ix]*std)*100),
                    'innovation_z':None if not m else m['innovation_z'],
                    'KF_mean_before':None if not c['prior'] else c['prior']['mean'],
                    'KF_mean_after':None if not post else post['mean'],
                    'bank_key':c['bank_key'],'bank_updates_before':None if not c['bank'] else c['bank']['updates'],
                    'has_regional_prior':c['region'] is not None,
                    'bank_uncertainty_a_before':None if not c['prior'] else float(np.sqrt(c['prior']['covariance'][1][1])),
                    'base_top5':[{'id':ids[j],'point':point(pool,j),'y_percent':float(y[j]*100),
                        'base_qLogEI':float(a['base_qLogEI'][j]),'loss':float(a['base_qLogEI'][ref]-a['base_qLogEI'][j]),
                        'within_gate_budget':bool(eligible[j]),'mu_percent':float(a['mu_y'][j]*100),'sd_pp':float(a['sd_y'][j]*100)} for j in top],
                    'best_eligible_y_percent':float(y[eligible].max()*100),
                    'best_available_y_percent':float(y[a['available']].max()*100),
                    'number_above95_in_budget':int(((y>=.95)&eligible).sum()),
                    'selected_matches_initial_knowledge':bool(member(pool,ready['knowledge'])[ix]),
                    'selected_matches_current_knowledge':bool(mask[ix])}
                if method=='ReasoningBO_point':
                    insight=d['insight'];rank_conf={'high':0,'medium':1,'low':2};ranked=sorted(insight['hypotheses'],key=lambda h:rank_conf[h['confidence'].lower()])
                    r['selected_hypothesis']=ranked[0]
                ms['rounds'].append(r)
            for event in s['calls']:
                key=event['context_sha256'];raw=event['validated']['raw']
                ms['refreshes'].append({'query':event['iteration'],'reasons':event['reasons'],'request_context':key,
                    'before_knowledge':quality(pool,y,s['initial_knowledge'] if not ms['refreshes'] else s['calls'][len(ms['refreshes'])-1]['validated']['plan']['knowledge']),
                    'after_knowledge':quality(pool,y,event['validated']['plan']['knowledge']),
                    'actions':raw['actions'],'summary':raw['summary'],
                    'before_decision':event['before']})
            rows=ms['rounds'];changed=[r for r in rows if r['selected']!=r['reference']]
            ms['summary']={'nonbase_choices':len(changed),'one_step_positive':sum(r['one_step_delta_vs_reference_pp']>1e-8 for r in changed),
                'one_step_negative':sum(r['one_step_delta_vs_reference_pp']< -1e-8 for r in changed),
                'one_step_delta_sum_pp':sum(r['one_step_delta_vs_reference_pp'] for r in rows),
                'choices_outside_hypothetical_R2_budget':[r['query'] for r in rows if r['loss']>r['budget']],
                'selected_initial_region_count':sum(r['selected_matches_initial_knowledge'] for r in rows),
                'fallback_queries':[r['query'] for r in rows if r['fallback']],
                'active_amplitude_queries':[r['query'] for r in rows if r['amplitude']>0 and r['has_regional_prior']],
                'base_top5_contains_above95_queries':[r['query'] for r in rows if max(v['y_percent'] for v in r['base_top5'])>=95],
                'selected_rank_counts':dict(Counter(r['base_rank'] for r in rows)),
                'base_selected_MAE_pp':float(np.mean([abs(r['base_mu_percent']-r['yield_percent']) for r in rows])),
                'deployed_selected_MAE_pp':float(np.mean([abs(r['deployed_mean_percent']-r['yield_percent']) for r in rows]))}
            ds['methods'][method]=ms
        result['datasets'][task]=ds
    for path,h in before.items():assert hashed(BASE/path)==h,path
    save(OUT/'analysis.json',result)
    print(json.dumps({t:{m:{'AUC':v['SearchAUC'],'best':v['final_best_percent'],'summary':v.get('summary')}
        for m,v in ds['methods'].items() if m in ('New_R2','ReasoningBO_point','Pure_BO')} for t,ds in result['datasets'].items()},indent=2))


if __name__=='__main__':main()
