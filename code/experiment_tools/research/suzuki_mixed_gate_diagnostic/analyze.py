"""Post-hoc prefix interventions, never new trajectories, GP fits or LLM calls.

Alternative outcomes and region-quality statistics use the fixed oracle only
for retrospective evaluation; hidden labels never enter candidate score inputs.
They are local, deterministic one-step counterfactuals, not additive trajectory
AUC attributions. Each historical posterior/knowledge plan stays fixed.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import csv
import json
import statistics as st

import numpy as np
import torch
from chem.benchmarks import mixed_comparison_suzuki_engine as e
from chem.benchmarks import mixed_comparison_suzuki_study as shared
from chem.benchmarks import mixed_kalman_suzuki_study as original
from chem.benchmarks import kalman_knowledge_suzuki_engine as p
from chem.benchmarks import kalman_knowledge_suzuki_report as algebra
from chem.benchmarks.suzuki_domain import CATALYSTS,digest

OUT=Path('research/suzuki_mixed_gate_diagnostic')
read=shared.read;write=shared.write;hashfile=shared.filehash
RHO0=float(1-np.sqrt(.2525))


def guarded_paths():
    paths=[Path(__file__),Path(e.__file__),Path(e.mixed.__file__),Path(p.__file__),
           original.OUT/'results.json',original.OUT/'release_receipt.json',
           shared.OUT/'results.json',shared.OUT/'release_receipt.json']
    for case in (2,4):
        for name in ('pool.json','oracle.npz','advice.json'):paths.append(shared.original.ROOT/'inputs'/f'case{case}'/name)
        for folder in ('runs','frozen','base'):paths.append(original.ROOT/folder/f'c{case}_s0'/'state.json')
        paths.append(shared.ROOT/'runs'/f'c{case}_s0_assistant_initial_LGBO_gate'/'state.json')
    return {f.as_posix():hashfile(f) for f in paths}


def worker_init():shared.worker_init()


def analyze_case(case):
    pool=shared.inputs(case);ids=pool['ids'];ix={k:i for i,k in enumerate(ids)};z=np.asarray(pool['z'])
    ys=np.load(shared.original.ROOT/'inputs'/f'case{case}'/'oracle.npz')['values']
    new=read(original.ROOT/'runs'/f'c{case}_s0'/'state.json')
    gate=read(shared.ROOT/'runs'/f'c{case}_s0_assistant_initial_LGBO_gate'/'state.json')
    assert new['history'][:4]==gate['history'][:4]
    prior_advice=shared.advice(case,'assistant_initial')
    full_region=e.regional.region_for(prior_advice,[h['id'] for h in prior_advice['clauses']])
    sources=[root/folder/f'c{case}_s0' for root in (original.ROOT,shared.ROOT)
             for folder in ('cache','audit_cache','qualification_cache')]
    cache=e.MixedRegionalCache(pool,OUT/'cache'/f'c{case}',case,sources=sources)
    rows=[];cf=[];quality=[];summaries=[]
    target_id='p03631' if case==2 else 'p00874';target=ix[target_id]
    categories=np.asarray([CATALYSTS.index(x['catalyst']) for x in pool['points']])

    def point_info(j):return {'candidate_id':ids[j],'point':pool['points'][j],'category':int(categories[j])}
    def outcome(j,inc):return {'yield_pp':float(ys[j]*100),'immediate_incumbent_pp':float(max(inc,ys[j])*100),**point_info(j)}

    for trajectory,state in (('new',new),('gate',gate)):
        for t,event in enumerate(state['decisions'],1):
            history=state['history'][:t+3];inc=max(h['y'] for h in history)
            base,bmeta=cache.score(history);stats=cache.posterior_stats(history);fit,value=cache.model(history)
            available=~np.isin(ids,[h['candidate_id'] for h in history]);ref=e.regional.argmax(base,available,ids)
            budget=e.regional.budget_at(4,t);eligible=available&((base[ref]-base)<=budget)
            oracle_residual=ys-stats['mu_y'];local_scores={};current=ix[event['selected_id']]
            if trajectory=='new':
                assert bmeta['key']==event['base_score_key']
                actual_retained=event['diagnosis']['retained_ids'];hs=event['hypotheses'];weights=event['trust']
                if actual_retained:actual_scores,ameta=cache.knowledge_score(history,hs,weights,actual_retained)
                else:actual_scores=base
                actual_delta=p.correction(z,hs,weights,stats['sd_y'],actual_retained)
            else:
                assert bmeta==event['base_acquisition']
                actual_scores,ameta=cache.score(history,full_region,prior_advice['confidence'])
                with np.load(cache.directory/(ameta['key']+'.npz'),allow_pickle=False) as d:
                    actual_delta=d['shift_std'].copy()*float(value['std'])
                hs=[];weights={};actual_retained=[]
            assert e.regional.argmax(actual_scores,eligible,ids)==current
            assert abs((base[ref]-base[current])-event['base_logei_loss'])<1e-12
            actual=outcome(current,inc)

            def residual_scores(hypotheses,w,retained,cap=True):
                if not retained:return base,np.zeros(len(ids))
                key=digest({'h':hypotheses,'w':w,'r':sorted(retained),'cap':cap})
                if key not in local_scores:
                    if cap:delta=p.correction(z,hypotheses,w,stats['sd_y'],retained)
                    else:
                        delta=np.zeros(len(ids));phi=p.features(z,hypotheses)
                        for h in hypotheses:
                            if h['id'] in retained:delta+=w.get(h['id'],0)*phi[:,p.SLOTS.index(h['id'])+1]
                    score=algebra.marginal_scores(stats['mu_std'],stats['sd_std'],delta/float(value['std']),
                                                   value['best_std'],bmeta['mc_seed'],cache.z)
                    local_scores[key]=score,delta
                return local_scores[key]

            def diagnose(hypotheses,w,b=budget):
                return p.diagnose(base,z,hypotheses,w,b,~available,ids)

            def add_variant(name,scores,mask,delta=None,retained=None):
                j=e.regional.argmax(scores,mask,ids)
                row={'case':case,'trajectory':trajectory,'round':t,'variant':name,
                     'actual_candidate':ids[current],**outcome(j,inc),
                     'actual_yield_pp':actual['yield_pp'],'actual_incumbent_pp':actual['immediate_incumbent_pp'],
                     'actual_minus_variant_yield_pp':actual['yield_pp']-float(ys[j]*100),
                     'actual_minus_variant_incumbent_pp':actual['immediate_incumbent_pp']-max(inc,ys[j])*100,
                     'same_choice':bool(j==current),'base_logEI_loss':float(base[ref]-base[j]),
                     'inside_actual_budget':bool(eligible[j]),'retained_ids':list(retained or []),
                     'selected_GP_mean_pp':float(stats['mu_y'][j]*100),'selected_GP_sd_pp':float(stats['sd_y'][j]*100),
                     'mean_correction_pp':None if delta is None else float(delta[j]*100)}
                cf.append(row);return row

            add_variant('actual',actual_scores,eligible,actual_delta,actual_retained)
            add_variant('base_only',base,available,np.zeros(len(ids)))
            add_variant('no_gate_hold_guidance',actual_scores,available,actual_delta,actual_retained)
            if trajectory=='new':
                replay,dd=residual_scores(hs,weights,actual_retained)
                err=float(np.max(np.abs(replay-actual_scores)));assert err<1e-8
                initial_w={h['id']:RHO0 for h in hs}
                s,d=residual_scores(hs,initial_w,actual_retained)
                add_variant('frozen_trust_hold_B4',s,eligible,d,actual_retained)
                dg=diagnose(hs,initial_w);s,d=residual_scores(hs,initial_w,dg.retained_ids)
                add_variant('frozen_trust_rediagnose_B4',s,eligible,d,dg.retained_ids)
                active=[h['id'] for h in hs if weights.get(h['id'],0)>.05]
                s,d=residual_scores(hs,weights,active)
                add_variant('no_B4_hold_trust',s,eligible,d,active)
                cover=float(np.max(base)-np.min(base)+1)
                dg=diagnose(hs,weights,cover);s,d=residual_scores(hs,weights,dg.retained_ids)
                add_variant('no_gate_rediagnose_B4',s,available,d,dg.retained_ids)
                s,d=residual_scores(hs,weights,actual_retained,False)
                add_variant('no_mean_cap',s,eligible,d,actual_retained)
                if actual_retained:
                    masks=p.membership(z,hs);hard=eligible.copy()
                    for k,h in enumerate(hs):
                        if h['id'] in actual_retained:hard&=masks[k]
                    add_variant('hard_execute_same_retained',actual_scores,hard,actual_delta,actual_retained)
                else:add_variant('hard_execute_same_retained',base,eligible,np.zeros(len(ids)),[])
                if any(c['iteration']==t for c in state['calls']):
                    old_h=event['hypotheses_before_call'];forecast=p.forecast(event['filter_before_call'],old_h)
                    w=forecast['weights'] if event['enabled_before_call'] else {h['id']:0 for h in old_h}
                    dg=diagnose(old_h,w);s,d=residual_scores(old_h,w,dg.retained_ids)
                    add_variant('no_LLM_refresh_this_round',s,eligible,d,dg.retained_ids)
                for h in hs:
                    w=dict(weights);w[h['id']]=0.;dg=diagnose(hs,w);s,d=residual_scores(hs,w,dg.retained_ids)
                    add_variant('drop_'+h['id'],s,eligible,d,dg.retained_ids)
                    mask=p.membership(z,[h])[0];inside=available&mask;outside=available&~mask
                    j=p.SLOTS.index(h['id'])+1
                    q={'case':case,'round':t,'id':h['id'],'version':h['version'],
                       **p.canonical(h),'trust':weights.get(h['id'],0),
                       'theta':event['filter_prior']['theta'][j],
                       'theta_sd':float(np.sqrt(event['filter_prior']['covariance'][j][j])),
                       'retained':h['id'] in actual_retained,
                       'selected_satisfies':bool(mask[current]),'selected_component_raw_pp':float(weights.get(h['id'],0)*p.features(z[[current]],hs)[0,j]*100),
                       'offline_residual_contrast_pp':float((oracle_residual[inside].mean()-oracle_residual[outside].mean())*100),
                       'offline_yield_mean_inside_pp':float(ys[inside].mean()*100),'offline_yield_mean_outside_pp':float(ys[outside].mean()*100),
                       'offline_max_inside_pp':float(ys[inside].max()*100),'offline_max_outside_pp':float(ys[outside].max()*100)}
                    quality.append(q)
                region_scores,region_meta=cache.score(history,full_region,prior_advice['confidence'])
                with np.load(cache.directory/(region_meta['key']+'.npz'),allow_pickle=False) as d:
                    region_delta=d['shift_std'].copy()*float(value['std'])
                add_variant('swap_to_baseline_region_system',region_scores,eligible,region_delta,[])
                add_variant('baseline_region_ungated',region_scores,available,region_delta,[])
            else:
                err=0.

            selected_mu=float(stats['mu_y'][current]*100);dy=float(actual_delta[current]*100);observed=float(ys[current]*100)
            target_available=bool(available[target])
            target_rank_base=1+int(np.sum(available&(base>base[target]))) if target_available else None
            target_rank_guide=1+int(np.sum(eligible&(actual_scores>actual_scores[target]))) if eligible[target] else None
            r={'case':case,'trajectory':trajectory,'round':t,'total_query':t+4,
               'candidate_id':ids[current],'point':pool['points'][current],'category':int(categories[current]),
               'yield_pp':observed,'incumbent_before_pp':float(inc*100),'incumbent_after_pp':actual['immediate_incumbent_pp'],
               'base_candidate':ids[ref],'base_candidate_yield_pp':float(ys[ref]*100),
               'budget':budget,'eligible_count':int(eligible.sum()),'loss':float(base[ref]-base[current]),
               'GP_mean_pp':selected_mu,'GP_sd_pp':float(stats['sd_y'][current]*100),'correction_pp':dy,
               'deployed_squared_error_delta_pp2':(observed-selected_mu-dy)**2-(observed-selected_mu)**2,
               'GP_residual_pp':observed-selected_mu,'target_id':target_id,'target_available':target_available,
               'target_eligible':bool(eligible[target]),'target_base_loss':float(base[ref]-base[target]) if target_available else None,
               'target_base_rank':target_rank_base,'target_guided_rank_in_E':target_rank_guide,
               'score_replay_error':err}
            if trajectory=='new':
                r.update(trust=weights,retained_ids=actual_retained,revoked_ids=event['diagnosis']['revoked_ids'],
                         conflict=event['diagnosis']['conflict_kind'],innovation_z=event['measurement']['innovation_z'],
                         cap_binding=abs(dy)>=.5*r['GP_sd_pp']-1e-10 and abs(dy)>0,
                         calls=[{'reasons':c['reasons'],'plan':c['plan']} for c in state['calls'] if c['iteration']==t],
                         filter_theta=event['filter_prior']['theta'],filter_covariance=event['filter_prior']['covariance'])
            rows.append(r)

    new_rows=[r for r in rows if r['trajectory']=='new'];gate_rows=[r for r in rows if r['trajectory']=='gate']
    gaps=[g['incumbent_after_pp']-a['incumbent_after_pp'] for a,g in zip(new_rows,gate_rows)]
    features=np.asarray([v['measurement']['feature'] for v in new['decisions'][:7]])
    initial_ids=[h['id'] for h in new['decisions'][0]['hypotheses']]
    columns=[0]+[p.SLOTS.index(key)+1 for key in initial_ids]
    rank=int(np.linalg.matrix_rank(features[:,columns]))
    result={'case':case,'trajectory':rows,'counterfactuals':cf,'knowledge_quality':quality,
            'gate_minus_new_AUC_pp':st.mean(gaps),'round_incumbent_gaps_pp':gaps,
            'new_AUC':st.mean(r['incumbent_after_pp'] for r in new_rows),
            'gate_AUC':st.mean(r['incumbent_after_pp'] for r in gate_rows),
            'first7_filter_feature_rank':rank,'first7_filter_feature_columns':columns,
            'first7_feature_rows':features[:,columns].tolist(),'cache_counters':cache.counters,
            'operator_compiles_in_loop':sum(original.JIT.values())}
    assert cache.counters['fits']==0
    write(OUT/f'case{case}.json',result)
    return {k:v for k,v in result.items() if k not in ('trajectory','counterfactuals','knowledge_quality')}


def save_csv(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def main():
    OUT.mkdir(parents=True,exist_ok=True);before=guarded_paths();write(OUT/'input_manifest.json',before)
    with ProcessPoolExecutor(max_workers=2,initializer=worker_init) as pool:summary=list(pool.map(analyze_case,(2,4)))
    results=[read(OUT/f'case{c}.json') for c in (2,4)]
    for field in ('trajectory','counterfactuals','knowledge_quality'):
        save_csv(OUT/(field+'.csv'),[r for case in results for r in case[field]])
    assert before==guarded_paths()
    receipt={'status':'PASS','prefixes_replayed':64,'actual_choices_reproduced':64,'new_GP_fits':0,'LLM_calls':0,
             'original_inputs_unchanged':True,'max_native_score_replay_error':max(r['score_replay_error'] for c in results for r in c['trajectory']),
             'counterfactual_choices':sum(len(c['counterfactuals']) for c in results),
             'counterfactual_scope':'One step at each observed historical prefix. Alternative labels are offline fixed-oracle lookups; no rollout or additive AUC module attribution.',
             'frozen_trust_definition':RHO0,'summary':summary}
    assert all(r['operator_compiles_in_loop']==0 for r in summary)
    write(OUT/'validation_receipt.json',receipt);print(json.dumps(receipt,separators=(',',':')))


if __name__=='__main__':main()
