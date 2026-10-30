"""Validate and summarize post-hoc controls without changing primary records."""
from __future__ import annotations
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json
import math
import statistics as st
import numpy as np
from scripts.diagnose_suzuki_comparison import ROOT,OUT,CASES,SEEDS,read,write,digest,filehash,old_result,native_state


def aggregate(rows):
    return {'n':len(rows),
        **{key+'_pct':st.mean(r[key]*100 for r in rows) for key in ('initial_best','search_auc','final_best')},
        'by_case':{str(c):{key+'_pct':st.mean(r[key]*100 for r in rows if r['case']==c)
                    for key in ('initial_best','search_auc','final_best')} for c in CASES}}


def main():
    p=read(OUT/'protocol.json');raw=dict(p);h=raw.pop('protocol_sha256');assert digest(raw)==h
    assert filehash('scripts/diagnose_suzuki_comparison.py')==p['driver_sha256']
    for path,h in (p['sources']|p['input_sha256']).items():assert filehash(path)==h,path
    q=read(OUT/'regional_lift_protocol.json');raw=dict(q);h=raw.pop('protocol_sha256');assert digest(raw)==h
    assert filehash('scripts/diagnose_lgbo_regional_lift.py')==q['driver_sha256']
    primary_inventory=read('research/suzuki_official/release_inventory.json')
    for entry in primary_inventory['files']:assert filehash(entry['path'])==entry['sha256'],entry['path']
    # A second arithmetic implementation derives incumbents from raw outcomes.
    rows=[];max_metric_error=0.;recomputed_positions=0
    for f in sorted((ROOT/'controls').glob('*/result.json')):
        r=read(f)
        if 'history' in r:ys=[x['y'] for x in r['history']]
        else:ys=[x['value'] for x in read(f.parent/'state.json')['oracle_ledger']]
        curve=[max(ys[:i]) for i in range(1,21)]
        assert len(ys)==r['actual_queries']==20 and r['status']=='COMPLETE'
        error=abs(sum(curve[4:])/16-r['search_auc']);max_metric_error=max(max_metric_error,error)
        assert error<1e-14 and curve==r['best_curve']
        assert r['LLM_calls']==0 and r['operator_compiles_in_loop']==0
        recomputed_positions+=20;rows.append(r)
    assert len(rows)==91
    controls=[r for r in rows if r['recipe']!='legacy_bridge_qualification']
    counts=Counter(r['recipe'] for r in controls);assert len(counts)==6 and set(counts.values())=={15}
    assert all({(r['case'],r['seed']) for r in controls if r['recipe']==recipe}=={(c,s) for c in CASES for s in SEEDS} for recipe in counts)
    original=[{**old_result(c,s),'case':c} for c in CASES for s in SEEDS]
    groups={'legacy_original':aggregate(original)}
    groups.update({recipe:aggregate([r for r in controls if r['recipe']==recipe]) for recipe in counts})
    official=read('research/suzuki_official/results.json')['runs']
    for method in ('LLINBO','LGBO','HEBO'):
        a=[r for r in official if r['method']==method and r['case'] in CASES]
        groups[method]={'n':15,**{key:st.mean(r[key] for r in a) for key in ('initial_best_pct','search_auc_pct','final_best_pct')},
            'by_case':{str(c):{key:st.mean(r[key] for r in a if r['case']==c) for key in ('initial_best_pct','search_auc_pct','final_best_pct')} for c in CASES}}
    ll_checks=[];paired={}
    for active,neutral in [('LLINBO','LLINBO_without_LLM'),('LGBO','LGBO_neutral_prior'),('LGBO','LGBO_zero_regional_lift')]:
        differences=[]
        for r in controls:
            if r['recipe']!=neutral:continue
            s=native_state(r['case'],r['seed'],active)
            actual=sum(max(h['y'] for h in s['history'][:i]) for i in range(5,21))/16*100
            differences.append({'case':r['case'],'seed':r['seed'],'delta_auc_pct':actual-r['search_auc']*100})
            assert r['history'][:4]==s['history'][:4]
            if active=='LLINBO':
                accepted=[i for i,d in enumerate(s['decisions']) if d['accepted_llm']]
                equality=[h['z'] for h in r['history']]==[h['z'] for h in s['history']]
                if not accepted:assert equality
                else:
                    k=accepted[0]
                    assert [h['z'] for h in r['history'][:4+k]]==[h['z'] for h in s['history'][:4+k]]
                    assert r['history'][4+k]['z']==s['decisions'][k]['native_gp_proposal']
                ll_checks.append({'case':r['case'],'seed':r['seed'],'accepted':len(accepted),'entire_trajectory_equal':equality})
            elif neutral=='LGBO_zero_regional_lift':
                assert [d['plan']['mode'] for d in r['plans']]==[d['plan']['mode'] for d in s['decisions']]
                assert all(call['lambda']==0 for call in r['regional_fit_calls'])
        paired[active+'_minus_'+neutral]={'mean_delta_pct':st.mean(d['delta_auc_pct'] for d in differences),
            'by_case':{str(c):st.mean(d['delta_auc_pct'] for d in differences if d['case']==c) for c in CASES},
            'positive_seed_pairs':sum(d['delta_auc_pct']>1e-9 for d in differences),
            'negative_seed_pairs':sum(d['delta_auc_pct'] < -1e-9 for d in differences),'seed_pairs':differences}
    a=groups['legacy_original']['search_auc_pct'];b=groups['old_pool_new_init']['search_auc_pct']
    c=groups['expanded_pool_old_init']['search_auc_pct'];d=groups['expanded_pool_new_init']['search_auc_pct']
    factorial={'pool_effect_with_original_init_pct':c-a,'pool_effect_with_new_init_pct':d-b,
        'init_effect_with_original_pool_pct':b-a,'init_effect_with_expanded_pool_pct':d-c,
        'interaction_pct':d-c-b+a,'note':'Controlled legacy B0 effects for this diagnostic grid; not a causal decomposition of every native method.'}
    all_states=[native_state(c,s,m) for c in CASES for s in SEEDS for m in ('LLINBO','LGBO','HEBO')]
    lg_branches=Counter(x['plan']['mode'] for s in all_states if s['method']=='LGBO' for x in s['decisions'])
    old_groups=read(OUT/'raw_reconciliation.json')['groups']
    best_old=max((r for r in old_groups if r['family']=='legacy'),key=lambda r:r['search_auc_pct'])
    cap=st.mean(r['old_pool_max_pct'] for r in read(OUT/'pool_inspection.json'))
    output={'study':'Suzuki cross-protocol diagnosis','scope':{'cases':list(CASES),'seeds':list(SEEDS),
        'paired_trajectories_per_recipe':15,'initial_plus_additional':[4,16],'posthoc':True,'blind_LLM_evaluation':False},
        'groups':groups,'best_original_legacy_setting':best_old,
        'legacy_initial_only_factorial':factorial,'paired_guidance_ablation':paired,
        'old_pool_unlimited_query_mean_final_ceiling_pct':cap,
        'native_mean_final_above_old_pool_ceiling_pct':{m:groups[m]['final_best_pct']-cap for m in ('LLINBO','LGBO','HEBO')},
        'LLINBO_branch_checks':ll_checks,'LGBO_original_common_case_branches':dict(lg_branches),
        'legacy_B4_trace':read(OUT/'raw_reconciliation.json')['legacy_B4_actual_assistant'],
        'point_prior_equal_weight_proof':q['point_prior_uniform_weight_checks'],
        'limits':['Old and new studies have different search domains and initialization; only cases and metric units are aligned here.',
            'New initialization has higher observed incumbent but worsens the unchanged legacy B0 under both candidate sets.',
            'The regional zero-lift ablation conditions on the saved active preference/branch schedule; no new on-policy assistant calls were made.',
            'Other model/kernel/encoding/acquisition differences have not been individually isolated.',
            'Only three reaction scenarios and five seeds; post-hoc exploratory evidence does not establish broad superiority or clean chemical-knowledge causality.',
            'PointBumpPrior equal path weights are an observed property of the pinned official code, which remains unchanged.'],
        'validation':{'status':'PASS','diagnostic_trajectories':90,'extra_legacy_qualification_trajectories':1,
            'diagnostic_query_budget_used':1800,'qualification_replay_queries':20,
            'independently_recomputed_budget_positions':recomputed_positions,
            'max_metric_error_normalized':max_metric_error,'source_files_and_primary_records_unchanged':True,
            'protected_primary_files_checked':len(primary_inventory['files']),
            'LLINBO_zero_accept_exact_trajectory_matches':sum(r['entire_trajectory_equal'] for r in ll_checks),
            'LLM_calls':0,'operator_compiles_in_loop':0},
        'source_receipts':{'parent_diagnostic_protocol':p['protocol_sha256'],'regional_protocol':q['protocol_sha256']}}
    write(OUT/'comparison_diagnosis.json',output)
    # Compact independently auditable row table, before any chart is authored.
    import csv
    with (OUT/'diagnostic_metrics.csv').open('w',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['case','seed','recipe','initial_best_pct','search_auc_pct','final_best_pct'])
        writer.writeheader();writer.writerows({'case':r['case'],'seed':r['seed'],'recipe':r['recipe'],
            **{k+'_pct':r[k]*100 for k in ('initial_best','search_auc','final_best')}} for r in controls)
    print(json.dumps({'validation':output['validation'],'groups':groups,'paired':{k:{a:b for a,b in v.items() if a!='seed_pairs'} for k,v in paired.items()},'factorial':factorial},separators=(',',':')))


if __name__=='__main__':main()
