"""Read-only aggregation and plots for the fixed-prefix diagnostic."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import json
from pathlib import Path
import statistics as st
import numpy as np
from chem.benchmarks import mixed_comparison_suzuki_study as s
from chem.benchmarks import kalman_knowledge_suzuki_engine as p

OUT=Path('research/suzuki_mixed_gate_diagnostic')
data=[s.read(OUT/f'case{case}.json') for case in (2,4)]
result={}
for d in data:
    case=d['case'];new=[r for r in d['trajectory'] if r['trajectory']=='new']
    gate=[r for r in d['trajectory'] if r['trajectory']=='gate']
    variants=[r for r in d['counterfactuals'] if r['trajectory']=='new']
    segments=[(1,7),(8,10),(11,16)] if case==2 else [(1,7),(8,11),(12,16)]
    part=[{'rounds':[lo,hi],'AUC_gap_pp':sum(d['round_incumbent_gaps_pp'][lo-1:hi])/16} for lo,hi in segments]
    for row in part:row['fraction_of_total_gap']=row['AUC_gap_pp']/d['gate_minus_new_AUC_pp']
    gains={}
    for name in sorted({r['variant'] for r in variants}):
        group=[r for r in variants if r['variant']==name]
        gains[name]={'changed_rounds':[r['round'] for r in group if not r['same_choice']],
                     'immediate_benefit_rounds':[{'round':r['round'],'effect_pp':r['actual_minus_variant_incumbent_pp']} for r in group if r['actual_minus_variant_incumbent_pp']>1e-8],
                     'immediate_loss_rounds':[{'round':r['round'],'effect_pp':r['actual_minus_variant_incumbent_pp']} for r in group if r['actual_minus_variant_incumbent_pp']<-1e-8]}
    state=s.read(s.previous.ROOT/'runs'/f'c{case}_s0'/'state.json')
    pre_call=[];ties=[]
    pool=s.inputs(case);ids=pool['ids'];ix={key:i for i,key in enumerate(ids)}
    for t,event in enumerate(state['decisions'],1):
        h=state['history'][:t+3]
        folder=s.previous.ROOT/'cache'/f'c{case}_s0'
        with np.load(folder/(event['base_score_key']+'.npz'),allow_pickle=False) as f:base=f['scores'].copy()
        with np.load(folder/(event['guided_score_key']+'.npz'),allow_pickle=False) as f:guided=f['scores'].copy()
        seen=np.isin(ids,[r['candidate_id'] for r in h]);av=~seen
        eligible=av&((float(base[av].max())-base)<=event['budget_logei'])
        target=ix[new[t-1]['target_id']]
        tied=[ids[j] for j in np.flatnonzero(eligible&(guided==float(guided[eligible].max())))]
        if eligible[target] and ids[target] in tied:
            ties.append({'round':t,'actual_id':event['selected_id'],'target_id':ids[target],
                         'guided_score':float(guided[target]),'exact_top_ties':tied})
        if any(c['iteration']==t for c in state['calls']):
            hs=event['hypotheses_before_call'];pr=p.forecast(event['filter_before_call'],hs)
            weights=pr['weights'] if event['enabled_before_call'] else {h['id']:0 for h in hs}
            dg=p.diagnose(base,pool['z'],hs,weights,event['budget_logei'],seen,ids)
            pre_call.append({'round':t,'retained_before_refresh':list(dg.retained_ids),'removed_before_refresh':list(dg.revoked_ids),
                             'conflict_before_refresh':dg.conflict_kind,'trust_before_refresh':weights})
    errs={name:float(np.sqrt(st.mean(e['measurement']['squared_errors'][name] for e in state['decisions']))*100)
          for name in ('GP','GP_offset_null','GP_joint_KF','GP_deployed')}
    result[str(case)]={'segments':part,'components':gains,'pre_call_diagnosis':pre_call,'exact_ties_with_target':ties,
                      'same_visited_points_RMSE_pp':errs,
                      'deployed_MSE_minus_GP_pp2':st.mean(r['deployed_squared_error_delta_pp2'] for r in new),
                      'deployed_worse_squared_error_rounds':[r['round'] for r in new if r['deployed_squared_error_delta_pp2']>1e-8],
                      'budget_direct_changed_rounds':gains['no_gate_hold_guidance']['changed_rounds']}
s.write(OUT/'summary.json',result)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,2,figsize=(12,6),layout='constrained')
for col,d in enumerate(data):
    new=[r for r in d['trajectory'] if r['trajectory']=='new'];gate=[r for r in d['trajectory'] if r['trajectory']=='gate']
    ax=axes[0,col]
    ax.plot(range(1,17),[r['incumbent_after_pp'] for r in new],label='KF/event/B4',color='#0072B2',marker='o',ms=3)
    ax.plot(range(1,17),[r['incumbent_after_pp'] for r in gate],label='Regional budget gate',color='#D55E00',marker='s',ms=3)
    for r in new:
        if r.get('calls'):ax.axvline(r['round'],color='#0072B2',alpha=.14,lw=1)
    ax.set(title=f"Case {d['case']}",ylabel='Best yield (%)',xticks=range(1,17));ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    ax=axes[1,col];ax.bar(range(1,17),d['round_incumbent_gaps_pp'],color='#777777')
    ax.set(xlabel='Added query round (1 = total query 5)',ylabel='Gate minus new (pp)',xticks=range(1,17))
    ax.set_title(f"Mean gap = {d['gate_minus_new_AUC_pp']:.4f} pp; same final best",fontsize=10);ax.grid(axis='y',alpha=.2)
for ax in axes.flat:ax.spines[['top','right']].set_visible(False)
fig.suptitle('Actual trajectory gaps; vertical lines mark LLM refreshes. Module counterfactuals are separate.',fontsize=11)
fig.savefig(OUT/'trajectory_gap.png',dpi=170,bbox_inches='tight');plt.close(fig)
print(json.dumps({c:{k:v for k,v in x.items() if k!='components'} for c,x in result.items()},separators=(',',':')))
