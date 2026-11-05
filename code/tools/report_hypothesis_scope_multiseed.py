"""Per-seed paired results, sample SD and inspectable English evidence report."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hypothesis_scope_multiseed import study as s

LABELS=dict(bo='Pure BO',first='Same LLM first + BO',hyp='Frozen hypothesis policy')
NAMES=dict(c1='Reizman Suzuki Case 1',c2='Reizman Suzuki Case 2',c3='Reizman Suzuki Case 3',c4='Reizman Suzuki Case 4',perera='Perera Suzuki',doyle='Doyle Buchwald-Hartwig',snar='SnAr',hplc='HPLC')


def states(p):
    for task in s.TASKS:
        for arm in s.ARMS:
            path=s.frozen.folder_for(task)/'state.json' if arm=='hyp' else ROOT/p['seed0_controls'][task][{'bo':'pure_BO','first':'LLM_first_then_BO'}[arm]]['path']
            yield task,0,arm,s.read(path),path
        for seed in s.SEEDS:
            for arm in s.ARMS:
                path=s.folder_for(task,seed,arm)/'state.json';yield task,seed,arm,s.read(path),path


def main():
    p=s.check(True);data=[];histories={};paths={};trajectory=[];mechanisms=[]
    for task,seed,arm,state,path in states(p):
        row=s.frozen.first.metrics(state,task,arm)
        row.update(seed=seed,seed0_imported=seed==0,state_path=path.relative_to(ROOT).as_posix(),state_sha256=s.sha(path))
        data.append(row);short=s.SHORT[task]
        raw=np.asarray([h['raw_value'] for h in state['history']]);histories[(short,seed,arm)]=np.maximum.accumulate(raw)
        paths[(short,seed,arm)]=tuple(h['candidate_id'] for h in state['history'])
        for q,(value,best) in enumerate(zip(raw,np.maximum.accumulate(raw)),1):trajectory.append(dict(task=short,seed=seed,arm=arm,query=q,objective_raw=float(value),best=float(best)))
        if arm=='hyp':
            changed=[r['query'] for r in state['decisions'] if r['decision'].get('changed_from_unrestricted_BO')]
            mechanisms.append(dict(task=short,seed=seed,logical_LLM_calls=len(state['calls']),
                initial_prior_reused=seed!=0,refresh_queries=';'.join(str(c['query']) for c in state['calls'][1:]),
                scoped_queries=sum(r['decision']['reason']=='hypothesis_constrained_native_qLogEI' for r in state['decisions']),
                changed_choices=len(changed),changed_queries=';'.join(map(str,changed)),
                KF_updates=sum(e['KF'] is not None for r in state['decisions'] for e in r['knowledge_feedback']),
                final_active_notes=sum(n['permission'] in ('active','provisional','probation') for n in state['notes'])))
    df=pd.DataFrame(data).sort_values(['task','seed','method']);df.to_csv(s.REPORT/'per_seed_results.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(trajectory).to_csv(s.REPORT/'trajectories.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(mechanisms).to_csv(s.REPORT/'mechanisms.csv',index=False,encoding='utf-8-sig')
    paired=[];summary=[]
    for task in s.SHORT.values():
        f=df[df.task==task]
        for seed in (0,*s.SEEDS):
            r=f[f.seed==seed].set_index('method');h=r.loc['hyp']
            paired.append(dict(task=task,seed=seed,delta_AUC_vs_BO=h.SearchAUC-r.loc['bo'].SearchAUC,
                delta_AUC_vs_same_first=h.SearchAUC-r.loc['first'].SearchAUC,
                delta_final_vs_BO=h.final_best-r.loc['bo'].final_best,
                delta_final_vs_same_first=h.final_best-r.loc['first'].final_best))
        for cohort,seeds in (('all_0_1_2_3',(0,1,2,3)),('new_1_2_3',(1,2,3))):
            g=f[f.seed.isin(seeds)];v={}
            for arm in s.ARMS:
                r=g[g.method==arm]
                v[f'{arm}_unique_paths']=len({paths[(task,seed,arm)] for seed in seeds})
                for metric in ('SearchAUC','final_best'):
                    v[f'{arm}_{metric}_mean']=float(r[metric].mean());v[f'{arm}_{metric}_SD']=float(r[metric].std(ddof=1))
            for comparator,column in (('BO','delta_AUC_vs_BO'),('same_first','delta_AUC_vs_same_first')):
                d=np.asarray([r[column] for r in paired if r['task']==task and r['seed'] in seeds])
                v[f'delta_vs_{comparator}_mean']=float(d.mean());v[f'delta_vs_{comparator}_SD']=float(d.std(ddof=1))
                v[f'W_T_L_vs_{comparator}']=f'{sum(d>1e-8)}/{sum(abs(d)<=1e-8)}/{sum(d<-1e-8)}'
            summary.append(dict(task=task,cohort=cohort,n=len(seeds),**v))
    pairs=pd.DataFrame(paired);pairs.to_csv(s.REPORT/'paired_differences.csv',index=False,encoding='utf-8-sig')
    aggregate=pd.DataFrame(summary);aggregate.to_csv(s.REPORT/'summary.csv',index=False,encoding='utf-8-sig')
    lines=['FROZEN HYPOTHESIS POLICY: 0+20 MULTISEED COMPARISON','',
        'Eight tasks; seed0 imported unchanged, seeds1-3 newly run. All methods use zero free observations and20 paid table queries. '
        'Both LLM-initialized arms share the exact frozen first physical input. Initial hypotheses are the previously blinded '
        'seed0 priors, explicitly reused. All-query feature-keyed exact-tie order changes with seed. Native GP fitting and MC512 '
        'streams stay numerical seed0. New own-history LLM refresh responses are independently generated and seed-isolated.','',
        'This is conditional robustness with fixed initial knowledge, not a test of new LLM initialization quality. Tie-order '
        'and downstream refresh-response randomness are not separated. Seed0 is diagnosed development evidence; seeds1-3 '
        'are also reported separately. Four replicates do not establish statistical superiority. No policy repair was introduced.','',
        'SearchAUC is the arithmetic mean of20 post-query cumulative maxima, including paid query1, zeros and nonimprovements. '
        'Final best is the maximum across those20 observations. SD is sample standard deviation (ddof=1), not a confidence interval. '
        'The first_final_best_query column is the first attainment of that trajectory\'s OWN final best, which may differ across '
        'methods. Original task units are preserved and raw task scores are not averaged across different objectives.','']
    for cohort in ('all_0_1_2_3','new_1_2_3'):
        lines+= [cohort.upper(),'', 'Task | BO AUC mean +/- SD | LLM-first AUC mean +/- SD | Hyp AUC mean +/- SD | Hyp-BO W/T/L | Hyp-first W/T/L']
        for _,r in aggregate[aggregate.cohort==cohort].iterrows():
            values=[f"{r[f'{a}_SearchAUC_mean']:.6f} +/- {r[f'{a}_SearchAUC_SD']:.6f}" for a in s.ARMS]
            lines.append(' | '.join([NAMES[r.task],*values,r.W_T_L_vs_BO,r.W_T_L_vs_same_first]))
        lines+= ['', 'Task | BO final mean +/- SD | LLM-first final mean +/- SD | Hyp final mean +/- SD']
        for _,r in aggregate[aggregate.cohort==cohort].iterrows():
            values=[f"{r[f'{a}_final_best_mean']:.6f} +/- {r[f'{a}_final_best_SD']:.6f}" for a in s.ARMS]
            lines.append(' | '.join([NAMES[r.task],*values]))
        lines.append('')
    lines+= ['PER-SEED PAIRED AUC CHANGES','',pairs.to_string(index=False),'',
        'MECHANISM EXPOSURE (COUNTS ARE NOT CAUSAL BENEFIT)','',pd.DataFrame(mechanisms).to_string(index=False),'',
        'With fixed starts, tasks having no acquisition ties can have identical LLM-first BO trajectories across all seeds. '
        'The unique_paths columns disclose the number of distinct selected-input sequences; zero SD in that setting does '
        'not show robustness to different initial observations or initial hypotheses.','',
        'The original older cold_s123 controls are not substituted here: their seeds changed only query1, whereas this study '
        'changes every query\'s exact-tie order. Both comparator arms were recomputed with the matching tie streams. '
        'Only exact observation-prefix numeric caches are reused, under fixed numerical recipes.','',
        'Eight persistent CPU workers, one numerical thread each, warmed once. Actual independent experimental LLMs use '
        'gpt-6.1-terra medium. No GP ranks or unmeasured outcomes enter refresh prompts. Agent completion is acknowledged '
        'before raw-output sealing. Initial-prior reuse is explicit and is not counted as a fresh LLM call.','']
    if (s.REPORT/'audit.json').exists():
        audit=s.read(s.REPORT/'audit.json');lines+= ['AUDIT','',json.dumps({a:b for a,b in audit.items() if a!='details'},indent=2),'']
    (s.REPORT/'report.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axs=plt.subplots(4,2,figsize=(12,11),constrained_layout=True)
    for ax,task in zip(axs.flat,s.SHORT.values()):
        f=pairs[pairs.task==task].sort_values('seed');values=f.delta_AUC_vs_same_first.to_numpy()
        bars=ax.bar(range(4),values,color=['#087f8c' if v>1e-8 else '#b55353' if v<-1e-8 else '#9099a5' for v in values],width=.6)
        bars[0].set_hatch('///');bars[0].set_edgecolor('#444e63');ax.axhline(0,color='#444e63',linewidth=.9)
        extent=max(float(np.max(abs(values))),.1);ax.set_ylim(min(float(values.min()),0)-extent*.22,max(float(values.max()),0)+extent*.24)
        for bar,value in zip(bars,values):ax.annotate(f'{value:+.3f}',(bar.get_x()+bar.get_width()/2,value),xytext=(0,5 if value>=0 else -5),
            textcoords='offset points',ha='center',va='bottom' if value>=0 else 'top',fontsize=9)
        ax.set_xticks(range(4),['0 (old)','1','2','3']);ax.set_title(NAMES[task],fontsize=11);ax.set_xlabel('Tie stream seed');ax.grid(axis='y',alpha=.16)
        ax.set_ylabel('Paired SearchAUC change')
    fig.suptitle('Frozen hypothesis policy minus same LLM first point + BO',fontsize=14)
    fig.savefig(s.REPORT/'paired_SearchAUC.png',dpi=180);fig.savefig(s.REPORT/'paired_SearchAUC.pdf');plt.close(fig)
    fig,axs=plt.subplots(4,2,figsize=(12,12),constrained_layout=True)
    colors=dict(bo='#444e63',first='#be7c19',hyp='#087f8c')
    for ax,task in zip(axs.flat,s.SHORT.values()):
        for arm in s.ARMS:
            y=np.asarray([histories[task,seed,arm] for seed in (0,1,2,3)]);mean=y.mean(axis=0);sd=y.std(axis=0,ddof=1)
            ax.plot(range(1,21),mean,color=colors[arm],label=LABELS[arm],linewidth=1.7)
            ax.fill_between(range(1,21),mean-sd,mean+sd,color=colors[arm],alpha=.10)
        ax.set_title(NAMES[task]);ax.set_xticks([1,4,8,12,16,20]);ax.set_xlabel('Paid query');ax.set_ylabel('Best yield (%)' if task not in ('snar','hplc') else ('Space-time yield' if task=='snar' else 'Peak area'))
        ax.grid(alpha=.16)
    h,l=axs.flat[0].get_legend_handles_labels();fig.legend(h,l,loc='outside upper center',ncol=3,frameon=False)
    fig.savefig(s.REPORT/'mean_cumulative_best.png',dpi=180);fig.savefig(s.REPORT/'mean_cumulative_best.pdf');plt.close(fig)
    print(aggregate[aggregate.cohort=='all_0_1_2_3'].to_string(index=False));print('Complete:',s.REPORT)


if __name__=='__main__':main()
