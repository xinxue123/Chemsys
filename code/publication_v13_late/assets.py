"""Reproducible phase and intervention assets from audited observation exports."""
from collections import defaultdict
from pathlib import Path
import csv
import hashlib
import json
import shutil
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'paper/v13_late'
R=ROOT/'experiments/reports/v13_late_validation_20261103'
OLD=ROOT/'paper/v13'
TASKS=('c1','c2','c3','c4','perera','doyle','snar','hplc')
NAMES=dict(zip(TASKS,('Suzuki 1','Suzuki 2','Suzuki 3','Suzuki 4','Perera','Doyle','SnAr','HPLC')))
NAMES['aryl']='Direct Arylation'
LABELS=dict(bo='BO',reasoning_bo='Reasoning BO',v13='V13',rbo='Reasoning BO',full='V13',
    kf_fixed='Fixed KF',no_refresh='No refresh',coupled='Coupled lifecycle')
COLORS=dict(bo='#647382',reasoning_bo='#BC7030',v13='#008577',rbo='#BC7030',full='#008577',
    kf_fixed='#9165A3',no_refresh='#3C83AC',coupled='#B54E59')


def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def put(name,value):
    path=P/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(value+'\n',encoding='utf-8')


def table(headers,rows,align=None):
    return '\\begin{tabular}{'+(align or 'l'+'r'*(len(headers)-1))+'}\\toprule\n'+ \
        ' & '.join(headers)+r'\\\midrule'+'\n'+ '\n'.join(' & '.join(map(str,row))+r'\\' for row in rows)+ \
        '\n'+r'\bottomrule\end{tabular}'


def save(fig,name):
    d=P/'figures';d.mkdir(exist_ok=True)
    fig.savefig(d/(name+'.pdf'),bbox_inches='tight',facecolor='white')
    fig.savefig(d/(name+'.png'),bbox_inches='tight',facecolor='white',dpi=170)
    plt.close(fig)


def stage():
    P.mkdir(parents=True,exist_ok=True)
    # Explicit copies only; no historical file is edited or moved.
    for folder in ('data','generated','figures','protocols'):
        shutil.copytree(OLD/folder,P/folder,dirs_exist_ok=True)
    for name in ('preamble.tex','references.bib','methods.tex'):
        shutil.copy2(OLD/name,P/name)
    audit=P/'historical_audit'
    shutil.copytree(OLD/'audit',audit,dirs_exist_ok=True)
    for name in ('historical_phase_metrics.csv','historical_phase_summary.csv','historical_late_record_events.csv','phase_analysis_receipt.json'):
        shutil.copy2(R/name,P/'data'/name)
    ledger=ROOT/'paper/v12_subagent/data/all_experiment_records.csv'
    shutil.copy2(ledger,P/'data/development_catalogue_at_V12_cutoff.csv')
    (P/'audit').mkdir(exist_ok=True)


def primary(rows,regime):
    seeds=(1,2,3) if regime=='warm' else (0,1,2,3)
    return [r for r in rows if r['regime']==regime and int(r['seed']) in seeds and r['method'] in ('bo','reasoning_bo','v13')]


def phase_tables(rows):
    facts={}
    for regime in ('warm','cold'):
        primary_rows=primary(rows,regime)
        data={(r['task'],r['method'],int(r['seed'])):r for r in primary_rows}
        seeds=(1,2,3) if regime=='warm' else (0,1,2,3)
        terminal=[];gains=[];targets=[]
        for task in TASKS:
            rr={m:[data[task,m,s] for s in seeds] for m in ('bo','reasoning_bo','v13')}
            values={m:st.mean(float(x['final_best']) for x in rs) for m,rs in rr.items()}
            best=max(values.values())
            final_cell=lambda m:('\\textbf{'+f'{values[m]:.2f}'+'}') if abs(values[m]-best)<1e-8 else f'{values[m]:.2f}'
            terminal.append([NAMES[task]]+[final_cell(m) for m in rr]+[f"{st.mean(float(x['M_cutoff']) for x in rr['v13']):.2f}",
                f"{st.mean(float(x['late_improvement']) for x in rr['v13']):.2f}"])
            gains.append([NAMES[task]]+[f"{st.mean(float(x['late_improvement']) for x in rr[m]):.2f}" for m in rr]+ \
                [f"{st.mean(float(x['late_incremental_SearchAUC']) for x in rr[m]):.2f}" for m in rr])
            targets.append([NAMES[task],f"{float(rr['bo'][0]['target_95percent_pool_max']):.2f}"]+ \
                [f"{sum(int(x['target_hit']) for x in rr[m])}/{len(seeds)}" for m in rr])
        put('generated/'+regime+'_terminal.tex',table(['Task','BO','Reasoning BO','V13',r'V13 $M_\tau$',r'V13 $\Delta_{\rm late}$'],terminal))
        put('generated/'+regime+'_late_gains.tex',table(['Task',r'$\Delta$: BO',r'$\Delta$: RBO',r'$\Delta$: V13',r'$I$: BO',r'$I$: RBO',r'$I$: V13'],gains))
        put('generated/'+regime+'_targets.tex',table(['Task','Target','BO','Reasoning BO','V13'],targets))
        if regime=='cold':
            native=read(P/'data/summary.csv');cells=[]
            for task in TASKS:
                rs={m:next(x for x in native if x['cohort']=='cold_multistart' and x['task']==task and x['method']==m) for m in ('bo','reasoning_bo','v13')}
                # The archive uses a stable cohort label; fail rather than guess
                # if future exports change that label.
                top_auc=max(float(x['mean_SearchAUC']) for x in rs.values())
                top_best=max(float(x['mean_final_best']) for x in rs.values())
                row=[NAMES[task]]
                for x in rs.values():
                    a=f"{float(x['mean_SearchAUC']):.2f}";b=f"{float(x['mean_final_best']):.2f}"
                    if abs(float(x['mean_SearchAUC'])-top_auc)<1e-8:a=r'\textbf{'+a+'}'
                    if abs(float(x['mean_final_best'])-top_best)<1e-8:b=r'\textbf{'+b+'}'
                    row.append(a+' / '+b)
                cells.append(row)
            put('generated/cold_joint.tex',table(['Task','BO','Reasoning BO','V13'],cells))
        facts[regime]={}
        for comparator in ('bo','reasoning_bo'):
            for metric in ('final_best','late_improvement','late_incremental_SearchAUC','normalized_terminal_regret','full_SearchAUC'):
                diffs=[float(data[t,'v13',s][metric])-float(data[t,comparator,s][metric]) for t in TASKS for s in seeds]
                if metric=='normalized_terminal_regret':diffs=[-d for d in diffs]
                counts=[sum(d>1e-8 for d in diffs),sum(abs(d)<=1e-8 for d in diffs),sum(d< -1e-8 for d in diffs)]
                facts[regime][comparator+'_'+metric]=dict(wins=counts[0],ties=counts[1],losses=counts[2])
        facts[regime]['targets']={m:sum(int(x['target_hit']) for x in primary_rows if x['method']==m) for m in ('bo','reasoning_bo','v13')}
    put('data/phase_facts.json',json.dumps(facts,indent=2))


def phase_figures(rows):
    rr=read(P/'data/rounds.csv')
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(8.5,5.7))
    for ax,task in zip(axes.flat,('perera','doyle','hplc','c4')):
        for method in ('bo','v13','reasoning_bo'):
            yy=[]
            for seed in range(4):
                values=sorted((x for x in rr if x['regime']=='cold' and x['task']==task and x['method']==method and int(x['seed'])==seed),key=lambda x:int(x['query']))
                arr=np.asarray([float(x['cumulative_best']) for x in values])
                yy.append(arr[9:]-arr[9])
            y=np.asarray(yy);x=np.arange(10,21)
            ax.plot(x,y.mean(0),lw=1.7,color=COLORS[method],label=LABELS[method])
            ax.fill_between(x,y.min(0),y.max(0),alpha=.09,color=COLORS[method])
        ax.set(title=NAMES[task],xlabel='Paid experiment',ylabel='Late gain (peak-area units)' if task=='hplc' else 'Late gain (percentage points)',xticks=(10,12,14,16,18,20))
        ax.grid(alpha=.16)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False)
    fig.tight_layout(rect=(0,0,1,.95));save(fig,'late_own_prefix')
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.6),sharey=True)
    for ax,regime in zip(axes,('warm','cold')):
        rs=primary(rows,regime);n=3 if regime=='warm' else 4
        x=np.arange(8)
        for k,method in enumerate(('bo','reasoning_bo','v13')):
            hits=[sum(int(r['target_hit']) for r in rs if r['task']==t and r['method']==method) for t in TASKS]
            ax.bar(x+(k-1)*.24,np.asarray(hits)/n,width=.22,color=COLORS[method],label=LABELS[method])
        ax.set(title=('Four initial + 16 queries' if regime=='warm' else 'Zero initial + 20 queries'),
            ylabel='Fraction reaching 95% of pool maximum',xticks=x,xticklabels=[NAMES[t].replace('Suzuki ','S') for t in TASKS],ylim=(0,1.13))
        ax.tick_params(axis='x',rotation=40);ax.grid(axis='y',alpha=.16)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False)
    fig.tight_layout(rect=(0,0,1,.93));save(fig,'target_attainment')


def historical_mechanisms():
    rounds=read(P/'data/rounds.csv');forecasts=read(P/'data/forecast_calculations.csv')
    fig,axes=plt.subplots(2,2,figsize=(8.5,5.5),gridspec_kw={'height_ratios':[1.3,1]})
    table_rows=[]
    for col,(task,seed) in enumerate((('hplc',3),('c4',0))):
        for method in ('bo','v13','reasoning_bo'):
            a=sorted((x for x in rounds if x['regime']=='cold' and x['task']==task and int(x['seed'])==seed and x['method']==method),key=lambda x:int(x['query']))
            axes[0,col].plot([int(x['query']) for x in a],[float(x['cumulative_best']) for x in a],color=COLORS[method],label=LABELS[method],lw=1.4 if method=='reasoning_bo' else 1.7,ls='--' if method=='reasoning_bo' else '-')
        axes[0,col].axvline(10,color='#BCC5CE',ls=':',lw=1)
        axes[0,col].set(title=f'{NAMES[task]}, start {seed}',ylabel='Best peak area' if task=='hplc' else 'Best yield (%)',xticks=(1,5,10,15,20));axes[0,col].grid(alpha=.15)
        ff=sorted((x for x in forecasts if x['regime']=='cold' and x['task']==task and int(x['seed'])==seed and x['note_id']=='H001' and int(x['version'])==1),key=lambda x:int(x['query']))
        axes[1,col].plot([int(x['query']) for x in ff],[float(x['reliability_after']) for x in ff],color=COLORS['v13'],lw=1)
        for improved,marker,label in (('1','^','Measured improvement'),('0','x','No improvement')):
            subset=[x for x in ff if x['improved']==improved]
            axes[1,col].scatter([int(x['query']) for x in subset],[float(x['reliability_after']) for x in subset],s=32,marker=marker,color=COLORS['v13'],label=label,zorder=3)
        axes[1,col].set(xlim=(.7,20.3),ylim=(-.03,.84),xlabel='Paid experiment',ylabel='H001 v1 reliability after feedback',xticks=(1,5,10,15,20));axes[1,col].grid(alpha=.15)
        for x in ff:
            if int(x['query'])>=10:
                table_rows.append([NAMES[task],seed,x['query'],x['use'].replace('_',r'\_'),f"{float(x['p_LLM']):.3f}",f"{float(x['p_GP']):.3f}",
                    x['improved'],f"{float(x['Brier_gain']):+.3f}",f"{float(x['reliability_before']):.3f} $\\to$ {float(x['reliability_after']):.3f}",
                    x['permission_before']+r' $\to$ '+x['permission_after']])
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False)
    handles,labels=axes[1,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False)
    fig.tight_layout(rect=(0,.06,1,.94));save(fig,'late_mechanism_examples')
    put('generated/historical_late_forecasts.tex',table(['Task','Start','Query','Use',r'$p_{\rm LLM}$',r'$p_{\rm GP}$',r'$s$',r'$z$',r'$r$ before/after','Permission'],table_rows,'lll lrrrrll'.replace(' ','')))


def fresh_assets():
    audit=json.loads((R/'final_audit.json').read_text(encoding='utf-8'))
    if audit['status']!='PASS_COMPLETE' or audit['complete_verified']!=75 or audit['newly_selected_queries_verified']!=840:
        raise ValueError('Do not publish incomplete or failed experiments')
    for name in ('final_audit.json','final_metrics.csv','final_rounds.csv'):
        shutil.copy2(R/name,P/'data'/name)
    metrics=read(P/'data/final_metrics.csv');rounds=read(P/'data/final_rounds.csv')
    for task in ('c4','perera','hplc'):
        tab=[]
        for seed in (0,3):
            row=[str(seed)]
            for arm in ('bo','rbo','full','kf_fixed','no_refresh','coupled'):
                a=[x for x in metrics if x['kind']=='prefix' and x['task']==task and int(x['prefix_seed'])==seed and x['arm']==arm]
                av=st.mean(float(x['continuation_SearchAUC']) for x in a);best=st.mean(float(x['final_best']) for x in a)
                row.append(f'{av:.2f} / {best:.2f}')
            tab.append(row)
        put('generated/prefix_'+task+'.tex',table(['Prefix','BO','RBO','V13','Fixed KF','No refresh','Coupled'],tab))
    effects=[]
    effect_rows=[]
    for task in ('c4','perera','hplc'):
        for seed in (0,3):
            a={arm:[x for x in metrics if x['kind']=='prefix' and x['task']==task and int(x['prefix_seed'])==seed and x['arm']==arm] for arm in ('bo','rbo','full','kf_fixed','no_refresh','coupled')}
            full={k:st.mean(float(x[k]) for x in a['full']) for k in ('continuation_SearchAUC','final_best')}
            row=[NAMES[task],str(seed)]
            for arm in ('bo','rbo','kf_fixed','no_refresh','coupled'):
                da=full['continuation_SearchAUC']-st.mean(float(x['continuation_SearchAUC']) for x in a[arm])
                db=full['final_best']-st.mean(float(x['final_best']) for x in a[arm])
                row.append(f'{da:+.2f} / {db:+.2f}')
                effects.append(dict(task=task,prefix_seed=seed,comparator=arm,delta_continuation_SearchAUC=da,delta_final_best=db,
                    full_n=len(a['full']),comparator_n=len(a[arm])))
            effect_rows.append(row)
    put('generated/prefix_effects.tex',table(['Task','Prefix','vs BO','vs RBO','vs fixed KF','vs no refresh','vs coupled'],effect_rows))
    with (P/'data/conditional_effects.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(effects[0]));w.writeheader();w.writerows(effects)
    facts={arm:{metric:{'wins':sum(x[metric]>1e-8 for x in effects if x['comparator']==arm),
        'ties':sum(abs(x[metric])<=1e-8 for x in effects if x['comparator']==arm),
        'losses':sum(x[metric]< -1e-8 for x in effects if x['comparator']==arm)}
        for metric in ('delta_continuation_SearchAUC','delta_final_best')}
        for arm in ('bo','rbo','kf_fixed','no_refresh','coupled')}
    put('data/conditional_facts.json',json.dumps(facts,indent=2))
    detail=[]
    for task in ('c4','perera','hplc'):
        for seed in (0,3):
            for arm in ('bo','rbo','full','kf_fixed','no_refresh','coupled'):
                for x in [r for r in metrics if r['kind']=='prefix' and r['task']==task and int(r['prefix_seed'])==seed and r['arm']==arm]:
                    detail.append([NAMES[task],seed,LABELS[arm],x['repetition'],f"{float(x['continuation_SearchAUC']):.2f}",f"{float(x['final_best']):.2f}",
                        f"{float(x['continuation_improvement']):.2f}",x['forecast_events'],x['editor_calls']])
    # Chunk by task to avoid a huge scaled, unreadable table.
    for task in ('c4','perera','hplc'):
        put('generated/prefix_all_'+task+'.tex',table(['Task','Prefix','Arm','Rep.','Tail AUC','Final best','Late gain','Forecasts','Edits'],[r for r in detail if r[0]==NAMES[task]],'lllrrrrrr'))
    holdout=[]
    for seed in range(3):
        row=[str(seed)]
        for arm in ('bo','rbo','full'):
            x=next(x for x in metrics if x['kind']=='holdout' and int(x['prefix_seed'])==seed and x['arm']==arm)
            row.append(f"{float(x['SearchAUC20']):.2f} / {float(x['final_best']):.2f}")
        holdout.append(row)
    put('generated/holdout.tex',table(['Seed','BO','Reasoning BO','V13'],holdout))
    fig,axes=plt.subplots(3,2,figsize=(8.6,8.4))
    for row,task in enumerate(('c4','perera','hplc')):
        for col,seed in enumerate((0,3)):
            ax=axes[row,col]
            for arm in ('bo','rbo','full','kf_fixed','no_refresh','coupled'):
                jobs=[x for x in metrics if x['kind']=='prefix' and x['task']==task and int(x['prefix_seed'])==seed and x['arm']==arm]
                yy=[]
                for job in jobs:
                    rr=sorted((x for x in rounds if x['job']==job['job']),key=lambda x:int(x['query']))
                    yy.append([float(job['prefix_best'])]+[float(x['incumbent_after']) for x in rr])
                y=np.asarray(yy);ax.plot(np.arange(10,21),y.mean(0),color=COLORS[arm],lw=1.8 if arm=='full' else 1.1,
                    ls='-' if arm in ('bo','rbo','full') else '--',label=LABELS[arm])
                if len(y)>1:ax.fill_between(np.arange(10,21),y.min(0),y.max(0),color=COLORS[arm],alpha=.055)
            ax.set(title=f'{NAMES[task]}, prefix {seed}',xlabel='Total paid experiments',ylabel='Best peak area' if task=='hplc' else 'Best yield (%)',xticks=(10,15,20));ax.grid(alpha=.15)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False)
    fig.tight_layout(rect=(0,0,1,.94));save(fig,'common_prefix_curves')
    fig,ax=plt.subplots(figsize=(6.8,3.3))
    for arm in ('bo','rbo','full'):
        y=np.asarray([[float(x['incumbent_after']) for x in sorted((r for r in rounds if r['kind']=='holdout' and r['arm']==arm and int(r['prefix_seed'])==s),key=lambda r:int(r['query']))] for s in range(3)])
        ax.plot(np.arange(1,21),y.mean(0),label=LABELS[arm],color=COLORS[arm],lw=1.8)
        ax.fill_between(np.arange(1,21),y.mean(0)-y.std(0,ddof=1),y.mean(0)+y.std(0,ddof=1),alpha=.10,color=COLORS[arm])
    ax.set(xlabel='Paid experiments from zero observations',ylabel='Best yield (%)',xticks=(1,5,10,15,20));ax.grid(alpha=.15);ax.legend(frameon=False,ncol=3)
    fig.tight_layout();save(fig,'holdout_curves')


def main(fresh=False):
    stage();rows=read(P/'data/historical_phase_metrics.csv');phase_tables(rows);phase_figures(rows);historical_mechanisms()
    if fresh:fresh_assets()
    print(json.dumps(dict(status='ASSETS_CREATED',namespace='paper/v13_late',fresh_results_included=fresh)))


if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('--fresh',action='store_true');args=a.parse_args();main(args.fresh)
