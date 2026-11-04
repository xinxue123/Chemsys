"""Scientific panels and exact tables for the separately identified subagent cohort."""
from pathlib import Path
from hashlib import sha256
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'paper/v12_subagent'
COL={'V12':'#006d89','Reasoning BO':'#c77d28','BO':'#626a73','No KF':'#9d5475','No refresh':'#70843d'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
def esc(s):return str(s).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
def save(fig,name):
    fig.savefig(P/'figures'/f'{name}.pdf',bbox_inches='tight');fig.savefig(P/'figures'/f'{name}.png',dpi=180,bbox_inches='tight');plt.close(fig)
def table(name,caption,headers,rows,align):
    text=r'\begingroup\small\setlength{\tabcolsep}{4pt}'+'\n'+r'\begin{longtable}{'+align+'}\n'
    text+=r'\caption{'+caption+r'}\\\toprule'+'\n'+' & '.join(headers)+r'\\\midrule\endfirsthead'+'\n'
    text+=r'\toprule '+' & '.join(headers)+r'\\\midrule\endhead'+'\n'
    text+='\n'.join(' & '.join(map(str,row))+r'\\' for row in rows)
    text+='\n'+r'\bottomrule\end{longtable}\endgroup'+'\n'
    (P/'generated'/name).write_text(text,encoding='utf-8')
def main():
    e=json.loads((P/'data/evidence.json').read_text(encoding='utf-8'));rows=e['rows'];pairs=e['paired'];tasks=[d['task'] for d in e['datasets']];labels={d['task']:d['label'] for d in e['datasets']}
    for name in ('figures','generated'):(P/name).mkdir(parents=True,exist_ok=True)
    lookup={(r['task'],r['initialization_seed'],r['method']):r for r in rows}
    get=lambda task,seed,method:lookup.get((task,seed,method))
    fmt=lambda x:'--' if x is None else f'{x:.3f}'
    starts=[]
    for task in tasks:
        controls=[get(task,seed,'BO') for seed in (1,2,3)]
        if not all(controls):raise ValueError('All primary starting controls must be present')
        starts.append([esc(labels[task])]+[fmt(r['initial_best']) for r in controls]+
                      [fmt(float(np.mean([r['final_best']-r['initial_best'] for r in controls])))])
    table('starting_conditions.tex','Best of the four shared initial observations at each primary initialization, and mean final-best gain along the three frozen BO trajectories. Values use task-native units. The gain is relative to the observed starting best, not to a known global optimum.',
          ['Task','Init. 1 best','Init. 2 best','Init. 3 best','Mean BO gain'],starts,'lrrrr')
    lines=[]
    for task in tasks:
        for seed in (0,1,2,3):
            for method in ('BO','V12','Reasoning BO','No KF','No refresh'):
                if method=='Reasoning BO' and seed==0:continue
                if method in ('No KF','No refresh') and (task not in ('Reizman_Suzuki_Case_2','Reizman_Suzuki_Case_4','HPLC') or seed not in (0,3)):continue
                r=get(task,seed,method)
                lines.append([esc(labels[task]),seed,method,fmt(r['SearchAUC']) if r else '--',fmt(r['final_best']) if r else '--',r['first_best_query'] if r else '--'])
    table('subagent_all.tex','Every registered subagent cell and its matched frozen BO control. Dashes are pending results, never imputed values. Objectives retain native units; BO controls are reused rather than rerun.',['Task','Init.','Method','SearchAUC','Final best','First best'],lines,'lllr rr'.replace(' ',''))
    lines=[]
    for r in pairs:
        if r['comparison'] not in ('KF_removal','refresh_removal'):continue
        removed=get(r['task'],r['seed'],r['control'])
        if removed is None:raise ValueError('Missing removal activity record')
        lines.append([esc(r['label']),r['seed'],r['control'],fmt(r['SearchAUC']),fmt(r['control_SearchAUC']),f"{r['AUC_delta']:+.3f}",f"{r['best_delta']:+.3f}",removed['forecasts'],removed['edits']])
    table('removal_pairs.tex','Full V12 minus each removal under the same subagent backend and initial observations. Each cell uses independent responses; these are whole-trajectory contrasts rather than an isolated causal effect. Forecasts and note calls count removal-arm forecast events and explicit knowledge-edit requests. A no-KF forecast is scored without applying a numerical filter update.',['Task','Init.','Removal','V12 AUC','Removal AUC',r'$\Delta$ AUC',r'$\Delta$ best','Forecasts','Note calls'],lines,'lllrrrrrr')
    lines=[]
    for r in rows:
        if r['method']!='V12':continue
        lines.append([esc(r['label']),r['initialization_seed'],r['forecasts'],r['weight_decreases'],r['pauses'],r['retractions'],r['resumes'],r['edits'],r['different_from_same_history_BO']])
    table('subagent_mechanisms.tex','Executed V12 forecast and lifecycle events. Changes in selection compare with the top acquisition point under the same current history, not the independent BO trajectory.',['Task','Init.','Forecast','Down','Pause','Retract','Resume','Edits','Non-top'],lines,'llrrrrrrr')
    # Paired data are individual initializations; lines show a descriptive zero reference.
    fig,axes=plt.subplots(3,1,figsize=(6.8,6.1),sharex=True,layout='constrained')
    kinds=('V12_vs_BO','RBO_vs_BO','V12_vs_RBO')
    titles=('V12 versus BO','Reasoning BO versus BO','V12 versus Reasoning BO')
    for k,(ax,kind,title) in enumerate(zip(axes,kinds,titles)):
        ax.axhline(0,color='#9aa0a5',lw=.8);ax.grid(axis='y',alpha=.15)
        for i,task in enumerate(tasks):
            rr=[r for r in pairs if r['task']==task and r['comparison']==kind and r['seed'] in (1,2,3)]
            for r in rr:ax.scatter(i+.15*(r['seed']-2),r['AUC_percent'],marker=('o','s','^')[r['seed']-1],s=28,color=COL['Reasoning BO'] if k==1 else COL['V12'])
            if len(rr)==3:ax.plot([i-.25,i+.25],[np.mean([r['AUC_percent'] for r in rr])]*2,color='#222222',lw=1.3)
        ax.set_title(title,loc='left',fontsize=10);ax.set_ylabel('SearchAUC difference (%)');ax.text(-.08,1.03,chr(97+k),transform=ax.transAxes,fontweight='bold')
    axes[-1].set_xticks(range(8),[labels[t] for t in tasks],rotation=30,ha='right');save(fig,'paired_search_efficiency')
    fig,axes=plt.subplots(2,4,figsize=(6.8,4.3),layout='constrained')
    for ax,task in zip(axes.flat,tasks):
        for method in ('BO','Reasoning BO','V12'):
            rr=[get(task,s,method) for s in (1,2,3)]
            if not all(rr):continue
            curves=np.array([r['best_curve'] for r in rr]);mean=curves.mean(axis=0);sd=curves.std(axis=0,ddof=1)
            ax.plot(range(17),mean,color=COL[method],label=method,lw=1.4)
            ax.fill_between(range(17),mean-sd,mean+sd,color=COL[method],alpha=.10,lw=0)
        ax.set_title(labels[task]);ax.set_xlabel('Query');ax.set_xticks((0,4,8,12,16));ax.grid(alpha=.12)
    handles,legend_labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc='upper center',bbox_to_anchor=(.5,1.10),ncol=3,fontsize=8.5,frameon=False)
    axes[0,0].set_ylabel('Cumulative best\n(native units)');axes[1,0].set_ylabel('Cumulative best\n(native units)');save(fig,'eight_task_curves')
    focus=('Reizman_Suzuki_Case_2','Reizman_Suzuki_Case_4','HPLC')
    fig,axes=plt.subplots(2,3,figsize=(6.8,4.0),layout='constrained')
    for col,task in enumerate(focus):
        for row,seed in enumerate((0,3)):
            ax=axes[row,col]
            for method in ('BO','V12','No KF','No refresh'):
                r=get(task,seed,method)
                if r:ax.plot(range(17),r['best_curve'],color=COL[method],label=method,lw=1.5)
            ax.set_title(f'{labels[task]}, init. {seed}');ax.set_xlabel('Query');ax.set_xticks((0,4,8,12,16));ax.grid(alpha=.12)
    handles,legend_labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc='upper center',bbox_to_anchor=(.5,1.10),ncol=4,fontsize=8.5,frameon=False)
    save(fig,'removal_trajectories')
    # Final best is shown separately from integrated search performance.
    fig,axes=plt.subplots(1,2,figsize=(6.8,3.3),layout='constrained')
    for ax,kind,title in zip(axes,('V12_vs_BO','V12_vs_RBO'),('V12 versus BO','V12 versus Reasoning BO')):
        ax.axhline(0,color='#9aa0a5',lw=.8)
        for i,task in enumerate(tasks):
            for r in pairs:
                if r['task']==task and r['comparison']==kind and r['seed'] in (1,2,3):
                    pct=100*r['best_delta']/r['control_final_best']
                    ax.scatter(i+.15*(r['seed']-2),pct,s=24,marker=('o','s','^')[r['seed']-1],color=COL['V12'])
        ax.set_title(title);ax.set_ylabel('Final-best difference (%)');ax.set_xticks(range(8),[labels[t] for t in tasks],rotation=40,ha='right');ax.grid(axis='y',alpha=.15)
    save(fig,'final_best')
    files=[P/'figures'/f'{name}.{extension}' for name in ('paired_search_efficiency','eight_task_curves','removal_trajectories','final_best') for extension in ('pdf','png')]
    files += [P/'generated'/name for name in ('starting_conditions.tex','subagent_all.tex','removal_pairs.tex','subagent_mechanisms.tex')]
    receipt=dict(evidence_sha256=sha256((P/'data/evidence.json').read_bytes()).hexdigest(),generator_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                 files={f.relative_to(P).as_posix():sha256(f.read_bytes()).hexdigest() for f in files})
    (ROOT/'experiments/reports/paper_v12_subagent/figure_generation.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(dict(complete=e['completed_trajectories'],figures=4,tables=4)))
if __name__=='__main__':main()
