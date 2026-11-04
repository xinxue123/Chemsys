"""Generate publication tables and vector figures from the verified V13 export."""
from pathlib import Path
import argparse
import csv
import json
import statistics
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'paper/v13'
TASKS=('c1','c2','c3','c4','perera','doyle','snar','hplc')
NAMES=dict(zip(TASKS,('Suzuki 1','Suzuki 2','Suzuki 3','Suzuki 4','Perera','Doyle','SnAr','HPLC')))
METHODS=('bo','reasoning_bo','v13')
LABELS=dict(bo='BO',reasoning_bo='Reasoning BO',v13='V13',v12_subagent='V12')
COLORS=dict(bo='#5D6C7B',reasoning_bo='#C67532',v13='#008577',v12_subagent='#8B6BA5')


def read(name):
    with (P/'data'/name).open(encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f))


def put(name,text):
    p=P/'generated'/name
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(text+'\n',encoding='utf-8')


def table(headers,rows,widths=None):
    spec=widths or ('l'+'r'*(len(headers)-1))
    return '\\begin{tabular}{'+spec+'}\\toprule\n'+' & '.join(headers)+r'\\\midrule'+'\n'+'\n'.join(' & '.join(map(str,r))+r'\\' for r in rows)+'\n'+r'\bottomrule\end{tabular}'


def save(fig,name):
    folder=P/'figures';folder.mkdir(parents=True,exist_ok=True)
    fig.savefig(folder/(name+'.pdf'),bbox_inches='tight',facecolor='white')
    fig.savefig(folder/(name+'.png'),dpi=170,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def workflow():
    fig,ax=plt.subplots(figsize=(8.2,4.75));ax.set(xlim=(0,10),ylim=(0,5));ax.axis('off')
    fig.subplots_adjust(left=.02,right=.98,bottom=.02,top=.98)
    constraints=[]
    def box(x,y,w,h,title,body,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.10,rounding_size=.08',facecolor=color,edgecolor='#B6C3CC',lw=.8))
        title_text=ax.text(x+.12,y+h-.17,title,fontsize=10.5,fontweight='bold',va='top',color='#1F3648')
        body_text=ax.text(x+.12,y+h-.52,body,fontsize=9.5,va='top',linespacing=1.5,color='#294657')
        constraints.extend([(title_text,x+w-.05),(body_text,x+w-.05)])
    def arrow(a,b,label=None):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=11,color='#6A8190',lw=1.2))
        if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.12,label,fontsize=8,ha='center',color='#526675')
    box(.15,3.12,2.5,1.5,'1  GP menu','Own data → native GP\nqLogEI → top five\nHide numeric GP scores','#EEF2F7')
    box(3.12,3.12,2.6,1.5,'2  Commit choice','Official confidence rule\nNotes + use + forecast\nCommit before outcome','#EAF5F1')
    box(6.28,3.12,3.45,1.5,'3  Observe outcome','Reveal one pool outcome\nScore LLM against GP\nUpdate both channels','#EEF2F7')
    arrow((2.75,3.87),(3.02,3.87));arrow((5.82,3.87),(6.18,3.87))
    box(.65,.45,3.8,1.8,'Predictive reliability','Relative Brier gain → scalar KF\nAccurate failure can improve reliability\nReliability enters the next LLM context\nGP mean and covariance stay unchanged','#E7F3F2')
    box(5.05,.45,4.05,1.8,'Recommendation permission','Tentative / allowed / suppressed / review\nAccurate failure inhibits promotion\nReview: stagnation or condition change\nRestore: two improving review tests','#FFF3E8')
    arrow((7.8,3.02),(7.8,2.35));arrow((6.75,3.02),(2.55,2.35))
    ax.text(5,0.03,'Event editing: retain / revise / retract / add; adverse facts survive applicable version changes.',ha='center',fontsize=9,color='#526675')
    ax.text(.15,4.85,'A credible failure forecast does not justify a positive recommendation.',fontsize=11,fontweight='bold',color='#1F3648')
    fig.canvas.draw()
    for t,right in constraints:
        if t.get_window_extent().x1>ax.transData.transform((right,0))[0]:
            raise ValueError('Workflow text exceeds its box: '+t.get_text())
    save(fig,'workflow')


def contrasts(paired):
    fig,axes=plt.subplots(1,2,figsize=(8.2,4.7),sharey=True)
    for ax,regime,seeds,title in zip(axes,('warm','cold'),((1,2,3),(0,1,2,3)),('a   Four shared observations + 16 queries','b   No initial observations + 20 queries')):
        for y,task in enumerate(TASKS):
            vals=[float(x['paired_relative_percent']) for x in paired if x['regime']==regime and x['task']==task and int(x['seed']) in seeds and x['comparator']=='bo']
            ax.plot([min(vals),max(vals)],[y,y],color='#B6C9CC',lw=2,zorder=1)
            for k,v in enumerate(vals):
                ax.scatter(v,y+(k-(len(vals)-1)/2)*.10,s=25,facecolors='white',edgecolors=COLORS['v13'],lw=1,zorder=3)
            ax.scatter(statistics.mean(vals),y,s=40,marker='D',color=COLORS['v13'],zorder=4)
        ax.axvline(0,color='#8C969E',lw=.8,ls='--');ax.set_title(title,loc='left',fontsize=10,fontweight='bold',pad=14)
        ax.set_xlabel('Paired SearchAUC change from BO (%)',fontsize=10);ax.grid(axis='x',alpha=.17)
        ax.set(yticks=range(8),yticklabels=[NAMES[t] for t in TASKS],ylim=(7.6,-.6))
        ax.spines[['top','right','left']].set_visible(False)
        ax.tick_params(axis='y',length=0)
    fig.text(.5,.01,'Circles: individual starts. Diamond: mean paired change. Line: observed range.',ha='center',fontsize=9.5)
    fig.tight_layout(rect=(0,.04,1,1),w_pad=2)
    save(fig,'paired_changes')


def learning(rounds,regime):
    seeds=(1,2,3) if regime=='warm' else (0,1,2,3)
    budget=16 if regime=='warm' else 20
    fig,axes=plt.subplots(2,4,figsize=(8.4,5.6))
    for ax,task in zip(axes.flat,TASKS):
        for method in METHODS:
            ys=[]
            for seed in seeds:
                rows=sorted((x for x in rounds if x['regime']==regime and x['task']==task and int(x['seed'])==seed and x['method']==method),key=lambda x:int(x['query']))
                ys.append([float(r['cumulative_best']) for r in rows])
            arr=np.array(ys);mean=arr.mean(0);sd=arr.std(0,ddof=1);q=np.arange(1,budget+1)
            ax.plot(q,mean,color=COLORS[method],lw=1.7,label=LABELS[method])
            ax.fill_between(q,mean-sd,mean+sd,color=COLORS[method],alpha=.08,lw=0)
        ax.set_title(NAMES[task],loc='left',fontweight='bold',fontsize=10)
        ax.set_xlim(1,budget);ax.set_xticks([1,5,10,15,budget] if budget==20 else [1,4,8,12,16])
        ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
        ax.tick_params(labelsize=10)
        if task=='snar':ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.52,-.005),fontsize=10.5)
    fig.supxlabel('Paid query after initialization' if regime=='warm' else 'Paid query (including the first experiment)',y=.065,fontsize=10.5)
    fig.supylabel('Cumulative best objective (native task units)',x=.005,fontsize=10.5)
    fig.tight_layout(rect=(.02,.10,1,1),h_pad=2,w_pad=1.1)
    save(fig,regime+'_curves')


def losses(rounds,forecasts):
    fig,axes=plt.subplots(2,2,figsize=(8.2,5.8),sharex='col')
    for col,(task,seed,title) in enumerate((('perera',2,'a   Perera, start 2'),('hplc',0,'b   HPLC, start 0'))):
        ax=axes[0,col]
        for method in METHODS:
            rows=sorted((x for x in rounds if x['regime']=='cold' and x['task']==task and int(x['seed'])==seed and x['method']==method),key=lambda x:int(x['query']))
            ax.step([int(x['query']) for x in rows],[float(x['cumulative_best']) for x in rows],where='post',label=LABELS[method],color=COLORS[method],lw=1.8)
        ax.set_title(title,loc='left',fontsize=10,fontweight='bold');ax.set_ylabel('Cumulative best')
        ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
        ax=axes[1,col]
        f=[x for x in forecasts if x['regime']=='cold' and x['task']==task and int(x['seed'])==seed]
        if f:
            ax.plot([int(x['query']) for x in f],[float(x['reliability_after']) for x in f],'o-',color=COLORS['v13'],lw=1.4,ms=4)
            last_permission=None
            for x in f:
                if x['permission_after']!=last_permission:
                    ax.annotate(x['permission_after'],(int(x['query']),float(x['reliability_after'])),xytext=(3,-14 if int(x['query'])%2 else 6),textcoords='offset points',fontsize=9,color='#485660')
                last_permission=x['permission_after']
            ax.axhline(.4,color='#9CA8B0',ls=':',lw=.8)
            ax.text(.04,.10,'q6–9: all five menu points in the same Note scope',transform=ax.transAxes,fontsize=8)
        else:
            ax.text(.5,.55,'No applicable Note forecasts\nNo KF or permission feedback at any query',ha='center',va='center',transform=ax.transAxes,fontsize=9,color='#5D6C7B')
        ax.set(xlim=(1,20),ylim=(0,.85),xticks=[1,5,10,15,20],xlabel='Paid query',ylabel='Note reliability')
        ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.01))
    fig.tight_layout(rect=(0,.055,1,1),h_pad=1.5,w_pad=2)
    save(fig,'failure_cases')


def tables(metrics,paired,summary):
    idx={(x['cohort'],x['task'],x['method']):x for x in summary}
    for cohort,out in (('warm_multistart','warm_table.tex'),('cold_multistart','cold_table.tex')):
        rows=[]
        for task in TASKS:
            r=[NAMES[task]]
            means=[float(idx[cohort,task,m]['mean_SearchAUC']) for m in METHODS]
            for method,mean in zip(METHODS,means):
                sd=float(idx[cohort,task,method]['sd_SearchAUC'])
                cell=f'{mean:.2f} $\\pm$ {sd:.2f}'
                r.append('\\textbf{'+cell+'}' if abs(mean-max(means))<1e-7 else cell)
            regime=cohort.split('_')[0];seeds=(1,2,3) if regime=='warm' else range(4)
            pp=[x for x in paired if x['regime']==regime and x['task']==task and x['comparator']=='bo' and int(x['seed']) in seeds]
            c=Counter(x['result'] for x in pp)
            r.append(f"{c['win']}/{c['tie']}/{c['loss']}")
            rows.append(r)
        put(out,table(['Task','BO','Reasoning BO','V13','W/T/L'],rows))
    for regime in ('warm','cold'):
        for part,tasks in enumerate((TASKS[:4],TASKS[4:]),1):
            rows=[]
            for task in tasks:
                for seed in range(4):
                    rr=[NAMES[task],str(seed)]
                    for method in METHODS:
                        x=next(x for x in metrics if x['regime']==regime and x['task']==task and int(x['seed'])==seed and x['method']==method)
                        rr.append(f"{float(x['SearchAUC']):.2f} / {float(x['final_best']):.2f} / {int(x['first_final_best_query'])}")
                    rows.append(rr)
            put(f'{regime}_all_{part}.tex',table(['Task','Start','BO','Reasoning BO','V13'],rows,'llrrr'))
    rows=[]
    for task in TASKS:
        rr=[NAMES[task]]
        for method in ('bo','reasoning_bo','v12_subagent','v13'):
            x=next(x for x in metrics if x['regime']=='warm' and x['task']==task and int(x['seed'])==0 and x['method']==method)
            rr.append(f"{float(x['SearchAUC']):.2f}")
        rows.append(rr)
    put('seed0.tex',table(['Task','BO','Reasoning BO (API)','V12 (agent)','V13 (agent)'],rows))
    rows=[]
    for task in TASKS:
        rr=[NAMES[task]]
        for method in ('bo','reasoning_bo','v12_subagent','v13'):
            x=idx['warm_multistart',task,method]
            rr.append(f"{float(x['mean_SearchAUC']):.2f} $\\pm$ {float(x['sd_SearchAUC']):.2f}")
        rows.append(rr)
    put('v12_context.tex',table(['Task','BO','Reasoning BO','V12','V13'],rows))
    mech=json.loads((P/'data/mechanisms.json').read_text(encoding='utf-8'))
    captions={'warm_development':'Warm start 0','warm_multistart':'Warm starts 1--3','cold_multistart':'Cold starts 0--3'}
    rows=[]
    for k in ('trajectories','decisions','note_forecasts','KF_increases','KF_decreases','accurate_failure_events','new_suppressions','new_allowed','changed_from_own_GP','refresh_calls','review_queries'):
        rows.append([k.replace('_',' '),*[str(mech[c][k]) for c in captions]])
    put('mechanisms.tex',table(['Event',*captions.values()],rows,'lrrr'))
    for regime,seeds in (('warm',(1,2,3)),('cold',tuple(range(4)))):
        for comparator in ('bo','reasoning_bo'):
            pp=[x for x in paired if x['regime']==regime and int(x['seed']) in seeds and x['comparator']==comparator]
            c=Counter(x['result'] for x in pp)
            key=('Warm' if regime=='warm' else 'Cold')+('BO' if comparator=='bo' else 'RBO')
            put(key+'.tex',f"{c['win']} wins, {c['tie']} ties and {c['loss']} losses")
    # Numeric source for V14 extension table, never blended with V13 efficacy.
    extension=P/'data/v14_selected_extension.csv'
    if not extension.exists():extension=ROOT/'experiments/reports/reason_cards_v14_20261103/comparison.csv'
    with extension.open(encoding='utf-8-sig') as f:
        ext=list(csv.DictReader(f))
    names={f'Reizman_Suzuki_Case_{i}':f'Suzuki {i}' for i in range(1,5)}|{'Perera_Suzuki':'Perera','Doyle_Buchwald_Hartwig':'Doyle','SnAr':'SnAr','HPLC':'HPLC'}
    erows=[[names[x['task']],x['seed'],*[f'{float(x[m]):.2f}' for m in ('bo_SearchAUC','v13_SearchAUC','reasoning_bo_SearchAUC','v14_SearchAUC')]] for x in ext]
    put('v14.tex',table(['Task','Start','BO','V13','Reasoning BO','V14'],erows,'llrrrr'))
    with (P/'data/v14_selected_extension.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(ext[0]));w.writeheader();w.writerows(ext)


def main():
    global P
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper-dir',type=Path,help='Standalone paper directory containing data/*.csv')
    args=parser.parse_args()
    if args.paper_dir:P=args.paper_dir.resolve()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42,'axes.labelcolor':'#344B5D','text.color':'#233C4E','axes.edgecolor':'#AEBCC4'})
    metrics,rounds,paired,summary,forecasts=[read(n+'.csv') for n in ('metrics','rounds','paired','summary','forecasts')]
    tables(metrics,paired,summary);workflow();contrasts(paired)
    learning(rounds,'warm');learning(rounds,'cold');losses(rounds,forecasts)
    print('Generated verified-result tables and five vector figures; no new optimization calls.')


if __name__=='__main__':main()
