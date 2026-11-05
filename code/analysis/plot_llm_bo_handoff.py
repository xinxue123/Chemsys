"""Descriptive outcome curves and actual control-transfer counts."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/reports/llm_bo_handoff_20261104'


def read(name):
    with (OUT/name).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def main():
    rows,metrics=read('rounds.csv'),read('metrics.csv')
    arms=('bo','llm_start_bo','handoff')
    names={'bo':'Random start + pure BO','llm_start_bo':'Same LLM start + pure BO','handoff':'Gradual handoff'}
    colors={'bo':'#23629F','llm_start_bo':'#7B8189','handoff':'#B66D1C'}
    fig,axes=plt.subplots(1,2,figsize=(12,5.2),layout='constrained')
    for method,style in zip(arms,('-','--','-')):
        rs=[r for r in rows if r['method']==method]
        curve=[np.mean([float(r['best']) for r in rs if int(r['query'])==q]) for q in range(1,9)]
        axes[0].plot(range(1,9),curve,color=colors[method],linestyle=style,lw=2.2,label=names[method])
        ms=[m for m in metrics if m['method']==method]
        rates=[sum(bool(m['first95']) and int(m['first95'])<=q for m in ms)/len(ms) for q in range(1,9)]
        axes[1].step(range(1,9),rates,where='post',color=colors[method],linestyle=style,lw=2.2)
    axes[0].set(ylabel='Mean cumulative best yield (%)',ylim=(0,100),title='Early-search outcome across ten trajectories')
    axes[1].set(ylabel='Fraction reaching 85.09% yield',ylim=(0,1),title='Good-yield target attainment')
    for ax in axes:
        ax.set(xlabel='Total experiment query',xticks=range(1,9))
        ax.grid(axis='y',color='#E6E8EC',lw=.7)
        ax.spines[['top','right']].set_visible(False)
    axes[0].legend(frameon=False,fontsize=9,loc='lower right')
    fig.suptitle('Case 1: same LLM first point, then gradual control transfer',fontsize=15)
    fig.savefig(OUT/'running_best.png',dpi=180)
    plt.close(fig)
    stages=[]
    for q in range(2,9):
        rs=[r for r in rows if r['method']=='handoff' and int(r['query'])==q]
        stages.append(dict(query=q,authority_ceiling=float(rs[0]['authority_ceiling']),
            positive_budget=sum(float(r['effective_logei_gap'])>0 for r in rs),
            changed_from_current_BO=sum(r['knowledge_changed_choice']=='True' for r in rs),
            fresh_refreshes=sum(r['fresh_refresh']=='True' for r in rs),
            mean_effective_logei_gap=float(np.mean([float(r['effective_logei_gap']) for r in rs]))))
    with (OUT/'control_by_query.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(stages[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(stages)
    fig,(left,right)=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    x=[r['query'] for r in stages]
    left.plot(x,[r['authority_ceiling'] for r in stages],marker='o',color='#23629F',label='Time ceiling')
    left.plot(x,[r['mean_effective_logei_gap']/np.log(4) for r in stages],marker='s',color='#B66D1C',label='Mean effective allowance / log(4)')
    left.set(ylim=(0,1),xticks=x,xlabel='Experiment query',ylabel='Normalized allowance',title='Ceiling and effective knowledge allowance')
    left.legend(frameon=False,fontsize=8)
    right.bar(x,[r['changed_from_current_BO'] for r in stages],color='#B66D1C')
    right.set(ylim=(0,10),xticks=x,xlabel='Experiment query',ylabel='Selections differing from current BO',title='Actual intervention per query (ten trajectories)')
    for ax in (left,right):
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',color='#E6E8EC',lw=.7)
    fig.savefig(OUT/'control_transfer.png',dpi=180)
    plt.close(fig)
    index=OUT/'README.md'
    text=index.read_text(encoding='utf-8')
    text+='\n## Observed transfer pattern\n\n'
    text+='The time ceiling and effective allowance are distinct. The query-level '
    text+='counts below show when knowledge actually changes an action; a positive '
    text+='ceiling alone is not evidence of intervention.\n\n'
    text+='| Query | Positive budget (of10) | Different from current BO | Fresh refreshes |\n'
    text+='|---|---|---|---|\n'
    for r in stages:
        text+=f"| {r['query']} | {r['positive_budget']} | {r['changed_from_current_BO']} | {r['fresh_refreshes']} |\n"
    text+='\n![Running-best and target attainment](running_best.png)\n\n'
    text+='![Control-transfer mechanism](control_transfer.png)\n'
    index.write_text(text,encoding='utf-8')
    print(json.dumps(stages))


if __name__=='__main__':
    main()
