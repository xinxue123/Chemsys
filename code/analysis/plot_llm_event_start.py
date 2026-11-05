"""Descriptive eight-query running-best and target-attainment figure."""
from collections import Counter
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/reports/llm_event_start_qlogei_20261104'


def read(name):
    with (OUT/name).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def main():
    rows,metrics,calls=read('rounds.csv'),read('metrics.csv'),read('LLM_calls.csv')
    arms=('bo','llm_start_bo','event')
    colors={'bo':'#245E99','llm_start_bo':'#B36B16','event':'#902F5B'}
    names={'bo':'Random start + qLogEI','llm_start_bo':'LLM start + qLogEI','event':'LLM start + KF events'}
    fig,axes=plt.subplots(1,2,figsize=(12,5.2),layout='constrained')
    for arm,style,width in zip(arms,('-','-',':'),(2.2,2.2,3.8)):
        selected=[r for r in rows if r['method']==arm]
        averages=[np.mean([float(r['best']) for r in selected if int(r['query'])==q]) for q in range(1,9)]
        axes[0].plot(range(1,9),averages,style,color=colors[arm],lw=width,label=names[arm])
        ms=[m for m in metrics if m['method']==arm]
        rates=[sum(bool(m['first95']) and int(m['first95'])<=q for m in ms)/len(ms) for q in range(1,9)]
        axes[1].step(range(1,9),rates,where='post',linestyle=style,color=colors[arm],lw=width,label=names[arm])
    axes[0].set(xlabel='Total experiment query',ylabel='Mean cumulative best yield (%)',ylim=(0,100),
        title='Mean running best across ten trajectories')
    axes[1].set(xlabel='Total experiment query',ylabel='Fraction reaching 85.09% yield',ylim=(0,1),
        title='Target attainment in ten trajectories per arm')
    for ax in axes:
        ax.set_xticks(range(1,9))
        ax.grid(axis='y',color='#E5E8EC',lw=.7)
        ax.spines[['top','right']].set_visible(False)
    axes[0].legend(frameon=False,fontsize=9,loc='lower right')
    fig.suptitle('Case 1: unrestricted first point + seven qLogEI queries',fontsize=15)
    fig.savefig(OUT/'running_best.png',dpi=180)
    plt.close(fig)
    refresh=[r for r in calls if r['kind']=='refresh']
    initial=[r for r in rows if r['method']=='event' and r['query']=='1']
    withdrawn=0
    for r in refresh:
        response=ROOT/Path(r['request_path']).parent/'response.raw.json'
        raw=json.loads(response.read_text(encoding='utf-8'))
        withdrawn+=not raw['hypotheses']
    diagnostic=dict(initial_candidate_counts=dict(Counter(r['candidate_id'] for r in initial)),
        initial_catalyst_counts=dict(Counter(r['catalyst'] for r in initial)),
        refreshes=len(refresh),refresh_abstentions=sum(not r['proposed_id'] for r in refresh),
        qLogEI_budget_rejections=sum(r['decision_rule']=='event_rejected_qLogEI_budget' for r in refresh),
        refreshes_with_no_retained_hypotheses=withdrawn)
    (OUT/'mechanism_diagnostic.json').write_text(json.dumps(diagnostic,indent=2)+'\n',encoding='utf-8')
    path=OUT/'README.md'
    content=path.read_text(encoding='utf-8')
    delta=next(np.mean([float(m['SearchAUC8']) for m in metrics if m['method']==arm])
               for arm in ('llm_start_bo',))-np.mean([float(m['SearchAUC8']) for m in metrics if m['method']=='bo'])
    content+=f"\nOf {len(refresh)} refreshes, {diagnostic['refresh_abstentions']} returned no candidate "
    content+=f"and {withdrawn} retained no hypotheses. Acquisition-budget rejections number "
    content+=f"{diagnostic['qLogEI_budget_rejections']}. The initial-only SearchAUC8 difference "
    content+=f"from random-start BO is {delta:+.4f}. See paired.csv for the isolated refresh effect.\n\n"
    content+='![Mean running best and target attainment](running_best.png)\n'
    path.write_text(content,encoding='utf-8')
    print(json.dumps(diagnostic))


if __name__=='__main__':
    main()
