"""Static scientific exports from frozen result JSON; no new evaluation."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT=Path('research/suzuki_acquisition')
METHODS=('qLogEI','UCB_official','UCB_fixed')
LABELS=('qLogEI','UCB original beta','UCB kappa=2')
COLORS=('#0072B2','#D55E00','#009E73')
REGIMES=('none','correct','wrong_category','mixed','assistant_initial')
NAMES=('No knowledge','Correct region','Wrong catalyst','Mixed errors','Initial assistant region')


def main():
    result=json.loads((OUT/'results.json').read_text(encoding='utf8'));tables=result['tables']['primary']
    fig,axes=plt.subplots(1,5,figsize=(15,3.2),sharey=True,layout='constrained')
    for ax,g,title in zip(axes,REGIMES,NAMES):
        for method,label,color in zip(METHODS,LABELS,COLORS):ax.plot(range(5,21),tables[g][method]['curve'],label=label,color=color,lw=1.8)
        ax.set(title=title,xlabel='Total queries',xticks=[5,10,15,20]);ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Mean incumbent yield (%)');axes[-1].legend(fontsize=8,loc='lower right')
    fig.suptitle('Cases 2-4: 15 paired trajectories per condition; fixed pool and GP',fontsize=11)
    fig.savefig(OUT/'incumbent_curves.png',dpi=180);fig.savefig(OUT/'incumbent_curves.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,3.7),layout='constrained');x=np.arange(4);width=.24
    for j,(method,label,color) in enumerate(zip(METHODS,LABELS,COLORS)):
        values=[result['region_minus_own_acquisition_B0']['primary'][g][method]['mean_AUC_delta_pp'] for g in REGIMES[1:]]
        axes[0].bar(x+(j-1)*width,values,width,color=color,label=label)
    axes[0].set(xticks=x,xticklabels=NAMES[1:],ylabel='Region minus own BO SearchAUC (pp)',title='What does regional knowledge add?')
    for i,case in enumerate(('2','3','4')):
        values=[result['paired_vs_qLogEI']['primary'][g]['UCB_fixed']['by_case'][case] for g in REGIMES]
        axes[1].plot(range(5),values,marker=('o','s','^')[i],label='Case '+case,lw=1.5)
    axes[1].set(xticks=range(5),xticklabels=NAMES,ylabel='UCB kappa=2 minus qLogEI AUC (pp)',title='Effect varies by reaction case')
    for ax in axes:
        ax.axhline(0,color='.35',lw=.8);ax.grid(axis='y',alpha=.18);ax.tick_params(axis='x',labelrotation=20);ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=8)
    fig.savefig(OUT/'regional_gain_and_cases.png',dpi=180);fig.savefig(OUT/'regional_gain_and_cases.pdf');plt.close(fig)
    with (OUT/'comparison_tables.csv').open('w',encoding='utf8',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['scope','condition','acquisition','n','SearchAUC_pp','final_yield_percent'])
        for scope,groups in result['tables'].items():
            for g,methods in groups.items():
                for method,row in methods.items():writer.writerow([scope,g,method,row['n'],row['AUC_mean'],row['final_mean']])
    print('Exported two figures (PNG/PDF) and comparison_tables.csv')


if __name__=='__main__':main()
