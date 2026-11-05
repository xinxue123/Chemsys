"""Static research figures from the audited initialization study."""
from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/reports/suzuki_initial_yield_20261104'


def rows(name):
    with (OUT/name).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def main():
    assert json.loads((OUT/'audit.json').read_text())['status']=='PASS'
    rr=rows('rounds.csv')
    gg=rows('groups.csv')
    colors={'low':'#2463A1','middle':'#A76B12','mixed':'#575B66'}
    styles={'low':'-','middle':'--','mixed':':'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.titlesize':12,'axes.labelsize':10})
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for case,ax in enumerate(axes.flat,1):
        max_y=float(next(x['maximum'] for x in gg if int(x['case'])==case))
        for arm in ('low','middle','mixed'):
            matrix=np.array([[float(x['best']) for x in sorted(
                [r for r in rr if int(r['case'])==case and r['arm']==arm and int(r['seed'])==seed],
                key=lambda x:int(x['query']))] for seed in range(5)])
            ax.plot(range(1,21),matrix.mean(0),color=colors[arm],linestyle=styles[arm],lw=2,label=arm.title())
            if arm!='mixed':
                ax.fill_between(range(1,21),matrix.min(0),matrix.max(0),color=colors[arm],alpha=.10)
        ax.axhline(.95*max_y,color='#60656D',linestyle='-.',lw=1)
        ax.set(title=f'Case {case}',xlabel='Total experiment query',ylabel='Running best yield (%)',
               xlim=(1,20),ylim=(0,max_y*1.07),xticks=[1,5,10,15,20])
        ax.grid(axis='y',color='#E2E5EA',lw=.7)
    fig.suptitle('Pure BO after one yield-controlled initial observation',fontsize=16)
    handles=[Line2D([0],[0],color=colors[a],linestyle=styles[a],lw=2,label=a.title()+' initial')
             for a in colors]
    handles.append(Line2D([0],[0],color='#60656D',linestyle='-.',label='95% of pool maximum'))
    fig.legend(handles=handles,loc='outside lower center',ncol=4,frameon=False)
    fig.savefig(OUT/'running_best.png',dpi=180)
    plt.close(fig)
    palette=['#CADBEF','#E9BD6D','#365A7A']
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for case,ax in enumerate(axes.flat,1):
        matrix=[]
        for arm in ('low','middle'):
            for seed in range(5):
                chosen=sorted([r for r in rr if int(r['case'])==case and r['arm']==arm and int(r['seed'])==seed],
                              key=lambda x:int(x['query']))
                matrix.append([('low','middle','high').index(r['yield_group']) for r in chosen])
        ax.imshow(matrix,aspect='auto',cmap=ListedColormap(palette),norm=BoundaryNorm([-.5,.5,1.5,2.5],3),
                  interpolation='nearest')
        ax.set(title=f'Case {case}',xticks=[0,4,9,14,19],xticklabels=[1,5,10,15,20],
               yticks=range(10),yticklabels=[f'{a.title()} s{s}' for a in ('low','middle') for s in range(5)],
               xlabel='Total experiment query')
        ax.axhline(4.5,color='white',lw=1.5)
    fig.suptitle('Observed yield groups along each pure BO trajectory',fontsize=16)
    labels=[Line2D([0],[0],marker='s',linestyle='',markersize=10,color=c,label=g)
            for c,g in zip(palette,['Low','Middle','High'])]
    fig.legend(handles=labels,loc='outside lower center',ncol=3,frameon=False)
    fig.savefig(OUT/'group_sequences.png',dpi=180)
    plt.close(fig)
    doc=OUT/'README.md'
    text=doc.read_text(encoding='utf-8')
    text+='\n![Running best by initialization group](running_best.png)\n\n'
    text+='Lines are five-seed means; shading spans the five observed starts and is not a confidence interval. '
    text+='Mixed initialization reuses low/middle trajectories, so its range is omitted.\n\n'
    text+='![Selected-point group sequences](group_sequences.png)\n\n'
    text+='Rows show the ten distinct low/middle starts per case. Group colors use each case’s own tertiles. '
    text+='Only query one was constrained by these labels.\n'
    doc.write_text(text,encoding='utf-8')


if __name__=='__main__':
    main()
