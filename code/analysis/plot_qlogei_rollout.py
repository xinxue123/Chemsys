"""Scientific figure for the corrected qLogEI-based rollout comparison."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/reports/fast_hit_qlogei_rollout_20261104'


def main():
    with (OUT/'metrics.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    fig,(left,right)=plt.subplots(1,2,figsize=(12,5.7),layout='constrained')
    arms=('qlogei','rollout_qlogei')
    colors={'qlogei':'#2463A1','rollout_qlogei':'#A76B12'}
    names={'qlogei':'qLogEI','rollout_qlogei':'qLogEI-based rollout'}
    for arm,style in zip(arms,('-','--')):
        selected=[r for r in rows if r['method']==arm]
        rates=[sum(bool(r['first95']) and int(r['first95'])<=q for r in selected)/len(selected)
               for q in range(1,9)]
        left.step(range(1,9),rates,where='post',color=colors[arm],linestyle=style,
                  lw=2.2,label=names[arm])
        left.scatter([8],[rates[-1]],s=32,color=colors[arm])
    left.set(xlim=(1,8.4),ylim=(0,1),xticks=range(1,9),xlabel='Total experiment query',
        ylabel='Fraction reaching 85.09% yield',title='Target attainment across ten paired starts')
    left.set_yticks(np.arange(0,1.01,.2),labels=['0%','20%','40%','60%','80%','100%'])
    left.legend(frameon=False,loc='upper left')
    order=[(g,s) for g in ('low','middle') for s in range(5)]
    for i,(group,seed) in enumerate(order):
        pair={r['method']:r for r in rows if r['initial_group']==group and int(r['seed'])==seed}
        xs=[int(pair[arm]['capped_hitting_index']) for arm in arms]
        right.plot(xs,[i,i],color='#C3C8D0',lw=1,zorder=1)
        for arm,x in zip(arms,xs):
            right.scatter(x,i,marker='o' if arm=='qlogei' else '^',s=65,
                facecolors=colors[arm] if pair[arm]['first95'] else 'none',
                edgecolors=colors[arm],lw=1.4,zorder=2)
    right.axvline(8.5,color='#C3C8D0',lw=1,linestyle=':')
    right.set(yticks=range(10),yticklabels=[f'{g.title()} s{s}' for g,s in order],
        xticks=range(1,10),xticklabels=[str(x) for x in range(1,9)]+['>8'],
        xlim=(.6,9.4),ylim=(9.6,-.6),xlabel='First target hit; open marks are unreached',
        title='Each shared initial point')
    for ax in (left,right):
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='x' if ax is right else 'y',color='#E4E7EB',lw=.7)
    fig.suptitle('Case 1: qLogEI at both rollout stages, one initial point + seven queries',fontsize=14)
    fig.savefig(OUT/'target_attainment.png',dpi=180)
    plt.close(fig)
    path=OUT/'README.md'
    content=path.read_text(encoding='utf-8')
    content+='\n![Target attainment and censored hit times](target_attainment.png)\n'
    path.write_text(content,encoding='utf-8')


if __name__=='__main__':
    main()
