"""Visualize target attainment and censored paired hit times."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/reports/fast_hit_bo_20261104'


def main():
    with (OUT/'metrics.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    fig,(left,right)=plt.subplots(1,2,figsize=(12,5.7),gridspec_kw={'width_ratios':[1,1.15]},layout='constrained')
    colors={'qlogei':'#2463A1','rollout':'#A76B12'}
    names={'qlogei':'qLogEI','rollout':'Two-step Rollout'}
    for arm,style in (('qlogei','-'),('rollout','--')):
        rs=[r for r in rows if r['method']==arm]
        rates=[sum(bool(r['first95']) and int(r['first95'])<=q for r in rs)/len(rs) for q in range(1,9)]
        left.step(range(1,9),rates,where='post',color=colors[arm],linestyle=style,lw=2.2,label=names[arm])
        left.scatter([8],[rates[-1]],s=32,color=colors[arm])
    left.set(xlim=(1,8.4),ylim=(0,1),xticks=range(1,9),xlabel='Total experiment query',
             ylabel='Fraction reaching 85.09% yield',title='Target attainment across ten paired starts')
    left.set_yticks(np.arange(0,1.01,.2),labels=['0%','20%','40%','60%','80%','100%'])
    left.legend(frameon=False,loc='upper left')
    order=[(g,s) for g in ('low','middle') for s in range(5)]
    for i,(group,seed) in enumerate(order):
        pair={r['method']:r for r in rows if r['initial_group']==group and int(r['seed'])==seed}
        xs=[int(pair[a]['capped_hitting_index']) for a in ('qlogei','rollout')]
        right.plot(xs,[i,i],color='#C3C8D0',lw=1,zorder=1)
        for arm,x in zip(('qlogei','rollout'),xs):
            hit=bool(pair[arm]['first95'])
            right.scatter(x,i,marker='o' if arm=='qlogei' else '^',s=65,
                          facecolors=colors[arm] if hit else 'none',edgecolors=colors[arm],lw=1.4,zorder=2)
    right.axvline(8.5,color='#C3C8D0',lw=1,linestyle=':')
    right.set(yticks=range(10),yticklabels=[f'{g.title()} s{s}' for g,s in order],
              xticks=range(1,10),xticklabels=[str(x) for x in range(1,9)]+['>8'],
              xlim=(.6,9.4),ylim=(9.6,-.6),xlabel='First target hit; open marks are unreached',
              title='Each shared initial point')
    for ax in (left,right):
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='x' if ax is right else 'y',color='#E4E7EB',lw=.7)
    fig.suptitle('Case 1 · one initial observation + seven BO queries',fontsize=15)
    fig.savefig(OUT/'target_attainment.png',dpi=180)
    plt.close(fig)
    p=OUT/'README.md'
    text=p.read_text(encoding='utf-8')
    text+='\n![Target attainment and censored hit times](target_attainment.png)\n\n'
    text+='Open marks at >8 indicate that the target was not reached within the budget. They are not observations at query9.\n'
    p.write_text(text,encoding='utf-8')


if __name__=='__main__':
    main()
