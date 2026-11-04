"""Illustrate registered score arithmetic; never evaluate alternative yields."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/reports/program_kf_weight_20261103'


def main():
    with (OUT/'prefix_scores.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    for ax,(task,q,title) in zip(axes,[('HPLC',9,'HPLC start 3, held query 9'),
                                     ('Perera_Suzuki',7,'Perera start 2, held query 7')]):
        rr=[r for r in rows if r['task']==task and int(r['query'])==q]
        weights=np.linspace(0,.8,161);dynamic=float(rr[0]['dynamic_weight'])
        groups={}
        for r in rr:
            key=float(r['GP_Borda']),float(r['LLM_Borda'])
            groups.setdefault(key,[]).append(r['candidate_id'])
        for (b,l),ids in groups.items():
            ax.plot(weights,(1-weights)*b+weights*l,label=', '.join(ids),lw=1.8)
        ax.axvline(.4,color='#b77917',ls='--',lw=1.5,label='Fixed w=0.4')
        ax.axvline(dynamic,color='#345f9b',ls=':',lw=2,label=f'Dynamic w={dynamic:.3f}')
        fixed=next(r['candidate_id'] for r in rr if r['fixed_selected']=='True')
        selected=next(r['candidate_id'] for r in rr if r['dynamic_selected']=='True')
        ax.set(title=title,xlabel='Program rank weight w',ylabel='Combined rank score',xlim=(0,.8))
        ax.grid(alpha=.2);ax.legend(fontsize=7,loc='best')
        ax.text(.02,.02,f'Fixed: {fixed}\nDynamic: {selected}',transform=ax.transAxes,fontsize=9,
            bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
    fig.suptitle('Identical GP and five forecasts; only the deployed weight changes',fontsize=12)
    fig.supxlabel('Same-score candidates are grouped. This is score sensitivity, with no alternative outcome queries.',fontsize=9)
    fig.savefig(OUT/'prefix_weight_sensitivity.png',dpi=180);plt.close(fig)
    print(str(OUT/'prefix_weight_sensitivity.png'))


if __name__=='__main__':main()
