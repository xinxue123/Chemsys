"""Compare frozen full trajectories; never participates in candidate selection."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from chem.benchmarks import stable_suzuki_study as s

STYLES={
    'Revised strategy':('#2563a6','o','-'),
    'LGBO':('#c37713','s','--'),
    'LLINBO-Justify':('#7c852b','^','-.'),
    'Pure BO (qLogEI)':('#42474d','x',':')}


def source(case,regime,label):
    if label=='Revised strategy':return s.ROOT/'runs'/f'c{case}_s0_{regime}_stable'/'state.json'
    if label=='Pure BO (qLogEI)':return s.prior.previous.ROOT/'base'/f'c{case}_s0'/'state.json'
    method='LGBO_region' if label=='LGBO' else 'LLINBO_matched'
    return s.prior.ROOT/'runs'/f'c{case}_s0_{regime}_{method}'/'state.json'


def main():
    assert s.status()['complete']==16
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
        'axes.labelcolor':'#30353b','text.color':'#30353b','axes.edgecolor':'#78818a','savefig.facecolor':'white'})
    records=[];sources={}
    for regime in s.REGIMES:
        fig,axes=plt.subplots(4,2,figsize=(13.2,14.2),sharex=True,sharey=True)
        for case in (1,2,3,4):
            for label,(color,marker,line) in STYLES.items():
                path=source(case,regime,label);state=s.read(path)
                assert state['status']=='COMPLETE' and len(state['history'])==20
                y=np.array([h['y']*100 for h in state['history']]);best=np.maximum.accumulate(y);x=np.arange(1,21)
                sources[path.as_posix()]=s.filehash(path)
                for k,values in enumerate((y,best)):
                    ax=axes[case-1,k]
                    ax.plot(x,values,color=color,marker=marker,linestyle=line,linewidth=1.7 if label=='Revised strategy' else 1.3,
                        markersize=3.8,label=label)
                for t,(a,b) in enumerate(zip(y,best),1):records.append({'case':case,'knowledge':regime,'method':label,'experiment':t,
                    'yield_percent':float(a),'incumbent_percent':float(b),'source':path.as_posix()})
            for k,ax in enumerate(axes[case-1]):
                ax.axvspan(.5,4.5,color='#e9edf2',alpha=.7,zorder=0)
                ax.set_ylim(0,102);ax.set_xlim(.5,20.5);ax.set_yticks([0,25,50,75,100]);ax.set_xticks([1,4,8,12,16,20])
                ax.grid(axis='y',color='#e1e5e9',linewidth=.7)
                ax.set_title(f"Case {case} | {'Observed yield' if k==0 else 'Best yield so far'}",loc='left',fontsize=12)
                if k==0:ax.set_ylabel('Yield (%)')
                if case==4:ax.set_xlabel('Experiment number (4 initialization + 16 queries)')
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.964),ncol=4,frameon=False,fontsize=11)
        fig.suptitle('Reizman Suzuki | '+regime.replace('_',' '),x=.07,ha='left',y=.99,fontsize=17)
        fig.text(.07,.019,'seed = 0; fixed 4,692-point pools. Grey: initialization. One trajectory per case/condition; no confidence intervals.\n'
            'Archived controls retain their original tie rule; revised strategy includes covariance-unit repair. Molecular rescue is inapplicable.',fontsize=9.5,color='#535e68')
        fig.subplots_adjust(top=.925,bottom=.085,left=.07,right=.98,hspace=.27,wspace=.10)
        target=s.OUT/f'comparison_{regime}.png';fig.savefig(target,dpi=160);plt.close(fig)
    with (s.OUT/'plot_source_rows.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    s.write(s.OUT/'plot_provenance.json',{'script_sha256':s.filehash(__file__),'sources':sources,
        'rows':len(records),'figures':4,'source_values':'each history y*100; cumulative maximum; no interpolation or smoothing',
        'visual_QA':'pending'})
    print({'figures':4,'rows':len(records)})


if __name__=='__main__':main()
