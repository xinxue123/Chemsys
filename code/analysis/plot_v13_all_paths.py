"""Scientific summary figure: all 227 main-comparison paths, no subsampling."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/reports/v13_all_paths_20261104'


def main():
    source=OUT/'all_path_profiles.csv'
    data=pd.read_csv(source,encoding='utf-8-sig');data=data[data.primary].copy()
    assert len(data)==227
    groups={'bo':'BO','v13':'V13','reasoning_bo':'Reasoning BO',
            'v12_subagent':'V12','v12_api':'V12'}
    data['display_method']=data.method.map(groups);assert data.display_method.notna().all()
    tasks=['c1','c2','c3','c4','perera','doyle','snar','hplc']
    labels=['Suzuki Case 1','Suzuki Case 2','Suzuki Case 3','Suzuki Case 4',
            'Perera Suzuki','Doyle B-H','SnAr','HPLC']
    colors={'BO':'#535d68','V13':'#2466a8','Reasoning BO':'#c8751c','V12':'#94618e'}
    markers={'BO':'s','V13':'o','Reasoning BO':'^','V12':'D'}
    offsets={'BO':-.25,'V13':-.08,'Reasoning BO':.09,'V12':.26}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.titlesize':12})
    fig,axes=plt.subplots(1,2,figsize=(13,7.3),sharex=True,sharey=True)
    count=0
    for ax,budget in zip(axes,['cold','warm']):
        b=data[data.regime==budget]
        for method in colors:
            for backend in ('none','fresh_subagent','historical_API'):
                g=b[(b.display_method==method)&(b.backend==backend)]
                if g.empty:continue
                y=g.task.map({t:i for i,t in enumerate(tasks)}).astype(float)+offsets[method]+(g.seed-1.5)*.035
                ax.scatter(g.AUC_normalized,y,s=44,marker=markers[method],linewidths=1.15,
                           facecolors='none' if backend=='historical_API' else colors[method],
                           edgecolors=colors[method],alpha=.82,zorder=3)
                count+=len(g)
        for y in range(len(tasks)-1):ax.axhline(y+.5,color='#e2e5e9',lw=.7,zorder=0)
        ax.set_xlim(0,1.02);ax.set_xticks([0,.2,.4,.6,.8,1])
        ax.set_xlabel('SearchAUC, normalized to each task pool range')
        ax.set_title(('0 + 20' if budget=='cold' else '4 + 16')+f' observations: all {len(b)} paths',pad=12)
        ax.set_yticks(range(len(tasks)),labels);ax.grid(axis='x',color='#e8ebee',lw=.7)
        ax.set_ylim(7.55,-.55)
    assert count==227
    handles=[Line2D([],[],marker=markers[m],color=colors[m],linestyle='',markersize=7,label=m) for m in colors]
    handles.append(Line2D([],[],marker='o',color='#535d68',markerfacecolor='none',linestyle='',
                          markersize=7,label='Open marker: historical API'))
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.54,.04),ncol=5,frameon=False,fontsize=10)
    fig.suptitle('All main-comparison trajectories across eight tasks',fontsize=15,y=.985)
    fig.text(.54,.012,'Each dot is one archived path. Vertical offsets separate methods and seeds; no path is omitted.',
             ha='center',fontsize=10,color='#48515b')
    fig.subplots_adjust(left=.13,right=.985,top=.91,bottom=.17,wspace=.1)
    target=OUT/'all_main_paths.png';fig.savefig(target,dpi=180,facecolor='white');plt.close(fig)
    data[['path_id','task','regime','seed','method','backend','AUC_normalized','SearchAUC','final_best']].to_csv(
        OUT/'all_main_paths_figure_data.csv',index=False,encoding='utf-8-sig')
    audit=dict(paths_plotted=count,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
               image_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),subsampling=False)
    (OUT/'figure_audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
