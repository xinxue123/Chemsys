"""Plot a saved cross-cohort analysis; no scientific run is invoked."""
from pathlib import Path
from hashlib import sha256
import json
import os
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'runtime/cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

OUT=ROOT/'experiments/reports/v12_api_subagent_diagnosis_20261102'
source=OUT/'analysis.json'
analysis=json.loads(source.read_text(encoding='utf-8'))
evidence_path=ROOT/'paper/v12_subagent/data/evidence.json'
evidence=json.loads(evidence_path.read_text(encoding='utf-8'))
assert analysis['status']=='PASS_READ_ONLY_CROSS_COHORT_DIAGNOSIS'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
colors={'API':'#c66b23','subagent':'#087fa4','BO':'#5b6671'}
fig,axs=plt.subplots(2,3,figsize=(12.7,7.7))
tasks=[('Reizman_Suzuki_Case_2','Suzuki Case 2',0),('Reizman_Suzuki_Case_4','Suzuki Case 4',1),('HPLC','HPLC',0)]
for col,(task,label,seed) in enumerate(tasks):
    ax=axs[0,col];rr=sorted([r for r in analysis['cross_cohort'] if r['task']==task],key=lambda r:r['seed'])
    assert [r['seed'] for r in rr]==[0,1,2,3]
    allv=[]
    for co,offset in [('API',-.18),('subagent',.18)]:
        values=[r[co+'_delta_percent'] for r in rr];allv.extend(values)
        bars=ax.bar(np.arange(4)+offset,values,width=.33,color=colors[co],zorder=3)
        for bar,value in zip(bars,values):
            ax.annotate(f'{value:+.1f}',(bar.get_x()+bar.get_width()/2,value),
                xytext=(-4 if co=='API' else 4,3 if value>=0 else -3),textcoords='offset points',
                ha='center',va='bottom' if value>=0 else 'top',fontsize=9)
    low=min(0,min(allv));high=max(0,max(allv));span=high-low
    ax.set_ylim(low-.20*span,high+.23*span)
    ax.axhline(0,color='#46525c',linewidth=.9);ax.grid(axis='y',alpha=.2,zorder=0)
    ax.set_xticks(range(4));ax.set_xlabel('Initialization seed');ax.set_title(label,pad=8)
    ax.set_ylabel('SearchAUC difference vs BO (%)')
    ax=axs[1,col]
    pair=[next(r for r in analysis['cells'] if r['cohort']==co and r['task']==task and r['initialization_seed']==seed and r['method']=='V12' and r['phase']!='LLM_repeat') for co in ('API','subagent')]
    control=next(r for r in evidence['rows'] if r['task']==task and r['initialization_seed']==seed and r['method']=='BO')
    for name,r in [('BO',control),('API',pair[0]),('subagent',pair[1])]:
        assert r['initial_ids']==control['initial_ids']
        ax.step(range(17),r['best_curve'],where='post',color=colors[name],linewidth=1.8,linestyle='--' if name=='BO' else '-')
    split=next(r['first_different_query'] for r in rr if r['seed']==seed)
    ax.axvline(split,color='#8b9297',linewidth=.9,linestyle=':')
    ax.text(.03,.96,f'First API/subagent split: q{split}',transform=ax.transAxes,va='top',fontsize=9)
    ax.set_title(f'{label}, seed {seed}',pad=9);ax.set_xlabel('Query after four initial observations')
    ax.set_ylabel('Best observed area (a.u.)' if task=='HPLC' else 'Best observed yield (%)')
    ax.set_xticks([0,4,8,12,16]);ax.set_xlim(0,16);ax.grid(alpha=.2)
    lo,hi=ax.get_ylim();ax.set_ylim(lo,hi+.10*(hi-lo))
fig.suptitle('V12 across API and subagent executions',fontsize=16,y=.985)
fig.text(.5,.94,'Matched pools, initial observations and 4 + 16 budget; the execution cohorts remain separate.',ha='center',fontsize=10)
fig.legend(handles=[Patch(facecolor=colors['API'],label='Original API V12'),Patch(facecolor=colors['subagent'],label='Subagent V12'),Line2D([0],[0],color=colors['BO'],linestyle='--',label='Pure BO')],loc='upper center',bbox_to_anchor=(.5,.922),ncol=3,frameon=False)
fig.subplots_adjust(top=.835,bottom=.105,left=.07,right=.98,hspace=.47,wspace=.30)
fig.text(.07,.025,'Top: each bar is one trajectory; panel y-scales differ. Bottom: diagnostic examples, not isolated module effects.',fontsize=9,color='#42515c')
png=OUT/'api_subagent_comparison.png';fig.savefig(png,dpi=170,facecolor='white');plt.close(fig)
receipt=dict(status='FIGURE_GENERATED_PENDING_VISUAL_REVIEW',source_sha256=sha256(source.read_bytes()).hexdigest(),
    evidence_sha256=sha256(evidence_path.read_bytes()).hexdigest(),generator_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
    figure_sha256=sha256(png.read_bytes()).hexdigest(),figure=png.relative_to(ROOT).as_posix(),new_experiments=0)
(OUT/'figure_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt,indent=2))
