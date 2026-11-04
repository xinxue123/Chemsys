"""Standalone scientific figures and exact LaTeX tables from reviewed records."""
from pathlib import Path
import json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parents[2];PAPER=ROOT/'paper/v12'
E=json.loads((PAPER/'data/evidence.json').read_text(encoding='utf-8'))
R=E['rows'];TASKS=[d['task'] for d in E['datasets']]
FOCUS=['Reizman_Suzuki_Case_2','Reizman_Suzuki_Case_4','HPLC']
LABEL={d['task']:d['label'] for d in E['datasets']}
GEN=PAPER/'generated';FIG=PAPER/'figures';GEN.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
COL={'BO':'#5f6973','Reasoning BO':'#c77d28','V12':'#006d89'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})

def esc(x):
    return str(x).replace('\\',r'\textbackslash{}').replace('_',r'\_').replace('%',r'\%').replace('&',r'\&').replace('#',r'\#')
def wr(name,text): (GEN/name).write_text(text,encoding='utf-8')
def save(fig,name):
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight');fig.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)
def find(task,method,phase='seed0',seed=0):
    return next((r for r in R if r['task']==task and r['method']==method and r['phase']==phase and r['initialization_seed']==seed),None)
def fmt(x):return '--' if x is None else f'{x:.3f}'
def longtable(name,caption,label,head,lines,align):
    text=r'\begingroup\small'+'\n'+r'\setlength{\tabcolsep}{4pt}'+'\n'+r'\begin{longtable}{'+align+'}\n'
    text+=r'\caption{'+caption+r'}\label{'+label+r'}\\'+'\n'+r'\toprule'+'\n'+' & '.join(head)+r'\\\midrule'+'\n'+r'\endfirsthead'+'\n'
    text+=r'\toprule'+'\n'+' & '.join(head)+r'\\\midrule'+'\n'+r'\endhead'+'\n'
    text+='\n'.join(' & '.join(map(str,line))+r'\\' for line in lines)
    text+='\n'+r'\bottomrule\end{longtable}\endgroup'+'\n';wr(name,text)

def tables():
    units={'P0':'Emulated yield (\%)','Perera':'UV product area (\%)','Doyle':'Reported yield (\%)','SnAr':r'STY (kg m$^{-3}$ h$^{-1}$)','HPLC':'Peak area (a.u.)'}
    lines=[]
    for d in E['datasets']:
        t=d['task'];ct=d['kinds'].count('continuous');cat=d['kinds'].count('categorical')
        lines.append(' & '.join([d['group'],esc(d['label']),str(d['candidate_count']),f'{ct} cont.; {cat} cat.',units.get(d['label'],units.get(d['group'],'')),'Complete' if find(t,'V12') else 'Pending'])+r'\\')
    wr('datasets.tex',r'\begin{table}[t]\centering\small'+'\n'+r'\caption{Frozen candidate pools and current V12 coverage. Pool counts describe this implementation, not the size of every original source dataset.}\label{tab:data}'+'\n'+r'\begin{tabularx}{\linewidth}{llrlXl}\toprule Group & Task & Points & Inputs & Objective & V12 seed 0\\\midrule'+'\n'+'\n'.join(lines)+'\n'+r'\bottomrule\end{tabularx}\end{table}')
    lines=[]
    for task in TASKS:
        for m in ('BO','Reasoning BO','V8','V9','V10','V11','V12'):
            r=find(task,m)
            lines.append([esc(LABEL[task]),esc(m),fmt(r['SearchAUC']) if r else '--',fmt(r['final_best']) if r else '--',r['first_best_query'] if r else '--'])
    longtable('seed0_all.tex','Complete modern development trajectories at seed 0; missing results are shown as dashes.','tab:seed0all',['Task','Method','SearchAUC','Final best','First best query'],lines,'llrrr')
    lines=[]
    for r in [x for x in R if x['phase']=='LLM_repeat']:
        b=find(r['task'],'BO');lines.append([esc(r['label']),r['replicate'],fmt(r['SearchAUC']),fmt(r['final_best']),r['first_best_query'],f"{100*(r['SearchAUC']/b['SearchAUC']-1):+.2f}"+r'\%'])
    longtable('repeats.tex','Independent fresh-response V12 repetitions, with the same fixed seed-0 BO reference.','tab:repeats',['Task','Repeat','SearchAUC','Final best','First best','AUC vs BO'],lines,'llrrrr')
    lines=[]
    for task in TASKS:
        for seed in (1,2,3):
            rs=[find(task,m,'initialization',seed) for m in ('BO','V12','Reasoning BO')]
            lines.append([esc(LABEL[task]),seed]+[fmt(r['SearchAUC']) if r else '--' for r in rs]+[fmt(rs[1]['final_best']-rs[0]['final_best']) if rs[1] else '--'])
    longtable('seeds.tex','Matched initialization results. Final-best difference is V12 minus BO in native units. Dashes denote incomplete registered comparisons.','tab:seeds',['Task','Init.','BO AUC','V12 AUC','RBO AUC',r'$\Delta$ final'],lines,'lrrrrr')
    lines=[]
    for r in [x for x in R if x['phase']=='initialization' and x['method']=='V12']:
        lines.append([esc(r['label']),r['initialization_seed'],r['forecasts'],r['weight_decreases'],r['pauses'],r['retractions'],r['resumes'],r['edits']])
    longtable('mechanisms.tex','Executed forecast and lifecycle events in the nine V12 initialization trajectories.','tab:mechanisms',['Task','Init.','Forecasts','Down','Pause','Retract','Resume','Edits'],lines,'lrrrrrrr')
    # An actual automatic-resume lineage, preserving the recorded current reference.
    r=find('Reizman_Suzuki_Case_4','V12','initialization',2);state=json.loads((ROOT/r['source']).read_text(encoding='utf-8'))
    all_events=[e for d in state['decisions'] for e in d['KF_feedback']]
    resume=next(e for e in all_events if e['reason']=='automatic_resume')
    events=[e for e in all_events if (e['id'],e['version'])==(resume['id'],resume['version'])]
    lines=[[e['query'],f"{e['p_LLM']:.3f}",f"{e['p_GP']:.3f}",int(e['success']),f"{e['Brier_gain']:+.4f}",f"{e['after']['mean']:+.4f}",f"{e['after']['weight']:.4f}",esc(e['after']['status'])] for e in events]
    longtable('kf_example.tex',f"Actual forecast sequence for note {esc(resume['id'])} version {resume['version']}, Suzuki Case 4 initialization 2. Each success indicator uses that query's current incumbent.",'tab:kf',['Query',r'$p$',r'$g$',r'$s$',r'$z$',r'$m$','Weight','State'],lines,'rrrrrrrl')
    inv=json.loads((ROOT/'experiments/reports/paper_v12/historical_inventory.json').read_text(encoding='utf-8'))
    lines=[]
    for g in inv['cohorts']:
        complete=sum(n for s,n in g['statuses'].items() if s in ('COMPLETE','COMPLETED'));other=g['records']-complete
        lines.append([esc(g['cohort']),g['records'],complete,other])
    longtable('historical_groups.tex','Catalogue coverage by historical experiment group. Complete is the recorded execution status, not a scientific acceptance decision.','tab:catalogue',['Experiment group','Records','Complete','Other status'],lines,'lrrr')

def workflow():
    fig,ax=plt.subplots(figsize=(10.5,3.5));ax.set(xlim=(0,10.5),ylim=(0,3.6));ax.axis('off')
    boxes=[(.1,2.05,2.0,1.05,'1  Observed history\nFit native GP\nTop-5 qLogEI menu'),(2.8,2.05,2.0,1.05,'2  LLM reasoning\nOne official choice\nScoped Notes'),(5.5,2.05,2.0,1.05,'3  Seal prediction\nP(beat incumbent)\nBefore outcome'),(8.2,2.05,2.0,1.05,'4  Observe result\nSelected point only\nUpdate GP history'),(5.5,.25,2.0,1.05,'5  Relative feedback\nBrier gain vs GP\nKF weight + lifecycle'),(2.8,.25,2.0,1.05,'6  Evidence memory\nRetain / revise / retire\nEvent-triggered edits')]
    for x,y,w,h,s in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.07',fc='#eef6f8',ec='#367b8b',lw=1.2));ax.text(x+w/2,y+h/2,s,ha='center',va='center',fontsize=10)
    arrow=dict(arrowstyle='->',lw=1.5,color='#52616b')
    for x in (2.15,4.85,7.55):ax.annotate('',xy=(x+.6,2.58),xytext=(x,2.58),arrowprops=arrow)
    ax.annotate('',xy=(6.5,1.36),xytext=(9.2,2.0),arrowprops=dict(arrow,connectionstyle='angle,angleA=-90,angleB=0,rad=8'))
    ax.annotate('',xy=(4.9,.78),xytext=(5.4,.78),arrowprops=dict(arrow,linestyle='--'))
    ax.annotate('',xy=(3.8,1.97),xytext=(3.8,1.37),arrowprops=dict(arrow,linestyle='--'))
    ax.text(.15,.75,'GP values stay unchanged\nby knowledge annotations',fontsize=9,color='#33434b')
    save(fig,'workflow')

def trajectories():
    fig,axs=plt.subplots(1,3,figsize=(10.5,3.2),layout='constrained')
    for i,(ax,t) in enumerate(zip(axs,FOCUS)):
        for m in ('BO','Reasoning BO','V12'):
            r=find(t,m);ax.plot(range(17),r['best_curve'],color=COL[m],lw=1.8,label=m,ls={'BO':'--','Reasoning BO':':','V12':'-'}[m])
        ax.set(title=LABEL[t],xlabel='Query after initialization',ylabel='Peak area (a.u.)' if t=='HPLC' else 'Cumulative best yield (%)',xticks=[0,4,8,12,16]);ax.grid(alpha=.18)
        ax.text(-.16,1.03,chr(97+i),transform=ax.transAxes,fontweight='bold')
    axs[0].legend(fontsize=8,loc='lower right');save(fig,'original_trajectories')

def repetitions():
    fig,axs=plt.subplots(1,3,figsize=(10.5,3.0),layout='constrained')
    for i,(ax,t) in enumerate(zip(axs,FOCUS)):
        rs=[r for r in R if r['task']==t and r['phase']=='LLM_repeat'];ys=[r['SearchAUC'] for r in rs];b=find(t,'BO')['SearchAUC'];orig=find(t,'V12')['SearchAUC']
        ax.axhline(b,color=COL['BO'],ls='--',lw=1.3,label='BO');ax.scatter([1,2,3],ys,s=45,color=COL['V12'],zorder=4,label='Fresh V12')
        ax.scatter([4],[orig],s=50,facecolor='white',edgecolor=COL['V12'],marker='D',zorder=4,label='Original V12')
        ax.set(title=LABEL[t],ylabel='SearchAUC (native units)',xticks=[1,2,3,4],xticklabels=['r1','r2','r3','orig.'],xlim=(.5,4.5));ax.grid(axis='y',alpha=.2)
        for x,y in zip([1,2,3],ys):ax.annotate(f'{y:.2f}',(x,y),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
        ax.margins(y=.23);ax.text(-.16,1.03,chr(97+i),transform=ax.transAxes,fontweight='bold')
    axs[0].legend(fontsize=7,loc='upper right');save(fig,'llm_repetitions')

def initialization():
    fig,axs=plt.subplots(2,1,figsize=(7.2,5.0),sharex=True,layout='constrained')
    colors=['#006d89','#c77d28','#7b5a9e'];marks=['o','s','^']
    for ax,key in zip(axs,('AUC_percent','best_percent')):
        ax.axhline(0,color='#667078',lw=1)
        for s in (1,2,3):
            ps=[next(x for x in E['paired'] if x['task']==t and x['phase']=='initialization' and x['method']=='V12' and x['seed']==s) for t in FOCUS]
            x=np.arange(3)+(s-2)*.18;y=[r[key] for r in ps];ax.scatter(x,y,color=colors[s-1],marker=marks[s-1],s=48,label=f'Init. {s}',zorder=4)
            for xx,yy in zip(x,y):ax.annotate(f'{yy:+.1f}',(xx,yy),xytext=(0,7 if yy>=0 else -13),textcoords='offset points',ha='center',fontsize=8)
        ax.grid(axis='y',alpha=.2);ax.set_ylabel(('SearchAUC' if key=='AUC_percent' else 'Final best')+' change vs BO (%)');ax.margins(y=.25)
    axs[0].legend(ncol=3,fontsize=9,loc='upper right');axs[1].set_xticks(range(3),[LABEL[t] for t in FOCUS]);save(fig,'initialization_effects')

def lifecycle():
    cases=[(FOCUS[0],3),(FOCUS[1],2),(FOCUS[2],1)]
    fig,axs=plt.subplots(3,1,figsize=(10.2,7.3),layout='constrained')
    for ax,(t,s) in zip(axs,cases):
        r=find(t,'V12','initialization',s);path=ROOT/r['source'];state=json.loads(path.read_text(encoding='utf-8'))
        hs=state['hypotheses'];keys=[(h['id'],h['version']) for h in hs];arr=np.full((len(keys),16),np.nan)
        for q in range(1,17):
            post=state if q==16 else json.loads((path.parent/'rounds'/f'{q+1:02d}_commit.json').read_text(encoding='utf-8'))['prestate']
            for h in post['hypotheses']:arr[keys.index((h['id'],h['version'])),q-1]=h['kf']['weight']
        im=ax.imshow(arr,vmin=0,vmax=.8,cmap='Blues',aspect='auto',extent=[.5,16.5,len(keys)-.5,-.5],interpolation='nearest')
        ax.set(yticks=range(len(keys)),yticklabels=[f'{k[0]} v{k[1]}' for k in keys],xticks=[1,4,8,12,16],title=f'{LABEL[t]}, initialization {s}')
        for d in state['decisions']:
            for e in d['KF_feedback']:
                mark={'automatic_pause':'s','automatic_retract':'x','automatic_resume':'^'}.get(e['reason'])
                if mark:ax.scatter(e['query'],keys.index((e['id'],e['version'])),marker=mark,s=60,c='#d47719',linewidths=1.4)
    axs[-1].set_xlabel('Query');cb=fig.colorbar(im,ax=axs,shrink=.65,pad=.02);cb.set_label('Calibration weight')
    for mark,label in [('s','Pause'),('x','Retract'),('^','Resume')]:axs[0].scatter([],[],marker=mark,s=40,c='#d47719',label=label)
    axs[0].legend(ncol=3,fontsize=8,loc='upper right',bbox_to_anchor=(1,1.36));save(fig,'lifecycle')

def development():
    methods=['Reasoning BO','V8','V9','V10','V11','V12'];a=np.full((8,6),np.nan)
    for i,t in enumerate(TASKS):
        bo=find(t,'BO')
        for j,m in enumerate(methods):
            r=find(t,m)
            if r:a[i,j]=100*(r['SearchAUC']/bo['SearchAUC']-1)
    lim=max(10,np.nanmax(abs(a)));fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    im=ax.imshow(a,cmap='RdBu',vmin=-lim,vmax=lim,aspect='auto');ax.set(xticks=range(6),xticklabels=methods,yticks=range(8),yticklabels=[LABEL[t] for t in TASKS])
    for i in range(8):
        for j in range(6):ax.text(j,i,'NA' if np.isnan(a[i,j]) else f'{a[i,j]:+.1f}%',ha='center',va='center',fontsize=10,color='white' if abs(a[i,j])>.65*lim else '#17222b')
    cb=fig.colorbar(im,ax=ax,pad=.025);cb.set_label('SearchAUC change vs matched BO (%)');save(fig,'development_heatmap')

if __name__=='__main__':
    tables();workflow();trajectories();repetitions();initialization();lifecycle();development()
    print('Generated six figure pairs and seven LaTeX tables.')
