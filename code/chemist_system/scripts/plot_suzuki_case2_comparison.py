"""Source-backed Case 2 comparison; reads frozen traces and never runs BO/LLM."""
from pathlib import Path
import csv
import hashlib
import json
import sys

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager as fm
from matplotlib import pyplot as plt
import numpy as np


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/suzuki_case2_method_comparison'
REGIMES=('correct','wrong_category','mixed','assistant_initial')
CONDITIONS=('正确知识','错误类别','混合错误','助手初始建议')
METHODS=('G3_gate_KF_B4_event','LGBO_region','LLINBO_matched','Mixed_qLogEI_B0')
LABELS=('新方案（G3）','区域LGBO','LLINBO-Justify','纯BO（qLogEI）')
COLORS=('#24689B','#B77B16','#77823E','#4E5967')
LINES=('-','--',':','-.');MARKERS=('o','s','^','D')


def read(path):return json.loads(path.read_text(encoding='utf8'))


def source(regime,method):
    if method==METHODS[0]:return ROOT/'data/runs/suzuki_regional_ablation/runs'/f'c2_s0_{regime}_{method}'/'state.json'
    if method==METHODS[3]:return ROOT/'data/runs/suzuki_mixed_kalman/base/c2_s0/state.json'
    return ROOT/'data/runs/suzuki_mixed_comparison/runs'/f'c2_s0_{regime}_{method}'/'state.json'


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf8')
    OUT.mkdir(parents=True,exist_ok=True)
    font=Path('C:/Windows/Fonts/msyh.ttc');fm.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':fm.FontProperties(fname=str(font)).get_name(),
        'font.size':11,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,
        'axes.edgecolor':'#8291A1','axes.labelcolor':'#26394A','text.color':'#26394A',
        'xtick.color':'#4D5D6C','ytick.color':'#4D5D6C','figure.facecolor':'white'})
    pool=read(ROOT/'data/runs/suzuki_guarded/inputs/case2/pool.json')
    with np.load(ROOT/'data/runs/suzuki_guarded/inputs/case2/oracle.npz',allow_pickle=False) as f:truth=dict(zip(pool['ids'],f['values']))
    traces={};metrics=[];points=[];sources={};recipes=set();first_fit=set();initial=None
    for regime in REGIMES:
        for method,label in zip(METHODS,LABELS):
            path=source(regime,method);state=read(path)
            assert state['status']=='COMPLETE' and state['case']==2 and state['seed']==0
            assert len(state['history'])==20 and len(state['decisions'])==16
            assert len({h['candidate_id'] for h in state['history']})==20
            if initial is None:initial=state['history'][:4]
            assert state['history'][:4]==initial
            for h in state['history']:assert h['y']==float(truth[h['candidate_id']])
            for t,decision in enumerate(state['decisions'],1):
                if 'base_acquisition' in decision:meta=decision['base_acquisition']
                else:
                    cache=ROOT/'data/runs/suzuki_mixed_kalman/cache/c2_s0'/(decision['base_score_key']+'.npz')
                    with np.load(cache,allow_pickle=False) as f:meta=json.loads(str(f['metadata']))
                    assert not decision['knowledge_enabled'] and decision['mean_change_y']==0
                    assert decision['base_score_key']==decision['guided_score_key']
                    assert decision['selected_id']==decision['reference_id']
                assert meta['gp_class']=='MixedSingleTaskGP' and meta['cat_dims']==[3]
                assert meta['category_map']==list(range(8)) and meta['mc_paths']==512
                assert meta['fit_seed']==28122026+t and meta['mc_seed']==38122026+t
                recipes.add(meta['recipe'])
                if t==1:first_fit.add(meta['fit_key'])
            y=np.array([h['y'] for h in state['history']])*100;best=np.maximum.accumulate(y)
            traces[regime,method]=(state,y,best)
            row={'knowledge_condition':regime,'method':method,'label':label,
                 'SearchAUC_percent':float(best[4:].mean()),'final_best_yield_percent':float(best[-1]),
                 'first_best_total_query':int(np.argmax(y)+1),'first_best_added_query':int(np.argmax(y)-3),
                 'added_query_mean_yield_percent':float(y[4:].mean()),
                 'logical_LLM_calls':len(state['calls']) if method==METHODS[0] else 16 if method==METHODS[2] else 0}
            metrics.append(row)
            sources[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
            for i,h in enumerate(state['history'],1):
                points.append({'knowledge_condition':regime,'method':method,'total_query':i,'added_query':max(i-4,0),
                    'candidate_id':h['candidate_id'],'yield_percent':float(y[i-1]),'best_so_far_percent':float(best[i-1]),
                    'source_knowledge_condition':'none' if method==METHODS[3] else regime})
    assert len(recipes)==len(first_fit)==1
    assert all(traces[g,METHODS[3]][0]['history']==traces['assistant_initial',METHODS[3]][0]['history'] for g in REGIMES)
    for filename,rows in [('case2_metrics.csv',metrics),('case2_trajectories.csv',points)]:
        with (OUT/filename).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

    def background(ax):
        ax.axvspan(.5,4.5,color='#EEF0F3',zorder=0)
        ax.axvline(4.5,color='#ACB6C1',ls=':',lw=1)
        ax.set(xlim=(.5,20.5),ylim=(0,45),xticks=[1,4,8,12,16,20],yticks=[0,10,20,30,40])
        ax.grid(axis='y',color='#E4E9EF',lw=.8);ax.set_axisbelow(True)

    fig,axes=plt.subplots(1,2,figsize=(14,5.5),sharey=True)
    handles=[]
    for i,(method,label,color,ls,marker) in enumerate(zip(METHODS,LABELS,COLORS,LINES,MARKERS)):
        _,y,best=traces['assistant_initial',method]
        line,=axes[0].plot(range(1,21),y,color=color,ls=ls,marker=marker,ms=4,lw=1.65,label=label,zorder=5 if i==0 else 2)
        axes[1].step(range(1,21),best,where='post',color=color,ls=ls,lw=1.9,zorder=5 if i==0 else 2)
        at=int(np.argmax(y)+1);axes[1].plot(at,max(y),marker=marker,ms=6,color=color,mfc='white',mew=1.5,zorder=8-i)
        handles.append(line)
    for ax,title in zip(axes,('当次实验产率','累计最佳产率（比较搜索进度）')):
        background(ax);ax.set_title(title,fontsize=12,loc='left',pad=13);ax.set_xlabel('实验次数（含前4次初始化）',labelpad=10)
    axes[0].set_ylabel('产率 / %',labelpad=9)
    fig.suptitle('Case 2：新方案与LGBO、LLINBO-Justify、纯BO',fontsize=17,y=.98)
    fig.text(.5,.915,'助手初始建议 · 同一Mixed GP、4692候选和初始化 · seed=0 · 各4＋16次实验',ha='center',fontsize=10.5)
    fig.legend(handles,LABELS,ncol=4,loc='upper center',bbox_to_anchor=(.5,.875),frameon=False,fontsize=10.5)
    fig.subplots_adjust(left=.06,right=.985,bottom=.18,top=.72,wspace=.12)
    fig.text(.06,.035,'灰区为初始化。LLINBO为统一底座的qLogEI适配版；实验查询数相同，LLM调用次数按各自机制保留。',fontsize=9,color='#617282')
    fig.savefig(OUT/'case2_methods_comparison.png',dpi=190);plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(13,8.5),sharex=True,sharey=True)
    for ax,regime,title in zip(axes.flat,REGIMES,CONDITIONS):
        background(ax);ax.set_title(title,loc='left',fontsize=12,pad=12)
        for i,(method,color,ls,marker) in enumerate(zip(METHODS,COLORS,LINES,MARKERS)):
            _,y,best=traces[regime,method]
            ax.step(range(1,21),best,where='post',color=color,ls=ls,lw=1.8,zorder=5 if i==0 else 2)
            at=int(np.argmax(y)+1);ax.plot(at,max(y),marker=marker,ms=5.5,color=color,mfc='white',mew=1.3,zorder=8-i)
    for ax in axes[:,0]:ax.set_ylabel('累计最佳产率 / %')
    for ax in axes[-1]:ax.set_xlabel('实验次数（含4次初始化）',labelpad=9)
    fig.suptitle('Case 2：四种知识条件下的累计最佳产率',fontsize=17,y=.98)
    fig.text(.5,.935,'同一Mixed GP、候选集、初始化与4＋16预算 · seed=0',ha='center',fontsize=10.5)
    fig.legend(handles,LABELS,ncol=4,loc='upper center',bbox_to_anchor=(.5,.905),frameon=False,fontsize=10)
    fig.subplots_adjust(left=.07,right=.98,bottom=.13,top=.82,wspace=.13,hspace=.28)
    fig.text(.07,.04,'纯BO不接收知识，四个面板复用同一条纯BO轨迹；各知识条件不是重复种子。圆点/方块/三角/菱形标出首次最终最佳。',fontsize=9,color='#617282')
    fig.savefig(OUT/'case2_methods_four_conditions.png',dpi=190);plt.close(fig)
    natural=[v for v in metrics if v['knowledge_condition']=='assistant_initial']
    new_auc=natural[0]['SearchAUC_percent']
    metadata={'case':2,'seed':0,'GP_recipe':next(iter(recipes)),'same_initial_fit_key':next(iter(first_fit)),
        'source_sha256':sources,'natural_metrics':natural,
        'new_minus_comparator_AUC_pp':{v['method']:new_auc-v['SearchAUC_percent'] for v in natural[1:]},
        'definition':'SearchAUC = mean incumbent yields at total queries5..20, in percent units.',
        'boundary':'Single seed. LLINBO uses the existing MixedGP/qLogEI/budget-protection Justify adapter. Fixed-pool system comparison; LLM-call budgets differ. No new BO/LLM execution.'}
    (OUT/'source_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'output':str(OUT),'natural_metrics':natural,'new_minus_comparator_AUC_pp':metadata['new_minus_comparator_AUC_pp']},ensure_ascii=False))


if __name__=='__main__':main()
