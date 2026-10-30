"""Plot actual selected yields and incumbents of the current G3 Case 2 policy."""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager as fm
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import numpy as np


REPO=Path(__file__).resolve().parents[1]
OUT=REPO/'outputs/suzuki_case2_yield_curves'
RUNS=REPO/'data/runs/suzuki_regional_ablation/runs'
METHOD='G3_gate_KF_B4_event'
REGIMES=('correct','wrong_category','mixed','assistant_initial')
LABELS=('正确知识','错误类别','混合错误','助手初始建议')
BLUE='#24689B';GOLD='#B77B16'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    font=Path('C:/Windows/Fonts/msyh.ttc')
    fm.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':fm.FontProperties(fname=str(font)).get_name(),
        'font.size':11,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,
        'axes.edgecolor':'#8090A0','xtick.color':'#4D5D6C','ytick.color':'#4D5D6C',
        'axes.labelcolor':'#26394A','text.color':'#26394A','figure.facecolor':'white'})
    with np.load(REPO/'data/runs/suzuki_guarded/inputs/case2/oracle.npz',allow_pickle=False) as f:
        truth=f['values'].copy()
    pool=json.loads((REPO/'data/runs/suzuki_guarded/inputs/case2/pool.json').read_text(encoding='utf8'))
    oracle=dict(zip(pool['ids'],truth))
    states={};rows=[];sources={};summary={}
    for regime,label in zip(REGIMES,LABELS):
        path=RUNS/f'c2_s0_{regime}_{METHOD}'/'state.json'
        state=json.loads(path.read_text(encoding='utf8'))
        assert state['status']=='COMPLETE' and len(state['history'])==20 and len(state['decisions'])==16
        assert state['case']==2 and state['seed']==0 and state['method']==METHOD
        for h in state['history']:assert h['y']==float(oracle[h['candidate_id']])
        y=np.array([h['y'] for h in state['history']])*100;best=np.maximum.accumulate(y)
        states[regime]=(state,y,best)
        summary[regime]={'label':label,'final_best_percent':float(best[-1]),
            'first_best_total_query':int(np.argmax(y)+1),'first_best_added_query':int(np.argmax(y)+1-4),
            'initial_best_percent':float(max(y[:4]))}
        sources[path.relative_to(REPO).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
        refreshed={c['iteration']+4 for c in state['calls']}
        for i,(h,value,incumbent) in enumerate(zip(state['history'],y,best),1):
            rows.append({'case':2,'seed':0,'method':METHOD,'knowledge_condition':regime,
                'total_query':i,'added_query':max(i-4,0),'stage':'initialization' if i<=4 else 'BO',
                'candidate_id':h['candidate_id'],'yield_percent':float(value),
                'best_so_far_percent':float(incumbent),'LLM_refresh_before_query':i in refreshed})
    assert all(s[0]['history'][:4]==states['correct'][0]['history'][:4] for s in states.values())
    with (OUT/'case2_yield_points.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

    def draw(ax, regime, title=None):
        _,y,best=states[regime];x=np.arange(1,21)
        ax.axvspan(.5,4.5,color='#ECEFF2',zorder=0)
        ax.axvline(4.5,color='#AAB5C1',ls=':',lw=1)
        raw,=ax.plot(x,y,color=BLUE,marker='o',ms=4.3,lw=1.6,label='当次实验产率',zorder=3)
        incumbent,=ax.step(x,best,where='post',color=GOLD,ls=(0,(5,3)),lw=1.8,label='累计最佳产率',zorder=2)
        first=int(np.argmax(y)+1);peak=float(max(y))
        ax.annotate(f'{peak:.3f}%\n第{first}次实验',xy=(first,peak),xytext=(first,43.5),
            ha='center',va='center',fontsize=9,color='#6E501C',
            arrowprops={'arrowstyle':'-','color':'#B49C70','lw':.8})
        ax.set(xlim=(.5,20.5),ylim=(0,47),xticks=[1,4,8,12,16,20],yticks=[0,10,20,30,40])
        ax.grid(axis='y',color='#E4E9EF',lw=.8);ax.set_axisbelow(True)
        if title:ax.set_title(title,loc='left',fontsize=12,pad=12)
        return raw,incumbent

    fig,ax=plt.subplots(figsize=(11,5.5))
    handles=draw(ax,'assistant_initial')
    ax.set_xlabel('实验次数（包含前4次初始化）',labelpad=12);ax.set_ylabel('产率 / %',labelpad=10)
    fig.suptitle('新方案在Case 2上的产率轨迹',fontsize=18,y=.98)
    fig.text(.5,.91,'助手初始建议 · 区域引导＋KF＋B4＋事件刷新 · seed=0 · 4＋16预算',ha='center',fontsize=10.5)
    fig.legend([*handles,Patch(facecolor='#ECEFF2')],['当次实验产率','累计最佳产率','前4次初始化'],
               loc='upper center',bbox_to_anchor=(.5,.87),ncol=3,frameon=False,fontsize=10)
    fig.subplots_adjust(left=.085,right=.98,bottom=.18,top=.77)
    fig.text(.085,.035,'数据：冻结模拟器实际查询结果。低产率实验会使蓝线下降，累计最佳值保持不变。',fontsize=9,color='#627282')
    fig.savefig(OUT/'case2_new_scheme_yields.png',dpi=190);plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(13,8.6),sharex=True,sharey=True)
    for ax,regime,label in zip(axes.flat,REGIMES,LABELS):handles=draw(ax,regime,label)
    for ax in axes[-1]:ax.set_xlabel('实验次数（含4次初始化）',labelpad=9)
    for ax in axes[:,0]:ax.set_ylabel('产率 / %',labelpad=10)
    fig.suptitle('新方案在Case 2上的产率轨迹：四种知识条件',fontsize=17,y=.98)
    fig.text(.5,.935,'同一Mixed GP、候选集和初始化 · seed=0 · 每组4＋16次实验',ha='center',fontsize=10.5)
    fig.legend([*handles,Patch(facecolor='#ECEFF2')],['当次实验产率','累计最佳产率','前4次初始化'],
               loc='upper center',bbox_to_anchor=(.5,.908),ncol=3,frameon=False,fontsize=10)
    fig.subplots_adjust(left=.07,right=.98,bottom=.12,top=.82,wspace=.13,hspace=.28)
    fig.text(.07,.025,'每个面板是一条完整轨迹，四个面板使用同一纵轴；不将不同知识条件视为重复种子。',fontsize=9,color='#627282')
    fig.savefig(OUT/'case2_new_scheme_four_conditions.png',dpi=190);plt.close(fig)
    result={'case':2,'method':METHOD,'seed':0,'observations_per_condition':20,
            'initialization_queries':4,'BO_queries':16,'series':'observed simulator yield; cumulative maximum',
            'summary':summary,'source_sha256':sources,'new_GP_fits':0,'new_LLM_calls':0}
    (OUT/'source_metadata.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'output':str(OUT),'summary':summary},ensure_ascii=False))


if __name__=='__main__':main()
