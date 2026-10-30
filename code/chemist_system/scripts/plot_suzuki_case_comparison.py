"""Plot matched assistant-initial traces for requested Suzuki cases; read only."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.plot_suzuki_case2_comparison import METHODS,LABELS,COLORS,LINES,MARKERS

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager as fm
from matplotlib import pyplot as plt
import numpy as np


REGIME='assistant_initial'


def read(path):return json.loads(path.read_text(encoding='utf8'))


def source(case,method):
    if method==METHODS[0]:return ROOT/'data/runs/suzuki_regional_ablation/runs'/f'c{case}_s0_{REGIME}_{method}'/'state.json'
    if method==METHODS[3]:return ROOT/f'data/runs/suzuki_mixed_kalman/base/c{case}_s0/state.json'
    return ROOT/'data/runs/suzuki_mixed_comparison/runs'/f'c{case}_s0_{REGIME}_{method}'/'state.json'


def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main(cases):
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf8')
    out=ROOT/'outputs/suzuki_method_comparisons'/('cases_'+'_'.join(map(str,cases)))
    out.mkdir(parents=True,exist_ok=True)
    fm.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':fm.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),
        'font.size':11,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,
        'axes.edgecolor':'#8291A1','axes.labelcolor':'#26394A','text.color':'#26394A',
        'xtick.color':'#4D5D6C','ytick.color':'#4D5D6C','figure.facecolor':'white'})
    reports={
        'new':read(ROOT/'research/suzuki_regional_ablation/results.json'),
        'comparison':read(ROOT/'research/suzuki_mixed_comparison/results.json'),
        'bo':read(ROOT/'research/suzuki_mixed_kalman/results.json')}
    rows=[];points=[];provenance={};checks={}
    for case in cases:
        pool_path=ROOT/f'data/runs/suzuki_guarded/inputs/case{case}/pool.json'
        oracle_path=pool_path.with_name('oracle.npz')
        pool=read(pool_path)
        with np.load(oracle_path,allow_pickle=False) as f:truth=dict(zip(pool['ids'],f['values']))
        traces={};initial=None;recipes=set();first_fit=set();case_rows=[]
        for method,label in zip(METHODS,LABELS):
            path=source(case,method);state=read(path)
            assert state['status']=='COMPLETE' and state['case']==case and state['seed']==0
            assert len(state['history'])==20 and len(state['decisions'])==16 and len(pool['ids'])==4692
            assert len({h['candidate_id'] for h in state['history']})==20
            if initial is None:initial=state['history'][:4]
            assert state['history'][:4]==initial
            for h in state['history']:assert h['y']==float(truth[h['candidate_id']])
            for t,d in enumerate(state['decisions'],1):
                if 'base_acquisition' in d:meta=d['base_acquisition']
                else:
                    with np.load(ROOT/f'data/runs/suzuki_mixed_kalman/cache/c{case}_s0'/(d['base_score_key']+'.npz'),allow_pickle=False) as f:
                        meta=json.loads(str(f['metadata']))
                    assert not d['knowledge_enabled'] and d['mean_change_y']==0
                    assert d['base_score_key']==d['guided_score_key'] and d['selected_id']==d['reference_id']
                assert meta['gp_class']=='MixedSingleTaskGP' and meta['cat_dims']==[3]
                assert meta['category_map']==list(range(8)) and meta['mc_paths']==512
                assert meta['fit_seed']==28102026+case*10000+t and meta['mc_seed']==38102026+case*10000+t
                recipes.add(meta['recipe'])
                if t==1:first_fit.add(meta['fit_key'])
            y=np.array([h['y'] for h in state['history']])*100;best=np.maximum.accumulate(y)
            traces[method]=(y,best)
            row={'case':case,'seed':0,'knowledge_condition':REGIME,'method':method,'label':label,
                 'SearchAUC_percent':float(best[4:].mean()),'final_best_yield_percent':float(best[-1]),
                 'first_best_total_query':int(np.argmax(y)+1),'first_best_added_query':int(np.argmax(y)-3),
                 'best_candidate_id':state['history'][int(np.argmax(y))]['candidate_id']}
            if method==METHODS[0]:reported=reports['new']['tables'][f'Case{case}'][REGIME][method]
            elif method==METHODS[3]:reported=reports['bo']['tables'][f'Case{case}'][method]
            else:reported=reports['comparison']['tables'][f'Case{case}'][REGIME][method]
            assert abs(row['SearchAUC_percent']-reported['SearchAUC_mean'])<1e-10
            assert abs(row['final_best_yield_percent']-reported['final_yield_mean'])<1e-10
            rows.append(row);case_rows.append(row)
            for i,h in enumerate(state['history'],1):
                points.append({'case':case,'method':method,'total_query':i,'added_query':max(i-4,0),
                    'candidate_id':h['candidate_id'],'yield_percent':float(y[i-1]),'best_so_far_percent':float(best[i-1])})
            provenance[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
        assert len(recipes)==len(first_fit)==1
        for path in (pool_path,oracle_path):provenance[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
        checks[str(case)]={'same_initialization':True,'same_GP_recipe':next(iter(recipes)),
                           'same_initial_fit_key':next(iter(first_fit)),'all_80_observed_yields_match_oracle':True,
                           'metrics_match_existing_reports':True}
        write_csv(out/f'case{case}_metrics.csv',case_rows)
        fig,axes=plt.subplots(1,2,figsize=(14,5.5),sharey=True)
        handles=[]
        for i,(method,label,color,ls,marker) in enumerate(zip(METHODS,LABELS,COLORS,LINES,MARKERS)):
            y,best=traces[method]
            handle,=axes[0].plot(range(1,21),y,color=color,ls=ls,marker=marker,ms=4,lw=1.65,
                                 label=label,zorder=5 if i==0 else 2)
            axes[1].step(range(1,21),best,where='post',color=color,ls=ls,lw=1.9,zorder=5 if i==0 else 2)
            at=int(np.argmax(y)+1)
            axes[1].plot(at,max(y),marker=marker,ms=6,color=color,mfc='white',mew=1.5,zorder=8-i)
            handles.append(handle)
        for ax,title in zip(axes,('当次实验产率','累计最佳产率（比较搜索进度）')):
            ax.axvspan(.5,4.5,color='#EEF0F3',zorder=0)
            ax.axvline(4.5,color='#ACB6C1',ls=':',lw=1)
            ax.set(xlim=(.5,20.5),ylim=(0,100),xticks=[1,4,8,12,16,20],yticks=[0,20,40,60,80,100])
            ax.grid(axis='y',color='#E4E9EF',lw=.8);ax.set_axisbelow(True)
            ax.set_title(title,fontsize=12,loc='left',pad=13)
            ax.set_xlabel('实验次数（含前4次初始化）',labelpad=10)
        axes[0].set_ylabel('产率 / %',labelpad=9)
        reference='（参考案例）' if case==1 else ''
        fig.suptitle(f'Case {case}{reference}：新方案与LGBO、LLINBO-Justify、纯BO',fontsize=17,y=.98)
        fig.text(.5,.915,'助手初始建议 · 同一Mixed GP、4692候选和初始化 · seed=0 · 各4＋16次实验',ha='center',fontsize=10.5)
        fig.legend(handles,LABELS,ncol=4,loc='upper center',bbox_to_anchor=(.5,.875),frameon=False,fontsize=10.5)
        fig.subplots_adjust(left=.06,right=.985,bottom=.18,top=.72,wspace=.12)
        fig.text(.06,.035,'灰区为初始化；标记为首次达到各自最终最佳。纵轴统一0–100%。LLINBO为现有qLogEI适配版；LLM调用次数按各自机制保留。',fontsize=8.8,color='#617282')
        fig.savefig(out/f'case{case}_methods_comparison.png',dpi=190);plt.close(fig)
    write_csv(out/'metrics.csv',rows);write_csv(out/'trajectories.csv',points)
    metadata={'cases':cases,'knowledge_condition':REGIME,'seed':0,'methods':METHODS,
        'observations_per_trajectory':20,'init_queries':4,'BO_queries':16,'metrics':rows,
        'SearchAUC_definition':'Mean cumulative maximum yield after total queries5..20, in percentage units.',
        'first_best_definition':'First query attaining each trajectory own final best, not a common target across methods.',
        'checks':checks,'source_sha256':provenance,'new_GP_fits':0,'new_LLM_calls':0,
        'limits':'One seed per case; Case1 reference. Existing MixedGP/qLogEI/budget-protected LLINBO Justify adapter; LLM budgets differ.'}
    (out/'source_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'output':str(out),'metrics':rows},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',nargs='+',type=int,choices=(1,2,3,4),default=[1,3,4])
    main(list(dict.fromkeys(parser.parse_args().cases)))
