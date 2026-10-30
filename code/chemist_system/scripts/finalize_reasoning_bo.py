"""Registered metrics, complete-only presentation and trace comparisons for M2."""
from pathlib import Path
from collections import Counter
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from chem.benchmarks import reasoning_bo_data as data
from chem.benchmarks import reasoning_bo_engine as engine
from chem.benchmarks import reasoning_bo_study as study
from chem.benchmarks import reasoning_bo_report as report

COLORS={'ReasoningBO':'#d47a16','New_R2':'#176bad','Gate_only':'#8853a5','LGBO_region':'#26856b',
    'BO_shared':'#333333','BO_q3_shared':'#7a828a','BO_q3_Sobol':'#a4a7aa'}
MARKERS={'ReasoningBO':'s','New_R2':'o','Gate_only':'^','LGBO_region':'D','BO_shared':'v',
    'BO_q3_shared':None,'BO_q3_Sobol':None}
NAMES={'ReasoningBO':'Reasoning BO','New_R2':'New R2 (batch)','Gate_only':'Gate only',
    'LGBO_region':'Regional LGBO','BO_shared':'BO: best3 of5','BO_q3_shared':'BO q3: shared init','BO_q3_Sobol':'BO q3: Sobol init'}


def figures(ss):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x=np.arange(31);evidence={};fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True)
    fig2,deltas=plt.subplots(1,2,figsize=(13,4.9),sharex=True)
    for row,task in enumerate(data.SPECS):
        evidence[task]={};all_values={}
        for method in engine.METHODS:
            selected=sorted([s for s in ss if s['task']==task and s['method']==method],key=lambda s:s['seed'])
            assert len(selected)==10
            best=np.array([[max(h['y'] for h in s['history'][:3])*100]+[d['incumbent_after']*100 for d in s['decisions']] for s in selected])
            actual=np.array([[np.mean([h['y'] for h in s['history'][:3]])*100]+[np.mean(d['observed_y'])*100 for d in s['decisions']] for s in selected])
            all_values[method]=best
            evidence[task][method]={'n':10,'batch':x.tolist(),'mean_best':best.mean(0).tolist(),
                'SEM_best':(best.std(0,ddof=1)/np.sqrt(10)).tolist(),'mean_batch_yield':actual.mean(0).tolist()}
            for col,values in enumerate((actual,best)):
                axes[row,col].plot(x,values.mean(0),color=COLORS[method],label=NAMES[method],
                    ls=('--' if method=='BO_q3_shared' else ':' if method=='BO_q3_Sobol' else '-'),
                    marker=MARKERS[method],markevery=5,ms=3,lw=1.5)
                if col==1:
                    sem=values.std(0,ddof=1)/np.sqrt(10)
                    axes[row,col].fill_between(x,values.mean(0)-sem,values.mean(0)+sem,color=COLORS[method],alpha=.09,lw=0)
                axes[row,col].set_ylim(0,103);axes[row,col].set_xlim(0,30);axes[row,col].set_xticks(np.arange(0,31,5))
                axes[row,col].grid(alpha=.18);axes[row,col].set_ylabel('Reported yield (%)')
                axes[row,col].set_title(task.replace('_',' ')+' | '+('batch mean' if col==0 else 'cumulative best'))
                axes[row,col].set_xlabel('Batch (3 simultaneous experiments)')
        for method in ('ReasoningBO','New_R2','Gate_only','LGBO_region'):
            diff=all_values[method]-all_values['BO_shared'];mean=diff.mean(0);sem=diff.std(0,ddof=1)/np.sqrt(10)
            ax=deltas[row];ax.plot(x,mean,color=COLORS[method],label=NAMES[method],
                marker=MARKERS[method],markevery=5,ms=3,lw=1.6)
            ax.fill_between(x,mean-sem,mean+sem,color=COLORS[method],alpha=.12,lw=0)
            evidence[task][method]['paired_best_delta_vs_BO_shared']=mean.tolist()
            evidence[task][method]['SEM_paired_best_delta'] = sem.tolist()
        deltas[row].axhline(0,color='#333333',ls='--',lw=1)
        deltas[row].set_title(task.replace('_',' '));deltas[row].set_ylabel('Cumulative best - BO best3 of5 (pp)')
        deltas[row].set_xlabel('Batch (same LLM initialization; n=10 paired repeats)')
        deltas[row].set_xlim(0,30);deltas[row].set_xticks(np.arange(0,31,5));deltas[row].grid(alpha=.18)
    handles,labels=axes[0,1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.018),ncol=4,fontsize=9,frameon=False)
    fig.suptitle('M2 complete replay: Mixed GP / qLogNEI / 10 fixed repeats per dataset',fontsize=13)
    fig.text(.5,.005,'Bands: +/-1 SEM. Batch0: 3 initial observations. After batch t: 3+3t observations. Regional methods share BO5-to3.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.125,1,.96]);fig.savefig(study.OUT/'comparison.png',dpi=180);plt.close(fig)
    h,l=deltas[0].get_legend_handles_labels();fig2.legend(h,l,loc='lower center',bbox_to_anchor=(.5,.025),ncol=4,frameon=False)
    fig2.suptitle('Increment from knowledge during optimization (shared initial experiments)',fontsize=13)
    fig2.text(.5,.006,'Paired differences; bands = +/-1 SEM over10 repeats, not95% confidence intervals. No Sobol-initialization difference is included.',ha='center',fontsize=8)
    fig2.tight_layout(rect=[0,.14,1,.94]);fig2.savefig(study.OUT/'knowledge_increment.png',dpi=180);plt.close(fig2)
    data.write(study.OUT/'chart_data.json',evidence)


def main():
    p=study.check_protocol();ss=report.states()
    assert len(ss)==140 and all(s['status']=='COMPLETE' for s in ss),'Never finalize an incomplete study'
    print(report.build(),flush=True)
    bykey={(s['task'],s['seed'],s['method']):s for s in ss};metrics={key:report.metric(s) for key,s in bykey.items()}
    paired=[];regret=[];divergences=[]
    for task in data.SPECS:
        maximum=float(np.load(data.ROOT/task/'oracle.npz')['values'].max()*100)
        for method in engine.METHODS:
            for seed in range(10):
                s=bykey[(task,seed,method)];m=metrics[(task,seed,method)]
                curve=[max(h['y'] for h in s['history'][:3])*100]+[d['incumbent_after']*100 for d in s['decisions']]
                reach=next((i for i,y in enumerate(curve) if abs(maximum-y)<1e-8),None)
                regret.append({'task':task,'method':method,'seed':seed,'table_max_percent':maximum,
                    'final_simple_regret_pp':maximum-m['final_best_percent'],'first_table_max_batch':reach})
        for control in ('ReasoningBO','BO_shared','Gate_only','LGBO_region','BO_q3_shared','BO_q3_Sobol'):
            diff=np.array([metrics[(task,seed,'New_R2')]['SearchAUC']-metrics[(task,seed,control)]['SearchAUC'] for seed in range(10)])
            paired.append({'task':task,'comparison':'New_R2 - '+control,'n':10,'mean_SearchAUC_delta_pp':float(diff.mean()),
                'SD_paired_delta':float(diff.std(ddof=1)),'wins':int((diff>1e-10).sum()),'ties':int((np.abs(diff)<=1e-10).sum()),'losses':int((diff< -1e-10).sum())})
        for left,right in (('New_R2','Gate_only'),('LGBO_region','Gate_only'),('ReasoningBO','BO_shared')):
            for seed in range(10):
                a,b=bykey[(task,seed,left)],bykey[(task,seed,right)]
                first=next((t for t in range(30) if set(a['decisions'][t]['selected_ids'])!=set(b['decisions'][t]['selected_ids'])),None)
                if first is None:continue
                da,db=a['decisions'][first],b['decisions'][first]
                prefix_equal=a['history'][:3*(first+1)]==b['history'][:3*(first+1)]
                # Observation ordering may differ within a simultaneous batch.
                prefix_equal={h['candidate_id']:h['y'] for h in a['history'][:3*(first+1)]}=={h['candidate_id']:h['y'] for h in b['history'][:3*(first+1)]}
                divergences.append({'task':task,'seed':seed,'comparison':left+' - '+right,
                    'first_divergent_batch':first+1,'same_observed_set_before_divergence':prefix_equal,
                    'left_selected':';'.join(da['selected_ids']),'right_selected':';'.join(db['selected_ids']),
                    'left_batch_max_percent':max(da['observed_y'])*100,'right_batch_max_percent':max(db['observed_y'])*100,
                    'immediate_incumbent_delta_pp':(da['incumbent_after']-db['incumbent_after'])*100,
                    'left_fallback':da['fallback'],'left_amplitude':da['amplitude'],
                    'knowledge_geometry_equal':da['knowledge']==db['knowledge'],
                    'left_refreshed_at_divergence':any(c['iteration']==first+1 for c in a.get('calls',[]))})
    report.write_csv(study.OUT/'paired_summary.csv',paired,list(paired[0]))
    report.write_csv(study.OUT/'regret.csv',regret,list(regret[0]))
    report.write_csv(study.OUT/'first_divergence.csv',divergences,list(divergences[0]))
    phases={}
    for folder in (study.ROOT/'requests').iterdir():
        if not (folder/'context.json').exists():continue
        context=data.read(folder/'context.json');mode=context.get('mode','region')
        phase='ReasoningBO_optimization' if mode.startswith('optimization_') else mode
        if mode=='region':phase='New_R2_refresh' if context['public']['phase']=='event_refresh' else 'shared_initial_knowledge'
        row=phases.setdefault(phase,{'phase':phase,'valid_decisions':0,'API_responses':0,'prompt_tokens':0,'completion_tokens':0})
        row['valid_decisions']+=int((folder/'validated.json').exists())
        for path in folder.glob('attempt*/response.json'):
            usage=data.read(path)['usage'];row['API_responses']+=1
            row['prompt_tokens']+=usage.get('prompt_tokens',0);row['completion_tokens']+=usage.get('completion_tokens',0)
    report.write_csv(study.OUT/'api_phases.csv',list(phases.values()),['phase','valid_decisions','API_responses','prompt_tokens','completion_tokens'])
    figures(ss)
    text=(study.OUT/'report.md').read_text(encoding='utf8')
    paired_values={(r['task'],r['comparison'].split(' - ')[1]):r['mean_SearchAUC_delta_pp'] for r in paired}
    su,do='Perera_Suzuki','Doyle_Buchwald_Hartwig'
    conclusion=(f"本轮未支持新方案全面优于对照：相对仅预算门限，Suzuki的平均SearchAUC增量为{paired_values[(su,'Gate_only')]:+.4f}个百分点，"
        f"Doyle为{paired_values[(do,'Gate_only')]:+.4f}个百分点；相对Reasoning BO分别为{paired_values[(su,'ReasoningBO')]:+.4f}、{paired_values[(do,'ReasoningBO')]:+.4f}个百分点。"
        f"Doyle最高SearchAUC均值来自Sobol初始化的直接q=3纯BO（{np.mean([metrics[(do,s,'BO_q3_Sobol')]['SearchAUC'] for s in range(10)]):.4f}），"
        f"新方案为{np.mean([metrics[(do,s,'New_R2')]['SearchAUC'] for s in range(10)]):.4f}。"
        '这些是10次重复的描述性结果，不作统计显著性声明。\n\n'
        f"优化循环内有效LLM决策：新方案{phases['New_R2_refresh']['valid_decisions']}次、Reasoning BO{phases['ReasoningBO_optimization']['valid_decisions']}次；"
        f"对应API成功响应（含格式重试）分别为{phases['New_R2_refresh']['API_responses']}与{phases['ReasoningBO_optimization']['API_responses']}次。"
        '二者均完整执行每条93次实验查询。该分项不含共享初始化及共同初始知识，详见[api_phases.csv](api_phases.csv)。\n\n')
    text=text.replace('每套数据10个预定种子；',conclusion+'每套数据10个预定种子；',1)
    text=text.replace('未完成时表格仅列可用完成数，禁止以缺失轨迹推断完整胜负。\n','')
    text=text.replace('SearchAUC：30个追加批次结束后的累计最佳产率的均值（百分点）。同时批次不人为排序。',
        'SearchAUC：30个追加批次结束后的累计最佳产率的均值（百分点）。同时批次不人为排序。首达最终最佳指本轨迹自己的最终最佳，并非统一产率目标；[regret.csv](regret.csv)另列全表最大值的达标批次。')
    text+='\n## 逐批曲线\n\n![逐批产率与累计最佳](comparison.png)\n\n'
    text+='阴影为10次重复的±1 SEM；指标表中的±SD是重复间标准差。每批三个点同时揭示，曲线不人为给批内点排序。\n\n'
    text+='![共享初始化下的知识增量](knowledge_increment.png)\n\n'
    text+='第二张图显示相对BO_shared的配对累计最佳差值；Sobol初始化差异没有混入该图。两个子图分别取纵轴范围，请按百分点刻度读取。\n\n'
    text+='## 新方案的配对SearchAUC增量\n\n| 数据 | 对照 | 平均增量/百分点 | 胜/平/负 |\n|---|---|---:|---:|\n'
    for row in paired:text+=f"| {row['task']} | {row['comparison'].split(' - ')[1]} | {row['mean_SearchAUC_delta_pp']:+.4f} | {row['wins']}/{row['ties']}/{row['losses']} |\n"
    text+='\n同初始化只隔离整体策略的增量；New_R2同时含KF、B4与事件刷新，不能把其整条轨迹增量归因给单个模块。首个分歧批次详见[first_divergence.csv](first_divergence.csv)。\n'
    text+='\n## 证据与适配边界\n\n'
    text+='- 原始数据对应：[Suzuki](perera_original_reconciliation.json)、[Doyle](doyle_original_reconciliation.json)。Suzuki使用作者保留两位小数的UV面积百分比，Doyle使用原始实验产率。\n'
    text+='- [官方边界差异](source_scope_differences.json)：Suzuki启动脚本漏逗号拼接了两个配体；Doyle启动列表有重复添加剂并漏一个类别。本次从作者CSV枚举，未采用拼接/遗漏选项。原协议的净计数简写已作[事实澄清](protocol_factual_errata.json)，候选池与实验未改变。\n'
    text+='- [运行资源调整](operational_concurrency.json)只改变并行度；[空记录兼容](execution_compatibility.json)保留全部旧结果并从中断前缀继续，未修改采集或KF公式。\n'
    text+='- [数值审计](numeric_audit.json)、[真实模型输入审计](input_audit.json)、[全轨迹校验](validation_receipt.json)与[13项代码校验](qualification_receipt.json)给出各自检验边界。\n'
    text+='- 每套数据10个不同的LLM初始实验集合。API未固定服务端采样种子，使用已登记的模型别名及可见响应/指纹记录。\n'
    text+='- 作者Suzuki表中LiOtBu对应的结构字符串含三碳烷氧基片段，需另行核对化学标注；本次为保持作者数据表示未修改。不能仅凭此观察归因方法胜负。\n'
    text+='- 提供给区域Agent的任务元数据含数据集名称；所有方法均禁止调用未观测产率，但离线公开数据实验不能证明没有预训练记忆。\n'
    (study.OUT/'report.md').write_text(text,encoding='utf8')
    data.write(study.OUT/'presentation_receipt.json',{'protocol_sha256':p['protocol_sha256'],
        'finalizer_sha256':data.filehash(__file__),'complete_trajectories':140,'n_per_task_method':10,
        'metric_definitions_changed':False,'comparison_png_sha256':data.filehash(study.OUT/'comparison.png'),
        'increment_png_sha256':data.filehash(study.OUT/'knowledge_increment.png')})
    print(json.dumps({'status':'COMPLETE','paired':paired},indent=2),flush=True)


if __name__=='__main__':main()
