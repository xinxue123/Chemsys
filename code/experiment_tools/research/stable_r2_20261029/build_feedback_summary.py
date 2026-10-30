"""Summarize all frozen feedback tie tests; this script never selects experiments."""
from pathlib import Path
import json
import sys
import numpy as np
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from chem.benchmarks import stable_feedback_study as study,reasoning_bo_data as data


def main():
    out=Path(__file__).resolve().parent;group=data.read(out/'feedback_stability.json');states={};reports=[];certificates=[]
    for seed in (0,1,2):
        variant=f'feedback_t{seed}';folder=out/variant;receipt=data.read(folder/'release_receipt.json')
        assert data.filehash(folder/'release_inventory.json')==receipt['inventory_sha256']
        for row in data.read(folder/'release_inventory.json')['files']:assert data.filehash(row['path'])==row['sha256'],row['path']
        reports.append(data.read(folder/'results.json'));certificates.append(data.read(folder/'validation_receipt.json'))
        for s in study.states(variant):
            assert s['status']=='COMPLETE' and len(s['history'])==20 and len(s['decisions'])==16
            states[seed,s['task']]=s
    comparisons=[]
    for task in data.SPECS:
        rows=[r for r in group['metrics'] if r['task']==task];baseline=reports[0]['frozen_controls']
        for name in ('Pure_BO','ReasoningBO'):
            b=next(r for r in baseline if r['task']==task and r['method']==name)
            comparisons.append({'task':task,'baseline':name,'baseline_AUC':b['SearchAUC'],
                'new_minimum_AUC_delta':min(r['SearchAUC'] for r in rows)-b['SearchAUC'],
                'new_minimum_best_delta_pp':min(r['final_best_percent'] for r in rows)-b['final_best_percent']})
    paths=study.preservation();steps=[]
    for (seed,task),s in states.items():
        for e in s['decisions']:
            if e['execution_mode']!='normal_R2':steps.append({'tie_seed':seed,'task':task,'query':e['iteration'],
                'mode':e['execution_mode'],'selected_id':e['selected_id'],'actual_percent':e['y']*100,
                'base_loss':e['base_opportunity_loss'],'budget':e['budget'],'gate_exception':e['acquisition_gate_exception']})
    api={'public_decision_requests':sum(r['API']['requests'] for r in reports),
        'received_new_responses_including_invalid':sum(r['API']['new_responses'] for r in reports),
        'reused_requests':sum(r['API']['reused_requests'] for r in reports),
        'prompt_tokens':sum(r['API']['usage'].get('prompt_tokens',0) for r in reports),
        'completion_tokens':sum(r['API']['usage'].get('completion_tokens',0) for r in reports)}
    result={'acceptance':group,'comparisons':comparisons,'selected_controlled_steps':steps,'API':api,
        'choices_replayed':sum(c['choices'] for c in certificates),'KF_updates_replayed':sum(c['KF_updates'] for c in certificates),
        'declared_gate_exceptions':sum(c['declared_acquisition_gate_exceptions'] for c in certificates),
        'protected_releases':paths,'scope':'Known seed0 development benchmarks and three fixed tie streams with memoized identical-input Agent replies; not independent LLM noise or initialization validation.'}
    data.write(out/'stable_strategy_results.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True)
    for row,task in enumerate(data.SPECS):
        for seed,color in zip((0,1,2),('#125eaa','#278bb5','#73b9c5')):
            y=np.array([h['y']*100 for h in states[seed,task]['history']]);x=np.arange(1,21)
            for col,values in enumerate((y,np.maximum.accumulate(y))):axes[row,col].plot(x,values,color=color,label=f'New tie{seed}',lw=1.5,alpha=.85)
        for name,key,color in [('Pure_BO','Pure_BO','#333333'),('ReasoningBO','ReasoningBO_point','#cc7b19')]:
            s=data.read(study.trust.parent.ROOT/'runs'/f'{task}_s0_{key}'/'state.json');y=np.array([h['y']*100 for h in s['history']])
            for col,values in enumerate((y,np.maximum.accumulate(y))):axes[row,col].plot(x,values,color=color,label=name,ls='--',lw=1.4)
        for col in (0,1):
            ax=axes[row,col];ax.set(xlim=(1,20),ylim=(0,103),xticks=[1,4,8,12,16,20],xlabel='Experiment index (first4 shared)',
                ylabel='UV product area (%)' if row==0 else 'Reported yield (%)',
                title=task.replace('_',' ')+' | '+('observed' if col==0 else 'cumulative best'))
            ax.axvline(4.5,color='#888888',lw=1,ls=':');ax.grid(alpha=.15)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=5,frameon=False)
    fig.suptitle('Fixed optimizer seed0 / tie seeds0,1,2 / same4+16 budget',fontsize=13)
    fig.tight_layout(rect=[0,.06,1,.96]);fig.savefig(out/'stable_strategy_comparison.png',dpi=180);plt.close(fig)
    passed=group['status']=='STABILITY_PASS'
    text=['**稳定化学决策策略：'+('通过约定的有限验收。' if passed else '未通过约定验收。')+'**','',
        '同一策略、同一源码与参数，在Perera Suzuki和Doyle Buchwald–Hartwig分别完成并列seed0/1/2，共6条完整轨迹。优化/初始化seed始终0，原共同4点、初始知识、全候选及16次追加查询不变。SearchAUC为16次追加查询后累计最佳的均值，首达序号包含前4次初始化。','',
        '| 数据集 | 最低AUC | 平均AUC | AUC极差 | 最低最终最佳/% | 所有门槛通过 |','|---|---:|---:|---:|---:|---|']
    for task,c in group['checks'].items():text.append(f"| {task} | {c['AUC_min']:.6f} | {c['AUC_mean']:.6f} | {c['AUC_range']:.6f} | {c['minimum_final_best_percent']:.6f} | {c['floor_all_pass'] and c['range_pass']} |")
    for task in data.SPECS:
        text += ['',f'**{task}全部结果**','', '| 方法 | 并列seed | SearchAUC | 最终最佳/% | 首达最佳实验序号 |','|---|---:|---:|---:|---:|']
        for name in ('Pure_BO','ReasoningBO'):
            r=next(r for r in reports[0]['frozen_controls'] if r['task']==task and r['method']==name)
            text.append(f"| {name}（冻结原对照） | — | {r['SearchAUC']:.6f} | {r['final_best_percent']:.6f} | {r['first_final_best_experiment']} |")
        for r in group['metrics']:
            if r['task']==task:text.append(f"| 新方案 | {r['tie_seed']} | {r['SearchAUC']:.6f} | {r['final_best_percent']:.6f} | {r['first_final_best_experiment']} |")
    text += ['', '![全部预定顺序及冻结对照](stable_strategy_comparison.png)','',
        '**策略实际改了什么**','',
        '原生Mixed GP仍作为健康路径；完全同分的采集值之间，用新单轴对照数×log(2+未观察类别数)/(1+已有单轴对照数)的信息代理，再按潜在SD和独立随机优先级裁决。该代理不是精确互信息，不改变任何不同分候选的顺序。普通反例保留覆盖轴完整集合，原Agent只提出一个受控比较，不依赖未校准的逐点产率分。','',
        '极端、未辨识的类别尺度诊断触发后，使用固定Morgan-Tanimoto主效应＋类别交互核的MixedSingleTaskGP，并安排最多两次联合检验。首次实验若比该关键类别原自身最佳高≥5个百分点，就继续该类别、换新背景；第二次组合≥75%时第3轮只改一项其余因素作局部跟进。模型是自定义类别核，不能说它与所有旧对照的默认核完全相同。','',
        '每轮先保存全候选GP均值/SD、基础与引导qLogEI、信息代理、预算集合及选择承诺，之后才读取该点真实值，再按原Joseph公式更新KF和同一区域信任。B4只检验知识与采集预算的兼容性，不把兼容当作化学正确。成功反例在原3次知识刷新预算内触发重审。','',
        '所有路径规范化JSON序列类型；相同系统、公开语义输入和完整自身前缀复用首个有效原始Agent输出，在各自运行重新生成计划。历史归档匹配按真实响应时间最早者选，不能按后续产率挑选。原始提示/回复、复用证明及模型响应记录均进入封存清单。','',
        f"已回放{result['choices_replayed']}次选择和{result['KF_updates_replayed']}次KF更新。控制实验明确越过正常机会损失门限{result['declared_gate_exceptions']}次，均计入16次查询；正常选择保持原门限。虚拟决策请求{api['public_decision_requests']}次，实际收到新回复{api['received_new_responses_including_invalid']}次（含格式失败），复用{api['reused_requests']}次。不是每个并列测试都重新独立抽样LLM。",'',
        '**结果能支持的结论**','',
        '通过只表示达到TASK.md预先固定的两套效果下限和三次AUC极差要求。Perera与ReasoningBO的AUC差、Doyle与纯BO的最终最佳差仍须按表解读，不能宣称所有指标全面胜出。结果来自已多次分析的seed0基准及并列集合，不是独立初始化、独立LLM噪声或跨任务普适性的证明。后期逐轮实测仍可波动；这里验收的是搜索效率与累计最佳。','',
        '改进主要体现在获得反例与继续新上下文的机会，以及减少任意同分/短名单影响；本轮没有单独证明KF、B4和事件各自的独立增量。前序所有负面完整轨迹和complete_t1/t2技术中断/Perera真实失败都保留，不以修复通过替代效果验收。','',
        '**复核和查看**','',
        '推荐入口为项目根目录stable_strategy.py。status读取全部结果；audit离线重放并拦截回执写入，不改封存文件、不调用新API；inspect指定数据集和查询轮次，显示选择的GP均值/SD、qLogEI、预算、信息代理、B4/KF及真实值，并列出完整数组和LLM档案路径。','',
        '```powershell\n.venv-official/Scripts/python.exe stable_strategy.py status\n.venv-official/Scripts/python.exe stable_strategy.py audit --tie-seed 0\n.venv-official/Scripts/python.exe stable_strategy.py inspect --dataset Perera_Suzuki --round 3 --tie-seed 0\n```','',
        '阶段设计：[STAGE11.md](STAGE11.md)；冻结标准：[TASK.md](TASK.md)；全部开发结果：[CURRENT_STATUS.md](CURRENT_STATUS.md)；组验收：[feedback_stability.md](feedback_stability.md)；机器汇总：[stable_strategy_results.json](stable_strategy_results.json)。',
        '数据集两进程并行、每进程Torch1线程、算子一次预热，循环内编译0次。全部旧对照与源文件哈希已验证，配置密钥不进入交付。']
    (out/'STABLE_STRATEGY.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    print(json.dumps({'status':group['status'],'checks':group['checks'],'API':api,'choices_replayed':result['choices_replayed']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
