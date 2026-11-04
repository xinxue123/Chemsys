"""Report the frozen direct-weight comparison without new scientific queries."""
from collections import Counter
from pathlib import Path
from statistics import fmean
import argparse,csv,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from program_kf_weight import study,VARIANT,CELLS,ARMS
from v13_origin_ablation import queue

OUT=ROOT/'experiments/reports'/VARIANT
NAMES=dict(fixed='Fixed weight',dynamic='Dynamic KF',v13='V13',bo='BO',reasoning_bo='Reasoning BO')


def csv_write(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def measure(task,seed,method,state,path):
    best=np.maximum.accumulate([h['raw_value'] for h in state['history']])
    return dict(task=task,seed=seed,method=method,queries=len(best),SearchAUC=fmean(best),
        final_best=float(best[-1]),first_final_best_query=int(np.flatnonzero(best==best[-1])[0])+1,
        state_path=study.rel(path))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prefix-only',action='store_true');args=parser.parse_args()
    p=study.check(True);OUT.mkdir(parents=True,exist_ok=True)
    prefix=study.read(OUT/'prefix_audit.json');assert prefix['status']=='PASS'
    stats=[]
    with (OUT/'prefix_scores.csv').open(encoding='utf-8-sig') as f:pr=list(csv.DictReader(f))
    for r in prefix['summaries']:
        rr=[x for x in pr if x['task']==r['task'] and int(x['seed'])==r['seed']]
        r=dict(r,min_dynamic_weight=min(float(x['dynamic_weight']) for x in rr),
            max_dynamic_weight=max(float(x['dynamic_weight']) for x in rr))
        stats.append(r)
    (OUT/'prefix_summary.json').write_text(json.dumps(stats,indent=2)+'\n',encoding='utf-8')
    if args.prefix_only:
        print(json.dumps(stats));return
    audit=study.read(OUT/'audit.json');assert audit['status']=='PASS_COMPLETE'
    metrics=[];rounds=[];states={};effects=[];round_effects=[]
    for task,seed in CELLS:
        cell=f'{study.SHORT[task]}{seed}'
        for method in ('bo','reasoning_bo','v13','fixed','dynamic'):
            path=study.base()/'runs'/f'trace_{cell}_{method}'/'state.json' if method in ARMS else study.reference(p,task,seed,method)
            s=study.read(path);assert len(s['history'])==20
            states[task,seed,method]=s;m=measure(task,seed,method,s,path);metrics.append(m)
            best=-float('inf')
            for q,(h,d) in enumerate(zip(s['history'],s['decisions']),1):
                best=max(best,h['raw_value']);packet=None
                if method in ARMS:packet=study.read(path.parent/'rounds'/f'{q:02}_commit.json')['decision']
                rounds.append(dict(task=task,seed=seed,method=method,query=q,selected_id=h['candidate_id'],
                    objective=h['raw_value'],cumulative_best=best,
                    program_weight=packet['ranking']['weight'] if packet else None,
                    LLM_probability=packet['actual_p_LLM'] if packet else None,
                    GP_probability=packet['actual_p_GP'] if packet else None,
                    proposed_id=packet['proposed_id'] if packet else None,
                    differs_from_GP=h['candidate_id']!=d['BO_reference_id'],
                    KF_Brier_gain=d.get('ranker_feedback',{}).get('Brier_gain'),
                    fixed_weight_shadow_selected=packet['fixed_weight_shadow']['selected_id'] if packet else None,
                    GP_all_tied=packet['ranking']['GP_all_tied'] if packet else None))
        look={m['method']:m for m in metrics if (m['task'],m['seed'])==(task,seed)}
        f,d=look['fixed'],look['dynamic']
        fb=np.maximum.accumulate([h['raw_value'] for h in states[task,seed,'fixed']['history']])
        db=np.maximum.accumulate([h['raw_value'] for h in states[task,seed,'dynamic']['history']])
        for q,(x,y) in enumerate(zip(fb,db),1):
            round_effects.append(dict(task=task,seed=seed,query=q,fixed_cumulative_best=float(x),
                dynamic_cumulative_best=float(y),difference=float(y-x),SearchAUC_contribution=float((y-x)/len(fb))))
        different=[q for q,(a,b) in enumerate(zip(states[task,seed,'fixed']['history'],states[task,seed,'dynamic']['history']),1)
                   if a['candidate_id']!=b['candidate_id']]
        shadow=[]
        for q in range(1,21):
            c=study.read((ROOT/d['state_path']).parent/'rounds'/f'{q:02}_commit.json')['decision']
            if c['selected_id']!=c['fixed_weight_shadow']['selected_id']:shadow.append(q)
        details=None
        if different:
            q=different[0]
            fc=study.read((ROOT/f['state_path']).parent/'rounds'/f'{q:02}_commit.json')
            dc=study.read((ROOT/d['state_path']).parent/'rounds'/f'{q:02}_commit.json')
            assert fc['GP_sha256']==dc['GP_sha256']
            assert fc['reasoning']['raw']==dc['reasoning']['raw']
            details=dict(query=q,fixed_id=fc['selected_id'],dynamic_id=dc['selected_id'],
                fixed_observed=states[task,seed,'fixed']['history'][q-1]['raw_value'],
                dynamic_observed=states[task,seed,'dynamic']['history'][q-1]['raw_value'],
                incumbent_before=max(h['raw_value'] for h in states[task,seed,'fixed']['history'][:q-1]),
                fixed_weight=fc['decision']['ranking']['weight'],dynamic_weight=dc['decision']['ranking']['weight'],
                GP_reference_id=fc['decision']['BO_reference_id'],
                input_and_all_five_forecasts_identical=True)
        effects.append(dict(task=task,seed=seed,delta_SearchAUC=d['SearchAUC']-f['SearchAUC'],
            delta_final_best=d['final_best']-f['final_best'],first_physical_divergence=different[0] if different else None,
            different_query_positions=different,dynamic_same_prefix_fixed_shadow_disagreements=shadow,
            first_divergence_details=details))
    csv_write('metrics.csv',metrics);csv_write('rounds.csv',rounds)
    csv_write('round_effects.csv',round_effects)
    (OUT/'paired_effects.json').write_text(json.dumps(effects,indent=2)+'\n',encoding='utf-8')
    lines=['程序化KF权重对照结果','',
        '仅Perera seed2和HPLC seed3。两阶段规则预先冻结；本轮没有改动V13或其他历史对照。',
        'KF校正整体候选预测通道的相对GP预测技能；不是单条化学知识的因果贡献。',
        'KF数值不进入LLM或编辑提示。两组共用预测格式、Note语义/权限/刷新规则与原生GP。',
        'q>=2时：S=(1-w)×GP同分平均Borda+w×LLM五点概率同分平均Borda。',
        '固定w=0.4；动态w为原生KF可信度。LLM完全同分时执行原GP首选。',
        'q1无观测、无GP、无改善概率，两组执行同一份官方提议，并计入20次查询。','',
        '阶段一：40个历史固定前缀，不新增实验查询；每个前缀仅一份五点预测，比较两种程序权重。']
    for r in stats:
        lines.append(f"{r['task']} seed{r['seed']}: 不同选点轮次{r['selection_disagreement_queries']}；GP五点完全同分轮次{r['GP_all_tied_queries']}；LLM完全同分轮次{r['LLM_uniform_queries']}；动态权重范围{r['min_dynamic_weight']:.6g}～{r['max_dynamic_weight']:.6g}。")
    lines+=['','阶段二：4条新轨迹×20次=80次模拟实验查询，均从0条观测开始。',
        '完整轨迹的KF均重置为原生先验；不继承离线诊断的KF、实验观测或知识状态。',
        'SearchAUC为实际20次查询后累计最佳值的均值，使用任务原始单位，不跨任务平均。',
        '任务 | 方法 | SearchAUC | 最终最佳 | 首次达到自身最终最佳的查询轮次']
    for m in metrics:
        lines.append(f"{m['task']} seed{m['seed']} | {NAMES[m['method']]} | {m['SearchAUC']:.6f} | {m['final_best']:.6f} | {m['first_final_best_query']}")
    lines+=['','动态KF相对固定权重的完整轨迹增量（唯一主对照）：']
    for e in effects:
        lines.append(f"{e['task']} seed{e['seed']}: SearchAUC {e['delta_SearchAUC']:+.6f}；最终最佳{e['delta_final_best']:+.6f}；首次轨迹分化q{e['first_physical_divergence']}；动态轨迹上固定权重影子不同选点轮次{e['dynamic_same_prefix_fixed_shadow_disagreements']}。")
        x=e['first_divergence_details']
        if x:
            lines.append(f"首处分化：固定w={x['fixed_weight']:.6f}选{x['fixed_id']}，实测{x['fixed_observed']:.6f}；动态w={x['dynamic_weight']:.6f}选{x['dynamic_id']}，实测{x['dynamic_observed']:.6f}；共同incumbent={x['incumbent_before']:.6f}。该轮GP、原始LLM输入及五点预测一致。")
        contributed=[r['query'] for r in round_effects if r['task']==e['task'] and abs(r['difference'])>1e-10]
        lines.append(f'累计最佳有差异的轮次：{contributed}。逐轮SearchAUC贡献见round_effects.csv。')
    lines+=['','本轮判断：',
        'Perera两种权重的20个选点完全一致：新公共程序框架相对V13的SearchAUC改善不能归于动态KF。',
        'HPLC观察到动态KF相对固定权重的SearchAUC增量，最终最佳和首次达到最终最佳的轮次相同。',
        'HPLC的新动态策略仍未超过原V13及Reasoning BO的SearchAUC；局部模块增量不等于整体方法更优。']
    lines+=['','解释边界：',
        '固定前缀差异只能证明数值权重改变程序选择，不能单独证明优化收益；没有为未选候选查询标签。',
        '完整轨迹在首次分化前，GP、菜单、预测原始响应严格一致。之后各自依据实际历史调用LLM并拟合GP，无法完全消除分化后的LLM响应波动。',
        '固定组同样计算影子KF，但不部署；两组的19次预测反馈更新均只针对实际执行点。',
        'V13/BO/Reasoning BO为原封存轨迹，仅作描述性参照。新公共预测/程序框架相对它们的差异不能全部归于KF。',
        '这里只有两个用户指定的任务/起始种子，各一份完整响应映射，不能声称多种子稳定优势。',
        'Brier技能与排序效用不是同一指标；此实验检验将其作为排序权重这一假设，并不预设其有效。',
        '所有相同科学输入在同一任务/起始编号内共用响应；没有跨任务、跨起始编号或按产率挑选响应。',
        '未执行的LLM提议不获得Note预测反馈；程序改选后的点只更新它预先登记的候选预测通道。',
        '完整预观测承诺、五候选分数、原始响应和真实子智能体调用回执均保留。']
    exceptions=list((study.base()/'responses').glob('*/*/schema_exception.json'))
    if exceptions:
        lines+=['',f'注释修复：{len(exceptions)}份响应在三次常规格式尝试后，另做一次预登记的作用域注释修复。',
            '额外修复仅删除所提议点作用域之外的Note预测；原始第一次有效的五点预测、提议点和其他字段完全锁定。',
            '初始失败、所有实际调用及失败批次回执全部保留；评分规则、GP、权重参数和查询预算没有改变。']
    if (study.base()/'alias_recovery_registration.json').exists():
        lines+=['', '共享响应恢复：Perera固定组曾因另一个对照已拥有相同响应请求而暂停。',
            '恢复过程等待原请求完成，复用完全相同的已准备输入和预测；未覆盖原请求，也未重采样原预测。',
            '两组在首次物理分化前的输入字节指纹和原始响应均经完整审计保持一致。']
    (OUT/'程序化KF权重对照结果.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8-sig')
    plots(rounds,pr)
    print(json.dumps(dict(status='COMPLETE',new_queries=80,metrics=metrics,paired_effects=effects)))


def plots(rows,prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=dict(bo='#666666',reasoning_bo='#bd6e91',v13='#2489a3',fixed='#d89024',dynamic='#456bab')
    fig,axes=plt.subplots(2,2,figsize=(12,8.2),layout='constrained')
    for col,(task,seed) in enumerate(CELLS):
        ax=axes[0,col]
        for method in ('bo','reasoning_bo','v13','fixed','dynamic'):
            rr=[r for r in rows if (r['task'],r['seed'],r['method'])==(task,seed,method)]
            ax.plot([r['query'] for r in rr],[r['cumulative_best'] for r in rr],label=NAMES[method],
                color=colors[method],lw=2,ls='--' if method in ('bo','fixed') else '-',
                marker='o' if method=='dynamic' else None,ms=3)
        ax.set(title=f"{'Perera' if task=='Perera_Suzuki' else 'HPLC'} start {seed}",xlabel='Paid query',
            ylabel='Best product UV area (%)' if task=='Perera_Suzuki' else 'Best peak area',xticks=[1,5,10,15,20])
        ax.grid(alpha=.2)
        bottom=axes[1,col]
        # Each prefix contributes five score rows; select one deterministically.
        seen={}
        for r in prefix:
            if r['task']==task and int(r['seed'])==seed:seen.setdefault(int(r['query']),r)
        qs=sorted(seen);weights=[float(seen[q]['dynamic_weight']) for q in qs]
        bottom.axhline(.4,color=colors['fixed'],ls='--',label='Fixed weight')
        bottom.plot(qs,weights,color=colors['dynamic'],label='Dynamic weight',marker='o',ms=3)
        differences=[q for q in qs if len({r['candidate_id'] for r in prefix if r['task']==task and int(r['query'])==q
                     and (r['fixed_selected']=='True' or r['dynamic_selected']=='True')})>1]
        for q in differences:bottom.axvline(q,color='#bf524c',alpha=.35,lw=1)
        bottom.set(xlabel='Frozen historical prefix query',ylabel='Program weight',ylim=(-.02,.82),xticks=[1,5,10,15,20])
        bottom.grid(alpha=.2);bottom.legend(fontsize=8)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=5,frameon=False)
    fig.supxlabel('Top: new complete trajectories. Bottom: held V13 histories; red lines mark different program selections.',fontsize=9)
    fig.savefig(OUT/'comparison.png',dpi=180);fig.savefig(OUT/'comparison.pdf');plt.close(fig)


if __name__=='__main__':main()
