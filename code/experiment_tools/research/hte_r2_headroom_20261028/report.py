"""Summarize remaining headroom; all counterfactuals remain explicitly local."""
from pathlib import Path
from collections import Counter
import json
import numpy as np
from analyze import OUT,BASE,ROOT,OLD,TASKS,read,save,hashed,preserved


def facts(results):
    out={'coverage':{},'recovery_episodes':{},'prediction_gain_not_selection_gain':{},'target_candidate':{},
        'local_switches':{},'candidate_pools':{}}
    for task,r in results.items():
        pool=read(BASE/'data/benchmarks/reasoning_bo'/task/'pool.json');lookup={k:i for i,k in enumerate(pool['ids'])}
        states={'corrected_R2':read(ROOT/'runs'/f'{task}_s0_R2_region_trust'/'state.json'),
            'Pure_BO':read(OLD/'runs'/f'{task}_s0_Pure_BO'/'state.json'),
            'ReasoningBO':read(OLD/'runs'/f'{task}_s0_ReasoningBO_point'/'state.json')}
        out['coverage'][task]={}
        for method,s in states.items():
            rows=[pool['points'][lookup[h['candidate_id']]] for h in s['history']]
            changes={a:{'initial_levels':len({p[a] for p in rows[:4]}),'final_levels':len({p[a] for p in rows}),
                'new_levels_after_init':len({p[a] for p in rows}-{p[a] for p in rows[:4]})} for a in pool['axes']}
            record={'axes':changes,'coverage_by_experiment':{a:[len({p[a] for p in rows[:n]}) for n in range(1,21)] for a in pool['axes']}}
            if 'Additive' in pool['axes']:
                record['added_query_additive_ids']=[pool['levels']['Additive'].index(p['Additive'])+1 for p in rows[4:]]
                record['added_query_additive_counts']=dict(Counter(record['added_query_additive_ids']))
            out['coverage'][task][method]=record
        out['recovery_episodes'][task]=[{'query':v['query'],'positive_evidence_query':v['recovery_evidence']['query'],
            'predictive_gain_pp_squared':v['recovery_evidence']['squared_error_gain_pp_squared'],
            'selected':v['selected_id'],'actual_percent':v['yield_percent'],
            'reference_percent':v['reference_y_percent'],'difference_pp':v['selected_vs_reference_pp']}
            for v in r['rows'] if v['trust_recovered']]
        out['prediction_gain_not_selection_gain'][task]=[{'query':v['query'],'predictive_gain':v['actual_predictive_error_gain_pp_squared'],
            'selected':v['yield_percent'],'reference':v['reference_y_percent']}
            for v in r['rows'] if v['amplitude']>0 and v['actual_predictive_error_gain_pp_squared']>0 and v['selected_vs_reference_pp']<0]
        switches={}
        for variant in r['rows'][0]['variants']:
            if variant=='actual':continue
            changed=[]
            for row in r['rows']:
                a,b=row['variants']['actual'],row['variants'][variant]
                if a['id']!=b['id']:
                    changed.append({'query':row['query'],'actual':a['yield_percent'],'alternative':b['yield_percent'],
                        'point_delta_pp':b['yield_percent']-a['yield_percent'],
                        'immediate_incumbent_delta_pp':b['immediate_incumbent_gain_pp']-a['immediate_incumbent_gain_pp']})
            switches[variant]=changed
        out['local_switches'][task]=switches
        out['candidate_pools'][task]={name:{'queries_with95':[row['query'] for row in r['rows'] if row['candidate_pool_coverage'][name]['contains95']],
            'best_covered_percent':max(row['candidate_pool_coverage'][name]['oracle_best_percent'] for row in r['rows'])}
            for name in r['rows'][0]['candidate_pool_coverage']}
        if task.startswith('Doyle'):
            target=lookup['p01032'];detail=[]
            for row in r['rows']:
                t=row['query'];folder=ROOT/'runs'/states['corrected_R2']['run_id']/'rounds'
                with np.load(folder/f'{t:02d}_candidates.npz') as a:
                    ranked=sorted(np.flatnonzero(a['available']),key=lambda j:(-float(a['base_qLogEI'][j]),pool['ids'][j]))
                    ref=ranked[0];detail.append({'query':t,'rank':ranked.index(target)+1,
                        'loss':float(a['base_qLogEI'][ref]-a['base_qLogEI'][target]),'budget':row['budget'],
                        'eligible':bool(a['eligible'][target]),'GP_mean_percent':float(a['mu_y'][target]*100),
                        'GP_latent_SD_pp':float(a['sd_y'][target]*100)})
            out['target_candidate'][task]={'candidate_id':'p01032','true_yield_percent':98.28722602,
                'point':pool['points'][target],'rows':detail}
    p=results['Perera_Suzuki'];q6=p['rows'][5];upper=p['global_oracle_max_percent']
    assert q6['reference_y_percent']==upper==100.
    conditional=(sum(v['incumbent_after_percent'] for v in p['rows'][:5])+11*upper)/16
    out['Perera_headroom']={'endpoint_headroom_pp':upper-p['final_best_percent'],
        'absolute_SearchAUC_ceiling':upper,'ceiling_gap_pp':upper-p['SearchAUC'],
        'pick_existing_BO_reference_at_query6_conditional_SearchAUC':conditional,
        'conditional_gain_pp':conditional-p['SearchAUC'],
        'limit':'ONLY conditional on the actual first5 queries. No deployable oracle-free trigger has been demonstrated.'}
    return out


def plot(results,f):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,8.5))
    d=results['Doyle_Buchwald_Hartwig']['rows'];p=results['Perera_Suzuki']['rows']
    ax=axes[0,0];x=np.arange(1,17)
    ax.plot(x,[r['yield_percent'] for r in d],'-o',c='#176bad',ms=3,label='Actual corrected R2')
    ax.plot(x,[r['eligible_oracle_best_percent'] for r in d],'--',c='#d47a16',label='Best truth inside budget (audit only)')
    ax.axhline(95,c='#888888',ls=':',label='95%');ax.axvline(4.5,c='#aaaaaa',ls=':')
    ax.set(ylim=(0,125),xlim=(1,16),xticks=[1,4,8,12,16],xlabel='Query after initialization4',ylabel='Doyle yield (%)',title='Later budget sets already miss high-yield regions')
    ax.legend(fontsize=8,loc='upper right');ax.grid(alpha=.15)
    ax=axes[0,1]
    for method,color in [('corrected_R2','#176bad'),('Pure_BO','#444444'),('ReasoningBO','#d47a16')]:
        values=f['coverage']['Doyle_Buchwald_Hartwig'][method]['coverage_by_experiment']['Additive']
        ax.plot(range(1,21),values,label=method,c=color,lw=1.6,marker='o',ms=3)
    ax.set(ylim=(0,7),xlim=(1,20),xticks=[1,4,8,12,16,20],xlabel='Experiment index (first4 shared)',ylabel='Distinct additive levels',title='Corrected R2 tests no new additive after initialization')
    ax.axvline(4.5,c='#888888',ls=':');ax.legend(fontsize=8);ax.grid(alpha=.15)
    ax=axes[1,0];xx=np.arange(6);width=.25
    for off,key,color,label in [(-width,'actual','#176bad','Current region guidance'),(0,'guided_top5_budget','#d47a16','Hard BO top5'),(width,'half_amplitude','#888888','Half amplitude')]:
        ax.bar(xx+off,[r['variants'][key]['yield_percent'] for r in p[:6]],width,color=color,label=label)
    ax.set(ylim=(0,125),xticks=xx,xticklabels=[f'q{i}' for i in range(1,7)],ylabel='Perera UV product area (%)',title='Simple restrictions can remove early regional gains')
    ax.legend(fontsize=8);ax.grid(axis='y',alpha=.15)
    ax=axes[1,1];eps=f['recovery_episodes']['Perera_Suzuki']+f['recovery_episodes']['Doyle_Buchwald_Hartwig'];xx=np.arange(len(eps));width=.35
    ax.bar(xx-width/2,[e['actual_percent'] for e in eps],width,color='#176bad',label='Choice after trust recovery')
    ax.bar(xx+width/2,[e['reference_percent'] for e in eps],width,color='#555555',label='Own BO reference (audit truth)')
    ax.set(ylim=(0,105),xticks=xx,xticklabels=['Perera q8','Perera q12','Doyle q11'],ylabel='Reported objective (%)',title='Recovery evidence does not certify the next choice')
    ax.legend(fontsize=8);ax.grid(axis='y',alpha=.15)
    fig.suptitle('Remaining headroom: frozen seed0 / same-history diagnostics',fontsize=13)
    fig.tight_layout(rect=[0,.02,1,.95]);fig.savefig(OUT/'headroom.png',dpi=180);plt.close(fig)


def report(results,f):
    text=['**还有改进空间，重点在早期知识验证与探索；单纯收紧候选或缩小KF幅度没有统一收益。**','',
        '对象为修正信任恢复后的R2正式v2两条轨迹。两套数据仍为seed=0、共同4次初始化＋16次查询；本轮不修改算法，不调用LLM，不拟合新GP，不运行新的完整优化轨迹。对32个冻结观测前缀重算局部替代选择，并保留全部负面结果。','',
        '**Perera：终点已满，剩下的是更早找到好点。**','',
        '当前最终最佳100%，SearchAUC98.5888；Reasoning BO为98.8325。终点没有提升空间，绝对理论AUC余量至多1.4113个百分点。第6轮自身BO首选已经是100%，却被知识引导换成95.06%。若仅在相同前5轮后改选这个BO首选，累计最佳从第6轮起保持100%，条件计算AUC为98.8031。该数值不是新实验结果，也尚无无需真值的在线判据保证只在正确时机改选。','',
        '**Doyle：第5轮以后再换局部排序已经很难追上。**','',
        '当前AUC80.1478、最终81.2755%，分别低于纯BO87.4571/98.7313%和Reasoning BO91.4567/98.2872%。在实际第5–16轮历史下，预算集里真实最高值均低于90%；当前GP把高产区域排到了预算之外。预算最初允许高产点，但前4轮的错误区域引导改变了后续数据与GP。这是路径依赖证据，不是新轨迹的性能上界。','',
        '16次追加实验中，15次为Additive_11，1次为Additive_12，未测试任何新添加剂。初始已观察4种添加剂，最后仍只有这4种；纯BO和Reasoning BO都增加了2种。高产p01032需要Additive_01和Base_03，而当前最佳81.28%的点使用Additive_11。关键是验证添加剂与其他条件的交互，不能把少量混杂观测概括为全局添加剂偏好。','',
        '**恢复判据仍然偏弱：预测增量不等于选点增量。**','',
        '| 场景 | 恢复后实际值/% | 同历史BO首选真值/% |','|---|---:|---:|']
    for task,episodes in f['recovery_episodes'].items():
        for e in episodes:text.append(f"| {task} 第{e['query']}轮 | {e['actual_percent']:.4f} | {e['reference_percent']:.4f} |")
    text+=['','这3次均有满足既定规则的先前真实预测改善证据，恢复逻辑没有算错；但一个位置上的改善不足以证明重拟合GP后整块区域的引导都有效。Perera第8轮损失约67.45个百分点，因之前已找到100%，SearchAUC不会反映这次较差选择。Doyle第11轮恢复后得到25.62%，低于BO首选43.31%。风险锁目前也只由B4冲突触发，不会在所有知识预测恶化时主动锁定。','',
        'Doyle第2/3轮还是更直接的反例：知识使所选点的预测误差下降，但所选真实值68.74/62.00%，都低于当时BO首选81.28%。依靠预测校准一项指标不能认证优化收益。','',
        '**已检查几种简单改法，不能直接当成改进方案。**','',
        '| 同历史单步改法 | 结果与限制 |','|---|---|',
        '| 硬限制BO前5 | Doyle第3/4轮由62.00/23.04%变成81.28/75.12%；Perera第1/2轮却由87.88/91.29%变成58.51/4.12%，会丢掉区域引导收益。 |',
        '| 幅度减半 | Doyle第3/4轮仍为62.00/23.04%；Perera前两轮同样退回58.51/4.12%。 |',
        '| 按系数减去1个后验标准差 | 没有阻止Doyle第3/4轮坏选择；也削弱了Perera早期收益。 |',
        '| 连续两次部署误差超过2.5σ且预测被知识恶化，则回退 | Perera未触发；Doyle仅第13轮触发，但该轮原本已回退，实际选择未改变。这个具体报警规则不值得直接推进。 |',
        '| 预算内用Hamming距离构造多样化5候选 | 某些前缀候选覆盖改善，但Doyle所有16轮的候选池仍均无95%点；它无法修复已偏离的固定历史。 |','',
        '这些都是同一实际历史的一步替代，不可拼成完整轨迹或相加宣称AUC收益。尤其不能将某轮替代值较好直接当作该方案已优于基线。','',
        '**最值得下一轮验证的是知识反例探索与分条诊断。**','',
        '第一优先：在原16次查询内预留少量验证实验，针对最强知识条目做受控反例；例如保持高产框架，测试偏好集合之外的添加剂。将“支持某个已观察组合”与“排除其他添加剂”区分，避免所有知识取交集后长期只试同一添加剂。反例选择必须依赖当时数据、化学假设与不确定性，不能照抄本次事后知道的最佳点。需要明确这种验证查询是否允许突破原机会损失门限，仍须计入同一总预算。','',
        '第二优先：把联合区域校正拆成可诊断条目或条件化知识。当前Doyle同历史去掉H2添加剂偏好，第2/3/4轮分别由68.74/62.00/23.04%改为81.28/81.28/75.12%，而只去掉Base条目并没有同样收益。该结果支持重点检查H2，但删除哪个条目不能在线使用未查询真值决定。','',
        '第三优先：恢复后仍持续审计冻结预测，避免一次改善让整块区域长期恢复满幅；结合不同位置的证据和系数不确定性。它主要提升可靠性，不能单独承诺解决Doyle前4轮造成的路径损失。','',
        '因此建议下一轮只做一个完整轨迹增量：预算内知识反例探索。对照继续保持原样，先验证它能否在Doyle更早测试新的添加剂，同时保住Perera的区域收益。本轮没有执行该新实验。','',
        '![剩余空间诊断](headroom.png)','',
        '计算与来源：[Perera逐轮局部诊断](Perera_Suzuki.json)、[Doyle逐轮局部诊断](Doyle_Buchwald_Hartwig.json)、[汇总与覆盖](facts.json)、[原结果保护与零新增调用凭据](receipt.json)。analyze.py与report.py可复现。',
        'Perera数值为UV产物面积百分比，Doyle为报告反应产率；所有真值替代和覆盖度均仅用于事后审核。单种子、基于已知轨迹提出的改法不是独立泛化证据。']
    (OUT/'report.md').write_text('\n'.join(text)+'\n',encoding='utf8')


def main():
    r={t:read(OUT/(t+'.json')) for t in TASKS};f=facts(r);save(OUT/'facts.json',f)
    plot(r,f);report(r,f);preserved()
    receipt=read(OUT/'receipt.json');receipt['artifacts_sha256']={p.name:hashed(p) for p in OUT.iterdir()
        if p.is_file() and p.name!='receipt.json'};save(OUT/'receipt.json',receipt)
    print(json.dumps({'status':'PASS','Perera_headroom':f['Perera_headroom'],
        'recovery_episodes':f['recovery_episodes'],'new_GP_fits':0,'new_API_calls':0,'new_experiments':0},indent=2))


if __name__=='__main__':main()
