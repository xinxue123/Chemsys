"""Analyze all three sealed followup tie tests; no new oracle reads."""
from pathlib import Path
import sys
import numpy as np
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from chem.benchmarks import reasoning_bo_data as data


def main():
    out=Path(__file__).resolve().parent;group=data.read(out/'followup_stability.json');allstates={};rows=[];probes=[]
    for seed in (0,1,2):
        variant=f'followup_t{seed}';root=Path('data/runs/stable_r2_20261029')/variant
        for p in (root/'runs').glob('*/state.json'):allstates[seed,data.read(p)['task']]=(data.read(p),p)
    successful=allstates[0,'Doyle_Buchwald_Hartwig'][0]['decisions'][5]['selected_id']
    pool=data.load_pool('Doyle_Buchwald_Hartwig');ix={k:i for i,k in enumerate(pool['ids'])}
    for seed in (0,1,2):
        s,path=allstates[seed,'Doyle_Buchwald_Hartwig'];r={'tie_seed':seed,'q3':{},'controls':[]}
        e=s['decisions'][2];r['q3']={'id':e['selected_id'],'actual_percent':e['y']*100,'guided_ties':e['tie_order']['guided_max_ties']}
        for t in (4,6):
            e=s['decisions'][t-1];commit=data.read(path.parent/'rounds'/f'{t:02d}_committed.json');prep=commit['validation']['prepared']
            options=[c['candidate_id'] for c in prep['candidates']]
            with np.load(path.parent/'rounds'/f'{t:02d}_candidates.npz') as z:
                j=ix[successful];maximum=float(z['base_qLogEI'][z['available']].max())
                r['controls'].append({'query':t,'selected':e['selected_id'],'actual_percent':e['y']*100,'pool_size':len(options),
                    'previous_s0_q6_observed_high_point_in_pool':successful in options,
                    'same_high_point_GP_mean_percent':float(z['mu_y'][j])*100,'same_high_point_GP_SD_pp':float(z['sd_y'][j])*100,
                    'same_high_point_base_loss':maximum-float(z['base_qLogEI'][j]),'normal_budget':e['budget']})
        rows.append(r)
        perera=allstates[seed,'Perera_Suzuki'][0]
        probes.append({'tie_seed':seed,'probes':[{'query':e['iteration'],'selected':e['selected_id'],'actual_percent':e['y']*100,
            'expected_percent':e['recombination']['validated']['plan']['prediction'],
            'gate_exception':e['acquisition_gate_exception'],'base_loss':e['base_opportunity_loss'],'normal_budget':e['budget']}
            for e in perera['decisions'][:3]]})
    payload={'stability':group,'Doyle_first_divergence':rows,'Perera_three_probe_sequences':probes,
        'comparison_point':{'source':'own s0 q6 actual observation; used only for retrospective analysis, never in a chooser','id':successful},
        'no_new_oracle_reads':True}
    data.write(out/'followup_diagnosis.json',payload)
    text=['**局部跟进版：Perera有限检验通过，整体稳定性失败。**','',
        '所有6条完整轨迹、96次选择及KF回放已核验封存，不能选择其中seed0作为稳定策略。优化seed仍0，变化的是独立并列种子。','',
        '| 数据集 | 并列seed | SearchAUC | 最佳/% | 首达自身最佳实验序号 |','|---|---:|---:|---:|---:|']
    for r in sorted(group['metrics'],key=lambda r:(r['task'],r['tie_seed'])):
        text.append(f"| {r['task']} | {r['tie_seed']} | {r['SearchAUC']:.6f} | {r['final_best_percent']:.6f} | {r['first_final_best_experiment']} |")
    text += ['','Perera三次都在追加第3轮、总实验第7次达到99.05%。首轮联合组合不同，但第二轮收敛到另一新配体组合81.70%，再保留两类别只换底物的局部跟进得到99.05%。seed1/2第二轮使用了明确记录的机会损失门限例外；因此不能声称全部探索都在原门限内。','',
        'Doyle第3轮初次分叉，三个引导分完全相同的候选分别改变配体、底物或添加剂，实测61.9973／68.7407／23.0413%。这不是GP能确定谁更好的情况。不同反馈改变了后续GP和事件输入。','',
        '| 并列seed | q3实测/% | q4池数／实测 | q6池数／实测 | s0已观测高产点是否仍在q6池中 |','|---:|---:|---|---|---|']
    for r in rows:
        a,b=r['controls'];text.append(f"| {r['tie_seed']} | {r['q3']['actual_percent']:.4f} | {a['pool_size']}／{a['actual_percent']:.4f} | {b['pool_size']}／{b['actual_percent']:.4f} | {b['previous_s0_q6_observed_high_point_in_pool']} |")
    text += ['','q4三次都是完整18个反例，Agent仍可选不同结构。q6的seed1/2则被GP前5名短名单限制，排除了另一条自身对照轨迹已观测的高产类别。这里只用先前实测作事后机制比较，没有给新策略提供其ID或真实值。','',
        '因此下一项先保留Perera工作路径；健康路径用完整反例评分、完整最高GP同分组审查、按自身实际前缀缓存第一次评分函数，并对事件摘要保留完整同分信息。不能用相同公共输入的反复LLM抽样混淆GP并列敏感性。该设计也不代表已排除LLM自身随机性或所有知识错误。','',
        '机器表：[followup_diagnosis.json](followup_diagnosis.json)；冻结验收：[followup_stability.md](followup_stability.md)；新方案：[STAGE9.md](STAGE9.md)。']
    (out/'followup_diagnosis.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    print({'status':group['status'],'Doyle_AUC_range':group['checks']['Doyle_Buchwald_Hartwig']['AUC_range'],'six_trajectories_checked':True})


if __name__=='__main__':main()
