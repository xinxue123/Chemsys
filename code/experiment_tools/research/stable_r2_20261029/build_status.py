"""Live overview; sealed negative evidence and frozen acceptance remain authoritative."""
from pathlib import Path
import json
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from chem.benchmarks import reasoning_bo_data as data
from chem.benchmarks import stable_feedback_study as study

LABELS={'robust_fit_t0':'仅稳健拟合','robust_anchor_t0':'锚点同分','robust_agent_t0':'完整同分类与高产重组',
    'additive_t0':'加性类别核','tie_agent_t0':'完整同分类Agent仲裁','model_challenge_t0':'联合模型质疑',
    'structure_t0':'结构相似性核','productive_t0':'期望收益反例','diverse_t0':'对照与结构多样性',
    'followup_t0':'新组合局部跟进','card_t0':'完整反例评分与同分审查','complete_t0':'可辨识性同分与完整反例','feedback_t0':'规范化原始建议与反馈延续'}


def main():
    out=Path(__file__).resolve().parent;results=[];pending=[];partial=[]
    variants=list(LABELS)+sorted(p.name for p in out.iterdir() if p.is_dir() and (p/'protocol.json').exists() and p.name not in LABELS)
    for v in variants:
        folder=out/v
        if not (folder/'release_receipt.json').exists():
            if (folder/'partial_receipt.json').exists():
                partial.append(data.read(folder/'partial_receipt.json'));continue
            if (folder/'protocol.json').exists():pending.append({'variant':v,'states':[{'task':s['task'],'status':s['status'],'queries':len(s['decisions']),'error':s.get('error')} for s in study.states(v)]})
            continue
        release=data.read(folder/'release_receipt.json');result=data.read(folder/'results.json')
        assert release['status']=='PASS_COMPLETE' and data.filehash(folder/'results.json')==release['results_sha256']
        assert data.read(folder/'validation_receipt.json')['choices']==32
        results.append(result)
    protected=study.preservation();count=2*len(results);queries=32*len(results)
    groups=[data.read(p) for p in out.glob('*_stability.json')]
    accepted=[g for g in groups if g['status']=='STABILITY_PASS']
    payload={'goal_status':'LIMITED_ACCEPTANCE_VERIFIED' if accepted else 'ACTIVE_NOT_ACHIEVED',
        'variants':results,'archived_partial_versions':partial,'new_complete_trajectories':count,'new_queries':queries,'in_progress':pending,
        'stability_groups':groups,'accepted_groups':[g['group'] for g in accepted],
        'prescribed_stability_tie_seeds':[0,1,2],'optimizer_seed':0,'per_trajectory_budget':'4+16',
        'acceptance':{'Perera':{'minimum_AUC':98.,'minimum_final_best':99.},'Doyle':{'minimum_AUC':91.4567435325,'minimum_final_best':98.28722602},'maximum_AUC_range_each_task':1.},
        'protected_releases':protected,'scope':'Development on known seed0 benchmarks; repeated traces and equivalent cached Agent replies are not independent validation.'}
    data.write(out/'CURRENT_STATUS.json',payload)
    text=['**稳定策略目标：'+('已通过规定的有限验收。' if accepted else '进行中，尚未达成。')+'**','',
        '冻结验收：同一策略的并列seed0/1/2全部通过；Perera每次AUC≥98、最佳≥99%；Doyle每次AUC≥91.4567435325、最佳≥98.28722602%；各数据集三次AUC极差≤1。优化seed0、原共同初始化、完整候选池、4＋16预算不变。','',
        f'已封存{len(results)}个版本、{count}条完整离线轨迹，共{queries}次追加查询。每版本32次选择和KF回放全部核验；负面结果全部保留。同前缀和同公开上下文的真实Agent回复可复用，重复轨迹或复用回复不算独立验证。','',
        '| 策略/并列版本 | Perera AUC | Perera最佳/% | Doyle AUC | Doyle最佳/% | 两套单次达标 |','|---|---:|---:|---:|---:|---|']
    for r in results:
        v=r['variant'];p=next(x for x in r['metrics'] if x['task']=='Perera_Suzuki');d=next(x for x in r['metrics'] if x['task']=='Doyle_Buchwald_Hartwig')
        text.append(f"| [{LABELS.get(v,v)}]({v}/report.md) | {p['SearchAUC']:.4f} | {p['final_best_percent']:.4f} | {d['SearchAUC']:.4f} | {d['final_best_percent']:.4f} | {r['formal_target_met']} |")
    text += ['','仅默认核重启、并列排序或结构相似性都不足以让Perera达标。结构核消除了初始最高分并列，仍持续搜索同类。LLM逐点评分也出现91%期望、12.01%实测的严重错误。新方案仍须以实际轨迹证明增量，不能以预测区分度或回放通过代替。','',
        '反例候选规则先后暴露两种限制：把混杂观测次数当作类别已被辨识；把旧区域最佳背景直接移植给新类别组合。第八阶段局部跟进使Perera三次均AUC98.615、最佳99.05%，但Doyle极差5.5268且两次未达标。第九阶段针对健康路径的完整反例池和最高GP同分组做化学评分审查；同输入首次评分卡缓存以固定评分函数，不能解释为独立LLM随机性验证。第九阶段完整收益评分在Doyle只得到81.8351／92.2560%，已否决。第十阶段恢复原单点控制建议，仅在GP完全同分时使用新单轴对照信息代理；不让未校准的LLM收益分接管排序。该版扩展检验又有Perera真实失败及Doyle缓存tuple/list类型断言；第十一阶段规范化原始建议缓存，并在新类别自身改善≥5个百分点时继续不同背景检验。所有阈值和预算在新轨迹前冻结，不按oracle或旧高产ID选择。','']
    for item in pending:
        text.append(f"尚未封存：{item['variant']}。")
        for s in item['states']:text.append(f"{s['task']}：{s['status']}，{s['queries']}/16次查询。")
        text.append('')
    for item in partial:
        text += [f"部分封存：{item['variant']}，Doyle5/16技术中断；Perera16/16完整结果见该目录interruption.json，不计入完整配对轨迹验收。",'']
    for g in groups:text += [f"稳定性组[{g['group']}]({g['group']}_stability.md)：{g['status']}。",'']
    text += ['所有原对照、里程碑和负面结果保持哈希。数据集两进程并行、每进程Torch1线程、算子一次预热、循环编译0次。单条COMPLETE仅代表16次查询结束；不能自动替换生产默认。','',
        '任务：[TASK.md](TASK.md)；最近方案：[STAGE11.md](STAGE11.md)；结构核逐轮诊断：[structure_diagnosis.md](structure_diagnosis.md)。',
        '当前执行入口stable_feedback_r2.py；动作prepare/run/status/validate/report/seal，均用--variant显式选择版本。']
    (out/'CURRENT_STATUS.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    print(json.dumps({'goal_status':payload['goal_status'],'sealed_versions':len(results),'complete_trajectories':count,'queries':queries,'in_progress':pending},ensure_ascii=False))


if __name__=='__main__':main()
