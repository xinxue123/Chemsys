"""Render the completed, audited V13 results without changing any experiment."""
from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'experiments/reports/v13_state_split_experiment_20261102'
NAMES = dict(c1='Suzuki Case 1', c2='Suzuki Case 2', c3='Suzuki Case 3',
             c4='Suzuki Case 4', perera='Perera Suzuki',
             doyle='Doyle Buchwald-Hartwig', snar='Summit SnAr', hplc='HPLC')


def main():
    audit = json.loads((OUT / 'audit.json').read_text(encoding='utf-8'))
    assert audit['status'] == 'PASS_COMPLETE' and audit['queries_replayed'] == 128
    with (OUT / 'comparison.csv').open(encoding='utf-8-sig', newline='') as f:
        comparisons = list(csv.DictReader(f))
    with (OUT / 'metrics.csv').open(encoding='utf-8-sig', newline='') as f:
        metrics = {(r['task'], r['method']): r for r in csv.DictReader(f)}
    assert len(comparisons) == 8
    wins = {}
    for arm in ('bo', 'reasoning_bo', 'v12_subagent'):
        delta = [float(r['V13_SearchAUC']) - float(r[arm + '_SearchAUC']) for r in comparisons]
        wins[arm] = (sum(x > 1e-8 for x in delta), sum(abs(x) <= 1e-8 for x in delta),
                     sum(x < -1e-8 for x in delta))
    lines = [
        'V13：预测可信度与推荐权限分离——完整实验结果（2026-11-02）', '',
        '8个任务全部完成，128次新查询通过逐轮重放检查。',
        '相对纯BO：%d胜、%d平、%d负；相对Reasoning BO：%d胜、%d平、%d负；相对V12子智能体：%d胜、%d平、%d负。'
        % (*wins['bo'], *wins['reasoning_bo'], *wins['v12_subagent']), '',
        '实验设置',
        '保持原候选池、各任务原GP/采集函数/并列顺序/五候选菜单、seed=0、相同4次初始化与16次新查询。'
        '知识不修改GP均值或方差，只在GP给定菜单中参与选择。本轮只运行V13，历史对照均未修改。',
        '原API接口额度耗尽；按此前授权，采用逐请求新建的gpt-6.1-terra medium子智能体。'
        '每次只提供本轨迹已测观测及公共决策上下文，不提供未查询目标、其他方法结果或GP数值。'
        '模型名称是编排请求记录，不是服务端版本证明。',
        'Reasoning BO采用冻结的历史API结果；V12主对照采用冻结子智能体结果。Case 2、Case 4、HPLC的V12 API结果也保存在metrics.csv。'
        'API和子智能体不是同一实验批次，V13与V12的新知识响应也未配对固定。', '',
        '指标',
        'M0=max(4次初始化的实测目标)，Mt=max(M(t-1),第t次新查询目标)。SearchAUC=(M1+...+M16)/16。'
        '本轮每条轨迹都完成16次新查询，因此分母为16。未改善、低值或零值查询均计入预算，不删除。'
        'SnAr/HPLC采用各自原目标量纲，不与产率直接求平均。', '',
        '任务 | V13 SearchAUC | 纯BO | Reasoning BO(API) | V12(子智能体) | V13相对纯BO',
    ]
    for r in comparisons:
        lines.append(f"{NAMES[r['task']]} | {float(r['V13_SearchAUC']):.4f} | "
                     f"{float(r['bo_SearchAUC']):.4f} | {float(r['reasoning_bo_SearchAUC']):.4f} | "
                     f"{float(r['v12_subagent_SearchAUC']):.4f} | {float(r['delta_vs_bo_pct']):+.2f}%")
    lines += ['', '任务 | V13最终最佳目标 | 首次达到该值的新查询轮次（0表示初始化）']
    for r in comparisons:
        m = metrics[(r['task'], 'V13')]
        lines.append(f"{NAMES[r['task']]} | {float(m['final_best']):.5f} | {m['first_final_best_query']}")
    changes = audit['calibration_changes']
    counts = audit['trajectory_counts']
    lines += [
        '', '机制实际执行情况',
        f"共{counts['forecast_count']}条知识预测收到反馈；KF可信度上调{changes['increased']}次、下调{changes['decreased']}次、"
        f"不变{changes['unchanged']}次。这里统计知识条目更新次数，不等同于查询次数。",
        f"出现{counts['accurate_failure_events']}次准确失败预测，失败反馈导致的正向权限恢复为"
        f"{audit['failure_based_positive_restorations']}次。共发生{counts['new_suppressions']}次进入suppressed、"
        f"{counts['newly_allowed']}次首次进入allowed。可信度与推荐权限分别存储、更新和重放核对。",
        '两次复核测试发生在Perera第8轮和Doyle第15轮，均未改善当时incumbent，因此未恢复正向推荐。'
        '本轮没有发生成功恢复，不构成恢复成功路径的实验验证。',
        '知识使用记录：' + json.dumps(audit['knowledge_use_counts'], ensure_ascii=False) + '。',
        '知识编辑记录：' + json.dumps(audit['memory_edit_counts'], ensure_ascii=False) + '。'
        '8个任务各调用4次知识编辑，包括初始调用；本轮实际动作只有新增和保留，没有修订或撤回。',
        f"{counts['changed_from_GP']}/128次选择不同于同一轮GP首选；这只是轨迹差异，不能归因于KF或某条知识的独立因果收益。",
        '', '执行检查',
        f"逐轮重放{audit['queries_replayed']}次，通过选点、实验前预测、反馈、知识状态及最终状态一致性检查。"
        '候选输入、冻结对照与注册科学代码哈希未变化，8个任务首次GP结果与各自纯BO完全一致。',
        f"保存{audit['unique_fresh_agents']}个独立子智能体响应：128次决策、32次知识编辑及5次额外格式尝试。"
        '5次格式拒绝分布于4轮；修复均在查询真实目标之前完成，并保留首次合法选点。没有根据真实结果重选。',
        '采用3个独立单线程CPU工作进程，复用数值缓存；循环内重复算子编译为0。'
        '逐轮审计不调用实验oracle，不产生额外实验查询。',
        '', '结论与边界',
        '本轮多数任务高于纯BO，Doyle、HPLC和Case 4的收益较明显；Case 2仍比纯BO低15.17%，是主要损失。'
        'Case 3仅提升0.40%，不应表述为稳定优势。Perera没有超过初始化最好值。',
        'SnAr与冻结V12子智能体16轮选点完全一致；与Reasoning BO有4轮不同，但累计最佳轨迹一致。'
        '因此SnAr的相对纯BO收益不能作为本次状态分离新增收益的证据。',
        '这次确认了准确失败预测不会自动解封正向推荐，并完成8条完整轨迹；'
        '单seed、每任务一次LLM响应实现不能证明稳定性，也不能分离状态修正与新LLM响应的因果效应。'
        '自由文本推理仍可能使用未显式引用的概念，审计能约束声明的知识使用，不能完全观测隐藏语义依赖。',
        '', '证据文件',
        'metrics.csv：所有对照的SearchAUC、最终最佳、首次达到最佳轮次及后端批次。',
        'comparison.csv：逐任务SearchAUC差值百分比。',
        'rounds.csv：128轮选点、目标值、GP首选、知识用途、状态及刷新摘要。',
        'audit.json：完整重放与机制计数。',
        '原始实验：experiments/runs/v13_state_split_20261102；入口：code/experiment_tools/run_state_split_v13.py。',
    ]
    target = OUT / 'report_zh.txt'
    target.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(str(target))


if __name__ == '__main__':
    main()
