"""Read-only interpretation of completed repair trajectories, no new queries."""
from pathlib import Path
import json
import statistics

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]
ROOT = BASE / 'data/runs/suzuki_grounded_regional'
data = json.loads((OUT / 'results.json').read_text(encoding='utf8'))
states = [json.loads(p.read_text(encoding='utf8')) for p in (ROOT / 'runs').glob('*/state.json')]
by = {(s['case'], s['regime'], s['method']): s for s in states}
raw = {(s['case'], s['regime'], s['method']): s for s in data['raw_metrics']}
regimes = ('correct', 'wrong_category', 'mixed', 'assistant_initial')
methods = ('R1_grounded', 'R2_conflict_fallback', 'R3_conservative_region')


def paired(method, reference):
    rows = [{'case': c, 'regime': r, 'delta': raw[c, r, method]['SearchAUC'] - raw[c, r, reference]['SearchAUC']}
            for c in (2, 3, 4) for r in regimes]
    return {'mean_delta': statistics.mean(v['delta'] for v in rows),
        'wins': sum(v['delta'] > 1e-10 for v in rows), 'ties': sum(abs(v['delta']) <= 1e-10 for v in rows),
        'losses': sum(v['delta'] < -1e-10 for v in rows),
        'worst': min(rows, key=lambda v: v['delta']), 'best': max(rows, key=lambda v: v['delta'])}


def split(case, left, right):
    a, b = by[case, 'assistant_initial', left], by[case, 'assistant_initial', right]
    for i, (da, db) in enumerate(zip(a['decisions'], b['decisions'])):
        if da['selected_id'] == db['selected_id']:
            continue
        ha, hb = a['history'][:4+i], b['history'][:4+i]
        ya, yb = a['history'][4+i]['y'], b['history'][4+i]['y']
        return {'query': i+1, 'same_observed_prefix': ha == hb,
            'same_knowledge_geometry': da['knowledge'] == db['knowledge'],
            'same_field_key': da['field_acquisition']['key'] == db['field_acquisition']['key'],
            'same_bank_forecast': da['bank_prior'] == db['bank_prior'],
            'left': {'method': left, 'id': da['selected_id'], 'yield_percent': ya*100,
                'amplitude': da['amplitude'], 'fallback': da['deployment']['neutral_fallback']},
            'right': {'method': right, 'id': db['selected_id'], 'yield_percent': yb*100,
                'amplitude': db['amplitude'], 'fallback': db['deployment']['neutral_fallback']},
            'incumbent_difference_pp': (max(max(v['y'] for v in ha), ya)-max(max(v['y'] for v in hb), yb))*100}
    return None


def curve(state):
    y = [h['y']*100 for h in state['history']]
    return [max(y[:i]) for i in range(5, 21)]


case4_LL = json.loads((BASE / 'data/runs/suzuki_mixed_comparison/runs/c4_s0_assistant_initial_LLINBO_matched/state.json').read_text(encoding='utf8'))
case4_R2 = by[4, 'assistant_initial', methods[1]]
gap = [a-b for a, b in zip(curve(case4_LL), curve(case4_R2))]
summary = {'primary_pairs': {m: {ref: paired(m, ref) for ref in ('G3_original', 'Gate_only')} for m in methods},
    'first_split_case4_R2_vs_R1': split(4, methods[1], methods[0]),
    'first_split_case3_R3_vs_R2': split(3, methods[2], methods[1]),
    'first_split_case4_R3_vs_R2': split(4, methods[2], methods[1]),
    'Case4_LL_minus_R2_AUC_gap': sum(gap)/16,
    'Case4_gap_contributions': {name: sum(gap[lo:hi])/16 for name, lo, hi in (('query1_5', 0, 5), ('query6_10', 5, 10), ('query11_16', 10, 16))},
    'Case3_R3_base_argmax_rounds': [d['iteration'] for d in by[3, 'assistant_initial', methods[2]]['decisions'] if d['selected_id'] == d['reference_id']],
    'worst_R3_minus_R2': min(({'case': c, 'regime': r,
        'delta': raw[c, r, methods[2]]['SearchAUC']-raw[c, r, methods[1]]['SearchAUC'],
        'R3_first_best_query': raw[c, r, methods[2]]['first_own_best_added'],
        'R2_first_best_query': raw[c, r, methods[1]]['first_own_best_added']} for c in (2,3,4) for r in regimes), key=lambda v:v['delta']),
    'new_GP_fits': 0, 'new_LLM_calls': 0,
    'scope': 'Observed full trajectories and actual first divergences only; not a factorial or all-round causal decomposition.'}
(OUT / 'analysis.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf8')

lines = ['# 修改后是否带来增量', '',
    '48/48 条完整新轨迹，Case 1–4 × 四种知识条件 × 三档修复；所有档位保持 Mixed GP、候选集、seed=0、4＋16 预算。原对照没有改写。', '',
    '**结论：R2 冲突回退取得小幅累计收益，但尚未超过仅预算门限；R3 的一标准差折扣没有通过整体增量检验。** 输入字段与证据校验提供了可审计的数据合同，不能据此宣称化学知识更正确或搜索性能必然提高。', '',
    '## 主测试 Case 2–4，四条件平均', '',
    '| 方法 | SearchAUC | 对原 G3 | 对仅预算门限 |', '|---|---:|---:|---:|']
average = data['summary']['primary']['all_conditions']
for method in ('G3_original', *methods, 'Gate_only', 'LGBO', 'LLINBO_Justify', 'Pure_BO'):
    value = average[method]['SearchAUC_mean']
    lines.append(f"| {method} | {value:.4f} | {value-average['G3_original']['SearchAUC_mean']:+.4f} | {value-average['Gate_only']['SearchAUC_mean']:+.4f} |")
lines += ['', '这些是同三种反应在四种初始知识下的描述性结果，不是 12 个独立反应。三个人工知识条件包含参考结果信息，四条件均值也不是实际部署条件的发生概率加权收益。', '',
    '## 助手初始知识：逐 Case 对照', '',
    '| Case | 原 G3 | R1 | R2 | R3 | 纯 BO | LLINBO-Justify |', '|---|---:|---:|---:|---:|---:|---:|']
for case in (1,2,3,4):
    order = ('G3_original', *methods, 'Pure_BO', 'LLINBO_Justify')
    lines.append('| '+str(case)+' | '+' | '.join(f"{raw[case,'assistant_initial',m]['SearchAUC']:.4f}" for m in order)+' |')
lines += ['', 'Case 1 是参考案例；首次达到各自最终最佳的轮次和最终产率见 `report.md` / `metrics.csv`。', '',
    '## 完整轨迹中的收益与损失', '']
for method in methods:
    p = summary['primary_pairs'][method]['G3_original']
    lines.append(f"- {method} 对原 G3：12 个 Case×条件中 {p['wins']} 胜、{p['ties']} 平、{p['losses']} 负；平均 {p['mean_delta']:+.4f} 个 AUC 百分点。最差为 Case {p['worst']['case']} / {p['worst']['regime']}：{p['worst']['delta']:+.4f}。")
lines += ['', '## Case 3：输入修复没有消除早期路径损失', '',
    'R3 与 R2 首次在新增第 4 次查询分叉：', '', '```json', json.dumps(summary['first_split_case3_R3_vs_R2'], ensure_ascii=False, indent=2), '```', '',
    '这一轮刷新后的知识几何也不同，不能将分叉单独归因于一标准差折扣。候选产率变化也不能等同于累计最佳的提升：两点均未超过当时的 incumbent。R3 最终最佳为 88.1865%，在新增第 15 次才取得；纯 BO 在新增第 12 次已取得 89.6450%。完整 SearchAUC 仍落后。R1 虽将最终最佳提高到 88.1865%，但直到第 16 次才取得，AUC 提升有限。', '',
    f"R3 在自身前缀上选择基础 qLogEI 参考点的轮次为 {summary['Case3_R3_base_argmax_rounds']}。这些轮次不能解释成知识直接强迫选择较差点；其 GP 训练轨迹已经不同。", '',
    '## Case 4：回退修复了一处关键早期损失，但没有超过 LLINBO', '',
    'R2 与 R1 第一次分叉的前缀、知识几何、区域场与 KF 预测均可在 JSON 中直接核对：', '',
    '```json', json.dumps(summary['first_split_case4_R2_vs_R1'], ensure_ascii=False, indent=2), '```', '',
    '新增第 2 次查询的产率从 85.0228% 变为 91.0978%，当轮累计最佳增加约 5.0638 个百分点。此处可以归因于冲突回退；R3 的一标准差折扣被冲突规则覆盖，不能把此收益归给折扣。', '',
    'R2 的 SearchAUC 从原 G3 的 91.8422 提高到 93.7384；最终达到 96.5158%，新增第 12 次取得，比 LLINBO 第 13 次早一轮。但 LLINBO 第 2 次就已取得 95.9214%，早期平台更高，因此 SearchAUC 仍以 95.4887 领先。', '',
    'LLINBO−R2 的 AUC 差距按实际 incumbent 曲线拆分：', '', '```json', json.dumps(summary['Case4_gap_contributions'], indent=2), '```', '',
    '## 新区保守化的反例', '',
    f"R3−R2 平均为 {average[methods[2]]['SearchAUC_mean']-average[methods[1]]['SearchAUC_mean']:+.4f}；最差反例：", '',
    '```json', json.dumps(summary['worst_R3_minus_R2'], ensure_ascii=False, indent=2), '```', '',
    '这里说明一标准差惩罚可能压低有用区域引导。完整轨迹分叉后，后续 GP、LLM 响应和事件时序都可能不同，不能把整段差值当作某一轮折扣的独立因果作用。', '',
    '## 当前取舍', '',
    '保留具名输入和事实校验作为数据合同；R2 可保留为改进候选，但当前主测试均值仍低于仅预算门限，尚不能替代这个基线。R3 不建议默认启用，当前证据不支持把一标准差扣减加入主线。没有追加种子或根据本轮结果修改参数。', '',
    '数据与实现完整性通过不代表研究已证明优于基线；本轮保持单 seed 的探索性结论。', '']
(OUT / 'analysis.md').write_text('\n'.join(lines), encoding='utf8')
print(json.dumps({'Case4_first_split': summary['first_split_case4_R2_vs_R1'],
    'Case3_first_split': summary['first_split_case3_R3_vs_R2'],
    'worst_R3_minus_R2': summary['worst_R3_minus_R2']}, ensure_ascii=False))
