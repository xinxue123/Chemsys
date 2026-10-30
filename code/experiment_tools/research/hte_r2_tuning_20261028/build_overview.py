"""Consolidate sealed development trajectories, including failed variants."""
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from chem.benchmarks import r2_tuning_study_v3 as study

OUT = PROJECT / study.OUT
VARIANTS = ['countercheck_v1', 'adaptive_countercheck_v2', 'focused_countercheck_v3']
LABELS = {
    'Pure_BO': '纯BO', 'ReasoningBO_point': 'Reasoning BO',
    'R2_region_trust': 'R2 同区域信任修正版',
    'countercheck_v1': 'V1：固定反例验证',
    'adaptive_countercheck_v2': 'V2：自适应验证与类别覆盖',
    'focused_countercheck_v3': 'V3：覆盖＋短名单',
}


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def main():
    assert Path.cwd().resolve() == PROJECT
    preserved = study.preservation()
    results = {v: read(OUT / v / 'results.json') for v in VARIANTS}
    for v in VARIANTS:
        release = read(OUT / v / 'release_receipt.json')
        assert release['status'] == 'PASS_COMPLETE'
        assert study.data.filehash(OUT / v / 'results.json') == release['results_sha256']
    original = read(PROJECT / study.trust.parent.OUT / 'results.json')['metrics']
    corrected = [r for r in read(PROJECT / study.trust.OUT / 'results.json')['metrics']
                 if r['method'] == 'R2_region_trust']
    rows = original + corrected + [row for v in VARIANTS for row in results[v]['metrics']]
    final = results[VARIANTS[-1]]
    met = final['formal_target_met']
    root = PROJECT / study.ROOT
    def commit(v):
        return read(root / v / 'runs' / ('Doyle_Buchwald_Hartwig_s0_' + v) /
                    'rounds' / '04_committed.json')
    c2, c3 = commit(VARIANTS[1]), commit(VARIANTS[2])
    assert c2['history_sha256'] == c3['history_sha256']
    full, shortlist = c2['validation']['prepared']['candidates'], c3['validation']['prepared']['candidates']
    tie_audit = {
        'identical_prequery_history': True,
        'full_focused_pool_size': len(full),
        'unique_base_qLogEI_values': len({c['base_qLogEI'] for c in full}),
        'shortlist_ids': [c['candidate_id'] for c in shortlist],
        'GP_mean_percent': shortlist[0]['GP_mean_percent'],
        'GP_latent_SD_pp': shortlist[0]['GP_latent_SD_pp'],
        'qLogEI': shortlist[0]['base_qLogEI'],
        'sorting': 'Descending qLogEI, then ascending candidate ID; all18 scores tie here.',
        'warning': 'This observation does not isolate GP ranking benefit; shortlist enumeration sensitivity is unresolved.',
    }
    overview = {
        'status': 'FIRST_BATCH_COMPLETE', 'formal_development_target_met': met,
        'seed': 0, 'initial_queries': 4, 'additional_queries': 16,
        'scope': 'Development on already inspected benchmarks; no independent validation or statistical dominance claim.',
        'all_metrics': rows, 'variant_results': results,
        'tie_audit': tie_audit, 'protected_releases': preserved,
        'source_result_sha256': {v: study.data.filehash(OUT / v / 'results.json') for v in VARIANTS},
    }
    study.data.write(OUT / 'overview.json', overview)
    lines = [
        '**Perera Suzuki 与 Doyle Buchwald–Hartwig：R2 首批调优结果**', '',
        f"预先冻结的正式开发数值目标：**{'达到' if met else '未达到'}**。三项策略各完成两条真实 Agent 轨迹，未增加每条轨迹的实验预算。", '',
        '所有版本使用完整实测候选池（3696／3955）、Mixed GP、相同初始4点和初始知识、seed=0、16次追加查询。对照结果直接读取原封存文件；没有重新运行或改写。Perera数值是UV产物面积百分比，Doyle是报告反应产率。', '',
        'SearchAUC定义为16次追加查询后的累计最佳值均值；首达序号包含初始4次。原八种方案和新增版本的全部指标保存在overview.json。', '',
    ]
    methods = list(LABELS)
    for task in study.data.SPECS:
        lines += [f'**{task}**', '',
                  '| 方法 | SearchAUC | 最终最佳/% | 首达最佳实验序号 |',
                  '|---|---:|---:|---:|']
        for method in methods:
            row = next(r for r in rows if r['task'] == task and r['method'] == method)
            lines.append(f"| {LABELS[method]} | {row['SearchAUC']:.4f} | {row['final_best_percent']:.4f} | {row['first_final_best_experiment']} |")
        lines.append('')
    lines += [
        '**策略变化及证据**', '',
        'V1在追加第2、4轮用控制变量反例替换正常R2查询。Perera初始化已达95.57%，但仍消耗两次验证，后续没有进步；Doyle的一次配体反例也造成损失。这一负面结果完整保留。', '',
        'V2增加统一的95%观测最佳值门槛，达到后跳过反例验证；未达到时，后续反例优先覆盖未见类别最多的知识维度。它保留了Perera原修正版的完整轨迹。Doyle第4轮验证Additive_14，产率78.253%，第12轮才找到98.2872%；类别覆盖有帮助，选到什么类别仍很关键。', '',
        'V3继续使用同一门槛和覆盖规则，把后续反例候选按基础qLogEI筛至5个，再交给Agent，并预留第6轮作为第三次机会。第一个验证仍保留原完整反例池；正常轮次仍使用原R2，包括原KF、B4和事件刷新。两个数据集没有名称分支，也没有硬编码高产候选。', '',
        'Doyle第2轮：固定配体、添加剂和芳基卤化物，把Base_02换成Base_03，77.6669%→81.2755%。第4轮：以该点为锚点，只把Additive_11换成Additive_01，81.2755%→98.2872%。Agent给出的事前理由是保留异噁唑骨架、改变取代基以检验H2的排除是否过强；其事前产率预测65–85%仍低估了实际值。', '',
        '第4轮的选择为p01032。它当时基础qLogEI总排名19，GP均值35.635%、潜在标准差27.904个百分点；处于原预算集内。第6轮因已达到95%门槛而跳过第三次验证。因此第6轮额外机会不是本次收益的来源。两次实际验证均未突破采集值机会损失门限。后续全部16次查询仍照常完成。', '',
        'Perera在初始化时已经达到95%门槛，所以V3没有执行反例验证，完整轨迹与原信任修正版相同：追加第7轮达到100%。这说明本次收益来自保留原路径，不能作为反例模块改善Perera的证据。', '',
        '**必须保留的解释边界**', '',
        f"Doyle第4轮完整的{len(full)}个新添加剂反例，其qLogEI完全相同（均为{shortlist[0]['base_qLogEI']:.9f}）。因此缩至5个实际依赖固定候选编号的并列裁决，不能声称GP识别出了更优的五个添加剂。编号只用于确定性裁决，没有当作离散变量的数值距离，也没有使用未查询真值排序；但枚举顺序敏感性仍未解决。", '',
        'V2到V3在第4轮前的真实实验历史完全相同。结果支持“缩小反例选择范围后，本次Agent挑到了更有收益的实验”这一轨迹事实，尚不能分别归因于化学知识、短名单大小或并列排序。KF、B4和事件刷新均未改公式；这轮实验也不构成它们各自带来增量的消融证据。', '',
        '95%门槛与上述规则是在已分析过的两个基准上逐步开发的。正式数值门槛达到也不代表独立泛化成立。Perera的SearchAUC仍低于Reasoning BO，且累计最佳图掩盖了找到高产点后仍出现低产查询；原始逐轮产率应同时查看。', '',
        '首批限定的三项研究已经结束，不自动扩大种子、实验预算或参数网格。下一阶段最有价值的验证是排除候选编号影响，并在相同额外LLM调用预算下比较反例选择与一般Agent选择；这些尚未执行。', '',
        '**运行、日志与核验**', '',
        '入口：pointwise_r2_tuning_v3.py。可执行 status、validate、report；run支持已完成状态和请求缓存，不会在完整轨迹之后追加查询。V3不是新的官方算法实现，而是原R2的受控知识反例研究分支。', '',
        '每轮查询前保存全部候选GP预测、基础和知识修正采集值、正常R2提案、实际提案及承诺哈希；随后才读取该点模拟真实结果，并保存KF、区域信任及控制变量比较。公开提示、回复和复用出处均保留，不含API密钥或模型隐藏思维内容。', '',
        '计算按两个数据集进程并行，每进程Torch1线程，小矩阵GP复用已观测历史缓存；算子一次预热，循环中编译次数为0。', '',
        '| 版本 | 完整轨迹 | 已核验选择 | 实际反例实验 | 门限例外 | 新API回复 | 复用请求 |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for v in VARIANTS:
        receipt = read(OUT / v / 'validation_receipt.json')
        api = results[v]['API']
        lines.append(f"| {v} | 2 | {receipt['choices']} | {receipt['controlled_experiments']} | {receipt['declared_acquisition_gate_exceptions']} | {api['new_responses']} | {api['reused_requests']} |")
    lines += ['', 'V3曲线：', '', '![逐轮产率与累计最佳](focused_countercheck_v3/comparison.png)', '',
              '可复核文件：[冻结任务](TASK.md)、[统一机器结果](overview.json)、[V3报告](focused_countercheck_v3/report.md)、[V3回放验证](focused_countercheck_v3/validation_receipt.json)、[V3封存回执](focused_countercheck_v3/release_receipt.json)。', '',
              'V1、V2的独立报告、协议、请求和失败轨迹一并保留。所有原对照、信任修正版与前序研究版本的文件哈希均已核对；不把验证程序通过等同于研究效果提高。']
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print(json.dumps({'status': overview['status'], 'formal_development_target_met': met,
                      'report': str(OUT / 'report.md')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
