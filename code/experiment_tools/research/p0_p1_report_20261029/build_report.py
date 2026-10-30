"""Read-only consolidation of sealed P0/P1 experiments; no fits or API calls."""
from collections import Counter
from copy import deepcopy
from decimal import Decimal, ROUND_HALF_UP
from hashlib import sha256
from pathlib import Path
import json
import math
import os
import statistics

PROJECT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
INPUTS = {}
CACHE = {}
STATES = {}
REGIMES = ('assistant_initial', 'correct', 'wrong_category', 'mixed')
TASKS = ('Perera_Suzuki', 'Doyle_Buchwald_Hartwig')
KNOWLEDGE = {'assistant_initial': '助手初始建议', 'correct': '正确知识（人工）',
             'wrong_category': '错误类别（人工）', 'mixed': '混合错误（人工）'}
P0_METHODS = (
    'Pure_BO', 'LGBO', 'Gate_only', 'B4_hard', 'B4_soft', 'LLINBO_Justify',
    'G3_original', 'R1_grounded', 'R2_conflict_fallback',
    'R3_conservative_region', 'Stable_revised')
P1_METHODS = (
    'Pure_BO', 'LGBO_region', 'Gate_only', 'B4_hard', 'B4_soft', 'LLINBO_Justify',
    'ReasoningBO_point', 'New_R2', 'R2_region_trust', 'feedback_t0')
NAMES = {
    'Pure_BO': '纯BO（qLogEI）', 'LGBO': '区域LGBO', 'LGBO_region': '区域LGBO',
    'Gate_only': '仅预算门限', 'B4_hard': '硬B4', 'B4_soft': '软B4＋区域注入',
    'LLINBO_Justify': 'LLINBO-Justify', 'G3_original': '旧G3（KF/B4/事件）',
    'R1_grounded': '旧R1（具名事实校验）',
    'R2_conflict_fallback': '旧R2（冲突回退）',
    'R3_conservative_region': '旧R3（新区域保守部署）',
    'Stable_revised': '**修订版Agent**',
    'ReasoningBO_point': 'Reasoning BO（逐点适配）',
    'New_R2': '旧R2（原逐点版）',
    'R2_region_trust': '旧R2（信任恢复修复）',
    'feedback_t0': '**修订版Agent（并列seed=0）**',
    'feedback_t1': '修订版Agent（并列seed=1）',
    'feedback_t2': '修订版Agent（并列seed=2）',
    'focused_countercheck_v3': 'V3（原编号并列）',
    'focused_countercheck_v3_shuffle_s0': 'V3（随机并列seed=0）',
}


def rel(path):
    p = Path(path)
    return p.relative_to(PROJECT).as_posix() if p.is_absolute() else p.as_posix()


def filehash(path):
    p = Path(path)
    return sha256((p if p.is_absolute() else PROJECT / p).read_bytes()).hexdigest()


def observe(path):
    p = rel(path)
    INPUTS[p] = filehash(p)
    return PROJECT / p


def read(path):
    key = rel(path)
    if key not in CACHE:
        CACHE[key] = json.loads(observe(key).read_text(encoding='utf-8'))
    return CACHE[key]


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8')


def link(path, label):
    p = observe(path)
    return f'[{label}]({p.as_posix()})'


def image(path, alt):
    p = observe(path)
    return f'![{alt}]({p.as_posix()})'


def metrics(path):
    state = read(path)
    assert state['status'] == 'COMPLETE', path
    history = state['history']
    assert len(history) == 20 and len(state['decisions']) == 16, path
    assert len({h['candidate_id'] for h in history}) == 20, path
    vals = [float(h['y']) * 100 for h in history]
    assert all(math.isfinite(v) and 0 <= v <= 100 for v in vals)
    best, running = [], -math.inf
    for v in vals:
        running = max(running, v)
        best.append(running)
    result = {'SearchAUC': math.fsum(best[4:]) / 16,
              'final_best_percent': max(vals),
              'first_final_best_experiment': vals.index(max(vals)) + 1,
              'initial_best_percent': max(vals[:4]),
              'mean_queried_percent': math.fsum(vals[4:]) / 16,
              'state_source': rel(path)}
    STATES[rel(path)] = state
    return result


def p0_path(case, regime, method):
    if method == 'Pure_BO':
        return f'data/runs/suzuki_mixed_kalman/base/c{case}_s0/state.json'
    if method == 'Stable_revised':
        return f'data/runs/suzuki_stable_20261029/runs/c{case}_s0_{regime}_stable/state.json'
    if method in ('R1_grounded', 'R2_conflict_fallback', 'R3_conservative_region'):
        return f'data/runs/suzuki_grounded_regional/runs/c{case}_s0_{regime}_{method}/state.json'
    if method == 'G3_original':
        return f'data/runs/suzuki_regional_ablation/runs/c{case}_s0_{regime}_G3_gate_KF_B4_event/state.json'
    name = {'LGBO': 'LGBO_region', 'Gate_only': 'LGBO_gate',
            'B4_hard': 'B4_base', 'B4_soft': 'B4_LGBO',
            'LLINBO_Justify': 'LLINBO_matched'}[method]
    return f'data/runs/suzuki_mixed_comparison/runs/c{case}_s0_{regime}_{name}/state.json'


def p1_path(task, method):
    if method.startswith('feedback_'):
        return f'data/runs/stable_r2_20261029/{method}/runs/{task}_s0_{method}/state.json'
    if method == 'R2_region_trust':
        return f'data/runs/hte_r2_region_trust_seed0_v2/runs/{task}_s0_R2_region_trust/state.json'
    if method.startswith('focused_countercheck'):
        return f'data/runs/hte_r2_tuning_20261028/{method}/runs/{task}_s0_{method}/state.json'
    return f'data/runs/hte_pointwise_seed0/runs/{task}_s0_{method}/state.json'


def check_metrics(actual, expected):
    for a, b in [('SearchAUC', 'SearchAUC'), ('final_best_percent', 'final_best_percent'),
                 ('first_final_best_experiment', 'first_final_best_experiment')]:
        assert abs(actual[a] - expected[b]) < 1e-10, (a, actual, expected)


def verify_inventory(folder, stem='release'):
    base = Path(folder)
    receipt = read(base / f'{stem}_receipt.json')
    inventory_path = base / f'{stem}_inventory.json'
    assert filehash(inventory_path) == receipt['inventory_sha256'], str(inventory_path)
    inventory = read(inventory_path)
    for row in inventory['files']:
        assert filehash(row['path']) == row['sha256'], row['path']
    return {'folder': base.as_posix(), 'stem': stem, 'files_verified': len(inventory['files']),
            'inventory_sha256': receipt['inventory_sha256']}


def extract():
    p0 = read('research/suzuki_stable_20261029/results.json')
    p1 = read('research/stable_r2_20261029/stable_strategy_results.json')
    original = read('research/hte_pointwise_seed0/results.json')
    trust = read('research/hte_r2_region_trust_seed0/results.json')
    a0, a1, dev = [], [], []
    for c in (1, 2, 3, 4):
        for regime in REGIMES:
            for method in P0_METHODS:
                path = p0_path(c, regime, method)
                m = metrics(path)
                exp = next(r for r in p0['comparison_rows'] if r['case'] == c
                           and r['regime'] == regime and r['method'] == method)
                check_metrics(m, {'SearchAUC': exp['SearchAUC'],
                                 'final_best_percent': exp['final_best'],
                                 'first_final_best_experiment': exp['first_own_best_total']})
                ref = read(p0_path(c, regime, 'Pure_BO'))['history'][:4]
                assert read(path)['history'][:4] == ref
                a0.append({'stage': 'P0', 'case': c, 'task': f'Reizman Suzuki Case {c}',
                           'knowledge_condition': regime, 'method': method, 'optimizer_seed': 0,
                           'tie_seed': 0 if method == 'Stable_revised' else None, **m})
    for task in TASKS:
        for method in (*P1_METHODS, 'feedback_t1', 'feedback_t2'):
            path = p1_path(task, method)
            m = metrics(path)
            if method.startswith('feedback_'):
                expected = next(r for r in p1['acceptance']['metrics'] if r['task'] == task and r['method'] == method)
            else:
                source = trust if method == 'R2_region_trust' else original
                expected = next(r for r in source['metrics'] if r['task'] == task and r['method'] == method)
            check_metrics(m, expected)
            assert read(path)['history'][:4] == read(p1_path(task, 'Pure_BO'))['history'][:4]
            a1.append({'stage': 'P1', 'task': task, 'knowledge_condition': 'shared_observed_only_initial',
                       'method': method, 'optimizer_seed': 0,
                       'tie_seed': int(method[-1]) if method.startswith('feedback_') else None, **m})
        for method in ('focused_countercheck_v3', 'focused_countercheck_v3_shuffle_s0'):
            m = metrics(p1_path(task, method))
            source = read(f'research/hte_r2_tuning_20261028/{method}/results.json')
            expected = next(r for r in source['metrics'] if r['task'] == task)
            check_metrics(m, expected)
            dev.append({'stage': 'P1_development', 'task': task, 'method': method, **m})
    assert len(a0) == 176 and len(a1) == 24 and len(dev) == 4
    # Check histories against frozen lookup tables, solely for retrospective reporting.
    import numpy as np
    for stage, tasks in [('P0', range(1, 5)), ('P1', TASKS)]:
        for task in tasks:
            root = (f'data/runs/suzuki_guarded/inputs/case{task}' if stage == 'P0'
                    else f'data/benchmarks/reasoning_bo/{task}')
            pool = read(root + '/pool.json')
            with np.load(observe(root + '/oracle.npz'), allow_pickle=False) as f:
                vals = f['values'].copy()
            index = {key: j for j, key in enumerate(pool['ids'])}
            rows = [r for r in a0 if r['case'] == task] if stage == 'P0' else [r for r in [*a1, *dev] if r['task'] == task]
            for row in rows:
                for h in read(row['state_source'])['history']:
                    assert float(h['y']) == float(vals[index[h['candidate_id']]])
    return {'P0': a0, 'P1': a1, 'P1_development': dev,
            'P0_aggregate_source': p0, 'P1_aggregate_source': p1}


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '|' + '|'.join(['---'] + ['---:'] * (len(headers) - 1)) + '|',
                      *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]]) + '\n'


def fmt(value, places=3, signed=False):
    # Remove sub-picopoint float round-off before ordinary decimal half-up display.
    # The machine-readable measurements and calculations retain full precision.
    d = Decimal(str(value)).quantize(Decimal('0.000000000001'), rounding=ROUND_HALF_UP)
    d = d.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    if d == 0:
        d = abs(d)
    return ('+' if signed and d >= 0 else '') + format(d, 'f')


def metric_table(rows, methods):
    by_method = {r['method']: r for r in rows}
    base = by_method['Pure_BO']['SearchAUC']
    return table(['方法', 'SearchAUC', '最佳值/%', '首次最佳/总次序', 'AUC较纯BO/百分点'], [
        [NAMES[m], fmt(by_method[m]['SearchAUC']),
         fmt(by_method[m]['final_best_percent']),
         by_method[m]['first_final_best_experiment'],
         fmt(by_method[m]['SearchAUC'] - base, signed=True)] for m in methods])


METHODS = r"""
## 一、Agent怎样完成一次实验决策

本报告中的Agent由LLM和数值决策程序共同组成。LLM提出可检验的化学偏好、修订知识，并在指定检验环节建议候选；数值程序拟合GP、计算采集值、检查约束、更新KF和执行选点。每次观察到新结果后，下一次选择都会使用更新后的自身经验。

优化始终发生在预先固定的候选集合中。P0的正常选择要给全部4692个候选计算GP预测和采集值；P1分别对3696、3955个实测组合评分。LLM并不在每轮给整个候选池逐点打分：只有诊断触发的小型联合/局部试验池需要其完整评分，普通反例环节则由LLM建议一个受控候选。LLM可以修改知识区域，执行点仍须属于原池。

### 1.1 决策与反馈顺序

| 步骤 | 输入与计算 | 对决策的作用 |
|---|---|---|
| 1. 读取自身经验 | 初始4点＋本轨迹此前实际查询结果，以及当前知识、区域校准状态库与信任记录 | 每条轨迹只使用自己的已观测前缀 |
| 2. 拟合或读取GP | Mixed GP给出全部候选的均值、潜在SD和含观测噪声SD；P1还检查类别核的辨识问题 | 形成预测、探索不确定性和拟合健康诊断 |
| 3. 计算基础采集值 | q=1、512路Sobol MC的qLogEI；已查询点排除 | 得到纯BO参考点及采集机会预算集合 |
| 4. 联合诊断知识 | B4对当前全部偏好检查预算兼容性，保留最大兼容子集 | 确定区域几何，并识别采集预算冲突 |
| 5. 形成知识修正 | 区域协方差场×KF预测幅度，再经过冲突回退和同一区域的停用/恢复规则 | 仅修改后验均值用于采集，基础GP不确定性继续保留 |
| 6. 判断是否刷新 | 根据停滞、预测失准、B4冲突、空知识或成功反例决定是否调用LLM | LLM保留、修订、撤销或补充偏好；修订后重做诊断和采集计算 |
| 7. 选择执行模式 | 优先检查第2/4/6轮普通受控试验；P1无普通试验时再检查早期联合/局部跟进；其余为正常区域引导BO | 定向检验占用同一个16次追加预算，不额外赠送实验 |
| 8. 解开完全同分 | 健康路径先看新增单轴对照信息代理，再看潜在SD，最后使用独立随机优先级 | 不使用候选编号大小裁决；不同分数的顺序不变 |
| 9. 承诺选择 | 保存全候选评分表、预算掩码、知识、诊断、KF先验与选中ID | 选择记录写入后才读取该点模拟值/实测表值 |
| 10. 接收并学习 | 保存真实结果，更新GP数据、区域KF和信任证据，再进入下一轮 | 反馈能改变后续知识权重、调用时机和访问轨迹 |

这里“真实结果”在P0是已冻结模拟器的输出，在P1是作者提供的实测表值。本报告没有新增真实合成实验。

### 1.2 GP均值、不确定性与基础BO

GP的预测来自已观测数据，而不是由LLM填入预测表。以标量后验的常见表示为例：

$$
\mu_t(x)=m(x)+k_{xX}[K_{XX}+\sigma_n^2 I]^{-1}(y-m_X),
$$

$$
s^2_{f,t}(x)=k(x,x)-k_{xX}[K_{XX}+\sigma_n^2 I]^{-1}k_{Xx}.
$$

连续量作物理范围归一化；名义变量仅进入相等性/集合成员判断，或进入明确的分子相似度查表，不对任意类别编号计算欧氏距离。GP参数通过已观测前缀的边际似然拟合。表中潜在SD对应未知函数的不确定性；含观测噪声SD还包括观测项，KF使用后者构造测量噪声。

令未查询集合为U，基础采集值为A₀(x)，第t次追加查询t=1…16的正常预算集合为：

$$
x_{\rm BO}\in\arg\max_{x\in U} A_0(x),\qquad
\epsilon_t=\log(4)\frac{16-t}{15},
$$

$$
E_t=\{x\in U:A_0(x_{\rm BO})-A_0(x)\leq\epsilon_t\}.
$$

A₀使用BoTorch的稳定MC-qLogEI实现，包含平滑近似。ε是对数采集分数上的机会限制，不是“产率最多损失ε个百分点”的保证。第16轮ε=0，知识只能在基础最高分集合内影响正常选择。受控试验的例外另行标记，仍计入16次实验。

## 二、修订版方案的关键特征

### 2.1 区域知识保留LGBO式协方差引导

知识写成连续区间或名义类别集合；兼容偏好的交集形成区域R。区域场可概括为：

$$
g_R(x)=\lambda\,{\rm Cov}_{GP}\!\left[f(x),\sum_{z\in R}w_z f(z)\right].
$$

区域场来自当前GP的后验协方差，不是给区域内所有点直接加一个固定产率。P0使用64个连续Sobol位置×8个催化剂的积分表示，P1使用实测域中的名义区域积分，积分点超过512时按既定规则抽取512点。区域权重不读取未查询产率。

修订版把缓存内部核协方差统一到GP后验输出单位，涉及outcome_transform.stdvs²的倍率。P0旧区域对照没有这一修复；P1采用修复后的逐点底座。新旧性能差包含这一实现差异，不能全部归因于信任规则。

### 2.2 KF校正“知识能否提供GP之外的增量”

每个实际区域有独立的联合状态θ=[b,a]：b吸收整体偏移，a校准区域场的幅度。测量模型为：

$$
r_t=y_t-\mu_{GP,t}(x_t)=[1,g_R(x_t)]\theta_t+\nu_t.
$$

采用标准线性KF预测、创新和Joseph协方差更新。产率在滤波中用0…1分数：

| 项 | 固定设置 |
|---|---|
| 初始均值 | [0,1] |
| 初始协方差P₀ | diag(0.0025,0.25) |
| 每轮过程协方差Q | diag(0.000025,0.0025)，区域停用期间按间隔累加 |
| 测量方差R | GP含观测噪声SD²＋0.0009，即3个百分点噪声地板 |
| 部署幅度 | a的预测均值截断到[0,1]，再经过B4及信任门控 |
| 实际部署均值 | μ_GP＋αg_R；b作为干扰项估计，不加到采集均值 |

即使α被门控为0，仍用未缩放区域场和新结果作影子KF更新，使被停用知识有机会积累新的支持证据。a是启发式预测幅度，不是化学命题为真的概率；历史残差还会随GP重拟合而变化。

### 2.3 B4与同区域信任恢复分工

B4检查偏好与当前采集机会预算是否兼容，通过最大兼容子集决定注入区域。修订版遇到B4冲突时把部署幅度置0，并保留影子校正。B4的“冲突”是采集预算冲突，不能直接当作化学证伪。

同一区域一旦被封锁，后续仅因B4暂时变清晰、改写条款名称或修改置信度，都不能自动恢复。恢复需要自身后续实验对实验前冻结的知识预测提供支持：

$$
\Delta SE=(y-\mu_{GP})^2-
\left(y-\mu_{GP}-a_{\rm pre}g_R\right)^2>10^{-12},
$$

同时探测修正非零、更新后KF系数为正，并且后续B4检查清晰。不同几何区域保留自己的证据，不把旧区域的收益自动转给新区域。该判据是局部预测证据，不足以证明整个区域普遍更优。

### 2.4 事件调用LLM，并检查其事实引用

普通刷新事件包括：3轮累计最佳改善不足0.5个百分点、连续2次|创新z|>2.5、B4冲突、无活动知识。常规刷新至少间隔2轮、最晚第14轮，每条轨迹最多3次新增区域刷新，初始知识单独复用。P1早期成功联合反例可在剩余额度内触发下一轮重审；P0成功单轴反例保留既有冷却规则。

LLM输出保留、修订、撤销、补充四类结构化动作。其引用的观测编号、对应轴取值与显示产率均要核对；格式或事实引用失败时只修复同一输入的JSON，不执行新实验，不按产率好坏挑选回复。检查引用正确并不代表其化学解释或因果归纳正确。

### 2.5 用明确预算取得反例和局部反馈

| 分支 | 触发与操作 | 重要限制 |
|---|---|---|
| 普通受控反例 | 第2/4/6轮，当前最佳不足95%；从自己最佳观测出发，只改一个轴且违反当前偏好；后续优先覆盖尚未观察的排除取值 | 仅原池中的精确单轴邻居；完整聚焦池，无top5截断；无邻居则跳过 |
| P1联合检验 | 当前最佳<99%，类别核极端且缺少独立单轴对照时，最多前两轮改变关键类别＋一个新背景，其他轴保持自己最佳观测的设置 | 是两因素联合检验，不作单因素因果结论；后续背景与既往背景的Tanimoto相似度须<0.5 |
| 类别延续 | 第一次联合试验比该类别在本次实验前自身已观察的最佳高至少5个百分点，则第二次继续该类别、换新背景 | 比较对象是“同类别的自身旧观测”，不是全球最佳；没有该类别旧数据则不强行延续 |
| 局部跟进 | 已完成两次联合试验，第二次结果≥75%、当前最佳<99%时，第3轮固定关键类别和背景，只改另一轴 | LLM评分完整小池，选最高预期值；评分并非校准产率预测 |

这些检验都从16次追加查询中划拨。正常区域引导在E内；检验可按协议显式越过门限，报告实际例外次数。P0缺少催化剂分子结构且只有一个名义轴，联合与结构恢复分支不启用。

### 2.6 处理GP失辨识、完全同分和建议复用

P1先检查原生类别GP是否出现≥90%的未查询点并列最高采集值，且≥90%的未查询点与训练集的最大类别核相关性<10⁻¹²；同时满足时，用固定的0.05、5两组长度尺度起点重拟合同一核，只有总边际似然目标改善>0.001才替换。进一步在最小类别长度尺度≤10⁻⁴、最大/最小≥10⁶，且最敏感轴没有单轴对照时，切换并锁定到固定分子相似度核：

$$
K=\sigma_f^2\left[
0.75\,\operatorname{mean}_j T_{\rm Morgan}(x_j,x'_j)
+0.25\exp\!\left(-\frac{\operatorname{mean}_j{\bf1}[x_j\ne x'_j]}{0.2}\right)
\right].
$$

Morgan指纹为radius=2、2048位，包含手性。类别代码仅作矩阵索引与相等性判断。分子特征由输入结构生成，未用产率筛选。P1-Perera实际进入该分支，Doyle保持原生核；因此“同为MixedSingleTaskGP”不等于所有方法使用同一协方差核。

健康路径只在采集值完全相同时，用新增单轴对照数×log(2+未见取值数)/(1+已有对照数)的信息代理，再按潜在SD、独立PCG64优先级裁决；代理不是精确互信息。分子恢复路径保留既定随机并列规则。GP摘要覆盖完整同分组，避免任意截断同分集合。

同一系统提示、公开语义输入和完整自身观测前缀才可复用首个有效原始LLM输出；每次运行重新校验自己的计划。P1三个并列顺序测试使用这一缓存，不能当成三次独立LLM随机采样。
"""


DESIGN = r"""
## 三、两级数据与比较口径

### 3.1 P0和P1分别回答什么

| 项目 | P0：Reizman Suzuki Case 1–4 | P1：Perera Suzuki＋Doyle Buchwald–Hartwig |
|---|---|---|
| 定位 | 混合域机制与错误知识压力检验 | 实测高通量表上的全名义变量搜索及并列顺序敏感性 |
| 目标来源 | 已冻结Summit反应模拟器的产率输出 | 作者打包实测数据表，逐点查表回放 |
| 候选数 | 每Case 4692 | Perera 3696；Doyle 3955 |
| 变量 | 停留时间60–600 s、温度30–110 ℃、负载0.5–2.5 mol%，8个匿名催化剂 | Perera：亲电体4、亲核体3、配体11、碱7、溶剂4；Doyle：配体4、添加剂22、碱3、芳基卤化物15 |
| 数据完整性 | 固定有限池，候选坐标构建不依赖产率 | Perera完整3696组合；Doyle 3960种笛卡尔组合中3955条实测记录，仅实测组合可执行 |
| 被优化的底物 | Case任务内部固定 | 表中列出的底物也是优化变量，不是固定底物后的纯条件优化 |
| 评价单位 | 模拟产率/% | Perera为UV色谱产物峰面积%；Doyle为来源表报告的反应产率%；不能统称分离收率 |
| GP | 连续＋名义Mixed GP | 全名义Mixed GP；Perera诊断后使用分子核 |
| 知识重复 | 四种初始知识条件，每Case每条件一条seed0轨迹 | 一组共同初始知识；优化/初始化seed固定0，并列seed0/1/2 |
| 修订版规模 | 16条完整轨迹，256次追加选择 | 6条完整轨迹，96次追加选择 |

每条均为共同4点初始化＋16次逐点查询（总20点，q=1），各方法使用对应任务的同一候选池和初始4点。本报告只整理已有封存轨迹，没有新增API调用、重拟合或补做实验。P0/P1按任务层级组织，顺序不等同于开发时序：修订版先在P1开发，再迁移重跑P0；两级都不能视为从未查看过的数据盲测。

P0的前三种知识是参考结果知情的人工压力条件：

| 条件 | 实际构造 | 研究含义 |
|---|---|---|
| 正确知识 | 围绕候选池最优参考点构造数值区间，催化剂集合包含最优催化剂 | 正向先验能否转化为搜索收益；不是独立发现的化学真理 |
| 错误类别 | 保留前述连续区间，替换为参照输出较差的催化剂类别 | 检查单个错误类别偏好的损害与保护 |
| 混合错误 | 保留参考催化剂及温度偏好，替换停留时间和负载区间，使联合区域较差 | 检查多条表面可分辨偏好联合误导的风险 |
| 助手初始建议 | 复用最初只看共同4个观测时形成的助手区域建议及置信度 | 当前主系统比较；后续刷新使用deepseek-flash |

P1复用共同4点形成的同一初始知识，没有再做上述四种人工知识条件。P0四条件和P1三个并列顺序不是同一种重复单位；不把它们混算成七个随机种子。

### 3.2 指标定义

令第j次总观测产率为yⱼ，Bⱼ=max(y₁,…,yⱼ)：

$$
\operatorname{SearchAUC}=\frac1{16}\sum_{j=5}^{20}B_j.
$$

本指标在0…100尺度上报告，是16个追加位置累计最佳的均值，不是除以候选池最优值的归一化积分。最佳值取20次中的最大值；首次最佳是第一次达到“该轨迹最终最佳”的总实验次序，包含4个初始化，≤4说明初始化中已取得最终最佳。各方法的“首次最佳”目标可能不同，不能只按该次序排优劣。

主比较表保留三位小数，并列与逐轮示例保留六位；采用十进制四舍五入，显示前仅消除10⁻¹²百分点量级的浮点末位误差。差值单位为百分点，计算及data.json保留完整精度。相同SearchAUC及最终值并不表示访问了相同点；这些指标完全相同的方法不强行排序。两级指标单位及候选域不同，不对P0与P1求一个总体“总分”。

### 3.3 对照方法与必要适配

| 方法 | 本项目的执行方式 |
|---|---|
| 纯BO | 最大化基础qLogEI，不注入知识 |
| 区域LGBO | 固定初始区域的官方解析均值倾斜组件，在全池选择；不受修订版正常机会门限约束 |
| 仅预算门限 | 固定相同区域注入，选择限定在基础qLogEI预算集合E内 |
| 硬B4 | 用基础采集值诊断兼容子集，执行保留偏好的硬可行交集 |
| 软B4＋区域注入 | 兼容子集决定注入区域，选择仍在E内，不要求执行点满足所有保留偏好 |
| LLINBO-Justify | 实际调用官方Justify分支审查LLM提案；有限池与名义变量接口适配，基础qLogEI/机会门限替换原实现相应的UCB/方差门限 |
| Reasoning BO（P1） | 官方推理与high/medium/low稳定排序流程，接收纯BO前5个候选，再选1个逐点查询；原批量3改为同4＋16预算的5选1 |
| 旧G3/R1/R2/R3 | 保留开发阶段已封存的知识更新、事实校验、冲突回退与保守部署版本，名称按所属阶段区分 |
| 修订版Agent | 上述KF、B4、事件刷新、区域信任、并列治理，以及满足条件时的拟合恢复和定向反例 |

这些结果比较的是明确适配后的系统，不能称为所有原论文默认协议的逐字复现。LLINBO、Reasoning BO的逐轮提案调用数，和事件调度方案的调用数并不相同；这里只统一实验查询预算。

P0新旧区域实现存在协方差单位修复差异，P1-Perera还改变了核。因此，结果可说明整个系统在既定预算下的表现，不能仅据总AUC差宣称KF、B4或事件刷新各自具有独立因果增益。早期编号GP、不同协议的HEBO及旧多种子批量实验不与本报告主表混算。
"""


def render(data):
    a0, a1, dev = data['P0'], data['P1'], data['P1_development']
    p0, p1 = data['P0_aggregate_source'], data['P1_aggregate_source']
    lines = ['# P0–P1两级实验数据报告', '',
             '## 报告范围', '',
             '先说明Agent的执行逻辑与修订版特征，再列出P0四个Suzuki Case和P1两套HTE数据的对比结果。结果整理截至2026-10-29，依据已封存的逐轮轨迹重新计算。', '',
             METHODS.strip(), '', DESIGN.strip(), '',
             '## 四、P0：Suzuki Case 1–4结果', '',
             '### 4.1 助手初始建议：四个Case分别比较', '',
             '这是P0的主系统比较。表中的旧R2是P0的R2_conflict_fallback；P1同名历史方案见其独立表，不跨阶段替换结果。', '']
    for case in range(1, 5):
        lines += [f'#### Case {case}', '',
                  metric_table([r for r in a0 if r['case'] == case and r['knowledge_condition'] == 'assistant_initial'], P0_METHODS)]
    natural = [r for r in a0 if r['knowledge_condition'] == 'assistant_initial']
    new = [r for r in natural if r['method'] == 'Stable_revised']
    base = [r for r in natural if r['method'] == 'Pure_BO']
    deltas = []
    for r in new:
        b = next(v for v in base if v['case'] == r['case'])
        old = next(v for v in natural if v['case'] == r['case'] and v['method'] == 'R2_conflict_fallback')
        deltas.append([f"Case {r['case']}", fmt(r['SearchAUC']-b['SearchAUC'], signed=True),
                       fmt(r['final_best_percent']-b['final_best_percent'], signed=True),
                       fmt(r['SearchAUC']-old['SearchAUC'], signed=True)])
    lines += ['### 4.2 P0的增量与压力条件', '',
              table(['Case', '新版AUC－纯BO', '新版最佳－纯BO', '新版AUC－旧R2'], deltas),
              '助手初始建议下，新版相对纯BO的AUC为2胜2负；相对旧R2，两例AUC持平、两例下降。Case 2和Case 3的累计最佳曲线与旧R2指标相同，部分实际查询仍不同。Case 4的新版仍低于仅门限和LLINBO-Justify。', '']
    stress_rows = []
    for method in P0_METHODS:
        vals = [statistics.mean(r['SearchAUC'] for r in a0 if r['method'] == method and r['knowledge_condition'] == regime) for regime in REGIMES]
        stress_rows.append([NAMES[method], *[fmt(v) for v in vals]])
    lines += [table(['方法／四Case均值AUC', *[KNOWLEDGE[r] for r in REGIMES]], stress_rows)]
    stress_delta = []
    for regime in REGIMES:
        subset = [r for r in a0 if r['knowledge_condition'] == regime]
        v = [r for r in subset if r['method'] == 'Stable_revised']
        pure = [r for r in subset if r['method'] == 'Pure_BO']
        stress_delta.append([KNOWLEDGE[regime], fmt(statistics.mean(r['SearchAUC'] for r in v)),
                             fmt(statistics.mean(r['SearchAUC'] for r in v)-statistics.mean(r['SearchAUC'] for r in pure), signed=True)])
    lines += [table(['条件', '新版平均AUC', '新版－纯BO/百分点'], stress_delta),
              '正确人工先验产生明显收益；错误类别条件的四Case平均增量接近0；混合错误条件平均低于纯BO约3.894个百分点。现有结果不支持把修订版称为在Suzuki上稳定抗错误知识的方案。各压力条件的逐Case完整值见附录A。', '',
              '### 4.3 P0中哪些新分支实际执行', '',
              '修订版16条轨迹共48次区域刷新、3次精确单轴试验，正常机会门限外选择0次；6个没有活动区域校准状态的轮次未做KF更新，因此256次选择对应250次实际KF更新。', '',
              '3次单轴试验均发生于Case 2：助手知识第4、6轮，以及错误类别第6轮。其余控制机会按95%阈值或缺少精确邻居等规则跳过。四个Case初始4点附近均没有严格单轴邻居，匿名催化剂也无法提供分子结构核或双类别联合背景；这些分支的缺席限制了P1策略向P0的迁移。', '',
              image('research/suzuki_stable_20261029/comparison_assistant_initial.png',
                    'P0助手初始建议条件：四Case逐轮值与累计最佳'),
              '', '其他条件图：' + '；'.join(link(f'research/suzuki_stable_20261029/comparison_{r}.png', KNOWLEDGE[r])
                                                for r in ('correct', 'wrong_category', 'mixed')) + '。', '',
              '## 五、P1：两套HTE数据结果', '',
              '### 5.1 Perera Suzuki', '',
              '下表主比较取并列seed=0；其余两个并列顺序单列。共同初始最佳为95.570%，距离表值100%的上限较近，需保留指标天花板的解释。表内“最佳值”指UV产物峰面积百分比。', '',
              metric_table([r for r in a1 if r['task'] == TASKS[0]], P1_METHODS),
              '### 5.2 Doyle Buchwald–Hartwig', '',
              '共同初始最佳为77.667%。表内“最佳值”为来源表报告的反应产率百分比。', '',
              metric_table([r for r in a1 if r['task'] == TASKS[1]], P1_METHODS),
              '### 5.3 与主要对照的差值', '']
    comparison_rows = []
    for task in TASKS:
        n = next(r for r in a1 if r['task'] == task and r['method'] == 'feedback_t0')
        for method in ('Pure_BO', 'ReasoningBO_point'):
            b = next(r for r in a1 if r['task'] == task and r['method'] == method)
            comparison_rows.append([task.replace('_', ' '), NAMES[method],
                                    fmt(n['SearchAUC']-b['SearchAUC'], signed=True),
                                    fmt(n['final_best_percent']-b['final_best_percent'], signed=True),
                                    f"{n['first_final_best_experiment']}／{b['first_final_best_experiment']}"])
    lines += [table(['数据集', '对照', '新版AUC－对照', '新版最佳－对照/百分点', '首次各自最佳：新版／对照'], comparison_rows),
              'Perera新版相对纯BO提高2.970个AUC百分点，最终值提高3.400个百分点；相对Reasoning BO最终值相同，但AUC低0.218，晚1次总观测达到该值。Doyle新版比Reasoning BO的AUC高1.289，最终值相同，早1次总观测达到；比纯BO的AUC高5.288，但最终值低0.444个百分点。两个优化指标反映了搜索速度与最终最优值的不同取舍。', '',
              '例如Perera新版的累计最佳为前2个追加位置95.570%、后14个位置99.050%，故AUC=(2×95.570＋14×99.050)/16=98.615；Reasoning BO提前一个位置达到同值，AUC=98.8325。两者0.2175个百分点的差距对应这一轮等待，而不是最终值不同。', '',
              '### 5.4 三组并列顺序验证', '']
    tie_rows = []
    for task in TASKS:
        for tie in range(3):
            r = next(x for x in a1 if x['task'] == task and x['method'] == f'feedback_t{tie}')
            tie_rows.append([task.replace('_', ' '), tie, fmt(r['SearchAUC'], 6),
                             fmt(r['final_best_percent'], 6), r['first_final_best_experiment']])
    lines += [table(['数据集', '并列seed', 'SearchAUC', '最佳值/%', '首次最佳/总次序'], tie_rows),
              '两套数据三组SearchAUC极差均为0，且满足预定有限验收下限。这里变化的是完全同分集合的随机优先级，优化/初始化seed始终0；相同输入的LLM回复按规则复用。Doyle后期实际查询和查询均值有变化，累计最佳保持一致。这不是独立初始化、独立LLM随机性或未知数据集的稳定性证明。', '',
              image('research/stable_r2_20261029/stable_strategy_comparison.png',
                    'P1三组并列顺序与冻结纯BO、Reasoning BO对照'), '',
              '### 5.5 用实际轨迹说明新分支的工作方式', '',
              '以下采用并列seed=0。追加轮次t和包含初始化的总次序t+4分开表示。', '']
    example_rows = []
    for step in p1['selected_controlled_steps']:
        if step['tie_seed'] != 0:
            continue
        name = '联合检验' if step['task'] == TASKS[0] and step['query'] < 3 else (
            '局部跟进' if step['task'] == TASKS[0] else '单轴受控反例')
        example_rows.append([step['task'].replace('_', ' '), step['query'], step['query'] + 4, name,
                             fmt(step['actual_percent']), fmt(step['base_loss'], 6),
                             fmt(step['budget'], 6), '是' if step['gate_exception'] else '否'])
    lines += [table(['数据集', '追加轮次', '总次序', '执行分支', '观测值/%', '基础采集损失', '正常门限', '门限例外'], example_rows),
              'Perera第一轮联合检验得到70.650%，低于本轨迹初始最佳95.570%，但比该关键类别先前自身最佳17.260%高53.390个百分点，因此第二轮继续该类别而更换新背景；得到81.700%，满足≥75%的局部跟进条件；第3轮固定新组合的关键类别与背景，只改另一轴，得到99.050%。第二轮的采集损失1.422917高于正常门限1.293875，是预先允许且占用16次预算的例外。成功局部反例随后在第4轮触发知识重审。', '',
              'Doyle保持原生类别GP，在第2、4、6轮执行3次单轴反例；第6轮取得98.287%，采集损失0.900820仍小于门限0.924196。该结果发生在第10次总实验，比Reasoning BO取得相同最终值早1次。局部观测支持本次搜索中的作用，未证明某一变量在全域具有独立因果效应。', '',
              '## 六、两级结果共同支持什么', '',
              '1. P0支持完整反馈链可执行、可回放，且部分Case能从区域知识中受益；它同时保留了Case 1/3的落后和混合错误知识的损失，不能宣称普遍稳健。', '',
              '2. P1支持在两套已反复分析的seed0基准、固定初始知识和三组并列顺序下保持约定效果。Perera的核恢复与联合/局部试验、Doyle的单轴反例提供了不同的执行路径。', '',
              '3. P1的Perera核不同于原生对照，P0也存在单位修复等共同变化。现有主表不能把全部收益归到KF、B4或事件刷新；尚缺冻结同核、同知识、同提示输入的最终版模块消融及独立初始化验证。', '',
              '4. 当前更稳妥的研究结论是：在固定实验预算下，用实验反馈约束知识部署，并为不确定的知识/模型保留定向检验机会，可以改善部分任务的搜索效率；这还不是化学知识正确率已被校准或跨任务全面胜出的证明。', '',
              '## 附录A：P0人工压力条件的逐Case完整表', '',
              '以下与主表相同，均为各Case各条件一条seed0轨迹；不把四个Case当作四次同分布随机初始化。', '']
    for case in range(1, 5):
        lines += [f'### A.{case} Case {case}', '']
        for regime in ('correct', 'wrong_category', 'mixed'):
            lines += [f'#### {KNOWLEDGE[regime]}', '',
                      metric_table([r for r in a0 if r['case'] == case and r['knowledge_condition'] == regime], P0_METHODS)]
    lines += ['## 附录B：P1关键开发版本与编号敏感性', '',
              '这些是已保留的开发记录，不替换主表冻结对照；尤其不选择结果较好的候选编号顺序作为最终成绩。', '']
    development_rows = []
    for task in TASKS:
        for method in ('R2_region_trust', 'focused_countercheck_v3', 'focused_countercheck_v3_shuffle_s0', 'feedback_t0'):
            source = dev if method.startswith('focused_') else a1
            r = next(x for x in source if x['task'] == task and x['method'] == method)
            development_rows.append([task.replace('_', ' '), NAMES[method], fmt(r['SearchAUC']),
                                     fmt(r['final_best_percent']), r['first_final_best_experiment']])
    lines += [table(['数据集', '版本', 'SearchAUC', '最佳值/%', '首次最佳/总次序'], development_rows),
              'V3由原编号并列改成随机并列后，Perera的AUC从98.589降至95.812；这促使后续同时处理类别核失辨识、完整候选/同分摘要和自身反馈延续，而不是保留编号优势。修订版并没有超过所有历史开发版本的所有指标，例如Doyle的V3单次AUC仍更高。', '',
              '全部完整负面版本及技术中断版本继续保留：' + link('research/stable_r2_20261029/CURRENT_STATUS.md', '开发记录总表') + '。', '',
              '## 附录C：执行核验与数据出处', '',
              table(['项', 'P0修订版', 'P1修订版（三并列顺序合计）'], [
                  ['完整轨迹', '16/16', '6/6'],
                  ['追加选择／已核验', '256／256', '96／96'],
                  ['实际KF更新／已核验', '250／250', '96／96'],
                  ['新增区域刷新（不含复用初始知识）', '48', '18'],
                  ['定向实验', '3次单轴', '9次单轴＋6次联合＋3次局部跟进'],
                  ['正常机会门限外的声明检验', '0', '3（每组Perera第2轮各1次）'],
                  ['逻辑LLM决策请求', '51', '36'],
                  ['当前分支实际收到新API响应（含格式失败）', '58', '9；另28个请求复用原始建议'],
                  ['新响应prompt／completion tokens', '472798／339869', '75564／68964'],
                  ['循环内算子编译', '0', '0'],
              ]),
              '两级后续执行均使用config.txt配置的deepseek-flash；密钥不进入报告。P1新响应数量只统计该最终分支，复用回复的原始调用和共同初始知识已有成本，不能据上述计数直接声称整个研究只花费9次调用。', '',
              'P0使用4个单线程CPU进程，P1使用2个单线程进程；核与线性代数提前预热，精确前缀缓存复用。GP训练点最多19个，未通过增加实验预算换取效果。', '',
              '### C.1 结果与轨迹', '',
              '- ' + link('research/suzuki_stable_20261029/results.json', 'P0结果汇总') + '；' +
              link('research/suzuki_stable_20261029/query_trace.csv', 'P0逐轮决策表') + '；' +
              link('research/suzuki_stable_20261029/ADAPTATIONS.md', 'P0适配说明') + '。',
              '- ' + link('research/stable_r2_20261029/stable_strategy_results.json', 'P1最终汇总') + '；' +
              link('research/stable_r2_20261029/feedback_stability.json', 'P1并列顺序检查') + '；' +
              link('research/stable_r2_20261029/TASK.md', 'P1预定验收口径') + '。',
              '- ' + link('research/hte_pointwise_seed0/results.json', 'P1八方法原逐点对照') + '；' +
              link('research/hte_r2_region_trust_seed0/results.json', 'P1信任修复对照') + '。',
              '', '### C.2 实现入口与方法出处', '',
              table(['环节', '源码入口'], [
                  ['P0执行与选点', link('chem/benchmarks/stable_suzuki_study.py', 'stable_suzuki_study.py') + '；' + link('chem/benchmarks/stable_suzuki_engine.py', 'stable_suzuki_engine.py')],
                  ['P0精确单轴候选与具名提示', link('chem/benchmarks/stable_suzuki_policy.py', 'stable_suzuki_policy.py')],
                  ['P1完整执行流程', link('chem/benchmarks/stable_feedback_study.py', 'stable_feedback_study.py')],
                  ['KF与区域注入', link('chem/benchmarks/regional_ablation_suzuki_engine.py', 'regional_ablation_suzuki_engine.py') + '；' + link('chem/benchmarks/pointwise_hte_engine.py', 'pointwise_hte_engine.py')],
                  ['同区域风险与恢复', link('chem/benchmarks/r2_region_trust_engine.py', 'r2_region_trust_engine.py')],
                  ['P1结构核与健康诊断', link('chem/benchmarks/stable_structure_fit.py', 'stable_structure_fit.py') + '；' + link('chem/benchmarks/stable_r2_fit.py', 'stable_r2_fit.py')],
                  ['P1联合检验、类别延续及局部跟进', link('chem/benchmarks/stable_feedback_bridge.py', 'stable_feedback_bridge.py')],
                  ['信息代理、原始建议缓存', link('chem/benchmarks/stable_information_engine.py', 'stable_information_engine.py') + '；' + link('chem/benchmarks/stable_feedback_requests.py', 'stable_feedback_requests.py')],
                  ['P1基线LLM接口与官方推理适配', link('chem/benchmarks/pointwise_hte_llm.py', 'pointwise_hte_llm.py')],
              ]),
              '只读查看入口：' + link('stable_strategy.py', 'P1状态、离线审计与逐轮inspect') +
              '。复现本报告可运行项目虚拟环境Python及research/p0_p1_report_20261029/build_report.py；脚本仅读取封存结果，不启动优化。', '',
              '本次整理从完整20点轨迹重算全部表值，对照原结果JSON，并核查相同初始化、查询唯一性及冻结查表结果。读取过的输入哈希见sources.json，完整精度行记录见data.json，核验范围见validation_receipt.json。没有重写原实验、发布回执或既有图片。', '']
    return '\n'.join(lines)


def main():
    os.chdir(PROJECT)
    assert fmt(98.83249999999998) == '98.833'
    assert fmt(-.217500000000001, signed=True) == '-0.218'
    inventories = [
        verify_inventory('research/suzuki_stable_20261029'),
        verify_inventory('research/stable_r2_20261029', 'milestone'),
        *[verify_inventory(f'research/stable_r2_20261029/feedback_t{i}') for i in range(3)],
        verify_inventory('research/hte_pointwise_seed0'),
        verify_inventory('research/hte_r2_region_trust_seed0'),
    ]
    data = extract()
    document = render(data)
    (OUT / 'REPORT.md').write_text(document, encoding='utf-8')
    public = {k: v for k, v in data.items() if not k.endswith('_source')}
    public['as_of'] = '2026-10-29'
    public['metric_definition'] = 'mean cumulative maximum percent at total experiment indices5..20'
    public['same_initialization_verified'] = True
    write('data.json', public)
    # Check only actual links, not LaTeX parentheses or Markdown cell formatting.
    import re
    targets = re.findall(r'\]\(([^)]+)\)', document)
    for target in targets:
        assert Path(target).exists(), target
    for path, expected in INPUTS.items():
        assert filehash(path) == expected, path
    write('sources.json', {'scope': 'read-only report inputs', 'files': INPUTS})
    receipt = {'status': 'PASS', 'P0_metric_rows_recomputed': len(data['P0']),
               'P1_metric_rows_recomputed': len(data['P1']),
               'P1_development_rows_recomputed': len(data['P1_development']),
               'unique_complete_trajectories_read': len(STATES),
               'all_histories_checked_against_frozen_tables': True,
               'protected_inventory_checks': inventories,
               'source_files_read_and_unchanged': len(INPUTS),
               'report_links_verified': len(targets),
               'new_model_fits': 0, 'new_API_calls': 0, 'new_experiments': 0,
               'script_sha256': filehash(__file__), 'report_sha256': filehash(OUT / 'REPORT.md'),
               'visual_review': 'existing figures reused unchanged; final review pending'}
    write('validation_receipt.json', receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != 'protected_inventory_checks'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
