# CeAl₂Ge₂ 贝叶斯优化决策流程 (2026-05-29)
# ===============================================

## 目标

最大化 CeAl₂Ge₂ 目标相产率，找到最优 AlCe / GeCe 比例组合。

## 问题本质

```
CeAl₂Ge₂ 化学计量比：
  Al/Ce = 2.0
  Ge/Ce = 2.0

现有 XRD 数据库网格（17 个可靠样本）：
  AlCe ∈ [1.0, 25.0]
  GeCe ∈ [10.0, 90.0]

★ CeAl₂Ge₂ (AlCe=2, GeCe=2) 区域完全空白 — 核心问题
★ 模型在 AlCe=2 处外推失效（GPR 预测恒为 0% ± 38%）
```

## 模型配置

- Surrogate: GPR (sklearn, Matern nu=2.5, alpha = sigma_obs^2)
- Acquisition: UCB (kappa=2.0)
- Heteroscedastic noise: alpha = yield_sigma^2 per observation
- Training: 12 个可靠样本 (pass / pass_caution / near status)

## 当前最优（数据库已确认）

| 排名 | Sample | AlCe | GeCe | Yield% | Sigma% | Status |
|------|--------|------|------|--------|--------|--------|
| 1 | CeAlGe_1-30-15 | 1.0 | 30.0 | 72.19% | 12.0 | near |
| 2 | CeAlGe_5-70-30-DSC | 5.0 | 70.0 | 68.65% | 3.0 | pass |
| 3 | CeAlGe_10-60-30 | 10.0 | 60.0 | 68.57% | 8.0 | pass_caution |
| 4 | CeAlGe_10-70-30 | 10.0 | 70.0 | 53.59% | 12.0 | near |
| 5 | CeAlGe_5-70-30 | 5.0 | 70.0 | 53.09% | 3.0 | pass |

## BO 决策迭代

### 迭代 0 — 初始化

种子样本（5 个，按 AlCe 分 band + 全局最优）：
- CeAlGe_1-30-10  (AlCe=1, GeCe=30, Yield=53.05%)
- CeAlGe_1-30-15  (AlCe=1, GeCe=30, Yield=72.19%) ← 全局最优
- CeAlGe_5-70-30-DSC (AlCe=5, GeCe=70, Yield=68.65%)
- CeAlGe_10-50-50 (AlCe=10, GeCe=50, Yield=1.14%)
- CeAlGe_20-70-30 (AlCe=20, GeCe=70, Yield=5.44%)

### 迭代 1 — 模型决策

**分析：**
- UCB 最高点：AlCe=1.0, GeCe≈19 (UCB=93.2)
- 这是已探索区域的延伸（AlCe=1 已有数据，GeCe 向 10-20 方向）
- AlCe=2 区域（CeAl₂Ge₂ 理论组分）模型完全无法外推

**BO 决策：**
```
合成样品：CeAlGe_1-20-20
  AlCe = 1.0
  GeCe = 20.0
  预测产率：~30-37%（高 sigma≈30%）
  预期 Rwp：预计 pass/near（AlCe=1 区域模型拟合良好）
```

**探索意义：**
- 若 Yield > 50% → 确认低 AlCe 路线可行，向 AlCe=1-2 区域深入
- 若 Yield < 20% → 低 AlCe 路线失效，需转向 AlCe=5-10 探索
- 若 Yield ≈ 30-40% → 确认趋势，模型更新后继续迭代

**实验预期：**
- 此样品合成后，无论结果如何，都极大提升模型对低 AlCe 区域的认知
- 是向 CeAl₂Ge₂ (AlCe=2) 迈出的最稳健第一步

### 迭代 2 — 取决于迭代 1 结果

**Case A：迭代 1 产率高（>50%）**
```
模型建议：AlCe=1.5-2.5, GeCe≈15-25
策略：向 AlCe=2 方向精炼，缩小步长
预期实验：CeAlGe_1.5-20-20 或 CeAlGe_2-20-20
```

**Case B：迭代 1 产率中等（20-50%）**
```
模型建议：AlCe≈1, GeCe 保持探索
策略：扩大 GeCe 范围搜索
预期实验：CeAlGe_1-15-15
```

**Case C：迭代 1 产率低（<20%）**
```
模型建议：AlCe=5-10, GeCe=15-30
策略：转向中 AlCe 区域探索
预期实验：CeAlGe_5-15-20
```

### 迭代 3 — 精炼最优区域

无论前两轮结果如何，第 3 次迭代在：
- 若找到 Yield > 70% 区域 → 锁定该区域，高密度采样
- 若所有探索产率 < 50% → 重新评估 CeAl₂Ge₂ 是否为最优目标相

## 决策约束

1. **不跳过 AlCe=1 区域直接到 AlCe=2**
   - 理由：AlCe=2 完全无数据，GPR 外推不可信
   - 理由：AlCe=1 已有 2 个高 yield 样本（72%, 53%），模型在低 AlCe 区可信

2. **每次迭代只选 1 个样品**
   - 理由：arc-melt 实验成本高，每次运行后更新模型
   - 理由：顺序决策比批量决策更高效（信息增益最大化）

3. **sigma blow-up 保护**
   - 当 sigma > 50% 时，结果标记为 `failed_current_model`
   - 不进入 BO 训练集，防止低质量数据扭曲模型

## BO 训练集更新协议

每次 XRD 实验完成后：

```
1. 运行 GSAS-II dynamic Rietveld
2. 获取 YieldResult(yield_pct, sigma_pct, status, label_type)
3. 若 status in {pass, pass_caution, near} 且 sigma < 50%：
     → bo.add_observation(AlCe=X, GeCe=Y, yield_value=V, yield_sigma=S)
     → 模型自动重拟合
4. 若 status = failed_current_model 或 sigma > 50%：
     → 仅记录，不进入 BO（label_type = penalty_pseudo_yield）
5. 若 unknown_not_quantifiable（Ga 体系等）：
     → 排除在 Ce-Al-Ge BO 训练集外
```

## 核心结论

```
⚠️  CeAl₂Ge₂ (AlCe=2, GeCe=2) 目标相在数据库中完全无数据
⚠️  模型在 AlCe=2 处外推失效
⚠️  当前最优：CeAlGe_1-30-15 → 72.19%

BO 策略：
  Step 1: 先探索 AlCe≈1, GeCe≈15-20 区域（稳健，信息增益最大）
  Step 2: 根据结果，逐步向 AlCe=2 方向精炼
  Step 3: 最终锁定 CeAl₂Ge₂ 高产率组合

迭代 1 推荐实验：CeAlGe_1-20-20
  AlCe=1, GeCe=20
  位于已知最优区域（AlCe=1）的延伸方向
  模型置信度：低（sigma≈30%），但结构上是合理的探索方向
```

## 文件位置

- 数据来源：`/home/ucas/workspace/arc_melt/rietveld_engine/runs/recompute_all_dynamic/20260529_113235/`
- BO 实现：`/home/ucas/.hermes/plugins_scripts/llm_driven_bo.py`
- QPA 后端：`/home/ucas/.hermes/plugins_scripts/autonomous_chemist_lib/rietveld/dynamic_gsas2.py`