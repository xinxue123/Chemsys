# 化学合成家系统架构 (2026-05-29)

## 一、顶层定位

化学合成家 = `autonomous_chemist.py` (单文件，~2900行)  
是整个系统的**唯一入口和决策引擎**，不是"一个模块"，是一套完整的推理循环。

```
用户提问 → autonomous_chemist.py → 知识检索 → 协议生成 → 审计 → 执行计划 → XRD反馈 → BO建议
```

---

## 二、目录结构

```
/home/ucas/.hermes/plugins_scripts/
├── autonomous_chemist.py          ★ 主入口，单文件闭环（Step 1-7 全在里面）
├── autonomous_bo.py               旧 BO（废弃，不用）
├── llm_driven_bo.py               ★ 生产 BO 引擎（替代 autonomous_bo）
├── audit/                         ★ 审计模块
│   ├── audit.py                   PhaseGraphAuditor — audit_protocol() + audit_arc_melt()
│   ├── engine.py                  审计引擎底层
│   ├── physics.py                 热力学/扩散模型
│   ├── data.py                    数据层
│   ├── ion_transport.py           KD 扩散系数
│   └── path_cost.py               反应路径代价
├── xrd_yield_estimator/           ★ XRD 定量（QPA 后端）
│   ├── __init__.py                create_estimator() 工厂
│   ├── dynamic_gsas2.py           兼容 wrapper（re-export 到 autonomous_chemist_lib）
│   ├── peak_based.py              Pawlik RIR 实现（CeAl₂Ge₂ 快速 QPA）
│   ├── rietveld.py                纯 Python Rietveld（已不用）
│   ├── rietveld_core.py           Rietveld 底层
│   └── xrd_bo_loop.py             XRD BO 闭环脚本
├── autonomous_chemist_lib/        ★ 化学合成家专属实现（独立包）
│   ├── __init__.py
│   └── rietveld/                  GSAS-II dynamic Rietveld 正式位置
│       ├── __init__.py
│       ├── dynamic_gsas2.py       正式实现（不再在 xrd_yield_estimator 里）
│       └── batch_recompute_dynamic.py  批量重算脚本
├── knowledge_guided_bo/           WikiCache（KB 检索工具，BO 历史层）
│   ├── wiki_cache.py
│   ├── core.py
│   ├── knowledge_base.py
│   └── acquisition.py
├── gsas2_refinement/              GSAS-II 单次 workflow（legacy，非 dynamic）
│   ├── workflow.py                GSAS2WorkflowEstimator（backend="gsas2_legacy" 时用）
│   ├── gsas2_runner.py
│   ├── phase_manager.py
│   └── cif_manager.py
├── xrd_analyzer.py                 XRD 相鉴定（SQLite + scan_dat）
├── tests/
│   └── test_dynamic_gsas2_integration.py   集成回归测试
└── references/                    文档（设计决策、技术债务）
```

---

## 三、七步闭环（Step 1 → Step 7）

### Step 1 — 知识库检索
```
full_loop() → wiki_lookup() → WikiCache (knowledge_guided_bo/wiki_cache.py)
```
- WikiCache 在 `/home/ucas/wiki/` 上建立内存索引（O(1) 查询）
- 后备：Hindsight（语义记忆）+ deliverables/ JSON 文件

### Step 2 — 文献检索
```
full_loop() → literature_search() → Semantic Scholar Graph API + arXiv
```
- 仅在 KB 上下文不足时触发
- LLM 解析摘要，提取温度/时间/气氛/压力参数

### Step 3 — JSON 合成协议生成
```
full_loop() → generate_protocol()
```
- 输出格式：前驱体配比、晶体结构、热处理分段工艺、预期物相组合、XRD 指纹峰

### Step 4 — 审计技能评估
```
full_loop() → audit_loop() → audit/audit.py → PhaseGraphAuditor
```
- **氧化物体系**（ErTa₇O₁₉）：`audit_protocol()` → SISSO E_hull + Tammann gate + KD 模型
- **金属间化合物**（CeAl₂Ge₂）：`audit_arc_melt()` → 5 类 10 项物理检查（水冷/真空/元素挥发/退火工艺）
- 拒绝 → Step 5 迭代精炼；通过 → Step 6

### Step 5 — 迭代精炼
- 审计失败时，查阅 KB 替代方案，重生 JSON 协议
- 默认最多 5 次迭代

### Step 6 — rxn-ca 元胞自动机模拟
```
full_loop() → _run_rxn_ca_simulation()
```
- **仅适用氧化物**（ErTa₇O₁₉ 等固相烧结体系）
- **金属间化合物**（CeAl₂Ge₂）：arc-melt 是非平衡过程，CA 不可用 → 跳过
- 数据流：MP API → GibbsEntrySet → reaction-network → ΔG Score → CA 演化 → 物相分数

### Step 7 — 执行 + XRD 反馈
```
full_loop() → generate_execution_plan() → 用户执行 → XRD → update_from_xrd()
```
- `update_from_xrd()`：写 wiki / Hindsight / deliverables JSON

---

## 四、Step 7 的详细实现（XRD QPA 流水线）

```
XRD 数据 → scan_dat → 相鉴定
  → create_estimator('ceal2ge2', backend='auto') → 定量
  → constrained_yield() → 投料约束校正
  → _run_audit_result() → 语义先验
  → add_observation() → BO 反馈
```

### 4.1 create_estimator 工厂

```
create_estimator('ceal2ge2', backend='auto')
  └→ AutoXRDYieldEstimator
       ├→ primary:   DynamicGSAS2RietveldEstimator
       └→ fallback: PeakBasedRI
```

`backend` 参数路由：

| backend 值 | 效果 |
|---|---|
| `gsas2` | 强制 `DynamicGSAS2RietveldEstimator` |
| `gsas2_legacy` | 旧单次 `GSAS2WorkflowEstimator` |
| `pawlik` / `peak` | `PeakBasedRI` (Pawlik RIR) |
| `auto` | ceal2ge2 用 AutoXRDYieldEstimator；其他用 PeakBasedRI |

### 4.2 DynamicGSAS2RietveldEstimator（正式实现）

路径：`autonomous_chemist_lib.rietveld.dynamic_gsas2`  
兼容路径：`xrd_yield_estimator.dynamic_gsas2`（re-export）

封装三层：
1. `iter_agent.py` — GSAS-II dynamic Rietveld agent（retry + divergence guard）
2. `residual_peak_diagnose.py` — residual peak 诊断（no_candidate / phase_limited）
3. `_report_to_yield_result()` — 分类逻辑（pass / near / failed_current_model / unknown_not_quantifiable）

输出 `YieldResult`：

| 字段 | 含义 |
|---|---|
| `yield_pct` | 目标相 CeAl₂Ge₂ 质量分数（%） |
| `sigma_pct` | 不确定性（%） |
| `phase_fractions` | 所有检测相的质量分数 |
| `phase_peaks_used['status']` | pass / near / pass_caution / failed_current_model / unknown_not_quantifiable |
| `phase_peaks_used['use_for_bo']` | 是否可作为 BO 数值反馈 |
| `phase_peaks_used['no_fallback']` | 是否阻断 Pawlik fallback |
| `phase_peaks_used['authoritative']` | 是否为权威结果 |
| `phase_peaks_used['label_type']` | rietveld_CeAl2Ge2_wt_percent / penalty_pseudo_yield_not_physical / ... |
| `phase_peaks_used['residual_diagnosis']` | residual peak 诊断结果 |
| `phase_peaks_used['metrics']` | {rwp, gof} |
| `phase_peaks_used['unknown_residual_peaks_2theta']` | 未解释峰 2θ 位置列表 |

分类规则（`_classify_result`）：

- **Ga 四元体系** → `unknown_not_quantifiable` + `no_fallback=True`
- **sigma 非有限或 >50%** → `failed_current_model` + `no_fallback=True`（sigma blow-up 防护）
- **无候选相解释 residual peaks 且 n_resid≥2** → `unknown_not_quantifiable`
- **Rwp≥25 或 GOF≥5** → `failed_current_model`
- **15<Rwp<20 或 3<GOF<5** → `near`（低 BO 权重）
- **12<Rwp≤15 或 2.3<GOF≤3** → `pass_caution`
- **Rwp≤12 且 GOF≤2.3** → `pass`

### 4.3 batch_recompute_dynamic.py

路径：`autonomous_chemist_lib/rietveld/batch_recompute_dynamic.py`  
功能：批量重算 `/home/ucas/workspace/arc_melt/*.raw` → 输出 `summary.json` + `summary.csv`

每次运行输出：
- `n_total`：总样品数
- `n_ok` / `n_error`：成功/错误数
- `n_use_for_bo`：QPA 有数值的样品数
- `n_bo_training_eligible`：可进入 Ce-Al-Ge BO 训练的样品数（CeAlGe_* + 非 Ga）
- `rows[].bo_training_eligible`：布尔值

---

## 五、贝叶斯优化（BO）

### 生产引擎：`llm_driven_bo.py`

架构：三层自适应
- n < 3：`ConservativePrior`
- 3 ≤ n < 10：`LLMGuided`（提供自然语言系统描述）
- n ≥ 10：`GPR`（高斯过程回归）

接口：
```python
from llm_driven_bo import LLMDrivenBO, heuristic_reasoner
bo = LLMDrivenBO(system_description="...", llm_reasoner=heuristic_reasoner)
bo.add_observation(AlCe_ratio=6.0, anneal_temp=0, yield_value=64.5, yield_sigma=3.0)
suggestion = bo.suggest(method="EI")
```

`yield_value` + `yield_sigma` 传入 `add_observation()`，`yield_sigma` 用于 heteroscedastic GPR（待 patch 进入 GPR alpha）。

**⚠️ CeAl₂Ge₂ 特殊处理**：
- yield 响应是阶梯函数（AlCe<5 → 0%, AlCe=6 → ~65%, AlCe>10 → 伪值）
- 标准 4D BO 无法处理 → **固定 AlCe=6.0 → 2D BO（退火温度 + 时间）**
- 不推荐 anneal_temp > 200°C（Ce₂Al₃Ge₄ ground state 会捕获）

### 旧引擎（废弃）：
- `autonomous_bo.py` — 废弃
- `knowledge_guided_bo/` — 废弃（SystemPlugin 架构，不兼容）

---

## 六、审计（Audit）

### PhaseGraphAuditor（`audit/audit.py`）

两种模式：

**SISSO 模式**（氧化物，固相烧结）：
```
audit_protocol(json_proposal) → {status, regime, E_hull, KD_diag, ...}
```
- E_hull：从 PhaseDiagram(raw_entries) 计算（不用 SISSO-corrected，后者对 Er-Ta-O 全零）
- Tammann gate：0.67 × T_m onset
- KD 扩散：ion_transport.py

**arc-melt 模式**（金属间化合物）：
```
audit_arc_melt(recipe_dict) → {status, security_token, checks[], ...}
```
- 5 类 10 项检查：HW（水冷/真空/Ti-getter）、EL（元素沸点/未知）、PROC（翻转次数/电流/气压）、PT（退火保护）
- 高挥发性元素（BP < 2500°C）：Zn/Mg/Cd/Na/K → 一票否决
- 未知元素 → HOLD（不静默拒绝）
- `security_token`：带 token 才允许起弧

---

## 七、rxn-ca（仅氧化物）

`_run_rxn_ca_simulation()`（`autonomous_chemist.py` 内）：

数据流：
```
MP API → MPRester.get_entries_in_chemsys("Er-Ta-O")
  → GibbsEntrySet → BasicEnumerator(precursors, targets)
  → ComputedReaction.energy_per_atom @ T
  → _DeltaGScore(phase_set) → sigmoid(ΔG)
  → ScoredReactionSet → ReactionLibrary
  → HeatingSchedule → ReactionRecipe
  → setup_noise_reaction() → CA evolution (size³ sites)
  → RxnCAResultDoc.final_simulation.state
  → _extract_phase_fractions(state) → {phase: fraction}
```

⚠️ 默认 `TammanScore` 在 T << T_m 时失效 → 使用 `_DeltaGScore`（ΔG 驱动）。

⚠️ CA precursor retention 是物理正确的，不是 bug：
- ErTa₇O₁₉ E_hull=+9.3 meV（亚稳态），ErTaO₄ 是真正基态
- 单次 CA step 不能代表 6.5h 连续扩散

⚠️ arc-melt 体系不适用 CA（需要 arc melting 设备）。

---

## 八、知识库

### WikiCache（`knowledge_guided_bo/wiki_cache.py`）
- 路径：`/home/ucas/wiki/`
- 格式：`.md` 实体文件（name / aliases / chemsys / mp_id / synthesis_history）
- 性能：首次 ~1ms 建立索引，之后 O(1) 查询
- mtime 失效：自动刷新

### Hindsight（向量记忆）
- bank: hermmes
- 语义搜索：`hindsight_recall(query)` / `hindsight_reflect(query)`
- 保留：实验决策、失败模式、验证结果

### deliverables/
- JSON 合成计划文件（跨 session 持久化）

---

## 九、已废弃/历史模块

| 文件/模块 | 状态 | 原因 |
|---|---|---|
| `autonomous_bo.py` | 废弃 | llm_driven_bo.py 替代 |
| `knowledge_guided_bo/` (SystemPlugin) | 废弃 | 架构不兼容，已被 llm_driven_bo.py 替代 |
| `gsas2_refinement/workflow.py` (GSAS2WorkflowEstimator) | 保留（legacy） | backend="gsas2_legacy" 时用 |
| `xrd_yield_estimator/rietveld.py` | 不再用 | 纯 Python Rietveld，Rwp 64%，无法达到 publication 级别 |
| `xrd_yield_estimator/dynamic_gsas2.py`（旧位置） | 转为 re-export wrapper | 正式实现在 autonomous_chemist_lib/rietveld/ |

---

## 十、依赖环境

- **python312**（主环境）：autonomous_chemist.py、llm_driven_bo.py、audit.py
- **gsas2**（Conda 环境）：GSAS-II refinement、iter_agent.py、residual_peak_diagnose.py
- **python12**（rxn-ca）：rxn-ca cellular automaton、reaction-network、pymatgen

环境变量：
```
ARC_MELT_ROOT=/home/ucas/workspace/arc_melt
RIETVELD_ENGINE_DIR=/home/ucas/workspace/arc_melt/rietveld_engine
WIKI_PATH=/home/ucas/wiki
```

---

## 十一、关键文件大小（近似行数）

| 文件 | 行数 | 用途 |
|---|---|---|
| `autonomous_chemist.py` | ~2900 | 主闭环 |
| `audit/audit.py` | ~900 | 审计引擎 |
| `llm_driven_bo.py` | ~700 | BO 引擎 |
| `xrd_yield_estimator/__init__.py` | ~430 | QPA 工厂 |
| `autonomous_chemist_lib/rietveld/dynamic_gsas2.py` | ~280 | GSAS-II dynamic backend |
| `knowledge_guided_bo/wiki_cache.py` | ~470 | KB 索引 |

---

## 十二、典型调用链

**CeAl₂Ge₂ arc-melt 全流程**：

```
用户 → autonomous_chemist.py full_loop('CeAl2Ge2')
  ├→ Step 1: wiki_lookup('ceal2ge2') → WikiCache
  ├→ Step 2: literature_search() [if KB insufficient]
  ├→ Step 3: generate_protocol() → JSON
  ├→ Step 4: audit_loop() → audit.audit.PhaseGraphAuditor.audit_arc_melt()
  │            → {status: WARNING, security_token: AUTH_TOKEN_ARC_PASS_...}
  ├→ Step 6: (跳过 rxn-ca — arc-melt 不适用)
  ├→ Step 7: generate_execution_plan() → arc-melt 工艺指令
  │
  └→ XRD 反馈回圈:
       create_estimator('ceal2ge2', backend='auto')
         └→ AutoXRDYieldEstimator
              ├→ primary: DynamicGSAS2RietveldEstimator
              │    ├→ iter_agent.py (gsas2 env)
              │    ├→ residual_peak_diagnose.py (gsas2 env)
              │    └→ _report_to_yield_result() → YieldResult
              └→ fallback: PeakBasedRI
       constrained_yield() → 投料约束校正
       _run_audit_result() → 语义先验
       llm_driven_bo.add_observation(yield_value, yield_sigma) → BO
```

**ErTa₇O₁ Nine 氧化物全流程**：

```
用户 → autonomous_chemist.py full_loop('ErTa7O19')
  ├→ Step 1-5: 同上，但 audit_protocol()（SISSO）不是 audit_arc_melt()
  ├→ Step 6: _run_rxn_ca_simulation()
  │    ├→ MP API → reaction-network → 1个反应
  │    └→ _DeltaGScore → rxn-ca → phase_fractions
  └→ Step 7: XRD 反馈
       create_estimator('erta7o19', backend='auto')
         └→ PeakBasedRI（ErTa₇O₁₉ 无 dynamic GSAS-II backend）
       constrained_yield() (ErTa₇O₁₉ 不支持，返回 {})
```