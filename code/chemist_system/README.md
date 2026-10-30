# chemist_system — 自主材料合成闭环

新增：[官方源码 LLINBO、LGBO、HEBO 的 Suzuki Case 1–4 对照](docs/suzuki_official_sources.md)。
60 条轨迹、每条共同 4 点初始化 + 16 次新增查询，共 1200 次模拟器查询；官方算法文件原样保存。
SearchAUC 平均值为 LLINBO 68.08%、LGBO 67.62%、HEBO 64.20%；HEBO 原生数值回退计入结果。
新入口为 `python -m chem.benchmarks.suzuki_official`，代码与原始回答、协议和结果位于独立的 `suzuki_official` 目录。

[此前的手写适配对照与 HEBO 精度核验](docs/suzuki_three_baselines.md)属于历史结果，不能引用为 LLINBO/LGBO 官方源码基线。

第5章入口：[Chem 论文正文与完整实验结论](docs/chapter5_chem.md)，
配套[实现、复现与证据索引](docs/chapter5_implementation.md)和[全部方法结果表](research/chapter5/results_tables.md)。
已完成3,755条测试轨迹，保留531条提前停止；B4的7,600次测试决策均经独立穷举核对。
结果支持有界修订和来源检查，但固定先验SearchAUC未超过单候选校验B3，材料部分因来源未合格保持HOLD。
统一研究入口为 `python -m chem.cli decision`；原有 `bo` 命令属于历史流程。

从 `plugins_scripts/` 整理出的独立系统：知识检索 → 协议生成 → 物理审计 → 执行计划 →
XRD 定量 → 贝叶斯优化建议。两条实证路线贯穿全部代码：**CeAl₂Ge₂ 电弧熔炼** 与
**ErTa₇O₁₉ 固相烧结**。

## 目录

```
chemist_system/
├── chem/                        可安装包（python -m chem.cli 为唯一入口）
│   ├── paths.py                 ★ 唯一路径来源（全部可用环境变量覆盖）
│   ├── config.py                MP key / GSAS-II / rxn 环境探测（惰性，不在 import 时抛错）
│   ├── protocol.py              化学计量规则 → 协议（arc_melt / solid_state 两条路线）
│   ├── cli.py                   统一命令行（doctor/loop/protocol/audit/xrd/bo/energetics/screen）
│   ├── loop.py                  七步闭环（原 autonomous_chemist.py，审计改为进程内调用）
│   ├── audit/
│   │   ├── gate.py              PhaseGraphAuditor：audit_protocol + audit_arc_melt
│   │   └── experimental/        torch+CANTERA 动力学支线（未接入主流程，显式导入）
│   ├── energetics/              periodic / miedema / sisso / mp / hull（离线可用）
│   ├── xrd/
│   │   ├── __init__.py          create_estimator 工厂 + SYSTEM_CONFIGS
│   │   ├── peak_based.py        Pawlik/Zevin RIR（ErTa 与 CeAlGe 两套路径）
│   │   ├── pattern.py           CIF → 衍射谱（纯 numpy）
│   │   ├── analyzer.py          相鉴定（SQLite 参考库）
│   │   ├── arcmelt.py           目录级 QPA 数据集构建
│   │   └── gsas2/
│   │       ├── dynamic.py       dynamic Rietveld 后端（生产 agent 或打包 agent）
│   │       ├── agent.py         通用迭代精修 agent（无外部 agent 时使用）
│   │       ├── batch.py         批量重算目录
│   │       └── refinement/      单次精修 / BIC 选相流水线（legacy 后端）
│   ├── bo/                      llm（生产三层 BO）/ gpr（GPR 驱动）/ rietveld（基准）
│   ├── knowledge/                wiki 缓存（WikiCache）
│   └── screening/               MP 候选筛选脚本
├── compat/gsas2_refinement/     旧包名兼容垫片（外部 iter_agent.py 使用）
├── data/                        ★ 数据与代码分离
│   ├── ce_al_ge/                CeAl₂Ge₂ 电弧熔炼数据（唯一数据集目录）
│   │   ├── arc_melt/raw/        原始谱（.raw/.xy/.txt/.mdi，47 个）
│   │   ├── arc_melt/cifs/       该体系物相 CIF（121 个）
│   │   ├── datasets/            arcmelt_dataset*.json
│   │   ├── energetics/          DFT hull / MP 缓存
│   │   ├── bench/               BO 基准曲线
│   │   └── rietveld/            GSAS-II 报告与产物
│   ├── mp_screens/              MP 筛选输出（furnace_xrd, order_magnetocaloric）
│   ├── reference/               XRD 参考库（运行时生成）
│   ├── runs/                    运行产物（batch/iter/residual）
│   ├── deliverables/            阶段 JSON
│   └── cache/                   相注册表等缓存
├── docs/                        架构说明、CeAl₂Ge₂ 决策记录、旧 BO 文档
├── tests/                       62项测试（无需 MP key / 不启动 GSAS-II）
└── pyproject.toml
```

## 环境

| 用途 | 解释器 | 说明 |
|---|---|---|
| 主流程 / 审计 / BO / Pawlik | `python12`（numpy scipy pymatgen rxn_ca rxn_network） | `/home/ucas/miniconda3/envs/python12/bin/python` |
| GSAS-II 动态精修 | `gsas2` conda env | 通过 `conda run -n gsas2 python ...` 调用 |
| MP API | 任意 | `MP_API_KEY` 环境变量，或 `<repo>/local/mp_api_key.txt` |

```bash
cd /home/ucas/workspace/chemist_system
./scripts/chem doctor           # 推荐：包装器会补 LD_LIBRARY_PATH（见下）
PY=/home/ucas/miniconda3/envs/python12/bin/python
$PY -m chem.cli doctor
$PY -m pytest -q                # 15 passed
```

> **必须用包装器的情况**：本机 python12 的 scipy/mp_api 扩展需要
> `CXXABI_1.3.15`，未 `conda activate` 的 shell 里会回落到系统 libstdc++ 而报
> `ImportError`。`./scripts/chem` 会导出 `$CONDA_PREFIX/lib` 到 `LD_LIBRARY_PATH`；
> `chem doctor` 也会打印 `mp_api import` 探针结果。

## 常用命令

```bash
# 七步闭环（CeAl2Ge2 走电弧熔炼路线：审计 10 项安全闸，自动跳过 rxn-ca）
$PY -m chem.cli loop CeAl2Ge2 --force-regen --max-iter 2 -o /tmp/loop.json

# 相鉴定（先播种参考库，再扫谱）
$PY -m chem.cli xrd seed
$PY -m chem.cli xrd identify /home/ucas/workspace/xrd/20260428-1400度1200度/ErTaO_20260428_162240.txt

# 生成协议 / 审计
$PY -m chem.cli protocol ErTa7O19 -o /tmp/proto.json
$PY -m chem.cli audit arc-melt --demo            # 无需 MP key，输出 security_token
$PY -m chem.cli audit protocol /tmp/proto.json   # 需要 MP key（DFT hull）

# XRD 定量（Pawlik 交叉校验：无需 GSAS-II）
$PY -m chem.cli xrd estimate data/ce_al_ge/arc_melt/raw/CeAlGe_10-70-30.xy \
      --system ceal2ge2 --backend pawlik

# 动态 Rietveld（conda gsas2 + GSAS-II，耗时数分钟）
$PY -m chem.cli xrd estimate <file.raw> --system ceal2ge2 --backend gsas2

# BO
$PY -m chem.cli bo suggest       # 读 data/ce_al_ge/datasets/arcmelt_dataset_rietveld.json
$PY -m chem.cli bo bench --reps 300

# 离线能量学
$PY -m chem.cli energetics hull --ce 0.25 --al 0.40
$PY -m chem.cli energetics enthalpy --formula CeAl2Ge2
```

## 路径覆盖（全部在 `chem/paths.py`）

| 变量 | 默认 | 说明 |
|---|---|---|
| `CHEMIST_HOME` | 仓库根 | 包与数据根 |
| `CHEMIST_DATA` | `<home>/data` | 数据根 |
| `CHEMIST_CE_AL_GE` | `<data>/ce_al_ge` | CeAl₂Ge₂ 数据集 |
| `CHEMIST_ARC_RAW` | `<ce_al_ge>/arc_melt/raw` | 原始 XRD |
| `CHEMIST_ARC_CIFS` | `<ce_al_ge>/arc_melt/cifs` | 物相 CIF |
| `CHEMIST_RIETVELD_ENGINE` | 若存在则用外部 `arc_melt/rietveld_engine`，否则 `<ce_al_ge>/rietveld/engine` | dynamic agent 目录 |
| `CHEMIST_WIKI` | `/home/ucas/wiki`（存在时）否则 `<data>/wiki` | 知识库 |
| `CHEMIST_GSAS2_HOME` | `/home/ucas/GSASIIv2` | GSAS-II 安装 |
| `CHEMIST_RXN_PYTHON` | 自动发现 python12 | rxn-ca 子进程解释器 |
| `CHEMIST_XRD_DB` | `<data>/reference/xrd_reference.db` | 相鉴定参考库 |

## 相对旧目录的实质变化

1. **路径唯一化**：删除了 11 处指向 `/home/ucas/.hermes/plugins_scripts` 与
   `.hermes/deliverables` 的硬编码（这些目录已不存在，旧代码的审计子进程必然失败）。
2. **审计改为进程内**：`run_audit()` / `_run_audit_result()` 直接调用
   `chem.audit.gate`，不再写临时 JSON 再起子进程；`audit.py` 导入时不再因缺 MP key 崩溃。
3. **协议生成自包含**：新增 `chem/protocol.py`（化学计量规则，arc-melt/solid-state
   双路线），不再依赖已消失的 `protocol_optimizer/tools.py`。
4. **电弧熔炼真正接入闭环**：`loop` 依据 `route` 分流到 `audit_arc_melt()`，
   APPROVED/WARNING 视为通过，金属间化合物自动跳过 rxn-ca（旧代码从不调用 arc 闸门）。
5. **修复 Pawlik 后端**：`PeakBasedRI.estimate()` 原先只支持 ErTa₇O₁₉ 的
   经验比 R，`ceal2ge2` + `--backend pawlik` 恒返回 0；现按 `method="PawlikQPA"`
   走 RIR 质量分数（实测 68.55% ± 0.83%）。
6. **动态 GSAS-II 后端可在无外部 agent 时工作**：优先使用
   `<engine>/iter_agent.py`，缺失则调用打包的 `chem.xrd.gsas2.agent`；
   `compat/gsas2_refinement` 让外部 agent 的旧 import 名解析到新包。

## 删除清单（已确认无调用方）

| 删除 | 原因 |
|---|---|
| `autonomous_bo.py`, `knowledge_guided_bo/{core,knowledge_base,acquisition}.py` | 被 `bo/llm.py` 的 llm→GPR 三层引擎取代 |
| `framework.py` | Module/Proposer 插件框架，无任何调用方 |
| `hull_auditor.py` | 迁为 `chem/energetics/hull.py` 并接入 CLI |
| `demo_run.py` | 演示旧 BO/hull 组合，已失效 |
| `xrd_yield_estimator/rietveld.py`, `rietveld_core.py` | 纯 Python Rietveld（Rwp≈64%），已被 GSAS-II 取代 |
| `xrd_yield_estimator/dynamic_gsas2.py`, `xrd_bo_loop.py` | 前者是 re-export 包装；后者依赖已删的 KB BO |
| `minimax_mcp/bridge.py` | 与化学系统无关的视觉 MCP 桥 |
| `xrd_database.db` | 空库（phases/peaks 均 0 行） |
| `_run_rxn_ca_simulation.py` | 文件自述“粘贴覆盖用”的过期草稿 |
| `gsas2_refinement/{bo_reservoir,chem_entry}.py` | 写死 `.hermes` 路径的一次性脚本 |
| `rietveld_engine/*.bak_*`, 全部 `__pycache__` | 备份/字节码 |

保留但标注为非主流程：`chem/audit/experimental/`（torch+CANTERA 动力学，
未被 `loop` 引用）、`chem/bo/gpr.py`、`chem/screening/`。

## 已知限制

- **MP key 未配置**：`audit protocol`（DFT hull）与 rxn-ca 需要 key；arc-melt 审计、
  Pawlik QPA、BO、energetics 均不需要。key 就绪后无需改代码。
- **dynamic Rietveld 未在本机端到端跑通验证**（需 conda gsas2 + 数分钟/样品）；
  已验证命令构造与报告路径选择，Pawlik 路径已用真实谱验证。
- 旧目录 `/home/ucas/workspace/plugins_scripts` 未删除，作为回退快照保留。
- 原仓库内曾硬编码过一个 Materials Project API key（`mp_key.py`、`dynamic_gsas2.py`、
  `autonomous_chemist.py` 内联）；新树已全部改为惰性读取。**建议在 MP 后台轮换该 key。**


## 验证记录（本机实测）

| 检查 | 命令 | 结果 |
|---|---|---|
| 语法编译 | `compileall chem compat` | 通过 |
| 导入矩阵 | 59 个模块逐个独立进程导入 | 59/59 |
| 单元测试 | `pytest -q` | 15 passed |
| 环境自检 | `chem doctor` | 路径/依赖/工具链全部就位（MP key 未配置） |
| 协议生成 | `chem protocol ErTa7O19` | 输出 solid_state 协议 |
| 电弧熔炼审计 | `chem audit arc-melt --demo` | WARNING + `AUTH_TOKEN_ARC_PASS_…`，10 项检查 |
| 氧化物审计 | `chem audit protocol` | 无 key 时明确报错并退出码 2（不抛栈） |
| Pawlik 定量 | `chem xrd estimate … --backend pawlik` | 68.55% ± 0.83%，4 相匹配 |
| 相鉴定 | `chem xrd seed` + `xrd identify` 真实 ErTa 谱 | 20 相入库，ErTaO4/Er2O3 等命中 |
| BO 建议 | `chem bo suggest` | 16 条观测 → GPR 模式出建议 |
| BO 基准 | `chem bo bench --reps 20` | ETTE agent 3.65 ± 0.79 |
| 离线能量学 | `chem energetics hull/enthalpy` | 相组合 + ΔHf = −46.7 kJ/mol-atom |
| 批量重算接线 | `chem xrd batch --dir <空目录>` | n_ok=0，正常收尾 |
| 七步闭环 | `chem loop CeAl2Ge2 --force-regen` | 6.8 s，success=True，audit=PASS，7 步执行计划（S2 429 时自动降级到规则协议） |
| 目录无关性 | 在 `/tmp` 下调用包装器 | 正常 |
| 数据根覆盖 | `CHEMIST_CE_AL_GE=/tmp/fake … doctor` | 路径随之切换 |

未端到端验证（环境原因，非代码缺陷）：
- `--backend gsas2` 实跑：需 conda `gsas2` + GSAS-II，每样品数分钟；只验证了命令构造与报告路径选择。
- 带 MP key 的 `audit protocol` / rxn-ca：本机无 key。
- `chem screen ...`：本机 key 为占位符，请求返回 401（脚本已改为导入安全，调用时才发请求）。
