# G3 修改与完整反馈验证

入口：`chem/benchmarks/suzuki_grounded_regional.py`。算法：`grounded_regional_suzuki_engine.py`；实验编排：`grounded_regional_suzuki_study.py`；独立复核与表格：`grounded_regional_suzuki_report.py`。

原 `regional_ablation_suzuki_*` 源码与原实验结果保持冻结，避免修复后覆盖原对照。

## 三档累计修改

| 档位 | 具名输入与数值校验 | B4 冲突部署回退 | 新区域幅度保守化 |
|---|---|---|---|
| R1_grounded | 是 | 原 G3 机制 | 原 G3 机制 |
| R2_conflict_fallback | 是 | 是 | 原 G3 机制 |
| R3_conservative_region | 是 | 是 | 是 |

1. **输入与证据**：每个已观测实验和 GP 摘要候选同时提供具名的归一化值、物理值和单位，取消裸 `z` 数组。LLM 每个动作必须引用已观测行、该槽位的变量名、对应归一化值、物理值和产率；数值与显示值误差不得超过 1e-8。催化剂用名称，不能以连续数值代替。不符合时按原来的最多 3 次已接收响应规则修复，首先有效的方案生效。校验的是数值引用，不是自由文本中的化学因果结论。
2. **冲突时回退**：B4 诊断删除至少一个条款或报告非 `none` 冲突时，该轮部署幅度为零，精确选择基础 qLogEI 最大候选。知识不永久删除。仍保留原 B4 保留区域的未缩放场，使用新实验残差进行影子 KF 更新。刷新可以先解决冲突，然后重新诊断；最终诊断决定该次选择。
3. **新区保守部署**：当实际区域几何与完整初始区域不同时，部署 `alpha=clip(mean_a-sqrt(Paa_forecast),0,1)`。初始几何继续使用原来的 `clip(mean_a,0,1)`。新区第一次约为 0.4975，之后仅用其自身反馈与过程噪声更新。返回已见几何复用该几何的银行；不移植其他区域的校准。此规则贯穿新区的使用期，不仅第一轮；B4 冲突优先使 alpha 为零。这是保守启发式，不是经过校准的置信保证。

## 不变项与比较边界

Reizman Suzuki Case 1–4；`correct/wrong_category/mixed/assistant_initial` 四种初始知识。统一 MixedSingleTaskGP、4692 候选、seed=0、4 次初始化＋16 次新增查询、GP 拟合种子、MC512 qLogEI、r4 递减基础分预算、官方区域协方差均值场及原 KF Joseph 更新。每个新档位按自己的完整反馈轨迹重新调用 DeepSeek，无历史答案移植。共 48 条新轨迹、768 次新增查询，最多 144 次逻辑刷新。

R1 与原 G3 的比较同时包括新提示词与新 LLM 响应。R2/R3 分叉后的 GP、知识和刷新内容也会不同；相邻差值属于完整策略的累计比较，不是与路径无关的模块因果效应。单 seed 无显著性结论。三个人工知识条件仍属于使用参考结果构造的压力测试。Pure BO 在四种知识表中引用同一条已有轨迹。

## 本地运行

从 `chemist_system` 目录运行；配置使用工作区上一级 `config.txt`，不记录密钥内容。

```powershell
.venv-official\Scripts\python.exe -m pytest tests/test_suzuki_grounded_regional.py tests/test_suzuki_regional_ablation.py -q --basetemp outputs/test_scratch_grounded_user --junitxml=research/suzuki_grounded_regional/component_tests.xml
.venv-official\Scripts\python.exe -m chem.benchmarks.suzuki_grounded_regional preflight
.venv-official\Scripts\python.exe -m chem.benchmarks.suzuki_grounded_regional qualify
.venv-official\Scripts\python.exe -m chem.benchmarks.suzuki_grounded_regional prepare
.venv-official\Scripts\python.exe -m chem.benchmarks.suzuki_grounded_regional run
.venv-official\Scripts\python.exe -m chem.benchmarks.suzuki_grounded_regional report
.venv-official\Scripts\python.exe -m chem.benchmarks.suzuki_grounded_regional verify
```

`run` 可恢复尚未完成的轮次；精确相同上下文的已验证响应可恢复使用，发现上下文不符则失败。不会用不匹配的历史提示词答案替换在线调用。完整的科学代码哈希在 prepare 时锁定，后续修改会拒绝继续原协议。

独立实验根目录 `data/runs/suzuki_grounded_regional/` 包含每条轨迹、按上下文哈希存储的请求/响应及独立缓存。`research/suzuki_grounded_regional/` 包含协议、测试、完整统一指标、累计差值、逐轮表格、图及校验收据。

## 查看单次选择

可在 `grounded_regional_suzuki_engine.setup` 依次查看基础分、预算候选、B4 保留集合、未缩放区域场、KF 预测、部署规则与最终排序。`grounded_regional_suzuki_study.step` 中结果读取发生在最终候选确定之后；该处之前没有隐藏产率。`original.update_bank` 前后可观察 KF 更新。

`rounds.csv` 按 Case、知识条件、方法、查询序号记录上述标量和选择依据；完整 4692 点评分保存在事件引用的缓存文件中。LLM `context.json` 与 `attemptXX/request.json` 是实际输入，`response.json` 是实际输出，`validated.json` 是事实校验结果。旧调试脚本仍服务于原 G0–G3，不会静默切换到新档位。

## 资源与复核

小样本 GP 使用 CPU；2 个独立 API/轨迹进程，每进程 1 个 BLAS/Torch 线程；资格检查与独立复核使用 4 个进程。算子在进入计数循环前预热，循环内 JIT 次数必须为零。按精确历史和模型配方复用缓存，不改变科学计算。当前任务不需要集群或 GPU。

复核覆盖每轮上下文、首先有效响应、事实数值、名义类别重标不变性、独立区域协方差场、独立 qLogEI 数值、B4 子集、幅度规则、独立 KF 代数、未重复查询、真实模拟产率、预算及原文件哈希。通过仅说明实现与实验完整性；是否改善搜索效率以完整结果表为准。
