# Suzuki Case 1–4：官方源码三方法对照

本研究按用户要求直接运行 LLINBO、LGBO 的官方算法文件，与官方 HEBO 统一初始化和查询预算。此前自行实现的两个方法保存在 `suzuki_baselines`，属于历史适配版本，不能作为本次官方源码结果引用。

## 来源与执行路径

| 方法 | 官方来源与固定版本 | 实际执行路径 |
|---|---|---|
| LLINBO | [UMDataScienceLab/LLM-in-the-Loop-BO](https://github.com/UMDataScienceLab/LLM-in-the-Loop-BO/tree/105701dffee2999f31de1c0f42241d0e82b48e55)，commit `105701dffee2999f31de1c0f42241d0e82b48e55` | `LLM_agent_BBFO.LLMIBO_BFO('justify').run()` → 原 `_run_justify` |
| LGBO | [Xinzhe309/LGBO](https://github.com/Xinzhe309/LGBO/tree/80baeff72758caa370a715e3ffb3a16afe7cab18)，commit `80baeff72758caa370a715e3ffb3a16afe7cab18` | 原 `lgbo_core.propose_lgbo_batch`，原提示构建与解析器 |
| HEBO | [huawei-noah/HEBO](https://github.com/huawei-noah/HEBO/tree/77636146d2a2208d971e589062d4ec3d050029f0)，安装包 `0.3.6` | 原 `HEBO.observe/suggest`，GP、MACE、100×100 演化 |

下载的 13 个官方文件保存在 `third_party/suzuki_official/`。每个文件均记录 Git blob SHA、SHA256、commit 和原始下载链接，见 [sources.json](../research/suzuki_official/sources.json)。文件按二进制原样保存，没有修改换行或算法正文。HEBO 的安装源码哈希也写入协议。

LLINBO 的 GP 由原 `helper_func.train_gp` 创建和拟合，UCB 由原 `optimize_acqf_ucb` 连续优化，20 次重启、200 个原始样本保持原值。初始最大后验方差使用原 10⁴ 网格；beta 和逐轮阈值、比较方向及分支条件完全由源码执行。没有替换为 sklearn GP、离散候选池搜索或重新编写的审查器。

LGBO 保留原 confidence→delta 映射 `3 Φ⁻¹(c)`、原均值/协方差计算、8192 个候选和 512 点区域网格。原区域分支使用 512 路 Sobol MC 的 qLogEI。原 `decide.py` 会把网格覆盖不足的小区域转为点先验；本次同样保留该分支及原 WeightedQLogEI，不能把所有步骤都描述为区域 MC 分支。未替换为解析 EI 或另一套论文公式。

## 仅做的接口适配

1. 通过用户指定的当前助手返回真实逐轮 LLM 回答；原提示与解析保留。没有用规则生成器冒充 LLM。当前助手无法提供不可变的后端模型快照、API token 账单或独立调用隔离，因此报告为共享聊天中的逻辑回答。
2. 三方法使用同一 case/seed 的四个物理初始点及其观测，替代 LLINBO 原来的 LLM warm-start。这是统一初始化所需的明确变更。
3. 在官方连续算法与 Suzuki 混合参数模拟器之间进行单位换算和催化剂标签解码。
4. 记录原算法选择、LLM 输入输出、数值分支、资源和随机数状态。LLINBO 暂停等待助手时缓存已执行的原生 GP/采集函数结果，并恢复对应 RNG；缓存不产生新建议。独立回放直接执行官方入口，不使用这些缓存。

两种上游接口要求连续参数。输入为 `[归一化停留时间, 归一化温度, 归一化负载量, 催化剂区间码]`，物理范围分别为 60–600 s、30–110 °C、0.5–2.5 mol%。固定标签顺序为 `P1-L1, P2-L1, P1-L2, P1-L3, P1-L4, P1-L5, P1-L6, P1-L7`。标签 k 对应 `[k/8,(k+1)/8)`，码 1 归最后一类；初始点和助手明确指定的标签使用区间中心。

原生提案的连续码原样保留到 GP 历史，只有模拟器入口解码标签。未把连续提案投影到类别中心，也未给官方 GP 增加 one-hot 核。代价是分段目标及任意标签顺序引入的距离偏差；本次不声称具有类别置换不变性或原生混合类别支持。HEBO 保留自身原生类别接口。因此这次是固定必要接口下的对照，不能把差异全部归因于 LLM 知识。

## 统一协议和证据边界

- 仅 Reizman Suzuki Case 1、2、3、4；每个 case 五个种子 0–4；三方法共 60 条预先列明的轨迹。
- 每条轨迹四次共同 LHS 初始化、16 次新增查询，共 20 次。重复物理条件保留并计费；没有因产率低、重复或不符合化学直觉而替换官方选择。
- 目标是固定 Summit 0.8.9 预训练反应模拟器的平均产率，按 100 归一化，无新增观测噪声。结果属于历史模拟器评估，不是 1200 次真实合成。
- 主指标为第 5–20 次查询后 incumbent 产率的算术平均，记作 SearchAUC；还报告最终最佳产率、绝对 80%/90% 达标数、重复数、失败和官方数值回退。没有已知连续全局最优，不能报告全局 regret。
- 若轨迹停止，保留其在 60 条计划中的分母，并把最后 incumbent 延续到预算终点；不能只统计有效轨迹。
- 当前聊天曾接触此前 Suzuki 结果；逐轨迹前缀约束和新回答留痕不能消除这种暴露。因此这不是盲评或独立 LLM 样本，不能用跨轨迹结果推断知识机制或推广到材料合成。
- 未调参后挑选赢家，未把旧手写方法结果与新结果合并，也未将 HEBO 的精度修复混入本次主对照。HEBO 保留原默认数值行为，并统计其回退。

初始协议哈希 `86cfbe06b59deda1d778f4828498dd21d903735d986864bdaa2a128c018bfcbe`。运行中适配层曾错误要求 LGBO 每步必须走区域构造器，导致三个合法小区域→点先验步骤在模拟器调用前停止。修订只放行该原生分支并保留跨修订的原始回答上下文，没有改动任何官方文件、科学参数或已消费回答，没有丢弃任何查询。修订后协议为 `7218d19f755d66bcaf86a5eb6e770481a5d092f42cb8cf5d2f8e8db31263badd`；旧协议、停止快照和修订记录见 [amendment_01](../research/suzuki_official/amendment_01/receipt.json)。这个适配层错误不能归为 LGBO 算法失败。

## 结果

60/60 条轨迹全部完成，三方法分别 400 次模拟器查询（每种方法 320 次新增查询），合计 1200 次；640 条逐轮助手回答全部消费。无提前停止轨迹。下表每个 case 为五种子的均值 ± 样本标准差，单位为产率百分点。

| SearchAUC | LLINBO 官方 | LGBO 官方 | HEBO 官方 |
|---|---:|---:|---:|
| Case 1 | 70.46 ± 6.49 | 65.55 ± 7.24 | 63.76 ± 7.74 |
| Case 2 | 24.70 ± 6.22 | 26.25 ± 9.89 | 21.34 ± 9.48 |
| Case 3 | 85.65 ± 3.41 | 87.06 ± 2.33 | 84.03 ± 4.24 |
| Case 4 | 91.50 ± 1.36 | 91.64 ± 1.64 | 87.66 ± 4.19 |
| 20 条等权平均 | **68.08** | **67.62** | **64.20** |

| 最终最佳产率 | LLINBO 官方 | LGBO 官方 | HEBO 官方 |
|---|---:|---:|---:|
| Case 1 | 76.62 ± 4.78 | 72.70 ± 2.35 | 74.74 ± 11.39 |
| Case 2 | 30.86 ± 8.59 | 31.66 ± 9.49 | 28.23 ± 10.73 |
| Case 3 | 90.57 ± 1.71 | 91.26 ± 1.46 | 89.70 ± 3.71 |
| Case 4 | 93.72 ± 1.65 | 95.24 ± 1.08 | 92.48 ± 4.71 |
| 20 条等权平均 | **72.94** | **72.72** | **71.29** |

| 审计指标（每方法 20 条轨迹） | LLINBO | LGBO | HEBO |
|---|---:|---:|---:|
| 最终产率 ≥80% 的轨迹 | 11/20 | 10/20 | 11/20 |
| 最终产率 ≥90% 的轨迹 | 7/20 | 9/20 | 7/20 |
| 重复物理查询，仍计费 | 16 | 17 | 0 |
| 预热后 script/trace 编译调用 | 0 | 0 | 0 |

LLINBO 原生审查仅采纳 8/320 条助手提案（2.5%），其余 312 次查询由原生 GP/UCB 分支决定；340 次原生拟合记录均为 BoTorch SingleTaskGP、RBFKernel（含各轨迹一次初始拟合）。LGBO 的 320 步实际分支为 region 39、region-soft 152、point 129。点分支是官方的小区域回退，未人为放大区域以规避它。

HEBO 有 8/20 条轨迹出现原生随机预测回退，共 824 次预测回退调用、460 次放弃的拟合 epoch。它们保留在主结果里，没有改 dtype、删掉坏步或补跑替换。涉及 `c1_s0, c1_s3, c2_s0, c2_s3, c2_s4, c3_s2, c3_s4, c4_s2`。这里的 824 是模型预测调用数，不能解释为 824 次实验查询。

本次均值下，LLINBO/LGBO 的 SearchAUC 分别比 HEBO 高 3.88/3.43 个百分点；Case 1 的主指标由 LLINBO 更高，Case 2–4 由 LGBO 更高。它们不是跨设置普遍胜出的证据：不同 acquisition 与类别表示、HEBO 数值回退、五种子规模及共享聊天暴露均限制归因。LLINBO 的大多数选择由 GP 决定，也不能将其提升全部解释为 LLM 化学知识的收益。上述结果适合第五章的官方基线对照小节，不能替代可靠知识来源与独立机制验证的证据。

![五种子平均最佳产率曲线，阴影为标准差](../research/suzuki_official/learning_curves.png)

独立标准库重算的 1200 个预算位置、960 个 SearchAUC 位置以及 90 个聚合均值/标准差检查均与报告一致，误差为 0。8 项新接口测试与 70 项原有回归测试分别在其配套环境通过。

60/60 条轨迹的直接官方入口回放全部通过，960/960 个新增查询逐点一致：LLINBO 320 个原始连续向量、原始提示及 20 条最终 RNG；LGBO 320 个原始连续向量及原生 plan；HEBO 320 个物理提案。回放未使用 LLINBO 的拟合/采集结果缓存，也未新增模拟器查询或 LLM 回答。13 个下载文件与 4 个 HEBO 核心文件的 Git blob 均与固定官方提交一致。

回放采用与主实验相同的导入/预热后设种子顺序。开发回放器时，曾因把导入放在设种子之后而造成 HEBO RNG 不一致，以及对暂停状态取错末尾 RNG 事件；纠正后的完整回放通过，早期审计失败保留在 `replay_preflight.json` 和日志中，主实验数据未因此改写。

## 本地资源、环境与复现

本机实测 20 个逻辑 CPU、31.84 GiB 内存，开始时约 12.96 GiB 可用。该任务可在本地 CPU 完成，无需集群/GPU或在线 LLM API 密钥。使用四个持久进程，每进程一个 BLAS/PyTorch 线程，HEBO 同时最多两条轨迹；等待助手期间不占用集群资源。

科学依赖隔离在 `.venv-official`：Python 3.10.6、torch 2.5.1+cpu、BoTorch 0.14.0、GPyTorch 1.14、NumPy 1.24.4、SciPy 1.12.0、HEBO 0.3.6。完整版本见 [requirements lock](../research/suzuki_official/requirements-windows-py310.lock)。上游依赖未完全锁定；本次验证的是固定兼容环境中的原样官方代码，不能声称逐位重现原论文的数值表。

算子使用预编译二进制；导入与矩阵预热在持久进程开始执行，科学循环使用 eager 执行。`operator_compiles_in_loop` 监测的是预热后 `torch.jit.script/trace` 的调用次数，不是对系统所有潜在编译器入口的追踪；GP 拟合是算法步骤，不应被计作编译。

```powershell
# 新环境，仅安装锁定依赖；不改原 .venv-reproduce
py -3.10 -m venv .venv-official
.venv-official\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r research/suzuki_official/requirements-windows-py310.lock
.venv-official\Scripts\python.exe -m chem.benchmarks.suzuki_official status

# 已完成记录的完整校验与官方源码回放，不产生新实验查询或新的 LLM 回答
.venv-official\Scripts\python.exe -m chem.benchmarks.suzuki_official report --replay-official
.venv-official\Scripts\python.exe -m scripts.check_suzuki_official_metrics
.venv-official\Scripts\python.exe -m chem.benchmarks.suzuki_official verify
```

重新使用新的助手回答开展新实验应创建独立运行目录和新协议；不要覆盖已消费的 `requests/*/response.json`。`serve --workers 4` 为现有轨迹的可恢复执行入口，完成的轨迹不会重新查询模拟器。

## 验证与文件索引

- [protocol.json](../research/suzuki_official/protocol.json)：60 条轨迹的固定预算、接口、来源、环境和哈希。
- [qualification.json](../research/suzuki_official/qualification.json)：原样来源、连续 oracle 对齐、LLINBO 直接/暂停运行历史和最终 RNG 对齐；固定回答仅是接口 fixture，不计科学结果。
- [results.json](../research/suzuki_official/results.json)、[run_metrics.csv](../research/suzuki_official/run_metrics.csv)：全分母结果和每条轨迹数据。
- [native_replay.json](../research/suzuki_official/native_replay.json)：直接执行官方入口的逐点回放证据。
- [independent_metrics.json](../research/suzuki_official/independent_metrics.json)：仅用 Python 标准库从原始历史重算指标。
- [validation_receipt.json](../research/suzuki_official/validation_receipt.json)：来源、上下文、预算、模拟器值及直接回放的最终核验。
- [release_inventory.json](../research/suzuki_official/release_inventory.json)：最终代码、文档、来源、原始上下文与结果的 SHA256 清单。
- `research/suzuki_official/assistant_round*.json`：当前助手手工逐条给出的条件、区域、置信度、理由和前缀引用；`.expanded.json` 只负责确定性换算和原解析器格式。
- `data/runs/suzuki_official/requests`、`runs`：完整原始提示/回答、每次查询、原生分支、错误与资源日志。
- 新增接口测试在 BO 环境执行；原有材料与决策回归测试在其原配套环境执行，避免为了可选材料依赖改变科学环境。首次在 BO 环境运行全项目测试的可选依赖错误保留在日志中。
