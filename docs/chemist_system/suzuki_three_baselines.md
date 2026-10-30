# Suzuki Case 1–4：三种知识辅助优化方案

**历史版本：本文件中的 LLINBO/LGBO 为自行实现的适配版本，不能作为官方源码结果引用。当前结果见 [官方源码三方法对照](suzuki_official_sources.md)。本文件保留旧协议、结果和 HEBO 精度核验证据。**

本轮仅比较 LLINBO-Justify、LGBO 和官方 HEBO。研究对象是 Summit 0.8.9 的四个预训练 Reizman–Suzuki 模拟案例，目标为产率；没有新增实验室反应，也没有扩展到其他数据集。与此前第5章的有限候选池实验分开保存。

## 实现边界

| 方法 | LLM 与 GPR 的分工 | 本轮实现 |
|---|---|---|
| LLINBO-Justify | LLM 提一个条件，GPR 用 UCB 差距接受或拒绝 | 拒绝时采用搜索到的 GP-UCB 最大值；容忍量随轮次衰减 |
| LGBO | LLM 给偏好区域和信心，区域知识平移 GP 均值 | 使用后验交叉协方差计算区域指数倾斜，保持协方差不变，再最大化单点 EI |
| HEBO | 不使用 LLM | 官方 HEBO 0.3.6，保留 GP、MACE 和混合变量 NSGA2 进化选点 |

[LLINBO 论文](https://arxiv.org/abs/2505.14756)包含三个变体，这里选择与“GPR 审查 LLM”对应的 Justify。采用[官方代码](https://github.com/UMDataScienceLab/LLM-in-the-Loop-BO/tree/105701dffee2999f31de1c0f42241d0e82b48e55)的数值约定：`beta_argument=2 log(t·4·π²/0.1·6)`，UCB 为 `μ+sqrt(beta_argument)σ`，`ψ_t=初始最大后验方差/t`。论文的理论参数形式与该公开实验代码不同，本轮不把代码参数解释成已验证的理论保证。

[LGBO 论文](https://arxiv.org/abs/2605.17976)的区域倾斜为 `μ'(X)=μ(X)+λ Cov_post(X,G)a`，`λ=c/sqrt(aᵀCov_post(G,G)a)`。本轮采用这一直接信心归一化，并设置 `1e-12` 数值下限；[公开代码](https://github.com/Xinzhe309/LGBO/tree/80baeff72758caa370a715e3ffb3a16afe7cab18)则使用 `guidance_scale·Φ⁻¹(c)` 和 Monte Carlo qLogEI。这里每次只选一个条件，使用解析 log EI，避免单点 EI 的蒙特卡洛误差。论文理论分析针对固定倾斜，不为本轮持续更新的 LLM 区域提供直接保证。

两种 LLM 方法共享 Matérn-5/2 ARD GPR，输入为三个归一化连续变量和八维完整催化剂 one-hot，目标在每次拟合中标准化。每轮用每类512个 Sobol 条件近似连续域采集函数最大值；LLINBO 额外加入实际 LLM 提议。LGBO 的区域积分网格有512个有效混合条件，各所选催化剂分配相同总权重。催化剂没有被编码成有序连续数，也不会产生无效 one-hot 插值。

[官方 HEBO](https://github.com/huawei-noah/HEBO/tree/77636146d2a2208d971e589062d4ec3d050029f0/HEBO)直接在连续混合域运行，不投影到有限候选池。GP每轮100次训练更新，进化种群100、100代。包内优化器、GP、进化器和模型工厂的文件与该官方版本逐字节一致，见[源码核对](../research/suzuki_baselines/hebo_source_comparison.json)。仅将随机初始化阈值固定为4，以接受三方法共同的四次初始实验。不同方法的采集函数搜索方式仍有差异，因此这是注明适配的本地比较，不是三篇论文的逐设置复现。

## 冻结设置与来源

每个 Case 使用种子0—4，每个种子三方法共享四次 Latin hypercube 初始实验和相同催化剂取样，然后各做16次新增查询，共60条轨迹、计划1,200次模拟器查询。硬域为停留时间60—600秒、温度30—110°C、催化剂用量0.5—2.5 mol%，以及八个官方类别。目标是预训练集成平均产率，不额外添加噪声。

两个 LLM 方法每轮都取得当前助手回答，计划共640条逻辑回答，按16轮批量提交；这不是640次独立 API 请求。原始回答、逐轮观测、精确上下文哈希和来源查询编号均留档；明确的“保留上轮区域”指令也由助手逐条提交，转录脚本只复制内容，不生成或排序条件。初始实验统一为随机设计，因此两种 LLM 方法没有使用原论文的 LLM warm-start。

用户指定当前助手充当 LLM。配置别名与推理设置记录为 `gpt-6.1-sol / max`，不可变后端快照和每次调用 token 计数不可取得。所有回答来自同一共享聊天，且此前已接触过 Suzuki 工作，因此本轮不是隔离会话的盲测；不同种子不代表独立 LLM 实例。请求文件只含该轨迹已揭示观测和定性文献资料，不能据此排除共享聊天或模型预训练产生的先验接触。

案例信息保留匿名催化剂标签，没有给这些标签臆造配体结构身份。四案比较能反映给定信息下的优化行为，不能验证具体配体机理。三方法的采集函数和部分代理模型也不同，因此性能差异不能单独归因于化学知识注入。

模拟器由原始权重导出为仅推断的 NumPy 前向。每案用原512点加64个新连续条件与官方接口核对，四案最大误差为0.0000153个产率百分点，小于预定0.0001。没有训练模拟器，也没有更新其参数。完整冻结设置见[protocol.json](../research/suzuki_baselines/protocol.json)，案例与种子清单见[manifest.json](../research/suzuki_baselines/manifest.json)。

## 指标与复现

主要指标 SearchAUC 是16个新增预算位置上 best-so-far 归一化产率的算术平均；表中乘100显示。次要指标为第20次查询后的最好产率，以及达到绝对产率80%/90%的轨迹数。每案报告五个种子的均值和标准差；配对差以同一 Case、同一种子计算。连续域全局最优值未知，不报告真实全局后悔值。失败轨迹保留在全部计划分母中，并将最后已知 incumbent 延续至预算末尾。

在 `chemist_system` 目录执行：

```powershell
python -m venv .venv-baselines
.\.venv-baselines\Scripts\python.exe -m pip install --only-binary=:all: --extra-index-url https://download.pytorch.org/whl/cpu -r research/suzuki_baselines/requirements-windows-py310.lock
.\.venv-baselines\Scripts\python.exe -m pip check
.\.venv-baselines\Scripts\python.exe -m pytest tests/test_suzuki_baselines.py
# 导出仅在已有 Summit 推断环境进行；已有导出后无需重复执行。
.\.venv-summit\Scripts\python.exe -m chem.benchmarks.suzuki export-oracles
.\.venv-baselines\Scripts\python.exe -m chem.benchmarks.suzuki prepare
.\.venv-baselines\Scripts\python.exe -m chem.benchmarks.suzuki run --workers 4
.\.venv-baselines\Scripts\python.exe -m chem.benchmarks.suzuki pending --compact
# 按 pending 上下文生成实际助手回答，通过 submit 导入后再 run。
.\.venv-baselines\Scripts\python.exe -m chem.benchmarks.suzuki submit --answers research/suzuki_baselines/assistant_round01.expanded.json
.\.venv-baselines\Scripts\python.exe -m chem.benchmarks.suzuki report --verify-selections
```

已有终态不会重新采样；保留请求和回答即可重放本次 LLM 数据，而不能保证重新调用当前助手得到完全相同回答。权重导出会生成新文件，已冻结实验必须沿用已有导出和 protocol，不能覆盖后继续旧轨迹。另建实验应使用新的 `--root`。

主实验每批最多四个独立 CPU 进程；精度核验单独使用两个进程，每进程一个 BLAS/torch 线程。依赖通过二进制 wheel 安装，worker 进入优化循环前预热；没有运行时 JIT 调用。这个规模无需 GPU 或集群。

## HEBO 数值核验

原始官方 HEBO 的320次选点全部逐轮重放一致。其中5轮出现 GP NaN 并触发官方随机预测回退，分布在 `c1_s2`、`c1_s3`、`c2_s3`、`c2_s4` 四条轨迹；285次训练更新被放弃，515次预测调用回退。原始60条主实验全部保留，不删除受影响轨迹。见[原生数值审查](../research/suzuki_baselines/hebo_numerical_audit.json)和[异常诊断](../research/suzuki_baselines/hebo_exception_diagnostic.json)。

在这五个相同观测前缀上，仅设置 torch 默认参数精度为 float64 即消除了回退，因此另外冻结[精度核验协议](../research/suzuki_baselines/hebo_precision_protocol.json)，补做20条相同案例、种子、初始化和预算的 HEBO 轨迹。官方源码、训练步数、核函数配置、MACE 与进化参数不变；每次实际预测的均值和方差必须为 float64。上游显式 `FloatTensor` 输入仍是 float32，所以本报告称其为“默认参数双精度修复”，不声称全流程都是 float64。

精度变化也会改变随机初始化的数值及后续轨迹，不能视为与单精度逐位等价。本核验是数值故障发现后的补充实验，不替换原先冻结的主对照；两个版本都会报告。修复版若仍出现放弃更新或随机预测回退，则该轨迹停止并保留分母。运行命令为 `python scripts/suzuki_hebo_precision.py`；原生逐轮重放命令为 `python scripts/suzuki_hebo_audit.py`，均使用 `.venv-baselines` 解释器。

## 结果

60/60条主轨迹全部完成，共1,200次查询；另20/20条默认参数双精度 HEBO 核验完成，共400次查询。640条助手逻辑回答全部按原上下文接收。所有优化查询均在硬域内，没有轨迹内重复查询。

| 方法 / 数值模式 | 四案平均 SearchAUC % | 平均最终最好产率 % |
|---|---:|---:|
| LLINBO-Justify | 65.77 | 75.47 |
| LGBO 区域倾斜 | 68.83 | 76.44 |
| HEBO 原生 | 62.64 | 69.73 |
| HEBO 默认参数双精度，补充核验 | 60.09 | 66.85 |

每行20条轨迹，四案等权；补充核验不并入原生 HEBO 的分母。LGBO 的 SearchAUC 均值在四案中均高于本轮另外两种原生实现，但最终最好产率不全面领先：Case 1、2 的 LLINBO 均值更高，Case 3 的 HEBO 均值更高。不能将这些差异单独归因为知识注入，因为采集函数、代理模型、连续采集函数搜索及共享 LLM 上下文也存在差异。五个种子不足以支持对新化学任务的普遍优势结论。

双精度修复消除了本轮训练放弃与随机预测回退，但没有提高平均产率或 SearchAUC。它还改变了随机初始化与后续选点路径，因此不能把两个数值模式的性能差异归因于回退的单一机制。

LLINBO 共接受20/320条 LLM 提议（6.25%），拒绝300条。LGBO 在95/320个观测前缀上改变了相对于同一前缀无倾斜 EI 的最大值选择（29.69%）；这是采集函数诊断，没有另跑一组无 LLM 优化轨迹，不能据此推断净产率收益。

独立数值重建覆盖640次 LLM 方法选点：639次点位完全一致，`c2_s2_LGBO` 第10轮存在一次机器精度并列，EI仅相差 `3.469446951953614e-18`。原始选择仍是64 ULP内等价的最大值，单独记录而不声称点位相同。最初的严格点位审查失败记录也保留，优化代码、阈值和轨迹没有为通过审查而改写。原生 HEBO 的320次选点全部逐点重放一致；精度修复版另外抽查四案各一个末轮前缀，均一致，这四次抽查不替代320次完整重放。独立标准库计算另行核对了80条轨迹的指标。70项回归检查及独立环境中的8项方法检查通过；这些检查支持实现与证据一致性，不验证真实实验安全性或 GP 置信区间的物理覆盖率。

完整数值见[四案结果表](../research/suzuki_baselines/results_tables.md)、[精度核验比较](../research/suzuki_baselines/hebo_precision_comparison.md)、[逐轨迹指标](../research/suzuki_baselines/results.json)、[学习曲线](../research/suzuki_baselines/figures/suzuki_learning_curves.png)和[交付回执](../research/suzuki_baselines/validation_receipt.json)。计算耗时只记录数值选点阶段，不包含助手书写和等待，也没有每次 token 成本。种子重采样区间属于固定四案和这些回答的探索性描述，不能视为独立 LLM 或新任务的置信区间。

最终证据检查可运行：

```powershell
.\.venv-baselines\Scripts\python.exe scripts/verify_suzuki_baselines.py
```
