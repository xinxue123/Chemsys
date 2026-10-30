# 第二实验入口

在 `chemist_system` 目录使用 **`.venv-official/Scripts/python.exe reasoning_bo.py`**。
这是当前支持入口，包含已记录的空测量记录兼容层；原 `reasoning_bo_comparison.py` 保留为冻结协议来源。

1. `prepare`：固定完整方案、作者提供的 Suzuki 3696 / Doyle 3955 实测池、10 个种子和所有源文件哈希。
2. `bootstrap`：每个数据集和种子调用官方初始化模板，锁定并共享 3 个初始实验；另外生成一份共同区域知识。
3. `run`：可续跑全部 140 条轨迹；`offline` / `online` 分开运行本地对照和真实 LLM 方法。`--limit 1` 只分阶段执行预定 manifest 的第一个任务种子，不修改全实验规模。
4. `status`：显示完成数量、API 用量及未完成轨迹。
5. `validate`：核验同时批次提交、查询预算、无重复、真实产率、基础预算、官方置信度排序和联合 KF。
6. `report`：生成两套数据分开的统一结果表、逐批产率和累计最佳曲线。未完成会明确标记。

默认读取工作区上层已授权的 `config.txt`，模型为 `deepseek-flash`。密钥不会进入结果目录。

每条轨迹为 **3 次初始化＋30 批×3＝93 次查询**。批内先共同提交 3 个点，再同时揭示结果。
全部类别使用 Mixed GP 的 CategoricalKernel；编码只作相等性标识，没有数值距离。

主要对照：ReasoningBO、BO_shared（相同 5 选 3 底座）、Gate_only、New_R2、LGBO_region。
另有 BO_q3_shared 与 BO_q3_Sobol，区分直接 q=3 的决策空间和初始化。
区域方法在此采用已登记的 5 选 3 批次扩展，不能直接当作上一实验逐点算法的原样复现。
ReasoningBO 保留官方提示及置信度排序；必要适配和与论文不同之处详见 `research/reasoning_bo_comparison/protocol.json`。

调试某一批可进入 `chem/benchmarks/reasoning_bo_study.py` 的 `run_one`：
`engine.setup` 后查看 10 个三元候选的基础/知识采集分数、门限和 B4；
`iteration_reasoning` / `regional_call` 查看真实请求与响应；
`commit_and_reveal` 前查看 3 个点的联合提交与 GP 预测；
`joint_kf` 前后检查矩阵、残差、增益和状态。

每批的独立证据位于 `data/runs/reasoning_bo_comparison/runs/<run_id>/batches/<批次>_committed.json`。
所有提示和可见回复位于 `data/runs/reasoning_bo_comparison/requests/<context_sha256>/`。
完整 GP checkpoint 按实际观察前缀缓存，6 个 CPU 工作进程并行；2 个 API 工作进程限流。
每个进程预热一次现成算子，不在循环中 JIT/torch.compile，不需要训练 GPU。

完整实验实际并行调度见 `scripts/dispatch_reasoning_bo.py` 与 `research/reasoning_bo_comparison/operational_concurrency.json`。
运行中发现未进行KF更新时日志中的 `measurement=None` 与事件检查的字典接口不兼容。
`chem/benchmarks/reasoning_bo_compat.py` 仅为事件检查提供只读视图，将None视为缺失创新值；保留原观测、知识、KF状态、采集数值和原始记录。
受影响轨迹使用 `scripts/resume_reasoning_bo.py` 原前缀续跑，详情见 `execution_compatibility.json`。
