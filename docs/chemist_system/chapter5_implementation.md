# 第5章实现、复现与证据索引

[论文正文](chapter5_chem.md)包含方法、完整实验及负结果；[全部方法结果表](../research/chapter5/results_tables.md)保存各先验下八方法数值。[实验前方案](chapter5_research_design.md)和各 protocol_snapshot.json 保留历史设计，最新状态见[交付回执](../research/chapter5/validation_receipt.json)。

已完成3,755条测试、325条开发轨迹。数学473条与反应固定先验58条测试提前停止，均保留完整预算位置；实际助手75条全部完成。独立核验涵盖53,821次新增决策、68,841条查询及7,600次B4精确子集决策。固定先验SearchAUC未超过B3，实际助手也没有最终后悔值优势。

## 环境与资源

本地20逻辑CPU、31.84 GiB内存足够。BO使用四个独立CPU进程，每进程一个BLAS线程，复用特征、同版本谓词矩阵和相同观测前缀预测。模拟器单线程预热后一次缓存512点，无须GPU或集群。所有轨迹 operator_compiles_in_loop=0，无循环内JIT。

研究环境为Windows Python 3.10.6、NumPy 2.2.6、SciPy 1.15.3、scikit-learn 1.7.2。全新 .venv-reproduce 按46个版本锁重建，不继承系统包；依赖检查与62项测试通过，26个任务的直接sklearn前缀重拟合最大误差0。见[环境复现回执](../research/chapter5/environment_reproduction.json)。在 chemist_system 目录执行：

```powershell
python -m venv .venv-reproduce
.\.venv-reproduce\Scripts\python.exe -m pip install --no-deps -r research/chapter5/requirements-research-windows-py310.lock
.\.venv-reproduce\Scripts\python.exe -m pip check
.\.venv-reproduce\Scripts\python.exe -m pytest --junitxml=data/runs/chapter5/reproduction_tests.xml
```

Summit采用独立 .venv-summit，固定Summit 0.8.9、PyTorch 1.13.1+cpu、NumPy 1.23.5、scikit-learn 1.0.2。只安装预训练推断子集，排除无关旧优化器；常规 import summit 会触发额外策略，不能代替受限入口。

```powershell
python -m venv .venv-summit
.\.venv-summit\Scripts\python.exe -m pip install --no-deps --extra-index-url https://download.pytorch.org/whl/cpu -r research/chapter5/requirements-summit-inference-windows-py310.lock
.\.venv-summit\Scripts\python.exe -m chem.cli decision summit-cache --case 1
```

受限入口不修改官方模拟器源码，初始化训练轮数为0，禁止优化器更新与反向传播，加载后核对全部权重和缩放器。四个案例独立前向及公开API误差均小于预先固定的0.0001个百分点，资产SHA256见各 cache_receipt.json。

## 统一入口与复核

研究入口统一为 python -m chem.cli decision，旧 bo 命令属于历史流程。

```powershell
.\.venv-reproduce\Scripts\python.exe -m chem.cli decision mechanism -o research/chapter5/mechanism_receipt.json
.\.venv-reproduce\Scripts\python.exe -m chem.cli decision data-audit -o research/chapter5/material_data_audit.json
.\.venv-reproduce\Scripts\python.exe -m chem.cli decision material-demo -o research/chapter5/material_interface_demo.json
.\.venv-reproduce\Scripts\python.exe -m chem.cli decision report --suite-dir data/runs/chapter5/math_test -o research/chapter5/math_test_results.json
.\.venv-reproduce\Scripts\python.exe -m chem.cli decision report --suite-dir data/runs/chapter5/summit_test -o research/chapter5/summit_test_results.json
.\.venv-reproduce\Scripts\python.exe -m chem.cli decision report --suite-dir data/runs/chapter5/assistant_test -o research/chapter5/assistant_test_results.json
.\.venv-reproduce\Scripts\python.exe scripts/chapter5_figures.py
.\.venv-reproduce\Scripts\python.exe -m chem.cli decision verify -o research/chapter5/validation_receipt.json
```

report 不重新优化。独立评价器不调用原选点、谓词编译或评分函数，而是重建来源前缀、谓词、全部方法选择、穷举、停止与指标。重新审查会更新耗时和审查文件哈希，之后运行绘图入口更新图源回执；原始状态和回答不改写。最后的 verify 检查冻结、原始文件哈希、干净环境、文档链接和图件，并生成全证据文件清单。

## 固定先验运行顺序

已有manifest、冻结、观测和终态不被新设置覆盖，恢复只补缺失分组。另建实验使用 --suite-dir 和对应 --development-dir。

```powershell
.\.venv-reproduce\Scripts\python.exe -m chem.cli decision math-prepare --split development
.\.venv-reproduce\Scripts\python.exe -m chem.cli decision math-run --split development --workers 4
.\.venv-reproduce\Scripts\python.exe -m chem.cli decision math-prepare --split test
.\.venv-reproduce\Scripts\python.exe -m chem.cli decision math-freeze
.\.venv-reproduce\Scripts\python.exe -m chem.cli decision math-run --split test --workers 4
```

反应轨道先在 .venv-summit 逐案执行 summit-cache --case 1 至 --case 4，再在研究环境执行同序列 summit-prepare / summit-run / summit-freeze / summit-run。Case 1开发，Case 2—4测试。固定先验构造使用oracle安排压力条件，不调用LLM，不代表自然化学知识效能。

## 当前助手闭环与重放

用户指定当前助手充当LLM，配置别名gpt-6.1-sol、推理设置max。assistant_provider.json 是实验前身份快照，“已完成事件为0”属于当时状态；实际数目见[助手执行回执](../research/chapter5/assistant_execution_receipt.json)，不为更新计数改写冻结身份。

开发16个逻辑事件、12份不同上下文回答；测试240个逻辑事件、174份回答。两次测试来源引用错误保留原文，消耗额度并沿用上一有效版本。每轨迹最多4次，固定在新增查询0、4、8、12之后。

assistant-prepare / assistant-run / assistant-pending 按 --split 管理开发或测试。无回答时保存 WAITING_LLM，请求位于 requests/<context_sha256>/request.json。助手根据该上下文提交完整数据回答，通过 assistant-submit --answers-file ... 导入，再 assistant-run 继续；开发通过后必须 assistant-freeze 才能测试。启发式脚本不能冒充LLM。

assistant_test_c*_v*.json 等文件保存本次输出。最后一轮显式保留指令由助手逐上下文作出；转录工具只复制该请求上份助手谓词、更新版本和转录助手指定候选，不读取oracle、不计算排名、不生成新条件。保留指令、展开回答及转录哈希均存档。

所有回答来自共享聊天，缺少独立会话、不可变后端权重快照和逐事件token记账；逻辑事件和相同上下文复用不是独立API采样。存档回答可精确重放，新模型生成不能据此宣称完全复现。恢复终态不会重查oracle或重耗模型额度。

## 代码与证据

| 模块 | 职责 |
| --- | --- |
| chem/bo/llm.py | 固定完整输入GPR、批量预测、显式噪声和随机状态；旧流程限额 |
| chem/decision/hypothesis.py | 三类谓词、来源检查、版本、预算和恢复 |
| chem/decision/domain_guard.py | 八种规则、最多16子集、最少撤销与冲突回执 |
| chem/decision/replay.py | 公共任务、隐藏表查询、官方推断、材料输入审查 |
| chem/decision/evaluation.py | 共用顺序控制器、冻结输入、固定先验实验和检查点 |
| chem/decision/assistant_suite.py | 实际回答队列、身份冻结、提交与运行 |
| chem/decision/reporting.py | 独立全轨迹重算、任务配对与家族聚类统计 |
| chem/decision/material_adapter.py | 未合格材料来源返回HOLD和空推荐 |
| scripts/chapter5_figures.py | 从独立审查数据生成三图和完整数值表 |

62项测试含对篡改oracle、伪造撤销子集和覆盖旧回答的拒绝检查。26个GPR直接重拟合属于抽样数值核对，不宣称重拟合全部53,821轮。

材料主表16条，暂定数值非DSC候选12条；质量比、共同热历史和目标相未核实，旧方案I4/mmm与本地元数据P-3m1描述不一致。material-demo 保留全部16条审查及12个完整组成向量，返回 HOLD / needs_validation 和空推荐，实际查询0，不生成材料2+6效能曲线。

| 位置 | 证据 |
| --- | --- |
| research/chapter5/protocol.json | 不改写的实验前常量与范围 |
| research/chapter5/*_test_freeze.json | 各轨道测试前冻结 |
| research/chapter5/*_test_results.json | 全方法、任务差值、探索性区间与恢复分母 |
| research/chapter5/figures/ | 三图，每图PNG、PDF、SVG |
| data/runs/chapter5/<suite>/inputs/ | 冻结任务、先验、初始化、oracle |
| data/runs/chapter5/<suite>/trajectories/ | 全部原始状态、停止和结果 |
| data/runs/chapter5/<suite>/predictions/ | 按观测前缀寻址的完整GPR数组 |
| data/runs/chapter5/<suite>/independent_audit.json | 独立重算行、文件哈希与计数 |
| data/runs/chapter5/assistant_*/requests/ | 请求、实际展示和原始回答 |
| research/chapter5/math_io_recovery_receipt.json | 文件锁恢复的前缀和事务证据 |

运行产物和虚拟环境不进入Git；原始证据仍保存在上述本地目录。迁移时保留输入、轨迹、预测和回执，仅传递源码不等于完整实验交付，排除凭据和虚拟环境。
