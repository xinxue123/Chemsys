# Chem实验决策项目

GitHub检出说明见[代码包与实验归档](docs/GITHUB_EXPORT.md)。仓库只保存代码与必要文档；历史实验、环境和密钥留在本地，复现实验需另行恢复归档。

代码、实验记录和运行环境已分开。日常修改代码请进入`code/`，运行使用本目录的`run.ps1`。原`chemist_system`路径保留为兼容入口，用于读取封存协议和历史实验，不再作为另一份开发副本。

| 位置 | 内容 |
|---|---|
| `code/chemist_system/chem/benchmarks/` | 优化方法、GP、KF、知识检查与LLM接口；当前新方案为`continuous_r5_*` |
| `code/chemist_system/` | 原Python入口、包、测试、脚本及官方第三方算法源码 |
| `code/experiment_tools/research/` | 各阶段分析、导出、审计与TeX/PDF构建脚本，沿原研究目录分类 |
| `experiments/datasets/` | 固定候选集与离线实验数据 |
| `experiments/runs/` | 原生逐轮记录、候选评分、LLM公开提示与回复、GP缓存 |
| `experiments/reports/` | 对比CSV、图、PDF、协议、验证回执与历史归档 |
| `experiments/logs/` | 控制台日志、调试导出与排版中间文件 |
| `runtime/environments/` | 原5套Python环境 |
| `runtime/tools/`、`runtime/cache/` | RDKit、TeX工具与缓存 |
| `docs/chemist_system/` | 方法、实现和历史说明文档 |
| `runtime/layout_20261030/` | 本次迁移映射、验证和回退记录 |

## 常用入口

从项目根目录运行，不需要先进入实验目录或激活环境：

```powershell
.\run.ps1 list
.\run.ps1 r5 status
.\run.ps1 r5 inspect --task HPLC --round 14
.\run.ps1 p0 status
.\run.ps1 p1 status --variant feedback_t0
.\run.ps1 tests tests.test_continuous_r5 -v
```

`r5`为SnAr/HPLC局部版R5，`p0`为Suzuki四Case阶段修订版，`p1`为Perera/Doyle Stage11。它们是不同版本，不互相覆盖结果。其他旧入口也可以使用文件名，例如`run.ps1 continuous_r4 status`。

分析和报告脚本同样通过入口运行，参数使用原项目的相对路径：

```powershell
.\run.ps1 script scripts/debug_suzuki_case.py --help
```

运行前可用`run.ps1 list`查入口；报告构建和`run`、`prepare`、`report`、`seal`等动作仍具有原来的写入行为。`status`和`inspect`用于查看已有结果。本次目录整理没有重新优化或调用LLM。

配置仍使用根目录的`config.txt`，没有放入代码目录；主要快捷入口自动传递其绝对路径。其他支持配置的入口可以显式传入`--config`。Git已忽略配置、实验、环境和旧兼容视图。

## 如何定位R5代码与日志

一次选点的断点主要位于`code/chemist_system/chem/benchmarks/continuous_r5_study.py`的`plan`和`apply_feedback`。方向检查、准入、验证、调度和反馈在`continuous_r5_engine.py`；公开LLM输入在`continuous_r5_llm.py`。

对应原生记录在`experiments/runs/continuous_r5_20261030/runs/`；完整候选CSV与比较报告在`experiments/reports/continuous_r5_20261030/`；控制台日志在`experiments/logs/continuous_r5_run.log`。专家咨询稿在`experiments/reports/expert_consultation_20261030/`。

## 兼容路径与数据完整性

旧协议、调试脚本和封存清单包含原路径，因此保留`chemist_system`目录联接，以及归档中的源码硬链接兼容名。实验目录的四个便捷入口指向`experiments/chemist_system/`中的原归档结构，不复制数据。硬链接与`code/`的主文件共享同一份文件内容，不是复制的第二套源码；实验数据未放进`code/`。若编辑器以替换文件的方式保存，入口会在运行前重新绑定兼容名。所有科学哈希检查仍按原规则执行，修改算法后不会伪造旧协议通过。

请不要单独删除兼容链接或在归档兼容名下另外开发。冻结目录中的历史源码快照属于证据材料，保留原名字和内容。迁移前后源码哈希和已封存实验清单由`code/tools/verify_layout.py`验证；文件映射及回退信息保存在`runtime/layout_20261030/`。
