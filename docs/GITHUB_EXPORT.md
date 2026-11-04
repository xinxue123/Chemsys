# GitHub代码包与本地实验归档

Git仓库包含代码、测试、研究分析脚本、必要说明和官方第三方组件。不包含本机`config.txt`、Python环境、实验日志、GP缓存、候选预测数组或完整封存归档。

## 新检出代码

当前依赖版本以Python 3.10的本地BO环境为参考。Windows示例：

```powershell
python -m venv runtime/environments/.venv-official
runtime/environments/.venv-official/Scripts/python.exe -m pip install -r requirements-bo.txt
.\run.ps1 list
.\run.ps1 r5 --help
.\run.ps1 script scripts/debug_suzuki_case.py --help
```

Linux可创建同名环境后执行`runtime/environments/.venv-official/bin/python code/run.py r5 --help`。首次实际启动时，入口会建立被Git忽略的兼容运行目录和源码硬链接；不会下载实验数据、创建密钥或调用LLM。

如需调用模型，把`config.example.txt`复制为根目录`config.txt`，填入自己的接口和密钥。`config.txt`已被Git忽略。当前Agent模型名沿用`deepseek-flash`。

这些命令仅验证接口，不代表新检出代码已经包含历史实验。`status`、`inspect`、历史回放和21项R5行为测试中的真实前缀检查，需要另行恢复原始候选池、逐轮状态、封存协议和回执。文件结构保持`experiments/chemist_system/data/`、`research/`与`outputs/`。迁移工具只用于已有Windows工作区；新检出代码无需运行目录迁移脚本。

Perera/Doyle的分子核分支还需要RDKit；Summit模拟器、HEBO及材料/XRD旧分支分别依赖其专用环境。这些环境不由基础BO依赖文件替代。完整历史结果依赖原环境与归档，不能仅凭安装成功宣称复现实验。

## 第三方来源

`code/chemist_system/third_party/`保留LLINBO、LGBO与Reasoning BO的已有源码组件、来源记录及上游提供的许可证。`code/resources/`保存启动时校验官方组件所需的来源清单，不包含实验轨迹。项目适配与原始算法的区别见相关实现和方法文档。

Reasoning BO上游的`scripts/experiment.py`独立启动脚本含通知凭据常量，不纳入Git上传；本项目实际调用的`src/`算法组件保留。本地私有归档及其来源清单不修改。来源清单描述此前取得的完整上游文件集合，因此不应把本代码包称为原始上游仓库的完整镜像；旧完整来源校验需要原始归档。

本次上传不改变本地已封存算法、参数或实验结果，也未新增实验与LLM调用。

## 研究结果和独立验证入口

[结果索引](../results/README.md)包含整理后的指标与程序化 KF 的逐轮目标值。[LaTeX 增补稿](../paper/v13_confirmation/confirmation.tex)与代码及结果分开保存。公开校验入口为：

~~~powershell
python code/analysis/verify_exported_kf_results.py
~~~

完整同前缀或在线回放入口为 code/experiment_tools/run_program_kf_weight.py 和 code/experiment_tools/run_program_kf_confirmation.py。这些入口使用原始冻结协议、候选池、模型响应和状态归档。results 只支持独立复算公开指标。原始归档和用于还原本地完整研究环境的旧稿数据仍需另行恢复。
