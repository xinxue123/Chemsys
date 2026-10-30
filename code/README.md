# 代码目录

`chemist_system/`保存主Python包、各版本入口、测试及官方第三方源码。`experiment_tools/`集中保存研究分析、审计、导出和报告构建脚本。`run.py`是统一启动器，`tools/`是本次目录迁移与核验工具。

实验结果、运行日志、候选数组和环境在项目根目录的`experiments/`与`runtime/`。推荐从根目录执行`run.ps1 r5 status`或`run.ps1 list`；详细路径与调试说明见根目录`README.md`。

源码保留原内容，封存协议仍校验原哈希；迁移没有改变算法、参数或已有实验结果。
