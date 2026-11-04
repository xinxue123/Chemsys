# Chem 实验结果

这里保存用于阅读和复算的结果导出。方法源码在 [code](../code)，论文增补源稿在 [paper](../paper/v13_confirmation)。逐轮原始状态、模型请求与响应、GP 检查点和数据集仍由本地实验归档保存。

| 目录 | 内容 |
|---|---|
| [v13](v13) | V13 历史 4＋16 与 0＋20 指标，以及后半程同前缀消融和 Direct Arylation 迁移结果 |
| [kf_weight](kf_weight) | 程序化 KF 初次验证，Perera seed=2 和 HPLC seed=3 |
| [kf_confirmation](kf_confirmation) | 独立验证，Perera seed=1、HPLC seed=1、Direct Arylation seed=0，含逐轮目标值、配对差值和离线前缀结果 |

独立验证的动态 KF 相对固定权重 SearchAUC 变化分别为 0.00%、+2.21% 和 −0.37%。HPLC 的最佳值提前一轮出现，动态组的累计最佳曲线与封存纯 BO 一致。Direct Arylation 的最终最佳低 1.81 个产率百分点。27 个同前缀核验中只有一次改选，该轮没有提高累计最佳。预注册的跨任务确认门槛未达到。

SearchAUC 按查询后的累计最佳均值计算。两个 KF 研究都从零观测开始，每条轨迹 20 次查询。历史 V13 的 4＋16 结果有 4 条初始观测，SearchAUC 使用后续 16 轮并计入初始最佳。指标保留任务原单位。HPLC 的目标为检测器面积，反应任务使用各数据集记录的响应。

在仓库根目录执行以下命令可用已提交的逐轮值独立复算两个 KF 研究的 SearchAUC、最终最佳及首次达标轮次。

~~~powershell
python code/analysis/verify_exported_kf_results.py
~~~

完整历史重放依赖原始归档。入口、依赖与归档边界见[检出说明](../docs/GITHUB_EXPORT.md)。
