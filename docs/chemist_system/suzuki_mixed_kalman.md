# Mixed GP重跑卡尔曼知识校正方案

本实验将旧方案的四维连续RBF GP替换为BoTorch 0.14.0的 `MixedSingleTaskGP`，继续检验卡尔曼残差校正、事件触发LLM刷新和B4软知识引导。候选池、初始化、观测预算、LLM提示词和知识使用规则保持一致。

**12/12条轨迹已完成，类别标签数值不再影响GP预测。** Case 2–4中新方案SearchAUC为67.993，同Mixed GP纯BO为66.849；增量主要来自Case 4，另两案例略低。新方案最终产率均值略低于同底座纯BO，持续刷新知识也没有优于冻结初始知识，不能据此声称框架稳定胜出。

## 完整结果

主指标是第5–20次查询后的incumbent产率均值，SearchAUC越高越好。以下每格分母为3，即Case 2–4各一个seed0轨迹；未与旧五种子均值混算。

| 方法 | Case 2–4 SearchAUC | 最终最佳产率均值 |
|---|---:|---:|
| Mixed GP完整新方案 | 67.993 | 74.633% |
| Mixed GP冻结初始知识+KF/B4 | 68.058 | 74.633% |
| Mixed GP纯qLogEI BO | 66.849 | 74.748% |
| 旧编号GP完整新方案 | 65.301 | 69.674% |
| 旧编号GP纯qLogEI BO | 64.339 | 69.201% |

同Mixed GP底座，新方案AUC增量为+1.144个百分点，分解为Case 2 −0.140、Case 3 −0.046、Case 4 +3.617，1胜2负。最终产率均值差为−0.116个百分点。冻结知识的AUC反而高0.065个百分点，两者三案例最终最好值完全相同。

纯BO模型替换的对照没有LLM参与：Mixed GP相对旧编号GP的主要三案例AUC提高2.510个百分点，最终产率提高5.547个百分点，2胜1负。Case 2的AUC提高9.082，Case 3提高2.311，Case 4下降3.863，说明正确的类别处理也不意味着每案例性能都提高。

| 案例 | Mixed完整方案AUC | Mixed冻结知识AUC | Mixed纯BO AUC | 旧编号GP完整方案AUC | Mixed完整方案最终产率 |
|---|---:|---:|---:|---:|---:|
| Case 1，参考 | 72.329 | 72.329 | 72.798 | 55.754 | 89.569% |
| Case 2 | 27.517 | 27.517 | 27.656 | 16.878 | 40.085% |
| Case 3 | 85.239 | 85.864 | 85.285 | 84.648 | 88.186% |
| Case 4 | 91.223 | 90.792 | 87.606 | 94.377 | 95.626% |

全四案例平均AUC：Mixed完整方案69.077、Mixed冻结知识69.125、Mixed纯BO68.336、旧编号GP完整方案62.914、旧编号GP纯BO65.926。对应Mixed完整方案最终产率均值78.367%，Mixed纯BO为78.185%。主要三案例和全四案例都完整保留。

Case 2已从旧完整方案的最终23.143%提高到40.085%，但Mixed纯BO也达到40.085%，且AUC略高。因此本轮Case 2改善支持更换GP底座的价值，不能归因于知识模块。新方案相对旧完整框架的主要三案例AUC增量为+2.692，但这个跨系统差值还受新LLM回答和后端指纹差异影响。

旧自然知识系统在seed0主要三案例上的区域LGBO、硬B4、软B4、LLINBO-Matched AUC分别为72.499、66.609、72.297、71.369。它们没有换成Mixed GP，只能作为历史系统参考；完整同seed0比较表保存在results.json和metrics.csv。

![四案例固定预算曲线](C:/Users/lihao/Documents/ChatGPT/Chemsys/chemist_system/research/suzuki_mixed_kalman/comparison.png)

## 知识增量与调度诊断

仅在新完整方案自己访问的Case 2–4共48个追加点上，实验前预测RMSE为：原Mixed GP 16.693个百分点、仅整体偏移16.923、联合KF预测17.127、实际可信度/诊断/限幅修正后16.747。部署修正相对原GP的MSE增加1.790平方百分点，仍没有观察到预测增量。不同策略访问点不同，不把这些RMSE当作全候选池误差，也不直接据此比较新旧GP的全域预测准确度。

主要三案例48轮中，知识改变同前缀基础BO选择21次，B4冲突10轮，预算集仅有1个候选7轮，限幅18轮，实际机会预算违反为0。全四案例64轮中，知识改变25次、B4冲突14轮、单点预算集10轮。不能把这些执行计数直接解释为收益。

各案例实际LLM调用轮次：Case 1为1、4、6、12；Case 2为1、8、10、13；Case 3为1、6、8、10；Case 4为1、8、12、14。全四案例共有4次初始调用，8次触发原因包含停滞，6次包含预算冲突，部分原因同时出现；没有连续大创新触发。Case 2第13轮撤销了全部活动知识，此后执行基础BO；该案例仍没有获得相对Mixed纯BO的AUC优势。

Mixed GP改善了类别几何表示，原有知识校准和调度问题仍需独立检验。当前结果不支持把持续刷新作为已经证实的增益模块，也不支持把KF权重解释为可靠化学知识的正确概率。

## 执行与完整性

- 12/12轨迹完成，240次计划逻辑观测、192次新增选择；其中4条纯BO对照不产生LLM成本。
- 16轮逻辑LLM调用均取得合格知识回答。共18份API响应，其中2份因16384-token长度限制截断后自动修复；HTTP错误0，无人工改写或根据新产率重选回答。
- token共238713，prompt70958、completion167755；reasoning160813已包含在completion中，不重复相加。
- 返回模型均为deepseek-flash，指纹为 `bc19ba97cbe016479903222bb1578db6`。旧编号GP框架的指纹为 `e72d3c7f663d3e1383b00b3884cb4501`；相同模型别名不能保证后端快照相同。无LLM的纯BO对照不受这一项影响。
- 34项组件测试通过；正式192个前缀均对催化剂标签任意一一重命名，后验均值和标准差差异均为0。初始四案例的全候选基础与知识修正采集分数置换差异也为0。
- 独立复算1801728个qLogEI分数，最大差1.677e-10；192次选择和KF更新逐项核验，最大滤波代数差1.11e-16；16份LLM上下文核对只含各自已揭示前缀。
- 正式运行144次新GP拟合，精确前缀复用缓存，循环内JIT编译0。旧7620份受保护文件、所有旧缓存树及9份额外保护文件哈希不变，包括已交付Case 2 CSV；凭据泄漏0。

机器可读结果位于 `research/suzuki_mixed_kalman/results.json`、`metrics.csv`、`audit_cases.json`、`validation_receipt.json`；逐轮原始请求、响应、假设、滤波状态与选择记录位于 `data/runs/suzuki_mixed_kalman/`。本轮记录与旧编号GP实验分开保存。

## 离散变量的处理

GP输入为三个归一化连续变量和一个名义催化剂标签。连续核仅作用于停留时间、温度、催化剂负载量。催化剂标签只进入基于是否相同的 `CategoricalKernel`：

`k_cat(c,c') = exp(-I[c != c']/lengthscale_cat)`。

数字标签只是识别符，不计算标签之差、欧氏距离或大小关系。任意一一重命名标签都不改变GP的核矩阵。没有对催化剂编号做[0,1]缩放或连续优化。

标准Mixed GP核包括连续效应、类别效应和交互：

`Scale(RBF_cont1 + Scale(Categorical1)) + Scale(RBF_cont2 * Categorical2)`。

两个连续RBF核都只读取前3列。该类别核不提供分子结构相似性；未知催化剂之间的先验是可交换的，不能据此声称学到了配体结构规律。Stock Mixed GP同时引入额外幅度和核参数，这不是仅替换一个数值列的纯编码实验。

旧候选文件中的第4个z坐标仅作为旧知识解析器的标签兼容表示，通过集合成员判断解释类别，不用于新GP距离。实际GP类别列直接从候选的催化剂名称产生，模型缓存同时保存真实Mixed输入。

## 冻结协议

- 仅Case 1–4、固定seed=0、相同4692个候选和4次初始化，再执行16次新增查询；主要比较Case 2–4，Case 1作参考。
- 三个新方法：完整 `Mixed_KF_event_B4`、冻结初始知识但继续KF/B4的 `Mixed_KF_frozen_initial_B4`、不调用LLM的 `Mixed_qLogEI_B0`。共12条轨迹、240次逻辑模拟器观测，其中192次新增选择，不是真实合成次数。
- 真实LLM仍为 `config.txt` 的 `deepseek-flash`，与旧版同一系统提示词、thinking enabled/high和max_tokens=16384。每案例最多4次逻辑调用，初始调用计入额度。仅发送自己的已观测前缀和当前Mixed GP摘要。
- KF先验、过程噪声、3pp测量噪声地板、联合偏移、单标准差降权、0.5潜在标准差限幅、B4最大兼容子集与事件条件全部直接调用旧的不可变实现。
- 保留原512路Sobol MC-qLogEI、拟合/MC种子、标准化和协方差jitter=1e-6。预算仍为 `log(4)*(16-t)/15`。本轮只替换GP，不同时修改采集函数或门限；已知后期MC尾部敏感性和单点预算集仍属于限制。
- 完整Mixed GP轨迹全部冻结之后才运行两个新对照。冻结知识对照复用各案例第一份LLM回答，不产生额外API请求。纯BO对照区分模型更换收益和知识策略增量。

协议SHA256：`e5087d4145c23fef0c295c8a979c2488a1323f13bca65169cc5a57db282843eb`。新模型、接口、测试和共享策略源码哈希已在新实验前固定，不复用不兼容的旧编号GP模型和采集缓存。

## 验证与证据边界

34项组件检查通过，包括原有KF/B4组件和新的类别处理检查。四案例首轮实际Mixed拟合都将标签0..7重命名为 `[19,-3,101,7,0,12,-9,4]`，全4692候选的均值、标准差、基础与知识修正采集分数差均为0，选点一致。

正式轨迹后再对所有前缀验证标签置换不变性，不重新拟合或读取未查询产率；独立从边缘矩复算qLogEI并以另一代数形式回放滤波更新。旧实验、旧缓存树、Case 2已交付CSV和后续诊断文件以哈希保护。

同Mixed GP底座的完整方案对纯BO比较，才是本轮知识策略的主要增量证据。新旧完整框架之间的差异还包括GP改变后产生的新LLM假设、触发时机和采样轨迹，不属于固定动态知识的GP因果消融。旧区域LGBO、B4和LLINBO仍作为不同知识来源的历史系统参考。

每案例只有一个初始化，主要比较分母为3，不计算显著性或种子间方差。卡尔曼可信度仍为启发式，Mixed GP的额外参数在4个初始观测下仍有识别限制。

## 执行

工作目录 `chemist_system`，入口：

```powershell
.venv-official/Scripts/python.exe -m chem.benchmarks.suzuki_mixed_kalman qualify
.venv-official/Scripts/python.exe -m chem.benchmarks.suzuki_mixed_kalman prepare
.venv-official/Scripts/python.exe -m chem.benchmarks.suzuki_mixed_kalman run
.venv-official/Scripts/python.exe -m chem.benchmarks.suzuki_mixed_kalman controls
.venv-official/Scripts/python.exe -m chem.benchmarks.suzuki_mixed_kalman report
.venv-official/Scripts/python.exe -m chem.benchmarks.suzuki_mixed_kalman verify
.venv-official/Scripts/python.exe -m chem.benchmarks.suzuki_mixed_kalman release
```

两个单线程持久CPU进程；Mixed核与Cholesky在计数循环前预热，循环内JIT调用记录，模型按精确观测前缀缓存。该规模本地CPU即可完成。

方法依据：[官方MixedSingleTaskGP实现](https://botorch.readthedocs.io/en/v0.14.0/_modules/botorch/models/gp_regression_mixed.html)。
