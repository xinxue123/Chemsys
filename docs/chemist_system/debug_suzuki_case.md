# 按阶段查看每一次Suzuki选择

入口：`scripts/debug_suzuki_case.py`。仍调用原始 `setup()`、`step()`、`update_bank()`；新增的是表格、页面、日志和断点。

## 推荐从一条轨迹开始

```powershell
Set-Location C:\Users\lihao\Documents\ChatGPT\Chemsys\chemist_system
.\.venv-official\Scripts\python.exe scripts/debug_suzuki_case.py --case 2 --regimes assistant_initial --methods G3_gate_KF_B4_event --pause kf round
```

每轮在实际KF更新前、选择与反馈全部完成后各停一次。输入 `c` 继续，`n` 逐行，`s` 进入函数，`q` 退出。终端首行打印 `DEBUG_OUTPUT`；在该目录双击 `index.html`，与调试器并排查看。页面无需服务器或联网，暂停期间点击“刷新已完成轮次”查看刚写出的结果。

若每个展示阶段也要暂停，把参数改为 `--pause stages kf round`。`--pdb-round 3` 将暂停限制在第3次追加查询。去掉 `--pause` 只输出、不自动停；IDE断点仍可用。`--rounds 4` 只截取前4轮，预算公式仍按16轮计算。

IDE直接运行时可修改脚本顶部 `DEFAULT_CASE`、`DEFAULT_REGIMES`、`DEFAULT_METHODS`、`DEFAULT_PAUSE`。解释器选择 `.venv-official/Scripts/python.exe`。

## 每轮的阶段与表格

| 阶段 | 文件 / 页面 | 主要内容 |
|---|---|---|
| 1 GPR | `01_GPR.csv` | 全部候选的物理条件、模拟真实产率、GP均值、潜在函数SD、含观测噪声SD、均值排名 |
| 2 基础qLogEI和门限 | `02_base_qLogEI_and_budget.csv` | 基础评分、损失、门限、是否进入预算集及判断依据 |
| 3 知识、B4和LLM | 页面“知识 / LLM / B4”及提示词文本 | 当前知识、保留/撤下原因、B4子集检查、事件触发、完整提示词、原响应和采用动作 |
| 4 区域修正 | `03_region_guidance.csv` | 原区域偏移、修正均值、α=1的qLogEI、相对基础排名变化 |
| 5 KF与最终排序 | `04_KF_ranking_and_choice.csv` | 本轮α、实际偏移、最终均值/评分、预算内排名、最终选中标记 |
| 6 最终选定 | 页面“5–6 KF / 最终选择”及日志 | 候选ID、三种评分的变化、基础损失≤门限、预算内排名第一 |
| 7 实验反馈与KF更新 | `05_KF_update.json`及“反馈 / KF更新” | 实际结果、残差、H、P、R、创新、S、K、Joseph更新及前后状态 |

每张候选表都有4692行和 `oracle_yield_percent_DEBUG_ONLY` 真实产率对照列。该列只加入观察层副本，不进入GP拟合、评分、筛选或LLM提示词。

`eligible` 只表示通过预算门限；最终看 `is_selected=true`、`guided_rank_eligible=1`。B4控制软注入区域，`matches_retained_region=false` 不自动排除候选。完全同分按候选ID字典序选择。

终端及 `run.log` 按上述顺序打印，每阶段只输出公式、阈值、候选数和前3名，再打印最终候选的均值/评分变化。`--top 0` 可关闭前3名预览，所有表格仍保留全部候选。

## 怎样查看和对比

- 左侧选择轨迹与追加轮次，再按阶段标签查看。
- 输入某个候选ID后切换阶段，查询会保留，方便对比同一点的变化。
- 用“仅预算候选”看门限结果；用“定位最终选中”看实际选择。
- 点击列头排序；页面显示6位小数，悬停看原始值，排序及CSV保留完整精度。
- 下载本阶段全部4692行CSV；分页只影响显示。
- “知识 / LLM / B4”可展开完整请求、原响应及子集检查；“反馈 / KF更新”提供公式和实际矩阵；“本轮日志”保留整个过程。

有事件刷新时，只展示一次不变的GP和基础预算集，再给出刷新前后知识/B4，以及真正用于查询的区域/KF排序。底层 `candidates/roundNN_phase*.npz/.json` 仍保留原计算阶段，`final_for_round=true` 表示最后采用的阶段。

展示阶段来自同一历史的原生计算输出，未另写决策算法。要进入GP拟合、后验或采集函数内部，加 `--recompute`；默认缓存命中会跳过部分内部计算。

## 提示词与响应

第一轮使用固定初始知识，没有新LLM调用，页面明确标为“本轮未调用”。未触发事件的轮次也标为沿用已有知识，不伪造提示词。

触发时，请求/回放之前先保存 `LLM_request_prompt.txt`。每次实际尝试另保存 `LLM_attempt01_prompt.txt` 和 `LLM_attempt01_response.txt`，格式修复尝试按02、03保留。`steps.json`包含动作、摘要、请求参数、B4和各阶段元数据。页面可完整展开，终端默认只打印摘要和精简理由；`--print-llm` 才把完整消息也打印出来。

默认 `--llm replay` 仅使用上下文、响应哈希精确匹配且重新校验合格的旧响应，不发送网络请求。要依据新历史真实请求模型，用 `--llm live`，读取父目录 `config.txt` 的 deepseek-flash 配置，会产生API用量，轨迹可能变化。密钥不写入调试输出。

## 建议的IDE断点

| 函数 | 可查看变量 |
|---|---|
| `debug_stage()` | `stage_name`、`candidate_table`、`stage_info` |
| `debug_llm_request()` | `llm_messages`、`request_context`；请求尚未执行 |
| `debug_kf_update()` | `kf_info`、`bank`、`prior`、`H`、`innovation`、`K`；原KF更新尚未执行 |
| `debug_round_complete()` | `state`、`candidate_table`、`round_info`、`llm`、`kf`、`output_directory`；本轮全部信息已写出 |

推荐在 `debug_kf_update()` 的 `H, innovation, K = ...`、`debug_round_complete()` 的 `stage_tables, selected_id = ...` 两行设断点。

```python
p candidate_table[candidate_table.candidate_id == 'p03631']
p candidate_table[candidate_table.eligible & (candidate_table.guided_rank_eligible <= 3)]
p kf_info['theta_prior']
p kf_info['P_prior']
p kf_info['R']
p kf_info['K']
p kf['bank_after_native']
p round_info['llm']
```

`candidate_table`在阶段/轮次断点可用，`kf_info`在KF更新前可用，`kf`和`round_info`在轮次完成断点可用。本轮选择使用更新前α，新反馈的α只影响未来相同区域；b不加入采集均值，KF参数方差也不额外加入GP方差。

## 一个Case的全部轨迹

```powershell
.\.venv-official\Scripts\python.exe scripts/debug_suzuki_case.py --case 2
```

默认4种知识条件×4个方案，共16条4＋16轨迹。输出写入新的 `outputs/suzuki_debug/` 子目录，正式数据和源文件不覆盖。完整Case的CSV较多，可加 `--no-stage-csv` 只保留页面表格，再按需下载CSV。

根目录看 `index.html`、`run.log`、`decision_summary.csv`；每轮表格、提示词和KF文件位于 `runs/<run_id>/roundNN/`。原生核心仍在 `regional_ablation_suzuki_engine.setup()` 和 `regional_ablation_suzuki_study.step()`，GP拟合在 `MixedKnowledgeCache.model()`，后验在 `UncertaintyCache.posterior_stats()`。

## 验证范围

11项Python测试通过；Case 2全部16条轨迹、256次选择重放保持一致；两个PDB暂停均实际执行；离线DOM测试覆盖分页、跨阶段搜索、数值排序、4692行CSV导出、轮次切换、完整LLM消息、KF和日志。内核与Cholesky循环外预热，循环内JIT编译0。

浏览器自动化工具禁止本地file协议预览，因此未作自动浏览器视觉检查；本机可双击页面打开。
