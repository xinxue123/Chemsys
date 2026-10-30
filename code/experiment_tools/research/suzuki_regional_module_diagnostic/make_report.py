"""Build the source-backed diagnostic narrative and timing figure."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager as fm
from matplotlib.lines import Line2D
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf8'))


def main():
    r=read(OUT/'results.json');v=read(OUT/'validation_receipt.json');assert v['status']=='PASS'
    data={c:read(OUT/f'case{c}.json') for c in (3,4)}
    fm.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':fm.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),
        'font.size':10,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(13,8),gridspec_kw={'height_ratios':[1.4,1]},sharex=True)
    for col,case in enumerate((3,4)):
        d=data[case];rows=d['rounds'];x=np.arange(1,17)
        ax=axes[0,col]
        ax.step(x,[q['new_incumbent_percent'] for q in rows],where='post',color='#24689B',lw=2)
        ax.step(x,[q['baseline_incumbent_percent'] for q in rows],where='post',color='#545D69',ls='--',lw=1.8)
        ax.set_title('Case 3：新方案 / 纯BO' if case==3 else 'Case 4：新方案 / LLINBO-Justify',loc='left',fontsize=12)
        ax.set_ylabel('累计最佳产率 / %（局部纵轴）')
        ax.set_ylim((76,92) if case==3 else (82,98));ax.grid(axis='y',color='#E3E8EE')
        for event in d['refresh']:
            t=event['round'];ax.axvline(t,color='#9FAFC0',ls=':',lw=1)
            ax.text(t+.08,91.1 if case==3 else 97.1,f'刷新{t}',fontsize=8,color='#6C7C8E')
        ax=axes[1,col];values=np.array([q['baseline_minus_new_AUC_contribution_pp'] for q in rows])
        ax.bar(x,values,color=['#B77B16' if val>=0 else '#24689B' for val in values],width=.68)
        ax.axhline(0,color='#6D7A88',lw=.9);ax.set_ylim(-.15,.7);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        ax.set_ylabel('当轮对AUC差的贡献 / pp');ax.set_xlabel('追加查询轮次（总实验次数＝轮次＋4）')
        ax.set_xticks([1,2,4,6,8,10,12,14,16]);ax.set_xlim(.5,16.6)
    fig.suptitle('差距在哪些轮次形成：完整G3的实际轨迹',fontsize=17,y=.98)
    fig.legend([Line2D([],[],color='#24689B',lw=2),Line2D([],[],color='#545D69',ls='--',lw=1.8)],
               ['新方案G3','对应基线'],loc='upper center',bbox_to_anchor=(.5,.94),ncol=2,frameon=False)
    fig.subplots_adjust(left=.08,right=.98,bottom=.14,top=.84,hspace=.18,wspace=.21)
    fig.text(.08,.025,'柱高＝（基线累计最佳−新方案累计最佳）/16；金色为基线领先，蓝色为新方案领先。可加总为AUC差，不能视作独立模块因果贡献。',fontsize=9,color='#607183')
    fig.savefig(OUT/'gap_timeline.png',dpi=180);plt.close(fig)

    text='''# 当前区域G3：Case 3与Case 4的逐轮损益诊断

对象是最新区域注入消融中的G3_gate_KF_B4_event，不是旧版加性残差KF方案。均为助手初始建议、Mixed GP、固定4692候选、seed0、4次初始化＋16次查询。Case 3对照Mixed纯BO，Case 4对照统一底座的LLINBO-Justify（qLogEI/预算门限适配版）。下文第r轮均指追加查询，总实验次数=r+4。

## 结论

可以继续改进，最优先的是具名特征输入及证据校验、B4撤销后的中性回退、区域更新后的保守校准。Case 3更像早期路径改变后GP排序偏移；Case 4有明确的第2轮B4动作损失。不能将所有损失归结为KF，也不能把错误LLM理由自动等同于已发生的产率损失。

本次复现32个真实历史前缀，计算257个确定性单步替换，共289条选点记录。所有替换先独立完成评分与选点，再查询冻结oracle评估结果。没有新GP拟合或LLM调用，没有生成替换后的未来轨迹。

## 整体差距及发生区间

| 案例 | 新方案AUC / 最终最佳 | 基线AUC / 最终最佳 | AUC差（基线−新方案） |
|---|---:|---:|---:|
'''
    for c,label in ((3,'纯BO'),(4,'LLINBO-Justify')):
        p=r['gap'][str(c)];text+=f"| Case {c} / {label} | {p['new']['AUC']:.6f} / {p['new']['final']:.6f}% | {p['baseline']['AUC']:.6f} / {p['baseline']['final']:.6f}% | {p['baseline_minus_new_AUC_pp']:.6f}pp |\n"
    text+='\n'
    for c in (3,4):
        for seg in r['gap'][str(c)]['segments']:
            text+=f"- Case {c}第{seg['first_round']}–{seg['last_round']}轮贡献{seg['AUC_gap_pp']:+.6f}pp，占该案例净差距{seg['gap_share_percent']:.2f}%。\n"
    text+='\n![差距时序]('+str((OUT/'gap_timeline.png').as_posix())+')\n'
    text+='''
## 已完成的完整轨迹消融

下面是各自反馈驱动的完整轨迹之间的有序条件增量，不是当前同一历史上的单步替换，也不是顺序无关的因果主效应。

| 比较 | Case 3 ΔAUC | Case 4 ΔAUC |
|---|---:|---:|
'''
    for key,label in (('initial_policy_difference','G0区域门限相对目标基线'),('KF','加KF：G1−G0'),
                      ('B4_given_KF','加B4：G2−G1'),('refresh_given_KF_B4','加刷新：G3−G2')):
        vals=[r['full_ladder_conditional_increments'][str(c)][key]['AUC_delta_pp'] for c in (3,4)]
        text+=f'| {label} | {vals[0]:+.6f} | {vals[1]:+.6f} |\n'
    text+='''
Case 3起点是G0相对纯BO，包含区域知识在预算内引导的整体作用。Case 4起点是G0相对LLINBO的系统差异，不能称作纯知识增量。KF和事件刷新在两个案例的这一加入顺序上均提高AUC；B4下降。刷新后的最终产率却分别比无刷新G2低1.275230、1.420052pp，说明更好的早期搜索进度与最终找到更高点不是同一个指标。

## Case 3：早期影响路径，晚期当前知识已经不改变选择

初始知识偏好P1-L3、较长停留时间、高温和较高loading，confidence=0.75。

| 追加轮次 | 机制和实际选择 | 同前缀单步对照 | 能确认的损益 |
|---|---|---|---|
| 1 | 初始区域知识选p02402，56.677% | 不加知识选p02266，67.922% | 两者均未超过初始化最佳77.593%；首次路径分叉，但当轮累计最佳差0 |
| 2 | α=0.728398，选p02200，71.242% | 完全关闭知识选p02302，79.959% | 当轮累计最佳少2.365990pp |
| 2 | 同一保留区域，使用KF | α固定1会选p02560，58.978% | KF提高当次产率12.264423pp，但两者累计最佳仍相同 |
| 4 | 刷新时间/loading区间，新几何α回1，选p02110，60.543% | 本轮不刷新选p02080，81.561% | 刷新造成当轮累计最佳损失3.968384pp |
| 4 | 检查内容变更与信任重置 | 旧区域即使α=1仍选81.561%；新区域沿用旧α=0.689646只得70.666% | 内容/几何变化是这一局部损失的主要证据，单独降低冷启动α不足以恢复改进 |
| 5–9 | 新方案累计最佳85.685% | 纯BO同期83.989–84.445% | 这段完整路径反而领先；不应只报告损失区间 |
| 6 | 撤销时间下界H2 | 本轮不刷新仍选p02587，84.775% | 没有本轮选点增量，原B4已处理了相应冲突 |
| 8 | 刷新后选p01547，79.793% | 不刷新选p04667，48.280% | 当次产率高31.513710pp；均未超过已有85.685%，即时累计最佳增量0 |
| 8–16 | 实际选择与当前GP基础qLogEI最大点完全一致 | 全部关闭知识仍选相同点 | 晚期不能归因为“当前知识持续强行选错”；早期采样已改变GP及后续路径 |
| 11、14–16 | 预算集只有1个候选 | 改知识、KF系数通常已无选点空间 | 调整晚期知识不能直接恢复漏掉的高值 |

第10–16轮形成96.14%的净AUC差：纯BO依次到88.186%、88.346%、89.645%，新方案停在85.685%。在新方案自己的GP下，纯BO最终高点p00953从第5轮起一直在预算集外；第2–4轮虽在集合内，修正排名分别171、48、87。它不是被某个最终硬知识条件直接排除，而是当前GP基础分数及门限已不支持。

撤掉B4在本条G3实际前缀上16轮均不改变选点；撤掉KF仅第2轮改变。完整消融中B4为−0.827562pp，发生在未刷新的G1/G2各自路径；G3从第4轮刷新后已走另一条路径。因此完整消融与局部零变化并不矛盾。

简单去掉门限也未修复：第11–16轮保持当前知识场、放开预算后均会选p02585，真实产率只有9.304%，没有累计最佳收益。当前证据不支持直接放宽门限作为主要修复。

## Case 4：第2轮B4损失明显；知识与KF也有应保留的收益

初始知识偏好P1-L3、高温、较高loading和非极短停留时间，confidence=0.7。

| 追加轮次 | 模块 / 实际结果 | 同前缀单步对照 | 实际−对照的即时累计最佳变化 |
|---|---|---|---:|
| 1 | 区域引导选p02292，86.034% | 纯GP选p02374，82.638% | +3.125648pp |
| 2 | B4撤下H1类别偏好，扩展区域，选p01486，85.023% | 保留完整知识选p01589，94.174% | −8.139740pp |
| 5 | 刷新移除类别/时间限制，新几何α=1，选p00457，27.123% | 本轮不刷新得42.273% | 0；两者均未超过86.034% |
| 6 | α=0.756096，选p01625，94.042% | 同区域α=1得89.870% | KF贡献+4.171989pp |
| 6 | 区域知识＋KF仍选94.042% | 纯GP选p02414，85.521% | +8.007645pp |
| 10 | 刷新改区域后α=1，选p00701，79.012% | 不刷新选p00598，94.890% | −0.847900pp |
| 10 | 同一实际前缀 | 完全关闭知识选p00906，95.117% | 知识引导整体少提升1.075798pp |
| 11 | 知识引导选94.726% | 纯GP选94.630% | +0.096329pp |
| 14 | α=0.510416，选93.576% | α固定1会选p00167，95.096% | KF造成−0.369484pp的当轮损失 |
| 15–16 | 预算集均只有1个候选 | 调整幅度没有选择空间 | 当前知识不能再改善选点 |

第2轮原历史与GP完全相同。取消B4但保持α=1，或恢复完整区域自身的已学α=0.964071，都选p01589；只把扩展区域的α降为0.964071仍选85.023%的p01486。因此这一轮的主要问题是B4扩展区域后重新构造软均值场，而不是3.6%的系数差。B4检验预算兼容性，不等于化学真伪；删掉类别条件会改变整个区域权重和lambda，可能加强其他候选的引导。

第5轮则不同：旧区域只把α改成1也选27.123%，新区域使用旧后验α=0.843338会回到42.273%。所以冷启动信任恢复足以解释此轮候选变化，但即时累计最佳未受影响。第10轮新旧区域α都为1，差异由知识/几何变更而非系数重置产生。

第1–5轮贡献68.79%的相对LLINBO净AUC差。完整轨迹中，B4加入造成−4.408957pp，刷新补回+1.276930pp，KF增加+0.129195pp；不能据此删除所有知识或KF，第1、6轮有明确局部收益。

### LLINBO的领先来自哪些分支

本条LLINBO轨迹16次提案接受4次、拒绝12次。两个关键高点来自拒绝LLM提案后的GP回退：

| 轮次 | LLM提案 | 基础分数损失 / 门限 | 实际GP回退 | 事后产率对照 |
|---|---|---:|---|---|
| 2 | p02484 | 1.403622 > 1.293875 | p00730，95.921% | 原提案87.198% |
| 13 | p01841 | 0.979254 > 0.277259 | p00946，96.516% | 原提案95.513% |

这支持借鉴“知识无效时明确退回基础BO”的机制，不能解释成LLM始终比GP更懂化学。两个系统在第1轮已经分叉，上表是LLINBO自己的历史，不能把其GP候选直接移植到G3并宣称会复现同样整轨结果。

## 已确认的变量对照错误，以及它们实际有没有影响

当前提示输入使用三值z数组；模型须结合fixed_slots与coordinate信息间接对应变量。动作槽固定能防止修改轴编号，不能保证理由中的数值与变量对应正确。

### Case 3第8轮刷新

| 观测索引 | 候选 | z_时间 | z_温度 | z_loading | 产率 |
|---|---|---:|---:|---:|---:|
| 7 | p02591 | 1 | 1 | 1 | 77.583900% |
| 10 | p02587 | 0 | 1 | 1 | 84.774712% |

LLM据此声称温度从1变0更好，提出低温[0,0.35]。实际变化的是停留时间600→60s，温度均为110°C、loading均为2.5mol%。B4本轮撤下该错误H4；移除B4仍选择同一点，因此不能把这条错误低温知识直接计为本条轨迹的选点损失。

同时将loading收窄至[0.85,1]只由少量高loading成功样本支持；事后看，P1-L1的高点p00613/p00953分别位于loading=0.542492/0.570240。现有知识表示缺少催化剂条件下的适用范围，容易把某类催化剂上的经验推广为所有类别通用。但第8轮起当前选点与纯GP相同，也不能把所有漏点直接归因于这个loading区间。

### Case 4第10轮刷新

以观测8为例，真实z是[时间0.0225，温度0.9337，loading0.6163]，产率84.432%。LLM却将0.0225当作温度，声称高产也出现在低温；还将其他观测的温度作为loading、loading作为停留时间，形成了循环错位的数值论据。更新撤销高温H2、收窄loading、添加时间下界。本轮不刷新能得到94.890%，刷新仅79.012%，所以这里同时具备“事实错误”和“局部决策损失”的证据。

这些是可程序核验的数据读取错误，区别于尚不确定的化学机理推理。应优先修复输入/证据契约。

## 下一轮改进顺序（尚未验证收益）

1. **具名特征与事实校验。** 观测及GP摘要改为带变量名、物理单位和归一化值的字典，避免裸z数组。要求动作引用结构化证据，如观测ID、变量名、实际值/阈值，并在接受前逐项对原数据核验；不一致则修复或保留旧知识。这保护输入事实，不认证化学因果关系。
2. **B4冲突后使用中性回退。** 暂停有争议的知识推力或回退基础qLogEI，避免“删掉一个约束→扩展并重新归一化区域→更强地推其他候选”。Case 4第2轮是明确检验点。化学知识记录与本轮是否注入分开，不能把预算冲突当作永久撤销的理由。
3. **新几何的校准保守起步。** 旧区域已有失败折扣时，LLM修改区域不应自动恢复完整强度。检验基于新区域不确定性/样本覆盖的渐进部署，同时保留初始知识α=1用于公平对照；不能盲目转移不同区域的校准状态。Case 4第5轮有局部冷启动证据，但Case 3第4轮只改α仍不能恢复收益，故内容校验优先。可同时记录已观察结果上的D_t=(y−μ_GP)²−(y−μ_GP−αd_R)²：正值表示知识改善本次实验前的预测，负值表示恶化。Case 3前3轮D_t均为负，随后刷新仍把新区域α恢复为1。D_t应作为有样本量/不确定性约束的在线证据，不能当作知识真值概率或直接替代完整轨迹搜索指标。
4. **只在知识会影响决策时花刷新预算。** Case 3第8轮后知识没有改变选择，第11、14–16轮候选集又是单点。可检验根据预算集大小、基础/引导候选是否不同和前瞻预测误差增量分配调用，保留早期探索机会。不能用事后oracle挑刷新时机，也不宜直接全程放松门限。

建议按上述顺序逐项测试完整轨迹，保留同一候选、初始化、Mixed GP和4＋16预算，记录正负结果。Case 3早期r2/r4、Case 4 r2/r5/r6/r10/r14是预先确定的调试观察点；不能只优化这些案例后声称已获得跨任务优势。

## 文件与证据边界

'''
    for name,desc in [('rounds.csv','32轮实际记录与AUC差贡献'),('variants.csv','32个实控＋257个单步替代，含即时产率与累计最佳差'),
                      ('targets.csv','事后选定的基线高点在G3前缀下的资格、排名与预测'),('ladder_metrics.csv','完整反馈轨迹的有序条件增量'),
                      ('LLINBO_case4_branches.csv','16次实际Justify接受/拒绝分支'),('coordinate_evidence.json','原观测三坐标及物理条件'),
                      ('validation_receipt.json','复算与旧文件保护记录')]:
        text+=f'- [{name}]({(OUT/name).as_posix()})：{desc}。\n'
    text+='\n固定oracle只用于选点之后的回顾评价及明确标注的残差/目标诊断，不反馈给知识提供者或排序过程。局部替代不展开未来，原始全轨迹未改写；单种子结果不证明统计显著或改进方案已获益。\n'
    (ROOT/'docs/suzuki_regional_module_diagnostic.md').write_text(text,encoding='utf8')
    print({'report':'docs/suzuki_regional_module_diagnostic.md','timeline':'gap_timeline.png'})


if __name__=='__main__':main()
