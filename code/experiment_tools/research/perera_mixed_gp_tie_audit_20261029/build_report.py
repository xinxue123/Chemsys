"""Source-backed audit findings; no trajectory or oracle changes."""
import json
import math
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from chem.benchmarks import perera_gp_tie_audit as a


def main():
    assert Path.cwd().resolve()==PROJECT
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows={task:a.data.read(a.OUT/(task+'_rounds.json')) for task in a.data.SPECS}
    fit=a.data.read(a.OUT/'Perera_Suzuki_q3_fitting_probes.json')
    tails=a.data.read(a.OUT/'acquisition_tail_checks.json')
    result=a.data.read(a.OUT/'results.json')
    proposal={
        'status':'PROPOSED_NOT_FULL_TRAJECTORY_VALIDATED',
        'name':'health-triggered robust Mixed-GP fit, feature-based tie decisions',
        'trigger':{'base_top_tie_fraction_at_least':.9,'fraction_max_train_kernel_correlation_below_1e_12_at_least':.9},
        'trigger_uses':'only current GP and observed feature prefix; no oracle, task-name or candidate-ID priority',
        'observed_guard_queries':{task:{v:[r['query'] for r in rs if r['variant']==v and r['collapse_guard']] for v in a.VARIANTS} for task,rs in rows.items()},
        'fit_recovery':'Only when triggered, keep original fitted GP as one candidate and refit same native kernel with fixed alternative initial lengthscales0.05 and5. Select with identical observed-prefix MLL; require meaningful objective improvement and reject nonfinite fits. Cache once per prefix.',
        'fit_tolerance_proposal':'Total native log-MLL improvement>1e-3 before replacing; this is a development rule requiring prospective validation, not a tuned-yield optimum.',
        'remaining_ties':'Use nominal edit count to best observed anchor and posterior integrated variance-reduction/coverage, then isolated seeded random tie priority. No numeric differences between category codes. Residual exchangeability is acknowledged.',
        'preserve_Doyle':'Keep original random-tie V3 on all healthy prefixes; do not change qLogEI, prompt schema, refresh schedule, KF/B4, countercheck rounds2/4/6 or95-percent threshold. Guard false on all32 inspected Doyle prefixes. No claim on unobserved trajectories.',
        'LLM_budget':'No mandatory additional LLM call in the first fitting-only variant.',
        'not_automatic':'Do not globally replace the kernel, deploy an arbitrary tight lengthscale prior, replace qLogEI with analytic EI, alter initialization, add experiments or run seeds in this audit.',
        'next_validation':'One isolated robust-fit-only4+16 trajectory per dataset, same random tie policy and frozen controls; test extra tie geometry separately only if needed.'}
    a.data.write(a.OUT/'proposal.json',proposal)
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True)
    for i,(task,rs) in enumerate(rows.items()):
        axesnames=list(rs[0]['lengthscales'])
        colors=plt.cm.tab10.colors
        for j,v in enumerate(a.VARIANTS):
            seq=sorted([r for r in rs if r['variant']==v],key=lambda r:r['query']);x=[r['query'] for r in seq]
            for k,axis in enumerate(axesnames):
                label=axis.replace('_SMILES','') if j==0 else None
                axes[i,0].plot(x,[r['lengthscales'][axis] for r in seq],color=colors[k],ls='-' if j==0 else '--',lw=1.4,label=label)
            label='ID ties' if j==0 else 'Random ties';color='#8e64ad' if j==0 else '#176bad'
            axes[i,1].plot(x,[r['base_top_ties'] for r in seq],color=color,label=label+' / base',lw=1.5)
            axes[i,1].plot(x,[r['guided_top_ties'] for r in seq],color=color,ls=':',label=label+' / guided',lw=1.7)
        for j in range(2):
            ax=axes[i,j];ax.set_yscale('log');ax.axvline(3,color='#aaaaaa',ls=':',lw=1)
            ax.grid(alpha=.15);ax.set(xlim=(1,16),xticks=[1,3,6,10,16],xlabel='Additional query (first4 observed already)',title=task.replace('_',' '))
            ax.legend(fontsize=8,loc='best',ncol=2)
        axes[i,0].set_ylabel('Categorical ARD lengthscale (log scale)')
        axes[i,1].set_ylabel('Candidates sharing maximum score (log scale)')
    fig.suptitle('Frozen GP audit | solid lengthscales: ID trace; dashed: random trace',fontsize=12)
    fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(a.OUT/'diagnostics.png',dpi=180);plt.close(fig)
    text=[
        '**Perera大量GP同分：主要是拟合退化及类别核的可辨识性问题，未发现后验或qLogEI批量计算错误。**','',
        '审计范围：原V3和随机并列V3，Perera与Doyle共64条逐轮前缀记录（含共享前缀）；其中16项采集值检查。全部只读旧轨迹，将GP状态复制到新审计目录后计算。无新LLM调用，无新完整优化轨迹，不读取未查询候选产率。','',
        '**1. 第3次追加查询前发生了什么**','',
        '| 追加查询 | 已观测数 | 不同GP均值数量 | 不同基础qLogEI数量 | 最高基础分并列数 | 最高知识修正分并列数 |','|---:|---:|---:|---:|---:|---:|']
    seq=[r for r in rows['Perera_Suzuki'] if r['variant']==a.VARIANTS[1]]
    for q in [1,2,3,4,6,16]:
        r=next(r for r in seq if r['query']==q)
        text.append(f"| {q} | {r['n']} | {r['unique_mu']} | {r['unique_base']} | {r['base_top_ties']} | {r['guided_top_ties']} |")
    p=next(r for r in seq if r['query']==3)
    text += ['',
        '表为随机并列V3。原V3与它的前6条观测完全相同，所以第3轮用的是同一GP；第4轮之后两条历史已经分叉。','',
        '| 类别变量 | 第3轮ARD长度尺度 |','|---|---:|']
    for axis,value in p['lengthscales'].items():text.append(f'| {axis} | {value:.9g} |')
    text += ['',
        '本地安装为BoTorch0.14.0／GPyTorch1.14。五维全类别分支使用Scale(CategoricalKernel)，其精确公式为', '',
        r'\[ k(x,x_i)=s_f^2\exp\left[-\frac{1}{5}\sum_{j=1}^5\frac{\mathbf1[x_j\ne x_{ij}]}{\ell_j}\right]. \]', '',
        '它只比较类别是否相等，不把类别编号当连续距离。对编码做任意单射重标记，核保持不变；本次逐轮检查也验证了这一点。官方对应源码：[Mixed GP](https://botorch.readthedocs.io/en/v0.14.0/_modules/botorch/models/gp_regression_mixed.html)、[CategoricalKernel](https://botorch.readthedocs.io/en/v0.14.0/_modules/botorch/models/kernels/categorical.html)。','',
        '在第3轮，亲电试剂、配体或碱只要有一项不同，核相关因子就约为exp(−120.84)，即3.3×10⁻⁵³；核亲试剂不同的衰减更强。几乎所有新组合与已观测点失去联系。核矩阵没有被程序整列填零；极小相关项在双精度后验输出中被舍去。','',
        f"结果是3690个未查询候选中的3680个（99.729%）退回同一先验：均值{p['prior_mu_percent']:.4f}%，潜在标准差{p['prior_latent_SD_pp']:.4f}个百分点。对应标准GP公式中的k*近零，μ=c+k*ᵀ(K+σ²I)⁻¹(y−c)≈c，方差≈s_f²。训练K+noise条件数约{p['train_Knoise_condition']:.1f}，没有Cholesky病态证据。",'',
        '**2. 拟合层存在实质性稳健性缺陷**','',
        '前6条数据中，亲电试剂、配体和碱的“哪些实验取值相同”矩阵完全一致，三项ARD尺度无法被分别辨识。模型同时没有类别长度尺度先验（只注册了噪声先验）。单一默认优化初值可能落在相关性近零的局部解，优化器报告成功也不能排除这种情况。','',
        '仅使用同样6条观测，固定三种初值进行诊断重拟合，未接触未来真值：','',
        '| 拟合 | 训练数据log边际似然（越大越好） | 最高解析EI并列数 |','|---|---:|---:|',
        f"| 原缓存拟合 | {fit['cached']['data_log_marginal_likelihood_internal']:.6f} | {fit['cached']['analytic_top_ties']} |"]
    for r in fit['refits']:text.append(f"| 长度尺度初值{r['initial_lengthscale']:g} | {r['data_log_marginal_likelihood_internal']:.6f} | {r['analytic_top_ties']} |")
    text += ['',
        '初值5找到的数据log边际似然提高约2.9491，原生含噪声先验的每样本负MLL也从0.169369降至−0.321552。它仍是同一个Mixed GP、同一类别核，没有靠编号或产率挑点。这说明至少部分退化是拟合局部最优，而非数据强迫只能得到3680重并列。','',
        '但不能把“并列更少”当作更好的优化方法。初值0.05几乎没有改善训练似然，却把并列降到3；LogNormal(log0.2,0.75)长度尺度先验探针把并列降到1，但其训练数据log边际似然降至−7.6455。不同先验的MAP总目标不能与原MLL直接比较，也没有任何新完整轨迹证明它提升SearchAUC。','',
        'Doyle诊断中，初值5只获得约2.7×10⁻⁷的微小数据log似然改善，却使99.85%的候选与训练集近乎无关，并列从3增至38。不能对两个数据集无条件启用“换初值取数值最高”或收紧尺度。','',
        '**3. qLogEI和尺度转换检查**','',
        f"独立用NumPy核矩阵和线性代数重算64条前缀，标准化后验均值最大误差{result['max_manual_mu_error']:.2e}，方差误差{result['max_manual_variance_error']:.2e}。外层总体标准差归一化和BoTorch默认样本标准差变换都被正确逆变换；best_f映回真实观测最佳值也一致。两层归一化冗余，但未发现单位错误。",'',
        '16项采集检查中，缓存与重新计算、256与128分块结果均完全一致；使用相同Sobol正态样本手算边际MC与库输出误差小于3×10⁻¹¹。q=1的解析LogEI在全部16项检查中给出相同的最高分候选集合和预算集合。Perera第3轮解析EI仍是3680个最高分并列。因此增加MC样本数或改解析EI不能解决这里的主要同分。','',
        'qLogEI是平滑MC近似，并不保证与解析LogEI的所有数值相同：在极低EI尾部，有限采样及fat平滑使对数值差异很大；本次16项检查中的这些大差异都位于预算集外。预算集内最大对数差小于0.017，未改变本次检查的预算集和最高分集合。依据官方[logEI源码](https://botorch.readthedocs.io/en/v0.14.0/_modules/botorch/acquisition/logei.html)，这一差异应与计算错误分开解释。','',
        '另有一项精度债：best_f作为Python float传入后，库保存为float32，而模型为float64。显式传float64 Tensor可以修整这一点；它不是3680重并列的来源，若修改仍应单独验证排名、门限与提示变化，不与主改进混在一起。','',
        '**4. 与逐轮收益和损失的联系**','',
        '两条Perera轨迹的第3轮：相同模型的最高知识修正分有24个候选并列。编号规则选p00192（96.57%），随机规则选p00276（78.26%）；qLogEI完全相等。这轮数据分歧随后改变GP、KF创新和事件刷新。Perera本轮反例模块始终因初始化已达95%门槛而跳过，所以不能将该损失归于反例验证。','',
        'Doyle随机版第4轮反例得到78.253%，第6轮预留反例选择Additive_01达到98.2872%，第10轮正常查询达到98.7313%；最终SearchAUC92.9398仍高于Reasoning BO91.4567及纯BO87.4571。类别核对未见添加剂仍有交换对称性，但当前优势来自实际的反例探索路径，不应为修Perera而无条件重写它。','',
        '**5. 建议先做“健康门限触发的稳健拟合”**','',
        '第一个最小验证版本只改拟合选择：在一次选择前，同时满足“至少90%未查询候选并列最高基础qLogEI”和“至少90%候选与训练点最大核相关性小于10⁻¹²”时，才启动同一类别核的额外固定初值重拟合。保留原拟合，使用相同观测前缀MLL比较；改善小于预声明数值容差时保留原解，避免Doyle中微小似然差造成任意切换。拟合与诊断按历史前缀缓存，避免在一次决策中反复拟合。','',
        '这条诊断门限在64条记录中只触发Perera的第3轮（两条轨迹各一次），在32条Doyle记录中触发0次。它不读数据集名称或候选编号。若实现只在该门限触发时改拟合、健康分支严格复用原提示和随机流，已观察Doyle路径具备保持不变的条件；这仍须完整轨迹核验，不能当作已经跑出了新策略成绩。','',
        '第一版继续保留随机同分处理、qLogEI512、KF/B4、区域信任、95%门槛和第2/4/6轮反例机制，不加LLM调用、不增加实验次数。暂不全局换核、强加先验或换采集函数。','',
        '若稳健拟合后仍有大量正常选择并列，再单独验证与候选ID无关的次级选择：在同分集合中优先与已观测最佳锚点做较少类别修改（名义Hamming编辑数），然后按后验信息增益/未覆盖类别选择，最后仍无法区分时才用独立随机流。编辑数是变量是否改变，不是类别编号相减；用完整同分集合或公平的类别覆盖摘要，避免按ID裁出前5。该层应作为独立增量验证，避免与拟合修复混在一起归因。','',
        '单纯Hamming核无法从4–6条数据区分所有未见配体或添加剂。若仍受这一结构限制，再研究按分子指纹/化学描述符构建的类别相似核或加性＋交互核；这些会改变模型假设，目前没有其保留Doyle收益的证据。','',
        '90%等门限是本轮开发诊断规则，并非独立验证得到的普适阈值。下一次完整实验应先只验证稳健拟合这一项，仍用相同初始化、seed=0、4＋16及随机并列，不自动扩展多种子。','',
        '![核尺度与并列候选数](diagnostics.png)','',
        '机器证据：[逐轮Perera](Perera_Suzuki_rounds.json)、[逐轮Doyle](Doyle_Buchwald_Hartwig_rounds.json)、[拟合探针](Perera_Suzuki_q3_fitting_probes.json)、[采集尾部与预算集合检查](acquisition_tail_checks.json)、[改进方案](proposal.json)、[总核验](results.json)。执行入口为项目根目录audit_perera_gp_ties.py，额外采集敏感性入口为python -m chem.benchmarks.perera_gp_acquisition_tail。']
    (a.OUT/'report.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    preserved=protected=a.protected.preservation()
    assert preserved==result['protected_before']
    paths=[Path(a.__file__),PROJECT/'chem/benchmarks/perera_gp_acquisition_tail.py',PROJECT/'audit_perera_gp_ties.py',Path(__file__)]
    paths += [p for p in a.OUT.rglob('*') if p.is_file() and p.name not in ('release_inventory.json','release_receipt.json') and '__pycache__' not in p.parts]
    files=[{'path':str(p),'sha256':a.data.filehash(p)} for p in sorted(set(paths))]
    a.data.write(a.OUT/'release_inventory.json',{'files':files})
    a.data.write(a.OUT/'release_receipt.json',{'status':'AUDIT_COMPLETE_PROPOSAL_UNTESTED','round_records':64,'unique_fit_keys':len({r['fit_key'] for rs in rows.values() for r in rs}),
        'acquisition_checks':16,'diagnostic_refits':8,'new_optimization_trajectories':0,'LLM_calls':0,
        'protected_releases':preserved,'inventory_sha256':a.data.filehash(a.OUT/'release_inventory.json'),
        'visual_QA':'pending diagnostics.png inspection','operator_compiles_in_loop':0})
    print(json.dumps({'report':str(a.OUT/'report.md'),'status':'AUDIT_COMPLETE_PROPOSAL_UNTESTED'}))


if __name__=='__main__':main()
