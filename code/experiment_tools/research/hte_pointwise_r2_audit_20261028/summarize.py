"""Audited calculations, scientific figure and narrative for the frozen traces."""
from pathlib import Path
from hashlib import sha256
from collections import defaultdict
import json
import numpy as np

OUT=Path(__file__).resolve().parent
BASE=OUT.parents[1]
ROOT=BASE/'data/runs/hte_pointwise_seed0'


def read(p):return json.loads(p.read_text(encoding='utf8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def hashed(p):return sha256(p.read_bytes()).hexdigest()


def calculate(x,replays):
    result={'module_switches':{},'refreshes':{},'AUC_gap_decomposition':{},'KF_checks':0,'API':{}}
    for task,ds in x['datasets'].items():
        result['module_switches'][task]={}
        rr=replays[task]['rows'];actual=ds['methods']['New_R2']['rounds']
        pairs={'KF_given_B4_fallback':('actual_R2','B4_fallback_alpha1'),
            'conflict_fallback_given_KF':('actual_R2','B4_region_KF_before_fallback'),
            'B4_region_admission_given_alpha1':('B4_region_alpha1','full_region_alpha1')}
        for label,(left,right) in pairs.items():
            changed=[]
            for r in rr:
                a,b=r['variants'][left],r['variants'][right]
                if a['selected']!=b['selected']:
                    inc=r['incumbent_before_percent']
                    changed.append({'query':r['query'],'left_y':a['yield_percent'],'right_y':b['yield_percent'],
                        'delta_pp':a['yield_percent']-b['yield_percent'],
                        'immediate_incumbent_delta_pp':max(inc,a['yield_percent'])-max(inc,b['yield_percent'])})
            result['module_switches'][task][label]=changed
        refresh=[]
        for ev in ds['methods']['New_R2']['refreshes']:
            t=ev['query'];r=rr[t-1];before=ev['before_decision'];post=actual[t-1]
            a,b=r['variants']['pre_refresh_actual_R2'],r['variants']['actual_R2']
            refresh.append({'query':t,'same_admitted_region':before['bank_key']==post['bank_key'],
                'fallback_before':before['fallback'],'fallback_after':post['fallback'],
                'alpha_before':before['amplitude'],'alpha_after':post['amplitude'],
                'selected_before':a['selected'],'selected_after':b['selected'],
                'before_y':a['yield_percent'],'after_y':b['yield_percent'],
                'one_step_delta_pp':b['yield_percent']-a['yield_percent']})
        result['refreshes'][task]=refresh
        a=ds['methods']['Pure_BO']['rounds'];b=ds['methods']['ReasoningBO_point']['rounds']
        delta=[r['incumbent_percent']-s['incumbent_percent'] for s,r in zip(a,b)]
        gap=ds['methods']['ReasoningBO_point']['SearchAUC']-ds['methods']['Pure_BO']['SearchAUC']
        assert abs(np.mean(delta)-gap)<1e-10
        result['AUC_gap_decomposition'][task]={'delta':gap,'round_deltas_pp':delta,
            'early_positive_q7_to10_contribution':sum(delta[6:10])/16 if task.startswith('Doyle') else None,
            'top5_nearly_tied_rounds_1e_minus6':[r['query'] for r in b if np.ptp([p['base_qLogEI'] for p in r['base_top5']])<=1e-6]}
        # Independent Joseph equations and units, without calling the implementation.
        state=read(ROOT/'runs'/f'{task}_s0_New_R2'/'state.json')
        for d in state['decisions']:
            t=d['iteration'];p=ROOT/'runs'/state['run_id']/'rounds'/f'{t:02d}_committed.json'
            c=read(p)['decision'];m=d['measurement'];P=np.asarray(c['prior']['covariance']);theta=np.asarray(c['prior']['mean'])
            H=np.asarray(m['feature']);R=m['measurement_variance'];res=d['y']-m['prequery_mu_y']
            assert abs(res-m['residual'])<1e-14
            assert abs(R-m['prequery_predictive_sd_y']**2-.0009)<1e-14
            assert abs(H[1]-m['raw_region_shift_y'])<1e-14
            S=float(H@P@H+R);gain=P@H/S;innovation=res-H@theta
            expected_mean=theta+gain*innovation;A=np.eye(2)-np.outer(gain,H);expected_cov=A@P@A.T+R*np.outer(gain,gain)
            np.testing.assert_allclose(expected_mean,d['calibration_after']['mean'],atol=1e-12,rtol=0)
            np.testing.assert_allclose(expected_cov,d['calibration_after']['covariance'],atol=1e-12,rtol=0)
            assert abs(innovation/S**.5-m['innovation_z'])<1e-12;result['KF_checks']+=1
    for folder in (ROOT/'requests').iterdir():
        ctx=read(folder/'context.json');key=ctx['task']+'/'+ctx['phase']
        g=result['API'].setdefault(key,{'requests':0,'responses':0,'prompt_tokens':0,'completion_tokens':0})
        g['requests']+=1
        for p in folder.glob('attempt*/response.json'):
            r=read(p);g['responses']+=1
            for k in ('prompt_tokens','completion_tokens'):g[k]+=r['usage'].get(k,0)
    return result


def plot(x,replays):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(13,8))
    doyle=x['datasets']['Doyle_Buchwald_Hartwig'];perera=x['datasets']['Perera_Suzuki']
    ax=axs[0,0]
    for method,color,marker in [('Pure_BO','#444444','v'),('New_R2','#176bad','o'),('ReasoningBO_point','#d47a16','s')]:
        m=doyle['methods'][method];ax.plot(range(1,17),[r['incumbent_percent'] for r in m['rounds']],
            c=color,label=f"{method} (AUC {m['SearchAUC']:.2f})",marker=marker,ms=4)
    ax.set(xlim=(1,16),ylim=(0,105),title='Doyle: high yield reached four queries earlier',xlabel='Query after common initialization4',ylabel='Cumulative best yield (%)')
    ax.legend(fontsize=8,loc='lower right');ax.set_xticks([1,4,7,11,16]);ax.grid(alpha=.2)
    ax=axs[0,1];rows=[r for r in doyle['methods']['New_R2']['rounds'] if r['selected']!=r['reference']]
    pos=np.arange(len(rows));ax.bar(pos-.18,[r['reference_yield_percent'] for r in rows],.36,color='#555555',label='Own GP first choice')
    ax.bar(pos+.18,[r['yield_percent'] for r in rows],.36,color='#176bad',label='Actual R2 choice')
    ax.set(xticks=pos,xticklabels=[f"q{r['query']}\nrank {r['base_rank']}" for r in rows],ylim=(0,105),ylabel='One-step actual yield (%)',title='Doyle: all five R2 diversions reduce immediate yield')
    ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    ax=axs[1,0];rr=[r for r in replays['Perera_Suzuki']['rows'] if 'pre_refresh_actual_R2' in r['variants']]
    pos=np.arange(len(rr));pre=[r['variants']['pre_refresh_actual_R2']['yield_percent'] for r in rr]
    post=[r['variants']['actual_R2']['yield_percent'] for r in rr]
    for values,off,color,label in [(pre,-.18,'#777777','Before LLM refresh'),(post,.18,'#176bad','After LLM refresh')]:
        bars=ax.bar(pos+off,values,.36,color=color,label=label);ax.bar_label(bars,fmt='%.2f',fontsize=8,padding=2)
    ax.set(xticks=pos,xticklabels=[f"q{r['query']}" for r in rr],ylim=(0,113),ylabel='One-step UV product area (%)',title='Perera: same admitted region; fallback switches')
    ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    ax=axs[1,1];r=perera['methods']['ReasoningBO_point']['rounds'][1];rows=r['base_top5'];pos=np.arange(5)
    ax.bar(pos,[a['y_percent'] for a in rows],.65,color=['#d47a16' if p['id']==r['selected'] else '#a3a3a3' for p in rows])
    mu=rows[0]['mu_percent'];sd=rows[0]['sd_pp'];ax.axhline(mu,color='#176bad',label=f'GP mean {mu:.2f}% (nearly equal)')
    ax.axhspan(mu-sd,mu+sd,color='#176bad',alpha=.12,label='GP latent SD (not a trial CI)')
    ax.set(xticks=pos,xticklabels=[a['id'] for a in rows],ylim=(0,110),ylabel='UV product area (%)',title='Perera q2: agent selects99.05% among GP near-ties')
    ax.legend(fontsize=8,loc='lower right');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Frozen seed0 traces and same-history one-step diagnostics',fontsize=13)
    fig.tight_layout(rect=[0,.02,1,.95]);fig.savefig(OUT/'diagnostics.png',dpi=180);plt.close(fig)


def report(x,r):
    p=x['datasets']['Perera_Suzuki'];d=x['datasets']['Doyle_Buchwald_Hartwig']
    text=['**R2 的主要问题是知识信任与候选重排机制；本轮未发现新的数值执行错误。**','',
        '检查对象是已封存的 Perera / Doyle、seed=0、共同4次初始化＋16次查询。两条 R2 轨迹的32次选择均用冻结的同历史GP重放一致，GP均值与标准差同日志一致；滤波均值、Joseph协方差更新与单位独立重算通过。没有拟合新GP、调用LLM或运行新的优化轨迹。旧865个封存文件在分析前后哈希一致。','',
        '| 数据集 | 纯BO SearchAUC | R2 | Reasoning BO |','|---|---:|---:|---:|']
    for task,ds in x['datasets'].items():text.append(f"| {task} | {ds['methods']['Pure_BO']['SearchAUC']:.4f} | {ds['methods']['New_R2']['SearchAUC']:.4f} | {ds['methods']['ReasoningBO_point']['SearchAUC']:.4f} |")
    text+=['','**Doyle：错误的联合知识把早期搜索引向低收益区域。**','',
        '初始四条偏好知识的交集只有16个点，事后真实最大值为77.6669%；第4、6、8轮刷新后分别是16、12、18个点，最高值依次为77.6669%、77.6669%、81.2755%。各次完整偏好交集中均无90%以上的点。其中 Base_01/02 的建议排除了 Base_03，而随后81.28%、98.29%的关键组合使用Base_03；Additive_11/12 的建议也没有覆盖98.29%组合的Additive_01。','',
        '这些真值仅用于事后审核。R2的区域注入是软引导，候选没有被硬限制在上述交集中，因此区域最高值不是算法的性能上界；它说明了引导方向的质量。','',
        'R2偏离自身同历史BO首选的5次全部较差：','',
        '| 追加查询轮次 | R2真实产率/% | 同历史BO首选/% | 差值/百分点 | R2点的基础qLogEI名次 |','|---:|---:|---:|---:|---:|']
    for v in d['methods']['New_R2']['rounds']:
        if v['selected']!=v['reference']:text.append(f"| {v['query']} | {v['yield_percent']:.4f} | {v['reference_yield_percent']:.4f} | {v['one_step_delta_vs_reference_pp']:.4f} | {v['base_rank']} |")
    text+=['','第4轮预算集仍有3409/3948个可用点，R2把基础排名3338的点选出来；GP均值47.93%叠加26.33个百分点知识改变量得到约74.26%，实际仅23.04%。机会损失预算约束的是模型采集值，并不保证真实产率损失小。','',
        '**KF 的计算正确，但没有直接检验知识能否胜过 BO。**','',
        '滤波状态是整块区域共享的[b,a]，不是逐条知识的独立可信度。观测方程是 y−mu_GP=b+a·delta_region+noise；部署只使用clip(a,0,1)·delta_region，b作为干扰项不部署。滤波创新与最终部署预测误差因此不是同一指标。新几何默认a=1，决策采用后验均值，不因a的后验不确定性而主动打折。Doyle的16轮中出现6个几何首次校正状态，已有反馈不会自动转移到新几何。','',
        'Doyle第4轮后a从约1降为0.5235，第5–7轮的B4几何切换又使用新的校正状态；第8轮实际幅度0.9734，第9轮降至0.4407，但第2、3、4、8、9轮在同历史无KF时仍选择相同点。第9轮后a变负，第10轮才归零。滤波系数收缩没有及时转化为排名纠正。','',
        '在控制B4回退不变的单步回放中，KF只改变3次最终选择：Perera第13/14轮分别由52.95/48.62%变为92.06/65.80%；Doyle第13轮由80.80%变为62.56%。这些点均未超过各轮已见最佳，所以这些单步切换没有当轮累计最佳收益。不能据此推断完整移除KF后的轨迹完全相同。','',
        '另一个例子是Perera第6轮：b约−29.58个百分点，部署均值约88.75+21.72=110.47%，实际95.06%；滤波包含b的预测约80.89%，产生正创新，a反而升至1.0881（部署截到1）。这是当前观测模型与部署指标之间的设计取舍，不是Joseph公式算错；小创新或较大a不等于知识带来增量。','',
        '**B4 有保护作用，但刷新可在没有新实验时恢复同一块引导。**','',
        'B4诊断的是偏好集合与模型预算集的相容性，没有验证每条化学知识的真实性。Doyle第5/6/7轮，冲突回退选择75.12/81.28/74.59%，而关闭整体回退、保留同样KF与B4区域时会选33.91/31.16/31.16%；保护作用清楚。Perera第9轮回退则将88.37%改为64.55%，安全回退并不保证每步真实产率更好。','',
        '更关键的是刷新和回退的交互：','',
        '| 数据集 | 轮次 | 实际保留区域是否相同 | 刷新前幅度→后幅度 | 刷新前候选真值→后候选真值/% |','|---|---:|---|---|---|']
    for task,rows in r['refreshes'].items():
        for v in rows:text.append(f"| {task} | {v['query']} | {'相同' if v['same_admitted_region'] else '改变'} | {v['alpha_before']:.4f}→{v['alpha_after']:.4f} | {v['before_y']:.4f}→{v['after_y']:.4f} |")
    text+=['','Perera第7/11轮撤销H3，B4实际已将它排除；区域和观测未变，仅“还有被撤销条目”的标记消失，就重新开启引导。第9轮重新添加H3又关闭同一块区域。Doyle第8轮撤销H3也发生相同的解除回退。此处没有新的实验反馈来证明区域已恢复可靠；这是明确的策略缺口，而非本轮适配新增的算术错误。','',
        'Perera第6轮知识引导将同历史BO首选的100%改为95.06%；第7轮刷新又将原会选中的100%改为84.23%。第11轮刷新有局部收益17.81→54.52%，但没有提高96.57%的已见最佳。Doyle的3次刷新在第4/6轮没有改变当轮选择，第8轮从39.21%降至31.16%；全部额外调用在第8轮已经耗尽，后续误差无法再触发LLM。','',
        '**Reasoning BO 的优势集中在小候选集内的反馈利用和搜索路径。**','',
        '它不把整块知识加入GP均值，始终在基础BO的前5个候选内选1个，保留基础GP的均值与不确定性。R2只受机会损失预算约束，早期可能从几千个点中被区域场拉走。Reasoning BO能直接选择违背初始偏好的点，不必先走“撤销知识—解除B4冲突—更新区域”流程。','',
        'Perera第2轮是清晰例子：前5个点GP均值均约93.5218%、标准差均约11.4719%，采集值差距不到3e−9，真实值却为8.09、99.05、19.41、16.64、90.40%。Agent选了第2个点p02178（碘代物/另一类硼酸酯），基础BO首选p01254只有8.09%。可见解释针对底物反应性；这支持模型近并列候选中化学信息的价值，但不是对化学因果解释的独立验证。','',
        'Doyle第3轮选择初始知识排斥的Base_03，得到81.2755%；第6轮在GP打分完全并列的5个点中选Arylhalide_04，得到57.9775%。更新这些数据后，第7轮BO本身已把98.2872%的p01032放在首位，Agent也选择它。纯BO到第11轮才访问同一点。优势既包含前期重排，也包含后续BO从不同数据路径中受益，不能把第7轮高产全部解释为LLM推翻BO。','',
        'Doyle SearchAUC优势3.9996个百分点，精确分解为第7–10轮提前获得高产的+4.2529，减去第2轮的−0.2255和第16轮纯BO最终更高的−0.0278。Reasoning BO在Doyle偏离自身首选的10次仅5次即时产率更好、5次更差，其优势并非每轮都判断正确。','',
        'LLM计算预算也不同：每个数据集R2只有3次额外刷新，Reasoning BO有16次逐轮调用，另均共享1次初始化知识。实验查询预算相同，LLM调用量没有匹配。本轮不能把全部差异归因于推理算法本身，也不能由seed=0推出总体优势。','',
        '**下一步应优先验证的改进。**','',
        '先让同一保留区域的信任恢复依赖实际实验反馈，避免仅撤销条目就解除风险记录；再把知识重排范围限制在基础BO的小候选集中，并保留机会损失预算。将知识是否改善选点与KF残差拟合分别记录。分条可信度与刷新预算重分配可随后验证，本轮没有执行这些新实验。','',
        '![逐轮诊断](diagnostics.png)','',
        '证据与复现：[静态逐轮分析](analysis.json)、[同历史模块比较](mechanisms.json)、[Perera单步回放](Perera_Suzuki_replay.json)、[Doyle单步回放](Doyle_Buchwald_Hartwig_replay.json)。脚本依次运行analyze.py、replay.py、summarize.py；replay只复制冻结检查点到本审计cache，不重新拟合，不读取任何API配置。','',
        '源码位置：pointwise_hte_engine.py:97（选择与回退）、pointwise_hte_engine.py:140（事件调度）、regional_ablation_suzuki_engine.py:157/164（滤波预测/更新）、kalman_knowledge_suzuki_engine.py:155（事件门限）。','',
        '以上替代候选真值均为事后、同历史的一步审核，不能把各模块的局部差值相加当作完整轨迹AUC提升。Perera单位为UV色谱产物面积百分比，Doyle单位为报告反应产率百分比。']
    (OUT/'report.md').write_text('\n'.join(text)+'\n',encoding='utf8')


def main():
    x=read(OUT/'analysis.json');replays={t:read(OUT/(t+'_replay.json')) for t in x['datasets']}
    r=calculate(x,replays);save(OUT/'mechanisms.json',r);plot(x,replays);report(x,r)
    inv=read(BASE/'research/hte_pointwise_seed0/release_inventory.json')
    for row in inv['files']:assert hashed(BASE/row['path'])==row['sha256'],row['path']
    receipt={'status':'PASS_WITH_POLICY_LIMITATIONS','old_sealed_files_unchanged':len(inv['files']),
        'actual_R2_choices_replayed':32,'frozen_GP_fits_performed':0,'new_API_calls':0,
        'independent_KF_Joseph_checks':r['KF_checks'],
        'findings':'No new arithmetic/execution discrepancy found in the32 R2 choices; policy limitations and one-step losses documented.',
        'scope':'seed0, two datasets; one-step alternatives are not full causal ablations',
        'files':{p.name:hashed(p) for p in OUT.iterdir() if p.is_file() and p.name!='receipt.json'}}
    save(OUT/'receipt.json',receipt)
    print(json.dumps({'status':receipt['status'],'R2_choices':32,'KF_checks':r['KF_checks'],'API_calls':0,'GP_fits':0,
        'refreshes':r['refreshes'],'module_switches':r['module_switches']},indent=2))


if __name__=='__main__':main()
