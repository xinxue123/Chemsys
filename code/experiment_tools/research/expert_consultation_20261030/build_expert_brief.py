"""Read-only synthesis of archived Chem experiments for expert consultation.

No model fitting, oracle calls or LLM requests. Builds TeX/PDF, a complete
results attachment and precise source custody in a new output directory.
"""
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
from hashlib import sha256
import csv
import json
import os
import re
import shutil
import subprocess
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
TYPESET=ROOT/'outputs/expert_consultation_20261030'
NAME='Chem_Expert_Consultation'
SOURCES={}
P0_NAMES={'Pure_BO':'纯BO','LGBO':'区域LGBO','Gate_only':'仅预算门限','B4_hard':'硬B4','B4_soft':'软B4＋区域',
    'LLINBO_Justify':'LLINBO-Justify','G3_original':'旧G3','R1_grounded':'旧R1','R2_conflict_fallback':'旧R2',
    'R3_conservative_region':'旧R3','Stable_revised':'P0修订版'}
P1_NAMES={'Pure_BO':'纯BO','LGBO_region':'区域LGBO','Gate_only':'仅预算门限','B4_hard':'硬B4','B4_soft':'软B4＋区域',
    'LLINBO_Justify':'LLINBO-Justify','ReasoningBO_point':'Reasoning BO','New_R2':'原逐点R2',
    'R2_region_trust':'信任恢复R2','feedback_t0':'P1修订版'}
P2_NAMES={'Pure_BO':'纯BO','LGBO_region':'区域LGBO','Gate_only':'仅预算门限','B4_hard':'硬B4','B4_soft':'软B4＋区域',
    'LLINBO_Justify':'LLINBO-Justify','ReasoningBO':'Reasoning BO','New_Agent':'原新方案',
    'New_Agent_R1':'R1','New_Agent_R2':'R2','New_Agent_R2_Subset':'R2子集恢复',
    'New_Agent_R3':'R3','New_Agent_R4':'R4','New_Agent_R5':'R5'}
REGIMES={'assistant_initial':'助手初始建议','correct':'正确知识','wrong_category':'错误类别','mixed':'混合错误'}


def sha(p):return sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    p=Path(p);p=p if p.is_absolute() else ROOT/p
    SOURCES[p.relative_to(ROOT).as_posix()]=sha(p)
    return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def fmt(v):return str(Decimal(str(v)).quantize(Decimal('.001'),rounding=ROUND_HALF_UP))
def esc(v):
    return ''.join({'&':r'\&','%':r'\%','_':r'\_','#':r'\#','{':r'\{','}':r'\}'}.get(c,c) for c in str(v))
def table(headers,rows,spec=None,small=True):
    spec=spec or ('l'+'r'*(len(headers)-1))
    out=[r'\begingroup'+(r'\small' if small else ''),r'\begin{tabularx}{\linewidth}{'+spec+'}',
         r'\toprule',' & '.join(esc(x) for x in headers)+r'\\',r'\midrule']
    for row in rows:
        cells=[esc(x) for x in row]
        if str(row[0]) in {'P0修订版','P1修订版','R5'}:cells=[r'\textbf{'+x+'}' for x in cells]
        out.append(' & '.join(cells)+r'\\')
    return '\n'.join(out+[r'\bottomrule',r'\end{tabularx}\endgroup'])
def section(title):return r'\section*{'+title+'}'
def note(text):return r'{\footnotesize '+esc(text)+'}'


def collect():
    previous=read('research/p0_p1_report_20261029/data.json')
    latest=read('research/continuous_r5_20261030/results.json')
    read('research/p0_p1_report_20261029/validation_receipt.json')
    for path in ['research/suzuki_stable_20261029/protocol.json','research/suzuki_stable_20261029/results.json',
        'research/stable_r2_20261029/feedback_t0/protocol.json','research/stable_r2_20261029/feedback_t0/results.json',
        'research/continuous_r5_20261030/protocol.json','research/continuous_r5_20261030/validation_receipt.json',
        'research/continuous_r4_diagnosis_20261030/findings.json']:
        read(path)
    stability=read('research/stable_r2_20261029/feedback_stability.json')
    rows=[];states={};init={};error=0.
    for stage in ['P0','P1','P1_development']:
        for r in previous[stage]:
            source=r['state_source'];state=states.setdefault(source,None)
            if state is None:state=states[source]=read(source)
            assert state['status']=='COMPLETE' and len(state['history'])==20 and len(state['decisions'])==16
            values=np.array([h['y']*100 for h in state['history']]);best=np.maximum.accumulate(values)
            err=max(abs(float(best[4:].mean())-r['SearchAUC']),abs(float(best[-1])-r['final_best_percent']))
            assert err<1e-8,(source,err);error=max(error,err)
            assert int(np.flatnonzero(values==best[-1])[0]+1)==r['first_final_best_experiment']
            group=(stage.replace('_development',''),r['task'])
            prefix=[(h['candidate_id'],h['y']) for h in state['history'][:4]]
            if group in init:assert init[group]==prefix
            else:init[group]=prefix
            unit='模拟产率/%' if stage=='P0' else 'UV产物峰面积/%' if r['task']=='Perera_Suzuki' else '来源表反应产率/%'
            rows.append(dict(stage=stage,task=r['task'],case=r.get('case',''),knowledge_condition=r.get('knowledge_condition','shared_initial'),
                method=r['method'],optimizer_seed=0,tie_seed=r.get('tie_seed'),SearchAUC=r['SearchAUC'],final_best=r['final_best_percent'],
                first_final_best_total=r['first_final_best_experiment'],initial_best=r['initial_best_percent'],unit=unit,state_source=source))
    pools={};calls=[]
    for task in ['SnAr','HPLC']:
        pools[task]=read(f'data/benchmarks/continuous_20261029/{task}/pool.json')
    for r in latest['metrics']:
        method=r['method'];task=r['task']
        folder='continuous_20261029' if method in list(P2_NAMES)[:8] else 'continuous_revised_20261030' if method=='New_Agent_R1' else 'continuous_r2_20261030' if method in {'New_Agent_R2','New_Agent_R2_Subset'} else 'continuous_'+method.replace('New_Agent_','').lower()+'_20261030'
        source=f'data/runs/{folder}/runs/{task}_{method}/state.json';state=read(source);states[source]=state
        assert state['status']=='COMPLETE' and len(state['history'])==20 and len(state['decisions'])==16
        values=np.array([h['raw_value'] for h in state['history']]);best=np.maximum.accumulate(values)
        err=max(abs(float(best[4:].mean())-r['SearchAUC']),abs(float(best[-1])-r['final_best']))
        assert err<1e-8,(source,err);error=max(error,err)
        assert int(np.flatnonzero(values==best[-1])[0]+1)==r['first_final_best_total']
        prefix=[(h['candidate_id'],h['raw_value']) for h in state['history'][:4]]
        if ('P2',task) in init:assert init[('P2',task)]==prefix
        else:init[('P2',task)]=prefix
        rows.append(dict(stage='P2',task=task,case='',knowledge_condition='shared_initial',method=method,optimizer_seed=0,tie_seed=0,
            SearchAUC=r['SearchAUC'],final_best=r['final_best'],first_final_best_total=r['first_final_best_total'],
            initial_best=r['initial_best'],unit=r['unit'],state_source=source))
        if method!='New_Agent_R5':continue
        pool=pools[task];ix={v:j for j,v in enumerate(pool['ids'])}
        for call in state['calls']:
            t=call['query'];base=Path(source).parent/'rounds';commit=read(base/f'{t:02d}_committed.json')
            context=read(f'data/runs/continuous_r5_20261030/requests/{call["context_sha256"]}/context.json')
            npz=ROOT/base/f'{t:02d}_candidates.npz';SOURCES[npz.relative_to(ROOT).as_posix()]=sha(npz)
            with np.load(npz) as a:
                E=int(a['eligible'].sum());menu=context['public']['candidate_menu']
                inside=sum(bool(a['eligible'][ix[v['candidate_id']]]) for v in menu)
            event=state['decisions'][t-1]
            calls.append({'task':task,'query':t,'menu_size':len(menu),'menu_inside_gate':inside,'eligible_count':E,
                'high_references':[r['index'] for r in context['public']['high_references']],
                'semantic_ok':commit['projection']['semantic_review']['admitted'],
                'GP_accepted':event['GP_review_accepted'],'GP_loss':event['GP_review_loss'],'epsilon':event['budget'],
                'executed_value':event['raw_value'],'mode':event['execution_mode']})
    assert len(rows)==232
    with (HERE/'experiment_results_complete.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (HERE/'R5_call_feasibility.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(calls[0]));writer.writeheader();writer.writerows(calls)
    snapshot={'as_of':'2026-10-30','rows':rows,'P0':previous['P0'],'P1':previous['P1'],'P1_development':previous['P1_development'],
        'P2':latest['metrics'],'R5':latest['new_metrics'],'R5_calls':calls,'P1_tie_stability':stability,
        'verification':{'metric_rows_recomputed':len(rows),'unique_state_files':len(states),'max_metric_error':error,
            'same_initialization_within_task':True,'new_experiments':0,'new_LLM_calls':0,'GP_fits':0}}
    write(HERE/'reviewed_snapshot.json',snapshot)
    return snapshot


HEADER=r'''\documentclass[UTF8,fontset=fandol,10pt,a4paper]{ctexart}
\usepackage{fontspec}
\setmainfont[BoldFont=texgyretermes-bold.otf,ItalicFont=texgyretermes-italic.otf]{texgyretermes-regular.otf}
\setmonofont{texgyretermes-regular.otf}
\setCJKmonofont{FandolSong-Regular.otf}
\usepackage[left=19mm,right=19mm,top=18mm,bottom=18mm,headheight=14pt,headsep=5mm]{geometry}
\usepackage{booktabs,tabularx,array,graphicx,fancyhdr,amsmath,amssymb,xcolor}
\usepackage{unicode-math}
\setmathfont{latinmodern-math.otf}
\usepackage[hidelinks,unicode]{hyperref}
\hypersetup{pdftitle={LLM辅助化学实验决策的当前方法与专家咨询问题}}
\newcolumntype{L}{>{\raggedright\arraybackslash}X}\newcolumntype{R}{>{\raggedleft\arraybackslash}X}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.12}
\linespread{1.08}\setlength{\emergencystretch}{2em}
\ctexset{section={format=\large\bfseries\heiti,beforeskip=1.2ex,afterskip=.6ex},subsection={format=\normalsize\bfseries\heiti,beforeskip=.9ex,afterskip=.4ex}}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small LLM辅助化学实验决策\quad 专家咨询材料}
\fancyhead[R]{\small 资料截至2026年10月30日}\fancyfoot[C]{\thepage}
\begin{document}
'''


def overview_figure():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    curves=read('research/continuous_r5_20261030/curves.json')
    methods={'Pure_BO':('Pure BO','#555555','-'),'LLINBO_Justify':('LLINBO-Justify','#2676ad','-'),
        'ReasoningBO':('Reasoning BO','#b57932','--'),'New_Agent_R4':('R4','#16836b','--'),'New_Agent_R5':('R5','#172b4d','-')}
    with plt.rc_context({'font.size':7,'axes.labelsize':7,'pdf.fonttype':42}):
        fig,axes=plt.subplots(1,2,figsize=(6.75,1.95),layout='constrained')
        for ax,task in zip(axes,['SnAr','HPLC']):
            for method,(label,color,line) in methods.items():
                ax.plot(range(1,21),curves[task][method]['best'],label=label,color=color,linestyle=line,
                    linewidth=1.8 if method=='New_Agent_R5' else 1.,marker='o' if method=='New_Agent_R5' else None,markersize=1.5)
            ax.set_title(task+' cumulative best');ax.set_xlabel('Total experiment (4 + 16)');ax.grid(alpha=.2);ax.set_xticks([1,4,8,12,16,20])
            ax.set_ylabel('STY (kg m^-3 h^-1)' if task=='SnAr' else 'Peak area (source units)')
            ax.axvspan(.5,4.5,color='#ddd',alpha=.18)
        handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=5,fontsize=6)
        fig.savefig(HERE/'continuous_best.pdf');plt.close(fig)


def document(d):
    text=[HEADER]
    def add(*parts):text.extend(parts)
    add(r'\begin{center}{\LARGE\bfseries\heiti LLM辅助化学实验决策}\\[5pt]{\large 当前方法 实验结果与专家咨询问题}\\[5pt]{\small 博士论文第5章研究讨论稿\quad 2026年10月30日}\end{center}')
    add(section('一 希望解决的问题与当前判断'),
        '本研究希望在只有少量实验数据时，让大语言模型提出可检验的化学假设，并通过实验反馈判断这些假设是否提供了高斯过程模型尚未掌握的信息。目标是在保留有用知识引导的同时，限制错误知识造成的搜索损失。使用现成LLM与GP，不训练新的语言模型；目前验证均为模拟器或历史实测数据回放，没有新增真实合成实验。',
        r'\textbf{当前结论：}整个决策链已实现并可回放，但还没有形成跨任务稳定优于纯BO或现有LLM辅助BO的统一策略。区域版在Perera和Doyle的已知任务上有收益；迁移到Suzuki后仍受错误知识影响。最新局部版R5保住了SnAr和HPLC的早期最佳，但SnAr低于R4，局部验证与均值部署均未激活。希望专家重点判断知识表示、证据准入和预算分配是否应调整，而非继续增加规则。')
    add(section('二 当前存在两条实现路线'),
        r'\textbf{区域路线（P0/P1阶段修订版）：}把LLM建议写成区间或类别集合；用LGBO式后验协方差形成区域均值引导，由KF校准幅度，B4检查多个偏好与采集机会是否兼容，并通过事件刷新和受控反例修订知识。',
        r'\textbf{局部路线（SnAr/HPLC的R5）：}先让LLM在没有GP分数的信息条件下，独立选择一个真实候选并声明局部对比假设；由GP审查能否执行。假设先进入影子验证，只有两个不同实验点的有效正证据才允许局部均值修正。R5新阶段没有继续执行旧版全局B4最大兼容子集求解，不能把它描述成原KF＋B4框架已经全面奏效。')
    add(section('三 R5怎样完成一次选择'),
        r'\begin{enumerate}\setlength{\itemsep}{3pt}',
        r'\item 读取本轨迹初始4点及后续实测数据，用原GP计算完整候选池的均值、潜在不确定性和预测不确定性；计算基础qLogEI、BO参考与预算集合。',
        r'\item 同步点级反证，检查旧局部假设是否有足够的新正证据；确认本轮是否存在真正可执行的验证点。',
        r'\item 事件与额度允许时调用deepseek-flash。提示只含自己的观测、原初知识、真实输入菜单、最多3个高值参照和旧假设声明，不含GP预测、排名、BO参考或门限。',
        r'\item 检查提议的实际变化方向与声明是否一致，检查GP预算。依次选择：已验证局部修正的严格改选、主动验证、GP接纳的独立提议、基础BO。',
        r'\item 先保存候选评分、假设、KF先验及选择承诺，再读取被选点的结果；结果进入GP，同时更新适用假设的影子证据和实际部署记录，供下一轮使用。',
        r'\end{enumerate}',
        note('“独立提议”指LLM没有收到GP信息。控制器可先计算GP可执行性，再决定是否调用LLM。LLM没有每轮给全候选池打分，最终实验也没有越出原固定候选池。'))

    add(r'\newpage',section('四 R5的数值模型与保护机制'),
        r'\subsection*{4.1 基础GP与采集预算}',
        '设本轨迹已观测数据为$(X,y)$。GP参数由这些数据的边际似然拟合，后验均值与潜在方差为：',
        r'\[\begin{aligned}\mu_t(x)&=c+k_{xX}(K_{XX}+\sigma_n^2I)^{-1}(y-c\mathbf1),\\s^2_f(x)&=k_{xx}-k_{xX}(K_{XX}+\sigma_n^2I)^{-1}k_{Xx}.\end{aligned}\]',
        '名义类别使用类别核或已声明的分子相似度核，不对任意类别编号作欧氏运算。SnAr和HPLC均为连续变量，采用原连续GP。预测方差还包含观测噪声。各方法的核与适配差异见实验说明。',
        r'令$A_0$为原$q=1$、512路Sobol MC的qLogEI实现，$U_t$为未查询点。正常决策约束为',
        r'\[\begin{gathered}x_{\rm BO}\in\arg\max_{x\in U_t}A_0(x),\qquad\epsilon_t=\log(4)\frac{16-t}{15},\\E_t=\{x\in U_t:A_0(x_{\rm BO})-A_0(x)\leq\epsilon_t\}.\end{gathered}\]',
        '这控制的是对数采集分数差，不能解释为对真实产率或后悔值的保证。R5所有分支都受原门限限制；第16轮门限为0。仅当分数完全相同时，才用既定信息代理、潜在SD、独立随机优先级裁决，不按候选编号大小选择。',
        r'\subsection*{4.2 局部假设与KF分别估计什么}',
        'LLM选择真实未查询候选，给出观测参照、1至2个聚焦轴、增减方向及该点的数值预测。局部仿射对比由程序构造，使用第32近邻的紧支撑核；方向与背景条件继续限制适用范围。非聚焦轴相对参照的归一化变化不得超过0.25。该对比不是已辨识的单因素因果斜率。',
        r'在工程归一化目标单位下，设局部预测为$\widehat f_h$、支持权重为$w_h$，则',
        r'\[\begin{aligned}H_h(x)&=\operatorname{clip}\{w_h(x)[\widehat f_h(x)-\mu_t(x)],-s_{\rm pred}(x),s_{\rm pred}(x)\},\\\widetilde\mu_t(x)&=\mu_t(x)+a_hH_h(x).\end{aligned}\]',
        '基础GP的拟合和协方差不改，只在准入后改采集时使用的均值；同一点至多由一个已准入模型负责。系数$a_h$表示预测修正幅度，并非化学知识为真的概率。影子银行与实际部署银行分开。',
        r'标量KF以$r=y-\mu_t(x)$为残差、$H=H_h(x)$为特征。$P$包含固定过程噪声，更新为',
        r'\[\begin{gathered}R=(s_{\rm pred}^2+0.0009)/v,\qquad K=\frac{PH}{PH^2+R},\\a^+=\operatorname{clip}[a+K(r-aH),0,a_{\max}],\qquad P^+=\frac{PR}{PH^2+R}.\end{gathered}\]',
        '初始幅度0.25、方差0.25，每轮过程方差0.0025；弱信号权重$v$由揭晓结果前的信号比确定。系数截断后仍使用原方差更新，因此这里是受约束的在线校正器，不是截断后验的精确计算。',
        r'\subsection*{4.3 影子证据与实际收益不能混为一谈}',
        r'准入使用冻结的$\delta=0.25H$探针，比较$\Delta SE=(y-\mu)^2-(y-\mu-\delta)^2$；信号比至少0.05才记为有效，弱正、弱负按$v$保留。最近6轮、最后4次自身检查中，须有两个不同实验点为正、加权评分为正且最近有效检查为正，才允许下一轮均值修正；两点上限0.25，三点上限0.5。',
        '实际KF只按真正部署的预测计算增益，负增益不得提高系数。暂停或本模型实际失败后需两个新的有效正点才能恢复。继承只传递适用点的负上下文，不能把追溯预测当成正验证。主动检验每轨迹最多3次，最多1次无正证据初检，确认最晚至第15轮，均占用原16次查询。')

    add(r'\newpage',section('五 当前遇到的主要问题'),
        r'\subsection*{5.1 知识可能正确 但当前数值代理未必有增量}',
        '“方向一致”只说明执行条件忠于声明，不能证明化学结论正确。少量混杂观测被转换成一个数值预测和局部仿射函数，这一步可能引入幅度、作用范围及交互错误。GP重拟合后，残差基准也会改变。R5的SnAr局部模型在第8、12、16轮有3次有效影子检查，全部使平方误差变大；HPLC第4轮的局部预测也被反驳。这证明对应冻结探针在这些点没有额外预测价值，不能据此证伪整个化学命题。',
        r'\subsection*{5.2 小支持集与收紧门限使验证很难发生}',
        '每个局部模型约束在小邻域内，而实验由正在变化的GP选择。R5在SnAr的12个新决策中只产生3次局部检查，HPLC的13个新决策产生5次检查，其中仅2次有效。两项均没有主动验证或局部均值部署；HPLC第14轮新模型虽在当轮BO点获得首次有效正证据，第15、16轮预算集均仅1点，不能再安排确认。',
        r'\subsection*{5.3 LLM独立性与可执行性存在冲突}',
        '真实菜单由输入几何和自身高值参照生成，保证不泄露GP，却未必覆盖预算集。SnAr第14轮有6个预算内候选，但发给LLM的33个点全部在门限外，因此该次调用无论如何选择都不能通过GP审查。HPLC第14轮45个展示点中仅1个在门限内。R5修复了“无可执行验证仍等待”的问题，但尚未解决“可以调用、菜单却没有可执行提议”的问题。',
        r'\subsection*{5.4 安全准入减少损失 也可能消除知识作用}',
        '8次R5提议中2次被GP接纳、6次超门限；SnAr唯一被接纳的第8轮提议与BO参考相同，新阶段实际全部由BO选点。HPLC仅第4轮改选，未创造新峰值。不能把这种结果称为知识有效参与了优化，也不能单凭0部署认定循环未执行。门限、方向限制、两点证据及时间窗口共同作用，需检验它们是否形成机制上的闭锁。',
        r'\subsection*{5.5 保护前缀与超越纯BO的目标可能不相容}',
        r'SnAr保护前4次追加查询后，即使剩余12次都立即达到候选池最高STY，SearchAUC的事后上限也只有\textbf{10227.031}，低于纯BO的\textbf{10373.540}。在这次受保护协议下，要求总AUC超过该纯BO在数学上已不可达。这个上限仅用于事后诊断，没有进入策略。应区分“保护既有早期收益的续跑”与“从共同初始化验证整体优越性”。',
        r'\subsection*{5.6 旧错误已修复 但收益归因仍不足}',
        'R4曾把连续提议投影成相反方向，等待不可能执行的验证，全局重叠阈值漏传点级反证，重复暂停又推迟恢复时钟。R5已分别修复；8次实际提议语义均一致，后期第11、14轮均有调用。但本轮无跨模型适用负证据继承、无暂停及恢复事件，相关行为测试通过不能当作性能证据。P0/P1还存在核恢复、区域协方差单位修复和显式门限例外，现有总分不足以分别归因于KF、B4或刷新。')

    add(r'\newpage',section('六 希望专家重点帮助判断的问题'),
        r'\subsection*{6.1 知识应该以什么对象进入优化}',
        '当前把语言声明转成数值局部均值，是否要求LLM承担了不适合它的定量校准任务？是否更适合只提出候选、局部排序、可证伪对比或交互结构，再由数值模型学习幅度？希望明确最适合本研究的小数据知识表示，以及哪些部分可以成为方法贡献。',
        r'\subsection*{6.2 如何给验证机会 而不是等待偶然命中}',
        '两次不同实验点的前瞻正证据与不断缩小的采集预算是否过于矛盾？是否应预先划分少量检验预算，或采用能体现信息价值的统一采集目标？若允许检验越过正常门限，怎样定义并报告风险和代价？第16轮单点门限是否适合知识修订实验？',
        r'\subsection*{6.3 是否应向LLM展示可执行集合}',
        '完全屏蔽GP能检验独立知识，但会出现菜单与预算集不相交。若仅展示GP筛出的可执行条件、隐藏预测值与排序，应如何描述信息独立性？应怎样与“全菜单独立提议后审查”“只在BO候选中排序”进行公平消融，而非把可达空间变化误判为推理能力提升？',
        r'\subsection*{6.4 KF是否是合适的统计对象}',
        '残差基准随GP拟合变化，采样自适应，多个假设又会共享同一观测。现用线性KF、噪声地板、弱信号加权和系数截断是否有充分解释？是否需要联合残差模型、动态模型平均或在线预测比较？固定探针的正证据能否合理授权不同幅度的部署，应怎样校准并避免重复使用证据？',
        r'\subsection*{6.5 B4应保留在哪个层面}',
        '区域版B4检验的是多条偏好与采集预算是否兼容，不等同于化学冲突。R5后半段已变成局部模型选择，尚未沿用全局B4。希望判断应恢复哪种联合诊断、怎样定义真正有意义的冲突，还是应删减模块使假设更清楚。',
        r'\subsection*{6.6 如何建立有说服力的最终实验}',
        'Perera/Doyle经过多轮开发，P0也是已查看过的迁移任务；P1三组并列顺序使用相同初始化与可复用回复。希望确定独立初始化、LLM重复与新任务的验证规模，是否固定底物后再优化条件，以及如何统一GP核、提示信息、候选可达空间与调用成本。应怎样同时报告搜索效率、最终最优值、错误知识损失和知识真正改变决策的频率？',
        r'\subsection*{建议优先讨论的最小验证方案}',
        '先确定一种知识对象和同一个GP底座，再冻结：纯BO、LLM提议＋GP审查、加入主动检验、加入证据准入、再加入KF、刷新和继承的逐项消融。各分支使用同一候选池、初始化及总查询预算，明确哪些调用与预算例外被允许。先证明最小机制带来增量，再增加复杂度；具体方案等待专家建议，本次未运行新实验。',
        note('希望专家最终给出三项明确意见：研究主线应保留什么；最需要修改的统计或决策环节是什么；支持论文结论的最小对照实验应如何设置。'))

    add(r'\newpage',section('七 实验数据与统一读表口径'),
        '以下均为本项目的已完成离线实验。每轨迹初始4点、逐点追加16次，总20次；优化seed固定0，使用各任务约定的共同初始化。P0/P1结果截至10月29日，连续任务R5截至10月30日。P0/P1阶段修订版与R5不是同一实现，不能把前者的收益当作R5跨数据集验证。',
        table(['层级与任务','候选数','输入与目标'],[
            ['P0 Suzuki Case 1–4','每例4692','3个连续条件＋8类匿名催化剂；Summit模拟产率/%'],
            ['P1 Perera Suzuki','3696','5个名义轴，含底物；实测UV产物峰面积/%'],
            ['P1 Doyle Buchwald–Hartwig','3955','4个名义轴，含底物；来源表反应产率/%'],
            ['P2 Summit SnAr','4096','4个连续条件；空间时产率STY；E-factor仅记录'],
            ['P2 Olympus HPLC','1007','6个连续进样条件；实测峰面积，不是反应产率'],
        ],r'>{\hsize=.75\hsize}L >{\hsize=.35\hsize}L >{\hsize=1.9\hsize}L'),
        r'SnAr的STY单位为kg\,m$^{-3}$\,h$^{-1}$。P1把底物也作为变量，并非固定底物的纯条件优化。Doyle仅保留实测组合；HPLC对重复输入沿用首条测量，不均值化。不同单位的AUC不汇总成总分。',
        r'\[B_j=\max_{i\leq j}y_i,\qquad \mathrm{SearchAUC}=\frac1{16}\sum_{j=5}^{20}B_j.\]',
        '最终最佳是20个观测中的最大值；附表“首次最佳”统一用含初始化的总实验次序，不同方法对应的是各自最终值。AUC反映提早找到高值的效率，最终最大值反映预算结束时的结果，两者可能不同。',
        r'\subsection*{7.1 采用的知识条件}',
        table(['P0条件','实际构造'],[
            ['助手初始建议','复用最初只看共同4个观测形成的助手建议；后续刷新使用deepseek-flash'],
            ['正确知识','围绕已知候选池最优参考点的连续区间，包含参考催化剂'],
            ['错误类别','保留正确连续区间，改用参照输出较差的催化剂'],
            ['混合错误','保留参考催化剂和温度，替换停留时间与负载，使联合区域较差'],
        ],r'>{\hsize=.5\hsize}L >{\hsize=1.5\hsize}L'),
        '后三种为参考结果知情的人工压力测试，不能当作独立化学知识发现。P1与P2各复用一组共同初始知识，未重做四种压力条件。SnAr初始偏好涉及较长停留时间、较低胺当量、较高浓度和温度；HPLC涉及进样环、附加体积、管路体积及流速。初始置信度0.7是设定值，不是校准概率。',
        r'\subsection*{7.2 与原论文及统一底座的差异}',
        'LGBO使用官方解析均值倾斜组件并适配有限池；LLINBO使用官方Justify分支，但共同底座将原实现相应的UCB/方差审查改成qLogEI/机会门限。P1的Reasoning BO按官方推理和分档排序流程，从BO前5点中选1点，适配为4＋16的逐点预算。这里比较的是项目适配系统，不能称为原论文默认协议的原样复现。',
        '实验次数一致不代表LLM调用数一致。P1-Perera的阶段修订版触发了分子相似度核恢复，而原生对照使用原类别核；P0修订版还修复区域协方差输出单位。R5只重跑两项，六种主要对照复用封存结果。各版本共同改变的部分需要单独控制，不能仅凭AUC归因。')

    add(r'\newpage',section('八 P0 Suzuki Case 1至4结果'),
        note('助手初始建议条件，表内为SearchAUC、最终最佳/%及首次最佳总次序。保留已封存11种方法；旧R1–R3仅指该P0阶段版本。'))
    for pair in [(1,2),(3,4)]:
        mini=[]
        for case in pair:
            group={r['method']:r for r in d['P0'] if r['case']==case and r['knowledge_condition']=='assistant_initial'}
            part=r'\textbf{Case '+str(case)+r'}\par\vspace{3pt}'+table(['方法','AUC','最佳','首次'],[
                [label,fmt(group[m]['SearchAUC']),fmt(group[m]['final_best_percent']),group[m]['first_final_best_experiment']]
                for m,label in P0_NAMES.items()],r'>{\hsize=1.5\hsize}L >{\hsize=.9\hsize}R >{\hsize=1\hsize}R >{\hsize=.6\hsize}R')
            mini.append(r'\begin{minipage}[t]{.487\linewidth}\fontsize{8.1}{10}\selectfont '+part+r'\end{minipage}')
        add(r'\begingroup\renewcommand{\arraystretch}{1.04}\setlength{\tabcolsep}{2pt}'+mini[0]+r'\hfill '+mini[1]+r'\endgroup\par\vspace{7pt}')
    add(r'\textbf{四种知识条件的四Case平均SearchAUC}',
        note('均值仅描述这4个指定Case，不是4次同分布随机初始化。逐Case逐条件176行完整结果见随附CSV。'))
    means=[]
    for m,label in P0_NAMES.items():
        means.append([label]+[fmt(np.mean([r['SearchAUC'] for r in d['P0'] if r['method']==m and r['knowledge_condition']==regime])) for regime in REGIMES])
    add(r'\begingroup\fontsize{9}{10.8}\selectfont\renewcommand{\arraystretch}{1.04}'+table(['方法',*REGIMES.values()],means,r'L R R R R',small=False)+r'\endgroup',
        'P0修订版在助手建议下对纯BO为2胜2负；混合错误下平均AUC低于纯BO3.894个百分点。正确先验的收益较明显，但结果不支持“已稳定抵抗错误知识”。[E1,E2]')

    add(r'\newpage',section('九 P1 Perera与Doyle结果'),
        note('共同初始化、优化seed=0、并列seed=0。P1修订版为Stage11 feedback_t0；两数据单位均按各自来源表解释。首次最佳为总实验次序。'))
    for task,title in [('Perera_Suzuki','Perera Suzuki'),('Doyle_Buchwald_Hartwig','Doyle Buchwald–Hartwig')]:
        group={r['method']:r for r in d['P1'] if r['task']==task}
        add(r'\subsection*{'+title+'}',table(['方法','SearchAUC','最终最佳/%','首次最佳'],[
            [label,fmt(group[m]['SearchAUC']),fmt(group[m]['final_best_percent']),group[m]['first_final_best_experiment']]
            for m,label in P1_NAMES.items()],r'>{\hsize=1.6\hsize}L >{\hsize=.8\hsize}R >{\hsize=.9\hsize}R >{\hsize=.7\hsize}R'))
    add(r'\textbf{结果解释。}Perera修订版比纯BO的AUC高2.970，较Reasoning BO低0.218，最终值同为99.050\%；初始最佳已95.570\%，存在天花板。Doyle修订版比Reasoning BO的AUC高1.289，最终值相同，早1次总实验达到；较纯BO搜索更快，但最终值低0.444个百分点。',
        r'\textbf{收益来自哪些实际实验。}Perera前3次追加依次执行联合检验、换背景检验、局部跟进，结果70.650\%、81.700\%、99.050\%；第2次明确越过正常门限，仍计入16次预算。Doyle第2、4、6次单轴反例得到81.275\%、78.253\%、98.287\%，没有门限例外。不能把这些收益全归因于KF或语言解释。',
        r'\textbf{稳定性范围。}修订版并列seed0/1/2的两项AUC极差均为0；初始化始终相同，相同公开输入允许复用首次有效LLM回复。该结果只支持指定并列顺序下的一致性，不是独立初始化、独立LLM抽样或未知任务验证。Perera分子核不同于原生对照，仍缺同核最终消融。[E1,E3]')

    add(r'\newpage',section('十 SnAr与HPLC连续任务结果'),
        note('14种封存方法，包含最新R5与历史方案。每项为AUC、最终最佳、首次最佳总次序；SnAr为STY，HPLC为峰面积。'))
    groups={task:{r['method']:r for r in d['P2'] if r['task']==task} for task in ['SnAr','HPLC']}
    combined=[]
    for m,label in P2_NAMES.items():
        row=[label]
        for task in groups:
            r=groups[task][m];row += [fmt(r['SearchAUC']),fmt(r['final_best']),r['first_final_best_total']]
        combined.append(row)
    add(r'\begingroup\fontsize{9}{10.8}\selectfont\setlength{\tabcolsep}{3pt}'+table(
        ['方法','SnAr AUC','最佳','首次','HPLC AUC','最佳','首次'],combined,
        r'>{\hsize=1.6\hsize}L >{\hsize=1.1\hsize}R >{\hsize=1.1\hsize}R >{\hsize=.5\hsize}R >{\hsize=1\hsize}R >{\hsize=1.1\hsize}R >{\hsize=.6\hsize}R',small=False)+r'\endgroup',
        r'\begin{center}\includegraphics[width=\linewidth]{continuous_best.pdf}\end{center}',
        r'\textbf{R5结果。}SnAr较R4的AUC下降194.158、最终最佳下降621.307；HPLC与R2–R4指标相同。早期最佳全部保留，但没有后期峰值增量。R5与R2/R3部分累计最佳曲线重叠，并不代表所有查询点相同。',
        table(['R5本轮实际激活','SnAr','HPLC'],[
            ['新LLM调用／其中第11轮起','4／2','4／2'],['语义通过／GP接纳','4／1','4／1'],
            ['主动验证／均值部署','0／0','0／0'],['跨模型反证链接／暂停动作','0／0','0／0'],
        ],r'>{\hsize=1.6\hsize}L >{\hsize=.7\hsize}R >{\hsize=.7\hsize}R'),
        'SnAr保护前4轮、HPLC保护前3轮完整复用原R2状态，共7个位置；其余25个位置重新决策。21项行为测试、32轮独立复算通过，3322个既有证据文件未变。这些是实现与追溯证据，不是性能优越性的证明。[E4,E5]',
        note('表中14种方法并非相同开发版本的模块消融；R2子集恢复为单独分支。原新方案、R1等负面结果均保留。'))

    add(r'\newpage',section('附录 关键逐轮证据与结果附件'),
        '下面列出R5实际调用时的菜单可达性。门限和损失均是基础qLogEI分数差；菜单内预算点数为0时，独立提议必然无法通过GP。没有使用未查询目标值判断可达性。',
        table(['任务','追加轮','菜单/预算内','全预算集','提议损失','门限','接纳'],[
            [c['task'],c['query'],str(c['menu_size'])+'/'+str(c['menu_inside_gate']),c['eligible_count'],fmt(c['GP_loss']),fmt(c['epsilon']),
             '是' if c['GP_accepted'] else '否'] for c in d['R5_calls']],r'L R R R R R R'),
        r'\textbf{对照解释。}R4的SnAr优势表现为第5轮独立改选后，基础BO在第12轮找到新峰值；当时没有主动验证或均值修正，不能归因于KF。HPLC的LLINBO-Justify在第9轮是提案被拒后由BO执行得到更高峰值，并非LLM当轮直接命中。这些轨迹说明早期改选会改变后续GP路径，单点“谁提出”不能代替完整轨迹归因。[E5]',
        r'\subsection*{附件的覆盖范围}',
        '随附完整CSV共232行：P0的4 Case×4知识条件×11方法共176行；P1主对比及3组并列顺序24行，另保留早期V3相关4行；SnAr/HPLC共14方法×2任务28行。所有指标重新从各自20点轨迹计算，重复列示的纯BO等不会被当作独立重复。早期任意编号GP、其他预算和批量多种子协议不与这些表混算。',
        r'\subsection*{证据索引}',
        r'\begin{description}\setlength{\itemsep}{3pt}',
        r'\item[E1] P0/P1汇总与原始状态路径：\path{research/p0_p1_report_20261029/data.json}；配套报告核对了数据单位与共同初始化。',
        r'\item[E2] P0修订版设计与16条新轨迹：\path{research/suzuki_stable_20261029/protocol.json}和同目录\path{results.json}。',
        r'\item[E3] P1 Stage11：\path{research/stable_r2_20261029/feedback_t0/}；三组并列顺序见父目录\path{feedback_stability.json}。',
        r'\item[E4] 连续任务全部指标及核验：\path{research/continuous_r5_20261030/results.json}、\path{validation_receipt.json}。',
        r'\item[E5] R5原生选择前记录：\path{data/runs/continuous_r5_20261030/runs/*/rounds/}；完整公开提示与脱敏回复在对应\path{requests/}。R4机制诊断：\path{research/continuous_r4_diagnosis_20261030/findings.json}。',
        r'\end{description}',
        '资料包包含本PDF的TeX源、图资产、完整指标CSV、调用可达性CSV及带源文件哈希的核验记录。原始大型候选表与全部调用证据仍保留在项目中，本次不重复拷贝。文档未发送给任何外部人员，没有重新调用LLM、拟合GP或运行实验。',
        note('当前材料用于请教方法与实验设计，不代表已经建立新颖性、统计显著性或真实化学合成成功。'))
    add(r'\end{document}')
    return '\n\n'.join(text)


def main():
    os.chdir(ROOT);HERE.mkdir(parents=True,exist_ok=True);TYPESET.mkdir(parents=True,exist_ok=True)
    snapshot=collect();overview_figure()
    tex=HERE/(NAME+'.tex');tex.write_text(document(snapshot),encoding='utf-8')
    env=dict(os.environ,TECTONIC_CACHE_DIR=str(ROOT/'outputs/tex_cache'))
    proc=subprocess.run([str(ROOT/'outputs/tex_tools/tectonic/tectonic.exe'),'--only-cached','--keep-logs','--outdir',str(TYPESET),str(tex)],
        cwd=HERE,env=env,text=True,capture_output=True,encoding='utf-8',errors='replace')
    (TYPESET/'compile_console.log').write_text(proc.stdout+'\n'+proc.stderr,encoding='utf-8')
    if proc.returncode:raise RuntimeError((proc.stdout+'\n'+proc.stderr)[-4000:])
    shutil.copyfile(TYPESET/(NAME+'.pdf'),HERE/(NAME+'.pdf'))
    for name,value in SOURCES.items():assert sha(ROOT/name)==value,name
    write(HERE/'sources.json',SOURCES)
    write(HERE/'build_receipt.json',{'status':'BUILT_PENDING_VISUAL_REVIEW','verification':snapshot['verification'],
        'source_files_unchanged':len(SOURCES),'sources_sha256':sha(HERE/'sources.json'),'builder_sha256':sha(__file__),
        'PDF_sha256':sha(HERE/(NAME+'.pdf')),'TeX_sha256':sha(tex),'new_experiment_or_API_calls':0})
    print(json.dumps({'PDF':str(HERE/(NAME+'.pdf')),'verification':snapshot['verification'],'sources':len(SOURCES)},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
