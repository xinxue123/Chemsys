"""Readable TeX/PDF diagnosis from the read-only same-prefix audit outputs."""
from pathlib import Path
from decimal import Decimal,ROUND_HALF_UP
import csv
import json
import os
import shutil
import subprocess
import hashlib

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/continuous_audit_20261030'
NAME='SnAr_HPLC_Module_Audit'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def rows(name):return list(csv.DictReader((OUT/name).open(encoding='utf-8-sig')))
def fmt(x,n=3):return format(Decimal(str(x)).quantize(Decimal(1).scaleb(-n),rounding=ROUND_HALF_UP),'f')
def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def table(head,body,spec):
    lines=[r'\par\begingroup\small',r'\begin{tabularx}{\linewidth}{'+spec+'}',r'\toprule',' & '.join(head)+r'\\',r'\midrule']
    lines += [' & '.join(str(v) for v in row)+r'\\' for row in body]
    return '\n'.join(lines+[r'\bottomrule',r'\end{tabularx}\endgroup\par'])

HEADER=r'''\documentclass[UTF8,fontset=fandol,10pt,a4paper]{ctexart}
\usepackage{fontspec}
\setmainfont[BoldFont=texgyretermes-bold.otf,ItalicFont=texgyretermes-italic.otf]{texgyretermes-regular.otf}
\usepackage[left=19mm,right=19mm,top=18mm,bottom=18mm,headheight=14pt,headsep=6mm]{geometry}
\usepackage{booktabs,tabularx,array,graphicx,fancyhdr,amsmath}
\usepackage[hidelinks,unicode]{hyperref}
\hypersetup{pdftitle={SnAr与HPLC新方案逐轮诊断}}
\newcolumntype{L}{>{\raggedright\arraybackslash}X}
\newcolumntype{R}{>{\raggedleft\arraybackslash}X}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.15}\linespread{1.08}\setlength{\emergencystretch}{2em}
\ctexset{section={format=\large\bfseries\heiti,beforeskip=1.3ex,afterskip=.8ex},
subsection={format=\normalsize\bfseries\heiti,beforeskip=.9ex,afterskip=.5ex}}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small SnAr与HPLC：新方案逐轮诊断}
\fancyhead[R]{\small 冻结轨迹审计\quad 同历史对照}\fancyfoot[C]{\thepage}
\begin{document}
'''

def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    curves=read(OUT/'curves.json')
    for task in ('SnAr','HPLC'):
        methods=['Pure_BO','New_Agent'] if task=='SnAr' else ['Pure_BO','LLINBO_Justify','New_Agent']
        palette={'Pure_BO':'#666666','LLINBO_Justify':'#2676ad','New_Agent':'#cf3030'}
        with plt.rc_context({'font.size':7,'pdf.fonttype':42,'axes.titlesize':8,'xtick.labelsize':7,'ytick.labelsize':7}):
            fig,ax=plt.subplots(figsize=(6.7,1.85),layout='constrained')
            for m in methods:
                ax.plot(range(17),curves[task][m]['incumbent'][3:],color=palette[m],label=m,
                    linewidth=1.7 if m=='New_Agent' else 1.2,marker='o' if m=='New_Agent' else 's',markersize=2.5,
                    linestyle='--' if m=='Pure_BO' else '-')
            ax.set_xlabel('Added query q (q=0: shared initial best)');ax.set_ylabel('STY (kg m^-3 h^-1)' if task=='SnAr' else 'Peak area (source units)')
            ax.set_xticks(range(0,17,2));ax.grid(alpha=.2);ax.legend(loc='lower right',fontsize=6.5)
            fig.savefig(OUT/f'{task}_audit_curve.pdf');fig.savefig(OUT/f'{task}_audit_curve.png',dpi=180);plt.close(fig)

def main():
    os.chdir(ROOT);checks=read(OUT/'execution_checks.json');assert checks['status']=='PASS' and checks['KF_updates_independently_checked']==30
    wide=rows('module_round_effects.csv');stage=rows('stage_choices.csv');gaps=rows('incumbent_gaps.csv');kf=rows('KF_prediction_diagnostics.csv')
    refresh=read(OUT/'refresh_audit.json');figures()
    W=lambda task,t:next(r for r in wide if r['task']==task and int(r['query_round'])==t)
    S=lambda task,t,name:next(r for r in stage if r['task']==task and int(r['query_round'])==t and r['stage']==name)
    K=lambda task,t:next(r for r in kf if r['task']==task and int(r['query_round'])==t)
    def comparison(task,t,a,b,label):
        aa=S(task,t,a);bb=S(task,t,b)
        return [t,label,fmt(aa['objective_after_choice']),fmt(bb['objective_after_choice']),fmt(float(bb['objective_after_choice'])-float(aa['objective_after_choice']))]
    # Machine-readable result digest keeps numerical provenance close to prose.
    summary={'execution':'32 decisions and 6 refreshes replay exactly; 30 KF updates independently checked; no numerical selection mismatch',
        'GP_refits':0,'LLM_calls':0,'KF_scaling_changed_candidate_count':sum(r['KF_actual_changed_choice']=='True' for r in wide),
        'SnAr_SearchAUC_gap_vs_BO':sum(float(r['contribution_to_SearchAUC_gap']) for r in gaps if r['task']=='SnAr' and r['baseline']=='Pure_BO'),
        'HPLC_SearchAUC_gap_vs_LLINBO':sum(float(r['contribution_to_SearchAUC_gap']) for r in gaps if r['task']=='HPLC' and r['baseline']=='LLINBO_Justify'),
        'initial_HPLC_region':{'total_candidates':1,'unqueried_candidates':0},
        'scope':'All effects are local same-prefix comparisons unless explicitly described as arithmetic actual-trajectory AUC decomposition.'}
    (OUT/'findings.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    text=[HEADER,r'\begin{center}{\LARGE\bfseries\heiti 执行检查与模块判定}\end{center}',
        r'\textbf{未发现GP/qLogEI/KF数值或选点执行不一致；发现知识可执行性校验缺口，以及知识、滤波和刷新调度的设计问题。}本次只读取封存轨迹：32次新方案选择、6次刷新完整重建，30次KF更新独立复算；最大KF误差$5.6\times10^{-17}$。1161个原文件保持原哈希，新增LLM调用和GP拟合均为0。',
        r'\section*{怎样解释本报告的“收益/损失”}',
        r'轮次$q=1\ldots16$指共同4点初始化后的追加查询，总实验次序为$q+4$。同历史对照固定该轮实际已有数据，逐项改变一个决策阶段，再回查其已封存的模拟/实测目标值。表中差值是\textbf{单步目标值差}，不等于整条轨迹的因果SearchAUC贡献；低目标实验本身不会降低累计最佳。',
        table(['模块','本轮直接证据','判断'],[
            ['初始知识','SnAr第1轮负向改选；HPLC初始交集只有1个已测点、0个新点。','知识可用性与覆盖不足。'],
            ['KF','32轮中，相对同一区域固定幅度1，KF缩放没有一次改变候选；两任务均无信任恢复。','算式正确，但未证明选点增量。'],
            ['B4/信任门','有保护，也放弃了更高目标候选；后期多轮部署幅度为0。','采集预算诊断不等于化学证伪。'],
            ['事件刷新','SnAr第3轮当轮负向改选；另5次刷新当轮建议不变。HPLC第3/5/7轮用尽3次额度。','改变知识和重置状态不自动产生增益。'],
        ],'L L L'),
        r'B4还有规定性的假设槽位偏置：HPLC第1轮有4个同样保留3条的可行子集，按ID字典序选择H1/H2/H3；交换H1/H4标签后，保留几何就改变。SnAr第4轮也出现此现象。\textbf{这是假设子集排序规则，不是候选编号排序错误，也不是唯一的化学证伪结论。}当前冲突回退仍使当轮部署幅度为0。',
        r'\section*{KF为什么没有及时削弱错误知识}',
        r'滤波观测模型为$r=y-\mu_{GP}=b+a g_R+\nu$，实际部署却是$\mu_{GP}+\operatorname{clip}(a,0,1)g_R$：\textbf{偏移$b$不部署，未截断的$a$参与滤波。}因此，拟合联合残差与检验实际部署修正是否优于GP，使用了不同的预测。少量数据时$b$和$a$还会相互补偿。',
        table(['示例','GP均值','部署修正','实测值','更新后$a$'],[
            ['SnAr第6轮',fmt(K('SnAr',6)['GP_mean']),fmt(K('SnAr',6)['deployed_correction']),fmt(K('SnAr',6)['objective']),fmt(K('SnAr',6)['a_after_unclipped'])],
            ['HPLC第5轮',fmt(K('HPLC',5)['GP_mean']),fmt(K('HPLC',5)['deployed_correction']),fmt(K('HPLC',5)['objective']),fmt(K('HPLC',5)['a_after_unclipped'])],
        ],'L R R R R'),
        r'SnAr第6轮加修正后预测由7358.386变为6612.300，比GP更不准，但$a$仍从1.626升至1.756；第7/8轮部署继续取满幅1。HPLC第5轮实测仅71.528，$a$仍为0.918；第7轮换区域又从1开始。两项现象均符合代码，但不等同于可靠的增量知识检验。',
        r'\section*{应优先修正的接口与调度}',
        r'知识准入只检查“全池交集非空”，B4却检查“未查询候选中的交集”。HPLC初始知识满足前者而不满足后者，形成口径缺口。刷新改变几何后使用新KF状态是既定规则；应避免把新几何的默认幅度1解释成已获得新实验信任。',
        r'优先项：统一知识与B4的未查询候选口径；用实验前冻结的实际部署预测相对GP的误差增益约束幅度；为后期冲突保留刷新额度；把SnAr的交互检验纳入同一实验预算。\textbf{本次没有实施修复或重跑。}',
        r'{\footnotesize HPLC按封存协议取首条测量。参考最优条件的两条重复值为2569.880/2080.323，新方案最佳条件为2317.816/2145.964；本报告未改用均值重新计分，也不据这条seed=0轨迹宣称真实均值或统计显著优势。}',
        r'\newpage',r'\begin{center}{\LARGE\bfseries\heiti SnAr：早期延迟与交互遗漏}\end{center}',
        r'新方案SearchAUC为9378.551，纯BO为10373.540，差$-994.989$。纯BO第4次追加查询达到11559.605；新方案第5轮得到10938.298后再未提高，始终未达到纯BO的最终值。',
        r'\section*{初始知识方向与参考最优不一致}',
        r'初始偏好：停留时间1.25--1.70 min、当量1.0--1.6、浓度0.34--0.44 M、温度79.5--109.2℃。四条都不包含该固定池的参考最优坐标（0.5 min、5当量、0.5 M、30℃）。初始交集12点中最高STY仅3147.694。这是事后知识质量检查；这些目标值没有进入LLM或原始选点。',
        table(['固定停留时间0.5 min、浓度0.5 M','30℃','120℃'],[
            ['吡咯烷1当量','6106.957','10938.298'],['吡咯烷5当量','11559.605','3556.043'],
        ],'L R R'),
        r'当量效应随温度反转。第6轮只在120℃背景下增加当量，无法检验“高当量＋低温”的联合优势；不能把单轴结果外推为全局规律。区域注入是软修正，上述坐标未被硬删除。',
        r'\section*{关键轮次：同历史的阶段替换}',
        table(['轮次','改变阶段','替换前目标','替换后目标','差值'],[
            comparison('SnAr',1,'base_BO','executed','初始区域引导'),
            comparison('SnAr',2,'normal_after_trust','executed','单轴反例检验'),
            comparison('SnAr',3,'before_refresh_normal','normal_after_trust','事件刷新'),
            comparison('SnAr',4,'normal_after_trust','executed','单轴反例检验'),
            comparison('SnAr',5,'before_refresh_normal','normal_after_trust','事件刷新'),
            comparison('SnAr',6,'normal_after_trust','executed','单轴反例检验'),
            comparison('SnAr',7,'base_BO','normal_after_trust','区域引导'),
            comparison('SnAr',9,'B4_conflict_gate','normal_after_trust','持续信任门'),
        ],'c L R R R'),
        r'第2轮反例救回了错误区域建议，但选点恰与基础BO相同。第4轮B4先撤销H4，几何改变使建议从10063.327变为9492.476，再通过冲突回退回到BO的9650.715；随后\textbf{反例检验覆盖了这个回退点}，实际仅6110.828。损失不能全归给B4。',
        r'第5轮的10938.298也是同历史基础BO会选的点。第6轮检验超出正常门限，已按协议记为例外；第6/7/9轮候选都低于当时最佳10938.298，表中单步损失不能直接折算成AUC扣分。',
        r'第9轮刷新撤销H2/H3后，几何与刚封锁的保留区域相同；B4转清晰也未自动恢复信任，符合既定证据锁规则。第9--16轮知识部署幅度均为0；第10--16轮预算集都只有1点。相对纯BO的AUC差中，前4轮贡献$-529.009$，后12轮未达到相同最优值贡献$-465.980$。',
        r'\begin{center}\includegraphics[width=\linewidth]{SnAr_audit_curve.pdf}\end{center}',
        r'\newpage',r'\begin{center}{\LARGE\bfseries\heiti HPLC：区域过窄与GP历史分化}\end{center}',
        r'新方案SearchAUC为2153.558，LLINBO-Justify为2287.240，差$-133.682$；最终最佳分别为2317.816和2569.880。相对纯BO，新方案仅提前1轮取得同一最终值，使AUC增加$438.022/16=27.376$。',
        r'\section*{知识、KF与刷新分别做了什么}',
        r'初始交集只有已测初始最佳点HPLC\_00640，未查询候选为0；第1/2轮B4因此冲突，正常选择均回退BO。样品环、附加体积和管路体积偏好在所有知识版本中都不包含池参考最优HPLC\_00457。',
        table(['轮次','区域引导相对同历史BO','BO目标','实际目标','差值'],[
            comparison('HPLC',4,'base_BO','executed','正向单步变化')[0:2]+comparison('HPLC',4,'base_BO','executed','')[2:],
            comparison('HPLC',5,'base_BO','executed','负向单步变化'),
            comparison('HPLC',6,'base_BO','executed','负向单步变化'),
            comparison('HPLC',7,'base_BO','executed','提高当轮累计最佳'),
        ],'c L R R R'),
        r'第4/5/6轮均未提高已有最佳1879.794。第7轮提高至2317.816，但刷新前已会选同一点；\textbf{不能把该轮收益直接归给第7轮刷新或KF}。第3/5/7轮刷新当轮建议全部不变，3次额度却已耗尽；第7轮的新结果及第8轮后的冲突未再进入一次新知识重审。',
        r'第7轮知识交集仅4点、2点未查询，交集最高峰面积仅1783.437；实际选中的2317.816位于交集外。这符合LGBO协方差软引导的数学实现，不能视为已发现正确的高峰面积区域。HPLC没有精确单轴邻居，第2/4/6轮反例全部跳过。',
        r'\section*{B4有保护，也有局部机会损失}',
        r'第1轮冲突回退把B4保留区域建议82.330换成BO点1783.437；第8/9轮分别把1726.283换成2142.062/2103.951。反方向例子是第11轮：回退把保留区域建议1726.283换成36.812。上述点都未超过当时累计最佳，不能据此给B4分配整条AUC因果收益。第8--16轮部署幅度均为0，未发生信任恢复。',
        r'\section*{LLINBO的优势实际来自哪次选择}',
        table(['LLINBO轮次','LLM提案处理','执行目标','关键事实'],[
            [6,'拒绝','2105.820','GP回退先提高最佳'],[7,'拒绝','2266.007','GP回退继续提高'],
            [8,'接受','1408.580','GP此时已建议2569.880，提案反而延迟发现'],
            [9,'拒绝','2569.880','最高值来自GP参考点，不是被接受的LLM提案'],
        ],'c L R L'),
        r'同为第9轮，参考最优点在新方案GP中为$1586.494\pm130.017$，基础采集排名604、机会损失39.497，超出门限0.647；在LLINBO自己的GP历史中为$2278.081\pm240.926$，排名1。\textbf{不同采样历史造成了后验与不确定性的分化}。新方案的门限保护了一个对该点过度自信且偏低的预测，单靠晚期B4不能修复。',
        r'第9--16轮最佳差252.063持续8轮，对AUC差贡献$-126.032$，是主要差距；早期第6轮贡献$-14.127$，第7/8轮新方案领先抵消$+6.476$。这不能简化成“LLINBO的每次化学建议更准”。',
        r'\begin{center}\includegraphics[width=\linewidth]{HPLC_audit_curve.pdf}\end{center}',
        r'\end{document}']
    tex=OUT/(NAME+'.tex');tex.write_text('\n\n'.join(text),encoding='utf-8')
    tmp=ROOT/'outputs/continuous_audit_typeset';tmp.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,TECTONIC_CACHE_DIR=str(ROOT/'outputs/tex_cache'))
    result=subprocess.run([str(ROOT/'outputs/tex_tools/tectonic/tectonic.exe'),'--keep-logs','--outdir',str(tmp),str(tex)],
        cwd=OUT,env=env,text=True,capture_output=True,encoding='utf-8',errors='replace')
    (tmp/'compile_console.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
    if result.returncode:raise RuntimeError((result.stdout+'\n'+result.stderr)[-4000:])
    shutil.copyfile(tmp/(NAME+'.pdf'),OUT/(NAME+'.pdf'))
    provenance={'results':h(OUT/'findings.json'),'checks':h(OUT/'execution_checks.json'),'TeX':h(tex),'PDF':h(OUT/(NAME+'.pdf'))}
    (OUT/'report_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'PDF':str(OUT/(NAME+'.pdf')),'TeX':str(tex),'metrics':summary},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
