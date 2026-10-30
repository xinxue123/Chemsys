"""Three-page display report from verified P0/P1 data; no experimental work."""
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
from hashlib import sha256
import json
import os
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
TMP=ROOT/'outputs/p0_p1_summary'
NAME='P0_P1_Summary'
DATA=HERE.parent/'data.json'


def h(p):return sha256(Path(p).read_bytes()).hexdigest()
def fmt(v,n=3):return format(Decimal(str(v)).quantize(Decimal('0.000000000001'),rounding=ROUND_HALF_UP).quantize(Decimal(1).scaleb(-n),rounding=ROUND_HALF_UP),'f')
def esc(s):
    for a,b in [('\\',r'\textbackslash{}'),('&',r'\&'),('%',r'\%'),('_',r'\_'),('#',r'\#')]:s=s.replace(a,b)
    return s


NAMES={'Pure_BO':'纯BO（qLogEI）','LGBO':'区域LGBO','LGBO_region':'区域LGBO',
    'Gate_only':'仅预算门限','B4_hard':'硬B4','B4_soft':'软B4＋区域引导',
    'LLINBO_Justify':'LLINBO-Justify','ReasoningBO_point':'Reasoning BO','Stable_revised':'修订版Agent',
    'feedback_t0':'修订版Agent'}
P0=('Pure_BO','LGBO','Gate_only','B4_hard','B4_soft','LLINBO_Justify','Stable_revised')
P1=('Pure_BO','LGBO_region','Gate_only','B4_hard','B4_soft','LLINBO_Justify','ReasoningBO_point','feedback_t0')


def table(headers,rows,spec,small=True):
    size=r'\small' if small else ''
    out=[r'\par\begingroup'+size,r'\begin{tabularx}{\linewidth}{'+spec+'}',r'\toprule',
        ' & '.join(r'\textbf{'+str(v)+'}' for v in headers)+r'\\',r'\midrule']
    for row in rows:
        bold='修订版Agent' in str(row[0])
        out.append(' & '.join((r'\textbf{'+str(v)+'}') if bold else str(v) for v in row)+r'\\')
    out += [r'\bottomrule',r'\end{tabularx}',r'\endgroup\par']
    return '\n'.join(out)


HEADER=r"""
\documentclass[UTF8,fontset=fandol,11pt,a4paper]{ctexart}
\usepackage{fontspec}
\setmainfont[BoldFont=texgyretermes-bold.otf,ItalicFont=texgyretermes-italic.otf,BoldItalicFont=texgyretermes-bolditalic.otf]{texgyretermes-regular.otf}
\usepackage[left=22mm,right=22mm,top=20mm,bottom=20mm,headheight=14pt,headsep=7mm]{geometry}
\usepackage{booktabs,tabularx,array}
\usepackage{fancyhdr}
\usepackage[hidelinks,unicode,hypertexnames=false]{hyperref}
\hypersetup{pdftitle={P0与P1两级实验结果简报},pdfauthor={}}
\newcolumntype{L}{>{\raggedright\arraybackslash}X}
\newcolumntype{R}{>{\raggedleft\arraybackslash}X}
\newcolumntype{C}{>{\centering\arraybackslash}X}
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.18}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\linespread{1.12}
\setlength{\emergencystretch}{2em}
\ctexset{section={format=\large\bfseries\heiti,beforeskip=1.4ex,afterskip=.8ex},
subsection={format=\normalsize\bfseries\heiti,beforeskip=1.1ex,afterskip=.5ex}}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small P0与P1两级实验结果简报}
\fancyhead[R]{\small 数据集\quad 知识条件\quad 对比结果}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.3pt}
\begin{document}
\begin{center}
{\LARGE\bfseries\heiti P0与P1两级实验结果简报}\\[4pt]
{\small 固定候选集\quad Mixed GP\quad seed=0\quad 4次初始化＋16次查询}
\end{center}
"""


def main():
    os.chdir(ROOT);TMP.mkdir(parents=True,exist_ok=True)
    data=json.loads(DATA.read_text(encoding='utf-8'));source_hash=h(DATA)
    p0,p1=data['P0'],data['P1']
    def a(case,regime,method):return next(r for r in p0 if r['case']==case and r['knowledge_condition']==regime and r['method']==method)
    def b(task,method):return next(r for r in p1 if r['task']==task and r['method']==method)
    text=[HEADER,
        r'\section*{1\quad 数据集与统一设置}',
        table(['层级与数据集','候选数','优化变量','评价指标'],[
            ['P0：Reizman Suzuki\\newline Case 1--4','每Case 4692','停留时间、温度、催化剂负载及8个匿名催化剂','模拟产率/\\%'],
            ['P1：Perera Suzuki','3696','亲电体、亲核体、配体、碱、溶剂','UV产物峰面积/\\%'],
            ['P1：Doyle Buchwald--Hartwig','3955','芳基卤化物、配体、添加剂、碱','来源表报告产率/\\%']],
            r'@{}>{\raggedright\arraybackslash}p{.24\linewidth}>{\centering\arraybackslash}p{.13\linewidth}LL@{}'),
        r'P0使用冻结反应模拟器回放；P1使用作者提供的实测表逐点回放。P1中底物也是优化变量，不是固定底物后的条件优化；Perera指标不等同于分离收率。',
        r'同一任务的各方法使用相同候选池和初始4点，追加16次逐点查询。以下选取主要对照方案，不展开开发版本和逐轮日志。',
        r'\section*{2\quad 采用的知识条件}',
        r'\subsection*{P0：四种初始知识条件}',
        table(['条件','具体设置'],[
            ['助手初始建议','复用仅根据共同4个初始观测形成的助手区域建议，作为主系统比较。'],
            ['正确知识（人工）','围绕候选池参考最优点构造连续条件区间，并包含参考最优催化剂。'],
            ['错误类别（人工）','保留连续条件区间，但改用参考表现较差的催化剂类别。'],
            ['混合错误（人工）','保留催化剂与温度偏好，替换停留时间及负载区间，使联合区域较差。']],
            r'@{}>{\raggedright\arraybackslash}p{.25\linewidth}L@{}'),
        r'{\small 三种人工条件利用参考结果构造，用于压力测试，不是独立发现的化学真理。每Case每条件一条seed=0轨迹。}',
        r'\subsection*{P1：共同初始知识与并列顺序检查}',
        r'LLM根据共同4个观测形成一组初始偏好，各方法复用这组待检验知识。修订版另外使用并列seed=0、1、2检查完全同分候选的随机顺序；初始化仍固定seed=0，没有增加四种人工知识条件。',
        r'\section*{3\quad Agent的简要执行逻辑}',
        r'每轮先由Mixed GP预测全部候选，再计算qLogEI。LLM提供或修订化学偏好区域，KF根据实验反馈调整知识幅度，B4与预算门限控制知识影响；停滞、预测失准或冲突时触发知识重审。因冲突停用的知识，须有新实验支持其预测增益才可恢复。选定候选后读取结果，继续更新经验。定向检验也占用同16次追加预算。',
        r'{\small P0缺少催化剂结构，未启用分子核及联合检验；P1-Perera启用分子核恢复与联合/局部检验。因此两级存在适配差异，不能把全部收益归到单个模块。}',
        r'\newpage',
        r'\section*{4\quad P0：Suzuki Case 1--4对比}',
        r'{\small SearchAUC为16次追加查询后累计最佳值的均值，越高越好；首次最佳按含4个初始化的总实验次序计。所有表值由封存轨迹重算。}',
        r'\subsection*{4.1\quad 助手初始建议下的SearchAUC}',
        table(['方法','Case 1','Case 2','Case 3','Case 4'],[
            [NAMES[m],*[fmt(a(c,'assistant_initial',m)['SearchAUC']) for c in range(1,5)]] for m in P0],
            r'@{}>{\raggedright\arraybackslash}p{.28\linewidth}RRRR@{}'),
        r'\subsection*{4.2\quad 四种知识条件的平均SearchAUC}',
        r'{\small 每列为四个Case的均值，不能当作四次独立初始化。}',
        table(['方法','助手建议','正确知识','错误类别','混合错误'],[
            [NAMES[m],*[fmt(sum(a(c,r,m)['SearchAUC'] for c in range(1,5))/4) for r in ['assistant_initial','correct','wrong_category','mixed']]] for m in P0],
            r'@{}>{\raggedright\arraybackslash}p{.28\linewidth}RRRR@{}'),
        r'\subsection*{4.3\quad 助手初始建议下的最终最佳值}',
        table(['Case','修订版/\\%','纯BO/\\%','LLINBO/\\%','修订版首次最佳'],[
            [str(c),fmt(a(c,'assistant_initial','Stable_revised')['final_best_percent']),
             fmt(a(c,'assistant_initial','Pure_BO')['final_best_percent']),
             fmt(a(c,'assistant_initial','LLINBO_Justify')['final_best_percent']),
             str(a(c,'assistant_initial','Stable_revised')['first_final_best_experiment'])] for c in range(1,5)],
            r'@{}>{\raggedright\arraybackslash}p{.14\linewidth}RRRR@{}'),
        r'\subsection*{P0主要观察}',
        r'修订版在Case 2、4的SearchAUC高于纯BO，在Case 1、3低于纯BO；混合错误知识下平均AUC较纯BO低3.894个百分点。现有结果不支持“在Suzuki上稳定抗错误知识”的结论。Case 2各方法最终值接近，主要差异体现为更早找到高值。',
        r'\newpage',
        r'\section*{5\quad P1：两套实测HTE数据对比}',
        r'\subsection*{5.1\quad Perera Suzuki}',
        table(['方法','SearchAUC','最佳值/\\%','首次最佳'],[
            [NAMES[m],fmt(b('Perera_Suzuki',m)['SearchAUC']),
             fmt(b('Perera_Suzuki',m)['final_best_percent']),
             str(b('Perera_Suzuki',m)['first_final_best_experiment'])] for m in P1],
            r'@{}>{\raggedright\arraybackslash}p{.40\linewidth}RRR@{}'),
        r'\subsection*{5.2\quad Doyle Buchwald--Hartwig}',
        table(['方法','SearchAUC','最佳值/\\%','首次最佳'],[
            [NAMES[m],fmt(b('Doyle_Buchwald_Hartwig',m)['SearchAUC']),
             fmt(b('Doyle_Buchwald_Hartwig',m)['final_best_percent']),
             str(b('Doyle_Buchwald_Hartwig',m)['first_final_best_experiment'])] for m in P1],
            r'@{}>{\raggedright\arraybackslash}p{.40\linewidth}RRR@{}'),
        r'\subsection*{5.3\quad 修订版三组并列顺序检查}',
        table(['数据集','并列seed 0','并列seed 1','并列seed 2'],[
            [label,*[fmt(b(task,f'feedback_t{t}')['SearchAUC']) for t in range(3)]]
            for task,label in [('Perera_Suzuki','Perera Suzuki'),('Doyle_Buchwald_Hartwig','Doyle Buchwald--Hartwig')]],
            r'@{}>{\raggedright\arraybackslash}p{.40\linewidth}RRR@{}'),
        r'{\small 表中为SearchAUC。三组的最终最佳值及首次最佳也分别一致；主对比取并列seed=0。相同输入复用LLM原始回复，不是独立初始化或独立LLM随机性检验。}',
        r'\subsection*{P1主要观察}',
        r'Perera修订版与Reasoning BO的最终值相同，AUC低0.218，晚1次总观测达到该值。Doyle修订版比Reasoning BO的AUC高1.289、最终值相同，早1次总观测达到；比纯BO搜索更快，但最终值低0.444个百分点。',
        r'{\small LGBO、LLINBO与Reasoning BO均采用本项目固定候选集和4＋16预算下的接口适配。结论限于固定初始化、既有数据和有限并列顺序检查。数据来源为封存轨迹，整理日期2026年10月29日。}',
        r'\end{document}']
    tex='\n\n'.join(text)
    assert r'\tableofcontents' not in tex
    (HERE/(NAME+'.tex')).write_text(tex,encoding='utf-8')
    engine=ROOT/'outputs/tex_tools/tectonic/tectonic.exe'
    env=os.environ.copy();env['TECTONIC_CACHE_DIR']=str(ROOT/'outputs/tex_cache')
    command=[str(engine),'--keep-logs','--keep-intermediates','--print','--outdir',str(TMP),str(HERE/(NAME+'.tex'))]
    with (TMP/'compile_console.log').open('wb') as log:
        completed=subprocess.run(command,cwd=HERE,env=env,stdout=log,stderr=subprocess.STDOUT)
    if completed.returncode:
        print((TMP/'compile_console.log').read_text(encoding='utf-8',errors='replace')[-4000:]);raise SystemExit(completed.returncode)
    shutil.copy2(TMP/(NAME+'.pdf'),HERE/(NAME+'.pdf'))
    assert h(DATA)==source_hash
    proof={'scope':'concise display report; no contents or development appendix','source_data_sha256':source_hash,
        'tex_sha256':h(HERE/(NAME+'.tex')),'pdf_sha256':h(HERE/(NAME+'.pdf')),
        'P0_methods':list(P0),'P1_methods':list(P1),'new_API_calls':0,'new_experiments':0,
        'log':str(TMP/'compile_console.log')}
    (HERE/'summary_provenance.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(proof,ensure_ascii=True))


if __name__=='__main__':main()
