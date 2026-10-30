"""Two-page comparison of one frozen R1 rerun against unchanged old controls."""
from pathlib import Path
from decimal import Decimal,ROUND_HALF_UP
import json,os,subprocess,shutil,hashlib

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];NAME='SnAr_HPLC_Revised_R1'
NAMES={'Pure_BO':'纯BO（qLogEI）','LGBO_region':'区域LGBO','Gate_only':'仅预算门限','B4_hard':'硬B4',
    'B4_soft':'软B4＋区域注入','LLINBO_Justify':'LLINBO-Justify','ReasoningBO':'Reasoning BO','New_Agent':'旧新方案','New_Agent_R1':'修正版新方案R1'}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def fmt(v,n=3):return format(Decimal(str(v)).quantize(Decimal(1).scaleb(-n),rounding=ROUND_HALF_UP),'f')
def signed(v):return ('+' if v>=0 else '')+fmt(v)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def table(headers,rows,spec):
    out=[r'\begingroup\small',r'\begin{tabularx}{\linewidth}{'+spec+'}',r'\toprule',' & '.join(headers)+r'\\',r'\midrule']
    for row in rows:
        bold=row[0]=='修正版新方案R1';out.append(' & '.join((r'\textbf{'+str(v)+'}') if bold else str(v) for v in row)+r'\\')
    return '\n'.join(out+[r'\bottomrule',r'\end{tabularx}\endgroup'])
HEADER=r'''\documentclass[UTF8,fontset=fandol,10pt,a4paper]{ctexart}
\usepackage{fontspec}
\setmainfont[BoldFont=texgyretermes-bold.otf,ItalicFont=texgyretermes-italic.otf]{texgyretermes-regular.otf}
\usepackage[left=19mm,right=19mm,top=18mm,bottom=18mm,headheight=14pt,headsep=6mm]{geometry}
\usepackage{booktabs,tabularx,array,graphicx,fancyhdr}
\usepackage[hidelinks,unicode]{hyperref}
\hypersetup{pdftitle={SnAr与HPLC修正版新方案R1}}
\newcolumntype{L}{>{\raggedright\arraybackslash}X}\newcolumntype{R}{>{\raggedleft\arraybackslash}X}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.15}
\linespread{1.08}\setlength{\emergencystretch}{2em}
\ctexset{section={format=\large\bfseries\heiti,beforeskip=1.3ex,afterskip=.7ex}}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small SnAr与HPLC：修正版新方案R1}
\fancyhead[R]{\small 固定协议单次重跑}\fancyfoot[C]{\thepage}
\begin{document}
'''

def figures(curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'Pure_BO':'#666666','LLINBO_Justify':'#2676ad','ReasoningBO':'#ae6a37','New_Agent':'#b56a83','New_Agent_R1':'#d02b2b'}
    for task,group in curves.items():
        with plt.rc_context({'font.size':7,'axes.labelsize':6.5,'axes.titlesize':8,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'pdf.fonttype':42}):
            fig,axes=plt.subplots(1,2,figsize=(6.75,2.5),layout='constrained')
            for method,series in group.items():
                for ax,k in zip(axes,('objective','incumbent')):
                    ax.plot(range(1,21),series[k],color=colors[method],label=method,linewidth=1.6 if method=='New_Agent_R1' else .8,
                        marker='o' if method=='New_Agent_R1' else None,markersize=2,linestyle='--' if method=='New_Agent' else '-')
            for ax,title in zip(axes,('Observed objective','Cumulative best')):
                ax.set_title(title);ax.set_xlabel('Total experiment (4 + 16)');ax.set_ylabel('STY (kg m^-3 h^-1)' if task=='SnAr' else 'Peak area (source units)')
                ax.set_xticks([1,4,8,12,16,20]);ax.grid(alpha=.2);ax.axvspan(.5,4.5,color='#dddddd',alpha=.12)
            handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=6.5)
            fig.savefig(HERE/f'{task}_figure.pdf');plt.close(fig)

def main():
    os.chdir(ROOT);validation=read(HERE/'validation_receipt.json');assert validation['status']=='PASS' and validation['added_choices_verified']==32
    results=read(HERE/'results.json');curves=read(HERE/'curves.json');figures(curves);text=[HEADER]
    for task in ('SnAr','HPLC'):
        if task=='HPLC':text.append(r'\newpage')
        p=read(ROOT/'data/benchmarks/continuous_20261029'/task/'pool.json')
        state=read(ROOT/'data/runs/continuous_revised_20261030/runs'/(task+'_New_Agent_R1')/'state.json')
        group={r['method']:r for r in results['metrics'] if r['task']==task};new=group['New_Agent_R1'];old=group['New_Agent']
        text += [r'\begin{center}{\LARGE\bfseries\heiti '+('Summit SnAr' if task=='SnAr' else 'Olympus HPLC')+r'}\\[3pt]',
            r'{\small seed=0\quad 原候选集与GP\quad 共同4次初始化＋16次查询\quad deepseek-flash}\end{center}',
            r'\section*{本轮修改与比较范围}',
            r'仅重跑两条修正版新方案轨迹，原16条对照及旧新方案结果全部只读。规则在本轮结果之前封存；没有按成绩调整参数或挑选运行。初始4点及原始初始知识建议复用。',
            table(['修改','固定实现'],[
                ['知识准入','每次部署及修订验证均要求联合区域内至少1个未查询候选；空知识允许。'],
                ['KF校正','无偏移项的标量KF，创新与实际部署预测一致；负配对误差增益不允许系数上升。新区域初始幅度0.5。'],
                ['刷新调度','全程最多3次，前8轮最多2次，至少保留1次给第9--14轮；间隔至少2轮。'],
                ['交互检验','第2、6轮各最多1次，均满足原基础qLogEI门限，不增加查询或门限例外。'],
            ],r'>{\hsize=.35\hsize}L >{\hsize=1.65\hsize}L')]
        if task=='SnAr':
            text.append(r'固定4096点、4个连续变量。优化STY（kg\,m$^{-3}$\,h$^{-1}$），E-factor仅记录；交互候选为自身最佳点的精确双轴邻居，其余变量完全一致。')
        else:
            text.append(r'固定1007组不同实测输入，仍按原首条峰面积回放。原始知识交集只有1个已测点、0个新点，先进行准入处理，计入同一刷新额度。候选池没有精确双轴邻居；使用主要两轴变化、其余背景接近的现有联合条件，并记录所有变化轴，\textbf{不宣称独立因果交互}。')
        text += [r'\section*{全部方案结果}',table(['方法','SearchAUC','最终最佳','首次最佳/总次序','AUC较纯BO'],
            [[NAMES[m],fmt(group[m]['SearchAUC']),fmt(group[m]['final_best']),group[m]['first_final_best_total'],signed(group[m]['SearchAUC']-group['Pure_BO']['SearchAUC'])] for m in NAMES],
            r'>{\hsize=1.6\hsize}L >{\hsize=.85\hsize}R >{\hsize=.85\hsize}R >{\hsize=.85\hsize}R >{\hsize=.85\hsize}R'),
            r'{\footnotesize SearchAUC是第5--20次实验位置累计最佳目标的均值；首次最佳针对本轨迹自身最终值。SnAr和HPLC单位不同，不合并排名。}',
            r'\begin{center}\includegraphics[width=\linewidth]{'+task+r'_figure.pdf}\end{center}',
            'R1的SearchAUC较旧新方案 '+signed(new['SearchAUC']-old['SearchAUC'])+'，较纯BO '+signed(new['SearchAUC']-group['Pure_BO']['SearchAUC'])+'，较LLINBO-Justify '+signed(new['SearchAUC']-group['LLINBO_Justify']['SearchAUC'])+'；最终最佳较旧新方案 '+signed(new['final_best']-old['final_best'])+'。',
            '实际刷新轮次：'+str([c['iteration'] for c in state['calls']])+'；交互检验轮次：'+str([c['query'] for c in state['interaction_tests']])+'；越过原采集门限次数：0。']
        negative=[v for v in state['decisions'] if v['measurement']['actual_paired_SE_gain']< -1e-12]
        text.append('共'+str(len(negative))+'次负部署增益，系数均未上升。GP实现未改；共同初始4点上的预测和基础qLogEI逐点一致。全部32次追加选择已审计。')
        text.append(r'{\footnotesize 本轮仅每任务一个初始化；技术检查通过不等于性能提高，也不支持统计显著性或跨任务普遍优势。原始目标尺度及HPLC重复测量处理均未改变。}')
    text.append(r'\end{document}');tex=HERE/(NAME+'.tex');tex.write_text('\n\n'.join(text),encoding='utf-8')
    tmp=ROOT/'outputs/continuous_revised_typeset';tmp.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,TECTONIC_CACHE_DIR=str(ROOT/'outputs/tex_cache'))
    result=subprocess.run([str(ROOT/'outputs/tex_tools/tectonic/tectonic.exe'),'--keep-logs','--outdir',str(tmp),str(tex)],cwd=HERE,env=env,text=True,capture_output=True,encoding='utf-8',errors='replace')
    (tmp/'compile_console.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
    if result.returncode:raise RuntimeError((result.stdout+'\n'+result.stderr)[-4000:])
    pdf=HERE/(NAME+'.pdf');shutil.copyfile(tmp/(NAME+'.pdf'),pdf)
    (HERE/'summary_provenance.json').write_text(json.dumps({'PDF_sha256':sha(pdf),'TeX_sha256':sha(tex),'results_sha256':sha(HERE/'results.json'),'validation_sha256':sha(HERE/'validation_receipt.json')},indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'PDF':str(pdf),'TeX':str(tex),'PDF_bytes':pdf.stat().st_size},ensure_ascii=False))

if __name__=='__main__':main()
