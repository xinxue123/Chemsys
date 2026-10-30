"""Two-page TeX/PDF comparison derived from frozen R4 outputs."""
from pathlib import Path
from decimal import Decimal,ROUND_HALF_UP
import hashlib
import json
import os
import shutil
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RUN=ROOT/'data/runs/continuous_r4_20261030'
NAME='SnAr_HPLC_R4_Comparison'
NAMES={'Pure_BO':'纯BO（qLogEI）','LLINBO_Justify':'LLINBO-Justify','ReasoningBO':'Reasoning BO',
    'New_Agent_R2':'R2','New_Agent_R3':'R3','New_Agent_R4':'R4'}


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def fmt(v):return format(Decimal(str(v)).quantize(Decimal('.001'),rounding=ROUND_HALF_UP),'f')
def signed(v):return ('+' if v>=0 else '')+fmt(v)


def table(headers,rows,spec):
    out=[r'\begingroup\small',r'\begin{tabularx}{\linewidth}{'+spec+'}',r'\toprule',' & '.join(headers)+r'\\',r'\midrule']
    for row in rows:out.append(' & '.join(r'\textbf{'+str(v)+'}' if row[0]=='R4' else str(v) for v in row)+r'\\')
    return '\n'.join(out+[r'\bottomrule',r'\end{tabularx}\endgroup'])


HEADER=r'''\documentclass[UTF8,fontset=fandol,10pt,a4paper]{ctexart}
\usepackage{fontspec}
\setmainfont[BoldFont=texgyretermes-bold.otf,ItalicFont=texgyretermes-italic.otf]{texgyretermes-regular.otf}
\usepackage[left=19mm,right=19mm,top=18mm,bottom=18mm,headheight=14pt,headsep=6mm]{geometry}
\usepackage{booktabs,tabularx,array,graphicx,fancyhdr}
\usepackage[hidelinks,unicode]{hyperref}
\hypersetup{pdftitle={SnAr与HPLC：R4独立提议与局部验证}}
\newcolumntype{L}{>{\raggedright\arraybackslash}X}\newcolumntype{R}{>{\raggedleft\arraybackslash}X}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.12}
\linespread{1.05}\setlength{\emergencystretch}{2em}
\ctexset{section={format=\large\bfseries\heiti,beforeskip=1.2ex,afterskip=.6ex}}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small SnAr与HPLC：R4独立提议与局部验证}
\fancyhead[R]{\small 固定协议・受保护前缀续跑}\fancyfoot[C]{\thepage}
\begin{document}
'''


def figures(curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles={'Pure_BO':('#555555','-',.9),'LLINBO_Justify':('#2676ad','-',.95),'ReasoningBO':('#b57932','--',1.),
        'New_Agent_R2':('#8c64a7','--',1.),'New_Agent_R3':('#ce665f',':',1.5),'New_Agent_R4':('#16836b','-',1.8)}
    labels={'Pure_BO':'Pure BO','LLINBO_Justify':'LLINBO-Justify','ReasoningBO':'Reasoning BO','New_Agent_R2':'R2','New_Agent_R3':'R3','New_Agent_R4':'R4'}
    for task,group in curves.items():
        with plt.rc_context({'font.size':7,'axes.labelsize':6.5,'axes.titlesize':8,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'pdf.fonttype':42}):
            fig,axes=plt.subplots(1,2,figsize=(6.75,2.3),layout='constrained')
            for method in NAMES:
                color,line,width=styles[method]
                for ax,key in zip(axes,['observed','best']):ax.plot(range(1,21),group[method][key],color=color,linestyle=line,linewidth=width,
                    label=labels[method],marker='o' if method=='New_Agent_R4' else None,markersize=2.1)
            for ax,title in zip(axes,['Observed objective','Cumulative best']):
                ax.set_title(title);ax.set_xlabel('Total experiment (4 + 16)');ax.set_xticks([1,4,8,12,16,20]);ax.grid(alpha=.2)
                ax.set_ylabel('STY (kg m^-3 h^-1)' if task=='SnAr' else 'Peak area (source units)')
                ax.axvspan(.5,4.5,color='#ddd',alpha=.18);ax.axvspan(4.5,8.5 if task=='SnAr' else 7.5,color='#b4d9cd',alpha=.18)
            handles,text=axes[0].get_legend_handles_labels();fig.legend(handles,text,loc='outside lower center',ncol=6,fontsize=5.8)
            fig.savefig(HERE/(task+'_summary_figure.pdf'));plt.close(fig)


def main():
    os.chdir(ROOT);audit=read(HERE/'validation_receipt.json');assert audit['status']=='PASS' and audit['budgeted_added_choices_verified']==32
    results=read(HERE/'results.json');figures(read(HERE/'curves.json'));text=[HEADER]
    for task,prefix in [('SnAr',4),('HPLC',3)]:
        if task=='HPLC':text.append(r'\newpage')
        group={r['method']:r for r in results['metrics'] if r['task']==task};r4=group['New_Agent_R4'];r3=group['New_Agent_R3']
        state=read(RUN/'runs'/(task+'_New_Agent_R4')/'state.json');calls=[c['query'] for c in state['calls']]
        text.extend([r'\begin{center}{\LARGE\bfseries\heiti '+('Summit SnAr' if task=='SnAr' else 'Olympus HPLC')+r'}\\[3pt]',
            r'{\small seed=0\quad 同一候选集、GP与4＋16预算\quad deepseek-flash}\end{center}',r'\section*{数据与R4执行逻辑}'])
        text.append(r'固定4096个候选、4个连续变量；优化STY（kg\,m$^{-3}$\,h$^{-1}$），E-factor仅记录。' if task=='SnAr'
            else '固定1007组不同实测输入、6个连续变量；优化峰面积，沿用原首条测量值，不平均或插值。')
        text.append('知识条件仍为共同LLM初始建议，仅使用自身观测更新。完整复用R2前'+str(prefix)+'次追加查询及状态，占用原16次预算；后续没有引用其他轨迹的实验结果。')
        text.append(table(['修改','实际规则'],[
            ['独立提议','先依据物理域与自身观测提议，再计算本轮GP并审查。隐藏所有GP预测、排名和BO参考；投影到原未查询候选池，每项任务最多4次调用。'],
            ['局部验证','作用轴控制局部仿射对比与背景条件；两次不同实验点的有效正证据才准入均值修正。最多3次主动验证，占用原实验预算与原门限。'],
            ['证据与恢复','统一账本同步新旧模型；按失败点适用范围与预测增量判断继承。暂停后继续影子检查；弱正、弱负均保留，不永久删除。'],
            ['选择顺序','已验证局部修正严格改选；否则主动验证；否则GP接纳的独立提议；否则BO。GP模型、方差、qLogEI和机会损失门限不改。'],
        ],r'>{\hsize=.35\hsize}L >{\hsize=1.65\hsize}L'))
        text.append(r'\section*{六种方法对比}')
        text.append(table(['方法','SearchAUC','最终最佳','首次最佳/追加轮'],[
            [NAMES[m],fmt(group[m]['SearchAUC']),fmt(group[m]['final_best']),group[m]['first_final_best_total']-4] for m in NAMES],
            r'>{\hsize=1.5\hsize}L >{\hsize=.85\hsize}R >{\hsize=.9\hsize}R >{\hsize=.75\hsize}R'))
        text.extend([r'{\footnotesize SearchAUC为16个追加查询位置的累计最佳目标均值；首次最佳针对各自最终值。灰色区域为初始化，浅绿色为受保护追加前缀。其他对照未重跑；LLM调用预算不同。}',
            r'\begin{center}\includegraphics[width=\linewidth]{'+task+r'_summary_figure.pdf}\end{center}'])
        if task=='SnAr':
            text.append(r'\textbf{结果：}R4的SearchAUC较R3、R2均增加 '+fmt(r4['SearchAUC']-r3['SearchAUC'])+
                '，较LLINBO-Justify增加 '+fmt(r4['SearchAUC']-group['LLINBO_Justify']['SearchAUC'])+
                '。第12次追加查询由BO找到 '+fmt(r4['final_best'])+'，最终最佳与纯BO、Reasoning BO相同，但到达更晚，AUC仍低于两者。')
            text.append('盲提议轮次为 '+str(calls)+'，2次接纳、2次拒绝，提议与当轮GP参考重合0次。第10轮两个模型各得到一次正证据；第11、12轮没有合格的预算内区分点，主动验证0次，局部均值部署0次。')
        else:
            text.append(r'\textbf{结果：}R4与R3、R2的SearchAUC及最终最佳相同；较纯BO的AUC高 '+
                fmt(r4['SearchAUC']-group['Pure_BO']['SearchAUC'])+'，较LLINBO-Justify低 '+
                fmt(group['LLINBO_Justify']['SearchAUC']-r4['SearchAUC'])+'。收益保留自第3轮保护前缀，没有新的峰值提升。')
            text.append('盲提议轮次为 '+str(calls)+'，2次接纳、2次拒绝，提议与GP参考重合0次。第4、6轮的局部预测被实测反驳；第14轮一个模型获得首次有效正证据，已不足两点准入。暂停后的旧模型仍完成了影子检查。')
        text.append(r'\textbf{归因限制：}主动局部验证、局部均值部署和相关模型的负证据继承激活均为0；继承一致性与恢复已通过行为测试，尚无本轮性能增量证据。SnAr改善表现为独立提议后BO轨迹改变，未做模块消融，不能归功于KF或验证调度。')
        text.append(r'{\footnotesize 两条轨迹共32个追加预算位置，其中7个复用、25个重新决策；8个新LLM逻辑请求。17项测试、全部32轮独立复算通过，既有2775个证据文件未变。仅一个seed及受保护前缀续跑，不作为跨seed或端到端优越性结论。}')
    text.append(r'\end{document}');tex=HERE/(NAME+'.tex');tex.write_text('\n\n'.join(text),encoding='utf-8')
    temporary=ROOT/'outputs/continuous_r4_typeset';temporary.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,TECTONIC_CACHE_DIR=str(ROOT/'outputs/tex_cache'))
    proc=subprocess.run([str(ROOT/'outputs/tex_tools/tectonic/tectonic.exe'),'--keep-logs','--outdir',str(temporary),str(tex)],
        cwd=HERE,env=env,text=True,capture_output=True,encoding='utf-8',errors='replace')
    (temporary/'compile_console.log').write_text(proc.stdout+'\n'+proc.stderr,encoding='utf-8')
    if proc.returncode:raise RuntimeError((proc.stdout+'\n'+proc.stderr)[-4000:])
    pdf=HERE/(NAME+'.pdf');shutil.copyfile(temporary/pdf.name,pdf)
    write(HERE/'summary_provenance.json',{'PDF_sha256':sha(pdf),'TeX_sha256':sha(tex),'results_sha256':sha(HERE/'results.json'),
        'validation_sha256':sha(HERE/'validation_receipt.json'),'builder_sha256':sha(__file__)})
    print(json.dumps({'PDF':str(pdf),'TeX':str(tex),'PDF_bytes':pdf.stat().st_size},ensure_ascii=False))


if __name__=='__main__':main()
