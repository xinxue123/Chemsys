"""Build a concise two-page R5 TeX/PDF from audited, immutable results."""
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import os
import shutil
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RUN=ROOT/'data/runs/continuous_r5_20261030'
NAME='SnAr_HPLC_R5_Comparison'
NAMES={'Pure_BO':'纯BO（qLogEI）','LLINBO_Justify':'LLINBO-Justify','ReasoningBO':'Reasoning BO',
       'New_Agent_R2':'R2','New_Agent_R3':'R3','New_Agent_R4':'R4','New_Agent_R5':'R5'}


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def fmt(v):return format(Decimal(str(v)).quantize(Decimal('.001'),rounding=ROUND_HALF_UP),'f')


def table(headers,rows,spec):
    out=[r'\begingroup\small',r'\begin{tabularx}{\linewidth}{'+spec+'}',r'\toprule',' & '.join(headers)+r'\\',r'\midrule']
    for row in rows:out.append(' & '.join(r'\textbf{'+str(v)+'}' if row[0]=='R5' else str(v) for v in row)+r'\\')
    return '\n'.join(out+[r'\bottomrule',r'\end{tabularx}\endgroup'])


HEADER=r'''\documentclass[UTF8,fontset=fandol,10pt,a4paper]{ctexart}
\usepackage{fontspec}
\setmainfont[BoldFont=texgyretermes-bold.otf,ItalicFont=texgyretermes-italic.otf]{texgyretermes-regular.otf}
\usepackage[left=19mm,right=19mm,top=18mm,bottom=18mm,headheight=14pt,headsep=6mm]{geometry}
\usepackage{booktabs,tabularx,array,graphicx,fancyhdr}
\usepackage[hidelinks,unicode]{hyperref}
\hypersetup{pdftitle={SnAr与HPLC：R5语义一致性与可执行验证}}
\newcolumntype{L}{>{\raggedright\arraybackslash}X}\newcolumntype{R}{>{\raggedleft\arraybackslash}X}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.10}
\linespread{1.03}\setlength{\emergencystretch}{2em}
\ctexset{section={format=\large\bfseries\heiti,beforeskip=1.1ex,afterskip=.5ex}}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small SnAr与HPLC：R5语义一致性与可执行验证}
\fancyhead[R]{\small 固定协议・受保护前缀续跑}\fancyfoot[C]{\thepage}
\begin{document}
'''


def figures(curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles={'Pure_BO':('#555555','-',.9),'LLINBO_Justify':('#2676ad','-',.95),'ReasoningBO':('#b57932','--',1.),
        'New_Agent_R2':('#8c64a7','--',1.),'New_Agent_R3':('#ce665f',':',1.4),'New_Agent_R4':('#16836b','--',1.3),
        'New_Agent_R5':('#172b4d','-',1.8)}
    for task,group in curves.items():
        with plt.rc_context({'font.size':7,'axes.labelsize':6.5,'axes.titlesize':8,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'pdf.fonttype':42}):
            fig,axes=plt.subplots(1,2,figsize=(6.75,2.25),layout='constrained')
            for method in NAMES:
                color,line,width=styles[method]
                label={'Pure_BO':'Pure BO','LLINBO_Justify':'LLINBO-Justify','ReasoningBO':'Reasoning BO'}.get(method,method.replace('New_Agent_',''))
                for ax,key in zip(axes,['observed','best']):ax.plot(range(1,21),group[method][key],color=color,linestyle=line,linewidth=width,
                    label=label,marker='o' if method=='New_Agent_R5' else None,markersize=2.)
            for ax,title in zip(axes,['Observed objective','Cumulative best']):
                ax.set_title(title);ax.set_xlabel('Total experiment (4 + 16)');ax.set_xticks([1,4,8,12,16,20]);ax.grid(alpha=.2)
                ax.set_ylabel('STY (kg m^-3 h^-1)' if task=='SnAr' else 'Peak area (source units)')
                ax.axvspan(.5,4.5,color='#ddd',alpha=.18);ax.axvspan(4.5,8.5 if task=='SnAr' else 7.5,color='#b4d9cd',alpha=.18)
            handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=7,fontsize=5.5)
            fig.savefig(HERE/(task+'_summary_figure.pdf'));plt.close(fig)


def main():
    os.chdir(ROOT);audit=read(HERE/'validation_receipt.json');assert audit['status']=='PASS' and audit['budgeted_added_choices_verified']==32
    results=read(HERE/'results.json');figures(read(HERE/'curves.json'));text=[HEADER]
    for task,prefix in [('SnAr',4),('HPLC',3)]:
        if task=='HPLC':text.append(r'\newpage')
        group={r['method']:r for r in results['metrics'] if r['task']==task};new=group['New_Agent_R5'];old=group['New_Agent_R4']
        state=read(RUN/'runs'/(task+'_New_Agent_R5')/'state.json');calls=[c['query'] for c in state['calls']]
        text.extend([r'\begin{center}{\LARGE\bfseries\heiti '+('Summit SnAr' if task=='SnAr' else 'Olympus HPLC')+r'}\\[3pt]',
            r'{\small seed=0\quad 同一候选集、GP与4＋16预算\quad deepseek-flash}\end{center}',r'\section*{数据、知识与R5逻辑}'])
        text.append(r'固定4096个候选、4个连续变量；优化STY（kg\,m$^{-3}$\,h$^{-1}$），E-factor仅记录。' if task=='SnAr'
            else '固定1007组不同实测输入、6个连续变量；优化峰面积，沿用原首条测量值，不平均或插值。')
        text.append('知识条件为共同LLM初始建议，仅依据自身观测更新。完整复用R2前'+str(prefix)+'次追加查询及状态，与R3、R4相同，计入16次追加预算。GP、qLogEI、原机会损失门限及同分规则不改。')
        text.append(table(['修改','R5实际规则'],[
            ['方向一致','LLM从真实未查询输入菜单选点，方向与参照逐项核对；身份映射距离为0。菜单只用自身观测与输入几何，最多3个高值参照，无GP信息。'],
            ['等待与恢复','只有现在存在合格验证点才延迟LLM；无候选则释放调用。反证按失败点适用性继承，取消全局重叠门槛；重复暂停不重置恢复时钟。'],
            ['后期与准入','每项最多4次新调用，第11轮前最多2次。主动验证仍占原实验预算；两处不同实验点的有效前瞻正证据才允许局部均值修正。'],
        ],r'>{\hsize=.35\hsize}L >{\hsize=1.65\hsize}L'))
        text.append(r'\section*{七种方法对比}')
        text.append(table(['方法','SearchAUC','最终最佳','首次最佳/追加轮'],[
            [NAMES[m],fmt(group[m]['SearchAUC']),fmt(group[m]['final_best']),group[m]['first_final_best_total']-4] for m in NAMES],
            r'>{\hsize=1.5\hsize}L >{\hsize=.85\hsize}R >{\hsize=.9\hsize}R >{\hsize=.75\hsize}R'))
        text.extend([r'{\footnotesize SearchAUC为16个追加位置的累计最佳目标均值；首次最佳针对各自最终值。灰色为初始化，浅绿色为保护前缀。六种旧对照未重跑；LLM调用预算不同。}',
            r'\begin{center}\includegraphics[width=\linewidth]{'+task+r'_summary_figure.pdf}\end{center}'])
        if task=='SnAr':
            text.append(r'\textbf{结果：}R5的SearchAUC较R4下降 '+fmt(old['SearchAUC']-new['SearchAUC'])+
                '，最终最佳下降 '+fmt(old['final_best']-new['final_best'])+'；与R2、R3持平，低于纯BO和Reasoning BO。保护第4轮峰值保留，未复现R4第12轮的后期最优。')
            text.append('新调用轮次 '+str(calls)+'，第8轮接纳点恰好等于BO参考，另3次超门限，因此新阶段实际选点全部为BO。第8、12、16轮的局部探针均为负增益，没有可准入的两点正证据。')
        else:
            text.append(r'\textbf{结果：}R5与R2、R3、R4的SearchAUC及最终最佳相同；较纯BO的AUC高 '+
                fmt(new['SearchAUC']-group['Pure_BO']['SearchAUC'])+'，较LLINBO-Justify低 '+
                fmt(group['LLINBO_Justify']['SearchAUC']-new['SearchAUC'])+'。第3轮保护峰值保留，后期未进一步提高。')
            text.append('新调用轮次 '+str(calls)+'，第4轮接受独立提议，其局部预测被实测反驳；另3次提议超门限。第14轮新模型仅获得一次有效正证据，第15、16轮预算集均为1点，不能完成主动确认。')
        text.append(r'\textbf{激活与限制：}8次提议语义均一致，每项均保留并使用2次后期调用；主动验证、局部均值部署、跨模型反证继承和重复暂停事件仍均为0。修复通过行为与轨迹核验，但没有这些模块的性能增量证据。')
        text.append(r'{\footnotesize 两项任务共32个追加位置，7个复用、25个重新决策；8个新逻辑调用。21项测试及全部32轮独立复算通过，3322个既有证据文件未变。仅一个seed的保护前缀续跑；不能宣称稳定优于基线。}')
    text.append(r'\end{document}');tex=HERE/(NAME+'.tex');tex.write_text('\n\n'.join(text),encoding='utf-8')
    temporary=ROOT/'outputs/continuous_r5_typeset';temporary.mkdir(parents=True,exist_ok=True)
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
