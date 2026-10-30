"""Two-page TeX comparison, including protected-prefix and inactive-ablation limits."""
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import os
import shutil
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
NAME='SnAr_HPLC_R2_Comparison'
NAMES={'Pure_BO':'纯BO（qLogEI）','New_Agent':'旧新方案','New_Agent_R1':'R1',
       'New_Agent_R2':'R2','New_Agent_R2_Subset':'R2＋B4子集恢复'}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def fmt(value):return format(Decimal(str(value)).quantize(Decimal('.001'),rounding=ROUND_HALF_UP),'f')
def signed(value):return ('+' if value>=0 else '')+fmt(value)


def table(headers,rows,spec,bold_label=None):
    out=[r'\begingroup\small',r'\begin{tabularx}{\linewidth}{'+spec+'}',r'\toprule',
         ' & '.join(headers)+r'\\',r'\midrule']
    for row in rows:
        bold=row[0]==bold_label
        out.append(' & '.join(r'\textbf{'+str(value)+'}' if bold else str(value) for value in row)+r'\\')
    return '\n'.join(out+[r'\bottomrule',r'\end{tabularx}\endgroup'])


HEADER=r'''\documentclass[UTF8,fontset=fandol,10pt,a4paper]{ctexart}
\usepackage{fontspec}
\setmainfont[BoldFont=texgyretermes-bold.otf,ItalicFont=texgyretermes-italic.otf]{texgyretermes-regular.otf}
\usepackage[left=19mm,right=19mm,top=18mm,bottom=18mm,headheight=14pt,headsep=6mm]{geometry}
\usepackage{booktabs,tabularx,array,graphicx,fancyhdr}
\usepackage[hidelinks,unicode]{hyperref}
\hypersetup{pdftitle={SnAr与HPLC：R2及独立B4子集恢复消融}}
\newcolumntype{L}{>{\raggedright\arraybackslash}X}\newcolumntype{R}{>{\raggedleft\arraybackslash}X}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.15}
\linespread{1.08}\setlength{\emergencystretch}{2em}
\ctexset{section={format=\large\bfseries\heiti,beforeskip=1.3ex,afterskip=.7ex}}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small SnAr与HPLC：R2及B4子集恢复消融}
\fancyhead[R]{\small 固定协议・受保护前缀续跑}\fancyfoot[C]{\thepage}
\begin{document}
'''


def figures(curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles={'Pure_BO':('#555555','-',.85),'New_Agent':('#b57628','--',.9),
            'New_Agent_R1':('#c5607f','--',.9),'New_Agent_R2':('#d62728','-',1.6),
            'New_Agent_R2_Subset':('#7450ac',':',2.3)}
    labels={'Pure_BO':'Pure BO','New_Agent':'Old new agent','New_Agent_R1':'R1',
            'New_Agent_R2':'R2','New_Agent_R2_Subset':'R2 + subset (same path)'}
    for task,group in curves.items():
        with plt.rc_context({'font.size':7,'axes.labelsize':6.5,'axes.titlesize':8,
                             'xtick.labelsize':6.5,'ytick.labelsize':6.5,'pdf.fonttype':42}):
            fig,axes=plt.subplots(1,2,figsize=(6.75,2.4),layout='constrained')
            for method in ['Pure_BO','New_Agent','New_Agent_R1','New_Agent_R2_Subset','New_Agent_R2']:
                series=group[method];color,line,width=styles[method]
                for ax,key in zip(axes,['observed','best']):
                    ax.plot(range(1,21),series[key],color=color,linestyle=line,linewidth=width,label=labels[method],
                            marker='o' if method=='New_Agent_R2' else None,markersize=2,
                            zorder=4 if method=='New_Agent_R2' else 2)
            for ax,title in zip(axes,['Observed objective','Cumulative best']):
                ax.set_title(title);ax.set_xlabel('Total experiment (4 + 16)')
                ax.set_ylabel('STY (kg m^-3 h^-1)' if task=='SnAr' else 'Peak area (source units)')
                ax.set_xticks([1,4,8,12,16,20]);ax.grid(alpha=.2)
                ax.axvspan(.5,4.5,color='#ddd',alpha=.15)
            handles,text=axes[0].get_legend_handles_labels()
            fig.legend(handles,text,loc='outside lower center',ncol=3,fontsize=6.5)
            fig.savefig(HERE/f'{task}_summary_figure.pdf');plt.close(fig)


def main():
    os.chdir(ROOT);validation=read(HERE/'validation_receipt.json')
    assert validation['status']=='PASS' and validation['budgeted_added_choices_verified']==64
    results=read(HERE/'results.json');curves=read(HERE/'curves.json');figures(curves);text=[HEADER]
    for task in ['SnAr','HPLC']:
        if task=='HPLC':text.append(r'\newpage')
        group={row['method']:row for row in results['metrics'] if row['task']==task}
        new=group['New_Agent_R2'];state=read(ROOT/'data/runs/continuous_r2_20261030/runs'/(task+'_New_Agent_R2')/'state.json')
        text.extend([r'\begin{center}{\LARGE\bfseries\heiti '+('Summit SnAr' if task=='SnAr' else 'Olympus HPLC')+r'}\\[3pt]',
            r'{\small seed=0\quad 候选集与GP不变\quad 4次初始化＋16次查询\quad deepseek-flash}\end{center}',
            r'\section*{数据与R2执行逻辑}'])
        if task=='SnAr':
            text.append(r'固定4096个候选、4个连续变量；优化STY（kg\,m$^{-3}$\,h$^{-1}$），E-factor仅记录。知识条件为原始LLM初始建议，随后只使用各自观测修订。前2次追加查询复用R1未改动的早期机制，隔离后续规则差异。')
        else:
            text.append(r'固定1007组不同实测输入、6个连续变量；优化峰面积，仍使用原首条测量值，不平均或插值。知识条件为原始LLM初始建议。前3次追加查询完整复用R1的选点、预测、知识、KF与LLM回复，保护已有早期收益。')
        text.append(table(['修改','固定实现'],[
            ['刷新与部署分离','刷新当轮执行基础BO；新区域先冻结影子预测，有正的前瞻配对误差增益后，最早下一轮部署。失准提案继续暂停。'],
            ['跨版本负证据','重叠几何继承较低的实际部署KF系数，不靠修改边界重置；影子收益不计作实际部署收益。'],
            ['后续联合挑战','第6轮可保留BO；刷新验证优先。联合挑战始终受原qLogEI门限约束，不额外增加实验。'],
            ['独立B4消融','仅打开子集恢复开关：至少2次BO影子观测、正累计及最近增益、半/全强度选点一致。主R2维持原回退。'],
        ],r'>{\hsize=.35\hsize}L >{\hsize=1.65\hsize}L'))
        text.append(r'\section*{对比结果}')
        rows=[[NAMES[m],fmt(group[m]['SearchAUC']),fmt(group[m]['final_best']),
               group[m]['first_final_best_total']-4,signed(group[m]['SearchAUC']-group['Pure_BO']['SearchAUC'])] for m in NAMES]
        text.append(table(['方法','SearchAUC','最终最佳','首次最佳/追加轮','AUC较纯BO'],rows,
                    r'>{\hsize=1.6\hsize}L >{\hsize=.85\hsize}R >{\hsize=.85\hsize}R >{\hsize=.85\hsize}R >{\hsize=.85\hsize}R','R2'))
        text.extend([r'{\footnotesize SearchAUC为16次追加查询位置的累计最佳值均值；首次最佳针对各自轨迹最终值。两项任务目标单位不同，不合并排名。}',
            r'\begin{center}\includegraphics[width=\linewidth]{'+task+r'_summary_figure.pdf}\end{center}',
            r'R2的SearchAUC较R1 '+signed(new['SearchAUC']-group['New_Agent_R1']['SearchAUC'])+
            '，较旧新方案 '+signed(new['SearchAUC']-group['New_Agent']['SearchAUC'])+
            '，较纯BO '+signed(new['SearchAUC']-group['Pure_BO']['SearchAUC'])+'。'])
        if task=='SnAr':
            text.append('第3轮先执行BO，获得 '+fmt(state['decisions'][2]['raw_value'])+'；第4轮达到 '+fmt(new['final_best'])+'。比R1更早进入高值区域，但最终最佳仍低于纯BO。')
        else:
            text.append('第3轮已达到 '+fmt(new['final_best'])+'，因此完整保留R1的SearchAUC。原旧新方案与纯BO分别到第7、8轮才达到同一最佳；本轮未获得更高最终值。')
        pending=[event for event in state['decisions'] if event.get('pending_validation')]
        text.append('刷新轮次为 '+str([call['iteration'] for call in state['calls']])+'。新增'+str(len(pending))+
            '个提案均未通过影子检验，提升为活动知识0次；受保护前缀之后均执行BO。第6轮联合挑战因暂停状态而跳过。')
        text.append(r'\textbf{消融限制：}B4子集恢复触发0次，主方案与消融的完整轨迹相同；这不能证明恢复机制无效。本轮收益体现为避免错误引导及早期BO回退，尚无新知识成功重新部署的收益证据。')
        text.append(r'{\footnotesize 仅一个seed；两条R2主轨迹与两条独立消融合计64个追加预算位置，其中10个为前缀复用。原对照未重跑。一次JSON缓存等价校验技术修复已登记，参数、已完成前缀和原对照未改变。全部64轮通过独立复算。}')
    text.append(r'\end{document}')
    tex=HERE/(NAME+'.tex');tex.write_text('\n\n'.join(text),encoding='utf-8')
    temporary=ROOT/'outputs/continuous_r2_typeset';temporary.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,TECTONIC_CACHE_DIR=str(ROOT/'outputs/tex_cache'))
    result=subprocess.run([str(ROOT/'outputs/tex_tools/tectonic/tectonic.exe'),'--keep-logs','--outdir',str(temporary),str(tex)],
        cwd=HERE,env=env,text=True,capture_output=True,encoding='utf-8',errors='replace')
    (temporary/'compile_console.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
    if result.returncode:raise RuntimeError((result.stdout+'\n'+result.stderr)[-4000:])
    pdf=HERE/(NAME+'.pdf');shutil.copyfile(temporary/pdf.name,pdf)
    (HERE/'summary_provenance.json').write_text(json.dumps({'PDF_sha256':sha(pdf),'TeX_sha256':sha(tex),
        'results_sha256':sha(HERE/'results.json'),'validation_sha256':sha(HERE/'validation_receipt.json'),
        'builder_sha256':sha(__file__)},indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'PDF':str(pdf),'TeX':str(tex),'PDF_bytes':pdf.stat().st_size},ensure_ascii=False))


if __name__=='__main__':main()
