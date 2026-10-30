"""Two-page TeX/PDF display report from the sealed continuous experiment."""
from pathlib import Path
from decimal import Decimal,ROUND_HALF_UP
from hashlib import sha256
import json
import os
import shutil
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
NAME='SnAr_HPLC_Summary'
NAMES={'Pure_BO':'纯BO（qLogEI）','LGBO_region':'区域LGBO','Gate_only':'仅预算门限','B4_hard':'硬B4',
       'B4_soft':'软B4＋区域注入','LLINBO_Justify':'LLINBO-Justify','ReasoningBO':'Reasoning BO','New_Agent':'新方案Agent'}
METHODS=list(NAMES)
AXIS_LABELS={'tau':'停留时间/min','equiv_pldn':'吡咯烷当量','conc_dfnb':'底物浓度/(mol/L)','temperature':'温度/℃',
             'sample_loop':'样品环体积/mL','additional_volume':'附加体积/mL','tubing_volume':'管路体积/mL','sample_flow':'样品流速/(mL/min)'}

def fmt(v,n=3):return format(Decimal(str(v)).quantize(Decimal(1).scaleb(-n),rounding=ROUND_HALF_UP),'f')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def digest(p):return sha256(Path(p).read_bytes()).hexdigest()
def esc(s):return str(s).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
def table(headers,rows,spec):
    lines=[r'\begingroup\small',r'\begin{tabularx}{\linewidth}{'+spec+'}',r'\toprule',' & '.join(headers)+r'\\',r'\midrule']
    for row in rows:
        lines.append(' & '.join((r'\textbf{'+str(v)+'}') if row[0]=='新方案Agent' else str(v) for v in row)+r'\\')
    return '\n'.join(lines+[r'\bottomrule',r'\end{tabularx}',r'\endgroup'])

def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    methods=['Pure_BO','LGBO_region','LLINBO_Justify','ReasoningBO','New_Agent']
    colors=['#777777','#ce8a2e','#3078ab','#9e5a5a','#d32f2f']
    for task in ('SnAr','HPLC'):
        p=read(ROOT/'data/benchmarks/continuous_20261029'/task/'pool.json')
        with plt.rc_context({'font.size':7,'axes.labelsize':6.5,'axes.titlesize':8,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'pdf.fonttype':42,'ps.fonttype':42}):
            fig,axes=plt.subplots(1,2,figsize=(6.75,2.4),layout='constrained')
            for method,color in zip(methods,colors):
                state=read(ROOT/'data/runs/continuous_20261029/runs'/(task+'_'+method)/'state.json')
                y=np.array([h['y']*p['objective_scale'] for h in state['history']])
                for ax,values in zip(axes,(y,np.maximum.accumulate(y))):
                    ax.plot(range(1,21),values,label=method,color=color,linewidth=1.4 if method=='New_Agent' else .8,
                        marker='o' if method=='New_Agent' else None,markersize=2)
            for ax,title in zip(axes,('Observed objective','Cumulative best')):
                ax.set_title(title);ax.set_xlabel('Total experiment (4 + 16)');ax.set_ylabel('STY (kg m^-3 h^-1)' if task=='SnAr' else 'Peak area (source units)')
                ax.set_xticks([1,4,8,12,16,20]);ax.grid(alpha=.2);ax.axvspan(.5,4.5,alpha=.12,color='#dddddd')
            handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=6.5)
            fig.savefig(HERE/f'{task}_figure.pdf');plt.close(fig)

HEADER=r'''\documentclass[UTF8,fontset=fandol,10pt,a4paper]{ctexart}
\usepackage{fontspec}
\setmainfont[BoldFont=texgyretermes-bold.otf,ItalicFont=texgyretermes-italic.otf]{texgyretermes-regular.otf}
\usepackage[left=19mm,right=19mm,top=18mm,bottom=18mm,headheight=14pt,headsep=6mm]{geometry}
\usepackage{booktabs,tabularx,array,graphicx,fancyhdr}
\usepackage[hidelinks,unicode]{hyperref}
\hypersetup{pdftitle={SnAr与HPLC实验结果简报}}
\newcolumntype{L}{>{\raggedright\arraybackslash}X}
\newcolumntype{R}{>{\raggedleft\arraybackslash}X}
\setlength{\parindent}{0pt}\setlength{\parskip}{4pt}\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.12}\linespread{1.06}\setlength{\emergencystretch}{2em}
\ctexset{section={format=\large\bfseries\heiti,beforeskip=1ex,afterskip=.6ex}}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small SnAr与HPLC实验结果简报}
\fancyhead[R]{\small 固定流程迁移检验}\fancyfoot[C]{\thepage}
\begin{document}
'''

def main():
    os.chdir(ROOT)
    receipt=read(HERE/'validation_receipt.json');assert receipt['status']=='PASS' and receipt['queries_checked']==256
    results=read(HERE/'results.json')['metrics'];text=[HEADER]
    figures()
    for task in ('SnAr','HPLC'):
        if task=='HPLC':text.append(r'\newpage')
        p=read(ROOT/'data/benchmarks/continuous_20261029'/task/'pool.json')
        ready=read(ROOT/'data/runs/continuous_20261029/initialization'/task/'ready.json')
        group={r['method']:r for r in results if r['task']==task}
        text += [r'\begin{center}{\LARGE\bfseries\heiti '+('Summit SnAr' if task=='SnAr' else 'Olympus HPLC')+r'}\\[3pt]',
                 r'{\small seed=0\quad 共同4次初始化＋16次逐点查询\quad deepseek-flash}\end{center}']
        if task=='SnAr':
            text += [r'\section*{数据集与执行设置}',
                r'官方动力学模拟器；调节停留时间、吡咯烷当量、底物入口浓度和温度。每轴8个等距水平，固定4096个候选；噪声设为0。主目标为最大化时空收率STY（kg\,m$^{-3}$\,h$^{-1}$），同时记录E-factor，本轮未做双目标优化。']
        else:
            text += [r'\section*{数据集与执行设置}',
                r'六个连续进样参数：样品环体积、附加体积、管路体积、样品流速、推动速度和等待时间。1386条实测记录对应1007组不同输入；326组重复输入的峰面积不一致。沿用官方LGBO精确匹配取首条规则，固定1007个候选，不插值、不挑选最佳重复值。目标为最大化峰面积（来源任意单位），不等同于反应产率。']
        text += [r'所有方法共享候选池、初始4点及初始知识。全连续域统一使用SingleTaskGP、MC512 qLogEI和相同并列裁决；新方案沿用区域引导、KF、B4、信任恢复与事件刷新，普通单轴反例也占用16次预算。不启用分子核及类别联合分支。',
                 r'\section*{采用的知识条件}']
        kr=[]
        for h in ready['knowledge']['clauses']:
            a=h['axis'];lo,hi=p['bounds'][a];physical=[lo+z*(hi-lo) for z in h['bounds']]
            digits=5 if task=='HPLC' and ('volume' in a or a=='sample_loop') else 3
            kr.append([AXIS_LABELS[a],fmt(h['bounds'][0],2)+'--'+fmt(h['bounds'][1],2),fmt(physical[0],digits)+'--'+fmt(physical[1],digits)])
        text.append(table(['变量','归一化偏好区间','对应物理区间'],kr,'L R R'))
        text.append(r'{\footnotesize 每个任务仅一组初始知识，由LLM根据共同4个观测生成，置信参数固定0.7；均为待检验偏好，没有人工正确/错误条件。新方案可事件刷新；固定区域对照保留初始知识。}')
        if task=='HPLC':text.append(r'{\footnotesize 推动速度与等待时间未设置初始偏好。}')
        text += [r'\section*{对比结果}',table(['方法','SearchAUC','最终最佳','首次最佳/总次序','AUC较纯BO'],
            [[NAMES[m],fmt(group[m]['SearchAUC']),fmt(group[m]['final_best']),group[m]['first_final_best_total'],('+' if group[m]['SearchAUC']>=group['Pure_BO']['SearchAUC'] else '')+fmt(group[m]['SearchAUC']-group['Pure_BO']['SearchAUC'])] for m in METHODS],
            r'>{\hsize=1.6\hsize}L >{\hsize=.85\hsize}R >{\hsize=.85\hsize}R >{\hsize=.85\hsize}R >{\hsize=.85\hsize}R')]
        text.append(r'{\footnotesize SearchAUC是第5--20次实验位置累计最佳目标值的均值，保留原始目标单位。首次最佳针对各轨迹自身最终最佳，不能单独据此排名。}')
        text.append(r'\begin{center}\includegraphics[width=\linewidth]{'+task+r'_figure.pdf}\end{center}')
        n=group['New_Agent'];b=group['Pure_BO'];r=group['ReasoningBO']
        direction=lambda d:'高于' if d>=0 else '低于'
        text.append('新方案的SearchAUC'+direction(n['SearchAUC']-b['SearchAUC'])+'纯BO '+fmt(abs(n['SearchAUC']-b['SearchAUC']))+'，'+direction(n['SearchAUC']-r['SearchAUC'])+'Reasoning BO '+fmt(abs(n['SearchAUC']-r['SearchAUC']))+'。新方案执行'+str(n['controlled_tests'])+'次受控检验、'+str(n['refreshes'])+'次知识刷新。')
        if task=='HPLC':text.append('新方案最终最佳峰面积仍低于Reasoning BO与LLINBO-Justify。')
        if task=='SnAr':
            text.append(r'{\footnotesize 必要适配：官方LGBO区域组件、LLINBO Justify分支及Reasoning BO提示/置信排序；统一有限池、qLogEI与5选1预算，详见protocol.json。原95\%产率停止检验阈值不适用于STY和峰面积，故关闭。}')
        else:
            text.append(r'{\footnotesize HPLC候选池没有精确单轴邻居，故未执行反例检验。单初始化仅支持描述性比较。两个目标的固定归一化尺度为13000、10000；KF与事件常数未按本轮成绩调参。全部256次选择已审计；既有P0/P1结果未修改。}')
        text.append(r'{\footnotesize 来源：'+(r'\href{https://gosummit.readthedocs.io/en/latest/experiments_benchmarks/implemented_benchmarks.html}{Summit SnAr官方文档}。' if task=='SnAr' else r'\href{https://github.com/aspuru-guzik-group/olympus/tree/440b6b58ebfcaa2391cff7e94b570fb4fda98d68/src/olympus/datasets/dataset_hplc}{Olympus HPLC固定版本}；\href{https://github.com/Xinzhe309/LGBO/blob/main/docs/benchmark_data.md}{LGBO数据协议}。')+r'}')
    text += [r'\end{document}']
    tex=HERE/(NAME+'.tex');tex.write_text('\n\n'.join(text),encoding='utf-8')
    tmp=ROOT/'outputs/continuous_typeset';tmp.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,TECTONIC_CACHE_DIR=str(ROOT/'outputs/tex_cache'))
    cmd=[str(ROOT/'outputs/tex_tools/tectonic/tectonic.exe'),'--keep-logs','--outdir',str(tmp),str(tex)]
    result=subprocess.run(cmd,cwd=HERE,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace')
    (tmp/'compile_console.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
    if result.returncode:raise RuntimeError((result.stdout+'\n'+result.stderr)[-5000:])
    pdf=HERE/(NAME+'.pdf');shutil.copyfile(tmp/(NAME+'.pdf'),pdf)
    provenance={'PDF_sha256':digest(pdf),'TeX_sha256':digest(tex),'results_sha256':digest(HERE/'results.json'),
        'validation_receipt_sha256':digest(HERE/'validation_receipt.json'),'compiler':'official Tectonic0.17.0','no_contents':True}
    (HERE/'summary_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'PDF':str(pdf),'TeX':str(tex),'PDF_bytes':pdf.stat().st_size}),flush=True)

if __name__=='__main__':main()
