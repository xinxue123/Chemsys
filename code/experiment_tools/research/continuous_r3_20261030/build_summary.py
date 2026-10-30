"""Inspect frozen R3 outcomes and build a short, source-derived TeX report."""
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
from collections import Counter
import hashlib
import json
import os
import shutil
import subprocess
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RUN=ROOT/'data/runs/continuous_r3_20261030'
NAME='SnAr_HPLC_R3_Comparison'
NAMES={'Pure_BO':'纯BO（qLogEI）','LLINBO_Justify':'LLINBO-Justify',
       'ReasoningBO':'Reasoning BO','New_Agent_R2':'R2','New_Agent_R3':'R3'}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def fmt(v):return format(Decimal(str(v)).quantize(Decimal('.001'),rounding=ROUND_HALF_UP),'f')
def write(path,v):Path(path).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def diagnostics():
    result={}
    for task,prefix in [('SnAr',4),('HPLC',3)]:
        state=read(RUN/'runs'/(task+'_New_Agent_R3')/'state.json')
        r2=read(ROOT/'data/runs/continuous_r2_20261030/runs'/(task+'_New_Agent_R2')/'state.json')
        scale=read(ROOT/f'data/benchmarks/continuous_20261029/{task}/pool.json')['objective_scale']
        rounds=[];gains=Counter();positive=[];negative=[]
        for d in state['decisions'][prefix:]:
            t=d['iteration'];folder=RUN/'runs'/state['run_id']/'rounds'
            c=read(folder/f'{t:02d}_committed.json')
            with np.load(folder/f'{t:02d}_candidates.npz') as f:
                a={key:f[key].copy() for key in f.files}
            checks=d['local_feedback']['shadow_checks'];useful=[v for v in checks if v['informative']]
            for v in useful:
                gains['positive' if v['paired_SE_gain']>1e-12 else 'negative' if v['paired_SE_gain']< -1e-12 else 'zero']+=1
            if d['local_feedback']['actual_local_update']:
                v=d['local_feedback']['actual_local_update']['measurement']
                (positive if v['actual_paired_SE_gain']>0 else negative).append(t)
            rounds.append({'query':t,'eligible_count':int(a['eligible'].sum()),
                'GP_reference':d['reference_id'],'LLM_proposed':d['LLM_proposed_id'],'selected':d['selected_id'],
                'same_as_R2_selected':d['selected_id']==r2['decisions'][t-1]['selected_id'],
                'GP_review_loss':d['GP_review_loss'],'budget':d['budget'],
                'active_models_at_commit':sum(h['active_cap']>0 for h in c['local_models_at_commit'].values()),
                'full_pool_nonzero_local_shifts':int(np.sum(abs(a['local_shift_y'])>1e-12)),
                'eligible_nonzero_local_shifts':int(np.sum(a['eligible']&(abs(a['local_shift_y'])>1e-12))),
                'maximum_local_shift_raw':float(np.max(abs(a['local_shift_y'])))*scale,
                'local_ranking_changed':d['local_ranking_changed'],
                'all_shadow_checks':len(checks),'informative_shadow_checks':len(useful),
                'actual_owner':d['local_feedback']['actual_owner_hypothesis']})
        result[task]={'fresh_rounds':rounds,'authorization_events':state['authorization_events'],
            'informative_shadow_signs':dict(gains),'positive_actual_queries':positive,'negative_actual_queries':negative,
            'fresh_LLM_nonreference_queries':[d['iteration'] for d in state['decisions'][prefix:] if d['selected_id']!=d['reference_id']],
            'fresh_selected_equal_R2_count':sum(row['same_as_R2_selected'] for row in rounds),
            'withdrawals':sum(len(h['withdrawal_records']) for h in state['hypotheses'].values()),
            'failed_models':[{'id':key,'failure_query':h['last_actual_failure_query'],
                'withdrawal_queries':[v['query'] for v in h['withdrawal_records']],
                'later_informative_queries':[v['query'] for v in h['shadow_records'] if v['informative'] and v['query']>h['last_actual_failure_query']],
                'final_status':h['status']} for key,h in state['hypotheses'].items() if h['last_actual_failure_query']>=0]}
    write(HERE/'round_diagnostics.json',result);return result


def table(headers,rows,spec):
    out=[r'\begingroup\small',r'\begin{tabularx}{\linewidth}{'+spec+'}',r'\toprule',
         ' & '.join(headers)+r'\\',r'\midrule']
    for row in rows:
        out.append(' & '.join(r'\textbf{'+str(v)+'}' if row[0]=='R3' else str(v) for v in row)+r'\\')
    return '\n'.join(out+[r'\bottomrule',r'\end{tabularx}\endgroup'])


HEADER=r'''\documentclass[UTF8,fontset=fandol,10pt,a4paper]{ctexart}
\usepackage{fontspec}
\setmainfont[BoldFont=texgyretermes-bold.otf,ItalicFont=texgyretermes-italic.otf]{texgyretermes-regular.otf}
\usepackage[left=19mm,right=19mm,top=18mm,bottom=18mm,headheight=14pt,headsep=6mm]{geometry}
\usepackage{booktabs,tabularx,array,graphicx,fancyhdr}
\usepackage[hidelinks,unicode]{hyperref}
\hypersetup{pdftitle={SnAr与HPLC：R3局部知识验证与对比}}
\newcolumntype{L}{>{\raggedright\arraybackslash}X}\newcolumntype{R}{>{\raggedleft\arraybackslash}X}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.12}
\linespread{1.05}\setlength{\emergencystretch}{2em}
\ctexset{section={format=\large\bfseries\heiti,beforeskip=1.2ex,afterskip=.6ex}}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small SnAr与HPLC：R3局部知识验证}
\fancyhead[R]{\small 固定协议・受保护前缀续跑}\fancyfoot[C]{\thepage}
\begin{document}
'''


def figures(curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles={'Pure_BO':('#555555','-',.9),'LLINBO_Justify':('#2676ad','-',.95),
            'ReasoningBO':('#b57932','--',1.05),'New_Agent_R2':('#8b62a5','--',1.5),'New_Agent_R3':('#d62728','-',1.7)}
    labels={'Pure_BO':'Pure BO','LLINBO_Justify':'LLINBO-Justify','ReasoningBO':'Reasoning BO','New_Agent_R2':'R2','New_Agent_R3':'R3'}
    for task,group in curves.items():
        with plt.rc_context({'font.size':7,'axes.labelsize':6.5,'axes.titlesize':8,
                             'xtick.labelsize':6.5,'ytick.labelsize':6.5,'pdf.fonttype':42}):
            fig,axes=plt.subplots(1,2,figsize=(6.75,2.3),layout='constrained')
            for method in NAMES:
                color,line,width=styles[method]
                for ax,key in zip(axes,['observed','best']):
                    ax.plot(range(1,21),group[method][key],color=color,linestyle=line,linewidth=width,
                            label=labels[method],marker='o' if method=='New_Agent_R3' else None,markersize=2)
            for ax,title in zip(axes,['Observed objective','Cumulative best']):
                ax.set_title(title);ax.set_xlabel('Total experiment (4 + 16)')
                ax.set_ylabel('STY (kg m^-3 h^-1)' if task=='SnAr' else 'Peak area (source units)')
                ax.set_xticks([1,4,8,12,16,20]);ax.grid(alpha=.2);ax.axvspan(.5,4.5,color='#ddd',alpha=.18)
            handles,labels_=axes[0].get_legend_handles_labels()
            fig.legend(handles,labels_,loc='outside lower center',ncol=5,fontsize=6)
            fig.savefig(HERE/f'{task}_summary_figure.pdf');plt.close(fig)


def main():
    os.chdir(ROOT);audit=read(HERE/'validation_receipt.json');assert audit['status']=='PASS'
    assert audit['budgeted_added_choices_verified']==32
    result=read(HERE/'results.json');diag=diagnostics();figures(read(HERE/'curves.json'));text=[HEADER]
    for task,prefix in [('SnAr',4),('HPLC',3)]:
        if task=='HPLC':text.append(r'\newpage')
        group={v['method']:v for v in result['metrics'] if v['task']==task};new=group['New_Agent_R3'];d=diag[task]
        text.extend([r'\begin{center}{\LARGE\bfseries\heiti '+('Summit SnAr' if task=='SnAr' else 'Olympus HPLC')+r'}\\[3pt]',
            r'{\small seed=0\quad 同一候选集、GP与4＋16预算\quad deepseek-flash}\end{center}',
            r'\section*{数据与R3执行逻辑}'])
        text.append('固定4096个候选、4个连续变量；优化STY（kg\,m$^{-3}$\,h$^{-1}$），E-factor仅记录。' if task=='SnAr'
                    else '固定1007组不同实测输入、6个连续变量；优化峰面积，沿用原首条测量值，不平均或插值。')
        text.append('知识条件为原LLM初始建议；完整复用R2前'+str(prefix)+'次追加查询及状态，计入原16次预算。后续只使用自身已观测数据；旧KF系数归档，R3局部模型重新积累前瞻证据。')
        text.append(table(['步骤','固定实现'],[
            ['提出与审查','每轮LLM从提议卡选一个候选并给局部响应预测；GP仍评分完整候选池。原qLogEI机会损失门限独立审查，高LLM预测不能绕过。'],
            ['反馈后修正','第32近输入界定局部范围；未验证权重为0。前瞻影子预测有正增益，最早下一轮准入，系数上限0.25或0.5；GP方差不改。'],
            ['恢复通路','实际与影子KF分账，失败只隔离单模型并继续影子评估；失败后两次新正反馈才能恢复，重叠失败模型继承更严格准入。'],
            ['实际选择','已验证局部修正若严格改变预算内最优点则优先；否则使用GP接纳的LLM提议，再回退BO。所有实验占用原预算。'],
        ],r'>{\hsize=.35\hsize}L >{\hsize=1.65\hsize}L'))
        text.append(r'\section*{对比结果}')
        text.append(table(['方法','SearchAUC','最终最佳','首次最佳/追加轮'],[
            [NAMES[m],fmt(group[m]['SearchAUC']),fmt(group[m]['final_best']),group[m]['first_final_best_total']-4] for m in NAMES],
            r'>{\hsize=1.5\hsize}L >{\hsize=.85\hsize}R >{\hsize=.9\hsize}R >{\hsize=.75\hsize}R'))
        text.extend([r'{\footnotesize SearchAUC为16个追加查询位置的累计最佳目标值均值；首次最佳针对各自轨迹最终值。对照全部读取旧封存结果；仅实验预算相同，LLM调用预算不同。}',
            r'\begin{center}\includegraphics[width=\linewidth]{'+task+r'_summary_figure.pdf}\end{center}'])
        if task=='SnAr':
            text.append(r'\textbf{逐轮发现：}第8、9、13、16轮部署局部均值修正，其中第8、9、16轮降低预测误差，第13轮增加误差并隔离该模型（系数0.250降至0.237）。这4轮均未因局部修正改变选点。LLM仅第9、11轮偏离当轮BO参考，保护前缀后最佳值无增长。')
            text.append('恢复通路单元测试通过，但该失败模型随后被LLM撤销，失败后没有新的有信息反馈，实际恢复0次。保护前4轮后，即使第5轮立即找到池内最优值，SearchAUC上限也为'+fmt(new['protected_prefix_SearchAUC_upper_bound'])+'，低于纯BO '+fmt(group['Pure_BO']['SearchAUC'])+'；该上限仅作事后分析。')
        else:
            text.append(r'\textbf{逐轮发现：}13个新局部模型中，7次有信息影子检验全部为负增益，获准均值修正0次；这次拒绝有数值反证。LLM第4、11轮改选，其余11次提议与当轮BO参考相同。后续未超过第3轮已获得的最佳值。')
            text.append('R3较纯BO的AUC高 '+fmt(new['SearchAUC']-group['Pure_BO']['SearchAUC'])+'，但全部来自受保护早期轨迹；较LLINBO-Justify低 '+fmt(group['LLINBO_Justify']['SearchAUC']-new['SearchAUC'])+'，最终最佳也较低。没有证据将保住的收益归因于新局部修正。')
        text.append(r'\textbf{结论与范围：}R3与R2的SearchAUC及最终最佳相同，尚无搜索效率增量。25次新LLM请求、32个预算位置全部独立复算通过，旧2210个证据文件未变；仅一个seed及受保护前缀续跑，不作为跨seed或端到端优越性结论。')
    text.append(r'\end{document}');tex=HERE/(NAME+'.tex');tex.write_text('\n\n'.join(text),encoding='utf-8')
    temporary=ROOT/'outputs/continuous_r3_typeset';temporary.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,TECTONIC_CACHE_DIR=str(ROOT/'outputs/tex_cache'))
    completed=subprocess.run([str(ROOT/'outputs/tex_tools/tectonic/tectonic.exe'),'--keep-logs','--outdir',str(temporary),str(tex)],
        cwd=HERE,env=env,text=True,capture_output=True,encoding='utf-8',errors='replace')
    (temporary/'compile_console.log').write_text(completed.stdout+'\n'+completed.stderr,encoding='utf-8')
    if completed.returncode:raise RuntimeError((completed.stdout+'\n'+completed.stderr)[-4000:])
    pdf=HERE/(NAME+'.pdf');shutil.copyfile(temporary/pdf.name,pdf)
    write(HERE/'summary_provenance.json',{'PDF_sha256':sha(pdf),'TeX_sha256':sha(tex),
        'results_sha256':sha(HERE/'results.json'),'validation_sha256':sha(HERE/'validation_receipt.json'),'builder_sha256':sha(__file__)})
    print(json.dumps({'PDF':str(pdf),'TeX':str(tex),'PDF_bytes':pdf.stat().st_size},ensure_ascii=False))


if __name__=='__main__':main()
