"""Inspect complete evidence and render the registered batch-level metrics."""
from collections import Counter
from pathlib import Path
import csv
import json
import math

import numpy as np

from . import reasoning_bo_data as data
from . import reasoning_bo_engine as engine
from . import reasoning_bo_study as study
from .suzuki_domain import digest


def states():
    rows=[]
    for spec in data.read(study.ROOT/'manifest.json'):
        path=study.ROOT/'runs'/spec['run_id']/'state.json'
        rows.append(data.read(path) if path.exists() else {**spec,'status':'NOT_STARTED','history':[],'decisions':[]})
    return rows


def metric(s):
    if s['status']!='COMPLETE':return {}
    ds=s['decisions'];curve=np.array([d['incumbent_after']*100 for d in ds])
    initial=max(h['y'] for h in s['history'][:3])*100
    best=float(curve[-1]);full=np.r_[initial,curve]
    return {'SearchAUC':float(curve.mean()),'final_best_percent':best,
        'initial_best_percent':initial,'first_final_best_batch':int(np.flatnonzero(full>=best-1e-10)[0]),
        'mean_queried_yield_percent':float(np.mean([h['y'] for h in s['history'][3:]])*100),
        'refresh_calls':len(s.get('calls',[]))}


def write_csv(path,rows,fields):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)


def build():
    p=study.check_protocol();ss=states();rows=[];summary=[];paired=[]
    for s in ss:
        rows.append({k:s[k] for k in ('task','seed','method','run_id','status')} | metric(s))
    write_csv(study.OUT/'metrics.csv',rows,['task','seed','method','run_id','status','SearchAUC','final_best_percent',
        'initial_best_percent','first_final_best_batch','mean_queried_yield_percent','refresh_calls'])
    bykey={(s['task'],s['seed'],s['method']):s for s in ss}
    for task in data.SPECS:
        for method in engine.METHODS:
            selected=[metric(s) for s in ss if s['task']==task and s['method']==method and s['status']=='COMPLETE']
            row={'task':task,'method':method,'completed':len(selected),'planned':10}
            for key in ('SearchAUC','final_best_percent','first_final_best_batch','initial_best_percent'):
                vals=[r[key] for r in selected]
                row[key+'_mean']=float(np.mean(vals)) if vals else None
                row[key+'_SD']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
            summary.append(row)
        for control in ('ReasoningBO','BO_shared','Gate_only','LGBO_region','BO_q3_shared','BO_q3_Sobol'):
            deltas=[]
            for seed in study.SEEDS:
                a,b=bykey[(task,seed,'New_R2')],bykey[(task,seed,control)]
                if a['status']=='COMPLETE' and b['status']=='COMPLETE':
                    d=metric(a)['SearchAUC']-metric(b)['SearchAUC'];deltas.append(d)
                    paired.append({'task':task,'seed':seed,'comparison':'New_R2 - '+control,'SearchAUC_delta_pp':d})
    write_csv(study.OUT/'summary.csv',summary,['task','method','completed','planned',
        *[k for name in ('SearchAUC','final_best_percent','first_final_best_batch','initial_best_percent') for k in (name+'_mean',name+'_SD')]])
    write_csv(study.OUT/'paired_deltas.csv',paired,['task','seed','comparison','SearchAUC_delta_pp'])
    rounds=[]
    for s in ss:
        for d in s['decisions']:
            rounds.append({'task':s['task'],'method':s['method'],'seed':s['seed'],
                **{k:d[k] for k in ('iteration','incumbent_before','incumbent_after','incumbent_gain','amplitude','fallback',
                    'budget','eligible_batches','opportunity_loss')},'selected_ids':';'.join(d['selected_ids']),
                'queried_yields_percent':';'.join(str(y*100) for y in d['observed_y']),
                'batch_mean_yield_percent':float(np.mean(d['observed_y'])*100),
                'retained':';'.join(d['retained'])})
    write_csv(study.OUT/'rounds.csv',rounds,['task','method','seed','iteration','selected_ids','queried_yields_percent',
        'batch_mean_yield_percent','incumbent_before','incumbent_after','incumbent_gain','amplitude','fallback',
        'budget','eligible_batches','opportunity_loss','retained'])
    status='COMPLETE' if all(s['status']=='COMPLETE' for s in ss) else 'INCOMPLETE'
    result={'status':status,'protocol_sha256':p['protocol_sha256'],'completed':sum(s['status']=='COMPLETE' for s in ss),
        'planned':len(ss),'summary':summary,'paired_deltas':paired,
        'limits':['Same-model/common-Mixed-GP comparison, not numerical reproduction of QWQ/Ax published scores.',
            'Doyle is an official supplementary task, distinct from the Direct Arylation main-table task.',
            'SearchAUC is our explicitly defined batch-incumbent mean; not asserted identical to an undocumented paper log-AUC.',
            'Our regional methods share the BO5-to3 action space in this experiment; this is a registered batch adaptation.']}
    data.write(study.OUT/'results.json',result)
    text=['# 第二实验：Reasoning BO 对照','',f'状态：**{status}**；完成 {result["completed"]}/{len(ss)} 条轨迹。',
        '', '每套数据10个预定种子；每条3次初始化＋30批×3次＝93次查询。批内不反馈。',
        'Suzuki为作者提供的3696条子集，Doyle为3955条完整实测池。底物是优化变量。',
        '使用deepseek-flash、全类别Mixed GP与官方实际导入的qLogNEI。',
        'Reasoning BO执行官方提示模板及稳定置信度排序；新方案为已登记的同时批次扩展。',
        '区域LGBO/门限/新方案均在同一基础BO生成的5个候选的10个三元子集中决策。',
        'BO_shared隔离5选3知识增量；BO_q3_shared隔离该选择空间；BO_q3_Sobol另用于观察初始化差异。','',
        'SearchAUC：30个追加批次结束后的累计最佳产率的均值（百分点）。同时批次不人为排序。',
        '未完成时表格仅列可用完成数，禁止以缺失轨迹推断完整胜负。']
    for task in data.SPECS:
        text+=['',f'## {task}','', '| 方法 | 完成/计划 | SearchAUC 均值±SD | 最终最佳产率均值±SD | 首达最终最佳批次均值 |',
            '|---|---:|---:|---:|---:|']
        def show(mean,sd):
            return '—' if mean is None else f'{mean:.4f}'+(f' ± {sd:.4f}' if sd is not None else '')
        for row in summary:
            if row['task']!=task:continue
            text.append(f"| {row['method']} | {row['completed']}/10 | {show(row['SearchAUC_mean'],row['SearchAUC_SD'])} | "
                f"{show(row['final_best_percent_mean'],row['final_best_percent_SD'])} | {show(row['first_final_best_batch_mean'],None)} |")
    text+=['','全部必要适配见 [protocol.json](protocol.json) 的 official_adaptations。',
        '未对原论文的QWQ模型、Ax连续松弛、知识图谱或GRPO训练结果作数值复现声明。','',
        '参考：[论文](https://onlinelibrary.wiley.com/doi/10.1002/mgea.70086)、[官方代码](https://github.com/little1d/Reasoning-BO)。']
    (study.OUT/'report.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    if any(s['status']=='COMPLETE' for s in ss):plot(ss)
    return {'status':status,'completed':result['completed'],'planned':len(ss),'report':str(study.OUT/'report.md')}


def plot(ss):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    colors={'ReasoningBO':'#e38128','New_R2':'#176bad','Gate_only':'#8853a5','LGBO_region':'#2a9776',
        'BO_shared':'#333333','BO_q3_shared':'#88929c','BO_q3_Sobol':'#aaadaf'}
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True)
    for row,task in enumerate(data.SPECS):
        for method in engine.METHODS:
            selected=[s for s in ss if s['task']==task and s['method']==method and s['status']=='COMPLETE']
            if not selected:continue
            x=np.arange(31)
            best=np.array([[max(h['y'] for h in s['history'][:3])*100]+[d['incumbent_after']*100 for d in s['decisions']] for s in selected])
            actual=np.array([[np.mean([h['y'] for h in s['history'][:3]])*100]+[np.mean(d['observed_y'])*100 for d in s['decisions']] for s in selected])
            for col,values in enumerate((actual,best)):
                mean=values.mean(0)
                axes[row,col].plot(x,mean,color=colors[method],label=f'{method} (n={len(selected)})',
                    ls='--' if method.startswith('BO_q3') else '-',lw=1.3 if method.startswith('BO_q3') else 1.8)
                if len(selected)>1 and col==1:
                    sem=values.std(0,ddof=1)/math.sqrt(len(selected))
                    axes[row,col].fill_between(x,mean-sem,mean+sem,color=colors[method],alpha=.1,lw=0)
                axes[row,col].set_title(task.replace('_',' ')+' — '+('batch mean yield' if col==0 else 'cumulative best'))
                axes[row,col].set_ylim(0,103);axes[row,col].grid(alpha=.2)
                axes[row,col].xaxis.set_major_locator(MaxNLocator(integer=True,nbins=7))
                axes[row,col].set_ylabel('Reported yield (%)')
        axes[row,0].set_xlabel('Batch (0 = initialization; 3 experiments per batch)')
        axes[row,1].set_xlabel('Batch (0 = initialization; 3 experiments per batch)')
    handles,labels=axes[0,1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=9)
    fig.suptitle('M2: shared empirical domains / Mixed GP / qLogNEI',fontsize=13)
    fig.tight_layout(rect=[0,.11,1,.96]);fig.savefig(study.OUT/'comparison.png',dpi=180);plt.close(fig)


def validate():
    p=study.check_protocol();ss=states();counts=Counter();max_kf_error=0.
    for s in ss:
        if not s['history']:continue
        pool=data.load_pool(s['task']);lookup={key:i for i,key in enumerate(pool['ids'])}
        values=np.load(data.ROOT/s['task']/'oracle.npz')['values']
        init=data.read(study.ROOT/'initialization'/f"{s['task']}_s{s['seed']:02d}"/'ready.json')
        if s['method']!='BO_q3_Sobol':assert s['history'][:3]==init['history']
        assert s['initial_knowledge_context_sha256']==init['knowledge_context_sha256']
        assert len(s['history'])==3+3*len(s['decisions'])
        ids=[h['candidate_id'] for h in s['history']];assert len(ids)==len(set(ids))
        for h in s['history']:assert h['y']==float(values[lookup[h['candidate_id']]])
        for i,d in enumerate(s['decisions'],1):
            path=study.ROOT/'runs'/s['run_id']/'batches'/f'{i:02d}_committed.json'
            assert data.filehash(path)==d['committed_sha256'];c=data.read(path)
            assert c['selected_ids']==d['selected_ids'] and c['history_count']==3*i
            assert c['history_sha256']==digest(s['history'][:3*i]) and c['simultaneous_commit'] is True
            decision=c['decision'];selected=decision['selected'];ref=decision['reference']
            score=np.array(decision['base']);assert np.isfinite(score).all()
            assert ref==min(range(len(score)),key=lambda j:(-score[j],decision['batch_ids'][j]))
            eligible=score[ref]-score<=decision['budget'];assert np.array_equal(eligible,decision['eligible'])
            if s['method'] not in ('ReasoningBO','LGBO_region'):assert eligible[selected]
            if d['fallback']:assert selected==ref and d['amplitude']==0
            if s['method']=='ReasoningBO':
                valid=llm_validate(d,pool,s['history'][:3*i],decision['shortlist']['ids'])
                assert valid==d['selected_ids']
            m=d['measurement']
            if m:
                H=np.array(m['feature']);prior=decision['prior'];P=np.array(prior['covariance']);theta=np.array(prior['mean'])
                R=np.array(m['prequery_observation_covariance'])+np.eye(3)*.0009
                residual=np.array(d['observed_y'])-np.array(m['prequery_mu_y'])
                S=H@P@H.T+R
                direct_mean=theta+P@H.T@np.linalg.solve(S,residual-H@theta)
                direct_cov=P-P@H.T@np.linalg.solve(S,H@P)
                err=max(np.max(np.abs(direct_mean-np.array(m['posterior']['mean']))),
                    np.max(np.abs(direct_cov-np.array(m['posterior']['covariance']))))
                max_kf_error=max(max_kf_error,float(err));assert err<1e-10
                counts['joint_KF_updates']+=1
            counts['audited_batches']+=1
        if s['status']=='COMPLETE':assert len(s['decisions'])==30 and len(s['history'])==93
        counts['audited_trajectories']+=1
    complete=all(s['status']=='COMPLETE' for s in ss)
    result={'status':'PASS_COMPLETE' if complete else 'PASS_AVAILABLE_EVIDENCE_INCOMPLETE',
        'protocol_sha256':p['protocol_sha256'],'counts':dict(counts),'planned_trajectories':len(ss),
        'max_joint_KF_error':max_kf_error,'operator_compiles_in_loop':0}
    data.write(study.OUT/'validation_receipt.json',result)
    return result


def llm_validate(d,pool,history,shortlist):
    from .reasoning_bo_llm import validate_insight
    return validate_insight(d['insight'],pool,history,shortlist)['selected_ids']
