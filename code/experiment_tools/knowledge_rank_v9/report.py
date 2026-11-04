"""Four-method eight-task comparison; inspectable per-round KF lifecycle."""
from collections import Counter
import csv,json
from pathlib import Path
import numpy as np
from chem.benchmarks import knowledge_study as old
from . import study


def build(root):
    root=Path(root);p=study.check(root);folder=study.directory(root);rows=[];states={};lifecycle=[]
    for task in p['tasks']:
        for arm in study.ARMS:
            path=folder/'runs'/f'{task}_{arm}/state.json'
            state=old.read(path)
            if state['status']!='COMPLETE' or len(state['history'])!=20 or len(state['decisions'])!=16:
                raise ValueError('All24new complete trajectories are required')
            if state['task']!=task or state['arm']!=arm or state['protocol_sha256']!=p['protocol_sha256']:
                raise ValueError('Wrong task/arm/protocol')
            if len({h['candidate_id'] for h in state['history']})!=20:raise ValueError('Repeated experiment')
            states[task,arm]=state;r=study.summary(state);r['method']=arm;rows.append(r)
            for d in state['decisions']:
                for e in d['KF_feedback']:
                    lifecycle.append(dict(task=task,query=d['query'],id=e['id'],version=e['version'],point=e['point_id'],
                        Brier_gain=e['Brier_gain'],KF_mean=e['after']['mean'],KF_variance=e['after']['variance'],
                        weight_before=e['deployment_weight_before'],weight_after=e['deployment_weight_after'],
                        status_before=e['before']['status'],status_after=e['after']['status'],reason=e['reason']))
        bo=states[task,'bo'];v8bo=old.read(old.paths(root,'v8')/'runs'/f'{task}_bo_agent_initial_terra/state.json')
        if bo['history']!=v8bo['history']:raise ValueError('Full20point BO does not reproduce V8')
        for arm in ('reasoning_bo','kf_rank'):
            if states[task,arm]['history'][:4]!=bo['history'][:4]:raise ValueError('Unmatched initialization')
        v8=old.read(old.paths(root,'v8')/'runs'/f'{task}_full_agent_initial_terra/state.json');vr=old.summarize(v8)
        if v8['history'][:4]!=bo['history'][:4]:raise ValueError('V8 initial4 differ')
        rows.append(dict(task=task,method='V8',arm='V8',SearchAUC=vr['SearchAUC'],final_best=vr['final_best'],
            first_best_total=vr['first_best_total'],queries=16,status='COMPLETE',
            non_BO_choices=vr['knowledge_choices']+vr['optional_verifications'],LLM_calls=vr['calls']))
    index={(r['task'],r['method']):r for r in rows}
    for r in rows:
        b=index[r['task'],'bo'];r.update(AUC_delta_vs_BO=r['SearchAUC']-b['SearchAUC'],
            AUC_relative_percent=100*(r['SearchAUC']/b['SearchAUC']-1),best_delta_vs_BO=r['final_best']-b['final_best'])
    arms={}
    for method in ('kf_rank','reasoning_bo','V8'):
        subset=[r for r in rows if r['method']==method]
        wins=sum(r['AUC_delta_vs_BO']>1e-9 for r in subset);losses=sum(r['AUC_delta_vs_BO']<-1e-9 for r in subset)
        arms[method]=dict(wins=wins,ties=8-wins-losses,losses=losses,
            final_best_losses=sum(r['best_delta_vs_BO']<-1e-9 for r in subset),
            median_relative_AUC_percent=float(np.median([r['AUC_relative_percent'] for r in subset])))
    comparison={method:dict(wins=sum(index[t,'kf_rank']['SearchAUC']-index[t,method]['SearchAUC']>1e-9 for t in p['tasks']),
        losses=sum(index[t,'kf_rank']['SearchAUC']-index[t,method]['SearchAUC']<-1e-9 for t in p['tasks'])) for method in ('V8','reasoning_bo','bo')}
    counts=Counter(e['reason'] for e in lifecycle)
    audit_path=root/'experiments/reports/knowledge_loop_20261101/v9_audit.json'
    audit=old.read(audit_path) if audit_path.exists() else {}
    audited=audit.get('status')=='PASS_LISTED_CHECKS' and not audit.get('failures') and audit.get('totals',{}).get('runs')==24
    result=dict(protocol_sha256=p['protocol_sha256'],rows=rows,arms=arms,KF_rank_vs=comparison,
        KF_event_counts=dict(counts),KF_weight_selection_changes=sum(r.get('KF_weight_selection_changes',0) for r in rows if r['method']=='kf_rank'),
        lifecycle=lifecycle,majority_AUC_gate=arms['kf_rank']['wins']>=5,independent_confirmation=False,
        audit_complete=audited,goal_achieved=False,
        limitations=['Seed0 real-agent development; no multi-seed or pressure-condition claim.',
            'Top5 rank fusion replaces V8 opportunity gate and numerical expert, so whole-policy gains do not isolate KF.',
            'Fixed-weight shadow shares actual versions/statuses; it is not a complete no-KF ablation.',
            'KF Gaussian approximation measures prospective Brier skill; it does not establish chemical causality.'])
    reportdir=root/'experiments/reports/knowledge_loop_20261101';write=study.write
    write(reportdir/'v9_results.json',result)
    fields=['task','method','SearchAUC','final_best','first_best_total','AUC_delta_vs_BO','AUC_relative_percent',
        'best_delta_vs_BO','non_BO_choices','KF_weight_selection_changes','automatic_pauses','automatic_retractions','automatic_resumes','LLM_calls']
    with (reportdir/'v9_comparison.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    if lifecycle:
        with (reportdir/'v9_KF_lifecycle.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(lifecycle[0]));writer.writeheader();writer.writerows(lifecycle)
    lines=['V9-KF-Rank：8任务，seed0，共同4初始化＋16查询',
        '原GP及qLogEI不变；两个LLM臂同5选1、Terra medium；V8只读。',
        '任务\t方法\tSearchAUC\t最终最佳\t首达总轮次\tAUC较BO\tAUC相对BO/%\t改点']
    for task in p['tasks']:
        for method in ('bo','V8','reasoning_bo','kf_rank'):
            r=index[task,method]
            lines.append(f"{task}\t{method}\t{r['SearchAUC']:.8f}\t{r['final_best']:.8f}\t{r['first_best_total']}\t"
                f"{r['AUC_delta_vs_BO']:+.8f}\t{r['AUC_relative_percent']:+.3f}\t{r['non_BO_choices']}")
    lines+=['',json.dumps(arms,ensure_ascii=False),f'KF事件：{dict(counts)}',
        f"KF当前权重相对固定.4影子实际改点：{result['KF_weight_selection_changes']}",
        f"至少5/8严格AUC胜出：{result['majority_AUC_gate']}；24轨迹完整审计：{audited}",*result['limitations']]
    (reportdir/'v9_results.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return dict(arms=arms,comparison=comparison,KF_event_counts=dict(counts),audit_complete=audited)
