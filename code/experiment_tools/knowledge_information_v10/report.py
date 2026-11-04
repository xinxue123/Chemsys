"""Compare only the new policy with three sealed controls; export full traces."""
from collections import Counter
from pathlib import Path
import csv,json
import numpy as np
from . import study


def table(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def build(root):
    root=Path(root);protocol=study.check(root);states={};metrics=[];rounds=[];probes=[];updates=[]
    for task in protocol['tasks']:
        for label,arm in [('Pure BO','bo'),('Reasoning BO','reasoning_bo'),('V9','kf_rank'),('V10','information')]:
            base=study.directory(root) if arm=='information' else study.v9.directory(root)
            state=study.old.read(base/'runs'/f'{task}_{arm}/state.json')
            if state['status']!='COMPLETE' or len(state['history'])!=20 or len(state['decisions'])!=16:raise ValueError('Require all complete4+16 trajectories')
            states[task,label]=state
            if label!='Pure BO' and state['history'][:4]!=states[task,'Pure BO']['history'][:4]:raise ValueError('Unmatched initialization')
            values=np.array([r['raw_value'] for r in state['history']]);best=np.maximum.accumulate(values)
            metrics.append(dict(task=task,method=label,SearchAUC=float(best[4:].mean()),final_best=float(best[-1]),
                first_best_total=int(np.flatnonzero(values==best[-1])[0])+1,
                non_BO_choices=sum(d['selected_id']!=d['BO_reference_id'] for d in state['decisions']),
                information_queries=state.get('information_used',0),dissent_used=state.get('dissent_used',False),
                LLM_calls=len(state['insights'])+(len(state['edit_calls']) if label=='V10' else 0)))
            for t,d in enumerate(state['decisions'],1):
                rounds.append(dict(task=task,method=label,query=t,total_experiment=t+4,selected_id=d['selected_id'],
                    objective=d['raw_value'],incumbent=float(best[t+3]),mode=d['mode'],BO_reference_id=d['BO_reference_id']))
            if label=='V10':
                for t,d in enumerate(state['decisions'],1):
                    c=study.old.read(base/'runs'/f'{task}_{arm}'/'rounds'/f'{t:02d}_commit.json')
                    for p in c['probe_reviews']:
                        probes.append(dict(task=task,query=t,**{k:v for k,v in p.items() if k!='blocked_reasons'},
                            blocked_reasons=';'.join(p['blocked_reasons']),selected=p['candidate_id']==c['selected_id'],
                            outside_top5=p['candidate_id'] not in c['ordered_ids'],mode=c['mode']))
                    for e in d['KF_feedback']:
                        updates.append(dict(task=task,query=t,id=e['id'],version=e['version'],point=e['point_id'],
                            p_LLM=e['p_LLM'],p_GP=e['p_GP'],success=e['success'],Brier_gain=e['Brier_gain'],
                            kalman_gain=e['kalman_gain'],mean_before=e['before']['mean'],mean_after=e['after']['mean'],
                            variance_after=e['after']['variance'],weight_before=e['deployment_weight_before'],
                            weight_after=e['deployment_weight_after'],status_before=e['before']['status'],
                            status_after=e['after']['status'],reason=e['reason']))
    index={(r['task'],r['method']):r for r in metrics};comparisons={}
    for r in metrics:
        bo=index[r['task'],'Pure BO'];r.update(AUC_delta_vs_BO=r['SearchAUC']-bo['SearchAUC'],
            AUC_relative_percent=100*(r['SearchAUC']/bo['SearchAUC']-1),final_best_delta_vs_BO=r['final_best']-bo['final_best'])
    for method in ('Pure BO','V9','Reasoning BO'):
        deltas=[index[task,'V10']['SearchAUC']-index[task,method]['SearchAUC'] for task in protocol['tasks']]
        wins=sum(d>1e-9 for d in deltas);losses=sum(d<-1e-9 for d in deltas)
        comparisons[method]=dict(wins=wins,ties=8-wins-losses,losses=losses)
    out=root/'experiments/reports/knowledge_loop_20261101';audit=out/'v10_audit.json'
    result=dict(protocol_sha256=protocol['protocol_sha256'],rows=metrics,V10_vs=comparisons,
        majority_AUC_gate=comparisons['Pure BO']['wins']>=5,independent_confirmation=False,
        information_queries=sum(s['information_used'] for (task,label),s in states.items() if label=='V10'),
        KG_queries=sum(r['mode']=='information_KG' for r in rounds if r['method']=='V10'),
        disagreement_queries=sum(r['mode']=='information_disagreement' for r in rounds if r['method']=='V10'),
        queries_outside_top5=len({(r['task'],r['query']) for r in probes if r['selected'] and r['outside_top5']}),
        KF_events=dict(Counter(e['reason'] for e in updates)),KF_feedback_count=len(updates),
        audit_complete=audit.exists() and study.old.read(audit).get('status')=='PASS_LISTED_CHECKS',
        limitations=['Same seed0 development, no independent confirmation.',
            'Controls reused as requested; no information-only or no-KF ablation, so whole-policy comparison only.',
            'V10 uses up to20 logical LLM calls versus16 for the sealed controls; objective budgets match.',
            'KG assumes fixed hyperparameters for hypothetical updates; the next real GP is normally refitted.'])
    study.write(out/'v10_results.json',result)
    table(out/'v10_comparison.csv',metrics);table(out/'v10_rounds.csv',rounds)
    table(out/'v10_probe_decisions.csv',probes);table(out/'v10_KF_updates.csv',updates)
    lines=['V10与封存对照：seed0，4初始化+16查询；GP与候选池配方一致','任务\t方法\tSearchAUC\t最终最佳\t首达总实验序号\tAUC较BO/%']
    for r in metrics:lines.append(f"{r['task']}\t{r['method']}\t{r['SearchAUC']:.8f}\t{r['final_best']:.8f}\t{r['first_best_total']}\t{r['AUC_relative_percent']:+.3f}")
    lines += ['',json.dumps(comparisons,ensure_ascii=False),*result['limitations']]
    (out/'v10_results.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles={'Pure BO':'#444444','Reasoning BO':'#3a78b8','V9':'#e99b32','V10':'#20865a'}
    for kind in ('observed','incumbent'):
        fig,axes=plt.subplots(4,2,figsize=(13,13),constrained_layout=True)
        for ax,task in zip(axes.flat,protocol['tasks']):
            for label,color in styles.items():
                v=np.array([r['raw_value'] for r in states[task,label]['history']])
                y=np.maximum.accumulate(v)[3:] if kind=='incumbent' else v[4:]
                x=np.arange(17) if kind=='incumbent' else np.arange(1,17)
                ax.plot(x,y,label=label,color=color,linewidth=1.8)
            ax.set_title(task.replace('_',' '));ax.set_xlabel('Query (0 = initial four)' if kind=='incumbent' else 'Query')
            ax.set_ylabel('Best objective' if kind=='incumbent' else 'Observed objective');ax.grid(alpha=.22)
        handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4)
        fig.suptitle('Frozen pool / native GP / seed 0 / 4 + 16 queries',fontsize=14)
        fig.savefig(out/f'v10_{kind}_comparison.png',dpi=180);fig.savefig(out/f'v10_{kind}_comparison.pdf');plt.close(fig)
    return {k:result[k] for k in ('V10_vs','majority_AUC_gate','information_queries','KG_queries','disagreement_queries','KF_events','audit_complete')}
