"""Whole-policy comparison against four frozen controls; no control reruns."""
from collections import Counter
from pathlib import Path
import json
import numpy as np
from knowledge_information_v10.report import table
from . import study


def build(root):
    root=Path(root);protocol=study.check(root);states={};metrics=[];rounds=[];notes=[];updates=[]
    methods=[('Pure BO',study.v9.directory(root),'bo'),('Reasoning BO',study.v9.directory(root),'reasoning_bo'),
        ('V9',study.v9.directory(root),'kf_rank'),('V10',study.v10.directory(root),'information'),('V11',study.directory(root),'rbo_memory')]
    for task in protocol['tasks']:
        for label,base,arm in methods:
            run=base/'runs'/f'{task}_{arm}';state=study.old.read(run/'state.json')
            if state['status']!='COMPLETE' or len(state['history'])!=20 or len(state['decisions'])!=16:raise ValueError('Require complete4+16')
            states[task,label]=state
            if label!='Pure BO' and state['history'][:4]!=states[task,'Pure BO']['history'][:4]:raise ValueError('Unmatched initialization')
            values=np.array([r['raw_value'] for r in state['history']]);best=np.maximum.accumulate(values)
            metrics.append(dict(task=task,method=label,SearchAUC=float(best[4:].mean()),final_best=float(best[-1]),
                first_best_total=int(np.flatnonzero(values==best[-1])[0])+1,
                non_BO_choices=sum(d['selected_id']!=d['BO_reference_id'] for d in state['decisions']),
                logical_LLM_calls=len(state['insights'])+(len(state['edit_calls']) if label in ('V10','V11') else 0)))
            for t,d in enumerate(state['decisions'],1):
                rounds.append(dict(task=task,method=label,query=t,total_experiment=t+4,selected_id=d['selected_id'],
                    objective=d['raw_value'],incumbent=float(best[t+3]),mode=d['mode'],BO_reference_id=d['BO_reference_id']))
                if label!='V11':continue
                c=study.old.read(run/'rounds'/f'{t:02d}_commit.json')
                notes.append(dict(task=task,query=t,selected_id=d['selected_id'],BO_reference_id=d['BO_reference_id'],
                    same_as_BO=d['selected_id']==d['BO_reference_id'],official_hypothesis_index=c['winning_hypothesis_index'],
                    refresh=c['refresh']['allowed'],refresh_reasons=';'.join(c['refresh']['reasons']),
                    citations=json.dumps(c['memory_references'],ensure_ascii=False),
                    actions=json.dumps(c['reviews'],ensure_ascii=False),prospective_forecasts=len(c['predictions']),
                    opportunity_gap=c['opportunity_gap']))
                for e in d['KF_feedback']:
                    updates.append(dict(task=task,query=t,id=e['id'],version=e['version'],point=e['point_id'],
                        p_LLM=e['p_LLM'],p_GP=e['p_GP'],success=e['success'],Brier_gain=e['Brier_gain'],kalman_gain=e['kalman_gain'],
                        mean_before=e['before']['mean'],mean_after=e['after']['mean'],variance_after=e['after']['variance'],
                        weight_before=e['before']['weight'],weight_after=e['after']['weight'],
                        status_before=e['before']['status'],status_after=e['after']['status'],reason=e['reason']))
    index={(r['task'],r['method']):r for r in metrics};comparisons={}
    for r in metrics:
        bo=index[r['task'],'Pure BO'];r.update(AUC_delta_vs_BO=r['SearchAUC']-bo['SearchAUC'],
            AUC_relative_percent=100*(r['SearchAUC']/bo['SearchAUC']-1),final_best_delta_vs_BO=r['final_best']-bo['final_best'])
    for method in ('Pure BO','Reasoning BO','V9','V10'):
        deltas=[index[task,'V11']['SearchAUC']-index[task,method]['SearchAUC'] for task in protocol['tasks']]
        wins=sum(d>1e-9 for d in deltas);losses=sum(d<-1e-9 for d in deltas)
        comparisons[method]=dict(wins=wins,ties=len(deltas)-wins-losses,losses=losses)
    out=root/'experiments/reports/knowledge_loop_20261101';audit=out/'v11_audit.json'
    result=dict(protocol_sha256=protocol['protocol_sha256'],rows=metrics,V11_vs=comparisons,
        majority_AUC_gate=comparisons['Pure BO']['wins']>=5,independent_confirmation=False,
        notes_cited_decisions=sum(bool(json.loads(r['citations'])) for r in notes),
        counterevidence_citations=sum(ref['use']=='counterevidence' for r in notes for ref in json.loads(r['citations'])),
        KF_events=dict(Counter(e['reason'] for e in updates)),KF_feedback_count=len(updates),
        audit_complete=audit.exists() and study.old.read(audit).get('status')=='PASS_LISTED_CHECKS',
        limitations=['Same seed0 development, not independent confirmation.',
            'Whole-policy comparison only: note citations and KF changes do not establish causal module benefit.',
            'Up to20 logical LLM calls for V11/V10 versus16 for frozen Reasoning BO/V9; objective budgets match.',
            'Official confidence selection and prompt retained; shared native GP, initialization and sequential5-to1 are matched-interface adaptations.',
            'Notes are advisory. There is no numerical weight fusion, GP correction, additional gate or outside-menu proposal.'])
    study.write(out/'v11_results.json',result);table(out/'v11_comparison.csv',metrics);table(out/'v11_rounds.csv',rounds)
    table(out/'v11_note_decisions.csv',notes);table(out/'v11_KF_updates.csv',updates)
    lines=['V11 Reasoning BO + evidence memory：seed0，4初始化+16查询；候选池与原始GP配方一致',
        '任务\t方法\tSearchAUC\t最终最佳\t首达总实验序号\tAUC较BO/%']
    for r in metrics:lines.append(f"{r['task']}\t{r['method']}\t{r['SearchAUC']:.8f}\t{r['final_best']:.8f}\t{r['first_best_total']}\t{r['AUC_relative_percent']:+.3f}")
    (out/'v11_results.txt').write_text('\n'.join(lines+['',json.dumps(comparisons),*result['limitations']])+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles={'Pure BO':'#444444','Reasoning BO':'#3a78b8','V9':'#e99b32','V10':'#999999','V11':'#20865a'}
    for kind in ('observed','incumbent'):
        fig,axes=plt.subplots(4,2,figsize=(13,13),constrained_layout=True)
        for ax,task in zip(axes.flat,protocol['tasks']):
            for label,color in styles.items():
                v=np.array([r['raw_value'] for r in states[task,label]['history']])
                y=np.maximum.accumulate(v)[3:] if kind=='incumbent' else v[4:]
                x=np.arange(17) if kind=='incumbent' else np.arange(1,17)
                ax.plot(x,y,label=label,color=color,linewidth=2.2 if label=='V11' else 1.4,linestyle='--' if label=='V10' else '-')
            ax.set_title(task.replace('_',' '));ax.set_xlabel('Query (0 = initial four)' if kind=='incumbent' else 'Query')
            ax.set_ylabel('Best objective' if kind=='incumbent' else 'Observed objective');ax.grid(alpha=.22)
        handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=5)
        fig.suptitle('Frozen pool / native GP / seed 0 / 4 + 16 queries',fontsize=14)
        fig.savefig(out/f'v11_{kind}_comparison.png',dpi=180);fig.savefig(out/f'v11_{kind}_comparison.pdf');plt.close(fig)
    return {k:result[k] for k in ('V11_vs','majority_AUC_gate','notes_cited_decisions','counterevidence_citations','KF_events','audit_complete')}
