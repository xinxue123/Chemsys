"""Export recorded ranking/KF stages and scientific comparison figures."""
import csv,json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/chemist_system'));sys.path.insert(0,str(ROOT/'code/experiment_tools'))
from knowledge_rank_v9 import study
from chem.benchmarks import knowledge_study as old


def table(path,rows):
    if not rows:return
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    protocol=study.check(ROOT);out=ROOT/'experiments/reports/knowledge_loop_20261101'
    rankings=[];updates=[];edits=[]
    for task in protocol['tasks']:
        run=study.directory(ROOT)/'runs'/f'{task}_kf_rank';state=old.read(run/'state.json')
        if state['status']!='COMPLETE':raise ValueError('Complete all trajectories first')
        for t,d in enumerate(state['decisions'],1):
            c=old.read(run/'rounds'/f'{t:02d}_commit.json');ranking=c['ranking']
            for i,key in enumerate(c['ordered_ids']):
                rankings.append(dict(task=task,query=t,candidate_id=key,base_qLogEI=c['raw_qLogEI'][i],
                    base_Borda=ranking['base_rank_scores'][i],combined_Borda=ranking['combined_rank_scores'][i],
                    selected=key==c['selected_id'],BO_selected=key==c['BO_reference_id'],
                    observed_objective=d['raw_value'] if key==c['selected_id'] else '',
                    outside_V8_gate=c['outside_V8_gate'] if key==c['selected_id'] else '',
                    rule_terms=json.dumps([dict(id=r['id'],version=r['version'],p_LLM=r['probabilities'][i],
                        inside_scope=r['inside'][i],effective_weight=r['effective_weight'],rank_score=r['rank_scores'][i]) for r in ranking['terms']],ensure_ascii=False),
                    fixed_weight_shadow=c['fixed_initial_weight_selected_id'],KF_changed_selection=c['KF_changes_selection_from_fixed_weights']))
            for e in d['KF_feedback']:
                updates.append(dict(task=task,query=t,id=e['id'],version=e['version'],point=e['point_id'],
                    reference_id=e['reference_id'],observed_y=e['observed_y'],success=e['success'],p_LLM=e['p_LLM'],p_GP=e['p_GP'],
                    Brier_gain=e['Brier_gain'],measurement_variance=e['measurement_variance'],kalman_gain=e['kalman_gain'],
                    mean_before=e['before']['mean'],mean_after=e['after']['mean'],variance_before=e['before']['variance'],variance_after=e['after']['variance'],
                    positive_skill_probability=e['after']['probability_positive'],weight_before=e['deployment_weight_before'],
                    weight_after=e['deployment_weight_after'],status_before=e['before']['status'],status_after=e['after']['status'],reason=e['reason']))
            for review in c['reviews']:
                action=c['LLM_raw']['actions'][review['action']]
                edits.append(dict(task=task,query=t,id=action['id'],op=action['op'],accepted=review['accepted'],
                    outcome=review.get('outcome','rejected'),event_reasons=';'.join(c['refresh']['reasons']),
                    scientific_reason=action['reason'],review_reason=review.get('reason',''),
                    inherited_negative_count=review.get('inherited_negative_count',0)))
    table(out/'v9_decision_stages.csv',rankings);table(out/'v9_KF_updates.csv',updates);table(out/'v9_LLM_edits.csv',edits)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles={'bo':('#444444','Pure BO'),'V8':('#e99b32','V8'),'reasoning_bo':('#3a78b8','Reasoning BO'),'kf_rank':('#20865a','V9 KF-Rank')}
    short={**{f'Reizman_Suzuki_Case_{i}':f'Reizman Suzuki Case {i}' for i in range(1,5)},
        'Perera_Suzuki':'Perera Suzuki','Doyle_Buchwald_Hartwig':'Doyle Buchwald-Hartwig','SnAr':'SnAr','HPLC':'HPLC'}
    for kind in ('incumbent','observed'):
        fig,axes=plt.subplots(4,2,figsize=(13,13),constrained_layout=True)
        for ax,task in zip(axes.flat,protocol['tasks']):
            for arm,(color,label) in styles.items():
                path=(old.paths(ROOT,'v8')/'runs'/f'{task}_full_agent_initial_terra/state.json') if arm=='V8' else study.directory(ROOT)/'runs'/f'{task}_{arm}/state.json'
                s=old.read(path);values=np.asarray([r['raw_value'] for r in s['history']])
                y=np.maximum.accumulate(values)[3:] if kind=='incumbent' else values[4:]
                x=np.arange(17) if kind=='incumbent' else np.arange(1,17)
                ax.plot(x,y,label=label,color=color,linewidth=1.8,alpha=.95)
            ax.set_title(short[task]);ax.set_xlabel('Query (0 = initial four)' if kind=='incumbent' else 'Query')
            ax.set_ylabel('Best objective' if kind=='incumbent' else 'Observed objective');ax.grid(alpha=.22)
        handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4)
        fig.suptitle('Shared pool / GP / seed 0 / 4 + 16 queries',fontsize=14)
        fig.savefig(out/f'v9_{kind}_comparison.png',dpi=180);fig.savefig(out/f'v9_{kind}_comparison.pdf');plt.close(fig)
    print(json.dumps(dict(candidate_stage_rows=len(rankings),KF_updates=len(updates),LLM_actions=len(edits),figures=2)))


if __name__=='__main__':main()
