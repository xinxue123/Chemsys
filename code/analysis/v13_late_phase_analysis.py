"""Post-hoc all-task phase accounting; frozen labels are EVALUATION ONLY."""
from pathlib import Path
from collections import defaultdict
import csv
import json
import statistics
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'paper/v13/data'
OUT=ROOT/'experiments/reports/v13_late_validation_20261103'


def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def write(name,values):
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)


def main():
    metrics=rows(SOURCE/'metrics.csv');rounds=rows(SOURCE/'rounds.csv')
    grouped=defaultdict(list)
    for x in rounds:grouped[x['regime'],x['task'],int(x['seed']),x['method']].append(x)
    truth={}
    for d in rows(SOURCE/'datasets.csv'):
        with np.load(ROOT/d['oracle_path'],allow_pickle=False) as z:
            raw=z['raw_values'] if 'raw_values' in z else z['values']*100
            truth[d['task']]=(float(np.max(raw)),float(np.min(raw)))
    phase=[];hits=[]
    for m in metrics:
        key=(m['regime'],m['task'],int(m['seed']),m['method']);rr=sorted(grouped[key],key=lambda x:int(x['query']))
        B=int(m['queries']);tau=B//2;M=[float(x['cumulative_best']) for x in rr]
        fstar,fmin=truth[m['task']];target=.95*fstar
        first_target=0 if m['initial_best']!='' and float(m['initial_best'])>=target else next((int(x['query']) for x in rr if float(x['cumulative_best'])>=target),None)
        phase.append(dict(regime=m['regime'],task=m['task'],seed=m['seed'],method=m['method'],backend=m['backend'],
            initial=m['initial'],queries=B,phase_cutoff=tau,M_cutoff=M[tau-1],final_best=M[-1],
            final_simple_regret=fstar-M[-1],normalized_terminal_regret=(fstar-M[-1])/(fstar-fmin),
            early_SearchAUC=statistics.mean(M[:tau]),late_absolute_SearchAUC=statistics.mean(M[tau:]),
            full_SearchAUC=statistics.mean(M),late_improvement=M[-1]-M[tau-1],
            late_incremental_SearchAUC=statistics.mean(x-M[tau-1] for x in M[tau:]),
            late_record_events=sum(float(x['cumulative_best'])>float(x['incumbent_before'])+1e-8 for x in rr[tau:]),
            target_95percent_pool_max=target,target_hit=int(first_target is not None),first_target_query='' if first_target is None else first_target,
            first_final_best_query=m['first_final_best_query']))
        if m['method']=='v13':
            state=json.loads((ROOT/m['state_path']).read_text(encoding='utf-8'))
            previous_updates=0;previous_transitions=0
            for x,d in zip(rr,state['decisions']):
                q=int(x['query']);feedback=d.get('feedback',[])
                if q>tau and float(x['cumulative_best'])>float(x['incumbent_before'])+1e-8:
                    uses=d.get('knowledge_uses',[])
                    hits.append(dict(regime=m['regime'],task=m['task'],seed=m['seed'],query=q,candidate_id=x['candidate_id'],
                        incumbent_before=x['incumbent_before'],objective=x['objective'],improvement=float(x['cumulative_best'])-float(x['incumbent_before']),
                        applicable_note_forecasts_on_hit=len(feedback),prior_note_updates=previous_updates,
                        prior_permission_transitions=previous_transitions,
                        latest_editor_before_or_at_hit=max((q0 for q0 in state['edit_calls'] if q0<=q),default=0),
                        editor_this_query=int(q in state['edit_calls']),review_this_query=int(q in state['review_queries']),
                        selected_is_own_GP_top=int(x['candidate_id']==x['BO_reference_id']),
                        declared_note_uses=json.dumps(uses,ensure_ascii=False),
                        feedback_precedes_future_choice_not_this_choice=True,state_path=m['state_path']))
                previous_updates+=len(feedback)
                previous_transitions+=sum(e['recommendation_before']!=e['recommendation_after'] for e in feedback)
    write('historical_phase_metrics.csv',phase);write('historical_late_record_events.csv',hits)
    summary=[]
    for regime,seeds in (('warm',(1,2,3)),('cold',(0,1,2,3))):
        for task in truth:
            for method in ('bo','reasoning_bo','v13'):
                rs=[x for x in phase if x['regime']==regime and x['task']==task and x['method']==method and int(x['seed']) in seeds]
                value=dict(regime=regime,task=task,method=method,n=len(rs))
                for k in ('M_cutoff','final_best','full_SearchAUC','late_improvement','late_incremental_SearchAUC','normalized_terminal_regret'):
                    value[k+'_mean']=statistics.mean(x[k] for x in rs)
                    value[k+'_sd']=statistics.stdev(x[k] for x in rs)
                value['target_hits']=sum(x['target_hit'] for x in rs);summary.append(value)
    write('historical_phase_summary.csv',summary)
    facts=dict(status='COMPLETE_POST_HOC_ANALYSIS',registered_native_trajectories=len(phase),late_V13_record_events=len(hits),
        late_record_events_with_selected_note_forecast=sum(x['applicable_note_forecasts_on_hit']>0 for x in hits),
        retrospective_cutoff='half of paid sequential queries; not selected separately by task',
        target='95 percent of exact frozen candidate-pool maximum; evaluator only',
        limitations=['Own-prefix late gains have different headroom; not a causal intervention',
          'A forecast update after a hit cannot have caused that same selection',
          'No selected-point forecast does not exclude indirect prior-memory effects',
          'API warm-development trajectories remain separately labeled'],
        decision_GP_fits=0,LLM_calls=0,new_experimental_queries=0,offline_evaluation_labels_read=True)
    (OUT/'phase_analysis_receipt.json').write_text(json.dumps(facts,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(facts))


if __name__=='__main__':main()
