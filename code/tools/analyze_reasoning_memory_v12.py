"""Read-only matched-prefix explanation and independently checked comparison metrics."""
from pathlib import Path
from statistics import fmean
import json,sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/chemist_system'));sys.path.insert(0,str(ROOT/'code/experiment_tools'))
from reasoning_memory_v12 import study
from knowledge_information_v10.report import table


def build():
    out=ROOT/'experiments/reports/knowledge_loop_20261101';result=study.old.read(out/'v12_results.json')
    paths={'Pure BO':(study.v9.directory(ROOT),'bo'),'Reasoning BO':(study.v9.directory(ROOT),'reasoning_bo'),
        'V11':(study.v11.directory(ROOT),'rbo_memory'),'V12':(study.directory(ROOT),'memory'),'V12 category':(study.directory(ROOT),'category')}
    states={};curves={};observed={};checked=0;prefixes=[];deltas=[];challenges=[]
    for row in result['rows']:
        task,label=row['task'],row['method'];base,arm=paths[label];run=base/'runs'/f'{task}_{arm}'
        state=study.old.read(run/'state.json');states[task,label]=state;values=[r['raw_value'] for r in state['history']]
        curve=[max(values[:i+1]) for i in range(len(values))];curves[task,label]=curve
        assert abs(row['SearchAUC']-fmean(curve[4:]))<1e-10
        assert row['final_best']==max(values) and row['first_best_total']==values.index(max(values))+1
        for r in state['history']:
            key=(task,r['candidate_id']);value=(r['y'],r['raw_value'])
            assert key not in observed or observed[key]==value
            observed[key]=value
        checked+=1
    for task in study.TASKS:
        a=states[task,'V11'];b=states[task,'V12']
        first=next(t for t,(x,y) in enumerate(zip(a['decisions'],b['decisions']),1) if x['selected_id']!=y['selected_id'])
        entry=dict(task=task,first_divergence_query=first,arms={})
        commits=[]
        for label in ('V11','V12'):
            base,arm=paths[label];c=study.old.read(base/'runs'/f'{task}_{arm}'/'rounds'/f'{first:02d}_commit.json');commits.append(c)
            d=states[task,label]['decisions'][first-1];h=c['LLM_raw']['hypotheses'][c['winning_hypothesis_index']]
            entry['arms'][label]=dict(selected_id=c['selected_id'],objective=d['raw_value'],strategy=h['strategy'],rationale=h['rationale'],
                refresh=c['refresh'],memory_references=c['memory_references'],GP_sha256=c['GP_sha256'],ordered_ids=c['ordered_ids'])
        assert commits[0]['prestate']['history']==commits[1]['prestate']['history']
        assert commits[0]['GP_sha256']==commits[1]['GP_sha256'] and commits[0]['ordered_ids']==commits[1]['ordered_ids']
        entry['matched_observations_GP_menu']=True;entry['causal_boundary']='First joint-policy divergence; individual memory/KF/refresh effects not isolated.'
        prefixes.append(entry)
        for method in ('Pure BO','Reasoning BO','V11'):
            total=0
            for t in range(1,17):
                gap=curves[task,'V12'][t+3]-curves[task,method][t+3];total+=gap/16
                deltas.append(dict(task=task,comparator=method,query=t,incumbent_difference=gap,SearchAUC_contribution=gap/16))
            index={(r['task'],r['method']):r for r in result['rows']}
            assert abs(total-(index[task,'V12']['SearchAUC']-index[task,method]['SearchAUC']))<1e-10
    for task in study.TASKS[:2]:
        for t in range(1,17):
            c=study.old.read(study.directory(ROOT)/'runs'/f'{task}_category/rounds'/f'{t:02d}_commit.json')
            if not c['category']['inserted']:continue
            cid=c['category']['challenger_id'];mentions=[h for h in c['LLM_raw']['hypotheses'] if any(x['candidate_id']==cid for x in h['parameter_sets'])]
            challenges.append(dict(task=task,query=t,challenger_id=cid,selected_id=c['selected_id'],selected=cid==c['selected_id'],
                challenger_qLogEI=c['raw_qLogEI'][-1],native_best_qLogEI=c['raw_qLogEI'][0],
                challenger_rationales=json.dumps(mentions,ensure_ascii=False),gate=json.dumps(c['category']['details'],ensure_ascii=False)))
    study.write(out/'v12_matched_prefix_divergences.json',prefixes);table(out/'v12_AUC_contributions.csv',deltas);table(out/'v12_category_challengers.csv',challenges)
    verification=dict(status='PASS',independently_recomputed_metric_rows=checked,consistent_observed_task_points=len(observed),
        matched_prefix_divergences=3,new_queries=0,new_API_calls=0,new_GP_fits=0,
        source_states_sha256=result['source_state_sha256'],script_sha256=study.old.sha(__file__))
    study.write(out/'v12_metric_verification.json',verification);print(json.dumps(verification,ensure_ascii=False,indent=2))


if __name__=='__main__':build()
