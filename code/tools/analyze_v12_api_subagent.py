"""Compare saved V12 API/subagent trajectories; never run an experiment.

Only JSON commitments, states, exported metrics and saved GP arrays are read.
No policy module, model client, emulator or GP fitter is imported.
"""
from collections import Counter
from hashlib import sha256
from itertools import combinations
from pathlib import Path
import csv
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'paper/v12_subagent/data'
OUT = ROOT / 'experiments/reports/v12_api_subagent_diagnosis_20261102'
HASHES = {}


def read(path):
    path = ROOT / path
    data = path.read_bytes()
    HASHES[path.relative_to(ROOT).as_posix()] = sha256(data).hexdigest()
    return json.loads(data)


def wins(values):
    return dict(wins=sum(v > 1e-9 for v in values), ties=sum(abs(v) <= 1e-9 for v in values), losses=sum(v < -1e-9 for v in values))


def first_diff(a, b):
    return next((q for q, (x, y) in enumerate(zip(a['selected_ids'], b['selected_ids']), 1) if x != y), None)


def load_cell(row, cohort):
    state = read(row['source'])
    assert HASHES[row['source']] == row['source_sha256']
    assert state['status'] == 'COMPLETE' and len(state['history']) == 20 and len(state['decisions']) == 16
    values = [h['raw_value'] for h in state['history']]
    auc = sum(max(values[:4+q]) for q in range(1, 17))/16
    assert math.isclose(auc, row['SearchAUC'], rel_tol=0, abs_tol=1e-10)
    assert max(values) == row['final_best']
    commits = [read(Path(row['source']).parent / 'rounds' / f'{q:02d}_commit.json') for q in range(1, 17)]
    events, rounds = [], []
    for q, (c, d) in enumerate(zip(commits, state['decisions']), 1):
        assert c['selected_id'] == d['selected_id'] == row['selected_ids'][q-1]
        assert c['commit_sha256'] == d['commit_sha256']
        feedback = d.get('KF_feedback', [])
        for event in feedback:
            assert event['success'] == (values[q+3] > max(values[:q+3]))
            gain = (int(event['success'])-event['p_GP'])**2-(int(event['success'])-event['p_LLM'])**2
            assert math.isclose(gain,event['Brier_gain'],rel_tol=0,abs_tol=1e-10)
            events.append(dict(query=q, **{k:event[k] for k in ('id','version','success','p_LLM','p_GP','Brier_gain','reason','before','after')}))
        refs = d.get('memory_references', [])
        rounds.append(dict(query=q,selected_id=c['selected_id'],raw_value=values[q+3],incumbent_before=max(values[:q+3]),
            incumbent_after=max(values[:q+4]),BO_reference_id=c.get('BO_reference_id'),
            expected_forecasts=c.get('expected_forecasts'),forecast_events=len(feedback),
            edit_call=q in state.get('edit_calls',[]),refresh=c.get('refresh'),reviews=c.get('reviews',[]),
            accepted_support=sum(r.get('use')=='support' for r in refs),references=refs,
            sidecar_warnings=d.get('sidecar_warnings',[]),decision_purpose=c.get('decision_purpose'),
            initial_or_refreshed_notes=c.get('hypotheses_at_commit',[]) if q==1 or q in state.get('edit_calls',[]) else None,
            insight=c.get('LLM_raw'),memory_proposal=c.get('memory_raw'),commit_source=(Path(row['source']).parent/'rounds'/f'{q:02d}_commit.json').as_posix()))
    notes = commits[0].get('hypotheses_at_commit',[])
    result=dict(row, cohort=cohort, initial_notes=notes, edit_queries=state.get('edit_calls',[]),
        first_forecast_query=min((e['query'] for e in events),default=None),
        forecasts=len(events),queries_with_forecast=sum(bool(r['forecast_events']) for r in rounds),
        positive_improving=sum(e['Brier_gain']>0 and e['success'] for e in events),
        positive_nonimproving=sum(e['Brier_gain']>0 and not e['success'] for e in events),
        negative_improving=sum(e['Brier_gain']<0 and e['success'] for e in events),
        negative_nonimproving=sum(e['Brier_gain']<0 and not e['success'] for e in events),
        expected_forecasts=sum(r['expected_forecasts'] or 0 for r in rounds),
        sidecar_warnings=sum(len(r['sidecar_warnings']) for r in rounds),
        accepted_support=sum(r['accepted_support'] for r in rounds),
        non_top_without_forecast=sum(r['selected_id']!=r['BO_reference_id'] and not r['forecast_events'] for r in rounds),
        events=events,rounds=rounds)
    return result,commits


def response_style(commits):
    distinct=Counter();hypotheses=Counter();confidence=Counter()
    for c in commits:
        raw=c['LLM_raw'];raw=raw.get('insight',raw)
        hh=raw['hypotheses']
        distinct[len({p['candidate_id'] for h in hh for p in h['parameter_sets']})]+=1
        hypotheses[len(hh)]+=1
        winner=min(hh,key=lambda h:{'high':0,'medium':1,'low':2}[h['confidence'].lower()])
        confidence[winner['confidence'].lower()]+=1
    return dict(responses=len(commits),distinct_proposed_candidates=dict(distinct),hypotheses=dict(hypotheses),winning_confidence=dict(confidence))


def main():
    old=read(DATA/'legacy_evidence.json');new=read(DATA/'evidence.json')
    audit=read('experiments/reports/paper_v12_subagent/audit.json')
    assert audit['status']=='PASS'
    protected=read('experiments/runs/paper_v12_subagent/protocol.json')['protected_controls_sha256']
    cells=[];commit_sets={}
    for cohort,rows in [('API',old['rows']),('subagent',new['rows'])]:
        for row in rows:
            if row['method'] not in ('V12','No KF','No refresh'):continue
            cell,cc=load_cell(row,cohort)
            cells.append(cell);commit_sets[row['source']]=cc
    api={(r['task'],r['initialization_seed']):r for r in cells if r['cohort']=='API' and r['phase']!='LLM_repeat'}
    sub={(r['task'],r['initialization_seed']):r for r in cells if r['cohort']=='subagent' and r['method']=='V12'}
    bo={(r['task'],r['initialization_seed']):r for r in new['rows'] if r['method']=='BO'}
    cross=[]
    for key,a in api.items():
        b=sub[key];control=bo[key]
        assert a['initial_ids']==b['initial_ids']==control['initial_ids']
        assert a['values'][:4]==b['values'][:4]==control['values'][:4]
        first=first_diff(a,b)
        checks=[]
        for q in range(1,(first or 16)+1):
            ca,cb=(commit_sets[r['source']][q-1] for r in (a,b))
            assert a['selected_ids'][:q-1]==b['selected_ids'][:q-1]
            arrays=[]
            for r in (a,b):
                path=ROOT/Path(r['source']).parent/'rounds'/f'{q:02d}_GP.npz'
                HASHES[path.relative_to(ROOT).as_posix()]=sha256(path.read_bytes()).hexdigest()
                with np.load(path) as p:arrays.append({k:p[k].copy() for k in p.files})
            equal={k:np.array_equal(arrays[0][k],arrays[1][k],equal_nan=True) for k in arrays[0]}
            checks.append(dict(query=q,arrays_exact_equal=equal,menu_equal=ca['ordered_ids']==cb['ordered_ids'],display_equal=ca['display_ids']==cb['display_ids']))
        index=(first or 16)-1
        a0,b0=commit_sets[a['source']][0],commit_sets[b['source']][0]
        exact_fields=('claim','effect','focus','scope','reference_id')
        extract=lambda ns:[{k:n.get(k) for k in exact_fields} for n in ns]
        cross.append(dict(task=key[0],label=a['label'],seed=key[1],API_AUC=a['SearchAUC'],subagent_AUC=b['SearchAUC'],BO_AUC=control['SearchAUC'],
            API_delta_percent=100*(a['SearchAUC']/control['SearchAUC']-1),subagent_delta_percent=100*(b['SearchAUC']/control['SearchAUC']-1),
            API_best=a['final_best'],subagent_best=b['final_best'],BO_best=control['final_best'],
            API_first_best=a['first_best_query'],subagent_first_best=b['first_best_query'],
            first_different_query=first,differing_positions=sum(x!=y for x,y in zip(a['selected_ids'],b['selected_ids'])),
            API_prior_KF=sum(e['query']<first for e in a['events']) if first else None,
            subagent_prior_KF=sum(e['query']<first for e in b['events']) if first else None,
            API_initial_notes=len(a['initial_notes']),subagent_initial_notes=len(b['initial_notes']),
            exact_initial_note_fields_equal=extract(a['initial_notes'])==extract(b['initial_notes']),
            initial_memory_context_hash_equal=a0.get('memory_context_sha256')==b0.get('memory_context_sha256'),
            API_selected_at_first_diff=a['selected_ids'][index] if first else None,
            subagent_selected_at_first_diff=b['selected_ids'][index] if first else None,
            API_outcome_at_first_diff=a['values'][index+4] if first else None,
            subagent_outcome_at_first_diff=b['values'][index+4] if first else None,
            shared_prefix_checks=checks,API_source=a['source'],subagent_source=b['source']))
    totals=[]
    for cohort in ('API','subagent'):
        groups={'matched_initializations_0_3':[r for r in cells if r['cohort']==cohort and r['method']=='V12' and (r['task'],r['initialization_seed']) in api and r['phase']!='LLM_repeat'],
                'matched_primary_1_3':[r for r in cells if r['cohort']==cohort and r['method']=='V12' and (r['task'],r['initialization_seed']) in api and r['initialization_seed'] in (1,2,3) and r['phase']!='LLM_repeat']}
        if cohort=='API':groups['fresh_seed0_repeats']=[r for r in cells if r['cohort']==cohort and r['phase']=='LLM_repeat']
        else:groups['all_eight_tasks_0_3']=[r for r in cells if r['cohort']==cohort and r['method']=='V12']
        for label,rr in groups.items():
            keys=('forecasts','queries_with_forecast','positive_improving','positive_nonimproving','negative_improving','negative_nonimproving','pauses','retractions','resumes','edits','expected_forecasts','sidecar_warnings','accepted_support','non_top_without_forecast')
            totals.append(dict(cohort=cohort,group=label,trajectories=len(rr),queries=16*len(rr),**{k:sum(r[k] for r in rr) for k in keys},
                AUC_vs_BO=wins([r['SearchAUC']-bo[(r['task'],r['initialization_seed'])]['SearchAUC'] for r in rr]),
                best_vs_BO=wins([r['final_best']-bo[(r['task'],r['initialization_seed'])]['final_best'] for r in rr])))
    repeats=[]
    for task in sorted({r['task'] for r in cells if r['phase']=='LLM_repeat'}):
        rr=[r for r in cells if r['task']==task and r['phase']=='LLM_repeat']
        disagreements=[sum(x!=y for x,y in zip(a['selected_ids'],b['selected_ids']))/16 for a,b in combinations(rr,2)]
        repeats.append(dict(task=task,SearchAUC_min=min(r['SearchAUC'] for r in rr),SearchAUC_max=max(r['SearchAUC'] for r in rr),
            mean_pairwise_position_disagreement=sum(disagreements)/len(disagreements),
            AUC_vs_BO=wins([r['SearchAUC']-bo[(task,0)]['SearchAUC'] for r in rr])))
    styles=[];edit_profiles=[];panels=[]
    focus=set(k[0] for k in api)
    for co in ('API','subagent'):
        group=[r for r in cells if r['cohort']==co and r['method']=='V12' and r['task'] in focus and r['phase']!='LLM_repeat']
        styles.append(dict(cohort=co,method='V12',scope='matched_12_cells',**response_style([c for r in group for c in commit_sets[r['source']]])))
        calls=[q for r in group for q in r['rounds'] if q['edit_call']]
        reviews=[a for q in calls for a in q['reviews']]
        edit_profiles.append(dict(cohort=co,calls=len(calls),query_positions=dict(Counter(q['query'] for q in calls)),
            empty_actions=sum(not q['memory_proposal']['actions'] for q in calls),
            accepted_actions=dict(Counter(a['op'] for a in reviews if a.get('accepted'))),
            rejected_actions=sum(not a.get('accepted') for a in reviews),carried_fact_count=sum(a.get('carried_fact_count',0) for a in reviews)))
        rr=[r for r in (old if co=='API' else new)['rows'] if r['method']=='Reasoning BO']
        cc=[]
        for r in rr:
            state=read(r['source']);assert HASHES[r['source']]==r['source_sha256']
            cc.extend(read(Path(r['source']).parent/'rounds'/f'{q:02d}_commit.json') for q in range(1,17))
        styles.append(dict(cohort=co,method='Reasoning BO',scope='available_8_seed0_API_or_24_primary_subagent_cells',**response_style(cc)))
    for task in sorted(focus):
        pp=[r for r in cross if r['task']==task and r['seed'] in (1,2,3)]
        panels.append(dict(task=task,n=3,API_ratio_of_mean_AUC_percent=100*(sum(r['API_AUC'] for r in pp)/sum(r['BO_AUC'] for r in pp)-1),
            subagent_ratio_of_mean_AUC_percent=100*(sum(r['subagent_AUC'] for r in pp)/sum(r['BO_AUC'] for r in pp)-1),
            API_wins=wins([r['API_AUC']-r['BO_AUC'] for r in pp]),subagent_wins=wins([r['subagent_AUC']-r['BO_AUC'] for r in pp])))
    examples=[]
    for task,seed in [('Reizman_Suzuki_Case_2',0),('Reizman_Suzuki_Case_4',1),('HPLC',0)]:
        a,b=api[(task,seed)],sub[(task,seed)];q=first_diff(a,b)
        one=dict(task=task,seed=seed,first_difference=q)
        for label,r in [('API',a),('subagent',b)]:
            c=commit_sets[r['source']][q-1]
            one[label]=dict(selected_id=c['selected_id'],GP_reference=c['BO_reference_id'],GP_rank=c['ordered_ids'].index(c['selected_id'])+1,
                logEI_gap=c['opportunity_gap'],observed_value=r['values'][q+3],prequery_incumbent=max(r['values'][:q+3]))
        one['AUC_delta_query_summands']=[(b['best_curve'][q]-a['best_curve'][q])/16 for q in range(1,17)]
        assert math.isclose(sum(one['AUC_delta_query_summands']),b['SearchAUC']-a['SearchAUC'],abs_tol=1e-10)
        examples.append(one)
    native_comparisons=read('experiments/reports/paper_v12_subagent/contrast_trace_review.json')
    for name in ('engine.py','llm.py'):
        p=ROOT/'code/experiment_tools/reasoning_memory_v12'/name
        data=p.read_bytes();HASHES[p.relative_to(ROOT).as_posix()]=sha256(data).hexdigest()
        if name=='engine.py':assert data.decode('utf-8').count('KF_changes_selection_from_fixed_weights=False')==2
    verified_old=verified_new=0
    for name,value in HASHES.items():
        if name in protected:assert protected[name]==value;verified_old+=1
        if name in audit['completed_input_sha256']:assert audit['completed_input_sha256'][name]==value;verified_new+=1
        assert sha256((ROOT/name).read_bytes()).hexdigest()==value
    OUT.mkdir(parents=True,exist_ok=True)
    result=dict(status='PASS_READ_ONLY_CROSS_COHORT_DIAGNOSIS',script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256=HASHES,verified_against_protected_history=verified_old,verified_against_new_native_audit=verified_new,
        matched_cells=len(cross),cross_cohort=cross,totals=totals,API_repeats=repeats,cells=cells,
        response_style=styles,edit_profiles=edit_profiles,matched_primary_panels=panels,first_divergence_examples=examples,
        subagent_paired_contrast_summary=native_comparisons['summaries'],
        scientific_code_findings={'forecast_target':'current-incumbent binary improvement probability, not candidate ranking or information gain',
            'KF_changes_selection_from_fixed_weights':'The native field is hard-coded False, not a measured counterfactual. It must not be counted as proof that advisory memory never influenced an LLM choice.',
            'sources':['code/experiment_tools/reasoning_memory_v12/engine.py','code/experiment_tools/reasoning_memory_v12/llm.py']},
        boundary='Cross-environment observations, not randomized backend effects. Same initial observations are verified. GP arrays are compared only while observed histories match. Each selected-point forecast is relative to the current incumbent. Fresh responses and independently generated initial notes prevent isolated component attribution.',
        new_experiments=0,new_model_calls=0,new_GP_fits=0)
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    scalar=[{k:v for k,v in row.items() if k!='shared_prefix_checks'} for row in cross]
    with (OUT/'cross_cohort_metrics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(scalar[0]));writer.writeheader();writer.writerows(scalar)
    print(json.dumps({k:result[k] for k in ('status','matched_cells','matched_primary_panels','response_style','edit_profiles','verified_against_protected_history','verified_against_new_native_audit')},indent=2))


if __name__=='__main__':main()
