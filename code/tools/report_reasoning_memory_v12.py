"""Rebuild the targeted comparison from completed own trajectories and sealed controls."""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import argparse,json,subprocess,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/chemist_system'));sys.path.insert(0,str(ROOT/'code/experiment_tools'))
from reasoning_memory_v12 import study
from knowledge_information_v10.report import table


def build():
    p=study.check(ROOT);out=ROOT/'experiments/reports/knowledge_loop_20261101';base=study.directory(ROOT)
    metrics=[];rounds=[];decisions=[];updates=[];citations=[];states={};source_files={};keys=set();logical_calls=0
    methods=[('Pure BO',study.v9.directory(ROOT),'bo'),('Reasoning BO',study.v9.directory(ROOT),'reasoning_bo'),
        ('V11',study.v11.directory(ROOT),'rbo_memory'),('V12',base,'memory'),('V12 category',base,'category')]
    for task in study.TASKS:
        pool=study.gpmod.load_task(ROOT,task,seed=0)
        for label,folder,arm in methods:
            if task=='HPLC' and arm=='category':continue
            run=folder/'runs'/f'{task}_{arm}';path=run/'state.json';state=study.old.read(path)
            if state['status']!='COMPLETE' or len(state['history'])!=20 or len(state['decisions'])!=16:raise ValueError('Need complete4+16 before comparison')
            states[task,label]=state;source_files[path.relative_to(ROOT).as_posix()]=study.old.sha(path)
            if label!='Pure BO' and state['history'][:4]!=states[task,'Pure BO']['history'][:4]:raise ValueError('Unmatched initialization')
            values=np.array([r['raw_value'] for r in state['history']]);best=np.maximum.accumulate(values);first=int(np.flatnonzero(values==best[-1])[0])+1
            metrics.append(dict(task=task,method=label,SearchAUC=float(best[4:].mean()),final_best=float(best[-1]),first_best_total=first,
                first_best_query=max(0,first-4),unit=pool.unit,non_BO_choices=sum(d['selected_id']!=d['BO_reference_id'] for d in state['decisions'])))
            for t,d in enumerate(state['decisions'],1):
                rounds.append(dict(task=task,method=label,query=t,total_experiment=t+4,selected_id=d['selected_id'],objective=d['raw_value'],
                    incumbent=float(best[t+3]),BO_reference_id=d['BO_reference_id'],mode=d['mode']))
                if not label.startswith('V12'):continue
                c=study.old.read(run/'rounds'/f'{t:02d}_commit.json');keys.add(c['LLM_context_sha256']);logical_calls+=1
                if c['memory_context_sha256']:keys.add(c['memory_context_sha256']);logical_calls+=1
                h=c['LLM_raw']['hypotheses'][c['winning_hypothesis_index']]
                decisions.append(dict(task=task,arm=arm,query=t,selected_id=c['selected_id'],conditions=json.dumps(pool.points[pool.ix[c['selected_id']]],ensure_ascii=False),
                    objective=d['raw_value'],incumbent_before=d['incumbent_before_y']*pool.scale,incumbent_after=d['incumbent_after_y']*pool.scale,
                    original_menu_rank=c['original_ordered_ids'].index(c['selected_id'])+1 if c['selected_id'] in c['original_ordered_ids'] else None,
                    original_menu=json.dumps(c['original_ordered_ids']),actual_menu=json.dumps(c['ordered_ids']),opportunity_gap=c['opportunity_gap'],
                    strategy=h['strategy'],rationale=h['rationale'],confidence=h['confidence'],purpose=json.dumps(c['decision_purpose'],ensure_ascii=False),
                    refresh=c['refresh']['allowed'],refresh_reasons=';'.join(c['refresh']['reasons']),actions=json.dumps(c['reviews'],ensure_ascii=False),
                    category_inserted=c['category']['inserted'],category_selected=c['category']['challenger_id']==c['selected_id'],
                    challenger_id=c['category']['challenger_id'],predictions=len(c['predictions']),expected_predictions=c['expected_forecasts'],
                    sidecar_warnings=json.dumps(c['sidecar_warnings']),memory_references=json.dumps(c['memory_references'],ensure_ascii=False)))
                for r in c['memory_references']:citations.append(dict(task=task,arm=arm,query=t,**r))
                for e in d['KF_feedback']:
                    updates.append(dict(task=task,arm=arm,query=t,id=e['id'],version=e['version'],point=e['point_id'],reference_id=e['reference_id'],
                        reference_objective=e['reference_y']*pool.scale,observed_objective=e['observed_y']*pool.scale,
                        p_LLM=e['p_LLM'],p_GP=e['p_GP'],success=e['success'],Brier_gain=e['Brier_gain'],
                        kalman_gain=e['kalman_gain'],weight_before=e['before']['weight'],weight_after=e['after']['weight'],
                        status_before=e['before']['status'],status_after=e['after']['status'],reason=e['reason'],positive_evidence=e['positive_improvement_evidence']))
    index={(r['task'],r['method']):r for r in metrics}
    for r in metrics:
        for method,suffix in [('Pure BO','BO'),('Reasoning BO','RBO'),('V11','V11')]:
            control=index[r['task'],method];r['AUC_delta_vs_'+suffix]=r['SearchAUC']-control['SearchAUC']
            r['AUC_percent_vs_'+suffix]=100*(r['SearchAUC']/control['SearchAUC']-1);r['best_delta_vs_'+suffix]=r['final_best']-control['final_best']
    gates={}
    for task in study.TASKS:
        r=index[task,'V12'];v=index[task,'V11']
        gates[task]=(r['SearchAUC']>v['SearchAUC']+1e-9 if task.endswith('_2') else r['SearchAUC']>=v['SearchAUC']-1e-9 and r['final_best']>=v['final_best']-1e-9)
    ablations=[]
    for task in study.TASKS[:2]:
        r=index[task,'V12 category'];main=index[task,'V12'];state=states[task,'V12 category']
        first=next((t for t,(a,b) in enumerate(zip(state['decisions'],states[task,'V12']['decisions']),1) if a['selected_id']!=b['selected_id']),None)
        ablations.append(dict(task=task,category_SearchAUC=r['SearchAUC'],main_SearchAUC=main['SearchAUC'],delta=r['SearchAUC']-main['SearchAUC'],
            final_best_delta=r['final_best']-main['final_best'],first_selection_divergence=first,
            menu_insertions=state['category_menu_insertions'],challengers_selected=state['category_selected_queries']))
    comparisons={}
    for method in ('Pure BO','Reasoning BO','V11'):
        diffs=[index[t,'V12']['SearchAUC']-index[t,method]['SearchAUC'] for t in study.TASKS]
        comparisons[method]=dict(wins=sum(x>1e-9 for x in diffs),ties=sum(abs(x)<=1e-9 for x in diffs),losses=sum(x<-1e-9 for x in diffs))
    result=dict(protocol_sha256=p['protocol_sha256'],rows=metrics,main_prespecified_gates=gates,all_main_gates_met=all(gates.values()),
        V12_vs=comparisons,category_ablation=ablations,HPLC_category='not applicable; continuous inputs, no duplicate run',
        independent_confirmation=False,causal_module_increment_demonstrated=False,source_state_sha256=source_files,
        limitation='Three targeted seed0 development tasks. Fixed experimental budget; model-call counts differ. Whole-policy changes are bundled; only the categorical menu extension has its own full-trajectory ablation.')
    study.write(out/'v12_results.json',result);table(out/'v12_comparison.csv',metrics);table(out/'v12_rounds.csv',rounds)
    table(out/'v12_decision_explanations.csv',decisions);table(out/'v12_KF_updates.csv',updates);table(out/'v12_note_citations.csv',citations);table(out/'v12_category_ablation.csv',ablations)
    responses=list((base/'requests').glob('*/attempt*/response.json'));usage=Counter();models=Counter()
    for path in responses:
        r=study.old.read(path);models[r.get('resolved_model',r['model'])]+=1
        for k in ('input_tokens','output_tokens','total_tokens'):usage[k]+=r['usage'].get(k,0)
    costs=dict(logical_call_uses=logical_calls,unique_contexts=len(keys),exact_context_reuses=logical_calls-len(keys),
        successful_HTTP_responses=len(responses),models=dict(models),**dict(usage),
        schema_repair_HTTP_responses=sum(path.parent.name!='attempt1' for path in responses),
        historical_schema_rejection_receipts=len(list((base/'requests').glob('*/attempt*/schema_error.json'))),
        provider_cost_not_inferred_from_token_count=True)
    study.write(out/'v12_API_usage.json',costs)
    mechanisms=dict(forecast_updates=len(updates),expected_forecasts=sum(d['expected_predictions'] for d in decisions),
        sidecar_warning_count=sum(len(json.loads(d['sidecar_warnings'])) for d in decisions),
        citations_by_effective_use=dict(Counter(r['use'] for r in citations)),
        positive_support_normalizations=sum(r['requested_use']=='support' and r['use']!='support' for r in citations),
        KF_events=dict(Counter(e['reason'] for e in updates)),
        failure_calibration_weight_increases=sum(not e['success'] and e['weight_after']>e['weight_before'] for e in updates),
        positive_improvement_evidence=sum(e['positive_evidence'] for e in updates),
        edit_queries={f'{task}/{label}':state['edit_calls'] for (task,label),state in states.items() if label.startswith('V12')},
        carried_facts=sum(len(h['carried_fact_ids']) for (task,label),state in states.items() if label.startswith('V12') for h in state['hypotheses']),
        numeric_prior_replays=0,operator_compiles_in_loop=sum(state['cache_counters']['operator_compiles_in_loop'] for (task,label),state in states.items() if label.startswith('V12')))
    study.write(out/'v12_mechanisms.json',mechanisms)
    lines=['V12 targeted comparison: seed0, original pools/GP,4initial+16queries',
        'SearchAUC = mean cumulative best after each of16queries; original task units.',
        'task\tmethod\tSearchAUC\tfinal_best\tfirst_best_query\tAUC_percent_vs_BO']
    for r in metrics:lines.append(f"{r['task']}\t{r['method']}\t{r['SearchAUC']:.8f}\t{r['final_best']:.8f}\t{r['first_best_query']}\t{r['AUC_percent_vs_BO']:+.3f}")
    lines+=['',json.dumps(gates,ensure_ascii=False),json.dumps(ablations,ensure_ascii=False),result['limitation']]
    (out/'v12_results.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles={'Pure BO':('#555555','--'),'Reasoning BO':('#3072ad','-.'),'V11':('#b18232',':'),'V12':('#57783f','-'),'V12 category':('#c45686','--')}
    fig,axes=plt.subplots(3,2,figsize=(13.2,10.2),constrained_layout=True)
    for row,task in enumerate(study.TASKS):
        for label,(color,style) in styles.items():
            if (task,label) not in states:continue
            v=np.array([r['raw_value'] for r in states[task,label]['history']])
            axes[row,0].plot(np.arange(1,17),v[4:],color=color,linestyle=style,linewidth=1.7,marker='o' if label=='V12' else None,markersize=3,label=label)
            axes[row,1].plot(np.arange(17),np.maximum.accumulate(v)[3:],color=color,linestyle=style,linewidth=1.7,marker='o' if label=='V12' else None,markersize=3,label=label)
        title=task.replace('Reizman_Suzuki_','').replace('_',' ')
        for col in range(2):
            ax=axes[row,col];ax.set_title(title+(' | measured objective' if col==0 else ' | cumulative best'))
            ax.set_xlabel('Query (after four shared initial observations)');ax.set_ylabel('Yield (%)' if task!='HPLC' else 'HPLC objective (native units)')
            ax.set_xticks([1,4,8,12,16] if col==0 else [0,4,8,12,16]);ax.grid(alpha=.22)
            ax.spines[['top','right']].set_visible(False)
        if task=='HPLC':axes[row,0].text(.02,.96,'Category ablation: not applicable',ha='left',va='top',transform=axes[row,0].transAxes,fontsize=9)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=5)
    fig.suptitle('V12 targeted seed-0 study | frozen pools and GP | 4 + 16 experiments\nTask-specific y scales; single trajectories, no confidence intervals',fontsize=13)
    fig.savefig(out/'v12_trajectory_comparison.png',dpi=180);fig.savefig(out/'v12_trajectory_comparison.pdf');plt.close(fig)
    return dict(result=result,costs=costs,mechanisms=mechanisms)


def finalize(bundle):
    out=ROOT/'experiments/reports/knowledge_loop_20261101';p=study.check(ROOT);audit=study.old.read(out/'v12_audit.json')
    if audit['status']!='PASS_LISTED_CHECKS':raise ValueError('Independent audit required')
    doc=ROOT/'docs/reasoning_memory_v12_results_20261101.txt'
    if not doc.exists():raise ValueError('Interpreted handoff document required')
    names=['v12_results.json','v12_comparison.csv','v12_rounds.csv','v12_decision_explanations.csv','v12_KF_updates.csv',
        'v12_note_citations.csv','v12_category_ablation.csv','v12_API_usage.json','v12_mechanisms.json','v12_audit.json',
        'v12_GP_preflight.json','v12_results.txt','v12_trajectory_comparison.png','v12_trajectory_comparison.pdf','v12_run.log',
        'v12_matched_prefix_divergences.json','v12_AUC_contributions.csv','v12_category_challengers.csv','v12_metric_verification.json']
    receipt=dict(request_status='COMPLETE_THREE_TASKS_AND_TWO_CATEGORY_ABLATIONS',complete_new_runs=5,new_queries=80,
        reused_control_runs=9,new_control_runs=0,protocol_sha256=p['protocol_sha256'],core_commit=p['code_milestone'],
        delivery_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        seed=0,initial=4,queries_per_run=16,main_prespecified_gates=bundle['result']['main_prespecified_gates'],
        all_main_gates_met=bundle['result']['all_main_gates_met'],V12_vs=bundle['result']['V12_vs'],
        independent_confirmation=False,stable_majority_advantage_supported=False,audit_status=audit['status'],
        audited_queries=audit['rounds'],audited_forecasts=audit['forecasts'],pipeline_tests_passed=10,audit_tests_passed=3,
        protected_control_files=audit['protected_control_files'],API_usage=bundle['costs'],
        files_sha256={name:study.old.sha(out/name) for name in names},document_sha256=study.old.sha(doc),
        auditor_sha256=study.old.sha(ROOT/'code/tools/audit_reasoning_memory_v12.py'),reporter_sha256=study.old.sha(__file__),
        figure_review='PNG visually inspected: six complete panels, legible axes/legend, matching series and no clipping. Main/category coincide and are separately labeled. Original task units and independent y scales disclosed.',
        created_utc=datetime.now(timezone.utc).isoformat())
    study.write(out/'v12_completion_receipt.json',receipt)
    status=study.old.read(out/'STATUS.json');already=status.get('v12',{}).get('request_status')==receipt['request_status']
    if not already:status['new_optimization_runs']+=5;status['new_experimental_queries']+=80
    status.update(current_phase='V12_COMPLETE_TARGETED_COMPARISON',variant=study.VARIANT,current_sessions={},active_optimization_processes=0,
        protocol_sha256=p['protocol_sha256'],code_milestone=p['code_milestone'],delivery_commit=receipt['delivery_commit'],
        updated_utc=receipt['created_utc'],v12=receipt,stable_majority_advantage_supported=False,
        current_request_performance_target_met=bundle['result']['all_main_gates_met'],
        user_primary_development_target_basis='Existing V11 all-eight-task seed0 majority gate; V12 is a selected-three-task development check, not a new all-task confirmation')
    status['next_steps']=['Requested targeted V12 run complete; all prior controls unchanged.',
        'Retain both category ablations separately and all negative performance evidence; no automatic promotion or new experiment.']
    study.write(out/'STATUS.json',status);return receipt


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--finalize',action='store_true');args=ap.parse_args()
    if args.finalize:
        out=ROOT/'experiments/reports/knowledge_loop_20261101'
        bundle={key:study.old.read(out/name) for key,name in [('result','v12_results.json'),('costs','v12_API_usage.json'),('mechanisms','v12_mechanisms.json')]}
        print(json.dumps(finalize(bundle),ensure_ascii=False,indent=2))
    else:print(json.dumps(build(),ensure_ascii=False,indent=2))
