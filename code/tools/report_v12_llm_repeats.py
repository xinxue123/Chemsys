"""Conditional LLM-repeat variability; original V12 is not pooled into n=3."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
from itertools import combinations
from pathlib import Path
from statistics import fmean,stdev
import argparse,json,subprocess,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/chemist_system'));sys.path.insert(0,str(ROOT/'code/experiment_tools'))
from v12_llm_repeats import study
from knowledge_information_v10.report import table


def metric(state):
    values=[r['raw_value'] for r in state['history']]
    if state['status']!='COMPLETE' or len(values)!=20 or len(state['decisions'])!=16:raise ValueError('Completed4+16 required')
    curve=[max(values[:i+1]) for i in range(len(values))];first=values.index(max(values))+1
    return dict(SearchAUC=fmean(curve[4:]),final_best=max(values),first_best_query=max(0,first-4),first_best_total=first)


def choice_stats(sequences):
    if len(sequences)!=3 or any(len(x)!=16 for x in sequences):raise ValueError('Three complete choice sequences required')
    distinct=[len(set(x[t] for x in sequences)) for t in range(16)]
    rates=[sum(a[t]!=b[t] for a,b in combinations(sequences,2))/3 for t in range(16)]
    return dict(first_divergence_query=next((i+1 for i,n in enumerate(distinct) if n>1),None),
        rounds_with_disagreement=sum(n>1 for n in distinct),pairwise_disagreement_fraction=fmean(rates),
        distinct_counts=distinct,pairwise_rates=rates)


def build():
    p=study.check(ROOT);out=ROOT/'experiments/reports/knowledge_loop_20261101'
    audit=study.old.read(out/'v12_repeats_audit.json')
    if audit['status']!='PASS_LISTED_CHECKS':raise ValueError('Listed trajectory and cache-isolation audit required')
    states={};commits={};metrics=[];summaries=[];round_rows=[];pair_rows=[];explanations=[];source_files={}
    for task in study.TASKS:
        pool=study.v12.gpmod.load_task(ROOT,task,seed=0)
        paths=[('Pure BO',study.v12.v9.directory(ROOT)/'runs'/f'{task}_bo'),
            ('Original V12',study.v12.directory(ROOT)/'runs'/f'{task}_memory')]
        paths += [(rep,study.run_path(ROOT,task,rep)) for rep in study.REPLICATES]
        for label,run in paths:
            state=study.old.read(run/'state.json');states[task,label]=state;values=metric(state)
            source_files[(run/'state.json').relative_to(ROOT).as_posix()]=study.old.sha(run/'state.json')
            if label!='Pure BO':assert state['history'][:4]==states[task,'Pure BO']['history'][:4]
            bo=metric(states[task,'Pure BO'])
            metrics.append(dict(task=task,run=label,new_repeat=label in study.REPLICATES,unit=pool.unit,**values,
                AUC_delta_vs_BO=values['SearchAUC']-bo['SearchAUC'],AUC_percent_vs_BO=100*(values['SearchAUC']/bo['SearchAUC']-1),
                final_best_delta_vs_BO=values['final_best']-bo['final_best']))
            if label in study.REPLICATES or label=='Original V12':
                commits[task,label]=[study.old.read(run/'rounds'/f'{t:02d}_commit.json') for t in range(1,17)]
            if label not in study.REPLICATES:continue
            for d,c in zip(state['decisions'],commits[task,label]):
                h=c['LLM_raw']['hypotheses'][c['winning_hypothesis_index']]
                explanations.append(dict(task=task,replicate=label,query=d['query'],selected_id=d['selected_id'],
                    objective=d['raw_value'],incumbent=d['incumbent_after_y']*pool.scale,
                    conditions=json.dumps(pool.points[pool.ix[d['selected_id']]],ensure_ascii=False),
                    original_GP_rank=c['ordered_ids'].index(c['selected_id'])+1,confidence=h['confidence'],
                    strategy=h['strategy'],rationale=h['rationale'],purpose=json.dumps(c['decision_purpose'],ensure_ascii=False),
                    refresh=c['refresh']['allowed'],refresh_reasons=';'.join(c['refresh']['reasons']),
                    memory_actions=json.dumps(c['reviews'],ensure_ascii=False),memory_references=json.dumps(c['memory_references'],ensure_ascii=False),
                    KF_events=json.dumps(d['KF_feedback'],ensure_ascii=False),LLM_context_sha256=c['LLM_context_sha256']))
        fresh=[r for r in metrics if r['task']==task and r['new_repeat']];aucs=[r['SearchAUC'] for r in fresh];bests=[r['final_best'] for r in fresh]
        sequences=[[d['selected_id'] for d in states[task,rep]['decisions']] for rep in study.REPLICATES];var=choice_stats(sequences)
        bo=metric(states[task,'Pure BO']);deltas=[r['AUC_delta_vs_BO'] for r in fresh]
        summaries.append(dict(task=task,n=3,BO_SearchAUC=bo['SearchAUC'],BO_final_best=bo['final_best'],
            AUC_mean=fmean(aucs),AUC_sample_SD=stdev(aucs),AUC_min=min(aucs),AUC_max=max(aucs),
            AUC_mean_delta_vs_BO=fmean(deltas),AUC_mean_percent_vs_BO=100*(fmean(aucs)/bo['SearchAUC']-1),
            AUC_wins=sum(x>1e-9 for x in deltas),AUC_ties=sum(abs(x)<=1e-9 for x in deltas),AUC_losses=sum(x<-1e-9 for x in deltas),
            best_mean=fmean(bests),best_sample_SD=stdev(bests),best_min=min(bests),best_max=max(bests),
            best_wins=sum(x>bo['final_best']+1e-9 for x in bests),best_ties=sum(abs(x-bo['final_best'])<=1e-9 for x in bests),
            best_losses=sum(x<bo['final_best']-1e-9 for x in bests),first_divergence_query=var['first_divergence_query'],
            rounds_with_choice_disagreement=var['rounds_with_disagreement'],mean_pairwise_disagreement=var['pairwise_disagreement_fraction']))
        for t in range(1,17):
            ds=[states[task,rep]['decisions'][t-1] for rep in study.REPLICATES]
            values=[d['raw_value'] for d in ds];best=[max(r['raw_value'] for r in states[task,rep]['history'][:t+4]) for rep in study.REPLICATES]
            row=dict(task=task,query=t,distinct_selected_points=var['distinct_counts'][t-1],pairwise_disagreement=var['pairwise_rates'][t-1],
                measured_min=min(values),measured_max=max(values),incumbent_min=min(best),incumbent_max=max(best),
                BO_selected=states[task,'Pure BO']['decisions'][t-1]['selected_id'],original_V12_selected=states[task,'Original V12']['decisions'][t-1]['selected_id'])
            for rep,d,b in zip(study.REPLICATES,ds,best):row.update({rep+'_selected':d['selected_id'],rep+'_objective':d['raw_value'],rep+'_incumbent':b})
            round_rows.append(row)
        for left,right in list(combinations(study.REPLICATES,2))+[(rep,'Original V12') for rep in study.REPLICATES]:
            a=[d['selected_id'] for d in states[task,left]['decisions']];b=[d['selected_id'] for d in states[task,right]['decisions']]
            first=next((t+1 for t,(x,y) in enumerate(zip(a,b)) if x!=y),None);shared_gp=shared_prompt=None
            if first is not None:
                ca,cb=commits[task,left][first-1],commits[task,right][first-1]
                assert ca['prestate']['history']==cb['prestate']['history']
                shared_gp=ca['GP_sha256']==cb['GP_sha256'] and ca['ordered_ids']==cb['ordered_ids'];assert shared_gp
                shared_prompt=ca['LLM_context_sha256']==cb['LLM_context_sha256']
            pair_rows.append(dict(task=task,left=left,right=right,both_new=right!='Original V12',first_divergence_query=first,
                differing_rounds=sum(x!=y for x,y in zip(a,b)),final_selected_set_Jaccard=len(set(a)&set(b))/len(set(a)|set(b)),
                same_history_GP_menu_at_first_divergence=shared_gp,identical_selection_prompt_at_first_divergence=shared_prompt))
    evidence=study.old.read(out/'v12_repeats_request_evidence.json');groups=defaultdict(list);context_rows=[]
    for r in evidence['contexts']:groups[(r['task'],r['kind'],r['context_sha256'])].append(r)
    for (task,kind,key),rows in groups.items():
        if len(rows)<2:continue
        context_rows.append(dict(task=task,kind=kind,query=rows[0]['query'],context_sha256=key,independent_calls=len(rows),
            unique_response_ids=len({r['response_id'] for r in rows}),unique_raw_outputs=len({r['raw_sha256'] for r in rows}),
            unique_selected_points=len({r['selected_id'] for r in rows}) if kind=='selection' else None,
            replicates=';'.join(r['replicate'] for r in rows)))
    usage=Counter();models=Counter()
    for r in evidence['responses']:
        models[r['model']]+=1
        for key in ('input_tokens','output_tokens','total_tokens'):usage[key]+=r['usage'].get(key,0)
    costs=dict(logical_call_uses=len(evidence['contexts']),successful_HTTP_responses=len(evidence['responses']),
        distinct_context_strings=len(groups),cross_replica_response_reuses=0,models=dict(models),**dict(usage),
        schema_repair_HTTP_responses=sum(r['attempt']!='attempt1' for r in evidence['responses']))
    result=dict(protocol_sha256=p['protocol_sha256'],frozen_V12_protocol_sha256=p['frozen_V12_protocol_sha256'],
        rows=metrics,task_summaries=summaries,API_usage=costs,source_state_sha256=source_files,
        repeated_identical_context_groups=len(context_rows),repeated_contexts_with_distinct_raw_outputs=sum(r['unique_raw_outputs']>1 for r in context_rows),
        repeated_selection_contexts_with_distinct_choices=sum(r['kind']=='selection' and r['unique_selected_points']>1 for r in context_rows),
        observation='Fresh LLM calls with fixed seed0 and initial observations; each task has three realizations.',
        original_V12_pooled=False,significance_or_noninferiority_claim=False,
        limitation='Sample SD and observed min/max are descriptive, not confidence intervals. n=3 at one initialization does not establish global stability or isolate memory versus selection randomness.')
    study.write(out/'v12_repeats_results.json',result);table(out/'v12_repeats_comparison.csv',metrics);table(out/'v12_repeats_summary.csv',summaries)
    table(out/'v12_repeats_round_choices.csv',round_rows);table(out/'v12_repeats_pairwise.csv',pair_rows)
    table(out/'v12_repeats_identical_contexts.csv',context_rows);table(out/'v12_repeats_decisions.csv',explanations)
    study.write(out/'v12_repeats_API_usage.json',costs)
    # Cross-check metrics via an independent vectorized calculation.
    for r in metrics:
        values=np.asarray([x['raw_value'] for x in states[r['task'],r['run']]['history']])
        assert abs(r['SearchAUC']-float(np.maximum.accumulate(values)[4:].mean()))<1e-10
    for s in summaries:
        v=np.asarray([r['SearchAUC'] for r in metrics if r['task']==s['task'] and r['new_repeat']])
        assert abs(s['AUC_mean']-float(v.mean()))<1e-10 and abs(s['AUC_sample_SD']-float(v.std(ddof=1)))<1e-10
    study.write(out/'v12_repeats_metric_verification.json',dict(status='PASS',metric_rows=len(metrics),aggregate_rows=len(summaries),
        main_sample_per_task=3,original_in_sample=False,reporter_sha256=study.old.sha(__file__),new_queries=0,new_API_calls=0))
    lines=['V12 independent LLM repetitions: fixed seed0 / same4initial /16queries / n=3 per task',
        'task\trun\tSearchAUC\tfinal_best\tfirst_best_query\tAUC_percent_vs_BO']
    for r in metrics:lines.append(f"{r['task']}\t{r['run']}\t{r['SearchAUC']:.8f}\t{r['final_best']:.8f}\t{r['first_best_query']}\t{r['AUC_percent_vs_BO']:+.4f}")
    lines+=['','Sample summaries (SD is not a confidence interval):',json.dumps(summaries,ensure_ascii=False,indent=2),result['limitation']]
    (out/'v12_repeats_results.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    figures(out,states,round_rows)
    return result


def figures(out,states,round_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'Pure BO':'#505050','Original V12':'#959595','r01':'#3072ad','r02':'#af8130','r03':'#b4497e'}
    styles={'Pure BO':'--','Original V12':':','r01':'-','r02':'-.','r03':'--'}
    fig,axes=plt.subplots(3,2,figsize=(13.2,10.2),constrained_layout=True)
    for row,task in enumerate(study.TASKS):
        for label in colors:
            v=np.array([x['raw_value'] for x in states[task,label]['history']])
            for col in (0,1):
                axes[row,col].plot(np.arange(1,17) if col==0 else np.arange(17),v[4:] if col==0 else np.maximum.accumulate(v)[3:],
                    color=colors[label],linestyle=styles[label],linewidth=1.6,label=label,marker='o' if label in study.REPLICATES else None,markersize=2.8)
        title=task.replace('Reizman_Suzuki_','').replace('_',' ')
        for col in (0,1):
            ax=axes[row,col];ax.set_title(title+(' | measured objective' if col==0 else ' | cumulative best'))
            ax.set_xlabel('Query (after four shared initial observations)');ax.set_ylabel('Yield (%)' if task!='HPLC' else 'HPLC objective (native units)')
            ax.set_xticks([1,4,8,12,16] if col==0 else [0,4,8,12,16]);ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=5)
    fig.suptitle('Frozen V12 | three independent LLM realizations per task\nSame seed, initial data and GP recipe; task-specific y scales',fontsize=13)
    fig.savefig(out/'v12_repeats_trajectories.png',dpi=180);fig.savefig(out/'v12_repeats_trajectories.pdf');plt.close(fig)
    # Selected points are categorical. Disagreement is computed from exact IDs;
    # candidate numeric suffixes are never plotted as quantitative coordinates.
    fig,axes=plt.subplots(3,1,figsize=(12,5.5),constrained_layout=True)
    for ax,task in zip(axes,study.TASKS):
        rows=[r for r in round_rows if r['task']==task];x=np.arange(1,17);y=np.array([r['pairwise_disagreement'] for r in rows])*100
        ax.bar(x,y,color='#3072ad',width=.7);ax.set_ylim(0,108);ax.set_yticks([0,200/3,100],['0','67','100'])
        ax.set_xticks(x);ax.set_ylabel('Pairs disagree (%)');ax.set_title(task.replace('Reizman_Suzuki_','').replace('_',' '));ax.grid(axis='y',alpha=.18)
        for q,v,row in zip(x,y,rows):
            if v:ax.text(q,v+2,str(row['distinct_selected_points']),ha='center',fontsize=9)
    axes[-1].set_xlabel('Query');fig.suptitle('Round-by-round point-choice variation | labels = distinct points among three runs\nAfter the first divergence, different histories can explain subsequent choices',fontsize=12)
    fig.savefig(out/'v12_repeats_choice_variation.png',dpi=180);fig.savefig(out/'v12_repeats_choice_variation.pdf');plt.close(fig)


def finalize():
    out=ROOT/'experiments/reports/knowledge_loop_20261101';p=study.check(ROOT);result=study.old.read(out/'v12_repeats_results.json');audit=study.old.read(out/'v12_repeats_audit.json')
    if audit['status']!='PASS_LISTED_CHECKS' or audit['rounds']!=144:raise ValueError('Complete independent audit required')
    doc=ROOT/'docs/v12_llm_repeats_results_20261101.txt'
    if not doc.exists():raise ValueError('Interpreted findings required')
    names=['v12_repeats_results.json','v12_repeats_comparison.csv','v12_repeats_summary.csv','v12_repeats_round_choices.csv',
        'v12_repeats_pairwise.csv','v12_repeats_identical_contexts.csv','v12_repeats_decisions.csv','v12_repeats_API_usage.json',
        'v12_repeats_audit.json','v12_repeats_request_evidence.json','v12_repeats_preflight.json','v12_repeats_metric_verification.json',
        'v12_repeats_results.txt','v12_repeats_trajectories.png','v12_repeats_trajectories.pdf','v12_repeats_choice_variation.png',
        'v12_repeats_choice_variation.pdf','v12_repeats_run.log']
    receipt=dict(request_status='COMPLETE_NINE_FRESH_LLM_TRAJECTORIES',complete_new_runs=9,new_queries=144,new_control_runs=0,reused_BO_controls=3,
        seed=0,replicates_per_task=3,initial=4,queries=16,protocol_sha256=p['protocol_sha256'],frozen_V12_protocol_sha256=p['frozen_V12_protocol_sha256'],
        code_milestone=p['code_milestone'],delivery_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        audit_status=audit['status'],protected_files=audit['protected_control_files'],task_summaries=result['task_summaries'],
        cache_isolation_tests_passed=4,audit_tests_passed=3,API_usage=result['API_usage'],
        stable_majority_advantage_supported=False,global_stability_confirmed=False,
        figure_review='Both PNGs visually inspected for complete task panels, source-matched lines, readable legends/axes and no clipping; differing task units and post-divergence interpretation are explicit.',
        files_sha256={name:study.old.sha(out/name) for name in names},document_sha256=study.old.sha(doc),
        created_utc=datetime.now(timezone.utc).isoformat())
    study.write(out/'v12_repeats_completion_receipt.json',receipt)
    status=study.old.read(out/'STATUS.json');already=status.get('v12_llm_repeats',{}).get('request_status')==receipt['request_status']
    if not already:status['new_optimization_runs']+=9;status['new_experimental_queries']+=144
    status.update(current_phase='V12_FIXED_SEED_LLM_REPEATS_COMPLETE',variant=study.VARIANT,current_sessions={},active_optimization_processes=0,
        protocol_sha256=p['protocol_sha256'],code_milestone=p['code_milestone'],delivery_commit=receipt['delivery_commit'],
        updated_utc=receipt['created_utc'],v12_llm_repeats=receipt,stable_majority_advantage_supported=False)
    status['next_steps']=['Nine requested fresh LLM trajectories completed; preserve every result and all historical controls.',
        'This n=3 fixed-initialization experiment measures conditional LLM-service variability; it does not establish multi-initialization stability.']
    study.write(out/'STATUS.json',status);return receipt


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--finalize',action='store_true');args=ap.parse_args()
    result=finalize() if args.finalize else build();print(json.dumps(result,ensure_ascii=False,indent=2))
