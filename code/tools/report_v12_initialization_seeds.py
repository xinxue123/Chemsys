"""Paired initialization comparisons; never pool the historical seed0 runs."""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from statistics import fmean,stdev
import argparse,json,subprocess,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/chemist_system'));sys.path.insert(0,str(ROOT/'code/experiment_tools'))
from v12_initialization_seeds import study
from knowledge_information_v10.report import table
from report_v12_llm_repeats import metric


def summarize_pairs(pairs):
    rows=[]
    for task in study.TASKS:
        ps=[r for r in pairs if r['task']==task]
        if len(ps)!=3 or {r['initialization_seed'] for r in ps}!={1,2,3}:raise ValueError('Exactly three registered paired seeds required')
        row=dict(task=task,n_paired_seeds=3)
        for label in ('AUC','best'):
            bo=[r['BO_'+label] for r in ps];v=[r['V12_'+label] for r in ps];diff=[r[label+'_delta'] for r in ps]
            row.update({label+'_BO_mean':fmean(bo),label+'_BO_sample_SD':stdev(bo),label+'_V12_mean':fmean(v),label+'_V12_sample_SD':stdev(v),
                label+'_paired_delta_mean':fmean(diff),label+'_paired_delta_sample_SD':stdev(diff),
                label+'_paired_delta_min':min(diff),label+'_paired_delta_max':max(diff),
                label+'_ratio_of_means_percent':100*(fmean(v)/fmean(bo)-1),label+'_mean_pair_percent':fmean([r[label+'_percent'] for r in ps]),
                label+'_wins':sum(x>1e-9 for x in diff),label+'_ties':sum(abs(x)<=1e-9 for x in diff),label+'_losses':sum(x<-1e-9 for x in diff)})
        rows.append(row)
    return rows


def build():
    p=study.check(ROOT,verify_controls=True);out=ROOT/'experiments/reports/knowledge_loop_20261101'
    audit=study.old.read(out/'v12_seeds_audit.json')
    if audit['status']!='PASS_LISTED_CHECKS' or audit['rounds']!=288:raise ValueError('Complete paired audit required')
    rows=[];pairs=[];rounds=[];initial=[];states={};sources={}
    for task,seed in study.PAIRS:
        pool=study.registered_pool(ROOT,task,seed,p);scores={}
        for arm in study.ARMS:
            path=study.run_path(ROOT,task,seed,arm)/'state.json';state=study.old.read(path);states[task,seed,arm]=state
            values=metric(state);scores[arm]=values
            sources[path.relative_to(ROOT).as_posix()]=study.old.sha(path)
            rows.append(dict(task=task,initialization_seed=seed,arm=arm,unit=pool.unit,initial_best=max(r['raw_value'] for r in state['history'][:4]),**values))
        bs=states[task,seed,'bo'];vs=states[task,seed,'memory'];assert bs['history'][:4]==vs['history'][:4]
        b,v=scores['bo'],scores['memory']
        pair=dict(task=task,initialization_seed=seed,unit=pool.unit,initial_best=max(r['raw_value'] for r in bs['history'][:4]),
            BO_AUC=b['SearchAUC'],V12_AUC=v['SearchAUC'],AUC_delta=v['SearchAUC']-b['SearchAUC'],AUC_percent=100*(v['SearchAUC']/b['SearchAUC']-1),
            BO_best=b['final_best'],V12_best=v['final_best'],best_delta=v['final_best']-b['final_best'],best_percent=100*(v['final_best']/b['final_best']-1),
            BO_first_best_query=b['first_best_query'],V12_first_best_query=v['first_best_query'],
            first_choice_divergence=next((i+1 for i,(x,y) in enumerate(zip(bs['decisions'],vs['decisions'])) if x['selected_id']!=y['selected_id']),None))
        pairs.append(pair)
        for i,r in enumerate(bs['history'][:4],1):
            initial.append(dict(task=task,initialization_seed=seed,initial_order=i,candidate_id=r['candidate_id'],raw_value=r['raw_value'],
                conditions=json.dumps(pool.points[pool.ix[r['candidate_id']]],ensure_ascii=False)))
        for t,(bd,vd) in enumerate(zip(bs['decisions'],vs['decisions']),1):
            rr=dict(task=task,initialization_seed=seed,query=t)
            for label,d,state in [('BO',bd,bs),('V12',vd,vs)]:
                rr.update({label+'_selected_id':d['selected_id'],label+'_objective':d['raw_value'],
                    label+'_incumbent':max(r['raw_value'] for r in state['history'][:t+4])})
            rr['same_selected_point']=bd['selected_id']==vd['selected_id'];rr['incumbent_delta']=rr['V12_incumbent']-rr['BO_incumbent'];rounds.append(rr)
    summaries=summarize_pairs(pairs);evidence=study.old.read(out/'v12_seeds_request_evidence.json');usage=Counter();models=Counter()
    for r in evidence['responses']:
        models[r['model']]+=1
        for key in ('input_tokens','output_tokens','total_tokens'):usage[key]+=r['usage'].get(key,0)
    costs=dict(logical_calls=len(evidence['contexts']),successful_HTTP_responses=len(evidence['responses']),
        memory_edits=sum(r['kind']=='memory' for r in evidence['contexts']),selection_calls=sum(r['kind']=='selection' for r in evidence['contexts']),
        schema_repair_HTTP_responses=sum(r['attempt']!='attempt1' for r in evidence['responses']),cross_run_response_reuses=0,models=dict(models),**dict(usage))
    # Independent vectorized calculation of every metric and the paired mean/SD.
    for r in rows:
        vals=np.asarray([x['raw_value'] for x in states[r['task'],r['initialization_seed'],r['arm']]['history']])
        assert abs(r['SearchAUC']-float(np.maximum.accumulate(vals)[4:].mean()))<1e-10
        assert r['final_best']==float(vals.max())
    for s in summaries:
        a=np.asarray([[r['BO_AUC'],r['V12_AUC']] for r in pairs if r['task']==s['task']]);d=a[:,1]-a[:,0]
        assert abs(s['AUC_paired_delta_mean']-float(d.mean()))<1e-10
        assert abs(s['AUC_paired_delta_sample_SD']-float(d.std(ddof=1)))<1e-10
    result=dict(protocol_sha256=p['protocol_sha256'],frozen_V12_protocol_sha256=p['frozen_V12_protocol_sha256'],
        rows=rows,pairs=pairs,task_summaries=summaries,API_usage=costs,source_state_sha256=sources,
        AUC_observed_pair_wins=sum(r['AUC_delta']>1e-9 for r in pairs),AUC_observed_pair_ties=sum(abs(r['AUC_delta'])<=1e-9 for r in pairs),
        AUC_observed_pair_losses=sum(r['AUC_delta']<-1e-9 for r in pairs),
        final_best_observed_pair_wins=sum(r['best_delta']>1e-9 for r in pairs),final_best_observed_pair_ties=sum(abs(r['best_delta'])<=1e-9 for r in pairs),
        final_best_observed_pair_losses=sum(r['best_delta']<-1e-9 for r in pairs),
        seed0_in_sample=False,initialization_seeds=[1,2,3],native_GP_tie_display_streams_frozen=True,
        statistical_significance_or_noninferiority_claim=False,
        limitation='Three paired initializations per task, one new LLM trajectory per initialization. Seeds1/2 have prior development exposure; initialization and LLM-service variance are not separated. Sample SD/min/max are descriptive, not confidence intervals; nine pairs are not nine independent datasets.')
    for name,data in [('v12_seeds_metrics.csv',rows),('v12_seeds_paired_comparison.csv',pairs),('v12_seeds_summary.csv',summaries),
                      ('v12_seeds_round_choices.csv',rounds),('v12_seeds_initialization.csv',initial)]:table(out/name,data)
    study.write(out/'v12_seeds_results.json',result);study.write(out/'v12_seeds_API_usage.json',costs)
    study.write(out/'v12_seeds_metric_verification.json',dict(status='PASS',runs=18,pairs=9,task_summaries=3,seed0_pooled=False,
        reporter_sha256=study.old.sha(__file__),new_queries=0,new_API_calls=0))
    lines=['Frozen V12 versus BO | seeds1/2/3 | paired initialization |4+16',
           'task\tseed\tBO AUC\tV12 AUC\tAUC delta\tBO final\tV12 final\tBO/V12 first best q']
    for r in pairs:lines.append(f"{r['task']}\t{r['initialization_seed']}\t{r['BO_AUC']:.8f}\t{r['V12_AUC']:.8f}\t{r['AUC_delta']:+.8f}\t{r['BO_best']:.8f}\t{r['V12_best']:.8f}\t{r['BO_first_best_query']}/{r['V12_first_best_query']}")
    lines+=['',json.dumps(summaries,ensure_ascii=False,indent=2),result['limitation']]
    (out/'v12_seeds_results.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    figures(out,states,pairs);return result


def figures(out,states,pairs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(13.2,9.1),constrained_layout=True,sharex=True,sharey='row')
    for i,task in enumerate(study.TASKS):
        for j,seed in enumerate(study.SEEDS):
            ax=axes[i,j]
            for arm,label,color,style in [('bo','Pure BO','#505050','--'),('memory','V12','#3072ad','-')]:
                vals=np.array([r['raw_value'] for r in states[task,seed,arm]['history']]);curve=np.maximum.accumulate(vals)[3:]
                ax.plot(np.arange(17),curve,label=label,color=color,linestyle=style,linewidth=1.8,marker='o' if arm=='memory' else None,markersize=2.4)
            ax.set_title(task.replace('Reizman_Suzuki_','').replace('_',' ')+f' | seed {seed}')
            ax.set_xticks([0,4,8,12,16]);ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
            if j==0:ax.set_ylabel('Best yield (%)' if task!='HPLC' else 'Best peak area (native units)')
            if i==2:ax.set_xlabel('Query after four shared initial experiments')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2)
    fig.suptitle('Frozen V12 versus pure BO | paired initialization seeds 1, 2, 3\nOne fresh LLM trajectory per pair; common y scale within each task row',fontsize=13)
    fig.savefig(out/'v12_seeds_trajectories.png',dpi=180);fig.savefig(out/'v12_seeds_trajectories.pdf');plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(11.6,7.2),constrained_layout=True)
    for i,task in enumerate(study.TASKS):
        rs=[r for r in pairs if r['task']==task]
        for j,(field,label) in enumerate([('AUC_delta','SearchAUC difference'),('best_delta','Final-best difference')]):
            ax=axes[i,j];v=np.array([r[field] for r in rs]);y=np.arange(3)
            ax.barh(y,v,color=['#3072ad' if x>=0 else '#b4497e' for x in v],height=.55);ax.axvline(0,color='#555555',linewidth=.8)
            ax.set_yticks(y,[f'seed {r["initialization_seed"]}' for r in rs]);ax.invert_yaxis()
            ax.set_title(task.replace('Reizman_Suzuki_','').replace('_',' ')+' | '+label)
            bound=float(abs(v).max()) or 1.;ax.set_xlim(min(0,v.min())-.32*bound,max(0,v.max())+.32*bound)
            for yy,x in zip(y,v):ax.text(x+(.025*bound if x>=0 else -.025*bound),yy,f'{x:+.3f}',ha='left' if x>=0 else 'right',va='center',fontsize=10)
            ax.set_xlabel('V12 minus BO (yield percentage points)' if task!='HPLC' else 'V12 minus BO (peak-area units)')
            ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.18)
    fig.suptitle('Paired differences | positive favors V12\nEach bar is one complete 4 + 16 trajectory comparison; independent panel scales',fontsize=13)
    fig.savefig(out/'v12_seeds_paired_differences.png',dpi=180);fig.savefig(out/'v12_seeds_paired_differences.pdf');plt.close(fig)


def finalize():
    out=ROOT/'experiments/reports/knowledge_loop_20261101';p=study.check(ROOT,verify_controls=True)
    result=study.old.read(out/'v12_seeds_results.json');audit=study.old.read(out/'v12_seeds_audit.json')
    if audit['status']!='PASS_LISTED_CHECKS' or audit['rounds']!=288 or len(result['pairs'])!=9:raise ValueError('Complete paired results required')
    doc=ROOT/'docs/v12_initialization_seeds_results_20261101.txt'
    if not doc.exists():raise ValueError('Interpreted result document required')
    names=['v12_seeds_results.json','v12_seeds_metrics.csv','v12_seeds_paired_comparison.csv','v12_seeds_summary.csv',
        'v12_seeds_round_choices.csv','v12_seeds_initialization.csv','v12_seeds_API_usage.json','v12_seeds_metric_verification.json',
        'v12_seeds_audit.json','v12_seeds_request_evidence.json','v12_seeds_preflight.json','v12_seeds_results.txt',
        'v12_seeds_trajectories.png','v12_seeds_trajectories.pdf','v12_seeds_paired_differences.png','v12_seeds_paired_differences.pdf','v12_seeds_run.log']
    receipt=dict(request_status='COMPLETE_18_PAIRED_INITIALIZATION_TRAJECTORIES',complete_new_runs=18,new_queries=288,new_V12_runs=9,new_BO_runs=9,
        initialization_seeds=[1,2,3],initial=4,queries=16,native_numerical_tie_menu_seed=0,LLM_runs_per_initialization=1,
        protocol_sha256=p['protocol_sha256'],frozen_V12_protocol_sha256=p['frozen_V12_protocol_sha256'],code_milestone=p['code_milestone'],
        delivery_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        audit_status=audit['status'],protected_files=audit['protected_control_files'],task_summaries=result['task_summaries'],API_usage=result['API_usage'],
        adapter_tests_passed=4,BO_audit_tests_passed=3,seed0_in_sample=False,global_stability_confirmed=False,
        figure_review='Both PNGs visually inspected for all nine pair panels, task units, common within-task trajectory scales, readable labels, zero-centered signed differences and no clipping.',
        files_sha256={name:study.old.sha(out/name) for name in names},document_sha256=study.old.sha(doc),created_utc=datetime.now(timezone.utc).isoformat())
    study.write(out/'v12_seeds_completion_receipt.json',receipt)
    status=study.old.read(out/'STATUS.json');already=status.get('v12_seed123',{}).get('request_status')==receipt['request_status']
    if not already:status['new_optimization_runs']+=18;status['new_experimental_queries']+=288
    status.update(current_phase='V12_SEEDS_1_2_3_PAIRED_COMPLETE',variant=study.VARIANT,current_sessions={},active_optimization_processes=0,
        protocol_sha256=p['protocol_sha256'],code_milestone=p['code_milestone'],delivery_commit=receipt['delivery_commit'],
        updated_utc=receipt['created_utc'],v12_seed123=receipt,stable_majority_advantage_supported=False)
    status['next_steps']=['Requested nine paired initialization comparisons complete; retain every trajectory and historical control.',
        'No additional repeats, tuning, control reruns, merge or push are part of this request.']
    study.write(out/'STATUS.json',status);return receipt


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--finalize',action='store_true');args=ap.parse_args()
    result=finalize() if args.finalize else build();print(json.dumps(result,ensure_ascii=False,indent=2))
