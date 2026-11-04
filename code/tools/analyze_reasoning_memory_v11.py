"""Saved-log-only V11 diagnostics. No LLM, GP refit or oracle read permitted."""
from collections import Counter
from pathlib import Path
from unittest.mock import patch
import json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/chemist_system'));sys.path.insert(0,str(ROOT/'code/experiment_tools'))
from reasoning_memory_v11 import study
from knowledge_information_v10.report import table
from audit_reasoning_memory_v11 import audit_task,audit_amendment,scope

TASKS=('Reizman_Suzuki_Case_2','Reizman_Suzuki_Case_4','HPLC')
OUT=ROOT/'experiments/reports/knowledge_loop_20261101/v11_diagnostic_20261101'
SOURCES={}


def read(path):
    path=Path(path);SOURCES[path.relative_to(ROOT).as_posix()]=study.old.sha(path)
    return study.old.read(path)


def forbidden(*args,**kwargs):raise RuntimeError('Analysis-only: API/oracle/model execution forbidden')


def build():
    OUT.mkdir(parents=True,exist_ok=True);p=study.check(ROOT)
    rows=[];menus=[];notes=[];events=[];all_metrics=[];phase=[];traces={};details={};divergences=[];visibility=[]
    for task in TASKS:
        pool=study.gpmod.load_task(ROOT,task,seed=0);SOURCES[pool.source_path.relative_to(ROOT).as_posix()]=study.old.sha(pool.source_path)
        states={};commits={};paths={}
        for label,base,arm in [('BO',study.v9.directory(ROOT),'bo'),('RBO',study.v9.directory(ROOT),'reasoning_bo'),('V11',study.directory(ROOT),'rbo_memory')]:
            paths[label]=base/'runs'/f'{task}_{arm}';states[label]=read(paths[label]/'state.json')
            commits[label]=[read(paths[label]/'rounds'/f'{t:02d}_commit.json') for t in range(1,17)]
            s=states[label];assert s['status']=='COMPLETE' and len(s['history'])==20
            values=np.array([r['raw_value'] for r in s['history']]);best=np.maximum.accumulate(values)[4:]
            traces[task,label]=best
            all_metrics.append(dict(task=task,method=label,SearchAUC=float(best.mean()),final_best=float(best[-1]),
                first_best_query=int(np.argmax(values==best[-1]))-3))
        assert states['V11']['history'][:4]==states['RBO']['history'][:4]==states['BO']['history'][:4]
        first=next(t for t in range(1,17) if commits['V11'][t-1]['selected_id']!=commits['RBO'][t-1]['selected_id'])
        vc,rc=commits['V11'][first-1],commits['RBO'][first-1]
        first_arrays=[]
        for label in ('V11','RBO'):
            path=paths[label]/'rounds'/f'{first:02d}_GP.npz';SOURCES[path.relative_to(ROOT).as_posix()]=study.old.sha(path)
            with np.load(path,allow_pickle=False) as saved:first_arrays.append({k:saved[k].copy() for k in saved.files})
        for key in ('mu_y','sd_y','sd_predictive_y','base_logei','available','tie_priority'):
            np.testing.assert_array_equal(first_arrays[0][key],first_arrays[1][key])
        divergences.append(dict(task=task,query=first,same_observed_prefix=vc['prestate']['history']==rc['prestate']['history'],
            same_GP_sha256=vc['GP_sha256']==rc['GP_sha256'],saved_GP_arrays_bitwise_equal=True,same_display_menu=vc['display_ids']==rc['display_ids'],
            V11_selected=vc['selected_id'],RBO_selected=rc['selected_id'],V11_references=vc['memory_references']))
        targets=['p03631','p01031'] if task.endswith('Case_2') else ['p00514','p00874','p00946'] if task.endswith('Case_4') else ['HPLC_00850','HPLC_00806','HPLC_00457']
        for label,cs in commits.items():
            for target in targets:
                visible=[dict(query=t,rank=c['ordered_ids'].index(target)+1,selected=c['selected_id']==target)
                    for t,c in enumerate(cs,1) if target in c['ordered_ids']]
                visibility.append(dict(task=task,method=label,candidate_id=target,menu_appearances=visible))
        known={};sources={}
        for label,s in states.items():
            for r in s['history']:
                key=r['candidate_id']
                if key in known:assert abs(known[key]-r['raw_value'])<1e-10
                known[key]=r['raw_value'];sources.setdefault(key,[]).append(label)
        details[task]=dict(initial=states['V11']['history'][:4],knowledge_versions=states['V11']['hypotheses'],rounds=[])
        for t,c in enumerate(commits['V11'],1):
            s=states['V11'];d=s['decisions'][t-1];key=c['selected_id'];j=pool.ix[key]
            gpfile=paths['V11']/'rounds'/f'{t:02d}_GP.npz';SOURCES[gpfile.relative_to(ROOT).as_posix()]=study.old.sha(gpfile)
            with np.load(gpfile,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
            ctx=read(study.directory(ROOT)/'requests'/c['LLM_context_sha256']/'context.json')['public']
            win=c['LLM_raw']['hypotheses'][c['winning_hypothesis_index']]
            pre=c['prestate'];best_before=max(r['raw_value'] for r in pre['history'])
            row=dict(task=task,query=t,V11_selected=key,V11_y=d['raw_value'],V11_best=traces[task,'V11'][t-1],
                BO_selected=states['BO']['history'][t+3]['candidate_id'],BO_y=states['BO']['history'][t+3]['raw_value'],BO_best=traces[task,'BO'][t-1],
                RBO_selected=states['RBO']['history'][t+3]['candidate_id'],RBO_y=states['RBO']['history'][t+3]['raw_value'],RBO_best=traces[task,'RBO'][t-1],
                GP_rank=c['ordered_ids'].index(key)+1,GP_mean=a['mu_y'][j]*pool.scale,GP_sd=a['sd_y'][j]*pool.scale,
                GP_predictive_sd=a['sd_predictive_y'][j]*pool.scale,qLogEI=a['base_logei'][j],opportunity_gap=c['opportunity_gap'],
                best_before=best_before,refresh=c['refresh']['allowed'],refresh_reasons=';'.join(c['refresh']['reasons']),
                confidence=win['confidence'],strategy=win['strategy'],rationale=win['rationale'],
                conditions=json.dumps(pool.points[j],ensure_ascii=False),cited_notes=json.dumps(c['memory_references']),
                scored_notes=len(c['predictions']),KF_events=';'.join(e['reason'] for e in d['KF_feedback']))
            row.update(delta_best_vs_BO=row['V11_best']-row['BO_best'],delta_best_vs_RBO=row['V11_best']-row['RBO_best'])
            rows.append(row)
            for rank,cid in enumerate(c['ordered_ids'],1):
                k=pool.ix[cid]
                menus.append(dict(task=task,query=t,candidate_id=cid,rank=rank,selected=cid==key,
                    GP_mean=a['mu_y'][k]*pool.scale,GP_sd=a['sd_y'][k]*pool.scale,GP_predictive_sd=a['sd_predictive_y'][k]*pool.scale,
                    qLogEI=a['base_logei'][k],gap=a['base_logei'][pool.ix[c['ordered_ids'][0]]]-a['base_logei'][k],
                    already_observed_in_comparison_logs=cid in known,logged_y=known.get(cid),logged_y_sources=';'.join(sources.get(cid,[])),
                    conditions=json.dumps(pool.points[k],ensure_ascii=False)))
            for h in c['hypotheses_at_commit']:
                cited=next((r['use'] for r in c['memory_references'] if (r['id'],r['version'])==(h['id'],h['version'])),None)
                inside=scope(pool,h,j);retrieved=any((h['id'],h['version'])==(n['id'],n['version']) for g in ('supporting_notes','cautionary_notes') for n in ctx['retrieved_notes'][g])
                forecast=c['forecast_probabilities'].get(h['id'],{}).get(key) if h['kf']['status']!='retracted' else None
                notes.append(dict(task=task,query=t,id=h['id'],version=h['version'],claim=h['claim'],status=h['kf']['status'],
                    weight=h['kf']['weight'],prior_only=h['kf']['local_updates']==0,local_updates=h['kf']['local_updates'],
                    inherited=len(h['inherited_negative']),retrieved=retrieved,selected_in_scope=inside,citation=cited,
                    fixed_reference=h['reference_id'],fixed_reference_y=h['reference_y']*pool.scale,current_incumbent=best_before,
                    p_LLM=forecast,present_in_forecast_notes=h['kf']['status']!='retracted'))
            for e in d['KF_feedback']:
                events.append(dict(task=task,query=t,id=e['id'],version=e['version'],p=e['p_LLM'],g=e['p_GP'],success=e['success'],
                    gain=e['Brier_gain'],kalman_gain=e['kalman_gain'],weight_before=e['before']['weight'],weight_after=e['after']['weight'],
                    status_before=e['before']['status'],status_after=e['after']['status'],reason=e['reason']))
            details[task]['rounds'].append(dict(query=t,winning=win,notes_edit=c['memory_raw'],reviews=c['reviews'],
                original_RBO=commits['RBO'][t-1]['LLM_raw'],menu_order=c['display_ids'],
                complete_menu=ctx['candidate_menu'],retrieved_notes=ctx['retrieved_notes'],forecast_notes=ctx['forecast_notes']))
        for control in ('BO','RBO'):
            delta=traces[task,'V11']-traces[task,control]
            for lo,hi in [(1,4),(5,8),(9,12),(13,16)]:
                phase.append(dict(task=task,control=control,query_from=lo,query_to=hi,AUC_contribution=float(delta[lo-1:hi].sum()/16)))
            assert abs(sum(r['AUC_contribution'] for r in phase if r['task']==task and r['control']==control)-float(delta.mean()))<1e-10
    table(OUT/'rounds.csv',rows);table(OUT/'menus.csv',menus);table(OUT/'notes.csv',notes);table(OUT/'KF_events.csv',events)
    table(OUT/'metrics.csv',all_metrics);table(OUT/'AUC_phase_decomposition.csv',phase);study.write(OUT/'details.json',details)
    study.write(OUT/'matched_prefix_divergences.json',divergences);study.write(OUT/'candidate_visibility.json',visibility)
    facts={}
    for task in TASKS:
        rr=[r for r in rows if r['task']==task];nn=[n for n in notes if n['task']==task and n['citation']]
        mm=[m for m in menus if m['task']==task];ee=[e for e in events if e['task']==task]
        homogeneous=[]
        if task!='HPLC':
            for t in range(1,17):
                cats={json.loads(m['conditions'])['catalyst'] for m in mm if m['query']==t}
                if len(cats)==1:homogeneous.append(dict(query=t,catalyst=next(iter(cats))))
        facts[task]=dict(citations=len(nn),outside_selected_scope=sum(not n['selected_in_scope'] for n in nn),
            paused_support_citations=[dict(query=n['query'],id=n['id']) for n in nn if n['status']=='paused' and n['citation']=='support'],
            homogeneous_menus=homogeneous,KF_event_counts=dict(Counter(e['reason'] for e in ee)),
            gain_despite_no_incumbent_improvement=[dict(query=e['query'],id=e['id'],gain=e['gain']) for e in ee
                if e['gain']>0 and next(r for r in rr if r['query']==e['query'])['V11_y']<=next(r for r in rr if r['query']==e['query'])['best_before']],
            phase_1_8_vs_RBO=sum(r['delta_best_vs_RBO'] for r in rr if r['query']<=8)/16,
            phase_9_14_vs_RBO=sum(r['delta_best_vs_RBO'] for r in rr if 9<=r['query']<=14)/16)
    study.write(OUT/'mechanism_facts.json',facts)
    audits=[audit_task(ROOT,task) for task in TASKS]
    amendment=audit_amendment(ROOT)
    study.write(OUT/'verification.json',dict(status='PASS_SAVED_LOG_CHECKS',audits=audits,amendment=amendment,
        new_API_calls=0,new_GP_fits=0,new_experiment_queries=0,oracle_files_read=0,
        comparison_outcomes='Only previously observed values in the three existing arms; no unknown candidate outcomes read.',
        source_sha256=SOURCES,analysis_source_sha256=study.old.sha(__file__)))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,3.8),constrained_layout=True)
    styles={'BO':('#444444','-'),'RBO':('#3676b5','--'),'V11':('#188662','-')}
    for ax,task in zip(axes,TASKS):
        for label,(color,style) in styles.items():ax.step(range(1,17),traces[task,label],where='post',label=label,color=color,linestyle=style)
        ax.set_title(task.replace('Reizman_Suzuki_','').replace('_',' '));ax.set_xlabel('Query after shared four initial points')
        ax.set_ylabel('Best yield (%)' if task!='HPLC' else 'Best peak area');ax.set_xticks([1,4,8,12,16]);ax.grid(alpha=.2)
    axes[0].legend();fig.suptitle('V11 diagnostic: saved seed-0 trajectories; task-specific y scales')
    fig.savefig(OUT/'three_task_incumbents.png',dpi=180);fig.savefig(OUT/'three_task_incumbents.pdf');plt.close(fig)
    print(json.dumps(dict(metrics=all_metrics,phase=phase,checked_rounds=sum(x['rounds'] for x in audits),new_queries=0),ensure_ascii=False,indent=2))


if __name__=='__main__':
    with patch.object(study.gpmod,'read_observation',side_effect=forbidden),patch.object(study.gpmod,'GPAdapter',side_effect=forbidden),patch.object(study.api,'APIClient',side_effect=forbidden):build()
