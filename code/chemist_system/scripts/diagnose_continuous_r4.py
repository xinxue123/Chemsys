"""Read-only R4 mechanisms and trajectory attribution on archived prefixes.

No experiment runs, outcome-provider access, LLM calls or GP fitting. Geometric
filter counts and acquisition recalculation use saved inputs/marginals only.
Comparator outcomes are used solely as labelled post-hoc observations.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from pathlib import Path
import csv
import json
import os
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));os.chdir(ROOT)
from chem.benchmarks import continuous_r4_engine as e
from chem.benchmarks import continuous_benchmark_engine as base
from chem.benchmarks.continuous_r4_report import independent_local
from chem.benchmarks.suzuki_domain import digest

RUN=ROOT/'data/runs/continuous_r4_20261030'
STUDY=ROOT/'research/continuous_r4_20261030'
OUT=ROOT/'research/continuous_r4_diagnosis_20261030'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return sha256(Path(p).read_bytes()).hexdigest()
def clean(v):
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [clean(x) for x in v]
    return v
def write(p,v):Path(p).write_text(json.dumps(clean(v),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def refuse(*a,**kw):raise RuntimeError('Read-only R4 diagnosis forbids fitting, GP caches and outcome-provider calls.')


def csv_write(p,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for row in rows:w.writerow({k:json.dumps(clean(v),ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else clean(v) for k,v in row.items()})


def state_for(task,method):
    root=RUN if method=='New_Agent_R4' else ROOT/'data/runs/continuous_r3_20261030' if method=='New_Agent_R3' else ROOT/'data/runs/continuous_r2_20261030' if method=='New_Agent_R2' else ROOT/'data/runs/continuous_20261029'
    return read(root/'runs'/(task+'_'+method)/'state.json'),root


def rank(a,j,mask):
    ids=np.flatnonzero(mask)
    order=ids[np.lexsort((a['tie_priority'][ids],-a['sd_y'][ids],-a['information_proxy'][ids],-a['base_qLogEI'][ids]))]
    found=np.flatnonzero(order==j);return int(found[0]+1) if len(found) else None


def task_analysis(task):
    base.warm();base.fit_gpytorch_mll=refuse;base.data.reveal=refuse;base.Cache.__init__=refuse
    state,_=state_for(task,'New_Agent_R4');r2,_=state_for(task,'New_Agent_R2');r3,r3root=state_for(task,'New_Agent_R3')
    ll,lr=state_for(task,'LLINBO_Justify');pool=read(ROOT/f'data/benchmarks/continuous_20261029/{task}/pool.json')
    ids=pool['ids'];ix={key:j for j,key in enumerate(ids)};z=np.asarray(pool['z']);scale=pool['objective_scale'];prefix=e.BOOTSTRAP[task]
    known={};comparators=[]
    for method in ['New_Agent_R4','New_Agent_R3','New_Agent_R2','LLINBO_Justify','ReasoningBO','Pure_BO']:
        s,_=state_for(task,method);assert s['history'][:4]==state['history'][:4]
        for i,h in enumerate(s['history'],1):
            if h['candidate_id'] in known:assert known[h['candidate_id']]['value']==h['raw_value']
            else:known[h['candidate_id']]={'value':h['raw_value'],'method':method,'total_experiment':i}
        for d in s['decisions']:
            llm=d.get('LLM') or {};comparators.append({'task':task,'method':method,'query':d['iteration'],
                'selected':d['selected_id'],'reference':d['reference_id'],'objective':d['raw_value'],
                'best':d['incumbent_after']*scale,'LLM_accepted':llm.get('accepted'),
                'LLM_proposal':llm.get('proposed_id'),'differs_from_reference':d['selected_id']!=d['reference_id']})
    peak=max(state['history'],key=lambda h:h['raw_value']);first_peak=next(i for i,h in enumerate(state['history'],1) if h['raw_value']==peak['raw_value'])-4
    target='HPLC_00457' if task=='HPLC' else peak['candidate_id'];jt=ix[target]
    rounds=[];funnels=[];projections=[];shadows=[];targets=[];inheritance=[];lifecycle=[];firstprobes=[];maxerr=0.
    sparse_checks=potential_checks=0
    for t,d in enumerate(state['decisions'],1):
        folder=RUN/'runs'/state['run_id']/'rounds';cp=folder/f'{t:02d}_committed.json';c=read(cp)
        assert sha(cp)==d['commit_sha256'] and c['history']==state['history'][:t+3]
        assert sha(folder/f'{t:02d}_candidates.npz')==c['candidate_table_sha256']
        with np.load(folder/f'{t:02d}_candidates.npz') as f:a={k:f[k].copy() for k in f.files}
        targets.append({'task':task,'query':t,'target_posthoc_observed_id':target,
            'target_value_posthoc_from_existing_observation':known[target]['value'],
            'target_GP_mean':a['mu_y'][jt]*scale,'target_GP_predictive_SD':a['sd_predictive_y'][jt]*scale,
            'target_available':bool(a['available'][jt]),'target_inside_budget':bool(a['eligible'][jt]),
            'target_acquisition_rank':rank(a,jt,a['available']),
            'target_base_loss':float(a['base_qLogEI'][ix[d['reference_id']]]-a['base_qLogEI'][jt]),'budget':d['budget']})
        if t<=prefix:continue
        assert np.array_equal(a['guided_qLogEI'],a['base_qLogEI']) and not np.any(a['local_shift_y'])
        assert np.array_equal(a['eligible'],a['available']&(a['base_qLogEI'][ix[d['reference_id']]]-a['base_qLogEI']<=d['budget']))
        recalculated=base.acquisition(a['mu_std'],a['sd_std'],np.zeros(len(ids)),c['best_std'],c['base_meta']['mc_seed'])
        error=float(np.max(abs(recalculated-a['base_qLogEI'])));maxerr=max(maxerr,error);assert error<1e-9
        ref=ix[d['reference_id']];R=a['sd_predictive_y']**2+.0009;nonref=a['eligible'].copy();nonref[ref]=False
        eligible_count=int(a['eligible'].sum());localrows=[]
        for k,key in enumerate(c['local_hypothesis_ids']):
            h=c['local_models_at_commit'][key];w,f,H=independent_local(pool,h,a)
            assert np.max(abs(H-a['local_features_y'][k]))<1e-12 and np.max(abs(w-a['local_weights'][k]))<1e-12
            positive=e.evidence(h,t);probe=.25*H;signal=abs(probe)/np.sqrt(R);contrast=abs(probe-probe[ref])/np.sqrt(R+R[ref])
            raw_w=e.old.scope_weight(pool,h);raw_H=np.clip(raw_w*(f-a['mu_y']),-a['sd_predictive_y'],a['sd_predictive_y'])
            raw_probe=.25*raw_H;raw_signal=abs(raw_probe)/np.sqrt(R);raw_contrast=abs(raw_probe-raw_probe[ref])/np.sqrt(R+R[ref])
            premise=bool(h['created_query']<t and not h['paused_by_LLM'] and not positive['eligible'] and positive['positive_count']>=1 and positive['score']>0)
            physical=nonref&(w>1e-12);informative=nonref&(signal>=.05);distinguished=informative&(contrast>=.05)
            all_probes=a['available']&(signal>=.05)&(contrast>=.05);all_probes[ref]=False
            minloss=float(np.min(a['base_qLogEI'][ref]-a['base_qLogEI'][all_probes])) if all_probes.any() else None
            row={'task':task,'query':t,'hypothesis_created_query':h['created_query'],'hypothesis_id':key,
                'paused':h['paused_by_LLM'],'positive_count':positive['positive_count'],'weighted_score':positive['score'],
                'passes_prior_positive_gate':premise,'eligible_count':eligible_count,
                'unqueried_scope_points':int(np.sum(a['available']&(w>1e-12))),
                'budget_scope_points':int(np.sum(a['eligible']&(w>1e-12))),
                'nonreference_budget_scope_points':int(physical.sum()),
                'nonreference_informative_points':int(informative.sum()),'nonreference_distinguishable_points':int(distinguished.sum()),
                'radial_only_distinguishable_points':int(np.sum(nonref&(raw_signal>=.05)&(raw_contrast>=.05))),
                'any_eligible_probe_max_signal_ratio':float(signal[a['eligible']].max()),
                'nonreference_max_contrast':float(contrast[nonref].max()) if nonref.any() else 0.,
                'minimum_distinguishable_probe_loss_without_gate':minloss,'current_budget':d['budget'],
                'eligible_probe_ids':','.join(ids[j] for j in np.flatnonzero(distinguished)),
                'selected_inside_scope':bool(w[ix[d['selected_id']]]>1e-12),
                'scope_at_posthoc_comparator_best':float(w[jt])}
            if h['created_query']>=t:reason='new_this_round'
            elif h['paused_by_LLM']:reason='paused'
            elif not premise:reason='no_prior_positive_evidence'
            elif t>14:reason='after_validation_window'
            elif eligible_count<2:reason='singleton_budget'
            elif not physical.any():reason='no_nonreference_scope_in_budget'
            elif not informative.any():reason='signal_below_threshold'
            elif not distinguished.any():reason='contrast_below_threshold'
            else:reason='feasible'
            row['first_failed_filter']=reason;funnels.append(row);localrows.append(row)
            if h['created_query']<t and not h['paused_by_LLM'] and not premise:
                for j in np.flatnonzero(distinguished):
                    observed=known.get(ids[j]);value=observed['value']/scale if observed else None
                    gain=(value-a['mu_y'][j])**2-(value-a['mu_y'][j]-probe[j])**2 if value is not None else None
                    firstprobes.append({'task':task,'query':t,'created_query':h['created_query'],'hypothesis_id':key,
                        'candidate_id':ids[j],'GP_mean':float(a['mu_y'][j])*scale,'probe_mean':float(a['mu_y'][j]+probe[j])*scale,
                        'base_loss':float(a['base_qLogEI'][ref]-a['base_qLogEI'][j]),'budget':d['budget'],
                        'known_value_posthoc_only':observed['value'] if observed else None,
                        'known_value_source':observed['method'] if observed else None,
                        'frozen_pointwise_probe_gain_posthoc_only':gain*scale**2 if gain is not None else None})
            potential_checks+=1;sparse_checks+=bool(w[ix[d['selected_id']]]>1e-12)
        available_probes=sum(r['nonreference_distinguishable_points'] for r in localrows if r['passes_prior_positive_gate'])
        assert available_probes==len(d['validation']['candidates'])
        knownref=known.get(d['reference_id'])
        rounds.append({'task':task,'query':t,'mode':d['execution_mode'],'selected_id':d['selected_id'],'reference_id':d['reference_id'],
            'objective':d['raw_value'],'incumbent':d['incumbent_after']*scale,'eligible_count':eligible_count,
            'fresh_call':c['answer'] is not None,'call_reason':d['call_permission']['reason'],
            'positive_pending_models':sum(r['passes_prior_positive_gate'] for r in localrows),
            'feasible_validation_candidate_model_pairs':available_probes,
            'feasible_first_probe_pairs_if_prior_positive_rule_removed_analysis_only':sum(r['nonreference_distinguishable_points'] for r in localrows if not r['paused'] and r['hypothesis_created_query']<t),
            'immediate_reference_value_posthoc':knownref['value'] if knownref else None,
            'immediate_difference_posthoc':d['raw_value']-knownref['value'] if knownref else None,
            'before_first_final_best':t<first_peak})
        if c['answer'] and c['projection']:
            plan=c['answer']['plan'];projected=c['projection'];h=c['local_models_at_commit'][d['registered_hypothesis']]
            r=c['history'][plan['hypothesis']['reference_index']-1];reference=pool['points'][ix[r['candidate_id']]]
            p=plan['point'];q=projected['point'];focus=plan['hypothesis']['focus_axes'];axis_changes=[]
            for axis in focus:
                length=pool['bounds'][axis][1]-pool['bounds'][axis][0];requested=p[axis]-reference[axis];actual=q[axis]-reference[axis]
                axis_changes.append({'axis':axis,'reference':reference[axis],'requested':p[axis],'projected':q[axis],
                    'requested_delta':requested,'projected_delta':actual,
                    'direction_reversed':requested*actual< -1e-12,'change_ratio':actual/requested if abs(requested)>1e-12 else None})
            projections.append({'task':task,'query':t,'hypothesis_id':d['registered_hypothesis'],
                'proposed_id':projected['candidate_id'],'GP_accepted':d['GP_review_accepted'],
                'proposal_GP_loss':d['GP_review_loss'],'budget':d['budget'],
                'projection_distance':projected['normalized_Euclidean_distance'],
                'focus_axes':focus,'reversed_focus_axes':[v['axis'] for v in axis_changes if v['direction_reversed']],
                'focus_direction_preserved':not any(v['direction_reversed'] for v in axis_changes),
                'axis_changes':axis_changes,'prediction_raw':plan['prediction'],'reference_objective':r['raw_value'],
                'reference_candidate_id':r['candidate_id'],'basis_kind':h['basis']['kind'],
                'declared_statement':plan['hypothesis']['statement'],'reason':plan['reason']})
        for v in d['local_feedback']['shadow_checks']:
            h=c['local_models_at_commit'][v['hypothesis_id']]
            shadows.append({'task':task,'query':t,'created_query':h['created_query'],'hypothesis_id':v['hypothesis_id'],
                'selected_id':d['selected_id'],'GP_mean_raw':v['GP_mean_y']*scale,'probe_mean_raw':v['probe_prediction_y']*scale,
                'observed_raw':d['raw_value'],'signal_ratio':v['probe_over_effective_SD'],'kernel_weight':v['kernel_weight'],
                'informative':v['informative'],'positive':v['paired_SE_gain']>1e-12,
                'weighted_score':v['weighted_standardized_gain'],'after_pause':h['paused_by_LLM']})
    for source in state['evidence_ledger']:
        ws=e.scope_weight(pool,source['source_model'])>1e-12;j=ix[source['candidate_id']]
        for key,h in state['hypotheses'].items():
            if key==source['source_hypothesis_id']:continue
            w=e.scope_weight(pool,h);wt=w>1e-12;den=min(int(ws.sum()),int(wt.sum()));overlap=float(np.sum(ws&wt)/den) if den else 0.
            f=e.local_forecast(pool,h);sd=source['GP_predictive_SD_y'];H=float(np.clip(w[j]*(f[j]-source['GP_mean_y']),-sd,sd));delta=.25*H
            residual=source['observed_y']-source['GP_mean_y'];gain=float(2*residual*delta-delta**2)
            own=any(v['query']==source['query'] and v['candidate_id']==source['candidate_id'] for v in h['shadow_records'])
            if overlap<.5:block='overlap_below_0.5'
            elif not wt[j]:block='failed_point_outside_target_support'
            elif gain>=-1e-12:block='target_probe_not_negative'
            elif own:block='own_prospective_check_precedes_inheritance'
            else:block='would_inherit'
            inheritance.append({'task':task,'source_query':source['query'],'source_created_query':state['hypotheses'][source['source_hypothesis_id']]['created_query'],
                'source_kind':source['kind'],'source_id':source['source_hypothesis_id'],'target_created_query':h['created_query'],
                'target_id':key,'support_overlap':overlap,'failed_point_in_target':bool(wt[j]),'target_probe_gain':gain,
                'own_check_already_present':own,'first_blocker':block,
                'direct_negative_point_excluded_only_by_global_overlap':overlap<.5 and bool(wt[j]) and gain< -1e-12 and not own})
    for key,h in state['hypotheses'].items():
        checks=[v for v in shadows if v['hypothesis_id']==key];strong=[v for v in checks if v['informative']]
        repeated=[]
        for a,b in zip(h['pause_records'],h['pause_records'][1:]):
            new=[v for v in h['shadow_records'] if a['query']<v['query']<b['query']]
            if not new:repeated.append({'first_pause':a['query'],'repeated_pause':b['query'],'new_scope_observations_between':0})
        lifecycle.append({'task':task,'hypothesis_id':key,'created_query':h['created_query'],'status':h['status'],
            'observed_checks':len(checks),'strong_checks':len(strong),'positive_distinct_experiments':len({v['query'] for v in strong if v['positive']}),
            'positive_queries':[v['query'] for v in strong if v['positive']],'pause_queries':[v['query'] for v in h['pause_records']],
            'repeated_pause_without_new_observation':repeated,'inherited_contexts':len(h['inherited_evidence'])})
    yr=np.maximum.accumulate([h['raw_value'] for h in state['history']])[4:];yl=np.maximum.accumulate([h['raw_value'] for h in ll['history']])[4:]
    aucrows=[{'task':task,'query':i+1,'R4_best':a,'LLINBO_best':b,'R4_minus_LLINBO':a-b,'AUC_contribution':(a-b)/16} for i,(a,b) in enumerate(zip(yr,yl))]
    llrows=[]
    for d in ll['decisions']:
        llm=d['LLM'];j=ix[d['selected_id']];ref=known.get(d['reference_id'])
        llrows.append({'query':d['iteration'],'selected':d['selected_id'],'reference':d['reference_id'],'objective':d['raw_value'],
            'cumulative_best':d['incumbent_after']*scale,'LLM_accepted':llm['accepted'],'proposal':llm['proposed_id'],
            'nonreference_choice':d['selected_id']!=d['reference_id'],'reference_known_value_posthoc':ref['value'] if ref else None,
            'physical_point':pool['points'][j],'proposal_reason':llm['raw']['reason']})
    prefix_error=None
    if task=='SnAr':
        with np.load(RUN/'runs'/state['run_id']/'rounds/05_candidates.npz') as f:current={k:f[k].copy() for k in f.files}
        with np.load(r3root/'runs'/r3['run_id']/'rounds/05_candidates.npz') as f:
            prefix_error=max(float(np.max(abs(current[k]-f[k]))) for k in ['mu_y','sd_y','sd_predictive_y','base_qLogEI'])
        assert prefix_error==0.
    summary={'task':task,'fresh_decisions':16-prefix,'source_local_checks':sparse_checks,'possible_model_round_pairs':potential_checks,
        'out_of_scope_model_round_pairs':potential_checks-sparse_checks,'projection_focus_reversals':[p['query'] for p in projections if p['reversed_focus_axes']],
        'pending_without_feasible_probe_queries':[r['query'] for r in rounds if r['positive_pending_models'] and not r['feasible_validation_candidate_model_pairs']],
        'first_probe_feasible_but_no_positive_queries':sorted({r['query'] for r in funnels if r['hypothesis_created_query']<r['query'] and not r['paused'] and not r['passes_prior_positive_gate'] and r['nonreference_distinguishable_points']>0}),
        'inheritance_blockers':dict(Counter(r['first_blocker'] for r in inheritance)),
        'actual_failure_ledger_events':sum(v['kind']=='actual_failure' for v in state['evidence_ledger']),
        'shadow_failure_ledger_events':sum(v['kind']=='shadow_failure' for v in state['evidence_ledger']),
        'R4_SearchAUC':float(yr.mean()),'LLINBO_SearchAUC':float(yl.mean()),
        'early_R4_AUC_advantage_contribution':float(np.maximum(yr-yl,0).sum()/16),
        'later_R4_AUC_deficit_contribution':float(np.minimum(yr-yl,0).sum()/16),
        'first_final_best_query':first_peak,'nonreference_choices_before_first_final_best':[r['query'] for r in rounds if r['selected_id']!=r['reference_id'] and r['query']<first_peak],
        'same_prefix_GP_error_vs_R3_at_query5':prefix_error,'max_saved_acquisition_error':maxerr,
        'LLINBO_accepted_queries':[r['query'] for r in llrows if r['LLM_accepted']],
        'LLINBO_nonreference_queries':[r['query'] for r in llrows if r['nonreference_choice']],
        'model_lifecycles':lifecycle,'GP_fits':0,'new_LLM_calls':0,'new_experiment_calls':0,'operator_compiles_in_loop':sum(base.JIT.values())}
    for name,rows in [('rounds',rounds),('validation_funnel',funnels),('projection_semantics',projections),('shadow_checks',shadows),
        ('inheritance_funnel',inheritance),('model_lifecycle',lifecycle),('known_best_GP_tracking',targets),('LLINBO_trace',llrows),
        ('AUC_difference',aucrows),('comparator_rounds',comparators),('first_probe_opportunities',firstprobes)]:csv_write(OUT/(task+'_'+name+'.csv'),rows)
    return summary


def main():
    OUT.mkdir(parents=True,exist_ok=True);keep=read(STUDY/'protected_snapshot.json');keep.update(read(STUDY/'release_inventory.json'))
    for name in ['release_inventory.json','release_receipt.json']:
        path=STUDY/name;keep[path.relative_to(ROOT).as_posix()]=sha(path)
    for name,value in keep.items():assert sha(ROOT/name)==value,name
    with ProcessPoolExecutor(max_workers=2) as executor:results=list(executor.map(task_analysis,['SnAr','HPLC']))
    for name,value in keep.items():assert sha(ROOT/name)==value,name
    write(OUT/'source_snapshot.json',keep)
    report={'status':'PASS','scope':'archived-prefix mechanism diagnosis only; no rerun, fitting, LLM or outcome-provider calls',
        'source_script_sha256':sha(__file__),'existing_files_unchanged':len(keep),'checks':results,
        'counterfactual_limit':'Filter counts and known point comparisons do not estimate alternate-trajectory SearchAUC.'}
    write(OUT/'diagnosis.json',report);print(json.dumps(clean(report),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
