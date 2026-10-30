"""Read-only R3 diagnosis: archived marginals, proposals and observed feedback.

No trajectory execution, LLM requests, GP fitting, checkpoint writes or outcome
provider calls. Recalculating acquisition values on a frozen prefix is an audit,
not a new optimization experiment. Known comparator observations are labelled
post hoc and are never supplied to a decision policy.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch
import csv
import json
import os
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));os.chdir(ROOT)
from chem.benchmarks import continuous_benchmark_engine as base
from chem.benchmarks import continuous_r3_engine as r3
from chem.benchmarks.suzuki_domain import digest

RUN=ROOT/'data/runs/continuous_r3_20261030'
STUDY=ROOT/'research/continuous_r3_20261030'
OUT=ROOT/'research/continuous_r3_diagnosis_20261030'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def filehash(p):return sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def refuse(*a,**kw):raise RuntimeError('R3 diagnosis forbids fitting, outcome calls and new GP caches.')


def csv_write(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def pick(a,score,mask):
    js=np.flatnonzero(mask);assert len(js)
    js=js[score[js]==score[js].max()];js=js[a['information_proxy'][js]==a['information_proxy'][js].max()]
    js=js[a['sd_y'][js]==a['sd_y'][js].max()];return int(js[np.argmin(a['tie_priority'][js])])


def observed_registry(task):
    records={}
    for root,methods in [(RUN,['New_Agent_R3']),
        (ROOT/'data/runs/continuous_r2_20261030',['New_Agent_R2']),
        (ROOT/'data/runs/continuous_20261029',['Pure_BO','LLINBO_Justify','ReasoningBO'])]:
        for method in methods:
            state=read(root/'runs'/(task+'_'+method)/'state.json')
            for i,h in enumerate(state['history'],1):
                key=h['candidate_id'];value=float(h['raw_value'])
                if key in records:assert records[key]['raw']==value
                else:records[key]={'raw':value,'first_source':method,'total_experiment':i}
    return records


def task_analysis(task):
    base.warm();base.fit_gpytorch_mll=refuse;base.data.reveal=refuse;base.Cache.__init__=refuse
    state=read(RUN/'runs'/(task+'_New_Agent_R3')/'state.json')
    parent=read(ROOT/'data/runs/continuous_r2_20261030/runs'/(task+'_New_Agent_R2')/'state.json')
    pool=read(ROOT/f'data/benchmarks/continuous_20261029/{task}/pool.json')
    ids=pool['ids'];ix={key:j for j,key in enumerate(ids)};scale=pool['objective_scale'];prefix=r3.BOOTSTRAP[task]
    known=observed_registry(task);rounds=[];proposals=[];shadows=[];deployments=[];modelrows=[];retired=[];maxerr=0.
    positive_keys=set();informative_keys=set();Rfloor_raw=scale*.03;inheritance=[]
    for t,event in enumerate(state['decisions'],1):
        folder=RUN/'runs'/state['run_id']/'rounds';path=folder/f'{t:02d}_committed.json';c=read(path)
        assert filehash(path)==event['commit_sha256']
        assert filehash(folder/f'{t:02d}_candidates.npz')==c['candidate_table_sha256']
        assert c['history']==state['history'][:t+3] and c['history_sha256']==digest(c['history'])
        if t<=prefix:
            assert state['history'][t+3]==parent['history'][t+3]
            continue
        with np.load(folder/f'{t:02d}_candidates.npz') as f:a={key:f[key].copy() for key in f.files}
        ref=ix[event['reference_id']];selected=ix[event['selected_id']];eligible=a['eligible'];available=a['available']
        assert ref==pick(a,a['base_qLogEI'],available)
        assert np.array_equal(eligible,available&(a['base_qLogEI'][ref]-a['base_qLogEI']<=c['budget']))
        assert eligible[selected] and available[selected]
        base_re=base.acquisition(a['mu_std'],a['sd_std'],np.zeros(len(ids)),c['best_std'],c['base_meta']['mc_seed'])
        guided_re=base.acquisition(a['mu_std'],a['sd_std'],a['local_shift_y']/c['GP_scale_y'],c['best_std'],c['base_meta']['mc_seed'])
        error=max(float(np.max(abs(base_re-a['base_qLogEI']))),float(np.max(abs(guided_re-a['guided_qLogEI']))))
        maxerr=max(maxerr,error);assert error<1e-9
        guided=pick(a,a['guided_qLogEI'],eligible);assert ids[guided]==c['guided_proposal_id']
        altered=eligible&(abs(a['local_shift_y'])>1e-12);altered_js=np.flatnonzero(altered)
        if eligible.sum()==1:explanation='singleton_budget'
        elif not altered.any():explanation='zero_local_overlap_with_budget'
        elif np.all((altered_js==ref)&(a['local_shift_y'][altered_js]>0)):explanation='only_reinforces_existing_BO_reference'
        elif not altered[ref] and np.all(a['local_shift_y'][altered_js]<0):explanation='only_demotes_nonreference_candidates'
        else:explanation='other'
        alt=np.flatnonzero(eligible&(np.arange(len(ids))!=ref))
        bestalt=pick(a,a['guided_qLogEI'],eligible&(np.arange(len(ids))!=ref)) if len(alt) else None
        card=c['candidate_card'];allowed=[r for r in card if eligible[ix[r['candidate_id']]]]
        plan=c['answer']['plan'];key=event['registered_hypothesis'];model=c['local_models_at_commit'][key]
        mu=a['mu_y'][selected]*scale;sd=a['sd_predictive_y'][selected]*scale;observed=event['raw_value']
        proposal={'task':task,'query':t,'candidate_id':plan['candidate_id'],'BO_reference':ids[ref],
            'follows_own_GP':plan['candidate_id']==ids[ref],'full_eligible_count':int(eligible.sum()),
            'eligible_in_card':len(allowed),'choice_forced_by_card_and_gate':len(allowed)==1,
            'prediction_raw':plan['prediction'],'GP_mean':mu,'GP_predictive_SD':sd,'observed':observed,
            'forecast_minus_GP_over_SD':float((plan['prediction']-mu)/sd),
            'LLM_absolute_error':abs(observed-plan['prediction']),'GP_absolute_error':abs(observed-mu),
            'GP_review_loss':event['GP_review_loss'],'budget':event['budget'],'reason':plan['reason'],
            'statement':plan['hypothesis']['statement'],'focus_axes':','.join(plan['hypothesis']['focus_axes']),
            'support_indices':','.join(map(str,plan['support'])),'hypothesis_id':key,
            'same_round_actual_mean_shift':event['measurement']['actual_deployed_correction_y']*scale,
            'local_radius_normalized':model['scope']['radius_RMS_normalized']}
        proposals.append(proposal)
        known_ref=known.get(ids[ref]);reference_value=known_ref['raw'] if known_ref else None
        row={'task':task,'query':t,'eligible_count':int(eligible.sum()),'eligible_in_card':len(allowed),
            'active_models':sum(h['active_cap']>0 for h in c['local_models_at_commit'].values()),
            'positive_shifts_inside_budget':int(np.sum(altered&(a['local_shift_y']>0))),
            'negative_shifts_inside_budget':int(np.sum(altered&(a['local_shift_y']<0))),
            'shifted_candidates_inside_budget':','.join(ids[j] for j in altered_js),
            'shifts_inside_budget_raw':','.join(format(a['local_shift_y'][j]*scale,'.12g') for j in altered_js),
            'BO_reference':ids[ref],'guided_reference':ids[guided],'LLM_proposal':plan['candidate_id'],'selected':ids[selected],
            'why_local_cannot_change_choice':explanation,'guided_runner_up':ids[bestalt] if bestalt is not None else '',
            'guided_gap_to_runner_up':float(a['guided_qLogEI'][ref]-a['guided_qLogEI'][bestalt]) if bestalt is not None else None,
            'base_gap_to_same_runner_up':float(a['base_qLogEI'][ref]-a['base_qLogEI'][bestalt]) if bestalt is not None else None,
            'selected_objective':observed,'BO_reference_objective_from_archived_observations_only':reference_value,
            'immediate_difference_from_known_reference':observed-reference_value if reference_value is not None else None,
            'reference_value_posthoc_source':known_ref['first_source'] if known_ref else '',
            'incumbent_improvement':(event['incumbent_after']-event['incumbent_before'])*scale,
            'same_R2_selected':ids[selected]==parent['decisions'][t-1]['selected_id']}
        rounds.append(row)
        for k,modelkey in enumerate(c['local_hypothesis_ids']):
            h=c['local_models_at_commit'][modelkey]
            if h['active_cap']<=0:continue
            for j in np.flatnonzero(eligible&(a['local_weights'][k]>1e-12)):
                deployments.append({'task':task,'query':t,'hypothesis_created_query':h['created_query'],'hypothesis_id':modelkey,
                    'candidate_id':ids[j],'is_BO_reference':j==ref,'selected':j==selected,'model_owns_candidate':a['local_owner'][j]==k,
                    'kernel_weight':float(a['local_weights'][k,j]),'LLM_prediction_raw':h['prediction_raw'],
                    'GP_mean':float(a['mu_y'][j])*scale,'feature_raw':float(a['local_features_y'][k,j])*scale,
                    'actual_owner_coefficient':float(a['local_coefficient'][j]),'actual_owner_shift_raw':float(a['local_shift_y'][j])*scale})
        for v in event['local_feedback']['shadow_checks']:
            h=c['local_models_at_commit'][v['hypothesis_id']];delta=.25*v['feature_y'];residual=event['y']-v['GP_mean_y']
            gain=2*residual*delta-delta**2;R=v['GP_predictive_SD_y']**2+.0009
            assert abs(gain-v['paired_SE_gain'])<1e-12
            ratio=abs(delta)/np.sqrt(R);assert v['informative']==bool(ratio>=.05 and abs(v['feature_y'])>1e-12)
            if v['informative']:informative_keys.add(v['hypothesis_id'])
            if gain>1e-12:positive_keys.add(v['hypothesis_id'])
            shadows.append({'task':task,'query':t,'hypothesis_id':v['hypothesis_id'],'created_query':h['created_query'],
                'same_round_as_creation':t==h['created_query'],'candidate_id':event['selected_id'],
                'prediction_raw':h['prediction_raw'],'GP_mean':v['GP_mean_y']*scale,'observed':observed,
                'kernel_weight':v['kernel_weight'],'probe_shift':delta*scale,'shadow_prediction':(v['GP_mean_y']+delta)*scale,
                'effective_SD':np.sqrt(R)*scale,'probe_over_effective_SD':ratio,
                'probe_over_GP_predictive_SD':abs(delta)/v['GP_predictive_SD_y'],
                'paired_SE_gain_raw_squared':gain*scale**2,'standardized_gain':gain/(2*R),
                'informative':v['informative'],'direction_correct':bool(residual*delta>0),
                'positive_gain':gain>1e-12,'reason':'informative_wrong_direction' if v['informative'] and residual*delta<0
                    else 'informative_positive' if v['informative'] and gain>1e-12 else 'weak_positive' if gain>1e-12 else 'weak_nonpositive',
                'best_nonnegative_coefficient_at_this_observation_analysis_only':float(np.clip(residual/v['feature_y'],0,.5)) if v['feature_y'] else 0.})
        # Frozen-history diagnostic of checks disabled by withdrawal. These
        # values never update a bank, authorization or trajectory.
        for modelkey,h in c['local_models_at_commit'].items():
            if not h['withdrawn']:continue
            feature,weight=r3.features(pool,h,{'mu_y':a['mu_y'],'sd_predictive_y':a['sd_predictive_y']})
            if weight[selected]<=1e-12:continue
            H=float(feature[selected]);delta=.25*H;r=event['y']-a['mu_y'][selected]
            R=float(a['sd_predictive_y'][selected]**2+.0009);ratio=abs(delta)/np.sqrt(R)
            retired.append({'task':task,'query':t,'hypothesis_id':modelkey,'created_query':h['created_query'],
                'withdrawal_queries':','.join(str(v['query']) for v in h['withdrawal_records']),
                'last_actual_failure_query':h['last_actual_failure_query'],'candidate_id':event['selected_id'],
                'hypothetical_shadow_shift_raw':delta*scale,'hypothetical_paired_gain_raw_squared':(2*r*delta-delta**2)*scale**2,
                'informative':bool(ratio>=.05 and abs(H)>1e-12),'positive':2*r*delta-delta**2>1e-12,
                'scope':'offline check of already observed points only, not actual recovery evidence'})
    for key,h in state['hypotheses'].items():
        checks=[r for r in shadows if r['hypothesis_id']==key];info=[r for r in checks if r['informative']]
        modelrows.append({'task':task,'hypothesis_id':key,'created_query':h['created_query'],'anchor':h['candidate_id'],
            'prediction_raw':h['prediction_raw'],'shadow_count':len(checks),'informative_count':len(info),
            'informative_positive_count':sum(r['positive_gain'] for r in info),'weak_positive_count':sum(r['positive_gain'] and not r['informative'] for r in checks),
            'final_status':h['status'],'last_actual_failure_query':h['last_actual_failure_query'],
            'withdrawal_queries':','.join(str(v['query']) for v in h['withdrawal_records']),
            'inherits_actual_failure_ids':','.join(h['inherited_actual_failure_ids']),
            'post_failure_informative_checks':sum(r['query']>h['last_actual_failure_query'] for r in info) if h['last_actual_failure_query']>=0 else None,
            'final_stored_positive_count':h['evidence']['positive_count'],
            'negative_shadow_only_evidence_not_inherited':h['last_actual_failure_query']<0 and any(r['informative'] and not r['positive_gain'] for r in checks)})
    for source,h in state['hypotheses'].items():
        if h['last_actual_failure_query']<0:continue
        ws=r3.scope_weight(pool,h)>1e-12
        for target,g in state['hypotheses'].items():
            if source==target:continue
            wt=r3.scope_weight(pool,g)>1e-12;den=min(int(ws.sum()),int(wt.sum()))
            overlap=float(np.sum(ws&wt)/den) if den else 0.
            if overlap<.5:continue
            used=[d['iteration'] for d in state['decisions'][prefix:] if d['local_feedback']['actual_owner_hypothesis']==target and d['iteration']>h['last_actual_failure_query']]
            inheritance.append({'task':task,'failed_created_query':h['created_query'],'failure_query':h['last_actual_failure_query'],
                'other_created_query':g['created_query'],'support_overlap':overlap,
                'failure_id_inherited':source in g['inherited_actual_failure_ids'],
                'existed_before_failure_outcome':g['created_query']<=h['last_actual_failure_query'],
                'later_actual_deployment_queries':','.join(map(str,used)),
                'later_deployments_all_singleton_budget':all(state['decisions'][q-1]['eligible_count']==1 for q in used)})
    y=np.array([h['raw_value'] for h in state['history']]);yp=np.array([h['raw_value'] for h in parent['history']])
    best=np.maximum.accumulate(y);bp=np.maximum.accumulate(yp)
    assert np.array_equal(best,bp)
    following=sum(p['follows_own_GP'] for p in proposals);forced=[p for p in proposals if p['choice_forced_by_card_and_gate']]
    discretionary=[p for p in proposals if not p['choice_forced_by_card_and_gate']]
    summary={'task':task,'new_rounds':len(proposals),'LLM_follows_own_GP':following,'LLM_follow_fraction':following/len(proposals),
        'forced_by_card_and_gate':len(forced),'discretionary_rounds':len(discretionary),
        'follows_GP_when_card_has_alternatives':sum(p['follows_own_GP'] for p in discretionary),
        'LLM_nonbase_queries':[p['query'] for p in proposals if not p['follows_own_GP']],
        'LLM_all_predictions_above_GP':all(p['prediction_raw']>p['GP_mean'] for p in proposals),
        'own_new_anchor_forecast_closer_than_GP_count':int(sum(p['LLM_absolute_error']<p['GP_absolute_error'] for p in proposals)),
        'own_new_anchor_LLM_MAE':float(np.mean([p['LLM_absolute_error'] for p in proposals])),
        'own_new_anchor_GP_MAE':float(np.mean([p['GP_absolute_error'] for p in proposals])),
        'own_new_anchor_LLM_MSE':float(np.mean([p['LLM_absolute_error']**2 for p in proposals])),
        'own_new_anchor_GP_MSE':float(np.mean([p['GP_absolute_error']**2 for p in proposals])),
        'median_absolute_forecast_difference_in_GP_SD':float(np.median([abs(p['forecast_minus_GP_over_SD']) for p in proposals])),
        'models':len(modelrows),'models_with_informative_feedback':len(informative_keys),
        'models_without_informative_feedback':len(modelrows)-len(informative_keys),
        'models_with_any_weak_positive':[m['created_query'] for m in modelrows if m['weak_positive_count']>0],
        'shadow_reason_counts':dict(Counter(r['reason'] for r in shadows)),
        'informative_shadow_counts_by_creation':dict(Counter(r['same_round_as_creation'] for r in shadows if r['informative'])),
        'all_actual_deployments':sum(d['local_feedback']['actual_local_update'] is not None for d in state['decisions'][prefix:]),
        'local_selection_changes':sum(d['local_ranking_changed'] for d in state['decisions'][prefix:]),
        'local_nonchange_reasons':dict(Counter(r['why_local_cannot_change_choice'] for r in rounds)),
        'actual_recoveries':sum(v['from']=='quarantined' for v in state['authorization_events']),
        'overlapping_existing_models_not_updated_after_failure':[r['other_created_query'] for r in inheritance if not r['failure_id_inherited']],
        'raw_noise_floor_SD':Rfloor_raw,'SearchAUC':float(best[4:].mean()),'R2_SearchAUC':float(bp[4:].mean()),
        'full_incumbent_sequence_identical_to_R2':True,'fresh_choices_different_from_R2':sum(not r['same_R2_selected'] for r in rounds),
        'max_independent_acquisition_error':maxerr,'GP_fits':0,'new_LLM_calls':0,'new_experiment_calls':0,
        'operator_compiles_in_loop':sum(base.JIT.values())}
    for name,rows in [('rounds',rounds),('proposals',proposals),('shadow_checks',shadows),('active_scope',deployments),
                      ('model_lifecycle',modelrows),('withdrawn_hypothetical_checks',retired),('inheritance_consistency',inheritance)]:
        csv_write(OUT/(task+'_'+name+'.csv'),rows)
    return summary


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    keep=read(STUDY/'protected_snapshot.json');keep.update(read(STUDY/'release_inventory.json'))
    for name in ['release_inventory.json','release_receipt.json']:
        path=STUDY/name;keep[path.relative_to(ROOT).as_posix()]=filehash(path)
    for path,expected in keep.items():assert filehash(ROOT/path)==expected,path
    with ProcessPoolExecutor(max_workers=2) as executor:summary=list(executor.map(task_analysis,['SnAr','HPLC']))
    for path,expected in keep.items():assert filehash(ROOT/path)==expected,path
    write(OUT/'source_snapshot.json',keep)
    write(OUT/'diagnosis.json',{'scope':'frozen-history analysis only; no trajectories, LLM, fitting, oracle calls, or parameter changes',
        'source_script_sha256':filehash(__file__),'checks':summary,'preserved_files':len(keep),
        'interpretation':'Known reference objective comparisons are single-point post hoc comparisons, not alternate-trajectory SearchAUC estimates.'})
    print(json.dumps({'status':'PASS','sources_unchanged':len(keep),'checks':summary},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
