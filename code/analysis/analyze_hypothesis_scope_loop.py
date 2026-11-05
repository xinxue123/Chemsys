"""Read-only diagnosis from sealed decisions and posterior arrays; no new optimization."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from copy import deepcopy
from math import erf,sqrt
import json
import hashlib
import numpy as np
import pandas as pd
from hypothesis_scope_loop import study as s,knowledge as k,policy

OUT=ROOT/'experiments/reports/hypothesis_scope_loop_diagnosis_20261105'
TASKS=('Reizman_Suzuki_Case_1','Reizman_Suzuki_Case_2','Doyle_Buchwald_Hartwig','HPLC')
KF_REASONS={'low_relative_predictive_skill','two_negative_relative_forecasts'}


def independent_KF(before,p_llm,p_gp,improved):
    from math import erfc,sqrt
    z=(float(improved)-p_gp)**2-(float(improved)-p_llm)**2
    noise=max(.0025,4*(p_llm-p_gp)**2*p_gp*(1-p_gp))
    predicted=before['variance']+.0025;gain=predicted/(predicted+noise)
    mean=before['mean']+gain*(z-before['mean'])
    variance=(1-gain)**2*predicted+gain**2*noise
    prob=.5*erfc(-mean/sqrt(2*variance));r=.8*prob
    if z<=1e-12:r=min(r,before['reliability'])
    return dict(z=z,kalman_gain=gain,reliability=r,mean=mean,variance=variance)


def context(arrays,pool):
    c={key:arrays[key].copy() for key in ('base_logei','available','tie_priority')}
    c['bo_index']=policy.argmax(c,c['available']);return c


def ignore_only_KF(notes):
    out=deepcopy(notes)
    for note in out:
        if note['permission']=='paused' and set(note['pause_reasons']) and set(note['pause_reasons'])<=KF_REASONS:
            note['permission']='active'
        note['KF']['reliability']=.4
    return out


def posterior_row(pool,path,candidate):
    j=pool.ix[candidate]
    with np.load(path) as a:
        valid=a['available'];scores=a['base_logei'];best=float(scores[valid].max())
        return dict(candidate=candidate,mean_raw=float(a['mu_y'][j]*pool.scale),SD_raw=float(a['sd_y'][j]*pool.scale),
            qLogEI=float(scores[j]),qLogEI_gap=best-float(scores[j]),target_available=bool(valid[j]),
            strictly_better_available=int(np.sum(valid&(scores>scores[j]))),
            exact_equal_available=int(np.sum(valid&(scores==scores[j]))))


def main():
    OUT.mkdir(parents=True,exist_ok=True);p=s.check(True)
    frozen=s.read(s.BASE/'completed_artifacts_sha256.json')['files']
    for name,h in frozen.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
    rounds=[];updates=[];hyp_rows=[];refresh_rows=[];static=[];predictions=[];summaries=[];details={};kernel_rows=[];geometry=[]
    for task in TASKS:
        short=s.SHORT[task];pool=s.pool_for(task);scopes=k.Scopes(pool);folder=s.folder_for(task)
        new=s.read(folder/'state.json');old=s.read(ROOT/p['controls'][task]['LLM_first_then_BO']['path'])
        assert new['history'][0]==old['history'][0]
        values=np.asarray([h['raw_value'] for h in new['history']]);oldvalues=np.asarray([h['raw_value'] for h in old['history']])
        best=np.maximum.accumulate(values);oldbest=np.maximum.accumulate(oldvalues)
        d_total=float(np.sum(best-oldbest)/20)
        priors=[];task_static=[]
        known={h['candidate_id']:h['raw_value'] for h in new['history']+old['history']}
        for i,(r,br) in enumerate(zip(new['decisions'],old['decisions'])):
            q=i+1;d=r['decision'];commit=s.read(folder/'rounds'/f'{q:02d}_commit.json')
            assert commit['commit_sha256']==s.digest({a:b for a,b in commit.items() if a!='commit_sha256'})
            notes=commit['working_knowledge']['notes'];inc=d['incumbent_before']
            prior_reviewed,_=k.pause_for_recommendation(priors,inc)
            item=dict(task=short,query=q,selected=r['candidate_id'],observed=r['raw_value'],best=float(best[i]),
                control_selected=br['candidate_id'],control_observed=br['raw_value'],control_best=float(oldbest[i]),
                cumulative_gap=float(best[i]-oldbest[i]),AUC_contribution=float((best[i]-oldbest[i])/20),
                actual_changed_from_current_prefix_BO=d.get('changed_from_unrestricted_BO',False),
                BO_id=d.get('unrestricted_BO_id'),selector=d['reason'],scope_count=d['scope_count'],
                acquisition_gap=d.get('qLogEI_opportunity_gap'),refresh=d['refresh_event']['call'],
                refresh_reasons=';'.join(d['refresh_event']['reasons']),refresh_hold=d['refresh_event'].get('hold'),
                conditions=json.dumps(d['conditions'],ensure_ascii=False))
            if q>1:
                with np.load(folder/'rounds'/f'{q:02d}_GP.npz') as arrays:
                    assert s.sha(folder/'rounds'/f'{q:02d}_GP.npz')==commit['GP_arrays_sha256']
                    ctx=context(arrays,pool)
                    actual=policy.select(ctx,scopes,notes,inc)
                    assert pool.ids[actual['index']]==r['candidate_id'] and pool.ids[ctx['bo_index']]==d['unrestricted_BO_id']
                    no_kf=policy.select(ctx,scopes,ignore_only_KF(notes),inc)
                    no_refresh=policy.select(ctx,scopes,prior_reviewed,inc) if d['refresh_event']['call'] else actual
                    scores=ctx['base_logei'];maximum=float(scores[ctx['bo_index']]);avail=ctx['available']
                    item.update(available_count=int(avail.sum()),exact_max_ties=int(np.sum(avail&(scores==maximum))),
                        max_ties_within_1e_12=int(np.sum(avail&(abs(scores-maximum)<=1e-12))),
                        scope_passes_unrestricted_BO=bool(actual['mask'][ctx['bo_index']]),
                        local_no_KF_choice=pool.ids[no_kf['index']],local_no_KF_changes_choice=no_kf['index']!=actual['index'],
                        local_no_refresh_choice=pool.ids[no_refresh['index']],local_no_refresh_changes_choice=no_refresh['index']!=actual['index'])
                    if no_kf['index']!=actual['index']:
                        row=dict(task=short,query=q,actual=r['candidate_id'],no_KF_choice=pool.ids[no_kf['index']],
                            actual_observed=r['raw_value'],no_KF_value_if_already_observed=known.get(pool.ids[no_kf['index']]),
                            note='Same saved prefix/scores/LLM proposals; remove only KF pauses/credibility gate. Not a trajectory ablation.')
                        static.append(row);task_static.append(row)
                    for note in notes:
                        own=scopes.mask(note['scope']);geom=ctx['available']&(own if note['direction']=='favour' else ~own)
                        if not geom.any():continue
                        j=policy.argmax(ctx,geom);candidate=pool.ids[j]
                        geometry.append(dict(task=short,query=q,id=note['id'],direction=note['direction'],permission=note['permission'],
                            pause_reasons=';'.join(note['pause_reasons']),p_improve=k.p_improve(note,inc),
                            actual=r['candidate_id'],geometry_only_choice=candidate,
                            geometry_only_value_if_already_observed=known.get(candidate),
                            geometry_only_changes_choice=j!=actual['index'],incumbent_before=inc,
                            note='One rule geometry only; bypass all recommendation/KF permissions, same saved prefix. Diagnostic, not a proposed policy.'))
                    if d['refresh_event']['call']:
                        call=next(c for c in new['calls'] if c['query']==q);pro=call['projection'];pid=pro['candidate_id'] if pro else None
                        refresh_rows.append(dict(task=short,query=q,preferred_id=pid,actual=r['candidate_id'],
                            preferred_executed=pid==r['candidate_id'],preferred_value_if_observed_in_saved_trajectories=known.get(pid),
                            no_refresh_choice=pool.ids[no_refresh['index']],refresh_changes_choice=no_refresh['index']!=actual['index'],
                            event_reasons=';'.join(d['refresh_event']['reasons'])))
                if q in (3,5,6,10):
                    targets={'c1':['p04671','p01031'],'doyle':['p00672','p03638'],'c2':['p03631']}.get(short,[])
                    for target in targets:
                        for arm,path in (('new',folder/'rounds'/f'{q:02d}_GP.npz'),
                            ('control',ROOT/p['controls'][task]['LLM_first_then_BO']['path'])):
                            if arm=='control':path=path.parent/'rounds'/f'{q:02d}_GP.npz'
                            predictions.append(dict(task=short,query=q,arm=arm,**posterior_row(pool,path,target)))
            rounds.append(item)
            for note in notes:
                if d['refresh_event']['call']:
                    mask=scopes.mask(note['scope'])
                    hyp_rows.append(dict(task=short,query=q,id=note['id'],direction=note['direction'],scope=json.dumps(note['scope']),
                        claim=note['claim'],forecast_mean=note['forecast']['mean_raw'],forecast_SD=note['forecast']['sd_raw'],
                        permission=note['permission'],reliability=note['KF']['reliability'],p_improve=k.p_improve(note,inc),
                        region_size=int(mask.sum()),semantic_key=note['semantic_key'],inherited_from=json.dumps(note['inherited_from'])))
            for event in r['knowledge_feedback']:
                kf=event['KF']
                if kf:
                    f=event['forecast'];pll=.5*(1+erf((f['mean_raw']-inc)/f['sd_raw']/sqrt(2)))
                    with np.load(folder/'rounds'/f'{q:02d}_GP.npz') as saved:
                        j=pool.ix[r['candidate_id']]
                        gpmean=float(saved['mu_y'][j]*pool.scale);gpsd=max(float(saved['sd_predictive_y'][j]*pool.scale),1e-12)
                    pgp=.5*(1+erf((gpmean-inc)/gpsd/sqrt(2)))
                    assert abs(pll-event['p_LLM'])<1e-12 and abs(pgp-event['p_GP'])<1e-12
                    independent=independent_KF(kf['before'],event['p_LLM'],event['p_GP'],event['improved'])
                    for actual_name,ind_name in (('Brier_gain','z'),('kalman_gain','kalman_gain')):assert abs(kf[actual_name]-independent[ind_name])<1e-12
                    for name in ('reliability','mean','variance'):assert abs(kf['after'][name]-independent[name])<1e-12
                updates.append(dict(task=short,query=q,id=event['id'],caused_selection=event['caused_selection'],
                    p_LLM=event['p_LLM'],p_GP=event['p_GP'],improved=event['improved'],
                    Brier_gain=kf['Brier_gain'] if kf else None,reliability_before=kf['before']['reliability'] if kf else None,
                    reliability_after=kf['after']['reliability'] if kf else None,kalman_gain=kf['kalman_gain'] if kf else None,
                    residual_SD=event['residual_SD'],permission_before=event['permission_before'],permission_after=event['permission_after'],
                    reasons=';'.join(event['reasons'])))
            priors=s.read(folder/'rounds'/f'{q:02d}_feedback.json')['notes_after_feedback']
        rtask=[r for r in rounds if r['task']==short]
        summaries.append(dict(task=short,control_SearchAUC=float(oldbest.mean()),new_SearchAUC=float(best.mean()),delta_SearchAUC=d_total,
            control_final=float(oldbest[-1]),new_final=float(best[-1]),
            scoped_rounds=sum(r['selector']=='hypothesis_constrained_native_qLogEI' for r in rtask),
            changed_choices=sum(r['actual_changed_from_current_prefix_BO'] for r in rtask),
            KF_updates=sum(r['Brier_gain'] is not None for r in updates if r['task']==short),
            direct_refresh_changes=[r['query'] for r in refresh_rows if r['task']==short and r['refresh_changes_choice']],
            same_prefix_KF_gate_changes=[r['query'] for r in task_static],
            identical_to_control=[r['candidate_id'] for r in new['history']]==[r['candidate_id'] for r in old['history']]))
        if short=='c1':
            contributions={label:float(np.sum((best-oldbest)[lo-1:hi])/20) for label,lo,hi in (('q1_5',1,5),('q6_11',6,11),('q12_19',12,19),('q20',20,20))}
        elif short=='c2':
            contributions={label:float(np.sum((best-oldbest)[lo-1:hi])/20) for label,lo,hi in (('q1_3',1,3),('q4_9',4,9),('q10_12',10,12),('q13_20',13,20))}
        elif short=='doyle':
            contributions={label:float(np.sum((best-oldbest)[lo-1:hi])/20) for label,lo,hi in (('q1_2',1,2),('q3_5',3,5),('q6_9',6,9),('q10_13',10,13),('q14_20',14,20))}
        else:contributions={}
        details[short]=dict(AUC_decomposition=contributions,initial_first_equal=True,
            initial_conditions=new['decisions'][0]['decision']['conditions'],calls=[c['query'] for c in new['calls']],
            original_permission_events=new['permission_events'])
        if short in ('c1','doyle'):
            for q in (5,6):
                for arm,folder2 in (('new',folder),('control',(ROOT/p['controls'][task]['LLM_first_then_BO']['path']).parent)):
                    params=s.read(folder2/'rounds'/f'{q:02d}_commit.json')['GP_parameters']
                    for name,value in params.items():kernel_rows.append(dict(task=short,query=q,arm=arm,parameter=name,raw_value=json.dumps(value)))
    for name,data in (('rounds',rounds),('KF_feedback',updates),('hypotheses',hyp_rows),('refresh_choices',refresh_rows),
        ('same_prefix_KF_checks',static),('saved_posteriors',predictions),('summary',summaries),('raw_kernel_parameters',kernel_rows),('geometry_only_checks',geometry)):
        pd.DataFrame(data).to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
    result=dict(status='READ_ONLY_ANALYSIS_COMPLETE',new_experimental_queries=0,new_GP_fits=0,new_LLM_calls=0,
        new_oracle_reads=0,checked_frozen_artifacts=len(frozen),protected_historical_artifacts=len(p['protected_sha256']),
        summaries=summaries,details=details,
        caveat='Static one-step checks hold actual histories, GP arrays and LLM responses fixed; not full-trajectory causal ablations.',
        source=str(Path(__file__).relative_to(ROOT)),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,ensure_ascii=True))


if __name__=='__main__':
    from unittest.mock import patch
    def forbidden(*args,**kwargs):raise AssertionError('This diagnostic forbids GP runs, LLM calls and oracle reads')
    with patch.object(s.native,'read_observation',side_effect=forbidden),patch.object(s.ColdGP,'__init__',side_effect=forbidden),patch.object(s.transport,'request',side_effect=forbidden):main()
