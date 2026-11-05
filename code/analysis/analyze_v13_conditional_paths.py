"""Explain frozen path divergence after removing high-start advantage."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import io
import json
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/analysis'),str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
import profile_v13_all_paths as profile
from knowledge_rank_v9 import engine

SOURCE=ROOT/'experiments/reports/v13_all_paths_20261104'
OUT=ROOT/'experiments/reports/v13_conditional_paths_20261104'
READS={};POOLS={};TRUTH={};STATES={};GP={}
CUTOFFS=(.50,.65,.80,.90)
START_FEATURES=['response_SD4_normalized','Gower4','category_contrast_pairs4',
    'GP_mean_SD_normalized_start','GP_coverage95_start','GP_RMSE_normalized_start',
    'GP_high_underestimated_start','GP_mixed_category_correlation_start',
    'GP_mean_truth_rank_start','GP_best90_acquisition_midrank_fraction_start']
WINDOW_FEATURES=['menu90_rate_5to8','reference90_rate_5to8','menu_regret_norm_5to8',
    'changed_reference_rate_5to8','one_factor_rate_5to8','low_one_factor_count_5to8',
    'KF_evaluations_5to8','KF_weight_changes_5to8','editor_calls_5to8']
FORWARD_GP_FEATURES=['GP_mean_SD_normalized_8','GP_coverage95_8','GP_RMSE_normalized_8',
    'GP_mean_truth_rank_8','GP_best90_acquisition_midrank_fraction_8']
OUTCOMES=['AUC_normalized','final_normalized','native_gain_AUC_normalized',
          'headroom_gain_AUC','post8_headroom_gain_AUC']


def tracked(path):
    p=Path(path).resolve();raw=p.read_bytes()
    READS[p.relative_to(ROOT).as_posix()]=hashlib.sha256(raw).hexdigest()
    return raw


def csvread(path):return pd.read_csv(io.BytesIO(tracked(path)),encoding='utf-8-sig')
def jsread(path):return json.loads(tracked(path).decode('utf-8-sig'))
def save(name,data):
    (data if isinstance(data,pd.DataFrame) else pd.DataFrame(data)).to_csv(
        OUT/name,index=False,encoding='utf-8-sig',lineterminator='\n')


def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def tier(frac):return 'high_80plus' if frac>=.8 else 'middle_50to80' if frac>=.5 else 'low_below50'


def metrics(r):
    s=STATES[r.path_id];v=np.array([h['raw_value'] for h in s['history']]);best=np.maximum.accumulate(v)
    truth=TRUTH[r.task];lo=float(truth.min());hi=float(truth.max());span=hi-lo
    k=10 if r.regime=='continuation10+10' else 4
    scored_from=10 if r.regime=='continuation10+10' else int(r.initial)
    T=20-scored_from;H=20-k;start=best[k-1];gain=best[-1]-start
    A=float(best[scored_from:].mean())
    hold=best.copy();hold[k:]=start;baseline=float(hold[scored_from:].mean())
    delta=np.maximum(v[k:]-best[k-1:-1],0)
    np.testing.assert_allclose(delta.sum(),gain,rtol=1e-10,atol=1e-8)
    weighted=float(np.dot(np.arange(H,0,-1)/H,delta))
    timing=weighted/gain if gain>1e-10 else 0.
    np.testing.assert_allclose(A,baseline+H/T*gain*timing,rtol=1e-11,atol=1e-8)
    prefix=[(h['candidate_id'],h['raw_value']) for h in s['history'][:k]]
    d=r.to_dict();d.update(landmark=k,start_incumbent=float(start),start_fraction_max=float(start/hi),
        start_tier=tier(start/hi),pool_max=hi,pool_min=lo,pool_range=span,
        prefix_id=digest(dict(task=r.task,landmark=k,history=prefix)),analysis_AUC=A,
        analysis_queries=T,remaining_queries=H,prefix_hold_AUC=baseline,
        native_gain_AUC=A-baseline,native_gain_AUC_normalized=(A-baseline)/span,
        final_gain=float(gain),final_gain_normalized=float(gain/span),record_timing_factor=timing,
        headroom_gain_AUC=weighted/(hi-start) if hi-start>1e-8 else None,
        post_start_records=int((delta>1e-9).sum()),
        post_start_failures=int((delta<=1e-9).sum()),
        first90_delay=next((i for i,x in enumerate(v[k:],1) if x>=.9*hi),H+1),
        first95_delay=next((i for i,x in enumerate(v[k:],1) if x>=.95*hi),H+1),
        first_point_fraction_max=float(v[0]/hi),
        post8_headroom_gain_AUC=float((best[8:].mean()-best[7])/(hi-best[7])) if hi-best[7]>1e-8 else None,
        best8_fraction_max=float(best[7]/hi))
    return d


def gp_probe(spec):
    r,n=spec;pool=POOLS[r.task];truth=TRUTH[r.task]
    folder,q,reused=profile.asset_folder(r.to_dict(),n+1)
    with np.load(io.BytesIO(tracked(folder/'rounds'/f'{q:02}_GP.npz')),allow_pickle=False) as z:
        arrays={key:z[key].copy() for key in ('mu_y','sd_y','sd_predictive_y','base_logei','available','tie_priority')}
    c=jsread(folder/'rounds'/f'{q:02}_commit.json')
    ctx=dict(stats={key:arrays[key] for key in ('mu_y','sd_y','sd_predictive_y')},
             **{key:arrays[key] for key in ('base_logei','available','tie_priority')})
    h=engine.gp_hash(ctx)
    if 'GP_sha256' in c:assert h==c['GP_sha256']
    available=arrays['available'];ix=np.flatnonzero(available);hi=float(truth.max());span=float(np.ptp(truth))
    np.testing.assert_array_equal(available,~np.isin(pool.ids,[h['candidate_id'] for h in STATES[r.path_id]['history'][:n]]))
    mu=arrays['mu_y']*pool.scale;sd=arrays['sd_y']*pool.scale;base=arrays['base_logei']
    high=available&(truth>=.9*hi);high95=available&(truth>=.95*hi)
    high_score=base[high].max() if high.any() else None
    midrank=(np.count_nonzero(base[ix]>high_score)+(np.count_nonzero(base[ix]==high_score)+1)/2)/len(ix) if high.any() else None
    params=jsread(folder/'rounds'/f'{q:02}_fit.json');params=params.get('parameters',params)
    tag='perera' if pool.group=='P1' else r.task
    cat=profile.helpers.category_parameters(params,tag)
    corr=pd.Series(mu[ix]).rank().corr(pd.Series(truth[ix]).rank()) if np.ptp(mu[ix])>0 else None
    menu=c.get('display_ids',c.get('ordered_ids',c.get('menu_ids',[])))
    return dict(path_id=r.path_id,observed=n,GP_sha256=h,
        menu_set_sha256=digest(sorted(menu)),menu_size=len(menu),reference_id=c.get('BO_reference_id'),
        mean_SD_normalized=float(sd[ix].mean()/span),
        coverage95=float(np.mean(np.abs(mu[ix]-truth[ix])<=1.96*sd[ix])),
        RMSE_normalized=float(np.sqrt(np.mean((mu[ix]-truth[ix])**2))/span),
        high_underestimated=float(np.mean((mu+1.96*sd)[high95]<.95*hi)) if high95.any() else None,
        mixed_category_correlation=cat['category_correlation'],mean_truth_rank=corr,
        best90_acquisition_midrank_fraction=midrank)


def behavior(df,rounds):
    out=[]
    for _,r in df.iterrows():
        b=rounds[(rounds.path_id==r.path_id)&rounds.total_observations.between(5,8)]
        assert len(b)==4
        out.append(dict(path_id=r.path_id,
            menu90_rate_5to8=float((b.menu_best>=.9*r.pool_max).mean()),
            reference90_rate_5to8=float((b.reference_objective>=.9*r.pool_max).mean()),
            menu_regret_norm_5to8=float(((b.menu_best-b.objective)/r.pool_range).mean()),
            changed_reference_rate_5to8=float(b.changed_from_reference.mean()),
            one_factor_rate_5to8=float(b.one_factor_vs_incumbent.mean()),
            low_one_factor_count_5to8=int(b.low_incumbent_one_factor.sum()),
            KF_evaluations_5to8=int(b.KF_updates.sum()),
            KF_weight_changes_5to8=int((b.KF_increases+b.KF_decreases).sum()),
            editor_calls_5to8=int(b.editor_call.sum())))
    return pd.DataFrame(out)


def correlation(b,x,y,controls,method_adjust=False,prefix_adjust=False):
    cols=list(dict.fromkeys([x,y]+controls+(['method'] if method_adjust else [])+
                           (['prefix_id'] if prefix_adjust else [])))
    z=b[cols].dropna();n=len(z)
    if n<6:return None,n,'fewer_than_6'
    if z[x].nunique()<2:return None,n,'constant_feature'
    if z[y].nunique()<2:return None,n,'constant_outcome'
    X=[np.ones(n)]
    for ctrl in controls:
        if z[ctrl].nunique()>1:X.append(z[ctrl].rank().to_numpy())
    if method_adjust:
        dummy=pd.get_dummies(z.method,drop_first=True,dtype=float)
        X.extend(dummy[col].to_numpy() for col in dummy)
    if prefix_adjust:
        dummy=pd.get_dummies(z.prefix_id,drop_first=True,dtype=float)
        X.extend(dummy[col].to_numpy() for col in dummy)
    X=np.column_stack(X);rank=np.linalg.matrix_rank(X)
    if n-rank<3:return None,n,'insufficient_residual_df'
    a=z[x].rank().to_numpy();v=z[y].rank().to_numpy()
    a=a-X@np.linalg.lstsq(X,a,rcond=None)[0];v=v-X@np.linalg.lstsq(X,v,rcond=None)[0]
    if np.std(a)<1e-9:return None,n,'constant_feature_residual'
    if np.std(v)<1e-9:return None,n,'constant_outcome_residual'
    return float(np.corrcoef(a,v)[0,1]),n,'informative'


def correlations(df):
    rows=[]
    main=df[df.primary&(df.backend!='historical_API')]
    for cutoff in CUTOFFS:
        low=main[main.start_fraction_max<cutoff]
        for regime,reg in low.groupby('regime'):
            for mode in ('start_adjusted','start_and_method_adjusted'):
                for feature in START_FEATURES+WINDOW_FEATURES+FORWARD_GP_FEATURES:
                    for outcome in (['post8_headroom_gain_AUC'] if feature in FORWARD_GP_FEATURES else OUTCOMES):
                        ctrls=['best8_fraction_max'] if outcome=='post8_headroom_gain_AUC' else ['start_fraction_max']
                        for task,b in reg.groupby('task'):
                            # Warm GP/design-at-start features repeat across methods
                            # within only four physical prefixes. Do not count these
                            # copies as six or more independent starting states.
                            effective=b.prefix_id.nunique()
                            if feature in START_FEATURES and effective<6:
                                value,n,status=None,len(b),'fewer_than_6_distinct_start_prefixes'
                            else:
                                value,n,status=correlation(b,feature,outcome,ctrls,
                                    mode=='start_and_method_adjusted',regime=='warm')
                            rows.append(dict(cutoff=cutoff,regime=regime,adjustment=mode,
                                task=task,feature=feature,outcome=outcome,n=n,distinct_prefixes=effective,
                                prefix_fixed_effects=regime=='warm',rho=value,status=status))
    blocks=pd.DataFrame(rows);save('conditional_association_blocks.csv',blocks)
    totals=[]
    keys=['cutoff','regime','adjustment','feature','outcome']
    for group,b in blocks.groupby(keys):
        valid=b[b.status=='informative'];a=valid.rho.to_numpy();tasks=valid.task.unique()
        leave=[valid[valid.task!=t].rho.mean() for t in tasks if (valid.task!=t).any()]
        totals.append(dict(zip(keys,group),informative_tasks=len(valid),paths_used=int(valid.n.sum()),
            mean_rho=float(a.mean()) if len(a) else None,median_rho=float(np.median(a)) if len(a) else None,
            positive_tasks=int((a>1e-8).sum()),negative_tasks=int((a< -1e-8).sum()),
            zero_tasks=int((np.abs(a)<=1e-8).sum()),
            leave_one_task_min=min(leave) if leave else None,leave_one_task_max=max(leave) if leave else None))
    result=pd.DataFrame(totals);save('conditional_association_summary.csv',result)
    return result


def paired(df):
    main=df[df.primary];bo=main[main.method=='bo'].set_index(['task','regime','seed'])
    out=[]
    for _,a in main[main.method!='bo'].iterrows():
        b=bo.loc[(a.task,a.regime,a.seed)]
        same=a.prefix_id==b.prefix_id
        if a.regime=='warm':
            assert same
            assert a.GP_sha256_start==b.GP_sha256_start,(a.path_id,b.path_id)
            assert a.GP_menu_set_sha256_start==b.GP_menu_set_sha256_start,(a.path_id,b.path_id)
            assert a.GP_reference_id_start==b.GP_reference_id_start,(a.path_id,b.path_id)
        gain_effect=(a.final_gain-b.final_gain)*(a.record_timing_factor+b.record_timing_factor)/2
        timing_effect=(a.record_timing_factor-b.record_timing_factor)*(a.final_gain+b.final_gain)/2
        factor=a.remaining_queries/a.analysis_queries
        base_diff=a.prefix_hold_AUC-b.prefix_hold_AUC
        assert np.isclose(a.analysis_AUC-b.analysis_AUC,base_diff+factor*(gain_effect+timing_effect))
        delta=(a.analysis_AUC-b.analysis_AUC)/a.pool_range
        out.append(dict(path_id=a.path_id,BO_path_id=b.path_id,task=a.task,regime=a.regime,seed=a.seed,
            method=a.method,backend=a.backend,same_prefix=same,start_fraction_max=a.start_fraction_max,
            BO_start_fraction_max=b.start_fraction_max,close_start_5pct=abs(a.start_incumbent-b.start_incumbent)<=.05*a.pool_max,
            delta_AUC_normalized=delta,delta_final_normalized=a.final_normalized-b.final_normalized,
            baseline_effect_normalized=base_diff/a.pool_range,
            final_gain_effect_normalized=factor*gain_effect/a.pool_range,
            timing_effect_normalized=factor*timing_effect/a.pool_range,
            native_post_start_gain_diff_normalized=(a.native_gain_AUC-b.native_gain_AUC)/a.pool_range,
            AUC_result='win' if delta>1e-9 else 'loss' if delta< -1e-9 else 'tie',
            **{'delta_'+f:a[f]-b[f] for f in START_FEATURES+WINDOW_FEATURES}))
    data=pd.DataFrame(out);save('all_main_BO_pairs_decomposed.csv',data)
    result=[]
    for cutoff in CUTOFFS:
        for label,mask in [('both_below', (data.start_fraction_max<cutoff)&(data.BO_start_fraction_max<cutoff)),
                           ('both_below_and_close', (data.start_fraction_max<cutoff)&(data.BO_start_fraction_max<cutoff)&data.close_start_5pct),
                           ('both_high', (data.start_fraction_max>=cutoff)&(data.BO_start_fraction_max>=cutoff))]:
            for group,b in data[mask].groupby(['regime','method','backend']):
                result.append(dict(cutoff=cutoff,cohort=label,**dict(zip(('regime','method','backend'),group)),n=len(b),
                    AUC_wins=int((b.AUC_result=='win').sum()),AUC_ties=int((b.AUC_result=='tie').sum()),
                    AUC_losses=int((b.AUC_result=='loss').sum()),
                    mean_delta_AUC_normalized=b.delta_AUC_normalized.mean(),
                    mean_delta_final_normalized=b.delta_final_normalized.mean(),
                    mean_baseline_effect_normalized=b.baseline_effect_normalized.mean(),
                    mean_final_gain_effect_normalized=b.final_gain_effect_normalized.mean(),
                    mean_timing_effect_normalized=b.timing_effect_normalized.mean()))
    summary=pd.DataFrame(result);save('matched_BO_cohort_summary.csv',summary)
    return data,summary


def strata(df):
    out=[]
    for group,b in df.groupby(['family','task','regime','backend','method','start_tier']):
        out.append(dict(zip(('family','task','regime','backend','method','start_tier'),group),n=len(b),
            mean_start_fraction_max=b.start_fraction_max.mean(),mean_SearchAUC=b.analysis_AUC.mean(),
            mean_native_gain_AUC_normalized=b.native_gain_AUC_normalized.mean(),
            mean_headroom_gain_AUC=b.headroom_gain_AUC.mean(),mean_final_normalized=b.final_normalized.mean(),
            mean_final_gain_normalized=b.final_gain_normalized.mean(),
            mean_record_timing_factor=b.record_timing_factor.mean(),reached95=int((b.final_fraction_max>=.95).sum())))
    result=pd.DataFrame(out);save('all_start_strata.csv',result)
    save('high_start_paths_separate.csv',df[df.start_fraction_max>=.8])
    save('low_middle_start_paths.csv',df[df.start_fraction_max<.8])
    first=[]
    for cutoff in CUTOFFS:
        for (regime,method,backend),b in df[df.primary].groupby(['regime','method','backend']):
            first.append(dict(cutoff=cutoff,regime=regime,method=method,backend=backend,n=len(b),
                low_first_point=int((b.first_point_fraction_max<cutoff).sum()),
                low_first_four=int((b.start_fraction_max<cutoff).sum())))
    save('first_point_vs_four_point_exclusion.csv',first)
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    tracked(__file__);tracked(ROOT/'docs/v13_conditional_paths_20261104.txt')
    df=csvread(SOURCE/'all_path_profiles.csv');rounds=csvread(SOURCE/'all_decision_features.csv')
    for task,full in profile.FULL.items():
        pool=profile.late.pool_for(full) if task=='aryl' else profile.knowledge_gp.load_task(ROOT,full,seed=0)
        POOLS[task]=pool;tracked(pool.source_path)
        with np.load(io.BytesIO(tracked(pool.source_path.with_name('oracle.npz'))),allow_pickle=False) as z:
            TRUTH[task]=z['raw_values'].copy() if 'raw_values' in z else z['values'].copy()*pool.scale
    for _,r in df.iterrows():
        raw=tracked(ROOT/r.source_path);assert hashlib.sha256(raw).hexdigest()==r.source_sha256
        STATES[r.path_id]=json.loads(raw.decode('utf-8-sig'))
    profile.STATES=STATES
    data=pd.DataFrame([metrics(r) for _,r in df.iterrows()])
    specs=[(r,n) for _,r in data.iterrows() for n in ([10] if r.landmark==10 else [4,8])]
    with ThreadPoolExecutor(max_workers=8) as executor:
        probes=list(executor.map(gp_probe,specs))
    for p in probes:GP[p['path_id'],p['observed']]=p
    for i,r in data.iterrows():
        for key,value in GP[r.path_id,int(r.landmark)].items():
            if key in ('path_id','observed'):continue
            data.loc[i,('GP_sha256_start' if key=='GP_sha256' else 'GP_'+key+'_start')]=value
        if r.landmark==4:
            for key,value in GP[r.path_id,8].items():
                if key not in ('path_id','observed','GP_sha256'):
                    data.loc[i,'GP_'+key+'_8']=value
    data=data.merge(behavior(data,rounds),on='path_id',validate='one_to_one')
    save('conditional_path_profiles.csv',data);save('archived_start_GP_diagnostics.csv',probes)
    stratum=strata(data);assoc=correlations(data);pairs,paired_summary=paired(data)
    for path,h in READS.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h,path
    audit=dict(status='PASS_CONDITIONAL_PATH_ANALYSIS',catalogue_states=len(data),main_paths=int(data.primary.sum()),
        from_start_nonreplay=int(((data.regime!='continuation10+10')&(~data.explicit_replay)).sum()),
        main_below80=int((data.primary&(data.start_fraction_max<.8)).sum()),
        main_high80=int((data.primary&(data.start_fraction_max>=.8)).sum()),
        exact_warm_BO_pairs=int((pairs.regime=='warm').sum()),
        GP_snapshots_checked=len(probes),source_files_checked=len(READS),
        metric_decompositions_checked=len(data),new_GP_fits=0,new_LLM_calls=0,new_experiments=0)
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    (OUT/'source_hashes.json').write_text(json.dumps(READS,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(audit,indent=2))
    print(paired_summary[(paired_summary.cutoff==.8)&(paired_summary.method=='v13')].to_string(index=False))


if __name__=='__main__':main()
