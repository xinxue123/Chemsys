"""Population summaries of frozen trajectories; no new optimization runs."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/reports/v13_all_paths_20261104'
FEATURES=[
    'best4_normalized','best8_normalized','response_SD4_normalized',
    'Gower4','Gower8','one_factor_pairs8','category_contrast_pairs8',
    'GP_mean_SD_normalized_4','GP_mean_SD_normalized_8','GP_coverage95_4','GP_coverage95_8',
    'GP_high_underestimated_fraction_4','GP_mixed_category_correlation_4',
    'missed_reference_record_through8','low_incumbent_one_factor_through8',
    'KF_updates_through8','editor_call_through8']
OUTCOMES=['AUC_normalized','final_normalized','future_AUC_after8_normalized',
          'remaining_best_gain_after8_normalized']
READS={}


def read(name):
    p=OUT/name;READS[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    return pd.read_csv(p,encoding='utf-8-sig')


def save(name,records):
    frame=records if isinstance(records,pd.DataFrame) else pd.DataFrame(records)
    frame.to_csv(OUT/name,index=False,encoding='utf-8-sig',lineterminator='\n')


def rho(frame,x,y):
    xy=frame[[x,y]].dropna()
    if len(xy)<4:return None,len(xy),'fewer_than_4'
    if xy[x].nunique()<2:return None,len(xy),'constant_feature'
    if xy[y].nunique()<2:return None,len(xy),'constant_outcome'
    return float(xy[x].rank(method='average').corr(xy[y].rank(method='average'))),len(xy),'informative'


def associations(scopes):
    blocks=[];totals=[]
    for scope,df,keys in scopes:
        for regime,reg in df.groupby('regime'):
            for feature in FEATURES:
                for outcome in OUTCOMES:
                    selected=[]
                    for group,b in reg.groupby(keys[0] if len(keys)==1 else keys,dropna=False):
                        if not isinstance(group,tuple):group=(group,)
                        value,n,status=rho(b,feature,outcome)
                        row=dict(scope=scope,regime=regime,feature=feature,outcome=outcome,
                                 **dict(zip(keys,group)),n=n,rho=value,status=status)
                        blocks.append(row)
                        if value is not None:selected.append(row)
                    if not selected:
                        totals.append(dict(scope=scope,regime=regime,feature=feature,outcome=outcome,
                                           informative_blocks=0,paths_in_informative_blocks=0));continue
                    arr=np.array([r['rho'] for r in selected]);tasks=sorted({r['task'] for r in selected})
                    leave=[np.mean([r['rho'] for r in selected if r['task']!=t]) for t in tasks
                           if any(r['task']!=t for r in selected)]
                    totals.append(dict(scope=scope,regime=regime,feature=feature,outcome=outcome,
                        informative_blocks=len(selected),paths_in_informative_blocks=sum(r['n'] for r in selected),
                        mean_rho=arr.mean(),median_rho=np.median(arr),
                        positive_blocks=int((arr>1e-9).sum()),negative_blocks=int((arr< -1e-9).sum()),
                        zero_blocks=int((np.abs(arr)<=1e-9).sum()),
                        leave_one_task_out_min=min(leave) if leave else None,
                        leave_one_task_out_max=max(leave) if leave else None))
    save('association_blocks.csv',blocks);save('association_summary.csv',totals)
    return pd.DataFrame(totals)


def summary(df,keys):
    result=[]
    for group,b in df.groupby(keys[0] if len(keys)==1 else keys,dropna=False,sort=True):
        if not isinstance(group,tuple):group=(group,)
        row=dict(zip(keys,group));row.update(n=len(b),unique_sequences=b.physical_sequence_sha256.nunique(),
            SearchAUC_mean=b.SearchAUC.mean(),SearchAUC_SD=b.SearchAUC.std(ddof=1),
            SearchAUC_min=b.SearchAUC.min(),SearchAUC_max=b.SearchAUC.max(),
            final_best_mean=b.final_best.mean(),final_best_min=b.final_best.min(),final_best_max=b.final_best.max(),
            mean_AUC_normalized=b.AUC_normalized.mean(),mean_final_normalized=b.final_normalized.mean(),
            reached95=int(b.first95_total.notna().sum()),censored95=int(b.first95_total.isna().sum()),
            median_first95_among_hits=b.first95_total.dropna().median() if b.first95_total.notna().any() else None,
            mean_best4_normalized=b.best4_normalized.mean(),
            mean_response_SD4_normalized=b.response_SD4_normalized.mean(),mean_Gower8=b.Gower8.mean(),
            mean_GP_SD4_normalized=b.GP_mean_SD_normalized_4.mean(),
            mean_GP_coverage95_4=b.GP_coverage95_4.mean(),
            total_scored_KF_feedback=b.KF_updates_through20.sum(),
            total_KF_weight_changes=(b.KF_increases_through20+b.KF_decreases_through20).sum(),
            paths_with_scored_KF_feedback=int((b.KF_updates_through20>0).sum()),
            total_editor_calls=b.editor_call_through20.sum(),
            missed_reference_records_first8=b.missed_reference_record_through8.sum())
        for tier in ('at_least_95pct_max','80_to_95pct_max','below_80pct_max'):
            row[tier]=int((b.terminal_tier==tier).sum())
        result.append(row)
    return pd.DataFrame(result)


def paired(main):
    bo=main[main.method=='bo'].set_index(['task','regime','seed'])
    records=[]
    for _,r in main[main.method!='bo'].iterrows():
        b=bo.loc[(r.task,r.regime,r.seed)]
        if r.regime=='warm':
            prefixes=[]
            for item in (r,b):
                p=ROOT/item.source_path;raw=p.read_bytes()
                assert hashlib.sha256(raw).hexdigest()==item.source_sha256
                READS[p.relative_to(ROOT).as_posix()]=item.source_sha256
                history=json.loads(raw.decode('utf-8-sig'))['history'][:4]
                prefixes.append([(h['candidate_id'],h['raw_value']) for h in history])
            assert prefixes[0]==prefixes[1],(r.path_id,b.path_id)
        records.append(dict(path_id=r.path_id,BO_path_id=b.path_id,task=r.task,regime=r.regime,seed=r.seed,
            method=r.method,backend=r.backend,SearchAUC=r.SearchAUC,BO_SearchAUC=b.SearchAUC,
            delta_SearchAUC=r.SearchAUC-b.SearchAUC,delta_AUC_normalized=r.AUC_normalized-b.AUC_normalized,
            final_best=r.final_best,BO_final_best=b.final_best,
            delta_final_normalized=r.final_normalized-b.final_normalized,
            AUC_result='win' if r.AUC_normalized>b.AUC_normalized+1e-9 else
                       'loss' if r.AUC_normalized<b.AUC_normalized-1e-9 else 'tie',
            final_result='win' if r.final_normalized>b.final_normalized+1e-9 else
                         'loss' if r.final_normalized<b.final_normalized-1e-9 else 'tie',
            initial_best_difference=float(r.initial_best)-float(b.initial_best) if r.regime=='warm' else None,
            **{'delta_'+f:r[f]-b[f] for f in FEATURES if f in r}))
    pairs=pd.DataFrame(records);save('main_matched_BO_pairs.csv',pairs)
    totals=[]
    for keys,g in pairs.groupby(['regime','method','backend']):
        totals.append(dict(zip(('regime','method','backend'),keys),n=len(g),
            AUC_wins=int((g.AUC_result=='win').sum()),AUC_ties=int((g.AUC_result=='tie').sum()),
            AUC_losses=int((g.AUC_result=='loss').sum()),
            final_wins=int((g.final_result=='win').sum()),final_ties=int((g.final_result=='tie').sum()),
            final_losses=int((g.final_result=='loss').sum()),
            mean_delta_AUC_normalized=g.delta_AUC_normalized.mean(),
            median_delta_AUC_normalized=g.delta_AUC_normalized.median()))
    save('main_matched_BO_summary.csv',totals)
    assert (pairs[pairs.regime=='warm'].initial_best_difference.abs()<1e-9).all()
    return pairs,pd.DataFrame(totals)


def continuation_stats(df):
    data=df[df.family=='late_prefix'];groups=[]
    for (task,seed),b in data.groupby(['task','seed']):
        f=b[b.method=='full'].continuation_SearchAUC.mean()
        for arm,g in b.groupby('method'):
            mean=g.continuation_SearchAUC.mean();delta=f-mean
            groups.append(dict(task=task,seed=seed,method=arm,n=len(g),
                continuation_SearchAUC_mean=mean,full_continuation_mean=f,full_minus_arm=delta,
                full_vs_arm='win' if delta>1e-7 else 'loss' if delta< -1e-7 else 'tie'))
    save('fixed_prefix_continuations.csv',groups)
    return pd.DataFrame(groups)


def mdtable(frame,cols):
    head='| '+' | '.join(cols)+' |\n| '+' | '.join('---' for _ in cols)+' |\n'
    for _,r in frame.iterrows():
        values=[]
        for c in cols:
            v=r[c]
            values.append('NA' if pd.isna(v) else f'{v:.3f}' if isinstance(v,(float,np.floating)) else str(v))
        head+='| '+' | '.join(values)+' |\n'
    return head


def analysis_record(df,assocs,pairs,comparisons,cont):
    counts=df.groupby(['family','regime']).size().rename('paths').reset_index()
    main=df[df.primary];fresh=main[main.backend!='historical_API'];v13=main[main.method=='v13']
    findings=['## Population findings\n',
      'The whole population does not support a universal explanation that poor paths simply have lower '
      'geometric diversity. Early response spread, GP scale/calibration and the ability to exploit useful '
      'candidates are distinct factors. The strongest association with SearchAUC is early incumbent quality; '
      'this partly follows from the definition of SearchAUC and is not a causal estimate.\n']
    for budget in ('cold','warm'):
        def val(feature,field='mean_rho'):
            return assocs[(assocs.scope=='V13_only')&(assocs.regime==budget)&
                          (assocs.outcome=='AUC_normalized')&(assocs.feature==feature)].iloc[0][field]
        findings.append(f"- V13 {budget}: early-best4 rho={val('best4_normalized'):.3f}; "
            f"response-SD4 rho={val('response_SD4_normalized'):.3f}; "
            f"Gower8 rho={val('Gower8'):.3f}; "
            f"GP-SD4 rho={val('GP_mean_SD_normalized_4'):.3f}; "
            f"missed-GP-records-through8 rho={val('missed_reference_record_through8'):.3f}; "
            f"scored-KF-feedback-through8 rho={val('KF_updates_through8'):.3f}.\n")
    findings += ['The scalar category-correlation association changes direction across tasks and outcomes. '
      'For V13 SearchAUC it is approximately zero in cold paths and -0.100 in warm paths; the corresponding '
      'terminal-value association is positive but has only two informative Suzuki task blocks per budget. '
      'This is not population evidence that category correlation alone determines success or failure.\n',
      'In the four Suzuki tasks, V13 has category correlation above 0.99 in 5/16 snapshots after four '
      'observations for both budgets. BO also has 5/16. At eight observations V13 has 1/16 cold and 5/16 '
      'warm, compared with BO 1/16 and 2/16. Both methods have zero such cases at 12 and 19 observations. '
      'Early category indistinguishability is real in some paths and is not unique to LLM intervention.\n']
    text=['# All contemporaneous trajectories: frozen V13 population audit\n',
      'All 398 distinct archived state files are listed; the principal matched comparison contains 227 paths. '
      'There are 328 complete from-start paths, 66 fixed-prefix continuations and four frozen-response replays. '
      'The 570 registry rows contain 172 repeated state-file references; these are not additional runs. '
      'Paths with identical selected sequences can still be separate method runs, so sequence identity is also reported.\n',
      mdtable(counts,['family','regime','paths']),
      *findings,
      '## Main matched comparison\n',
      'Warm = four shared initial observations plus 16 optimization queries; cold = zero initial observations '
      'plus 20 queries. API and fresh-subagent responses are separate groups. The BO files are shared references, '
      'not duplicated observations. Objective units differ among tasks, so cross-task raw means are not calculated.\n',
      mdtable(comparisons,['regime','method','backend','n','AUC_wins','AUC_ties','AUC_losses',
                           'final_wins','final_ties','final_losses','mean_delta_AUC_normalized']),
      '![All 227 main-comparison paths](all_main_paths.png)\n',
      '## What characterizes high- and low-performing paths across the whole population?\n',
      'The following associations use every available path in each stratum, not selected extremes. '
      'For each task and budget, Spearman ranks use average ties; correlations are averaged equally over informative '
      'task blocks. Constant features/outcomes and blocks with fewer than four paths are explicitly retained as '
      'uninformative in association_blocks.csv. No p-values or causal effects are claimed.\n',
      'The main fresh/no-LLM population contains 216 paths (120 warm and 96 cold). V13-only sensitivity '
      'contains 64 paths (four seeds for each of eight tasks and two budgets). Additional analyses stratify by method; '
      'the 11 main API paths are descriptive only.\n']
    shortlist=['best4_normalized','response_SD4_normalized','Gower8','category_contrast_pairs8',
               'GP_mean_SD_normalized_4','GP_coverage95_4','GP_mixed_category_correlation_4',
               'missed_reference_record_through8','KF_updates_through8','editor_call_through8']
    for scope in ('main_fresh','V13_only','main_method_separated'):
        a=assocs[(assocs.scope==scope)&(assocs.outcome=='AUC_normalized')&assocs.feature.isin(shortlist)]
        text+=['### '+scope+'\n',mdtable(a,['regime','feature','informative_blocks','mean_rho',
          'positive_blocks','negative_blocks','leave_one_task_out_min','leave_one_task_out_max'])]
    text+=['## Final-value strata and early path features\n',
      'Strata include all outcomes: final value at least 95% of the pool maximum, 80% to below 95%, '
      'and below 80%. These are retrospective descriptions, not prospective predictors.\n']
    tiers=summary(main,['regime','method','backend','terminal_tier'])
    save('main_terminal_tier_profiles.csv',tiers)
    text += [mdtable(tiers[tiers.method=='v13'],['regime','terminal_tier','n','mean_best4_normalized',
       'mean_response_SD4_normalized','mean_Gower8','mean_GP_SD4_normalized','mean_GP_coverage95_4',
       'total_scored_KF_feedback','paths_with_scored_KF_feedback']),
       '## GP category correlation and apparent certainty\n',
       'For the four Suzuki tasks only, category correlation is the frozen Mixed-GP correlation between '
       'different categorical values at identical continuous settings. Correlation above 0.99 means that '
       'the fitted kernel barely distinguishes the categories. It is a fitted-model diagnostic, not proof of '
       'a software error. Other tasks use different kernels and are not pooled under this scalar.\n']
    cat=[]
    for keys,g in main[main.task.isin(['c1','c2','c3','c4'])].groupby(['regime','method','backend']):
        for n in (4,8,12,19):
            x=g[f'GP_mixed_category_correlation_{n}'];cat.append(dict(zip(('regime','method','backend'),keys),
                observed=n,n=len(g),correlation_gt099=int((x>.99).sum()),
                mean_coverage95=g[f'GP_coverage95_{n}'].mean(),mean_normalized_SD=g[f'GP_mean_SD_normalized_{n}'].mean()))
    cats=pd.DataFrame(cat);save('main_suzuki_kernel_summary.csv',cats)
    text += [mdtable(cats[cats.method=='v13'],['regime','observed','n','correlation_gt099','mean_coverage95',
                                             'mean_normalized_SD']),
       '## Archived decision opportunities and feedback\n',
       'A missed-reference record means that the selected point did not improve the current incumbent while '
       'that path\'s own GP reference would have improved it. This is an oracle-labelled local diagnostic after '
       'the fact, not the outcome of a counterfactual full BO trajectory. Menu-best values are likewise used '
       'only for audit. No labels were exposed to any historical decision.\n',
       'Scored KF feedback counts structured pre-experiment prediction evaluations. A score need not change '
       'a weight: fixed-weight ablations may still log evaluations. Increases and decreases count actual '
       'recorded reliability/weight changes. Editor calls count available memory/refresh commit fields; '
       'legacy V8 uses a different schema and its zero structured-KF count is not evidence that it lacked '
       'other calibration. Continuation totals contain the shared first-ten-observation prefix and are '
       'not independent opportunities.\n',
       mdtable(summary(v13,['regime']),['regime','n','total_scored_KF_feedback','total_KF_weight_changes',
         'paths_with_scored_KF_feedback','total_editor_calls','missed_reference_records_first8']),
       'Across all 64 V13 paths, there are 338 scored feedback events and 310 recorded weight/reliability '
       'changes. Six of 32 cold paths have no scored KF feedback. Counts show execution, not stable ranking '
       'benefit: the V13-only early feedback-count correlation with SearchAUC is -0.097 cold and -0.109 warm. '
       'The mixed-method fresh population instead gives +0.117 cold and +0.035 warm, showing why pooling '
       'module presence with method performance is misleading.\n',
       '## Fixed-prefix continuations and development extensions\n',
       'The 66 continuation files form six shared prefixes (Case 4, Perera and HPLC; seeds 0 and 3). '
       'The next table averages the two response replicates for each LLM arm; BO has one branch. '
       'Continuation SearchAUC averages the final ten cumulative-best observations, carrying the prefix incumbent. '
       'These branches do not enlarge the from-start main comparison.\n',
       mdtable(cont,['task','seed','method','n','continuation_SearchAUC_mean','full_minus_arm']),
       'All 52 historical predecessor API paths, 12 V12 ablations, 12 targeted V13 ablations, six selected V14 '
       'paths, ten programmatic-KF paths and nine Direct Arylation holdout paths remain in all_path_profiles.csv '
       'and family_method_task_summary.csv. Their selected tasks and development history prevent treating '
       'them as a balanced confirmatory sample. Four frozen-response replays are separately labelled.\n',
       '## Interpretation limits\n',
       'Early best value correlates mechanically with SearchAUC because the incumbent is retained. The post-eight '
       'cumulative-best metric also carries the earlier incumbent. Remaining gain has less headroom when early '
       'performance is already high. Larger observed response spread can increase learned GP scale and uncertainty; '
       'larger SD alone is not improved calibration. Geometric diversity, categorical contrasts, empirical coverage '
       'and missed opportunities must therefore be considered separately.\n',
       'KF/editor counts are endogenous: poor progress can trigger more updates. Counts do not establish module '
       'benefit, and these retrospective trajectories do not isolate the causal effect of diversity, KF or refresh. '
       'Method-separated and V13-only rank tables are sensitivity checks, not independent replications. '
       'There are only eight original tasks and four seeds per primary method/budget; initialization and response '
       'variation remain important.\n',
       '## Reproducibility and files\n',
       'SearchAUC = mean of the cumulative maximum at each actual optimization query: 16 terms for warm paths '
       'and 20 for cold paths. Final values and labels were checked against frozen oracle arrays. '
       'All 7,180 archived commits are hash-checked and all 1,592 requested GP snapshots are available; '
       '32 legacy V8 snapshots lack a commit-embedded GP hash but have input-file hashes and prefix-mask checks. '
       'No GP was fitted, no LLM called and no new experiment executed.\n',
       '- path_catalogue.csv: every distinct state and source hash.\n'
       '- registry_references.csv: all 570 registry references and duplicate links.\n'
       '- all_observed_sequences.csv: 7,960 stored observation rows, including shared prefixes/replays.\n'
       '- all_path_profiles.csv: all 398 paths and path features.\n'
       '- all_decision_features.csv: all 7,180 decisions.\n'
       '- all_GP_prefixes.csv: 1,592 archived GP snapshots at 4, 8, 12 and 19 observations.\n'
       '- main_task_method_summary.csv / main_matched_BO_pairs.csv: task-specific results and matched controls.\n'
       '- association_blocks.csv / association_summary.csv: every predefined association and missing/constant block.\n'
       '- family_method_task_summary.csv: all development and continuation families, separated.\n'
       '- census_audit.json / profile_audit.json / statistics_audit.json: custody and calculation checks.\n']
    (OUT/'analysis.md').write_text('\n'.join(text)+'\n',encoding='utf-8')


def main():
    df=read('all_path_profiles.csv');main=df[df.primary]
    assert len(df)==398 and len(main)==227
    assert df.path_id.is_unique
    fresh=main[main.backend!='historical_API'];v13=main[main.method=='v13']
    assert len(fresh)==216 and len(v13)==64
    scopes=[('main_fresh',fresh,['task']),('V13_only',v13,['task']),
            ('main_method_separated',fresh,['task','method','backend'])]
    assocs=associations(scopes)
    save('main_task_method_summary.csv',summary(main,['regime','task','method','backend']))
    save('family_method_task_summary.csv',summary(df,['family','regime','task','method','backend']))
    pairs,comparisons=paired(main);cont=continuation_stats(df)
    analysis_record(df,assocs,pairs,comparisons,cont)
    # Independent reconstruction from the long observation table, without using
    # the catalogue's stored trajectory metrics.
    obs=read('all_observed_sequences.csv')
    for _,r in df.iterrows():
        b=obs[obs.path_id==r.path_id].sort_values('total_observations')
        values=b.objective.to_numpy();cum=np.maximum.accumulate(values)
        start=0 if r.regime=='continuation10+10' else int(r.initial)
        assert np.isclose(cum[start:].mean(),r.SearchAUC,rtol=1e-11,atol=1e-8),r.path_id
        assert np.isclose(cum[-1],r.final_best,rtol=1e-11,atol=1e-8),r.path_id
    for name,h in READS.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h
    audit=dict(status='PASS_POPULATION_STATISTICS',distinct_states=len(df),main_paths=len(main),
       fresh_or_no_LLM_main_paths=len(fresh),V13_paths=len(v13),matched_main_BO_pairs=len(pairs),
       metric_reconstructions=len(df),association_hypotheses=len(assocs),
       new_GP_fits=0,new_LLM_calls=0,new_experiments=0)
    (OUT/'statistics_audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    READS[Path(__file__).resolve().relative_to(ROOT).as_posix()]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT/'statistics_source_hashes.json').write_text(json.dumps(READS,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(audit,indent=2))
    print(comparisons.to_string(index=False))


if __name__=='__main__':main()
