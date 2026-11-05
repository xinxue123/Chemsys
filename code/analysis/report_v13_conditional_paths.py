"""Inspectible readout of conditional, initialization-controlled path analyses."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/analysis'))
import analyze_v13_conditional_paths as analysis
OUT=analysis.OUT
READS={}


def read(name):
    p=OUT/name;READS[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    return pd.read_csv(p,encoding='utf-8-sig')


def save(name,records):
    (records if isinstance(records,pd.DataFrame) else pd.DataFrame(records)).to_csv(
        OUT/name,index=False,encoding='utf-8-sig',lineterminator='\n')


def table(d,cols):
    out='| '+' | '.join(cols)+' |\n| '+' | '.join('---' for _ in cols)+' |\n'
    for _,r in d.iterrows():
        out+='| '+' | '.join('NA' if pd.isna(r[c]) else f'{r[c]:.3f}' if isinstance(r[c],(float,np.floating)) else str(r[c]) for c in cols)+' |\n'
    return out


def module_pairs(d):
    main=d[d.primary];states={}
    def history(r):
        if r.path_id not in states:
            p=ROOT/r.source_path;raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest();assert h==r.source_sha256
            READS[p.relative_to(ROOT).as_posix()]=h
            states[r.path_id]=json.loads(raw.decode('utf-8-sig'))['history']
        return states[r.path_id]
    records=[]
    def add(family,variant,control,scope):
        ha,hb=history(variant),history(control)
        shared=next((i for i,(a,b) in enumerate(zip(ha,hb)) if a['candidate_id']!=b['candidate_id']),20)
        records.append(dict(family=family,task=variant.task,seed=variant.seed,
            variant_method=variant.method,control_method=control.method,variant_path_id=variant.path_id,
            control_path_id=control.path_id,common_initial_observations=shared,
            same_four_prefix=variant.prefix_id==control.prefix_id,
            both_below80=variant.start_fraction_max<.8 and control.start_fraction_max<.8,
            delta_AUC_normalized=(variant.analysis_AUC-control.analysis_AUC)/variant.pool_range,
            delta_final_normalized=variant.final_normalized-control.final_normalized,interpretation=scope))
    for family,method in [('v12_ablation','v12_subagent'),('v13_targeted','v13')]:
        for _,r in d[d.family==family].iterrows():
            target=main[(main.task==r.task)&(main.seed==r.seed)&(main.method==method)&(main.regime==r.regime)]
            assert len(target)==1
            add(family,r,target.iloc[0],'whole_trajectory_conditional_ablation')
    for family in ('programmatic_KF','programmatic_KF_confirmation'):
        part=d[d.family==family]
        for _,r in part[part.method=='dynamic'].iterrows():
            fixed=part[(part.task==r.task)&(part.seed==r.seed)&(part.method=='fixed')]
            assert len(fixed)==1
            add(family,r,fixed.iloc[0],'dynamic_minus_fixed_programmatic_weights')
    save('registered_module_comparisons.csv',records)
    late=[]
    for (task,seed),b in d[d.family=='late_prefix'].groupby(['task','seed']):
        assert b.prefix_id.nunique()==1 and b.GP_sha256_start.nunique()==1
        full=b[b.method=='full'];reference=full.analysis_AUC.mean();span=full.pool_range.iloc[0]
        for arm,g in b[b.method!='full'].groupby('method'):
            delta=(reference-g.analysis_AUC.mean())/span
            late.append(dict(task=task,seed=seed,start_tier=b.start_tier.iloc[0],method=arm,n=len(g),
                full_minus_arm_AUC_normalized=delta,
                full_minus_arm_final_normalized=full.final_normalized.mean()-g.final_normalized.mean(),
                result='win' if delta>1e-9 else 'loss' if delta< -1e-9 else 'tie'))
    save('same_ten_prefix_module_comparisons.csv',late)
    return pd.DataFrame(records),pd.DataFrame(late)


def plot(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    pairs=[('GP_mean_truth_rank_start','GP mean ranking accuracy at start'),
           ('Gower4','Geometric dispersion at start'),
           ('GP_mean_SD_normalized_start','GP mean uncertainty at start'),
           ('menu90_rate_5to8','High-value menu offerings, steps 5-8'),
           ('changed_reference_rate_5to8','Overrides of GP reference, steps 5-8'),
           ('KF_weight_changes_5to8','KF weight changes, steps 5-8')]
    targets=[('native_gain_AUC_normalized','Subsequent SearchAUC gain','#2466a8','o'),
             ('final_normalized','Final best value','#c8751c','^')]
    b=a[(a.cutoff==.8)&(a.adjustment=='start_and_method_adjusted')]
    fig,axes=plt.subplots(1,2,figsize=(13.5,5.8),sharex=True,sharey=True)
    plotted=[]
    for ax,budget in zip(axes,['cold','warm']):
        for j,(feature,label) in enumerate(pairs):
            for i,(outcome,name,color,marker) in enumerate(targets):
                row=b[(b.regime==budget)&(b.feature==feature)&(b.outcome==outcome)].iloc[0]
                if row.informative_tasks:
                    ax.scatter(row.mean_rho,j+(i-.5)*.18,c=color,marker=marker,s=54,zorder=3)
                    plotted.append(row.to_dict())
                elif i==0:
                    ax.text(.02,j,'Not estimable',fontsize=10,color='#7b848d',va='center')
        ax.axvline(0,color='#8b9299',lw=.8);ax.grid(axis='x',color='#e5e8ec',lw=.7)
        ax.set_xlim(-1,1);ax.set_ylim(len(pairs)-.5,-.5)
        ax.set_yticks(range(len(pairs)),[v for _,v in pairs],fontsize=10)
        ax.set_title('0 + 20' if budget=='cold' else '4 + 16',fontsize=13,pad=10)
        ax.set_xlabel('Mean partial rank correlation within task')
        for side in ('right','top'):ax.spines[side].set_visible(False)
    fig.suptitle('Path associations after excluding high starts',fontsize=15,y=.985)
    handles=[Line2D([],[],marker=m,color=c,linestyle='',label=n,markersize=7) for _,n,c,m in targets]
    fig.legend(handles=handles,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.62,.045))
    fig.text(.61,.018,'Start best <80% of pool maximum. Warm comparisons control the exact physical prefix.',
             ha='center',fontsize=10,color='#535d68')
    fig.subplots_adjust(left=.285,right=.985,bottom=.24,top=.87,wspace=.1)
    target=OUT/'conditional_factors.png';fig.savefig(target,dpi=180,facecolor='white');plt.close(fig)
    save('conditional_factors_figure_data.csv',plotted)


def main():
    d=read('conditional_path_profiles.csv');a=read('conditional_association_summary.csv')
    p=read('all_main_BO_pairs_decomposed.csv');pairs=read('matched_BO_cohort_summary.csv')
    modules,late=module_pairs(d)
    main=d[d.primary];fresh=main[main.backend!='historical_API']
    counts=main.groupby(['regime','start_tier']).size().rename('paths').reset_index()
    save('main_start_cohort_counts.csv',counts)
    # Every low-start V13/BO pair, not best/worst examples.
    v=p[(p.method=='v13')&(p.regime=='warm')&(p.start_fraction_max<.8)&(p.BO_start_fraction_max<.8)]
    task_summary=[]
    for task,b in v.groupby('task'):
        task_summary.append(dict(task=task,n=len(b),AUC_wins=int((b.AUC_result=='win').sum()),
            AUC_losses=int((b.AUC_result=='loss').sum()),mean_delta_AUC=b.delta_AUC_normalized.mean(),
            final_gain_component=b.final_gain_effect_normalized.mean(),timing_component=b.timing_effect_normalized.mean(),
            mean_delta_final=b.delta_final_normalized.mean()))
    tasks=pd.DataFrame(task_summary);save('all_low_start_V13_warm_pairs_by_task.csv',tasks)
    selected=a[(a.cutoff==.8)&(a.adjustment=='start_and_method_adjusted')]
    initial_features=['GP_mean_truth_rank_start','GP_mean_SD_normalized_start','Gower4']
    behavioral=['menu90_rate_5to8','menu_regret_norm_5to8','changed_reference_rate_5to8',
                'KF_weight_changes_5to8','editor_calls_5to8']
    chosen=selected[selected.feature.isin(initial_features+behavioral)&
                    selected.outcome.isin(['native_gain_AUC_normalized','final_normalized'])]
    save('headline_conditional_associations.csv',chosen)
    sensitivity=a[(a.adjustment=='start_and_method_adjusted')&
        a.feature.isin(['menu90_rate_5to8','GP_mean_truth_rank_start','Gower4','KF_weight_changes_5to8'])&
        a.outcome.isin(['native_gain_AUC_normalized','final_normalized'])]
    save('threshold_sensitivity.csv',sensitivity)
    low_late=late[late.start_tier!='high_80plus']
    module_late=low_late.groupby(['method','result']).size().rename('prefix_groups').reset_index()
    save('low_start_late_module_summary.csv',module_late)
    plot(a)
    text=['# What drives path divergence after excluding high starts?\n',
      'The high-start paths are now a separate cohort. Among initially lower paths, the evidence points '
      'to different factors for search speed and terminal quality. It does not support a universal rule '
      'that larger uncertainty, larger geometric dispersion, more KF updates or more LLM overrides improve both.\n',
      '## Population and separation of high starts\n',
      f'All {len(d)} catalogue states remain visible. The227 main paths contain '
      f'{int((main.start_fraction_max<.8).sum())} lower-start paths and '
      f'{int((main.start_fraction_max>=.8).sum())} high-start paths. '
      f'The primary fresh/no-LLM lower-start population contains {len(fresh[fresh.start_fraction_max<.8])} paths. '
      'A high start means best of observations1--4 is at least80% of the fixed pool maximum. '
      'For warm runs these are the actual four initial observations; for cold runs they are algorithm-selected '
      'startup observations. Conditioning on cold startup is retrospective and cannot establish a causal method ranking. '
      'For fixed-prefix continuations, the start is their ten-observation prefix.\n',
      table(counts,['regime','start_tier','paths']),
      'Cutoffs50%,65%,80% and90% were specified before these conditional associations were computed. '
      'First-point-only versus first-four-point exclusion counts are also supplied. API runs, selected development '
      'extensions, reused prefixes and frozen replays are not pooled as independent fresh primary trajectories.\n',
      '## Remove the built-in initial-incumbent advantage without changing SearchAUC\n',
      'For a four-observation landmark, let G be the final improvement beyond the landmark incumbent. '
      'Let E be the normalized timing-weighted mean of the later record improvements. Then exactly\n',
      '$$\\mathrm{SearchAUC}=C_{\\mathrm{prefix}}+\\frac{16}{T}G E.$$\n',
      'T is the original number of scored queries (16 warm,20 cold). '
      'C_prefix is the SearchAUC the recorded prefix would obtain if its incumbent never improved afterward. '
      'For warm paths C_prefix is exactly the initial best. E=1 means all later improvement is achieved '
      'at the first remaining query; E=1/16 means it occurs only at the final query; E=0 is the no-improvement convention. '
      'The identity was checked for all398 states, using ten remaining queries for registered continuations. '
      'This decomposition explains arithmetic; it does not by itself identify the decision mechanism.\n',
      '## Identical initial observations and GP: all lower-start warm V13 versus BO pairs\n',
      'All99 main warm method-versus-BO pairs have identical initial observations and archived initial GP hashes. '
      'Restricting V13 and BO to starts below80% leaves24 exact-prefix pairs, with12 SearchAUC wins and12 losses. '
      'The normalized mean difference is+0.026137, composed of+0.004233 from the terminal-gain term and '
      '+0.021904 from the timing term. Thus approximately84% of the mean net advantage is the arithmetic '
      'timing component; this is not84% of paths and is not a causal module attribution. '
      'The mean terminal-best difference is only+0.007319 of pool range.\n',
      table(tasks,['task','n','AUC_wins','AUC_losses','mean_delta_AUC','final_gain_component',
                   'timing_component','mean_delta_final']),
      'Cold V13/BO comparisons with both starts below80% contain20 pairs, but their startup incumbents still '
      'differ. Their mean SearchAUC difference+0.078662 includes+0.054712 from the startup-prefix baseline, '
      'so simply removing very high starts does not remove all startup confounding. Requiring starts within '
      '5% of the pool maximum leaves only seven pairs: five wins and two losses, with mean normalized '
      'terminal-best difference+0.103071. These are similar-value starts, not identical observations or GPs.\n',
      table(pairs[(pairs.cutoff==.8)&(pairs.method=='v13')],['cohort','regime','n','AUC_wins','AUC_ties',
            'AUC_losses','mean_delta_AUC_normalized','mean_delta_final_normalized']),
      '## Factors associated with later improvement and terminal quality\n',
      'The following are partial Spearman associations within tasks, controlling starting incumbent and method. '
      'Warm behavioral comparisons additionally control the exact physical prefix. Task means are equally weighted; '
      'constant and undersized blocks remain listed in the block file. No significance or causal dominance is claimed. '
      'At least six distinct starting prefixes are required for starting-GP/design associations. Warm runs have '
      'only four physical starts per task, so their repeated method rows do not provide estimable warm starting-feature effects.\n',
      table(chosen,['regime','feature','outcome','informative_tasks','paths_used','mean_rho','positive_tasks','negative_tasks']),
      'The most consistent early behavioral association is whether the offered menu contains genuinely high-valued '
      'candidates (at least90% of pool maximum): its partial association with post-start SearchAUC gain is '
      '+0.425 cold (six positive tasks of six) and+0.353 warm (four positive, two negative). '
      'For terminal quality it is+0.391 cold (five of five positive) and+0.352 warm (four of five positive). '
      'This is an oracle-labelled retrospective opportunity diagnostic; menu quality was not known to the historical policy.\n',
      'Among cold low/middle starts, the accuracy of the GP mean ranking over unqueried candidates is associated '
      'with terminal quality (+0.633 across four informative tasks, all positive), while mean GP uncertainty '
      'is not consistently favorable (-0.086 across those four tasks). Geometric dispersion has a moderate '
      'terminal-quality association (+0.309), but essentially no association with post-start SearchAUC gain '
      '(-0.054). Larger dispersion can therefore accompany better eventual coverage without faster discovery.\n',
      'The frequency of overriding the GP reference has opposite associations in the two budgets: '
      '-0.194 cold versus+0.237 warm for SearchAUC gain. More or fewer overrides is not a universal policy rule. '
      'Early menu regret can also accompany useful exploration, so selecting the menu\'s best immediate true '
      'value is not proved to maximize the full trajectory.\n',
      '![Conditional path factors](conditional_factors.png)\n',
      '## Does an early association predict improvement later, after observation8?\n',
      'A forward check uses only observations9--20 for the outcome and controls incumbent8. '
      'The remaining-headroom diagnostic avoids carrying incumbent8 directly into its numerator. '
      'For menu high-value offerings at5--8, the association is+0.319 cold but-0.159 warm. '
      'For GP mean-ranking accuracy at8, it is only+0.069 cold and+0.040 warm. Thus the main opportunity '
      'association does not establish a stable universal driver of late improvement; it is strongest for '
      'the full post-start discovery process.\n',
      table(selected[(selected.outcome=='post8_headroom_gain_AUC')&selected.feature.isin(
        ['menu90_rate_5to8','KF_weight_changes_5to8','GP_mean_truth_rank_8','GP_RMSE_normalized_8'])],
        ['regime','feature','informative_tasks','mean_rho','positive_tasks','negative_tasks']),
      '## KF and refresh: execution counts do not isolate benefit\n',
      'The partial association of KF weight-change count at5--8 with SearchAUC gain is-0.177 cold and-0.227 warm; '
      'with terminal quality it is-0.091 and-0.333. These are not causal harms: feedback is selective and difficult '
      'paths may generate different knowledge-use and editing opportunities. At this window most editor-call '
      'variation disappears after method adjustment, so there is insufficient independent variation to estimate '
      'an editor-count effect. A positive pooled count association should not be presented as refresh benefit.\n',
      'Registered whole-trajectory ablations and programmatic fixed/dynamic-weight comparisons are listed below, '
      'with their actual common-prefix lengths. A changed initial prefix is explicitly not treated as an '
      'identical-start module experiment.\n',
      table(modules,['family','task','seed','variant_method','control_method','common_initial_observations',
                     'both_below80','delta_AUC_normalized','delta_final_normalized']),
      'The low/middle ten-observation continuation prefixes are HPLC seeds0 and3; Case4 and Perera continuation '
      'prefixes are high-start and are separate. All continuation arms share the same actual ten observations '
      'and initial GP snapshot. The following compares the two full-response replicates\' mean to each arm\'s '
      'mean; BO has one branch.\n',
      table(low_late,['task','seed','method','n','full_minus_arm_AUC_normalized','full_minus_arm_final_normalized','result']),
      '## Threshold sensitivity and high-start cohort\n',
      table(sensitivity[sensitivity.feature=='menu90_rate_5to8'],['cutoff','regime','outcome',
            'informative_tasks','mean_rho','positive_tasks','negative_tasks']),
      'The menu-offering association stays positive under all four exclusion cutoffs for both outcomes and budgets, '
      'though the50% threshold leaves only two to four informative tasks. All other threshold results are in '
      'threshold_sensitivity.csv, including weakened or changing signs. High-start paths are preserved in '
      'high_start_paths_separate.csv and all_start_strata.csv; they are useful for studying ceiling effects '
      'and maintenance, not evidence for low-start discovery.\n',
      '## What can and cannot be concluded\n',
      'The useful working explanation is a sequence: the collected observations affect GP discrimination; '
      'the GP/shortlist must expose good regions; the selection policy must exploit opportunities while '
      'preserving subsequent learning. SearchAUC and terminal best reward different aspects of this sequence. '
      'The archived evidence supports opportunity quality and cold-start ranking accuracy more directly than '
      'a generic uncertainty/diversity or update-count story. It does not identify one universally dominant '
      'causal variable or prove that KF or refreshing improves the next selection.\n',
      'The frozen runs were not randomized interventions on every hypothesized mechanism. Conditioning on cold '
      'startup can alter method composition. Rank residualization is not full causal adjustment. The number '
      'of informative tasks is at most eight and frequently smaller; near-ceiling ties reduce terminal-quality '
      'information. Oracle-labelled features are explanatory audits, not available online control signals.\n',
      '## Reproduction\n',
      'Run code/analysis/analyze_v13_conditional_paths.py followed by '
      'code/analysis/report_v13_conditional_paths.py. All original run states remain frozen. '
      'No new GP fit, LLM call, experiment or counterfactual trajectory was performed. '
      'The conditional_path_profiles.csv file keeps every catalogue path, cohort label and decomposed metric. '
      'Source hashes, exact-prefix checks, original metric reconstruction and all rejected association blocks '
      'are supplied beside this analysis record.\n']
    (OUT/'analysis.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    for path,h in READS.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h,path
    code=Path(__file__);READS[code.relative_to(ROOT).as_posix()]=hashlib.sha256(code.read_bytes()).hexdigest()
    (OUT/'report_source_hashes.json').write_text(json.dumps(READS,indent=2)+'\n',encoding='utf-8')
    print('REPORT_COMPLETE',len(d),'states;',len(modules),'registered module comparisons;',len(late),'late-prefix comparisons')
    print(module_late.to_string(index=False))


if __name__=='__main__':main()
