"""Kernel/condition evidence, controlled response effects and fitting sensitivity."""
from pathlib import Path
import hashlib
import io
import json
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/analysis'))
import profile_v13_kernel_conditions as design
OUT=design.OUT
READS={}


def read(name,base=OUT):
    p=base/name;raw=p.read_bytes();READS[p.relative_to(ROOT).as_posix()]=hashlib.sha256(raw).hexdigest()
    return pd.read_csv(io.BytesIO(raw),encoding='utf-8-sig')


def table(data,columns):
    s='| '+' | '.join(columns)+' |\n| '+' | '.join('---' for _ in columns)+' |\n'
    for _,r in data.iterrows():
        s+='| '+' | '.join('NA' if pd.isna(r[k]) else f'{r[k]:.4f}' if isinstance(r[k],(float,np.floating)) else str(r[k]) for k in columns)+' |\n'
    return s


def unique_transitions(meta,edges,axes):
    hist={}
    for _,r in meta[meta.primary&(meta.start_fraction_max<.8)].iterrows():
        path=ROOT/r.source_path;raw=path.read_bytes();h=hashlib.sha256(raw).hexdigest();assert h==r.source_sha256
        READS[path.relative_to(ROOT).as_posix()]=h;hist[r.path_id]=json.loads(raw.decode('utf-8-sig'))['history']
    keys=[]
    for _,r in edges.iterrows():
        h=hist[r.path_id];n=int(r.measured_total)
        key=design.origin.profile.helpers.digest(dict(task=r.task,
            before=sorted([dict(candidate_id=x['candidate_id'],y=x['y']) for x in h[:n-1]],key=lambda x:x['candidate_id']),
            new=dict(candidate_id=h[n-1]['candidate_id'],y=h[n-1]['y']),
            before_parameters=r.old_parameters_sha256,after_parameters=r.new_parameters_sha256))
        keys.append(key)
    all_edges=edges.assign(transition_key=keys);unique=all_edges.drop_duplicates('transition_key')
    amap=all_edges[['path_id','measured_total','transition_key']]
    all_axes=axes.merge(amap,on=['path_id','measured_total'],validate='many_to_one')
    unique_axes=all_axes.drop_duplicates(['transition_key','axis'])
    design.save('transition_reference_map.csv',amap)
    design.save('unique_kernel_transitions.csv',unique);design.save('unique_axis_condition_responses.csv',unique_axes)
    return unique,unique_axes


def controlled_summary(variants,receipts,registry,edges):
    receipt=receipts.set_index('case_id');records=[];selections=[]
    for case,b in variants.groupby('case_id'):
        assert (b.status=='PASS').all()
        a=b[b.variant=='A_actual_native'].iloc[0];e=b[b.variant=='E_GP_expected'].iloc[0]
        s=b[b.variant=='S_old_data_next_seed'].iloc[0];w=b[b.variant=='W_actual_warm_start'].iloc[0]
        ew=b[b.variant=='EW_GP_expected_warm_start'].iloc[0]
        rho=lambda row:np.array(json.loads(row.rho));old=np.array(json.loads(receipt.loc[case].before_rho))
        source=registry[registry.case_id==case].iloc[0]
        edge=edges[(edges.path_id==source.path_id)&(edges.measured_total==source.measured_total)].iloc[0]
        data_effect=float(np.max(np.abs(rho(a)-rho(e))));warm_effect=float(np.max(np.abs(rho(a)-rho(w))))
        row=dict(case_id=case,task=a.task,path_id=a.path_id,regime=a.regime,method=a.method,
            measured_total=int(a.measured_total),source_path_references=int((registry.case_id==case).sum()),
            conditions=source.conditions,largest_change_axis=source.largest_change_axis,
            measured_result=receipt.loc[case].observed_result,GP_expected_result=receipt.loc[case].GP_expected_result,
            innovation_z=edge.innovation_z,isolated_axes=edge.isolated_axes,
            nominal_isolated_axes=edge.nominal_isolated_axes,
            actual_vs_expected_shape_change=data_effect,actual_vs_warm_shape_change=warm_effect,
            same_old_data_seed_shape_change=float(np.max(np.abs(rho(s)-old))),
            actual_vs_warm_MLL_gap=float(abs(a.MLL-w.MLL)),
            weakly_identified_two_start=bool(warm_effect>.25 and abs(a.MLL-w.MLL)<.01),
            actual_high_MAE=a.high_MAE,expected_high_MAE=e.high_MAE,warm_high_MAE=w.high_MAE,
            expected_warm_high_MAE=ew.high_MAE,
            real_outcome_high_MAE_gain=e.high_MAE-a.high_MAE,
            real_outcome_high_MAE_gain_warm=ew.high_MAE-w.high_MAE,
            minimum_gain_across_starts=min(e.high_MAE-a.high_MAE,ew.high_MAE-w.high_MAE),
            improves_under_both_starts=bool(e.high_MAE>a.high_MAE+1e-6 and ew.high_MAE>w.high_MAE+1e-6),
            worsens_under_both_starts=bool(e.high_MAE<a.high_MAE-1e-6 and ew.high_MAE<w.high_MAE-1e-6),
            outcome_initialization_interaction=float(np.max(np.abs((rho(a)-rho(e))-(rho(w)-rho(ew))))),
            actual_global_rank=a.global_rank,expected_global_rank=e.global_rank,warm_global_rank=w.global_rank,
            expected_warm_global_rank=ew.global_rank,
            real_response_brings_top5=bool(a.global_rank<=5<e.global_rank),
            real_response_loses_top5=bool(e.global_rank<=5<a.global_rank),
            start_changes_top5=bool((a.global_rank<=5)!=(w.global_rank<=5)),
            native_replay_rho_error=a.archived_rho_max_abs_error,
            native_replay_mean_error=a.archived_mu_max_abs_error,
            actual_MLL=a.MLL,warm_MLL=w.MLL)
        records.append(row)
        # This selection uses training objective only; held-out labels evaluate
        # the selection afterward and never select the fitted model.
        picked=w if w.MLL>a.MLL+1e-10 else a
        selections.append(dict(case_id=case,task=a.task,picked=picked.variant,
            high_MAE_gain_vs_native=a.high_MAE-picked.high_MAE,
            native_global_rank=a.global_rank,picked_global_rank=picked.global_rank))
    result=pd.DataFrame(records);design.save('controlled_case_effects.csv',result)
    design.save('higher_MLL_selection_diagnostic.csv',selections)
    return result,pd.DataFrame(selections)


def plots(cases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'P0':'#2466a8','P1':'#b9741b','P2':'#87618d'}
    groups={'P0':['c1','c2','c3','c4'],'P1':['perera','doyle'],'P2':['snar','hplc']}
    labels={'P0':'Reizman Suzuki','P1':'Perera + Doyle','P2':'SnAr + HPLC'}
    markers={'P0':'o','P1':'^','P2':'s'}
    fig,axs=plt.subplots(1,2,figsize=(12,5.5))
    for group,tasks in groups.items():
        b=cases[cases.task.isin(tasks)]
        axs[0].scatter(b.actual_vs_expected_shape_change,b.real_outcome_high_MAE_gain,
            color=colors[group],marker=markers[group],s=35,alpha=.72,label=labels[group])
        axs[1].scatter(b.actual_vs_warm_MLL_gap,b.actual_vs_warm_shape_change,
            color=colors[group],marker=markers[group],s=35,alpha=.72,label=labels[group])
    axs[0].axhline(0,color='#7c858d',lw=.8)
    axs[0].set_xlabel('Kernel-correlation change: real vs GP-expected outcome')
    axs[0].set_ylabel('High-region MAE reduction from real outcome\n(fraction of the task pool range)')
    axs[0].set_title('Real vs GP-expected outcome: native start')
    axs[0].set_xlim(-.025,1.025)
    axs[1].set_xscale('symlog',linthresh=1e-5)
    axs[1].axvline(.01,color='#7c858d',lw=.8,ls='--');axs[1].axhline(.25,color='#7c858d',lw=.8,ls='--')
    axs[1].set_xlabel('Absolute per-observation MLL gap: two starts')
    axs[1].set_ylabel('Maximum kernel-correlation difference: two starts')
    axs[1].set_title('Same data, different optimization starting points')
    axs[1].set_ylim(-.025,1.025)
    for ax in axs:
        ax.grid(color='#e8ebee',lw=.6)
        for side in ('top','right'):ax.spines[side].set_visible(False)
    row=cases[cases.case_id=='kernel_b4d805c15631f473'].iloc[0]
    axs[1].annotate('Case 2, seed 0',xy=(row.actual_vs_warm_MLL_gap,row.actual_vs_warm_shape_change),
        xytext=(.02,.83),textcoords='axes fraction',fontsize=10,arrowprops=dict(arrowstyle='-',color='#59636d'))
    handles,labs=axs[0].get_legend_handles_labels()
    fig.legend(handles,labs,loc='lower center',bbox_to_anchor=(.52,.025),ncol=3,frameon=False)
    fig.suptitle('Controlled kernel-response diagnostics: all 131 unique cases',fontsize=14,y=.985)
    fig.subplots_adjust(left=.085,right=.985,bottom=.22,top=.85,wspace=.28)
    fig.savefig(OUT/'kernel_condition_diagnostics.png',dpi=180,facecolor='white');plt.close(fig)
    design.save('kernel_condition_figure_data.csv',cases)


def readable_observation_catalogue(cases,registry,variants,axes):
    """One inspectible row per measured condition and native kernel axis."""
    pools={task:design.origin.profile.knowledge_gp.load_task(ROOT,full,seed=0)
           for task,full in design.origin.profile.FULL.items() if task!='aryl'}
    raw=(OUT/'controlled_jobs.json').read_bytes()
    READS[(OUT/'controlled_jobs.json').relative_to(ROOT).as_posix()]=hashlib.sha256(raw).hexdigest()
    jobs={j['case_id']:j for j in json.loads(raw.decode('utf-8'))};rows=[]
    for _,case in cases.iterrows():
        ref=registry[registry.case_id==case.case_id].iloc[0];pool=pools[case.task]
        ax=axes[(axes.path_id==ref.path_id)&(axes.measured_total==ref.measured_total)]
        v={r.variant:r for _,r in variants[variants.case_id==case.case_id].iterrows()}
        value=json.loads(ref.conditions)
        observed={h['candidate_id']:h['raw_value'] for h in jobs[case.case_id]['history']}
        for i,(_,a) in enumerate(ax.iterrows()):
            row=dict(case_id=case.case_id,task=case.task,path_id=ref.path_id,measured_total=int(ref.measured_total),
                candidate_id=ref.candidate_id,axis=a.axis,kind=a.kind,new_axis_value=value[a.axis],
                isolated_contrast=bool(a.isolated_contrast),contrast_anchor=a.contrast_anchor,
                anchor_axis_value=pool.points[pool.ix[a.contrast_anchor]][a.axis] if isinstance(a.contrast_anchor,str) else None,
                observed_response=case.measured_result,expected_response=case.GP_expected_result,
                observed_contrast_abs=a.contrast_response_abs,rho_before=a.rho_before,
                anchor_response=observed[a.contrast_anchor] if isinstance(a.contrast_anchor,str) else None,
                signed_response_contrast=case.measured_result-observed[a.contrast_anchor] if isinstance(a.contrast_anchor,str) else None,
                ell1_before=a.ell1_before,ell2_before=a.ell2_before,
                real_outcome_high_MAE_gain=case.real_outcome_high_MAE_gain,
                optimizer_start_shape_disagreement=case.actual_vs_warm_shape_change)
            for short,name in [('A','A_actual_native'),('E','E_GP_expected'),('W','W_actual_warm_start'),('EW','EW_GP_expected_warm_start')]:
                for key in ('rho','ell1','ell2'):row[key+'_'+short]=json.loads(v[name][key])[i]
            rows.append(row)
    design.save('key_observations_by_condition_and_axis.csv',rows)


def main():
    meta=read('conditional_path_profiles.csv',design.origin.SOURCE)
    edges=read('all2475_kernel_transitions.csv');axes=read('all_axis_condition_responses.csv')
    variants=pd.concat([read('controlled_refit_variants.csv'),read('expected_warm_refits.csv')],ignore_index=True)
    assert len(variants)==655
    design.save('factorial_refit_variants.csv',variants)
    receipts=read('case_receipts.csv');registry=read('controlled_case_registry.csv')
    unique,uaxes=unique_transitions(meta,edges,axes)
    cases,selector=controlled_summary(variants,receipts,registry,edges)
    catalog=[]
    for task,g in unique.groupby('task'):
        cat=uaxes[(uaxes.task==task)&(uaxes.kind=='categorical')&(uaxes.delta_rho.abs()>=.25)]
        catalog.append(dict(task=task,unique_transitions=len(g),large_kernel_responses=int((g.kernel_response>=.25).sum()),
            large_nominal_axis_changes=len(cat),without_direct_nominal_contrast=int((~cat.isolated_contrast).sum())))
    catalog=pd.DataFrame(catalog);design.save('kernel_response_by_task.csv',catalog)
    contrasts=uaxes[uaxes.isolated_contrast].copy()
    contrasts['contrast_size']=np.where(contrasts.contrast_response_fraction>=.2,'at_least20pct_range',
        np.where(contrasts.contrast_response_fraction<=.05,'at_most5pct_range','intermediate'))
    grouped=contrasts.groupby(['task','axis','kind','contrast_size']).agg(n=('delta_rho','size'),
        median_delta_rho=('delta_rho','median'),lower_correlation=('delta_rho',lambda z:(z< -.05).sum()),
        higher_correlation=('delta_rho',lambda z:(z>.05).sum())).reset_index()
    design.save('isolated_contrast_kernel_responses.csv',grouped)
    bytask=cases.groupby('task').agg(cases=('case_id','size'),large_real_response_effect=('actual_vs_expected_shape_change',lambda z:(z>.25).sum()),
        large_start_effect=('actual_vs_warm_shape_change',lambda z:(z>.25).sum()),weak_two_start=('weakly_identified_two_start','sum'),
        actual_MAE_better=('real_outcome_high_MAE_gain',lambda z:(z>1e-6).sum()),
        actual_MAE_worse=('real_outcome_high_MAE_gain',lambda z:(z< -1e-6).sum()),
        actual_MAE_better_both_starts=('improves_under_both_starts','sum'),
        actual_MAE_worse_both_starts=('worsens_under_both_starts','sum')).reset_index()
    design.save('controlled_effects_by_task.csv',bytask)
    c2=cases[(cases.path_id.isin(['path0028','path0033','path0037','path0041']))&
        cases.measured_total.isin([14,7,10])]
    # Use the exact four event references, not an accidental second case on a path.
    fixed={('path0028',14),('path0033',7),('path0037',7),('path0041',10)}
    c2=cases[cases.apply(lambda r:(r.path_id,int(r.measured_total)) in fixed,axis=1)]
    design.save('case2_four_real_vs_expected_controls.csv',c2)
    readable_observation_catalogue(cases,registry,variants,axes)
    menu=read('case2_before_control_candidate_disagreement.csv');post=read('case2_actual_category_control_kernel_fits.csv')
    plots(cases)
    seedmax=cases.same_old_data_seed_shape_change.max()
    nbetter=int((cases.real_outcome_high_MAE_gain>1e-6).sum());nworse=int((cases.real_outcome_high_MAE_gain< -1e-6).sum())
    cat=uaxes[(uaxes.kind=='categorical')&(uaxes.delta_rho.abs()>=.25)]
    text=['# Which experimental observations change the GP kernel usefully?\n',
      'Real measurements often improve predictions of unseen high-value regions, but a large fitted-kernel '
      'change is not by itself a reliable key-point criterion. The evidence separates response information, '
      'experimental contrast and optimization-start sensitivity. No historical optimization trajectory was changed.\n',
      '## Scope and controlled comparisons\n',
      f'The165 lower-start main paths provide2475 archived GP updates (observations5--19), corresponding to '
      f'{len(unique)} distinct prefix/measurement/parameter transitions after exact reference deduplication. '
      'The62 high-start main paths remain separate. For each lower-start path, the largest pre-discovery '
      'kernel-correlation update was selected; all four documented Case2 admission observations were added. '
      'This gives167 path/observation references and131 distinct controlled datasets. This is a retrospective '
      'influential-update sample, not a random sample of prospective experimental choices.\n',
      'Each controlled dataset is fitted four ways: A actual response/native initialization; E the same '
      'conditions with response replaced by the preceding GP mean; S preceding data unchanged with the '
      'next-step fitting seed; W actual response initialized from the preceding fitted parameters. '
      'A and E have the same data count, fitting seed and default initialization. A and W have exactly '
      'the same dataset and objective. E uses synthetic feedback only as a diagnostic. A follow-up adds '
      'EW: the same expected-response data as E, initialized from the preceding fitted parameters. '
      'A/E and W/EW complete a response-by-initialization2x2 comparison on every controlled dataset.\n',
      'All candidate comparisons hold the actual next-step incumbent, acquisition normalization,512 Sobol '
      'draws, availability and tie priorities fixed. True unqueried yields are used only for retrospective '
      'prediction checks; they do not enter model fitting or the disagreement score. No synthetic SearchAUC '
      'trajectory is reported. The524 initial controlled fits,131 expected-response warm-start fits '
      'and two exploratory contrast fits make657 isolated '
      'GP fits, with zero new experimental queries and zero LLM calls.\n',
      '## What is learned from actual feedback?\n',
      f'Actual rather than GP-expected feedback lowers the mean absolute prediction error on still-unqueried '
      f'high-value candidates in{nbetter}/131 cases and increases it in{nworse}/131. '
      'The high region is at least90% of the fixed pool maximum; errors are normalized by task pool range. '
      'These comparisons test the incremental effect of the actual outcome at the same measured condition, '
      'rather than correlating high yields with cumulative-best metrics.\n',
      f'The completed2x2 comparison finds{int(cases.improves_under_both_starts.sum())}/131 cases in which '
      'the real response improves high-region MAE under both tested optimization starts, '
      f'{int(cases.worsens_under_both_starts.sum())} where it worsens under both, and '
      f'{len(cases)-int(cases.improves_under_both_starts.sum())-int(cases.worsens_under_both_starts.sum())} '
      'with differing directions or below-tolerance changes. '
      f'{int((cases.minimum_gain_across_starts>=.01).sum())} cases improve by at least1% of pool range '
      'under both starts. This is evidence for useful information-bearing observations in this selected '
      'sample; it is not a guarantee that an observation can be identified prospectively or improves SearchAUC.\n',
      f'Large real-versus-expected changes in effective kernel correlations occur in{int((cases.actual_vs_expected_shape_change>.25).sum())}/131 cases. '
      f'In{int(cases.real_response_brings_top5.sum())} cases, actual feedback brings a global-optimum candidate '
      f'into the fixed top-five interface while expected feedback would not; the reverse occurs in'
      f'{int(cases.real_response_loses_top5.sum())}. Improved predictive accuracy and improved acquisition '
      'exposure are different outcomes and are tabulated separately.\n',
      table(bytask,['task','cases','large_real_response_effect','large_start_effect','weak_two_start',
                    'actual_MAE_better_both_starts','actual_MAE_worse_both_starts']),
      '## How do kernels respond to concrete conditions?\n',
      'For each axis, full-kernel correlation is evaluated while holding other coordinates fixed: a '
      'continuous change of20% of input range or one change of nominal level. Continuous RBF lengthscales '
      'and likelihood noise use the native identity-constrained parameters; categorical lengthscales '
      'use the native Softplus transform. Eight task-specific decoders were checked against actual kernel matrices.\n',
      table(catalog,['task','unique_transitions','large_kernel_responses','large_nominal_axis_changes','without_direct_nominal_contrast']),
      f'Among{len(cat)} distinct large nominal-axis correlation changes, '
      f'{int((~cat.isolated_contrast).sum())} follow an observation with no isolated contrast on that axis. '
      'Thus changing temperature, loading or several conditions together can substantially change the fitted '
      'catalyst relationship. Such a change is a joint marginal-likelihood inference, not direct experimental '
      'identification of a catalyst effect. The complete axis table retains exact contrast anchors, '
      'response differences, boundary flags, novelty, lengthscales and correlation changes.\n',
      'There is no general rule that a larger observed response difference must shorten a particular '
      'lengthscale. Background, noise, empirical output scaling, kernel amplitudes and all lengthscales '
      'are fitted jointly. Isolated contrasts make the interpretation more defensible than simultaneous '
      'changes in several factors.\n',
      '## A reproducible information-bearing point: SnAr concentration contrast\n',
      'For SnAr cold V13 seed0, hold residence time0.5, piperidine equivalents5 and temperature30C fixed. '
      'Increase concentration from0.385714 to0.442857. The measured objective rises from8805.17 '
      'to10199.89, while the preceding GP predicted9116.81 for the new point. This is an identified '
      'one-factor contrast with the preceding measured point, not an attribution from a confounded design.\n',
      'The normalized concentration lengthscale changes from0.32276 to0.48359, and correlation at a20% '
      'input-range displacement rises from0.82532 to0.91803. The new response supports smoother transfer '
      'along concentration. High-region MAE falls from0.36379 to0.17778 of pool range. Using the expected '
      'response instead gives0.33484. The warm-start fit gives0.17777 and a nearly identical effective kernel, '
      'so this example is not a large optimization-start bifurcation.\n',
      'Under the common post-step acquisition settings, actual feedback leaves the optimum ranked1; '
      'expected feedback would rank it2425. Before the measurement it was ranked4, and V13 chose '
      'the GP reference at that step. The next experiment measures the pool optimum. This is evidence '
      'of useful ordinary GP learning; it is not a specific KF or LLM benefit.\n',
      '## A different kind of key point: distinguish plausible kernel explanations\n',
      'Case2 warm V13 seed0 has14 observations, nine usingP1-L1 and none forming a matched cross-catalyst '
      'comparison at identical continuous settings. The new23.3325% point changes time, temperature and '
      'loading together while retainingP1-L1. Its result is only1.6919 percentage points above the old '
      'GP mean. Nevertheless the product categorical lengthscale changes from0.09577 to3.68489; '
      'the full same-setting cross-category correlation changes from0.000029 to0.762337.\n',
      'Replacing that measured response by its GP expectation21.6407% keeps category correlation near '
      '0.000008 and moves the optimum from rank5 to175. But fitting the same actual14 observations '
      'from the old parameters retains correlation0.000029 and ranks the optimum90. The two actual-data '
      'solutions differ by only0.000159 in per-observation marginal likelihood. The measurement affects '
      'the numerical solution, while these data do not decisively choose the two catalyst-sharing explanations.\n',
      table(c2,['path_id','measured_total','measured_result','GP_expected_result','actual_global_rank',
                'expected_global_rank','warm_global_rank','actual_vs_warm_MLL_gap']),
      'The earlier interpretation of Case2 seed2 as an ordering example needs qualification. Removing '
      'candidates and updating tie order is sufficient if the old GP is frozen. In the actual refitting '
      'pipeline, replacing the real response by its predicted mean ranks the optimum494 instead of3. '
      'Ordering and outcome-driven refitting interact; neither is an exclusive explanation.\n',
      'Before measuring the next point, compare the two already fitted14-observation models on the '
      'five offered candidates. Standardized mean disagreement is computed from their predictions only. '
      'P1-L5 at600s,110C and2.5mol% is both the only catalyst-only control in this menu and its largest '
      'model-disagreement point. No true yield is used in that ranking.\n',
      table(menu,['candidate_id','native_rank','mean_A','SD_A','mean_W','SD_W',
                  'standardized_model_disagreement','category_only_control','measured_next']),
      'The actual experiment returns40.0853%, versus23.3325% forP1-L1 under those same settings. '
      'This supplies an interpretable16.7528 percentage-point catalyst contrast and discovers the optimum. '
      'It is an example of a potentially informative point that can be recognized from existing data '
      'and design structure before knowing its yield. It is not a multi-task policy validation.\n',
      table(post,['stage','variant','observed','category_correlation','MLL']),
      'After that comparison, the best of the three tested fits has correlation0.50334 and per-observation '
      'MLL-0.92863. Fits initialized from the two preceding solutions have MLL-0.96914 and-0.96466. '
      'Their gaps to the tested best now exceed0.01, unlike the nearly tied14-observation solutions. '
      'However they still converge to different local solutions: the contrast adds discrimination but '
      'does not make the optimizer or global kernel uniquely identified.\n',
      '## Separate experimental information from fitting sensitivity\n',
      f'All131 native actual-response refits reproduce the archived kernel correlations exactly. '
      f'The largest change from refitting the old data with the next-step seed is{seedmax:.3g}. '
      f'By contrast,{int((cases.actual_vs_warm_shape_change>.25).sum())}/131 actual datasets change effective '
      'kernel correlations by more than0.25 when the optimizer starts at preceding parameters. '
      f'{int(cases.weakly_identified_two_start.sum())} pairs have that large shape difference while '
      'their per-observation MLL gap is below0.01. This is a local two-start weak-identification diagnostic, '
      'not proof that all possible kernel explanations are equivalent.\n',
      '![All131 controlled response and optimizer diagnostics](kernel_condition_diagnostics.png)\n',
      f'A diagnostic that simply selects the higher-MLL solution chooses the warm-start fit in'
      f'{int((selector.picked=="W_actual_warm_start").sum())} cases. Relative to native fitting, hidden '
      f'high-region MAE improves in{int((selector.high_MAE_gain_vs_native>1e-6).sum())}, worsens in'
      f'{int((selector.high_MAE_gain_vs_native< -1e-6).sum())}, and is unchanged in the rest. '
      'Therefore multiple starts can reveal a risk, but maximum training evidence alone is not a '
      'demonstrated fix for predicting the unobserved high region.\n',
      '## Implication for identifying useful experiments\n',
      'A point is useful relative to the existing observations and remaining decision, not solely because '
      'its own yield is high. Two supported roles emerge: an anchored one-factor response that improves '
      'transfer along a coordinate, and a condition contrast that discriminates models with similar '
      'training support. A large parameter jump without either interpretability or fitting robustness '
      'should not automatically be counted as learned chemical knowledge.\n',
      'A research direction grounded in these results is to expose kernel-model disagreement and available '
      'single-factor contrasts within the existing qLogEI interface, then use real feedback to validate '
      'the competing condition hypotheses. This study diagnoses that direction; it does not establish '
      'SearchAUC gain, justify extra experimental cost, or attribute a benefit to KF or event refresh.\n',
      '## Evidence files and limits\n',
      'The complete2475 updates and10,845 axis records are retained, including non-improving paths. '
      'Exact duplicate references are removed only for the distinct-transition and controlled-fit summaries. '
      'Prefixes still overlap, the selected cases emphasize large model responses, and the task counts '
      'are unbalanced. Counts and paired numerical diagnostics are descriptive; no p-values or independent '
      'sample claims are made. A/E marginal likelihoods cannot be compared as model evidence because '
      'their target data differ. A/W share the same data and can be compared directly.\n',
      '- all2475_kernel_transitions.csv / all_axis_condition_responses.csv: every update and its physical conditions.\n'
      '- unique_kernel_transitions.csv / unique_axis_condition_responses.csv: exact-reference deduplication.\n'
      '- controlled_case_registry.csv / controlled_jobs.json: all167 references and131 unique fitting jobs.\n'
      '- controlled_refit_variants.csv: the original524 fitting controls.\n'
      '- expected_warm_refits.csv / factorial_refit_variants.csv / controlled_case_effects.csv: '
      'the completed2x2 comparison plus old-data seed controls,655 fits in total.\n'
      '- key_observations_by_condition_and_axis.csv: raw experimental values, contrast anchors, native '
      'lengthscales and correlations under actual, expected and warm-start controls.\n'
      '- case2_before_control_candidate_disagreement.csv: prospectively available menu comparison.\n'
      '- case2_actual_category_control_kernel_fits.csv: actual catalyst-contrast follow-up.\n'
      '- native_decoder_checks.json / parameter_constraints.json: native parameter-transform validation.\n'
      '- profile_audit.json / refit_audit.json / contrast_audit.json: scope, failures and computation receipts.\n'
      'Original states and GP caches remain read-only. New fitting checkpoints are isolated under '
      'experiments/runs/v13_kernel_conditions_20261104. CPU worker processes preserve the native fitting '
      'procedure, use one numerical thread each, and perform no in-loop operator compilation.\n',
      'Reproduce from the project root with the existing runtime/environments/.venv-official/Scripts/python.exe '
      'interpreter, in this order: profile_v13_kernel_conditions.py; refit_v13_kernel_conditions.py '
      '--workers 12; complete_v13_kernel_factorial.py; check_case2_kernel_contrast.py; '
      'report_v13_kernel_conditions.py. These scripts are under code/analysis. Reuse only the frozen '
      'pools and inputs recorded in the source hash manifests.\n']
    (OUT/'analysis.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    audit=dict(status='PASS_KERNEL_CONDITION_REPORT',all_transitions=len(edges),unique_transitions=len(unique),
        controlled_cases=len(cases),actual_feedback_improves_high_MAE=nbetter,actual_feedback_worsens_high_MAE=nworse,
        actual_feedback_improves_under_both_starts=int(cases.improves_under_both_starts.sum()),
        actual_feedback_worsens_under_both_starts=int(cases.worsens_under_both_starts.sum()),
        improves_at_least_1pct_range_under_both_starts=int((cases.minimum_gain_across_starts>=.01).sum()),
        optimizer_start_large_disagreements=int((cases.actual_vs_warm_shape_change>.25).sum()),
        weak_two_start_cases=int(cases.weakly_identified_two_start.sum()),seed_only_max_shape_change=float(seedmax),
        new_experiments=0,new_LLM_calls=0)
    for path,h in READS.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h,path
    READS[Path(__file__).relative_to(ROOT).as_posix()]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    design.write(OUT/'report_source_hashes.json',READS);design.write(OUT/'report_audit.json',audit)
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
