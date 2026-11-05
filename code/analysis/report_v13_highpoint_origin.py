"""Report computational origins, without high-yield/AUC tautologies."""
from pathlib import Path
import hashlib
import io
import json
import sys

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/analysis'))
import trace_v13_highpoint_origin as origin
OUT=origin.OUT
READS={}


def read(name,base=OUT):
    p=base/name;raw=p.read_bytes();READS[p.relative_to(ROOT).as_posix()]=hashlib.sha256(raw).hexdigest()
    return pd.read_csv(io.BytesIO(raw),encoding='utf-8-sig')


def table(data,columns):
    out='| '+' | '.join(columns)+' |\n| '+' | '.join('---' for _ in columns)+' |\n'
    for _,r in data.iterrows():
        out+='| '+' | '.join('NA' if pd.isna(r[k]) else f'{r[k]:.3f}' if isinstance(r[k],(float,np.floating)) else str(r[k]) for k in columns)+' |\n'
    return out


def plot_case2(d,traces):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig,axes=plt.subplots(2,2,figsize=(11.5,7),sharex=True,sharey=True)
    data=[]
    styles={'bo':('#59636d','o','-'),'v13':('#2466a8','^','--')}
    for seed,ax in enumerate(axes.flat):
        for method,(color,marker,ls) in styles.items():
            meta=d[(d.primary)&(d.regime=='warm')&(d.task=='c2')&(d.seed==seed)&(d.method==method)].iloc[0]
            b=traces[(traces.path_id==meta.path_id)&(traces.target_kind=='pool_global_max')&traces.target_available]
            ax.plot(b.total_observations,b.native_rank,color=color,marker=marker,ms=4,lw=1.5,ls=ls,label=method)
            chosen=b[b.target_selected==True]
            if len(chosen):ax.scatter(chosen.total_observations,chosen.native_rank,s=160,c='#c8751c',marker='*',zorder=5)
            data.extend(b.to_dict('records'))
        ax.axhspan(1,5,color='#e8eef5',alpha=.8,zorder=0);ax.axhline(5,color='#8f979f',lw=.7)
        ax.set_yscale('log');ax.set_ylim(4000,.8);ax.set_xlim(4.5,20.5)
        ax.set_xticks([5,10,15,20]);ax.set_yticks([1,5,20,100,1000],['1','5','20','100','1000'])
        ax.set_title(f'seed {seed}',fontsize=12,pad=7);ax.grid(axis='x',color='#e9ecef',lw=.7)
        for side in ('top','right'):ax.spines[side].set_visible(False)
    for ax in axes[:,0]:ax.set_ylabel('Native qLogEI rank of global maximum\n(lower is earlier in the shortlist)')
    for ax in axes[-1,:]:ax.set_xlabel('Total observations: four initial + subsequent queries')
    handles=[Line2D([],[],color=c,marker=m,ls=ls,label='BO' if k=='bo' else 'V13') for k,(c,m,ls) in styles.items()]
    handles.append(Line2D([],[],color='#c8751c',marker='*',ms=11,ls='',label='Global maximum measured'))
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.54,.015),ncol=3,frameon=False)
    fig.suptitle('Case 2: how the same global optimum becomes selectable',fontsize=15,y=.98)
    fig.text(.54,.925,'All four matched seeds; shaded band = native top five. The target yield is unknown to the policy.',ha='center',fontsize=10)
    fig.subplots_adjust(left=.11,right=.98,top=.86,bottom=.14,wspace=.16,hspace=.25)
    fig.savefig(OUT/'case2_highpoint_rank_origins.png',dpi=180,facecolor='white');plt.close(fig)
    origin.save('case2_highpoint_rank_figure_data.csv',data)


def main():
    d=read('conditional_path_profiles.csv',origin.SOURCE);traces=read('all_candidate_origin_traces.csv')
    events=read('all_first_exposure_events.csv');numeric=read('one_step_numerical_mechanisms.csv')
    params=read('refit_parameter_channel_restorations.csv');branches=read('all_lower_start_warm_first_divergences.csv')
    selections=read('all_first_high_selections_with_reasons.csv')
    ids=set(d[d.primary&(d.start_fraction_max<.8)].path_id)
    ev=events[events.path_id.isin(ids)];first=ev[ev.target_kind=='first90_hit']
    global_=ev[ev.target_kind=='pool_global_max'];new=first[first.entry_origin=='new_exposure_after_feedback']
    source=[]
    for (method,regime),b in global_.groupby(['method','regime']):
        source.append(dict(method=method,regime=regime,paths=len(b),never_in_menu=int(b.never_exposed.sum()),
            in_menu_never_selected=int((b.first_exposure_total.notna()&b.selection_total.isna()).sum()),
            selected_global=int(b.selection_total.notna().sum()),
            selected_after_prior_menu_offer=int((b.exposed_before_selection>0).sum())))
    source=pd.DataFrame(source);origin.save('global_optimum_exposure_vs_selection.csv',source)
    phases=[]
    for kind,g in numeric[~numeric.initial].groupby('target_kind'):
        phases.append(dict(target_kind=kind,admission_events=len(g),
            fixed_parameter_update_sufficient=int((g.fixed_kernel_observation_update_rank<=5).sum()),
            refit_or_rescaling_required=int((g.fixed_kernel_observation_update_rank>5).sum()),
            observed_SD_increased=int((g.target_SD_after>g.target_SD_before).sum())))
    origin.save('exposure_update_channels.csv',phases)
    par=[]
    for (kind,variant),g in params.groupby(['target_kind','variant']):
        par.append(dict(target_kind=kind,variant=variant,applicable_events=len(g),
            still_in_top5=int(g.remains_in_top5.sum()),loses_top5=int((~g.remains_in_top5).sum())))
    par=pd.DataFrame(par);origin.save('parameter_necessity_summary.csv',par)
    # Keep all24 V13 matched low-start branches, not selected successful ones.
    v=branches[branches.method=='v13'];assert len(v)==24
    origin.save('all24_v13_low_start_branches.csv',v)
    c2=numeric[(numeric.task=='c2')&(numeric.regime=='warm')&(numeric.method=='v13')&
               (numeric.target_kind=='pool_global_max')].copy()
    c2=c2.merge(events[['path_id','target_kind','preceding_observation_id','preceding_observation_value',
                        'selection_total','first_exposure_total']],on=['path_id','target_kind'],validate='one_to_one')
    origin.save('case2_all_four_admission_mechanisms.csv',c2)
    plot_case2(d,traces)
    highrole=selections.groupby(['method','regime']).apply(lambda g:pd.Series(dict(n=len(g),
        chosen_below_GP_first=int((~g.selected_is_GP_reference).sum()),
        declared_knowledge_use=int((g.knowledge_uses>0).sum()),
        declared_use_of_previously_scored_knowledge=int((g.prior_feedback_knowledge_uses>0).sum()),
        refresh_at_selection=int(g.refresh_at_selection.sum()),category_only_control=int(g.pure_single_category_control.sum())))).reset_index()
    origin.save('first_high_selection_policy_roles.csv',highrole)
    example=params[(params.path_id=='path0028')&(params.target_kind=='pool_global_max')]
    stages=['before_rank','depletion_only_rank','new_tie_order_rank','new_incumbent_rank','new_MC_draw_rank',
            'new_acquisition_scale_rank','fixed_kernel_observation_update_rank','actual_refitted_GP_rank']
    text=['# How do high-value candidates become selectable?\n',
      'This analysis traces the computation that exposes a candidate, rather than correlating high yields '
      'with SearchAUC. It uses archived predictions, observed feedback, exact kernel conditioning, '
      'and parameter-restoration diagnostics. The policy never receives the target\'s unknown true yield.\n',
      '## Scope\n',
      'All398 catalogue states and718 target histories are retained, giving11,812 archived target-step rows. '
      'The primary population contains165 main paths with best of first four observations below80% of the '
      'pool maximum;62 high-start main paths remain separate. Warm prefixes are actual initializations; '
      'cold prefixes are algorithm-selected startup. Frozen replays and ten-observation continuations '
      'remain separate archive families. A high discovery means at least90% of the frozen pool maximum, '
      'not90% absolute yield. Global maximum means the actual highest pool objective, keeping all tied points.\n',
      '## Computational bottlenecks: exposure and selection are distinct\n',
      'Among165 lower-start main paths, the exact global maximum never entered the offered menu in79 paths. '
      'It entered in86;69 eventually measured it, while17 did not. In51 of the69 successful paths it had '
      'already been offered at an earlier decision. These are counts of actual feasible actions, not correlations '
      'with the optimization metric. BO must take rank1, whereas the LLM interfaces can select within the five-point menu.\n',
      table(source,['method','regime','paths','never_in_menu','in_menu_never_selected','selected_global',
                    'selected_after_prior_menu_offer']),
      'This separates failure to expose the optimum from failure/delay in selecting an exposed optimum. '
      'It does not imply that the policy could know which unmeasured candidate is optimal.\n',
      '## What changed immediately before first exposure?\n',
      'There are133 actual first-high-discovery points in this population. One had already appeared in the '
      'no-data uniform menu;132 first entered after feedback. Of those132 preceding measurements,101 were '
      'below80% of the pool maximum,28 below50%, and43 did not improve the current incumbent. '
      'Thus a modest or unsuccessful measurement can expose the later high point through its information '
      'about the response surface. No yield/AUC correlation is needed for this statement.\n',
      'The old fitted kernel was conditioned on the single actual new observation, without reoptimizing '
      'parameters. After applying the same next-step incumbent, candidate mask, tie priorities, scaling and '
      'MC draws,57 of132 first-high targets enter the top five under this frozen update;75 do not. '
      'The actual refitted model admits those75.\n',
      table(pd.DataFrame(phases),['target_kind','admission_events','fixed_parameter_update_sufficient',
                                 'refit_or_rescaling_required','observed_SD_increased']),
      'For45 of the132 first-high targets, the actual posterior SD increases. Pure fixed-parameter Gaussian '
      'conditioning cannot produce that increase; the difference comes from refitting and output rescaling. '
      'Therefore “the GP became more uncertain after this result” must not be attributed to conditioning alone.\n',
      'A sequential acquisition check also finds11 cases where removing the measured candidate is already '
      'sufficient for admission, four where changing the tie order becomes sufficient, and13 where changing '
      'the incumbent becomes sufficient. These are order-dependent sufficiency checks; they are not additive '
      'causal shares, and subsequent conditioning can reverse an earlier-stage admission.\n',
      '## Which fitted quantities matter?\n',
      'For the75 first-high events requiring refit/rescaling, hold the actual new dataset, output '
      'normalization, incumbent, candidate mask and MC draws fixed. Restore all fitted parameters to their '
      'preceding values: eight targets remain in the top five;67 leave it. Thus output rescaling and '
      'conditioning alone do not explain those67 admissions. No GP optimization was performed in this test.\n',
      table(par[par.target_kind=='first90_hit'],['variant','applicable_events','still_in_top5','loses_top5']),
      'Restoring just continuous lengthscales removes38 of66 applicable targets; restoring the categorical '
      'kernel removes28 of58. Kernel amplitudes remove18 of49, background mean six of75 and noise five '
      'of75. These effects overlap: several parameter changes can be jointly necessary. They identify '
      'sensitive computational channels, not universally valid chemical relationships or additive importance.\n',
      '## Case2: all four matched seeds\n',
      'The pool optimum is40.0853157% at catalystP1-L5,600s,110C and2.5mol% loading. '
      'All four warm seeds have matching V13/BO initial observations and initial GP. '
      'The plot shows when that same fixed point moves into the native top-five menu and when each '
      'method measures it. It does not use the point\'s true yield to generate any prediction.\n',
      '![All four Case2 candidate-rank histories](case2_highpoint_rank_origins.png)\n',
      table(c2,['seed','preceding_observation_value','before_rank','fixed_kernel_observation_update_rank',
                'actual_refitted_GP_rank','first_exposure_total','selection_total']),
      'Seed0 gives a complete source-linked chain. At total observation14, V13 measures23.3325% at '
      'P1-L1,600s,110C,2.5mol%. Before incorporating it, the unmeasured optimum ranks213. '
      'Removing the measured point gives212; updating incumbent/MC settings gives157; conditioning '
      'with the old physical kernel gives144. The actual next GP puts it at rank5. '
      'Its predicted mean changes from5.4658% to14.4497%, while SD falls from6.3259 to5.2039 percentage points. '
      'The same-setting cross-category correlation changes from0.000029 to0.762337.\n',
      table(example,['variant','rank','strict_rank','ties']),
      'Restoring only the preceding categorical-kernel parameters on the actual new dataset moves the '
      'optimum back to rank132; all other fitted quantities and acquisition settings remain those of the '
      'actual next step. This isolates the categorical-kernel update as locally necessary for that admission. '
      'The LLM then chooses the fifth-ranked point, changing only catalyst fromP1-L1 to the unmeasuredP1-L5. '
      'Its recorded reason explicitly proposes a catalyst comparison at the strongest observed continuous '
      'settings. The observed yield is40.0853%. This explains exposure, experimental design and selection. '
      'The GP did not predict40% yield: its14.45% mean was below the23.33% incumbent, so this was still an '
      'uncertain alternative. Both V13 and BO measure the maximum at total observation15 in this seed. '
      'V13 still loses to BO on SearchAUC because its earlier incumbent trajectory is worse; '
      'this is not a selected winning example.\n',
      'Seed1 illustrates a different mechanism. The preceding measurement13.9685% is6.1183 percentage '
      'points below the old GP prediction. Its conditional influence coefficient on the target is-0.08459. '
      'The product is+0.5176 percentage points, so the target mean rises from22.3411 to22.8587% under '
      'the old kernel, and the rank becomes3. This is negative posterior covariance after conditioning on '
      'the existing data; it is not a claim that catalysts have a negative prior similarity.\n',
      'Seed2 is mainly an ordering example: the target initially has native rank6, strict score rank2 '
      'and a six-point tie class. Removing the higher-score measured point makes its native rank5; '
      'the next tie order makes it rank3. Its first appearance is not evidence that a chemical rule '
      'has newly identified the optimum. Seed3 goes the other way under the frozen kernel: rank73 '
      'becomes1150 after conditioning, but the actual refitted model gives rank5, with SD rising '
      'from2.5825 to4.3518 percentage points. These paths do not share a single universal mechanism.\n',
      '## Matched branches: why a higher immediate yield need not build a better path\n',
      'All99 main warm method/BO pairs were checked at their first divergent choice, with identical '
      'physical history, GP, incumbent and candidate menu. The lower-start subset contains74 pairs, '
      'including all24 V13/BO pairs. Their next GP snapshots are actual observed branches, not simulated '
      'optimization rollouts. All rows are supplied in all24_v13_low_start_branches.csv.\n',
      'For Case2 seed0, the first divergent V13 choice yields8.5073% versus BO\'s6.6840%. '
      'Nevertheless the next GP ranks the fixed optimum534 under the V13 branch and187 under BO. '
      'The immediate result was higher, but the induced model ordering was less favorable. '
      'This is a concrete branch difference, not a monotone link from observed yield to future GP quality.\n',
      table(v,['task','seed','first_divergence_total','agent_selected_value','BO_selected_value',
                'agent_global_rank_after_actual_observation','BO_global_rank_after_actual_observation',
                'delta_AUC_normalized']),
      '## Where are LLM knowledge, KF and refresh in this chain?\n',
      'GP parameters change because the observed dataset is refitted; V13 does not directly modify GP '
      'means with KF weights. LLM/context/knowledge can affect which point is measured, which then changes '
      'that dataset. At the actual first-high selection, the recorded roles are:\n',
      table(highrole,['method','regime','n','chosen_below_GP_first','declared_knowledge_use',
                      'declared_use_of_previously_scored_knowledge','refresh_at_selection']),
      'In the Case2 seed0 optimum-selection commit, a refresh is present but there is no declared '
      'structured knowledge use and no structured forecast. Therefore the successful catalyst comparison '
      'cannot be specifically attributed to KF. Refresh may influence the LLM context; a logged refresh '
      'alone cannot identify its net causal contribution. The complete recorded rationale, conditions '
      'and source commit are saved for every first-high selection.\n',
      '## Interpretation and remaining uncertainty\n',
      'High actual objective values are properties of the fixed response surface. The optimization method '
      'determines when their conditions become exposed and selected. The demonstrable channels here are '
      'observation-conditioned posterior changes, refitted lengthscales/category sharing, ordinary '
      'candidate/tie ordering, and the selection among offered points. These are prior-to-measurement '
      'computations; they explain more than a correlation between high values and SearchAUC.\n',
      'Parameter restoration identifies local computational necessity under stated fixed settings. It does '
      'not prove the refitted chemical similarity is physically correct, identify a unique global cause '
      'of path success, or establish that a particular next experimental design will reliably improve '
      'performance. Refitting also includes standardization and numerical optimization behavior; those '
      'must not all be renamed chemical learning. No new model fitting, LLM response, experimental query '
      'or counterfactual BO trajectory was executed.\n',
      '## Evidence and reproduction\n',
      'The218 one-step GP/acquisition replays passed; the95 distinct parameter-restoration transitions '
      'produce751 diagnostic target rows without unresolved replays. Frozen conditioning was independently '
      'checked with native condition_on_observations for the mixed, categorical and continuous GP families. '
      'A transition can expose both the pool maximum and the first-high target, so these target replays '
      'are not218 independent experimental trajectories. '
      'Native q=1 qLogEI is reproduced using its exact512 Sobol draws, smoothing, jitter and best_f dtype; '
      'no analytic-EI surrogate is substituted.\n',
      '- all_candidate_origin_traces.csv: every archived target-step trace across398 states.\n'
      '- all_first_exposure_events.csv: all718 targets, including never exposed.\n'
      '- one_step_numerical_mechanisms.csv: conditioning, refit and acquisition-stage ranks.\n'
      '- refit_parameter_channel_restorations.csv: actual-data parameter restorations.\n'
      '- all24_v13_low_start_branches.csv: every lower-start V13/BO warm branch pair.\n'
      '- all_first_high_selections_with_reasons.csv:133 actual first-high choices and source-linked reasons.\n'
      '- independent_conditioning_checks.csv: native conditional-model verification.\n'
      '- audit.json, branch_audit.json, parameter_audit.json and source hash manifests: custody and scope.\n']
    (OUT/'analysis.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    for path,h in READS.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h,path
    READS[Path(__file__).relative_to(ROOT).as_posix()]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT/'report_source_hashes.json').write_text(json.dumps(READS,indent=2)+'\n',encoding='utf-8')
    print('ORIGIN_REPORT_COMPLETE',len(d),'catalogue states;',len(ids),'primary low-start paths')


if __name__=='__main__':main()
