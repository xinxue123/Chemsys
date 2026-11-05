"""Source-backed comparison and finite-pool hypothesis checks after completion."""
from collections import defaultdict
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from chemical_hypothesis_bo import study as s,hypotheses as h
import chemical_hypothesis_active as active


def check_claims():
    claims={}
    for path in sorted(active.responses()):
        saved=s.read(path);req=s.read(ROOT/saved['request'])
        for note in saved['plan']['notes']:claims.setdefault(req['task'],{}).setdefault(note['id'],note)
    result=[]
    for task in s.native.TASKS:
        pool=s.native.load_task(ROOT,task,seed=0)
        assert all(s.read(s.folder_for(task,arm)/'state.json')['status']=='COMPLETE' for arm in ('fixed','adaptive'))
        with np.load(pool.source_path.with_name('oracle.npz'),allow_pickle=False) as f:
            y=f['raw_values'].copy() if 'raw_values' in f else f['values'].copy()*pool.scale
        for note in claims.get(s.SHORT[task],{}).values():
            plus,minus=h.masks(pool,note);axes=[a for a in pool.axes if a not in note['focus']]
            grouped=defaultdict(lambda:[[],[]])
            for side,mask in enumerate((plus,minus)):
                for j in np.flatnonzero(mask):grouped[tuple(pool.points[j][a] for a in axes)][side].append(float(y[j]))
            gaps=[np.mean(v[0])-np.mean(v[1]) for v in grouped.values() if v[0] and v[1]]
            result.append(dict(task=s.SHORT[task],id=note['id'],claim=note['claim'],
                preferred_count=int(plus.sum()),comparison_count=int(minus.sum()),
                preferred_pool_mean=float(y[plus].mean()),comparison_pool_mean=float(y[minus].mean()),
                unmatched_region_mean_gap=float(y[plus].mean()-y[minus].mean()),
                exactly_matched_nonfocus_contexts=len(gaps),
                matched_mean_gap=float(np.mean(gaps)) if gaps else None,
                matched_fraction_positive=float(np.mean(np.asarray(gaps)>0)) if gaps else None,
                interpretation='Finite-pool post hoc descriptive check; not prospective validation or a causal chemical law.'))
    df=pd.DataFrame(result);df.to_csv(s.REPORT/'posthoc_claim_validation.csv',index=False,encoding='utf-8-sig')
    return df


def plot(frame):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tasks=list(s.SHORT.values());names=['Case 1','Case 2','Case 3','Case 4','Perera Suzuki','Doyle B-H','SnAr','HPLC']
    methods=['previous_kernel_F','fixed','adaptive'];colors=['#45688D','#B48622','#CF6429'];markers=['s','o','^']
    labels=['Previous kernel F','Grounded, fixed','Grounded, adaptive']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,ax=plt.subplots(figsize=(11,6.6));positions=np.arange(8)
    for k,method in enumerate(methods):
        values=[]
        for task in tasks:
            d=frame[frame.task==task].set_index('method');values.append(100*(d.loc[method,'SearchAUC']/d.loc['pure_BO','SearchAUC']-1))
        yy=positions+(k-1)*.23
        ax.scatter(values,yy,c=colors[k],marker=markers[k],s=46,label=labels[k],zorder=3)
        for x,y in zip(values,yy):
            label=f'{x:+.2f}%' if abs(x)>=.005 else f'{x:+.4f}%'
            ax.annotate(label,(x,y),xytext=(6 if x>=0 else -6,0),textcoords='offset points',
                ha='left' if x>=0 else 'right',va='center',fontsize=9,color=colors[k])
    ax.axvline(0,color='#333333',lw=1);ax.set_yticks(positions,names);ax.invert_yaxis()
    ax.set_xlim(-32,16);ax.set_xticks([-30,-20,-10,0,10]);ax.grid(axis='x',color='#e5e5e5',linewidth=.7,zorder=0)
    ax.set_xlabel('SearchAUC difference from matched pure BO (%)')
    ax.set_title('Eight matched tasks: earlier scheme and grounded chemical knowledge',loc='left',pad=44,fontsize=14)
    ax.legend(loc='lower left',bbox_to_anchor=(-.01,1.02),ncol=3,frameon=False,fontsize=10)
    for side in ('top','right'):ax.spines[side].set_visible(False)
    fig.text(.135,.025,'Seed 0; identical four initial observations + 16 queries. One trajectory per method/task; no uncertainty intervals.',fontsize=9,color='#444444')
    fig.subplots_adjust(left=.16,right=.96,bottom=.13,top=.82)
    fig.savefig(s.REPORT/'SearchAUC_comparison.png',dpi=180,facecolor='white');plt.close(fig)


def main():
    active.activate()
    assert s.read(s.REPORT/'audit.json')['status']=='PASS_REPLAY_AND_CUSTODY'
    assert s.read(s.REPORT/'posthoc_scope.json')['experimental_queries_added']==0
    frame=pd.read_csv(s.REPORT/'unified_results.csv');claim=check_claims();plot(frame)
    methods=['pure_BO','original_strict_contrast','previous_kernel_F','fixed','adaptive']
    text=['Grounded chemical hypotheses: unified comparison and mechanism analysis','',
        'Protocol: unchanged input pools and native GPs, seed 0, four identical initial observations and 16 queries. '
        'SearchAUC is the mean of the 16 post-query cumulative best objectives, including initialization in each maximum. '
        'Final best is the maximum of all 20 measured values. Objectives differ by task; raw values are not averaged across tasks.','',
        'Methods: pure_BO = frozen native qLogEI control; original_strict_contrast = earlier exact-match kernel diagnostic; '
        'previous_kernel_F = the subsequent kernel-contrast revision with predictive feedback; fixed = initial grounded LLM hypotheses and fixed weights; '
        'adaptive = the same initial response with predictive-likelihood weight updates and event refresh. '
        'The current update rule is Gaussian predictive likelihood. No claim is made that this is a V13 KF ablation.','',
        'SEARCH AUC','Task | '+' | '.join(methods)]
    for task in s.SHORT.values():
        f=frame[frame.task==task].set_index('method');text.append(task+' | '+' | '.join(f'{f.loc[m,"SearchAUC"]:.6f}' for m in methods))
    text+=['','FINAL BEST','Task | '+' | '.join(methods)]
    for task in s.SHORT.values():
        f=frame[frame.task==task].set_index('method');text.append(task+' | '+' | '.join(f'{f.loc[m,"final_best"]:.6f}' for m in methods))
    for reference in ('pure_BO','previous_kernel_F','fixed'):
        differences=[g.set_index('method').loc['adaptive','SearchAUC']-g.set_index('method').loc[reference,'SearchAUC'] for _,g in frame.groupby('task')]
        text+=['',f'Adaptive versus {reference}: {sum(v>1e-9 for v in differences)} higher, {sum(v< -1e-9 for v in differences)} lower, {sum(abs(v)<=1e-9 for v in differences)} ties.']
    text+=['','INTERPRETATION',
        'Case 3 differs from BO by only -0.000943. These are single-seed development comparisons, not stability evidence.','',
        'Early knowledge has concrete point-selection effects. Case 4 query 3: the 75-100 C temperature preference gives 87.333908 versus '
        'the same-prefix native BO reference 85.029747. Perera query 2: the boronic-acid versus pinacol-ester hypothesis is necessary '
        'for selecting 96.70; removing that expert gives 12.49 and the native reference gives 23.20. '
        'SnAr query 1: the short-residence-time expert is necessary for 5213.902057 versus 4225.408289 without that expert '
        '(native reference 4214.072701). These are numerical selection effects, not isolated mechanistic chemistry experiments.','',
        'Doyle uses the isolated identity-corrected rerun. CN1CCCN2C1=NCCC2 is MTBD (C8H15N3, PubChem CID123583); '
        'the original common-name annotation incorrectly said DBU. Correct structures and graph facts were present, but the alias was inconsistent. '
        'Both original Doyle trajectories, responses and first-pass reports are preserved. This correction also uses fresh independent LLM responses; '
        'differences between the two Doyle runs do not isolate the effect of a name correction. Current Doyle selection effects appear in the tables below.','',
        'There are explicit failures. Case 1 query 1 improves the immediate outcome, 28.498081 versus 18.881805, yet its changed '
        'history delays the search: pure BO reaches 67.139450 at query 5 and 88.441406 at query 9. Adaptive reaches its own final '
        '89.569000 only at query 16, so higher terminal yield coexists with much lower SearchAUC. HPLC larger-loop guidance '
        'gives 646.852600 versus the native query-1 reference 1783.436890; both are below the initial best 1879.794190, '
        'but the poor new observation changes subsequent GP fits. Retracting the hypothesis at query 4 cannot restore the earlier BO history.','',
        'FEEDBACK AND REFRESH',
        'Full fixed-versus-adaptive trajectories combine weight learning and library edits. The feedback shadow holds the actual GP prefix '
        'and actual library/refresh schedule fixed, propagates the initial prior through each library edit, and removes only predictive '
        'weight updates. It is narrower than an equal-prior-reset shadow, which also changes the new-library mass rule. The table below '
        'lists every selection changed by this feedback-only weight shadow.','',
        'Immediate refresh effects compare the new library to keeping the immediately preceding library and its preceding weights on '
        'the current actual prefix. This does not capture delayed refresh effects: Case 2 first diverges from fixed knowledge at query 10, '
        'measuring 40.085316 versus 23.332539 after its query-7 library revision. '
        'Conversely, Case 1 and Case 4 have later selection losses after library replacements even when the refresh round itself is unchanged.','',
        'Hypothesis-removal effects are nonlinear and must not be added together. The oracle counterfactuals are looked up only after '
        'all runs complete and are never supplied to the decision agents. They measure one-step changes on the actual prefix, '
        'not full-trajectory causal ablation effects. New LLM proposals, covariance-based preference transfer, changed observations and '
        'GP refitting can all alter later paths. A chemical statement can have a positive region-level association yet fail to identify '
        'the best candidate within the native budget.','',
        'COVARIANCE TRANSFER AND SCOPE',
        'The preference shift is not explicitly zeroed outside a hypothesis context. SnAr query 3 is influenced by hypotheses conditioned '
        'on 75-120 C, but selects at 42.857 C instead of the native reference at 30 C. Both points lie outside those contexts. '
        'The high-temperature short-residence-time trend is positive in all 256 exactly matched nonfocus grid contexts, yet it harms '
        'this lower-temperature ranking. HPLC query 1 selects sample_loop=0.065745 mL over a native reference at 0.071074 mL: '
        'both are already in the hypothesized 0.04-0.08 mL preferred band. That statement alone does not justify their relative ranking. '
        'Case 2 query 10 selects PCy3 over XPhos at identical 600 s, 110 C and 2.5 mol%; removing the temperature expert changes '
        'this choice. This is indirect covariance-mediated transfer, not an explicit pre-query PCy3 superiority claim. The LLM first '
        'adds that PCy3 comparison at query 11, after the 40.085316 result has already been observed.','',
        'AUDIT AND EVIDENCE',
        'The original 256 decisions and the isolated corrected 32 Doyle decisions replay. The active comparison uses 256 decisions. '
        'Original native qLogEI reconstruction error <= 1.154e-10. The active comparison uses 32 independent '
        'gpt-6.1-terra medium requests, with initial responses shared between arms; 36 requests were actually executed when including '
        'the four superseded Doyle requests. '
        '7425 protected historical artifacts retain their hashes. No operators compile inside the decision loops. '
        'A dispatch-only Windows junction normalization is recorded; scientific runner sources remain frozen. '
        'Token usage and provider-side execution attestation are unavailable.','',
        'unified_results.csv contains all five methods. posthoc_decisions.csv contains every same-prefix native comparison. '
        'posthoc_hypothesis_effects.csv records every point change after removing one current expert. '
        'posthoc_feedback_effects.csv and posthoc_refresh_effects.csv contain the controlled one-step diagnostics. '
        'posthoc_claim_validation.csv compares all submitted scopes against the finite pool after completion, with exact nonfocus '
        'matches where possible; unmatched regional means are descriptive and confounded. knowledge_versions.csv and the immutable '
        'run request folders preserve prompts, raw responses and hypotheses. audit.json and posthoc_scope.json specify validation and scope.']
    feedback=pd.read_csv(s.REPORT/'posthoc_feedback_effects.csv');refresh=pd.read_csv(s.REPORT/'posthoc_refresh_effects.csv')
    text+=['','FEEDBACK-ONLY POINT CHANGES (POST HOC, SAME ACTUAL PREFIX)',
        f'{len(feedback)}/128 selections changed; {(feedback.immediate_best_gain>1e-10).sum()} positive, {(feedback.immediate_best_gain< -1e-10).sum()} negative immediate best-value changes.',
        'Task | Query | Actual value | Without predictive updates | Immediate best-value gain']
    for row in feedback.itertuples():text.append(f'{row.task} | {row.query} | {row.selected_raw:.6f} | {row.without_raw:.6f} | {row.immediate_best_gain:+.6f}')
    text+=['','REFRESH-ROUND SUMMARY',
        f'{len(refresh)} noninitial calls; {int(refresh.changed.sum())} immediate point changes; {(refresh.immediate_best_gain>1e-10).sum()} positive, {(refresh.immediate_best_gain< -1e-10).sum()} negative immediate best-value changes.']
    effects=pd.read_csv(s.REPORT/'posthoc_hypothesis_effects.csv')
    effects=effects[(effects.arm=='adaptive')&(effects.immediate_best_gain.abs()>1e-10)]
    text+=['','EXPERT-REMOVAL EVENTS WITH NONZERO IMMEDIATE BEST-VALUE EFFECTS',
        'Effects are not additive; IDs map to knowledge_versions.csv.',
        'Task | Query | Hypothesis | Actual | Without expert | Best-value gain | Inside nonfocus context']
    for row in effects.itertuples():text.append(f'{row.task} | {row.query} | {row.id} | {row.full_raw:.6f} | {row.without_raw:.6f} | {row.immediate_best_gain:+.6f} | {row.selected_inside_nonfocus_context}')
    (s.REPORT/'comparison_and_mechanisms.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
    md=['# Active comparison after the Doyle MTBD identity correction','',
        'Same seed, inputs, GP, four initial observations and sixteen queries. Doyle uses the necessary corrected rerun; the other seven tasks are unchanged.','',
        '[Full English mechanism analysis](comparison_and_mechanisms.txt) · [All five methods](unified_results.csv) · [Active source selection](active_comparison.json)','',
        '| Task | Pure BO AUC / best | Previous kernel F | Grounded fixed | Grounded adaptive |','|---|---:|---:|---:|---:|']
    for task in s.SHORT.values():
        f=frame[frame.task==task].set_index('method')
        md.append('| '+task+' | '+' | '.join(f'{f.loc[m,"SearchAUC"]:.6f} / {f.loc[m,"final_best"]:.6f}' for m in ('pure_BO','previous_kernel_F','fixed','adaptive'))+' |')
    md+=['','![SearchAUC differences from matched BO](SearchAUC_comparison.png)','',
        'Each value retains its task objective units. The post hoc oracle lookups are diagnostics only and never enter the decision agents. The original incorrect-alias Doyle runs remain archived.']
    (s.REPORT/'analysis.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='COMPLETE',metrics_rows=len(frame),unique_structural_hypotheses=len(claim),
        report=str(s.REPORT/'comparison_and_mechanisms.txt'))))


if __name__=='__main__':main()
