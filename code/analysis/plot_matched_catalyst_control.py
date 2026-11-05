"""Post-study figures and interpretation; never used by the selector."""
from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/reports/matched_catalyst_control_20261104'


def rows(name):
    with (OUT/name).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def main():
    metrics=rows('metrics.csv')
    pairs=rows('paired.csv')
    profiles=rows('mechanisms_by_prefix.csv')
    audit=json.loads((OUT/'audit.json').read_text(encoding='utf-8'))
    native=json.loads((OUT/'fresh_native_check.json').read_text(encoding='utf-8'))
    assert audit['status']==native['status']=='PASS'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(13.5,7.2),sharex=True,sharey=True)
    colors={'original_handoff':'#456b94','one_catalyst_contrast':'#c86b28'}
    for column,(seed,title) in enumerate([(0,'Prefix A: labels 0, 7'),(2,'Prefix B: labels 2, 3'),(6,'Prefix C: label 6')]):
        for method,label in [('original_handoff','Original handoff'),('one_catalyst_contrast','One catalyst contrast')]:
            row=next(r for r in metrics if r['seed']==str(seed) and r['method']==method)
            y=np.array(json.loads(row['yield_sequence']))
            for axis,values in [(axes[0,column],y),(axes[1,column],np.maximum.accumulate(y))]:
                axis.plot(range(1,9),values,marker='o',ms=4,lw=1.8,color=colors[method],label=label)
        axes[0,column].set_title(title)
        for axis in axes[:,column]:
            axis.axvspan(.6,2.4,color='#e5e5e5',alpha=.65,zorder=-2)
            axis.axvline(3,color='#666666',ls=':',lw=1)
            axis.set_ylim(-2,100)
            axis.set_xlim(.7,8.3)
            axis.set_xticks(range(1,9))
            axis.grid(axis='y',alpha=.18)
        axes[1,column].set_xlabel('Total experiment query')
    axes[0,0].set_ylabel('Observed yield (%)')
    axes[1,0].set_ylabel('Cumulative best yield (%)')
    axes[0,0].legend(frameon=False,loc='upper left')
    fig.suptitle('One matched catalyst experiment: early cost and later recovery',fontsize=15,y=.98)
    fig.text(.5,.017,'Queries 1-2 share the exact prefix; query 3 is replaced once; queries 4-8 use native qLogEI.\nThree physical branches, five paired labels. Archived original and BO controls remain unchanged.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.075,1,.95])
    fig.savefig(OUT/'paired_trajectories.png',dpi=190)
    plt.close(fig)

    s0=next(r for r in profiles if r['seed']=='0' and r['query']=='8')
    pool=json.loads((ROOT/'experiments/runs/suzuki_guarded/inputs/case1/pool.json').read_text(encoding='utf-8'))
    recovery_id=next(r for r in rows('rounds.csv') if r['seed']=='0' and
        r['method']=='one_catalyst_contrast' and r['query']=='8')['candidate_id']
    j=pool['ids'].index(recovery_id)
    recovery={}
    for tag,variant,job in [('old','llm_bo_handoff_20261104','handoff_s0'),
                          ('new','matched_catalyst_control_20261104','contrast_s0')]:
        with np.load(ROOT/'experiments/runs'/variant/'runs'/job/'rounds/08_GP.npz') as z:
            score,available=z['logei'],z['available']
            recovery[tag]=dict(candidate_id=recovery_id,GP_mean=float(z['mu'][j])*100,
                GP_sd=float(np.sqrt(z['variance'][j]))*100,
                acquisition_rank=1+int(np.count_nonzero(available&(score>score[j]))),
                EI_relative_max=float(np.exp(score[j]-score[available].max())))
    (OUT/'branch_A_recovery.json').write_text(json.dumps(recovery,indent=2)+'\n',encoding='utf-8')
    lines=['# Interpretation of the one-action catalyst-control experiment','',
        '**The registered substitution does not improve early-search efficiency overall. It improves terminal best in two physical branches but weakens SearchAUC8 in two of the three branches.**','',
        f"Five paired labels give mean SearchAUC8 change {audit['paired_SearchAUC8_increment']:+.4f} and terminal-yield change {audit['paired_final_increment']:+.4f} percentage points. There are only three distinct physical branches; their equal-weight SearchAUC8 change is {audit['three_branch_mean_increment']:+.4f}. Repeated labels must not be counted as independent replications.",'',
        '| Physical branch | SearchAUC8, old -> new | Terminal best, old -> new |','|---|---:|---:|']
    for seed,label in [(0,'A (labels0,7)'),(2,'B (labels2,3)'),(6,'C (label6)')]:
        r=next(r for r in pairs if r['seed']==str(seed))
        lines.append(f"| {label} | {float(r['old_SearchAUC8']):.2f} -> {float(r['new_SearchAUC8']):.2f} | {float(r['old_final']):.2f} -> {float(r['new_final']):.2f} |")
    lines.extend(['','## What the successful branch shows','',
        'In branch A, query3 changes from1.58% to20.69%. The new point tests P1-L4 at the same60 s,110 degC and0.5 mol% as the observed P1-L7 anchor at8.59%. These two catalysts differ by12.09 percentage points at that specific condition.',
        f"Before the last query, the category kernel correlation changes from {float(s0['old_category_correlation']):.6f} to {float(s0['new_category_correlation']):.6f}. The latent SD at the actual89.57% optimum changes from {float(s0['old_optimum_sd']):.2f} to {float(s0['new_optimum_sd']):.2f} percentage points. The candidate is still underestimated; increased uncertainty helps prevent the previous near-exclusion of other catalysts.",
        'The branch reaches88.44% on query8, versus46.51% terminal best originally. Its SearchAUC8 gain is only3.90 because recovery is late. This is a local example of a contrast altering the model and subsequent path, not proof of a general contrast strategy.',
        f"At the actual recovery candidate {recovery_id}, the old/new query8 GP means are {recovery['old']['GP_mean']:.2f}/{recovery['new']['GP_mean']:.2f}%, while latent SDs are {recovery['old']['GP_sd']:.3f}/{recovery['new']['GP_sd']:.3f} percentage points. Its native acquisition rank moves from {recovery['old']['acquisition_rank']} to {recovery['new']['acquisition_rank']}. The repaired branch selects it despite a LOWER mean because uncertainty is no longer nearly zero. See branch_A_recovery.json.",'',
        '## Why the other branches lose early-search efficiency','',
        'Branches B and C originally obtain67.14% on query3. The matched replacement is8.68% from P1-L6, compared with the observed P2-L1 anchor at7.76%, at600 s,110 degC and0.5 mol%. The measured catalyst difference is only0.92 percentage points at that condition. It supplies little evidence against equivalence of those two catalysts there; it does not establish how every other catalyst behaves.',
        'The replacement spends an early query on this low-yield pair and removes the original early high-yield observation. The subsequent fitted category correlation is near1 in both branches. Branch B later reaches78.25%, but its delayed discovery still lowers SearchAUC8 by8.64. Branch C ends at56.05% and loses18.12 SearchAUC8.',
        'Thus adding one nominal catalyst contrast is insufficient: the contrasted catalysts and condition need to expose a decision-relevant difference, and any benefit must repay the opportunity cost within the remaining budget. This experiment does not support geometric diversity or contrast counts as sufficient selection objectives.','',
        '## Calibration and kernel-retention are not interchangeable','',
        'At seven observations, near-merged category kernels increase from2/5 to3/5 labels, even while paired common-pool empirical interval coverage increases from78.6% to91.0%. Coverage gains are strongest in repaired branch A; they are not uniform across branches. The remaining categorical effects and uncertainty depend on which outcomes the actual contrast reveals.',
        'A kernel retaining catalyst distinctions is therefore a mechanism diagnostic, not an optimization score. The primary registered SearchAUC8 objective declines for this substitution. The point intervention changes both design geometry and the measured outcome; its total effect is identified locally, while separate effects of geometry, outcome level and later adaptive sampling are not isolated.','',
        '## Execution and scope','',
        'The first two observations and decision-prefix GP arrays match exactly. The third action alone is an explicit override, chosen without candidate outcomes using native acquisition and physical tie priorities. All exact contrasts fail the old acquisition budget, retaining26-28% of maximum exp(qLogEI), so this does not demonstrate an improvement inside the original budget constraint.',
        f"Independent reconstruction checks {native['checked_prefixes']} prefixes with {native['new_GP_fits']} fresh native fits. Maximum errors in mean, variance and qLogEI are all0; no old fitted model or score cache is used for computation. There is one native line-search warning on branch C/query8. The frozen native routine returns the same model and scores; convergence to the best marginal-likelihood solution is not asserted.",
        'All original source/result hashes remain valid. No new LLM call is made. The fixed native-BO suffix follows the original controls\' actual choice rule and removes downstream response randomness; it is not a test of a freshly refreshed full agent. Only five labels are feasible in the fixed pool, with three physical branches. This already inspected benchmark does not provide blind or cross-task validation.','',
        '![Observed and cumulative-best trajectories](paired_trajectories.png)','',
        'Primary tables and custody: README.md, paired.csv, mechanisms_by_prefix.csv, audit.json and fresh_native_check.json.'])
    (OUT/'INTERPRETATION.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    readme=OUT/'README.md'
    text=readme.read_text(encoding='utf-8')
    marker='## Completed interpretation and trajectories'
    if marker not in text:
        text+='\n'+marker+'\n\nSee [mechanistic interpretation](INTERPRETATION.md) for the branch-specific outcome and limitations.\n\n![Paired trajectories](paired_trajectories.png)\n'
        readme.write_text(text,encoding='utf-8')
    print(json.dumps(dict(status='COMPLETE',figure=str(OUT/'paired_trajectories.png'),
        interpretation=str(OUT/'INTERPRETATION.md'),new_LLM_calls=0,new_queries=0)))


if __name__=='__main__':
    main()
