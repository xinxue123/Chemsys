"""Static scientific figure and English evidence note from read-only diagnostics."""
from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/reports/hypothesis_scope_loop_diagnosis_20261105'


def main():
    frame=pd.read_csv(OUT/'rounds.csv');summary=pd.read_csv(OUT/'summary.csv')
    fig,axs=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    for ax,short,title in zip(axs.flat,('c1','c2','doyle','hplc'),('Reizman Suzuki Case 1','Reizman Suzuki Case 2','Doyle Buchwald-Hartwig','HPLC')):
        f=frame[frame.task==short];row=summary[summary.task==short].iloc[0]
        ax.plot(f['query'],f.control_best,label='Same LLM first point + BO',color='#be7c19',linewidth=2.7,linestyle='--')
        ax.plot(f['query'],f.best,label='Hypotheses + KF + refresh',color='#087f8c',linewidth=1.8,marker='.',markersize=4)
        ax.fill_between(f['query'],f.best,f.control_best,where=f.best<f.control_best,color='#b55353',alpha=.13,interpolate=True)
        ax.fill_between(f['query'],f.best,f.control_best,where=f.best>f.control_best,color='#087f8c',alpha=.10,interpolate=True)
        ax.set_title(f'{title}  |  SearchAUC change {row.delta_SearchAUC:+.3f}',fontsize=11)
        ax.set_xlabel('Paid query');ax.set_ylabel('Best observed yield (%)' if short!='hplc' else 'Best peak area (arbitrary units)')
        ax.set_xticks([1,4,8,12,16,20]);ax.set_xlim(.7,20.5);ax.grid(alpha=.18)
        if short=='c1':
            ax.annotate('q6-11: delayed high yields',xy=(8,53.4123),xytext=(4,63),fontsize=9,arrowprops=dict(arrowstyle='->',color='#555'))
        elif short=='c2':
            ax.annotate('Refreshed scope: q4',xy=(4,23.3325),xytext=(1,30),fontsize=9,arrowprops=dict(arrowstyle='->',color='#555'))
            ax.annotate('BO best: q10 vs q13',xy=(10,40.0853),xytext=(10,33),fontsize=9,arrowprops=dict(arrowstyle='->',color='#555'))
        elif short=='doyle':
            ax.annotate('KF releases scope; q3 = 93.10%',xy=(3,93.1012),xytext=(4.2,65),fontsize=9,arrowprops=dict(arrowstyle='->',color='#555'))
        else:ax.text(.47,.32,'All 20 choices identical',transform=ax.transAxes,fontsize=10)
    handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=2,frameon=False)
    fig.savefig(OUT/'paired_paths.png',dpi=180);fig.savefig(OUT/'paired_paths.pdf');plt.close(fig)
    lines=['HYPOTHESIS-SCOPE LOOP: READ-ONLY MECHANISM DIAGNOSIS','',
        'Scope: Case 1, Case 2, Doyle, HPLC. Each comparison fixes the identical LLM-chosen first experiment, seed0, pool, native GP recipe and 20 paid queries. The new arm changes scopes/feedback/refresh. No experiments were rerun. The diagnostic reads saved posterior arrays and uses only outcomes already present in saved trajectories.',
        '',summary.to_string(index=False),'',
        'CASE 1: CONSTRAINTS DIVERT EARLY LEARNING; A RECOMMENDATION GATE ALSO REJECTS A USEFUL REGION',
        'The initial SPhos/high-temperature scope changes q2 to p01913 (0.245362%). KF reliability drops .400 -> .112826, pausing the scope. A q4 refreshed hot/long-residence region changes selection to p01951 (33.348198%); reliability drops .400 -> .156042. Both choices leave the incumbent at49.016312%. The original q2 value is0%, so the loss cannot be explained simply by a worse q2 objective.',
        'The two changed early input/output pairs change the GP fit. At q6, p04671 (71.305290%, P1-L7 at600s/110C/2.5mol%) is maximal qLogEI in the control, but has acquisition gap .09177085 in the new arm; two candidates rank strictly ahead. The new arm instead selects another SPhos condition (45.567333%). It reaches p04671 only atq11. These are saved native posteriors, not knowledge-edited means.',
        'At q4 the LLM preferred p01031 (XPhos,600s/110C/2.5mol%), but the broad temperature/time scope lets native qLogEI choose p01951 with SPhos. The same p01031 is proposed atq7, but qLogEI picks p02591 (RuPhos,53.412315%). Thus the intended matched catalyst experiment and the executed regional test differ. This follows the registered selector; it is not an execution deviation.',
        'After q7 improvement the H3 reliability is .772432. Before q8, its fixed forecast52 +/-30 is compared with the new incumbent53.412315: probability .481226 falls below .5 and pauses the positive scope. Ignoring only this scope permission/credibility layer at the saved q8 prefix selects p01031, known from q9 to yield67.139450%. The actual q8 point yields32.916119%. This is a one-step missed opportunity, not a demonstrated alternate trajectory AUC.',
        'The q9 in-scope BO observation raises H3 credibility again to .795718, but its forecast remains52 +/-30 and positive guidance stays paused. KF updates the Brier-skill state, not the raw regional mean/SD. For a normal forecast, p(improve)>=.5 is exactly equivalent to regional mean>=incumbent, irrespective of SD. A valid region can therefore become ineligible immediately after its own success without a new chemical contradiction. q8-9 refresh is blocked by cooldown; q10-14 is blocked by the reserved late-call rule.',
        'The cumulative-best deficit in q6-11 contributes -4.790157 to SearchAUC; q12-19 recovers +.214055, and q20 adds +.053634, for -4.522468 total. The final best improves from88.496315% to89.569000%, but too late to recover earlier AUC loss. The q15 avoidance refresh changes none of q15-20 choices.',
        '',
        'CASE 2: A USEFUL q4 REFRESH, FOLLOWED BY NECESSARY EXIT FROM BAD GUIDANCE',
        'The initial SPhos scope changes q2 to9.893713%, below the19.700253% incumbent, and KF drops to .108426. Atq4 a new XPhos/RuPhos scope70-110C chooses the LLM preferred p01031, yielding23.332539%. Reliability rises to .742328. Subsequent forced q5/q6 choices give23.142725% and21.669279%; reliability declines to .485867 and .295159. The q6 pause is caused by two negative relative forecasts AND two guidance-caused nonimprovements, not by crossing the .2 reliability cutoff.',
        'Atq7 the LLM switches to P2-XPhos; p01551 yields16.389189%, and KF drops to .162868. q8-14 then use unrestricted BO. Atq10 BO selects p03631 (P1-PCy3,600s/110C/2.5mol%) and observes40.085316%, three queries earlier than the matched-start control. Keeping only the q7 KF-paused scope active at this SAME q10 prefix excludes p03631 and selects p01544 instead (outcome not read by this diagnostic). Thus the exit mechanism is selection-relevant, while full-trajectory module causality remains untested.',
        'The q1 SPhos, q4 XPhos and q5 RuPhos inputs have identical600s/110C/2.5mol% settings; their yields19.700253,23.332539,23.142725 support the specific matched ligand contrast. H2 explicitly predicts exceeding the original19.70%; q4-6 all do. q5-6 failing to improve the NEW incumbent23.332539 does not falsify that original chemical claim. The program is withdrawing further optimization permission, based on a stricter moving target. Reliability .742 -> .486 -> .295 does not gradually soften the hard region before the separate two-failure rules pause it.',
        'The AUC increment decomposes exactly into +1.089686 fromq4-9 and +2.512917 fromq10-12; other rounds contribute0. The latter portion is69.75% of the total +3.602602. The best point is found by native BO after suspension, not directly proposed by the LLM. q15 avoidance changes no remaining point.',
        '',
        'DOYLE: INITIAL NEAR-TIE CHOICE PLUS KF RELEASE, NOT A DIRECT BENEFIT OF LATER REFRESH',
        'Both arms start atp00669,47.679377%. The initial favour scope fixes the para-CF3 aryl iodide. It changes q2 to p01237,40.302441%, while native BO would choose p03325,41.677715%. The acquisition gap is4.09e-14; all3954 available points are within1e-12 of the maximum. The chosen objective itself is not better than the control objective.',
        'The failed improvement forecast lowers H1 reliability from .400 to .106306, so q3 can leave this aryl-iodide scope. At the saved actual q3 prefix, reinstating only KF-paused eligibility selects p00309; that input is already measured in the control at32.148415%. The actual unrestricted choice p00672 yields93.101175%. This demonstrates a concrete same-prefix positive contribution of the KF gate at q3, but does not isolate its complete-trajectory SearchAUC effect.',
        'p00672 preserves the first point\'s ligand, additive and base and changes only the aryl halide to3-iodopyridine. Native BO reaches94.549813% atq6. This benchmark gain includes choosing a different substrate within the permitted pool, not solely improving conditions for a fixed substrate.',
        'p00672 is a maximal-qLogEI point in BOTH q3 histories. It ties with14 available points in the new arm and35 in the control. Its new-arm GP mean is46.527879%, not a prediction near93%. The early changed prefix alters the top tie set under the same feature-keyed random order. The magnitude is therefore tied to this initialization and tie realization, and does not prove stable chemical targeting of93% yields.',
        'The q4 refreshed 3-iodopyridine positive scope has mean85 below the93.101175 incumbent and is immediately paused (p=.342717). It gets three predictive feedback updates but changes no point. The q7 avoidance and q15 retained avoidance also change no point. q4,7,15 refreshes have no direct action increment on this observed path.',
        '',
        'HPLC: UNCERTAIN SCALE, HARD RECOMMENDATION ADMISSION, AND REGIONS THAT ALREADY CONTAIN THE BO CHOICE',
        'The initial large-loop hypothesis has forecast1 +/-1000000 peak-area units. After the first observed92.857560, its improvement probability is .499963354, so the .5 recommendation gate pauses it beforeq2 even with initial KF reliability .4. This is the exact implemented inequality, not floating-point noise. An uncertain regional mean below the incumbent fails admission even with an enormous SD.',
        'Atq4, wait8-10s leaves211 candidates and includes the native BO winner HPLC_00710. Its30.872940 observation drops reliability .400 -> .162016. Atq5 unrestricted BO finds1827.863530. The paused rule receives positive Brier skill, reliability rises to .514668, although the raw95 +/-80 forecast misses the actual objective by21.66SD. KF measures binary improvement skill, not absolute-objective forecast accuracy or a matched wait-time causal effect; those quantities must not be conflated.',
        'Atq7, flow1.0-1.6mL/min leaves296 candidates and again contains the native winner,52.737770. One failed forecast drops reliability to .137951. All q8-14 native winners are also inside this flow interval, so bypassing the KF gate would not change those choices. q10-14 refresh events are deferred to reserve a late call.',
        'Atq15, avoiding120-150Hz push frequencies leaves576 candidates and contains every remaining native BO winner. q16 reaches2097.936040 independently of this avoidance rule; the avoidance version receives no in-scope measurements/KF updates. All20 selected inputs exactly match the LLM-first-only control. The11 KF updates and three refreshes have zero action increment on this observed trajectory.',
        '',
        'EXECUTION VS MECHANISM',
        'The existing160-decision replay is consistent; this read-only diagnostic independently checks76 saved-array choices in the four tasks and27 KF updates, including raw-unit GP/LLM improvement probabilities. It verifies the1333-file completion manifest and9435 protected historical artifacts. No GP construction/fitting, LLM request or oracle read is allowed in this script.',
        'A previously disclosed HPLC q4 scratch-export change of3.75e-10mL/min does not alter the sealed response, hypothesis, forecast or any selected point. It is an export custody incident, not an explanation of performance. Original files are preserved.',
        'The selector has no global qLogEI-gap gate. Its active restriction is the geometric hypothesis scope. KF reliability is continuous internally, but selection uses a hard threshold and permissions; it is not a continuous qLogEI weight. Region forecasts and half-probability admission can suppress useful regions. The free-text falsification field is stored and supplied to the next LLM, but the program does not execute that statement: it applies generic improvement/Brier/residual/nonimprovement triggers. Those triggers do not test the literal conditional chemical contrast.',
        'Doyleq3 is also a reminder that GP uncertainty is not necessarily calibrated early: its selected point has predicted46.527879 +/-2.642961% but observed93.101175%. Both arms underestimated this already-queried point; the logs support a fortunate high-value discovery and different ranking/tie geometry, not an accurate quantitative forecast of the93% result.',
        'Most refreshes define different finite-pool scopes and start with reliability .4. Only identical scope+direction inherits KF evidence. Doyleq15 inherits an avoidance version with zero scored observations. This is consistent with the implementation, but limits reusable learned knowledge. No explicit probation/retest is exercised in these four paths.',
        '',
        'SUPPORTED NEXT DESIGN PRIORITIES (NOT IMPLEMENTED OR TESTED HERE)',
        'Preserve the KF exit that enables Doyleq3 and Case2q10. Repair the mismatch between matched preferred experiments and broad-scope execution, so feedback evaluates the reason that actually caused selection. Replace absolute regional-mean versus incumbent admission with a local, GP-relative value test that can express useful improvement probabilities below .5, while retaining predictive-only/avoidance states for accurate failure forecasts. Separate numeric forecast calibration, regional chemical evidence and recommendation utility; HPLC needs an honest unknown-scale option or measured-scale normalization. Refresh effectiveness should be counted by changed admissible top choices and verified tests, not by calls or text changes.',
        '',
        'LIMIT: Static counterfactuals hold the ACTUAL prefix, saved GP scores and LLM outputs fixed. Previously observed outcomes may describe the chosen alternative; unobserved alternatives remain unknown. These checks establish one-step mechanism relevance and accounting contributions, not full-trajectory KF/refresh AUC effects or multi-seed stability.']
    (OUT/'findings.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='COMPLETE',figure=str(OUT/'paired_paths.png'),findings=str(OUT/'findings.txt'))))


if __name__=='__main__':main()
