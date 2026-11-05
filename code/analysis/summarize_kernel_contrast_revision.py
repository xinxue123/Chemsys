"""Separate repaired mechanisms from optimization outcomes for all staged arms."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from kernel_contrast_revision import study


def main():
    protocol=study.check(True)
    frame=pd.read_csv(study.REPORT/'results.csv',encoding='utf-8-sig')
    decisions=pd.read_csv(study.REPORT/'round_diagnostics.csv',encoding='utf-8-sig')
    verification=study.read(study.REPORT/'final_verification.json')
    metrics=[];mechanisms=[];cases=[]
    for task in study.SHORT.values():
        f=frame[frame.task==task].set_index('method')
        ours=f.loc['feedback'];bo=f.loc['pure_BO'];fixed=f.loc['soft_fixed']
        metrics.append(dict(task=task,BO_SearchAUC=bo.SearchAUC,old_SearchAUC=f.loc['original_kernel_contrast','SearchAUC'],
            geometry_SearchAUC=f.loc['geometry','SearchAUC'],hypotheses_SearchAUC=f.loc['hypotheses','SearchAUC'],
            fixed_SearchAUC=fixed.SearchAUC,feedback_SearchAUC=ours.SearchAUC,
            delta_vs_BO=ours.SearchAUC-bo.SearchAUC,relative_delta_vs_BO=100*(ours.SearchAUC/bo.SearchAUC-1),
            delta_vs_fixed=ours.SearchAUC-fixed.SearchAUC,BO_final=bo.final_best,feedback_final=ours.final_best,
            BO_first_own_final_query=bo.first_final_best_query,feedback_first_own_final_query=ours.first_final_best_query,
            actual_interventions=int(ours.interventions),feedback_changes_next_choice=int(ours.feedback_changes_choice)))
    for arm in study.ARMS:
        part=decisions[decisions.arm==arm];winners=frame[frame.method==arm]
        mechanisms.append(dict(arm=arm,decisions=len(part),qualified_rounds=int((part.qualified>0).sum()),
            changed_decisions=int(part.changed.sum()),feedback_changes_next_choice=int(part.feedback_changes_choice.sum()),
            SearchAUC_wins=int((winners.delta_SearchAUC>1e-9).sum()),SearchAUC_losses=int((winners.delta_SearchAUC< -1e-9).sum()),
            mean_relative_SearchAUC_gain_percent=float(np.mean([100*(row.SearchAUC/frame[(frame.task==row.task)&(frame.method=='pure_BO')].iloc[0].SearchAUC-1)
                for _,row in winners.iterrows()]))))
    for task in study.native.TASKS:
        short=study.SHORT[task];pool=study.native.load_task(ROOT,task,seed=0)
        state=study.read(study.folder_for(task,'feedback')/'state.json')
        for record in state['decisions']:
            d=record['decision']
            if not d['differs_from_BO']:
                continue
            j=pool.ix[d['selected_id']];m=d['matching']
            cases.append(dict(task=short,query=record['query'],candidate_id=d['selected_id'],
                raw_value=record['raw_value'],incumbent_before=record['incumbent_before'],incumbent_after=record['incumbent_after'],
                match_kind=m['kind'],match_quality=m['quality'],nuisance_RMS=m['nuisance_RMS'],
                anchor_id=m['anchor_id'],focus_axes=json.dumps(m['focus_axes']),
                score=d['score'],ESS=d['effective_expert_count'],
                fixed_initial_id=d['fixed_initial_selected_id'],feedback_changes_choice=d['feedback_weights_change_selection'],
                P0_interior_or_noncorner=bool(pool.group=='P0' and not all(abs(float(x))<1e-8 or abs(float(x)-1)<1e-8 for x in pool.features[j][:3]))))
    result=pd.DataFrame(metrics);mechanism=pd.DataFrame(mechanisms);interventions=pd.DataFrame(cases)
    for name,f in [('priority_comparison',result),('priority_mechanisms',mechanism),('feedback_interventions',interventions)]:
        f.to_csv(study.REPORT/(name+'.csv'),index=False,encoding='utf-8-sig')
    wins=int((result.delta_vs_BO>1e-9).sum());losses=int((result.delta_vs_BO< -1e-9).sum())
    fwins=int((result.delta_vs_fixed>1e-9).sum());flosses=int((result.delta_vs_fixed< -1e-9).sum())
    lines=['# Priority fixes: what changed, what worked, and what did not','',
        'The implementation bottlenecks were repaired and tested, but the complete adaptive '
        'strategy is not a demonstrated optimization improvement. It beats BO in four of eight '
        'tasks and loses in four under the fixed seed0 development setting. No historical '
        'method or result is replaced.','',
        '## Three priorities and four controlled stages','',
        '1. Geometry: input-only local and joint matches replace impossible exact matching. '
        'Matching errors reduce the disagreement score. Fixed-probe correlation gaps no longer '
        'exclude a candidate that has a meaningful predictive difference. These interface changes '
        'are bundled in G and are not claimed as a single-variable causal ablation.','',
        '2. Kernel explanations: H adds the native GP and two component-correlation sensitivity '
        'hypotheses per axis. This guarantees genuinely distinct covariance tests without '
        'changing the native GP or adding an optimizer fit. They are explicit sensitivity '
        'experts, not posterior samples or proven chemistry. Nonfocal parameters remain native.','',
        '3. Admission and feedback: S replaces the hard current0.01 MLL cutoff with fixed soft '
        'initial weights; F starts identically to S and updates all expert weights from actual '
        'precommitted forecasts. The current training likelihood is never multiplied into '
        'those weights again. S separates soft admission from the effect of online feedback.','',
        'The original qLogEI gate, candidate pools, primary GP, initialization and4+16 budget '
        'are unchanged. Each arm/task is a full independently updated trajectory. Every '
        'result is retained; no threshold was changed after seeing outcomes.','',
        '| Task | BO SearchAUC | Fixed soft weights | Adaptive feedback | Adaptive minus BO | BO final best | Adaptive final best |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for _,r in result.iterrows():
        lines.append(f'| {r.task} | {r.BO_SearchAUC:.5f} | {r.fixed_SearchAUC:.5f} | {r.feedback_SearchAUC:.5f} | '
            f'{r.delta_vs_BO:+.5f} | {r.BO_final:.5f} | {r.feedback_final:.5f} |')
    lines+=['','Raw values have task-specific units. SearchAUC remains the primary metric. '
        'An improved final best does not undo a lower SearchAUC; Case1 is a clear example.','',
        '| Stage | Changed decisions | SearchAUC better / worse than BO |',
        '|---|---:|---:|']
    for _,r in mechanism.iterrows():
        lines.append(f'| {r.arm} | {int(r.changed_decisions)} | {int(r.SearchAUC_wins)} / {int(r.SearchAUC_losses)} |')
    lines+=['',f'F versus S gives {fwins} better and {flosses} worse SearchAUC results. '
        'Thus online feedback moderates some harmful interventions but does not consistently '
        'improve the full search trajectory.','',
        '## Proof that the mechanisms actually participate','',
        f'The audit replayed all {verification["decision_replays"]} decisions with maximum score error '
        f'{verification["maximum_score_error"]}. It checked {verification["sidecar_focal_parameter_checks"]} '
        'sidecar states: only declared focal lengthscale entries differ from the current native '
        'state, including preservation of all nonfocal parameters and outcome-scaling buffers. '
        'The S/F forecasts, initial weights and first selections match on all eight tasks.','',
        f'F has {verification["adaptive_weight_updates"]} probability-weight changes above1e-12. '
        f'On its own observed prefixes, updated weights change {verification["feedback_changes_next_choice"]} '
        f'choices relative to frozen initial weights; {verification["feedback_follows_BO"]} of these '
        'follow the native BO reference. These same-prefix checks establish participation, '
        'not the causal SearchAUC effect of any individual update.','',
        f'F makes {len(interventions)} selections different from its own native BO references. '
        f'{int(interventions.P0_interior_or_noncorner.sum())} of its Suzuki interventions are '
        'outside the old corner-only exact-contrast subset. HPLC interventions are now executable '
        'joint-condition tests, not claims of an isolated causal effect.','',
        '## Remaining problem','',
        'Larger disagreement is not equivalent to an experiment that accelerates discovery. '
        'The added sensitivity hypotheses can be mathematically distinct yet scientifically '
        'unhelpful. Soft admission allows such explanations to affect early points; feedback '
        'can lower their weights only after paying for an observation. Those early changes '
        'also alter the subsequent native GP trajectory. The preserved qLogEI opportunity '
        'constraint does not guarantee realized-yield or path-level non-regression.','',
        'SnAr illustrates partial damage control: F improves over S, but both remain below '
        'BO. Doyle illustrates an opposite effect: S finds a higher final best than F, while '
        'feedback weights favor forecasts that do not preserve that search benefit. Predictive '
        'credibility and recommendation value remain different objectives.','',
        'The deterministic component-correlation targets are development assumptions. The '
        'current nuisance parameters are copied from the native fit rather than separately '
        'inferred for each expert, so these are not a calibrated posterior over kernel models. '
        'This is an important scientific limit, not a hidden optimizer convergence failure.','',
        'This revision remains an isolated research branch. It should not be promoted as a '
        'replacement for BO or the previous contrast method. A next decision-value experiment '
        'would need a predeclared connection between a tested hypothesis and an improved '
        'subsequent candidate choice, rather than further gate relaxation. No such extra '
        'trajectory is claimed or launched here.','',
        'Reproduction and every staged result are in analysis.md and results.csv. The protocol '
        'is under experiments/runs/kernel_contrast_revision_20261104/protocol.json. Model '
        'predictions, candidate tables, commitments, matching distances and weight transitions '
        'are retained per round. No API or LLM resources were used.']
    (study.REPORT/'priority_fixes.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    files=[f for f in study.REPORT.iterdir() if f.is_file() and f.name not in ('report_hashes.json',)]
    hashes={f.relative_to(ROOT).as_posix():study.sha(f) for f in files}
    for name in ('summarize_kernel_contrast_revision.py','verify_kernel_contrast_revision.py'):
        path=ROOT/'code/analysis'/name;hashes[path.relative_to(ROOT).as_posix()]=study.sha(path)
    study.write(study.REPORT/'report_hashes.json',hashes,replace=True)
    value=dict(status='PASS_PRIORITY_FIXES_SUMMARY',SearchAUC_wins_vs_BO=wins,SearchAUC_losses_vs_BO=losses,
        SearchAUC_wins_vs_fixed=fwins,SearchAUC_losses_vs_fixed=flosses,
        interventions=len(interventions),new_reruns_in_analysis=0,method_promoted=False)
    print(json.dumps(value));print(mechanism.to_string(index=False))


if __name__=='__main__':main()
