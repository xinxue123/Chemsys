"""Summarize saved-log gate diagnostics without changing any experiment."""
from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'experiments/reports/kernel_contrast_gate_diagnosis_20261104'


def main():
    rounds = pd.read_csv(OUT/'round_gate_details.csv',encoding='utf-8-sig')
    scenarios = pd.read_csv(OUT/'scenario_counts_all_tasks.csv',encoding='utf-8-sig').set_index('scenario')
    counts = pd.read_csv(OUT/'scenario_counts_by_task.csv',encoding='utf-8-sig')
    geometry = pd.read_csv(OUT/'pool_geometry.csv',encoding='utf-8-sig')
    predictions = pd.read_csv(OUT/'excluded_model_prediction_summary.csv',encoding='utf-8-sig')
    audit = json.loads((OUT/'audit.json').read_text(encoding='utf-8'))
    lines = [
        '# Why only two decisions changed: a saved-log gate audit','',
        'The main bottleneck is the combination of exact-contrast geometry and a small, strongly '
        'screened set of same-family GP fits. The native qLogEI budget does suppress some tests, '
        'but widening that budget alone does not activate five of the six unchanged tasks.','',
        'This analysis uses the existing 128 round records, saved candidate predictions and fitted '
        'model metadata. It performs zero GP fits, zero LLM calls and zero new experimental queries. '
        'All 128 recorded selections are reproduced from the saved arrays. Relaxed gates below '
        'are hypothetical decisions on the original measured prefixes, not optimization reruns '
        'or counterfactual SearchAUC estimates.','',
        '## What “two interventions” means','',
        'Round1 deliberately follows BO because no preceding fitted model exists; round16 follows '
        'the exact native BO maximum. These endpoints remove 16 of the 128 rounds from intervention '
        'opportunities. Among the remaining 112 rounds, three satisfy every contrast criterion: '
        'Case1 query4, Perera query7 and Doyle query7. Doyle selects the same candidate as BO, so '
        'only two selections differ. An unchanged selection is therefore not synonymous with a '
        'mechanism that never ran.','',
        'The previous first-failure logs are short-circuit classifications. '
        '“No contrast in budget” does not distinguish an exclusion by qLogEI from the absence '
        'of a contrast anywhere in the fixed pool. Likewise a zero reported gated disagreement '
        'can mean that no candidate survived an earlier gate; it is not an unconditional statement '
        'that every model predicts exactly the same value.','',
        '## Exact contrasts are structurally unavailable for much of the pool','',
        '| Task | Candidates | Candidates with any exact one-axis neighbor | Fraction |',
        '|---|---:|---:|---:|']
    for _,r in geometry.iterrows():
        lines.append(f'| {r.task} | {int(r.candidates)} | {int(r.points_with_any_exact_single_axis_neighbor)} | {r.percent_with_exact_neighbor:.2f}% |')
    lines += ['',
        'Every Suzuki pool has only 64 such points, all corners of the three continuous axes '
        'combined with the eight catalysts. The other 4,628 inputs have no exact one-axis partner. '
        'None of the four initial inputs is a corner. Until the trajectory measures a corner, '
        'the exact-contrast policy cannot act at all, irrespective of the acquisition budget.','',
        'HPLC is even more restrictive: each of its six coordinates has 1,007 distinct values '
        'among 1,007 candidates. There is no exact one-variable contrast anywhere in the pool. '
        'This is a policy-to-pool compatibility failure, not an acquisition-score computation error. '
        'The check uses the original 1e-8 matching tolerance and treats nominal labels only by equality.','',
        '## Which gate changes decision opportunities?','',
        'All other thresholds and saved fits are fixed. The counts below are hypothetical '
        'selections different from native BO on the same recorded prefixes. A changed count '
        'does not imply a better yield, and equal counts can still involve different candidates.','',
        '| Modification | Qualifying rounds | Decisions different from BO |',
        '|---|---:|---:|']
    labels = [
        ('Original gates','factorial_budget1_MLL1_single1'),
        ('Remove qLogEI budget only','factorial_budget0_MLL1_single1'),
        ('MLL gap 0.01 to 0.02','MLL_gap_0.02'),
        ('MLL gap 0.01 to 0.05','MLL_gap_0.05'),
        ('Remove MLL admission only','factorial_budget1_MLL0_single1'),
        ('Remove exact-contrast requirement only','factorial_budget1_MLL1_single0'),
        ('Lower axis-correlation threshold to zero','rho_gap_0'),
        ('Lower standardized disagreement from 0.5 to 0.25','disagreement_0.25'),
        ('Allow 5% mismatch plus MLL gap 0.05','near05_MLL_0.05'),
        ('Remove MLL and exact contrast, keep budget','factorial_budget1_MLL0_single0')]
    for label,key in labels:
        r = scenarios.loc[key]
        lines.append(f'| {label} | {int(r.qualifying_rounds)} | {int(r.changed_rounds)} |')
    lines += ['',
        'Removing the budget increases the different-from-BO count from two to four. The newly '
        'active rounds are Case1 query14 and Doyle query7. Perera query7 stays active but switches '
        'candidate. Thus counting additional intervention rounds alone understates all point changes. '
        'Removing exact matching alone also changes one already-active candidate without adding '
        'an intervention round.','',
        'The newly chosen Case1 query14 point has native logEI deficit 41.308, compared with '
        'the allowed 0.18484; Doyle query7 has deficit 4.398 versus 0.83178. The latter has only '
        'about 1.23% of the best exponentiated acquisition score. These are not near-boundary '
        'exclusions. Perera query7 would choose p00192 without the budget, whose deficit is 40.955; '
        'that point was later actually measured at 96.57%. This is a genuine retrospective example '
        'of a valuable point that the native GP rates too low, but it does not validate unrestricted '
        'disagreement selection or establish the resulting full trajectory.','',
        '## Why the other six tasks did not change','',
        '| Task | Logged rounds and decisive limitation |',
        '|---|---|',
        '| Case2 | Queries1–11 have no exact contrast anywhere in the pool. The first measured corner arrives at query11; query12 finally has five budget contrasts. At query12 the largest admitted correlation difference is0.01679 and the largest admitted predictive disagreement anywhere is0.4494, both below the declared thresholds. Queries14–15 similarly have near-identical admitted fits. Removing the budget, MLL gate, or exact gate individually adds no selection change. |',
        '| Case3 | Queries1–7 have no full-pool contrast. Queries8–10 have3/2/1 budget contrasts, but admitted predictive disagreements are tiny; the largest over all rounds is0.0580 versus the0.5 threshold. The budget does exclude contrasts at queries11–13 and15, but widening it still adds no intervention. |',
        '| Case4 | No measured prefix ever has an exact contrast anywhere in the pool. Queries1–7 also retain only one MLL-admitted fit; later admitted fits usually converge closely. Removing MLL alone cannot overcome the geometry. Relaxing both matching and admission creates opportunities, whereas widening qLogEI alone does not. |',
        '| Doyle | Every round has exact budget contrasts. Query7 has36 qualified candidates, and the policy runs; its chosen p01513 is already the BO reference. The measured68.26% favors the native forecast74.20% over an overconfident alternative86.45%. Relaxing MLL can change queries8,10,14; removing the budget changes query7 to another candidate. |',
        '| SnAr | Budget contrasts exist in every round; queries2–15 retain2–3 fits. These are effectively the same fitted solution: maximum axis-correlation difference is0.0001044 and maximum full-pool standardized mean disagreement is0.002314. No tested single-gate relaxation produces a new selection. There is little generated model diversity to exploit. |',
        '| HPLC | Every prefix has zero exact contrasts even in the full pool. Most alternative fits also agree: admitted full-pool standardized disagreement never exceeds0.01028. The different fit at query10 is rejected with an MLL loss near0.02467; removing both admission and exact matching could change that round, but either change alone cannot. |','',
        'Alternative fits are local optima of the same GP family from preceding fitted parameters. '
        'Two or three fit records are not necessarily two or three materially different explanations, '
        'nor are they hyperparameter-posterior samples or independent chemistry hypotheses. '
        'SnAr illustrates this limitation especially clearly.','',
        '## Is the 0.01 MLL screen too severe?','',
        'It is conservative: relaxing to0.02 adds one different-from-BO round; relaxing to0.05 adds '
        'five. However, more disagreements are not necessarily useful disagreements. The following '
        'check compares each rejected fit with the same-round highest-MLL fit using the locked '
        'predictive density on the actual next measured point. Repeated fits/prefixes are not '
        'independent samples.','',
        '| Rejected MLL loss | Model-rounds | Better next-point predictive score | Worse |',
        '|---|---:|---:|---:|']
    for _,r in predictions.iterrows():
        lines.append(f'| {r.loss_bin} | {int(r.model_rounds)} | {int(r.better)} | {int(r.worse)} |')
    lines += ['',
        'Among rejected losses between0.01 and0.05, eight model-rounds predict the actual next '
        'point better and22 worse. Case4 query15 is a useful missed example: a fit rejected by '
        'only0.01175 MLL predicts the measured point better. These results support neither '
        'unconditional rejection nor unconditional admission. They also do not evaluate the '
        'unmeasured proposed alternatives.','',
        '## Implications before another optimization run','',
        'First make the contrast criterion compatible with the fixed candidate geometry. '
        'For irregular continuous pools, use an explicitly penalized local match or a joint-condition '
        'test; do not describe it as an experimentally identified one-variable effect. '
        'This is especially necessary for HPLC and interior Suzuki inputs.','',
        'Second, create genuinely distinct, defensible kernel explanations instead of assuming '
        'multiple warm starts provide model uncertainty. Identical optimizer solutions are a valid '
        'reason to follow BO and should not be artificially turned into disagreement.','',
        'Third, study an MLL penalty or admission sensitivity together with prospective predictive '
        'evidence. The logs show that0.01 is restrictive and that some rejected fits are poorly '
        'calibrated. Merely raising the threshold to increase trigger frequency is not justified.','',
        'Do not make broad qLogEI-budget expansion the first change. It is a secondary bottleneck '
        'in this run, and several excluded high-disagreement points have extreme acquisition deficits. '
        'A native-GP error can still exclude a valuable point, as Perera demonstrates; resolving '
        'that issue needs a separately declared decision-value test.','',
        'No selection implementation discrepancy was found in this saved-array audit. The main '
        'issues are policy applicability, model-generation limitations and ambiguous first-failure '
        'diagnostic labels. All original code, trajectories, caches and outcomes remain unchanged. '
        'The hypothetical decisions cannot recover discarded model ancestors or reproduce future '
        'fits after a changed observation, so no performance or stability claim is made.','',
        'Evidence: round_gate_details.csv; pool_geometry.csv; saved_prefix_scenarios.csv; '
        'scenario_counts_by_task.csv; excluded_opportunities.csv; fitted_model_pairs.csv; '
        'excluded_model_prediction_checks.csv.']
    (OUT/'analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    assert int(scenarios.loc['factorial_budget1_MLL1_single1','changed_rounds']) == 2
    assert int(scenarios.loc['factorial_budget0_MLL1_single1','changed_rounds']) == 4
    assert int(scenarios.loc['factorial_budget1_MLL0_single1','changed_rounds']) == 16
    receipts = {f.relative_to(ROOT).as_posix():hashlib.sha256(f.read_bytes()).hexdigest()
        for f in OUT.iterdir() if f.is_file() and f.name!='report_artifact_hashes.json'}
    receipts[Path(__file__).relative_to(ROOT).as_posix()] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT/'report_artifact_hashes.json').write_text(json.dumps(receipts,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='PASS_GATE_DIAGNOSIS_REPORT',rounds=len(rounds),
        new_GP_fits=0,new_experimental_queries=0,new_LLM_calls=0,
        report=str(OUT/'analysis.md'),scenario_evaluations=audit['scenario_evaluations'])))


if __name__=='__main__':
    main()
