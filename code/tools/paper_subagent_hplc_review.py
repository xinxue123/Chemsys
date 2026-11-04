"""Read-only, source-bound HPLC panel and lifecycle timing review.

No model fitting, outcome queries, response generation or policy changes.
"""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'paper/v12_subagent/data'
OUT = ROOT / 'experiments/reports/paper_v12_subagent'


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    paths = [DATA / 'evidence.json', DATA / 'round_decisions.csv',
             DATA / 'forecast_updates.csv', OUT / 'audit.json',
             OUT / 'contrast_trace_review.json']
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}
    evidence = json.loads(paths[0].read_text(encoding='utf-8'))
    audit = json.loads(paths[3].read_text(encoding='utf-8'))
    contrast = json.loads(paths[4].read_text(encoding='utf-8'))
    if audit['status'] != 'PASS' or evidence['completed_trajectories'] != 68 or contrast['evidence_cutoff'] != 68:
        raise ValueError('Complete audited cohort is required')
    rows = [r for r in evidence['rows'] if r['task'] == 'HPLC']
    rr, ff = [], []
    for path, target in ((paths[1], rr), (paths[2], ff)):
        with path.open(encoding='utf-8-sig', newline='') as handle:
            target.extend(r for r in csv.DictReader(handle) if r['task'] == 'HPLC')
    by_cell = {(r['initialization_seed'], r['method']): r for r in rows}
    if len(rows) != 15 or len(rr) != 240 or len(ff) != 63:
        raise ValueError('Complete HPLC panel denominator')
    memory_rounds = [r for r in rr if r['method'] in ('V12', 'No KF', 'No refresh')]
    if len(memory_rounds) != 128:
        raise ValueError('Memory-enabled round denominator')
    if sum(int(r['expected_forecasts']) for r in memory_rounds) != len(ff) or sum(int(r['sidecar_warning_count']) for r in memory_rounds):
        raise ValueError('Missing forecast or warning requires review')
    for r in rows:
        path = ROOT / r['source']
        if digest(path) != r['source_sha256']:
            raise ValueError('State changed')
        if r['cohort'] == 'codex_subagent' and audit['completed_input_sha256'].get(r['source']) != r['source_sha256']:
            raise ValueError('State not bound by audit')
        hashes[r['source']] = r['source_sha256']
    for r in rr:
        if digest(ROOT / r['commit_source']) != r['commit_file_sha256']:
            raise ValueError('Commitment changed')
        if r['cohort'] == 'codex_subagent' and audit['completed_input_sha256'].get(r['commit_source']) != r['commit_file_sha256']:
            raise ValueError('Commitment not bound by audit')
        hashes[r['commit_source']] = r['commit_file_sha256']
    full = [r for r in rr if r['method'] == 'V12']
    full_f = [r for r in ff if r['method'] == 'V12']
    recovery = [r for r in full if int(r['automatic_resumes'])]
    if [(r['initialization_seed'], r['query']) for r in recovery] != [('1', '10')]:
        raise ValueError('Recovery event changed')
    seed1_f = {int(r['query']): r for r in full_f if r['initialization_seed'] == '1'}
    if not (seed1_f[5]['status_after'] == 'paused' and seed1_f[7]['status_before'] == seed1_f[7]['status_after'] == 'paused'
            and int(seed1_f[7]['success']) and all(not int(seed1_f[q]['success']) and float(seed1_f[q]['Brier_gain']) > 0 for q in (9, 10))):
        raise ValueError('Recovery and earlier improvement timing changed')
    hcontrasts = [r for r in contrast['contrasts'] if r['task'] == 'HPLC']
    removal = [r for r in hcontrasts if r['comparison'] in ('KF_removal', 'refresh_removal')]
    if len(hcontrasts) != 14 or len(removal) != 4 or any(r['exact_initial_scope_structure_match'] for r in removal):
        raise ValueError('Removal knowledge comparison changed')
    firsts = {(r['seed'], r['comparison']): r['first_different_query'] for r in removal}
    if firsts != {(0, 'KF_removal'): 8, (0, 'refresh_removal'): 5,
                  (3, 'KF_removal'): 10, (3, 'refresh_removal'): 4}:
        raise ValueError('Removal first divergences changed')
    if by_cell[(2, 'V12')]['SearchAUC'] >= by_cell[(2, 'BO')]['SearchAUC'] or by_cell[(2, 'V12')]['final_best'] <= by_cell[(2, 'BO')]['final_best']:
        raise ValueError('Endpoint versus efficiency contrast changed')
    result = dict(
        status='PASS_COMPLETE_HPLC_DESCRIPTIVE_REVIEW', evidence_cutoff=68, audit_cutoff=68,
        generator_sha256=digest(Path(__file__)), input_sha256=hashes,
        task='HPLC', cells=len(rows), round_count=len(rr), forecast_events=len(ff),
        full_V12_forecasts=len(full_f), expected_forecasts=sum(int(r['expected_forecasts']) for r in memory_rounds),
        sidecar_warnings=0, method_cells=dict(Counter(r['method'] for r in rows)),
        summary=[r for r in contrast['task_panels'] if r['task'] == 'HPLC'],
        metrics=rows, contrasts=hcontrasts, rounds=rr, forecasts=ff,
        interpretations=[
            'Primary seeds 1-3: V12/BO SearchAUC wins 2 and loses 1; ratio of means +9.412309502%. Final best wins 2 and ties 1. Seed 1 contributes a +60.40371455% AUC contrast; seed 2 loses 9.35052331% AUC but gains 252.06323 a.u. final area.',
            'Seed 0: full V12/no-KF first differs at query 8 after three full-policy updates. Full V12 has +0.117152432% AUC but -21.44165 a.u. final best. Against no-refresh, first difference is query 5; full V12 gains 2.106951817% AUC and 142.24170 a.u. final best. Explicit LLM retraction at query 4 is distinct from automatic KF retraction.',
            'Seed 1: full V12 pauses H001 after query 5 and recovers after query 10. The recovery follows two positive Brier gains on non-improvements at queries 9 and 10. The large improvement at query 7 happened while paused. The query-7 retain proposal was rejected because its evidence did not name completed own queries. No accepted productive-support citation occurs before query 15, which does not improve the incumbent.',
            'Seed 2: full V12 reaches its final best at query 13, versus query 5 for BO. Its last applicable forecast occurs at query 8. The better final maximum does not negate its lower SearchAUC or establish a causal late knowledge effect.',
            'Seed 3: full V12/no-KF selections match through query 9; full V12 improves to its final maximum at query 10 and then pauses H001 for a negative relative gain on that improving experiment. Full/no-refresh selections already differ at query 4, before that pause; no-refresh is better in both AUC and final best. The pause cannot explain the earlier divergence.',
            'All four HPLC removal pairs have different exact initial scope structures. Initial experimental observations are matched, but initial knowledge is independently generated. Timing and metric comparisons describe complete policies, not isolated causal contributions.'
        ],
        claim_boundary='No retuning, additional queries or causal attribution. Same-history references are not independently executed policy counterfactuals. Forecasts sharing an outcome are not independent replicates.')
    if any(digest(ROOT / name) != value for name, value in hashes.items()):
        raise ValueError('Input changed during review')
    (OUT / 'hplc_panel_review_68.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('status', 'cells', 'round_count', 'forecast_events', 'full_V12_forecasts', 'sidecar_warnings')}, indent=2))


if __name__ == '__main__':
    main()
