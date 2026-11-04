"""Describe paired trajectories and event timing without causal attribution.

This reads audited exports only; it does not refit models, query outcomes,
dispatch language models or change an experimental policy.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'paper/v12_subagent/data'
OUT = ROOT / 'experiments/reports/paper_v12_subagent'


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def outcome_counts(rows, key):
    return dict(wins=sum(r[key] > 1e-9 for r in rows),
                ties=sum(abs(r[key]) <= 1e-9 for r in rows),
                losses=sum(r[key] < -1e-9 for r in rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    paths = [DATA / 'evidence.json', DATA / 'round_decisions.csv',
             DATA / 'forecast_updates.csv', OUT / 'audit.json']
    input_hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}
    evidence = json.loads(paths[0].read_text(encoding='utf-8'))
    audit = json.loads(paths[3].read_text(encoding='utf-8'))
    if audit['failures']:
        raise ValueError('Native audit has failures')
    rows = {r['source']: r for r in evidence['rows']}
    rounds = defaultdict(list)
    forecasts = defaultdict(list)
    for r in read_csv(paths[1]):
        rounds[r['source']].append(r)
    for r in read_csv(paths[2]):
        forecasts[r['source']].append(r)
    initial_notes = {}
    for source, r in rows.items():
        if digest(ROOT / source) != r['source_sha256']:
            raise ValueError('State changed: ' + source)
        if r['cohort'] == 'codex_subagent' and audit['completed_input_sha256'].get(source) != r['source_sha256']:
            raise ValueError('Unaudited state: ' + source)
        if len(rounds[source]) != 16 or len(r['selected_ids']) != 16:
            raise ValueError('Incomplete pair trajectory: ' + source)
        if [r['selected_id'] for r in sorted(rounds[source], key=lambda x: int(x['query']))] != r['selected_ids']:
            raise ValueError('Exported point order differs from the trajectory')
        commit_path = (ROOT / source).parent / 'rounds/01_commit.json'
        commit_name = commit_path.relative_to(ROOT).as_posix()
        commit_sha = digest(commit_path)
        if r['cohort'] == 'codex_subagent' and audit['completed_input_sha256'].get(commit_name) != commit_sha:
            raise ValueError('Unaudited first commitment: ' + commit_name)
        input_hashes[commit_name] = commit_sha
        notes = json.loads(commit_path.read_text(encoding='utf-8')).get('hypotheses_at_commit', [])
        initial_notes[source] = [
            {key: note.get(key) for key in ('claim', 'effect', 'focus', 'scope', 'reference_id')}
            for note in notes]
    contrasts = []
    for pair in evidence['paired']:
        left, right = rows[pair['source']], rows[pair['control_source']]
        if left['initial_ids'] != right['initial_ids'] or left['values'][:4] != right['values'][:4]:
            raise ValueError('Unmatched initial observations')
        differences = [q for q, (a, b) in enumerate(zip(left['selected_ids'], right['selected_ids']), 1) if a != b]
        first = differences[0] if differences else None
        identical_prefix = first - 1 if first else 16
        if left['values'][4:4 + identical_prefix] != right['values'][4:4 + identical_prefix]:
            raise ValueError('Equal selected prefix has different outcomes')
        item = {k: pair[k] for k in ('task', 'label', 'seed', 'comparison', 'method', 'control',
                                    'SearchAUC', 'control_SearchAUC', 'AUC_delta', 'AUC_percent',
                                    'final_best', 'control_final_best', 'best_delta', 'source', 'control_source')}
        item.update(first_different_query=first, identical_selected_prefix=identical_prefix,
                    differing_query_positions=len(differences),
                    common_selected_ids=len(set(left['selected_ids']) & set(right['selected_ids'])),
                    equal_final_best=left['final_best'] == right['final_best'])
        left_notes, right_notes = initial_notes[left['source']], initial_notes[right['source']]
        scope_fields = ('effect', 'focus', 'scope')
        item['method_initial_notes'] = len(left_notes)
        item['control_initial_notes'] = len(right_notes)
        item['exact_initial_claim_scope_match'] = left_notes == right_notes
        item['exact_initial_scope_structure_match'] = (
            [{key: note[key] for key in scope_fields} for note in left_notes] ==
            [{key: note[key] for key in scope_fields} for note in right_notes])
        for prefix, row in (('method', left), ('control', right)):
            source = row['source']
            ff = forecasts[source]
            numeric = [f for f in ff if int(f['numerical_KF_update_applied'])]
            rr = rounds[source]
            item[prefix + '_forecast_events'] = len(ff)
            item[prefix + '_numeric_update_events'] = len(numeric)
            item[prefix + '_first_numeric_update_query'] = min((int(f['query']) for f in numeric), default=None)
            item[prefix + '_memory_edit_queries'] = [int(r['query']) for r in rr if int(r['memory_edit_call'])]
            # Outcomes from query q cannot influence the already-sealed choice at q.
            item[prefix + '_numeric_updates_before_first_difference'] = (
                sum(int(f['query']) < first for f in numeric) if first else None)
            item[prefix + '_support_citations_before_first_difference'] = (
                sum(int(r['accepted_support_citations']) for r in rr if int(r['query']) < first) if first else None)
            item[prefix + '_source_sha256'] = row['source_sha256']
        contrasts.append(item)
    primary = [r for r in contrasts if r['comparison'] in ('V12_vs_BO', 'V12_vs_RBO', 'RBO_vs_BO') and r['seed'] in (1, 2, 3)]
    removal = [r for r in contrasts if r['comparison'] in ('KF_removal', 'refresh_removal')]
    if args.require_complete:
        if evidence['completed_trajectories'] != 68 or len(removal) != 12:
            raise ValueError('Full registered contrast matrix is incomplete')
        if Counter(r['comparison'] for r in primary) != {'V12_vs_BO': 24, 'V12_vs_RBO': 24, 'RBO_vs_BO': 24}:
            raise ValueError('Primary contrast denominator')
    summaries = []
    for name in sorted({r['comparison'] for r in primary + removal}):
        group = [r for r in primary + removal if r['comparison'] == name]
        summaries.append(dict(comparison=name, pairs=len(group),
                              SearchAUC=outcome_counts(group, 'AUC_delta'),
                              final_best=outcome_counts(group, 'best_delta'),
                              exact_initial_claim_scope_matches=sum(r['exact_initial_claim_scope_match'] for r in group),
                              exact_initial_scope_structure_matches=sum(r['exact_initial_scope_structure_match'] for r in group),
                              identical_full_selection_sequences=sum(r['first_different_query'] is None for r in group),
                              first_difference_at_query_one=sum(r['first_different_query'] == 1 for r in group),
                              differing_pairs_without_prior_method_KF_update=sum(
                                  r['first_different_query'] is not None and r['method_numeric_updates_before_first_difference'] == 0 for r in group)))
    task_panels = []
    for panel in evidence['summary']:
        group = [r for r in primary if r['task'] == panel['task'] and r['comparison'] == panel['comparison']]
        task_panels.append(dict(panel, final_best_counts=outcome_counts(group, 'best_delta')))
    if any(digest(ROOT / name) != value for name, value in input_hashes.items()):
        raise ValueError('Evidence changed during review')
    result = dict(status='PASS_DESCRIPTIVE_PAIRED_TRACE_REVIEW',
                  evidence_cutoff=evidence['completed_trajectories'], audit_cutoff=audit['completed_trajectories'],
                  generator_sha256=digest(Path(__file__)), input_sha256=input_hashes,
                  registered_primary_pairs=72, completed_primary_pairs=len(primary),
                  registered_removal_pairs=12, completed_removal_pairs=len(removal),
                  summaries=summaries, task_panels=task_panels, contrasts=contrasts,
                  claim_boundary='First divergence is descriptive timing, not an isolated component effect. Each arm uses fresh responses, and initial notes can differ before any numerical update. Initial-note comparisons test exact ordered claim/scope/reference fields and exact effect/focus/scope fields; they do not assert semantic equivalence of different wording or note order. After the first selection divergence, later histories and GP posteriors may differ. Same-history BO references are not independent-policy counterfactuals. Full-sequence equality does not establish that the prompts or reasoning were equal.')
    (OUT / 'contrast_trace_review.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('status', 'evidence_cutoff', 'completed_primary_pairs', 'completed_removal_pairs', 'summaries')}, indent=2))


if __name__ == '__main__':
    main()
