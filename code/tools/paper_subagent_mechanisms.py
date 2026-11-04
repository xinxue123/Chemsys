"""Export descriptive round/forecast evidence without new model or oracle calls.

Only complete trajectories in the current evidence snapshot are read. These
counts are branch coverage, not counterfactual effects on optimization.
"""
from pathlib import Path
from hashlib import sha256
import csv
import json
import math

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / 'paper/v12_subagent'
REPORT = ROOT / 'experiments/reports/paper_v12_subagent'


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_csv(path, rows):
    if not rows:
        raise ValueError('Do not leave an old nonempty export after an empty result')
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def require_close(a, b, context):
    if not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-10):
        raise ValueError('Numerical mismatch: ' + context)


def export():
    evidence_path = PAPER / 'data/evidence.json'
    evidence_sha = digest(evidence_path)
    evidence = read(evidence_path)
    units = {d['task']: d['unit'] for d in evidence['datasets']}
    sources, rounds, forecasts, coverage = {}, [], [], []
    for row in evidence['rows']:
        path = ROOT / row['source']
        sources[row['source']] = digest(path)
        if sources[row['source']] != row['source_sha256']:
            raise ValueError('State changed since evidence snapshot: ' + row['source'])
        state = read(path)
        history, decisions = state['history'], state['decisions']
        if state['status'] != 'COMPLETE' or len(history) != 20 or len(decisions) != 16:
            raise ValueError('Full 4+16 trajectory required')
        meta = {key: row[key] for key in ('cohort', 'task', 'method', 'arm', 'initialization_seed')}
        start_round, start_forecast = len(rounds), len(forecasts)
        for query, decision in enumerate(decisions, 1):
            if decision['query'] != query or history[query + 3]['candidate_id'] != decision['selected_id']:
                raise ValueError('Decision chronology mismatch')
            commit_path = path.parent / 'rounds' / f'{query:02d}_commit.json'
            commit = read(commit_path)
            commit_source = commit_path.relative_to(ROOT).as_posix()
            sources[commit_source] = digest(commit_path)
            if commit['commit_sha256'] != decision['commit_sha256'] or commit['selected_id'] != decision['selected_id']:
                raise ValueError('Decision and commitment differ')
            before = max(h['raw_value'] for h in history[:query + 3])
            value = history[query + 3]['raw_value']
            require_close(value, decision['raw_value'], 'queried raw objective')
            events = decision.get('KF_feedback', [])
            refs = decision.get('memory_references', [])
            refresh = commit.get('refresh') or {}
            expected = commit.get('expected_forecasts')
            if expected is not None and len(events) > expected:
                raise ValueError('More updates than applicable note forecasts')
            record = dict(meta, query=query, selected_id=decision['selected_id'],
                          BO_reference_id=decision.get('BO_reference_id'),
                          non_top_acquisition=int(decision['selected_id'] != decision['BO_reference_id']),
                          mode=decision['mode'], objective_unit=units[row['task']],
                          raw_value=value, incumbent_before_raw=before,
                          incumbent_after_raw=max(before, value), improved=int(value > before),
                          menu_ids_json=compact(commit['ordered_ids']),
                          selected_logEI_gap=commit.get('opportunity_gap'),
                          expected_forecasts=expected, forecast_updates=len(events),
                          positive_improvement_forecasts=sum(bool(e['positive_improvement_evidence']) for e in events),
                          negative_gain_forecasts=sum(e['Brier_gain'] < -1e-12 for e in events),
                          weight_decreases=sum(e['after']['weight'] < e['before']['weight'] for e in events),
                          automatic_pauses=sum(e['reason'] == 'automatic_pause' for e in events),
                          automatic_retractions=sum(e['reason'] == 'automatic_retract' for e in events),
                          automatic_resumes=sum(e['reason'] == 'automatic_resume' for e in events),
                          memory_edit_call=int(query in state.get('edit_calls', [])),
                          refresh_reasons_json=compact(refresh.get('reasons', [])),
                          edit_reviews_json=compact(commit.get('reviews', [])),
                          requested_support_citations=sum(r.get('requested_use') == 'support' for r in refs),
                          accepted_support_citations=sum(r.get('use') == 'support' for r in refs),
                          memory_references_json=compact(refs),
                          sidecar_warning_count=len(decision.get('sidecar_warnings', [])),
                          source=row['source'], source_sha256=row['source_sha256'],
                          commit_source=commit_source, commit_file_sha256=sources[commit_source],
                          commit_sha256=decision['commit_sha256'])
            if bool(refresh.get('allowed')) != bool(record['memory_edit_call']):
                raise ValueError('Memory edit chronology mismatch')
            rounds.append(record)
            for event in events:
                if event['point_id'] != decision['selected_id'] or event['query'] != query:
                    raise ValueError('Forecast event does not match selected point')
                if event['success'] != bool(value > before):
                    raise ValueError('Forecast targets a different improvement event')
                gain = (float(event['success']) - event['p_GP']) ** 2 - (float(event['success']) - event['p_LLM']) ** 2
                require_close(gain, event['Brier_gain'], 'relative Brier gain')
                before_kf, after_kf = event['before'], event['after']
                forecast = dict(meta, query=query, selected_id=decision['selected_id'],
                                note_id=event['id'], version=event['version'], event_id=event['event_id'],
                                reference_id=event['reference_id'],
                                p_LLM=event['p_LLM'], p_GP=event['p_GP'], success=int(event['success']),
                                Brier_gain=event['Brier_gain'],
                                positive_improvement_evidence=int(event['positive_improvement_evidence']),
                                reason=event['reason'],
                                numerical_KF_update_applied=int(event['reason'] != 'fixed_calibration_ablation'),
                                measurement_variance=event.get('measurement_variance'),
                                native_diagnostic_predicted_variance=event.get('predicted_variance'),
                                native_diagnostic_kalman_gain=event.get('kalman_gain'))
                for name in ('mean', 'variance', 'probability_positive', 'weight', 'status', 'local_updates', 'bad_streak', 'good_streak'):
                    forecast[name + '_before'] = before_kf[name]
                    forecast[name + '_after'] = after_kf[name]
                forecast.update(deployment_weight_before=event['deployment_weight_before'],
                                deployment_weight_after=event['deployment_weight_after'],
                                prediction_sha256=event['prediction_sha256'], source=row['source'],
                                source_sha256=row['source_sha256'], commit_sha256=decision['commit_sha256'])
                forecasts.append(forecast)
        rr, ff = rounds[start_round:], forecasts[start_forecast:]
        checks = {'forecasts': len(ff), 'weight_decreases': sum(r['weight_decreases'] for r in rr),
                  'pauses': sum(r['automatic_pauses'] for r in rr),
                  'retractions': sum(r['automatic_retractions'] for r in rr),
                  'resumes': sum(r['automatic_resumes'] for r in rr),
                  'edits': sum(r['memory_edit_call'] for r in rr),
                  'different_from_same_history_BO': sum(r['non_top_acquisition'] for r in rr)}
        if any(row[key] != count for key, count in checks.items()):
            raise ValueError('Mechanism counts differ from existing evidence')
        if row['arm'] == 'memory' and row['cohort'] == 'codex_subagent':
            coverage.append(dict(meta, queries=16,
                                 queries_with_applicable_notes=sum(bool(r['expected_forecasts']) for r in rr),
                                 queries_with_forecast_update=sum(r['forecast_updates'] > 0 for r in rr),
                                 expected_forecasts=sum(r['expected_forecasts'] for r in rr),
                                 missing_forecasts=sum(r['expected_forecasts'] - r['forecast_updates'] for r in rr),
                                 positive_improvement_forecasts=sum(r['positive_improvement_forecasts'] for r in rr),
                                 positive_gain_nonimprovement_forecasts=sum(e['Brier_gain'] > 1e-12 and not e['success'] for e in ff),
                                 non_top_queries_without_forecast=sum(r['non_top_acquisition'] and not r['forecast_updates'] for r in rr),
                                 requested_support_citations=sum(r['requested_support_citations'] for r in rr),
                                 accepted_support_citations=sum(r['accepted_support_citations'] for r in rr),
                                 **checks))
    if digest(evidence_path) != evidence_sha or any(digest(ROOT / name) != value for name, value in sources.items()):
        raise ValueError('Export inputs changed during read')
    totals = {key: sum(r[key] for r in coverage) for key in coverage[0]
              if key not in ('cohort', 'task', 'method', 'arm', 'initialization_seed')}
    summary = dict(cohort='codex_subagent', arm='memory', completed_trajectories=len(coverage),
                   totals=totals, trajectories=coverage,
                   boundary='Descriptive executed opportunities and events. Several forecasts may share one outcome. A non-top choice is not a causal memory effect; its same-history BO reference has no observed counterfactual outcome here.')
    files = []
    for folder in (PAPER / 'data', REPORT):
        folder.mkdir(parents=True, exist_ok=True)
        for name, rows in (('round_decisions.csv', rounds), ('forecast_updates.csv', forecasts)):
            output = folder / name
            write_csv(output, rows)
            if folder == PAPER / 'data':
                files.append(output)
        output = folder / 'mechanism_coverage.json'
        output.write_text(json.dumps(summary, indent=2), encoding='utf-8')
        if folder == PAPER / 'data':
            files.append(output)
    receipt = dict(evidence_sha256=evidence_sha, generator_sha256=digest(Path(__file__)),
                   completed_trajectories=evidence['completed_trajectories'],
                   exported_rounds=len(rounds), subagent_rounds=sum(r['cohort'] == 'codex_subagent' for r in rounds),
                   reused_BO_rounds=sum(r['cohort'] == 'reused_deterministic_BO' for r in rounds),
                   exported_forecasts=len(forecasts), source_sha256=sources,
                   files={f.relative_to(PAPER).as_posix(): digest(f) for f in files},
                   new_oracle_queries=0, new_LLM_calls=0, new_GP_fits=0)
    (REPORT / 'mechanism_generation.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    return {k: v for k, v in receipt.items() if k not in ('source_sha256', 'files')}


if __name__ == '__main__':
    print(json.dumps(export(), indent=2))
