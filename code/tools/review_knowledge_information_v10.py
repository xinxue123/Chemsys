"""Read-only V9 design diagnosis; no oracle, GP fitting, or LLM requests."""
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main():
    run = ROOT / 'experiments/runs/knowledge_loop_20261101/v9_kf_rank'
    protocol = json.loads((run / 'protocol.json').read_text(encoding='utf-8'))
    rows, sources = [], {}
    for task in protocol['tasks']:
        path = run / 'runs' / f'{task}_kf_rank' / 'state.json'
        sources[path.relative_to(ROOT).as_posix()] = sha256(path.read_bytes()).hexdigest()
        state = json.loads(path.read_text(encoding='utf-8'))
        if state['status'] != 'COMPLETE' or len(state['decisions']) != 16:
            raise ValueError('Design review requires completed frozen trajectories')
        decisions = state['decisions']
        feedback = [e for d in decisions for e in d['KF_feedback']]
        rows.append(dict(task=task, queries=len(decisions),
                         non_BO_queries=[d['query'] for d in decisions if d['mode'] != 'base_bo'],
                         queries_with_knowledge_feedback=sum(bool(d['KF_feedback']) for d in decisions),
                         forecast_updates=len(feedback),
                         GP_success_probability_at_least_095=sum(e['p_GP'] >= .95 for e in feedback),
                         positive_Brier_gains=sum(e['Brier_gain'] > 1e-12 for e in feedback),
                         material_positive_Brier_gains=sum(e['Brier_gain'] > .02 for e in feedback),
                         edit_queries=state['edit_calls']))
    summary = {key: sum(row[key] for row in rows) for key in (
        'queries', 'queries_with_knowledge_feedback', 'forecast_updates',
        'GP_success_probability_at_least_095', 'positive_Brier_gains', 'material_positive_Brier_gains')}
    summary['non_BO_choices'] = sum(len(row['non_BO_queries']) for row in rows)
    value = dict(status='DESIGN_DIAGNOSIS_ONLY', frozen_V9_protocol=protocol['protocol_sha256'],
                 summary=summary, tasks=rows, sources_sha256=sources,
                 new_oracle_queries=0, new_LLM_calls=0, new_GP_fits=0,
                 limitations=[
                     'Probability>=.95 is the archived GP forecast, not proof of true certainty.',
                     'Feedback count is not a causal explanation of SearchAUC differences.',
                     'No unqueried target or other-arm target is used to propose new candidates.',
                     'V10 information math is a synthetic-tested prototype, not an online optimizer.',
                 ])
    out = ROOT / 'experiments/reports/knowledge_loop_20261101/v10_design_diagnosis.json'
    with out.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(out), **summary), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
