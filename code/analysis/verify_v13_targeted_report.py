"""Check the report against saved observations without invoking policy or GP code."""
from collections import defaultdict
from pathlib import Path
import csv
import json
import math

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'experiments/reports/v13_targeted_ablation_20261103'


def close(a, b):
    if not math.isclose(float(a), float(b), abs_tol=1e-8, rel_tol=1e-12):
        raise ValueError(f'Report mismatch: {a!r} != {b!r}')


def rows(name):
    with (REPORT / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def main():
    metrics = rows('metrics.csv')
    observed = defaultdict(list)
    for r in rows('rounds.csv'):
        observed[r['task'], int(r['seed']), r['arm']].append(r)
    calculated = {}
    for m in metrics:
        key = m['task'], int(m['seed']), m['arm']
        state = json.loads((ROOT / m['state_path']).read_text(encoding='utf-8'))
        history = state['history']
        if state['status'] != 'COMPLETE' or len(history) != 20:
            raise ValueError(f'Incomplete native trajectory: {key}')
        trajectory = sorted(observed[key], key=lambda r: int(r['query']))
        if [int(r['query']) for r in trajectory] != list(range(1, 21)):
            raise ValueError(f'Missing or repeated report rounds: {key}')
        running = []
        best = -math.inf
        for h, r in zip(history, trajectory):
            if h['candidate_id'] != r['selected_id']:
                raise ValueError(f'Native point mismatch: {key}')
            close(h['raw_value'], r['objective'])
            best = max(best, float(h['raw_value']))
            running.append(best)
            close(best, r['cumulative_best'])
        auc = math.fsum(running) / len(running)
        close(auc, m['SearchAUC'])
        close(best, m['final_best'])
        close(running.index(best) + 1, m['first_final_best_query'])
        calculated[key] = (auc, best, [h['candidate_id'] for h in history])
    for r in rows('paired_effects.csv'):
        task, seed = r['task'], int(r['seed'])
        if r['contrast'] == 'interaction':
            values = [calculated[task, seed, a] for a in ('full', 'fixed', 'no_refresh', 'neither')]
            for i, field in enumerate(('delta_SearchAUC', 'delta_final_best')):
                close(values[0][i] - values[1][i] - values[2][i] + values[3][i], r[field])
        else:
            a = calculated[task, seed, r['left']]
            b = calculated[task, seed, r['right']]
            close(a[0] - b[0], r['delta_SearchAUC'])
            close(a[1] - b[1], r['delta_final_best'])
            if int(a[2] == b[2]) != int(r['identical_all_choices']):
                raise ValueError('Incorrect identical-point label')
    result = dict(status='PASS', native_trajectories=len(calculated),
                  native_observations=sum(len(v) for v in observed.values()),
                  independently_recomputed='mean of cumulative maximum of native raw_value',
                  denominator='actual completed query count; verified 20 in every trajectory',
                  raw_units_preserved=True, all_report_points_match_native_states=True,
                  all_metrics_and_paired_effects_match=True, new_scientific_queries=0)
    (REPORT / 'report_validation.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
