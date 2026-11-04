"""Verify compact KF results with the standard library, using no oracle."""
import csv
from collections import defaultdict
from math import isclose
from pathlib import Path
from statistics import mean

RESULTS = Path(__file__).resolve().parents[2] / 'results'


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as source:
        return list(csv.DictReader(source))


def check(directory, keys, best_column):
    rows = defaultdict(list)
    for row in read(directory / 'rounds.csv'):
        rows[tuple(row[key] for key in keys)].append(row)
    metrics = read(directory / 'metrics.csv')
    if len(rows) != len(metrics):
        raise ValueError(f'{directory.name}: metrics and traces differ in count')
    for row in metrics:
        key = tuple(row[name] for name in keys)
        trace = sorted(rows.pop(key), key=lambda item: int(item['query']))
        objectives = [float(item['objective']) for item in trace]
        running = [max(objectives[:i]) for i in range(1, len(objectives)+1)]
        if len(trace) != 20 or [int(t['query']) for t in trace] != list(range(1, 21)):
            raise ValueError(f'{key}: incorrect query sequence')
        if any(not isclose(value, float(t[best_column]), abs_tol=1e-9)
               for value, t in zip(running, trace)):
            raise ValueError(f'{key}: incorrect cumulative best')
        first_best = objectives.index(max(objectives))+1
        best_query = row.get('first_final_best') or row['first_final_best_query']
        if (not isclose(mean(running), float(row['SearchAUC']), abs_tol=1e-9)
                or not isclose(running[-1], float(row['final_best']), abs_tol=1e-9)
                or first_best != int(best_query)):
            raise ValueError(f'{key}: inconsistent metrics')
        if 'target_95' in row:
            hit = next((i for i,y in enumerate(objectives,1) if y>=float(row['target_95'])),None)
            if (str(hit) if hit is not None else '') != row['first_target']:
                raise ValueError(f'{key}: incorrect target hit')
    return len(metrics)


if __name__ == '__main__':
    original = check(RESULTS/'kf_weight', ('task','seed','method'), 'cumulative_best')
    confirm = check(RESULTS/'kf_confirmation', ('cell','arm'), 'best')
    print(f'PASS: {original + confirm} exported trajectories, {(original + confirm)*20} round values')
