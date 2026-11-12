"""Cross-machine noise floor: the same method and stream rerun on a second machine.

Standard library only: compares exported per-run tables and issues no model calls,
queries or GP fits. Effects are % of task pool maximum. Identical means the same
SearchAUC within 1e-9. Per-run noise is sqrt(mean(d^2)/2), which treats each run as
carrying independent noise; it is sensitive to the few large differences.

    python code/analysis/analyze_hypothesis_scope_v2_noise_floor.py
"""
import csv
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASKS = ('c1', 'c2', 'c3', 'c4', 'perera', 'doyle', 'snar', 'hplc')
ARMS = (('bo_env', 'bo', 'Pure BO'), ('first_env', 'first', 'Same-first BO'))
GROUPS = (('all four streams', (0, 1, 2, 3)), ('gate streams 1-2', (1, 2)), ('reference streams 0 and 3', (0, 3)))


def rows_of(path):
    with open(path, encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def summarize(values):
    return dict(runs=len(values), identical=sum(v < 1e-9 for v in values), mean_abs=st.mean(values), largest=max(values),
                per_run_noise=math.sqrt(st.mean(v*v for v in values)/2))


def main():
    top = {t: max(float(r['objective_raw']) for r in rows_of(ROOT/'data/pools'/t/'candidates.csv')) for t in TASKS}
    new = {(r['arm'], r['task'], int(r['seed'])): float(r['SearchAUC']) for r in rows_of(ROOT/'results/hypothesis_scope_v2_20261105/per_seed_results.csv')}
    old = {(r['method'], r['task'], int(r['seed'])): float(r['SearchAUC']) for r in rows_of(ROOT/'results/grounded_baselines_20261105/per_seed_results.csv')}
    gaps = {(env, t, s): abs(new[env, t, s] - old[frozen, t, s])*100/top[t] for env, frozen, _ in ARMS for t in TASKS for s in range(4)}
    result = {}
    for label, streams in GROUPS:
        result[label] = {name: summarize([gaps[env, t, s] for t in TASKS for s in streams]) for env, _, name in ARMS}
        result[label]['both arms'] = summarize([gaps[env, t, s] for env, _, _ in ARMS for t in TASKS for s in streams])
    result['by_task_both_arms'] = {t: st.mean(gaps[env, t, s] for env, _, _ in ARMS for s in range(4)) for t in TASKS}
    (ROOT/'results/hypothesis_scope_v2_20261105/noise_floor.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    for label, _ in GROUPS:
        for name, row in result[label].items():
            print(f"{label:26s} {name:14s} runs {row['runs']:2d} identical {row['identical']:2d} mean {row['mean_abs']:5.2f} largest {row['largest']:5.2f} per-run {row['per_run_noise']:5.2f}")
    print({t: round(v, 2) for t, v in result['by_task_both_arms'].items()})


if __name__ == '__main__':
    main()
