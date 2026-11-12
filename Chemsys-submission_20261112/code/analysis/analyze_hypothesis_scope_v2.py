"""Apply the registered development rule to the v2 arms.

Standard library only. Reads the exported v2 tables plus frozen comparison tables.
The rule comes from docs/hypothesis_scope_v2_registration_20261105.md: mean SearchAUC
gain of at least 1.0 point, gains in more tasks than losses, and mean final best not
lower by more than 0.5 point. Effects are % of pool maximum, with task as the unit.

    python code/analysis/analyze_hypothesis_scope_v2.py <folder with per_seed_results.csv>
"""
import csv
import json
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASKS = ('c1', 'c2', 'c3', 'c4', 'perera', 'doyle', 'snar', 'hplc')
SEEDS = range(4)
T90_DF7 = 1.894579


def rows_of(path):
    with open(path, encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def sign_flip_p(values):
    observed = abs(sum(values))
    hits = sum(abs(sum(v if m >> i & 1 else -v for i, v in enumerate(values))) >= observed - 1e-12
               for m in range(2 ** len(values)))
    return hits / 2 ** len(values)


def load(folder):
    top = {t: max(float(r['objective_raw']) for r in rows_of(ROOT/'data/pools'/t/'candidates.csv')) for t in TASKS}
    auc, final = {}, {}
    for r in rows_of(ROOT/'results/grounded_baselines_20261105/per_seed_results.csv'):
        key = (r['method'], r['task'], int(r['seed']))
        auc[key] = 100*float(r['SearchAUC'])/top[r['task']]
        final[key] = 100*float(r['final_best'])/top[r['task']]
    for r in rows_of(Path(folder)/'per_seed_results.csv'):
        key = (r['arm'], r['task'], int(r['seed']))
        auc[key] = 100*float(r['SearchAUC'])/top[r['task']]
        final[key] = 100*float(r['final_best'])/top[r['task']]
    return auc, final


def contrast(auc, final, a, b):
    cells = [(t, s) for t in TASKS for s in SEEDS if (a, t, s) in auc and (b, t, s) in auc]
    tasks = [t for t in TASKS if all((t, s) in cells for s in SEEDS)]
    by_task = {t: st.mean(auc[(a, t, s)] - auc[(b, t, s)] for s in SEEDS) for t in tasks}
    final_by_task = {t: st.mean(final[(a, t, s)] - final[(b, t, s)] for s in SEEDS) for t in tasks}
    raw = [auc[(a, t, s)] - auc[(b, t, s)] for t, s in cells]
    values = list(by_task.values())
    mean = st.mean(values)
    half = T90_DF7*st.stdev(values)/math.sqrt(len(values)) if len(values) > 1 else float('nan')
    up, down = sum(v > 1e-9 for v in values), sum(v < -1e-9 for v in values)
    final_mean = st.mean(final_by_task.values())
    return dict(contrast=f'{a} - {b}', tasks=len(tasks), mean_auc=mean, interval90=[mean - half, mean + half], tasks_up=up, tasks_down=down,
                sign_flip_p=sign_flip_p(values) if len(values) <= 12 else None, cells_better=sum(d > 1e-9 for d in raw),
                cells_tied=sum(abs(d) <= 1e-9 for d in raw), cells_worse=sum(d < -1e-9 for d in raw), worst_cell=min(raw),
                mean_final=final_mean, passes_rule=mean >= 1. and up > down and final_mean >= -.5, by_task=by_task, final_by_task=final_by_task)


def main():
    folder = sys.argv[1]
    auc, final = load(folder)
    arms = sorted({k[0] for k in auc})
    table = {a: dict(mean_auc=st.mean(auc[(a, t, s)] for t in TASKS for s in SEEDS), mean_final=st.mean(final[(a, t, s)] for t in TASKS for s in SEEDS))
             for a in arms if all((a, t, s) in auc for t in TASKS for s in SEEDS)}
    # Primary contrasts are within one environment. Frozen arms are shown as cross-machine context only.
    pairs = [('x0', 'n0'), ('n0', 'first_env'), ('x0', 'first_env'), ('x0', 'bo_env'), ('first_env', 'bo_env'),
             ('h1', 'first_env'), ('x1', 'h1'), ('x0', 'x1'), ('x1', 'first_env'), ('ex1', 'h1'), ('ex1', 'first_env'),
             ('x0', 'hyp'), ('x1', 'hyp'), ('ex1', 'hyp'), ('x0', 'lgbo'), ('x1', 'lgbo'), ('ex1', 'lgbo')]
    out = dict(arms=table, contrasts=[contrast(auc, final, a, b) for a, b in pairs if a in table and b in table])
    (Path(folder)/'development_rule.json').write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    for name, row in table.items():
        print(f"{name:6s} AUC {row['mean_auc']:6.2f}  final {row['mean_final']:6.2f}")
    for c in out['contrasts']:
        print(f"{c['contrast']:13s} AUC {c['mean_auc']:+6.2f} [{c['interval90'][0]:+.2f},{c['interval90'][1]:+.2f}] tasks {c['tasks_up']}up/{c['tasks_down']}down "
              f"p={c['sign_flip_p']:.3f} cells {c['cells_better']}/{c['cells_tied']}/{c['cells_worse']} worst {c['worst_cell']:+.1f} final {c['mean_final']:+.2f} rule={c['passes_rule']}")
        print('   ', {t: round(v, 2) for t, v in c['by_task'].items()})


if __name__ == '__main__':
    main()
