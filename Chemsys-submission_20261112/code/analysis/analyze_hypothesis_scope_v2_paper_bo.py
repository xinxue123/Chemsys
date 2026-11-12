"""Apply the registered reading for the Reasoning BO paper's BO engine on the held-out tasks.

Reads saved run states and derived tables only. No model calls, queries or GP fits.
Rules are in docs/hypothesis_scope_v2_registration_20261105.md, under
"Reasoning BO paper's BO engine on the held-out tasks".

    python code/analysis/analyze_hypothesis_scope_v2_paper_bo.py <v2 run folder> <output folder>
"""
import csv
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_hypothesis_scope_v2_first_point_streams import sign_flip

ROOT = Path(__file__).resolve().parents[2]
T90 = 1.714  # 23 degrees of freedom
TOP = dict(aryl=100.0, cpa=3.134624572)
EXISTING = dict(aryl=('bo', 'r1', 'n6m1', 'rbo', 'lgbo'), cpa=('bo', 'r1'))
NAMES = dict(paper='paper BO', bo='our plain BO', r1='controller', n6m1='neighbour search', rbo='Reasoning BO', lgbo='LGBO')


def rows_of(path):
    with open(path, encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def score(values):
    best = [max(values[:i + 1]) for i in range(len(values))]
    assert len(values) == 20
    return st.mean(best), best[-1]


def contrast(a, b, name):
    d = [x - y for x, y in zip(a, b)]
    mean, half = st.mean(d), T90*st.stdev(d)/math.sqrt(len(d))
    low, high = mean - half, mean + half
    first, second = name.split(' - ')
    reading = f'{first} leads' if low > 1 else f'{second} leads' if high < -1 else 'not separated'
    return dict(contrast=name, mean_auc=mean, interval90=[low, high], up=sum(v > 1e-9 for v in d), down=sum(v < -1e-9 for v in d),
                sign_flip_p=sign_flip(d), reading=reading)


def existing(task):
    runs = defaultdict(dict)
    if task == 'aryl':
        for r in rows_of(ROOT/'results/hypothesis_scope_v2_shared_start_20261107/trajectories.csv'):
            runs[r['arm'], int(r['pair'])][int(r['query'])] = (r['candidate_id'], 100*float(r['raw_value'])/TOP[task])
        starts = {int(r['pair']): r['first_point'] for r in rows_of(ROOT/'results/hypothesis_scope_v2_shared_start_20261107/pairs.csv')}
    else:
        for r in rows_of(ROOT/'results/hypothesis_scope_v2_cpa_20261107/trajectories.csv'):
            runs[r['arm'], int(r['pair'])][int(r['query'])] = (r['candidate_id'], float(r['pct_of_max']))
        starts = {int(r['pair']): r['start'] for r in rows_of(ROOT/'results/hypothesis_scope_v2_cpa_20261107/pairs.csv')}
    return runs, starts


def main():
    base, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    result, table, trajectories = {}, [], []
    for task in ('aryl', 'cpa'):
        runs, starts = existing(task)
        auc, final = defaultdict(list), defaultdict(list)
        recipe = None
        for t in range(24):
            state = json.loads((base/'paper_bo'/task/f's{t}'/'state.json').read_text(encoding='utf-8'))
            assert state['status'] == 'COMPLETE' and state['stream'] == t and len(state['history']) == 20
            assert state['history'][0]['candidate_id'] == starts[t] == runs['bo', t][1][0]
            assert len({h['candidate_id'] for h in state['history']}) == 20
            recipe = recipe or state['recipe_sha256']
            assert state['recipe_sha256'] == recipe
            values = [100*h['raw_value']/TOP[task] for h in state['history']]
            for q, (h, v) in enumerate(zip(state['history'], values), 1):
                trajectories.append(dict(task=task, pair=t, arm='paper', query=q, candidate_id=h['candidate_id'], pct_of_max=round(v, 6)))
            a, f = score(values)
            auc['paper'].append(a)
            final['paper'].append(f)
            row = dict(task=task, pair=t, start=starts[t], paper_auc=round(a, 4), paper_final=round(f, 4))
            for arm in EXISTING[task]:
                a, f = score([runs[arm, t][q][1] for q in range(1, 21)])
                auc[arm].append(a)
                final[arm].append(f)
                row[f'{arm}_auc'] = round(a, 4)
            table.append(row)
        ties = sum(h.get('exact_ties', 1) > 1 for t in range(24) for h in json.loads((base/'paper_bo'/task/f's{t}'/'state.json').read_text(encoding='utf-8'))['history'])
        primary = contrast(auc['paper'], auc['bo'], 'paper BO - our plain BO')
        primary['mean_final'] = st.mean(a - b for a, b in zip(final['paper'], final['bo']))
        result[task] = dict(pairs=24, recipe_sha256=recipe, queries_with_exact_ties=ties,
                            arms={arm: dict(mean_auc=st.mean(auc[arm]), mean_final=st.mean(final[arm])) for arm in auc},
                            primary=primary,
                            secondary={arm: contrast(auc[arm], auc['paper'], f'{NAMES[arm]} - paper BO') for arm in EXISTING[task] if arm != 'bo'})
    (out/'paper_bo.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    for name, rows in (('pairs.csv', table), ('trajectories.csv', trajectories)):
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with open(out/name, 'w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
