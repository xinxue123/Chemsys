"""Does the quality of the LLM's first experiment explain when LLM-started search beats BO?

Saved-evidence analysis only: no model calls, GP fits or oracle queries. It rereads exported
result tables and pool files and writes results/start_quality_saved_evidence_20261107/.
Every test here was chosen after the results existed, so all of it is exploratory.

    C:\\ChemEnv310\\Scripts\\python.exe code/analysis/analyze_start_quality_20261107.py
"""
import csv
import itertools
import json
import statistics as st
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT/'results'
OUT = RES/'start_quality_saved_evidence_20261107'
CPA_INPUT = ROOT/'data/pools/cpa'
DEV = ('c1', 'c2', 'c3', 'c4', 'perera', 'doyle', 'snar', 'hplc')
INPUTS = {'c1': (4, 3), 'c2': (4, 3), 'c3': (4, 3), 'c4': (4, 3), 'perera': (5, 0), 'doyle': (4, 0),
          'snar': (4, 4), 'hplc': (6, 6), 'aryl': (5, 0), 'cpa': (3, 0)}  # (inputs, continuous), Table 1


def rows(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def pool(task):
    """Candidate IDs and objective values, in pool order."""
    if task in DEV:
        r = rows(ROOT/f'data/pools/{task}/candidates.csv')
        return [x['candidate_id'] for x in r], np.array([float(x['objective_raw']) for x in r])
    folder = ROOT/'data/transfer/Direct_Arylation' if task == 'aryl' else CPA_INPUT
    ids = json.loads((folder/'pool.json').read_text(encoding='utf-8'))['ids']
    values = np.load(folder/'oracle.npz')['values']
    assert len(ids) == len(values)
    return ids, np.asarray(values, float)


def percentile(values, v):
    """Mid-rank percentile: share below plus half the share tied."""
    return 100*(np.mean(values < v) + 0.5*np.mean(values == v))


def ranks(x):
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0]*len(x)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j)/2 + 1
        i = j + 1
    return r


def spearman(x, y):
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def exact_p(x, y):
    """Two-sided exact permutation p for Spearman's rho (small n only)."""
    rho = abs(spearman(x, y))
    rx, ry = ranks(x), ranks(y)
    hits = total = 0
    for perm in itertools.permutations(ry):
        total += 1
        hits += abs(np.corrcoef(rx, perm)[0, 1]) >= rho - 1e-12
    return hits/total


def main():
    OUT.mkdir(exist_ok=True)
    traj = rows(RES/'hypothesis_scope_v2_20261105/trajectories.csv')
    gain = {r['task']: float(r['percent_of_pool_max']) for r in rows(RES/'first_point_saved_evidence_20261105/decomposition_by_task.csv')
            if r['contrast'] == 'same_first_minus_bo'}
    follow = json.loads((RES/'hypothesis_scope_v2_gp_followups_20261106/gp_followups.json').read_text(encoding='utf-8'))['test_F']
    shared = rows(RES/'hypothesis_scope_v2_shared_start_20261107/pairs.csv')
    cpa = rows(RES/'hypothesis_scope_v2_cpa_20261107/pairs.csv')

    table = []
    for task in DEV + ('aryl', 'cpa'):
        ids, values = pool(task)
        where = dict(zip(ids, values))
        top = values.max()
        if task in DEV:  # n0 starts from the LLM's first point; one draw per seed.
            picks = [r['candidate_id'] for r in traj if r['arm'] == 'n0' and r['task'] == task and r['query'] == '1']
            gap, source = gain[task], 'same-first BO minus pure BO, 4 streams'
        elif task == 'aryl':  # Draw counts and same-first runs from Test F; random starts from the 24 shared starts.
            draws = follow['same_first_reestimate']['draws']
            picks = [p for p, n in draws.items() for _ in range(n)]
            random_auc = st.mean(float(r['bo_auc']) for r in shared)
            gap = follow['same_first_reestimate']['weighted_auc'] - random_auc
            source = 'same-first BO (8 streams per pick, draw-weighted) minus plain BO from 24 shared starts'
        else:
            picks = [r['llm_first'] for r in cpa]
            gap, source = None, 'no BO run from the LLM pick exists; see within-task proxy'
        pct = [percentile(values, where[p]) for p in picks]
        table.append({
            'task': task, 'role': 'development' if task in DEV else 'held-out', 'pool_size': len(values),
            'inputs': INPUTS[task][0], 'continuous_inputs': INPUTS[task][1], 'llm_draws': len(picks),
            'distinct_llm_picks': len(set(picks)), 'llm_first_percentile': round(st.mean(pct), 2),
            'llm_first_pct_of_max': round(100*st.mean(where[p] for p in picks)/top, 2),
            'head_start_vs_pool_mean': round(100*(st.mean(where[p] for p in picks) - values.mean())/top, 2),
            'share_ge50_of_max': round(100*np.mean(values >= 0.5*top), 2),
            'share_ge90_of_max': round(100*np.mean(values >= 0.9*top), 2),
            'share_lt5_of_max': round(100*np.mean(values < 0.05*top), 2),
            'start_gain_vs_random_bo': None if gap is None else round(gap, 2), 'gain_source': source})
        if task == 'cpa':  # Cross-check: the CPA analysis counted only values strictly below.
            strict = st.mean(100*np.mean(values < where[p]) for p in picks)
            assert abs(strict - st.mean(float(r['llm_first_percentile']) for r in cpa)) < 0.01

    with open(OUT/'by_task.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)

    have = [t for t in table if t['start_gain_vs_random_bo'] is not None]
    g = [t['start_gain_vs_random_bo'] for t in have]
    cross = {}
    for key in ('llm_first_percentile', 'llm_first_pct_of_max', 'head_start_vs_pool_mean', 'share_ge50_of_max', 'share_ge90_of_max', 'share_lt5_of_max', 'continuous_inputs', 'pool_size'):
        x = [t[key] for t in have]
        cross[key] = {'tasks': len(x), 'spearman': round(spearman(x, g), 3), 'exact_p': round(exact_p(x, g), 4)}
    one_reizman = {}  # c1-c4 share chemistry: keep one case at a time (7 tasks).
    for keep in ('c1', 'c2', 'c3', 'c4'):
        sub = [t for t in have if not t['task'].startswith('c') or t['task'] == keep]
        y = [t['start_gain_vs_random_bo'] for t in sub]
        one_reizman[keep] = {key: {'spearman': round(spearman([t[key] for t in sub], y), 3), 'exact_p': round(exact_p([t[key] for t in sub], y), 4)}
                             for key in ('llm_first_percentile', 'head_start_vs_pool_mean')}
    cross['one_reizman_case'] = one_reizman
    above = [t for t in have if t['llm_first_percentile'] >= 80]
    below = [t for t in have if t['llm_first_percentile'] < 80]
    split = {'threshold_percentile': 80,
             'at_or_above': {'tasks': [t['task'] for t in above], 'mean_gain': round(st.mean(t['start_gain_vs_random_bo'] for t in above), 2)},
             'below': {'tasks': [t['task'] for t in below], 'mean_gain': round(st.mean(t['start_gain_vs_random_bo'] for t in below), 2)}}

    # Within a task: does a better start give better plain BO? Uses plain BO from 24 shared random starts.
    within = []
    for task, data, vkey, akey in (('aryl', shared, 'first_yield', 'bo_auc'), ('cpa', cpa, 'start_value', 'plain_auc')):
        v = [float(r[vkey]) for r in data]
        a = [float(r[akey]) for r in data]
        ordered = [auc for _, auc in sorted(zip(v, a))]  # Ties keep a stable order; thirds are 8 starts each.
        third = len(ordered)//3
        within.append({'task': task, 'starts': len(v), 'spearman_start_value_vs_bo_auc': round(spearman(v, a), 3),
                       'mean_auc_top_third_starts': round(st.mean(ordered[-third:]), 2),
                       'mean_auc_bottom_third_starts': round(st.mean(ordered[:third]), 2)})
    suz = rows(RES/'first_point_saved_evidence_20261105/low_middle_start_reference.csv')
    for r in suz:
        within.append({'task': r['task'], 'starts': int(r['distinct_starts']),
                       'spearman_start_value_vs_bo_auc': round(float(r['spearman_first_yield_vs_auc']), 3),
                       'mean_auc_top_third_starts': None, 'mean_auc_bottom_third_starts': None})
    with open(OUT/'within_task.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(within[0]))
        w.writeheader()
        w.writerows(within)

    summary = {'note': 'Saved-evidence, exploratory. No model calls, GP fits or oracle queries.',
               'cross_task': cross, 'split': split, 'within_task': within}
    (OUT/'summary.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
    for t in table:
        print({k: t[k] for k in ('task', 'llm_first_percentile', 'llm_first_pct_of_max', 'distinct_llm_picks', 'share_ge50_of_max', 'share_ge90_of_max', 'share_lt5_of_max', 'start_gain_vs_random_bo')})
    print(json.dumps(cross, indent=0))
    print(json.dumps(split))
    print(json.dumps(within, indent=0))


if __name__ == '__main__':
    main()
