"""Saved-evidence attribution of the LLM first point.

Offline and standard-library only. Reads exported tables and sealed archives in
memory. It issues no model calls, GP fits or oracle experiments, extracts no
archive member, and writes only to results/first_point_saved_evidence_20261105/.

Every analysis here was chosen after the underlying results were available, so
all p-values and intervals are exploratory.

    python code/analysis/analyze_first_point_saved_evidence.py
"""
import csv
import json
import math
import random
import statistics as st
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARCHIVES = ROOT / 'archives'
GROUNDED = ROOT / 'results/grounded_baselines_20261105'
OUT = ROOT / 'results/first_point_saved_evidence_20261105'
TASKS = ('c1', 'c2', 'c3', 'c4', 'perera', 'doyle', 'snar', 'hplc')
SEEDS = range(4)
REIZMAN_AXES = ('t_res', 'temperature', 'catalyst_loading')
CONTINUOUS = dict(c1=REIZMAN_AXES, c2=REIZMAN_AXES, c3=REIZMAN_AXES, c4=REIZMAN_AXES,
                  snar=('tau', 'equiv_pldn', 'conc_dfnb', 'temperature'),
                  hplc=('sample_loop', 'additional_volume', 'tubing_volume',
                        'sample_flow', 'push_speed', 'wait_time'))
T90_DF7 = 1.894579


def rows_of(path):
    with open(path, encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def member(archive, name):
    return json.loads(archive.read(name).decode('utf-8-sig'))


def write_rows(name, rows):
    with open(OUT / name, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def sign_flip_p(values):
    """Exact two-sided sign-flip p over all 2**n assignments."""
    observed = abs(sum(values))
    hits = sum(abs(sum(v if mask >> i & 1 else -v for i, v in enumerate(values))) >= observed - 1e-12
               for mask in range(2 ** len(values)))
    return hits / 2 ** len(values)


def interval90(values):
    half = T90_DF7 * st.stdev(values) / math.sqrt(len(values))
    return st.mean(values) - half, st.mean(values) + half


def spearman(a, b):
    def ranks(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        out = [0.] * len(x)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]] = (i + j) / 2
            i = j + 1
        return out
    ra, rb = ranks(a), ranks(b)
    ma, mb = st.mean(ra), st.mean(rb)
    top = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    bottom = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return top / bottom


def task_clustered_ci(cell, n=20000, seed=7):
    """Resample tasks, then streams inside each task."""
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        values = []
        for task in (rng.choice(TASKS) for _ in TASKS):
            values += [cell[(task, rng.randrange(4))] for _ in SEEDS]
        means.append(st.mean(values))
    means.sort()
    return means[int(.025 * n)], means[int(.975 * n)]


def load():
    pool = {t: rows_of(ROOT / 'data/pools' / t / 'candidates.csv') for t in TASKS}
    value = {t: {r['candidate_id']: float(r['objective_raw']) for r in pool[t]} for t in TASKS}
    top = {t: max(value[t].values()) for t in TASKS}
    scaled = {}
    for task, axes in CONTINUOUS.items():
        low = {a: min(float(r[a]) for r in pool[task]) for a in axes}
        high = {a: max(float(r[a]) for r in pool[task]) for a in axes}
        scaled[task] = {r['candidate_id']: [(float(r[a]) - low[a]) / (high[a] - low[a]) for a in axes]
                        for r in pool[task]}
    return value, top, scaled


def menu_percentile(value, selected, menu):
    values = [value[i] for i in menu]
    x = value[selected]
    return (sum(v < x for v in values) + .5 * (sum(v == x for v in values) - 1)) / (len(values) - 1) * 100


def five_point_menus(value, scaled):
    """Random five-point query-1 menus shared by V13, Reasoning BO and BO."""
    picks = []
    with zipfile.ZipFile(ARCHIVES / 'run__cold0_20261102.zip') as first, \
            zipfile.ZipFile(ARCHIVES / 'run__cold_s123_20261102.zip') as rest:
        for task in TASKS:
            for seed in SEEDS:
                for arm in ('v13', 'rbo', 'bo'):
                    if seed == 0:
                        commit = member(first, f'experiments/runs/cold0_20261102/runs/{task}_{arm}/rounds/01_commit.json')
                    else:
                        commit = member(rest, f'experiments/runs/cold_s123_20261102/s{seed}/runs/ms{seed}_{task}_{arm}/rounds/01_commit.json')
                    picks.append(dict(task=task, seed=seed, arm=arm, menu=commit['display_ids'], selected=commit['selected_id']))
    rows = []
    for p in picks:
        task, menu = p['task'], p['menu']
        row = dict(task=task, seed=p['seed'], arm=p['arm'], selected=p['selected'],
                   selected_value=value[task][p['selected']],
                   menu_mean=st.mean(value[task][i] for i in menu),
                   rank_in_menu=1 + sum(value[task][i] > value[task][p['selected']] for i in menu),
                   percentile=menu_percentile(value[task], p['selected'], menu),
                   rule_selected='', rule_percentile='')
        if task in scaled:
            rule = max(menu, key=lambda i: st.mean(scaled[task][i]))
            row.update(rule_selected=rule, rule_percentile=menu_percentile(value[task], rule, menu))
        rows.append(row)
    write_rows('menu_picks.csv', rows)
    arm_task = defaultdict(dict)
    for arm, members in (('v13', ('v13',)), ('rbo', ('rbo',)), ('bo', ('bo',)), ('llm', ('v13', 'rbo'))):
        for task in TASKS:
            arm_task[arm][task] = st.mean(r['percentile'] for r in rows if r['task'] == task and r['arm'] in members)
    llm = [r for r in rows if r['arm'] != 'bo']
    by_cell = defaultdict(dict)
    for r in rows:
        by_cell[(r['task'], r['seed'])][r['arm']] = r['selected']
    summary = {}
    for arm, table in arm_task.items():
        values = [table[t] for t in TASKS]
        summary[arm] = dict(mean_percentile=st.mean(values), tasks_above_50=sum(v > 50 for v in values),
                            tasks_below_50=sum(v < 50 for v in values),
                            sign_flip_p=sign_flip_p([v - 50 for v in values]),
                            by_task={t: table[t] for t in TASKS})
    summary['llm_best_of_five'] = sum(r['rank_in_menu'] == 1 for r in llm)
    summary['llm_picks'] = len(llm)
    summary['v13_rbo_same_pick'] = sum(c['v13'] == c['rbo'] for c in by_cell.values())
    rule = {}
    for task in CONTINUOUS:
        subset = [r for r in llm if r['task'] == task]
        rule[task] = dict(llm=st.mean(r['percentile'] for r in subset),
                          rule=st.mean(r['rule_percentile'] for r in subset),
                          agreement=sum(r['selected'] == r['rule_selected'] for r in subset), picks=len(subset))
    summary['rule_vs_llm'] = rule
    return summary


def grounded_picks(value, top, scaled):
    """Frozen grounded picks against their full example menus and the corner rule."""
    choices = {r['task']: r for r in rows_of(ROOT / 'results/grounded_first_start_20261105/first_choices.csv')}
    with zipfile.ZipFile(ARCHIVES / 'reports.zip') as archive:
        cards = {Path(n).stem: member(archive, n) for n in archive.namelist()
                 if 'grounded_first_start' in n and 'initial_inputs' in n and n.endswith('.json')}
    rows = []
    for task in TASKS:
        pick = choices[task]['candidate_id']
        ids = [e['candidate_id'] for e in cards[task]['input_only_examples']]
        values = sorted(value[task][i] for i in ids)
        row = dict(task=task, pick=pick, in_menu=pick in ids, menu_size=len(ids),
                   menu_share_of_pool_pct=100 * len(ids) / len(value[task]), pick_value=value[task][pick],
                   rank_in_menu=1 + sum(v > value[task][pick] for v in values),
                   menu_median=st.median(values), above_menu_median=value[task][pick] > st.median(values),
                   pool_percentile=100 * sum(v < value[task][pick] for v in value[task].values()) / len(value[task]),
                   pick_fraction_of_range='', all_axes_max=False, top_corner_points='', top_corner_mean='')
        if task in scaled:
            fractions = scaled[task][pick]
            score = {i: st.mean(scaled[task][i]) for i in ids}
            corner = [i for i in ids if abs(score[i] - max(score.values())) < 1e-9]
            row.update(pick_fraction_of_range=json.dumps([round(x, 3) for x in fractions]),
                       all_axes_max=min(fractions) > .999, top_corner_points=len(corner),
                       top_corner_mean=st.mean(value[task][i] for i in corner))
        rows.append(row)
    write_rows('grounded_picks.csv', rows)
    return dict(above_menu_median=sum(r['above_menu_median'] for r in rows), tasks=len(rows),
                all_in_menu=all(r['in_menu'] for r in rows),
                all_axes_max=[r['task'] for r in rows if r['all_axes_max']])


def warm_starts(value, top, grounded):
    """Warm random 4+16 BO rescored on all 20 observations, the 0+20 metric."""
    index = {}
    for archive in ARCHIVES.glob('run__*.zip'):
        with zipfile.ZipFile(archive) as handle:
            index.update({n: archive for n in handle.namelist() if n.endswith('state.json')})
    rows = []
    auc = {}
    for record in rows_of(ROOT / 'results/cohorts/warm_4plus16/metrics.csv'):
        if record['method'] != 'bo':
            continue
        path = record['state_path'].replace('experiments/chemist_system/data/runs/', 'experiments/runs/')
        with zipfile.ZipFile(index[path]) as handle:
            state = member(handle, path)
        task, seed = record['task'], int(record['seed'])
        observed = [h['raw_value'] for h in state['history']]
        if len(observed) != 20:
            start = state.get('initial_ids') or state.get('initial')
            observed = [value[task][i] for i in start] + observed
        assert len(observed) == 20, (task, seed, len(observed))
        best = total = 0.
        for y in observed:
            best = max(best, y)
            total += best
        auc[(task, seed)] = total / 20
        rows.append(dict(task=task, seed=seed, SearchAUC20=auc[(task, seed)], best_initial=max(observed[:4]),
                         reported_16_term_SearchAUC=float(record['SearchAUC'])))
    write_rows('warm_bo_searchauc20.csv', rows)
    out = {}
    for name, method in (('llm_first_minus_warm', 'first'), ('random_one_point_minus_warm', 'bo')):
        diff = [100 * (st.mean(grounded[(t, s, method)]['SearchAUC'] for s in SEEDS) -
                       st.mean(auc[(t, s)] for s in SEEDS)) / top[t] for t in TASKS]
        out[name] = dict(mean=st.mean(diff), by_task=dict(zip(TASKS, diff)), tasks_above_zero=sum(d > 0 for d in diff),
                         sign_flip_p=sign_flip_p(diff), interval90=interval90(diff))
    # Seeds 1-3 have a documented uniform input-only draw; seed 0's rule was not confirmed.
    diff = [100 * (st.mean(grounded[(t, s, 'first')]['SearchAUC'] for s in SEEDS) -
                   st.mean(auc[(t, s)] for s in (1, 2, 3))) / top[t] for t in TASKS]
    out['llm_first_minus_warm_seeds_1_3'] = dict(mean=st.mean(diff), by_task=dict(zip(TASKS, diff)),
                                                 tasks_above_zero=sum(d > 0 for d in diff),
                                                 sign_flip_p=sign_flip_p(diff), interval90=interval90(diff))
    return out


def head_start(top):
    trace = defaultdict(dict)
    for r in rows_of(GROUNDED / 'trajectories.csv'):
        trace[(r['task'], int(r['seed']), r['method'])][int(r['query'])] = float(r['cumulative_best'])
    gap = {t: {q: st.mean(100 * (trace[(t, s, 'first')][q] - trace[(t, s, 'bo')][q]) / top[t] for s in SEEDS)
               for q in range(1, 21)} for t in TASKS}
    write_rows('head_start_by_query.csv', [
        dict(query=q, **{t: gap[t][q] for t in TASKS}, mean=st.mean(gap[t][q] for t in TASKS)) for q in range(1, 21)])
    early = st.mean(sum(gap[t][q] for q in range(1, 11)) / 20 for t in TASKS)
    late = st.mean(sum(gap[t][q] for q in range(11, 21)) / 20 for t in TASKS)
    return dict(mean_gap_at_query={q: st.mean(gap[t][q] for t in TASKS) for q in (1, 3, 5, 10, 15, 20)},
                auc_gap_from_queries_1_10=early, auc_gap_from_queries_11_20=late, share_early=early / (early + late))


def decomposition(top, grounded):
    cells = {}
    rows = []
    out = {}
    for name, a, b in (('same_first_minus_bo', 'first', 'bo'), ('controller_minus_same_first', 'hyp', 'first'),
                       ('controller_minus_bo', 'hyp', 'bo'), ('lgbo_minus_same_first', 'lgbo', 'first'),
                       ('rbo_minus_same_first', 'rbo', 'first')):
        cell = {(t, s): 100 * (grounded[(t, s, a)]['SearchAUC'] - grounded[(t, s, b)]['SearchAUC']) / top[t]
                for t in TASKS for s in SEEDS}
        cells[name] = cell
        by_task = {t: st.mean(cell[(t, s)] for s in SEEDS) for t in TASKS}
        raw = [grounded[(t, s, a)]['SearchAUC'] - grounded[(t, s, b)]['SearchAUC'] for t in TASKS for s in SEEDS]
        wins, losses = sum(d > 1e-9 for d in raw), sum(d < -1e-9 for d in raw)
        out[name] = dict(mean=st.mean(by_task.values()), median=st.median(by_task.values()), by_task=by_task,
                         wins=wins, ties=32 - wins - losses, losses=losses,
                         sign_flip_p=sign_flip_p(list(by_task.values())), bootstrap95=task_clustered_ci(cell))
        rows += [dict(contrast=name, task=t, percent_of_pool_max=by_task[t]) for t in TASKS]
    write_rows('decomposition_by_task.csv', rows)
    return out


def funnel(top, grounded):
    mechanisms = rows_of(ROOT / 'results/hypothesis_scope_multiseed_20261105/mechanisms.csv')
    path = defaultdict(dict)
    for r in rows_of(GROUNDED / 'trajectories.csv'):
        path[(r['task'], int(r['seed']), r['method'])][int(r['query'])] = r['candidate_id']
    rows = []
    for m in mechanisms:
        task, seed = m['task'], int(m['seed'])
        changed = [int(q) for q in m['changed_queries'].split(';') if q]
        first_gap = next((q for q in range(1, 21) if path[(task, seed, 'hyp')][q] != path[(task, seed, 'first')][q]), '')
        rows.append(dict(task=task, seed=seed, scoped_queries=int(m['scoped_queries']), changed_choices=len(changed),
                         changed_queries=m['changed_queries'], changed_before_first_refresh=sum(q < 4 for q in changed),
                         first_divergence_from_same_first=first_gap,
                         delta_percent_of_pool_max=100 * (grounded[(task, seed, 'hyp')]['SearchAUC'] -
                                                          grounded[(task, seed, 'first')]['SearchAUC']) / top[task]))
    write_rows('funnel_by_trajectory.csv', rows)
    differ = [r for r in rows if abs(r['delta_percent_of_pool_max']) > 1e-9]
    feedback_only = [r for r in rows if r['changed_choices'] and not r['changed_before_first_refresh']]
    return dict(post_first_choices=32 * 19, scoped=sum(r['scoped_queries'] for r in rows),
                changed=sum(r['changed_choices'] for r in rows),
                changed_before_first_refresh=sum(r['changed_before_first_refresh'] for r in rows),
                trajectories_diverging=sum(r['first_divergence_from_same_first'] != '' for r in rows),
                trajectories_with_different_auc=len(differ),
                better=sum(r['delta_percent_of_pool_max'] > 0 for r in differ),
                worse=sum(r['delta_percent_of_pool_max'] < 0 for r in differ),
                refresh_only_trajectories=len(feedback_only),
                refresh_only_effect_range=[min(r['delta_percent_of_pool_max'] for r in feedback_only),
                                           max(r['delta_percent_of_pool_max'] for r in feedback_only)])


def low_middle_starts(grounded):
    metrics = rows_of(ROOT / 'results/suzuki_initial_yield_20261104/metrics.csv')
    rows = []
    out = {}
    for case, task in (('1', 'c1'), ('2', 'c2'), ('3', 'c3'), ('4', 'c4')):
        distinct = {r['initial_id']: r for r in metrics if r['case'] == case and r['arm'] in ('low', 'middle')}
        auc = [float(r['SearchAUC20']) for r in distinct.values()]
        first = [float(r['initial_yield']) for r in distinct.values()]
        percentile = lambda x: 100 * sum(a < x for a in auc) / len(auc)
        row = dict(task=task, distinct_starts=len(auc), median_SearchAUC20=st.median(auc), max_SearchAUC20=max(auc),
                   llm_first_mean_percentile=st.mean(percentile(grounded[(task, s, 'first')]['SearchAUC']) for s in SEEDS),
                   random_bo_mean_percentile=st.mean(percentile(grounded[(task, s, 'bo')]['SearchAUC']) for s in SEEDS),
                   spearman_first_yield_vs_auc=spearman(first, auc))
        rows.append(row)
        out[task] = row
    write_rows('low_middle_start_reference.csv', rows)
    return out


def main():
    OUT.mkdir(exist_ok=True)
    value, top, scaled = load()
    grounded = {}
    for r in rows_of(GROUNDED / 'per_seed_results.csv'):
        grounded[(r['task'], int(r['seed']), r['method'])] = dict(SearchAUC=float(r['SearchAUC']))
    summary = dict(status='EXPLORATORY_SAVED_EVIDENCE', new_model_calls=0, new_GP_fits=0, new_oracle_queries=0,
                   five_point_menus=five_point_menus(value, scaled), grounded_picks=grounded_picks(value, top, scaled),
                   warm_random_starts=warm_starts(value, top, grounded), head_start=head_start(top),
                   decomposition=decomposition(top, grounded), funnel=funnel(top, grounded),
                   low_middle_starts=low_middle_starts(grounded))
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
