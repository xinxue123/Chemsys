"""Method-level evidence for the hypothesis-scope controller from saved arrays.

Offline and standard-library only. Reads sealed archives in memory. It issues no
model calls, GP fits or oracle experiments, and writes only to
results/hypothesis_scope_method_evidence_20261105/.

Candidate outcomes are read from the post-hoc pool tables, as an evaluator. They
were never available to the controller. One-step counterfactuals hold the actual
prefix, saved GP scores and LLM outputs fixed, so they are not trajectory effects.
All analyses were chosen after the results existed and are exploratory.

    python code/analysis/analyze_hypothesis_scope_method.py
"""
import ast
import csv
import io
import json
import math
import statistics as st
import struct
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARCHIVES = ROOT / 'archives'
GROUNDED = ROOT / 'results/grounded_baselines_20261105'
OUT = ROOT / 'results/hypothesis_scope_method_evidence_20261105'
TASKS = ('c1', 'c2', 'c3', 'c4', 'perera', 'doyle', 'snar', 'hplc')
SEEDS = range(4)
METHODS = ('hyp', 'first', 'lgbo', 'rbo', 'bo')
TOP_FRACTION = .05
MIN_KF, MIN_P = .2, .5


def rows_of(path):
    with open(path, encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_rows(name, rows):
    with open(OUT / name, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def sign_flip_p(values):
    observed = abs(sum(values))
    hits = sum(abs(sum(v if m >> i & 1 else -v for i, v in enumerate(values))) >= observed - 1e-12
               for m in range(2 ** len(values)))
    return hits / 2 ** len(values)


def binomial_p(wins, losses):
    n = wins + losses
    if not n:
        return 1.
    return min(1., 2 * sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n)


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
    return top / bottom if bottom else float('nan')


def load_npy(raw):
    """Minimal .npy reader for the bool, int64 and float arrays saved by the study."""
    assert raw[:6] == b'\x93NUMPY'
    major = raw[6]
    size = struct.unpack('<H', raw[8:10])[0] if major == 1 else struct.unpack('<I', raw[8:12])[0]
    start = 10 if major == 1 else 12
    header = ast.literal_eval(raw[start:start + size].decode('latin1'))
    body = raw[start + size:]
    n = 1
    for d in header['shape']:
        n *= d
    code = header['descr'].lstrip('<|=')
    if code == 'b1':
        return [bool(x) for x in body[:n]]
    fmt = {'f8': 'd', 'f4': 'f', 'i8': 'q', 'i4': 'i'}[code]
    return list(struct.unpack('<%d%s' % (n, fmt), body[:n * struct.calcsize(fmt)]))


def load_npz(blob):
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        return {n[:-4]: load_npy(archive.read(n)) for n in archive.namelist()}


def p_improve(note, incumbent):
    f = note['forecast']
    return .5 * (1 + math.erf((f['mean_raw'] - incumbent) / f['sd_raw'] / math.sqrt(2)))


def eligible(note, incumbent):
    if note['permission'] not in ('provisional', 'active', 'probation'):
        return False
    p = p_improve(note, incumbent)
    if note['direction'] == 'favour' and p < MIN_P:
        return False
    if note['direction'] == 'avoid' and p > MIN_P:
        return False
    return note['permission'] == 'probation' or note['KF']['reliability'] >= MIN_KF


def scoped(notes, arrays, available):
    mask = list(available)
    for note in notes:
        region = arrays['scope_' + note['semantic_key']]
        mask = [m and (r if note['direction'] == 'favour' else not r) for m, r in zip(mask, region)]
    return mask


def argmax(arrays, mask):
    best = None
    for j, keep in enumerate(mask):
        if keep:
            key = (-arrays['base_logei'][j], arrays['tie_priority'][j])
            if best is None or key < best[0]:
                best = (key, j)
    return best[1]


def trajectories():
    seed0 = zipfile.ZipFile(ARCHIVES / 'run__hypothesis_scope_loop_20261105.zip')
    rest = zipfile.ZipFile(ARCHIVES / 'run__hypothesis_scope_multiseed_20261105.zip')
    for task in TASKS:
        for seed in SEEDS:
            if seed == 0:
                archive, base = seed0, f'experiments/runs/hypothesis_scope_loop_20261105/runs/{task}'
            else:
                archive, base = rest, f'experiments/runs/hypothesis_scope_multiseed_20261105/runs/s{seed}/{task}_hyp'
            commits = {q: json.loads(archive.read(f'{base}/rounds/{q:02d}_commit.json').decode('utf-8-sig')) for q in range(1, 21)}
            arrays = {q: load_npz(archive.read(f'{base}/rounds/{q:02d}_GP.npz')) for q in range(2, 21)}
            yield task, seed, commits, arrays


def method_table(top):
    data = {}
    for r in rows_of(GROUNDED / 'per_seed_results.csv'):
        data[(r['task'], int(r['seed']), r['method'])] = r
    auc = {k: 100 * float(v['SearchAUC']) / top[k[0]] for k, v in data.items()}
    final = {k: 100 * float(v['final_best']) / top[k[0]] for k, v in data.items()}
    calls = {k: int(v['post_first_logical_LLM_calls'] or 0) for k, v in data.items()}
    rows = []
    for task in TASKS:
        means = {m: st.mean(auc[(task, s, m)] for s in SEEDS) for m in METHODS}
        best = max(means.values())
        order = sorted(METHODS, key=lambda m: -means[m])
        for m in METHODS:
            rows.append(dict(task=task, method=m, auc_pct_of_max=means[m], rank=order.index(m) + 1, regret_to_best=best - means[m],
                             stream_sd=st.stdev(auc[(task, s, m)] for s in SEEDS),
                             final_pct_of_max=st.mean(final[(task, s, m)] for s in SEEDS)))
    write_rows('method_by_task.csv', rows)
    out = {}
    for m in METHODS:
        mine = [r for r in rows if r['method'] == m]
        vs_first = [auc[(t, s, m)] - auc[(t, s, 'first')] for t in TASKS for s in SEEDS]
        vs_bo = [auc[(t, s, m)] - auc[(t, s, 'bo')] for t in TASKS for s in SEEDS]
        out[m] = dict(mean_auc=st.mean(r['auc_pct_of_max'] for r in mine), mean_rank=st.mean(r['rank'] for r in mine),
                      ranks=[r['rank'] for r in mine], mean_regret=st.mean(r['regret_to_best'] for r in mine),
                      max_regret=max(r['regret_to_best'] for r in mine),
                      median_stream_sd=st.median(r['stream_sd'] for r in mine),
                      worst_cell_vs_same_first=min(vs_first), cells_5pt_below_same_first=sum(d < -5 for d in vs_first),
                      worst_cell_vs_bo=min(vs_bo), cells_5pt_below_bo=sum(d < -5 for d in vs_bo),
                      mean_post_first_calls=st.mean(calls[(t, s, m)] for t in TASKS for s in SEEDS),
                      mean_final=st.mean(r['final_pct_of_max'] for r in mine))
    return out


def main():
    OUT.mkdir(exist_ok=True)
    value, top = {}, {}
    ranked = {}
    for task in TASKS:
        pool = rows_of(ROOT / 'data/pools' / task / 'candidates.csv')
        value[task] = [float(r['objective_raw']) for r in pool]
        ids = [r['candidate_id'] for r in pool]
        ranked[task] = dict(ids=ids, ix={c: i for i, c in enumerate(ids)})
        top[task] = max(value[task])

    summary = dict(status='EXPLORATORY_SAVED_EVIDENCE', new_model_calls=0, new_GP_fits=0, new_oracle_queries=0,
                   methods=method_table(top))
    replay_errors = 0
    decisions = 0
    hypotheses = {}
    changed = []
    gate = []
    scope_rounds = []
    for task, seed, commits, arrays in trajectories():
        v, ix, ids = value[task], ranked[task]['ix'], ranked[task]['ids']
        n = len(v)
        cut = sorted(v, reverse=True)[max(1, int(TOP_FRACTION * n)) - 1]
        is_top = [x >= cut for x in v]
        best_ids = [j for j in range(n) if v[j] >= max(v) - 1e-9]  # Perera and Doyle have tied pool maxima.
        order = sorted(range(n), key=lambda j: v[j])
        pct = [0.] * n
        for rank, j in enumerate(order):
            pct[j] = 100 * rank / (n - 1)
        for q in range(2, 21):
            c, a = commits[q], arrays[q]
            d = c['decision']
            notes = c['working_knowledge']['notes']
            inc = d['incumbent_before']
            available = a['available']
            active = [x for x in notes if eligible(x, inc)]
            mask = scoped(active, a, available)
            fallback = not active or not any(mask)
            bo = ix[d['unrestricted_BO_id']]
            pick = bo if fallback else argmax(a, mask)
            decisions += 1
            replay_errors += ids[pick] != d['selected_id']
            sel = ix[d['selected_id']]
            record = lambda j: max(v[j], inc)
            if sel != bo:
                ids_active = sorted(x['id'] for x in active)
                installed = min((x['installed_query'] for x in active), default=None)
                # Greedy control: the exploitation pick a pure GP would make at this same prefix.
                greedy = max((j for j in range(n) if available[j]), key=lambda j: (a['mu_y'][j], -a['tie_priority'][j]))
                changed.append(dict(task=task, seed=seed, query=q, selected=ids[sel], unrestricted_BO=ids[bo],
                                    selected_value=v[sel], BO_value=v[bo], incumbent=inc,
                                    record_gain_pct=100 * (record(sel) - record(bo)) / top[task],
                                    value_gain_pct=100 * (v[sel] - v[bo]) / top[task],
                                    greedy_GP=ids[greedy], greedy_value=v[greedy],
                                    value_gain_vs_greedy_pct=100 * (v[sel] - v[greedy]) / top[task],
                                    record_gain_vs_greedy_pct=100 * (record(sel) - record(greedy)) / top[task],
                                    source='initial_hypothesis' if installed == 1 else 'refreshed_hypothesis',
                                    active=';'.join(ids_active), acquisition_gap=d['qLogEI_opportunity_gap']))
            # Gate bypass: every installed hypothesis constrains, ignoring permission and credibility.
            forced = scoped(notes, a, available) if notes else list(available)
            no_gate = bo if (not notes or not any(forced)) else argmax(a, forced)
            if no_gate != sel:
                blocked = [x for x in notes if x not in active]
                gate.append(dict(task=task, seed=seed, query=q, actual=ids[sel], no_gate=ids[no_gate], actual_value=v[sel],
                                 no_gate_value=v[no_gate], incumbent=inc,
                                 record_gain_pct=100 * (record(sel) - record(no_gate)) / top[task],
                                 value_gain_pct=100 * (v[sel] - v[no_gate]) / top[task],
                                 blocked=';'.join(sorted(x['id'] for x in blocked)),
                                 blocked_permissions=';'.join(sorted(x['permission'] for x in blocked))))
            if active and any(mask):
                unq = [j for j in range(n) if available[j]]
                top_left = sum(is_top[j] for j in unq)
                kept = sum(is_top[j] for j in unq if mask[j])
                scope_rounds.append(dict(task=task, seed=seed, query=q, share_of_unqueried=sum(mask) / len(unq),
                                         top_share_kept=kept / top_left if top_left else '',
                                         optimum_available=any(available[j] for j in best_ids), optimum_kept=any(available[j] and mask[j] for j in best_ids),
                                         changed=sel != bo))
            for note in notes:
                key = (task, note['semantic_key'], note['direction']) if note['installed_query'] == 1 else \
                      (task, seed, note['semantic_key'], note['direction'])
                region = a['scope_' + note['semantic_key']]
                row = hypotheses.get(key)
                if row is None:
                    unq = [j for j in range(n) if available[j]]
                    inside = [j for j in unq if region[j]]
                    if len(inside) < 4 or len(inside) == len(unq):
                        continue
                    base = sum(is_top[j] for j in unq) / len(unq)
                    lift = (sum(is_top[j] for j in inside) / len(inside)) / base if base else float('nan')
                    # Favour scopes face the GP's highest-mean set; avoid scopes its lowest-mean set.
                    sign = -1 if note['direction'] == 'favour' else 1
                    gp_rank = sorted(unq, key=lambda j: sign * a['mu_y'][j])[:len(inside)]
                    gp_lift = (sum(is_top[j] for j in gp_rank) / len(gp_rank)) / base if base else float('nan')
                    row = dict(task=task, seed='all' if note['installed_query'] == 1 else seed, id=note['id'],
                               direction=note['direction'], installed_query=note['installed_query'],
                               scope_share=len(inside) / len(unq), lift_top5=lift, gp_same_size_lift=gp_lift,
                               first_pause_reasons='',
                               mean_percentile=st.mean(pct[j] for j in inside), contains_optimum=any(region[j] for j in best_ids),
                               forecast_mean=note['forecast']['mean_raw'], in_scope_mean=st.mean(v[j] for j in inside),
                               rounds_constraining=0, rounds_installed=0, final_permission='', final_reliability='',
                               scored=0, ever_paused=False, claim=note['claim'][:160])
                    hypotheses[key] = row
                row['rounds_installed'] += 1
                row['rounds_constraining'] += note in active
                row['final_permission'] = note['permission']
                row['final_reliability'] = note['KF']['reliability']
                row['scored'] = len(note['feedback'])
                row['ever_paused'] = row['ever_paused'] or note['permission'] == 'paused'
                if note['permission'] == 'paused' and not row['first_pause_reasons']:
                    row['first_pause_reasons'] = ';'.join(note['pause_reasons'])

    rows = list(hypotheses.values())
    write_rows('hypothesis_quality.csv', rows)
    write_rows('changed_choices.csv', changed)
    write_rows('gate_counterfactuals.csv', gate)
    write_rows('scope_rounds.csv', scope_rounds)

    def quality(subset):
        lifts = [r['lift_top5'] for r in subset if r['lift_top5'] == r['lift_top5']]
        gp = [r['gp_same_size_lift'] for r in subset if r['gp_same_size_lift'] == r['gp_same_size_lift']]
        return dict(n=len(subset), median_lift=st.median(lifts) if lifts else None, mean_lift=st.mean(lifts) if lifts else None,
                    share_lift_above_1=sum(x > 1 for x in lifts) / len(lifts) if lifts else None,
                    median_percentile=st.median(r['mean_percentile'] for r in subset) if subset else None,
                    contains_optimum=sum(r['contains_optimum'] for r in subset),
                    median_scope_share=st.median(r['scope_share'] for r in subset) if subset else None,
                    median_gp_lift=st.median(gp) if gp else None)

    favour = [r for r in rows if r['direction'] == 'favour']
    avoid = [r for r in rows if r['direction'] == 'avoid']
    refreshed_favour = [r for r in favour if r['installed_query'] > 1]
    paired = [(r['lift_top5'], r['gp_same_size_lift']) for r in refreshed_favour
              if r['lift_top5'] == r['lift_top5'] and r['gp_same_size_lift'] == r['gp_same_size_lift']]
    by_task = []
    for task in TASKS:
        sub = [r for r in favour if r['task'] == task]
        lifts = [r['lift_top5'] for r in sub if r['lift_top5'] == r['lift_top5']]
        if lifts:
            by_task.append(st.mean(math.log(max(x, .05)) for x in lifts))
    scored = [r for r in favour if r['scored'] > 0 and r['lift_top5'] == r['lift_top5']]
    good = [r for r in scored if r['lift_top5'] > 1]
    bad = [r for r in scored if r['lift_top5'] <= 1]
    summary['hypotheses'] = dict(
        total=len(rows), favour=quality(favour), avoid=quality(avoid), initial_favour=quality([r for r in favour if r['installed_query'] == 1]),
        refreshed_favour=quality(refreshed_favour),
        early_refresh_favour=quality([r for r in refreshed_favour if r['installed_query'] <= 7]),
        late_refresh_favour=quality([r for r in refreshed_favour if r['installed_query'] > 7]),
        favour_task_log_lift=dict(zip(TASKS, by_task)), favour_log_lift_sign_flip_p=sign_flip_p(by_task) if len(by_task) == 8 else None,
        refreshed_llm_beats_gp=sum(a > b for a, b in paired), refreshed_gp_beats_llm=sum(a < b for a, b in paired),
        refreshed_llm_gp_ties=sum(a == b for a, b in paired), refreshed_llm_gp_binomial_p=binomial_p(sum(a > b for a, b in paired), sum(a < b for a, b in paired)),
        kf_vs_lift_spearman=spearman([r['final_reliability'] for r in scored], [r['lift_top5'] for r in scored]) if len(scored) > 2 else None,
        good_scored=len(good), good_ever_paused=sum(r['ever_paused'] for r in good),
        bad_scored=len(bad), bad_ever_paused=sum(r['ever_paused'] for r in bad),
        good_mean_constraining_rounds=st.mean(r['rounds_constraining'] for r in good) if good else None,
        bad_mean_constraining_rounds=st.mean(r['rounds_constraining'] for r in bad) if bad else None)

    def one_step(items, key='record_gain_pct'):
        better = sum(r[key] > 1e-9 for r in items)
        worse = sum(r[key] < -1e-9 for r in items)
        return dict(n=len(items), better=better, tied=len(items) - better - worse, worse=worse,
                    mean_gain=st.mean(r[key] for r in items) if items else None, binomial_p=binomial_p(better, worse))

    refreshed = [r for r in changed if r['source'] == 'refreshed_hypothesis']
    task_means = {}
    for task in TASKS:
        sub = [r for r in refreshed if r['task'] == task]
        if sub:
            task_means[task] = dict(n=len(sub), value_vs_BO=st.mean(r['value_gain_pct'] for r in sub),
                                    value_vs_greedy=st.mean(r['value_gain_vs_greedy_pct'] for r in sub),
                                    record_vs_BO=st.mean(r['record_gain_pct'] for r in sub),
                                    record_vs_greedy=st.mean(r['record_gain_vs_greedy_pct'] for r in sub))
    summary['refreshed_by_task'] = task_means
    summary['refreshed_task_sign_flip'] = {k: sign_flip_p([t[k] for t in task_means.values()])
                                           for k in ('value_vs_BO', 'value_vs_greedy', 'record_vs_BO', 'record_vs_greedy')}
    summary['refreshed_vs_greedy'] = dict(value=one_step(refreshed, 'value_gain_vs_greedy_pct'),
                                          record=one_step(refreshed, 'record_gain_vs_greedy_pct'))
    summary['acquisition_gap'] = dict(refreshed_median=st.median(r['acquisition_gap'] for r in refreshed),
                                      initial_median=st.median(r['acquisition_gap'] for r in changed if r['source'] == 'initial_hypothesis'))
    summary['install_timing'] = {d: dict(sorted(Counter(r['installed_query'] for r in rows if r['direction'] == d).items()))
                                 for d in ('favour', 'avoid')}
    summary['pause_reasons_favour'] = dict(Counter(x for r in favour for x in r['first_pause_reasons'].split(';') if x))
    summary['pause_reasons_avoid'] = dict(Counter(x for r in avoid for x in r['first_pause_reasons'].split(';') if x))
    avoid_pairs = [(r['lift_top5'], r['gp_same_size_lift']) for r in avoid if r['installed_query'] > 1]
    summary['avoid_vs_gp_lowest'] = dict(llm_lower=sum(a < b for a, b in avoid_pairs), gp_lower=sum(a > b for a, b in avoid_pairs),
                                         ties=sum(a == b for a, b in avoid_pairs), median_llm=st.median(a for a, _ in avoid_pairs),
                                         median_gp=st.median(b for _, b in avoid_pairs))
    summary['constraining_rounds'] = dict(favour_total=sum(r['rounds_constraining'] for r in favour),
                                          avoid_total=sum(r['rounds_constraining'] for r in avoid),
                                          favour_median=st.median(r['rounds_constraining'] for r in favour),
                                          avoid_median=st.median(r['rounds_constraining'] for r in avoid))
    summary['replay'] = dict(decisions=decisions, replay_errors=replay_errors)
    summary['changed_choices'] = dict(all_record=one_step(changed), all_value=one_step(changed, 'value_gain_pct'),
                                      initial_value=one_step([r for r in changed if r['source'] == 'initial_hypothesis'], 'value_gain_pct'),
                                      refreshed_value=one_step([r for r in changed if r['source'] == 'refreshed_hypothesis'], 'value_gain_pct'),
                                      refreshed_record=one_step([r for r in changed if r['source'] == 'refreshed_hypothesis']),
                                      by_task=dict(Counter(r['task'] for r in changed)))
    summary['gate'] = dict(record=one_step(gate), value=one_step(gate, 'value_gain_pct'), by_task=dict(Counter(r['task'] for r in gate)))
    kept = [r['top_share_kept'] for r in scope_rounds if r['top_share_kept'] != '']
    opt = [r for r in scope_rounds if r['optimum_available']]
    summary['scope_rounds'] = dict(n=len(scope_rounds), median_share_of_unqueried=st.median(r['share_of_unqueried'] for r in scope_rounds),
                                   median_top_share_kept=st.median(kept), rounds_optimum_available=len(opt),
                                   rounds_optimum_excluded=sum(not r['optimum_kept'] for r in opt))
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
