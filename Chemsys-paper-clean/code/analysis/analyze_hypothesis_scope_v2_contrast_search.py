"""Contrast-neighbourhood search: selection, success rule and ablations, from saved run states.

Standard library only: no model calls, queries or GP fits. Rules come from
docs/hypothesis_scope_v2_registration_20261105.md. Effects are % of task pool maximum.

    python code/analysis/analyze_hypothesis_scope_v2_contrast_search.py <v2 run folder> <output folder>
"""
import json
import random
import statistics as st
import sys
from pathlib import Path

from analyze_hypothesis_scope_v2_gp_followups import DEV, SEED, metrics, paired, read, rows_of, tops, write_rows

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ('n6m1', 'n6m2', 'n11m1', 'n11m2', 'a6m1')
BASELINES = dict(development=dict(LGBO=84.93, same_first=82.83, reasoning_BO=79.05, pure_BO=76.55), held_out=dict(pure_BO=76.05, same_first=56.48, LGBO=54.85))
BEST_FINAL = dict(development=96.42, held_out=89.66)


def done(path):
    return path.exists() and read(path)['status'] == 'COMPLETE'


def score(base, arm, task, seed, top):
    path = base/'gp_followup'/arm/task/f's{seed}'/'state.json'
    if not done(path):
        return None
    auc, final, _ = metrics(read(path), top)
    state = read(path)
    return dict(auc=auc, final=final, local=sum(r['decision'].get('reason') == 'incumbent_contrast_scope' for r in state['decisions']),
                hedged=sum(r['decision'].get('reason') == 'failed_first_hedge' for r in state['decisions']))


def references(base, top):
    saved = {(r['arm'], r['task'], int(r['seed'])): r for r in rows_of(ROOT/'results/hypothesis_scope_v2_20261105/per_seed_results.csv')}
    out = {}
    for t in DEV:
        for s in range(4):
            for arm, name in (('first_env', 'first'), ('bo_env', 'bo'), ('h1', 'h1')):
                r = saved[arm, t, s]
                out[name, t, s] = dict(auc=100*float(r['SearchAUC'])/top[t], final=100*float(r['final_best'])/top[t])
    for s in range(8):
        for arm in ('first', 'bo', 'h1', 'lgbo'):
            auc, final, _ = metrics(read(base/'heldout/runs'/arm/f's{s}'/'aryl/state.json'), 100.)
            out[arm, 'aryl', s] = dict(auc=auc, final=final)
    return out


def table(base, arm, top):
    rows = {(t, s): score(base, arm, t, s, top[t]) for t in DEV for s in range(4)}
    return rows if all(rows.values()) else None


def held(base, arm):
    rows = {('aryl', s): score(base, arm, 'aryl', s, 100.) for s in range(8)}
    return rows if all(rows.values()) else None


def by_task(rows):
    return {t: st.mean(rows[t, s]['auc'] for s in range(4)) for t in DEV}


def compare(rows, ref, name, rng):
    keys = list(rows)
    if keys[0][0] == 'aryl':
        return paired([rows[k]['auc'] - ref[(name,) + k]['auc'] for k in keys], [rows[k]['final'] - ref[(name,) + k]['final'] for k in keys], rng)
    auc = [st.mean(rows[t, s]['auc'] - ref[name, t, s]['auc'] for s in range(4)) for t in DEV]
    final = [st.mean(rows[t, s]['final'] - ref[name, t, s]['final'] for s in range(4)) for t in DEV]
    return paired(auc, final, rng)


def summary(rows):
    return dict(auc=st.mean(r['auc'] for r in rows.values()), final=st.mean(r['final'] for r in rows.values()), sd=st.stdev(r['auc'] for r in rows.values()),
                local=st.mean(r['local'] for r in rows.values()), hedged=sum(r['hedged'] > 0 for r in rows.values()))


def verdict(values, data):
    means = BASELINES[data]
    return dict(auc=values['auc'], final=values['final'], beats={k: values['auc'] > v for k, v in means.items()}, final_ok=values['final'] >= BEST_FINAL[data] - .5,
                passes=all(values['auc'] > v for v in means.values()) and values['final'] >= BEST_FINAL[data] - .5)


def main():
    base, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    top = tops()
    ref = references(base, top)
    first_final = st.mean(ref['first', t, s]['final'] for t in DEV for s in range(4))
    result = dict(status='EXPLORATORY', new_model_calls=0, variants={})
    for v in VARIANTS:
        rows = table(base, v, top)
        if rows:
            s = summary(rows)
            result['variants'][v] = dict(s, by_task={t: round(x, 2) for t, x in by_task(rows).items()}, eligible=s['final'] >= first_final - .5, vs_first=compare(rows, ref, 'first', rng), vs_bo=compare(rows, ref, 'bo', rng))
    eligible = {k: r['auc'] for k, r in result['variants'].items() if r['eligible']}
    result['first_final'] = first_final
    result['selected'] = max(eligible, key=eligible.get) if len(result['variants']) == len(VARIANTS) and eligible else None
    chosen = result['selected']
    if chosen:
        for name, arm in (('primary', 'h_' + chosen), ('no_hedge', chosen), ('no_LLM', 'b_' + chosen)):
            block = {}
            rows = table(base, arm, top)
            if rows:
                s = summary(rows)
                block['development'] = dict(s, by_task={t: round(x, 2) for t, x in by_task(rows).items()}, verdict=verdict(s, 'development'), vs_first=compare(rows, ref, 'first', rng), vs_bo=compare(rows, ref, 'bo', rng), vs_h1=compare(rows, ref, 'h1', rng))
            rows = held(base, arm)
            if rows:
                s = summary(rows)
                block['held_out'] = dict(s, by_stream={k[1]: round(r['auc'], 2) for k, r in rows.items()}, verdict=verdict(s, 'held_out'), vs_first=compare(rows, ref, 'first', rng), vs_bo=compare(rows, ref, 'bo', rng),
                                         vs_h1=compare(rows, ref, 'h1', rng), vs_lgbo=compare(rows, ref, 'lgbo', rng))
            result[name] = block
    round2 = {}
    for v in ('r2a', 'r2b', 'r2c'):
        rows = table(base, v, top)
        if rows:
            s = summary(rows)
            round2[v] = dict(s, by_task={t: round(x, 2) for t, x in by_task(rows).items()}, eligible=s['final'] >= first_final - .5, vs_first=compare(rows, ref, 'first', rng), vs_h1=compare(rows, ref, 'h1', rng), verdict=verdict(s, 'development'))
    result['round2'] = round2
    eligible = {k: r['auc'] for k, r in round2.items() if r['eligible']}
    pick = max(eligible, key=eligible.get) if len(round2) == 3 and eligible else None
    result['round2_selected'] = pick
    if pick:
        for name, arm in (('round2_primary', 'c_' + pick), ('round2_llm_first', pick)):
            rows = held(base, arm)
            if rows:
                s = summary(rows)
                result[name] = dict(s, by_stream={k[1]: round(r['auc'], 2) for k, r in rows.items()}, verdict=verdict(s, 'held_out'), vs_first=compare(rows, ref, 'first', rng), vs_bo=compare(rows, ref, 'bo', rng),
                                    vs_h1=compare(rows, ref, 'h1', rng), vs_lgbo=compare(rows, ref, 'lgbo', rng))
    if all(done(base/'gp_followup/c_r2a/aryl'/f's{s}'/'state.json') for s in range(24)):
        mine, bo = [], []
        for s in range(24):
            mine.append(metrics(read(base/'gp_followup/c_r2a/aryl'/f's{s}'/'state.json'), 100.))
            folder = base/'heldout/runs/bo'/f's{s}'/'aryl' if s < 8 else base/'gp_followup/bo_extra/aryl'/f's{s}'
            bo.append(metrics(read(folder/'state.json'), 100.))
        result['precision_24'] = dict(design_auc=st.mean(x[0] for x in mine), bo_auc=st.mean(x[0] for x in bo), design_final=st.mean(x[1] for x in mine), bo_final=st.mean(x[1] for x in bo),
                                      paired=paired([a[0] - b[0] for a, b in zip(mine, bo)], [a[1] - b[1] for a, b in zip(mine, bo)], rng), new_sixteen_design=st.mean(x[0] for x in mine[8:]), new_sixteen_bo=st.mean(x[0] for x in bo[8:]))
    (out/'contrast_search.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    if 'precision_24' in result:
        p = result['precision_24']
        c = p['paired']
        print(f"== 24 streams: design {p['design_auc']:.2f} final {p['design_final']:.2f} | pure BO {p['bo_auc']:.2f} final {p['bo_final']:.2f} | gap {c['mean_auc']:+.2f} [{c['interval90'][0]:+.2f},{c['interval90'][1]:+.2f}] {c['up']}/{c['down']}/{c['tied']} p={c['sign_flip_p']:.3f} | new 16: {p['new_sixteen_design']:.2f} vs {p['new_sixteen_bo']:.2f}")
    for v, r in round2.items():
        c = r['vs_first']
        print(f"R2 {v} AUC {r['auc']:.2f} final {r['final']:.2f} eligible {r['eligible']} passes {r['verdict']['passes']} | vs same-first {c['mean_auc']:+.2f} [{c['interval90'][0]:+.2f},{c['interval90'][1]:+.2f}] {c['up']}/{c['down']} | {r['by_task']}")
    print('R2 selected', pick)
    for name in ('round2_primary', 'round2_llm_first'):
        if name in result:
            r = result[name]
            print(f"== {name} held-out: AUC {r['auc']:.2f} SD {r['sd']:.2f} final {r['final']:.2f} passes {r['verdict']['passes']} {r['verdict']['beats']} final_ok {r['verdict']['final_ok']}")
            for key in ('vs_first', 'vs_bo', 'vs_h1', 'vs_lgbo'):
                c = r[key]
                print(f"   {key:8s} {c['mean_auc']:+.2f} [{c['interval90'][0]:+.2f},{c['interval90'][1]:+.2f}] {c['up']}/{c['down']}/{c['tied']} p={c['sign_flip_p']:.3f} final {c['mean_final']:+.2f}")
            print('   detail', r['by_stream'])
    for v, r in result['variants'].items():
        c = r['vs_first']
        print(f"{v:6s} AUC {r['auc']:.2f} final {r['final']:.2f} local {r['local']:.1f} eligible {r['eligible']} | vs same-first {c['mean_auc']:+.2f} [{c['interval90'][0]:+.2f},{c['interval90'][1]:+.2f}] {c['up']}/{c['down']} | {r['by_task']}")
    print('first final', round(first_final, 2), '| selected', result['selected'])
    for name in ('primary', 'no_hedge', 'no_LLM'):
        for data, r in result.get(name, {}).items():
            print(f"== {name} {data}: AUC {r['auc']:.2f} SD {r['sd']:.2f} final {r['final']:.2f} local {r['local']:.1f} hedged {r['hedged']} | passes {r['verdict']['passes']} {r['verdict']['beats']} final_ok {r['verdict']['final_ok']}")
            for key in ('vs_first', 'vs_bo', 'vs_h1', 'vs_lgbo'):
                if key in r:
                    c = r[key]
                    print(f"   {key:8s} {c['mean_auc']:+.2f} [{c['interval90'][0]:+.2f},{c['interval90'][1]:+.2f}] {c['up']}/{c['down']}/{c['tied']} p={c['sign_flip_p']:.3f} final {c['mean_final']:+.2f}")
            print('   detail', r.get('by_task') or r.get('by_stream'))


if __name__ == '__main__':
    main()
