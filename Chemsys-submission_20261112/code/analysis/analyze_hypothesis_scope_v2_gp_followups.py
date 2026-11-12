"""Apply the registered rules to the GP-only follow-ups, Tests F and S.

Standard library only: reads sealed run states and issues no model calls, queries or GP
fits. Rules come from docs/hypothesis_scope_v2_registration_20261105.md. Effects are % of
task pool maximum. Random streams are seeded, so results repeat exactly.

    python code/analysis/analyze_hypothesis_scope_v2_gp_followups.py <v2 run folder> <output folder>
"""
import csv
import json
import math
import random
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEED = 20261106
PERMUTATIONS = 100000
RESAMPLES = 20000
T90 = {7: 1.894579, 23: 1.713872}
DEV = ('c1', 'c2', 'c3', 'c4', 'perera', 'doyle', 'snar', 'hplc')
EARLY = range(1, 10)  # Queries 2-10.


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows_of(path):
    with open(path, encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def tops():
    out = {t: max(float(r['objective_raw']) for r in rows_of(ROOT/'data/pools'/t/'candidates.csv')) for t in DEV}
    out['aryl'] = 100.
    return out


def metrics(state, top):
    values = [r['raw_value'] for r in state['history']]
    assert state['status'] == 'COMPLETE' and len(values) == 20
    best = [max(values[:i + 1]) for i in range(20)]
    return 100*st.mean(best)/top, 100*best[-1]/top, values


def permutation_p(a, b, rng):
    observed = abs(st.mean(a) - st.mean(b))
    pooled, n = a + b, len(a)
    hits = 0
    for _ in range(PERMUTATIONS):
        rng.shuffle(pooled)
        hits += abs(st.mean(pooled[:n]) - st.mean(pooled[n:])) >= observed - 1e-12
    return (hits + 1)/(PERMUTATIONS + 1)


def bootstrap(a, b, rng):
    gaps = sorted(st.mean(rng.choices(a, k=len(a))) - st.mean(rng.choices(b, k=len(b))) for _ in range(RESAMPLES))
    return [gaps[int(.05*RESAMPLES)], gaps[int(.95*RESAMPLES)]]


def sign_flip(values, rng):
    observed = abs(sum(values))
    if len(values) <= 12:
        hits = sum(abs(sum(v if m >> i & 1 else -v for i, v in enumerate(values))) >= observed - 1e-12 for m in range(2**len(values)))
        return hits/2**len(values)
    hits = sum(abs(sum(v if rng.random() < .5 else -v for v in values)) >= observed - 1e-12 for _ in range(PERMUTATIONS))
    return (hits + 1)/(PERMUTATIONS + 1)


def paired(auc, final, rng):
    mean = st.mean(auc)
    half = T90[len(auc) - 1]*st.stdev(auc)/math.sqrt(len(auc))
    up, down = sum(v > 1e-9 for v in auc), sum(v < -1e-9 for v in auc)
    final_mean = st.mean(final)
    return dict(units=len(auc), mean_auc=mean, interval90=[mean - half, mean + half], up=up, down=down, tied=len(auc) - up - down,
                sign_flip_p=sign_flip(auc, rng), mean_final=final_mean, meets_rule=mean >= 1. and up > down and final_mean >= -.5)


def test_f(base, top, rng):
    g = base/'gp_followup'
    plan = read(g/'design.json')
    rows, out = [], {}
    for task, spec in plan.items():
        short, axis, best = spec['short'], spec['axis'], spec['best_level']
        groups = {}
        for group in ('inside', 'outside'):
            points = []
            for point in spec[group]:
                aucs, finals, shares, second = [], [], [], []
                for stream in spec['streams']:
                    state = read(g/'first_point'/short/point/f's{stream}'/'state.json')
                    auc, final, _ = metrics(state, top[short])
                    picks = [r['decision']['conditions'][axis] == best for r in state['decisions']]
                    aucs.append(auc)
                    finals.append(final)
                    shares.append(st.mean(picks[i] for i in EARLY))
                    second.append(picks[1])
                    rows.append(dict(task=short, group=group, first_id=point, stream=stream, first_value=state['history'][0]['raw_value'],
                                     SearchAUC_pct=auc, final_best_pct=final, best_level_share_q2_10=st.mean(picks[i] for i in EARLY)))
                points.append((st.mean(aucs), st.mean(finals), st.mean(shares), st.mean(second)))
            groups[group] = points
        inside, outside = ([p[0] for p in groups[k]] for k in ('inside', 'outside'))
        finals_in, finals_out = ([p[1] for p in groups[k]] for k in ('inside', 'outside'))
        gap = st.mean(outside) - st.mean(inside)
        out[short] = dict(axis=axis, best_level=best, points_per_group=len(inside), mean_auc_inside=st.mean(inside), mean_auc_outside=st.mean(outside),
                          gap_outside_minus_inside=gap, interval90_bootstrap=bootstrap(outside, inside, rng), permutation_p=permutation_p(list(outside), list(inside), rng),
                          final_inside=st.mean(finals_in), final_outside=st.mean(finals_out),
                          share_inside=100*st.mean(p[2] for p in groups['inside']), share_outside=100*st.mean(p[2] for p in groups['outside']),
                          q2_share_inside=100*st.mean(p[3] for p in groups['inside']), q2_share_outside=100*st.mean(p[3] for p in groups['outside']))
    aryl = plan['Direct_Arylation']
    out['aryl']['meets_rule'] = out['aryl']['gap_outside_minus_inside'] >= 5. and out['aryl']['permutation_p'] < .05
    llm = {}
    for point in aryl['llm_points']:
        states = [read(g/'first_point/aryl'/point/f's{s}'/'state.json') for s in aryl['llm_streams']]
        scored = [metrics(s, 100.) for s in states]
        picks = [[r['decision']['conditions'][aryl['axis']] == aryl['best_level'] for r in s['decisions']] for s in states]
        llm[point] = dict(first_value=states[0]['history'][0]['raw_value'], streams=len(states), mean_auc=st.mean(x[0] for x in scored), sd_auc=st.stdev(x[0] for x in scored),
                          mean_final=st.mean(x[1] for x in scored), best_solvent_share_q2_10=100*st.mean(st.mean(p[i] for i in EARLY) for p in picks))
    out['llm_points'] = llm
    # Same-first BO re-estimated over eight streams per point, weighted by how often each point was drawn.
    draws = [read(base/'heldout/runs/first'/f's{s}'/'aryl/state.json')['history'][0]['candidate_id'] for s in range(8)]
    out['same_first_reestimate'] = dict(draws={p: draws.count(p) for p in llm}, weighted_auc=sum(draws.count(p)*r['mean_auc'] for p, r in llm.items())/8,
                                        weighted_final=sum(draws.count(p)*r['mean_final'] for p, r in llm.items())/8,
                                        unweighted_auc=st.mean(r['mean_auc'] for r in llm.values()))
    # How far the second query moves from the first, over every Test F run.
    moves = {}
    for task, spec in plan.items():
        short = spec['short']
        shared = []
        for state_path in (g/'first_point'/short).glob('*/s*/state.json'):
            d = read(state_path)['decisions']
            one, two = d[0]['decision']['conditions'], d[1]['decision']['conditions']
            shared.append(sum(one[a] == two[a] for a in one))
        moves[short] = dict(runs=len(shared), axes=len(one), mean_shared_levels=st.mean(shared), runs_sharing_no_level=sum(v == 0 for v in shared))
    out['second_query_moves'] = moves
    return rows, out


def test_s(base, top, rng):
    g = base/'gp_followup'
    rows = []
    bo_dev = {(r['task'], int(r['seed'])): r for r in rows_of(ROOT/'results/hypothesis_scope_v2_20261105/per_seed_results.csv') if r['arm'] == 'bo_env'}
    task_auc, task_final, changed, identical = [], [], 0, 0
    for task in DEV:
        auc, final = [], []
        for seed in range(4):
            state = read(g/'r0'/task/f's{seed}'/'state.json')
            a, f, _ = metrics(state, top[task])
            b = bo_dev[task, seed]
            ba, bf = 100*float(b['SearchAUC'])/top[task], 100*float(b['final_best'])/top[task]
            count = sum(r['decision'].get('changed_from_unrestricted_BO', False) for r in state['decisions'][1:])
            changed += count
            identical += abs(a - ba) < 1e-9
            auc.append(a - ba)
            final.append(f - bf)
            rows.append(dict(data='development', task=task, stream=seed, r0_auc=a, bo_auc=ba, r0_final=f, bo_final=bf, changed_choices=count))
        task_auc.append(st.mean(auc))
        task_final.append(st.mean(final))
    dev = paired(task_auc, task_final, rng)
    dev.update(by_task=dict(zip(DEV, [round(v, 3) for v in task_auc])), changed_choices=changed, identical_runs=identical, runs=32,
               r0_mean=st.mean(r['r0_auc'] for r in rows), bo_mean=st.mean(r['bo_auc'] for r in rows))
    auc, final, held, changed, identical = [], [], [], 0, 0
    for seed in range(24):
        bo_folder = base/'heldout/runs/bo'/f's{seed}'/'aryl' if seed < 8 else g/'bo_extra/aryl'/f's{seed}'
        r0, bo = read(g/'r0/aryl'/f's{seed}'/'state.json'), read(bo_folder/'state.json')
        a, f, values = metrics(r0, 100.)
        ba, bf, bo_values = metrics(bo, 100.)
        assert values[0] == bo_values[0]
        count = sum(r['decision'].get('changed_from_unrestricted_BO', False) for r in r0['decisions'][1:])
        changed += count
        identical += abs(a - ba) < 1e-9
        auc.append(a - ba)
        final.append(f - bf)
        held.append(dict(data='held-out', task='aryl', stream=seed, r0_auc=a, bo_auc=ba, r0_final=f, bo_final=bf, changed_choices=count))
    rows += held
    out = paired(auc, final, rng)
    out.update(changed_choices=changed, identical_runs=identical, runs=24, r0_mean=st.mean(r['r0_auc'] for r in held), bo_mean=st.mean(r['bo_auc'] for r in held),
               r0_final=st.mean(r['r0_final'] for r in held), bo_final=st.mean(r['bo_final'] for r in held),
               first_eight=paired(auc[:8], final[:8], rng), new_sixteen=dict(mean_auc=st.mean(auc[8:]), bo_mean=st.mean(r['bo_auc'] for r in held[8:])))
    refresh = []
    for seed in range(8):
        r1, r0 = read(base/'heldout/runs/r1'/f's{seed}'/'aryl/state.json'), read(g/'r0/aryl'/f's{seed}'/'state.json')
        refresh.append(metrics(r1, 100.)[0] - metrics(r0, 100.)[0])
    out['refresh_r1_minus_r0'] = dict(mean_auc=st.mean(refresh), up=sum(v > 1e-9 for v in refresh), down=sum(v < -1e-9 for v in refresh), sign_flip_p=sign_flip(refresh, rng))
    return rows, dict(development=dev, held_out=out)


def test_g(base, top, rng):
    """Gate ablation: p0 keeps the gate, p0g removes it, and both skip later refreshes."""
    g = base/'gp_followup'
    first_dev = {(r['task'], int(r['seed'])): r for r in rows_of(ROOT/'results/hypothesis_scope_v2_20261105/per_seed_results.csv') if r['arm'] == 'first_env'}
    rows, out = [], {}
    gate_auc, gate_final, first_auc, first_final = [], [], [], []
    for task in DEV:
        ga, gf, fa, ff = [], [], [], []
        for seed in range(4):
            on, off = (read(g/arm/task/f's{seed}'/'state.json') for arm in ('p0', 'p0g'))
            assert on['history'][0]['candidate_id'] == off['history'][0]['candidate_id']
            a, f, _ = metrics(on, top[task])
            b, e, _ = metrics(off, top[task])
            c = first_dev[task, seed]
            ca, cf = 100*float(c['SearchAUC'])/top[task], 100*float(c['final_best'])/top[task]
            ga.append(a - b)
            gf.append(f - e)
            fa.append(a - ca)
            ff.append(f - cf)
            rows.append(dict(data='development', task=task, stream=seed, gate_auc=a, no_gate_auc=b, same_first_auc=ca, gate_final=f, no_gate_final=e))
        gate_auc.append(st.mean(ga))
        gate_final.append(st.mean(gf))
        first_auc.append(st.mean(fa))
        first_final.append(st.mean(ff))
    dev = [r for r in rows if r['data'] == 'development']
    out['development'] = dict(gate_vs_no_gate=dict(paired(gate_auc, gate_final, rng), by_task=dict(zip(DEV, [round(v, 3) for v in gate_auc]))),
                              gate_vs_same_first=paired(first_auc, first_final, rng), gate_mean=st.mean(r['gate_auc'] for r in dev), no_gate_mean=st.mean(r['no_gate_auc'] for r in dev))
    ga, gf, fa, ff, held = [], [], [], [], []
    for seed in range(8):
        on, off = (read(g/arm/'aryl'/f's{seed}'/'state.json') for arm in ('p0', 'p0g'))
        same = read(base/'heldout/runs/first'/f's{seed}'/'aryl/state.json')
        assert on['history'][0]['candidate_id'] == same['history'][0]['candidate_id']
        a, f, _ = metrics(on, 100.)
        b, e, _ = metrics(off, 100.)
        c, d, _ = metrics(same, 100.)
        ga.append(a - b)
        gf.append(f - e)
        fa.append(a - c)
        ff.append(f - d)
        held.append(dict(data='held-out', task='aryl', stream=seed, gate_auc=a, no_gate_auc=b, same_first_auc=c, gate_final=f, no_gate_final=e))
    rows += held
    out['held_out'] = dict(gate_vs_no_gate=paired(ga, gf, rng), gate_vs_same_first=paired(fa, ff, rng), gate_mean=st.mean(r['gate_auc'] for r in held), no_gate_mean=st.mean(r['no_gate_auc'] for r in held))
    return rows, out


def main():
    base, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    top = tops()
    first_rows, first = test_f(base, top, rng)
    scope_rows, scope = test_s(base, top, rng)
    gate_rows, gate = test_g(base, top, rng)
    write_rows(out/'first_point_runs.csv', first_rows)
    write_rows(out/'scope_runs.csv', scope_rows)
    write_rows(out/'gate_runs.csv', gate_rows)
    result = dict(test_F=first, test_S=scope, test_G=gate)
    (out/'gp_followups.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    for task in ('aryl', 'doyle', 'perera'):
        r = first[task]
        print(f"F {task:7s} inside {r['mean_auc_inside']:6.2f} outside {r['mean_auc_outside']:6.2f} gap {r['gap_outside_minus_inside']:+6.2f} "
              f"[{r['interval90_bootstrap'][0]:+.2f},{r['interval90_bootstrap'][1]:+.2f}] p={r['permutation_p']:.4f} final {r['final_inside']:.1f}/{r['final_outside']:.1f} "
              f"best-level share {r['share_inside']:.1f}/{r['share_outside']:.1f} q2 {r['q2_share_inside']:.1f}/{r['q2_share_outside']:.1f}")
    print('F rule met:', first['aryl']['meets_rule'])
    for point, r in first['llm_points'].items():
        print(f"  LLM {point} first {r['first_value']:.2f} AUC {r['mean_auc']:.2f} (SD {r['sd_auc']:.2f}) final {r['mean_final']:.2f} solvent share {r['best_solvent_share_q2_10']:.1f}")
    print('same-first re-estimate:', first['same_first_reestimate'])
    print('second query moves:', first['second_query_moves'])
    for name, r in scope.items():
        print(f"S {name:12s} r0 {r['r0_mean']:.2f} bo {r['bo_mean']:.2f} gain {r['mean_auc']:+.2f} [{r['interval90'][0]:+.2f},{r['interval90'][1]:+.2f}] up/down {r['up']}/{r['down']} tied {r['tied']} "
              f"p={r['sign_flip_p']:.4f} final {r['mean_final']:+.2f} rule={r['meets_rule']} changed {r['changed_choices']} identical {r['identical_runs']}/{r['runs']}")
    print('dev by task', scope['development']['by_task'])
    print('held first eight', {k: round(v, 3) if isinstance(v, float) else v for k, v in scope['held_out']['first_eight'].items() if k != 'interval90'}, '| new sixteen', scope['held_out']['new_sixteen'])
    print('refresh r1-r0', scope['held_out']['refresh_r1_minus_r0'])
    for name, r in gate.items():
        for label in ('gate_vs_no_gate', 'gate_vs_same_first'):
            c = r[label]
            print(f"G {name:12s} {label:18s} gain {c['mean_auc']:+.2f} [{c['interval90'][0]:+.2f},{c['interval90'][1]:+.2f}] up/down {c['up']}/{c['down']} tied {c['tied']} p={c['sign_flip_p']:.3f} final {c['mean_final']:+.2f} rule={c['meets_rule']}")
        print(f"  means: gate {r['gate_mean']:.2f} no gate {r['no_gate_mean']:.2f}")
    print('gate by task', gate['development']['gate_vs_no_gate']['by_task'])


if __name__ == '__main__':
    main()
