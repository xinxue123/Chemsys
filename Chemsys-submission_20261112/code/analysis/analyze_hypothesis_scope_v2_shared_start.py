"""Apply the registered readings to the shared-start comparison. Reads saved run states only.

Readings come from docs/hypothesis_scope_v2_registration_20261105.md.

    python code/analysis/analyze_hypothesis_scope_v2_shared_start.py <v2 run folder> <output folder>
"""
import csv
import json
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_hypothesis_scope_v2_first_point_streams import T90, read, score, sign_flip

ARMS = ('r1', 'n6m1', 'lgbo', 'rbo')
NAMES = dict(bo='plain BO', r1='controller', n6m1='neighbour search', lgbo='LGBO', rbo='Reasoning BO')
BETWEEN = (('r1', 'lgbo'), ('r1', 'rbo'), ('n6m1', 'lgbo'), ('n6m1', 'rbo'), ('n6m1', 'r1'))
T90.update({n: v for n, v in ((10, 1.812), (11, 1.796), (12, 1.782), (14, 1.761), (16, 1.746), (17, 1.740), (18, 1.734), (19, 1.729), (20, 1.725), (21, 1.721), (22, 1.717))})


def contrast(a, b, pairs, name):
    d = [a[t][0] - b[t][0] for t in pairs]
    f = [a[t][1] - b[t][1] for t in pairs]
    mean = st.mean(d)
    half = T90[len(d) - 1]*st.stdev(d)/math.sqrt(len(d))
    return dict(contrast=name, pairs=len(d), mean_auc=mean, interval90=[mean - half, mean + half], up=sum(v > 1e-9 for v in d), down=sum(v < -1e-9 for v in d),
                sign_flip_p=sign_flip(d), mean_final=st.mean(f))


def rule(c):
    return c['mean_auc'] >= 1. and c['up'] > c['down'] and c['mean_final'] >= -.5


def lead(c, a, b):
    low, high = c['interval90']
    return f'{a} leads' if low > 1. else f'{b} leads' if high < -1. else 'not separated'


def usage(base):
    out = {}
    for path in (base/'requests/shared_start').glob('**/validated.json'):
        saved = read(path)
        parts = path.relative_to(base/'requests/shared_start').parts
        name = parts[0] if parts[0] == 'r1' else parts[1].rsplit('_', 1)[-1]
        row = out.setdefault(name, dict(requests=0, total_tokens=0, schema_retries=0, models=[]))
        row['requests'] += 1
        row['total_tokens'] += saved['usage'].get('total_tokens', 0)
        row['schema_retries'] += saved['attempt'] - 1
        if saved['resolved_model'] not in row['models']:
            row['models'].append(saved['resolved_model'])
    return out


def main():
    base, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    runs = {}
    starts, rows = {}, []
    for t in range(24):
        s = (t - 1) % 24
        own = read((base/'heldout/runs/bo'/f's{s}'/'aryl' if s < 8 else base/'gp_followup/bo_extra/aryl'/f's{s}')/'state.json')
        point = own['history'][0]['candidate_id']
        starts[t] = (point, own['history'][0]['raw_value'])
        runs['bo', t] = read(base/'first_point_streams/crossover'/point/f's{t}'/'state.json')
        for arm in ARMS:
            path = base/'shared_start/runs'/arm/f's{t}'/'aryl/state.json'
            if path.exists():
                runs[arm, t] = read(path)
    pairs = [t for t in range(24) if all((a, t) in runs and runs[a, t]['status'] == 'COMPLETE' for a in ARMS)]
    for key, state in runs.items():
        if key[1] in pairs:
            assert state['history'][0]['candidate_id'] == starts[key[1]][0] and len(state['history']) == 20, key
            assert state['seed'] == key[1], key
    scores = {a: {t: score(runs[a, t]) for t in pairs} for a in ('bo',) + ARMS}
    poor = [t for t in pairs if starts[t][1] < 5]
    rest = [t for t in pairs if starts[t][1] >= 5]
    against = {}
    for arm in ARMS:
        c = contrast(scores[arm], scores['bo'], pairs, NAMES[arm] + ' - plain BO')
        against[arm] = dict(c, rule_met=rule(c))
    between = {}
    for a, b in BETWEEN:
        c = contrast(scores[a], scores[b], pairs, f'{NAMES[a]} - {NAMES[b]}')
        between[f'{a}_{b}'] = dict(c, reading=lead(c, NAMES[a], NAMES[b]))
    strata = {}
    for name, group in (('first_yield_below_5', poor), ('first_yield_at_least_5', rest)):
        strata[name] = dict(pairs=len(group), means={a: st.mean(scores[a][t][0] for t in group) for a in scores},
                            against_plain_bo={a: contrast(scores[a], scores['bo'], group, NAMES[a] + ' - plain BO') for a in ARMS})
    by_set = {}
    for a, b in BETWEEN + tuple((arm, 'bo') for arm in ARMS):
        d = [st.mean(scores[a][t][0] - scores[b][t][0] for t in pairs if t % 8 == h) for h in range(8)]
        half = 1.895*st.stdev(d)/math.sqrt(8)
        by_set[f'{a}_{b}'] = dict(mean=st.mean(d), interval90=[st.mean(d) - half, st.mean(d) + half], up=sum(v > 1e-9 for v in d), down=sum(v < -1e-9 for v in d), sign_flip_p=sign_flip(d))
    result = dict(by_hypothesis_set=by_set, pairs=len(pairs), arms={a: dict(mean_auc=st.mean(v[0] for v in scores[a].values()), sd_auc=st.stdev(v[0] for v in scores[a].values()),
                                                  mean_final=st.mean(v[1] for v in scores[a].values())) for a in scores},
                  against_plain_bo=against, between_methods=between, strata=strata, api_usage=usage(base))
    (out/'shared_start.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    with open(out/'pairs.csv', 'w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(('pair', 'first_point', 'first_yield') + tuple(f'{a}_{m}' for a in scores for m in ('auc', 'final')))
        for t in pairs:
            writer.writerow((t, *starts[t]) + tuple(round(v, 3) for a in scores for v in scores[a][t]))
    with open(out/'trajectories.csv', 'w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(('arm', 'pair', 'query', 'candidate_id', 'raw_value'))
        for (arm, t), state in sorted(runs.items()):
            if t in pairs:
                for q, r in enumerate(state['history'], 1):
                    writer.writerow((arm, t, q, r['candidate_id'], r['raw_value']))
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
