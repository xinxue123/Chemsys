"""Apply the registered CPA reading: controller minus plain BO from 24 shared starts.

Reads saved run states and the vendored source table only. No model calls, queries or GP fits.
Rules are in docs/hypothesis_scope_v2_registration_20261105.md, under "CPA held-out test".

    python code/analysis/analyze_hypothesis_scope_v2_cpa.py <v2 run folder> <output folder>
"""
import csv
import io
import json
import math
import statistics as st
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_hypothesis_scope_v2_first_point_streams import read, sign_flip

AXES = ('Catalyst', 'Imine', 'Thiol')
T90 = {19: 1.729, 20: 1.725, 21: 1.721, 22: 1.717, 23: 1.714}


def table(base):
    """Raw response by candidate ID, from the source table through the saved pool's ID order."""
    pool = read(base/'input/pool.json')
    archive = base.parents[3]/'code/chemist_system/third_party/reasoning_bo_official/data/chemistry_data.zip'
    with zipfile.ZipFile(archive) as handle:
        rows = list(csv.DictReader(io.StringIO(handle.read('CPA.csv').decode('utf-8-sig'))))
    value = {tuple(r[a] for a in AXES): float(r['yield']) for r in rows}
    return pool, {i: value[tuple(p[a] for a in AXES)] for i, p in zip(pool['ids'], pool['points'])}


def score(state, top):
    values = [100*r['raw_value']/top for r in state['history']]
    assert state['status'] == 'COMPLETE' and len(values) == 20
    best = [max(values[:i + 1]) for i in range(20)]
    return st.mean(best), best[-1]


def in_scope(point, scope):
    for axis, rule in scope.items():
        if set(rule) - {'allowed', 'excluded'}:
            return None
        if 'allowed' in rule and point[axis] not in rule['allowed']:
            return False
        if point[axis] in rule.get('excluded', ()):
            return False
    return True


def main():
    base, out = Path(sys.argv[1])/'cpa', Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    pool, raw = table(base)
    top = max(raw.values())
    capped = {i for i, v in raw.items() if v == top}
    ordered = sorted(raw.values())
    median = st.median(ordered)
    percentile = lambda v: 100*sum(x < v for x in ordered)/len(ordered)
    points = dict(zip(pool['ids'], pool['points']))
    rows, d, f, trajectories = [], [], [], []
    for t in range(24):
        bo_path = base/'runs/r1'/f's{t}'/'cpa/state.json'
        if not bo_path.exists():
            continue
        controller = read(bo_path)
        if controller['status'] != 'COMPLETE':
            continue
        start = read(base/'runs/bo'/f's{(t - 1) % 24}'/'cpa/state.json')['history'][0]['candidate_id']
        plain = read(base/'plain'/start/f's{t}'/'state.json')
        assert controller['history'][0]['candidate_id'] == plain['history'][0]['candidate_id'] == start and controller['seed'] == plain['seed'] == t
        draw = read(base/'draws'/f's{t}.json')
        notes = read(base/'runs/h0'/f's{t}'/'cpa/rounds/01_commit.json')['working_knowledge']['notes']
        favour = [n for n in notes if n['direction'] == 'favour']
        holds = [in_scope(points[i], n['scope']) for n in favour for i in capped]
        for arm, state in (('r1', controller), ('bo', plain)):
            trajectories.extend(dict(pair=t, arm=arm, query=q, candidate_id=r['candidate_id'], raw_value=r['raw_value'], pct_of_max=100*r['raw_value']/top)
                                for q, r in enumerate(state['history'], 1))
        c, p = score(controller, top), score(plain, top)
        d.append(c[0] - p[0])
        f.append(c[1] - p[1])
        rows.append(dict(pair=t, start=start, start_value=raw[start], start_above_median=raw[start] > median, plain_auc=p[0], plain_final=p[1],
                         controller_auc=c[0], controller_final=c[1], difference=c[0] - p[0], calls=len(controller['calls']),
                         llm_first=draw['candidate_id'], llm_first_value=raw[draw['candidate_id']], llm_first_percentile=percentile(raw[draw['candidate_id']]),
                         llm_first_capped=draw['candidate_id'] in capped, favour_scopes=len(favour),
                         favour_scopes_holding_capped=sum(any(in_scope(points[i], n['scope']) for i in capped) for n in favour),
                         scope_rule_unreadable=None in holds))
    n = len(d)
    mean = st.mean(d)
    half = T90[n - 1]*st.stdev(d)/math.sqrt(n)
    low, high = mean - half, mean + half
    up, down = sum(v > 1e-9 for v in d), sum(v < -1e-9 for v in d)
    reading = 'controller leads' if low > 1 else 'plain BO leads' if high < -1 else 'not separated'
    if reading == 'not separated':
        reading += ', no loss larger than 3 points' if low > -3 else ', unresolved'
    strata = {}
    for name, group in (('start_above_median', [r for r in rows if r['start_above_median']]), ('start_at_or_below_median', [r for r in rows if not r['start_above_median']])):
        strata[name] = dict(pairs=len(group), plain=st.mean(r['plain_auc'] for r in group) if group else None,
                            controller=st.mean(r['controller_auc'] for r in group) if group else None)
    usage = dict(requests=0, total_tokens=0, schema_retries=0, models=[])
    for path in (base.parent/'requests/cpa').glob('**/validated.json'):
        saved = read(path)
        usage['requests'] += 1
        usage['total_tokens'] += saved['usage'].get('total_tokens', 0)
        usage['schema_retries'] += saved['attempt'] - 1
        if saved['resolved_model'] not in usage['models']:
            usage['models'].append(saved['resolved_model'])
    result = dict(pairs=n, enough_pairs=n >= 20, pool_maximum_raw=top, capped_rows=len(capped),
                  arms=dict(plain_bo=dict(mean_auc=st.mean(r['plain_auc'] for r in rows), mean_final=st.mean(r['plain_final'] for r in rows)),
                            controller=dict(mean_auc=st.mean(r['controller_auc'] for r in rows), mean_final=st.mean(r['controller_final'] for r in rows),
                                            mean_calls=st.mean(r['calls'] for r in rows))),
                  confirmatory=dict(contrast='controller - plain BO', mean_auc=mean, interval90=[low, high], up=up, down=down, reading=reading),
                  secondary=dict(sign_flip_p=sign_flip(d), mean_final=st.mean(f), paper_rule_met=mean >= 1 and up > down and st.mean(f) >= -.5, strata=strata,
                                 recall=dict(llm_first_mean_percentile=st.mean(r['llm_first_percentile'] for r in rows),
                                             llm_first_capped=sum(r['llm_first_capped'] for r in rows), distinct_llm_first=len({r['llm_first'] for r in rows}),
                                             favour_scopes=sum(r['favour_scopes'] for r in rows),
                                             favour_scopes_holding_capped=sum(r['favour_scopes_holding_capped'] for r in rows),
                                             unreadable_scopes=sum(r['scope_rule_unreadable'] for r in rows))),
                  api_usage=usage)
    (out/'cpa.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    with open(out/'pairs.csv', 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()} for r in rows)
    with open(out/'trajectories.csv', 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trajectories[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows({k: round(v, 6) if isinstance(v, float) else v for k, v in r.items()} for r in trajectories)
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
