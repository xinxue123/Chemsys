"""Apply the registered crossover reading. Reads saved run states only.

    python code/analysis/analyze_hypothesis_scope_v2_first_point_crossover.py <v2 run folder> <output folder>
"""
import csv
import json
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_hypothesis_scope_v2_first_point_streams import T90, WEIGHTS, read, score, sign_flip


def llm_score(base, stream):
    total = sum(WEIGHTS.values())
    parts = []
    for point, weight in WEIGHTS.items():
        path = base/'first_point_streams/aryl'/point/f's{stream}'/'state.json'
        path = path if path.exists() else base/'gp_followup/first_point/aryl'/point/f's{stream}'/'state.json'
        parts.append(weight*score(read(path))[0])
    return sum(parts)/total


def reading(mean):
    return 'travels with the points' if mean >= 68 else 'does not travel' if mean <= 58 else 'mixed'


def summary(rows, name):
    d = [r['crossed_auc'] - r['llm_same_streams'] for r in rows]
    mean = st.mean(d)
    half = T90[len(d) - 1]*st.stdev(d)/math.sqrt(len(d))
    crossed = st.mean(r['crossed_auc'] for r in rows)
    return dict(contrast=name, points=len(rows), own_stream_auc=st.mean(r['own_auc'] for r in rows), crossed_auc=crossed, llm_same_streams=st.mean(r['llm_same_streams'] for r in rows),
                crossed_minus_llm=mean, interval90=[mean - half, mean + half], up=sum(v > 1e-9 for v in d), down=sum(v < -1e-9 for v in d), sign_flip_p=sign_flip(d), reading=reading(crossed))


def main():
    base, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    llm = {s: llm_score(base, s) for s in range(24)}
    rows, by_target = [], {t: [] for t in range(24)}
    for s in range(24):
        own = read((base/'heldout/runs/bo'/f's{s}'/'aryl' if s < 8 else base/'gp_followup/bo_extra/aryl'/f's{s}')/'state.json')
        point, targets = own['history'][0]['candidate_id'], [(s + k) % 24 for k in range(1, 6)]
        crossed = [score(read(base/'first_point_streams/crossover'/point/f's{t}'/'state.json')) for t in targets]
        for t, c in zip(targets, crossed):
            by_target[t].append(c[0])
        rows.append(dict(own_stream=s, point=point, first_yield=own['history'][0]['raw_value'], own_auc=score(own)[0], crossed_auc=st.mean(c[0] for c in crossed),
                         crossed_final=st.mean(c[1] for c in crossed), llm_same_streams=st.mean(llm[t] for t in targets)))
    excess = [r['own_auc'] - r['crossed_auc'] for r in rows]
    half = T90[23]*st.stdev(excess)/math.sqrt(24)
    pooled = [r['own_auc'] for r in rows] + [v for t in range(24) for v in by_target[t]]
    result = dict(own_minus_crossed=dict(mean=st.mean(excess), interval90=[st.mean(excess) - half, st.mean(excess) + half], sign_flip_p=sign_flip(excess)),
                  random_start_bo=dict(pooled_runs=len(pooled), pooled_auc=st.mean(pooled), crossed_on_streams_0_7=st.mean(v for t in range(8) for v in by_target[t]),
                                       crossed_on_streams_8_23=st.mean(v for t in range(8, 24) for v in by_target[t])),
                  crossover_1=summary(rows, 'all 24 points'), crossover_2=summary([r for r in rows if r['first_yield'] < 5], 'first yield below 5%'),
                  descriptive=dict(exact_zero=summary([r for r in rows if r['first_yield'] == 0], 'exact zero'), at_least_5=summary([r for r in rows if r['first_yield'] >= 5], 'first yield at least 5%')),
                  correlation_first_yield_crossed_auc=st.correlation([r['first_yield'] for r in rows], [r['crossed_auc'] for r in rows]),
                  correlation_own_crossed_auc=st.correlation([r['own_auc'] for r in rows], [r['crossed_auc'] for r in rows]))
    (out/'crossover.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    with open(out/'crossover_points.csv', 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()} for r in rows)
    print(json.dumps(result, indent=1))
    for r in rows:
        print(r['own_stream'], r['point'], r['first_yield'], round(r['own_auc'], 1), round(r['crossed_auc'], 1), round(r['llm_same_streams'], 1))


if __name__ == '__main__':
    main()
