"""Apply the registered readings to the LLM first points on all 24 pure BO streams.

Reads saved run states only: no model calls, queries or GP fits. Readings come from
docs/hypothesis_scope_v2_registration_20261105.md.

    python code/analysis/analyze_hypothesis_scope_v2_first_point_streams.py <v2 run folder> <output folder>
"""
import csv
import itertools
import json
import math
import random
import statistics as st
import sys
from pathlib import Path

WEIGHTS = dict(a00210=5, a00102=2, a00499=1)
POINTS = ('a00210', 'a00102', 'a00499', 'a00493')
T90 = {6: 1.943, 9: 1.833, 13: 1.771, 15: 1.753, 23: 1.714}
STREAMS = range(24)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def score(state):
    values = [r['raw_value'] for r in state['history']]
    assert state['status'] == 'COMPLETE' and len(values) == 20
    best = [max(values[:i + 1]) for i in range(20)]
    return st.mean(best), best[-1]


def sign_flip(d):
    observed = abs(sum(d))
    if len(d) <= 16:
        flips = itertools.product((1, -1), repeat=len(d))
        totals = [abs(sum(s*v for s, v in zip(f, d))) for f in flips]
        return sum(t >= observed - 1e-12 for t in totals)/len(totals)
    rng = random.Random(20261107)
    hits = sum(abs(sum(v if rng.random() < .5 else -v for v in d)) >= observed - 1e-12 for _ in range(200000))
    return (hits + 1)/200001


def contrast(bo, llm, streams, name):
    d = [bo[s][0] - llm[s][0] for s in streams]
    f = [bo[s][1] - llm[s][1] for s in streams]
    mean = st.mean(d)
    half = T90[len(d) - 1]*st.stdev(d)/math.sqrt(len(d))
    low, high = mean - half, mean + half
    return dict(contrast=name, streams=len(d), bo=st.mean(bo[s][0] for s in streams), llm=st.mean(llm[s][0] for s in streams), mean_auc=mean,
                interval90=[low, high], up=sum(v > 1e-9 for v in d), down=sum(v < -1e-9 for v in d), sign_flip_p=sign_flip(d), mean_final=st.mean(f),
                reading='random start leads' if low > 1. else 'LLM start leads' if high < -1. else 'not separated')


def main():
    base, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    new, old = base/'first_point_streams/aryl', base/'gp_followup/first_point/aryl'
    states = {}
    for point in POINTS:
        for s in STREAMS:
            path = new/point/f's{s}'/'state.json'
            states[point, s] = read(path if path.exists() else old/point/f's{s}'/'state.json')
            assert states[point, s]['history'][0]['candidate_id'] == point and states[point, s]['seed'] == s
    bo_states = [read((base/'heldout/runs/bo'/f's{s}'/'aryl' if s < 8 else base/'gp_followup/bo_extra/aryl'/f's{s}')/'state.json') for s in STREAMS]
    matched = 0
    for s in range(8):
        named = read(base/'heldout_named/runs/first'/f's{s}'/'aryl/state.json')
        point = named['history'][0]['candidate_id']
        if point in POINTS:
            assert [r['candidate_id'] for r in named['history']] == [r['candidate_id'] for r in states[point, s]['history']], f'{point} s{s}'
            matched += 1
    scores = {k: score(v) for k, v in states.items()}
    bo = [score(x) for x in bo_states]
    first = [x['history'][0]['raw_value'] for x in bo_states]
    total = sum(WEIGHTS.values())
    weighted = [tuple(sum(w*scores[p, s][i] for p, w in WEIGHTS.items())/total for i in (0, 1)) for s in STREAMS]
    equal = [tuple(st.mean(scores[p, s][i] for p in POINTS) for i in (0, 1)) for s in STREAMS]
    poor = [s for s in STREAMS if first[s] < 5]
    zero = [s for s in STREAMS if first[s] == 0]
    rest = [s for s in STREAMS if first[s] >= 5]
    rows = [contrast(bo, weighted, list(STREAMS), 'primary 1: all streams'), contrast(bo, weighted, poor, 'primary 2: first yield below 5%'),
            contrast(bo, weighted, zero, 'exact-zero streams'), contrast(bo, weighted, rest, 'first yield at least 5%'),
            contrast(bo, weighted, list(range(8, 24)), 'streams 8-23'), contrast(bo, equal, list(STREAMS), 'equal weight, four points')]
    grid = [[scores[p, s][0] for s in STREAMS] for p in POINTS]
    grand = st.mean(v for r in grid for v in r)
    stream_means = [st.mean(r[s] for r in grid) for s in STREAMS]
    point_means = [st.mean(r) for r in grid]
    n_p, n_s = len(POINTS), len(STREAMS)
    ss_stream = n_p*sum((v - grand)**2 for v in stream_means)
    ss_point = n_s*sum((v - grand)**2 for v in point_means)
    ss_total = sum((v - grand)**2 for r in grid for v in r)
    result = dict(contrasts=rows, named_first_runs_matched=matched, point_means=dict(zip(POINTS, point_means)), point_finals={p: st.mean(scores[p, s][1] for s in STREAMS) for p in POINTS},
                  variance_shares=dict(stream=ss_stream/ss_total, point=ss_point/ss_total, remainder=1 - (ss_stream + ss_point)/ss_total),
                  bo_auc_vs_llm_stream_mean_correlation=st.correlation([x[0] for x in bo], stream_means),
                  stream_classes=dict(poor=poor, zero=zero, rest=rest))
    (out/'first_point_streams.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    with open(out/'runs.csv', 'w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(('stream', 'bo_first_yield', 'bo_auc', 'bo_final') + tuple(f'{p}_{m}' for p in POINTS for m in ('auc', 'final')) + ('weighted_auc', 'weighted_final'))
        for s in STREAMS:
            writer.writerow((s, first[s], round(bo[s][0], 3), bo[s][1]) + tuple(round(v, 3) for p in POINTS for v in scores[p, s]) + tuple(round(v, 3) for v in weighted[s]))
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
