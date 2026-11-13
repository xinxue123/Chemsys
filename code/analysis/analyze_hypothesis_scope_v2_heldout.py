"""Apply the registered held-out rule to the Direct Arylation streams.

Standard library only: reads sealed run states and issues no model calls, queries or GP
fits. The rule comes from docs/hypothesis_scope_v2_registration_20261105.md: the revision
is supported if its mean SearchAUC exceeds h1 by at least 1.0 point of pool maximum and it
is higher in more streams than lower. The stream is the unit (n=8).

    python code/analysis/analyze_hypothesis_scope_v2_heldout.py <v2 run folder> <output folder>
"""
import csv
import json
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_hypothesis_scope_method import load_npz, sign_flip_p

ARMS = ('bo', 'first', 'n0', 'x0', 'h1')
STREAMS = range(8)
T90_DF7 = 1.894579
PRIMARY = ('x0', 'h1')
PAIRS = (PRIMARY, ('x0', 'n0'), ('x0', 'first'), ('x0', 'bo'), ('h1', 'first'), ('h1', 'bo'), ('n0', 'first'), ('first', 'bo'))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_rows(path, rows):
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def load(base):
    oracle = load_npz((base/'heldout/input/oracle.npz').read_bytes())['raw_values']
    top = max(oracle)
    metrics, traces = [], []
    for arm in ARMS:
        for stream in STREAMS:
            state = read(base/'heldout/runs'/arm/f's{stream}'/'aryl'/'state.json')
            assert state['status'] == 'COMPLETE' and len(state['history']) == 20
            values = [r['raw_value'] for r in state['history']]
            best = [max(values[:i + 1]) for i in range(20)]
            executed = next((r for r in state['decisions'] if r['decision'].get('reason') == 'executed_LLM_preferred_test'), None)
            metrics.append(dict(arm=arm, stream=stream, first_id=state['history'][0]['candidate_id'], first_value=values[0],
                                first_percentile=100*sum(v < values[0] for v in oracle)/len(oracle),
                                SearchAUC_pct=100*st.mean(best)/top, final_best_pct=100*best[-1]/top,
                                LLM_calls_after_first=sum(c['query'] > 1 for c in state.get('calls', [])),
                                executed_query=executed['query'] if executed else '', executed_value=executed['raw_value'] if executed else ''))
            traces += [dict(arm=arm, stream=stream, query=i + 1, candidate_id=r['candidate_id'], raw_value=r['raw_value'], cumulative_best=best[i])
                       for i, r in enumerate(state['history'])]
    return metrics, traces, top, len(oracle)


def contrast(rows, a, b):
    auc = {(r['arm'], r['stream']): r['SearchAUC_pct'] for r in rows}
    final = {(r['arm'], r['stream']): r['final_best_pct'] for r in rows}
    d = [auc[(a, s)] - auc[(b, s)] for s in STREAMS]
    half = T90_DF7*st.stdev(d)/math.sqrt(len(d))
    mean = st.mean(d)
    up, down = sum(v > 1e-9 for v in d), sum(v < -1e-9 for v in d)
    return dict(contrast=f'{a} - {b}', mean_auc=mean, interval90=[mean - half, mean + half], streams_up=up, streams_down=down,
                sign_flip_p=sign_flip_p(d), mean_final=st.mean(final[(a, s)] - final[(b, s)] for s in STREAMS),
                by_stream=[round(v, 3) for v in d], meets_rule=mean >= 1. and up > down)


def usage(base):
    total = dict(requests=0, input_tokens=0, output_tokens=0, total_tokens=0, models=[], schema_retries=0)
    for path in list((base/'requests/heldout_first').glob('*/*/validated.json')) + list((base/'requests/heldout_h1').glob('*/*/validated.json')):
        saved = read(path)
        total['requests'] += 1
        total['schema_retries'] += saved['attempt'] - 1
        for key in ('input_tokens', 'output_tokens', 'total_tokens'):
            total[key] += saved['usage'].get(key, 0)
        if saved['resolved_model'] not in total['models']:
            total['models'].append(saved['resolved_model'])
    return total


def main():
    base, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    rows, traces, top, size = load(base)
    write_rows(out/'per_stream_results.csv', rows)
    write_rows(out/'trajectories.csv', traces)
    arms = {a: dict(mean_auc=st.mean(r['SearchAUC_pct'] for r in rows if r['arm'] == a), sd_auc=st.stdev(r['SearchAUC_pct'] for r in rows if r['arm'] == a),
                    mean_final=st.mean(r['final_best_pct'] for r in rows if r['arm'] == a),
                    mean_calls_after_first=st.mean(r['LLM_calls_after_first'] for r in rows if r['arm'] == a)) for a in ARMS}
    contrasts = [contrast(rows, a, b) for a, b in PAIRS]
    firsts = [r for r in rows if r['arm'] == 'first']
    executed = [r for r in rows if r['arm'] == 'x0' and r['executed_query'] != '']
    result = dict(pool_size=size, pool_maximum=top, arms=arms, contrasts=contrasts, revision_supported=contrasts[0]['meets_rule'],
                  LLM_first=dict(distinct_points=len({r['first_id'] for r in firsts}), mean_percentile=st.mean(r['first_percentile'] for r in firsts),
                                 values=[r['first_value'] for r in firsts]),
                  BO_first=dict(mean_percentile=st.mean(r['first_percentile'] for r in rows if r['arm'] == 'bo')),
                  executed_streams=len(executed), executed_values=[r['executed_value'] for r in executed], api_usage=usage(base))
    (out/'heldout_rule.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    for name, row in arms.items():
        print(f"{name:6s} AUC {row['mean_auc']:6.2f} (SD {row['sd_auc']:5.2f})  final {row['mean_final']:6.2f}  calls {row['mean_calls_after_first']:.1f}")
    for c in contrasts:
        print(f"{c['contrast']:11s} AUC {c['mean_auc']:+7.2f} [{c['interval90'][0]:+.2f},{c['interval90'][1]:+.2f}] streams {c['streams_up']}up/{c['streams_down']}down "
              f"p={c['sign_flip_p']:.3f} final {c['mean_final']:+.2f} rule={c['meets_rule']}\n    {c['by_stream']}")
    print('revision supported:', result['revision_supported'])
    print('LLM first:', result['LLM_first'], '| BO first:', result['BO_first'])
    print('executed streams:', result['executed_streams'], result['executed_values'])
    print('usage:', result['api_usage'])


if __name__ == '__main__':
    main()
