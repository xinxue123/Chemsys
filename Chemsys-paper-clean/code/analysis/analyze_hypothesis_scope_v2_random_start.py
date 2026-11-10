"""Apply the registered exploratory rule to scopes from pure BO's random first points.

Standard library only: reads sealed run states and issues no model calls, queries or GP
fits. The rule comes from docs/hypothesis_scope_v2_registration_20261105.md: r1 beats pure
BO by at least 1.0 point of mean SearchAUC, more streams up than down, and mean final best
not lower by more than 0.5 point. The stream is the unit (n=8).

    python code/analysis/analyze_hypothesis_scope_v2_random_start.py <v2 run folder> <output folder>
"""
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_hypothesis_scope_v2_heldout as held
from analyze_hypothesis_scope_method import load_npz, sign_flip_p

held.ARMS = ('bo', 'first', 'h1', 'r1')
PAIRS = (('r1', 'bo'), ('r1', 'h1'), ('r1', 'first'), ('h1', 'bo'))
SENSITIVITY_DROPPED = (0, 3)
QUERIES = (1, 2, 3, 4, 5, 7, 10, 15, 20)


def usage(base):
    total = dict(requests=0, input_tokens=0, output_tokens=0, total_tokens=0, models=[], schema_retries=0)
    for path in (base/'requests/r1').glob('*/*/validated.json'):
        saved = held.read(path)
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
    rows, traces, top, size = held.load(base)
    held.write_rows(out/'per_stream_results.csv', rows)
    held.write_rows(out/'trajectories.csv', traces)
    arms = {a: dict(mean_auc=st.mean(r['SearchAUC_pct'] for r in rows if r['arm'] == a), sd_auc=st.stdev(r['SearchAUC_pct'] for r in rows if r['arm'] == a),
                    mean_final=st.mean(r['final_best_pct'] for r in rows if r['arm'] == a),
                    mean_calls_after_first=st.mean(r['LLM_calls_after_first'] for r in rows if r['arm'] == a)) for a in held.ARMS}
    contrasts = [held.contrast(rows, a, b) for a, b in PAIRS]
    for c in contrasts:
        c['meets_rule'] = c['meets_rule'] and c['mean_final'] >= -.5
    kept = [v for s, v in enumerate(contrasts[0]['by_stream']) if s not in SENSITIVITY_DROPPED]
    sensitivity = dict(dropped_streams=list(SENSITIVITY_DROPPED), mean_auc=st.mean(kept), streams_up=sum(v > 1e-9 for v in kept),
                       streams_down=sum(v < -1e-9 for v in kept), sign_flip_p=sign_flip_p(kept))

    best = {}
    for t in traces:
        best.setdefault((t['arm'], t['stream']), []).append(t['cumulative_best'])
    running = {a: {q: st.mean(best[a, s][q - 1] for s in held.STREAMS) for q in QUERIES} for a in held.ARMS}

    oracle = load_npz((base/'heldout/input/oracle.npz').read_bytes())['raw_values']
    streams = []
    reasons = Counter()
    for s in held.STREAMS:
        r1 = held.read(base/'heldout/runs/r1'/f's{s}'/'aryl/state.json')
        bo = held.read(base/'heldout/runs/bo'/f's{s}'/'aryl/state.json')
        same = [a['candidate_id'] == b['candidate_id'] for a, b in zip(r1['history'], bo['history'])]
        decisions = [r['decision'] for r in r1['decisions'][1:]]
        reasons.update(d['reason'] for d in decisions)
        streams.append(dict(stream=s, first_value=r1['history'][0]['raw_value'], first_divergence=next((i + 1 for i, v in enumerate(same) if not v), None),
                            call_queries=[c['query'] for c in r1['calls']], scoped_queries=sum(d['reason'] == 'hypothesis_constrained_native_qLogEI' for d in decisions),
                            changed_from_BO=sum(d.get('changed_from_unrestricted_BO', False) for d in decisions), reached_optimum_query=next((i + 1 for i, r in enumerate(r1['history']) if r['raw_value'] >= max(oracle)), None)))
    result = dict(pool_size=size, pool_maximum=top, arms=arms, contrasts=contrasts, primary_rule_met=contrasts[0]['meets_rule'], sensitivity=sensitivity,
                  running_best=running, streams=streams, r1_decision_reasons=dict(reasons), api_usage=usage(base))
    (out/'random_start_rule.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')

    for name, row in arms.items():
        print(f"{name:6s} AUC {row['mean_auc']:6.2f} (SD {row['sd_auc']:5.2f})  final {row['mean_final']:6.2f}  calls {row['mean_calls_after_first']:.2f}")
    for c in contrasts:
        print(f"{c['contrast']:11s} AUC {c['mean_auc']:+7.2f} [{c['interval90'][0]:+.2f},{c['interval90'][1]:+.2f}] streams {c['streams_up']}up/{c['streams_down']}down "
              f"p={c['sign_flip_p']:.3f} final {c['mean_final']:+.2f} rule={c['meets_rule']}\n    {c['by_stream']}")
    print('primary rule met:', result['primary_rule_met'])
    print('sensitivity:', sensitivity)
    print('arm   ', *[f'q{q:<5d}' for q in QUERIES])
    for a in held.ARMS:
        print(f'{a:6s}', *[f'{running[a][q]:6.1f}' for q in QUERIES])
    for row in streams:
        print(row)
    print('reasons:', dict(reasons))
    print('usage:', result['api_usage'])


if __name__ == '__main__':
    main()
