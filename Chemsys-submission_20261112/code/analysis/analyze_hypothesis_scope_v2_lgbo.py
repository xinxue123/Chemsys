"""Apply the registered reading to the held-out LGBO comparison.

Standard library only: reads sealed run states and issues no model calls, queries or GP
fits. The reading comes from docs/hypothesis_scope_v2_registration_20261105.md.

    python code/analysis/analyze_hypothesis_scope_v2_lgbo.py <v2 run folder> <output folder>
"""
import csv
import json
import math
import statistics as st
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_hypothesis_scope_method import sign_flip_p

STREAMS = range(8)
T90_DF7 = 1.894579
BEST_SOLVENT = 'CC(N(C)C)=O'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def score(state):
    values = [r['raw_value'] for r in state['history']]
    assert state['status'] == 'COMPLETE' and len(values) == 20
    best = [max(values[:i + 1]) for i in range(20)]
    return st.mean(best), best[-1], best


def contrast(a, b, name):
    d = [x[0] - y[0] for x, y in zip(a, b)]
    f = [x[1] - y[1] for x, y in zip(a, b)]
    mean = st.mean(d)
    half = T90_DF7*st.stdev(d)/math.sqrt(len(d))
    return dict(contrast=name, mean_auc=mean, interval90=[mean - half, mean + half], up=sum(v > 1e-9 for v in d), down=sum(v < -1e-9 for v in d),
                sign_flip_p=sign_flip_p(d), mean_final=st.mean(f), by_stream=[round(v, 3) for v in d])


def main():
    base, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    runs = {arm: [read(base/'heldout/runs'/arm/f's{s}'/'aryl/state.json') for s in STREAMS] for arm in ('lgbo', 'h1', 'first', 'bo')}
    scored = {arm: [score(s) for s in rows] for arm, rows in runs.items()}
    assert all(a['history'][0]['candidate_id'] == b['history'][0]['candidate_id'] for a, b in zip(runs['lgbo'], runs['h1']))
    primary = contrast(scored['h1'], scored['lgbo'], 'controller - LGBO')
    low, high = primary['interval90']
    reading = 'LGBO ahead' if high < -1. else 'controller ahead' if low > 1. else 'not separated'
    others = [contrast(scored['lgbo'], scored['first'], 'LGBO - same-first BO'), contrast(scored['lgbo'], scored['bo'], 'LGBO - pure BO')]
    modes, changed, shares = Counter(), 0, []
    for state in runs['lgbo']:
        changed += sum(r['decision'].get('changed_from_BO', False) for r in state['decisions'][1:])
        modes.update(p['mode'] for p in state['preferences'])
        shares.append(st.mean(r['decision']['conditions']['Solvent_SMILES'] == BEST_SOLVENT for r in state['decisions'][1:10]))
    usage = dict(requests=0, input_tokens=0, output_tokens=0, total_tokens=0, schema_retries=0, models=[])
    for path in (base/'requests/heldout_lgbo').glob('*/*/validated.json'):
        saved = read(path)
        usage['requests'] += 1
        usage['schema_retries'] += saved['attempt'] - 1
        for key in ('input_tokens', 'output_tokens', 'total_tokens'):
            usage[key] += saved['usage'].get(key, 0)
        if saved['resolved_model'] not in usage['models']:
            usage['models'].append(saved['resolved_model'])
    controller = dict(requests=0, total_tokens=0)
    for path in (base/'requests/heldout_h1').glob('*/*/validated.json'):
        if path.parent.name != 'query01':  # Post-first refreshes only.
            controller['requests'] += 1
            controller['total_tokens'] += read(path)['usage'].get('total_tokens', 0)
    with open(out/'trajectories.csv', 'w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(('arm', 'stream', 'query', 'candidate_id', 'raw_value', 'cumulative_best'))
        for s, state in enumerate(runs['lgbo']):
            for q, (row, best) in enumerate(zip(state['history'], scored['lgbo'][s][2]), 1):
                writer.writerow(('lgbo', s, q, row['candidate_id'], row['raw_value'], best))
    result = dict(controller_post_first_usage=controller, arms={a: dict(mean_auc=st.mean(x[0] for x in v), sd_auc=st.stdev(x[0] for x in v), mean_final=st.mean(x[1] for x in v)) for a, v in scored.items()},
                  primary=primary, registered_reading=reading, secondary=others, lgbo_modes=dict(modes), lgbo_changed_from_BO=changed, lgbo_decisions=8*19,
                  lgbo_best_solvent_share_q2_10=100*st.mean(shares), running_best={a: [round(st.mean(x[2][q] for x in v), 2) for q in range(20)] for a, v in scored.items()},
                  by_stream=[dict(stream=s, first=runs['lgbo'][s]['history'][0]['candidate_id'], lgbo=round(scored['lgbo'][s][0], 3), controller=round(scored['h1'][s][0], 3),
                                  same_first=round(scored['first'][s][0], 3), lgbo_final=scored['lgbo'][s][1], controller_final=scored['h1'][s][1]) for s in STREAMS], api_usage=usage)
    (out/'lgbo_heldout.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    for name, row in result['arms'].items():
        print(f"{name:6s} AUC {row['mean_auc']:6.2f} (SD {row['sd_auc']:5.2f}) final {row['mean_final']:6.2f}")
    for c in [primary] + others:
        print(f"{c['contrast']:22s} AUC {c['mean_auc']:+7.2f} [{c['interval90'][0]:+.2f},{c['interval90'][1]:+.2f}] up/down {c['up']}/{c['down']} p={c['sign_flip_p']:.3f} final {c['mean_final']:+.2f}\n    {c['by_stream']}")
    print('registered reading:', reading)
    print('modes', dict(modes), '| changed from BO', changed, 'of', 8*19, '| best-solvent share', round(result['lgbo_best_solvent_share_q2_10'], 1))
    print('usage', usage)
    print('controller post-first usage', controller)


if __name__ == '__main__':
    main()
