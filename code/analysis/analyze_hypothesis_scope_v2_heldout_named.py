"""Apply the registered readings to the named-substrate Direct Arylation rerun.

Standard library only: reads sealed run states and issues no model calls, queries or GP fits.
The readings come from docs/hypothesis_scope_v2_registration_20261105.md.

    python code/analysis/analyze_hypothesis_scope_v2_heldout_named.py <v2 run folder> <output folder>
"""
import csv
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_hypothesis_scope_method import sign_flip_p
from analyze_hypothesis_scope_v2_lgbo import STREAMS, contrast, read, score

LLM_ARMS = ('first', 'h1', 'lgbo', 'rbo', 'n6m1')
PUBLISHED = dict(Base_SMILES=('O=C([O-])C(C)(C)C.[Cs+]', 'O=C([O-])C.[Cs+]'), Ligand_SMILES=('C[C@]1(O2)O[C@](C[C@]2(C)P3C4=CC=CC=C4)(C)O[C@]3(C)C1',),
                 Solvent_SMILES=('CC(N(C)C)=O',), Concentration=('0.153',), Temp_C=('105',))
BO_24_STREAMS = 73.24


def states(root, arm):
    return [read(root/'runs'/arm/f's{s}'/'aryl/state.json') for s in STREAMS]


def rule(c):
    return c['mean_auc'] >= 1. and c['up'] > c['down'] and c['mean_final'] >= -.5


def recall(rows, points):
    """How the first picks sit against the published best conditions."""
    conditions = [points[r['history'][0]['candidate_id']] for r in rows]
    matches = {a: sum(c[a] in v for c in conditions) for a, v in PUBLISHED.items()}
    return dict(first_ids=[r['history'][0]['candidate_id'] for r in rows], first_yields=[r['history'][0]['raw_value'] for r in rows],
                distinct=len({r['history'][0]['candidate_id'] for r in rows}), mean_yield=st.mean(r['history'][0]['raw_value'] for r in rows),
                level_matches=matches, all_five_levels=sum(all(c[a] in v for a, v in PUBLISHED.items()) for c in conditions))


def usage(base):
    out = {}
    for path in base.glob('requests/named_*/**/validated.json'):
        saved = read(path)
        name = path.relative_to(base/'requests').parts[0]
        if name == 'named_lgbo':
            name = 'named_rbo' if path.parts[-3].endswith('_rbo') else name
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
    named, hidden = base/'heldout_named', base/'heldout'
    pool = read(hidden/'input/pool.json')
    points = dict(zip(pool['ids'], pool['points']))
    new = {a: states(named, a) for a in LLM_ARMS + ('bo',)}
    old = {a: states(hidden, a) for a in ('first', 'h1', 'lgbo', 'bo')}
    old['n6m1'] = [read(base/'gp_followup/n6m1/aryl'/f's{s}'/'state.json') for s in STREAMS]
    assert [[r['candidate_id'] for r in a['history']] for a in new['bo']] == [[r['candidate_id'] for r in a['history']] for a in old['bo']], 'Pure BO changed'
    for arm in ('h1', 'lgbo', 'rbo', 'n6m1'):
        assert all(a['history'][0]['candidate_id'] == b['history'][0]['candidate_id'] for a, b in zip(new[arm], new['first']))
    s_new = {a: [score(x) for x in v] for a, v in new.items()}
    s_old = {a: [score(x) for x in v] for a, v in old.items()}
    versus_bo = {}
    for arm in LLM_ARMS:
        c = contrast(s_new[arm], s_new['bo'], arm + ' - pure BO')
        versus_bo[arm] = dict(c, rule_met=rule(c), mean_minus_24_stream_bo=st.mean(x[0] for x in s_new[arm]) - BO_24_STREAMS)
    controller = {}
    for arm in ('lgbo', 'rbo'):
        c = contrast(s_new['h1'], s_new[arm], 'controller - ' + arm)
        low, high = c['interval90']
        controller[arm] = dict(c, reading=arm + ' ahead' if high < -1. else 'controller ahead' if low > 1. else 'not separated')
    card_effect = {a: dict(hidden=st.mean(x[0] for x in s_old[a]), named=st.mean(x[0] for x in s_new[a]), hidden_final=st.mean(x[1] for x in s_old[a]),
                           named_final=st.mean(x[1] for x in s_new[a])) for a in ('first', 'h1', 'lgbo', 'n6m1')}
    for row in card_effect.values():
        row['change'] = row['named'] - row['hidden']
    result = dict(arms={a: dict(mean_auc=st.mean(x[0] for x in v), sd_auc=st.stdev(x[0] for x in v), mean_final=st.mean(x[1] for x in v)) for a, v in s_new.items()},
                  versus_pure_bo=versus_bo, controller_vs_baselines=controller, card_effect=card_effect,
                  recall=dict(named=recall(new['first'], points), hidden=recall(old['first'], points)),
                  running_best={a: [round(st.mean(x[2][q] for x in v), 2) for q in range(20)] for a, v in s_new.items()},
                  by_stream=[dict(stream=s, first_id=new['first'][s]['history'][0]['candidate_id'], first_yield=new['first'][s]['history'][0]['raw_value'],
                                  **{a: round(s_new[a][s][0], 3) for a in LLM_ARMS + ('bo',)}) for s in STREAMS], api_usage=usage(base))
    (out/'heldout_named.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    with open(out/'trajectories.csv', 'w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(('arm', 'stream', 'query', 'candidate_id', 'raw_value', 'cumulative_best'))
        for arm in LLM_ARMS:
            for s, state in enumerate(new[arm]):
                for q, (row, best) in enumerate(zip(state['history'], s_new[arm][s][2]), 1):
                    writer.writerow((arm, s, q, row['candidate_id'], row['raw_value'], best))
    for name, row in result['arms'].items():
        print(f"{name:6s} AUC {row['mean_auc']:6.2f} (SD {row['sd_auc']:5.2f}) final {row['mean_final']:6.2f}")
    for c in list(versus_bo.values()) + list(controller.values()):
        print(f"{c['contrast']:22s} AUC {c['mean_auc']:+7.2f} [{c['interval90'][0]:+.2f},{c['interval90'][1]:+.2f}] {c['up']}/{c['down']} p={c['sign_flip_p']:.3f} final {c['mean_final']:+.2f} {c.get('rule_met', c.get('reading'))} {c.get('mean_minus_24_stream_bo', '')}")
    print(json.dumps(dict(card_effect=card_effect, recall=result['recall'], usage=result['api_usage'], by_stream=result['by_stream']), indent=0))


if __name__ == '__main__':
    main()
