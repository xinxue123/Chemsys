"""Apply the registered new-task screen readings: LLM-start BO minus random-start BO, per task.

Reads saved run states, pools, draws and the hashed predictions only. No model calls, queries or GP fits.
Rules are in docs/hypothesis_scope_v2_registration_20261105.md, under "New-task screen".

    python code/analysis/analyze_hypothesis_scope_v2_screen.py <v2 run folder> <output folder>
"""
import csv
import hashlib
import json
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_hypothesis_scope_v2_first_point_streams import read, sign_flip

TASKS = ('dfl', 'bor', 'chx')
TASK_NAMES = {'dfl': 'deoxyfluorination', 'bor': 'borylation', 'chx': 'Pd coupling'}
STREAMS = range(24)
T90_23 = 1.714
PREDICTIONS_SHA = 'e61f2c19af1daf7e7de294d629b623be208ee98d9ffe6b2f5cc25a3dddfb8f2b'


def score(state, top):
    values = [100*r['raw_value']/top for r in state['history']]
    assert state['status'] == 'COMPLETE' and len(values) == 20
    best = [max(values[:i + 1]) for i in range(20)]
    return st.mean(best), best[-1], best


def main():
    base, out = Path(sys.argv[1])/'screen', Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    assert hashlib.sha256((base/'predictions.json').read_bytes()).hexdigest() == PREDICTIONS_SHA, 'predictions changed'
    predictions = read(base/'predictions.json')['tasks']
    summary, pairs, curves = {}, [], []
    for short in TASKS:
        top = float(read(base/short/'input/pool.json')['objective_scale'])
        diffs, finals = [], []
        llm_auc, bo_auc = [], []
        for t in STREAMS:
            a, fa, ca = score(read(base/short/'runs/llm'/f's{t}'/'state.json'), top)
            b, fb, cb = score(read(base/short/'runs/bo'/f's{t}'/'state.json'), top)
            llm_auc.append(a), bo_auc.append(b), diffs.append(a - b), finals.append(fa - fb)
            pairs.append(dict(task=short, stream=t, llm_first=read(base/short/'draws'/f's{t}.json')['candidate_id'],
                              llm_auc=round(a, 4), llm_final=round(fa, 4), bo_auc=round(b, 4), bo_final=round(fb, 4), difference=round(a - b, 4)))
            curves += [dict(task=short, stream=t, arm=arm, query=q + 1, running_best=round(v, 4)) for arm, c in (('llm', ca), ('bo', cb)) for q, v in enumerate(c)]
        mean, sd = st.mean(diffs), st.stdev(diffs)
        half = T90_23*sd/math.sqrt(len(diffs))
        low, high = mean - half, mean + half
        reading = 'LLM start leads' if low > 1.0 else 'random start leads' if high < -1.0 else 'not separated'
        predicted = predictions[short]['prediction']
        consistent = None if predicted == 'no prediction' else (reading == 'LLM start leads') if predicted == 'gain' else (reading != 'LLM start leads')
        summary[short] = dict(prediction=predicted, mean_pick_percentile=predictions[short]['mean_pick_percentile'],
                              llm_start_auc=round(st.mean(llm_auc), 2), random_start_auc=round(st.mean(bo_auc), 2),
                              difference=round(mean, 2), interval90=[round(low, 2), round(high, 2)], reading=reading, consistent_with_screen_rule=consistent,
                              up=sum(d > 1e-9 for d in diffs), down=sum(d < -1e-9 for d in diffs), tied=sum(abs(d) <= 1e-9 for d in diffs),
                              sign_flip_p=round(sign_flip(diffs), 4), final_best_difference=round(st.mean(finals), 2))
    for short in TASKS:  # Step 4 (exploratory): the controller from the LLM pick, if its runs exist.
        r1 = [base/short/'runs/r1'/f's{t}'/'state.json' for t in STREAMS]
        if not all(p.exists() and read(p)['status'] == 'COMPLETE' for p in r1):
            continue
        top = float(read(base/short/'input/pool.json')['objective_scale'])
        ctrl = [score(read(p), top) for p in r1]
        step4 = dict(controller_auc=round(st.mean(c[0] for c in ctrl), 2), mean_calls=round(st.mean(len(read(p)['calls']) for p in r1), 2))
        for name, group in (('minus_llm_start_bo', 'llm'), ('minus_random_start_bo', 'bo')):
            other = [score(read(base/short/'runs'/group/f's{t}'/'state.json'), top) for t in STREAMS]
            d = [c[0] - o[0] for c, o in zip(ctrl, other)]
            m, half = st.mean(d), T90_23*st.stdev(d)/math.sqrt(len(d))
            step4[name] = dict(difference=round(m, 2), interval90=[round(m - half, 2), round(m + half, 2)],
                               reading='controller leads' if m - half > 1.0 else 'controller trails' if m + half < -1.0 else 'not separated',
                               up=sum(x > 1e-9 for x in d), down=sum(x < -1e-9 for x in d), tied=sum(abs(x) <= 1e-9 for x in d),
                               final_best_difference=round(st.mean(c[1] - o[1] for c, o in zip(ctrl, other)), 2))
        for row in pairs:
            if row['task'] == short:
                c = ctrl[row['stream']]
                row.update(controller_auc=round(c[0], 4), controller_final=round(c[1], 4))
        curves += [dict(task=short, stream=t, arm='controller', query=q + 1, running_best=round(v, 4)) for t, c in enumerate(ctrl) for q, v in enumerate(c[2])]
        summary[short]['step4_exploratory'] = step4
    tested = {s: v for s, v in summary.items() if v['consistent_with_screen_rule'] is not None}
    kinds = {v['prediction'] for v in tested.values()}
    if not all(v['consistent_with_screen_rule'] for v in tested.values()):
        overall = 'At least one outcome was inconsistent with the original screen rule.'
    elif kinds == {'gain', 'no gain'}:
        overall = 'Both branches were tested. All outcomes were consistent with the original screen rule.'
    elif kinds == {'no gain'}:
        count = 'three ' if len(tested) == 3 else ''
        overall = f'Only the no-gain branch was tested. All {count}outcomes were consistent with the original screen rule.'
        random_leads = [TASK_NAMES[task] for task, result in tested.items() if result['reading'] == 'random start leads']
        inconclusive = [TASK_NAMES[task] for task, result in tested.items() if result['reading'] == 'not separated']
        if random_leads:
            overall += f" Random-start BO led on {' and '.join(random_leads)}."
        if len(inconclusive) == 1:
            overall += f' The {inconclusive[0]} result was inconclusive under the interval criterion.'
        elif inconclusive:
            overall += f" The interval criterion left the results inconclusive on {' and '.join(inconclusive)}."
    elif kinds == {'gain'}:
        overall = 'Only the gain branch was tested. All outcomes were consistent with the original screen rule.'
    else:
        overall = 'No branch was tested.'
    record = dict(tasks=summary, overall=overall, no_task_reached_90=all(v['mean_pick_percentile'] < 90 for v in summary.values()),
                  predictions_sha256=PREDICTIONS_SHA, note='Registered readings; no model calls, queries or GP fits in this analysis.')
    (out/'screen.json').write_text(json.dumps(record, indent=1), encoding='utf-8')
    for name, rows in (('pairs.csv', pairs), ('trajectories.csv', curves)):
        with open(out/name, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader()
            w.writerows(rows)
    print(json.dumps(record, indent=1))


if __name__ == '__main__':
    main()
