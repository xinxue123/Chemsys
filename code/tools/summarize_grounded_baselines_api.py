"""Complete-population comparisons and independent report aggregate checks."""
from pathlib import Path
import csv
import hashlib
import json
import math
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'experiments/runs/grounded_baselines_20261105'
OUT = ROOT/'experiments/reports/grounded_baselines_20261105'
TASKS = ('c1','c2','c3','c4','perera','doyle','snar','hplc')
METHODS = ('hyp','lgbo','rbo','bo','first')


def read_csv(name):
    with (OUT/name).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def write_csv(name, rows):
    with (OUT/name).open('w',encoding='utf-8',newline='') as f:
        w = csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def bucket(value):
    return 'tie' if abs(value) <= 1e-9 else 'win' if value > 0 else 'loss'


def main():
    assert len(list(BASE.glob('runs/*/*/completion.json'))) == 64, 'Do not summarize an incomplete population'
    rows = read_csv('per_seed_results.csv')
    assert len(rows) == 160
    lookup = {(r['task'],int(r['seed']),r['method']):r for r in rows}
    assert len(lookup) == 160
    reports = {(r['task'],r['method']):r for r in read_csv('summary.csv')}
    stats = {}
    verified = []
    for t in TASKS:
        for m in METHODS:
            rs = [lookup[(t,s,m)] for s in range(4)]
            assert len({r['objective_unit'] for r in rs}) == 1
            rec = dict(task=t,method=m,completed_seeds=4,objective_unit=rs[0]['objective_unit'])
            for field in ('SearchAUC','final_best'):
                values = [float(r[field]) for r in rs]
                mean = math.fsum(values)/4
                sd = math.sqrt(math.fsum((v-mean)**2 for v in values)/3)
                assert abs(mean-float(reports[(t,m)][field+'_mean'])) < 1e-9
                assert abs(sd-float(reports[(t,m)][field+'_sample_SD'])) < 1e-9
                rec[field+'_mean'],rec[field+'_sample_SD'] = mean,sd
                stats[(t,m,field)] = (mean,sd)
            verified.append(rec)
    write_csv('complete_task_metric_comparison.csv',verified)
    paired = []
    for m in ('hyp','lgbo','rbo'):
        for ref in ('bo','first','hyp'):
            if m == ref:
                continue
            for metric in ('SearchAUC','final_best'):
                seed_counts = Counter(bucket(float(lookup[(t,s,m)][metric])-float(lookup[(t,s,ref)][metric])) for t in TASKS for s in range(4))
                task_counts = Counter(bucket(stats[(t,m,metric)][0]-stats[(t,ref,metric)][0]) for t in TASKS)
                paired.append(dict(method=m,reference=ref,metric=metric,task_seed_pairs=32,
                    seed_wins=seed_counts['win'],seed_ties=seed_counts['tie'],seed_losses=seed_counts['loss'],
                    tasks=8,task_mean_wins=task_counts['win'],task_mean_ties=task_counts['tie'],task_mean_losses=task_counts['loss']))
    write_csv('complete_paired_summary.csv',paired)
    backend_rows = read_csv('LLM_backend_by_round.csv')
    backend_counts = Counter(r['backend'] for r in backend_rows)
    assert len(backend_rows) == 1216
    assert backend_counts['independent_collaboration_subagent'] == 552
    # Include any schema-only response attempts in token and transport accounting.
    usage = Counter()
    response_attempts = Counter()
    provider_ids = []
    for p in BASE.glob('requests/*/*/attempt*/receipt.json'):
        receipt = json.loads(p.read_text(encoding='utf-8'))
        backend = receipt.get('transport_type','independent_collaboration_subagent')
        response_attempts[backend] += 1
        if backend == 'openai_responses_api':
            provider_ids.append(receipt['response_id'])
            folder = p.parent
            sequence = [folder/n for n in ('request.json','api_call_started.json','api_result.json','completion.json','receipt.json')]
            assert all(a.stat().st_mtime_ns <= b.stat().st_mtime_ns for a,b in zip(sequence,sequence[1:])), 'API request/result/seal timing'
            for key in ('input_tokens','output_tokens','total_tokens'):
                usage[key] += receipt['usage'].get(key,0)
            usage['reasoning_tokens'] += receipt['usage'].get('output_tokens_details',{}).get('reasoning_tokens',0)
            usage['cached_input_tokens'] += receipt['usage'].get('input_tokens_details',{}).get('cached_tokens',0)
    assert len(provider_ids) == len(set(provider_ids)), 'Every actual API response must have its own provider ID'
    text = ['Complete matched comparison: eight tasks, four exact-tie seeds, 0+20 paid queries.',
            'Values are arithmetic mean +/- sample SD. Initial LLM point/knowledge are fixed across seeds.',
            'Backend continuation: 552 preserved independent subagent requests; remaining post-first responses use stateless API. Not a pure-API replication.',
            'No cross-task mean of raw objectives is computed.\n']
    for metric in ('SearchAUC','final_best'):
        text += [metric,'Task | Hypothesis/KF/events | LGBO | Reasoning BO | Pure BO | Same first + BO']
        for t in TASKS:
            text.append(t+' | '+' | '.join(f'{stats[(t,m,metric)][0]:.2f} +/- {stats[(t,m,metric)][1]:.2f}' for m in METHODS))
        text.append('')
    text += ['Paired results (descriptive W/T/L; task-seed pairs are not all independent observations):']
    for r in paired:
        text.append(f"{r['method']} vs {r['reference']} {r['metric']}: {r['seed_wins']}/{r['seed_ties']}/{r['seed_losses']} over32 task-seed pairs; task-mean {r['task_mean_wins']}/{r['task_mean_ties']}/{r['task_mean_losses']} over8 tasks")
    text += ['', 'This comparison does not isolate a causal KF or event-refresh effect. Backend differences, fixed initial knowledge, and initial-point choice remain interpretation limits. Same-first BO is required when discussing benefits after the shared first experiment.']
    (OUT/'comparison_tables.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
    summary = dict(status='PASS_COMPLETE_POPULATION_AGGREGATE_RECONCILIATION',
        complete_trajectories=160,new_trajectories=64,frozen_control_trajectories=96,
        task_seed_groups=32,methods=5,per_task_method_aggregates_verified=40,
        common_seeds=[0,1,2,3],observations_per_trajectory=20,
        committed_post_first_requests_by_backend=dict(backend_counts),
        actual_response_attempts_by_backend=dict(response_attempts),API_usage_excluding_preflight=dict(usage),
        API_provider_response_ids_unique=True,API_request_result_seal_order_verified=True,
        independent_aggregate_formula='math.fsum/4; sqrt(fsum squared deviations/3)',
        no_cross_task_raw_objective_mean=True,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'aggregate_verification.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary))


if __name__ == '__main__':main()
