"""Independently recompute reported metrics from the actual stored observations."""
from collections import defaultdict
from pathlib import Path
from math import fsum,isclose
import csv,json

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/reports/program_kf_weight_20261103'


def read_csv(name):
    with (OUT/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def close(a,b):
    if not isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-9):raise ValueError((a,b))


def main():
    metrics=read_csv('metrics.csv');rounds=defaultdict(list);computed={};observations=0
    for r in read_csv('rounds.csv'):rounds[r['task'],r['seed'],r['method']].append(r)
    for m in metrics:
        state=json.loads((ROOT/m['state_path']).read_text(encoding='utf-8'))
        assert state['status']=='COMPLETE' and len(state['history'])==20
        key=m['task'],m['seed'],m['method'];rr=sorted(rounds[key],key=lambda r:int(r['query']))
        assert [int(r['query']) for r in rr]==list(range(1,21))
        best=-float('inf');curve=[]
        for h,r in zip(state['history'],rr):
            assert h['candidate_id']==r['selected_id'];close(h['raw_value'],r['objective'])
            best=max(best,h['raw_value']);curve.append(best);close(best,r['cumulative_best'])
        auc=fsum(curve)/len(curve)
        close(auc,m['SearchAUC']);close(best,m['final_best']);close(curve.index(best)+1,m['first_final_best_query'])
        computed[key]=auc;observations+=len(curve)
    effects=json.loads((OUT/'paired_effects.json').read_text(encoding='utf-8'))
    contribution=read_csv('round_effects.csv')
    for e in effects:
        key=e['task'],str(e['seed']);expected=computed[key+('dynamic',)]-computed[key+('fixed',)]
        close(expected,e['delta_SearchAUC'])
        selected=[r for r in contribution if (r['task'],r['seed'])==key]
        assert len(selected)==20
        close(fsum(float(r['SearchAUC_contribution']) for r in selected),expected)
    value=dict(status='PASS',native_trajectories=len(metrics),native_observations=observations,
        formula='sum(post-query cumulative maximum of raw observed objectives) / actual query count',
        denominator_verified=20,all_csv_points_and_metrics_match_native_states=True,
        paired_effects_equal_sum_of_round_contributions=True,new_experiment_queries=0)
    (OUT/'report_validation.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(value))


if __name__=='__main__':main()
