"""Independent raw-history metric arithmetic, without importing optimizer code."""
from pathlib import Path
import csv
import json
import math
import statistics

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/reports/hypothesis_scope_multiseed_20261105'


def main():
    rows=list(csv.DictReader((OUT/'per_seed_results.csv').open(encoding='utf-8-sig')));actual={};maximum=0.
    assert len(rows)==96
    for row in rows:
        state=json.loads((ROOT/row['state_path']).read_text(encoding='utf-8-sig'))
        values=[float(h['raw_value']) for h in state['history']];assert len(values)==20 and state['status']=='COMPLETE'
        incumbent=-math.inf;total=0.
        for y in values:incumbent=max(incumbent,y);total+=incumbent
        auc=total/len(values);error=abs(auc-float(row['SearchAUC']));maximum=max(maximum,error)
        assert error<1e-8 and abs(incumbent-float(row['final_best']))<1e-8
        key=row['task'],int(row['seed']),row['method'];actual[key]=dict(auc=auc,final=incumbent)
    pairs=list(csv.DictReader((OUT/'paired_differences.csv').open(encoding='utf-8-sig')))
    for row in pairs:
        task,seed=row['task'],int(row['seed'])
        for comparator,column in (('bo','delta_AUC_vs_BO'),('first','delta_AUC_vs_same_first')):
            delta=actual[task,seed,'hyp']['auc']-actual[task,seed,comparator]['auc']
            assert abs(delta-float(row[column]))<1e-8
    summary=list(csv.DictReader((OUT/'summary.csv').open(encoding='utf-8-sig')))
    for row in summary:
        seeds=(0,1,2,3) if row['cohort']=='all_0_1_2_3' else (1,2,3)
        for arm in ('bo','first','hyp'):
            for metric,raw in (('SearchAUC','auc'),('final_best','final')):
                values=[actual[row['task'],seed,arm][raw] for seed in seeds]
                assert abs(statistics.mean(values)-float(row[f'{arm}_{metric}_mean']))<1e-8
                assert abs(statistics.stdev(values)-float(row[f'{arm}_{metric}_SD']))<1e-8
    def wtl(column,seeds):
        values=[float(r[column]) for r in pairs if int(r['seed']) in seeds]
        return dict(wins=sum(v>1e-8 for v in values),ties=sum(abs(v)<=1e-8 for v in values),losses=sum(v<-1e-8 for v in values))
    audit=json.loads((OUT/'audit.json').read_text(encoding='utf-8-sig'));assert audit['status']=='PASS_MULTISEED_AUDIT'
    value=dict(status='PASS_INDEPENDENT_METRIC_CHECK',trajectories=96,budget_per_trajectory=20,
        maximum_SearchAUC_error=maximum,all_seeds_vs_BO=wtl('delta_AUC_vs_BO',(0,1,2,3)),
        all_seeds_vs_same_first=wtl('delta_AUC_vs_same_first',(0,1,2,3)),
        new_seeds_vs_BO=wtl('delta_AUC_vs_BO',(1,2,3)),new_seeds_vs_same_first=wtl('delta_AUC_vs_same_first',(1,2,3)),
        new_oracle_reads=0,new_GP_fits=0,new_LLM_calls=0)
    (OUT/'metric_validation.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    report=(OUT/'report.txt').read_text(encoding='utf-8')
    if '\nAUDIT\n' not in report:report+='\nAUDIT\n\n'+json.dumps({a:b for a,b in audit.items() if a!='details'},indent=2)+'\n'
    report+='\nINDEPENDENT METRIC CHECK\n\n'+json.dumps(value,indent=2)+'\n'
    report+='''
OBSERVED ROBUSTNESS LIMITS

Across all32 task/seed combinations, the frozen policy wins23 and loses9 against
pure BO on SearchAUC. Against the SAME LLM first point followed by BO, it has
8 wins,16 ties,8 losses. These are descriptive counts, not independent trials
or a significance test. Most advantages over pure BO are therefore not evidence
of incremental benefit from dynamic knowledge feedback.

Doyle is particularly variable. Its seed2 SearchAUC is53.594238 versus85.716582
for pure BO and69.057268 for same-first BO. Final best is55.565859 versus99.034462
and80.135870. Across all four seeds, its positive mean AUC difference versus BO
is only2.772842 with paired SD24.622189. Across the three NEW seeds alone the
mean difference is-3.533272. The seed0 large advantage is not a stable guarantee.

Case1 loses to the same-first control in3/4 seeds; Case2 is2 wins and2 losses.
Case3, Case4, Perera and SnAr usually preserve the same-first trajectory or its
AUC, with a few small losses/gains. HPLC reproduces the identical same-first
trajectory in all four seeds; its zero SD is conditional on the fixed first
input and the unchanged GP/tie setting, not varying initialization quality.

The current scheme remains useful relative to pure BO in many combinations,
but the frozen dynamic modules do not show broad stable incremental benefit.
Both large wins and severe negative cases remain in the report and raw logs.
'''
    (OUT/'report.txt').write_text(report,encoding='utf-8');print(json.dumps(value))


if __name__=='__main__':main()
