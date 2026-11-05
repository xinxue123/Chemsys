"""Trace backward from the known pool optimum through saved native GP decisions."""
from collections import defaultdict
from pathlib import Path
import csv
import hashlib
import json

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
NAME='suzuki_initial_yield_20261104'
BASE=ROOT/'experiments/runs'/NAME
OUT=ROOT/'experiments/reports'/NAME/'optimum_backtrace'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def main():
    p=read(BASE/'protocol.json')
    target_rows,rank_rows,summaries,paths,forward=[],[],[],[],[]
    OUT.mkdir(parents=True,exist_ok=True)
    for case in range(1,5):
        src=ROOT/f'experiments/runs/suzuki_guarded/inputs/case{case}'
        pool=read(src/'pool.json')
        with np.load(src/'oracle.npz',allow_pickle=False) as z:
            yy=z['values']*100
        targets=np.flatnonzero(yy==yy.max())
        assert len(targets)==1
        target=int(targets[0])
        target_id=pool['ids'][target]
        point=pool['points'][target]
        target_rows.append(dict(case=case,target_id=target_id,yield_percent=float(yy[target]),
                                **point))
        for arm in ('low','middle'):
            jobs=[j for j in p['jobs'] if j['case']==case and j['arm']==arm]
            records=[]
            for job in jobs:
                folder=BASE/'runs'/job['id']
                s=read(folder/'state.json')
                hit=next((q for q,h in enumerate(s['history'],1) if h['candidate_id']==target_id),None)
                cumulative=np.maximum.accumulate([h['raw_value'] for h in s['history']])
                auc=float(cumulative.mean())
                records.append(dict(job=job,state=s,first_target=hit,AUC=auc))
                for q in range(2,21):
                    c=read(folder/'rounds'/f'{q:02}_commit.json')
                    path=folder/'rounds'/f'{q:02}_GP.npz'
                    assert sha(path)==c['GP_arrays_sha256']
                    with np.load(path,allow_pickle=False) as z:
                        available=z['available']
                        scores=z['base_logei']
                        top=scores[available].max()
                        ties=np.flatnonzero(available&(scores==top))
                        selected=int(ties[np.argmin(z['tie_priority'][ties])])
                        assert c['selected_id']==pool['ids'][selected]
                        unqueried=bool(available[target])
                        gap=float(top-scores[target]) if unqueried else None
                        rank=1+int(np.count_nonzero(available&(scores>scores[target]))) if unqueried else None
                        ties_target=int(np.count_nonzero(available&(scores==scores[target]))) if unqueried else None
                        target_mean=float(z['mu_y'][target])*100
                        target_sd=float(z['sd_y'][target])*100
                        priority_rank=(1+int(np.count_nonzero(z['tie_priority'][ties]<z['tie_priority'][target]))
                                       if unqueried and gap==0 else None)
                    rank_rows.append(dict(case=case,arm=arm,seed=job['seed'],query=q,
                        target_id=target_id,target_unqueried=unqueried,
                        target_GP_mean=target_mean,target_GP_sd=target_sd,
                        target_qLogEI_rank=rank,top_score_ties=len(ties),target_score_ties=ties_target,
                        target_qLogEI_gap=gap,target_tie_priority_rank=priority_rank,
                        target_selected=selected==target,actual_selected=pool['ids'][selected]))
            successful=[r for r in records if r['first_target'] is not None]
            earliest=min(successful,key=lambda r:(r['first_target'],-r['AUC'])) if successful else None
            q2=[r for r in rank_rows if r['case']==case and r['arm']==arm and r['query']==2]
            unique=[r for r in rank_rows if r['case']==case and r['arm']==arm
                    and r['target_unqueried'] and r['target_qLogEI_gap']==0 and r['top_score_ties']==1]
            summary=dict(case=case,arm=arm,target_id=target_id,target_yield=float(yy[target]),
                initial_draws=5,actual_target_hits=len(successful),
                earliest_actual_query=earliest['first_target'] if earliest else None,
                earliest_actual_seed=earliest['job']['seed'] if earliest else None,
                earliest_unique_top_query=min(r['query'] for r in unique) if unique else None,
                q2_target_top_tie_starts=sum(r['target_qLogEI_gap']==0 for r in q2),
                q2_min_target_gap=min(r['target_qLogEI_gap'] for r in q2),
                q2_min_top_ties=min(r['top_score_ties'] for r in q2 if r['target_qLogEI_gap']==0)
                               if any(r['target_qLogEI_gap']==0 for r in q2) else None)
            summaries.append(summary)
            if earliest:
                seq=earliest['state']['history'][:earliest['first_target']]
                records_path=[]
                for q,o in enumerate(seq,1):
                    j=pool['ids'].index(o['candidate_id'])
                    item=dict(case=case,arm=arm,seed=earliest['job']['seed'],query=q,
                        candidate_id=o['candidate_id'],yield_percent=o['raw_value'],**pool['points'][j])
                    forward.append(item)
                    records_path.append(item)
                paths.append(dict(case=case,arm=arm,seed=earliest['job']['seed'],
                    target_id=target_id,earliest_query=earliest['first_target'],
                    forward=records_path,reverse=list(reversed(records_path))))
    for name,rows in (('global_optima.csv',target_rows),('target_GP_ranks.csv',rank_rows),
                      ('summary.csv',summaries),('earliest_actual_paths.csv',forward)):
        table(name,rows)
    (OUT/'reverse_paths.json').write_text(json.dumps(paths,indent=2)+'\n',encoding='utf-8')
    lines=[
        '# Backtracking from the known best-yield point','',
        'The endpoint is the highest-yield point in each frozen candidate pool. '
        'This analysis reads saved GP arrays and measured trajectories. It makes no new BO selections or GP fits. '
        'The endpoint is known to the retrospective evaluator. It was hidden from the BO controller.',
        '',
        '| Case | Target ID | Maximum yield (%) | Catalyst | Temperature | Residence time | Catalyst loading |',
        '|---|---|---|---|---|---|---|']
    for r in target_rows:
        lines.append(f"| {r['case']} | {r['target_id']} | {r['yield_percent']:.2f} | {r['catalyst']} | "
                     f"{r['temperature']:.2f} | {r['t_res']:.2f} | {r['catalyst_loading']:.3f} |")
    lines.extend(['',
        '| Case | Initial group | Starts reaching the exact target | Earliest actual query | Start seed | Starts with target in q2 maximum tie |',
        '|---|---|---|---|---|---|'])
    for r in summaries:
        hit='unreached' if r['earliest_actual_query'] is None else r['earliest_actual_query']
        seed='-' if r['earliest_actual_seed'] is None else r['earliest_actual_seed']
        lines.append(f"| {r['case']} | {r['arm']} | {r['actual_target_hits']}/5 | {hit} | {seed} | {r['q2_target_top_tie_starts']}/5 |")
    lines.extend(['','## Recorded routes to the endpoint','',
        'Each route below is the earliest exact-optimum visit among the five tested starts in that initial-yield group. '
        'All of its BO actions obey the recorded acquisition and fixed tie rule.'])
    for r in paths:
        reverse=' <- '.join(f"q{x['query']} {x['yield_percent']:.2f}%" for x in r['reverse'])
        lines.extend(['',f"Case {r['case']}, {r['arm']} start, seed {r['seed']}.",reverse])
    lines.extend(['','## An optimistic two-query route','',
        'In many one-observation GP states, the exact target already shares the maximum acquisition score with '
        'hundreds or thousands of candidates. Selecting it from that tie would yield a two-query route from an '
        'allowed low/middle start. This is an evaluator-informed choice within the maximum-score set. '
        'The registered physical-feature tie rule chooses a different member, as the target_tie_priority_rank column shows. '
        'A controller without the target label does not receive this backtracking advantage.',
        '',
        'The five Case 4 middle starts have no exact maximum-score tie containing the target at query two. '
        'Their deficits are at floating-point scale, roughly 1e-15 to 1e-14 in logEI. These tiny differences '
        'are retained exactly under the existing recipe. No tolerance tie or alternative rounding rule was introduced.',
        '',
        'Known target coordinates can therefore describe a desired endpoint and an optimistic path. '
        'A reproducible pure-BO route must also satisfy the acquisition and its tie rule at every preceding state. '
        'The saved paths establish feasibility for the displayed starts, without an exhaustive search over all 4,692 possible initial points.',
        '',
        'Files: global_optima.csv provides endpoint conditions; earliest_actual_paths.csv lists each executable step; '
        'reverse_paths.json reverses those paths; target_GP_ranks.csv records target rank, acquisition gap, '
        'ties and tie priority at all 760 saved GP decisions; summary.csv reports both actual and optimistic reachability.'])
    (OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    audit=dict(status='PASS',native_selections_verified=len(rank_rows),distinct_trajectories=40,
        exact_global_targets=4,new_experiment_queries=0,new_GP_fits=0,new_LLM_calls=0,
        no_approximate_ties_used=True,source_protocol_sha256=p['protocol_sha256'])
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(audit))


if __name__=='__main__':
    main()
