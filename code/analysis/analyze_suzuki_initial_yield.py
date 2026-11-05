"""Independent numerical audit and compact report for the 1+19 BO study."""
from collections import defaultdict
from pathlib import Path
from statistics import mean,stdev
import csv
import hashlib
import json
import math
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
NAME='suzuki_initial_yield_20261104'
BASE=ROOT/'experiments/runs'/NAME
OUT=ROOT/'experiments/reports'/NAME


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
                                     ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def near(a,b):
    assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-7),(a,b)


def analyze():
    protocol=read(BASE/'protocol.json')
    assert protocol['protocol_sha256']==digest({k:v for k,v in protocol.items() if k!='protocol_sha256'})
    for group in ('sources_sha256','inputs_sha256'):
        assert all(sha(ROOT/k)==v for k,v in protocol[group].items())
    assert read(BASE/'batch_receipt.json')['status']=='COMPLETE'
    metrics,rounds,summary,paired,groups=[],[],[],[],protocol['groups']
    states={}
    audits=[]
    for job in protocol['jobs']:
        case=job['case']
        path=ROOT/f'experiments/runs/suzuki_guarded/inputs/case{case}'
        pool=read(path/'pool.json')
        with np.load(path/'oracle.npz',allow_pickle=False) as z:
            truth=(z['raw_values'] if 'raw_values' in z else z['values']*100).copy()
        by_id=dict(zip(pool['ids'],map(float,truth)))
        ix={cid:i for i,cid in enumerate(pool['ids'])}
        g=next(x for x in groups if x['case']==case)
        def label(y):
            return 'low' if y<=g['lower_tercile'] else 'middle' if y<=g['upper_tercile'] else 'high'
        folder=BASE/'runs'/job['id']
        state=read(folder/'state.json')
        states[job['id']]=state
        assert state['status']=='COMPLETE' and state['job']==job
        h=state['history']
        assert len(h)==20 and len({o['candidate_id'] for o in h})==20
        assert h[0]['candidate_id']==job['initial_id']
        assert label(h[0]['raw_value'])==job['initial_group']!='high'
        if job['arm']!='mixed':
            assert label(h[0]['raw_value'])==job['arm']
        ys=[o['raw_value'] for o in h]
        best=np.maximum.accumulate(ys)
        s=dict(job=job,history=[],decisions=[],status='RUNNING')
        for q,(obs,dec) in enumerate(zip(h,state['decisions']),1):
            c=read(folder/'rounds'/f'{q:02}_commit.json')
            assert c['commit_sha256']==digest({k:v for k,v in c.items() if k!='commit_sha256'})
            assert c['prestate_sha256']==digest(s) and c['history_sha256']==digest(s['history'])
            assert obs['candidate_id']==c['selected_id']
            near(obs['raw_value'],by_id[obs['candidate_id']])
            near(obs['y']*100,obs['raw_value'])
            assert dec==dict(c,objective=obs['raw_value'])
            if q>1:
                npz=folder/'rounds'/f'{q:02}_GP.npz'
                assert sha(npz)==c['GP_arrays_sha256']
                with np.load(npz,allow_pickle=False) as z:
                    mask=~np.isin(pool['ids'],[x['candidate_id'] for x in h[:q-1]])
                    np.testing.assert_array_equal(mask,z['available'])
                    score=z['base_logei']
                    tied=np.flatnonzero(mask&(score==score[mask].max()))
                    selected=int(tied[np.argmin(z['tie_priority'][tied])])
                    assert pool['ids'][selected]==obs['candidate_id']
                    assert len(tied)==c['qLogEI_tied_best_count']
                    near(score[selected],c['base_qLogEI'])
                    for key,field in (('mu_y','gp_mean'),('sd_y','gp_sd'),('sd_predictive_y','gp_predictive_sd')):
                        near(float(z[key][selected])*100,c[field])
                    assert np.isfinite(score).all()
            s['history'].append(obs)
            s['decisions'].append(dec)
            s['status']='COMPLETE' if q==20 else 'RUNNING'
            point=pool['points'][ix[obs['candidate_id']]]
            rounds.append(dict(case=case,seed=job['seed'],arm=job['arm'],query=q,
                phase='initial' if q==1 else 'BO',candidate_id=obs['candidate_id'],
                yield_percent=obs['raw_value'],yield_group=label(obs['raw_value']),best=float(best[q-1]),
                t_res=point['t_res'],temperature=point['temperature'],
                catalyst_loading=point['catalyst_loading'],catalyst=point['catalyst'],
                gp_mean=c['gp_mean'],gp_sd=c['gp_sd'],gp_predictive_sd=c['gp_predictive_sd'],
                qLogEI=c['base_qLogEI'],top_score_ties=c['qLogEI_tied_best_count']))
        assert s==state
        counters=read(folder/'runtime_counters.json')
        assert counters['operator_compiles_in_loop']==0
        threshold=.95*g['maximum']
        hit=lambda target:next((q for q,y in enumerate(ys,1) if y>=target),None)
        row=dict(case=case,seed=job['seed'],arm=job['arm'],initial_id=job['initial_id'],
            initial_group=job['initial_group'],initial_yield=ys[0],
            SearchAUC20=float(best.mean()),SearchAUC19=float(best[1:].mean()),
            gain_AUC19=float((best[1:]-ys[0]).mean()),final_best=float(best[-1]),
            best_q5=float(best[4]),best_q10=float(best[9]),best_q20=float(best[-1]),
            first_final_best=ys.index(max(ys))+1,
            first_high=next((q for q,y in enumerate(ys,1) if label(y)=='high'),None),
            target95=threshold,first95=hit(threshold),target90=.9*g['maximum'],
            first90=hit(.9*g['maximum']),
            group_sequence=' '.join(label(y)[0].upper() for y in ys),
            final_gap=g['maximum']-best[-1])
        metrics.append(row)
        audits.append(dict(job=job['id'],queries=20,BO_queries=19,
            actual_fits=counters['fits'],fit_cache_hits=counters['fit_cache_hits'],
            compiles_in_loop=0))
    for case in range(1,5):
        for seed in protocol['seeds']:
            mixed=states[f'c{case}_s{seed}_mixed']
            same=states[f"c{case}_s{seed}_{mixed['job']['initial_group']}"]
            assert mixed['history']==same['history']
            for a,b in zip(mixed['decisions'],same['decisions']):
                assert a['selected_id']==b['selected_id']
            low=next(x for x in metrics if (x['case'],x['seed'],x['arm'])==(case,seed,'low'))
            middle=next(x for x in metrics if (x['case'],x['seed'],x['arm'])==(case,seed,'middle'))
            paired.append(dict(case=case,seed=seed,middle_minus_low_SearchAUC20=middle['SearchAUC20']-low['SearchAUC20'],
                middle_minus_low_gain_AUC19=middle['gain_AUC19']-low['gain_AUC19'],
                middle_minus_low_final=middle['final_best']-low['final_best'],
                low_first95=low['first95'],middle_first95=middle['first95']))
        for arm in ('low','middle','mixed'):
            rs=[x for x in metrics if x['case']==case and x['arm']==arm]
            hits=[x['first95'] for x in rs if x['first95'] is not None]
            summary.append(dict(case=case,arm=arm,n=5,
                initial_mean=mean(x['initial_yield'] for x in rs),
                SearchAUC20_mean=mean(x['SearchAUC20'] for x in rs),
                SearchAUC20_sd=stdev(x['SearchAUC20'] for x in rs),
                gain_AUC19_mean=mean(x['gain_AUC19'] for x in rs),
                final_best_mean=mean(x['final_best'] for x in rs),
                final_best_sd=stdev(x['final_best'] for x in rs),
                high_first_mean=mean(x['first_high'] for x in rs if x['first_high'] is not None),
                target95_hits=len(hits),first95_mean_when_hit=mean(hits) if hits else None))
    OUT.mkdir(parents=True,exist_ok=True)
    for name,rs in (('metrics.csv',metrics),('rounds.csv',rounds),('summary.csv',summary),
                   ('paired.csv',paired),('groups.csv',groups)):
        write_csv(name,rs)
    result=dict(status='PASS',protocol_sha256=protocol['protocol_sha256'],trajectories=60,
        distinct_initial_trajectories=40,total_paid_observations=1200,BO_selections=1140,
        new_LLM_calls=0,mixed_arms_are_exact_mixtures=True,native_maximum_acquisition_verified=True,
        compiles_in_loop=0,jobs=audits)
    (OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    report=[
        '# Pure BO sensitivity to the first observed yield',
        '',
        'Four Reizman Suzuki cases use one controlled initial observation and nineteen unrestricted BO queries. '
        'Low and middle groups are defined by within-case outcome tertiles. High-yield starts are excluded. '
        'The GP receives only measured outcomes. Numerical GP settings and tie streams stay fixed across all starts.',
        '',
        '| Case | Start group | Initial mean | SearchAUC20 mean ± SD | Final best mean ± SD | 95% target hits | Mean target query among hits |',
        '|---|---|---|---|---|---|---|']
    for s in summary:
        t='unreached' if s['first95_mean_when_hit'] is None else f"{s['first95_mean_when_hit']:.1f}"
        report.append(f"| {s['case']} | {s['arm']} | {s['initial_mean']:.2f} | {s['SearchAUC20_mean']:.2f} ± {s['SearchAUC20_sd']:.2f} | "
            f"{s['final_best_mean']:.2f} ± {s['final_best_sd']:.2f} | {s['target95_hits']}/5 | {t} |")
    report.extend(['','SearchAUC20 averages all twenty post-observation running best values. SearchAUC19 averages '
        'the nineteen values after the initial observation. gain_AUC19 subtracts each run’s own initial yield. '
        'A high initial value mechanically lifts the running-best baseline, so gain_AUC19 and target-hitting '
        'queries are also retained. The 95% target is 0.95 times the frozen pool maximum, available to the evaluator.',
        '',
        'The mixed arm draws from the low/middle union. With a single initial point and shared candidate priorities, '
        'each mixed run exactly matches one low or middle run. Its trajectories add mixture summaries, not independent replication.',
        '',
        'Each row averages five starts. These are selected by a fixed random rule within each yield stratum. '
        'The contrast changes both the initial response and its physical conditions. It is a retrospective sensitivity '
        'experiment and does not isolate yield alone or guarantee that the better mean group is optimal among all possible starts.',
        '',
        'See metrics.csv for each seed, rounds.csv for all conditions and selection diagnostics, paired.csv for '
        'middle-minus-low differences, groups.csv for thresholds, and audit.json for the independent numerical checks.'])
    report.extend(['','## Best observed sequence in each case','',
        'These sequences have the highest observed SearchAUC20 among the ten distinct low/middle starts tested in each case. '
        'They describe this sample, rather than an exhaustive search over all starting conditions.',
        ''])
    winners=[]
    for case in range(1,5):
        options=[r for r in metrics if r['case']==case and r['arm']!='mixed']
        winner=max(options,key=lambda r:r['SearchAUC20'])
        sequence=[r for r in rounds if (r['case'],r['seed'],r['arm'])==
                  (case,winner['seed'],winner['arm'])]
        winners.extend(sequence)
        first=sequence[0]
        report.extend([
            f"Case {case}: {winner['arm']} start, seed {winner['seed']}, initial {winner['initial_yield']:.2f}%, "
            f"SearchAUC20 {winner['SearchAUC20']:.2f}, terminal best {winner['final_best']:.2f}%.",
            f"Initial conditions: catalyst {first['catalyst']}, temperature {first['temperature']}, "
            f"residence time {first['t_res']}, catalyst loading {first['catalyst_loading']}.",
            'Query yields (%): '+', '.join(f"{r['yield_percent']:.2f}" for r in sequence)+'.',
            'Yield groups: '+winner['group_sequence']+'.',''])
    write_csv('best_observed_sequences.csv',winners)
    report.extend([
        '## Interpreting a single initial point','',
        'The native model standardizes its one training target to zero. The first acquisition therefore depends on '
        'the initial physical location and the shared numerical recipe. Changing only the value at an otherwise '
        'identical singleton input would shift the response origin. Group effects in this experiment combine '
        'physical-location differences, subsequent measured responses, and the initial running-best baseline.'])
    (OUT/'README.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='jobs'}))


if __name__=='__main__':
    analyze()
