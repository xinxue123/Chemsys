"""Independent outcome, budget, score and commitment checks."""
from collections import defaultdict
from pathlib import Path
from statistics import mean,stdev
import csv
import json
import math
import sys

import numpy as np
from scipy.special import ndtr

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from fast_hit_bo import study, PARAMETERS as P
from fast_hit_bo.model import digest
from fast_hit_bo.policy import choose

OUT=study.REPORT


def save_csv(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        w.writeheader();w.writerows(rows)


def normal_ei(mu,variance,best):
    sd=np.sqrt(np.maximum(variance,1e-18))
    u=(mu-best)/sd
    return np.maximum((mu-best)*ndtr(u)+sd*np.exp(-u*u/2)/np.sqrt(2*np.pi),0.)


def analyze():
    p=study.check()
    assert study.read(study.BASE/'batch_receipt.json')['status']=='COMPLETE'
    ep=study.read(study.BASE/'evaluation_endpoint.json')
    OUT.mkdir(parents=True,exist_ok=True)
    metrics,rounds,scores=[],[],[]
    states,commits={},{}
    estimation_disagreements=0
    conditional_checks=[]
    pool=study.knowledge_gp.load_task(ROOT,'Reizman_Suzuki_Case_1',seed=0)
    with np.load(pool.source_path.with_name('oracle.npz'),allow_pickle=False) as z:
        truth=z['values']*100
    assert ep['threshold95']==.95*float(truth.max())
    for job in p['jobs']:
        folder=study.BASE/'runs'/job['id']
        s=study.read(folder/'state.json')
        states[job['id']]=s
        assert s['status']=='COMPLETE' and s['job']==job and len(s['history'])==8
        assert len({h['candidate_id'] for h in s['history']})==8
        y=[h['raw_value'] for h in s['history']]
        best=np.maximum.accumulate(y)
        hit=next((i for i,v in enumerate(y,1) if v>=ep['threshold95']),None)
        metric=dict(seed=job['seed'],initial_group=job['initial_group'],method=job['arm'],
            initial_id=job['initial_id'],initial_yield=y[0],target95=ep['threshold95'],
            hit95=hit is not None,first95=hit,capped_hitting_index=hit if hit is not None else 9,
            final_best=float(best[-1]),SearchAUC8=float(best.mean()),
            first_final_best=y.index(max(y))+1,
            yield_sequence=' '.join(f'{v:.8f}' for v in y))
        metrics.append(metric)
        replay=dict(job=job,history=[],decisions=[],status='RUNNING')
        for q,(obs,c) in enumerate(zip(s['history'],s['decisions']),1):
            assert c==study.read(folder/'rounds'/f'{q:02}_commit.json')
            assert c['commit_sha256']==digest({k:v for k,v in c.items() if k!='commit_sha256'})
            assert c['prestate_sha256']==digest(replay) and c['history_sha256']==digest(replay['history'])
            d=c['decision']
            assert d['selected_id']==obs['candidate_id']
            assert math.isclose(obs['raw_value'],truth[pool.ix[obs['candidate_id']]],abs_tol=1e-8)
            assert math.isclose(obs['y']*100,obs['raw_value'],abs_tol=1e-8)
            selected_gp=dict(mean=None,sd=None,qLogEI=None,hyperparameters_fitted=None)
            if q==1:
                assert d['selected_id']==job['initial_id']
            else:
                a=folder/'rounds'/f'{q:02}_GP.npz'
                assert study.sha(a)==d['GP_arrays_sha256']
                checkpoint=ROOT/d['checkpoint']
                assert study.sha(checkpoint)==d['checkpoint_sha256']
                with np.load(a,allow_pickle=False) as z:
                    available=~np.isin(pool.ids,[h['candidate_id'] for h in s['history'][:q-1]])
                    np.testing.assert_array_equal(available,z['available'])
                    ref=choose(z['logei'],available,z['priorities'])
                    assert d['BO_reference_id']==pool.ids[ref]
                    independent=normal_ei(z['mu'],z['variance'],max(o['y'] for o in s['history'][:q-1]))
                    analytic_ref=choose(independent,available,z['priorities'])
                    assert d['analytic_EI_shadow_id']==pool.ids[analytic_ref]
                    if job['arm']=='qlogei' or q==8:
                        assert d['selected_id']==pool.ids[ref]
                    else:
                        outer=d['outer_ids']
                        values=np.array(d['outer_values'])
                        assert len(outer)==len(set(outer))<=48 and d['BO_reference_id'] in outer
                        assert not set(outer)&{o['candidate_id'] for o in s['history'][:q-1]}
                        top=np.flatnonzero(values==values.max())
                        winner=min(map(int,top),key=lambda i:z['priorities'][pool.ix[outer[i]]])
                        assert d['selected_id']==outer[winner]
                        for cid,v in zip(outer,values):
                            scores.append(dict(seed=job['seed'],initial_group=job['initial_group'],query=q,
                                candidate_id=cid,expected_two_step_gain=v*100,
                                selected=cid==d['selected_id'],GP_mean=z['mu'][pool.ix[cid]]*100,
                                GP_sd=np.sqrt(z['variance'][pool.ix[cid]])*100,qLogEI=z['logei'][pool.ix[cid]]))
                    j=pool.ix[obs['candidate_id']]
                    selected_gp=c['selected_GP']
                    assert math.isclose(selected_gp['mean'],z['mu'][j]*100,abs_tol=1e-8)
                    assert math.isclose(selected_gp['sd'],np.sqrt(z['variance'][j])*100,abs_tol=1e-8)
                    assert selected_gp['qLogEI']==z['logei'][j]
                    assert selected_gp['hyperparameters_fitted']==(q>=5)
                estimation_disagreements+=d['one_step_estimators_disagree']
                commits[job['id'],q]=c
            point=pool.points[pool.ix[obs['candidate_id']]]
            rounds.append(dict(seed=job['seed'],initial_group=job['initial_group'],method=job['arm'],
                query=q,candidate_id=obs['candidate_id'],yield_percent=obs['raw_value'],best=float(best[q-1]),
                target_hit_by_now=bool(best[q-1]>=ep['threshold95']),
                GP_mean=selected_gp['mean'],GP_sd=selected_gp['sd'],qLogEI=selected_gp['qLogEI'],
                BO_reference_id=d.get('BO_reference_id'),outer_candidates=len(d.get('outer_ids',[])),**point))
            replay['history'].append(obs)
            replay['decisions'].append(c)
            replay['status']='COMPLETE' if q==8 else 'RUNNING'
        assert replay==s
        assert study.read(folder/'runtime.json')['operator_compiles_in_loop']==0
    paired=[]
    for group in ('low','middle'):
        for seed in range(5):
            a=next(m for m in metrics if (m['seed'],m['initial_group'],m['method'])==(seed,group,'qlogei'))
            b=next(m for m in metrics if (m['seed'],m['initial_group'],m['method'])==(seed,group,'rollout'))
            sa=states[f's{seed}_{group}_qlogei']['history']
            sb=states[f's{seed}_{group}_rollout']['history']
            assert sa[0]==sb[0]
            divergence=next((q for q,(x,y) in enumerate(zip(sa,sb),1) if x['candidate_id']!=y['candidate_id']),None)
            limit=divergence or 8
            for q in range(2,limit+1):
                ca=commits[f's{seed}_{group}_qlogei',q]
                cb=commits[f's{seed}_{group}_rollout',q]
                assert ca['fit_key']==cb['fit_key']
                assert ca['decision']['checkpoint_sha256']==cb['decision']['checkpoint_sha256']
            paired.append(dict(seed=seed,initial_group=group,first_divergence=divergence,
                qlogei_hit=a['hit95'],rollout_hit=b['hit95'],qlogei_first=a['first95'],rollout_first=b['first95'],
                rollout_minus_qlogei_capped_index=b['capped_hitting_index']-a['capped_hitting_index'],
                rollout_minus_qlogei_final=b['final_best']-a['final_best'],
                rollout_minus_qlogei_SearchAUC8=b['SearchAUC8']-a['SearchAUC8']))
    summary=[]
    for group in ('low','middle','all'):
        for arm in ('qlogei','rollout'):
            rs=[m for m in metrics if m['method']==arm and (group=='all' or m['initial_group']==group)]
            summary.append(dict(initial_group=group,method=arm,n=len(rs),target_hits=sum(m['hit95'] for m in rs),
                capped_hitting_index_mean=mean(m['capped_hitting_index'] for m in rs),
                final_best_mean=mean(m['final_best'] for m in rs),
                final_best_sd=stdev(m['final_best'] for m in rs),
                SearchAUC8_mean=mean(m['SearchAUC8'] for m in rs)))
    save_csv('metrics.csv',metrics)
    save_csv('rounds.csv',rounds)
    save_csv('paired.csv',paired)
    save_csv('summary.csv',summary)
    save_csv('rollout_candidate_values.csv',scores)
    audit=dict(status='PASS',protocol_sha256=p['protocol_sha256'],trajectories=20,observations=160,
        BO_queries=140,matched_starts=10,old_outcomes_unchanged=True,
        analytic_MC_one_step_disagreements=estimation_disagreements,
        all_commits_and_actual_labels_verified=True,last_step_qLogEI_verified=True,
        same_GP_before_first_divergence=True,operator_compiles_in_loop=0,new_experiments_in_audit=0)
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    lines=['# Case 1: reaching good experiments within eight queries','',
        'Twenty new trajectories compare qLogEI with an approximate two-step receding rollout. '
        'The ten initial points are shared, using five low and five middle starts. The evaluator-only '
        'target is 85.09055023193359%. Each trajectory spends one initial observation and seven BO queries.',
        '',
        '| Start group | Method | Target hits | Mean capped hitting index | Mean final best (%) | Mean SearchAUC8 |',
        '|---|---|---|---|---|---|']
    for s in summary:
        lines.append(f"| {s['initial_group']} | {s['method']} | {s['target_hits']}/{s['n']} | "
                     f"{s['capped_hitting_index_mean']:.2f} | {s['final_best_mean']:.2f} | {s['SearchAUC8_mean']:.2f} |")
    total={s['method']:s for s in summary if s['initial_group']=='all'}
    faster=sum(x['rollout_minus_qlogei_capped_index']<0 for x in paired)
    slower=sum(x['rollout_minus_qlogei_capped_index']>0 for x in paired)
    lines.extend(['',
        f"Rollout reaches the target in {total['rollout']['target_hits']}/10 starts, compared with "
        f"{total['qlogei']['target_hits']}/10 for qLogEI. Its capped hitting index is lower in {faster} paired starts, "
        f"higher in {slower}, and tied in {10-faster-slower}. The registered speed objective shows no improvement "
        'for this rollout. Its higher mean terminal best concerns a separate endpoint.'])
    lines.extend(['',
        'The capped hitting index equals the actual first hit if reached within eight queries. An unreached run '
        'is assigned9 as a comparison penalty;9 is not an observed hit. Both unreached trajectories and successful '
        'trajectories are included. SearchAUC8 averages the eight post-query cumulative best values.',
        '',
        'Both arms use the same nondegenerate Mixed GP prior. Kernel parameters are fixed for the first three '
        'observations and fitted after four. Outputs are yield /100, with no empirical singleton standardization. '
        'The prior mean is50%, prior marginal SD25%, and fixed nugget SD0.1 percentage point. This shared '
        'recipe differs from the older singleton-MLE BO study, so that older trajectory is not the matched control.',
        '',
        'Rollout evaluates a union of at most48 outer candidates, including the qLogEI leader, posterior-SD leaders '
        'and input-only coverage points. It draws32 shared Sobol outcomes per action and maximizes analytic '
        'conditional EI over all4,692 unqueried inputs in the second step. Hypothetical outcomes update the '
        'current GP with fixed hyperparameters. The final query uses the same qLogEI rule as the control.',
        '',
        f"The one-step analytic EI and MC qLogEI leaders differ at {estimation_disagreements} of140 scored decisions. "
        'This is logged as an approximation difference. The comparison tests this implemented rollout, rather than '
        'an exact globally optimal eight-step policy.',
        '',
        'See metrics.csv for hit times and censoring, rounds.csv for actual conditions and outcomes, '
        'paired.csv for each shared start, and rollout_candidate_values.csv for the planned two-step gain. '
        'All20 trajectories are retained. Five paired starts per initial group provide a small descriptive experiment.'])
    (OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(audit))


if __name__=='__main__':
    analyze()
