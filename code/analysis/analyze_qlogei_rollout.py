"""Audit corrected rollout commitments and compare frozen matched controls."""
from pathlib import Path
from statistics import mean,stdev
import csv
import json
import math
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from qlogei_rollout import study
from fast_hit_bo.model import digest
from fast_hit_bo.policy import choose


def csv_write(name,rows):
    with (study.REPORT/name).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def analyze():
    p=study.check()
    assert study.read(study.BASE/'batch_receipt.json')['status']=='COMPLETE'
    ep=study.read(study.BASE/'evaluation_endpoint.json')
    study.REPORT.mkdir(parents=True,exist_ok=True)
    pool=study.knowledge_gp.load_task(ROOT,'Reizman_Suzuki_Case_1',seed=0)
    with np.load(pool.source_path.with_name('oracle.npz'),allow_pickle=False) as z:
        truth=z['values']*pool.scale
    assert ep['threshold95']==.95*float(truth.max())
    metrics,rounds,values=[],[],[]
    states,commits={},{}
    sampled_evaluations,total_future_candidates=0,0
    oldp=study.read(study.REFERENCE/'protocol.json')
    for job in p['reference_jobs']+p['jobs']:
        new=job['arm']=='rollout_qlogei'
        base=study.BASE if new else study.REFERENCE
        folder=base/'runs'/job['id']
        state=study.read(folder/'state.json')
        assert state['job']==job and state['status']=='COMPLETE'
        assert len(state['history'])==len(state['decisions'])==8
        assert len({o['candidate_id'] for o in state['history']})==8
        states[job['id']]=state
        y=[o['raw_value'] for o in state['history']]
        best=np.maximum.accumulate(y)
        hit=next((q for q,v in enumerate(y,1) if v>=ep['threshold95']),None)
        metrics.append(dict(seed=job['seed'],initial_group=job['initial_group'],method=job['arm'],
            initial_id=job['initial_id'],initial_yield=y[0],target95=ep['threshold95'],
            hit95=hit is not None,first95=hit,capped_hitting_index=hit or 9,
            final_best=float(best[-1]),SearchAUC8=float(best.mean()),
            first_final_best=y.index(max(y))+1,
            yield_sequence=' '.join(f'{v:.8f}' for v in y)))
        replay=dict(job=job,history=[],decisions=[],status='RUNNING')
        for q,(o,c) in enumerate(zip(state['history'],state['decisions']),1):
            d=c['decision']
            assert c==study.read(folder/'rounds'/f'{q:02}_commit.json')
            assert c['commit_sha256']==digest({k:v for k,v in c.items() if k!='commit_sha256'})
            assert c['prestate_sha256']==digest(replay)
            assert c['history_sha256']==digest(replay['history'])
            assert c['protocol_sha256']==(p if new else oldp)['protocol_sha256']
            assert d['selected_id']==o['candidate_id']
            j=pool.ix[o['candidate_id']]
            assert math.isclose(o['raw_value'],truth[j],abs_tol=1e-8)
            assert math.isclose(o['raw_value'],o['y']*100,abs_tol=1e-8)
            selected=dict(mean=None,sd=None,qLogEI=None)
            if q==1:
                assert d['selected_id']==job['initial_id']
            else:
                array=folder/'rounds'/f'{q:02}_GP.npz'
                assert study.sha(array)==d['GP_arrays_sha256']
                assert study.sha(ROOT/d['checkpoint'])==d['checkpoint_sha256']
                with np.load(array,allow_pickle=False) as z:
                    available=~np.isin(pool.ids,[h['candidate_id'] for h in replay['history']])
                    np.testing.assert_array_equal(available,z['available'])
                    reference=choose(z['logei'],available,z['priorities'])
                    assert d['BO_reference_id']==pool.ids[reference]
                    if not new or q==8:
                        assert d['selected_id']==pool.ids[reference]
                    else:
                        assert d['base_policy']=='qLogExpectedImprovement_MC512'
                        assert d['analytic_EI_used'] is False
                        assert d['future_seed']==38112000+q+1 and d['fantasy_seed']==4112026+q
                        outer=d['outer_ids']
                        assert len(outer)==len(set(outer))<=48 and d['BO_reference_id'] in outer
                        score=np.asarray(d['outer_values'])
                        top=np.flatnonzero(score==score.max())
                        winner=min(map(int,top),key=lambda i:z['priorities'][pool.ix[outer[i]]])
                        assert outer[winner]==d['selected_id']
                        assert len(outer)==len(d['first_logei'])==len(d['future_logei'])
                        for k,cid in enumerate(outer):
                            ix=pool.ix[cid]
                            assert available[ix]
                            assert d['first_logei'][k]==z['logei'][ix]
                            future=np.array(d['future_logei'][k])
                            assert len(future)==len(d['future_ids'][k])==32
                            assert np.isfinite(future).all()
                            assert all(available[pool.ix[fc]] and fc!=cid for fc in d['future_ids'][k])
                            independent=np.exp(d['first_logei'][k])+np.exp(future).mean()
                            assert abs(independent-score[k])<1e-12
                            counts=d['evaluated_counts'][k]
                            assert len(counts)==32 and all(1<=n<=available.sum()-1 for n in counts)
                            sampled_evaluations+=sum(counts)
                            total_future_candidates+=32*(available.sum()-1)
                            values.append(dict(seed=job['seed'],initial_group=job['initial_group'],
                                query=q,candidate_id=cid,selected=cid==d['selected_id'],
                                current_qLogEI=d['first_logei'][k],
                                current_MC_improvement_pp=float(np.exp(d['first_logei'][k])*100),
                                mean_future_MC_improvement_pp=float(np.exp(future).mean()*100),
                                expected_two_step_gain_pp=score[k]*100))
                    selected=c['selected_GP']
                    assert abs(selected['mean']-z['mu'][j]*100)<1e-8
                    assert abs(selected['sd']-np.sqrt(z['variance'][j])*100)<1e-8
                    assert selected['qLogEI']==z['logei'][j]
                    assert selected['hyperparameters_fitted']==(q>=5)
                commits[job['id'],q]=c
            rounds.append(dict(seed=job['seed'],initial_group=job['initial_group'],method=job['arm'],
                query=q,candidate_id=o['candidate_id'],yield_percent=o['raw_value'],best=float(best[q-1]),
                GP_mean=selected['mean'],GP_sd=selected['sd'],qLogEI=selected['qLogEI'],
                BO_reference_id=d.get('BO_reference_id'),outer_candidates=len(d.get('outer_ids',[])),
                **pool.points[j]))
            replay['history'].append(o)
            replay['decisions'].append(c)
            replay['status']='COMPLETE' if q==8 else 'RUNNING'
        assert replay==state
        assert study.read(folder/'runtime.json')['operator_compiles_in_loop']==0
    paired=[]
    for group in ('low','middle'):
        for seed in range(5):
            a,b=[next(m for m in metrics if (m['seed'],m['initial_group'],m['method'])==
                (seed,group,arm)) for arm in ('qlogei','rollout_qlogei')]
            sa,sb=[states[f's{seed}_{group}_{arm}'] for arm in ('qlogei','rollout_qlogei')]
            assert sa['history'][0]==sb['history'][0]
            divergence=next((q for q,(x,y) in enumerate(zip(sa['history'],sb['history']),1)
                if x['candidate_id']!=y['candidate_id']),None)
            for q in range(2,(divergence or 8)+1):
                ca,cb=[commits[f's{seed}_{group}_{arm}',q] for arm in ('qlogei','rollout_qlogei')]
                assert ca['fit_key']==cb['fit_key']
                assert ca['decision']['checkpoint_sha256']==cb['decision']['checkpoint_sha256']
            old=study.read(study.REFERENCE/'runs'/f's{seed}_{group}_rollout/state.json')
            old_difference=next((q for q,(x,y) in enumerate(zip(old['history'],sb['history']),1)
                if x['candidate_id']!=y['candidate_id']),None)
            paired.append(dict(seed=seed,initial_group=group,first_divergence=divergence,
                qlogei_hit=a['hit95'],rollout_hit=b['hit95'],qlogei_first=a['first95'],rollout_first=b['first95'],
                rollout_minus_qlogei_capped_index=b['capped_hitting_index']-a['capped_hitting_index'],
                rollout_minus_qlogei_final=b['final_best']-a['final_best'],
                rollout_minus_qlogei_SearchAUC8=b['SearchAUC8']-a['SearchAUC8'],
                first_divergence_from_archived_analytic_rollout=old_difference))
    summary=[]
    for group in ('low','middle','all'):
        for arm in ('qlogei','rollout_qlogei'):
            rows=[m for m in metrics if m['method']==arm and (group=='all' or m['initial_group']==group)]
            summary.append(dict(initial_group=group,method=arm,n=len(rows),
                target_hits=sum(m['hit95'] for m in rows),
                capped_hitting_index_mean=mean(m['capped_hitting_index'] for m in rows),
                final_best_mean=mean(m['final_best'] for m in rows),
                final_best_sd=stdev(m['final_best'] for m in rows),
                SearchAUC8_mean=mean(m['SearchAUC8'] for m in rows)))
    for name,data in (('metrics.csv',metrics),('rounds.csv',rounds),('paired.csv',paired),
                      ('summary.csv',summary),('rollout_candidate_values.csv',values)):
        csv_write(name,data)
    study.check()
    audit=dict(status='PASS',protocol_sha256=p['protocol_sha256'],
        comparison_trajectories=20,new_trajectories=10,new_observations=80,
        frozen_reference_observations=80,matched_starts=10,
        all_original_artifacts_unchanged=True,all_commit_chains_and_labels_verified=True,
        same_GP_before_first_divergence=True,last_step_qLogEI_exact=True,
        native_MC512_qLogEI_at_both_stages=True,analytic_EI_used=False,
        future_candidate_fantasy_pairs=int(total_future_candidates),
        candidate_fantasy_pairs_with_native_MC_evaluation=int(sampled_evaluations),
        operator_compiles_in_loop=0,new_experiments_in_audit=0)
    (study.REPORT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    lines=['# Case 1: qLogEI-based two-step rollout','',
        'This corrected study uses MC512 qLogEI at both stages. Ten new rollout trajectories '
        'are paired with ten frozen qLogEI controls. Each uses the same controlled initial '
        'point and eight total experiments (one initial point plus seven BO selections). '
        'The evaluator-only good-yield threshold is 85.09055023193359%.','',
        '| Start group | Method | Target hits | Mean capped hit index | Mean final best (%) | Mean SearchAUC8 |',
        '|---|---|---|---|---|---|']
    for row in summary:
        lines.append(f"| {row['initial_group']} | {row['method']} | {row['target_hits']}/{row['n']} | "
            f"{row['capped_hitting_index_mean']:.2f} | {row['final_best_mean']:.2f} | {row['SearchAUC8_mean']:.2f} |")
    lines.extend(['','The corrected rollout does not improve the speed objective in this experiment. '
        'It reaches the threshold in 1/10 starts, compared with 3/10 for qLogEI. Its slightly higher '
        'mean terminal best is a separate endpoint and does not establish faster discovery.'])
    faster=sum(r['rollout_minus_qlogei_capped_index']<0 for r in paired)
    slower=sum(r['rollout_minus_qlogei_capped_index']>0 for r in paired)
    lines.extend(['',f'The rollout has a lower capped hitting index in {faster} pairs, '
        f'a higher index in {slower}, and ties in {10-faster-slower}. '
        'An unreached trajectory receives 9 as a comparison penalty; this is not an observed hit. '
        'All successful and unsuccessful trajectories are retained. SearchAUC8 is the mean '
        'of eight cumulative-best post-query values.','',
        'For each of at most 48 first-action candidates, the policy draws 32 current-GP '
        'outcomes, conditions the GP without refitting, and maximizes next-step MC512 qLogEI '
        'over the full unqueried pool. The action value adds exp(current qLogEI) and the '
        'mean exp(best future qLogEI), so improvements are added in yield units. '
        'Both stages retain the native smoothing, Sobol samples and frozen base-policy '
        'incumbent conversion. The final real query directly maximizes qLogEI.','',
        'The outer action shortlist remains an approximation. It includes 16 qLogEI leaders, '
        '16 posterior-variance leaders and two feature-only coverage points per catalyst. '
        'Full-pool future maxima are accelerated using rigorous bounds on the same finite '
        'MC objective. Conditional native BoTorch checks, synthetic stress checks and '
        'commit/label audits are stored separately. No analytic Gaussian EI enters decisions.','',
        'The GP is the unchanged nondegenerate Mixed GP from the matched study: 50% prior '
        'mean, 25 percentage-point prior SD, fixed 0.1 percentage-point nugget SD, no output '
        'standardization, and kernel fitting only after four real observations. This differs '
        'from the older singleton-MLE experiment; its trajectories are not the matched controls.','',
        'The previous analytic-EI rollout is retained in fast_hit_bo_20261104 and is not relabelled. '
        f"The correction changes {sum(r['first_divergence_from_archived_analytic_rollout'] is not None for r in paired)}/10 "
        'selected-point trajectories relative to that archive. The result describes this '
        'small paired experiment and does not establish initialization-independent performance.','',
        'Files: metrics.csv gives every trajectory; rounds.csv gives all queried conditions; '
        'paired.csv gives shared-start differences; rollout_candidate_values.csv separates '
        'current and expected future gain. The run directory retains all pre-outcome '
        'commitments, full current-GP score arrays, fit checkpoints and future maxima.'])
    (study.REPORT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(audit))


if __name__=='__main__':
    analyze()
