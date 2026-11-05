"""Complete-population audit of unrestricted first points and sparse refresh."""
from copy import deepcopy
from pathlib import Path
from statistics import mean,stdev
import csv
import json
import math
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from llm_event_start import study,policy,prompts,PARAMETERS as P
from llm_screen_qlogei import knowledge
from fast_hit_bo.model import digest
from cold_start_v13 import transport


def csv_write(name,rows):
    if not rows:
        return
    with (study.REPORT/name).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def analyze():
    p=study.check(True)
    for phase in ('main','control'):
        assert study.read(study.BASE/f'{phase}_receipt.json')['status']=='COMPLETE'
    endpoint=study.read(study.BASE/'evaluation_endpoint.json')
    study.REPORT.mkdir(parents=True,exist_ok=True)
    pool=study.knowledge_gp.load_task(ROOT,'Reizman_Suzuki_Case_1',seed=0)
    with np.load(pool.source_path.with_name('oracle.npz'),allow_pickle=False) as z:
        truth=z['values']*100
    metrics,rounds,feedback_rows,calls=[],[],[],[]
    states={}
    for job in p['jobs']:
        folder=study.BASE/'runs'/job['id']
        state=study.read(folder/'state.json')
        assert state['status']=='COMPLETE' and state['job']==job
        assert len(state['history'])==len(state['decisions'])==8
        assert len({o['candidate_id'] for o in state['history']})==8
        states[job['id']]=state
        ys=[h['raw_value'] for h in state['history']]
        best=np.maximum.accumulate(ys)
        hit=next((q for q,value in enumerate(ys,1) if value>=endpoint['threshold95']),None)
        metrics.append(dict(seed=job['seed'],method=job['arm'],initial_id=state['history'][0]['candidate_id'],
            initial_yield=ys[0],first95=hit,hit95=hit is not None,capped_hitting_index=hit or 9,
            final_best=float(best[-1]),SearchAUC8=float(best.mean()),
            LLM_calls=len(state['knowledge_calls']),refresh_calls=len(state['refresh_queries']),
            yield_sequence=' '.join(f'{y:.8f}' for y in ys)))
        replay=study.initial_state(job)
        for q,(obs,commit) in enumerate(zip(state['history'],state['decisions']),1):
            assert commit==study.read(folder/'rounds'/f'{q:02}_commit.json')
            assert commit['commit_sha256']==digest({k:v for k,v in commit.items() if k!='commit_sha256'})
            assert commit['prestate_sha256']==digest(replay) and commit['history_sha256']==digest(replay['history'])
            assert commit['protocol_sha256']==p['protocol_sha256']
            assert commit['decision']['selected_id']==obs['candidate_id']
            j=pool.ix[obs['candidate_id']]
            assert abs(obs['raw_value']-truth[j])<1e-8
            notes=deepcopy(replay['notes'])
            proposal=None
            event=dict(allowed=False,reasons=['normal_full_pool_qLogEI'])
            if job['arm']=='event':
                event=dict(allowed=True,reasons=['zero_observation_initial_proposal']) if q==1 else knowledge.refresh_event(replay,q)
            assert event==commit['knowledge_event']
            if commit['LLM'] is not None:
                assert job['arm']=='event' and event['allowed']
                receipt=commit['LLM']
                with study.transport_scope():
                    request_path=ROOT/receipt['request_relpath']
                    req=transport.verify_request(request_path)
                    exp=study.read(request_path.parent/'export.json')
                    assert study.sha(ROOT/exp['input'])==exp['input_sha256']
                    supplied=study.read(ROOT/exp['input'])
                    assert supplied==dict(system=prompts.SYSTEM,public=prompts.public(pool,replay,q,event))
                raw_path=request_path.parent/'response.raw.json'
                assert study.sha(raw_path)==receipt['response_receipt']['raw_sha256']
                assert study.read(raw_path)==receipt['raw']
                assignment=study.read(request_path.parent/'assignment.json')
                invocation=study.read(request_path.parent/'invocation.json')
                assert assignment['spawn_request']==invocation
                assert assignment['spawn_result']['task_name']==assignment['agent_id']=='/root/'+invocation['task_name']
                assert (invocation['model'],invocation['reasoning_effort'],invocation['fork_turns'])==('gpt-6.1-terra','medium','none')
                plan=policy.validate(receipt['raw'],pool,replay['notes'],q)
                assert plan==receipt['plan']
                notes,proposal=plan['notes'],plan['proposal']
                calls.append(dict(seed=job['seed'],query=q,kind='initial' if q==1 else 'refresh',
                    trigger=json.dumps(event['reasons']),proposed_id=commit['decision']['projected_id'],
                    accepted=commit['decision']['proposal_accepted'],decision_rule=commit['decision']['rule'],
                    selected_id=obs['candidate_id'],qLogEI_opportunity_loss=commit['decision']['qLogEI_opportunity_loss'],
                    summary=receipt['raw']['summary'],request_path=receipt['request_relpath']))
            else:
                assert not event['allowed']
            assert notes==commit['notes_at_commit']
            context=None
            gp_mean=gp_sd=logei=None
            if q>1:
                packet=commit['decision']
                array=folder/'rounds'/f'{q:02}_GP.npz'
                assert study.sha(array)==packet['GP_arrays_sha256']
                assert study.sha(ROOT/packet['checkpoint'])==packet['checkpoint_sha256']
                with np.load(array,allow_pickle=False) as z:
                    context={k:z[k].copy() for k in z.files}
                context['incumbent']=max(h['y'] for h in replay['history'])
                for key in ('mu','variance','logei'):
                    assert np.isfinite(context[key]).all()
                assert (context['variance']>0).all()
                np.testing.assert_array_equal(context['available'],~np.isin(pool.ids,[h['candidate_id'] for h in replay['history']]))
                gp_mean=float(context['mu'][j])*100
                gp_sd=float(context['variance'][j]**.5)*100
                logei=float(context['logei'][j])
                selected=commit['selected_GP']
                assert abs(selected['mean']-gp_mean)<1e-8 and abs(selected['sd']-gp_sd)<1e-8
                assert selected['qLogEI']==logei and selected['hyperparameters_fitted']==(q>=5)
            expected=policy.decide(pool,context,notes,proposal,q,job['seed'],job['arm'])
            if q==1 and job['arm']=='llm_start_bo':
                source=commit['shared_initial_source']
                assert study.sha(ROOT/source['path'])==source['sha256']
                original=study.read(ROOT/source['path'])
                expected.update(selected_id=original['decision']['selected_id'],rule='exact_shared_LLM_initial_point')
            for key,value in expected.items():
                assert value==commit['decision'][key],(job['id'],q,key)
            if q>1:
                assert commit['decision']['qLogEI_opportunity_loss']<=P['logei_budget']+1e-12
                if not commit['decision']['proposal_accepted']:
                    assert obs['candidate_id']==commit['decision']['BO_reference_id']
            after,events=knowledge.feedback(notes,commit['decision']['forecasts'],obs,q)
            stored=study.read(folder/'rounds'/f'{q:02}_feedback.json')
            assert stored==dict(query=q,events=events,notes_after=after)
            for e in events:
                y=float(e['improved'])
                gain=(y-e['p_GP'])**2-(y-e['p_LLM'])**2
                assert abs(gain-e['Brier_gain'])<1e-12
                if gain<=1e-12:
                    assert e['after']['reliability']<=e['before']['reliability']
                if e['permission_before']=='paused':
                    assert e['permission_after']=='paused'
                feedback_rows.append(dict(seed=job['seed'],query=q,id=e['id'],version=e['version'],
                    candidate_id=obs['candidate_id'],p_LLM=e['p_LLM'],p_GP=e['p_GP'],
                    actual_yield=obs['raw_value'],improved=e['improved'],Brier_gain=e['Brier_gain'],
                    kalman_gain=e['kalman_gain'],reliability_before=e['before']['reliability'],
                    reliability_after=e['after']['reliability'],mean_after=e['after']['mean'],
                    variance_after=e['after']['variance'],permission_before=e['permission_before'],
                    permission_after=e['permission_after'],reasons=';'.join(e['reasons'])))
            if commit['LLM'] is not None:
                replay['knowledge_archive'].append(dict(query=q,notes=deepcopy(replay['notes'])))
                replay['knowledge_calls'].append(q)
                if q>1:
                    replay['refresh_queries'].append(q)
            replay['notes']=after
            replay['history'].append(obs)
            replay['decisions'].append(commit)
            replay['status']='COMPLETE' if q==8 else 'RUNNING'
            rounds.append(dict(seed=job['seed'],method=job['arm'],query=q,candidate_id=obs['candidate_id'],
                yield_percent=obs['raw_value'],best=float(best[q-1]),rule=commit['decision']['rule'],
                GP_mean=gp_mean,GP_sd=gp_sd,qLogEI=logei,LLM_called=commit['LLM'] is not None,
                proposal_accepted=commit['decision']['proposal_accepted'],
                BO_reference_id=commit['decision']['BO_reference_id'],KF_updates=len(events),**pool.points[j]))
        assert replay==state
        assert study.read(folder/'runtime.json')['operator_compiles_in_loop']==0
    paired=[]
    for seed in P['seeds']:
        a,b,c=[next(m for m in metrics if m['seed']==seed and m['method']==arm)
               for arm in ('bo','event','llm_start_bo')]
        event=states[f'event_s{seed}']
        control=states[f'llm_start_bo_s{seed}']
        assert event['history'][0]==control['history'][0]
        divergence=next((q for q,(x,y) in enumerate(zip(event['history'],control['history']),1)
                         if x['candidate_id']!=y['candidate_id']),None)
        for q in range(2,(divergence or 8)+1):
            x,y=event['decisions'][q-1],control['decisions'][q-1]
            assert x['fit_key']==y['fit_key']
            assert x['decision']['checkpoint_sha256']==y['decision']['checkpoint_sha256']
        paired.append(dict(seed=seed,random_initial=a['initial_yield'],LLM_initial=b['initial_yield'],
            event_minus_random_SearchAUC8=b['SearchAUC8']-a['SearchAUC8'],
            initial_only_minus_random_SearchAUC8=c['SearchAUC8']-a['SearchAUC8'],
            event_minus_same_start_SearchAUC8=b['SearchAUC8']-c['SearchAUC8'],
            event_minus_random_final=b['final_best']-a['final_best'],
            event_minus_same_start_final=b['final_best']-c['final_best'],
            first_event_control_divergence=divergence))
    summary=[]
    for arm in ('bo','llm_start_bo','event'):
        ms=[m for m in metrics if m['method']==arm]
        summary.append(dict(method=arm,n=len(ms),distinct_initial_points=len({m['initial_id'] for m in ms}),
            initial_yield_mean=mean(m['initial_yield'] for m in ms),
            target_hits=sum(m['hit95'] for m in ms),capped_hit_mean=mean(m['capped_hitting_index'] for m in ms),
            final_best_mean=mean(m['final_best'] for m in ms),SearchAUC8_mean=mean(m['SearchAUC8'] for m in ms),
            SearchAUC8_sd=stdev(m['SearchAUC8'] for m in ms)))
    for name,rows in (('metrics.csv',metrics),('rounds.csv',rounds),('paired.csv',paired),
                      ('summary.csv',summary),('knowledge_feedback.csv',feedback_rows),('LLM_calls.csv',calls)):
        csv_write(name,rows)
    refresh=[c for c in calls if c['kind']=='refresh']
    divergences=[r for r in rounds if r['method']=='event' and r['query']>1 and r['candidate_id']!=r['BO_reference_id']]
    audit=dict(status='PASS',protocol_sha256=p['protocol_sha256'],trajectories=30,observations=240,
        initial_yield_conditioning=False,full_pool_normal_qLogEI_verified=True,
        every_event_proposal_reviewed=True,GP_arrays_and_checkpoint_hashes_verified=True,
        all_preoutcome_commit_chains_and_actual_labels_verified=True,
        independent_relative_Brier_checks=len(feedback_rows),knowledge_feedback_updates=len(feedback_rows),
        initial_LLM_calls=sum(c['kind']=='initial' for c in calls),refresh_calls=len(refresh),
        accepted_refresh_proposals=sum(c['accepted'] for c in refresh),
        actual_postinitial_qLogEI_deviations=len(divergences),
        newly_paused_notes=sum(e['permission_before']!='paused' and e['permission_after']=='paused' for e in feedback_rows),
        same_LLM_initial_control_trajectories=10,all_protected_original_artifacts_unchanged=True,
        operator_compiles_in_loop=0,new_experiments_in_audit=0)
    study.check(True)
    study.write(study.REPORT/'audit.json',audit,replace=True)
    names={'bo':'Random first point + qLogEI','llm_start_bo':'LLM first point + qLogEI',
           'event':'LLM first point + confidence-triggered refresh'}
    lines=['# Case 1: unrestricted LLM first point and sparse confidence refresh','',
        'Thirty complete trajectories use eight observations each. The ten replicate labels are 0..9. '
        'No yield groups or unmeasured yields are used to select an initial point. All arms share '
        'the fixed pool, Mixed GP recipe, MC512 qLogEI, fitting schedule and feature-based tie rules.','',
        '| Method | Mean initial yield (%) | Distinct starts | Hits >=85.09% by query8 | Mean final best (%) | Mean SearchAUC8 |',
        '|---|---|---|---|---|---|']
    for s in summary:
        lines.append(f"| {names[s['method']]} | {s['initial_yield_mean']:.2f} | {s['distinct_initial_points']}/10 | "
            f"{s['target_hits']}/10 | {s['final_best_mean']:.2f} | {s['SearchAUC8_mean']:.2f} |")
    effect=mean(r['event_minus_same_start_SearchAUC8'] for r in paired)
    initial=mean(r['initial_only_minus_random_SearchAUC8'] for r in paired)
    lines.extend(['',f'The mean SearchAUC8 difference from random-start BO is separated into an '
        f'LLM-initialization component of {initial:+.4f} and a same-initial-point refresh component '
        f'of {effect:+.4f}. The latter uses exact shared initial observations, not different LLM '
        'responses or different starting conditions.','',
        f"There are {audit['initial_LLM_calls']} independent initial LLM requests, {len(refresh)} "
        f"confidence-triggered refreshes and {len(feedback_rows)} per-claim KF updates. "
        f"{audit['newly_paused_notes']} notes become paused. {audit['accepted_refresh_proposals']} "
        f"refreshed proposals pass review, and {len(divergences)} post-initial actions differ from "
        'the full-pool qLogEI maximum at their own actual history. Calls and state changes alone '
        'are not evidence of improved experimental selection.','',
        'The initial responder sees zero outcomes and proposes physical conditions with local '
        'knowledge forecasts. Projection chooses a real unqueried pool point using normalized '
        'continuous distance and an exactly matching nominal catalyst. Normal later choices '
        'maximize native qLogEI over the entire unqueried pool. Only scored knowledge degradation '
        'triggers another LLM request. A refreshed candidate must cite permitted favour knowledge '
        'and retain at least half the current maximum exp(qLogEI); rejected proposals return to '
        'the unmodified qLogEI maximum. There is no continuous knowledge screening or rollout.','',
        'From query2, applicable pre-outcome forecasts are scored by relative Brier gain against '
        'the current GP and update per-note KF confidence. Query1 has no incumbent and no relative '
        'improvement score. Recommendation permissions are separate; an accurate failure does '
        'not automatically restore a paused rule. Same-direction/same-scope revisions inherit '
        'their evidence. A fresh responder is used for each request, with no response sharing '
        'between replicates. The exact LLM-start control requires no extra LLM calls.','',
        'SearchAUC8 averages the eight post-query running best values, including the paid first '
        'observation. The good-yield threshold is the previously frozen evaluator-only 95% of '
        'the pool maximum. Unreached first-hit times remain missing; capped hit index9 is only '
        'a comparison penalty. Successful and unsuccessful trajectories are all retained.','',
        'This is a small single-task pilot. Repeated LLM starting points reduce coverage of '
        'distinct initial conditions; ten independent responses are not ten distinct physical '
        'starting experiments. Native fitting emitted line-search warnings on some prefixes; '
        'the inherited BoTorch fitting routine returned models and the completed score arrays '
        'are finite. Do not infer universal optimization benefit or optimizer convergence from '
        'the numerical audit. The stopped low/middle continuous-screening pilot is excluded.','',
        'Inspect metrics.csv, rounds.csv, paired.csv, LLM_calls.csv and knowledge_feedback.csv. '
        'The native run archive contains every prompt, response, projection, commit, GP score '
        'array and KF feedback event. All previous study files are hash-protected.'])
    (study.REPORT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(audit=audit,summary=summary)))


if __name__=='__main__':
    analyze()
