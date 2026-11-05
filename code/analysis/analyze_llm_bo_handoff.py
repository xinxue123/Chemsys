"""Prospective commitment replay and matched-start handoff comparison."""
from copy import deepcopy
from pathlib import Path
from statistics import mean,stdev
import csv
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from llm_bo_handoff import study,policy,prompts
from llm_event_start import policy as original_policy
from llm_screen_qlogei import knowledge
from cold_start_v13 import transport
from fast_hit_bo.model import digest


def csv_save(name,rows):
    if not rows:
        return
    with (study.REPORT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def metric(state,method,threshold):
    values=[h['raw_value'] for h in state['history']]
    best=np.maximum.accumulate(values)
    hit=next((q for q,y in enumerate(values,1) if y>=threshold),None)
    return dict(seed=state['job']['seed'],method=method,initial_id=state['history'][0]['candidate_id'],
        initial_yield=values[0],observations=len(values),SearchAUC8=float(best.mean()),
        final_best=float(best[-1]),hit95=hit is not None,first95=hit,capped_hitting_index=hit or 9,
        yield_sequence=' '.join(f'{v:.8f}' for v in values))


def analyze():
    p=study.check(True)
    assert study.read(study.BASE/'batch_receipt.json')['status']=='COMPLETE'
    assert study.read(study.BASE/'preflight.json')['status']=='PASS'
    study.REPORT.mkdir(parents=True,exist_ok=True)
    pool=study.knowledge_gp.load_task(ROOT,'Reizman_Suzuki_Case_1',seed=0)
    ep=study.read(study.BASE/'evaluation_endpoint.json')
    with np.load(pool.source_path.with_name('oracle.npz'),allow_pickle=False) as z:
        truth=z['values']*100
    assert ep['threshold95']==.95*float(truth.max())
    metrics,rounds,feedback_rows,calls=[],[],[],[]
    refs={}
    for job in p['references']:
        s=study.read(study.REFERENCE/'runs'/job['id']/'state.json')
        refs[job['id']]=s
        assert len(s['history'])==8 and s['status']=='COMPLETE'
        metrics.append(metric(s,job['arm'],ep['threshold95']))
        best=np.maximum.accumulate([h['raw_value'] for h in s['history']])
        for q,h in enumerate(s['history'],1):
            j=pool.ix[h['candidate_id']]
            assert abs(h['raw_value']-truth[j])<1e-8
            rounds.append(dict(seed=job['seed'],method=job['arm'],query=q,candidate_id=h['candidate_id'],
                yield_percent=h['raw_value'],best=float(best[q-1]),authority_ceiling=None,
                effective_logei_gap=None,actual_logei_gap=None,budget_candidates=None,
                eligible_candidates=None,knowledge_changed_choice=None,BO_reference_id=None,
                fresh_refresh=False,KF_updates=None,rule='frozen_reference',**pool.points[j]))
    states={}
    for job in p['jobs']:
        folder=study.BASE/'runs'/job['id']
        state=study.read(folder/'state.json')
        states[job['id']]=state
        assert state['job']==job and state['status']=='COMPLETE'
        assert len(state['history'])==len(state['decisions'])==8
        assert len({h['candidate_id'] for h in state['history']})==8
        assert state['history'][0]==refs[job['initial_source_job']]['history'][0]
        metrics.append(metric(state,'handoff',ep['threshold95']))
        replay=study.initial_state(job)
        best=np.maximum.accumulate([h['raw_value'] for h in state['history']])
        for q,(obs,c) in enumerate(zip(state['history'],state['decisions']),1):
            assert c==study.read(folder/'rounds'/f'{q:02}_commit.json')
            assert c['commit_sha256']==digest({k:v for k,v in c.items() if k!='commit_sha256'})
            assert c['prestate_sha256']==digest(replay) and c['history_sha256']==digest(replay['history'])
            assert c['protocol_sha256']==p['protocol_sha256']
            assert c['decision']['selected_id']==obs['candidate_id']
            j=pool.ix[obs['candidate_id']]
            assert abs(obs['raw_value']-truth[j])<1e-8 and abs(obs['y']*100-obs['raw_value'])<1e-8
            notes=deepcopy(replay['notes'])
            proposal=None
            context=None
            if q==1:
                source=c['initial_source']
                assert study.sha(ROOT/source['state_path'])==source['state_sha256']
                assert study.sha(ROOT/source['commit_path'])==source['commit_sha256']
                original=study.read(ROOT/source['commit_path'])
                assert digest(obs)==source['observation_sha256']
                assert original['LLM']['response_receipt']['raw_sha256']==source['original_LLM_raw_sha256']
                plan=original_policy.validate(original['LLM']['raw'],pool,[],1)
                notes,proposal=plan['notes'],plan['proposal']
                assert c['LLM'] is None and c['selected_GP'] is None
            else:
                event=study.event_for(replay,q)
                assert event==c['knowledge_event']
                assert event['allowed']==(c['LLM'] is not None)
                if c['LLM'] is not None:
                    r=c['LLM']
                    path=ROOT/r['request_relpath']
                    with study.transport_scope():
                        request=transport.verify_request(path)
                    export=study.read(path.parent/'export.json')
                    assert study.sha(ROOT/export['input'])==export['input_sha256']
                    assert study.read(ROOT/export['input'])==dict(system=prompts.SYSTEM,
                        public=prompts.public(pool,replay,q,event,policy.authority(q)))
                    assert study.sha(path.parent/'response.raw.json')==r['response_receipt']['raw_sha256']
                    assert study.read(path.parent/'response.raw.json')==r['raw']
                    assignment=study.read(path.parent/'assignment.json')
                    invocation=study.read(path.parent/'invocation.json')
                    assert assignment['spawn_request']==invocation
                    assert assignment['spawn_result']['task_name']==assignment['agent_id']=='/root/'+invocation['task_name']
                    assert (invocation['model'],invocation['reasoning_effort'],invocation['fork_turns'])==('gpt-6.1-terra','medium','none')
                    plan=original_policy.validate(r['raw'],pool,replay['notes'],q)
                    assert plan==r['plan']
                    notes,proposal=plan['notes'],plan['proposal']
                    calls.append(dict(seed=job['seed'],query=q,authority_ceiling=policy.authority(q),
                        trigger=json.dumps(event['reasons']),proposed_id=c['decision']['projected_id'],
                        accepted=c['decision']['proposal_accepted'],rejection=c['decision']['proposal_rejection'],
                        selected_id=obs['candidate_id'],summary=r['raw']['summary'],
                        returned_hypotheses=len(r['raw']['hypotheses']),request_path=r['request_relpath']))
                a=folder/'rounds'/f'{q:02}_GP.npz'
                assert study.sha(a)==c['decision']['GP_arrays_sha256']
                assert study.sha(ROOT/c['decision']['checkpoint'])==c['decision']['checkpoint_sha256']
                with np.load(a,allow_pickle=False) as z:
                    saved={key:z[key].copy() for key in z.files}
                context={key:saved[key] for key in ('mu','variance','logei','available','priorities')}
                context['incumbent']=max(h['y'] for h in replay['history'])
                if c['LLM'] is not None:
                    before_refresh,_=policy.decide(pool,context,replay['notes'],None,q,job['seed'])
                    calls[-1]['selected_without_refresh']=before_refresh['selected_id']
                    calls[-1]['refresh_changed_current_choice']=before_refresh['selected_id']!=obs['candidate_id']
                np.testing.assert_array_equal(context['available'],~np.isin(pool.ids,[h['candidate_id'] for h in replay['history']]))
                for key in ('mu','variance','logei'):
                    assert np.isfinite(context[key]).all()
                assert (context['variance']>0).all()
                stats=c['selected_GP']
                assert abs(stats['mean']-context['mu'][j]*100)<1e-8
                assert abs(stats['sd']-np.sqrt(context['variance'][j])*100)<1e-8
                assert stats['qLogEI']==context['logei'][j]
                assert stats['hyperparameters_fitted']==(q>=5)
            assert notes==c['notes_at_commit']
            packet,arrays=policy.decide(pool,context,notes,proposal,q,job['seed'])
            for key,value in packet.items():
                assert value==c['decision'][key],(job['id'],q,key)
            if q>1:
                for key,value in arrays.items():
                    np.testing.assert_array_equal(value,saved[key])
                assert packet['actual_logei_gap']<=packet['effective_logei_gap']+1e-12
                assert packet['handoff_authority_cap']==(8-q)/7
                if q==8:
                    assert packet['selected_id']==packet['BO_reference_id']
                    assert packet['phase']=='pure_BO' and packet['effective_logei_gap']==0
                    assert c['LLM'] is None
            updated,events=knowledge.feedback(notes,packet['forecasts'],obs,q)
            stored=study.read(folder/'rounds'/f'{q:02}_feedback.json')
            expected=dict(query=q,events=events,notes_after=updated)
            if q==1:
                expected['reused_initial_observation']=True
            assert stored==expected
            for e in events:
                y=float(e['improved'])
                independent=(y-e['p_GP'])**2-(y-e['p_LLM'])**2
                assert abs(independent-e['Brier_gain'])<1e-12
                if e['Brier_gain']<=1e-12:
                    assert e['after']['reliability']<=e['before']['reliability']
                if e['permission_before']=='paused':
                    assert e['permission_after']=='paused'
                feedback_rows.append(dict(seed=job['seed'],query=q,id=e['id'],version=e['version'],
                    candidate_id=obs['candidate_id'],yield_percent=obs['raw_value'],
                    p_LLM=e['p_LLM'],p_GP=e['p_GP'],improved=e['improved'],Brier_gain=e['Brier_gain'],
                    kalman_gain=e['kalman_gain'],reliability_before=e['before']['reliability'],
                    reliability_after=e['after']['reliability'],permission_before=e['permission_before'],
                    permission_after=e['permission_after'],reasons=';'.join(e['reasons']),
                    promoted_selection=e['promoted_selection']))
            if c['LLM'] is not None:
                replay['knowledge_archive'].append(dict(query=q,notes=deepcopy(replay['notes'])))
                replay['knowledge_calls'].append(q)
                replay['refresh_queries'].append(q)
            replay['notes']=updated
            replay['history'].append(obs)
            replay['decisions'].append(c)
            replay['status']='COMPLETE' if q==8 else 'RUNNING'
            rounds.append(dict(seed=job['seed'],method='handoff',query=q,candidate_id=obs['candidate_id'],
                yield_percent=obs['raw_value'],best=float(best[q-1]),authority_ceiling=packet['handoff_authority_cap'],
                effective_logei_gap=packet['effective_logei_gap'],actual_logei_gap=packet.get('actual_logei_gap'),
                budget_candidates=packet.get('budget_candidates'),eligible_candidates=packet.get('eligible_candidates'),
                knowledge_changed_choice=packet['knowledge_used'] if q>1 else None,
                BO_reference_id=packet['BO_reference_id'],fresh_refresh=c['LLM'] is not None,
                KF_updates=len(events),rule=packet['rule'],**pool.points[j]))
        assert replay==state
        assert study.read(folder/'runtime.json')['operator_compiles_in_loop']==0
    pairs=[]
    for seed in range(10):
        m={method:next(r for r in metrics if r['seed']==seed and r['method']==method)
           for method in ('bo','llm_start_bo','event','handoff')}
        new=states[f'handoff_s{seed}']
        old=refs[f'llm_start_bo_s{seed}']
        assert new['history'][0]==old['history'][0]
        divergence=next((q for q,(a,b) in enumerate(zip(new['history'],old['history']),1)
                         if a['candidate_id']!=b['candidate_id']),None)
        for q in range(2,(divergence or 8)+1):
            a,b=new['decisions'][q-1],old['decisions'][q-1]
            assert a['fit_key']==b['fit_key']
            assert a['decision']['checkpoint_sha256']==b['decision']['checkpoint_sha256']
        pairs.append(dict(seed=seed,initial_id=m['handoff']['initial_id'],initial_yield=m['handoff']['initial_yield'],
            first_divergence=divergence,control_SearchAUC8=m['llm_start_bo']['SearchAUC8'],
            handoff_SearchAUC8=m['handoff']['SearchAUC8'],
            increment_SearchAUC8=m['handoff']['SearchAUC8']-m['llm_start_bo']['SearchAUC8'],
            increment_final_best=m['handoff']['final_best']-m['llm_start_bo']['final_best'],
            versus_random_SearchAUC8=m['handoff']['SearchAUC8']-m['bo']['SearchAUC8'],
            handoff_first95=m['handoff']['first95'],control_first95=m['llm_start_bo']['first95']))
    summary=[]
    for method in ('bo','llm_start_bo','event','handoff'):
        rs=[r for r in metrics if r['method']==method]
        summary.append(dict(method=method,n=len(rs),distinct_initial_points=len({r['initial_id'] for r in rs}),
            initial_yield_mean=mean(r['initial_yield'] for r in rs),SearchAUC8_mean=mean(r['SearchAUC8'] for r in rs),
            SearchAUC8_sd=stdev(r['SearchAUC8'] for r in rs),final_best_mean=mean(r['final_best'] for r in rs),
            target_hits=sum(r['hit95'] for r in rs),capped_hit_mean=mean(r['capped_hitting_index'] for r in rs)))
    for name,rows in (('metrics.csv',metrics),('rounds.csv',rounds),('paired.csv',pairs),
                      ('summary.csv',summary),('knowledge_feedback.csv',feedback_rows),('refresh_calls.csv',calls)):
        csv_save(name,rows)
    new_rounds=[r for r in rounds if r['method']=='handoff' and r['query']>1]
    audit=dict(status='PASS',protocol_sha256=p['protocol_sha256'],new_trajectories=10,
        trajectory_observations=80,reused_initial_observations=10,new_prospective_queries=70,
        frozen_reference_trajectories=30,comparison_trajectories=40,comparison_observations=320,
        paired_initial_points_and_knowledge_exact=True,all_commit_chains_and_actual_labels_verified=True,
        all_screen_arrays_and_decisions_replayed=True,all_acquisition_budgets_verified=True,
        terminal_native_BO_choices_verified=10,knowledge_changed_choices=sum(bool(r['knowledge_changed_choice']) for r in new_rounds),
        KF_updates=len(feedback_rows),new_pauses=sum(e['permission_before']!='paused' and e['permission_after']=='paused' for e in feedback_rows),
        fresh_refresh_calls=len(calls),refresh_proposals=sum(bool(c['proposed_id']) for c in calls),
        accepted_refresh_proposals=sum(c['accepted'] for c in calls),
        refreshes_changing_current_choice=sum(c['refresh_changed_current_choice'] for c in calls),
        all_original_artifacts_unchanged=True,operator_compiles_in_loop=0,new_queries_in_audit=0)
    study.check(True)
    study.write(study.REPORT/'audit.json',audit,replace=True)
    names={'bo':'Random first point + pure qLogEI','llm_start_bo':'Same LLM first point + pure qLogEI',
        'event':'Previous sparse-event policy','handoff':'Gradual LLM-to-BO handoff'}
    lines=['# Case 1: completed gradual LLM-to-BO handoff test','',
        'Ten new trajectories inherit their exact first point, outcome and initial knowledge '
        'from the previous ten sealed LLM replies. Seven subsequent prospective choices per '
        'trajectory give70 new oracle selections. Each method is evaluated at the same '
        'eight-observation budget, including the inherited paid first observation. '
        'The other30 trajectories are frozen references, not new runs.','',
        '| Method | Mean SearchAUC8 | Mean final best (%) | Hits >=85.09% by query8 |',
        '|---|---|---|---|']
    for row in summary:
        lines.append(f"| {names[row['method']]} | {row['SearchAUC8_mean']:.2f} | {row['final_best_mean']:.2f} | {row['target_hits']}/10 |")
    gain=mean(r['increment_SearchAUC8'] for r in pairs)
    terminal=mean(r['increment_final_best'] for r in pairs)
    win=sum(r['increment_SearchAUC8']>1e-10 for r in pairs)
    loss=sum(r['increment_SearchAUC8']<-1e-10 for r in pairs)
    lines.extend(['',f'At exact shared LLM initializations, the mean handoff increment is '
        f'{gain:+.4f} SearchAUC8 and {terminal:+.4f} final-yield percentage points. '
        f'Paired SearchAUC8 wins/losses/ties are {win}/{loss}/{10-win-loss}. '
        'The random-start comparison includes an initialization-policy difference; the '
        'exact-LLM-start comparison isolates post-initial control.','',
        f"The controller changes {audit['knowledge_changed_choices']} of70 choices relative to "
        f"the global qLogEI maximum at its own actual prefix. There are {len(feedback_rows)} "
        f"prospective per-claim KF updates, {audit['new_pauses']} pauses and {len(calls)} fresh "
        f"LLM refreshes; {audit['refresh_proposals']} refreshes propose a point and "
        f"{audit['accepted_refresh_proposals']} proposals pass all reviews. All ten last choices "
        'use exactly native full-pool qLogEI and its original tie rule. A shared decision '
        'rule at the final query does not imply the same final point after different histories.','',
        'After the LLM first point, the authority ceiling is (8-query)/7. Cached eligible '
        'knowledge preferences act inside a qLogEI budget of log(4) times the time ceiling '
        'and current eligible KF confidence factor. The ceiling contracts deterministically; '
        'actual confidence and recommendation permissions can reduce the allowance further. '
        'Refresh never restarts the clock. A favour claim must have improvement probability '
        'at least0.5; paused claims cannot recommend. The GP mean, variance and qLogEI are '
        'unchanged. Relative-Brier feedback scores only actually executed points.','',
        'The Mixed GP is the same50% prior/25 percentage-point SD recipe as the controls. '
        'It conditions on every observation and first fits hyperparameters after four '
        'observations. MC512 seeds, numeric fit seeds, all4,692 inputs and feature-based '
        'tie priorities are unchanged. Six persistent single-thread CPU workers warm '
        'operators once, with zero in-loop JIT compilations.','',
        'SearchAUC8 is the mean of eight post-query cumulative-best values. The good-yield '
        'threshold is frozen and evaluator-only. Unreached times remain missing;9 is only '
        'the capped-index comparison penalty. All outcomes, including negative trajectories, '
        'are included. No transition parameter was tuned after new prospective outcomes.','',
        'There are five distinct physical initial conditions across the ten frozen LLM '
        'replies, with one repeated six times. This remains a small response-weighted '
        'Case1 comparison with fixed later tie/numerical seeds. It does not test a new '
        'initialization prompt, prove cross-task stability, or separately identify the '
        'increment of KF versus the time schedule, preference rule or refresh mechanism.','',
        'The protocol contains a zero-outcome prompt-clarification amendment: rejected '
        'fresh proposals retain the current reviewed cached-knowledge choice rather than '
        'necessarily reverting to the global maximum. The original protocol and prompt '
        'snapshot are retained; no prospective query or fresh LLM call preceded that '
        'clarification. The control algorithm was unchanged.','',
        'The controller-only mechanism notes describe the earlier pre-study stage. This '
        'completed report supersedes their performance-status statements. See paired.csv, '
        'rounds.csv, knowledge_feedback.csv and refresh_calls.csv for individual paths and '
        'mechanisms. The run archive retains every pre-outcome commit, full GP/screen array, '
        'checkpoint and fresh responder receipt.'])
    native_path=study.REPORT/'fresh_native_check.json'
    if native_path.exists():
        native=study.read(native_path)
        assert native['status']=='PASS'
        lines.extend(['','## Fresh native numerical reconstruction','',
            f"An independent native constructor performs {native['new_GP_fits']} fresh GP fits "
            f"and checks all {native['prefixes']} new decision prefixes without using old fitted "
            'models or score caches for computation. All GP means, variances and qLogEI '
            f"scores match with maximum absolute errors {native['maximum_mean_error']}, "
            f"{native['maximum_variance_error']} and {native['maximum_logei_error']}. "
            'Every selected candidate passes its declared native acquisition budget and '
            'all ten last choices match the native full-pool BO maximum. '
            f"Native fitting emitted warnings on {native['prefixes_with_native_fitting_warnings']} "
            'prefixes; their recalculated results also match. No new LLM or oracle query '
            'is used in this numerical audit.','',
            f"The paired SearchAUC8 effect is positive in {win} trajectories, negative in {loss}, "
            f"and tied in {10-win-loss}. Mean final-best yield decreases despite the positive "
            'SearchAUC8 increment. The registered handoff rule should therefore be reported '
            'as a partial early-search gain relative to the matched LLM-start control, '
            'not as superiority to the frozen random-start pure BO or a cross-task result.'])
    (study.REPORT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(audit=audit,summary=summary,paired_SearchAUC8_increment=gain)))


if __name__=='__main__':
    analyze()
