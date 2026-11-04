"""Replay all prospective V14 decisions, then compare unchanged controls."""
from collections import Counter
from pathlib import Path
import csv,json
import numpy as np
from cold_start_v13 import validation as old_validation,control as old
from cold_start_v13_seeds import study as ms
from knowledge_rank_v9 import engine
from reasoning_memory_v12.llm import compiled_extractor
from . import study,transport,policy,prompts,controller,VARIANT


def csv_write(path,rows):
    if not rows:return
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def replay(root,task,seed,state,protocol):
    folder=study.run_path(root,task,seed);pool=study.frozen.pool_for(root,task)
    initial=transport.read(folder/'initial_commit.json')
    assert initial['commit_sha256']==old.digest({k:v for k,v in initial.items() if k!='commit_sha256'})
    assert initial['initial']==0 and not initial['initial_ids'] and not initial['history']
    current=study.initial_state(task,seed,protocol,initial);rows=[]
    for q,observation in enumerate(state['history'],1):
        commit=transport.read(folder/f'rounds/{q:02d}_commit.json')
        with np.load(folder/f'rounds/{q:02d}_GP.npz',allow_pickle=False) as z:
            ctx=dict(stats={k:z[k].copy() for k in ('mu_y','sd_y','sd_predictive_y')},
                **{k:z[k].copy() for k in ('base_logei','available','tie_priority')})
        assert engine.gp_hash(ctx)==commit['GP_sha256']
        np.testing.assert_array_equal(ctx['available'],~np.isin(pool.ids,[r['candidate_id'] for r in current['history']]))
        np.testing.assert_array_equal(ctx['tie_priority'],study.PRIORITIES(pool,q,seed if q==1 else 0))
        fit=transport.read(folder/f'rounds/{q:02d}_fit.json')
        assert fit['acquisition']['observed_count']==q-1
        if q==1:
            assert not fit['fitted_GP'] and np.isnan(ctx['stats']['mu_y']).all()
        else:
            assert fit['fitted_GP'] and all(np.isfinite(x).all() for x in ctx['stats'].values())
        ordered,display=engine.candidates(pool,ctx,q);ids=[pool.ids[int(j)] for j in display]
        assert commit['display_ids']==ids and commit['ordered_ids']==[pool.ids[int(j)] for j in ordered]
        refresh=policy.edit_event(pool,current,ordered,q);assert refresh==commit['refresh']
        if commit['memory']:
            assert refresh['allowed']
            raw=old_validation._request_result(root,commit['memory'],prompts.MEMORY_SYSTEM,
                prompts.memory_public(pool,current,ids,q,refresh))
            plan=policy.apply_edits(raw,pool,current,q);assert plan==commit['memory']['plan']
            notes=plan['notes']
        else:
            assert not refresh['allowed'];notes=current['notes']
        assert notes==commit['edited_notes']
        public=prompts.reasoning_public(root,pool,current,notes,ids,q)
        raw=old_validation._request_result(root,commit['reasoning'],prompts.REASONING_SYSTEM,public)
        packet=policy.validate_decision(root,pool,notes,current,ctx,ids,q,raw)
        assert packet==commit['decision']
        current=study.feedback(pool,current,commit,observation)
        channel=current['decision_channel'];events=current['decisions'][-1]['feedback']
        assert channel['calibration']['updates']==max(0,q-1)
        assert len(current['review_queries'])+len(channel['review_queries'])<=2
        assert len(current['edit_calls'])<=4 and len(channel['learning_queries'])<=2
        assert all(not (e['recommendation_before']!='allowed' and e['recommendation_after']=='allowed' and not e['improved']) for e in events)
        d=packet['deployment'];card=current['decisions'][-1]['card_feedback']
        rows.append(dict(task=task,seed=seed,query=q,proposed=packet['proposed_id'],executed=packet['selected_id'],
            BO_reference=packet['BO_reference_id'],objective=observation['raw_value'],best=max(h['raw_value'] for h in current['history']),
            proposal_accepted=d['accepted'],gate_reason=d['reason'],qLogEI_gap=d['logEI_gap'],opportunity_budget=d['budget'],
            forecast_count=len(packet['all_probabilities']) if q>1 else 0,card_KF_updates=channel['calibration']['updates'],
            card_reliability=channel['calibration']['reliability'],card_Brier_gain=card.get('calibration',{}).get('Brier_gain'),
            local_suppression=card.get('local_suppression',False),card_review=bool(d['review']),
            note_tests_credited=sum(e['direction_test'] for e in events),
            note_tests_calibration_only=sum(e['declared_use'] in controller.TEST_USES and not e['direction_test'] for e in events),
            memory_refresh=bool(commit['memory']),refresh_reasons=';'.join(refresh['reasons'])))
    assert current==state
    assert transport.read(folder/'runtime_counters.json')['operator_compiles_in_loop']==0
    return rows


def audit(root):
    root=Path(root).resolve();protocol=study.check(root,True);compiled_extractor(str(root))
    out=root/'experiments/reports'/VARIANT;out.mkdir(parents=True,exist_ok=True)
    rows=[];metrics=[];missing=[];trajectories=[]
    for task,seed in study.JOBS:
        sf=study.run_path(root,task,seed)/'state.json'
        if not sf.exists() or transport.read(sf)['status']!='COMPLETE':
            missing.append(dict(task=task,seed=seed));continue
        state=transport.read(sf);rr=replay(root,task,seed,state,protocol);rows+=rr;trajectories.append(state)
        current=old_validation.metrics(state)
        baselines=[b for b in protocol['baselines'] if (b['task'],b['seed'])==(task,seed)]
        for b in baselines:
            metrics.append(dict(task=task,seed=seed,method=b['method'],**{k:b[k] for k in current}))
        metrics.append(dict(task=task,seed=seed,method='v14',**current))
    agents=[];attempts=Counter()
    for path in (transport.base(root)/'requests').glob('*/*/attempt*/response_receipt.json'):
        receipt=transport.read(path);assignment=transport.read(path.parent/'assignment.json')
        req=transport.frozen.verify_request(path.parent/'request.json')
        invocation=transport.read(path.parent/'invocation.json')
        assert receipt['raw_sha256']==transport.file_sha(path.parent/'response.raw.json')
        assert receipt['assignment_sha256']==transport.file_sha(path.parent/'assignment.json')
        assert receipt['agent_id']==assignment['agent_id']=='/root/'+invocation['task_name']
        assert req['fork_turns']=='none' and req['model']==study.MODEL and req['effort']==study.EFFORT
        vp=path.parent/'validation.json'
        if vp.exists():attempts[transport.read(vp)['status']]+=1
        agents.append(receipt['agent_id'])
    assert len(agents)==len(set(agents))
    comparison=[]
    for task,seed in study.JOBS:
        arms={r['method']:r for r in metrics if (r['task'],r['seed'])==(task,seed)}
        if 'v14' not in arms:continue
        v=arms['v14'];comparison.append(dict(task=task,seed=seed,
            **{a+'_SearchAUC':arms[a]['SearchAUC'] for a in ('bo','v13','reasoning_bo','v14')},
            V14_minus_V13=v['SearchAUC']-arms['v13']['SearchAUC'],V14_minus_BO=v['SearchAUC']-arms['bo']['SearchAUC'],
            **{a+'_final_best':arms[a]['final_best'] for a in ('bo','v13','reasoning_bo','v14')},
            **{a+'_first_final_best_query':arms[a]['first_final_best_query'] for a in ('bo','v13','reasoning_bo','v14')}))
    csv_write(out/'metrics.csv',metrics);csv_write(out/'comparison.csv',comparison);csv_write(out/'rounds.csv',rows)
    complete=not missing
    result=dict(status='PASS_COMPLETE' if complete else 'PARTIAL',trajectories=len(trajectories),queries_replayed=len(rows),
        expected_trajectories=6,expected_queries=120,missing=missing,unique_fresh_agents=len(agents),schema_attempts=dict(attempts),
        no_GP_change=True,all_frozen_controls_unchanged=True,operator_compiles_in_loop=0,
        correct_failure_positive_restorations=0,oracle_calls_in_audit=0,
        protocol_sha256=protocol['protocol_sha256'],candidate_card_KF_updates=sum(t['decision_channel']['calibration']['updates'] for t in trajectories),
        accepted_proposals=sum(r['proposal_accepted'] for r in rows),gate_reasons=dict(Counter(r['gate_reason'] for r in rows)))
    if complete:assert result['queries_replayed']==120 and result['candidate_card_KF_updates']==114
    study.check(root,True)
    (out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['V14 targeted diagnostic validation',result['status'],
        '0 initial observations +20 paid queries; frozen GP/pools; old controls not rerun.',
        'Two known failures and four known gains selected prospectively. Not an unbiased or multiseed stability estimate.','',
        'Task | seed | BO AUC | V13 AUC | Reasoning BO AUC | V14 AUC | V14-V13 | V14-BO']
    for r in comparison:
        lines.append(f"{r['task']} | {r['seed']} | {r['bo_SearchAUC']:.5f} | {r['v13_SearchAUC']:.5f} | {r['reasoning_bo_SearchAUC']:.5f} | {r['v14_SearchAUC']:.5f} | {r['V14_minus_V13']:+.5f} | {r['V14_minus_BO']:+.5f}")
    lines+=['','Final best / first query reaching final best are in comparison.csv and metrics.csv.',
            'Every executed query afterq1 has a forecast/KF update, including GP fallbacks. Unexecuted proposals remain unlabelled.',
            'Technical replay validity does not establish optimization advantage.',json.dumps(result,ensure_ascii=False)]
    (out/'report.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if complete:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
        styles={'bo':('Pure BO','#29568b','-'),'v13':('V13','#888888',':'),
                'reasoning_bo':('Reasoning BO','#24846b','--'),'v14':('V14','#c56321','-')}
        for ax,(task,seed),state in zip(axes.flat,study.JOBS,trajectories):
            for arm,(label,color,ls) in styles.items():
                hist=state['history'] if arm=='v14' else transport.read(study.baseline_path(root,task,seed,arm)/'state.json')['history']
                ax.step(range(1,21),np.maximum.accumulate([h['raw_value'] for h in hist]),where='post',label=label,color=color,ls=ls,lw=1.8)
            ax.set_title(f'{study.frozen.SHORT[task]} / seed={seed}');ax.set_xlabel('Paid query');ax.set_ylabel('Cumulative best objective')
            ax.set_xlim(1,20);ax.set_xticks([1,5,10,15,20]);ax.grid(alpha=.2)
        handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=4,frameon=False)
        fig.savefig(out/'cumulative_best.png',dpi=170);plt.close(fig)
    return result
