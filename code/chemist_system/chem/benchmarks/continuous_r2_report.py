"""Independent execution audit and read-only comparison exports for R2."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import csv
import json
import numpy as np
from . import continuous_r2_study as s
from . import continuous_r2_engine as e
from . import continuous_r2_llm as llm
from . import revised_continuous_engine as r1
from . import continuous_benchmark_data as data
from .suzuki_domain import digest

CRITICAL=('history','decisions','knowledge','initial_knowledge','banks','region_trust','calls',
          'refresh_checks','interaction_tests','interaction_checks','geometry_ledger','bank_initializations',
          'pending','active_suspended','promotions','proposal_archive','subset_evidence','bootstrap_receipts')
LABELS={'Pure_BO':'纯BO（qLogEI）','New_Agent':'旧新方案','New_Agent_R1':'R1',
        'New_Agent_R2':'R2','New_Agent_R2_Subset':'R2＋B4子集恢复'}


def csv_write(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def assert_equal(a,b,label):
    assert digest(a)==digest(b),label


def no_refit(*args,**kwargs):raise RuntimeError('R2 audit refuses GP fitting.')


def independent_KF(event):
    m=event['measurement'];prior=event['bank_prior']
    if prior is None:
        assert m['actual_deployed_correction_y']==0. and m['actual_paired_SE_gain']==0.
        return 0.,False
    H=m['raw_region_shift_y'] if event['gate_open'] else 0.
    P=prior['variance'];a=prior['mean'];R=m['prequery_predictive_sd_y']**2+.0009
    innovation=event['y']-m['prequery_mu_y']-a*H
    expected=float(np.clip(a+P*H/(P*H*H+R)*innovation,0,1))
    gain=(event['y']-m['prequery_mu_y'])**2-(event['y']-m['prequery_mu_y']-a*H)**2
    variance=P*R/(P*H*H+R)
    error=max(abs(expected-event['bank_after']['mean']),abs(variance-event['bank_after']['variance']),
              abs(innovation-m['innovation']),abs(gain-m['actual_paired_SE_gain']),
              abs(a*H-m['actual_deployed_correction_y']))
    assert error<1e-12
    if gain < -1e-12:assert expected<=a+1e-12
    return error,gain < -1e-12


def audit_one(job):
    task,method=job;s.old.warm();pool=s.frozen.pool(task);cache=e.Cache(pool)
    state=data.read(s.ROOT/'runs'/(task+'_'+method)/'state.json')
    assert state['status']=='COMPLETE' and len(state['history'])==20 and len(state['decisions'])==16
    protocol=s.check();replay=s.initial(task,method,protocol);folder=s.ROOT/'runs'/state['run_id']/'rounds'
    original=data.read(s.parent.ROOT/'runs'/(task+'_New_Agent_R1')/'state.json')
    assert state['history'][:4]==s.frozen.initial(task)
    assert len({h['candidate_id'] for h in state['history']})==20
    assert state['history'][:4+e.BOOTSTRAP[task]]==original['history'][:4+e.BOOTSTRAP[task]]
    maximum=0.;kf_max=0.;negative=0;contexts=set();pending_checks=0;subset_applied=0
    with patch.object(s.old,'fit_gpytorch_mll',no_refit):
        for t,event in enumerate(state['decisions'],1):
            cp=folder/f'{t:02d}_committed.json';c=data.read(cp)
            assert data.filehash(cp)==event['commit_sha256']
            assert data.filehash(folder/f'{t:02d}_candidates.npz')==c['candidate_table_sha256']
            assert c['protocol_sha256']==protocol['protocol_sha256']
            for key in CRITICAL:assert_equal(replay[key],c['state_before_decision'][key],f'{task}/{method}/{t}/before/{key}')
            assert c['history']==state['history'][:t+3]
            assert c['history_sha256']==digest(c['history'])
            with np.load(folder/f'{t:02d}_candidates.npz') as f:arrays={k:f[k].copy() for k in f.files}
            selected=cache.ix[c['selected_id']]
            assert arrays['available'][selected] and arrays['eligible'][selected]
            assert c['base_opportunity_loss']<=c['budget']+1e-12
            if t<=e.BOOTSTRAP[task]:
                source=s.parent.ROOT/'runs'/original['run_id']/'rounds'
                assert c['bootstrap_replay']['source_commit_sha256']==data.filehash(source/f'{t:02d}_committed.json')
                assert data.filehash(folder/f'{t:02d}_candidates.npz')==data.filehash(source/f'{t:02d}_candidates.npz')
                expected=deepcopy(original['decisions'][t-1])
                expected['bootstrap_replay']=c['bootstrap_replay'];expected['commit_sha256']=data.filehash(cp)
                assert_equal(event,expected,f'{task}/{t}/bootstrap_event')
                replay['knowledge']=deepcopy(event['knowledge']);replay['banks'][event['bank_key']]=deepcopy(event['bank_after'])
                replay['region_trust']=deepcopy(event['region_trust_after'])
                for key,source_key,index in [('calls','calls','iteration'),('refresh_checks','refresh_checks','iteration'),
                                             ('interaction_checks','interaction_checks','iteration'),('interaction_tests','interaction_tests','query')]:
                    replay[key].extend(deepcopy(v) for v in original[source_key] if v[index]==t)
                replay['bootstrap_receipts'].append(c['bootstrap_replay']);e.register_feedback(replay,event)
            else:
                def region_provider(own,d,reasons,quota):
                    saved=c['refresh'];assert saved is not None
                    context=data.read(s.ROOT/'requests'/saved['context_sha256']/'context.json')
                    assert context['observed_prefix']==own['history']
                    assert context['system_sha256']==digest(llm.REGION_SYSTEM)
                    assert_equal(context['public'],llm.region_payload(pool,own,t,d,reasons,quota),'region context')
                    answer=saved['answer'];validated=llm.previous.validate_region(answer['raw'],pool,own)
                    assert_equal(validated,answer['plan'],'region schema/grounded support')
                    contexts.add(saved['context_sha256']);return deepcopy(answer),saved['context_sha256']
                def joint_provider(own,d,prepared):
                    saved=c['interaction'];assert saved is not None
                    context=data.read(s.ROOT/'requests'/saved['context_sha256']/'context.json')
                    assert context['observed_prefix']==own['history']
                    assert context['system_sha256']==digest(llm.JOINT_SYSTEM)
                    assert_equal(context['public'],llm.joint_payload(pool,own,t,d,prepared),'joint context')
                    answer=saved['answer'];validated=llm.validate_joint(answer['raw'],prepared,own,llm.reference(pool,d))
                    assert_equal(validated,answer['plan'],'optional joint validation')
                    contexts.add(saved['context_sha256']);return deepcopy(answer),saved['context_sha256']
                plan=s.decide(replay,cache,t,region_provider,joint_provider);d=plan['d']
                assert pool['ids'][d['selected']]==c['selected_id']
                assert pool['ids'][d['normal_proposal']]==c['normal_proposal_id']
                assert pool['ids'][d['reference']]==c['reference_id']
                for key in ['promotion','refresh','refresh_check','interaction','interaction_check','execution_mode']:
                    assert_equal(plan[key],c[key],f'plan {key}')
                for key in ['knowledge','admission','region','retained','bank_key','bank_before','bank_prior','amplitude','gate_open']:
                    value=replay['knowledge'] if key=='knowledge' else d['bank'] if key=='bank_before' else d['prior'] if key=='bank_prior' else d[key]
                    assert_equal(value,c[key],f'decision {key}')
                comparisons=[('base_qLogEI',d['base']),('guided_qLogEI',d['score']),('raw_shift_std',d['raw_shift_std']),
                    ('deployed_shift_std',d['shift_std']),('mu_y',d['stats']['mu_y']),('sd_y',d['stats']['sd_y']),
                    ('sd_predictive_y',d['stats']['sd_predictive_y'])]
                frozen=plan['pending_forecast']
                comparisons.append(('pending_raw_shift_std',frozen['raw_shift_std'] if frozen else np.zeros(len(pool['ids']))))
                for name,value in comparisons:
                    error=float(np.max(abs(arrays[name]-value)));maximum=max(maximum,error)
                    assert error<1e-9,(task,method,t,name,error)
                assert np.array_equal(arrays['eligible'],d['eligible']) and np.array_equal(arrays['available'],d['available'])
                if c['refresh']:
                    assert c['selected_id']==c['reference_id'] and not c['gate_open'] and c['amplitude']==0.
                    assert c['knowledge']==c['state_before_decision']['knowledge'] or c['promotion'] is not None
                    inheritance=replay['pending']['inheritance']
                    if not inheritance.get('existing_geometry'):
                        bank=replay['pending']['initial_bank']
                        if bank:
                            assert bank['mean']<=.5 and bank['variance']>=.25
                            newmask=e.spec_mask(pool,replay['pending']['region'])
                            source_means=[.5];source_variances=[.25]
                            for row in inheritance['sources']:
                                source_key=row['geometry_key'];entry=replay['geometry_ledger'][source_key]
                                assert entry['negative_actual_count']>0
                                oldmask=e.spec_mask(pool,entry['region']);overlap=float(np.sum(newmask&oldmask)/min(newmask.sum(),oldmask.sum()))
                                assert abs(overlap-row['overlap_of_smaller_set'])<1e-12 and overlap>=.5
                                prior=r1.forecast(replay['banks'][source_key],t)
                                source_means.append(prior['mean']);source_variances.append(prior['variance'])
                            assert bank['mean']==min(source_means) and bank['variance']==max(source_variances)
                if frozen:
                    assert c['selected_id']==c['reference_id'] and not c['gate_open']
                    pending_checks+=1
                expected=s.feedback(replay,cache,plan,t,state['history'][t+3]);expected['commit_sha256']=data.filehash(cp)
                assert_equal(expected,event,f'{task}/{method}/{t}/feedback')
                if event['pending_validation']:
                    value=event['pending_validation'];mu=value['GP_mean_y'];shift=value['frozen_probe_prediction_y']-mu
                    gain=(event['y']-mu)**2-(event['y']-mu-shift)**2
                    assert abs(gain-value['paired_SE_gain'])<1e-12
                    assert event['measurement']['actual_paired_SE_gain']==0.
                if d['subset_recovery']['applied']:
                    subset_applied+=1
                    assert method==e.METHODS[1] and d['subset_recovery']['BO_shadow_count']>=2
                    assert d['subset_recovery']['sum_gain']>0 and d['subset_recovery']['latest_gain']>0
                    assert d['subset_recovery']['rank_stable']
            error,neg=independent_KF(event);kf_max=max(kf_max,error);negative+=int(neg)
            with np.load(data.ROOT/task/'oracle.npz') as f:
                assert abs(float(f['raw_values'][selected])-event['raw_value'])<1e-9
            replay['history'].append(deepcopy(state['history'][t+3]));replay['decisions'].append(deepcopy(event))
            cache.release()
    for key in CRITICAL:assert_equal(replay[key],state[key],f'{task}/{method}/final/{key}')
    assert len(state['calls'])<=3 and sum(c['iteration']<=8 for c in state['calls'])<=2
    assert all(b['iteration']-a['iteration']>=2 for a,b in zip(state['calls'],state['calls'][1:]))
    assert all(c['iteration']<=14 for c in state['calls'])
    assert state['operator_compiles_in_loop']==0 and cache.counters['fits']==0 and sum(s.old.JIT.values())==0
    return {'task':task,'method':method,'choices_checked':16,'protected_prefix_choices_checked':e.BOOTSTRAP[task],
        'fresh_choices_checked':16-e.BOOTSTRAP[task],'KF_updates_checked':16,'negative_gain_nonincrease_checks':negative,
        'max_prediction_or_acquisition_error':maximum,'max_independent_KF_error':kf_max,'GP_refits_in_audit':0,
        'fresh_LLM_contexts_checked':len(contexts),'pending_BO_checks':pending_checks,
        'promotions':[c['query'] for c in state['promotions']],
        'refresh_rounds':[c['iteration'] for c in state['calls']],
        'subset_recovery_activations':subset_applied,'operator_compiles_in_loop':0}


def audit():
    protocol=s.check();assert s.status()['complete']==4
    jobs=[(task,method) for task in s.TASKS for method in e.METHODS]
    with ProcessPoolExecutor(max_workers=4) as executor:checked=list(executor.map(audit_one,jobs))
    keep=data.read(s.OUT/'protected_snapshot.json');assert digest(keep)==protocol['protected_snapshot_sha256']
    for name,h in keep.items():assert data.filehash(name)==h,name
    amendment=data.read(s.OUT/'execution_amendment.json')
    for name,h in amendment['completed_subset_files_sha256'].items():assert data.filehash(name)==h,name
    receipt={'status':'PASS','completed_trajectories':4,'budgeted_added_choices_verified':64,
        'replayed_prefix_choices_verified':10,'fresh_choices_verified':54,'baselines_rerun':0,
        'protected_files_unchanged':len(keep),'completed_subset_files_unchanged_after_cache_fix':len(amendment['completed_subset_files_sha256']),
        'checks':checked,'HPLC_first_three_full_transitions_preserved':True,
        'execution_amendment_sha256':data.filehash(s.OUT/'execution_amendment.json'),
        'scope':'technical validation; single seed and protected-prefix continuation, not statistical superiority'}
    data.write(s.OUT/'validation_receipt.json',receipt);return receipt


def metric(state,pool):
    y=np.array([h['raw_value'] for h in state['history']]);best=np.maximum.accumulate(y)
    events=state['decisions']
    return {'task':state['task'],'method':state['method'],'SearchAUC':float(best[4:].mean()),
        'final_best':float(best[-1]),'first_final_best_total':int(np.flatnonzero(y==best[-1])[0]+1),
        'initial_best':float(best[3]),'refreshes':len(state['calls']),'controlled_tests':len(state['interaction_tests']),
        'gate_exceptions':0,'nonbase_choices':sum(d['selected_id']!=d['reference_id'] for d in events),
        'unit':pool['objective_unit'],'result_source':'R2_only_new_run' if state['method']==e.METHODS[0] else 'R2_independent_subset_ablation',
        'protected_prefix_queries':e.BOOTSTRAP[state['task']],'pending_checks':sum(d.get('pending_validation') is not None for d in events),
        'pending_promotions':len(state['promotions']),'subset_activations':sum(d.get('subset_recovery',{}).get('applied',False) for d in events)}


def report():
    protocol=s.check();validation=data.read(s.OUT/'validation_receipt.json');assert validation['status']=='PASS'
    old=data.read(s.parent.OUT/'results.json')['metrics'];metrics=deepcopy(old);new=[];curves={};events_rows=[]
    for state in s.states():
        task=state['task'];pool=s.frozen.pool(task);row=metric(state,pool);metrics.append(row);new.append(row)
        folder=s.ROOT/'runs'/state['run_id']/'rounds';target=s.OUT/task/state['method'];trajectory=[];history=[]
        with np.load(data.ROOT/task/'oracle.npz') as f:truth=f['raw_values'].copy()
        for i,observation in enumerate(state['history'],1):
            j=pool['ids'].index(observation['candidate_id'])
            h={'total_experiment':i,'query_round':max(0,i-4),'candidate_id':observation['candidate_id'],
               **pool['points'][j],'objective':observation['raw_value'],
               'cumulative_best':max(x['raw_value'] for x in state['history'][:i]),
               'phase':'initial4' if i<=4 else 'protected_R1_prefix' if i<=4+e.BOOTSTRAP[task] else 'R2_fresh_decision'}
            if 'e_factor' in observation:h['e_factor']=observation['e_factor']
            history.append(h)
        for t,event in enumerate(state['decisions'],1):
            c=data.read(folder/f'{t:02d}_committed.json');m=event['measurement'];scale=pool['objective_scale']
            j=pool['ids'].index(event['selected_id']);ref=pool['ids'].index(event['reference_id'])
            tr={'query_round':t,'total_experiment':t+4,'candidate_id':event['selected_id'],**pool['points'][j],
                'objective':event['raw_value'],'cumulative_best':event['incumbent_after']*scale,
                'mode':event['execution_mode'],'protected_prefix':bool(event.get('bootstrap_replay')),
                'same_prefix_BO_id':event['reference_id'],'BO_objective_analysis_only':float(truth[ref]),
                'GP_mean':m['prequery_mu_y']*scale,'GP_predictive_SD':m['prequery_predictive_sd_y']*scale,
                'deployed_correction':m['actual_deployed_correction_y']*scale,'deployed_mean':m['actual_deployed_mean_y']*scale,
                'coefficient_before':m['coefficient_before'],'coefficient_after':m['coefficient_after'],
                'actual_paired_SE_gain_raw_squared':m['actual_paired_SE_gain']*scale*scale,
                'gate_open':event['gate_open'],'B4_revoked_ids':','.join(event['diagnosis']['revoked_ids']),
                'base_opportunity_loss':event['base_opportunity_loss'],'budget':event['budget'],
                'refresh_proposed':any(x['iteration']==t for x in state['calls']),
                'promotion_applied':bool(event.get('promotion')),
                'pending_status_after':event.get('pending_after',{}).get('status') if event.get('pending_after') else '',
                'pending_shadow_gain_y_squared':event['pending_validation']['paired_SE_gain'] if event.get('pending_validation') else '',
                'subset_recovered':event.get('subset_recovery',{}).get('applied',False)}
            trajectory.append(tr);events_rows.append({'task':task,'method':state['method'],**tr})
            with np.load(folder/f'{t:02d}_candidates.npz') as f:arrays={k:f[k].copy() for k in f.files}
            candidates=[];pending=c.get('pending_forecast');alpha=event['amplitude'];std=c['GP_scale_y']
            for k,key in enumerate(pool['ids']):
                candidate={'candidate_id':key,**pool['points'][k],'true_objective_analysis_only':float(truth[k]),
                    'GP_mean':float(arrays['mu_y'][k])*scale,'GP_latent_SD':float(arrays['sd_y'][k])*scale,
                    'GP_predictive_SD':float(arrays['sd_predictive_y'][k])*scale,
                    'base_qLogEI':float(arrays['base_qLogEI'][k]),'deployed_qLogEI':float(arrays['guided_qLogEI'][k]),
                    'available':bool(arrays['available'][k]),'inside_original_budget_gate':bool(arrays['eligible'][k]),
                    'base_opportunity_loss':float(arrays['base_qLogEI'][ref]-arrays['base_qLogEI'][k]),
                    'raw_region_shift':float(arrays['raw_shift_std'][k])*std*scale,
                    'deployed_mean':float(arrays['mu_y'][k]+arrays['deployed_shift_std'][k]*std)*scale,
                    'deployed_alpha':alpha,'pending_shadow_prediction':
                        float(arrays['mu_y'][k]+pending['shadow_probe_coefficient']*arrays['pending_raw_shift_std'][k]*std)*scale if pending else '',
                    'selected':key==event['selected_id'],'BO_reference':key==event['reference_id'],
                    'normal_proposal':key==event['normal_proposal_id'],
                    'in_joint_pool':bool(arrays['interaction_pool'][k]),'tie_priority':int(arrays['tie_priority'][k])}
                candidates.append(candidate)
            csv_write(target/'candidate_tables'/f'{t:02d}_candidates.csv',candidates)
        csv_write(target/'trajectory.csv',trajectory);csv_write(target/'all20_history.csv',history)
        data.write(target/'proposal_history.json',{'calls':state['calls'],'promotions':state['promotions'],
            'proposal_archive':state['proposal_archive'],'pending_final':state['pending'],
            'bank_initializations':state['bank_initializations'],'geometry_ledger':state['geometry_ledger']})
    for task in s.TASKS:
        curves[task]={}
        for method in LABELS:
            root=s.ROOT if method in e.METHODS else s.parent.ROOT if method=='New_Agent_R1' else s.frozen.ROOT
            state=data.read(root/'runs'/(task+'_'+method)/'state.json')
            y=np.array([h['raw_value'] for h in state['history']])
            curves[task][method]={'observed':y.tolist(),'best':np.maximum.accumulate(y).tolist()}
    csv_write(s.OUT/'comparison.csv',metrics)
    selected=[r for r in metrics if r['method'] in LABELS]
    csv_write(s.OUT/'main_comparison.csv',selected);csv_write(s.OUT/'decision_events.csv',events_rows)
    data.write(s.OUT/'results.json',{'metrics':metrics,'new_metrics':new,'protocol_sha256':protocol['protocol_sha256']})
    data.write(s.OUT/'curves.json',curves);plot(curves)
    return {'new_metrics':new,'comparison_rows':len(metrics),'selected_comparison_rows':len(selected)}


def plot(curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.size':10})
    styles={'Pure_BO':('#555555','-',1.2),'New_Agent':('#d17a00','--',1.3),
            'New_Agent_R1':('#ce647a','--',1.3),'New_Agent_R2':('#d62728','-',2.1),
            'New_Agent_R2_Subset':('#6b45ae',':',1.8)}
    for task,methods in curves.items():
        fig,axes=plt.subplots(1,2,figsize=(12.6,3.65));x=np.arange(1,21)
        for method,values in methods.items():
            color,line,width=styles[method]
            label={'Pure_BO':'Pure BO','New_Agent':'Old new agent','New_Agent_R1':'R1',
                   'New_Agent_R2':'R2','New_Agent_R2_Subset':'R2 + subset ablation'}[method]
            for ax,key in zip(axes,['observed','best']):
                ax.plot(x,values[key],color=color,linestyle=line,linewidth=width,label=label,
                        marker='o' if method=='New_Agent_R2' else None,markersize=2.3)
        for ax,title in zip(axes,['Observed objective','Cumulative best']):
            ax.axvspan(.5,4.5,color='#efefef',zorder=-1)
            ax.grid(alpha=.22);ax.set_title(title);ax.set_xlabel('Total experiment (4 + 16)')
            ax.set_xticks([1,4,8,12,16,20]);ax.set_xlim(.5,20.5)
            ax.set_ylabel('STY (kg m^-3 h^-1)' if task=='SnAr' else 'Peak area (source units)')
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.015),ncol=5,fontsize=8)
        fig.tight_layout(rect=[0,.09,1,1])
        fig.savefig(s.OUT/(task+'_comparison.png'),dpi=180,bbox_inches='tight')
        fig.savefig(s.OUT/(task+'_figure.pdf'),bbox_inches='tight');plt.close(fig)


def inspect(task,method,t):
    state=data.read(s.ROOT/'runs'/(task+'_'+method)/'state.json')
    return {'task':task,'method':method,'decision':state['decisions'][t-1],
            'candidate_CSV':str((s.OUT/task/method/'candidate_tables'/f'{t:02d}_candidates.csv').resolve())}
