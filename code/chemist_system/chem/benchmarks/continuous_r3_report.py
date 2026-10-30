"""R3 independent forecast audit, branch accounting and frozen comparisons."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import csv
import json
import numpy as np
from . import continuous_r3_study as s
from . import continuous_r3_engine as e
from . import continuous_r3_llm as llm
from . import continuous_benchmark_data as data
from .suzuki_domain import digest

CRITICAL=('history','decisions','initial_knowledge','hypotheses','authorization_events','calls',
          'bootstrap_receipts','protected_legacy_state','legacy_context')
NAMES={'Pure_BO':'纯BO（qLogEI）','LLINBO_Justify':'LLINBO-Justify','ReasoningBO':'Reasoning BO',
       'New_Agent_R2':'R2','New_Agent_R3':'R3'}


def csv_write(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def same(a,b,label):assert digest(a)==digest(b),label
def no_refit(*args,**kwargs):raise RuntimeError('R3 audit refuses new GP fitting.')


def scalar_check(prior,measurement,mu,sd,y,feature,cap,shadow=False):
    a=prior['mean'];P=prior['variance'];R=sd**2+.0009
    K=P*feature/(P*feature**2+R);innovation=y-mu-a*feature
    coefficient=float(np.clip(a+K*innovation,0,cap));variance=P*R/(P*feature**2+R)
    gain=(y-mu)**2-(y-mu-a*feature)**2
    key='hypothetical_paired_SE_gain' if shadow else 'actual_paired_SE_gain'
    error=max(abs(measurement['coefficient_after']-coefficient),abs(measurement['innovation']-innovation),
              abs(measurement[key]-gain))
    assert error<1e-12
    if gain < -1e-12:assert coefficient<=a+1e-12
    return error,coefficient,variance


def independent_evidence(h,t):
    assert all(r['query']<t for r in h['shadow_records'])
    rows=[r for r in h['shadow_records'] if r['informative'] and r['query']>h['last_actual_failure_query']][-4:]
    score=sum(r['standardized_gain'] for r in rows);positives=sum(r['paired_SE_gain']>1e-12 for r in rows)
    required=2 if h['last_actual_failure_query']>=0 or h['inherited_actual_failure_ids'] else 1
    eligible=bool(len(rows)>=required and positives>=required and score>0 and rows[-1]['paired_SE_gain']>1e-12)
    return eligible,score,positives,required


def audit_task(task):
    s.base.warm();protocol=s.check();pool=s.original.pool(task);cache=e.Cache(pool)
    state=data.read(s.ROOT/'runs'/(task+'_'+e.METHOD)/'state.json')
    assert state['status']=='COMPLETE' and len(state['history'])==20 and len(state['decisions'])==16
    assert len({h['candidate_id'] for h in state['history']})==20
    replay=s.initial(task,protocol);folder=s.ROOT/'runs'/state['run_id']/'rounds'
    source=data.read(s.parent.ROOT/'runs'/(task+'_New_Agent_R2')/'state.json')
    assert state['history'][:4+e.BOOTSTRAP[task]]==source['history'][:4+e.BOOTSTRAP[task]]
    maximum=0.;kf_error=0.;local_updates=negative=checks=informative=0;contexts=set();outside=0
    with patch.object(s.base,'fit_gpytorch_mll',no_refit):
        for t,event in enumerate(state['decisions'],1):
            cp=folder/f'{t:02d}_committed.json';c=data.read(cp)
            assert data.filehash(cp)==event['commit_sha256']
            assert c['protocol_sha256']==protocol['protocol_sha256']
            assert data.filehash(folder/f'{t:02d}_candidates.npz')==c['candidate_table_sha256']
            for key in CRITICAL:same(replay[key],c['state_before_decision'][key],f'{task}/{t}/before/{key}')
            assert c['history']==state['history'][:t+3] and c['history_sha256']==digest(c['history'])
            with np.load(folder/f'{t:02d}_candidates.npz') as f:a={k:f[k].copy() for k in f.files}
            j=cache.ix[c['selected_id']]
            assert a['available'][j] and a['eligible'][j]
            if t<=e.BOOTSTRAP[task]:
                src=s.parent.ROOT/'runs'/(task+'_New_Agent_R2')/'rounds'
                old_c=data.read(src/f'{t:02d}_committed.json')
                after=data.read(src/f'{t+1:02d}_committed.json')['state_before_decision']
                receipt=c['protected_R2_replay']
                assert receipt['source_commit_sha256']==data.filehash(src/f'{t:02d}_committed.json')
                assert data.filehash(folder/f'{t:02d}_candidates.npz')==data.filehash(src/f'{t:02d}_candidates.npz')
                same(c['legacy_state_before'],old_c['state_before_decision'],'legacy source state')
                same(c['protected_legacy_state_after'],after,'full protected post-state')
                assert after['history']==source['history'][:t+4] and len(after['decisions'])==t
                assert all(call['iteration']<=t for call in after['calls'])
                expected=deepcopy(source['decisions'][t-1]);expected['protected_R2_replay']=receipt;expected['commit_sha256']=data.filehash(cp)
                same(event,expected,'protected event')
                replay['protected_legacy_state']=deepcopy(after);replay['bootstrap_receipts'].append(receipt)
                replay['legacy_context']={'initial_knowledge':deepcopy(after['initial_knowledge']),
                    'last_prefix_active_knowledge':deepcopy(after['knowledge']),
                    'past_proposals':[{'query':call['iteration'],'knowledge':call['answer']['plan']['knowledge']}
                                     for call in after['calls'] if call['iteration']<=t],
                    'past_shadow_validations':[deepcopy(d['pending_validation']) for d in after['decisions'] if d.get('pending_validation')]}
            else:
                def provider(own,d,query,card):
                    context=data.read(s.ROOT/'requests'/c['context_sha256']/'context.json')
                    assert context['observed_prefix']==own['history']
                    assert context['system_sha256']==digest(llm.SYSTEM)
                    same(context['public'],llm.payload(pool,own,d,query,card),'LLM observed-only payload')
                    validated=llm.validate(c['answer']['raw'],pool,own,card)
                    same(validated,c['answer']['plan'],'LLM schema')
                    contexts.add(c['context_sha256']);return deepcopy(c['answer']),c['context_sha256']
                prepared=s.plan(replay,cache,t,provider);d=prepared['d'];decision=prepared['decision'];g=decision['guidance']
                assert pool['ids'][decision['selected']]==c['selected_id']
                assert pool['ids'][d['reference']]==c['reference_id']
                same(prepared['card'],c['candidate_card'],'candidate card')
                same(prepared['authorizations'],c['authorizations'],'causal authorizations')
                same({key:value for key,value in decision.items() if key!='guidance'},c['decision'],'decision review')
                same(g['priors'],c['local_priors'],'prequery priors')
                same(replay['hypotheses'],c['local_models_at_commit'],'local registry before result')
                checks_arrays=[('base_qLogEI',d['base']),('guided_qLogEI',g['score']),('mu_y',d['stats']['mu_y']),
                    ('sd_y',d['stats']['sd_y']),('sd_predictive_y',d['stats']['sd_predictive_y']),
                    ('local_features_y',g['features_y']),('local_weights',g['weights']),
                    ('local_shift_y',g['shift_y']),('local_coefficient',g['coefficient'])]
                for name,value in checks_arrays:
                    error=float(np.max(abs(a[name]-value))) if value.size else 0.
                    maximum=max(maximum,error);assert error<1e-9,(task,t,name,error)
                assert np.array_equal(a['eligible'],d['eligible']) and np.array_equal(a['available'],d['available'])
                assert np.array_equal(a['local_owner'],g['owner'])
                z=np.asarray(pool['z']);n=len(pool['ids'])
                for k,key in enumerate(g['keys']):
                    h=replay['hypotheses'][key];center=np.asarray(h['scope']['center'])
                    distance=np.sqrt(np.mean((z-center)**2,axis=1));radius=float(np.sort(distance)[min(32,n)-1])
                    assert radius==h['scope']['radius_RMS_normalized']
                    w=np.maximum(0.,1-(distance/radius)**2)**2;w[distance>=radius]=0.
                    H=np.clip(w*(h['prediction_y']-d['stats']['mu_y']),-d['stats']['sd_predictive_y'],d['stats']['sd_predictive_y'])
                    assert float(np.max(abs(H-a['local_features_y'][k])))<1e-12
                    assert np.all(a['local_features_y'][k,distance>=radius]==0.)
                    if h['status'] in ('probation','supported'):
                        eligible,score,positive,required=independent_evidence(h,t)
                        assert eligible and not h['withdrawn']
                        assert abs(score-h['evidence']['score'])<1e-12
                        assert required==h['evidence']['required_positives']
                    if distance[j]>=radius:outside+=1
                assert abs(g['shift_y'][j])<=.5*d['stats']['sd_predictive_y'][j]+1e-12
                if decision['proposed_index'] is not None:
                    proposed=decision['proposed_index'];expected_review=bool(d['base'][d['reference']]-d['base'][proposed]<=d['budget'])
                    assert expected_review==decision['GP_review_accepted']
                expected=s.apply_feedback(replay,cache,prepared,t,state['history'][t+3]);expected['commit_sha256']=data.filehash(cp)
                same(expected,event,f'{task}/{t}/frozen feedback')
                mu=float(d['stats']['mu_y'][j]);sd=float(d['stats']['sd_predictive_y'][j]);y=event['y'];R=sd**2+.0009
                for row in event['local_feedback']['shadow_checks']:
                    checks+=1;H=row['feature_y'];delta=.25*H;gain=(y-mu)**2-(y-mu-delta)**2
                    assert abs(gain-row['paired_SE_gain'])<1e-12
                    assert abs(gain/(2*R)-row['standardized_gain'])<1e-12
                    assert row['informative']==bool(abs(delta)/np.sqrt(R)>=.05 and abs(H)>1e-12)
                    if row['informative']:
                        informative+=1;k=g['keys'].index(row['hypothesis_id']);prior=g['priors'][k]['shadow']
                        err,coefficient,variance=scalar_check(prior,row['shadow_KF'],mu,sd,y,H,.5,True)
                        kf_error=max(kf_error,err)
                        bank=replay['hypotheses'][row['hypothesis_id']]['shadow_bank']
                        assert abs(bank['mean']-coefficient)<1e-12 and abs(bank['variance']-variance)<1e-12
                actual=event['local_feedback']['actual_local_update']
                if actual:
                    local_updates+=1;prior=actual['prior'];H=actual['measurement']['feature']
                    err,coefficient,variance=scalar_check(prior,actual['measurement'],mu,sd,y,H,actual['measurement']['cap'])
                    kf_error=max(kf_error,err)
                    assert abs(actual['bank_after']['mean']-coefficient)<1e-12
                    assert abs(actual['bank_after']['variance']-variance)<1e-12
                    negative+=int(actual['measurement']['actual_paired_SE_gain']< -1e-12)
                else:assert event['local_feedback']['actual_paired_SE_gain']==0.
            with np.load(data.ROOT/task/'oracle.npz') as f:assert abs(float(f['raw_values'][j])-event['raw_value'])<1e-9
            replay['history'].append(deepcopy(state['history'][t+3]));replay['decisions'].append(deepcopy(event));cache.release()
    for key in CRITICAL:same(replay[key],state[key],f'{task}/final/{key}')
    assert cache.counters['fits']==0 and state['operator_compiles_in_loop']==0 and sum(s.base.JIT.values())==0
    assert len(state['calls'])==16-e.BOOTSTRAP[task]
    return {'task':task,'choices_checked':16,'protected_choices':e.BOOTSTRAP[task],
        'new_choices':16-e.BOOTSTRAP[task],'LLM_contexts_checked':len(contexts),
        'shadow_checks':checks,'informative_shadow_checks':informative,'actual_local_updates':local_updates,
        'negative_actual_KF_nonincrease_checks':negative,'out_of_scope_zero_checks':outside,
        'max_prediction_acquisition_error':maximum,'max_independent_KF_error':kf_error,
        'GP_refits_in_audit':0,'operator_compiles_in_loop':0}


def audit():
    p=s.check();assert s.status()['complete']==2
    with ProcessPoolExecutor(max_workers=2) as executor:results=list(executor.map(audit_task,s.TASKS))
    keep=data.read(s.OUT/'protected_snapshot.json');assert digest(keep)==p['protected_snapshot_sha256']
    for name,value in keep.items():assert data.filehash(name)==value,name
    receipt={'status':'PASS','completed_trajectories':2,'budgeted_added_choices_verified':32,
        'protected_prefix_choices_verified':7,'new_choices_verified':25,'baselines_rerun':0,
        'protected_files_unchanged':len(keep),'checks':results,
        'scope':'technical validation of a single-seed protected-prefix continuation, not general superiority'}
    data.write(s.OUT/'validation_receipt.json',receipt);return receipt


def metric(state,pool):
    y=np.array([h['raw_value'] for h in state['history']]);best=np.maximum.accumulate(y)
    fresh=state['decisions'][e.BOOTSTRAP[state['task']]:]
    return {'task':state['task'],'method':e.METHOD,'SearchAUC':float(best[4:].mean()),'final_best':float(best[-1]),
        'first_final_best_total':int(np.flatnonzero(y==best[-1])[0]+1),'initial_best':float(best[3]),
        'unit':pool['objective_unit'],'result_source':'R3_only_new_run','protected_prefix_queries':e.BOOTSTRAP[state['task']],
        'fresh_LLM_calls':len(state['calls']),'GP_accepted':sum(d['GP_review_accepted'] for d in fresh),
        'GP_rejected':sum(d['LLM_proposed_id'] is not None and not d['GP_review_accepted'] for d in fresh),
        'LLM_abstained':sum(d['LLM_proposed_id'] is None for d in fresh),
        'fresh_nonbase_choices':sum(d['selected_id']!=d['reference_id'] for d in fresh),
        'local_mean_deployments':sum(d['local_feedback']['actual_local_update'] is not None for d in fresh),
        'local_mean_ranking_changes':sum(d['local_ranking_changed'] for d in fresh),
        'positive_actual_local_gains':sum(d['local_feedback']['actual_paired_SE_gain']>1e-12 for d in fresh),
        'negative_actual_local_gains':sum(d['local_feedback']['actual_paired_SE_gain']< -1e-12 for d in fresh),
        'registered_models':len(state['hypotheses']),'authorization_events':len(state['authorization_events']),
        'recoveries_after_actual_failure':sum(a['from']=='quarantined' for a in state['authorization_events']),
        'post_prefix_best_gain':float(best[-1]-best[3+e.BOOTSTRAP[state['task']]]),'gate_exceptions':0}


def report():
    p=s.check();assert data.read(s.OUT/'validation_receipt.json')['status']=='PASS'
    old=data.read(s.parent.OUT/'results.json')['metrics'];metrics=deepcopy(old);new=[];curves={};activation=[]
    for state in s.states():
        task=state['task'];pool=s.original.pool(task);scale=pool['objective_scale'];ids=pool['ids'];ix={key:j for j,key in enumerate(ids)}
        row=metric(state,pool);metrics.append(row);new.append(row)
        folder=s.ROOT/'runs'/state['run_id']/'rounds';target=s.OUT/task;trajectory=[];history=[];shadows=[];actual=[];reviews=[]
        with np.load(data.ROOT/task/'oracle.npz') as f:truth=f['raw_values'].copy()
        prefix=state['history'][:4+e.BOOTSTRAP[task]]
        prefix_best=np.maximum.accumulate([h['raw_value'] for h in prefix])[4:]
        ceiling=float((prefix_best.sum()+(16-e.BOOTSTRAP[task])*float(truth.max()))/16)
        row['protected_prefix_SearchAUC_upper_bound']=ceiling;row['pool_maximum_analysis_only']=float(truth.max())
        for i,h in enumerate(state['history'],1):
            record={'total_experiment':i,'query_round':max(0,i-4),'candidate_id':h['candidate_id'],**pool['points'][ix[h['candidate_id']]],
                'objective':h['raw_value'],'cumulative_best':max(v['raw_value'] for v in state['history'][:i]),
                'phase':'initial4' if i<=4 else 'protected_R2_prefix' if i<=4+e.BOOTSTRAP[task] else 'R3_fresh_decision'}
            if 'e_factor' in h:record['e_factor']=h['e_factor']
            history.append(record)
        for t,event in enumerate(state['decisions'],1):
            c=data.read(folder/f'{t:02d}_committed.json');m=event['measurement'];legacy=t<=e.BOOTSTRAP[task]
            j=ix[event['selected_id']];ref=ix[event['reference_id']]
            with np.load(folder/f'{t:02d}_candidates.npz') as f:a={key:f[key].copy() for key in f.files}
            shift=a['deployed_shift_std']*c['GP_scale_y'] if legacy else a['local_shift_y']
            local=event.get('local_feedback')
            trajectory.append({'query_round':t,'total_experiment':t+4,'candidate_id':event['selected_id'],**pool['points'][j],
                'objective':event['raw_value'],'cumulative_best':event['incumbent_after']*scale,
                'mode':'protected_R2_prefix' if legacy else event['execution_mode'],'reference_id':event['reference_id'],
                'LLM_proposed_id':event.get('LLM_proposed_id'),'GP_review_accepted':event.get('GP_review_accepted'),
                'GP_mean':m['prequery_mu_y']*scale,'GP_predictive_SD':m['prequery_predictive_sd_y']*scale,
                'deployed_mean':m['actual_deployed_mean_y']*scale,'deployed_shift':m['actual_deployed_correction_y']*scale,
                'actual_paired_SE_gain_raw_squared':m['actual_paired_SE_gain']*scale**2,
                'local_ranking_changed':event.get('local_ranking_changed',False),
                'local_owner':local['actual_owner_hypothesis'] if local else '',
                'base_opportunity_loss':event['base_opportunity_loss'],'budget':event['budget']})
            if not legacy:
                reviews.append({'query_round':t,'proposed_id':event['LLM_proposed_id'],'GP_accepted':event['GP_review_accepted'],
                    'GP_loss':event['GP_review_loss'],'epsilon':event['budget'],'selected_id':event['selected_id'],
                    'mode':event['execution_mode'],'proposal_selected':event['proposal_selected'],
                    'local_ranking_changed':event['local_ranking_changed'],'context_sha256':event['context_sha256']})
                for check in local['shadow_checks']:
                    shadows.append({'query_round':t,**check,'GP_mean_raw':check['GP_mean_y']*scale,
                        'probe_prediction_raw':check['probe_prediction_y']*scale,'observed_raw':check['observed_y']*scale})
                if local['actual_local_update']:
                    update=local['actual_local_update'];v=update['measurement']
                    actual.append({'query_round':t,'hypothesis_id':update['hypothesis_id'],'candidate_id':event['selected_id'],
                        'coefficient_before':v['coefficient_before'],'coefficient_after':v['coefficient_after'],'cap':v['cap'],
                        'GP_mean_raw':v['prequery_mu_y']*scale,'deployed_mean_raw':v['actual_deployed_mean_y']*scale,
                        'observed_raw':event['raw_value'],'actual_gain_raw_squared':v['actual_paired_SE_gain']*scale**2})
            candidates=[];menu={r['candidate_id'] for r in c.get('candidate_card',[])}
            for k,key in enumerate(ids):
                owner='' if legacy or a['local_owner'][k]<0 else c['local_hypothesis_ids'][int(a['local_owner'][k])]
                candidates.append({'candidate_id':key,**pool['points'][k],'true_objective_analysis_only':float(truth[k]),
                    'GP_mean':float(a['mu_y'][k])*scale,'GP_latent_SD':float(a['sd_y'][k])*scale,
                    'GP_predictive_SD':float(a['sd_predictive_y'][k])*scale,'base_qLogEI':float(a['base_qLogEI'][k]),
                    'guided_qLogEI':float(a['guided_qLogEI'][k]),'available':bool(a['available'][k]),
                    'inside_original_budget':bool(a['eligible'][k]),'base_loss':float(a['base_qLogEI'][ref]-a['base_qLogEI'][k]),
                    'deployed_mean':float(a['mu_y'][k]+shift[k])*scale,'local_shift':float(shift[k])*scale,
                    'local_owner':owner,'local_coefficient':float(a['local_coefficient'][k]) if not legacy else '',
                    'in_LLM_card':key in menu,'LLM_proposed':key==event.get('LLM_proposed_id'),
                    'BO_reference':key==event['reference_id'],'guided_proposal':key==event.get('guided_proposal_id'),
                    'selected':key==event['selected_id'],'tie_priority':int(a['tie_priority'][k])})
            csv_write(target/'candidate_tables'/f'{t:02d}_candidates.csv',candidates)
        csv_write(target/'trajectory.csv',trajectory);csv_write(target/'all20_history.csv',history)
        csv_write(target/'candidate_reviews.csv',reviews);csv_write(target/'shadow_checks.csv',shadows)
        csv_write(target/'actual_local_updates.csv',actual)
        data.write(target/'local_hypotheses.json',{'hypotheses':state['hypotheses'],'authorizations':state['authorization_events']})
        activation.append(row)
    for task in s.TASKS:
        curves[task]={}
        for method in NAMES:
            root=s.ROOT if method==e.METHOD else s.parent.ROOT if method=='New_Agent_R2' else s.original.ROOT
            state=data.read(root/'runs'/(task+'_'+method)/'state.json');y=np.array([h['raw_value'] for h in state['history']])
            curves[task][method]={'observed':y.tolist(),'best':np.maximum.accumulate(y).tolist()}
    csv_write(s.OUT/'comparison.csv',metrics);csv_write(s.OUT/'main_comparison.csv',[r for r in metrics if r['method'] in NAMES])
    data.write(s.OUT/'results.json',{'metrics':metrics,'new_metrics':new,'protocol_sha256':p['protocol_sha256']})
    data.write(s.OUT/'module_activation.json',activation);data.write(s.OUT/'curves.json',curves);plot(curves)
    return {'new_metrics':new,'comparison_rows':len(metrics)}


def plot(curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.size':10})
    styles={'Pure_BO':('#555555','-',1.1),'LLINBO_Justify':('#2676ad','-',1.1),
            'ReasoningBO':('#b57932','--',1.1),'New_Agent_R2':('#9b6aac','--',1.2),'New_Agent_R3':('#d62728','-',2.1)}
    for task,methods in curves.items():
        fig,axes=plt.subplots(1,2,figsize=(12.6,3.7));x=np.arange(1,21)
        for method,values in methods.items():
            color,line,width=styles[method]
            for ax,key in zip(axes,['observed','best']):
                ax.plot(x,values[key],color=color,linestyle=line,linewidth=width,label=method,
                        marker='o' if method==e.METHOD else None,markersize=2.3)
        for ax,title in zip(axes,['Observed objective','Cumulative best']):
            ax.axvspan(.5,4.5,color='#eee',zorder=-1);ax.grid(alpha=.2);ax.set_title(title)
            ax.set_xlabel('Total experiment (4 + 16)');ax.set_xticks([1,4,8,12,16,20]);ax.set_xlim(.5,20.5)
            ax.set_ylabel('STY (kg m^-3 h^-1)' if task=='SnAr' else 'Peak area (source units)')
        handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=5,fontsize=8)
        fig.tight_layout(rect=[0,.10,1,1]);fig.savefig(s.OUT/(task+'_comparison.png'),dpi=180,bbox_inches='tight')
        fig.savefig(s.OUT/(task+'_figure.pdf'),bbox_inches='tight');plt.close(fig)


def inspect(task,t):
    state=data.read(s.ROOT/'runs'/(task+'_'+e.METHOD)/'state.json')
    return {'task':task,'decision':state['decisions'][t-1],
            'candidate_CSV':str((s.OUT/task/'candidate_tables'/f'{t:02d}_candidates.csv').resolve())}
