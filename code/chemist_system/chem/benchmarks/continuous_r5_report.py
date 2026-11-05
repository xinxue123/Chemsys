"""R5 replay audit, semantic contracts and seven-way frozen comparison."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import numpy as np
from . import continuous_r5_study as s
from . import continuous_r5_engine as e
from . import continuous_r5_llm as llm
from . import continuous_benchmark_data as data
from .continuous_r3_report import csv_write,same,no_refit
from .suzuki_domain import digest

CRITICAL=('history','decisions','initial_knowledge','hypotheses','authorization_events','calls','bootstrap_receipts',
    'protected_legacy_state','legacy_context','evidence_ledger','inheritance_events','scheduled_validations')
NAMES={'Pure_BO':'纯BO（qLogEI）','LLINBO_Justify':'LLINBO-Justify','ReasoningBO':'Reasoning BO',
    'New_Agent_R2':'R2','New_Agent_R3':'R3','New_Agent_R4':'R4','New_Agent_R5':'R5'}


def scalar_check(prior,m,H,mu,sd,y,cap,weight=1.,shadow=False):
    a=prior['mean'];P=prior['variance'];R=(sd**2+.0009)/weight;K=P*H/(P*H*H+R)
    innovation=y-mu-a*H;expected=float(np.clip(a+K*innovation,0,cap));variance=P*R/(P*H*H+R)
    gain=(y-mu)**2-(y-mu-a*H)**2
    name='hypothetical_paired_SE_gain' if shadow else 'actual_paired_SE_gain'
    err=max(abs(m['coefficient_after']-expected),abs(m['innovation']-innovation),abs(m[name]-gain),abs(m['measurement_variance']-R))
    assert err<1e-12
    if gain < -1e-12:assert expected<=a+1e-12
    return err,variance


def independent_local(pool,h,a):
    z=np.asarray(pool['z']);center=np.asarray(h['scope']['center']);distance=np.sqrt(np.mean((z-center)**2,axis=1))
    radius=float(np.sort(distance)[min(32,len(z))-1]);assert radius==h['scope']['radius_RMS_normalized']
    w=np.maximum(0.,1-(distance/radius)**2)**2;w[distance>=radius]=0.
    focus=h['basis']['focus_indices'];background=[i for i in range(z.shape[1]) if i not in focus]
    if background:
        delta=np.max(abs(z[:,background]-center[background]),axis=1)
        w*=np.maximum(0.,1-(delta/.25)**2)**2;w[delta>=.25]=0.
    reference=np.asarray(pool['z'][pool['ids'].index(h['basis']['reference_candidate_id'])])
    for axis,direction in h['semantic_contract']['directions'].items():
        k=pool['axes'].index(axis);sign=1 if direction=='increase' else -1
        w[(z[:,k]-reference[k])*sign<=1e-12]=0.
    if background:w[np.max(abs(z[:,background]-reference[background]),axis=1)>.25+1e-12]=0.
    v=np.asarray(h['basis']['focus_displacement']);den=float(v@v)
    u=(z[:,focus]-center[focus])@v/den if den>1e-12 else np.zeros(len(z));u=np.clip(u,-1.,.5)
    forecast=np.clip(h['prediction_y']+(h['prediction_y']-h['basis']['reference_y'])*u,0.,2.)
    H=np.clip(w*(forecast-a['mu_y']),-a['sd_predictive_y'],a['sd_predictive_y'])
    return w,forecast,H


def independent_evidence(h,t):
    assert all(r['query']<t and r['query']>=h['created_query'] for r in h['shadow_records'])
    cut=max(h['last_actual_failure_query'],h['inherited_actual_failure_query'],h['last_pause_query'])
    rows=[r for r in h['shadow_records'] if r['query']>cut and r['query']>=t-6][-4:]
    inherited=[r for r in h['inherited_evidence'] if r['query']>=t-6 and r['query']>cut]
    positive={r['candidate_id'] for r in rows if r['informative'] and r['paired_SE_gain']>1e-12}
    strong=[r for r in rows if r['informative']];score=sum(r['weighted_standardized_gain'] for r in rows+inherited)
    eligible=bool(len(positive)>=2 and score>0 and strong[-1]['paired_SE_gain']>1e-12)
    # Authorization occurs before the blind call. A pause issued in that call
    # immediately zeros deployment but does not rewrite the pre-call evidence
    # receipt. Validate both states instead of treating the receipt as post-pause.
    stored_eligible=eligible;stored_positive=len(positive);stored_score=score
    if h['last_pause_query']==t:
        assert h['active_cap']==0 and h['paused_by_LLM']
        current=[r for r in h['pause_records'] if r['query']==t and r['new_pause_epoch']]
        assert len(current)==1
        prior_pause=current[0]['epoch_before']
        old_cut=max(h['last_actual_failure_query'],h['inherited_actual_failure_query'],prior_pause)
        old_rows=[r for r in h['shadow_records'] if r['query']>old_cut and r['query']>=t-6][-4:]
        old_inherited=[r for r in h['inherited_evidence'] if r['query']>=t-6 and r['query']>old_cut]
        old_strong=[r for r in old_rows if r['informative']]
        stored_positive=len({r['candidate_id'] for r in old_strong if r['paired_SE_gain']>1e-12})
        stored_score=sum(r['weighted_standardized_gain'] for r in old_rows+old_inherited)
        stored_eligible=bool(stored_positive>=2 and stored_score>0 and old_strong[-1]['paired_SE_gain']>1e-12)
    assert h['evidence']['eligible']==stored_eligible and h['evidence']['positive_count']==stored_positive
    assert abs(h['evidence']['score']-stored_score)<1e-12
    if h['active_cap']>0:assert eligible and not h['paused_by_LLM']


def independent_context(pool,h,ledger,t,a):
    w,f,unused=independent_local(pool,h,a);records={};failed=[]
    own={(r['query'],r['candidate_id']) for r in h['shadow_records']}
    for event in ledger:
        if event['query']>=t or event['source_hypothesis_id']==h['id']:continue
        j=pool['ids'].index(event['candidate_id'])
        if w[j]<=1e-12:continue
        sw,unused1,unused2=independent_local(pool,event['source_model'],a)
        den=min(int(np.sum(w>1e-12)),int(np.sum(sw>1e-12)))
        overlap=float(np.sum((w>1e-12)&(sw>1e-12))/den) if den else 0.
        sd=event['GP_predictive_SD_y'];H=float(np.clip(w[j]*(f[j]-event['GP_mean_y']),-sd,sd));delta=.25*H
        residual=event['observed_y']-event['GP_mean_y'];gain=float(2*residual*delta-delta**2)
        if gain>=-1e-12:continue
        if event['kind']=='actual_failure':failed.append(event['event_id'])
        token=(event['query'],event['candidate_id'])
        if token in own:continue
        R=sd**2+.0009;weight=min(1.,(abs(delta)/np.sqrt(R)/.05)**2)
        row={'query':event['query'],'candidate_id':event['candidate_id'],'source_event_id':event['event_id'],
            'source_kind':event['kind'],'support_overlap':overlap,'target_probe_y':delta,'paired_SE_gain':gain,
            'weighted_standardized_gain':float(weight*gain/(2*R)),
            'kind':'inherited counterevidence only, never prospective positive validation'}
        if token not in records or row['weighted_standardized_gain']<records[token]['weighted_standardized_gain']:records[token]=row
    expected=sorted(records.values(),key=lambda r:(r['query'],r['candidate_id']))
    same(expected,h['inherited_evidence'],'independent point-level inheritance')
    assert h['inherited_actual_failure_query']==-1 and h['inherited_failure_ids']==sorted(failed)
    for pause in h['pause_records']:
        if not pause['new_pause_epoch']:assert pause['epoch_after']==pause['epoch_before']


def audit_task(task):
    s.base.warm();protocol=s.check();pool=s.original.pool(task);cache=e.Cache(pool)
    state=data.read(s.ROOT/'runs'/(task+'_'+e.METHOD)/'state.json');folder=s.ROOT/'runs'/state['run_id']/'rounds'
    assert state['status']=='COMPLETE' and len(state['history'])==20 and len(state['decisions'])==16
    assert len({h['candidate_id'] for h in state['history']})==20
    replay=s.initial(task,protocol);source=data.read(s.prefix_study.parent.ROOT/'runs'/(task+'_New_Agent_R2')/'state.json')
    assert state['history'][:4+e.BOOTSTRAP[task]]==source['history'][:4+e.BOOTSTRAP[task]]
    maximum=kf_error=0.;contexts=set();shadow_count=informative=actual_count=negative=scheduled=0
    with patch.object(s.base,'fit_gpytorch_mll',no_refit):
        for t,event in enumerate(state['decisions'],1):
            cp=folder/f'{t:02d}_committed.json';c=data.read(cp)
            assert data.filehash(cp)==event['commit_sha256'] and c['protocol_sha256']==protocol['protocol_sha256']
            assert data.filehash(folder/f'{t:02d}_candidates.npz')==c['candidate_table_sha256']
            for key in CRITICAL:same(replay[key],c['state_before_decision'][key],f'{task}/{t}/before/{key}')
            assert c['history']==state['history'][:t+3] and c['history_sha256']==digest(c['history'])
            with np.load(folder/f'{t:02d}_candidates.npz') as f:a={key:f[key].copy() for key in f.files}
            j=cache.ix[c['selected_id']];assert a['eligible'][j] and a['available'][j]
            if t<=e.BOOTSTRAP[task]:
                src=s.prefix_study.parent.ROOT/'runs'/(task+'_New_Agent_R2')/'rounds'
                old=data.read(src/f'{t:02d}_committed.json');after=data.read(src/f'{t+1:02d}_committed.json')['state_before_decision']
                receipt=c['protected_R2_replay'];assert receipt['source_commit_sha256']==data.filehash(src/f'{t:02d}_committed.json')
                assert data.filehash(folder/f'{t:02d}_candidates.npz')==data.filehash(src/f'{t:02d}_candidates.npz')
                same(c['legacy_state_before'],old['state_before_decision'],'legacy before');same(c['protected_legacy_state_after'],after,'legacy after')
                expected=deepcopy(source['decisions'][t-1]);expected['protected_R2_replay']=receipt;expected['commit_sha256']=data.filehash(cp)
                same(expected,event,'protected event')
                replay['protected_legacy_state']=deepcopy(after);replay['bootstrap_receipts'].append(receipt)
                replay['legacy_context']={'initial_knowledge':deepcopy(after['initial_knowledge']),
                    'last_prefix_active_knowledge':deepcopy(after['knowledge']),
                    'past_proposals':[{'query':call['iteration'],'knowledge':call['answer']['plan']['knowledge']} for call in after['calls'] if call['iteration']<=t],
                    'past_shadow_validations':[deepcopy(d['pending_validation']) for d in after['decisions'] if d.get('pending_validation')]}
            else:
                proposal_file=folder/f'{t:02d}_blind_proposal.json';assert data.filehash(proposal_file)==c['blind_proposal_sha256']
                preGP=data.read(proposal_file)
                assert preGP['controller_GP_computed'] is True and preGP['LLM_received_GP_information'] is False
                assert preGP['GP_review_completed'] is False
                def provider(own,query):
                    context=data.read(s.ROOT/'requests'/c['context_sha256']/'context.json');contexts.add(c['context_sha256'])
                    assert context['observed_prefix']==own['history'] and context['system_sha256']==digest(llm.SYSTEM)
                    same(context['public'],llm.payload(pool,own,query),'blind public payload')
                    assert all(set(row)=={'index','point','objective'} for row in context['public']['observations'])
                    for row in context['public']['candidate_menu']:
                        index=cache.ix[row['candidate_id']]
                        assert row['point']==pool['points'][index] and row['candidate_id'] not in {h['candidate_id'] for h in own['history']}
                    raw=c['answer']['raw'];same(llm.validate(raw,pool,own,query),c['answer']['plan'],'validated schema')
                    for request in (s.ROOT/'requests'/c['context_sha256']).glob('attempt*/request.json'):
                        messages=data.read(request)['messages'];same(__import__('json').loads(messages[1]['content']),context['public'],'actual API public payload')
                        assert messages[0]['content']==llm.SYSTEM
                    return deepcopy(c['answer']),c['context_sha256']
                prepared=s.plan(replay,cache,t,provider,lambda receipt:same(receipt,preGP,'pre-GP proposal receipt'))
                d=prepared['d'];g=prepared['g'];decision=prepared['decision']
                assert pool['ids'][decision['selected']]==c['selected_id']
                same(decision,c['decision'],'decision');same(g['priors'],c['local_priors'],'priors')
                same(replay['hypotheses'],c['local_models_at_commit'],'models before result')
                same(prepared['projection'],c['projection'],'projection');same(prepared['authorizations'],c['authorizations'],'authorizations')
                same(prepared['inheritance_changes'],c['inheritance_changes'],'inheritance')
                for name,value in [('base_qLogEI',d['base']),('guided_qLogEI',g['score']),('mu_y',d['stats']['mu_y']),
                    ('sd_y',d['stats']['sd_y']),('sd_predictive_y',d['stats']['sd_predictive_y']),('local_features_y',g['features_y']),
                    ('local_weights',g['weights']),('local_shift_y',g['shift_y']),('local_coefficient',g['coefficient'])]:
                    error=float(np.max(abs(a[name]-value))) if value.size else 0.;maximum=max(maximum,error)
                    assert error<1e-9,(task,t,name,error)
                for name,value in [('eligible',d['eligible']),('available',d['available']),('local_owner',g['owner'])]:assert np.array_equal(a[name],value)
                assert np.all(abs(a['local_shift_y'])<=.5*a['sd_predictive_y']+1e-12)
                for k,key in enumerate(g['keys']):
                    h=replay['hypotheses'][key];w,forecast,H=independent_local(pool,h,a)
                    assert np.max(abs(w-a['local_weights'][k]))<1e-12 and np.max(abs(H-a['local_features_y'][k]))<1e-12
                    if h['created_query']<t:independent_evidence(h,t)
                    independent_context(pool,h,replay['evidence_ledger'],t,a)
                if prepared['projection']:
                    projected=prepared['projection'];plan=c['answer']['plan'];chosen=projected['index']
                    assert plan['candidate_id']==pool['ids'][chosen] and projected['point']==pool['points'][chosen]
                    assert projected['proposed_physical_point']==projected['point'] and projected['normalized_Euclidean_distance']==0.
                    hypothesis=plan['hypothesis'];reference=c['history'][hypothesis['reference_index']-1]
                    rp=pool['points'][cache.ix[reference['candidate_id']]];point=projected['point'];violations=[]
                    for axis in hypothesis['focus_axes']:
                        sign=1 if hypothesis['directions'][axis]=='increase' else -1
                        if (point[axis]-rp[axis])*sign<=1e-12:violations.append('focus_direction_mismatch:'+axis)
                    for axis in pool['axes']:
                        if axis not in hypothesis['focus_axes']:
                            lo,hi=pool['bounds'][axis]
                            if abs(point[axis]-rp[axis])/(hi-lo)>.25+1e-12:violations.append('background_outside_contract:'+axis)
                    assert projected['semantic_review']=={'admitted':not violations,'violations':violations}
                    assert decision['GP_review_accepted']==bool(not violations and d['eligible'][chosen])
                    if violations:assert prepared['registered_hypothesis'] is None
                validation=decision['validation'];vmask=np.zeros(len(pool['ids']),dtype=bool)
                for row in validation['candidates']:
                    index=row['index'];assert index!=d['reference'] and d['eligible'][index]
                    assert row['probe_ratio']>=.05 and row['contrast_proxy']>=.05;vmask[index]=True
                assert np.array_equal(vmask,a['validation_candidate'])
                if decision['mode']=='scheduled_local_validation':
                    scheduled+=1;assert len(replay['scheduled_validations'])<=3 and t<=15
                    if validation['stage']=='initial_probe':assert t<=14
                assert sum(v['stage']=='initial_probe' for v in replay['scheduled_validations'])<=1
                if prepared['call_permission']['reason']=='validation_executable_now':
                    assert decision['mode']=='scheduled_local_validation' and decision['selected']==prepared['call_permission']['validation_index']
                expected=s.apply_feedback(replay,cache,prepared,t,state['history'][t+3]);expected['commit_sha256']=data.filehash(cp)
                same(expected,event,f'{task}/{t}/feedback')
                mu=float(a['mu_y'][j]);sd=float(a['sd_predictive_y'][j]);y=event['y'];R=sd**2+.0009
                for row in event['local_feedback']['shadow_checks']:
                    shadow_count+=1;H=row['feature_y'];delta=.25*H;gain=(y-mu)**2-(y-mu-delta)**2
                    ratio=abs(delta)/np.sqrt(R);weight=min(1.,(ratio/.05)**2)
                    assert abs(gain-row['paired_SE_gain'])<1e-12 and abs(weight-row['signal_weight'])<1e-12
                    assert row['informative']==bool(ratio>=.05 and abs(H)>1e-12);informative+=row['informative']
                    if 'shadow_KF' in row:
                        k=g['keys'].index(row['hypothesis_id']);err,variance=scalar_check(g['priors'][k]['shadow'],row['shadow_KF'],H,mu,sd,y,.5,weight,True)
                        kf_error=max(kf_error,err);assert abs(replay['hypotheses'][row['hypothesis_id']]['shadow_bank']['variance']-variance)<1e-12
                actual=event['local_feedback']['actual_local_update']
                if actual:
                    actual_count+=1;m=actual['measurement'];err,variance=scalar_check(actual['prior'],m,m['feature'],mu,sd,y,m['cap'])
                    kf_error=max(kf_error,err);assert abs(actual['bank_after']['variance']-variance)<1e-12
                    negative+=int(m['actual_paired_SE_gain']< -1e-12)
                else:assert event['local_feedback']['actual_paired_SE_gain']==0.
            with np.load(data.ROOT/task/'oracle.npz') as f:assert abs(float(f['raw_values'][j])-event['raw_value'])<1e-9
            replay['history'].append(deepcopy(state['history'][t+3]));replay['decisions'].append(deepcopy(event));cache.release()
    for key in CRITICAL:same(replay[key],state[key],f'{task}/final/{key}')
    assert len(contexts)==len(state['calls'])<=4 and cache.counters['fits']==0
    assert state['operator_compiles_in_loop']==0 and sum(s.base.JIT.values())==0
    return {'task':task,'choices_checked':16,'protected_choices':e.BOOTSTRAP[task],'new_choices':16-e.BOOTSTRAP[task],
        'blind_LLM_contexts_checked':len(contexts),'proposal_before_review_receipts_checked':16-e.BOOTSTRAP[task],
        'scheduled_validations_checked':scheduled,'shadow_checks':shadow_count,'informative_shadow_checks':informative,
        'actual_local_updates':actual_count,'negative_actual_KF_nonincrease_checks':negative,
        'max_prediction_acquisition_error':maximum,'max_independent_KF_error':kf_error,'GP_refits_in_audit':0,'operator_compiles_in_loop':0}


def audit():
    p=s.check();assert s.status()['complete']==2
    with ProcessPoolExecutor(max_workers=2) as ex:results=list(ex.map(audit_task,s.TASKS))
    keep=data.read(s.OUT/'protected_snapshot.json');assert digest(keep)==p['protected_snapshot_sha256']
    for name,value in keep.items():assert data.filehash(name)==value,name
    receipt={'status':'PASS','completed_trajectories':2,'budgeted_added_choices_verified':32,
        'protected_prefix_choices_verified':7,'new_choices_verified':25,'baselines_rerun':0,
        'protected_files_unchanged':len(keep),'checks':results,'scope':'single-seed protected-prefix technical audit, not general superiority'}
    data.write(s.OUT/'validation_receipt.json',receipt);return receipt


def metric(state,pool):
    y=np.array([h['raw_value'] for h in state['history']]);best=np.maximum.accumulate(y);fresh=state['decisions'][e.BOOTSTRAP[state['task']]:]
    calls=[d for d in fresh if d['context_sha256'] is not None]
    return {'task':state['task'],'method':e.METHOD,'SearchAUC':float(best[4:].mean()),'final_best':float(best[-1]),
        'first_final_best_total':int(np.flatnonzero(y==best[-1])[0]+1),'initial_best':float(best[3]),'unit':pool['objective_unit'],
        'result_source':'R5_only_new_run','protected_prefix_queries':e.BOOTSTRAP[state['task']],
        'fresh_LLM_calls':len(state['calls']),'GP_accepted':sum(d['GP_review_accepted'] for d in calls),
        'GP_rejected':sum(d['LLM_proposed_id'] is not None and d['semantic_review_accepted'] and not d['GP_review_accepted'] for d in calls),
        'semantic_rejected':sum(d['LLM_proposed_id'] is not None and not d['semantic_review_accepted'] for d in calls),
        'LLM_abstained':sum(d['LLM_proposed_id'] is None for d in calls),
        'LLM_follows_GP':sum(d['LLM_proposed_id']==d['reference_id'] for d in calls if d['LLM_proposed_id'] is not None),
        'fresh_nonbase_choices':sum(d['selected_id']!=d['reference_id'] for d in fresh),
        'scheduled_validations':len(state['scheduled_validations']),
        'scheduled_initial_probes':sum(v['stage']=='initial_probe' for v in state['scheduled_validations']),
        'scheduled_confirmations':sum(v['stage']=='confirmation' for v in state['scheduled_validations']),
        'late_LLM_calls':sum(v['query']>=11 for v in state['calls']),
        'idempotent_pause_actions':sum(not v['new_pause_epoch'] for h in state['hypotheses'].values() for v in h['pause_records']),
        'local_mean_deployments':sum(d['local_feedback']['actual_local_update'] is not None for d in fresh),
        'local_mean_ranking_changes':sum(d['local_ranking_changed'] for d in fresh),
        'positive_actual_local_gains':sum(d['local_feedback']['actual_paired_SE_gain']>1e-12 for d in fresh),
        'negative_actual_local_gains':sum(d['local_feedback']['actual_paired_SE_gain']< -1e-12 for d in fresh),
        'registered_models':len(state['hypotheses']),'authorization_events':len(state['authorization_events']),
        'inheritance_updates':len(state['inheritance_events']),'evidence_ledger_events':len(state['evidence_ledger']),
        'inherited_negative_context_links':sum(len(h['inherited_evidence']) for h in state['hypotheses'].values()),
        'shadow_checks_after_LLM_pause':sum(any(p['query']<=r['query'] for p in h['pause_records']) for h in state['hypotheses'].values() for r in h['shadow_records']),
        'recoveries_after_failure_or_pause':sum(a['from'] in ('quarantined','paused') for a in state['authorization_events']),
        'post_prefix_best_gain':float(best[-1]-best[3+e.BOOTSTRAP[state['task']]]),'gate_exceptions':0}


def report():
    p=s.check();assert data.read(s.OUT/'validation_receipt.json')['status']=='PASS'
    metrics=deepcopy(data.read(s.parent.OUT/'results.json')['metrics']);new=[];curves={}
    for state in s.states():
        task=state['task'];pool=s.original.pool(task);ids=pool['ids'];ix={key:j for j,key in enumerate(ids)};scale=pool['objective_scale']
        row=metric(state,pool);metrics.append(row);new.append(row);folder=s.ROOT/'runs'/state['run_id']/'rounds';target=s.OUT/task
        trajectory=[];history=[];reviews=[];shadows=[];actual=[];validation=[];modelrow=[]
        with np.load(data.ROOT/task/'oracle.npz') as f:truth=f['raw_values'].copy()
        prefix=e.BOOTSTRAP[task];best=np.maximum.accumulate([h['raw_value'] for h in state['history']])
        row['protected_prefix_SearchAUC_upper_bound']=float((best[4:4+prefix].sum()+(16-prefix)*truth.max())/16)
        for i,h in enumerate(state['history'],1):
            history.append({'total_experiment':i,'query_round':max(0,i-4),'candidate_id':h['candidate_id'],**pool['points'][ix[h['candidate_id']]],
                'objective':h['raw_value'],'cumulative_best':best[i-1],
                'phase':'initial4' if i<=4 else 'protected_R2_prefix' if i<=4+prefix else 'R5_fresh_decision'})
        for t,event in enumerate(state['decisions'],1):
            c=data.read(folder/f'{t:02d}_committed.json');legacy=t<=prefix;j=ix[event['selected_id']];ref=ix[event['reference_id']]
            with np.load(folder/f'{t:02d}_candidates.npz') as f:a={key:f[key].copy() for key in f.files}
            shift=a['deployed_shift_std']*c['GP_scale_y'] if legacy else a['local_shift_y'];m=event['measurement'];local=event.get('local_feedback')
            trajectory.append({'query_round':t,'total_experiment':t+4,'candidate_id':event['selected_id'],**pool['points'][j],
                'objective':event['raw_value'],'cumulative_best':best[t+3],'mode':'protected_R2_prefix' if legacy else event['execution_mode'],
                'reference_id':event['reference_id'],'LLM_proposed_id':event.get('LLM_proposed_id'),
                'GP_review_accepted':event.get('GP_review_accepted'),'eligible_count':int(a['eligible'].sum()),
                'GP_mean':m['prequery_mu_y']*scale,'GP_predictive_SD':m['prequery_predictive_sd_y']*scale,
                'deployed_mean':m['actual_deployed_mean_y']*scale,'deployed_shift':m['actual_deployed_correction_y']*scale,
                'actual_paired_SE_gain_raw_squared':m['actual_paired_SE_gain']*scale**2,
                'local_ranking_changed':event.get('local_ranking_changed',False),'local_owner':local['actual_owner_hypothesis'] if local else '',
                'base_opportunity_loss':event['base_opportunity_loss'],'budget':event['budget']})
            if not legacy:
                if c['answer']:
                    plan=c['answer']['plan'];context=data.read(s.ROOT/'requests'/event['context_sha256']/'context.json')
                    reviews.append({'query_round':t,'proposed_id':event['LLM_proposed_id'],
                        'physical_proposal':c['projection']['point'] if c['projection'] else None,'forecast_raw':plan['prediction'],'hypothesis':plan['hypothesis'],
                        'semantic_accepted':event['semantic_review_accepted'],
                        'semantic_violations':c['projection']['semantic_review']['violations'] if c['projection'] else [],
                        'high_reference_indices':[r['index'] for r in context['public']['high_references']],
                        'real_input_menu_size':len(context['public']['candidate_menu']),
                        'GP_accepted':event['GP_review_accepted'],'GP_loss':event['GP_review_loss'],'epsilon':event['budget'],
                        'projection_distance':c['projection']['normalized_Euclidean_distance'] if c['projection'] else None,
                        'selected_id':event['selected_id'],'mode':event['execution_mode'],'reason':plan['reason'],
                        'context_sha256':event['context_sha256'],'public_GP_information':False})
                for v in local['shadow_checks']:
                    shadows.append({'query_round':t,**v,'GP_mean_raw':v['GP_mean_y']*scale,
                        'probe_prediction_raw':v['probe_prediction_y']*scale,'observed_raw':v['observed_y']*scale})
                if local['actual_local_update']:
                    v=local['actual_local_update'];m2=v['measurement'];actual.append({'query_round':t,'hypothesis_id':v['hypothesis_id'],
                        'coefficient_before':m2['coefficient_before'],'coefficient_after':m2['coefficient_after'],'cap':m2['cap'],
                        'GP_mean_raw':m2['prequery_mu_y']*scale,'deployed_mean_raw':m2['actual_deployed_mean_y']*scale,
                        'observed_raw':event['raw_value'],'actual_gain_raw_squared':m2['actual_paired_SE_gain']*scale**2})
                validation.append({'query_round':t,'eligible_count':int(a['eligible'].sum()),'candidate_count':len(event['validation']['candidates']),
                    'validation_hypothesis':event['validation']['hypothesis_id'],'validation_selected':event['execution_mode']=='scheduled_local_validation',
                    'stage':event['validation']['stage'],
                    'reason':event['validation']['reason'],'call_reason':event['call_permission']['reason']})
            candidates=[]
            for k,key in enumerate(ids):
                owner='' if legacy or a['local_owner'][k]<0 else c['local_hypothesis_ids'][int(a['local_owner'][k])]
                candidates.append({'candidate_id':key,**pool['points'][k],'true_objective_analysis_only':float(truth[k]),
                    'GP_mean':float(a['mu_y'][k])*scale,'GP_latent_SD':float(a['sd_y'][k])*scale,'GP_predictive_SD':float(a['sd_predictive_y'][k])*scale,
                    'base_qLogEI':float(a['base_qLogEI'][k]),'guided_qLogEI':float(a['guided_qLogEI'][k]),
                    'available':bool(a['available'][k]),'inside_original_budget':bool(a['eligible'][k]),
                    'base_loss':float(a['base_qLogEI'][ref]-a['base_qLogEI'][k]),'deployed_mean':float(a['mu_y'][k]+shift[k])*scale,
                    'local_shift':float(shift[k])*scale,'local_owner':owner,'local_coefficient':float(a['local_coefficient'][k]) if not legacy else '',
                    'validation_candidate':bool(a['validation_candidate'][k]) if not legacy else False,
                    'LLM_proposed':key==event.get('LLM_proposed_id'),'BO_reference':key==event['reference_id'],
                    'guided_proposal':key==event.get('guided_proposal_id'),'selected':key==event['selected_id'],'tie_priority':int(a['tie_priority'][k])})
            csv_write(target/'candidate_tables'/f'{t:02d}_candidates.csv',candidates)
        for name,rows in [('trajectory',trajectory),('all20_history',history),('candidate_reviews',reviews),('shadow_checks',shadows),('actual_local_updates',actual),('validation_schedule',validation)]:csv_write(target/(name+'.csv'),rows)
        data.write(target/'local_hypotheses.json',{'hypotheses':state['hypotheses'],'authorizations':state['authorization_events'],
            'evidence_ledger':state['evidence_ledger'],'inheritance_events':state['inheritance_events']})
    for task in s.TASKS:
        curves[task]={}
        for method in NAMES:
            root=s.ROOT if method==e.METHOD else s.parent.ROOT if method=='New_Agent_R4' else s.prefix_study.ROOT if method=='New_Agent_R3' else s.prefix_study.parent.ROOT if method=='New_Agent_R2' else s.original.ROOT
            state=data.read(root/'runs'/(task+'_'+method)/'state.json');y=np.array([h['raw_value'] for h in state['history']])
            curves[task][method]={'observed':y.tolist(),'best':np.maximum.accumulate(y).tolist()}
    csv_write(s.OUT/'comparison.csv',metrics);csv_write(s.OUT/'main_comparison.csv',[r for r in metrics if r['method'] in NAMES])
    data.write(s.OUT/'results.json',{'metrics':metrics,'new_metrics':new,'protocol_sha256':p['protocol_sha256']})
    data.write(s.OUT/'module_activation.json',new);data.write(s.OUT/'curves.json',curves);plot(curves)
    return {'new_metrics':new,'comparison_rows':len(metrics)}


def plot(curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles={'Pure_BO':('#555555','-',1.),'LLINBO_Justify':('#2676ad','-',1.),'ReasoningBO':('#b57932','--',1.),
        'New_Agent_R2':('#8c64a7','--',1.),'New_Agent_R3':('#ce665f',':',1.4),
        'New_Agent_R4':('#16836b','--',1.4),'New_Agent_R5':('#172b4d','-',2.)}
    plt.rcParams.update({'pdf.fonttype':42,'font.size':9})
    for task,group in curves.items():
        fig,axes=plt.subplots(1,2,figsize=(12.6,3.7));x=np.arange(1,21)
        for method,series in group.items():
            color,line,width=styles[method]
            for ax,key in zip(axes,['observed','best']):ax.plot(x,series[key],color=color,linestyle=line,linewidth=width,label=method,
                marker='o' if method==e.METHOD else None,markersize=2.5)
        for ax,title in zip(axes,['Observed objective','Cumulative best']):
            ax.set_title(title);ax.set_xlabel('Total experiment (4 + 16)');ax.set_xticks([1,4,8,12,16,20]);ax.grid(alpha=.2);ax.axvspan(.5,4.5,color='#ddd',alpha=.2)
            ax.set_ylabel('STY (kg m^-3 h^-1)' if task=='SnAr' else 'Peak area (source units)')
        handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=7,fontsize=7.0)
        fig.tight_layout(rect=[0,.10,1,1]);fig.savefig(s.OUT/(task+'_comparison.png'),dpi=180,bbox_inches='tight');fig.savefig(s.OUT/(task+'_figure.pdf'),bbox_inches='tight');plt.close(fig)


def inspect(task,t):
    state=data.read(s.ROOT/'runs'/(task+'_'+e.METHOD)/'state.json')
    return {'task':task,'decision':state['decisions'][t-1],
        'candidate_CSV':str((s.OUT/task/'candidate_tables'/f'{t:02d}_candidates.csv').resolve())}
