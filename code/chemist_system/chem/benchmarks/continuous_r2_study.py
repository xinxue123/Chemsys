"""Four frozen new-agent trajectories; all previous results remain read-only."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
import json
import shutil
import time
import traceback
import numpy as np
from . import continuous_benchmark_data as data
from . import continuous_benchmark_study as frozen
from . import continuous_benchmark_engine as old
from . import revised_continuous_study as parent
from . import revised_continuous_engine as r1
from . import continuous_r2_engine as e
from . import continuous_r2_llm as llm
from . import deepseek_suzuki_client as api
from .suzuki_domain import digest

ROOT = e.ROOT
OUT = Path('research/continuous_r2_20261030')
TASKS = frozen.TASKS


def sources():
    names = [Path(__file__),Path(e.__file__),Path(llm.__file__),Path('continuous_r2.py'),
             Path('tests/test_continuous_r2.py'),OUT/'DESIGN.md']
    return {p.as_posix():data.filehash(p) for p in names}


def protected():
    keep = data.read(parent.OUT/'protected_snapshot.json')
    keep.update(data.read(parent.OUT/'release_inventory.json'))
    for name in ['release_inventory.json','release_receipt.json']:
        path = parent.OUT/name; keep[path.as_posix()] = data.filehash(path)
    diagnosis = Path('research/continuous_r1_diagnosis_20261030')
    keep.update(data.read(diagnosis/'inventory.json'))
    keep[(diagnosis/'inventory.json').as_posix()] = data.filehash(diagnosis/'inventory.json')
    keep['scripts/diagnose_continuous_r1.py'] = data.filehash('scripts/diagnose_continuous_r1.py')
    for name,h in keep.items():
        assert data.filehash(name)==h,name
    return keep


def initial(task,method,protocol):
    state = parent.initial(task,data.read(parent.OUT/'protocol.json'))
    state.update(method=method,run_id=task+'_'+method,protocol_sha256=protocol['protocol_sha256'],
                 geometry_ledger={},bank_initializations=[],pending=None,active_suspended=False,
                 promotions=[],proposal_archive=[],subset_evidence={},bootstrap_receipts=[])
    return state


def prepare(config='../config.txt'):
    if (OUT/'protocol.json').exists():
        return check()
    assert data.read(OUT/'qualification.json')['status']=='PASS'
    keep = protected(); parent_protocol = parent.check()
    bootstrap_files = {}
    for task in TASKS:
        source = data.read(parent.ROOT/'runs'/(task+'_New_Agent_R1')/'state.json')
        for t in range(1,e.BOOTSTRAP[task]+1):
            folder = parent.ROOT/'runs'/source['run_id']/'rounds'
            for suffix in ['committed.json','candidates.npz']:
                path=folder/f'{t:02d}_{suffix}'; bootstrap_files[path.as_posix()] = data.filehash(path)
        keys = {c['context_sha256'] for c in source['calls'] if c['iteration']<=e.BOOTSTRAP[task]}
        keys.update(d['interaction_context'] for d in source['decisions'][:e.BOOTSTRAP[task]] if d['interaction_context'])
        for key in keys:
            for path in (parent.ROOT/'requests'/key).rglob('*'):
                if path.is_file() and path.suffix=='.json':
                    target=ROOT/'bootstrap_requests'/key/path.relative_to(parent.ROOT/'requests'/key)
                    target.parent.mkdir(parents=True,exist_ok=True)
                    if not target.exists():shutil.copyfile(path,target)
                    assert data.filehash(path)==data.filehash(target)
    protocol={'study':'Continuous_R2_prospective_deployment_with_independent_subset_ablation',
        'tasks':list(TASKS),'methods':list(e.METHODS),'planned_trajectories':4,'seed':0,
        'initial_queries':4,'additional_queries':16,'planned_budgeted_added_queries':64,
        'parameters':e.PARAMETERS,'source_sha256':sources(),'parent_protocol_sha256':parent_protocol['protocol_sha256'],
        'input_sha256':parent_protocol['input_sha256'],'protected_snapshot_sha256':digest(keep),
        'bootstrap_source_sha256':bootstrap_files,'qualification_sha256':data.filehash(OUT/'qualification.json'),
        'LLM':api.load_config(config).public(),'maximum_fresh_logical_LLM_calls':14,
        'comparison':'R2 and independent R2_Subset versus read-only R1, old New_Agent, Pure_BO; rerun no comparator',
        'bootstrap_policy':'replay only own early prefix and corresponding frozen forecasts/responses; SnAr2 and protected HPLC3 count inside the same16 queries',
        'ablation_control':'only subset recovery toggle differs; identical observed prefixes/prompts reuse the same LLM response',
        'resources':{'persistent_independent_streams':4,'torch_threads_each':1,'GP':'unchanged CPU exact GP <=20 observations',
                     'warmup':'once per process before trajectory loop','operator_compiles_in_loop':0},
        'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'interpretation':'single seed, one frozen version; protected early-prefix continuation, not a fresh end-to-end LLM replication'}
    protocol['protocol_sha256']=digest(protocol)
    data.write(OUT/'protected_snapshot.json',keep);data.write(OUT/'protocol.json',protocol)
    return protocol


def check():
    p=data.read(OUT/'protocol.json'); q=deepcopy(p); key=q.pop('protocol_sha256')
    assert digest(q)==key,'R2 protocol changed after freeze'
    current=sources()
    if p['source_sha256']!=current:
        amendment=data.read(OUT/'execution_amendment.json')
        assert amendment['original_protocol_sha256']==key
        assert amendment['original_source_sha256']==p['source_sha256']
        assert amendment['corrected_source_sha256']==current
        assert amendment['scientific_parameters_unchanged'] and amendment['original_parameters_sha256']==digest(p['parameters'])
    for name,h in p['input_sha256'].items():assert data.filehash(name)==h,name
    for name,h in p['bootstrap_source_sha256'].items():assert data.filehash(name)==h,name
    assert data.filehash(OUT/'qualification.json')==p['qualification_sha256']
    return p


def qualify():
    old.warm();results=[]
    for task in TASKS:
        pool=frozen.pool(task);copied=e.seed_cache(pool);cache=e.Cache(pool);history=frozen.initial(task)
        scores,_=cache.score(history);stats=cache.model(history)
        with np.load(parent.ROOT/'runs'/(task+'_New_Agent_R1')/'rounds/01_candidates.npz') as f:
            error=max(float(np.max(abs(scores-f['base_qLogEI']))),
                *[float(np.max(abs(stats[k]-f[k]))) for k in ['mu_y','sd_y','sd_predictive_y']])
        assert error==0. and cache.counters['fits']==0
        results.append({'task':task,'pool_size':len(pool['ids']),'same_initial_ids':pool['initial_ids'],
                        'max_initial_prediction_acquisition_difference':error,'GP_fits':0,'checkpoints_copied':copied})
    receipt={'status':'PASS','tasks':results,'operator_compiles_in_loop':sum(old.JIT.values()),
             'scope':'same GP recipe, inputs, initial forecasts and acquisition; no performance claim'}
    data.write(OUT/'qualification.json',receipt);return receipt


def decide(state,cache,t,region_provider,joint_provider):
    promotion=e.promote(state,t);d=e.setup(cache,state,t)
    reasons=r1.event_reasons(state,d,t)
    if state.get('pending') and state['pending']['status']=='rejected':reasons.append('rejected_prospective_proposal')
    quota=e.refresh_permission(state,t);check_record={'iteration':t,'reasons':reasons,'quota':quota}
    state['refresh_checks'].append(deepcopy(check_record));refresh=None
    if reasons and quota['allowed']:
        answer,key=region_provider(state,d,reasons,quota)
        refresh={'answer':answer,'context_sha256':key}
        e.stage_refresh(cache.pool,state,t,answer,key,reasons,quota)
        d=e.setup(cache,state,t)
    mode='R2_BO_pending' if state['active_suspended'] else 'R2_normal'
    joint=None;joint_check=None
    if t==6:
        if state['active_suspended']:
            joint_check={'iteration':t,'action':'skip','reason':'prospective_BO_validation_or_rejected_proposal_has_priority'}
        else:
            prepared=r1.interaction_candidates(cache,state,d,t)
            if not prepared['candidates']:
                joint_check={'iteration':t,'action':'skip','reason':prepared['skip_reason']}
            else:
                answer,key=joint_provider(state,d,prepared)
                joint={'prepared':prepared,'answer':answer,'context_sha256':key}
                plan=answer['plan']
                if plan['action']=='keep_BO':
                    d=e.suspend(d,'LLM_optional_joint_abstention');mode='R2_BO_abstention'
                else:
                    d['selected']=cache.ix[plan['candidate_id']];mode='R2_bounded_joint_test'
                joint_check={'iteration':t,'action':plan['action'],'candidate_count':len(prepared['candidates']),
                             'reason':plan['reason'],'context_sha256':key}
        state['interaction_checks'].append(deepcopy(joint_check))
    shadow=e.pending_forecast(cache,state,d,t)
    assert d['eligible'][d['selected']] and d['available'][d['selected']]
    if shadow is not None:assert d['selected']==d['reference'] and not d['gate_open']
    return {'d':d,'promotion':promotion,'refresh':refresh,'refresh_check':check_record,
            'interaction':joint,'interaction_check':joint_check,'pending_forecast':shadow,'execution_mode':mode}


def feedback(state,cache,plan,t,observed):
    d=plan['d'];j=d['selected'];y=observed['y'];pool=cache.pool
    mu=float(d['stats']['mu_y'][j]);sd=float(d['stats']['sd_predictive_y'][j]);raw=float(d['raw_shift_std'][j]*d['stats']['std'])
    bank=trust_feedback=None
    if d['bank'] is not None:
        bank,m=r1.update_bank(d['bank'],d['prior'],t,raw,mu,sd,y,d['gate_open'])
        state['banks'][d['bank_key']]=bank;d['_measurement']=m
        trust_feedback=r1.trust_feedback(state,d,t,y,bank)
    else:
        residual=y-mu;variance=sd**2+.0009
        m={'feature':0.,'gate_open':False,'prequery_mu_y':mu,'prequery_predictive_sd_y':sd,
           'raw_region_shift_y':raw,'actual_deployed_correction_y':0.,'actual_deployed_mean_y':mu,
           'observed_y':y,'residual':residual,'innovation':residual,'innovation_variance':variance,
           'measurement_variance':variance,'innovation_z':residual/np.sqrt(variance),'gain':0.,
           'coefficient_before':0.,'coefficient_after':0.,'actual_paired_SE_gain':0.,
           'paired_SE_GP':residual**2,'paired_SE_deployed':residual**2,'calibration_target':'no active region'}
    assert abs(m['actual_deployed_correction_y']-d['shift_std'][j]*d['stats']['std'])<1e-12
    pending_check=e.finish_pending(state,plan['pending_forecast'],d,t,y,pool['ids'][j])
    subset_check=e.subset_feedback(state,d,t,y)
    if d['subset_recovery']['applied'] and m['actual_paired_SE_gain'] < -1e-12:
        state.setdefault('subset_evidence',{})[d['bank_key']]={'observations':[], 'reset_query':t,
                                                              'reason':'negative_actual_deployed_gain'}
    best=max(h['y'] for h in state['history'])
    event={'iteration':t,'selected_id':pool['ids'][j], 'normal_proposal_id':pool['ids'][d['normal_proposal']],
        'reference_id':pool['ids'][d['reference']], 'knowledge':deepcopy(state['knowledge']),
        'admission':d['admission'],'region':d['region'],'diagnosis':d['diagnosis'].to_dict(),'retained':d['retained'],
        'amplitude':d['amplitude'],'gate_open':d['gate_open'],'trust':d['trust'],
        'bank_key':d['bank_key'],'bank_before':d['bank'],'bank_prior':d['prior'],'bank_after':bank,
        'budget':d['budget'],'execution_mode':plan['execution_mode'],'normal_gate_exception':False,
        'base_opportunity_loss':float(d['base'][d['reference']]-d['base'][j]),
        'y':y,'raw_value':observed['raw_value'],'incumbent_before':best,'incumbent_after':max(best,y),
        'eligible_count':int(d['eligible'].sum()),'measurement':m,'trust_feedback':trust_feedback,
        'region_trust_after':deepcopy(state['region_trust']),'pending_validation':pending_check,
        'subset_shadow':subset_check,'subset_recovery':deepcopy(d['subset_recovery']),
        'active_suspended_after':state['active_suspended'],'pending_after':deepcopy(state['pending']),
        'promotion':plan['promotion'],'interaction_context':plan['interaction']['context_sha256'] if plan['interaction'] else None}
    if 'e_factor' in observed:event['e_factor']=observed['e_factor']
    e.register_feedback(state,event)
    joint=plan['interaction']
    if joint and joint['answer']['plan']['action']=='test':
        row=joint['answer']['plan'];state['interaction_tests'].append({'query':t,'candidate_id':pool['ids'][j],
            'anchor_candidate_id':joint['prepared']['anchor_candidate_id'],'geometry':row['geometry'],
            'focus_axes':row['focus_axes'],'all_changed_axes':row['all_changed_axes'],
            'background_RMS_normalized':row['background_RMS_normalized'],'observed_y':y,
            'prediction_raw':row['prediction'],'incumbent_before_y':best,
            'base_opportunity_loss':event['base_opportunity_loss'],'budget':d['budget'],'gate_exception':False})
    return event


def bootstrap(state,path,t):
    source=data.read(parent.ROOT/'runs'/(state['task']+'_New_Agent_R1')/'state.json')
    source_folder=parent.ROOT/'runs'/source['run_id']/'rounds'
    source_commit=data.read(source_folder/f'{t:02d}_committed.json')
    assert source_commit['history']==state['history']
    before=deepcopy(state);folder=path.parent/'rounds';folder.mkdir(parents=True,exist_ok=True)
    target=folder/f'{t:02d}_candidates.npz';shutil.copyfile(source_folder/f'{t:02d}_candidates.npz',target)
    receipt={'source_run_id':source['run_id'],'query':t,
             'source_commit_sha256':data.filehash(source_folder/f'{t:02d}_committed.json'),
             'source_candidate_table_sha256':data.filehash(target),'budget_counted':True,
             'source_protocol_sha256':source['protocol_sha256']}
    commit={**deepcopy(source_commit),'protocol_sha256':state['protocol_sha256'],
            'bootstrap_replay':receipt,'state_before_decision':before}
    cp=folder/f'{t:02d}_committed.json';data.write(cp,commit)
    event=deepcopy(source['decisions'][t-1]);event['bootstrap_replay']=receipt;event['commit_sha256']=data.filehash(cp)
    state['knowledge']=deepcopy(event['knowledge']);state['banks'][event['bank_key']]=deepcopy(event['bank_after'])
    state['region_trust']=deepcopy(event['region_trust_after'])
    state['calls'].extend(deepcopy(c) for c in source['calls'] if c['iteration']==t)
    state['refresh_checks'].extend(deepcopy(c) for c in source['refresh_checks'] if c['iteration']==t)
    state['interaction_checks'].extend(deepcopy(c) for c in source['interaction_checks'] if c['iteration']==t)
    state['interaction_tests'].extend(deepcopy(c) for c in source['interaction_tests'] if c['query']==t)
    e.register_feedback(state,event)
    state['history'].append(deepcopy(source['history'][t+3]));state['decisions'].append(event)
    state['bootstrap_receipts'].append(receipt);state['status']='RUNNING';data.write(path,state)
    print(json.dumps({'task':state['task'],'method':state['method'],'query':t,'mode':'protected_R1_prefix_replay',
                      'objective':event['raw_value'],'best':max(h['raw_value'] for h in state['history'])}),flush=True)


def step(state,path,cache,client):
    t=len(state['decisions'])+1;started=time.perf_counter();pool=cache.pool
    before=deepcopy(state)
    plan=decide(state,cache,t,
        lambda own,d,reasons,quota:llm.region_call(client,pool,own,t,d,reasons,quota),
        lambda own,d,prepared:llm.joint_call(client,pool,own,t,d,prepared))
    d=plan['d'];j=d['selected'];joint=plan['interaction'];pending=plan['pending_forecast']
    folder=path.parent/'rounds';folder.mkdir(parents=True,exist_ok=True);table=folder/f'{t:02d}_candidates.npz'
    joint_mask=np.zeros(len(pool['ids']),dtype=bool)
    if joint:
        for row in joint['prepared']['candidates']:joint_mask[cache.ix[row['candidate_id']]]=True
    np.savez_compressed(table,base_qLogEI=d['base'],guided_qLogEI=d['score'],available=d['available'],eligible=d['eligible'],
        mu_y=d['stats']['mu_y'],sd_y=d['stats']['sd_y'],sd_predictive_y=d['stats']['sd_predictive_y'],
        mu_std=d['stats']['mu_std'],sd_std=d['stats']['sd_std'],raw_shift_std=d['raw_shift_std'],deployed_shift_std=d['shift_std'],
        information_proxy=d['information_proxy'],tie_priority=d['tie_priority'],interaction_pool=joint_mask,
        full_region_unqueried=old.membership(pool,state['knowledge']).all(0)&d['available'],
        pending_raw_shift_std=pending['raw_shift_std'] if pending else np.zeros(len(pool['ids'])))
    pending_record={k:v for k,v in pending.items() if k!='raw_shift_std'} if pending else None
    commit={'iteration':t,'selected_id':pool['ids'][j],'normal_proposal_id':pool['ids'][d['normal_proposal']],
        'reference_id':pool['ids'][d['reference']],'protocol_sha256':state['protocol_sha256'],
        'state_before_decision':before,'history':deepcopy(state['history']),'history_sha256':digest(state['history']),
        'candidate_table_sha256':data.filehash(table),'knowledge':deepcopy(state['knowledge']),
        'admission':d['admission'],'region':d['region'],'diagnosis':d['diagnosis'].to_dict(),'retained':d['retained'],
        'amplitude':d['amplitude'],'gate_open':d['gate_open'],'trust':d['trust'],
        'bank_key':d['bank_key'],'bank_before':d['bank'],'bank_prior':d['prior'],'budget':d['budget'],
        'GP_scale_y':d['stats']['std'],'GP_mean_y':d['stats']['mean'],'best_std':d['stats']['best_std'],
        'base_meta':d['base_meta'],'guided_meta':d['guided_meta'],'field_meta':d['field_meta'],
        'execution_mode':plan['execution_mode'],'base_opportunity_loss':float(d['base'][d['reference']]-d['base'][j]),
        'normal_gate_exception':False,'promotion':plan['promotion'],'refresh':plan['refresh'],
        'refresh_check':plan['refresh_check'],'interaction':joint,'interaction_check':plan['interaction_check'],
        'pending_forecast':pending_record,'subset_recovery':d['subset_recovery'],
        'suspension_reason':d.get('suspension_reason'),'inheritance':d['inheritance']}
    cp=folder/f'{t:02d}_committed.json';data.write(cp,commit)
    # First selected outcome lookup is strictly after the complete decision commit.
    observed=data.reveal(state['task'],pool['ids'][j])
    event=feedback(state,cache,plan,t,observed);event['commit_sha256']=data.filehash(cp)
    state['history'].append(observed);state['decisions'].append(event)
    state['status']='COMPLETE' if t==16 else 'RUNNING';state['active_seconds']+=time.perf_counter()-started
    state['operator_compiles_in_loop']=sum(old.JIT.values());assert not state['operator_compiles_in_loop']
    data.write(path,state);cache.release()
    print(json.dumps({'task':state['task'],'method':state['method'],'query':t,'mode':plan['execution_mode'],
        'objective':observed['raw_value'],'best':event['incumbent_after']*pool['objective_scale'],
        'alpha':d['amplitude'],'refreshes':len(state['calls']),
        'pending_status':state['pending']['status'] if state['pending'] else None,
        'subset_recovered':d['subset_recovery']['applied']}),flush=True)


def run_one(task,method,config='../config.txt'):
    old.warm();p=check();path=ROOT/'runs'/(task+'_'+method)/'state.json'
    state=data.read(path) if path.exists() else initial(task,method,p)
    if state['status']=='COMPLETE':return {'task':task,'method':method,'status':'COMPLETE','queries':16}
    assert state['protocol_sha256']==p['protocol_sha256'];state['status']='RUNNING';state.pop('error',None)
    client=api.APIClient(api.load_config(config));cache=e.Cache(frozen.pool(task))
    stable=deepcopy(state)
    try:
        while len(state['decisions'])<16:
            stable=deepcopy(state)
            if len(state['decisions'])<e.BOOTSTRAP[task]:bootstrap(state,path,len(state['decisions'])+1)
            else:step(state,path,cache,client)
    except Exception as error:
        state=stable
        state['status']='INTERRUPTED';state['error']=api.safe_error(error,client.config)
        data.write(path.parent/'error.json',{'error':state['error'],
            'traceback_lines':[api.safe_error(RuntimeError(line),client.config) for line in traceback.format_exc().splitlines()]})
    finally:
        client.close();state['cache_counters']=cache.counters;data.write(path,state)
    return {'task':task,'method':method,'status':state['status'],'queries':len(state['decisions']),'error':state.get('error')}


def run(config='../config.txt'):
    check()
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures=[executor.submit(run_one,task,method,config) for task in TASKS for method in e.METHODS]
        for future in as_completed(futures):print(json.dumps(future.result()),flush=True)
    return status()


def states():return [data.read(p) for p in sorted((ROOT/'runs').glob('*/state.json'))]


def status():
    ss=states();return {'planned':4,'complete':sum(s['status']=='COMPLETE' for s in ss),
        'budgeted_added_queries':sum(len(s['decisions']) for s in ss),
        'runs':[{'task':s['task'],'method':s['method'],'queries':len(s['decisions']),
                 'status':s['status'],'error':s.get('error')} for s in ss]}


def preflight(config='../config.txt'):
    client=api.APIClient(api.load_config(config))
    try:
        result={'configured_model':api.MODEL,'model_available':any(r['id']==api.MODEL for r in client.models()['data'])}
        response=client.http.get('/user/balance')
        if response.status_code==200:
            result['balance']={k:v for k,v in response.json().items() if k in ('is_available','balance_infos')}
        data.write(OUT/'api_preflight.json',result);return result
    finally:client.close()
