"""Only two R1 new-agent trajectories; every original comparator stays sealed."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
from pathlib import Path
import json
import time
import traceback
import numpy as np
from . import continuous_benchmark_data as data
from . import continuous_benchmark_study as frozen
from . import continuous_benchmark_engine as old
from . import revised_continuous_engine as e
from . import revised_continuous_llm as llm
from . import deepseek_suzuki_client as api
from .suzuki_domain import digest

ROOT=e.ROOT;OUT=Path('research/continuous_revised_20261030');TASKS=frozen.TASKS

def sources():
    paths=[Path(__file__),Path(e.__file__),Path(llm.__file__),Path('continuous_revised.py'),
        Path('chem/benchmarks/revised_continuous_report.py'),Path('tests/test_revised_continuous.py'),OUT/'DESIGN.md']
    return {p.as_posix():data.filehash(p) for p in paths}

def protected():
    keep={}
    for folder in (frozen.OUT,Path('research/continuous_audit_20261030')):
        inv=folder/'release_inventory.json';receipt=data.read(folder/'release_receipt.json')
        assert data.filehash(inv)==receipt['inventory_sha256']
        keep[inv.as_posix()]=data.filehash(inv)
        keep[(folder/'release_receipt.json').as_posix()]=data.filehash(folder/'release_receipt.json')
        for row in data.read(inv)['files']:
            assert data.filehash(row['path'])==row['sha256'],row['path']
            keep[row['path']]=row['sha256']
    return keep

def initial(task,p):
    ready=data.read(frozen.ROOT/'initialization'/task/'ready.json')
    return {'task':task,'method':'New_Agent_R1','seed':0,'tie_seed':0,'run_id':task+'_New_Agent_R1',
        'protocol_sha256':p['protocol_sha256'],'history':deepcopy(ready['history']),
        'knowledge':deepcopy(ready['knowledge']),'initial_knowledge':deepcopy(ready['knowledge']),
        'initial_proposal_source_context':ready['context_sha256'],'initial_admission':e.admission(frozen.pool(task),ready['knowledge'],ready['history']),
        'banks':{},'region_trust':{},'calls':[],'refresh_checks':[],'interaction_tests':[],'interaction_checks':[],
        'decisions':[],'status':'READY','active_seconds':0.,'operator_compiles_in_loop':0}

def qualify():
    old.warm();result=[]
    for task in TASKS:
        pool=frozen.pool(task);history=frozen.initial(task);cache=e.Cache(pool);base,meta=cache.score(history);v=cache.model(history)
        path=frozen.ROOT/'runs'/(task+'_New_Agent')/'rounds/01_candidates.npz'
        with np.load(path) as f:
            error=max(float(np.max(abs(base-f['base_qLogEI']))),float(np.max(abs(v['mu_y']-f['mu_y']))),
                float(np.max(abs(v['sd_y']-f['sd_y']))),float(np.max(abs(v['sd_predictive_y']-f['sd_predictive_y']))))
        assert error<1e-9 and cache.counters['fits']==0,(task,error)
        result.append({'task':task,'pool_size':len(pool['ids']),'same_initial_ids':pool['initial_ids'],
            'max_GP_and_base_qLogEI_difference':error,'GP_recipe':cache.recipe,'initial_GP_fit_reused':True,
            'initial_knowledge_reused':True,'old_GP_source_sha256':data.filehash(old.__file__)})
    receipt={'status':'PASS','tasks':result,'operator_compiles_in_loop':sum(old.JIT.values()),'scope':'unchanged GP/input/acquisition qualification, not a performance claim'}
    data.write(OUT/'qualification.json',receipt);return receipt

def prepare(config='../config.txt'):
    if (OUT/'protocol.json').exists():return check()
    assert data.read(OUT/'qualification.json')['status']=='PASS';old_protocol=frozen.check();keep=protected()
    p={'study':'Continuous_R1_deployed_gain_and_joint_tests','tasks':list(TASKS),'methods':['New_Agent_R1'],
        'planned_trajectories':2,'seed':0,'initial_queries':4,'additional_queries':16,'planned_added_queries':32,
        'parameters':e.PARAMETERS,'source_sha256':sources(),'parent_protocol_sha256':old_protocol['protocol_sha256'],
        'input_sha256':old_protocol['input_sha256'],'qualification_sha256':data.filehash(OUT/'qualification.json'),
        'protected_snapshot_sha256':digest(keep),'LLM':api.load_config(config).public(),
        'initial_knowledge_policy':'reuse the original observed-only proposal unchanged; recheck unqueried witnesses; any repair is within the same maximum3 refresh allowance',
        'comparison_policy':'read only all original comparator results and old new-agent results; rerun no baseline',
        'commit_policy':'save all predictions, masks, knowledge, filter prior, LLM choices and hashes before selected outcome lookup',
        'max_logical_LLM_calls':10,'resources':{'persistent_independent_tasks':2,'torch_threads_each':1,
            'GP':'identical CPU exact GP on <=20 observations; frozen checkpoint copies or identical fit routine and seeds',
            'warmup':'same operators once per process before counted loops','operator_compiles_in_loop':0},
        'interpretation':'single initialization per task; revised rules frozen before outcomes; same experimental budget, LLM scheduling may differ; no post-result tuning',
        'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    p['protocol_sha256']=digest(p);data.write(OUT/'protected_snapshot.json',keep);data.write(OUT/'protocol.json',p);return p

def check():
    p=data.read(OUT/'protocol.json');q=deepcopy(p);key=q.pop('protocol_sha256');assert digest(q)==key
    assert p['source_sha256']==sources(),'R1 source changed after freeze'
    for path,h in p['input_sha256'].items():assert data.filehash(path)==h,path
    assert data.filehash(OUT/'qualification.json')==p['qualification_sha256']
    return p

def step(state,path,cache,client):
    t=len(state['decisions'])+1;started=time.perf_counter();pool=cache.pool;d=e.setup(cache,state,t)
    reasons=e.event_reasons(state,d,t);quota=e.refresh_permission(state,t)
    completed_call=next((c for c in state['calls'] if c['iteration']==t),None)
    if not completed_call:
        state['refresh_checks']=[x for x in state['refresh_checks'] if x['iteration']!=t]
        state['refresh_checks'].append({'iteration':t,'reasons':reasons,'quota':quota})
    if reasons and quota['allowed'] and not any(c['iteration']==t for c in state['calls']):
        answer,key=llm.region_call(client,pool,state,t,d,reasons,quota)
        state['calls'].append({'iteration':t,'reasons':reasons,'quota_before':quota,'knowledge_before':deepcopy(state['knowledge']),
            'answer':answer,'context_sha256':key})
        state['knowledge']=answer['plan']['knowledge'];data.write(path,state);d=e.setup(cache,state,t)
    joint=None;mode='normal_R1'
    if t in (2,6) and len(state['interaction_tests'])<2:
        prepared=e.interaction_candidates(cache,state,d,t)
        state['interaction_checks']=[x for x in state['interaction_checks'] if x['iteration']!=t]
        state['interaction_checks'].append({'iteration':t,'candidate_count':len(prepared['candidates']),
            'geometry':prepared['geometry'],'skip_reason':prepared['skip_reason']})
        if prepared['candidates']:
            answer,key=llm.interaction_call(client,pool,state,t,d,prepared)
            joint={'prepared':prepared,'answer':answer,'context_sha256':key};d['selected']=cache.ix[answer['plan']['candidate_id']];mode='bounded_interaction_test'
    j=d['selected'];key=pool['ids'][j];assert d['available'][j] and d['eligible'][j]
    folder=path.parent/'rounds';folder.mkdir(parents=True,exist_ok=True);table=folder/f'{t:02d}_candidates.npz'
    joint_mask=np.zeros(len(pool['ids']),dtype=bool)
    if joint:
        for row in joint['prepared']['candidates']:joint_mask[cache.ix[row['candidate_id']]]=True
    np.savez_compressed(table,base_qLogEI=d['base'],guided_qLogEI=d['score'],available=d['available'],eligible=d['eligible'],
        mu_y=d['stats']['mu_y'],sd_y=d['stats']['sd_y'],sd_predictive_y=d['stats']['sd_predictive_y'],
        mu_std=d['stats']['mu_std'],sd_std=d['stats']['sd_std'],raw_shift_std=d['raw_shift_std'],deployed_shift_std=d['shift_std'],
        information_proxy=d['information_proxy'],tie_priority=d['tie_priority'],interaction_pool=joint_mask,
        full_region_unqueried=old.membership(pool,state['knowledge']).all(0)&d['available'])
    commit={'iteration':t,'selected_id':key,'normal_proposal_id':pool['ids'][d['normal_proposal']],'reference_id':pool['ids'][d['reference']],
        'history':deepcopy(state['history']),'history_sha256':digest(state['history']),'protocol_sha256':state['protocol_sha256'],
        'candidate_table_sha256':data.filehash(table),'knowledge':deepcopy(state['knowledge']),'admission':d['admission'],
        'region':d['region'],'diagnosis':d['diagnosis'].to_dict(),'retained':d['retained'],'amplitude':d['amplitude'],
        'gate_open':d['gate_open'],'trust':d['trust'],'region_trust_at_commit':deepcopy(state['region_trust']),
        'bank_key':d['bank_key'],'bank_before':d['bank'],'bank_prior':d['prior'],'budget':d['budget'],'execution_mode':mode,
        'GP_scale_y':d['stats']['std'],'GP_mean_y':d['stats']['mean'],'best_std':d['stats']['best_std'],
        'base_meta':d['base_meta'],'guided_meta':d['guided_meta'],'field_meta':d['field_meta'],
        'base_opportunity_loss':float(d['base'][d['reference']]-d['base'][j]),'normal_gate_exception':False,
        'interaction':joint,'refresh_check':state['refresh_checks'][-1]}
    cp=folder/f'{t:02d}_committed.json';data.write(cp,commit)
    # No selected result is accessed until the decision has been persisted.
    observed=data.reveal(state['task'],key);y=observed['y'];mu=float(d['stats']['mu_y'][j]);sd=float(d['stats']['sd_predictive_y'][j]);raw=float(d['raw_shift_std'][j]*d['stats']['std'])
    bank=feedback=None
    if d['bank'] is not None:
        bank,measurement=e.update_bank(d['bank'],d['prior'],t,raw,mu,sd,y,d['gate_open'])
        assert abs(measurement['actual_deployed_correction_y']-d['shift_std'][j]*d['stats']['std'])<1e-12
        state['banks'][d['bank_key']]=bank;d['_measurement']=measurement;feedback=e.trust_feedback(state,d,t,y,bank)
    else:
        R=sd*sd+.0009;r=y-mu
        measurement={'feature':0.,'gate_open':False,'prequery_mu_y':mu,'prequery_predictive_sd_y':sd,
            'raw_region_shift_y':raw,'actual_deployed_correction_y':0.,'actual_deployed_mean_y':mu,'observed_y':y,'residual':r,
            'innovation':r,'innovation_variance':R,'measurement_variance':R,'innovation_z':float(r/np.sqrt(R)),
            'gain':0.,'coefficient_before':0.,'coefficient_after_unprojected':0.,'coefficient_after':0.,
            'paired_SE_GP':r*r,'paired_SE_deployed':r*r,'actual_paired_SE_gain':0.,'calibration_target':'no admitted region bank'}
    best=max(h['y'] for h in state['history'])
    if joint:
        plan=joint['answer']['plan'];state['interaction_tests'].append({'query':t,'candidate_id':key,
            'anchor_candidate_id':joint['prepared']['anchor_candidate_id'],'geometry':plan['geometry'],
            'focus_axes':plan['focus_axes'],'all_changed_axes':plan['all_changed_axes'],
            'background_RMS_normalized':plan['background_RMS_normalized'],'observed_y':y,'prediction_raw':plan['prediction'],
            'incumbent_before_y':best,'base_opportunity_loss':commit['base_opportunity_loss'],'budget':d['budget'],'gate_exception':False})
    event={**{k:commit[k] for k in ('iteration','selected_id','normal_proposal_id','reference_id','knowledge','admission','region','diagnosis','retained',
        'amplitude','gate_open','trust','bank_key','bank_before','bank_prior','budget','execution_mode','base_opportunity_loss','normal_gate_exception')},
        'y':y,'raw_value':observed['raw_value'],'incumbent_before':best,'incumbent_after':max(best,y),'eligible_count':int(d['eligible'].sum()),
        'measurement':measurement,'bank_after':bank,'trust_feedback':feedback,'region_trust_after':deepcopy(state['region_trust']),
        'interaction_context':joint['context_sha256'] if joint else None,'commit_sha256':data.filehash(cp)}
    if 'e_factor' in observed:event['e_factor']=observed['e_factor']
    state['history'].append(observed);state['decisions'].append(event);state['status']='COMPLETE' if t==16 else 'RUNNING'
    state['active_seconds']+=time.perf_counter()-started;state['operator_compiles_in_loop']=sum(old.JIT.values());assert not state['operator_compiles_in_loop']
    data.write(path,state);cache.release()
    print(json.dumps({'task':state['task'],'query':t,'mode':mode,'objective':observed['raw_value'],'best':max(best,y)*pool['objective_scale'],
        'alpha':d['amplitude'],'deployed_gain':measurement['actual_paired_SE_gain'],'refreshes':len(state['calls']),
        'remaining_refreshes':3-len(state['calls'])}),flush=True)

def run_one(task,config='../config.txt'):
    old.warm();p=check();path=ROOT/'runs'/(task+'_New_Agent_R1')/'state.json';state=data.read(path) if path.exists() else initial(task,p)
    if state['status']=='COMPLETE':return {'task':task,'status':'COMPLETE','queries':16}
    assert state['protocol_sha256']==p['protocol_sha256'];state['status']='RUNNING';state.pop('error',None);data.write(path,state)
    client=api.APIClient(api.load_config(config));cache=e.Cache(frozen.pool(task))
    try:
        while len(state['decisions'])<16:step(state,path,cache,client)
    except Exception as error:
        state['status']='INTERRUPTED';state['error']=api.safe_error(error,client.config)
        data.write(path.parent/'error.json',{'error':state['error'],'traceback':api.safe_error(RuntimeError(traceback.format_exc()),client.config)})
    finally:
        client.close();state['cache_counters']=cache.counters;data.write(path,state)
    return {'task':task,'status':state['status'],'queries':len(state['decisions']),'error':state.get('error')}

def run(config='../config.txt'):
    check()
    with ProcessPoolExecutor(max_workers=2) as ex:
        for f in as_completed([ex.submit(run_one,task,config) for task in TASKS]):print(json.dumps(f.result()),flush=True)
    return status()

def states():return [data.read(p) for p in sorted((ROOT/'runs').glob('*/state.json'))]
def status():
    ss=states();return {'planned':2,'complete':sum(s['status']=='COMPLETE' for s in ss),'additional_queries':sum(len(s['decisions']) for s in ss),
        'interrupted':[{'task':s['task'],'query':len(s['decisions']),'error':s.get('error')} for s in ss if s['status']=='INTERRUPTED']}

def preflight(config='../config.txt'):
    client=api.APIClient(api.load_config(config))
    try:
        result={'configured_model':api.MODEL,'model_available':any(r['id']==api.MODEL for r in client.models()['data'])}
        response=client.http.get('/user/balance')
        if response.status_code==200:
            raw=response.json();result['balance']={k:v for k,v in raw.items() if k in ('is_available','balance_infos')}
        data.write(OUT/'api_preflight.json',result);return result
    finally:client.close()
