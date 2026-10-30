"""Only two new R3 trajectories. Prior scientific runs are protected read-only."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
from pathlib import Path
import json
import shutil
import time
import traceback
import numpy as np
from . import continuous_benchmark_data as data
from . import continuous_benchmark_study as original
from . import continuous_benchmark_engine as base
from . import continuous_r2_study as parent
from . import continuous_r3_engine as e
from . import continuous_r3_llm as llm
from . import deepseek_suzuki_client as api
from .suzuki_domain import digest

ROOT=e.ROOT;OUT=Path('research/continuous_r3_20261030');TASKS=original.TASKS


def sources():
    names=[Path(__file__),Path(e.__file__),Path(llm.__file__),Path('continuous_r3.py'),
           Path('tests/test_continuous_r3.py'),OUT/'DESIGN.md']
    return {p.as_posix():data.filehash(p) for p in names}


def protected():
    keep=data.read(parent.OUT/'protected_snapshot.json');keep.update(data.read(parent.OUT/'release_inventory.json'))
    for name in ['release_inventory.json','release_receipt.json']:
        path=parent.OUT/name;keep[path.as_posix()]=data.filehash(path)
    diagnosis=Path('research/continuous_r2_diagnosis_20261030')
    keep.update(data.read(diagnosis/'inventory.json'))
    keep[(diagnosis/'inventory.json').as_posix()]=data.filehash(diagnosis/'inventory.json')
    keep['scripts/diagnose_continuous_r2.py']=data.filehash('scripts/diagnose_continuous_r2.py')
    for name,value in keep.items():assert data.filehash(name)==value,name
    return keep


def initial(task,protocol):
    ready=data.read(original.ROOT/'initialization'/task/'ready.json')
    return {'task':task,'method':e.METHOD,'run_id':task+'_'+e.METHOD,'seed':0,'tie_seed':0,
        'protocol_sha256':protocol['protocol_sha256'],'history':deepcopy(ready['history']),
        'initial_knowledge':deepcopy(ready['knowledge']),'hypotheses':{},'authorization_events':[],
        'calls':[],'decisions':[],'bootstrap_receipts':[],'protected_legacy_state':None,
        'legacy_context':{'initial_knowledge':deepcopy(ready['knowledge']),'past_proposals':[]},
        'status':'READY','active_seconds':0.,'operator_compiles_in_loop':0}


def qualify():
    base.warm();results=[]
    for task in TASKS:
        pool=original.pool(task);copied=e.seed_cache(pool);cache=e.Cache(pool)
        source=parent.ROOT/'runs'/(task+'_New_Agent_R2')/'rounds'
        checks=[]
        for t in [1,e.BOOTSTRAP[task]+1]:
            c=data.read(source/f'{t:02d}_committed.json');history=c['history'];scores,_=cache.score(history);v=cache.model(history)
            with np.load(source/f'{t:02d}_candidates.npz') as f:
                error=max(float(np.max(abs(scores-f['base_qLogEI']))),
                          *[float(np.max(abs(v[key]-f[key]))) for key in ['mu_y','sd_y','sd_predictive_y']])
            assert error==0. and cache.counters['fits']==0
            checks.append({'query':t,'max_GP_and_acquisition_difference':error});cache.release()
        results.append({'task':task,'pool_size':len(pool['ids']),'protected_queries':e.BOOTSTRAP[task],
                        'checks':checks,'GP_fits':0,'checkpoints_copied':copied})
    receipt={'status':'PASS','tasks':results,'operator_compiles_in_loop':sum(base.JIT.values())}
    data.write(OUT/'qualification.json',receipt);return receipt


def prepare(config='../config.txt'):
    if (OUT/'protocol.json').exists():return check()
    assert data.read(OUT/'qualification.json')['status']=='PASS';p2=parent.check();keep=protected();bootstrap={}
    for task in TASKS:
        folder=parent.ROOT/'runs'/(task+'_New_Agent_R2')/'rounds'
        for t in range(1,e.BOOTSTRAP[task]+2):
            for suffix in ['committed.json','candidates.npz']:
                path=folder/f'{t:02d}_{suffix}';bootstrap[path.as_posix()]=data.filehash(path)
    protocol={'study':'Continuous_R3_GP_review_and_prospective_local_forecasts','tasks':list(TASKS),
        'methods':[e.METHOD],'planned_trajectories':2,'seed':0,'initial_queries':4,'additional_queries':16,
        'planned_budgeted_added_queries':32,'protected_prefix_positions':7,'new_decisions':25,
        'maximum_fresh_logical_LLM_calls':25,'parameters':e.PARAMETERS,'source_sha256':sources(),
        'parent_protocol_sha256':p2['protocol_sha256'],'input_sha256':p2['input_sha256'],
        'protected_snapshot_sha256':digest(keep),'bootstrap_source_sha256':bootstrap,
        'qualification_sha256':data.filehash(OUT/'qualification.json'),'LLM':api.load_config(config).public(),
        'comparison':'read-only R2, LLINBO_Justify, ReasoningBO, Pure_BO; no comparator reruns',
        'LLM_budget':'one candidate/local-hypothesis call at each fresh R3 decision; experiment budgets matched, LLM call counts differ',
        'commit_policy':'freeze all forecasts, review, local evidence and choice before selected outcome lookup',
        'resources':{'persistent_streams':2,'torch_threads_each':1,'GP':'unchanged CPU exact GP <=20 observations',
                     'warmup':'once per process before trajectory loop','operator_compiles_in_loop':0},
        'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'interpretation':'protected-prefix continuation on one seed; a positive local prediction check is not proof of a chemical law'}
    protocol['protocol_sha256']=digest(protocol);data.write(OUT/'protected_snapshot.json',keep);data.write(OUT/'protocol.json',protocol)
    return protocol


def check():
    p=data.read(OUT/'protocol.json');q=deepcopy(p);key=q.pop('protocol_sha256')
    assert digest(q)==key and p['source_sha256']==sources(),'R3 source/protocol changed after freeze'
    for name,h in p['input_sha256'].items():assert data.filehash(name)==h,name
    for name,h in p['bootstrap_source_sha256'].items():assert data.filehash(name)==h,name
    assert data.filehash(OUT/'qualification.json')==p['qualification_sha256']
    return p


def bootstrap(state,path,t):
    task=state['task'];source_folder=parent.ROOT/'runs'/(task+'_New_Agent_R2')/'rounds'
    source=data.read(parent.ROOT/'runs'/(task+'_New_Agent_R2')/'state.json')
    original_commit=data.read(source_folder/f'{t:02d}_committed.json')
    assert original_commit['history']==state['history']
    after=data.read(source_folder/f'{t+1:02d}_committed.json')['state_before_decision']
    assert after['history']==source['history'][:t+4]
    before=deepcopy(state);folder=path.parent/'rounds';folder.mkdir(parents=True,exist_ok=True)
    table=folder/f'{t:02d}_candidates.npz';shutil.copyfile(source_folder/f'{t:02d}_candidates.npz',table)
    receipt={'source_run_id':source['run_id'],'query':t,'source_commit_sha256':data.filehash(source_folder/f'{t:02d}_committed.json'),
             'source_array_sha256':data.filehash(table),'source_post_state_sha256':digest(after),'budget_counted':True}
    commit={**deepcopy(original_commit),'protocol_sha256':state['protocol_sha256'],
        'legacy_state_before':deepcopy(original_commit['state_before_decision']),
        'state_before_decision':before,'protected_R2_replay':receipt,'protected_legacy_state_after':after}
    cp=folder/f'{t:02d}_committed.json';data.write(cp,commit)
    event=deepcopy(source['decisions'][t-1]);event['protected_R2_replay']=receipt;event['commit_sha256']=data.filehash(cp)
    state['history'].append(deepcopy(source['history'][t+3]));state['decisions'].append(event)
    state['protected_legacy_state']=deepcopy(after);state['bootstrap_receipts'].append(receipt)
    state['legacy_context']={'initial_knowledge':deepcopy(after['initial_knowledge']),
        'last_prefix_active_knowledge':deepcopy(after['knowledge']),
        'past_proposals':[{'query':c['iteration'],'knowledge':c['answer']['plan']['knowledge']}
                          for c in after['calls'] if c['iteration']<=t],
        'past_shadow_validations':[deepcopy(d['pending_validation']) for d in after['decisions'] if d.get('pending_validation')]}
    state['status']='RUNNING';data.write(path,state)
    if t==e.BOOTSTRAP[task]:data.write(path.parent/'protected_prefix_state.json',after)
    print(json.dumps({'task':task,'query':t,'mode':'protected_R2_prefix','objective':event['raw_value'],
                      'best':max(h['raw_value'] for h in state['history'])}),flush=True)


def plan(state,cache,t,provider):
    authorizations=e.authorize(state,t);d=e.setup(cache,state,t);card=e.candidate_card(cache,state,d)
    answer,key=provider(state,d,t,card)
    decision=e.decide(cache,state,d,t,answer,key)
    state['calls'].append({'query':t,'context_sha256':key,'answer':deepcopy(answer),
                           'card_size':len(card),'GP_review_accepted':decision['GP_review_accepted'],
                           'registered_hypothesis':decision['registered_hypothesis']})
    return {'d':d,'decision':decision,'answer':answer,'context_sha256':key,'card':card,'authorizations':authorizations}


def apply_feedback(state,cache,prepared,t,observed):
    d=prepared['d'];decision=prepared['decision'];j=decision['selected'];pool=cache.pool
    outcome=e.feedback(state,d,decision,t,observed['y'],pool['ids'][j]);g=decision['guidance'];best=max(h['y'] for h in state['history'])
    actual=outcome['actual_local_update'];m=actual['measurement'] if actual else None
    mu=outcome['GP_mean_y'];sd=outcome['GP_predictive_SD_y'];R=sd**2+.0009
    measurement={'prequery_mu_y':mu,'prequery_predictive_sd_y':sd,
        'raw_region_shift_y':m['feature'] if m else 0.,
        'innovation_z':m['innovation_z'] if m else float((observed['y']-mu)/np.sqrt(R)),
        'actual_deployed_correction_y':outcome['deployed_shift_y'],'actual_deployed_mean_y':outcome['deployed_mean_y'],
        'actual_paired_SE_gain':outcome['actual_paired_SE_gain'],
        'coefficient_before':m['coefficient_before'] if m else 0.,'coefficient_after':m['coefficient_after'] if m else 0.}
    event={'iteration':t,'selected_id':pool['ids'][j],'reference_id':pool['ids'][d['reference']],
        'guided_proposal_id':pool['ids'][g['guided']],
        'LLM_proposed_id':pool['ids'][decision['proposed_index']] if decision['proposed_index'] is not None else None,
        'execution_mode':decision['mode'],'GP_review_accepted':decision['GP_review_accepted'],
        'GP_review_loss':decision['GP_review_loss'],'local_ranking_changed':decision['local_ranking_changed'],
        'proposal_selected':decision['proposal_selected'],'registered_hypothesis':decision['registered_hypothesis'],
        'base_opportunity_loss':float(d['base'][d['reference']]-d['base'][j]),'budget':d['budget'],
        'eligible_count':int(d['eligible'].sum()),'normal_gate_exception':False,
        'y':observed['y'],'raw_value':observed['raw_value'],'incumbent_before':best,'incumbent_after':max(best,observed['y']),
        'measurement':measurement,'local_feedback':outcome,'authorizations':deepcopy(prepared['authorizations']),
        'active_hypotheses_at_commit':[p['hypothesis_id'] for p in g['priors'] if p['status'] in ('probation','supported') and p['active_cap']>0],
        'context_sha256':prepared['context_sha256']}
    if 'e_factor' in observed:event['e_factor']=observed['e_factor']
    return event


def step(state,path,cache,client):
    t=len(state['decisions'])+1;started=time.perf_counter();before=deepcopy(state)
    prepared=plan(state,cache,t,lambda own,d,query,card:llm.request(client,cache.pool,own,d,query,card))
    d=prepared['d'];decision=prepared['decision'];g=decision['guidance'];pool=cache.pool
    folder=path.parent/'rounds';folder.mkdir(parents=True,exist_ok=True);table=folder/f'{t:02d}_candidates.npz'
    np.savez_compressed(table,base_qLogEI=d['base'],guided_qLogEI=g['score'],available=d['available'],eligible=d['eligible'],
        mu_y=d['stats']['mu_y'],sd_y=d['stats']['sd_y'],sd_predictive_y=d['stats']['sd_predictive_y'],
        mu_std=d['stats']['mu_std'],sd_std=d['stats']['sd_std'],information_proxy=d['information_proxy'],tie_priority=d['tie_priority'],
        local_features_y=g['features_y'],local_weights=g['weights'],local_owner=g['owner'],local_shift_y=g['shift_y'],
        local_coefficient=g['coefficient'])
    commit={'iteration':t,'selected_id':pool['ids'][decision['selected']],'reference_id':pool['ids'][d['reference']],
        'guided_proposal_id':pool['ids'][g['guided']],'protocol_sha256':state['protocol_sha256'],
        'history':deepcopy(state['history']),'history_sha256':digest(state['history']),'state_before_decision':before,
        'candidate_table_sha256':data.filehash(table),'base_meta':d['base_meta'],'guided_meta':g['meta'],
        'GP_scale_y':d['stats']['std'],'GP_mean_y':d['stats']['mean'],'best_std':d['stats']['best_std'],
        'budget':d['budget'],'candidate_card':prepared['card'],'answer':prepared['answer'],'context_sha256':prepared['context_sha256'],
        'authorizations':prepared['authorizations'],'local_hypothesis_ids':g['keys'],'local_priors':g['priors'],
        'local_models_at_commit':deepcopy(state['hypotheses']),
        'decision':{key:value for key,value in decision.items() if key!='guidance'}}
    cp=folder/f'{t:02d}_committed.json';data.write(cp,commit)
    # No outcome is accessed until all GP and local forecasts and choice are frozen.
    observed=data.reveal(state['task'],commit['selected_id'])
    event=apply_feedback(state,cache,prepared,t,observed);event['commit_sha256']=data.filehash(cp)
    state['history'].append(observed);state['decisions'].append(event);state['status']='COMPLETE' if t==16 else 'RUNNING'
    state['active_seconds']+=time.perf_counter()-started;state['operator_compiles_in_loop']=sum(base.JIT.values())
    assert state['operator_compiles_in_loop']==0;data.write(path,state);cache.release()
    print(json.dumps({'task':state['task'],'query':t,'mode':event['execution_mode'],'objective':event['raw_value'],
        'best':event['incumbent_after']*pool['objective_scale'],'GP_accepted':event['GP_review_accepted'],
        'local_ranking_changed':event['local_ranking_changed'],'local_owner':event['local_feedback']['actual_owner_hypothesis'],
        'hypotheses':len(state['hypotheses']),'authorized':len(event['active_hypotheses_at_commit'])}),flush=True)


def run_one(task,config='../config.txt'):
    base.warm();protocol=check();path=ROOT/'runs'/(task+'_'+e.METHOD)/'state.json'
    state=data.read(path) if path.exists() else initial(task,protocol)
    if state['status']=='COMPLETE':return {'task':task,'status':'COMPLETE','queries':16}
    assert state['protocol_sha256']==protocol['protocol_sha256'];state['status']='RUNNING';state.pop('error',None)
    client=api.APIClient(api.load_config(config));cache=e.Cache(original.pool(task));stable=deepcopy(state)
    try:
        while len(state['decisions'])<16:
            stable=deepcopy(state)
            if len(state['decisions'])<e.BOOTSTRAP[task]:bootstrap(state,path,len(state['decisions'])+1)
            else:step(state,path,cache,client)
    except Exception as error:
        state=stable;state['status']='INTERRUPTED';state['error']=api.safe_error(error,client.config)
        data.write(path.parent/'error.json',{'error':state['error'],
            'traceback_lines':[api.safe_error(RuntimeError(line),client.config) for line in traceback.format_exc().splitlines()]})
    finally:
        client.close();state['cache_counters']=cache.counters;data.write(path,state)
    return {'task':task,'status':state['status'],'queries':len(state['decisions']),'error':state.get('error')}


def run(config='../config.txt'):
    check()
    with ProcessPoolExecutor(max_workers=2) as executor:
        for future in as_completed([executor.submit(run_one,task,config) for task in TASKS]):print(json.dumps(future.result()),flush=True)
    return status()


def states():return [data.read(p) for p in sorted((ROOT/'runs').glob('*/state.json'))]


def status():
    ss=states();return {'planned':2,'complete':sum(s['status']=='COMPLETE' for s in ss),
        'budgeted_added_queries':sum(len(s['decisions']) for s in ss),
        'runs':[{'task':s['task'],'queries':len(s['decisions']),'status':s['status'],'error':s.get('error')} for s in ss]}


def preflight(config='../config.txt'):
    client=api.APIClient(api.load_config(config))
    try:
        result={'configured_model':api.MODEL,'model_available':any(row['id']==api.MODEL for row in client.models()['data'])}
        response=client.http.get('/user/balance')
        if response.status_code==200:result['balance']={k:v for k,v in response.json().items() if k in ('is_available','balance_infos')}
        data.write(OUT/'api_preflight.json',result);return result
    finally:client.close()
