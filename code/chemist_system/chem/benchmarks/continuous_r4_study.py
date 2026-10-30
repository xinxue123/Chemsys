"""Two frozen R4 continuations; all comparator trajectories remain read-only."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
from pathlib import Path
import json
import time
import traceback
import numpy as np
from . import continuous_r3_study as parent
from . import continuous_r4_engine as e
from . import continuous_r4_llm as llm
from . import continuous_benchmark_data as data
from . import continuous_benchmark_study as original
from . import continuous_benchmark_engine as base
from . import deepseek_suzuki_client as api
from .suzuki_domain import digest

ROOT=e.ROOT;OUT=Path('research/continuous_r4_20261030');TASKS=original.TASKS


def sources():
    return {p.as_posix():data.filehash(p) for p in [Path(__file__),Path(e.__file__),Path(llm.__file__),
        Path('continuous_r4.py'),Path('tests/test_continuous_r4.py'),OUT/'DESIGN.md']}


def runtime_sources():
    names=['continuous_r3_engine','continuous_r3_study','continuous_benchmark_engine','continuous_benchmark_data',
        'continuous_benchmark_llm','revised_continuous_engine','reaction_transfer_llm','deepseek_suzuki_client',
        'stable_information_engine','suzuki_domain']
    return {str(Path('chem/benchmarks')/(n+'.py')).replace('\\','/'):data.filehash(Path('chem/benchmarks')/(n+'.py')) for n in names}


def protected():
    diagnosis=Path('research/continuous_r3_diagnosis_20261030')
    keep=data.read(diagnosis/'source_snapshot.json');keep.update(data.read(diagnosis/'inventory.json'))
    for name in ['inventory.json','receipt.json']:
        path=diagnosis/name;keep[path.as_posix()]=data.filehash(path)
    for name,value in keep.items():assert data.filehash(name)==value,name
    return keep


def initial(task,p):
    state=parent.initial(task,p);state['method']=e.METHOD;state['run_id']=task+'_'+e.METHOD
    state.update(evidence_ledger=[],inheritance_events=[],scheduled_validations=[])
    return state


def qualify():
    base.warm();rows=[]
    for task in TASKS:
        pool=original.pool(task);copies=e.seed_cache(pool);cache=e.Cache(pool);checks=[]
        source=parent.ROOT/'runs'/(task+'_New_Agent_R3')/'rounds'
        for t in [1,e.BOOTSTRAP[task]+1]:
            c=data.read(source/f'{t:02d}_committed.json');scores,_=cache.score(c['history']);stats=cache.model(c['history'])
            with np.load(source/f'{t:02d}_candidates.npz') as f:
                err=max(float(np.max(abs(scores-f['base_qLogEI']))),*[float(np.max(abs(stats[k]-f[k]))) for k in ['mu_y','sd_y','sd_predictive_y']])
            assert err==0. and cache.counters['fits']==0
            checks.append({'query':t,'max_GP_and_acquisition_difference':err});cache.release()
        rows.append({'task':task,'checks':checks,'GP_fits':0,'checkpoint_copies':copies})
    result={'status':'PASS','tasks':rows,'operator_compiles_in_loop':sum(base.JIT.values())}
    data.write(OUT/'qualification.json',result);return result


def prepare(config='../config.txt'):
    if (OUT/'protocol.json').exists():return check()
    qualification=data.read(OUT/'qualification.json');assert qualification['status']=='PASS'
    p3=parent.check();keep=protected()
    p={'study':'Continuous_R4_blind_proposal_local_validation_shared_evidence','tasks':list(TASKS),
        'methods':[e.METHOD],'planned_trajectories':2,'seed':0,'initial_queries':4,'additional_queries':16,
        'planned_budgeted_added_queries':32,'protected_prefix_positions':7,'new_decisions':25,
        'maximum_fresh_logical_LLM_calls':8,'parameters':e.PARAMETERS,'source_sha256':sources(),
        'runtime_sha256':runtime_sources(),'parent_protocol_sha256':p3['protocol_sha256'],
        'input_sha256':p3['input_sha256'],'bootstrap_source_sha256':p3['bootstrap_source_sha256'],
        'protected_snapshot_sha256':digest(keep),'qualification_sha256':data.filehash(OUT/'qualification.json'),
        'LLM':api.load_config(config).public(),'comparison':'read-only R3,R2,LLINBO_Justify,ReasoningBO,Pure_BO',
        'commit_policy':'blind proposal before current GP; final forecasts and choice before selected observation lookup',
        'LLM_budget':'up to4 blind calls per task, eight logical calls total; only experiment budgets matched',
        'resources':{'persistent_streams':2,'torch_threads_each':1,'GP':'unchanged CPU exact GP, <=20 observations',
            'warmup':'once per process, zero operator compilation in the trajectory loop'},
        'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'interpretation':'single-seed protected-prefix continuation; successful validation is not scientific superiority'}
    p['protocol_sha256']=digest(p);data.write(OUT/'protected_snapshot.json',keep);data.write(OUT/'protocol.json',p);return p


def check():
    p=data.read(OUT/'protocol.json');q=deepcopy(p);key=q.pop('protocol_sha256')
    assert digest(q)==key and p['source_sha256']==sources() and p['runtime_sha256']==runtime_sources(),'R4 frozen source changed'
    for group in ['input_sha256','bootstrap_source_sha256']:
        for name,h in p[group].items():assert data.filehash(name)==h,name
    assert data.filehash(OUT/'qualification.json')==p['qualification_sha256'];return p


def plan(state,cache,t,provider,proposal_commit=None):
    auth,inherit=e.authorize(cache.pool,state,t);permission=e.call_permission(state,t)
    answer=key=registered=projected=None
    if permission['allowed']:
        answer,key=provider(state,t)  # Intentionally no current GP, acquisition or review object.
        registered,projected=e.register(cache.pool,state,t,answer['plan'],key)
        changes=e.synchronize(cache.pool,state,t);state['inheritance_events'].extend(deepcopy(changes));inherit+=changes
        state['calls'].append({'query':t,'context_sha256':key,'answer':deepcopy(answer),
            'registered_hypothesis':registered,'projection':deepcopy(projected),'proposal_precedes_current_GP':True})
    receipt={'query':t,'history_sha256':digest(state['history']),'call_permission':permission,
        'answer':deepcopy(answer),'context_sha256':key,'projection':deepcopy(projected),
        'registered_hypothesis':registered,'GP_current_round_computed':False}
    if proposal_commit is not None:proposal_commit(deepcopy(receipt))
    d=e.setup(cache,state,t);g=e.guidance(cache,state,d,t);decision=e.decide(cache,state,d,g,t,projected)
    return {'d':d,'g':g,'decision':decision,'answer':answer,'context_sha256':key,'projection':projected,
        'registered_hypothesis':registered,'call_permission':permission,'authorizations':auth,
        'inheritance_changes':inherit,'blind_proposal_receipt':receipt}


def apply_feedback(state,cache,p,t,observed):
    d=p['d'];g=p['g'];decision=p['decision'];j=decision['selected'];pool=cache.pool
    outcome=e.feedback(pool,state,d,g,decision,t,observed['y']);actual=outcome['actual_local_update']
    m=actual['measurement'] if actual else None;mu=outcome['GP_mean_y'];sd=outcome['GP_predictive_SD_y'];R=sd**2+.0009
    measurement={'prequery_mu_y':mu,'prequery_predictive_sd_y':sd,'raw_region_shift_y':m['feature'] if m else 0.,
        'innovation_z':m['innovation_z'] if m else float((observed['y']-mu)/np.sqrt(R)),
        'actual_deployed_correction_y':outcome['deployed_shift_y'],'actual_deployed_mean_y':outcome['deployed_mean_y'],
        'actual_paired_SE_gain':outcome['actual_paired_SE_gain'],
        'coefficient_before':m['coefficient_before'] if m else 0.,'coefficient_after':m['coefficient_after'] if m else 0.}
    best=max(h['y'] for h in state['history']);proposal=decision['proposed_index']
    event={'iteration':t,'selected_id':pool['ids'][j],'reference_id':pool['ids'][d['reference']],
        'guided_proposal_id':pool['ids'][g['guided']],'LLM_proposed_id':pool['ids'][proposal] if proposal is not None else None,
        'execution_mode':decision['mode'],'GP_review_accepted':decision['GP_review_accepted'],
        'GP_review_loss':decision['GP_review_loss'],'local_ranking_changed':decision['local_ranking_changed'],
        'proposal_selected':decision['proposal_selected'],'registered_hypothesis':p['registered_hypothesis'],
        'base_opportunity_loss':float(d['base'][d['reference']]-d['base'][j]),'budget':d['budget'],
        'eligible_count':int(d['eligible'].sum()),'normal_gate_exception':False,'y':observed['y'],'raw_value':observed['raw_value'],
        'incumbent_before':best,'incumbent_after':max(best,observed['y']),'measurement':measurement,'local_feedback':outcome,
        'authorizations':p['authorizations'],'inheritance_changes_before_result':p['inheritance_changes'],
        'active_hypotheses_at_commit':[v['hypothesis_id'] for v in g['priors'] if v['active_cap']>0],
        'context_sha256':p['context_sha256'],'call_permission':p['call_permission'],'validation':decision['validation']}
    if 'e_factor' in observed:event['e_factor']=observed['e_factor']
    return event


def step(state,path,cache,client):
    t=len(state['decisions'])+1;started=time.perf_counter();before=deepcopy(state)
    folder=path.parent/'rounds';folder.mkdir(parents=True,exist_ok=True);proposal_file=folder/f'{t:02d}_blind_proposal.json'
    def commit_proposal(receipt):
        if proposal_file.exists():assert digest(data.read(proposal_file))==digest(receipt)
        else:data.write(proposal_file,receipt)
    p=plan(state,cache,t,lambda own,query:llm.request(client,cache.pool,own,query),commit_proposal)
    d=p['d'];g=p['g'];decision=p['decision'];pool=cache.pool;table=folder/f'{t:02d}_candidates.npz'
    validation_mask=np.zeros(len(pool['ids']),dtype=bool)
    for row in decision['validation']['candidates']:validation_mask[row['index']]=True
    np.savez_compressed(table,base_qLogEI=d['base'],guided_qLogEI=g['score'],available=d['available'],eligible=d['eligible'],
        mu_y=d['stats']['mu_y'],sd_y=d['stats']['sd_y'],sd_predictive_y=d['stats']['sd_predictive_y'],
        mu_std=d['stats']['mu_std'],sd_std=d['stats']['sd_std'],information_proxy=d['information_proxy'],tie_priority=d['tie_priority'],
        local_features_y=g['features_y'],local_weights=g['weights'],local_owner=g['owner'],local_shift_y=g['shift_y'],
        local_coefficient=g['coefficient'],validation_candidate=validation_mask)
    commit={'iteration':t,'selected_id':pool['ids'][decision['selected']],'reference_id':pool['ids'][d['reference']],
        'guided_proposal_id':pool['ids'][g['guided']],'protocol_sha256':state['protocol_sha256'],
        'history':deepcopy(state['history']),'history_sha256':digest(state['history']),'state_before_decision':before,
        'candidate_table_sha256':data.filehash(table),'blind_proposal_sha256':data.filehash(proposal_file),
        'base_meta':d['base_meta'],'guided_meta':g['meta'],'GP_scale_y':d['stats']['std'],'GP_mean_y':d['stats']['mean'],
        'best_std':d['stats']['best_std'],'budget':d['budget'],'answer':p['answer'],'context_sha256':p['context_sha256'],
        'projection':p['projection'],'call_permission':p['call_permission'],'authorizations':p['authorizations'],
        'inheritance_changes':p['inheritance_changes'],'local_hypothesis_ids':g['keys'],'local_priors':g['priors'],
        'local_models_at_commit':deepcopy(state['hypotheses']),'decision':deepcopy(decision)}
    cp=folder/f'{t:02d}_committed.json';data.write(cp,commit)
    observed=data.reveal(state['task'],commit['selected_id'])  # All predictions and choices already committed.
    event=apply_feedback(state,cache,p,t,observed);event['commit_sha256']=data.filehash(cp)
    state['history'].append(observed);state['decisions'].append(event);state['status']='COMPLETE' if t==16 else 'RUNNING'
    state['active_seconds']+=time.perf_counter()-started;state['operator_compiles_in_loop']=sum(base.JIT.values())
    assert state['operator_compiles_in_loop']==0;data.write(path,state);cache.release()
    print(json.dumps({'task':state['task'],'query':t,'mode':event['execution_mode'],'objective':event['raw_value'],
        'best':event['incumbent_after']*pool['objective_scale'],'fresh_LLM_calls':len(state['calls']),
        'GP_accepted':event['GP_review_accepted'],'scheduled_tests':len(state['scheduled_validations']),
        'local_ranking_changed':event['local_ranking_changed'],'local_owner':outcome_owner(event)},ensure_ascii=False),flush=True)


def outcome_owner(event):return event['local_feedback']['actual_owner_hypothesis']


def run_one(task,config='../config.txt'):
    base.warm();p=check();path=ROOT/'runs'/(task+'_'+e.METHOD)/'state.json'
    state=data.read(path) if path.exists() else initial(task,p)
    if state['status']=='COMPLETE':return {'task':task,'status':'COMPLETE','queries':16}
    assert state['protocol_sha256']==p['protocol_sha256'];state['status']='RUNNING';state.pop('error',None)
    client=api.APIClient(api.load_config(config));cache=e.Cache(original.pool(task));stable=deepcopy(state)
    try:
        while len(state['decisions'])<16:
            stable=deepcopy(state)
            if len(state['decisions'])<e.BOOTSTRAP[task]:parent.bootstrap(state,path,len(state['decisions'])+1)
            else:step(state,path,cache,client)
    except Exception as error:
        state=stable;state['status']='INTERRUPTED';state['error']=api.safe_error(error,client.config)
        data.write(path.parent/'error.json',{'error':state['error'],
            'traceback_lines':[api.safe_error(RuntimeError(line),client.config) for line in traceback.format_exc().splitlines()]})
    finally:client.close();state['cache_counters']=cache.counters;data.write(path,state)
    return {'task':task,'status':state['status'],'queries':len(state['decisions']),'error':state.get('error')}


def run(config='../config.txt'):
    check()
    with ProcessPoolExecutor(max_workers=2) as ex:
        for f in as_completed([ex.submit(run_one,task,config) for task in TASKS]):print(json.dumps(f.result()),flush=True)
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
