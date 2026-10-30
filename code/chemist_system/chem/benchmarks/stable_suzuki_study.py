"""Frozen Stage11 transport, sixteen full Reizman trajectories, seed0/4+16."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import time
import traceback
import numpy as np
import torch

from . import stable_suzuki_engine as engine
from . import stable_suzuki_policy as policy
from . import mixed_comparison_suzuki_study as prior
from . import grounded_regional_suzuki_study as grounded
from . import reaction_transfer_llm as transport
from . import stable_feedback_requests as canonical
from . import deepseek_suzuki_client as api
from .suzuki_domain import digest

ROOT=Path('data/runs/suzuki_stable_20261029')
OUT=Path('research/suzuki_stable_20261029')
REGIMES=prior.REGIMES
read,write,filehash=prior.read,prior.write,prior.filehash
WORKERS=4


def source_hashes():
    paths=[Path(__file__),Path(engine.__file__),Path(policy.__file__),Path('stable_suzuki.py'),
        Path('chem/benchmarks/stable_suzuki_report.py'),Path('tests/test_stable_suzuki.py'),
        Path(grounded.e.__file__),Path(engine.original.__file__),Path(engine.original.common.__file__),
        Path(engine.original.common.mixed.__file__),Path(engine.trust.__file__),Path(canonical.__file__),
        Path(transport.__file__),Path(api.__file__),Path('chem/decision/domain_guard.py'),
        Path('chem/benchmarks/stable_information_engine.py'),Path('chem/benchmarks/r2_tuning_policy.py'),
        Path('third_party/suzuki_official/lgbo/prior.py'),OUT/'DESIGN.md']
    return {p.as_posix():filehash(p) for p in paths}


def old_inventories():
    return [Path('research/suzuki_grounded_regional/release_inventory.json'),
        Path('research/stable_r2_20261029/milestone_inventory.json')]


def protected_files():
    result={}
    for inv in old_inventories():
        rows=read(inv)['files'];result[inv.as_posix()]=filehash(inv)
        for row in rows:
            assert filehash(row['path'])==row['sha256'],row['path']
            result[row['path']]=row['sha256']
    for p in [Path('research/milestones/M1_Reizman_seed0_20261028/milestone.json'),
              Path('research/suzuki_mixed_comparison/results.json'),Path('research/suzuki_mixed_kalman/results.json'),
              Path('research/suzuki_regional_ablation/results.json')]:result[p.as_posix()]=filehash(p)
    return result


def cache_for(case,pool,permutation=tuple(range(8)),qualification=False):
    group=f'c{case}_s0';directory=ROOT/('qualification_cache' if qualification else 'cache')/group
    if tuple(permutation)!=tuple(range(8)):directory=directory/'permuted'
    roots=[grounded.ROOT,grounded.previous.ROOT,prior.ROOT,prior.previous.ROOT]
    sources=[root/f/group for root in roots for f in ('cache','qualification_cache','audit_cache')]
    if not qualification:sources.insert(0,ROOT/'qualification_cache'/group)
    return engine.Cache(pool,directory,case,category_map=permutation,sources=sources)


def warm():prior.worker_init()


def initial(case,regime,p,pool):
    run_id=f'c{case}_s0_{regime}_stable';knowledge=grounded.previous.initial_knowledge(case,regime)
    return {'case':case,'regime':regime,'method':'Stable_revised','seed':0,'tie_seed':0,'run_id':run_id,
        'task':p['task_codes'][run_id],'protocol_sha256':p['protocol_sha256'],'status':'READY',
        'history':prior.initial(case,pool),'knowledge':knowledge,'initial_knowledge':deepcopy(knowledge),
        'slots':{h['id']:h['coordinate'] for h in knowledge['clauses']},'banks':{},'region_trust':{},
        'calls':[],'validations':[],'validation_checks':[],'decisions':[],'active_seconds':0.,'operator_compiles_in_loop':0}


def qualify():
    warm();rows=[]
    for case in (1,2,3,4):
        pool=prior.inputs(case);h=prior.initial(case,pool);c=cache_for(case,pool,qualification=True)
        perm=cache_for(case,pool,(19,-3,101,7,0,12,-9,4),True)
        a,_=c.score(h);b,_=perm.score(h)
        sa=c.posterior_stats(h);sb=perm.posterior_stats(h)
        err=max(float(np.max(np.abs(a-b))),float(np.max(np.abs(sa['mu_y']-sb['mu_y']))),float(np.max(np.abs(sa['sd_y']-sb['sd_y']))))
        assert err<1e-9
        knowledge=grounded.previous.initial_knowledge(case,'assistant_initial');kept=[x['id'] for x in knowledge['clauses']]
        spec=engine.original.region(knowledge,kept);tilt,v,_=c.make_tilt(h,spec);x=c.z[:19].unsqueeze(1)
        with torch.no_grad():cached=tilt.posterior(v['model'],x).mean.clone()
        saved=tilt._cache;tilt._cache=None
        with torch.no_grad():uncached=tilt.posterior(v['model'],x).mean.clone()
        tilt._cache=saved;unit_error=float((cached-uncached).abs().max());assert unit_error<1e-8
        ga,ra,_=c.calibrated_score(h,knowledge,kept,.63);gb,rb,_=perm.calibrated_score(h,knowledge,kept,.63)
        guided_error=max(float(np.max(np.abs(ga-gb))),float(np.max(np.abs(ra-rb))));assert guided_error<1e-8
        rows.append({'case':case,'candidate_count':len(pool['ids']),'category_relabel_error':err,
            'guided_relabel_error':guided_error,'cached_uncached_tilt_mean_error_std_units':unit_error,
            'outcome_covariance_multiplier':float(v['model'].outcome_transform.stdvs.square()),'initial_ids':pool['initial_ids']['0']})
        c.release();perm.release()
    result={'status':'PASS','cases':rows,'operator_compiles_in_loop':0};write(OUT/'qualification.json',result);return result


def prepare(config_path='../config.txt'):
    if (OUT/'protocol.json').exists():return check()
    assert read(OUT/'qualification.json')['status']=='PASS'
    kept=protected_files();manifest=[{'case':c,'regime':r,'run_id':f'c{c}_s0_{r}_stable'} for c in (1,2,3,4) for r in REGIMES]
    task_codes={row['run_id']:f"T{k:02d}" for k,row in enumerate(sorted(manifest,key=lambda row:digest(row['run_id'])))}
    inputs={}
    for case in (1,2,3,4):
        for name in ('pool.json','advice.json','oracle.npz'):
            path=prior.original.ROOT/'inputs'/f'case{case}'/name;inputs[path.as_posix()]=filehash(path)
    p={'study':'Reizman_Stage11_mixed_transport','cases':[1,2,3,4],'regimes':list(REGIMES),
        'seed':0,'tie_seed':0,'initial_queries':4,'additional_queries':16,'planned_trajectories':16,
        'planned_logical_observations':320,'planned_new_observations':256,'candidate_count':4692,
        'parameters':engine.PARAMETERS,'task_codes':task_codes,'LLM':api.load_config(config_path).public(),
        'request_parameters':transport.PARAMETERS,'max_refreshes_per_trace':3,'validation_rounds':[2,4,6],
        'max_new_logical_LLM_calls':96,'sources':source_hashes(),'input_sha256':inputs,
        'qualification_sha256':filehash(OUT/'qualification.json'),'protected_snapshot_sha256':digest(kept),
        'baseline_policy':'read-only frozen original Suzuki baselines; no baseline reruns or selection',
        'controlled_policy':'exact single-axis fixed-pool neighbors only; declared exceptions within16; empty pool skips with receipt',
        'molecular_policy':'not applicable without real structures or a second nominal context axis; no fabricated molecular encoding',
        'resources':{'CPU_processes':WORKERS,'Torch_threads_each':1,'warmup':'Mixed kernel and Cholesky before counted loops','operator_compiles_in_loop':0},
        'interpretation':'single fixed initialization, four prior-knowledge conditions; no multi-seed significance or universal transfer claim',
        'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    p['protocol_sha256']=digest(p);write(OUT/'protected_snapshot.json',kept);write(OUT/'protocol.json',p);write(ROOT/'manifest.json',manifest)
    return p


def check():
    p=read(OUT/'protocol.json');q=deepcopy(p);h=q.pop('protocol_sha256');assert digest(q)==h
    assert p['sources']==source_hashes(),'source changed after freeze'
    for path,key in p['input_sha256'].items():assert filehash(path)==key,path
    assert filehash(OUT/'qualification.json')==p['qualification_sha256']
    return p


def request(client,public,state,system,validator,phase):
    # JSON round trip is mandatory: B4 dataclass tuples serialize to lists.
    public=json.loads(json.dumps(canonical.semantic(public),ensure_ascii=False,allow_nan=False))
    context={'public':public,'observed_prefix':deepcopy(state['history']),'phase':phase,
        'system_sha256':digest(system),'protocol_sha256':state['protocol_sha256'],'task':state['task']}
    context['context_sha256']=digest(context);key=context['context_sha256'];folder=ROOT/'requests'/key
    write(folder/'context.json',context)
    return transport.get(client,context,folder,system,validator),key


def step(state,path,pool,cache,client):
    t=len(state['decisions'])+1;started=time.perf_counter();d=engine.setup(cache,state,pool,t)
    if not any(c['iteration']==t for c in state['calls']):
        reasons=engine.event_reasons(state,d,t)
        if reasons:
            public=policy.public_request(pool,state,d,reasons,'refresh')
            answer,key=request(client,public,state,policy.REFRESH_SYSTEM,
                lambda raw:grounded.e.validate_refresh(raw,state['knowledge'],state['slots'],public['observations'],pool),'refresh')
            state['calls'].append({'iteration':t,'reasons':reasons,'context_sha256':key,'knowledge_before':deepcopy(state['knowledge']),
                                  'validated':answer})
            state['knowledge']=answer['plan']['knowledge'];write(path,state);d=engine.setup(cache,state,pool,t)
    normal=pool['ids'][d['selected']];validation=None;prepared=None
    if t in engine.PARAMETERS['validation_rounds']:
        prepared=policy.build_candidates(pool,state,d)
        state['validation_checks']=[c for c in state['validation_checks'] if c['iteration']!=t]
        state['validation_checks'].append({'iteration':t,'candidate_count':len(prepared['candidates']),'skip_reason':prepared.get('skip_reason')})
        if prepared['candidates']:
            public=policy.public_request(pool,state,d,[],'controlled_validation');public['counterexample_pool']=prepared
            answer,key=request(client,public,state,policy.SYSTEM,lambda raw:policy.validate(raw,prepared,state['history']),'controlled_validation')
            validation={'prepared':prepared,'validated':answer,'context_sha256':key}
            d['selected']=pool['ids'].index(answer['plan']['candidate_id'])
    j=d['selected'];selected=pool['ids'][j];prefix=deepcopy(state['history']);best=max(h['y'] for h in prefix)
    assert d['available'][j];exception=bool(not d['eligible'][j]);assert not exception or validation is not None
    folder=path.parent/'rounds';folder.mkdir(parents=True,exist_ok=True);table=folder/f'{t:02d}_candidates.npz'
    with table.with_suffix('.tmp').open('wb') as f:
        np.savez_compressed(f,base_qLogEI=d['base'],guided_qLogEI=d['score'],available=d['available'],eligible=d['eligible'],
            raw_shift_std=d['raw_shift_std'],deployed_shift_std=d['shift_std'],
            tie_priority=engine.priority(len(pool['ids']),t,0),**d['stats'])
    table.with_suffix('.tmp').replace(table)
    _,v=cache.model(prefix);scale=float(v['std']);mu=float(d['stats']['mu_y'][j]);sd=float(d['stats']['sd_predictive_y'][j])
    raw_y=float(d['raw_shift_std'][j]*scale);deployed_y=float(d['shift_std'][j]*scale)
    committed={'iteration':t,'selected_id':selected,'normal_proposal_id':normal,'history':prefix,'history_sha256':digest(prefix),
        'protocol_sha256':state['protocol_sha256'],'candidate_table_sha256':filehash(table),
        'knowledge':deepcopy(state['knowledge']),'diagnosis':d['diagnosis'].to_dict(),'trust':deepcopy(d['trust']),
        'bank_before':d['bank'],'bank_prior':d['prior'],'amplitude':d['amplitude'],'validation':validation,
        'base_meta':d['base_meta'],'guided_meta':d['guided_meta'],'field_meta':d['field_meta'],'scale_y':scale,
        'reference_id':pool['ids'][d['reference']],'budget':d['budget'],'acquisition_gate_exception':exception,
        'GP_health':d['GP_health'],'tie_order':d['tie_order'],'region_trust_at_commit':deepcopy(state['region_trust'])}
    commit=folder/f'{t:02d}_committed.json';write(commit,committed)
    # Only after the irreversible decision receipt exists is this outcome read.
    with np.load(prior.original.ROOT/'inputs'/f"case{state['case']}"/'oracle.npz') as oracle:y=float(oracle['values'][j])
    after=feedback=None
    if d['bank'] is not None:
        after,m=engine.original.update_bank(d['bank'],d['prior'],t,raw_y,mu,sd,y)
        state['banks'][d['bank_key']]=after;d['_measurement']=m;feedback=engine.feedback(state,d,t,y,after)
    else:
        variance=sd*sd+.0009;m={'prequery_mu_y':mu,'prequery_predictive_sd_y':sd,'raw_region_shift_y':raw_y,
            'observed_y':y,'residual':y-mu,'innovation':y-mu,'measurement_variance':variance,
            'innovation_variance':variance,'innovation_z':float((y-mu)/np.sqrt(variance))}
    controlled=None
    if validation:
        plan=validation['validated']['plan'];anchor=prefix[plan['anchor_observation_index']-1]
        controlled={'query':t,'anchor_observation_index':plan['anchor_observation_index'],'anchor_candidate_id':anchor['candidate_id'],
            'anchor_yield_percent':anchor['y']*100,'tested_candidate_id':selected,'tested_yield_percent':y*100,
            'incumbent_before_percent':best*100,'observed_difference_pp':(y-anchor['y'])*100,'changed_axis':plan['changed_axis'],
            'tested_clause_id':plan['clause_id'],'prequery_prediction':plan['prediction']}
        state['validations'].append(controlled)
    event={**{k:committed[k] for k in ('iteration','selected_id','normal_proposal_id','reference_id','budget','acquisition_gate_exception','knowledge','diagnosis','trust','bank_before','bank_prior','amplitude','GP_health','tie_order')},
        'y':y,'incumbent_before':best,'incumbent_after':max(best,y),'execution_mode':'controlled_validation' if validation else 'normal_R2',
        'base_opportunity_loss':float(d['base'][d['reference']]-d['base'][j]),'eligible_count':int(d['eligible'].sum()),
        'base_qLogEI':float(d['base'][j]),'guided_qLogEI':float(d['score'][j]),'raw_shift_y':raw_y,'deployed_shift_y':deployed_y,
        'measurement':m,'bank_after':after,'trust_feedback':feedback,'region_trust_after_feedback':deepcopy(state['region_trust']),
        'controlled_comparison':controlled,'commit_sha256':filehash(commit)}
    state['history'].append({'candidate_id':selected,'y':y});state['decisions'].append(event)
    state['active_seconds']+=time.perf_counter()-started;state['status']='COMPLETE' if t==16 else 'RUNNING'
    state['operator_compiles_in_loop']=sum(prior.previous.JIT.values())
    assert state['operator_compiles_in_loop']==0
    write(path,state);cache.release()
    print(json.dumps({'task':state['task'],'query':t,'mode':event['execution_mode'],'yield_percent':y*100,
        'best_percent':max(best,y)*100,'refreshes':len(state['calls']),'trust_blocked':d['trust']['trust_blocked']}),flush=True)


def run_case(case,config_path='../config.txt'):
    p=check();pool=prior.inputs(case);cache=cache_for(case,pool);client=api.APIClient(api.load_config(config_path));states=[]
    try:
        for regime in REGIMES:
            state=initial(case,regime,p,pool);path=ROOT/'runs'/state['run_id']/'state.json'
            if path.exists():state=read(path)
            assert state['protocol_sha256']==p['protocol_sha256']
            if state['status']=='INTERRUPTED':state['status']='RUNNING';state.pop('error',None)
            states.append((state,path))
        for t in range(1,17):
            for state,path in states:
                if len(state['decisions'])!=t-1 or state['status']=='INTERRUPTED':continue
                try:step(state,path,pool,cache,client)
                except Exception as error:
                    state.update(status='INTERRUPTED',error=api.safe_error(error,client.config))
                    write(path,state);write(path.parent/'error.json',{'error':state['error'],'traceback':api.safe_error(RuntimeError(traceback.format_exc()),client.config)})
                    print(json.dumps({'task':state['task'],'status':'INTERRUPTED','query':t,'error':state['error']}),flush=True)
    finally:client.close()
    receipt={'case':case,'cache':cache.counters,'operator_compiles_in_loop':sum(prior.previous.JIT.values()),
        'complete':sum(s['status']=='COMPLETE' for s,_ in states),'planned':4}
    write(ROOT/'runtime'/f'c{case}.json',receipt);return receipt


def run(config_path='../config.txt'):
    check()
    with ProcessPoolExecutor(max_workers=WORKERS,initializer=warm) as executor:
        for f in as_completed([executor.submit(run_case,c,config_path) for c in (1,2,3,4)]):print(json.dumps(f.result()),flush=True)
    return status()


def status():
    rows=[read(p) for p in (ROOT/'runs').glob('*/state.json')]
    return {'planned':16,'complete':sum(s['status']=='COMPLETE' for s in rows),'queries_completed':sum(len(s['decisions']) for s in rows),
        'interrupted':[{'run_id':s['run_id'],'queries':len(s['decisions']),'error':s.get('error')} for s in rows if s['status']=='INTERRUPTED']}


def preflight(config_path='../config.txt'):
    client=api.APIClient(api.load_config(config_path))
    try:
        response=client.http.get('/user/balance');response.raise_for_status();raw=response.json()
        result={'available':raw.get('is_available'),'balances':[{k:r.get(k) for k in ('currency','total_balance')} for r in raw.get('balance_infos',[])],
                'model':api.MODEL,'CPU_workers':WORKERS}
        write(OUT/'resource_preflight.json',result);return result
    finally:client.close()
