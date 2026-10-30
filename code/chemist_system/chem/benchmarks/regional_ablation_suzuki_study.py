"""Complete feedback trajectories with matched initial regional knowledge."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import time

import httpx
import numpy as np

from . import regional_ablation_suzuki_engine as e
from . import mixed_comparison_suzuki_study as prior
from . import deepseek_small_suzuki_study as transport
from . import deepseek_suzuki_client as api
from .suzuki_domain import CATALYSTS,digest

ROOT=Path('data/runs/suzuki_regional_ablation');OUT=Path('research/suzuki_regional_ablation')
METHODS=e.METHODS;REGIMES=prior.REGIMES
read=prior.read;write=prior.write;filehash=prior.filehash
PARAMETERS=deepcopy(prior.PARAMETERS);GP_WORKERS=4;API_WORKERS=2


def inputs(case):return prior.inputs(case)
def initial_knowledge(case,regime):return e.from_advice(prior.advice(case,regime))


def hashes():
    paths=[Path(__file__),Path(e.__file__),Path(prior.__file__),Path(e.common.__file__),Path(e.common.mixed.__file__),
           Path(e.previous.__file__),Path(api.__file__),Path(transport.__file__),Path('chem/decision/domain_guard.py'),
           Path('chem/benchmarks/suzuki_regional_ablation.py'),Path('tests/test_suzuki_regional_ablation.py')]
    return {p.as_posix():filehash(p) for p in paths}


def cache_for(case,pool,qualification=False,category_map=e.common.mixed.CATEGORY_MAP):
    group=f'c{case}_s0';directory=ROOT/('qualification_cache' if qualification else 'cache')/group
    if tuple(category_map)!=e.common.mixed.CATEGORY_MAP:directory=directory/'permuted'
    sources=[root/folder/group for root in (prior.ROOT,prior.previous.ROOT) for folder in ('cache','audit_cache','qualification_cache')]
    if not qualification:sources.insert(0,ROOT/'qualification_cache'/group)
    return e.RegionalAblationCache(pool,directory,case,category_map=category_map,sources=sources)


def worker_init():prior.worker_init()


def qualify():
    worker_init();rows=[]
    for case in (1,2,3,4):
        pool=inputs(case);history=prior.initial(case,pool);cache=cache_for(case,pool,True)
        permuted=cache_for(case,pool,True,[19,-3,101,7,0,12,-9,4])
        for regime in REGIMES:
            knowledge=initial_knowledge(case,regime);ids=sorted(h['id'] for h in knowledge['clauses'])
            a,_,meta=cache.field(history,knowledge,ids);unity,_,_=cache.calibrated_score(history,knowledge,ids,1.)
            path=prior.ROOT/'runs'/f'c{case}_s0_{regime}_LGBO_gate'/'state.json';event=read(path)['decisions'][0]
            scoremeta=next(iter(event['guided_acquisitions'].values()))
            with np.load(prior.ROOT/'cache'/f'c{case}_s0'/(scoremeta['key']+'.npz'),allow_pickle=False) as f:expected=f['scores']
            assert np.array_equal(a,expected) and np.array_equal(unity,expected)
            b,_,_=permuted.calibrated_score(history,knowledge,ids,.63);c,_,_=cache.calibrated_score(history,knowledge,ids,.63)
            err=float(np.max(np.abs(b-c)));assert err<1e-9
            rows.append({'case':case,'regime':regime,'initial_native_scores_exact':True,'initial_KF_unity_exact':True,
                         'label_permutation_max_error':err})
    result={'status':'PASS','cases':rows,'operator_compiles_in_loop':sum(prior.previous.JIT.values())}
    assert result['operator_compiles_in_loop']==0
    write(OUT/'qualification.json',result);return result


def prepare(config_path='../config.txt'):
    if (OUT/'protocol.json').exists():return check_protocol()
    assert read(OUT/'qualification.json')['status']=='PASS';cfg=api.load_config(config_path)
    manifest=[{'case':c,'seed':0,'regime':r,'method':m,'run_id':f'c{c}_s0_{r}_{m}'} for c in (1,2,3,4) for r in REGIMES for m in METHODS]
    for i,v in enumerate(sorted(manifest,key=lambda x:digest({'salt':'regional-ablation-20261028','id':x['run_id']}))):v['task']=f'R{i:02}'
    pp=prior.check_protocol()
    protected=deepcopy(pp['protected_inventories'])
    protected['mixed_comparison']={'path':(prior.OUT/'release_inventory.json').as_posix(),'sha256':filehash(prior.OUT/'release_inventory.json')}
    extras={f.as_posix():filehash(f) for f in [Path('docs/suzuki_mixed_gate_diagnostic.md'),Path('research/suzuki_mixed_gate_diagnostic/release_receipt.json')]}
    protocol={'study':'Matched_regional_gate_KF_B4_refresh_ladder_seed0_v1',
        'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'scientific_sha256':hashes(),
        'manifest_sha256':digest(manifest),'qualification_sha256':filehash(OUT/'qualification.json'),
        'cases':[1,2,3,4],'primary_cases':[2,3,4],'seeds':[0],'regimes':REGIMES,'methods':METHODS,
        'planned_trajectories':64,'initial_queries':4,'additional_queries':16,'logical_queries':1280,
        'maximum_new_LLM_calls':48,'candidate_count':4692,'parameters':e.PARAMETERS,
        'GP':'Same immutable MixedSingleTaskGP adapter; exact input, normalization, fit/MC seeds, MC512 qLogEI, jitter1e-6; nominal cat kernel, same fixed pool.',
        'injection':'Same official regional covariance mean tilt in every arm: 64x8 nominal quadrature, numeric smooth0.08, confidence mapping3*Phi^-1(c). KF scales its mean field, covariance unchanged. No indicator-additive hypotheses, no0.5SD cap.',
        'initial_knowledge':'Byte-equivalent coordinate preferences and confidence from each existing case/seed0/condition; all four arms start with identical clauses and amplitude1. No initial LLM regeneration.',
        'initial_knowledge_sha256':{f'case{c}/{r}':digest(initial_knowledge(c,r)) for c in (1,2,3,4) for r in REGIMES},
        'ladder':{'G0_gate':'Fixed original region, amplitude1, maximize regional qLogEI within same base-score budget.',
                  'G1_gate_KF':'Add joint offset/regional-amplitude scalar KF, deployment clip(mean_a,0,1), initial mean_a1.',
                  'G2_gate_KF_B4':'Additionally unchanged maximum-cardinality/lexical B4 clause admission; same tilt of retained region; query still soft over entire eligible set.',
                  'G3_gate_KF_B4_event':'Additionally fresh own-prefix LLM region retain/revise/revoke/add, fixed confidence and locked slot-to-axis map.'},
        'KF':'State[b,a], P0diag(.0025,.25), Qdiag(.000025,.0025), R=prequery GP observation variance+.0009; measurement y-mu_GP with H=[1,unscaled regional shift_y]. Joseph update. b is nuisance, never added to acquisition. No retrospective refit on initial4/history.',
        'calibration_banks':'One state per actual regional geometry+confidence, independent of textual IDs/reasons. A changed or expanded region starts at same prior; returning to previously calibrated identical geometry reuses only its own updates and advances process noise by elapsed rounds. No trust transfer to a different region.',
        'B4':'Diagnose all currently supplied coordinate clauses on base qLogEI and same budget, regardless of current scalar amplitude; subset controls region geometry, not hard query feasibility. Predicate category comparisons use named-set membership only.',
        'refresh':'Treat supplied initial knowledge as virtual round1 call. Reuse prior event rules/cooldown/max4-including-initial/last14; at most3 fresh calls per case/condition, no round1 call. Triggers3-round stagnation<0.5pp,2 consecutive |innovation_z|>2.5,B4 conflicts,empty knowledge.',
        'refresh_schema':'At most4 locked coordinate slots. retain preserves exact geometry; revise/add numeric width0.1..0.9 with4decimals or nominal category set1..7. Confidence fixed at original value. Every active slot accounted for; geometry-specific KF bank remains authoritative.',
        'API':cfg.public(),'request_parameters':PARAMETERS,'system_prompt_sha256':sha256(e.SYSTEM_PROMPT.encode('utf8')).hexdigest(),
        'LLM_context':'Only own observed prefix, saved prequery measurements, current Mixed GP top3/catalyst, current clauses, current bank and B4 diagnosis. No case/condition/truth labels, unseen responses, other-method histories or previous-study LLM answers.',
        'retries':'First valid response; <=3 received schema responses and<=8 HTTP attempts. Transient retry5/10/20/30s. Failure consumes refresh and retains current knowledge; no human replacement or outcome-based resampling.',
        'evaluation':'SearchAUC mean incumbent at added positions1..16; final best; paired ladder increments G1-G0,G2-G1,G3-G2 within each case/condition, conditional on insertion order. Also total-vs-baseline deltas and downside. No module-additive causal AUC decomposition across all orders; one seed/case.',
        'stress_boundary':'correct/wrong_category/mixed initial priors are disclosed reference-outcome-aware stress constructions. LLM gets uncertain initial predicates but no truth/regime labels.',
        'protected_inventories':protected,'additional_protected_files':extras,
        'input_sha256':pp['input_sha256'],'resources':{'static_processes':4,'event_processes':2,'threads_each':1,'device':'cpu','warmup':'Mixed kernels and Cholesky before counted loops','cache':'Exact prefix Mixed GP reuse; no loops that compile operators'},
        'baseline_receipt':'Replay all16 rounds of all16 G0 trajectories and require exact candidate/y histories equal previous Mixed LGBO_gate, not just same aggregate AUC.'}
    protocol['protocol_sha256']=digest(protocol);write(OUT/'protocol.json',protocol);write(ROOT/'protocol.json',protocol);write(ROOT/'manifest.json',manifest)
    write(OUT/'start_receipt.json',{'states_before_start':len(list((ROOT/'runs').glob('*/state.json'))),
                                  'previous_results_sha256':filehash(prior.OUT/'results.json')})
    return protocol


def check_protocol():
    p=read(OUT/'protocol.json');raw=deepcopy(p);key=raw.pop('protocol_sha256')
    assert digest(raw)==key and p['scientific_sha256']==hashes()
    assert digest(read(ROOT/'manifest.json'))==p['manifest_sha256']
    for f,h in p['input_sha256'].items():assert filehash(f)==h,f
    for row in p['protected_inventories'].values():assert filehash(row['path'])==row['sha256']
    for f,h in p['additional_protected_files'].items():assert filehash(f)==h
    return p


def state_for(spec,p,pool):
    path=ROOT/'runs'/spec['run_id']/'state.json'
    if path.exists():return read(path),path
    knowledge=initial_knowledge(spec['case'],spec['regime']);slots={h['id']:h['coordinate'] for h in knowledge['clauses']}
    state={**spec,'protocol_sha256':p['protocol_sha256'],'status':'RUNNING','history':prior.initial(spec['case'],pool),
           'knowledge':knowledge,'initial_knowledge':deepcopy(knowledge),'slots':slots,'banks':{},'decisions':[],'calls':[],
           'active_seconds':0.,'operator_compiles_in_loop':0}
    write(path,state);return state,path


def public_request(state,pool,decision,reasons):
    ix={k:i for i,k in enumerate(pool['ids'])};observations=[]
    for i,h in enumerate(state['history'],1):
        j=ix[h['candidate_id']];row={'index':i,'cat':CATALYSTS.index(pool['points'][j]['catalyst']),
            'z':[round(float(x),4) for x in pool['z'][j][:3]],'yield_percent':round(h['y']*100,3)}
        if i>4:
            m=state['decisions'][i-5]['measurement'];row.update(prequery_GP_mean_percent=round(m['prequery_mu_y']*100,3),
                prequery_predictive_SD_pp=round(m['prequery_predictive_sd_y']*100,3),prequery_residual_pp=round(m['residual']*100,3),
                original_region_shift_pp=round(m['raw_region_shift_y']*100,3),joint_innovation_z=round(m['innovation_z'],4))
        observations.append(row)
    top=[]
    for cat in range(8):
        choices=[i for i,x in enumerate(pool['points']) if CATALYSTS.index(x['catalyst'])==cat and decision['available'][i]]
        for j in sorted(choices,key=lambda i:(-float(decision['base'][i]),pool['ids'][i]))[:3]:
            top.append({'id':pool['ids'][j],'cat':cat,'z':[round(float(x),4) for x in pool['z'][j][:3]],
                        'GP_mean_percent':round(float(decision['stats']['mu_y'][j])*100,3),
                        'GP_latent_SD_pp':round(float(decision['stats']['sd_y'][j])*100,3),
                        'base_logEI':round(float(decision['base'][j]),5)})
    return {'task':state['task'],'iteration':len(state['decisions'])+1,'remaining_queries':16-len(state['decisions']),
            'reasons':reasons,'observations':observations,'current_knowledge':state['knowledge'],
            'fixed_slots':{key:e.VARIABLES[j] for key,j in state['slots'].items()},'calibration_bank':decision['bank'],
            'calibration_forecast':decision['prior'],'deployment_amplitude':decision['amplitude'],
            'B4_diagnosis':decision['diagnosis'].to_dict(),'GP_top3_per_catalyst':top,'catalysts':dict(enumerate(CATALYSTS)),
            'physical_bounds':{'residence_time':[60,600],'temperature':[30,110],'catalyst_loading':[.5,2.5]}}


def get_refresh(client,context,state,pool,folder):
    if (folder/'validated.json').exists():return read(folder/'validated.json')
    base=[{'role':'system','content':e.SYSTEM_PROMPT},
          {'role':'user','content':json.dumps(context['public'],ensure_ascii=False,separators=(',',':'))}]
    messages=base;received=0
    for attempt in range(1,9):
        directory=folder/f'attempt{attempt:02d}';body=dict(PARAMETERS,messages=messages)
        if (directory/'request.json').exists():assert read(directory/'request.json')==body
        else:write(directory/'request.json',body)
        if (directory/'transport_error.json').exists():
            if not read(directory/'transport_error.json')['retryable']:raise RuntimeError('nonretryable API failure')
            continue
        if (directory/'response.json').exists():answer=read(directory/'response.json')
        else:
            try:answer=transport.complete(client,messages);write(directory/'response.json',answer)
            except Exception as error:
                status=error.response.status_code if isinstance(error,httpx.HTTPStatusError) else None
                retryable=isinstance(error,(httpx.TimeoutException,httpx.TransportError)) or status in (429,500,502,503,504)
                write(directory/'transport_error.json',{'error':api.safe_error(error,client.config),'retryable':retryable})
                if not retryable:raise RuntimeError('nonretryable API failure') from None
                if attempt<8:time.sleep(min(5*2**(attempt-1),30))
                continue
        received+=1
        try:
            if answer['model']!=api.MODEL or len(answer['choices'])!=1 or answer['choices'][0]['finish_reason']!='stop':raise ValueError('model mismatch or incomplete response')
            plan=e.validate_refresh(json.loads(answer['choices'][0]['message']['content']),state['knowledge'],state['slots'],len(state['history']),pool)
        except (ValueError,TypeError,KeyError,IndexError) as error:
            text=type(error).__name__+': '+str(error)[:240];write(directory/'validation_error.json',{'error':text})
            if received>=3:raise RuntimeError('received schema limit exhausted') from None
            content=answer['choices'][0]['message'].get('content') if answer.get('choices') else '{}'
            messages=base+[{'role':'assistant','content':content or '{}'},
                {'role':'user','content':'Schema validation failed: '+text+'. Correct only JSON, no experiment occurred. Return exactly actions,summary.'}]
            continue
        valid={'plan':plan,'accepted_attempt':attempt,'received_responses':received,'response_id':answer['id'],
               'response_model':answer['model'],'response_sha256':filehash(directory/'response.json'),'context_sha256':context['context_sha256']}
        write(folder/'validated.json',valid);return valid
    raise RuntimeError('HTTP attempt limit exhausted')


def step(state,path,pool,cache,values,client=None):
    t=len(state['decisions'])+1;start=time.perf_counter();before=sum(prior.previous.JIT.values())
    decision=e.setup(cache,state,pool,t);before_knowledge=deepcopy(state['knowledge']);refresh=None
    if state['method']==METHODS[3] and not any(c['iteration']==t for c in state['calls']):
        reasons=e.event_reasons(state,decision,t)
        if reasons:
            public=public_request(state,pool,decision,reasons)
            context={'public':public,'history':deepcopy(state['history']),'protocol_sha256':state['protocol_sha256']}
            context['context_sha256']=digest(context);folder=ROOT/'requests'/context['context_sha256'];write(folder/'context.json',context)
            refresh={'iteration':t,'reasons':reasons,'context_sha256':context['context_sha256'],
                     'knowledge_before':before_knowledge,'diagnosis_before':decision['diagnosis'].to_dict()}
            try:
                valid=get_refresh(client,context,state,pool,folder);state['knowledge']=valid['plan']['knowledge']
                refresh.update(status='ACCEPTED',**valid)
            except RuntimeError as error:refresh.update(status='FAILED',error=str(error))
            state['calls'].append(refresh);write(path,state)
            decision=e.setup(cache,state,pool,t)
    j=decision['selected'];fit,value=cache.model(state['history']);std=float(value['std'])
    raw_y=float(decision['raw_shift_std'][j]*std);shift_y=float(decision['shift_std'][j]*std)
    mu=float(decision['stats']['mu_y'][j]);sd=float(decision['stats']['sd_predictive_y'][j])
    # Outcome lookup happens after ranking; no hidden outcome enters setup().
    y=float(values[j]);after=None
    if decision['bank'] is not None:
        after,measurement=e.update_bank(decision['bank'],decision['prior'],t,raw_y,mu,sd,y)
        state['banks'][decision['bank_key']]=after
    else:
        variance=sd*sd+.0009;measurement={'prequery_mu_y':mu,'prequery_predictive_sd_y':sd,'raw_region_shift_y':raw_y,
            'observed_y':y,'residual':y-mu,'innovation':y-mu,'innovation_variance':variance,
            'measurement_variance':variance,'innovation_z':float((y-mu)/np.sqrt(variance))}
    preds={'GP':mu,'regional_unscaled':mu+raw_y,'deployed':mu+shift_y}
    if decision['prior'] is not None:preds['KF_joint']=mu+decision['prior']['mean'][0]+decision['prior']['mean'][1]*raw_y
    measurement.update(predictions=preds,squared_errors={k:float((y-v)**2) for k,v in preds.items()})
    event={'iteration':t,'selected_id':pool['ids'][j],'reference_id':pool['ids'][decision['reference']],
           'base_acquisition':decision['base_meta'],'field_acquisition':decision['field_meta'],'guided_acquisition':decision['guided_meta'],
           'budget_logei':decision['budget'],'base_logei_loss':float(decision['base'][decision['reference']]-decision['base'][j]),
           'eligible_candidates':int(decision['eligible'].sum()),'retained_ids':decision['retained'],
           'diagnosis':decision['diagnosis'].to_dict(),'knowledge':deepcopy(state['knowledge']),
           'bank_key':decision['bank_key'],'bank_before':decision['bank'],'bank_prior':decision['prior'],'bank_after':after,
           'amplitude':decision['amplitude'],'raw_shift_y':raw_y,'deployed_shift_y':shift_y,'measurement':measurement}
    assert event['base_logei_loss']<=event['budget_logei']
    state['decisions'].append(event);state['history'].append({'candidate_id':pool['ids'][j],'y':y})
    state['active_seconds']+=time.perf_counter()-start;state['operator_compiles_in_loop']+=sum(prior.previous.JIT.values())-before
    state['status']='COMPLETE' if t==16 else 'RUNNING';write(path,state)
    if state['method']==METHODS[3]:print(json.dumps({'task':state['task'],'round':t,'refreshes':len(state['calls'])},separators=(',',':')),flush=True)


def run_case(case,event=False,config_path='../config.txt'):
    p=check_protocol();pool=inputs(case);cache=cache_for(case,pool)
    values=np.load(prior.original.ROOT/'inputs'/f'case{case}'/'oracle.npz')['values']
    specs=[v for v in read(ROOT/'manifest.json') if v['case']==case and (v['method']==METHODS[3])==event]
    states=[state_for(v,p,pool) for v in specs]
    client=api.APIClient(api.load_config(config_path)) if event else None
    try:
        for t in range(1,17):
            for state,path in states:
                if state['status'] in ('COMPLETE','FAILED') or len(state['history'])!=t+3:continue
                try:step(state,path,pool,cache,values,client)
                except Exception as error:
                    state.update(status='FAILED',error={'kind':type(error).__name__,'message':str(error)[:300]});write(path,state)
    finally:
        if client:client.close()
    result={'case':case,'event':event,'planned':len(states),'complete':sum(s['status']=='COMPLETE' for s,_ in states),
            'failed':[s['run_id'] for s,_ in states if s['status']=='FAILED'],'cache':cache.counters,
            'operator_compiles_in_loop':sum(s['operator_compiles_in_loop'] for s,_ in states)}
    write(ROOT/'runtime'/f"c{case}_{'event' if event else 'static'}.json",result);return result


def run(event=False,config_path='../config.txt'):
    check_protocol()
    if event:
        states=[ROOT/'runs'/v['run_id']/'state.json' for v in read(ROOT/'manifest.json') if v['method']!=METHODS[3]]
        assert all(read(f)['status']=='COMPLETE' for f in states)
        write(OUT/'static_frozen.json',{'states':{f.as_posix():filehash(f) for f in states},
              'event_states_before_start':len(list((ROOT/'runs').glob('*_G3_gate_KF_B4_event/state.json')))})
    with ProcessPoolExecutor(max_workers=API_WORKERS if event else GP_WORKERS,initializer=worker_init) as executor:
        futures=[executor.submit(run_case,c,event,config_path) for c in (1,2,3,4)]
        for f in as_completed(futures):print(json.dumps(f.result(),separators=(',',':')),flush=True)
    return status()


def status():
    counts={};steps=0;failures=[]
    for f in (ROOT/'runs').glob('*/state.json'):
        row=read(f);counts[row['status']]=counts.get(row['status'],0)+1;steps+=len(row['decisions'])
        if row['status']=='FAILED':failures.append({'run':row['run_id'],'error':row.get('error')})
    return {'runs':counts,'additional_queries':steps,'planned':64,'failures':failures}
