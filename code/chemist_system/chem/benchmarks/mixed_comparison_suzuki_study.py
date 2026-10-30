"""Frozen seed0, four-condition Mixed-GP rerun of five existing policies."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import time
import traceback

import httpx
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from . import mixed_comparison_suzuki_engine as e
from . import mixed_kalman_suzuki_study as previous
from . import guarded_suzuki_study as original
from . import deepseek_small_suzuki_study as transport
from . import deepseek_suzuki_client as api
from .official_suzuki_adapter import modules,check_sources,AwaitingAssistant
from .suzuki_domain import digest

ROOT=Path('data/runs/suzuki_mixed_comparison');OUT=Path('research/suzuki_mixed_comparison')
METHODS=('LGBO_region','LLINBO_matched','LGBO_gate','B4_base','B4_LGBO')
REGIMES=e.regional.REGIMES;GP_WORKERS=4;API_WORKERS=2
read=original.read;write=original.write;filehash=original.filehash
JIT={};PARAMETERS=deepcopy(previous.PARAMETERS)


def inputs(case):return read(original.ROOT/'inputs'/f'case{case}'/'pool.json')
def advice(case,regime):return read(original.ROOT/'inputs'/f'case{case}'/'advice.json')['0'][regime]


def make_cache(case,pool,qualification=False,permutation=e.mixed.CATEGORY_MAP):
    group=f'c{case}_s0'
    folder=ROOT/('qualification_cache' if qualification else 'cache')/group
    if tuple(permutation)!=e.mixed.CATEGORY_MAP:folder=folder/'permuted'
    sources=[previous.ROOT/'cache'/group,previous.ROOT/'qualification_cache'/group]
    if not qualification:sources.insert(0,ROOT/'qualification_cache'/group)
    return e.MixedRegionalCache(pool,folder,case,0,permutation,sources)


def hashes():
    paths=[Path(__file__),Path(e.__file__),Path(e.mixed.__file__),Path(e.regional.__file__),
           Path(e.justify.__file__),Path(api.__file__),Path(transport.__file__),
           Path('chem/decision/domain_guard.py'),Path('chem/benchmarks/suzuki_mixed_comparison.py'),
           Path('tests/test_suzuki_mixed_comparison.py')]
    return {p.as_posix():filehash(p) for p in paths}


def initial(case,pool):
    values=np.load(original.ROOT/'inputs'/f'case{case}'/'oracle.npz')['values'];ix={k:i for i,k in enumerate(pool['ids'])}
    return [{'candidate_id':k,'y':float(values[ix[k]])} for k in pool['initial_ids']['0']]


def worker_init():
    threadpool_limits(1);previous.worker_init();modules('LLINBO')
    tilt=e.NominalRegionalTilt();tilt.set_box_region([0.]*4,[1.]*4)
    # The previous initializer begins counting only after Mixed/Cholesky warmup.


def qualification():
    worker_init();rows=[];permutation=(19,-3,101,7,0,12,-9,4)
    for case in (1,2,3,4):
        pool=inputs(case);h=initial(case,pool)
        a=make_cache(case,pool,True);b=make_cache(case,pool,True,permutation)
        base,meta=a.score(h);bp,_=b.score(h)
        event=read(previous.ROOT/'base'/f'c{case}_s0'/'state.json')['decisions'][0]
        with np.load(previous.ROOT/'cache'/f'c{case}_s0'/(event['base_score_key']+'.npz'),allow_pickle=False) as saved:
            expected=saved['scores'].copy()
        assert np.array_equal(base,expected)
        errors=[]
        for regime in REGIMES:
            prior=advice(case,regime);box=e.regional.region_for(prior,[c['id'] for c in prior['clauses']])
            ga,ma=a.score(h,box,prior['confidence']);gb,mb=b.score(h,box,prior['confidence'])
            error=float(np.max(np.abs(ga-gb)));assert error<1e-9
            errors.append({'regime':regime,'max_guided_qLogEI_difference':error,
                           'argmax_equal':int(np.argmax(ga))==int(np.argmax(gb))})
        rows.append({'case':case,'base_score_difference':float(np.max(np.abs(base-bp))),
                     'previous_Mixed_B0_scores_identical':True,'regimes':errors})
    result={'status':'PASS','cases':rows,'grid':e.GRID_RECIPE,
            'operator_compiles_in_loop':sum(previous.JIT.values())}
    assert result['operator_compiles_in_loop']==0
    write(OUT/'qualification.json',result);return result


def prepare(config_path='../config.txt',provider='deepseek-flash'):
    if (OUT/'protocol.json').exists():return check_protocol()
    assert read(OUT/'qualification.json')['status']=='PASS'
    assert provider in ('deepseek-flash','current_assistant')
    manifest=[{'run_id':f'c{c}_s0_{r}_{m}','case':c,'seed':0,'regime':r,'method':m}
              for c in (1,2,3,4) for r in REGIMES for m in METHODS]
    for i,spec in enumerate(sorted(manifest,key=lambda x:digest({'salt':'mixed-20261028','id':x['run_id']}))):
        spec['task']=f'X{i:02}'
    protected=dict(previous.PROTECTED,mixed_kalman=previous.OUT/'release_inventory.json')
    p={'study':'MixedGP_five_policy_four_condition_seed0_v1',
       'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
       'scientific_sha256':hashes(),'official_sources':check_sources(),
       'qualification_sha256':filehash(OUT/'qualification.json'),'manifest_sha256':digest(manifest),
       'protected_inventories':{k:{'path':v.as_posix(),'sha256':filehash(v)} for k,v in protected.items()},
       'additional_protected_files':previous.protected_extras(),
       'input_sha256':{(original.ROOT/'inputs'/f'case{c}'/name).as_posix():filehash(original.ROOT/'inputs'/f'case{c}'/name)
                       for c in (1,2,3,4) for name in ('pool.json','advice.json','oracle.npz')},
       'cases':[1,2,3,4],'primary_cases':[2,3,4],'seeds':[0],'regimes':REGIMES,'methods':METHODS,
       'planned_new_trajectories':80,'initial_queries':4,'added_queries':16,
       'planned_new_logical_queries':1600,'planned_LLINBO_proposals':256,'candidate_count':4692,
       'model':'Exact same MixedSingleTaskGP adapter, normalization, fit seeds, MC512 qLogEI and covariance jitter1e-6 as frozen Mixed KF experiment.',
       'regional_adaptation':{'grid':e.GRID_RECIPE,'reason':'Continuous category grid and sigmoid category intervals would retain arbitrary order effects. Decode preference to named set; uniform category counting measure and numeric smoothing only.',
           'reused_verbatim':'Official regional fit_lambda_by_delta, prepare_cache, posterior, TiltedModel, delta=3*Phi^-1(confidence).',
           'boundary':'Changes regional quadrature/representation necessarily; not a byte-identical old ordinal whole algorithm reproduction.'},
       'decision_rules':{'LGBO_region':'Full region analytic mean tilt, ungated guided-qLogEI argmax.',
           'LLINBO_matched':'Actual official run->_run_justify, strict base>proposal+epsilon rejects to base. All prior finite-pool hooks unchanged; only GP score cache replaced.',
           'LGBO_gate':'Full region tilt, guided argmax among base-score eligible candidates; no revocation.',
           'B4_base':'Unchanged maximum-cardinality/lexical subset diagnosis; hard retained intersection, base qLogEI argmax.',
           'B4_LGBO':'Same subset diagnosis, retained region soft tilt, guided argmax across eligible candidates.'},
       'budget':'Previously frozen r=4; log(4)*(16-t)/15. No retuning, extra tolerance or analytic-EI substitution.',
       'knowledge':'Use identical frozen seed0 four-clause advice per case/condition for all five methods. Fixed regional methods do not refresh. LLINBO receives that advice plus only its own revealed prefix and generates a fresh point each round.',
       'stress_boundary':'correct/wrong_category/mixed were reference-outcome-aware artificial priors. No hidden responses or truth/condition/case labels enter point-provider requests or decision rules.',
       'provider':provider,'api':api.load_config(config_path).public() if provider=='deepseek-flash' else None,
       'request_parameters':PARAMETERS if provider=='deepseek-flash' else None,
       'point_system_sha256':sha256(e.POINT_SYSTEM.encode('utf8')).hexdigest(),
       'provider_change':'Old LLINBO was shared-chat assistant. This rerun uses the specified provider; reviewer mechanism remains identical. Fresh isolated DeepSeek calls make provider type match the new KF framework, but alias is not an immutable backend snapshot.',
       'point_transport':'Physical JSON point+reason+observed support -> unchanged nearest unused same-catalyst numeric projection -> original source vector parser. No score-aware projection.',
       'retries':'First valid response; <=3 received schema attempts, <=8 HTTP attempts per logical proposal; transient retry5/10/20/30s. No outcome-based retries; exhausted failures stay in planned denominator.',
       'comparison':'Reuse immutable four Mixed KF, four frozen-KF and four Mixed B0 trajectories. Baseline changes are newly run. Primary system comparison uses assistant_initial; artificial conditions separately describe baseline stress response. KF was not seeded with these artificial priors, so do not claim knowledge-matched KF robustness under them.',
       'resources':{'GP_processes':GP_WORKERS,'API_processes':API_WORKERS,'threads_each':1,'device':'cpu',
           'warmup':'Mixed kernels/Cholesky/region imported before decision loops','cache':'Each exact Mixed prefix fit once; reuse only identical Mixed recipe/observations.'},
       'evaluation':'SearchAUC mean incumbent at query positions5..20; all planned runs including failures, carry last incumbent. One seed/case; descriptive, no significance claim.'}
    p['protocol_sha256']=digest(p);write(OUT/'protocol.json',p);write(ROOT/'protocol.json',p);write(ROOT/'manifest.json',manifest)
    write(OUT/'start_receipt.json',{'states_before_start':len(list((ROOT/'runs').glob('*/state.json'))),
          'new_framework_results_sha256':filehash(previous.OUT/'results.json')})
    return p


def check_protocol():
    p=read(OUT/'protocol.json');raw=deepcopy(p);key=raw.pop('protocol_sha256')
    assert digest(raw)==key and p['scientific_sha256']==hashes()
    assert p['manifest_sha256']==digest(read(ROOT/'manifest.json'))
    for f,h in p['input_sha256'].items():assert filehash(f)==h,f
    for row in p['protected_inventories'].values():assert filehash(row['path'])==row['sha256']
    assert p['additional_protected_files']==previous.protected_extras()
    return p


def state_for(spec,p,pool):
    path=ROOT/'runs'/spec['run_id']/'state.json'
    if path.exists():return read(path),path
    state={**spec,'status':'RUNNING','history':initial(spec['case'],pool),'decisions':[],
           'protocol_sha256':p['protocol_sha256'],'operator_compiles_in_loop':0,'active_seconds':0.}
    write(path,state);return state,path


def run_regional_case(case):
    p=check_protocol();pool=inputs(case);cache=make_cache(case,pool);ids=pool['ids']
    ix={k:i for i,k in enumerate(ids)};values=np.load(original.ROOT/'inputs'/f'case{case}'/'oracle.npz')['values']
    specs=[v for v in read(ROOT/'manifest.json') if v['case']==case and v['method']!='LLINBO_matched']
    states=[state_for(v,p,pool) for v in specs]
    for t in range(1,17):
        for state,path in states:
            if state['status'] in ('COMPLETE','FAILED') or len(state['history'])!=t+3:continue
            started=time.perf_counter();before=sum(previous.JIT.values())
            try:
                prior=advice(case,state['regime']);member=e.regional.clause_membership(pool['z'],prior)
                base,bmeta=cache.score(state['history']);gmeta={}
                def guided(retained):
                    if not retained:return base
                    g,m=cache.score(state['history'],e.regional.region_for(prior,retained),prior['confidence'])
                    gmeta[','.join(retained)]=m;return g
                queried=np.isin(ids,[h['candidate_id'] for h in state['history']])
                budget=e.regional.budget_at(4,t)
                chosen,event=e.regional.choose_policy(state['method'],base,prior,member,budget,queried,ids,guided)
                if state['method']=='LGBO_region':event.update(budget_logei=None,budget_violation=None)
                else:assert not event['budget_violation']
                state['decisions'].append({'iteration':t,**event,'base_acquisition':bmeta,'guided_acquisitions':gmeta})
                # Hidden surrogate response table is read only after a choice.
                state['history'].append({'candidate_id':ids[chosen],'y':float(values[chosen])})
                if len(state['history'])==20:state['status']='COMPLETE'
            except Exception:
                state.update(status='FAILED',error=traceback.format_exc())
            state['active_seconds']+=time.perf_counter()-started
            state['operator_compiles_in_loop']+=sum(previous.JIT.values())-before;write(path,state)
    receipt={'case':case,'complete':sum(s['status']=='COMPLETE' for s,_ in states),
             'planned':len(states),'failed':[s['run_id'] for s,_ in states if s['status']=='FAILED'],
             'cache_counters':cache.counters,'operator_compiles_in_loop':sum(s['operator_compiles_in_loop'] for s,_ in states)}
    write(ROOT/'groups'/f'c{case}_regional.json',receipt);return receipt


def get_proposal(client,context,folder):
    if (folder/'validated.json').exists():return read(folder/'validated.json')
    base=[{'role':'system','content':e.POINT_SYSTEM},
          {'role':'user','content':json.dumps(context['public'],ensure_ascii=False,separators=(',',':'))}]
    messages=base;received=0
    for attempt in range(1,9):
        directory=folder/f'attempt{attempt:02d}';body=dict(PARAMETERS,messages=messages)
        if (directory/'request.json').exists():assert read(directory/'request.json')==body
        else:write(directory/'request.json',body)
        if (directory/'transport_error.json').exists():
            if not read(directory/'transport_error.json')['retryable']:raise RuntimeError('nonretryable API resource failure')
            continue
        if (directory/'response.json').exists():answer=read(directory/'response.json')
        else:
            try:answer=transport.complete(client,messages);write(directory/'response.json',answer)
            except Exception as error:
                status=error.response.status_code if isinstance(error,httpx.HTTPStatusError) else None
                retryable=isinstance(error,(httpx.TimeoutException,httpx.TransportError)) or status in (429,500,502,503,504)
                write(directory/'transport_error.json',{'error':api.safe_error(error,client.config),'retryable':retryable})
                if not retryable:raise RuntimeError('nonretryable API resource failure') from None
                if attempt<8:time.sleep(min(5*2**(attempt-1),30))
                continue
        received+=1
        try:point=e.parse_point(answer,context['history'])
        except (ValueError,TypeError,KeyError,IndexError) as error:
            error_text=type(error).__name__+': '+str(error)[:200]
            write(directory/'validation_error.json',{'error':error_text})
            if received>=3:raise RuntimeError('received schema limit exhausted') from None
            content=answer['choices'][0]['message'].get('content') if answer.get('choices') else '{}'
            messages=base+[{'role':'assistant','content':content or '{}'},
                {'role':'user','content':'Schema validation failed: '+error_text+'. Correct only the JSON; no experiment has occurred. Return exactly point,reason,support.'}]
            continue
        valid={**point,'context_sha256':context['context_sha256'],'accepted_attempt':attempt,
               'received_responses':received,'response_model':answer['model'],'response_id':answer['id'],
               'response_sha256':filehash(directory/'response.json')}
        write(folder/'validated.json',valid);return valid
    raise RuntimeError('HTTP attempt limit exhausted')


def run_llinbo_case(case,config_path='../config.txt'):
    p=check_protocol();pool=inputs(case);cache=make_cache(case,pool)
    values=np.load(original.ROOT/'inputs'/f'case{case}'/'oracle.npz')['values'];ix={k:i for i,k in enumerate(pool['ids'])}
    specs=[v for v in read(ROOT/'manifest.json') if v['case']==case and v['method']=='LLINBO_matched']
    client=api.APIClient(api.load_config(config_path)) if p['provider']=='deepseek-flash' else None
    states=[]
    try:
        for spec in specs:
            state,path=state_for(spec,p,pool);states.append(state)
            if state['status'] in ('COMPLETE','FAILED'):continue
            started=time.perf_counter();before=sum(previous.JIT.values());prior=advice(case,spec['regime'])
            def assistant(history,prompt,t):
                assert history==state['history'][:len(history)]
                context={'public':e.public_point_request(history,prompt,t,pool,prior,spec['task']),
                         'history':history,'protocol_sha256':p['protocol_sha256']}
                context['context_sha256']=digest(context);folder=ROOT/'requests'/context['context_sha256']
                if (folder/'context.json').exists():assert read(folder/'context.json')==context
                else:write(folder/'context.json',context)
                if client:valid=get_proposal(client,context,folder)
                elif (folder/'validated.json').exists():valid=read(folder/'validated.json')
                else:
                    state.update(status='WAITING_LLM',pending_context=context['context_sha256'])
                    raise AwaitingAssistant(context['context_sha256'])
                projection=e.justify.project_proposal(valid['point'],pool,{r['candidate_id'] for r in history})
                write(folder/'projection.json',projection)
                return projection,valid
            def objective(candidate,event):
                position=event['iteration']+3
                if position<len(state['history']):
                    assert state['history'][position]['candidate_id']==candidate and state['decisions'][position-4]==event
                    return state['history'][position]['y']
                assert position==len(state['history'])
                y=float(values[ix[candidate]])
                state['decisions'].append(event);state['history'].append({'candidate_id':candidate,'y':y})
                state.update(status='RUNNING');state.pop('pending_context',None);write(path,state)
                print(json.dumps({'task':spec['task'],'iteration':event['iteration'],'status':'OBSERVED'},separators=(',',':')),flush=True)
                return y
            try:
                e.justify.execute(pool=pool,initial=state['history'][:4],advice=prior,case=case,seed=0,
                                  ratio=4,cache=cache,assistant=assistant,objective=objective)
                assert len(state['history'])==20;state.update(status='COMPLETE');state.pop('pending_context',None)
            except AwaitingAssistant:pass
            except Exception as error:state.update(status='FAILED',error=api.safe_error(error,client.config) if client else {'kind':type(error).__name__,'message':str(error)[:250]})
            state['active_seconds']+=time.perf_counter()-started
            state['operator_compiles_in_loop']+=sum(previous.JIT.values())-before;write(path,state)
    finally:
        if client:client.close()
    receipt={'case':case,'complete':sum(s['status']=='COMPLETE' for s in states),'planned':len(states),
             'failed':[s['run_id'] for s in states if s['status']=='FAILED'],'cache_counters':cache.counters,
             'operator_compiles_in_loop':sum(s['operator_compiles_in_loop'] for s in states)}
    write(ROOT/'groups'/f'c{case}_llinbo.json',receipt);return receipt


def run(stage,config_path='../config.txt'):
    check_protocol()
    with ProcessPoolExecutor(max_workers=GP_WORKERS if stage=='regional' else API_WORKERS,initializer=worker_init) as executor:
        futures=[executor.submit(run_regional_case,c) if stage=='regional' else executor.submit(run_llinbo_case,c,config_path) for c in (1,2,3,4)]
        for future in as_completed(futures):print(json.dumps(future.result(),separators=(',',':')),flush=True)
    return status()


def status():
    counts={};steps=0;failed=[]
    for f in (ROOT/'runs').glob('*/state.json'):
        row=read(f);counts[row['status']]=counts.get(row['status'],0)+1;steps+=len(row['decisions'])
        if row['status']=='FAILED':failed.append({'run':row['run_id'],'error':row.get('error')})
    return {'runs':counts,'added_queries':steps,'planned_runs':80,'failed':failed}
