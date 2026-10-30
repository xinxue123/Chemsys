"""Event-driven real-LLM residual hypotheses, independent four-case pilot."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import time

import httpx
import numpy as np
import torch

from . import kalman_knowledge_suzuki_engine as e
from . import deepseek_small_suzuki_study as previous
from . import deepseek_suzuki_client as api
from .official_suzuki_adapter import modules,check_sources
from .suzuki_domain import digest,CATALYSTS

ROOT=Path('data/runs/suzuki_kalman_knowledge');OUT=Path('research/suzuki_kalman_knowledge');WORKERS=2
PARENT=previous.PARENT;read=previous.read;write=previous.write;filehash=previous.filehash
PROTECTED=dict(previous.PROTECTED,deepseek_small=previous.OUT/'release_inventory.json')
PARAMETERS=deepcopy(previous.PARAMETERS)
METHODS=('KF_event_B4','KF_frozen_initial_B4');JIT={}
SYSTEM_PROMPT='''You are the chemical-knowledge component of an experimental optimizer.
Use ONLY this task's observed experiments and the supplied GP summaries. Do not recall benchmark optima. Catalyst labels are anonymous categorical identifiers, not molecular structures. Numeric z coordinates are normalized: residence_time 60..600 seconds, temperature 30..110 C, catalyst_loading 0.5..2.5 mol%. Catalyst ids 0..7 map to P1-L1,P2-L1,P1-L2,P1-L3,P1-L4,P1-L5,P1-L6,P1-L7; no ordinal relation is justified.
Propose at most FOUR testable hypotheses about POSITIVE RESIDUAL CONTRAST relative to the CURRENT GP, not raw yield. A hypothesis says that the favored region has effect_pp more yield percentage points of GP residual than its complement. The implementation centers its feature by pool coverage. If the GP already captures a pattern or evidence contradicts it, shrink effect_pp, revise or revoke rather than repeat it as certain chemistry. A separate nuisance offset handles GP global bias. The joint Kalman filter learns coefficients and uncertainty from saved pre-query residuals. Its trust is a conservative heuristic, not a probability of truth. Only added experiments have pre-query residuals; never infer evidence from training residuals of the refitted GP.
Use stable ids H1,H2,H3,H4. Account for EVERY currently active id by retain, revise or revocation. retain MUST keep variable,bounds,categories,effect_pp exactly identical. revise changes at least one of those fields and resets only that slot's calibration. add uses a currently vacant id. Retained slots keep their filter state. Revoked slots are removed. You may return zero hypotheses if none provide incremental information.
Return ONLY one JSON object with exactly hypotheses,revocations,summary. Each hypothesis has EXACTLY id,operation,variable,bounds,categories,effect_pp,reason,support. variable is residence_time,temperature,catalyst_loading or catalyst. Numeric bounds are two normalized numbers in [0,1], at most four decimal places, interval width 0.1..0.9, categories=[]; catalyst uses bounds=[] and 1..7 distinct categories from 0..7. Every predicate must partition the candidate pool. effect_pp is 1..20. reason is 1..1500 characters, support is a nonempty list of one-based indices of OBSERVED experiments. Each revocation has exactly id,reason,support. Summary is 1..2000 characters. Hypothesis ids may not repeat between lists.
Example initial response (illustrative only): {"hypotheses":[{"id":"H1","operation":"add","variable":"temperature","bounds":[0.5,1.0],"categories":[],"effect_pp":5,"reason":"A tentative positive residual contrast to test against the GP, supported by observed experiments 1 and 2.","support":[1,2]}],"revocations":[],"summary":"Test a modest residual contrast; evidence is sparse."}
Keep explanations brief. No arbitrary code or candidate outcome predictions from hidden data. Do not emit reasoning outside the JSON.'''


def scientific_hashes():
    files=[Path(__file__),Path(e.__file__),Path(previous.__file__),Path(api.__file__),Path(e.common.__file__),
           Path('chem/benchmarks/uncertain_suzuki_engine.py'),Path('chem/decision/domain_guard.py'),
           Path('chem/benchmarks/suzuki_kalman_knowledge.py'),Path('tests/test_suzuki_kalman_knowledge.py')]
    return {p.as_posix():filehash(p) for p in files}


def inputs(case):return read(PARENT/'inputs'/f'case{case}'/'pool.json')


def make_cache(case,pool,qualification=False):
    group=f'c{case}_s0';sources=[PARENT/'cache'/group,Path('data/runs/suzuki_llinbo_guarded/cache')/group]
    for label in ('uncertainty','acquisition'):
        sources += [Path(f'data/runs/suzuki_{label}/cache')/(group+'_'+g) for g in (*e.common.REGIMES,'none')]
    return e.KnowledgeCache(pool['z'],pool['ids'],ROOT/('qualification_cache' if qualification else 'cache')/group,case,0,sources=sources)


def worker_init():
    torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    modules('LGBO');__import__('boo');__import__('decide');__import__('prior_monte_carlo')
    torch.linalg.cholesky(torch.eye(4,dtype=torch.double))
    for name in ('script','trace'):
        original=getattr(torch.jit,name)
        # Workers are initialized exactly once, before any counted decision.
        def counted(*args,_original=original,_name=name,**kwargs):
            JIT[_name]+=1;return _original(*args,**kwargs)
        JIT[name]=0;setattr(torch.jit,name,counted)


def qualification():
    worker_init();receipts=[]
    for c in (1,2,3,4):
        pool=inputs(c);old=read(previous.ROOT/'runs'/f'c{c}_s0'/'state.json');h=old['history'][:4];cache=make_cache(c,pool,True)
        base,bmeta=cache.score(h);zero,zmeta=cache.knowledge_score(h,[],{},[]);stats=cache.posterior_stats(h)
        error=float(np.max(np.abs(base-zero)));assert error<1e-11
        assert cache.counters['fits']==0 and stats['fit_key']==bmeta['fit_key']==zmeta['fit_key']
        receipts.append({'case':c,'maximum_zero_shift_logei_error':error,'base_key':bmeta['key'],'zero_key':zmeta['key'],'counters':dict(cache.counters)})
    assert sum(JIT.values())==0
    r={'status':'PASS','cases':receipts,'new_GP_fits':0,'operator_compiles_in_loop':sum(JIT.values()),'torch_version':torch.__version__,
       'component':'Exact zero-knowledge recovery of frozen base qLogEI, same fit and 512 Sobol MC paths; physical-unit residuals.'}
    write(OUT/'qualification.json',r);return r


def prepare(config_path='../config.txt'):
    if (OUT/'protocol.json').exists():return check_protocol()
    q=read(OUT/'qualification.json');assert q['status']=='PASS'
    cfg=api.load_config(config_path);manifest=[{'case':c,'seed':0,'run_id':f'c{c}_s0','task':f'K{c:02d}'} for c in (1,2,3,4)]
    p={'study':'prequential_KF_event_LLM_B4_qLogEI_Suzuki_v1','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
       'scientific_sha256':scientific_hashes(),'protected_inventories':{k:filehash(v) for k,v in PROTECTED.items()},'official_sources':check_sources(),
       'manifest_sha256':digest(manifest),'qualification_sha256':filehash(OUT/'qualification.json'),
       'input_sha256':{f'case{c}/{name}':filehash(PARENT/'inputs'/f'case{c}'/name) for c in (1,2,3,4) for name in ('pool.json','oracle.npz')},
       'cases':[1,2,3,4],'primary_cases':[2,3,4],'seeds':[0],'initial_queries':4,'additional_queries':16,'candidate_count':4692,
       'methods':METHODS,'planned_trajectories':8,'planned_logical_queries':160,'maximum_logical_LLM_calls':16,'parameters':e.PARAMETERS,
       'api':cfg.public(),'request_parameters':PARAMETERS,'system_prompt_sha256':sha256(SYSTEM_PROMPT.encode('utf8')).hexdigest(),
       'knowledge':'Up to4 centered region-indicator residual contrasts in physical yield units, positive effect1..20pp. Anonymous catalysts remain categorical. GPR kernel and encoded inputs unchanged.',
       'filter':'Joint state offset+4 coefficients; random walk, prequery GP residual measurement, observation-inclusive GP variance+3pp floor; Joseph covariance; separate offset-only null. One-SD lower coefficient clipped0..1 supplies trust. Nuisance offset never changes acquisition.',
       'decision':'Unchanged B4 maximum-cardinality/lexical joint diagnosis on trusted predicates and base-qLogEI eligible set. Retained knowledge supplies capped (+/-0.5 latent SD) mean correction; covariance unchanged except common1e-6 jitter. Corrected qLogEI maximized across full eligible set, not hard predicate intersection.',
       'events':'Initial; then >=2-query cooldown, <=4 calls incl initial, no refresh after round14: 3-round stagnation(<0.5pp), two consecutive |innovation z|>2.5, current B4 conflict, or no active knowledge. All thresholds frozen before new outcomes.',
       'LLM_context':'Fresh isolated requests: ONLY own observed history, previously saved prequery residuals, current GP top3 per catalyst, current hypotheses and filter estimates, B4 diagnosis and trigger reasons. No unqueried outcomes, other-policy outcomes, previous-study cards or truth labels.',
       'lifecycle':'retain preserves predicate,effect and covariance; revise/add resets that coefficient and cross covariance; revoke zeroes slot; fixed stable ids, version counters. Existing data not retrospectively reused for filter updates.',
       'retry':'First valid response accepted; <=3 received schema attempts and <=8 HTTP attempts per logical call. Transient transport/429/5xx retries5,10,20,30 seconds. Failure consumes call budget and disables knowledge until valid refresh, falls back to original base BO.',
       'frozen_control':'After all adaptive runs are frozen, reuse ONLY each first accepted hypothesis plan; continue KF/B4/GP on its own new observations, no event refresh and no further API calls. This isolates availability of knowledge refresh at policy level, not individual module causal contributions.',
       'diagnostics':'All decision/innovation/trust/lifecycle/cap/guard receipts; same-prefix noKF,noB4,ungated suggestions offline, not trajectory AUC. Prequential errors compare GP, GP+offset-only, joint KF and deployed correction. No outcome-dependent parameter tuning.',
       'comparators':'Seed0 subsets only: qLogEI B0, UCBs, real DeepSeek direct adaptive/frozen scoring, natural-prior LGBO_region, budget-only LGBO_gate, hard B4_base, soft B4_LGBO, matched LLINBO. Old current-assistant natural knowledge is a system reference, not identical knowledge/model/cadence.',
       'resources':{'persistent_process_workers':WORKERS,'threads_per_worker':1,'device':'cpu','warmup':'LGBO modules and Cholesky before loop; jit script/trace calls counted; exact-prefix fit caching'},
       'limits':['Single fixed seed per case; primary Cases2-4 n=3, reference Case1; descriptive, no significance or chemical truth claim.',
                 'Changing GP, correlated residuals, collinear region features and16 queries limit Kalman identifiability and covariance calibration.',
                 'B4 opportunity guard bounds same-prefix Monte Carlo logEI loss, not actual yield or global regret. Alias not immutable provider model snapshot.'],
       'references':['https://www.cs.unc.edu/~welch/kalman/media/pdf/Kalman1960.pdf','https://botorch.readthedocs.io/en/v0.14.0/acquisition.html','https://api-docs.deepseek.com/guides/thinking_mode/']}
    p['protocol_sha256']=digest(p);write(OUT/'protocol.json',p);write(ROOT/'protocol.json',p);write(ROOT/'manifest.json',manifest)
    write(OUT/'system_prompt.json',{'text':SYSTEM_PROMPT,'sha256':p['system_prompt_sha256']})
    return p


def check_protocol():
    p=read(OUT/'protocol.json');raw=deepcopy(p);key=raw.pop('protocol_sha256');assert digest(raw)==key
    assert p['scientific_sha256']==scientific_hashes() and p['manifest_sha256']==digest(read(ROOT/'manifest.json'))
    assert p['protected_inventories']=={k:filehash(v) for k,v in PROTECTED.items()}
    return p


def public_request(state,pool,stats,base,diagnosis,reasons):
    ix={key:i for i,key in enumerate(pool['ids'])};seen=set(ix[h['candidate_id']] for h in state['history']);obs=[]
    for i,h in enumerate(state['history'],1):
        j=ix[h['candidate_id']];row={'index':i,'cat':CATALYSTS.index(pool['points'][j]['catalyst']),'z':[round(float(v),4) for v in pool['z'][j][:3]],'yield_percent':round(h['y']*100,3)}
        if i>4:
            m=state['decisions'][i-5]['measurement'];row.update(prequery_gp_mean_percent=round(m['prequery_mu_y']*100,3),prequery_gp_predictive_sd_pp=round(m['prequery_predictive_sd_y']*100,3),prequery_residual_pp=round(m['residual']*100,3),joint_innovation_z=round(m['innovation_z'],4))
        obs.append(row)
    summaries=[]
    for cat in range(8):
        available=[j for j,p in enumerate(pool['points']) if CATALYSTS.index(p['catalyst'])==cat and j not in seen]
        for j in sorted(available,key=lambda j:(-float(base[j]),pool['ids'][j]))[:3]:
            summaries.append({'candidate_id':pool['ids'][j],'cat':cat,'z':[round(float(v),4) for v in pool['z'][j][:3]],'gp_mean_percent':round(float(stats['mu_y'][j])*100,3),'gp_latent_sd_pp':round(float(stats['sd_y'][j])*100,3),'base_logei':round(float(base[j]),5)})
    prior=e.forecast(state['filter'],state['hypotheses'])
    return {'task':state['task'],'iteration':len(state['decisions'])+1,'remaining_queries':16-len(state['decisions']),'trigger_reasons':reasons,
            'observations':obs,'current_hypotheses':state['hypotheses'],'filter_forecast':prior,
            'B4_diagnosis':diagnosis.to_dict(),'gp_top3_per_catalyst':summaries,'domain':{'numeric':{'residence_time':[60,600],'temperature':[30,110],'catalyst_loading':[.5,2.5]},'catalysts':dict(enumerate(CATALYSTS))}}


def messages_for(public):return [{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':json.dumps(public,ensure_ascii=False,separators=(',',':'))}]


def get_plan(client,context,previous_hypotheses,z,folder):
    if (folder/'validated.json').exists():return read(folder/'validated.json')
    messages=messages_for(context['public']);received=0
    for attempt in range(1,9):
        directory=folder/f'attempt{attempt:02d}';body=dict(PARAMETERS,messages=messages)
        if (directory/'request.json').exists():assert read(directory/'request.json')==body
        else:write(directory/'request.json',body)
        if (directory/'transport_error.json').exists():
            if not read(directory/'transport_error.json')['retryable']:raise RuntimeError('nonretryable API failure')
            continue
        if (directory/'response.json').exists():answer=read(directory/'response.json')
        else:
            try:answer=previous.complete(client,messages);write(directory/'response.json',answer)
            except Exception as error:
                status=error.response.status_code if isinstance(error,httpx.HTTPStatusError) else None
                retryable=isinstance(error,(httpx.TimeoutException,httpx.TransportError)) or status in (429,500,502,503,504)
                write(directory/'transport_error.json',{'error':api.safe_error(error,client.config),'retryable':retryable})
                if not retryable:raise RuntimeError('nonretryable API failure') from None
                if attempt<8:time.sleep(min(5*2**(attempt-1),30))
                continue
        received+=1
        try:
            if answer['model']!=api.MODEL or answer['choices'][0]['finish_reason']!='stop':raise ValueError('model mismatch or incomplete JSON response')
            plan=e.validate_plan(json.loads(answer['choices'][0]['message']['content']),previous_hypotheses,len(context['history']),z)
        except (ValueError,TypeError,KeyError,IndexError) as error:
            text=type(error).__name__+': '+str(error)[:300];write(directory/'validation_error.json',{'error':text})
            if received>=3:raise RuntimeError('received schema limit exhausted') from None
            content=answer['choices'][0]['message'].get('content') if answer.get('choices') else '{}'
            messages=messages_for(context['public'])+[{'role':'assistant','content':content or '{}'}, {'role':'user','content':'Schema validation failed: '+text+'. Correct ONLY the JSON schema; no experiment has occurred. Return exactly hypotheses,revocations,summary.'}]
            continue
        valid={'plan':plan,'accepted_attempt':attempt,'received_responses':received,'response_id':answer['id'],'response_model':answer['model'],
               'response_sha256':filehash(directory/'response.json'),'context_sha256':context['context_sha256']}
        write(folder/'validated.json',valid);return valid
    raise RuntimeError('HTTP attempt limit exhausted')


def run_one(spec,config_path,frozen=False):
    p=check_protocol();case=spec['case'];pool=inputs(case);ids=pool['ids'];z=np.asarray(pool['z']);ix={key:i for i,key in enumerate(ids)}
    path=ROOT/('frozen' if frozen else 'runs')/spec['run_id']/'state.json';cache=make_cache(case,pool)
    if path.exists():state=read(path)
    else:
        history=deepcopy(read(previous.ROOT/'runs'/spec['run_id']/'state.json')['history'][:4])
        state={**spec,'method':METHODS[int(frozen)],'history':history,'decisions':[],'calls':[],'hypotheses':[],'filter':e.initial_filter(),
               'knowledge_enabled':False,'status':'RUNNING','protocol_sha256':p['protocol_sha256'],'active_seconds':0.,'operator_compiles_in_loop':0}
        if frozen:
            adaptive=read(ROOT/'runs'/spec['run_id']/'state.json');calls=adaptive['calls']
            if calls and calls[0]['status']=='ACCEPTED':
                state['filter'],state['hypotheses']=e.apply_plan(state['filter'],[],calls[0]['plan'],z);state['knowledge_enabled']=True
                state['frozen_source_context']=calls[0]['context_sha256'];state['frozen_initial_plan']=calls[0]['plan']
        write(path,state)
    if state['status']=='COMPLETE':return {'run':spec['run_id'],'method':state['method'],'status':'COMPLETE','steps':16}
    values=np.load(PARENT/'inputs'/f'case{case}'/'oracle.npz')['values']
    client=None if frozen else api.APIClient(api.load_config(config_path))
    if client:assert client.config.public()==p['api']
    try:
        while len(state['decisions'])<16:
            started=time.perf_counter();jit_before=sum(JIT.values());t=len(state['decisions'])+1
            filter_before_call=deepcopy(state['filter']);hypotheses_before_call=deepcopy(state['hypotheses'])
            if not frozen:
                base,_=cache.score(state['history']);stats=cache.posterior_stats(state['history']);prior=e.forecast(state['filter'],state['hypotheses'])
                queried=np.array([key in {h['candidate_id'] for h in state['history']} for key in ids]);weights=prior['weights'] if state['knowledge_enabled'] else {}
                diag=e.diagnose(base,z,state['hypotheses'],weights,e.common.budget_at(4,t),queried,ids)
                reasons=e.event_reasons(t,state['history'],state['decisions'],state['hypotheses'],diag,[c['iteration'] for c in state['calls']])
                if reasons:
                    public=public_request(state,pool,stats,base,diag,reasons);context={'public':public,'history':deepcopy(state['history']),'protocol_sha256':p['protocol_sha256']}
                    context['context_sha256']=digest(context);key=context['context_sha256'];directory=ROOT/'requests'/key;write(directory/'context.json',context)
                    call={'iteration':t,'reasons':reasons,'context_sha256':key}
                    try:
                        valid=get_plan(client,context,state['hypotheses'],z,directory)
                        state['filter'],state['hypotheses']=e.apply_plan(state['filter'],state['hypotheses'],valid['plan'],z);state['knowledge_enabled']=True
                        call.update(status='ACCEPTED',**valid)
                    except RuntimeError as error:
                        state['knowledge_enabled']=False;call.update(status='FAILED',error=str(error))
                    state['calls'].append(call);write(path,state)
            prior=e.forecast(state['filter'],state['hypotheses']);filter_before=deepcopy(state['filter'])
            selected,decision=e.choose(cache,state['history'],z,ids,state['hypotheses'],prior,t,state['knowledge_enabled'])
            # The oracle is read ONLY after the choice has been fixed. The saved
            # GP predictive mean/variance above are not refitted on this value.
            y=float(values[selected]);feature=e.features(z[[selected]],state['hypotheses'])[0]
            state['filter'],measurement=e.update_filter(state['filter'],prior,feature,decision['selected_gp_mu_y'],decision['selected_predictive_sd_y'],y)
            deployed=decision['selected_gp_mu_y']+decision['mean_change_y'];measurement['predictions']['GP_deployed']=deployed;measurement['squared_errors']['GP_deployed']=(y-deployed)**2
            event={'iteration':t,**decision,'filter_before_call':filter_before_call,'hypotheses_before_call':hypotheses_before_call,
                   'filter_before':filter_before,'filter_prior':prior,'hypotheses':deepcopy(state['hypotheses']),'measurement':measurement,'filter_after':deepcopy(state['filter'])}
            state['decisions'].append(event);state['history'].append({'candidate_id':ids[selected],'y':y})
            state['active_seconds']+=time.perf_counter()-started;state['operator_compiles_in_loop']+=sum(JIT.values())-jit_before
            state['status']='COMPLETE' if t==16 else 'RUNNING';write(path,state)
            print(json.dumps({'case':case,'method':state['method'],'round':t,'calls':len(state['calls']),'incumbent_percent':round(max(h['y'] for h in state['history'])*100,3),'trust':{k:round(v,3) for k,v in prior['weights'].items()},'B4':decision['diagnosis']['conflict_kind']},separators=(',',':')),flush=True)
        write(ROOT/'runtime'/f"{spec['run_id']}_{state['method']}.json",{'cache':cache.counters,'operator_compiles_in_loop':state['operator_compiles_in_loop'],'active_seconds':state['active_seconds']})
    finally:
        if client:client.close()
    return {'run':spec['run_id'],'method':state['method'],'status':state['status'],'steps':len(state['decisions'])}


def run(config_path='../config.txt',frozen=False):
    check_protocol();manifest=read(ROOT/'manifest.json')
    if frozen:
        freeze=read(OUT/'adaptive_frozen.json')
        for spec in manifest:assert filehash(ROOT/'runs'/spec['run_id']/'state.json')==freeze['states'][spec['run_id']]
    result=[]
    with ProcessPoolExecutor(max_workers=WORKERS,initializer=worker_init) as executor:
        futures=[executor.submit(run_one,spec,config_path,frozen) for spec in manifest]
        for f in as_completed(futures):result.append(f.result())
    if not frozen:
        assert all(r['status']=='COMPLETE' for r in result)
        write(OUT/'adaptive_frozen.json',{'states':{spec['run_id']:filehash(ROOT/'runs'/spec['run_id']/'state.json') for spec in manifest},
                                       'requests':{f.as_posix():filehash(f) for f in (ROOT/'requests').rglob('*.json')},'frozen_control_states_before_start':len(list((ROOT/'frozen').glob('*/state.json')))})
    return result
