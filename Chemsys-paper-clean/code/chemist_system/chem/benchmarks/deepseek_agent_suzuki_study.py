"""Isolated DeepSeek scorecard requests on the unchanged Suzuki candidate pool."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from copy import deepcopy
from pathlib import Path
import json
import time

import httpx
import numpy as np

from . import deepseek_suzuki_client as api
from . import agent_score_suzuki_engine as engine
from . import agent_score_suzuki_study as previous
from .suzuki_domain import digest,CATALYSTS

ROOT=Path('data/runs/suzuki_deepseek_agent');OUT=Path('research/suzuki_deepseek_agent');WORKERS=8
read=previous.read;write=previous.write;filehash=previous.filehash;PARENT=previous.PARENT
PROTECTED=dict(previous.PROTECTED,agent_score=previous.OUT/'release_inventory.json')


def hashes():return {p.name:filehash(p) for p in (Path(__file__),Path(api.__file__),Path(engine.__file__))}


def prepare(config_path='../config.txt'):
    if (OUT/'protocol.json').exists():return check_protocol()
    preflight=read(OUT/'preflight.json');assert preflight['status']=='PASS' and preflight['valid_card']
    assert preflight['system_prompt_sha256']==digest_text(api.SYSTEM_PROMPT)
    cfg=api.load_config(config_path);assert cfg.public()==preflight['api']
    manifest=[{'case':c,'seed':seed,'run_id':f'c{c}_s{seed}'} for c in (1,2,3,4) for seed in range(5)]
    manifest.sort(key=lambda r:digest({'run':r['run_id'],'salt':'deepseek-independent-prefix-20261027'}))
    for j,spec in enumerate(manifest):spec['task']=f'D{j:02d}'
    p={'study':'deepseek_flash_isolated_scorecard_Suzuki_v1','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
       'scientific_sha256':hashes(),'manifest_sha256':digest(manifest),'protected_inventories':{k:filehash(v) for k,v in PROTECTED.items()},
       'api':cfg.public(),'request_parameters':api.PARAMETERS,'system_prompt_sha256':digest_text(api.SYSTEM_PROMPT),
       'preflight_sha256':filehash(OUT/'preflight.json'),'model_catalog_sha256':filehash(OUT/'model_catalog.json'),
       'model_identity':'Requested deepseek-flash, preserve returned model ID/fingerprint. Official docs currently map the alias to DeepSeek-V4.1-Flash-1017; no immutable snapshot guarantee.',
       'provider_documents':['https://api-docs.deepseek.com/','https://api-docs.deepseek.com/guides/thinking_mode/','https://api-docs.deepseek.com/guides/json_mode/'],
       'cases':[1,2,3,4],'primary_cases':[2,3,4],'reference_case':[1],'seeds':list(range(5)),
       'initial_queries':4,'additional_queries':16,'candidate_count':4692,'repeat_candidates':False,
       'methods':['DeepSeek_score_adaptive','DeepSeek_score_frozen_initial'],'planned_trajectories':40,'planned_logical_queries':800,
       'planned_valid_scorecards':320,'maximum_API_attempts_per_decision':3,'new_trajectory_LLM_inputs':'No previous-study cards, results, other trajectories, GP scores, regional prior or unqueried yields.',
       'input_precision':'Same compact observed view as previous assistant study: z rounded4 decimals, yield% rounded3; exact observations retained locally.',
       'input_state':'Fresh API request with the fixed system prompt, own complete observed history and own immediately previous card. No shared conversation or provider conversation ID.',
       'policy':'Use unchanged agent_score_suzuki_engine scoring family and validation. Provider supplies all7 card fields. Vectorized full4692 scoring; mask queried IDs and deterministic lexical tie.',
       'schema_repair':'First valid card accepted, at most3 API attempts per decision. Repair only explains schema failure; no extra oracle outcomes or human card edits. Transient HTTP429/5xx/timeout may retry within the same bound, waiting2 then4seconds.',
       'no_fallback':'No local model, manual card, BO or other provider substitutes a failed API answer. Failed trajectories retain their incumbent over unused budget for reporting.',
       'frozen_control':'After all adaptive trajectories finish, first valid card and initial4 geometry remain fixed; exclude already queried points; no further LLM calls.',
       'comparators':'Read-only old qLogEI B0, UCB original and fixed-kappa2 B0; current-assistant rule study is separately labeled exploratory reference.',
       'evaluation':'SearchAUC mean incumbent over16 added queries, final yield, paired deltas, added-query yield and improvement counts; all40 planned rows retained.',
       'comparability_boundary':'Same rule executor, pool and budgets; API isolated high-thinking model versus old shared-chat assistant has different provider/prompt/context. No broad model-quality or chemical-knowledge causal claim.',
       'resources':{'parallel_trajectories':WORKERS,'device':'cpu_numpy','GP_fits':0,'operator_compilation':False,'API_connections':'one reused HTTP client per trajectory','HTTP_read_timeout_seconds':180},
       'privacy':'Credential read from user config at runtime; never copied, hashed or serialized. API bodies contain only observed public task state. Raw private reasoning text omitted, retain length/hash and usage.'}
    p['protocol_sha256']=digest(p);write(OUT/'protocol.json',p);write(ROOT/'protocol.json',p);write(ROOT/'manifest.json',manifest)
    write(OUT/'system_prompt.json',{'text':api.SYSTEM_PROMPT,'sha256':p['system_prompt_sha256']})
    for spec in manifest:
        pool=read(PARENT/'inputs'/f"case{spec['case']}"/'pool.json');ix={v:i for i,v in enumerate(pool['ids'])}
        values=np.load(PARENT/'inputs'/f"case{spec['case']}"/'oracle.npz')['values']
        h=[{'candidate_id':key,'y':float(values[ix[key]])} for key in pool['initial_ids'][str(spec['seed'])]]
        write(ROOT/'runs'/spec['run_id']/'state.json',{**spec,'method':'DeepSeek_score_adaptive','history':h,'decisions':[],
            'status':'READY','protocol_sha256':p['protocol_sha256'],'api_seconds':0.,'score_seconds':0.})
    return p


def digest_text(value):
    from hashlib import sha256
    return sha256(value.encode('utf8')).hexdigest()


def check_protocol():
    p=read(OUT/'protocol.json');raw=deepcopy(p);h=raw.pop('protocol_sha256');assert digest(raw)==h and p['scientific_sha256']==hashes()
    assert digest(read(ROOT/'manifest.json'))==p['manifest_sha256']
    assert p['protected_inventories']=={k:filehash(v) for k,v in PROTECTED.items()}
    return p


def public_request(state,pool):
    ix={key:i for i,key in enumerate(pool['ids'])};t=len(state['decisions'])+1
    return {'task':state['task'],'iteration':t,'remaining_queries':17-t,
            'domain':{'catalysts':list(CATALYSTS),'numeric':['residence_time','temperature','catalyst_loading'],
                      'physical_bounds':[[60,600],[30,110],[.5,2.5]],'coordinate_order':'normalized residence,temperature,loading'},
            'observations':[{'index':j+1,'cat':CATALYSTS.index(pool['points'][ix[h['candidate_id']]]['catalyst']),
                             'z':[round(float(v),4) for v in pool['z'][ix[h['candidate_id']]][:3]],'yield_percent':round(h['y']*100,3)} for j,h in enumerate(state['history'])],
            'previous_card':state['decisions'][-1]['card'] if state['decisions'] else None}


def get_card(client,request,history,folder):
    valid_path=folder/'validated.json'
    if valid_path.exists():
        validated=read(valid_path);answer=read(folder/f"attempt{validated['accepted_attempt']:02d}"/'response.json')
        assert api.parse_card(answer,history)==validated['card'];return validated
    messages=api.messages_for(request['public']);last_error=None
    for attempt in range(1,4):
        directory=folder/f'attempt{attempt:02d}';body=dict(api.PARAMETERS,messages=messages)
        path=directory/'request.json'
        if path.exists():assert read(path)==body
        else:write(path,body)
        response_path=directory/'response.json';error_path=directory/'transport_error.json'
        if response_path.exists():answer=read(response_path)
        elif error_path.exists():
            error=read(error_path)
            if not error['retryable']:raise RuntimeError('nonretryable API error')
            last_error=error;continue
        else:
            try:answer=client.complete(messages);write(response_path,answer)
            except Exception as error:
                status=error.response.status_code if isinstance(error,httpx.HTTPStatusError) else None
                retryable=isinstance(error,(httpx.TimeoutException,httpx.TransportError)) or status in (429,500,502,503,504)
                receipt={'error':api.safe_error(error,client.config),'retryable':retryable};write(error_path,receipt);last_error=receipt
                if not retryable:raise RuntimeError('nonretryable API error') from None
                if attempt<3:time.sleep(2**attempt)
                continue
        try:card=api.parse_card(answer,history)
        except Exception as error:
            text=type(error).__name__+': '+str(error)[:200];write(directory/'validation_error.json',{'error':text});last_error=text
            content=answer['choices'][0]['message'].get('content') if answer.get('choices') else ''
            messages=api.messages_for(request['public'])+[{'role':'assistant','content':content or '{}'},
                {'role':'user','content':'Your JSON scorecard failed the fixed schema: '+text+'. Return a corrected JSON object with exactly the7 fields. No new experiment has occurred.'}]
            continue
        validated={'card':card,'accepted_attempt':attempt,'response_model':answer['model'],'response_id':answer['id'],
                   'system_fingerprint':answer.get('system_fingerprint'),'response_sha256':filehash(response_path),
                   'context_sha256':request['context_sha256']}
        write(valid_path,validated);return validated
    raise RuntimeError('API/schema attempt limit exhausted')


def run_trajectory(spec,config_path):
    p=check_protocol();path=ROOT/'runs'/spec['run_id']/'state.json';state=read(path)
    if state['status'] in ('COMPLETE','FAILED'):return {'task':spec['task'],'status':state['status'],'steps':len(state['decisions'])}
    pool=read(PARENT/'inputs'/f"case{spec['case']}"/'pool.json');ids=pool['ids'];ix={key:i for i,key in enumerate(ids)}
    z=np.asarray(pool['z']);cats=np.array([CATALYSTS.index(v['catalyst']) for v in pool['points']]);client=api.APIClient(api.load_config(config_path))
    assert client.config.public()==p['api']
    try:
        while len(state['decisions'])<16:
            public=public_request(state,pool);request={'public':public,'history':deepcopy(state['history']),'protocol_sha256':p['protocol_sha256']}
            request['context_sha256']=digest(request);folder=ROOT/'requests'/request['context_sha256'];request_path=folder/'context.json'
            if request_path.exists():assert read(request_path)==request
            else:write(request_path,request)
            started=time.perf_counter();validated=get_card(client,request,state['history'],folder);state['api_seconds']+=time.perf_counter()-started
            card=validated['card'];hidx=[ix[h['candidate_id']] for h in state['history']]
            engine.score_pool(z[:1],cats[:1],z[hidx],cats[hidx],card)
            started=time.perf_counter();scores=engine.score_pool(z,cats,z[hidx],cats[hidx],card)
            selected=engine.select(scores,ids,{h['candidate_id'] for h in state['history']})
            score_key=digest({'context':request['context_sha256'],'card':card});target=ROOT/'scores'/(score_key+'.npz');target.parent.mkdir(parents=True,exist_ok=True)
            with target.with_suffix('.tmp').open('wb') as handle:np.savez_compressed(handle,scores=scores)
            target.with_suffix('.tmp').replace(target)
            event={'iteration':public['iteration'],'selected_id':ids[selected],'score':float(scores[selected]),'score_key':score_key,
                   'context_sha256':request['context_sha256'],'card':card,'accepted_attempt':validated['accepted_attempt'],
                   'response_model':validated['response_model'],'response_id':validated['response_id'],'candidates_scored':len(scores)}
            state['score_seconds']+=time.perf_counter()-started
            # Ranking/card/score artifacts are persisted before the outcome reveal.
            value=float(np.load(PARENT/'inputs'/f"case{spec['case']}"/'oracle.npz')['values'][selected])
            state['decisions'].append(event);state['history'].append({'candidate_id':ids[selected],'y':value})
            state['status']='COMPLETE' if len(state['decisions'])==16 else 'RUNNING';write(path,state)
    except Exception as error:
        state.update(status='FAILED',error=api.safe_error(error,client.config));write(path,state)
    finally:client.close()
    return {'task':spec['task'],'status':state['status'],'steps':len(state['decisions'])}


def run(config_path='../config.txt'):
    check_protocol();manifest=read(ROOT/'manifest.json')
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        for future in as_completed([executor.submit(run_trajectory,spec,config_path) for spec in manifest]):print(json.dumps(future.result(),separators=(',',':')),flush=True)
    states=[read(ROOT/'runs'/spec['run_id']/'state.json') for spec in manifest]
    receipt={'completed':sum(r['status']=='COMPLETE' for r in states),'failed':sum(r['status']=='FAILED' for r in states),
             'cards':sum(len(r['decisions']) for r in states),'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
             'frozen_states_before_start':len(list((ROOT/'frozen').rglob('state.json'))),
             'adaptive_states_sha256':{r['run_id']:filehash(ROOT/'runs'/r['run_id']/'state.json') for r in states}}
    frozen_path=OUT/'all_responses_frozen.json'
    if frozen_path.exists():
        prior=read(frozen_path);assert prior['adaptive_states_sha256']==receipt['adaptive_states_sha256'];return prior
    write(frozen_path,receipt);return receipt


def frozen_control():
    p=check_protocol();assert (OUT/'all_responses_frozen.json').exists()
    for spec in read(ROOT/'manifest.json'):
        path=ROOT/'frozen'/spec['run_id']/'state.json'
        if path.exists():continue
        source=read(ROOT/'runs'/spec['run_id']/'state.json');pool=read(PARENT/'inputs'/f"case{spec['case']}"/'pool.json');ids=pool['ids'];ix={key:i for i,key in enumerate(ids)}
        initial=source['history'][:4];row={**spec,'method':'DeepSeek_score_frozen_initial','history':deepcopy(initial),'decisions':[],
                                         'protocol_sha256':p['protocol_sha256'],'status':'FAILED'}
        if source['decisions']:
            card=source['decisions'][0]['card'];z=np.asarray(pool['z']);cats=np.array([CATALYSTS.index(v['catalyst']) for v in pool['points']]);hidx=[ix[h['candidate_id']] for h in initial]
            scores=engine.score_pool(z,cats,z[hidx],cats[hidx],card);values=np.load(PARENT/'inputs'/f"case{spec['case']}"/'oracle.npz')['values']
            for t in range(1,17):
                j=engine.select(scores,ids,{h['candidate_id'] for h in row['history']})
                row['decisions'].append({'iteration':t,'selected_id':ids[j],'score':float(scores[j]),'initial_card_context':source['decisions'][0]['context_sha256']})
                row['history'].append({'candidate_id':ids[j],'y':float(values[j])})
            row['status']='COMPLETE'
        write(path,row)
    return {'frozen_runs':20}
