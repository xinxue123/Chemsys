"""User-reduced four-case, seed0-only real-API study, separate from interrupted run."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import threading
import time

import httpx
import numpy as np

from . import deepseek_suzuki_client as api
from . import deepseek_agent_suzuki_study as initial
from . import agent_score_suzuki_engine as engine
from .suzuki_domain import digest,CATALYSTS

ROOT=Path('data/runs/suzuki_deepseek_small');OUT=Path('research/suzuki_deepseek_small');WORKERS=2
PARAMETERS=dict(api.PARAMETERS,max_tokens=16384);STOP=threading.Event()
read=initial.read;write=initial.write;filehash=initial.filehash;PARENT=initial.PARENT
PROTECTED=dict(initial.PROTECTED,deepseek_interrupted=initial.OUT/'interruption_inventory.json')


def hashes():return {p.name:filehash(p) for p in (Path(__file__),Path(api.__file__),Path(engine.__file__),Path(initial.__file__))}


def prepare(config_path='../config.txt'):
    if (OUT/'protocol.json').exists():return check_protocol()
    cfg=api.load_config(config_path);manifest=[{'case':c,'seed':0,'run_id':f'c{c}_s0','task':f'R{c:02d}'} for c in (1,2,3,4)]
    for spec in manifest:
        old=read(initial.ROOT/'runs'/spec['run_id']/'state.json');assert not old['decisions']
    p={'study':'deepseek_flash_reduced_seed0_four_case_v1','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
       'scientific_sha256':hashes(),'manifest_sha256':digest(manifest),'protected_inventories':{k:filehash(v) for k,v in PROTECTED.items()},
       'user_scope':'Explicitly reduce evaluations after funding: all4 cases, fixed seed0, no selection by results; preserve each trajectory4+16 query budget.',
       'cases':[1,2,3,4],'primary_cases':[2,3,4],'seeds':[0],'initial_queries':4,'additional_queries':16,'candidate_count':4692,
       'adaptive_trajectories':4,'frozen_control_trajectories':4,'planned_API_decisions':64,'planned_logical_queries':160,
       'api':cfg.public(),'request_parameters':PARAMETERS,'system_prompt_sha256':sha256(api.SYSTEM_PROMPT.encode('utf8')).hexdigest(),
       'input':'Same fixed system prompt, own complete observed history rounded z4/yield%3 decimals, own previous card; fresh isolated API request; no other-seed results, BO scores or unqueried outcomes.',
       'policy':'Unchanged agent_score_suzuki_engine vectorized4692 scoring; every coefficient supplied by deepseek-flash JSON. No local fitting, human card editing or BO fallback.',
       'inference_adaptation':'Increase max_tokens8192->16384 because4 interrupted responses were length-truncated; keep thinking enabled/high. Concurrency8->2 after observed429. This is not a model-only comparison to earlier shared-chat assistant.',
       'retries':'At most3 received model responses per decision for schema repair, at most8 HTTP attempts total; transient429/5xx/transport delays5,10,20,30s;402/401 stop all new calls. First valid card accepted, no outcome-based resampling.',
       'frozen_control':'After64 valid adaptive cards freeze, reuse first card and initial4 geometry for16 unique queries; no additional API calls.',
       'comparators':'Read-only seed0 subsets of qLogEI B0, UCB original and kappa2 B0; old current-assistant rules are separately labeled contextual reference.',
       'interrupted_run':'Original35 valid cards are other seeds; original20-run attempt retained unchanged and excluded from selected performance. No selected seed0 had any completed added observation.',
       'evaluation':'SearchAUC over16 added incumbent positions, final incumbent, actual-query mean and improvements. All8 planned trajectories retained; failures carry incumbent.',
       'limits':'One initialization per case: descriptive single-seed exploratory check, no variance estimate or statistical significance; alias not immutable snapshot; restricted scorecards, not per-point LLM calls.',
       'resources':{'parallel_trajectories':WORKERS,'device':'cpu_numpy','GP_fits':0,'operator_compiles_in_loop':0},
       'references':['https://api-docs.deepseek.com/','https://api-docs.deepseek.com/guides/thinking_mode/','https://api-docs.deepseek.com/quick_start/error_codes/']}
    p['protocol_sha256']=digest(p);write(OUT/'protocol.json',p);write(ROOT/'protocol.json',p);write(ROOT/'manifest.json',manifest)
    write(OUT/'system_prompt.json',{'text':api.SYSTEM_PROMPT,'sha256':p['system_prompt_sha256']})
    for spec in manifest:
        pool=read(PARENT/'inputs'/f"case{spec['case']}"/'pool.json');ix={v:i for i,v in enumerate(pool['ids'])};values=np.load(PARENT/'inputs'/f"case{spec['case']}"/'oracle.npz')['values']
        h=[{'candidate_id':key,'y':float(values[ix[key]])} for key in pool['initial_ids']['0']]
        write(ROOT/'runs'/spec['run_id']/'state.json',{**spec,'method':'DeepSeek_score_adaptive','history':h,'decisions':[],
              'status':'READY','protocol_sha256':p['protocol_sha256'],'api_seconds':0.,'score_seconds':0.})
    return p


def check_protocol():
    p=read(OUT/'protocol.json');raw=deepcopy(p);h=raw.pop('protocol_sha256');assert digest(raw)==h and p['scientific_sha256']==hashes()
    assert p['manifest_sha256']==digest(read(ROOT/'manifest.json'))
    assert p['protected_inventories']=={k:filehash(v) for k,v in PROTECTED.items()}
    return p


def complete(client,messages):
    started=time.perf_counter();response=client.http.post('/chat/completions',json=dict(PARAMETERS,messages=messages));response.raise_for_status()
    data=json.loads(response.text.replace(client.config.api_key,'[REDACTED]'));choices=[]
    for choice in data.get('choices',[]):
        msg=choice.get('message',{});reasoning=msg.get('reasoning_content') or ''
        choices.append({'index':choice.get('index'),'finish_reason':choice.get('finish_reason'),
                        'message':{'role':msg.get('role'),'content':msg.get('content')},'reasoning_characters':len(reasoning),
                        'reasoning_sha256':sha256(reasoning.encode('utf8')).hexdigest()})
    return {'id':data.get('id'),'created':data.get('created'),'model':data.get('model'),'system_fingerprint':data.get('system_fingerprint'),
            'choices':choices,'usage':data.get('usage',{}),'wall_seconds':time.perf_counter()-started,'retained_raw_reasoning':False}


def get_card(client,context,folder):
    if (folder/'validated.json').exists():return read(folder/'validated.json')
    messages=api.messages_for(context['public']);received=0
    for attempt in range(1,9):
        if STOP.is_set():raise RuntimeError('global API resource pause')
        directory=folder/f'attempt{attempt:02d}';body=dict(PARAMETERS,messages=messages);request_path=directory/'request.json'
        if request_path.exists():assert read(request_path)==body
        else:write(request_path,body)
        if (directory/'response.json').exists():answer=read(directory/'response.json')
        elif (directory/'transport_error.json').exists():
            error=read(directory/'transport_error.json')
            if not error['retryable']:STOP.set();raise RuntimeError('nonretryable API resource error')
            continue
        else:
            try:answer=complete(client,messages);write(directory/'response.json',answer)
            except Exception as error:
                status=error.response.status_code if isinstance(error,httpx.HTTPStatusError) else None
                retryable=isinstance(error,(httpx.TimeoutException,httpx.TransportError)) or status in (429,500,502,503,504)
                write(directory/'transport_error.json',{'error':api.safe_error(error,client.config),'retryable':retryable})
                if not retryable:STOP.set();raise RuntimeError('nonretryable API resource error') from None
                if attempt<8:time.sleep(min(5*2**(attempt-1),30))
                continue
        received+=1
        try:card=api.parse_card(answer,context['history'])
        except Exception as error:
            text=type(error).__name__+': '+str(error)[:200];write(directory/'validation_error.json',{'error':text})
            if received>=3:raise RuntimeError('received-model schema attempt limit exhausted')
            content=answer['choices'][0]['message'].get('content') if answer.get('choices') else ''
            messages=api.messages_for(context['public'])+[{'role':'assistant','content':content or '{}'},
                {'role':'user','content':'Your JSON scorecard failed the fixed schema: '+text+'. Correct it, returning exactly the7 JSON fields. No experiment has occurred.'}]
            continue
        valid={'card':card,'accepted_attempt':attempt,'received_responses':received,'response_id':answer['id'],'response_model':answer['model'],
               'response_sha256':filehash(directory/'response.json'),'context_sha256':context['context_sha256']}
        write(folder/'validated.json',valid);return valid
    raise RuntimeError('HTTP attempt limit exhausted')


def run_one(spec,config_path):
    if STOP.is_set():return {'task':spec['task'],'status':'NOT_STARTED_RESOURCE_PAUSE'}
    p=check_protocol();path=ROOT/'runs'/spec['run_id']/'state.json';state=read(path)
    if state['status'] in ('COMPLETE','FAILED'):return {'task':spec['task'],'status':state['status'],'steps':len(state['decisions'])}
    pool=read(PARENT/'inputs'/f"case{spec['case']}"/'pool.json');ids=pool['ids'];ix={v:i for i,v in enumerate(ids)};z=np.asarray(pool['z']);cats=np.array([CATALYSTS.index(v['catalyst']) for v in pool['points']])
    client=api.APIClient(api.load_config(config_path));assert client.config.public()==p['api']
    try:
        while len(state['decisions'])<16:
            public=initial.public_request(state,pool);context={'public':public,'history':deepcopy(state['history']),'protocol_sha256':p['protocol_sha256']};context['context_sha256']=digest(context)
            folder=ROOT/'requests'/context['context_sha256'];target=folder/'context.json'
            if target.exists():assert read(target)==context
            else:write(target,context)
            started=time.perf_counter();validated=get_card(client,context,folder);state['api_seconds']+=time.perf_counter()-started
            card=validated['card'];hidx=[ix[h['candidate_id']] for h in state['history']];engine.score_pool(z[:1],cats[:1],z[hidx],cats[hidx],card)
            started=time.perf_counter();scores=engine.score_pool(z,cats,z[hidx],cats[hidx],card);j=engine.select(scores,ids,{h['candidate_id'] for h in state['history']})
            key=digest({'context':context['context_sha256'],'card':card});score_path=ROOT/'scores'/(key+'.npz');score_path.parent.mkdir(parents=True,exist_ok=True)
            with score_path.with_suffix('.tmp').open('wb') as handle:np.savez_compressed(handle,scores=scores)
            score_path.with_suffix('.tmp').replace(score_path)
            event={'iteration':public['iteration'],'selected_id':ids[j],'score':float(scores[j]),'score_key':key,'context_sha256':context['context_sha256'],
                   'card':card,'accepted_attempt':validated['accepted_attempt'],'response_id':validated['response_id'],'response_model':validated['response_model'],'candidates_scored':4692}
            state['score_seconds']+=time.perf_counter()-started
            value=float(np.load(PARENT/'inputs'/f"case{spec['case']}"/'oracle.npz')['values'][j])
            state['decisions'].append(event);state['history'].append({'candidate_id':ids[j],'y':value});state['status']='COMPLETE' if len(state['decisions'])==16 else 'RUNNING';write(path,state)
    except Exception as error:state.update(status='FAILED',error=api.safe_error(error,client.config));write(path,state)
    finally:client.close()
    return {'task':spec['task'],'status':state['status'],'steps':len(state['decisions'])}


def run(config_path='../config.txt'):
    p=check_protocol()
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        for f in as_completed([executor.submit(run_one,spec,config_path) for spec in read(ROOT/'manifest.json')]):print(json.dumps(f.result(),separators=(',',':')),flush=True)
    rows=[read(ROOT/'runs'/v['run_id']/'state.json') for v in read(ROOT/'manifest.json')]
    result={'protocol_sha256':p['protocol_sha256'],'complete':sum(r['status']=='COMPLETE' for r in rows),'cards':sum(len(r['decisions']) for r in rows),
            'frozen_states_before_start':len(list((ROOT/'frozen').rglob('state.json'))),'state_sha256':{r['run_id']:filehash(ROOT/'runs'/r['run_id']/'state.json') for r in rows}}
    if (OUT/'all_responses_frozen.json').exists():assert read(OUT/'all_responses_frozen.json')['state_sha256']==result['state_sha256']
    else:write(OUT/'all_responses_frozen.json',result)
    return result


def frozen():
    # Reuse the prior fixed-card control logic, rebinding only its output namespace
    # in this dedicated process. The source files and all old artifacts stay intact.
    initial.ROOT=ROOT;initial.OUT=OUT;initial.check_protocol=check_protocol
    initial.frozen_control();assert len(list((ROOT/'frozen').glob('*/state.json')))==4
    return {'frozen_runs':4}
