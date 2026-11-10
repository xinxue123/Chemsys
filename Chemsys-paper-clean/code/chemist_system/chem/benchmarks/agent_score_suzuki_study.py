"""Resumable current-assistant scoring, outcome reveal and frozen-rule control."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import time

import numpy as np

from . import agent_score_suzuki_engine as e
from .suzuki_domain import digest,CATALYSTS,NUMERIC
from chem.decision.replay import write_json_atomic

ROOT=Path('data/runs/suzuki_agent_score');OUT=Path('research/suzuki_agent_score')
PARENT=Path('data/runs/suzuki_guarded');WORKERS=8
PROTECTED={k:Path(f'research/suzuki_{k}/release_inventory.json') for k in ('official','guarded','llinbo_guarded','uncertainty','acquisition')}


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def write(path,value):write_json_atomic(path,value)
def filehash(path):return sha256(Path(path).read_bytes()).hexdigest()
def hashes():return {p.name:filehash(p) for p in (Path(__file__),Path(e.__file__))}


def prepare():
    if (OUT/'protocol.json').exists():return check_protocol()
    assert read(OUT/'qualification.json')['status']=='PASS'
    assert read(OUT/'qualification.json')['scientific_sha256']==hashes()
    specs=[{'case':c,'seed':seed,'run_id':f'c{c}_s{seed}'} for c in (1,2,3,4) for seed in range(5)]
    specs.sort(key=lambda r:digest({'run':r['run_id'],'salt':'current-agent-score-v1-20261027'}))
    for i,r in enumerate(specs):r['task']=f'A{i:02d}'
    p={'study':'current_assistant_full_pool_scorecards_v1','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
       'scientific_sha256':hashes(),'manifest_sha256':digest(specs),'protected_inventories':{k:filehash(v) for k,v in PROTECTED.items()},
       'cases':[1,2,3,4],'primary_cases':[2,3,4],'reference_case':[1],'seeds':list(range(5)),
       'candidates_per_case':4692,'initial_queries':4,'additional_queries':16,'repeat_candidates':False,
       'methods':['Agent_score_adaptive','Agent_score_frozen_initial'],'new_trajectories':40,'logical_queries':800,'fresh_assistant_cards':320,
       'no_knowledge_conditions':'No supplied regional advice or quality labels. Match knowledge-free B0 inputs, pool and initial observations.',
       'provider':'Current assistant explicitly selected by user; normal chat authored scorecards, not independent model API calls or immutable model snapshot.',
       'policy':'Every round the assistant chooses eight catalyst biases, three numeric targets/curvatures, novelty weight and up to8 signed observation-centered Gaussian bumps; score all4692 points, mask observed IDs, maximize with lexical tie rule.',
       'formula':'b_c - sum(w_j*(z_j-target_j)^2) + eta*nearest_same_cat_normalized_distance_to_prefix + sum(A_a*exp(-||z-z_a||^2/(2*s_a^2))*same_cat)',
       'score_semantics':'Assistant-authored restricted nonlinear scoring-rule policy. Neither GP/BO nor independent LLM assessment of each point; no learned regression fit. Scores are utilities, not calibrated yields or uncertainties.',
       'experience':'Full own observed prefix and previous scorecard visible each round. Current assistant authors every new card, reason and supporting observation indices before reveal.',
       'frozen_control':'After all320 adaptive cards freeze, reuse first card and initial4 observations for every scoring step; only queried-ID exclusion changes. Outcomes do not update this control scoring.',
       'comparators':'Reuse frozen knowledge-free qLogEI B0, UCB original-beta B0 and previously selected kappa2 UCB B0; no retuning or fresh BO runs.',
       'evaluation':'Mean incumbent yield over16 added positions (SearchAUC pp), final yield, paired differences, feedback benefit versus frozen initial card. All planned rows retained; carry last incumbent after failure.',
       'exposure':'Shared chat has seen earlier case results. Opaque task IDs do not establish blinding; this is exploratory and model/rule-family specific.',
       'scope_limit':'No new candidate generation. No claim about unconstrained per-candidate LLM scoring, automatic knowledge generation or independent causal isolation of a language model architecture.',
       'resources':{'workers':WORKERS,'device':'cpu','vectorized_numpy':True,'GP_fits':0,'JIT_or_operator_compilation':False,'warmup':'One score call before measured selection; no dynamic code/eval.'}}
    p['protocol_sha256']=digest(p);write(OUT/'protocol.json',p);write(ROOT/'protocol.json',p);write(ROOT/'manifest.json',specs)
    for r in specs:
        pool=read(PARENT/'inputs'/f"case{r['case']}"/'pool.json');ix={key:i for i,key in enumerate(pool['ids'])}
        values=np.load(PARENT/'inputs'/f"case{r['case']}"/'oracle.npz')['values']
        history=[{'candidate_id':key,'y':float(values[ix[key]])} for key in pool['initial_ids'][str(r['seed'])]]
        state={**r,'method':'Agent_score_adaptive','history':history,'decisions':[],'status':'WAITING_AGENT','protocol_sha256':p['protocol_sha256'],'active_seconds':0.}
        write(ROOT/'runs'/r['run_id']/'state.json',state)
    return p


def check_protocol():
    p=read(OUT/'protocol.json');raw=deepcopy(p);h=raw.pop('protocol_sha256');assert digest(raw)==h and p['scientific_sha256']==hashes()
    assert p['manifest_sha256']==digest(read(ROOT/'manifest.json'))
    assert p['protected_inventories']=={k:filehash(v) for k,v in PROTECTED.items()}
    return p


def public_request(spec,state):
    pool=read(PARENT/'inputs'/f"case{spec['case']}"/'pool.json');ix={key:i for i,key in enumerate(pool['ids'])}
    request={'task':spec['task'],'iteration':len(state['decisions'])+1,'protocol_sha256':state['protocol_sha256'],
             'history':state['history'],'previous_card':state['decisions'][-1]['card'] if state['decisions'] else None,
             'domain':{'catalysts':list(CATALYSTS),'numeric':list(NUMERIC),'bounds':[[60,600],[30,110],[.5,2.5]]},
             'observations':[{'index':i+1,'id':row['candidate_id'],'cat':CATALYSTS.index(pool['points'][ix[row['candidate_id']]]['catalyst']),
                              'z':pool['z'][ix[row['candidate_id']]][:3],'yield_percent':row['y']*100} for i,row in enumerate(state['history'])]}
    request['context_sha256']=digest(request);return request


def view():
    check_protocol();result=[]
    for spec in read(ROOT/'manifest.json'):
        state=read(ROOT/'runs'/spec['run_id']/'state.json')
        if state['status']!='WAITING_AGENT':continue
        request=public_request(spec,state);path=ROOT/'requests'/request['context_sha256']/'request.json'
        if path.exists():assert read(path)==request
        else:write(path,request)
        result.append({'task':spec['task'],'t':request['iteration'],
                       'H':[[o['index'],o['cat'],*[round(v,4) for v in o['z']],round(o['yield_percent'],3)] for o in request['observations']]})
    return result


def execute_one(job):
    spec,state,card,batch_sha=job;request=public_request(spec,state);h=state['history'];e.validate_card(card,h)
    path=ROOT/'requests'/request['context_sha256']/'response.json'
    response={'context_sha256':request['context_sha256'],'card':card,'origin':'current_assistant','source_batch_sha256':batch_sha}
    if path.exists():assert read(path)==response
    else:write(path,response)
    pool=read(PARENT/'inputs'/f"case{spec['case']}"/'pool.json');ids=pool['ids'];ix={key:i for i,key in enumerate(ids)}
    cats=[CATALYSTS.index(p['catalyst']) for p in pool['points']];hidx=[ix[v['candidate_id']] for v in h]
    z=np.asarray(pool['z']);categories=np.asarray(cats)
    e.score_pool(z[:1],categories[:1],z[hidx],categories[hidx],card)  # fixed NumPy operator warmup
    start=time.perf_counter();scores=e.score_pool(z,categories,z[hidx],categories[hidx],card)
    selected=e.select(scores,ids,{r['candidate_id'] for r in h});score_key=digest({'context':request['context_sha256'],'card':card})
    target=ROOT/'scores'/(score_key+'.npz');target.parent.mkdir(parents=True,exist_ok=True)
    with target.with_suffix('.tmp').open('wb') as handle:np.savez_compressed(handle,scores=scores)
    target.with_suffix('.tmp').replace(target)
    event={'iteration':request['iteration'],'selected_id':ids[selected],'score':float(scores[selected]),'score_key':score_key,
           'context_sha256':request['context_sha256'],'card':card,'candidates_scored':len(scores)}
    # The oracle is opened only after the ranking and selected ID have frozen.
    value=float(np.load(PARENT/'inputs'/f"case{spec['case']}"/'oracle.npz')['values'][selected])
    state['history'].append({'candidate_id':ids[selected],'y':value});state['decisions'].append(event)
    state['active_seconds']+=time.perf_counter()-start
    if len(state['decisions'])==16:state['status']='COMPLETE'
    write(ROOT/'runs'/spec['run_id']/'state.json',state)
    return {'task':spec['task'],'iteration':request['iteration'],'selected_id':ids[selected],'observed_yield':value*100}


def submit(batch_path):
    p=check_protocol();batch=read(batch_path);specs=read(ROOT/'manifest.json');jobs=[]
    waiting={};round_number=batch['round'];assert 1<=round_number<=16
    for spec in specs:
        state=read(ROOT/'runs'/spec['run_id']/'state.json')
        waiting[spec['task']]=(spec,state)
    assert set(batch)=={'round','answers'} and set(batch['answers'])==set(waiting)
    for task,card in batch['answers'].items():
        # Compact list transport only expands fields; it never derives a policy.
        if isinstance(card,list):
            assert len(card)==7;card=dict(zip(('bias','target','curvature','novelty','anchors','reason','support'),card))
        spec,state=waiting[task]
        if len(state['decisions'])>=round_number:
            assert state['decisions'][round_number-1]['card']==card
            continue
        assert len(state['decisions'])+1==round_number;e.validate_card(card,state['history'])
        jobs.append((spec,state,card,filehash(batch_path)))
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:receipts=list(executor.map(execute_one,jobs))
    write(Path(batch_path).with_suffix('.receipt.json'),{'protocol_sha256':p['protocol_sha256'],'round':batch['round'],'selections':receipts})
    return receipts


def frozen_control():
    p=check_protocol();specs=read(ROOT/'manifest.json')
    adaptive=[read(ROOT/'runs'/r['run_id']/'state.json') for r in specs];assert len(adaptive)==20 and all(r['status']=='COMPLETE' for r in adaptive)
    for spec,source in zip(specs,adaptive):
        path=ROOT/'frozen'/spec['run_id']/'state.json'
        if path.exists():assert read(path)['status']=='COMPLETE';continue
        pool=read(PARENT/'inputs'/f"case{spec['case']}"/'pool.json');ids=pool['ids'];ix={key:i for i,key in enumerate(ids)};z=np.asarray(pool['z'])
        cats=np.array([CATALYSTS.index(v['catalyst']) for v in pool['points']]);initial=source['history'][:4];hidx=[ix[r['candidate_id']] for r in initial]
        card=source['decisions'][0]['card'];scores=e.score_pool(z,cats,z[hidx],cats[hidx],card)
        row={**spec,'method':'Agent_score_frozen_initial','history':deepcopy(initial),'decisions':[],'protocol_sha256':p['protocol_sha256'],'status':'RUNNING'}
        values=np.load(PARENT/'inputs'/f"case{spec['case']}"/'oracle.npz')['values']
        for t in range(1,17):
            selected=e.select(scores,ids,{r['candidate_id'] for r in row['history']})
            row['decisions'].append({'iteration':t,'selected_id':ids[selected],'score':float(scores[selected]),'frozen_card_context':source['decisions'][0]['context_sha256']})
            row['history'].append({'candidate_id':ids[selected],'y':float(values[selected])})
        row['status']='COMPLETE';write(path,row)
    return {'frozen_control_complete':20}
