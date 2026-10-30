"""Rerun ONLY R2 with persistent region recovery; all controls stay frozen."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import argparse
import json
import math
import time
import numpy as np

from . import r2_region_trust_engine as engine
from . import pointwise_hte_study as parent
from . import pointwise_hte_llm as llm
from . import reasoning_bo_data as data
from . import deepseek_suzuki_client as api
from .suzuki_domain import digest

ROOT=Path('data/runs/hte_r2_region_trust_seed0')
OUT=Path('research/hte_r2_region_trust_seed0')
METHOD='R2_region_trust'


def sources():
    paths=[Path(__file__),Path(engine.__file__),Path('chem/benchmarks/r2_region_trust_report.py'),
        Path('tests/test_r2_region_trust.py'),Path('pointwise_r2_trust.py')]
    return {str(p.as_posix()):data.filehash(p) for p in paths}


def preservation():
    inv=data.read(parent.OUT/'release_inventory.json');receipt=data.read(parent.OUT/'release_receipt.json')
    assert data.filehash(parent.OUT/'release_inventory.json')==receipt['inventory_sha256']
    for row in inv['files']:assert data.filehash(row['path'])==row['sha256'],row['path']
    folder=parent.ROOT/'cache'
    rows=[{'path':p.relative_to(folder).as_posix(),'bytes':p.stat().st_size,'sha256':data.filehash(p)}
        for p in sorted(folder.rglob('*')) if p.is_file() and p.suffix not in ('.lock','.tmp')]
    tree={'files':len(rows),'bytes':sum(r['bytes'] for r in rows),'sha256':digest(rows)}
    assert tree==receipt['cache_tree']
    return {'status':'PASS','old_evidence_files':len(inv['files']),'old_cache':tree,
        'old_inventory_sha256':data.filehash(parent.OUT/'release_inventory.json')}


def prepare(config='../config.txt'):
    if (OUT/'protocol.json').exists():return check()
    old=parent.check();preserved=preservation()
    p={'study':'R2_same_region_recovery_only_v1','parent_protocol_sha256':old['protocol_sha256'],
        'source_sha256':sources(),'datasets':list(data.SPECS),'seed':0,'new_trajectories':2,
        'initial_queries':4,'additional_queries':16,'q':1,'methods_rerun':[METHOD],
        'unchanged_controls':list(engine.original.METHODS),'parameters':engine.PARAMETERS,
        'same_initial_data_and_knowledge':True,'same_Mixed_GP_qLogEI512_candidate_pools_and_budget':True,
        'unchanged_KF_B4_event_formula_and_prompt':True,
        'LLM_reuse':'Reuse an old R2 reply only for the same ordered own observations and identical public prompt; tolerate only numeric roundoff at abs1e-10,rel1e-12. Revalidate and log provenance. New contexts call deepseek-flash.',
        'API':api.load_config(config).public(),'baseline_preservation':preserved,
        'scope':'Only the trust restoration policy changes. No BO topK restriction, coefficient retuning or multiseed.',
        'initialization_sources':{},'CPU_workers':2,'Torch_threads_per_worker':1,'operator_compiles_in_loop':0,
        'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    for task in data.SPECS:
        source=parent.ROOT/'initialization'/task/'ready.json';p['initialization_sources'][str(source)]=data.filehash(source)
        data.write(ROOT/'initialization'/task/'ready.json',data.read(source))
    p['protocol_sha256']=digest(p);data.write(OUT/'protocol.json',p);return p


def check():
    parent.check();p=data.read(OUT/'protocol.json');raw=deepcopy(p);key=raw.pop('protocol_sha256')
    assert digest(raw)==key and p['source_sha256']==sources()
    for path,h in p['initialization_sources'].items():assert data.filehash(path)==h
    return p


def equivalent(a,b):
    if type(a) in (int,float) and type(b) in (int,float):return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-10)
    if isinstance(a,dict) and isinstance(b,dict):return a.keys()==b.keys() and all(equivalent(a[k],b[k]) for k in a)
    if isinstance(a,list) and isinstance(b,list):return len(a)==len(b) and all(equivalent(x,y) for x,y in zip(a,b))
    return type(a)==type(b) and a==b


def request(client,public,state,system,validator,phase):
    context={'public':public,'task':state['task'],'method':'New_R2','observed_prefix':deepcopy(state['history']),
        'phase':phase,'seed':0,'protocol_sha256':state['protocol_sha256']}
    context['context_sha256']=digest(context);folder=ROOT/'requests'/context['context_sha256']
    data.write(folder/'context.json',context)
    if (folder/'validated.json').exists():
        answer=data.read(folder/'validated.json');assert answer['plan']==validator(answer['raw'])
        return answer,context['context_sha256']
    for path in sorted((parent.ROOT/'requests').glob('*/context.json')):
        old=data.read(path)
        if old['task']!=context['task'] or old['method']!='New_R2' or old['phase']!=phase:continue
        if old['observed_prefix']!=context['observed_prefix'] or not equivalent(old['public'],public):continue
        source=path.parent/'validated.json'
        if not source.exists():continue
        saved=data.read(source);plan=validator(saved['raw'])
        assert plan==saved['plan']
        answer={**deepcopy(saved),'context_sha256':context['context_sha256']}
        data.write(folder/'reused.json',{'source':str(source),'source_sha256':data.filehash(source),
            'source_context_sha256':old['context_sha256'],'same_ordered_observations':True,
            'identical_public_prompt':old['public']==public,'numeric_roundoff_equivalent':True})
        data.write(folder/'validated.json',answer);return answer,context['context_sha256']
    answer=llm.nominal.get(client,context,folder,system,validator)
    return answer,context['context_sha256']


def initial(task,p):
    ready=data.read(ROOT/'initialization'/task/'ready.json')
    return {'task':task,'method':'New_R2','display_method':METHOD,'seed':0,'run_id':task+'_s0_'+METHOD,
        'protocol_sha256':p['protocol_sha256'],'history':deepcopy(ready['history']),
        'knowledge':deepcopy(ready['knowledge']),'initial_knowledge':deepcopy(ready['knowledge']),
        'initial_knowledge_context_sha256':ready['request_context_sha256'],
        'banks':{},'region_trust':{},'calls':[],'decisions':[],'status':'READY','seconds':0.}


def run_one(task,config='../config.txt'):
    parent.warm();p=check();pool=data.load_pool(task);state=initial(task,p)
    folder=ROOT/'runs'/state['run_id'];path=folder/'state.json'
    if path.exists():state=data.read(path)
    if state['status']=='COMPLETE':return {'task':task,'status':'COMPLETE','queries':16}
    state.pop('error',None)
    cache=engine.Cache(pool,ROOT/'cache'/task);cache.sources=[parent.ROOT/'cache'/task]
    client=api.APIClient(api.load_config(config));start=time.perf_counter()
    try:
        while len(state['decisions'])<16:
            t=len(state['decisions'])+1;d=engine.setup(cache,state,t);reasons=engine.event_reasons(state,d,t)
            if reasons:
                with patch.object(llm,'request',request):answer,key=llm.regional_call(client,pool,state,d,reasons)
                state['calls'].append({'iteration':t,'reasons':reasons,'context_sha256':key,
                    'before':parent.decision_receipt(d),'validated':answer})
                state['knowledge']=answer['plan']['knowledge'];data.write(path,state)
                d=engine.setup(cache,state,t)
            j=d['selected'];key=pool['ids'][j];prefix=deepcopy(state['history']);best=max(h['y'] for h in prefix)
            assert d['available'][j] and d['eligible'][j]
            table=folder/'rounds'/f'{t:02d}_candidates.npz';table.parent.mkdir(parents=True,exist_ok=True)
            with table.with_suffix('.tmp').open('wb') as f:
                np.savez_compressed(f,base_qLogEI=d['base'],guided_qLogEI=d['score'],available=d['available'],eligible=d['eligible'],
                    raw_shift_std=d['raw_shift_std'],deployed_shift_std=d['shift_std'],**d['stats'])
            table.with_suffix('.tmp').replace(table)
            target=folder/'rounds'/f'{t:02d}_committed.json'
            data.write(target,{'task':task,'iteration':t,'observation_count':len(prefix),'selected_id':key,
                'history_sha256':digest(prefix),'decision':parent.decision_receipt(d),
                'region_trust_at_commit':deepcopy(state['region_trust']),
                'candidate_table_sha256':data.filehash(table),'protocol_sha256':p['protocol_sha256']})
            observed=data.reveal(task,[key])[0];measurement=bank=trust_feedback=None
            if d['bank_key']:
                _,v=cache.model(prefix)
                bank,measurement=engine.original.nominal.update_bank(d['bank'],d['prior'],t,
                    float(d['raw_shift_std'][j])*v['std'],float(d['stats']['mu_y'][j]),
                    float(d['stats']['sd_observation_y'][j]),observed['y'])
                state['banks'][d['bank_key']]=bank;d['_measurement']=measurement
                trust_feedback=engine.feedback(state,d,t,observed['y'],bank)
            loss=float(d['base'][d['reference']]-d['base'][j]);assert loss<=d['budget']
            event={'iteration':t,'selected_id':key,'y':observed['y'],'incumbent_before':best,
                'incumbent_after':max(best,observed['y']),'incumbent_gain':max(0.,observed['y']-best),
                'reference_id':pool['ids'][d['reference']],'base_qLogEI':float(d['base'][j]),
                'guided_qLogEI':float(d['score'][j]),'base_opportunity_loss':loss,'budget':d['budget'],
                'eligible_count':int(d['eligible'].sum()),'amplitude':d['amplitude'],'fallback':d['fallback'],
                'retained':d['retained'],'diagnosis':d['diagnosis'].to_dict(),'measurement':measurement,
                'calibration_after':deepcopy(bank),'knowledge':deepcopy(state['knowledge']),
                'trust':deepcopy(d['trust']),'trust_feedback':trust_feedback,
                'region_trust_after_feedback':deepcopy(state['region_trust']),'commit_sha256':data.filehash(target)}
            state['history'].append(observed);state['decisions'].append(event)
            state['status']='COMPLETE' if t==16 else 'RUNNING';data.write(path,state);cache.release()
            print(json.dumps({'task':task,'query':t,'best_percent':max(best,observed['y'])*100,
                'trust_blocked':d['trust']['trust_blocked'],'status':state['status']}),flush=True)
    except Exception as error:
        state['status']='INTERRUPTED';state['error']=api.safe_error(error,client.config)
    finally:
        state['seconds']+=time.perf_counter()-start;data.write(path,state);client.close()
    return {'task':task,'status':state['status'],'queries':len(state['decisions']),'error':state.get('error')}


def states():
    return [data.read(p) for p in (ROOT/'runs').glob('*/state.json')]


def run(config='../config.txt'):
    check()
    with ProcessPoolExecutor(max_workers=2,initializer=parent.warm) as ex:
        for f in as_completed([ex.submit(run_one,task,config) for task in data.SPECS]):print(json.dumps(f.result()),flush=True)
    return {'complete':sum(s['status']=='COMPLETE' for s in states()),'planned':2}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=['prepare','run','status','validate','report','seal'])
    ap.add_argument('--config',default='../config.txt');a=ap.parse_args()
    if a.action=='prepare':p=prepare(a.config);result={'protocol_sha256':p['protocol_sha256'],'trajectories':2}
    elif a.action=='run':result=run(a.config)
    elif a.action=='status':result=[{'run_id':s['run_id'],'status':s['status'],'queries':len(s['decisions']),'error':s.get('error')} for s in states()]
    else:
        from . import r2_region_trust_report as report
        result={'validate':report.validate,'report':report.build,'seal':report.seal}[a.action]()
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
