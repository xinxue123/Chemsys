"""Frozen, resumable assistant-in-the-loop extension of the matched study."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from copy import deepcopy
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from chem.decision.replay import write_json_atomic
from . import guarded_suzuki_engine as shared
from . import guarded_suzuki_study as parent
from . import llinbo_guarded_engine as engine
from .official_suzuki_adapter import AwaitingAssistant,check_sources,modules
from .suzuki_domain import CATALYSTS,NUMERIC,digest

ROOT=Path('data/runs/suzuki_llinbo_guarded')
OUT=Path('research/suzuki_llinbo_guarded')
JIT={'script':0,'trace':0};CACHES={}
REGIMES=shared.REGIMES
WORKERS=8

def read(path):return parent.read(path)
def write(path,value):write_json_atomic(path,value)
def filehash(path):return parent.filehash(path)
def scientific_hashes():return {f.name:filehash(f) for f in [Path(__file__),Path(engine.__file__)]}


def prepare():
    if (OUT/'protocol.json').exists():return check_protocol()
    p=parent.check_protocol();check_sources()
    manifest=[]
    for case in (1,2,3,4):
        for seed in range(5):
            for regime in REGIMES:
                key=f'c{case}_s{seed}_{regime}_LLINBO_matched'
                manifest.append({'run_id':key,'case':case,'seed':seed,'regime':regime})
    # Opaque public IDs; no condition quality label or case ID in LLM displays.
    ordered=sorted(manifest,key=lambda s:digest({'salt':'frozen-llinbo-20261027','run':s['run_id']}))
    for i,s in enumerate(ordered):s['public_task']=f'Q{i:02d}'
    protocol={'study':'official_LLINBO_matched_extension_v1','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'parent_protocol_sha256':p['protocol_sha256'],'parent_inventory_sha256':filehash(parent.OUT/'release_inventory.json'),
        'parent_receipt_sha256':filehash(parent.OUT/'release_receipt.json'),
        'cases':[1,2,3,4],'test_cases':[2,3,4],'reference_case':[1],'seeds':list(range(5)),
        'regimes':list(REGIMES),'planned_runs':80,'initial_queries':4,'additional_queries':16,
        'planned_logical_LLM_responses':1280,'planned_logical_queries':1600,
        'initialization':'Exact same four parent initial candidate IDs and observed values.',
        'pool':'Exact same4692 candidates per case; no additions; same-catalyst nearest normalized numeric unqueried projection.',
        'common_BO':'Parent ScoreCache unchanged: official LGBO GP/RBF, standardized target, MC512 qLogEI, shared seeds and covariance jitter1e-6.',
        'review_budget_ratio':read(parent.OUT/'selected_budget.json')['ratio'],
        'budget_selection':'Reuse previously frozen Case1 r=4; no new parameter search.',
        'official_entrypoint':'Unchanged vendor LLMIBO_BFO.run -> _run_justify; native strict > reject / else accept branch executes.',
        'adaptations':[
            'Warm-start replaced by the shared four observations.',
            'train_gp hook uses unchanged parent common GP/qLogEI score cache instead of LLINBO native fitting.',
            'optimize_acqf_ucb hook maximizes the common unshifted qLogEI table on unused pool IDs, replacing continuous UCB optimization.',
            'UpperConfidenceBound hook reads common qLogEI values; source beta is computed/logged but unused.',
            'find_max_variance_bound returns MatchedThreshold; its division hook supplies log(4)*(16-t)/15 instead of initial max variance/t.',
            'Current assistant provides a raw physical point; deterministic coordinates-only same-catalyst projection supplies the vendor JSON vector.',
            'Source sample_candidate_points prompt and parser retained; description adds physical/category mapping and the same four uncertain preference clauses.',
            'Per-prefix prompt shuffle seed removes dependence on cache hits/pause placement.',
            'No repeats: projection searches only unused same-category points.',
            'RNG-independent exact-prefix pause replay returns recorded observations without new oracle queries.'
        ],
        'comparators':['LGBO_region','LGBO_gate','B4_base','B4_LGBO'],
        'comparison_reuse':'Reuse frozen parent trajectories for the exact same case/seed/regime, selected ratio4; no reruns or winner selection.',
        'knowledge_fairness':'All methods receive the same initial four preference clauses. LLINBO turns them plus its own revealed prefix into fresh point advice each round; parent comparators keep the initial regional prior fixed. This is a system comparison, not an isolated comparison of identical evolving knowledge.',
        'LLM':{'provider':'current assistant, explicitly selected by user','fresh_each_round':True,'independent_API_calls':False,
            'immutable_model_snapshot':None,'separate_contexts':False,
            'visible_view':'Initial exact observations/preference bounds; later latest actual observation, incumbent, and observed-only category summaries. Full original prompts/prefixes saved. No BO scores, other methods outputs, unqueried yields or truth labels in view.',
            'exposure':'Current shared chat already contains all-case results and synthetic-prior descriptions; opaque task IDs reduce explicit labels but do not make the study blind.'},
        'evaluation':'SearchAUC mean incumbent over added positions1..16; all planned runs retained, carry incumbent after failures. Paired case/seed contrasts, no global regret or significance claim.',
        'resources':{'persistent_workers':WORKERS,'threads_each':1,'device':'cpu','operator_warmup':'Imports and cholesky before counted loop; jit script/trace calls audited.'},
        'official_sources':check_sources(),'scientific_sha256':scientific_hashes(),'manifest_sha256':digest(ordered)}
    protocol['protocol_sha256']=digest(protocol)
    OUT.mkdir(parents=True,exist_ok=True);ROOT.mkdir(parents=True,exist_ok=True)
    write(OUT/'protocol.json',protocol);write(ROOT/'protocol.json',protocol);write(ROOT/'manifest.json',ordered)
    write(OUT/'start_receipt.json',{'created_utc':protocol['created_utc'],'prior_LLInbo_states':0,
        'parent_results_sha256':filehash(parent.OUT/'results.json'),'selected_budget_sha256':filehash(parent.OUT/'selected_budget.json')})
    return protocol


def check_protocol():
    p=read(OUT/'protocol.json');value=deepcopy(p);h=value.pop('protocol_sha256');assert digest(value)==h
    assert scientific_hashes()==p['scientific_sha256']
    assert digest(read(ROOT/'manifest.json'))==p['manifest_sha256']
    assert filehash(parent.OUT/'release_inventory.json')==p['parent_inventory_sha256']
    return p


def worker_init():
    threadpool_limits(1);torch.set_num_threads(1);modules('LGBO');modules('LLINBO')
    torch.linalg.cholesky(torch.eye(4,dtype=torch.double))
    for name in JIT:
        original=getattr(torch.jit,name)
        def counted(*args,_name=name,_original=original,**kwargs):
            JIT[_name]+=1;return _original(*args,**kwargs)
        setattr(torch.jit,name,counted)


def get_state(spec,p):
    path=ROOT/'runs'/spec['run_id']/'state.json'
    if path.exists():return read(path)
    pool=read(parent.ROOT/'inputs'/f"case{spec['case']}"/'pool.json')
    values=np.load(parent.ROOT/'inputs'/f"case{spec['case']}"/'oracle.npz')['values'];ix={v:i for i,v in enumerate(pool['ids'])}
    initial=[{'candidate_id':v,'y':float(values[ix[v]])} for v in pool['initial_ids'][str(spec['seed'])]]
    state={**spec,'method':'LLINBO_matched','status':'CREATED','history':initial,'decisions':[],
        'protocol_sha256':p['protocol_sha256'],'operator_compiles_in_loop':0,'active_seconds':0.}
    write(path,state);return state


def run_group(specs):
    p=check_protocol();case,seed=specs[0]['case'],specs[0]['seed']
    folder=parent.ROOT/'inputs'/f'case{case}'
    pool=read(folder/'pool.json');priors=read(folder/'advice.json')[str(seed)]
    values=np.load(folder/'oracle.npz')['values'];lookup={v:i for i,v in enumerate(pool['ids'])}
    cache=CACHES.setdefault((case,seed),shared.ScoreCache(pool['z'],pool['ids'],ROOT/'cache'/f'c{case}_s{seed}',case,seed))
    for spec in specs:
        state=get_state(spec,p);path=ROOT/'runs'/spec['run_id']/'state.json'
        if state['status'] in ('COMPLETE','FAILED'):continue
        if state['status']=='WAITING_LLM' and not (ROOT/'requests'/state['pending_context']/'response.json').exists():continue
        advice=priors[spec['regime']];before=sum(JIT.values());started=time.perf_counter()
        def assistant(history,prompt,t):
            assert history==state['history'][:len(history)]
            request={'public_task':spec['public_task'],'iteration':t,'history':history,'advice':advice,
                'source_prompt':prompt,'protocol_sha256':p['protocol_sha256']}
            context=digest(request);request['context_sha256']=context
            req=ROOT/'requests'/context/'request.json';resp=req.with_name('response.json')
            if not req.exists():write(req,request)
            if not resp.exists():
                state['status']='WAITING_LLM';state['pending_context']=context
                raise AwaitingAssistant(context)
            response=read(resp);assert response['context_sha256']==context
            projection=engine.project_proposal(response['raw_point'],pool,{r['candidate_id'] for r in history})
            assert projection==response['projection']
            return projection,response
        def objective(candidate,event):
            t=event['iteration'];position=t+3
            if position<len(state['history']):
                assert state['history'][position]['candidate_id']==candidate
                assert state['decisions'][t-1]==event
                return state['history'][position]['y']
            assert position==len(state['history'])
            value=float(values[lookup[candidate]])
            state['history'].append({'candidate_id':candidate,'y':value});state['decisions'].append(event)
            state['status']='RUNNING';state.pop('pending_context',None);write(path,state)
            return value
        try:
            engine.execute(pool=pool,initial=state['history'][:4],advice=advice,case=case,seed=seed,
                ratio=p['review_budget_ratio'],cache=cache,assistant=assistant,objective=objective)
            assert len(state['history'])==20 and len(state['decisions'])==16
            state['status']='COMPLETE';state.pop('pending_context',None)
        except AwaitingAssistant:pass
        except Exception:
            state['status']='FAILED';state['error']=traceback.format_exc()
        finally:
            state['active_seconds']+=time.perf_counter()-started
            state['operator_compiles_in_loop']+=sum(JIT.values())-before;write(path,state)
    write(ROOT/'groups'/f'c{case}_s{seed}.json',{'case':case,'seed':seed,'score_cache_counters':cache.counters,'operator_compiles_in_loop':sum(JIT.values())})
    return True


def status():
    counts={};steps=0
    for f in (ROOT/'runs').glob('*/state.json'):
        s=read(f);counts[s['status']]=counts.get(s['status'],0)+1;steps+=len(s['decisions'])
    return {'runs':counts,'new_queries':steps,'planned_new_queries':1280}


def run():
    p=check_protocol();manifest=read(ROOT/'manifest.json')
    groups={}
    for s in manifest:groups.setdefault((s['case'],s['seed']),[]).append(s);get_state(s,p)
    inflight={};previous=None
    with ProcessPoolExecutor(max_workers=WORKERS,initializer=worker_init) as pool:
        while True:
            for key,specs in groups.items():
                if key in inflight.values():continue
                ready=False
                for spec in specs:
                    state=read(ROOT/'runs'/spec['run_id']/'state.json')
                    if state['status'] not in ('COMPLETE','FAILED') and (state['status']!='WAITING_LLM' or (ROOT/'requests'/state['pending_context']/'response.json').exists()):ready=True
                if ready and len(inflight)<WORKERS:inflight[pool.submit(run_group,specs)]=key
            if inflight:
                done,_=wait(inflight,timeout=.3,return_when=FIRST_COMPLETED)
                for future in done:future.result();inflight.pop(future)
            else:time.sleep(.3)
            current=status()
            if current!=previous:write(ROOT/'status.json',current);previous=current
            if sum(current['runs'].get(v,0) for v in ('COMPLETE','FAILED'))==80 and not inflight:return current


def pending():
    result={}
    for spec in read(ROOT/'manifest.json'):
        s=read(ROOT/'runs'/spec['run_id']/'state.json')
        if s['status']=='WAITING_LLM':
            q=read(ROOT/'requests'/s['pending_context']/'request.json')
            if not (ROOT/'requests'/s['pending_context']/'response.json').exists():result[spec['public_task']]=(spec,s,q)
    return result


def view(initial=False):
    """Observed-only provider view; no model scores, hidden values or labels."""
    rows=[]
    for public,(spec,state,q) in sorted(pending().items()):
        pool=read(parent.ROOT/'inputs'/f"case{spec['case']}"/'pool.json');ix={v:i for i,v in enumerate(pool['ids'])}
        def point(row):
            p=pool['points'][ix[row['candidate_id']]]
            return [CATALYSTS.index(p['catalyst']),*[round(p[n],3) for n in NUMERIC],round(row['y']*100,3)]
        h=q['history'];best=max(range(len(h)),key=lambda i:h[i]['y'])
        out={'task':public,'t':q['iteration'],'last':[len(h),*point(h[-1])], 'best':[best+1,*point(h[best])]}
        if initial:
            out['H']=[[i+1,*point(row)] for i,row in enumerate(h)]
            out['preference']=[{'j':c['coordinate'],'bounds':[round(v,5) for v in c['interval']]} for c in q['advice']['clauses']]
            out['confidence']=q['advice']['confidence']
        else:
            bycat={}
            for i,row in enumerate(h):
                cat=point(row)[0]
                if cat not in bycat or row['y']>h[bycat[cat]]['y']:bycat[cat]=i
            out['cat_best']=[[cat,i+1,round(h[i]['y']*100,2)] for cat,i in sorted(bycat.items())]
        rows.append(out)
    return rows


def submit(path):
    raw=read(path);waiting=pending();receipts=[]
    for public,a in raw['answers'].items():
        spec,state,q=waiting[public];assert q['iteration']==raw['round']
        # Every coordinate and rationale is authored by the current assistant.
        cat,t,temp,loading,reason,support=a
        assert isinstance(reason,str) and reason.strip() and support
        assert all(isinstance(i,int) and 1<=i<=len(q['history']) for i in support)
        assert isinstance(cat,int) and 0<=cat<8
        point={'catalyst':CATALYSTS[cat],'t_res':t,'temperature':temp,'catalyst_loading':loading}
        pool=read(parent.ROOT/'inputs'/f"case{spec['case']}"/'pool.json')
        projection=engine.project_proposal(point,pool,{r['candidate_id'] for r in q['history']})
        response={'context_sha256':q['context_sha256'],'raw_point':point,'projection':projection,
            'reasoning':reason,'supporting_queries':support,'origin':'current_assistant',
            'source_batch_sha256':filehash(path),'transport':'normalization and nearest unused same-catalyst coordinates only; no score/outcome ranking'}
        target=ROOT/'requests'/q['context_sha256']/'response.json'
        if target.exists():assert read(target)==response
        else:write(target,response)
        receipts.append({'public_task':public,'context_sha256':q['context_sha256'],'response_sha256':digest(response)})
    write(Path(path).with_suffix('.receipt.json'),{'responses':receipts,'source_sha256':filehash(path)})
    return {'submitted':len(receipts),'round':raw['round']}
