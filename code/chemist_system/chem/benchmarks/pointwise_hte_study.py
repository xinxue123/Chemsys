"""One seed, full-pool, sequential4+16 comparison after the M2 error audit."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
from pathlib import Path
import argparse
import json
import os
import time

import numpy as np
import torch

from . import reasoning_bo_data as data
from . import pointwise_hte_engine as engine
from . import pointwise_hte_llm as llm
from . import pointwise_hte_justify as justify
from . import deepseek_suzuki_client as api
from .suzuki_domain import digest

ROOT=llm.ROOT
OUT=Path('research/hte_pointwise_seed0')
_WARMED=False


def sources():
    paths=[Path(__file__),Path(engine.__file__),Path(llm.__file__),Path(justify.__file__),
        Path(engine.nominal.__file__),Path(llm.nominal.__file__),Path(llm.native_rbo.__file__),
        Path('chem/benchmarks/guarded_suzuki_engine.py'),Path('chem/decision/domain_guard.py'),
        Path('chem/benchmarks/regional_ablation_suzuki_engine.py'),Path('chem/benchmarks/kalman_knowledge_suzuki_engine.py'),
        Path('chem/benchmarks/reasoning_bo_data.py'),Path('chem/benchmarks/llinbo_guarded_engine.py'),
        Path('third_party/suzuki_official/lgbo/prior.py'),Path('third_party/suzuki_official/llinbo/LLM_agent_BBFO.py'),
        Path('chem/benchmarks/pointwise_hte_report.py'),Path('tests/test_pointwise_hte.py'),Path('pointwise_hte.py')]
    return {str(p.as_posix()):data.filehash(p) for p in paths}


def sobol_initial(pool):
    from scipy.stats import qmc
    rows=qmc.Sobol(len(pool['axes']),scramble=True,seed=0).random_base2(8);selected=[]
    for x in rows:
        point={a:pool['levels'][a][min(int(x[j]*len(pool['levels'][a])),len(pool['levels'][a])-1)] for j,a in enumerate(pool['axes'])}
        try:key=pool['ids'][llm.native_rbo.index_point(pool,point)]
        except ValueError:continue
        if key not in selected:selected.append(key)
        if len(selected)==4:return selected
    raise ValueError('insufficient unique observed feature combinations')


def warm():
    global _WARMED
    if _WARMED:return
    torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    from botorch.models.gp_regression_mixed import MixedSingleTaskGP
    from botorch.acquisition.logei import qLogExpectedImprovement
    from botorch.sampling.normal import SobolQMCNormalSampler
    x=torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]],dtype=torch.double)
    model=MixedSingleTaskGP(x,torch.tensor([[.1],[.4],[.6],[.3]],dtype=torch.double),cat_dims=[0,1]);model.eval()
    with torch.no_grad():qLogExpectedImprovement(model,best_f=.6,
        sampler=SobolQMCNormalSampler(torch.Size([512]),seed=20261028))(x[:1].unsqueeze(1))
    _WARMED=True


def prepare(config_path='../config.txt'):
    if (OUT/'protocol.json').exists():return check()
    manifest=[{'task':task,'seed':0,'method':method,'run_id':task+'_s0_'+method}
        for task in data.SPECS for method in engine.METHODS]
    p={'study':'HTE_pointwise_seed0_corrected_v1','source_sha256':sources(),
        'datasets':list(data.SPECS),'methods':list(engine.METHODS),'seeds':[0],
        'initial_queries':4,'additional_queries':16,'q':1,'trajectories':16,'logical_queries':320,
        'knowledge_condition':'one fresh observed-only deepseek-flash initial plan shared across all knowledge methods; no manual oracle stress priors',
        'parameters':engine.PARAMETERS,'API':api.load_config(config_path).public(),
        'API_parameters':llm.nominal.PARAMETERS,'initialization':'same seeded categorical Sobol4, all8 methods; no LLM warmstart advantage',
        'candidate_scope':'entire author measured pools3696/3955, no fixed-substrate task selection; substrates vary as in M2 data',
        'acquisition':'same MixedSingleTaskGP all categorical, qLogEI(q=1),512 MC,1e-6 common jitter; deterministic observed-prefix fit/MC seeds',
        'error_fixes':['Native regional cached covariance multiplied by scalar outcome stdvs squared; uncached branch untouched.',
            'RBO query_round separated from exact observation_count; first selection has4 observed experiments.',
            'No-KF measurement None is treated as absent innovation in event checks.'],
        'mechanisms':{
            'Pure_BO':'argmax base qLogEI over all unqueried points',
            'LGBO_region':'official regional covariance mean tilt with unit transport; full-pool guided qLogEI maximum',
            'Gate_only':'fixed same regional prior; guided maximum within decaying base opportunity budget',
            'B4_hard':'unchanged exact maximum-cardinality/lexical B4 and hard retained-clause witness, then base qLogEI',
            'B4_soft':'same B4 diagnosis; admitted subset enters soft regional field; guided maximum over full budget-eligible pool',
            'New_R2':'same sequential scalar KF/B4/conflict fallback/event mechanism as Suzuki Case R2, corrected regional units',
            'LLINBO_Justify':'official _run_justify body; nominal proposal I/O and same qLogEI loss-budget review hooks as Suzuki Case',
            'ReasoningBO_point':'official stable high/medium/low sorting n=1; exact top5 base-q1 candidates, full own history; no Notes'},
        'adaptations':['Reasoning BO5-to3 becomes5-to1 under common one-point queries; no claim of paper original protocol reproduction.',
            'LLINBO original UCB/maxvariance hooks are the same matched qLogEI/budget substitutions used in Suzuki Case; algorithm if/else source unchanged.',
            'Region quadrature uses all measured witnesses if<=512, otherwise fixed outcome-independent512 sample.',
            'All categories remain nominal; proposed absent/repeated LLINBO combinations use Hamming nearest unused measured point, fixed lexical tie.',
            'No previous responses/knowledge/GP checkpoints are reused from erroneous M2 trajectories.'],
        'protected':{str(p):data.filehash(p) for p in [Path('research/reasoning_bo_comparison/release_inventory.json'),
            Path('research/reasoning_bo_comparison/release_receipt.json'),Path('research/milestones/M1_Reizman_seed0_20261028/milestone.json')]},
        'inputs':{},'metrics':'SearchAUC mean16 post-query incumbents; final best; first query attaining OWN final best; complete pointwise curves and immutable prequery receipts.',
        'scope_limit':'One seed, one real initial knowledge condition per dataset, descriptive evidence only. No selection based on final outcomes.',
        'resources':{'CPU_workers':4,'API_workers':4,'Torch_threads_per_worker':1,'device':'CPU','operator_compiles_in_loop':0},
        'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    for task in data.SPECS:
        for name in ('pool.json','oracle.npz','compass.json'):
            path=data.ROOT/task/name;p['inputs'][str(path)]=data.filehash(path)
        ids=sobol_initial(data.load_pool(task));data.write(ROOT/'initialization'/task/'committed.json',
            {'initial_ids':ids,'selection':'seed0 nominal Sobol, feature-only, all4 fixed before outcomes'})
        data.write(ROOT/'initialization'/task/'observed.json',data.reveal(task,ids))
    p['manifest_sha256']=digest(manifest);p['protocol_sha256']=digest(p)
    data.write(OUT/'protocol.json',p);data.write(ROOT/'manifest.json',manifest)
    return p


def check():
    p=data.read(OUT/'protocol.json');q=deepcopy(p);h=q.pop('protocol_sha256')
    assert digest(q)==h and p['source_sha256']==sources(),'frozen source changed'
    for name,value in {**p['protected'],**p['inputs']}.items():assert data.filehash(name)==value,name
    assert p['manifest_sha256']==digest(data.read(ROOT/'manifest.json'))
    return p


def bootstrap(task,config_path='../config.txt'):
    warm();p=check();folder=ROOT/'initialization'/task
    if (folder/'ready.json').exists():return data.read(folder/'ready.json')
    pool=data.load_pool(task);history=data.read(folder/'observed.json')
    state={'task':task,'seed':0,'method':'shared_initial_knowledge','history':history,'decisions':[],
        'knowledge':{'confidence':.7,'clauses':[]},'protocol_sha256':p['protocol_sha256']}
    client=api.APIClient(api.load_config(config_path))
    try:
        answer,key=llm.regional_call(client,pool,state,initial=True)
        result={'history':history,'knowledge':answer['plan']['knowledge'],'request_context_sha256':key,
            'validated':answer,'protocol_sha256':p['protocol_sha256']}
        data.write(folder/'ready.json',result);return result
    finally:client.close()


def initial_state(spec,p):
    ready=data.read(ROOT/'initialization'/spec['task']/'ready.json')
    return {**spec,'protocol_sha256':p['protocol_sha256'],'history':deepcopy(ready['history']),
        'knowledge':deepcopy(ready['knowledge']),'initial_knowledge':deepcopy(ready['knowledge']),
        'initial_knowledge_context_sha256':ready['request_context_sha256'],
        'banks':{},'calls':[],'decisions':[],'status':'READY','seconds':0.}


def decision_receipt(d):
    return {k:(v.to_dict() if k=='diagnosis' else v.tolist() if isinstance(v,np.ndarray) else v)
        for k,v in d.items() if k not in ('stats','base','score','available','eligible','raw_shift_std','shift_std')}


def run_one(spec,config_path='../config.txt'):
    warm();p=check();pool=data.load_pool(spec['task']);folder=ROOT/'runs'/spec['run_id'];path=folder/'state.json'
    state=data.read(path) if path.exists() else initial_state(spec,p)
    if state['status']=='COMPLETE':return {'run_id':spec['run_id'],'status':'COMPLETE','queries':16}
    cache=engine.Cache(pool,ROOT/'cache'/spec['task'])
    online=spec['method'] in ('New_R2','LLINBO_Justify','ReasoningBO_point')
    client=api.APIClient(api.load_config(config_path)) if online else None
    reasoner=llm.make_reasoner(client,pool,folder/'native',p['protocol_sha256']) if spec['method']=='ReasoningBO_point' else None
    started=time.perf_counter();index={key:i for i,key in enumerate(pool['ids'])}
    def evaluate(key,d,extra=None):
        iteration=len(state['decisions'])+1;j=index[key]
        assert d['available'][j] and key not in {h['candidate_id'] for h in state['history']}
        prefix=deepcopy(state['history']);best=max(h['y'] for h in prefix)
        table=folder/'rounds'/f'{iteration:02d}_candidates.npz';table.parent.mkdir(parents=True,exist_ok=True)
        with table.with_suffix('.tmp').open('wb') as f:
            np.savez_compressed(f,base_qLogEI=d['base'],guided_qLogEI=d['score'],available=d['available'],eligible=d['eligible'],
                raw_shift_std=d['raw_shift_std'],deployed_shift_std=d['shift_std'],**d['stats'])
        table.with_suffix('.tmp').replace(table)
        receipt={'task':spec['task'],'method':spec['method'],'iteration':iteration,'observation_count':len(prefix),
            'selected_id':key,'history_sha256':digest(prefix),'decision':decision_receipt(d),'extra':extra,
            'candidate_table_sha256':data.filehash(table),'protocol_sha256':p['protocol_sha256']}
        target=folder/'rounds'/f'{iteration:02d}_committed.json';data.write(target,receipt)
        observed=data.reveal(spec['task'],[key])[0];measurement=None
        if spec['method']=='New_R2' and d['bank_key']:
            _,value=cache.model(prefix)
            bank,measurement=engine.nominal.update_bank(d['bank'],d['prior'],iteration,
                float(d['raw_shift_std'][j])*value['std'],float(d['stats']['mu_y'][j]),
                float(d['stats']['sd_observation_y'][j]),observed['y'])
            state['banks'][d['bank_key']]=bank
        loss=float(d['base'][d['reference']]-d['base'][j])
        if spec['method'] not in ('LGBO_region','ReasoningBO_point'):assert loss<=d['budget']
        event={'iteration':iteration,'selected_id':key,'y':observed['y'],'incumbent_before':best,
            'incumbent_after':max(best,observed['y']),'incumbent_gain':max(0.,observed['y']-best),
            'reference_id':pool['ids'][d['reference']],'base_qLogEI':float(d['base'][j]),
            'guided_qLogEI':float(d['score'][j]),'base_opportunity_loss':loss,'budget':d['budget'],
            'eligible_count':int(d['eligible'].sum()),'amplitude':d['amplitude'],'fallback':d['fallback'],
            'retained':d['retained'],'diagnosis':d['diagnosis'].to_dict(),'measurement':measurement,
            'calibration_after':deepcopy(bank) if measurement is not None else None,
            'knowledge':deepcopy(state['knowledge']),'commit_sha256':data.filehash(target),
            'extra':extra,'insight':extra.get('insight') if extra else None}
        state['history'].append(observed);state['decisions'].append(event)
        state['status']='COMPLETE' if iteration==16 else 'RUNNING';data.write(path,state)
        print(json.dumps({'run_id':spec['run_id'],'query':iteration,'status':state['status']}),flush=True)
        return observed['y']
    try:
        if spec['method']=='LLINBO_Justify':
            # Resume by replaying saved own decisions through the same native body;
            # already completed callbacks return saved outcomes without new queries.
            def propose(prefix,prompt,t):
                answer,context=llm.point_call(client,pool,state,prefix,prompt,t)
                projection=engine.nominal.project_proposal(answer['plan']['decoded_point'],pool,prefix)
                return projection,{**answer,'context_sha256':context}
            def objective(key,extra,prefix):
                t=len(prefix)-3
                if t<=len(state['decisions']):
                    old=state['decisions'][t-1];assert old['selected_id']==key;return old['y']
                assert prefix==state['history']
                d=engine.setup(cache,state,t);d['selected']=index[key]
                value=evaluate(key,d,extra);cache.release();return value
            state['native_execution']=justify.execute(pool,state,cache,propose,objective)
        else:
            while len(state['decisions'])<16:
                t=len(state['decisions'])+1;d=engine.setup(cache,state,t)
                reasons=engine.event_reasons(state,d,t)
                if reasons:
                    answer,key=llm.regional_call(client,pool,state,d,reasons)
                    state['calls'].append({'iteration':t,'reasons':reasons,'context_sha256':key,
                        'before':decision_receipt(d),'validated':answer})
                    state['knowledge']=answer['plan']['knowledge'];data.write(path,state)
                    d=engine.setup(cache,state,t)
                selected=pool['ids'][d['selected']];extra=None
                if reasoner:
                    top=engine.shortlist(cache,state,d)
                    selected,insight,context=llm.rbo_call(reasoner,pool,state,top)
                    extra={'shortlist_ids':[pool['ids'][i] for i in top],'insight':insight,'response_context_sha256':context}
                    d['selected']=index[selected]
                evaluate(selected,d,extra);cache.release()
    except Exception as error:
        state['status']='INTERRUPTED';state['error']=api.safe_error(error,client.config) if client else {'kind':type(error).__name__,'message':str(error)[:240]}
    finally:
        state['seconds']+=time.perf_counter()-started;data.write(path,state)
        if client:client.close()
    return {'run_id':spec['run_id'],'status':state['status'],'queries':len(state['decisions']),**({'error':state['error']} if state.get('error') else {})}


def status():
    manifest=data.read(ROOT/'manifest.json');ss=[]
    for spec in manifest:
        path=ROOT/'runs'/spec['run_id']/'state.json';ss.append(data.read(path) if path.exists() else {**spec,'status':'NOT_STARTED','decisions':[]})
    return {'planned':len(manifest),'complete':sum(s['status']=='COMPLETE' for s in ss),'queries':sum(len(s['decisions']) for s in ss),
        'unfinished':[{'run_id':s['run_id'],'status':s['status'],'queries':len(s['decisions']),'error':s.get('error')} for s in ss if s['status']!='COMPLETE']}


def run(config_path='../config.txt'):
    check()
    with ProcessPoolExecutor(max_workers=2,initializer=warm) as ex:
        for f in as_completed([ex.submit(bootstrap,task,config_path) for task in data.SPECS]):f.result()
    with ProcessPoolExecutor(max_workers=4,initializer=warm) as online,ProcessPoolExecutor(max_workers=4,initializer=warm) as offline:
        futures=[]
        for spec in data.read(ROOT/'manifest.json'):
            ex=online if spec['method'] in ('New_R2','LLINBO_Justify','ReasoningBO_point') else offline
            futures.append(ex.submit(run_one,spec,config_path))
        for f in as_completed(futures):print(json.dumps(f.result()),flush=True)
    return status()


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=['prepare','run','status','report','validate'])
    ap.add_argument('--config',default='../config.txt');a=ap.parse_args()
    if a.action=='prepare':result=prepare(a.config)
    elif a.action=='run':result=run(a.config)
    elif a.action=='status':result=status()
    else:
        from . import pointwise_hte_report as report
        result=report.build() if a.action=='report' else report.validate()
    if a.action=='prepare':result={'protocol_sha256':result['protocol_sha256'],'trajectories':16,'queries':320}
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
