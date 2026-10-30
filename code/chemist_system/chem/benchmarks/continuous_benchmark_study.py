"""Reproducible SnAr / HPLC comparison, seed0, fixed pools, common4+16."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from contextlib import ExitStack,redirect_stdout
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import csv
import io
import json
import time
import traceback
import numpy as np
import torch
from . import continuous_benchmark_data as data
from . import continuous_benchmark_engine as e
from . import continuous_benchmark_llm as llm
from . import deepseek_suzuki_client as api
from .official_suzuki_adapter import modules
from .suzuki_domain import digest

ROOT=Path('data/runs/continuous_20261029');OUT=data.OUT
TASKS=('SnAr','HPLC')

def sources():
    paths=[Path(__file__),Path(data.__file__),Path(e.__file__),Path(llm.__file__),Path('continuous_benchmarks.py'),
        Path('third_party/suzuki_official/lgbo/prior.py'),Path('third_party/suzuki_official/lgbo/decide.py'),
        Path('third_party/suzuki_official/llinbo/LLM_agent_BBFO.py'),Path('third_party/reasoning_bo_official/src/bo/reasoner/base.py'),
        Path('chem/benchmarks/r2_region_trust_engine.py'),Path('chem/benchmarks/regional_ablation_suzuki_engine.py'),
        Path('chem/decision/domain_guard.py'),Path('chem/benchmarks/stable_information_engine.py'),
        Path('chem/benchmarks/continuous_benchmark_report.py'),Path('tests/test_continuous_benchmarks.py')]
    return {p.as_posix():data.filehash(p) for p in paths}

def pool(task):return data.read(data.ROOT/task/'pool.json')
def initial(task):
    p=pool(task)
    return [data.reveal(task,k) for k in p['initial_ids']]

def qualify():
    e.warm();rows=[]
    for task in TASKS:
        p=pool(task);h=initial(task);cache=e.Cache(p);v=cache.model(h);base,meta=cache.score(h)
        assert np.isfinite(base).all() and np.isfinite(v['mu_y']).all() and (v['sd_y']>0).all()
        dim=len(p['axes']);tilt=e.CorrectedTilt(torch.tensor([[0.]*dim,[1.]*dim],dtype=torch.double),grid_size=512,smooth=.08)
        tilt.set_box_region([.15]*dim,[.85]*dim);tilt.fit_lambda_by_delta(v['model'],1.);tilt.prepare_cache(v['model'])
        xx=cache.X[:7].unsqueeze(1)
        with torch.no_grad():cached=tilt.posterior(v['model'],xx).mean
        saved=tilt._cache;tilt._cache=None
        with torch.no_grad():uncached=tilt.posterior(v['model'],xx).mean
        err=float((cached-uncached).abs().max());assert err<1e-8,(task,err)
        # Independent qLogEI against a real GP wrapper on sampled points.
        from botorch.models.model import Model
        from gpytorch.distributions import MultivariateNormal
        from botorch.posteriors.gpytorch import GPyTorchPosterior
        from botorch.acquisition.logei import qLogExpectedImprovement
        from botorch.sampling.normal import SobolQMCNormalSampler
        class JitterModel(Model):
            @property
            def num_outputs(self):return 1
            @property
            def batch_shape(self):return torch.Size()
            def posterior(self,X,**kwargs):
                post=v['model'].posterior(X,**kwargs)
                return GPyTorchPosterior(MultivariateNormal(post.mvn.mean,post.mvn.lazy_covariance_matrix.add_jitter(1e-6)))
        acq=qLogExpectedImprovement(JitterModel(),best_f=v['best_std'],sampler=SobolQMCNormalSampler(torch.Size([512]),seed=meta['mc_seed']))
        with torch.no_grad():direct=acq(cache.X[:19].unsqueeze(1)).numpy()
        score_error=float(np.max(np.abs(direct-base[:19])));assert score_error<1e-7,(task,score_error)
        rows.append({'task':task,'candidates':len(p['ids']),'initial_ids':p['initial_ids'],
            'cached_joint_mean_max_error':err,'marginal_direct_qLogEI_max_error_19_sampled_points':score_error,
            'GP_fit_seed':v['fit_seed'],'GP':'SingleTaskGP','all_continuous':True,'objective_scale':p['objective_scale']})
    receipt={'status':'PASS','tasks':rows,'scope':'GP/qLogEI and regional covariance adapter; not a performance claim'}
    data.write(OUT/'qualification.json',receipt);return receipt

def prepare(config='../config.txt'):
    if (OUT/'protocol.json').exists():return check()
    assert data.read(OUT/'qualification.json')['status']=='PASS'
    inputs={p.as_posix():data.filehash(p) for task in TASKS for p in (data.ROOT/task/'pool.json',data.ROOT/task/'oracle.npz')}
    protected_paths=[Path('research/p0_p1_report_20261029/REPORT.md'),Path('research/p0_p1_report_20261029/data.json'),
        Path('research/p0_p1_report_20261029/report_manifest.json'),Path('research/stable_r2_20261029/milestone_inventory.json'),
        Path('research/stable_r2_20261029/stable_strategy_results.json'),Path('research/suzuki_stable_20261029/results.json')]
    protected={p.as_posix():data.filehash(p) for p in protected_paths}
    p={'study':'Continuous_SnAr_HPLC_transport','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'tasks':list(TASKS),'methods':list(e.METHODS),'seed':0,'initial_queries':4,'additional_queries':16,'trajectories':16,
        'knowledge_conditions':'one common observed-prefix-only initial LLM region per task; no artificial oracle-based priors',
        'parameters':e.PARAMETERS,'LLM':api.load_config(config).public(),'source_sha256':sources(),'input_sha256':inputs,
        'protected_files':protected,'qualification_sha256':data.filehash(OUT/'qualification.json'),
        'resources':{'CPU_processes':8,'Torch_threads_per_process':1,'GPU':'no GPU needed for exact GP on <=20 observations','operator_compilation':'warm once before counted loop; no torch.compile or compiler in selection loop'},
        'adaptations':[
            'All inputs continuous: use SingleTaskGP, physical-range normalization, no dummy nominal variables.',
            'SnAr maximizes STY/13000 with noise=0, records E-factor without multiobjective scalarization; original Summit numerical class body unchanged.',
            'HPLC fixed measured unique-input pool uses first source row exact-match semantics; no IDW interpolation or outcome-based replicate choice.',
            'Fixed engineering normalization HPLC peak_area/10000; KF and event constants transferred without target-dependent tuning.',
            'Official LGBO regional tilt with cached covariance output-unit correction applied consistently to all regional methods.',
            'Official LLINBO _run_justify body runs unchanged; common4 start, qLogEI-loss threshold and finite-pool nearest unused projection replace UCB/max-var/continuous optimization hooks.',
            'Reasoning BO uses official optimization prompt and official stable confidence extraction; common GP top5 ->1 sequential adaptation, continuous physical I/O.',
            'New_Agent controlled exact single-axis tests at queries2/4/6 use same16 budget; original95-percent yield skip disabled because STY/peak-area are not percentage objectives.',
            'No molecular kernel recovery or multi-category joint tests in all-continuous domains.',
            'All methods share exact-score tie contrast proxy / latent SD / independent random priority, no numeric-ID ranking.'],
        'interpretation':'single fixed initialization per task; descriptive transfer experiment, no multi-seed significance; no tuning after comparing results'}
    p['protocol_sha256']=digest(p);data.write(OUT/'protocol.json',p);return p

def check():
    p=data.read(OUT/'protocol.json');q=deepcopy(p);h=q.pop('protocol_sha256');assert digest(q)==h
    assert p['source_sha256']==sources(),'source changed after freeze'
    for name,expected in p['input_sha256'].items():assert data.filehash(name)==expected,name
    return p

def base_state(task,method,p):
    k={'confidence':.7,'clauses':[]}
    return {'task':task,'method':method,'seed':0,'tie_seed':0,'run_id':task+'_'+method,'protocol_sha256':p['protocol_sha256'],
        'history':initial(task),'knowledge':k,'initial_knowledge':None,'banks':{},'region_trust':{},'calls':[],
        'validations':[],'validation_checks':[],'decisions':[],'status':'READY'}

def initialize(config='../config.txt'):
    p=check();client=api.APIClient(api.load_config(config));rows=[]
    try:
        for task in TASKS:
            folder=ROOT/'initialization'/task;state=base_state(task,'common_initialization',p)
            answer,key=llm.region_call(client,pool(task),state,1)
            ready={'history':state['history'],'knowledge':answer['plan']['knowledge'],'context_sha256':key,
                   'protocol_sha256':p['protocol_sha256']}
            data.write(folder/'ready.json',ready);rows.append({'task':task,'clauses':len(ready['knowledge']['clauses']),'context':key})
    finally:client.close()
    return rows

def save_step(state,path,cache,d,t,mode='normal',llm_record=None,validation=None):
    p=cache.pool;j=d['selected'];key=p['ids'][j];history=deepcopy(state['history']);v=d['stats']
    assert d['available'][j]
    exception=not bool(d['eligible'][j]);assert not exception or mode=='controlled_test' or state['method'] in ('LGBO_region','ReasoningBO')
    folder=path.parent/'rounds';folder.mkdir(parents=True,exist_ok=True);table=folder/f'{t:02d}_candidates.npz'
    np.savez_compressed(table,base_qLogEI=d['base'],guided_qLogEI=d['score'],available=d['available'],eligible=d['eligible'],
        mu_y=v['mu_y'],sd_y=v['sd_y'],sd_predictive_y=v['sd_predictive_y'],mu_std=v['mu_std'],sd_std=v['sd_std'],
        raw_shift_std=d['raw_shift_std'],deployed_shift_std=d['shift_std'],information_proxy=d['information_proxy'],tie_priority=d['tie_priority'])
    commit={'iteration':t,'selected_id':key,'normal_proposal_id':p['ids'][d['normal_proposal']],'reference_id':p['ids'][d['reference']],
        'history':history,'history_sha256':digest(history),'protocol_sha256':state['protocol_sha256'],'table_sha256':data.filehash(table),
        'knowledge':deepcopy(state['knowledge']),'diagnosis':d['diagnosis'].to_dict(),'retained':d['retained'],'region':d['region'],
        'amplitude':d['amplitude'],'fallback':d['fallback'],'trust':d['trust'],'region_trust_before':deepcopy(state['region_trust']),
        'bank_before':d['bank'],'bank_prior':d['prior'],'bank_key':d['bank_key'],'budget':d['budget'],'execution_mode':mode,
        'base_opportunity_loss':float(d['base'][d['reference']]-d['base'][j]),'normal_gate_exception':exception,
        'GP_scale_y':v['std'],'GP_mean_y':v['mean'],'best_std':v['best_std'],'base_meta':d['base_meta'],
        'field_meta':d['field_meta'],'guided_meta':d['guided_meta'],'LLM':llm_record,'validation':validation}
    committed=folder/f'{t:02d}_committed.json';data.write(committed,commit)
    # Outcome provider is consulted only after the complete decision is committed.
    observed=data.reveal(state['task'],key);y=observed['y'];mu=float(v['mu_y'][j]);sd=float(v['sd_predictive_y'][j]);raw=float(d['raw_shift_std'][j]*v['std'])
    bank=feedback=None
    if d['bank'] is not None:
        bank,measurement=e.update_bank(d['bank'],d['prior'],t,raw,mu,sd,y);state['banks'][d['bank_key']]=bank
        d['_measurement']=measurement;feedback=e.trust.feedback(state,d,t,y,bank)
    else:
        var=sd*sd+.0009;measurement={'prequery_mu_y':mu,'prequery_predictive_sd_y':sd,'raw_region_shift_y':raw,
            'observed_y':y,'residual':y-mu,'innovation':y-mu,'measurement_variance':var,'innovation_variance':var,'innovation_z':float((y-mu)/np.sqrt(var))}
    best=max(h['y'] for h in history)
    if validation:
        plan=validation['answer']['plan'];state['validations'].append({'query':t,'candidate_id':key,'observed_y':y,'incumbent_before_y':best,
            'anchor_candidate_id':validation['prepared']['anchor_candidate_id'],'changed_axis':plan['changed_axis'],
            'challenged_clause_ids':plan['challenged_clause_ids'],'prediction':plan['prediction']})
    event={**{k:commit[k] for k in ('iteration','selected_id','normal_proposal_id','reference_id','diagnosis','retained','region','amplitude','fallback','trust','bank_before','bank_prior','bank_key','budget','execution_mode','base_opportunity_loss','normal_gate_exception','knowledge')},
        'y':y,'raw_value':observed['raw_value'],'incumbent_before':best,'incumbent_after':max(best,y),'eligible_count':int(d['eligible'].sum()),
        'measurement':measurement,'bank_after':bank,'trust_feedback':feedback,'region_trust_after':deepcopy(state['region_trust']),
        'LLM':llm_record,'llm_insight':llm_record.get('raw') if state['method']=='ReasoningBO' and llm_record else None,
        'commit_sha256':data.filehash(committed)}
    if 'e_factor' in observed:event['e_factor']=observed['e_factor']
    state['history'].append(observed);state['decisions'].append(event);state['status']='COMPLETE' if t==16 else 'RUNNING'
    state['operator_compiles_in_loop']=sum(e.JIT.values());assert state['operator_compiles_in_loop']==0
    data.write(path,state);cache.release()
    print(json.dumps({'task':state['task'],'method':state['method'],'query':t,'mode':mode,'raw_value':observed['raw_value'],
        'best':max(best,y)*p['objective_scale'],'refreshes':len(state['calls'])}),flush=True)
    return y

def step(state,path,cache,client):
    t=len(state['decisions'])+1;p=cache.pool;d=e.setup(cache,state,t);record=None;validation=None;mode='normal'
    reasons=e.event_reasons(state,d,t)
    if reasons and not any(c['iteration']==t for c in state['calls']):
        answer,key=llm.region_call(client,p,state,t,d,reasons)
        state['calls'].append({'iteration':t,'reasons':reasons,'knowledge_before':deepcopy(state['knowledge']),'answer':answer,'context_sha256':key})
        state['knowledge']=answer['plan']['knowledge'];data.write(path,state);d=e.setup(cache,state,t)
    if state['method']=='New_Agent' and t in (2,4,6):
        prepared=e.validation_candidates(cache,state,d)
        state['validation_checks'].append({'iteration':t,'candidate_count':len(prepared['candidates']),'skip_reason':prepared['skip_reason']})
        if prepared['candidates']:
            answer,key=llm.test_call(client,p,state,t,d,prepared)
            validation={'prepared':prepared,'answer':answer,'context_sha256':key};d['selected']=cache.ix[answer['plan']['candidate_id']];mode='controlled_test'
    elif state['method']=='ReasoningBO':
        answer,key=llm.rbo_call(client,p,state,t,d);d['selected']=cache.ix[answer['plan']['selected_id']]
        record={'context_sha256':key,'raw':answer['raw'],'plan':answer['plan']};mode='ReasoningBO_top5'
    return save_step(state,path,cache,d,t,mode,record,validation)

class BudgetHook:
    def __truediv__(self,t):return float(np.log(4)*(16-t)/15)

def run_native_justify(state,path,cache,client):
    native=modules('LLINBO');p=cache.pool;dim=len(p['axes']);by_z={tuple(z):i for i,z in enumerate(p['z'])};live={}
    warm=[(tuple(p['z'][cache.ix[h['candidate_id']]]),h['y']) for h in state['history'][:4]]
    def convert(history):return [{'candidate_id':p['ids'][by_z[tuple(x)]],'y':float(y)} for x,y in history]
    def fitting(history):
        prefix=convert(history);t=len(prefix)-3
        view={**state,'history':prefix,'decisions':state['decisions'][:t-1]};live.update(d=e.setup(cache,view,t),t=t,view=view)
        return live['d']
    def maximize(model,bounds,beta):return torch.tensor([p['z'][model['reference']]],dtype=torch.double)
    def query(self,prompt,**unused):
        t=live['t'];view=live['view'];answer,key=llm.point_call(client,p,view,t,prompt)
        projection=llm.project(p,view,t,answer['plan']['point'])
        live['proposal']={'context_sha256':key,'raw':answer['raw'],'plan':answer['plan'],'projection':projection}
        return json.dumps(projection['z'])
    class Lookup:
        def __init__(self,model,**unused):self.d=model
        def __call__(self,x):return torch.tensor(float(self.d['base'][by_z[tuple(x.reshape(-1).tolist())]]),dtype=torch.double)
    def observe(vector):
        j=by_z[tuple(vector)];d=live['d'];t=live['t'];proposal=live['proposal'];prop=cache.ix[proposal['projection']['candidate_id']]
        rejected=bool(d['base'][d['reference']]>d['base'][prop]+d['budget']);expected=d['reference'] if rejected else prop
        assert j==expected;proposal.update(accepted=not rejected,proposed_id=p['ids'][prop]);d['selected']=j
        if t<=len(state['decisions']):
            assert state['decisions'][t-1]['selected_id']==p['ids'][j];cache.release();return state['history'][t+3]['y']
        return save_step(state,path,cache,d,t,'official_Justify',proposal)
    def start(self,num_warmstart,objective_function):assert num_warmstart==4;return list(warm)
    with ExitStack() as stack:
        hooks=[(native.LLAMAGENT_L,'llm_warmstarting',start),(native.LLAMAGENT_L,'query_llm',query),
            (native,'train_gp',fitting),(native,'find_max_variance_bound',lambda *a,**kw:BudgetHook()),
            (native,'optimize_acqf_ucb',maximize),(native,'UpperConfidenceBound',Lookup)]
        for obj,name,fn in hooks:stack.enter_context(patch.object(obj,name,fn))
        desc=p['description']+' Physical axes and normalization: '+json.dumps(p['bounds'])+'; x uses normalized coordinates in this exact axis order: '+json.dumps(p['axes'])
        # The original prompt randomly permutes history; freeze each generation's
        # RNG through a fixed trajectory seed, independent of worker scheduling.
        e.seed_all(40102026+(1 if p['task']=='SnAr' else 2)*10000)
        native.LLMIBO_BFO('justify',observe,dim=dim,desc=desc,T=16,T_ini=4,T_rep=1,verbose=False).run()

def run_one(task,method,config='../config.txt'):
    e.warm();p=check();ready=data.read(ROOT/'initialization'/task/'ready.json');path=ROOT/'runs'/(task+'_'+method)/'state.json'
    if path.exists():state=data.read(path)
    else:
        state=base_state(task,method,p);state['knowledge']=deepcopy(ready['knowledge']);state['initial_knowledge']=deepcopy(ready['knowledge']);data.write(path,state)
    if state['status']=='COMPLETE':return {'task':task,'method':method,'status':'COMPLETE','queries':16}
    assert state['protocol_sha256']==p['protocol_sha256'];state['status']='RUNNING';state.pop('error',None)
    client=api.APIClient(api.load_config(config)) if method in ('LLINBO_Justify','ReasoningBO','New_Agent') else None
    cache=e.Cache(pool(task));started=time.perf_counter()
    try:
        if method=='LLINBO_Justify':run_native_justify(state,path,cache,client)
        else:
            while len(state['decisions'])<16:step(state,path,cache,client)
    except Exception as error:
        state['status']='INTERRUPTED';state['error']=api.safe_error(error,client.config) if client else {'kind':type(error).__name__,'message':str(error)[:500]}
        data.write(path.parent/'error.json',{'error':state['error'],'traceback':api.safe_error(RuntimeError(traceback.format_exc()),client.config) if client else traceback.format_exc()})
    finally:
        if client:client.close()
        state['active_seconds']=state.get('active_seconds',0.)+time.perf_counter()-started;state['cache_counters']=cache.counters;data.write(path,state)
    return {'task':task,'method':method,'status':state['status'],'queries':len(state['decisions']),'error':state.get('error')}

def run(config='../config.txt',workers=8,methods=None):
    check();methods=methods or e.METHODS
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures=[ex.submit(run_one,task,m,config) for task in TASKS for m in methods]
        for f in as_completed(futures):print(json.dumps(f.result()),flush=True)
    return status()

def states():return [data.read(p) for p in sorted((ROOT/'runs').glob('*/state.json'))]
def status():
    ss=states();return {'planned':16,'complete':sum(s['status']=='COMPLETE' for s in ss),'queries':sum(len(s['decisions']) for s in ss),
        'interrupted':[{'task':s['task'],'method':s['method'],'queries':len(s['decisions']),'error':s.get('error')} for s in ss if s['status']=='INTERRUPTED']}

def preflight(config='../config.txt'):
    client=api.APIClient(api.load_config(config))
    try:
        result={'configured_model':api.MODEL,'model_available':any(r['id']==api.MODEL for r in client.models()['data'])}
        for endpoint in ('/user/balance','/balance'):
            try:
                r=client.http.get(endpoint)
                if r.status_code==200:
                    obj=r.json();result['balance']={k:v for k,v in obj.items() if k in ('balance_infos','is_available','balance','total_balance')};break
            except Exception:pass
        data.write(OUT/'api_preflight.json',result);return result
    finally:client.close()
