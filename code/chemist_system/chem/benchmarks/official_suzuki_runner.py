"""Frozen official-source experiment and a persistent, bounded CPU worker pool."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
from hashlib import sha256
from importlib import metadata
import importlib
import inspect
import io
import contextlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from threadpoolctl import threadpool_limits,threadpool_info

from chem.decision.replay import write_json_atomic
from . import official_suzuki_adapter as adapter
from .suzuki_domain import FrozenSuzukiOracle,initial_design,digest,same_point
from .suzuki_policies import hebo_select

ROOT=Path('data/runs/suzuki_official')
RESEARCH=Path('research/suzuki_official')
ORACLES=Path('data/runs/suzuki_baselines/oracles')
METHODS=('LLINBO','LGBO','HEBO')
JIT_CALLS={'script':0,'trace':0}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def hashes():
    paths=[Path(adapter.__file__),Path(__file__),Path(importlib.import_module('chem.benchmarks.suzuki_domain').__file__)]
    return {p.name:sha256(p.read_bytes()).hexdigest() for p in paths} | {
        'HEBO_transport':sha256(inspect.getsource(hebo_select).encode('utf8')).hexdigest()}


def installed_hebo():
    d=metadata.distribution('HEBO')
    return {str(p):sha256(Path(d.locate_file(p)).read_bytes()).hexdigest() for p in d.files
            if str(p).startswith('hebo/') and str(p).endswith('.py')}


def prepare():
    if (ROOT/'protocol.json').exists(): return check_protocol()
    qualification=read(RESEARCH/'qualification.json')
    if qualification['status']!='PASS': raise ValueError('source/interface qualification is required')
    protocol={'schema_version':1,'study':'Suzuki_official_sources_v1',
        'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'cases':[1,2,3,4],'seeds':[0,1,2,3,4],
        'methods':list(METHODS),'initial_queries':4,'additional_queries':16,'planned_runs':60,
        'initialization':'same physical 4-point Latin hypercube and 4 distinct catalyst labels per case/seed; no method-specific LLM warm-start',
        'objective':'maximize pretrained Summit0.8.9 ensemble mean yld/100; no added observation noise; not laboratory reactions',
        'continuous_interface':{'dimension':4,'order':['t_res normalized','temperature normalized','catalyst_loading normalized','catalyst_code'],
            'physical_numeric_bounds':[[60,600],[30,110],[.5,2.5]],'catalysts':list(adapter.CATALYSTS),
            'mapping':'label k occupies [k/8,(k+1)/8); z=1 maps to label7; initialization and explicit LLM catalyst points use bin centers',
            'GP_history':'native 4D proposals are stored without replacing their category code by a bin center; the oracle alone decodes categories',
            'limitation':'continuous-only upstream algorithms see a piecewise objective and an arbitrary label order; this does not imply chemical distances or permutation invariance'},
        'LLINBO':{'entrypoint':'LLM_agent_BBFO.LLMIBO_BFO(method=justify).run -> unchanged _run_justify',
            'GP':'unchanged helper_func.train_gp -> BoTorch SingleTaskGP + fit_gpytorch_mll, default constructor',
            'acquisition':'unchanged helper_func.optimize_acqf_ucb; num_restarts20 raw_samples200',
            'threshold':'unchanged source initial 10^4-grid maximum posterior variance / (t+1); unchanged source beta expression',
            'provider':'only LLAMAGENT_L.query_llm transport replaced by current assistant; source candidate prompt and JSON parsing retained',
            'pause_resume':'RNG-preserving memo of actual upstream train_gp/maxvariance/optimize_acqf_ucb; direct upstream vs resumed history and final RNG qualified identical',
            'forbidden_changes':['no sklearn substitute','no one-hot surrogate','no finite candidate replacement of UCB optimization','no rewritten acceptance rule']},
        'LGBO':{'entrypoint':'unchanged lgbo_core.propose_lgbo_batch; original prompt.py/parse_assistant_response',
            'policy':'tilt','GP':'unchanged prior_monte_carlo.BaseBOSampler.fit_model -> SingleTaskGP + fit_gpytorch_mll',
            'confidence':'unchanged decide.py confidence_to_delta with guidance_scale3',
            'acquisition':'unchanged boo.py qLogExpectedImprovement, SobolQMC512 for q=1',
            'candidate_grid':8192,'regional_grid':512,'all_defaults_retained':True,
            'forbidden_changes':['no analytic EI replacement','no paper-equation replacement of public confidence mapping','no rewritten covariance/mean tilt']},
        'HEBO':{'entrypoint':'unmodified HEBO0.3.6 suggest/observe','model':'gp','default_parameter_dtype':'torch.float32',
            'rand_sample':4,'gp_epochs':100,'population':100,'generations':100,'acquisition':'MACE',
            'numerical_fallback':'retain and count any upstream abandoned updates/random predictions; no precision repair in this primary rerun'},
        'randomness':{'LLINBO':'seed20261027+case*10000+seed*100 once at native loop start; resume restores source post-call RNG exactly',
            'LGBO_HEBO':'seed20261027+case*10000+seed*100+additional_iteration; LGBO uses its native seed argument'},
        'duplicates':'retain upstream repeated physical conditions and charge each query; no manual deduplication or replacement',
        'LLM':{'provider':'current assistant selected by user','planned_logical_responses':640,'refresh':'each additional query',
            'independent_API_calls':False,'immutable_backend_snapshot':None,'isolated_contexts':False,'tokens_available':False,
            'disclosure':'same chat already contains previous Suzuki outcomes; fresh exact-context answers do not constitute a blind independent LLM evaluation'},
        'evaluation':{'primary':'mean best-so-far normalized yield over 16 added-budget positions',
            'secondary':'final best yield, absolute80/90% attainment, actual repeats, failures, native numerical fallbacks',
            'failures':'retain all 60 planned trajectories; carry last incumbent across unfilled budget positions',
            'regret':'no global regret claim; continuous-domain optimum unknown'},
        'resources':{'persistent_workers':4,'max_hebo_concurrency':2,'threads_per_worker':1,'device':'cpu',
            'compilation':'prebuilt binaries and import/matrix warm-up before loops; count any later torch.jit.script/trace calls'},
        'engine_sha256':hashes(),'upstream_sources':adapter.check_sources(),'hebo_source_hashes':installed_hebo(),
        'oracles_sha256':{str(p):sha256(p.read_bytes()).hexdigest() for p in ORACLES.rglob('*') if p.is_file()},
        'qualification_sha256':sha256((RESEARCH/'qualification.json').read_bytes()).hexdigest(),
        'environment':{'python':__import__('platform').python_version(),'packages':{d.metadata['Name']:d.version for d in metadata.distributions()}},
        'historical_study':'previous adapted results are not merged with this study'}
    manifest=[{'run_id':f'c{c}_s{s}_{m}','case':c,'seed':s,'method':m,'initial_points':initial_design(c,s)}
              for c in protocol['cases'] for s in protocol['seeds'] for m in METHODS]
    protocol['manifest_sha256']=digest(manifest)
    protocol['protocol_sha256']=digest(protocol)
    write_json_atomic(ROOT/'manifest.json',manifest)
    write_json_atomic(ROOT/'protocol.json',protocol)
    write_json_atomic(RESEARCH/'manifest.json',manifest)
    write_json_atomic(RESEARCH/'protocol.json',protocol)
    return protocol


def check_protocol():
    p=read(ROOT/'protocol.json');a=deepcopy(p);h=a.pop('protocol_sha256')
    if h!=digest(a) or p['engine_sha256']!=hashes() or digest(read(ROOT/'manifest.json'))!=p['manifest_sha256']:
        raise ValueError('frozen official-source protocol/adapter/manifest changed')
    adapter.check_sources()
    if p['hebo_source_hashes']!=installed_hebo(): raise ValueError('official HEBO sources changed')
    return p


def request(spec,state,t,system,prompt,protocol):
    previous=state['responses'][t-2] if t>1 and len(state['responses'])>=t-1 else None
    value={'run_id':spec['run_id'],'method':spec['method'],'case':spec['case'],'seed':spec['seed'],'iteration':t,
        'history':state['history'][:t+3],'previous_response':previous,'system_prompt':system,'user_prompt':prompt,
        'protocol_sha256':protocol['protocol_sha256']}
    if t<=len(state['responses']):
        # Replayed prefixes retain their original exact context/protocol hash
        # across a documented transport-only amendment. Never re-query an LLM
        # or rewrite an answer already consumed by an upstream algorithm.
        answer=state['responses'][t-1]
        original=read(ROOT/'requests'/answer['context_sha256']/'request.json')
        for key,item in value.items():
            if key!='protocol_sha256' and original[key]!=item:
                raise ValueError('recorded assistant prefix changed after transport amendment')
        validate_answer(answer,original)
        return answer
    value['context_sha256']=digest(value)
    folder=ROOT/'requests'/value['context_sha256']
    write_json_atomic(folder/'request.json',value)
    if not (folder/'response.json').exists():
        state.update(status='WAITING_LLM',pending_context=value['context_sha256'])
        raise adapter.AwaitingAssistant('exact context response is pending')
    answer=read(folder/'response.json')
    validate_answer(answer,value)
    if len(state['responses'])<t: state['responses'].append(answer)
    elif state['responses'][t-1]!=answer: raise ValueError('recorded assistant answer changed')
    return answer


def validate_answer(answer,req):
    for k in ('run_id','iteration','context_sha256'):
        if answer.get(k)!=req[k]: raise ValueError('assistant context mismatch')
    if not isinstance(answer.get('reasoning'),str) or len(answer['reasoning'])<8: raise ValueError('rationale missing')
    q=answer.get('supporting_queries',[])
    if not q or any(type(i)is not int or not 1<=i<=len(req['history']) for i in q): raise ValueError('non-prefix citation')
    if req['method']=='LLINBO':
        z=json.loads(answer['upstream_reply']);adapter.validate_z(z)
    else:
        _,prompt=adapter.modules('LGBO')
        parsed=prompt.parse_assistant_response(answer['upstream_reply'])
        if parsed['mode'] not in ('point','region'): raise ValueError('invalid official LGBO response')
        vectors=[parsed['point']] if parsed['mode']=='point' else [parsed['lb'],parsed['ub']]
        for v in vectors:
            if len(v)!=4: raise ValueError('invalid LGBO dimension')
            for x,p in zip(v,adapter.PARAMETERS):
                if not np.isfinite(x) or not p['bounds'][0]<=x<=p['bounds'][1]: raise ValueError('out-of-domain preference')
        if parsed['mode']=='region' and any(a>b for a,b in zip(*vectors)): raise ValueError('reversed region bounds')
        if not 0<=parsed['confidence']<=1: raise ValueError('invalid confidence')


def _worker_init():
    threadpool_limits(1)
    torch.set_num_threads(1)
    adapter.modules('LLINBO');adapter.modules('LGBO')
    from hebo.optimizers.hebo import HEBO
    _=torch.linalg.cholesky(torch.eye(4,dtype=torch.double))
    # Record any compiler entry after this warm-up. No eager GP fitting is
    # counted as compilation, and no source algorithm is switched to JIT.
    for name in ('script','trace'):
        original=getattr(torch.jit,name)
        def counted(*args,_name=name,_original=original,**kwargs):
            JIT_CALLS[_name]+=1
            return _original(*args,**kwargs)
        setattr(torch.jit,name,counted)


def _trace_llinbo(state,trace):
    decisions=[]
    for i,g in enumerate(trace['gates']):
        if i>=len(state['history'])-4: break
        psi=trace['psi0']/(i+1)
        accepted=not(g['values'][0]>g['values'][1]+psi)
        actual=state['history'][i+4]['z']
        proposed=json.loads(state['responses'][i]['upstream_reply'])
        expected=proposed if accepted else trace['acquisition_points'][i]
        if not np.array_equal(np.array(actual),np.array(expected)):
            raise ValueError('logged native branch and objective input disagree')
        decisions.append({'iteration':i+1,'beta':g['beta'],'psi0':trace['psi0'],'psi':psi,
            'gp_ucb':g['values'][0],'llm_ucb':g['values'][1],'accepted_llm':accepted,'z':actual,
            'native_gp_proposal':trace['acquisition_points'][i],'upstream_method':'LLMIBO_BFO._run_justify'})
    state['decisions']=decisions
    state['native_fit_classes']=trace['fit_classes']
    state['memo_last_invocation']=trace['memo_events']


def run_one(spec,protocol):
    directory=ROOT/'runs'/spec['run_id'];file=directory/'state.json'
    directory.mkdir(parents=True,exist_ok=True)
    state=read(file) if file.exists() else {**spec,'status':'CREATED','history':[], 'responses':[], 'decisions':[],
        'protocol_sha256':protocol['protocol_sha256'],'optimization_seconds':0.,'oracle_queries':0,'replayed_queries':0}
    if state['status'] in ('COMPLETED','FAILED'): return spec['run_id']
    oracle=FrozenSuzukiOracle(ORACLES,spec['case'])
    if not state['history']:
        ys=oracle.predict(spec['initial_points'])
        state['history']=[{'query':i+1,'point':p,'z':adapter.encode_point(p),'y':float(y),'initial':True}
                          for i,(p,y) in enumerate(zip(spec['initial_points'],ys))]
        state['oracle_queries']=4
        write_json_atomic(file,state)
    before_compile=sum(JIT_CALLS.values())
    started=time.perf_counter();text=io.StringIO()
    try:
        with contextlib.redirect_stdout(text),contextlib.redirect_stderr(text):
            if spec['method']=='LLINBO':
                consumed=0
                def objective(z):
                    nonlocal consumed
                    j=4+consumed;consumed+=1
                    z=[float(v) for v in z]
                    if j<len(state['history']):
                        if z!=state['history'][j]['z']: raise ValueError('native source replay changed a prior query')
                        state['replayed_queries']+=1
                        return state['history'][j]['y']
                    y=float(oracle.predict([adapter.decode_z(z)])[0])
                    state['history'].append({'query':j+1,'z':z,'point':adapter.decode_z(z),'y':y,'initial':False})
                    state['oracle_queries']+=1
                    write_json_atomic(file,state)
                    return y
                def assistant(history,prompt):
                    t=len(history)-3
                    expected=[(tuple(r['z']),r['y']) for r in state['history'][:t+3]]
                    if expected!=history: raise ValueError('upstream LLM received a different history prefix')
                    a=request(spec,state,t,'You are an AI assistant that helps people find an maximum of a black box function.',prompt,protocol)
                    return a['upstream_reply']
                try:
                    trace=adapter.execute_llinbo(case=spec['case'],initial=state['history'][:4],objective=objective,
                        assistant=assistant,seed=20261027+spec['case']*10000+spec['seed']*100,
                        memo_directory=directory/'memo')
                    state['status']='COMPLETED'
                except adapter.AwaitingAssistant as exc:
                    trace=exc.trace
                _trace_llinbo(state,trace)
            else:
                while len(state['history'])<20:
                    t=len(state['history'])-3
                    seed=20261027+spec['case']*10000+spec['seed']*100+t
                    if spec['method']=='LGBO':
                        system,prompt=adapter.lgbo_prompts(spec['case'],state['history'])
                        a=request(spec,state,t,system,prompt,protocol)
                        z,detail=adapter.execute_lgbo(state['history'],a['upstream_reply'],seed)
                        point=adapter.decode_z(z)
                    else:
                        point,detail=hebo_select([h['point'] for h in state['history']], [h['y'] for h in state['history']],seed)
                        z=adapter.encode_point(point)
                    y=float(oracle.predict([point])[0])
                    state['history'].append({'query':len(state['history'])+1,'point':point,'z':z,'y':y,'initial':False})
                    state['oracle_queries']+=1
                    state['decisions'].append({'iteration':t,'z':z,**detail})
                    state['status']='RUNNING';state.pop('pending_context',None)
                    write_json_atomic(file,state)
                state['status']='COMPLETED'
    except adapter.AwaitingAssistant:
        pass
    except Exception as exc:
        state.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc())
    state['optimization_seconds']+=time.perf_counter()-started
    log=text.getvalue()
    state.setdefault('upstream_numerics',{'abandoned_fit_epochs':0,'random_prediction_fallbacks':0})
    state['upstream_numerics']['abandoned_fit_epochs']+=log.count('give up fitting GP')
    state['upstream_numerics']['random_prediction_fallbacks']+=log.count('output random predictions')
    if log:
        (directory/f"invocation_q{len(state['history']):02d}.log").write_text(log,encoding='utf8')
    state['operator_compiles_in_loop']=state.get('operator_compiles_in_loop',0)+sum(JIT_CALLS.values())-before_compile
    state['resources']={'device':'cpu','torch_threads':torch.get_num_threads(),'threadpools':threadpool_info()}
    if state['status']=='COMPLETED': state.pop('pending_context',None)
    write_json_atomic(file,state)
    return spec['run_id']


def status():
    rows=[]
    for spec in read(ROOT/'manifest.json'):
        f=ROOT/'runs'/spec['run_id']/'state.json'
        s=read(f) if f.exists() else {'status':'CREATED','history':[]}
        rows.append({'run_id':spec['run_id'],'status':s['status'],'queries':len(s['history']),'error':s.get('error')})
    return {'counts':{name:sum(r['status']==name for r in rows) for name in ('CREATED','RUNNING','WAITING_LLM','COMPLETED','FAILED')},'runs':rows}


def serve(workers=4):
    if not 1<=workers<=4: raise ValueError('worker limit1..4')
    protocol=check_protocol();manifest=read(ROOT/'manifest.json')
    inflight={};previous=None
    with ProcessPoolExecutor(max_workers=workers,initializer=_worker_init) as pool:
        while True:
            active={s['run_id'] for s in inflight.values()}
            candidates=[];finished=0
            for spec in manifest:
                f=ROOT/'runs'/spec['run_id']/'state.json'
                s=read(f) if f.exists() else {'status':'CREATED'}
                if s['status'] in ('COMPLETED','FAILED'): finished+=1;continue
                if spec['run_id'] in active: continue
                if s['status']=='WAITING_LLM' and not (ROOT/'requests'/s['pending_context']/'response.json').exists(): continue
                candidates.append(spec)
            candidates.sort(key=lambda s:(s['method']=='HEBO',s['case'],s['seed'],s['method']))
            for spec in candidates:
                if len(inflight)>=workers: break
                if spec['method']=='HEBO' and sum(s['method']=='HEBO' for s in inflight.values())>=min(2,workers): continue
                future=pool.submit(run_one,spec,protocol);inflight[future]=spec
            if inflight:
                ready,_=wait(inflight,timeout=.4,return_when=FIRST_COMPLETED)
                for f in ready: f.result();inflight.pop(f)
            else: time.sleep(.4)
            snapshot=status()
            if snapshot!=previous:
                write_json_atomic(ROOT/'status.json',snapshot);previous=snapshot
            if finished==len(manifest) and not inflight: return snapshot


def pending(compact=False):
    output=[]
    for spec in read(ROOT/'manifest.json'):
        file=ROOT/'runs'/spec['run_id']/'state.json'
        if not file.exists():continue
        state=read(file)
        if state['status']!='WAITING_LLM':continue
        req=read(ROOT/'requests'/state['pending_context']/'request.json')
        if compact:
            req={k:req[k] for k in ('run_id','method','iteration','context_sha256')} | {'history':req['history']}
        output.append(req)
    return output


def submit(path):
    answers=read(path)['responses'];receipt=[]
    for a in answers:
        req=read(ROOT/'requests'/a['context_sha256']/'request.json')
        validate_answer(a,req)
        file=ROOT/'requests'/a['context_sha256']/'response.json'
        if file.exists() and read(file)!=a: raise ValueError('an official-run assistant response already exists')
        write_json_atomic(file,a)
        receipt.append({'context_sha256':a['context_sha256'],'answer_sha256':digest(a)})
    write_json_atomic(Path(path).with_suffix('.receipt.json'),{'source_sha256':sha256(Path(path).read_bytes()).hexdigest(),'answers':receipt})
    return {'submitted':len(answers)}
