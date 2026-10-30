"""Mixed-GP replacement study, same finite pool and frozen knowledge policy."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import time

import numpy as np
import torch

from . import mixed_kalman_suzuki_engine as e
from . import kalman_knowledge_suzuki_study as old
from . import kalman_knowledge_suzuki_engine as policy
from .suzuki_domain import digest

ROOT=Path('data/runs/suzuki_mixed_kalman');OUT=Path('research/suzuki_mixed_kalman');WORKERS=2
PARENT=old.PARENT;read=old.read;write=old.write;filehash=old.filehash;api=old.api
PROTECTED=dict(old.PROTECTED,kalman_knowledge=old.OUT/'release_inventory.json')
PARAMETERS=deepcopy(old.PARAMETERS);SYSTEM_PROMPT=old.SYSTEM_PROMPT
METHODS=('Mixed_KF_event_B4','Mixed_KF_frozen_initial_B4','Mixed_qLogEI_B0');JIT={}


def inputs(case):return old.inputs(case)


def make_cache(case,pool,qualification=False,category_map=e.CATEGORY_MAP):
    group=f'c{case}_s0';own=ROOT/('qualification_cache' if qualification else 'cache')/group
    if tuple(category_map)!=e.CATEGORY_MAP:own=own/'permuted'
    sources=[] if qualification else [ROOT/'qualification_cache'/group]
    return e.MixedKnowledgeCache(pool,own,case,0,category_map,sources)


def hashes():
    paths=[Path(__file__),Path(e.__file__),Path(policy.__file__),Path(old.__file__),Path(old.previous.__file__),Path(api.__file__),
           Path('chem/benchmarks/uncertain_suzuki_engine.py'),Path('chem/decision/domain_guard.py'),
           Path('chem/benchmarks/suzuki_mixed_kalman.py'),Path('tests/test_suzuki_mixed_kalman.py')]
    return {p.as_posix():filehash(p) for p in paths}


def protected_extras():
    paths=list(Path('outputs/case2_kf_event_b4_seed0').glob('*.csv'))
    paths += [Path('outputs/case2_kf_event_b4_seed0/README.txt'),old.OUT/'release_receipt.json']
    for folder in ('suzuki_kalman_gate_diagnostic','suzuki_case2_gp_fit_trace'):
        paths += list((Path('research')/folder).glob('*.json'))+list((Path('research')/folder).glob('*.py'))
    return {p.as_posix():filehash(p) for p in paths if p.is_file()}


def worker_init():
    torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    # Import and warm the Mixed kernel / linear algebra before counted loops.
    x=torch.tensor([[.1,.2,.3,0.],[.2,.3,.4,1.],[.6,.7,.8,2.],[.9,.8,.7,3.]],dtype=torch.double)
    model=e.MixedSingleTaskGP(x,torch.tensor([[0.],[1.],[-1.],[.5]],dtype=torch.double),cat_dims=[3])
    with torch.no_grad():torch.linalg.cholesky(model.covar_module(x).to_dense()+.01*torch.eye(4,dtype=torch.double))
    for name in ('script','trace'):
        original=getattr(torch.jit,name)
        def counted(*args,_original=original,_name=name,**kwargs):
            JIT[_name]+=1;return _original(*args,**kwargs)
        JIT[name]=0;setattr(torch.jit,name,counted)


def qualification():
    worker_init();results=[];permutation=(19,-3,101,7,0,12,-9,4)
    for case in (1,2,3,4):
        pool=inputs(case);history=read(old.ROOT/'runs'/f'c{case}_s0'/'state.json')['history'][:4]
        first=make_cache(case,pool,True);second=make_cache(case,pool,True,permutation)
        base,_=first.score(history);permuted,_=second.score(history)
        p0=first.posterior_stats(history);p1=second.posterior_stats(history)
        plan=policy.validate_plan({'hypotheses':[{'id':'H1','operation':'add','variable':'catalyst','bounds':[],'categories':[0,2],
            'effect_pp':5.,'reason':'Synthetic interface qualification only','support':[1]}], 'revocations':[],'summary':'No adaptive experiment or future labels used'},[],4,pool['z'])
        _,hs=policy.apply_plan(policy.initial_filter(),[],plan,pool['z']);weights={'H1':.4975062189439555}
        g0,_=first.knowledge_score(history,hs,weights,['H1']);g1,_=second.knowledge_score(history,hs,weights,['H1'])
        errors={'mean_y':float(np.max(np.abs(p0['mu_y']-p1['mu_y']))),'sd_y':float(np.max(np.abs(p0['sd_y']-p1['sd_y']))),
                'base_logei':float(np.max(np.abs(base-permuted))),'guided_logei':float(np.max(np.abs(g0-g1)))}
        assert max(errors.values())<1e-9
        _,value=first.model(history);zero=e.MixedResidualMeanModel(value['model'],[],{},[],value['std'])
        x=first.z[:32].unsqueeze(1)
        with torch.no_grad():
            raw=value['model'].posterior(x);wrapped=zero.posterior(x)
            assert torch.equal(raw.mean,wrapped.mean) and torch.allclose(wrapped.variance,raw.variance+1e-6,rtol=0,atol=1e-12)
        results.append({'case':case,'permuted_labels':permutation,'maximum_differences':errors,
                        'base_argmax_equal':int(np.argmax(base))==int(np.argmax(permuted)),
                        'guided_argmax_equal':int(np.argmax(g0))==int(np.argmax(g1)),
                        'zero_shift_mean_equal':True,'common_jitter':1e-6,'cache_counters':first.counters})
    assert sum(JIT.values())==0
    result={'status':'PASS','cases':results,'operator_compiles_in_loop':0,
            'boundary':'Actual four-case initial fits, posterior and full4692 base/guided scores invariant to arbitrary bijective GP category relabeling.'}
    write(OUT/'qualification.json',result);return result


def prepare(config_path='../config.txt'):
    if (OUT/'protocol.json').exists():return check_protocol()
    q=read(OUT/'qualification.json');assert q['status']=='PASS';cfg=api.load_config(config_path)
    manifest=[{'case':c,'seed':0,'run_id':f'c{c}_s0','task':f'M{c:02d}'} for c in (1,2,3,4)]
    import botorch,gpytorch,scipy,inspect
    from botorch.models.kernels.categorical import CategoricalKernel
    p={'study':'MixedGP_KF_event_B4_Suzuki_seed0_v1','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
       'scientific_sha256':hashes(),'manifest_sha256':digest(manifest),'qualification_sha256':filehash(OUT/'qualification.json'),
       'protected_inventories':{k:filehash(v) for k,v in PROTECTED.items()},'additional_protected_files':protected_extras(),
       'input_sha256':{f'case{c}/{name}':filehash(PARENT/'inputs'/f'case{c}'/name) for c in (1,2,3,4) for name in ('pool.json','oracle.npz')},
       'cases':[1,2,3,4],'primary_cases':[2,3,4],'seeds':[0],'initial_queries':4,'additional_queries':16,'candidate_count':4692,
       'methods':METHODS,'planned_trajectories':12,'planned_logical_queries':240,'maximum_logical_LLM_calls':16,
       'model':{'class':'botorch.models.gp_regression_mixed.MixedSingleTaskGP','cat_dims':[3],
           'continuous_dims':[0,1,2],'encoding':'Only continuous coordinates normalized/clipped. Catalyst label taken from physical point, nominal integer0..7 passed to equality/Hamming kernel; never an ordinal RBF coordinate.',
           'kernel':'Scale(RBF_cont1+Scale(Categorical1))+Scale(RBF_cont2*Categorical2)',
           'implementation_sha256':{inspect.getfile(c):filehash(inspect.getfile(c)) for c in (e.MixedSingleTaskGP,CategoricalKernel)},
           'versions':{'torch':torch.__version__,'botorch':botorch.__version__,'gpytorch':gpytorch.__version__,'scipy':scipy.__version__}},
       'parameters':policy.PARAMETERS,'api':cfg.public(),'request_parameters':PARAMETERS,
       'system_prompt_sha256':sha256(SYSTEM_PROMPT.encode('utf8')).hexdigest(),
       'change':'Replace ordinal4D-RBF SingleTaskGP with stock MixedSingleTaskGP. Keep pool,observations,explicit Y standardization/default outcome transform,fit/MC seeds,512 Sobol paths,jitter,knowledge schema,Kalman/event/B4 logic and budget schedule unchanged.',
       'policy_reuse':'Original immutable kalman_knowledge_suzuki_engine functions; old pool z category bins remain a parser/set-membership representation only. GP inputs derive nominal labels from physical catalyst strings, not legacy bin distances.',
       'controls':'New MixedGP pure-qLogEI B0, same seeds/budget, no API; frozen-first-knowledge+KF+B4 after all adaptive states frozen. Old ordinal GP and older strategies are read-only contextual references.',
       'LLM':'Fresh deepseek-flash requests to own observed prefix/current MixedGP summaries, same old system prompt; no prior-study answers, no unqueried yields, max4 logical calls/case incl initial; schema/HTTP retries unchanged.',
       'resources':{'persistent_workers':2,'threads_each':1,'device':'cpu','warmup':'Mixed kernel and Cholesky before JIT counted loops','cache':'Exact prefix once; no incompatible old ordinal fit/score reuse'},
       'limits':['One seed per case. Primary n=3, no significance or chemical-truth claims.',
                 'Stock MixedGP changes kernel structure and parameterization; posterior category relations are equality-based, without molecular structure descriptors.',
                 'Fresh event/hypothesis trajectories differ after GP change; old/new framework differences are system effects, not a knowledge-matched causal GP ablation.',
                 '512MC qLogEI and decaying log4 budget retained for isolation; previously observed late-tail MC sensitivity and singleton gates are not corrected in this study.'],
       'references':['https://botorch.readthedocs.io/en/v0.14.0/_modules/botorch/models/gp_regression_mixed.html','https://api-docs.deepseek.com/guides/thinking_mode/']}
    p['protocol_sha256']=digest(p);write(OUT/'protocol.json',p);write(ROOT/'protocol.json',p);write(ROOT/'manifest.json',manifest)
    write(OUT/'system_prompt.json',{'text':SYSTEM_PROMPT,'sha256':p['system_prompt_sha256']});return p


def check_protocol():
    p=read(OUT/'protocol.json');raw=deepcopy(p);key=raw.pop('protocol_sha256');assert digest(raw)==key and p['scientific_sha256']==hashes()
    assert p['manifest_sha256']==digest(read(ROOT/'manifest.json')) and p['protected_inventories']=={k:filehash(v) for k,v in PROTECTED.items()}
    assert p['additional_protected_files']==protected_extras()
    return p


def run_one(spec,config_path,mode='adaptive'):
    p=check_protocol();case=spec['case'];pool=inputs(case);ids=pool['ids'];z=np.asarray(pool['z']);cache=make_cache(case,pool)
    folder={'adaptive':'runs','frozen':'frozen','base':'base'}[mode];method=METHODS[('adaptive','frozen','base').index(mode)]
    path=ROOT/folder/spec['run_id']/'state.json'
    if path.exists():state=read(path)
    else:
        initial=deepcopy(read(old.ROOT/'runs'/spec['run_id']/'state.json')['history'][:4])
        state={**spec,'method':method,'history':initial,'decisions':[],'calls':[],'filter':policy.initial_filter(),'hypotheses':[],
               'knowledge_enabled':False,'status':'RUNNING','protocol_sha256':p['protocol_sha256'],'active_seconds':0.,'operator_compiles_in_loop':0}
        if mode=='frozen':
            adaptive=read(ROOT/'runs'/spec['run_id']/'state.json');call=adaptive['calls'][0]
            if call['status']=='ACCEPTED':
                state['filter'],state['hypotheses']=policy.apply_plan(state['filter'],[],call['plan'],z);state['knowledge_enabled']=True
                state['frozen_initial_plan']=call['plan'];state['frozen_source_context']=call['context_sha256']
        write(path,state)
    if state['status'] in ('COMPLETE','FAILED'):return {'run':spec['run_id'],'method':method,'status':state['status'],'steps':len(state['decisions'])}
    values=np.load(PARENT/'inputs'/f'case{case}'/'oracle.npz')['values'];client=api.APIClient(api.load_config(config_path)) if mode=='adaptive' else None
    if client:assert client.config.public()==p['api']
    try:
        while len(state['decisions'])<16:
            start=time.perf_counter();before_jit=sum(JIT.values());t=len(state['decisions'])+1
            filter_before_call=deepcopy(state['filter']);hypotheses_before_call=deepcopy(state['hypotheses']);enabled_before_call=state['knowledge_enabled']
            pending=next((c for c in state['calls'] if c['iteration']==t),None)
            if pending:
                filter_before_call=pending['filter_before'];hypotheses_before_call=pending['hypotheses_before'];enabled_before_call=pending['enabled_before']
            if mode=='adaptive' and pending is None:
                base,_=cache.score(state['history']);stats=cache.posterior_stats(state['history']);prior=policy.forecast(state['filter'],state['hypotheses'])
                queried=np.array([key in {h['candidate_id'] for h in state['history']} for key in ids])
                diag=policy.diagnose(base,z,state['hypotheses'],prior['weights'] if state['knowledge_enabled'] else {},policy.common.budget_at(4,t),queried,ids)
                reasons=policy.event_reasons(t,state['history'],state['decisions'],state['hypotheses'],diag,[c['iteration'] for c in state['calls']])
                if reasons:
                    public=old.public_request(state,pool,stats,base,diag,reasons);context={'public':public,'history':deepcopy(state['history']),'protocol_sha256':p['protocol_sha256']}
                    context['context_sha256']=digest(context);directory=ROOT/'requests'/context['context_sha256'];write(directory/'context.json',context)
                    call={'iteration':t,'reasons':reasons,'context_sha256':context['context_sha256'],'filter_before':filter_before_call,
                          'hypotheses_before':hypotheses_before_call,'enabled_before':enabled_before_call}
                    try:
                        valid=old.get_plan(client,context,state['hypotheses'],z,directory)
                        state['filter'],state['hypotheses']=policy.apply_plan(state['filter'],state['hypotheses'],valid['plan'],z);state['knowledge_enabled']=True
                        call.update(status='ACCEPTED',**valid)
                    except RuntimeError as error:
                        state['knowledge_enabled']=False;call.update(status='FAILED',error=str(error))
                    state['calls'].append(call);write(path,state)
            prior=policy.forecast(state['filter'],state['hypotheses']);filter_before=deepcopy(state['filter'])
            selected,decision=policy.choose(cache,state['history'],z,ids,state['hypotheses'],prior,t,state['knowledge_enabled'])
            y=float(values[selected]);feature=policy.features(z[[selected]],state['hypotheses'])[0]
            state['filter'],measurement=policy.update_filter(state['filter'],prior,feature,decision['selected_gp_mu_y'],decision['selected_predictive_sd_y'],y)
            deployed=decision['selected_gp_mu_y']+decision['mean_change_y'];measurement['predictions']['GP_deployed']=deployed;measurement['squared_errors']['GP_deployed']=(y-deployed)**2
            state['decisions'].append({'iteration':t,**decision,'filter_before_call':filter_before_call,'hypotheses_before_call':hypotheses_before_call,
                'enabled_before_call':enabled_before_call,'filter_before':filter_before,'filter_prior':prior,'hypotheses':deepcopy(state['hypotheses']),
                'measurement':measurement,'filter_after':deepcopy(state['filter'])})
            state['history'].append({'candidate_id':ids[selected],'y':y});state['active_seconds']+=time.perf_counter()-start
            state['operator_compiles_in_loop']+=sum(JIT.values())-before_jit;state['status']='COMPLETE' if t==16 else 'RUNNING';write(path,state)
            print(json.dumps({'case':case,'method':method,'round':t,'calls':len(state['calls']),'best_percent':round(max(h['y'] for h in state['history'])*100,3),'eligible':decision['eligible_candidates']},separators=(',',':')),flush=True)
    except Exception as error:
        state['status']='FAILED';state['failure']={'kind':type(error).__name__,'message':str(error)[:300]};write(path,state)
        print(json.dumps({'case':case,'method':method,'failure':state['failure']}),flush=True)
    finally:
        if client:client.close()
    write(ROOT/'runtime'/f"{spec['run_id']}_{method}.json",{'cache':cache.counters,'operator_compiles_in_loop':state['operator_compiles_in_loop'],'active_seconds':state['active_seconds']})
    return {'run':spec['run_id'],'method':method,'status':state['status'],'steps':len(state['decisions'])}


def run_controls_one(spec,config):return [run_one(spec,config,'frozen'),run_one(spec,config,'base')]


def run(config_path='../config.txt',controls=False):
    check_protocol();manifest=read(ROOT/'manifest.json');result=[]
    if controls:
        freeze=read(OUT/'adaptive_frozen.json')
        for spec in manifest:assert filehash(ROOT/'runs'/spec['run_id']/'state.json')==freeze['states'][spec['run_id']]
    with ProcessPoolExecutor(max_workers=WORKERS,initializer=worker_init) as executor:
        futures=[executor.submit(run_controls_one if controls else run_one,spec,config_path) for spec in manifest]
        for future in as_completed(futures):
            value=future.result();result.extend(value if controls else [value])
    if not controls:
        write(OUT/'adaptive_frozen.json',{'states':{spec['run_id']:filehash(ROOT/'runs'/spec['run_id']/'state.json') for spec in manifest},
              'requests':{f.as_posix():filehash(f) for f in (ROOT/'requests').rglob('*.json')},
              'control_states_before_start':sum(len(list((ROOT/folder).glob('*/state.json'))) for folder in ('frozen','base'))})
    return result
