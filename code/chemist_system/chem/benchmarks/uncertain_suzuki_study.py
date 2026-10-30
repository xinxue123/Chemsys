"""Case1-only parameter selection and frozen uncertainty mechanism testing."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
from pathlib import Path
import json
import statistics as st
import time
import traceback

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from . import guarded_suzuki_study as parent
from . import guarded_suzuki_engine as common
from . import uncertain_suzuki_engine as e
from .official_suzuki_adapter import modules,check_sources
from .suzuki_domain import digest

ROOT=Path('data/runs/suzuki_uncertainty');OUT=Path('research/suzuki_uncertainty');WORKERS=8
JIT={'script':0,'trace':0}
read=parent.read;write=parent.write;filehash=parent.filehash

def hashes():return {f.name:filehash(f) for f in [Path(__file__),Path(e.__file__)]}
def prepare():
    if (OUT/'protocol.json').exists():return check_protocol()
    p=parent.check_protocol();assert read(OUT/'qualification.json')['status']=='PASS'
    protocol={'study':'uncertainty_knowledge_mechanisms_v1','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'parent_protocol_sha256':p['protocol_sha256'],'parent_inventory_sha256':filehash(parent.OUT/'release_inventory.json'),
        'LLINBO_inventory_sha256':filehash('research/suzuki_llinbo_guarded/release_inventory.json'),
        'development_cases':[1],'test_cases':[2,3,4],'seeds':list(range(5)),'regimes':list(common.REGIMES),
        'initial_queries':4,'additional_queries':16,'common_candidates':4692,'repeats':False,'budget_ratio':4,
        'common_BO':'Exactly parent GP/RBF/Y standardization, MC512 qLogEI, source regional component smooth.08 and confidence mapping. No GP refit changes.',
        'methods':list(e.METHODS),'controls':list(e.CONTROLS),'parameters':{m:list(v) for m,v in e.PARAMETERS.items()},
        'sigma_cap':'Preserve signed native posterior mean shift, cap absolute change at kappa times latent epistemic sd. Covariance is unchanged.',
        'credible_B4':'Source B4 solver on each original predicate AND (mu+k*sd >= max_available(mu-k*sd)); empty subset retains base fallback. Same logEI budget.',
        'boundary_probe':'On probe rounds maximize latent sd over budget-eligible points violating exactly one initial clause; if empty use all eligible. Other rounds LGBO_gate.',
        'probe_schedules':{str(k):list(v) for k,v in e.PROBES.items()},
        'residual_B4':'credible_B4 with k multiplied by prefix-only RMS predictive z after >=3 previous added queries; z squared clipped100, scale clipped1..3.',
        'sigma_probe_control':'Same probe schedule and off-round LGBO_gate, but uncertainty probe uses entire eligible set.',
        'sigma_BO_control':'Same schedule, entirely knowledge-free; uncertainty probe over eligible set, otherwise B0. Shared between knowledge regimes.',
        'uncertainty':'Latent GP sd drives decisions. Observation-inclusive predictive sd is separate, used only for sequential residual scaling and calibration audit.',
        'calibration_boundary':'Residual inflation heuristic; neither conformal prediction nor a calibrated coverage guarantee. Categorical interval code remains a misspecification risk.',
        'development_selection':{'correct_margin_pp':1.,'retained_positive_LGBO_gain_fraction':.8,'fraction_applies_if_gain_at_least_pp':1.,
            'ranking':'Among qualified settings, maximize min(wrong_category mean AUC,mixed mean AUC), then average, then smaller numeric parameter.',
            'fallback':e.FALLBACK,'if_none_qualified':'Test fixed fallback, preserve qualification failure; no extra search.'},
        'control_selection':'sigma_probe and sigma_BO inherit boundary_probe schedule, never selected separately on performance.',
        'planned_development_runs':210,'planned_test_runs':315,'planned_logical_queries':10500,
        'evaluation':'Mean incumbent over16 added positions, paired differences, clipped downside to B0, actual changed queries and predictive interval coverage. All planned rows retained.',
        'failures':'Carry last incumbent through unfilled budget positions. Source/interface failure separated from optimization results.',
        'resources':{'persistent_workers':WORKERS,'threads_per_worker':1,'device':'cpu','cache':'Exact read-through copies; all prior files remain immutable; import/cholesky warm-up before JIT counting.'},
        'LLM':'No new LLM calls. Fixed original four-clause advice; LLINBO dynamic-point results remain a separately qualified system comparator.',
        'exposure':'Current chat already knows earlier Case2-4 results; this is exploratory, not blind or confirmatory.',
        'source_notes':{'GP_UCB':'https://arxiv.org/abs/0912.3995','calibration':'https://arxiv.org/abs/2112.04620','calibration_caution':'https://proceedings.mlr.press/v216/foldager23a.html'},
        'qualification_sha256':filehash(OUT/'qualification.json'),
        'transport_amendment_sha256':filehash(OUT/'attempt_01/receipt.json') if (OUT/'attempt_01/receipt.json').exists() else None,
        'scientific_sha256':hashes(),'official_sources':check_sources()}
    protocol['protocol_sha256']=digest(protocol);write(OUT/'protocol.json',protocol);write(ROOT/'protocol.json',protocol)
    return protocol

def check_protocol():
    p=read(OUT/'protocol.json');raw=deepcopy(p);h=raw.pop('protocol_sha256');assert digest(raw)==h
    assert p['scientific_sha256']==hashes()
    assert p['parent_inventory_sha256']==filehash(parent.OUT/'release_inventory.json')
    assert p['LLINBO_inventory_sha256']==filehash('research/suzuki_llinbo_guarded/release_inventory.json')
    return p

def worker_init():
    threadpool_limits(1);torch.set_num_threads(1);modules('LGBO');torch.linalg.cholesky(torch.eye(4,dtype=torch.double))
    for name in JIT:
        original=getattr(torch.jit,name)
        def counted(*args,_name=name,_original=original,**kwargs):JIT[_name]+=1;return _original(*args,**kwargs)
        setattr(torch.jit,name,counted)

def run_id(case,seed,regime,method,parameter):return f'c{case}_s{seed}_{regime}_{method}_p{parameter:g}'

def specs(stage,regime):
    if stage=='development':
        if regime=='none':return [('sigma_BO',k) for k in (3,4)]
        return [(m,k) for m in e.METHODS for k in e.PARAMETERS[m]]+[('sigma_probe',k) for k in (3,4)]
    selected=read(OUT/'selected_parameters.json')['methods'];schedule=selected['boundary_probe']['parameter']
    if regime=='none':return [('sigma_BO',schedule)]
    return [(m,selected[m]['parameter']) for m in e.METHODS]+[('sigma_probe',schedule)]

def cache_for(case,seed,regime,folder='cache'):
    group=f'c{case}_s{seed}';own=ROOT/folder/(group+'_'+regime)
    sources=[parent.ROOT/'cache'/group,Path('data/runs/suzuki_llinbo_guarded/cache')/group]
    sources += [ROOT/'cache'/(group+'_'+g) for g in (*common.REGIMES,'none') if g!=regime]
    pool=read(parent.ROOT/'inputs'/f'case{case}'/'pool.json')
    return e.UncertaintyCache(pool['z'],pool['ids'],own,case,seed,sources=sources)

def run_group(job):
    case,seed,regime,stage=job;p=check_protocol();folder=parent.ROOT/'inputs'/f'case{case}'
    pool=read(folder/'pool.json');ids=pool['ids'];lookup={k:i for i,k in enumerate(ids)}
    y=np.load(folder/'oracle.npz')['values'];advice=read(folder/'advice.json')[str(seed)].get(regime,{'clauses':[],'confidence':.7})
    membership=common.clause_membership(pool['z'],advice);hids=[r['id'] for r in advice['clauses']]
    initial=[{'candidate_id':key,'y':float(y[lookup[key]])} for key in pool['initial_ids'][str(seed)]]
    states=[]
    for method,parameter in specs(stage,regime):
        key=run_id(case,seed,regime,method,parameter);file=ROOT/stage/'runs'/key/'state.json'
        state=read(file) if file.exists() else {'run_id':key,'case':case,'seed':seed,'regime':regime,'method':method,'parameter':parameter,
            'stage':stage,'history':deepcopy(initial),'decisions':[],'status':'RUNNING','active_seconds':0.,'operator_compiles_in_loop':0,'protocol_sha256':p['protocol_sha256']}
        accepted=[p['protocol_sha256'],*p.get('accepted_development_protocols',[])] if stage=='development' else [p['protocol_sha256']]
        assert state['protocol_sha256'] in accepted;states.append((state,file))
    cache=cache_for(case,seed,regime)
    for t in range(1,17):
        for state,file in states:
            if state['status'] in ('COMPLETE','FAILED') or len(state['history'])!=t+3:continue
            before=sum(JIT.values());started=time.perf_counter()
            try:
                h=state['history'];base,base_meta=cache.score(h);post=cache.posterior_stats(h);scale=e.residual_scale(state['decisions'])
                queried=np.array([key in {v['candidate_id'] for v in h} for key in ids])
                guidance={}
                def guided(kind,parameter):
                    box=common.region_for(advice,hids)
                    value,meta=cache.capped_score(h,box,advice['confidence'],parameter) if kind=='cap' else cache.score(h,box,advice['confidence'])
                    guidance[kind]=meta;return value
                index,event=e.select(state['method'],state['parameter'],base,post['mu_std'],post['sd_std'],membership,
                    common.budget_at(4,t),queried,ids,hids,guided,t,scale)
                # Outcome revealed only after the policy has selected its ID.
                observed=float(y[index]);mu=float(post['mu_y'][index]);sd=float(post['sd_y'][index]);sp=float(post['sd_predictive_y'][index])
                event.update(iteration=t,base_acquisition=base_meta,posterior_key=str(post['key']),guided_acquisitions=guidance,
                    predicted_y=mu,epistemic_sd_y=sd,predictive_sd_y=sp,
                    predictive_z=(observed-mu)/max(sp,1e-12),epistemic_z=(observed-mu)/max(sd,1e-12))
                state['decisions'].append(event);state['history'].append({'candidate_id':ids[index],'y':observed})
                if len(state['history'])==20:state['status']='COMPLETE'
            except Exception:
                state.update(status='FAILED',error=traceback.format_exc())
            state['active_seconds']+=time.perf_counter()-started;state['operator_compiles_in_loop']+=sum(JIT.values())-before;write(file,state)
    result={'case':case,'seed':seed,'regime':regime,'planned':len(states),'complete':sum(s['status']=='COMPLETE' for s,f in states),
        'failed':[s['run_id'] for s,f in states if s['status']=='FAILED'],'cache_counters':cache.counters,'operator_compiles_in_loop':sum(s['operator_compiles_in_loop'] for s,f in states)}
    write(ROOT/stage/'groups'/f'c{case}_s{seed}_{regime}.json',result);return result

def run_stage(stage):
    p=check_protocol()
    if stage=='test':
        selected=read(OUT/'selected_parameters.json');assert selected['protocol_sha256']==p['protocol_sha256']
        receipt=OUT/'test_start_receipt.json'
        if not receipt.exists():
            assert not list((ROOT/'test').rglob('state.json'))
            write(receipt,{'selected_parameters_sha256':filehash(OUT/'selected_parameters.json'),'test_states_before_start':0,'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    cases=(1,) if stage=='development' else (2,3,4)
    jobs=[(c,seed,g,stage) for c in cases for seed in range(5) for g in (*common.REGIMES,'none')]
    with ProcessPoolExecutor(max_workers=WORKERS,initializer=worker_init) as pool:
        for f in as_completed([pool.submit(run_group,j) for j in jobs]):print(json.dumps(f.result(),separators=(',',':')),flush=True)
    return {'stage':stage,'groups':len(jobs)}

def select_parameters():
    p=check_protocol();path=OUT/'selected_parameters.json'
    if path.exists():return read(path)
    assert not list((ROOT/'test').rglob('state.json'))
    states=[read(f) for f in (ROOT/'development/runs').glob('*/state.json')];assert len(states)==210 and all(r['status']=='COMPLETE' for r in states)
    old=read(parent.OUT/'selected_budget.json');assert old['ratio']==4
    def existing(method,regime):
        key=f'c1_s{{seed}}_{regime}_{method}'+('' if method=='LGBO_region' else '_r4')
        if method=='B0':key='c1_s{seed}_none_B0'
        return st.mean(parent.state_metrics(read(parent.ROOT/'development/runs'/key.format(seed=seed)/'state.json'))['auc'] for seed in range(5))
    b0=existing('B0','none');lg=existing('LGBO_region','correct');gain=lg-b0;chosen={}
    for method in e.METHODS:
        candidates=[]
        for parameter in e.PARAMETERS[method]:
            mean=lambda g:st.mean(parent.state_metrics(dict(r,ratio=4))['auc'] for r in states if r['method']==method and r['regime']==g and r['parameter']==parameter)
            good=mean('correct');wrong=mean('wrong_category');mixed=mean('mixed');ret=(good-b0)/gain if gain>=1 else None
            candidates.append({'parameter':parameter,'correct_auc':good,'wrong_auc':wrong,'mixed_auc':mixed,
                'qualified':good>=lg-1 and (ret is None or ret>=.8),'retained_gain':ret})
        qualified=[v for v in candidates if v['qualified']]
        winner=min(qualified,key=lambda v:(-min(v['wrong_auc'],v['mixed_auc']),-(v['wrong_auc']+v['mixed_auc'])/2,v['parameter'])) if qualified else next(v for v in candidates if v['parameter']==e.FALLBACK[method])
        chosen[method]={'parameter':winner['parameter'],'development_qualified':bool(qualified),'candidates':candidates}
    result={'protocol_sha256':p['protocol_sha256'],'methods':chosen,'Case1_B0_auc':b0,'Case1_LGBO_correct_auc':lg,'test_results_used':False}
    write(path,result);return result
