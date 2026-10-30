"""Frozen acquisition-only study; all policy inputs exclude unqueried outcomes."""
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
from . import acquisition_suzuki_engine as e
from .llinbo_guarded_report import metric,baseline
from .official_suzuki_adapter import modules,check_sources
from .suzuki_domain import digest

ROOT=Path('data/runs/suzuki_acquisition');OUT=Path('research/suzuki_acquisition')
WORKERS=8;JIT={'script':0,'trace':0}
read=parent.read;write=parent.write;filehash=parent.filehash
PROTECTED={name:Path(f'research/suzuki_{name}/release_inventory.json') for name in ('official','guarded','llinbo_guarded','uncertainty')}


def hashes():return {p.name:filehash(p) for p in (Path(__file__),Path(e.__file__))}


def cache_for(case,seed,regime,folder='cache'):
    group=f'c{case}_s{seed}';own=ROOT/folder/(group+'_'+regime)
    sources=[parent.ROOT/'cache'/group,Path('data/runs/suzuki_llinbo_guarded/cache')/group,
             Path('data/runs/suzuki_uncertainty/audit_cache')/(group+'_llinbo_audit')]
    sources += [Path('data/runs/suzuki_uncertainty/cache')/(group+'_'+g) for g in (*common.REGIMES,'none')]
    sources += [ROOT/'cache'/(group+'_'+g) for g in (*common.REGIMES,'none') if g!=regime]
    pool=read(parent.ROOT/'inputs'/f'case{case}'/'pool.json')
    return e.AcquisitionCache(pool['z'],pool['ids'],own,case,seed,sources=sources)


def worker_init():
    threadpool_limits(1);torch.set_num_threads(1);modules('LGBO')
    torch.linalg.cholesky(torch.eye(4,dtype=torch.double))
    for name in JIT:
        original=getattr(torch.jit,name)
        def counted(*args,_name=name,_original=original,**kwargs):JIT[_name]+=1;return _original(*args,**kwargs)
        setattr(torch.jit,name,counted)


def qualify():
    worker_init();pool=read(parent.ROOT/'inputs/case1/pool.json')
    row=baseline(1,0,'correct','B0');history=row['history'][:4]
    advice=read(parent.ROOT/'inputs/case1/advice.json')['0']['correct']
    cache=cache_for(1,0,'qualification','qualification_cache');base,meta=cache.score(history)
    key=row['decisions'][0]['base_acquisition']['key']
    with np.load(parent.ROOT/'cache/c1_s0'/(key+'.npz')) as d:assert np.array_equal(base,d['scores'])
    checks=[]
    for box in (None,common.region_for(advice,[r['id'] for r in advice['clauses']])):
        effective,value,_=cache.effective_model(history,box,advice['confidence'])
        comp=cache.components(history,box,advice['confidence'])
        with torch.no_grad():
            for beta in (e.beta_at('official',1),.25,1.,4.):
                native=torch.cat([e.UpperConfidenceBound(effective,beta=torch.tensor(beta,dtype=torch.double))(x.unsqueeze(1)).reshape(-1) for x in cache.z.split(256)]).numpy()
                own,_=cache.ucb(history,beta,box,advice['confidence'])
                error=float(np.max(np.abs(native-own)));assert error<1e-12
                checks.append({'box':box,'beta':beta,'max_native_UCB_error':error})
            for x in cache.z.split(256):
                p=effective.posterior(x.unsqueeze(1));p0=value['model'].posterior(x.unsqueeze(1))
                assert torch.allclose(p.variance,p0.variance+1e-6,atol=1e-12,rtol=1e-12)
    assert cache.counters['fits']==0 and sum(JIT.values())==0
    result={'status':'PASS','initial_qLogEI_bit_exact':True,'native_UCB_checks':checks,
            'shared_posterior_jitter_verified':1e-6,'new_GP_fits':0,'operator_compiles_in_loop':0,'scientific_sha256':hashes()}
    write(OUT/'qualification.json',result);return result


def prepare():
    if (OUT/'protocol.json').exists():return check_protocol()
    parent.check_protocol();q=read(OUT/'qualification.json');assert q['status']=='PASS' and q['scientific_sha256']==hashes()
    p={'study':'UCB_vs_qLogEI_common_Suzuki_v1','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
       'scientific_sha256':hashes(),'protected_inventories':{k:filehash(v) for k,v in PROTECTED.items()},
       'development_cases':[1],'primary_cases':[2,3,4],'supplementary_cases':[1,2,3,4],'seeds':list(range(5)),
       'regimes':list(common.REGIMES),'policies':['B0','LGBO_region'],'initial_queries':4,'additional_queries':16,
       'candidates':4692,'repeated_candidates':False,'GP':'Frozen common native LGBO fit, Y standardization, RBF, categorical ordinal coordinates; unchanged.',
       'region':'Original fixed native LGBO signed mean tilt, grid512, smooth.08, delta=3*Phi^-1(confidence), covariance jitter1e-6 for both acquisitions.',
       'qLogEI':'Reuse 20 frozen B0 plus80 frozen LGBO_region trajectories, MC512, q=1; do not rerun or charge as new queries.',
       'UCB_official':'mu+sqrt(beta_t)*sd, beta_t=2*log(t*4*pi**2/.1*6), t1..16; double precision beta. Original beta expression only; this is not full native LLINBO.',
       'UCB_fixed':{'development_kappas':list(e.KAPPAS),'beta':'kappa**2','selection':'Highest Case1 B0 mean SearchAUC across5 seeds, then smaller kappa; same winner for every policy/regime/case.'},
       'gate':'No admission gate in sequential comparison: logEI ratio budgets cannot transfer numerically to UCB.',
       'development_unique_runs':15,'main_UCB_runs':200,'main_reused_development_runs':5,'new_unique_trajectories':210,'new_logical_queries':4200,
       'metrics':'SearchAUC=mean of incumbent over16 added queries, yield percentage points; final yield, paired differences, same-prefix sigma choices, downside against own-acquisition B0.',
       'failures':'All planned trajectories retained; carry last incumbent through missing positions; failures not scientific successes.',
       'offline_LLMinbo':'1280 old actual LLINBO prefixes held fixed: UCB topN gate, N equals old qLogEI admissible set size. Equal capacity diagnostic only, not native psi or new sequential LLM run.',
       'offline_analytic_EI':'320 old B0 prefixes: analytic q1 LogEI versus MC512 qLogEI argmax on same GP/posterior; diagnostic only.',
       'LLM':'No new LLM answers; four original advice conditions held fixed. Current chat exposed to previous test results, exploratory comparison.',
       'resources':{'workers':WORKERS,'threads_per_worker':1,'device':'cpu','warmup':'imports and cholesky before counted loop','cache':'Atomic read-through, prior cache trees immutable'},
       'qualification_sha256':filehash(OUT/'qualification.json'),'official_sources':check_sources(),
       'references':['https://botorch.readthedocs.io/en/v0.14.0/acquisition.html','https://arxiv.org/abs/2310.20708']}
    p['protocol_sha256']=digest(p);write(OUT/'protocol.json',p);write(ROOT/'protocol.json',p);return p


def check_protocol():
    p=read(OUT/'protocol.json');raw=deepcopy(p);h=raw.pop('protocol_sha256');assert digest(raw)==h
    assert p['scientific_sha256']==hashes()
    assert p['protected_inventories']=={k:filehash(v) for k,v in PROTECTED.items()}
    return p


def run_id(case,seed,regime,mode,kappa):return f'c{case}_s{seed}_{regime}_{mode}'+(f'_k{kappa:g}' if kappa is not None else '')


def state_path(case,seed,regime,mode,kappa,stage='main'):
    # The selected Case1 B0 development trajectories are referenced, not copied.
    if stage=='main' and case==1 and regime=='none' and mode=='fixed':stage='development'
    return ROOT/stage/'runs'/run_id(case,seed,regime,mode,kappa)/'state.json'


def configurations(stage):
    if stage=='development':return [('fixed',k) for k in e.KAPPAS]
    return [('official',None),('fixed',read(OUT/'selected_kappa.json')['kappa'])]


def run_group(job):
    case,seed,regime,stage=job;p=check_protocol();folder=parent.ROOT/'inputs'/f'case{case}'
    pool=read(folder/'pool.json');ids=pool['ids'];ix={k:i for i,k in enumerate(ids)}
    y=np.load(folder/'oracle.npz')['values'];advice=read(folder/'advice.json')[str(seed)].get(regime,{'clauses':[],'confidence':.7})
    box=common.region_for(advice,[v['id'] for v in advice['clauses']]) if regime!='none' else None
    initial=[{'candidate_id':k,'y':float(y[ix[k]])} for k in pool['initial_ids'][str(seed)]]
    states=[]
    for mode,kappa in configurations(stage):
        path=state_path(case,seed,regime,mode,kappa,stage)
        row=read(path) if path.exists() else {'run_id':run_id(case,seed,regime,mode,kappa),'case':case,'seed':seed,
            'regime':regime,'method':'B0' if regime=='none' else 'LGBO_region','mode':mode,'kappa':kappa,'stage':stage,
            'history':deepcopy(initial),'decisions':[],'status':'RUNNING','active_seconds':0.,'operator_compiles_in_loop':0,'protocol_sha256':p['protocol_sha256']}
        assert row['protocol_sha256']==p['protocol_sha256'];states.append((row,path))
    cache=cache_for(case,seed,regime)
    for t in range(1,17):
        for row,path in states:
            if row['status'] in ('COMPLETE','FAILED') or len(row['history'])!=t+3:continue
            before=sum(JIT.values());started=time.perf_counter()
            try:
                h=row['history'];beta=e.beta_at(row['mode'],t,row['kappa']);scores,meta=cache.ucb(h,beta,box,advice['confidence'])
                observed={v['candidate_id'] for v in h};available=np.array([k not in observed for k in ids])
                selected=common.argmax(scores,available,ids);comp=cache.components(h,box,advice['confidence'])
                event={'iteration':t,'selected_id':ids[selected],'acquisition':meta,'score':float(scores[selected]),
                       'mean_std':float(comp['mu'][selected]),'sd_std':float(comp['sd'][selected])}
                row['decisions'].append(event)
                # Only this selected outcome is revealed to the next policy call.
                row['history'].append({'candidate_id':ids[selected],'y':float(y[selected])})
                if len(row['history'])==20:row['status']='COMPLETE'
            except Exception:row.update(status='FAILED',error=traceback.format_exc())
            row['active_seconds']+=time.perf_counter()-started;row['operator_compiles_in_loop']+=sum(JIT.values())-before;write(path,row)
    result={'case':case,'seed':seed,'regime':regime,'complete':sum(r['status']=='COMPLETE' for r,f in states),
            'failed':[r['run_id'] for r,f in states if r['status']=='FAILED'],'cache_counters':cache.counters}
    write(ROOT/stage/'groups'/f'c{case}_s{seed}_{regime}.json',result);return result


def run_stage(stage):
    check_protocol()
    if stage=='main':
        selection=read(OUT/'selected_kappa.json');assert selection['protocol_sha256']==check_protocol()['protocol_sha256']
        receipt=OUT/'main_start_receipt.json'
        if not receipt.exists():
            assert not list((ROOT/'main').rglob('state.json'))
            write(receipt,{'selected_kappa_sha256':filehash(OUT/'selected_kappa.json'),'main_states_before_start':0,'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    jobs=[(1,s,'none',stage) for s in range(5)] if stage=='development' else [(c,s,g,stage) for c in (1,2,3,4) for s in range(5) for g in ('none',*common.REGIMES)]
    with ProcessPoolExecutor(max_workers=WORKERS,initializer=worker_init) as executor:
        for future in as_completed([executor.submit(run_group,j) for j in jobs]):print(json.dumps(future.result(),separators=(',',':')),flush=True)
    return {'stage':stage,'groups':len(jobs)}


def select_kappa():
    p=check_protocol();path=OUT/'selected_kappa.json'
    if path.exists():return read(path)
    assert not list((ROOT/'main').rglob('state.json'))
    rows=[read(f) for f in (ROOT/'development/runs').glob('*/state.json')]
    assert len(rows)==15 and all(r['status']=='COMPLETE' for r in rows)
    candidates=[{'kappa':k,'Case1_B0_AUC':st.mean(metric(r)['auc'] for r in rows if r['kappa']==k)} for k in e.KAPPAS]
    winner=min(candidates,key=lambda v:(-v['Case1_B0_AUC'],v['kappa']))
    result={'protocol_sha256':p['protocol_sha256'],'kappa':winner['kappa'],'candidates':candidates,'test_results_used':False}
    write(path,result);return result
