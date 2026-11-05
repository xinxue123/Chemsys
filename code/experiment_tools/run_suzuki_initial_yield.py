"""One controlled initial point followed by nineteen native pure-BO choices."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from chem.benchmarks import knowledge_gp
from cold_start_v13.gp import ColdGP

NAME = 'suzuki_initial_yield_20261104'
BASE = ROOT/'experiments/runs'/NAME
REPORT = ROOT/'experiments/reports'/NAME
DESIGN = ROOT/'docs'/f'{NAME}.txt'
SEEDS = tuple(range(5))
ARMS = ('low','middle','mixed')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
                                     ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value,immutable=True):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and immutable:
        if read(path)!=value:
            raise ValueError('Immutable file changed: '+str(path))
        return
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def pool_for(case):
    return knowledge_gp.load_task(ROOT,f'Reizman_Suzuki_Case_{case}',seed=0)


def oracle_for(pool):
    with np.load(pool.source_path.with_name('oracle.npz'),allow_pickle=False) as z:
        return z['raw_values'].copy() if 'raw_values' in z else z['values'].copy()*pool.scale


def prepare():
    path=BASE/'protocol.json'
    if path.exists():
        return check()
    jobs,inputs,groups=[],{},[]
    for case in range(1,5):
        pool=pool_for(case)
        y=oracle_for(pool)
        lower,upper=map(float,np.quantile(y,[1/3,2/3]))
        label=np.where(y<=lower,'low',np.where(y<=upper,'middle','high'))
        order=sorted(range(len(pool.ids)),key=lambda j:pool.features[j])
        groups.append(dict(case=case,candidates=len(y),lower_tercile=lower,upper_tercile=upper,
            minimum=float(y.min()),maximum=float(y.max()),
            counts={g:int(np.count_nonzero(label==g)) for g in ('low','middle','high')}))
        for p in (pool.source_path,pool.source_path.with_name('oracle.npz')):
            inputs[p.relative_to(ROOT).as_posix()]=sha(p)
        for seed in SEEDS:
            priority=np.empty(len(y),dtype=np.int64)
            priority[order]=np.random.default_rng(np.random.SeedSequence([seed,case,20261104])).permutation(len(y))
            for arm in ARMS:
                allowed=label!='high' if arm=='mixed' else label==arm
                js=np.flatnonzero(allowed)
                j=int(js[np.argmin(priority[js])])
                jobs.append(dict(id=f'c{case}_s{seed}_{arm}',case=case,seed=seed,arm=arm,
                    initial_id=pool.ids[j],initial_group=str(label[j])))
    sources={Path(__file__).relative_to(ROOT).as_posix():sha(__file__),
        DESIGN.relative_to(ROOT).as_posix():sha(DESIGN),
        Path(knowledge_gp.__file__).relative_to(ROOT).as_posix():sha(knowledge_gp.__file__),
        'code/experiment_tools/cold_start_v13/gp.py':sha(ROOT/'code/experiment_tools/cold_start_v13/gp.py')}
    # Capture all existing scientific dependencies without editing them.
    # Portable registration captures the selected local source closure.
    # The measured inputs, RNG, GP, acquisition, and budget are unchanged.
    for folder in ('code/chemist_system/chem','code/experiment_tools'):
        for dependency in (ROOT/folder).rglob('*.py'):
            sources[dependency.relative_to(ROOT).as_posix()]=sha(dependency)
    p=dict(name=NAME,registered_utc=datetime.now(timezone.utc).isoformat(),initial=1,BO_queries=19,total=20,
        seeds=list(SEEDS),jobs=jobs,groups=groups,inputs_sha256=inputs,sources_sha256=sources,
        design=DESIGN.read_text(encoding='utf-8-sig'),LLM_calls=0)
    p['protocol_sha256']=digest(p)
    write(path,p)
    REPORT.mkdir(parents=True,exist_ok=True)
    return check()


def check():
    p=read(BASE/'protocol.json')
    if digest({k:v for k,v in p.items() if k!='protocol_sha256'})!=p['protocol_sha256']:
        raise ValueError('Protocol mismatch')
    for group in ('sources_sha256','inputs_sha256'):
        for name,h in p[group].items():
            if sha(ROOT/name)!=h:
                raise ValueError('Registered source/input changed: '+name)
    return p


def job_entry(job):
    (ROOT/'experiments/chemist_system').mkdir(parents=True,exist_ok=True)
    os.chdir(ROOT/'experiments/chemist_system')
    pool=pool_for(job['case'])
    folder=BASE/'runs'/job['id']
    path=folder/'state.json'
    s=read(path) if path.exists() else dict(job=job,history=[],decisions=[],status='RUNNING')
    if s['status']=='COMPLETE':
        return dict(id=job['id'],status='COMPLETE')
    gp=ColdGP(pool,BASE/'cache'/f"case{job['case']}",seed=0)
    while len(s['history'])<20:
        q=len(s['history'])+1
        cp=folder/'rounds'/f'{q:02}_commit.json'
        if cp.exists():
            c=read(cp)
            if c['prestate_sha256']!=digest(s):
                raise ValueError('State changed before feedback')
        else:
            if q==1:
                c=dict(query=q,selected_id=job['initial_id'],rule='group_controlled_initial_draw',
                    gp_mean=None,gp_sd=None,gp_predictive_sd=None,base_qLogEI=None,
                    qLogEI_tied_best_count=None)
            else:
                ctx=gp.context(s['history'],q)
                j=int(ctx['bo_index'])
                c=dict(query=q,selected_id=pool.ids[j],rule='maximum_native_qLogEI',
                    gp_mean=float(ctx['stats']['mu_y'][j])*pool.scale,
                    gp_sd=float(ctx['stats']['sd_y'][j])*pool.scale,
                    gp_predictive_sd=float(ctx['stats']['sd_predictive_y'][j])*pool.scale,
                    base_qLogEI=float(ctx['base_logei'][j]),
                    qLogEI_tied_best_count=int(np.count_nonzero(ctx['available']&
                                                (ctx['base_logei']==ctx['base_logei'][j]))),
                    fit_key=ctx['stats']['fit_key'],gp_fit_parameters=ctx['stats']['fit_parameters'])
                rp=folder/'rounds'/f'{q:02}_GP.npz'
                rp.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(rp,base_logei=ctx['base_logei'],available=ctx['available'],
                    tie_priority=ctx['tie_priority'],mu_y=ctx['stats']['mu_y'],sd_y=ctx['stats']['sd_y'],
                    sd_predictive_y=ctx['stats']['sd_predictive_y'])
                c['GP_arrays_sha256']=sha(rp)
            c.update(prestate_sha256=digest(s),history_sha256=digest(s['history']))
            c['commit_sha256']=digest(c)
            write(cp,c)
        if c['selected_id'] in {o['candidate_id'] for o in s['history']}:
            raise ValueError('Repeated query')
        obs=knowledge_gp.read_observation(ROOT,pool,c['selected_id'])
        s['history'].append(obs)
        s['decisions'].append(dict(c,objective=obs['raw_value']))
        s['status']='COMPLETE' if q==20 else 'RUNNING'
        write(path,s,immutable=False)
        if gp.counters['operator_compiles_in_loop']:
            raise ValueError('Compilation inside loop')
        write(folder/'runtime_counters.json',gp.counters,immutable=False)
        gp.release()
        print(json.dumps(dict(job=job['id'],query=q,best=max(o['raw_value'] for o in s['history']))),flush=True)
    return dict(id=job['id'],status='COMPLETE')


def batch(workers):
    p=check()
    errors,results=[],[]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        fs={executor.submit(job_entry,j):j for j in p['jobs']}
        for f in as_completed(fs):
            try:
                results.append(f.result())
            except Exception as e:
                errors.append(dict(job=fs[f]['id'],type=type(e).__name__,error=str(e)))
                print(json.dumps(errors[-1]),flush=True)
    check()
    result=dict(status='COMPLETE' if not errors else 'FAILED',results=results,errors=errors)
    write(BASE/'batch_receipt.json',result,immutable=False)
    if errors:
        raise RuntimeError(errors)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('prepare','run'))
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    result=prepare() if args.action=='prepare' else batch(args.workers)
    print(json.dumps(dict(status='REGISTERED',jobs=len(result['jobs']),protocol_sha256=result['protocol_sha256'])
                     if args.action=='prepare' else result))
