"""Frozen-control study with qLogEI at both rollout stages."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import shutil

from chem.benchmarks import knowledge_gp
from fast_hit_bo import study as legacy, PARAMETERS as P
from fast_hit_bo.model import SharedGP, digest
from . import VARIANT
from .policy import decision

ROOT=legacy.ROOT
BASE=ROOT/'experiments/runs'/VARIANT
REPORT=ROOT/'experiments/reports'/VARIANT
DESIGN=ROOT/'docs'/f'{VARIANT}.txt'
REFERENCE=ROOT/'experiments/runs/fast_hit_bo_20261104'
read,write,sha=legacy.read,legacy.write,legacy.sha


def check():
    p=read(BASE/'protocol.json')
    assert p['protocol_sha256']==digest({k:v for k,v in p.items() if k!='protocol_sha256'})
    for group in ('sources_sha256','inputs_sha256','protected_sha256'):
        for path,h in p[group].items():
            if sha(ROOT/path)!=h:
                raise ValueError('Frozen artifact changed: '+path)
    return p


def prepare():
    if (BASE/'protocol.json').exists():
        return check()
    old=read(REFERENCE/'protocol.json')
    for group in ('sources_sha256','inputs_sha256'):
        for path,h in old[group].items():
            assert sha(ROOT/path)==h,path
    refs=[j for j in old['jobs'] if j['arm']=='qlogei']
    jobs=[dict(j,id=j['id'].replace('_qlogei','_rollout_qlogei'),arm='rollout_qlogei') for j in refs]
    assert len(jobs)==10
    sources=dict(old['sources_sha256'])
    paths=list(Path(__file__).parent.glob('*.py'))+[DESIGN,ROOT/'code/experiment_tools/run_qlogei_rollout.py']
    for path in paths:
        sources[path.relative_to(ROOT).as_posix()]=sha(path)
    protected={}
    for directory in (REFERENCE/'runs',ROOT/'experiments/reports/fast_hit_bo_20261104'):
        for path in directory.rglob('*'):
            if path.is_file():
                protected[path.relative_to(ROOT).as_posix()]=sha(path)
    for path in (REFERENCE/'protocol.json',REFERENCE/'evaluation_endpoint.json',REFERENCE/'preflight.json'):
        protected[path.relative_to(ROOT).as_posix()]=sha(path)
    p=dict(variant=VARIANT,registered_utc=datetime.now(timezone.utc).isoformat(),
        jobs=jobs,reference_jobs=refs,reference_directory=REFERENCE.relative_to(ROOT).as_posix(),
        parameters=P,initial=1,total_queries=8,sources_sha256=sources,
        inputs_sha256=old['inputs_sha256'],protected_sha256=protected,
        design=DESIGN.read_text(encoding='utf-8'),new_observations_at_registration=0)
    p['protocol_sha256']=digest(p)
    write(BASE/'protocol.json',p)
    write(BASE/'evaluation_endpoint.json',read(REFERENCE/'evaluation_endpoint.json'))
    (BASE/'cache').mkdir(exist_ok=True)
    copied=0
    for path in (REFERENCE/'cache').iterdir():
        if path.suffix in ('.pt','.npz'):
            shutil.copyfile(path,BASE/'cache'/path.name)
            assert sha(path)==sha(BASE/'cache'/path.name)
            copied+=1
    write(BASE/'cache_copy.json',dict(files=copied,policy='Observed-history hash lookup only'))
    return check()


def initialize():
    check()
    (ROOT/'experiments/chemist_system').mkdir(parents=True,exist_ok=True)
    os.chdir(ROOT/'experiments/chemist_system')
    pool=knowledge_gp.load_task(ROOT,'Reizman_Suzuki_Case_1',seed=0)
    legacy.BASE=BASE
    legacy.POOL=pool
    legacy.GP=SharedGP(pool,BASE/'cache')
    legacy.decision=decision


def preflight():
    from .validation import qualify
    p=check()
    initialize()
    result=qualify(legacy.GP,read(REFERENCE/'runs/s4_middle_rollout/state.json')['history'])
    result['protocol_sha256']=p['protocol_sha256']
    result['new_experiments']=0
    assert sum(knowledge_gp.COMPILE_COUNTS.values())==0
    result['operator_compiles_in_loop']=0
    write(BASE/'preflight.json',result)
    check()
    return result


def batch(workers):
    p=check()
    assert read(BASE/'preflight.json')['status']=='PASS'
    results,errors=[],[]
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize) as executor:
        futures={executor.submit(legacy.run_job,j):j for j in p['jobs']}
        for f in as_completed(futures):
            try:
                results.append(f.result())
            except Exception as error:
                failure=dict(job=futures[f]['id'],type=type(error).__name__,error=str(error))
                errors.append(failure)
                print(json.dumps(failure),flush=True)
    check()
    receipt=dict(status='FAILED' if errors else 'COMPLETE',results=results,errors=errors)
    write(BASE/'batch_receipt.json',receipt,replace=True)
    if errors:
        raise RuntimeError(errors)
    return receipt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('prepare','preflight','run'))
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    result=prepare() if args.action=='prepare' else preflight() if args.action=='preflight' else batch(args.workers)
    if args.action=='prepare':
        result=dict(status='REGISTERED',jobs=len(result['jobs']),protocol_sha256=result['protocol_sha256'])
    print(json.dumps(result))
