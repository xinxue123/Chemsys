"""Exact singleton predecessor layer of Case 1 optimum reachability."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from chem.benchmarks import knowledge_gp
from cold_start_v13.gp import ColdGP

NAME='suzuki_case1_recursive_paths_20261104'
BASE=ROOT/'experiments/runs'/NAME
REPORT=ROOT/'experiments/reports'/NAME
DESIGN=ROOT/'docs'/f'{NAME}.txt'
GP=None
POOL=None
TARGET=None


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def write(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if read(path)!=value:
            raise ValueError('Existing evidence differs: '+str(path))
        return
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    temp.replace(path)


def prepare():
    if (BASE/'protocol.json').exists():
        return check()
    pool=knowledge_gp.load_task(ROOT,'Reizman_Suzuki_Case_1',seed=0)
    oracle=pool.source_path.with_name('oracle.npz')
    with np.load(oracle,allow_pickle=False) as z:
        y=z['values']*100
    lo,hi=map(float,np.quantile(y,[1/3,2/3]))
    targets=np.flatnonzero(y==y.max())
    assert len(targets)==1
    target=int(targets[0])
    jobs=[dict(id=pool.ids[j],index=int(j),group='low' if y[j]<=lo else 'middle')
          for j in np.flatnonzero(y<=hi)]
    assert len(jobs)==3128
    old=read(ROOT/'experiments/runs/suzuki_initial_yield_20261104/protocol.json')
    source=dict(old['sources_sha256'])
    source[Path(__file__).relative_to(ROOT).as_posix()]=sha(__file__)
    source[DESIGN.relative_to(ROOT).as_posix()]=sha(DESIGN)
    p=dict(name=NAME,registered_utc=datetime.now(timezone.utc).isoformat(),jobs=jobs,
        target_id=pool.ids[target],target_index=target,target_yield=float(y[target]),
        lower_tercile=lo,upper_tercile=hi,max_path_length=8,
        admissibility='exact_maximum_qLogEI_all_ties',initial_groups=['low','middle'],
        sources_sha256=source,inputs_sha256={str(x.relative_to(ROOT)).replace(chr(92),'/'):sha(x)
                                            for x in (pool.source_path,oracle)},
        design=DESIGN.read_text(encoding='utf-8-sig'),requested_depth=8)
    p['protocol_sha256']=digest(p)
    write(BASE/'protocol.json',p)
    return check()


def check():
    p=read(BASE/'protocol.json')
    assert p['protocol_sha256']==digest({k:v for k,v in p.items() if k!='protocol_sha256'})
    for group in ('sources_sha256','inputs_sha256'):
        for path,h in p[group].items():
            if sha(ROOT/path)!=h:
                raise ValueError('Source/input changed: '+path)
    return p


def initialize():
    global GP,POOL,TARGET
    os.chdir(ROOT/'experiments/chemist_system')
    p=check()
    TARGET=p['target_index']
    POOL=knowledge_gp.load_task(ROOT,'Reizman_Suzuki_Case_1',seed=0)
    GP=ColdGP(POOL,BASE/'cache',seed=0)
    GP.cache.sources=[ROOT/'experiments/runs/suzuki_initial_yield_20261104/cache/case1']


def one(job):
    path=BASE/'singleton_states'/f"{job['id']}.json"
    if path.exists():
        return read(path)
    observation=knowledge_gp.read_observation(ROOT,POOL,job['id'])
    history=[dict(candidate_id=job['id'],y=observation['y'])]
    score,meta=GP.cache.score(history)
    available=np.ones(len(POOL.ids),bool)
    available[job['index']]=False
    max_score=float(score[available].max())
    eligible=available&(score==max_score)
    ties=knowledge_gp.semantic_priorities(POOL,2,seed=0)
    js=np.flatnonzero(eligible)
    native_selected=int(js[np.argmin(ties[js])])
    score_path=GP.directory/(meta['key']+'.npz')
    record=dict(initial_id=job['id'],initial_index=job['index'],initial_group=job['group'],
        initial_yield=observation['raw_value'],history_sha256=digest(history),
        target_id=POOL.ids[TARGET],target_score=float(score[TARGET]),max_score=max_score,
        target_gap=max_score-float(score[TARGET]),target_eligible=bool(eligible[TARGET]),
        admissible_next_count=len(js),native_selected=POOL.ids[native_selected],
        score_file=score_path.relative_to(ROOT).as_posix(),score_sha256=sha(score_path),
        fit_key=meta['fit_key'],fitted_model_sha256=sha(GP.directory/(meta['fit_key']+'.pt')))
    assert GP.counters['operator_compiles_in_loop']==0
    record['record_sha256']=digest(record)
    write(path,record)
    GP.release()
    return record


def preflight():
    initialize()
    old=read(ROOT/'experiments/runs/suzuki_initial_yield_20261104/protocol.json')
    checks=[]
    for job in old['jobs']:
        if job['case']!=1 or job['arm']=='mixed':
            continue
        s=read(ROOT/'experiments/runs/suzuki_initial_yield_20261104/runs'/job['id']/'state.json')
        prefix=[dict(candidate_id=s['history'][0]['candidate_id'],y=s['history'][0]['y'])]
        score,meta=GP.cache.score(prefix)
        old_arrays=ROOT/'experiments/runs/suzuki_initial_yield_20261104/runs'/job['id']/'rounds/02_GP.npz'
        with np.load(old_arrays,allow_pickle=False) as z:
            np.testing.assert_array_equal(score,z['base_logei'])
        checks.append(dict(initial_id=prefix[0]['candidate_id'],scores_exact=True))
        GP.release()
    value=dict(status='PASS',checks=checks,operator_compiles_in_loop=GP.counters['operator_compiles_in_loop'])
    write(BASE/'preflight.json',value)
    return value


def collect(p):
    rows=[read(BASE/'singleton_states'/f"{job['id']}.json") for job in p['jobs']]
    assert len(rows)==3128
    REPORT.mkdir(parents=True,exist_ok=True)
    for row in rows:
        assert row['record_sha256']==digest({k:v for k,v in row.items() if k!='record_sha256'})
        assert sha(ROOT/row['score_file'])==row['score_sha256']
    cols=['initial_id','initial_group','initial_yield','target_id','target_score','max_score',
          'target_gap','target_eligible','admissible_next_count','native_selected']
    for filename,records in (('all_initial_states.csv',rows),
                              ('all_two_step_paths.csv',[r for r in rows if r['target_eligible']])):
        with (REPORT/filename).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore',lineterminator='\n')
            w.writeheader()
            w.writerows(records)
    summary=dict(status='COMPLETE_DEPTH_2_ONLY',singleton_states=len(rows),
        two_step_paths=sum(r['target_eligible'] for r in rows),
        outgoing_edges=sum(r['admissible_next_count'] for r in rows),
        nonterminal_outgoing_edges=sum(r['admissible_next_count']-r['target_eligible'] for r in rows),
        native_direct_target=sum(r['native_selected']==r['target_id'] for r in rows),
        per_group={g:dict(starts=sum(r['initial_group']==g for r in rows),
                          two_step_paths=sum(r['initial_group']==g and r['target_eligible'] for r in rows))
                   for g in ('low','middle')},requested_maximum_depth=8,
        full_depth8_complete=False,operator_compiles_in_loop=0)
    write(REPORT/'layer2_summary.json',summary)
    return summary


def batch(workers):
    p=check()
    assert read(BASE/'preflight.json')['status']=='PASS'
    todo=[j for j in p['jobs'] if not (BASE/'singleton_states'/f"{j['id']}.json").exists()]
    done=len(p['jobs'])-len(todo)
    started=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize) as executor:
        fs={executor.submit(one,j):j for j in todo}
        for f in as_completed(fs):
            f.result()
            done+=1
            if done%50==0 or done==len(p['jobs']):
                print(json.dumps(dict(complete_singletons=done,total=3128,
                    elapsed_seconds=round(time.monotonic()-started,1))),flush=True)
    check()
    return collect(p)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('action',choices=('prepare','preflight','singleton'))
    ap.add_argument('--workers',type=int,default=8)
    a=ap.parse_args()
    result=prepare() if a.action=='prepare' else preflight() if a.action=='preflight' else batch(a.workers)
    if a.action=='prepare':
        result=dict(protocol_sha256=result['protocol_sha256'],singleton_roots=len(result['jobs']))
    print(json.dumps(result))
