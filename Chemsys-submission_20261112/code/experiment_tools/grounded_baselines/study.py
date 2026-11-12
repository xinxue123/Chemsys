"""Registered common-first 0+20 baseline study; frozen controls are read-only."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import argparse
import csv
import json
import os
import shutil
import numpy as np
from filelock import FileLock
from hypothesis_scope_multiseed import study as matched
from knowledge_rank_v9.engine import candidates
from . import VARIANT,MODEL,EFFORT,SEEDS,ARMS,PARAMETERS as P,policy,prompts,transport

ROOT=matched.ROOT
BASE=ROOT/'experiments/runs'/VARIANT
REPORT=ROOT/'experiments/reports'/VARIANT
SHORT=matched.SHORT
TASKS=matched.TASKS
JOBS=tuple((task,seed,arm) for seed in SEEDS for task in TASKS for arm in ARMS)
read,write,sha,digest=matched.read,matched.write,matched.sha,matched.digest

def path_for(task,seed,arm):return BASE/'runs'/f's{seed}'/(SHORT[task]+'_'+arm)

def prior_for(task):
    saved=matched.initial_prior(task)
    return [dict(id=h['id'],claim=h['claim'],direction=h['direction'],scope=deepcopy(h['scope']),
                 forecast=deepcopy(h['forecast']),falsification=h['falsification'])
            for h in saved['raw']['hypotheses']]

def check(protected=False):
    p=read(BASE/'protocol.json')
    assert p['protocol_sha256']==digest({a:b for a,b in p.items() if a!='protocol_sha256'})
    assert p['parameters']==P and p['model']==MODEL and p['effort']==EFFORT
    for group in ('sources_sha256','inputs_sha256')+(('protected_sha256',) if protected else ()):
        for path,h in p[group].items():assert sha(ROOT/path)==h,path
    return p

def prepare():
    if (BASE/'protocol.json').exists():return check(True)
    assert not (BASE/'runs').exists() and not (BASE/'requests').exists()
    qa=read(BASE/'preflight.json');assert qa['status']=='PASS_GROUNDED_BASELINES_PREFLIGHT'
    old=matched.check(True)
    sealed=read(matched.BASE/'completed_artifacts_sha256.json')['files']
    for path,h in sealed.items():assert sha(ROOT/path)==h,path
    controls=[]
    with (matched.REPORT/'per_seed_results.csv').open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            assert sha(ROOT/row['state_path'])==row['state_sha256']
            controls.append(dict(task=row['task'],seed=int(row['seed']),method=row['method'],
                                 path=row['state_path'],sha256=row['state_sha256']))
    files=list(Path(__file__).parent.glob('*.py'))+[ROOT/'code/experiment_tools/run_grounded_baselines.py',
        ROOT/'code/tools/grounded_baselines_dispatch_server.py',ROOT/'docs'/f'{VARIANT}.txt']
    sources=dict(old['sources_sha256'])
    for path in files:sources[path.relative_to(ROOT).as_posix()]=sha(path)
    for name in ('code/experiment_tools/knowledge_rank_v9/engine.py','code/experiment_tools/knowledge_rank_v9/llm.py',
                 'code/chemist_system/chem/benchmarks/pointwise_hte_engine.py',
                 'code/chemist_system/chem/benchmarks/reaction_transfer_engine.py',
                 'code/experiment_tools/chemical_hypothesis_bo/experts.py'):
        sources[name]=sha(ROOT/name)
    for path in (ROOT/'code/chemist_system/third_party/suzuki_official/lgbo').glob('*.py'):
        sources[path.relative_to(ROOT).as_posix()]=sha(path)
    for name in ('code/chemist_system/third_party/reasoning_bo_official/src/bo/reasoner/base.py',
                 'code/chemist_system/third_party/reasoning_bo_official/src/prompts/test_prompts.json'):
        sources[name]=sha(ROOT/name)
    protected=dict(old['protected_sha256']);protected.update(old['sources_sha256']);protected.update(sealed)
    protected[(matched.BASE/'completed_artifacts_sha256.json').relative_to(ROOT).as_posix()]=sha(matched.BASE/'completed_artifacts_sha256.json')
    p=dict(variant=VARIANT,created_UTC=datetime.now(timezone.utc).isoformat(),parameters=P,
        model=MODEL,effort=EFFORT,backend='fresh independent collaboration subagents',fork_turns='none',
        seeds=list(SEEDS),arms=list(ARMS),tasks=list(TASKS),jobs=[list(j) for j in JOBS],
        sources_sha256=sources,inputs_sha256=old['inputs_sha256'],protected_sha256=protected,
        controls=controls,current_protocol_sha256=old['protocol_sha256'],preflight_sha256=sha(BASE/'preflight.json'),
        shared_first={t:matched.frozen.locked_first(t) for t in TASKS},
        shared_blinded_priors={t:prior_for(t) for t in TASKS},
        maximum_experimental_queries=1280,maximum_new_logical_LLM_requests=1216,
        maximum_schema_attempts=2432,operator_compiles_in_loop=0,persistent_CPU_workers=8,
        BLAS_threads_per_worker=1,
        adaptations=['Shared paid first input and blinded initial hypotheses, not fresh independent warmstarts',
            'Frozen ColdGP (Mixed GP on categorical inputs), all-query feature-tie seeds0..3, sequential q=1 over frozen pools',
            'LGBO official regional tilt with finite-pool uniform quadrature cap512; categorical membership rather than ordinal intervals',
            'LGBO cached posterior covariance mapped through scalar outcome stdvs squared, as previously audited',
            'LGBO mean lift followed by unchanged native MC512 qLogEI; no KF, hard scope restriction or opportunity gate',
            'Reasoning BO official Insight template and stable confidence extractor,5-to1 rather than paper5-to3',
            'RBO receives shared blinded priors as retrieved text; own Insight history retained, no evolving KG/vector database',
            'All LLMs receive verified enriched background; GP ranks/numerical scores hidden, RBO sees neutral top5 membership',
            'Initial forecasts are scientific priors, never GP training labels; no other-arm outcomes shown'],
        primary_metric='mean of all20 post-query cumulative maxima',
        inference='Matched decision-mechanism comparison conditional on a shared fixed initial point and knowledge; exploratory, not a reproduction of published protocols')
    p['protocol_sha256']=digest(p);write(BASE/'protocol.json',p)
    for name in sources:
        dst=BASE/'source_snapshot'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dst)
    for task in TASKS:
        src=matched.BASE/'cache'/SHORT[task];dst=BASE/'cache'/SHORT[task];dst.mkdir(parents=True,exist_ok=True)
        for path in src.iterdir():
            if path.suffix not in ('.pt','.npz','.json'):continue
            target=dst/path.name
            if not target.exists():shutil.copyfile(path,target)
            else:assert sha(path)==sha(target)
    return check(True)

def initialize():
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    matched.initialize()
    matched.frozen.native._legacy_import('pointwise_hte_engine',ROOT)
    policy.warm(ROOT)

def apply(pool,folder,state,commit):
    assert commit['prestate_sha256']==digest(state)
    assert commit['commit_sha256']==digest({a:b for a,b in commit.items() if a!='commit_sha256'})
    q=commit['query'];d=commit['decision'];assert q==len(state['history'])+1
    assert d['selected_id'] not in {h['candidate_id'] for h in state['history']}
    observed=matched.frozen.native.read_observation(ROOT,pool,d['selected_id'])
    write(folder/'rounds'/f'{q:02d}_feedback.json',dict(commit_sha256=commit['commit_sha256'],observation=observed))
    state['history'].append(observed)
    state['decisions'].append(dict(query=q,**observed,decision=d,best=max(h['raw_value'] for h in state['history'])))
    request=commit.get('LLM')
    if request is not None:
        state['calls'].append(dict(query=q,request=request['request']))
        state['preferences' if state['arm']=='lgbo' else 'insights'].append(deepcopy(request['raw']))
    state['status']='COMPLETE' if q==20 else 'RUNNING'
    write(folder/'state.json',state,replace=True)
    print(json.dumps(dict(task=SHORT[pool.task],seed=state['seed'],arm=state['arm'],query=q,
        value=observed['raw_value'],best=state['decisions'][-1]['best'])),flush=True)

def run_one(task,seed,arm):
    assert (task,seed,arm) in JOBS
    p=check();pool=matched.frozen.pool_for(task);folder=path_for(task,seed,arm);folder.mkdir(parents=True,exist_ok=True)
    with FileLock(str(folder/'.worker.lock'),timeout=1):
        sf=folder/'state.json'
        state=read(sf) if sf.exists() else dict(task=task,seed=seed,arm=arm,protocol_sha256=p['protocol_sha256'],
            initial_ids=[],initial=0,queries=20,history=[],decisions=[],preferences=[],insights=[],calls=[],status='RUNNING')
        assert state['protocol_sha256']==p['protocol_sha256']
        write(folder/'initial_commit.json',dict(task=task,seed=seed,arm=arm,initial_ids=[],
            protocol_sha256=p['protocol_sha256'],shared_first=p['shared_first'][task],
            shared_blinded_prior_sha256=digest(p['shared_blinded_priors'][task])))
        gp=None
        while len(state['history'])<20:
            q=len(state['history'])+1;cp=folder/'rounds'/f'{q:02d}_commit.json'
            if cp.exists():commit=read(cp)
            else:
                metadata={};request=None
                if q==1:
                    j=pool.ix[p['shared_first'][task]['candidate_id']]
                    d=dict(selected_id=pool.ids[j],conditions=deepcopy(pool.points[j]),
                        reason='shared_frozen_independent_LLM_first',GP_fitted=False,qLogEI_defined=False,
                        first_source=p['shared_first'][task])
                else:
                    if gp is None:gp=matched.SeededGP(pool,BASE/'cache'/SHORT[task],tie_seed=seed)
                    ctx=gp.context(state['history'],q);stats=ctx['stats'];prior=p['shared_blinded_priors'][task]
                    if arm=='lgbo':
                        public=prompts.lgbo(pool,state,prior)
                        request=transport.request(f'glr_s{seed}_{SHORT[task]}_{arm}',f'query{q:02d}',prompts.LGBO_SYSTEM,public,
                            lambda raw:policy.validate_lgbo(raw,pool,state['history'],q,seed))
                        score,shift,meta=policy.scores(gp,ctx,request['plan']);j=policy.select(ctx,score)
                        method_fields=dict(lgbo=meta,preference=deepcopy(request['raw']),
                            shifted_mean_raw=float((stats['mu_y'][j]+shift[j])*pool.scale))
                    else:
                        ordered,display=candidates(pool,ctx,q);ids=[pool.ids[int(j)] for j in display]
                        public=prompts.rbo(ROOT,pool,state,prior,ids)
                        request=transport.request(f'glr_s{seed}_{SHORT[task]}_{arm}',f'query{q:02d}',prompts.RBO_SYSTEM,public,
                            lambda raw:policy.validate_rbo(ROOT,raw,ids))
                        j=pool.ix[request['plan']['selected_id']];score=ctx['base_logei'].copy();shift=np.zeros(len(pool.ids))
                        method_fields=dict(ordered_ids=[pool.ids[int(i)] for i in ordered],display_ids=ids,
                            official_confidence_selection=True)
                    arrays={k:stats[k] for k in ('mu_y','sd_y','sd_predictive_y')}
                    arrays.update({k:ctx[k] for k in ('base_logei','available','tie_priority')})
                    arrays.update(final_logei=score,mean_shift_y=shift)
                    ap=folder/'rounds'/f'{q:02d}_GP.npz'
                    if not ap.exists():
                        with ap.open('xb') as f:np.savez_compressed(f,**arrays)
                    d=dict(selected_id=pool.ids[j],conditions=deepcopy(pool.points[j]),
                        reason='official_region_mean_tilt_qLogEI' if arm=='lgbo' else 'official_Insight_confidence_top5_to1',
                        GP_fitted=True,qLogEI_defined=True,BO_reference_id=pool.ids[ctx['bo_index']],
                        changed_from_BO=bool(j!=ctx['bo_index']),base_logEI=float(ctx['base_logei'][j]),
                        final_logEI=float(score[j]),native_opportunity_gap=float(ctx['base_logei'][ctx['bo_index']]-ctx['base_logei'][j]),
                        GP_mean_raw=float(stats['mu_y'][j]*pool.scale),GP_SD_raw=float(stats['sd_y'][j]*pool.scale),
                        GP_predictive_SD_raw=float(stats['sd_predictive_y'][j]*pool.scale),
                        GP_fit_key=stats['fit_key'],GP_stage=ctx['stage'],**method_fields)
                    metadata=dict(GP_arrays_sha256=sha(ap),GP_parameters=stats['fit_parameters'],acquisition=ctx['acquisition_meta'],
                                  output_scale_convention=stats.get('cold_scale_convention'))
                commit=dict(query=q,prestate_sha256=digest(state),decision=d,LLM=request,**metadata)
                commit['commit_sha256']=digest(commit);write(cp,commit)
            apply(pool,folder,state,commit)
            if gp is not None:
                write(folder/'runtime_counters.json',gp.counters,replace=True);gp.release()
        write(folder/'completion.json',dict(status='COMPLETE',queries=20,LLM_calls=len(state['calls'])))
    return dict(task=SHORT[task],seed=seed,arm=arm,status='COMPLETE')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('preflight','prepare','run','pending','status'))
    parser.add_argument('--workers',type=int,default=8);parser.add_argument('--arms',nargs='*',choices=ARMS,default=list(ARMS))
    args=parser.parse_args()
    if args.action=='preflight':
        from .validation import preflight
        result=preflight()
    elif args.action=='prepare':result=dict(protocol_sha256=prepare()['protocol_sha256'])
    elif args.action=='pending':result=transport.pending()
    elif args.action=='status':
        rows=[dict(task=SHORT[t],seed=s,arm=a,observed=len(read(path_for(t,s,a)/'state.json')['history']))
              for t,s,a in JOBS if (path_for(t,s,a)/'state.json').exists()]
        result=dict(complete=sum(r['observed']==20 for r in rows),planned=len(JOBS),queries=sum(r['observed'] for r in rows),jobs=rows)
    else:
        check(True)
        with ProcessPoolExecutor(max_workers=min(8,args.workers),initializer=initialize) as executor:
            for future in as_completed([executor.submit(run_one,*job) for job in JOBS if job[2] in args.arms]):
                print(json.dumps(future.result()),flush=True)
        result=dict(status='COMPLETE',trajectories=len([j for j in JOBS if j[2] in args.arms]))
    print(json.dumps(result,ensure_ascii=True))

if __name__=='__main__':main()
