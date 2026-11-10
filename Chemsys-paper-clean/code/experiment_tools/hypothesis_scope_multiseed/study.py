"""Seed/I/O adapter; frozen single-seed scientific loop is called unchanged."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
from unittest.mock import patch
import argparse
import builtins
import json
import os
import shutil
import numpy as np
from filelock import FileLock
from hypothesis_scope_loop import study as frozen,policy,knowledge,prompts,PARAMETERS as P
from . import VARIANT,SEEDS,MODEL,EFFORT,transport
ROOT=frozen.ROOT;BASE=ROOT/'experiments/runs'/VARIANT;REPORT=ROOT/'experiments/reports'/VARIANT
SHORT=frozen.SHORT;TASKS=tuple(frozen.native.TASKS);ARMS=('bo','first','hyp')
JOBS=tuple((task,seed,arm) for arm in ARMS for seed in SEEDS for task in TASKS)
read,write,sha,digest=frozen.read,frozen.write,frozen.sha,frozen.digest
NativeColdGP=frozen.ColdGP
original_check=frozen.check


def folder_for(task,seed,arm):return BASE/'runs'/f's{seed}'/(SHORT[task]+'_'+arm)


def initial_state(task,seed,arm,p):
    value=dict(task=task,protocol_sha256=p['protocol_sha256'],initial_ids=[],initial=0,queries=20,
        seed=seed,tie_seed=seed,numerical_seed=0,arm=arm,history=[],decisions=[],status='RUNNING')
    if arm=='hyp':value.update(notes=[],archive=[],calls=[],permission_events=[])
    return value


def retie(context,pool,query,seed):
    out=dict(context);out['tie_priority']=frozen.native.semantic_priorities(pool,query,seed)
    out['bo_index']=policy.argmax(out,out['available'])
    if out.get('epsilon') is not None:
        out['eligible']=out['available']&(out['base_logei'][out['bo_index']]-out['base_logei']<=out['epsilon'])
    out['tie_seed']=seed;out['numerical_seed']=0
    return out


class SeededGP(NativeColdGP):
    def __init__(self,pool,path,*,tie_seed):
        super().__init__(pool,path,seed=0);self.tie_seed=tie_seed
    def context(self,history,query):return retie(super().context(history,query),self.pool,query,self.tie_seed)


def initial_prior(task):
    old=read(frozen.folder_for(task)/'state.json');call=old['calls'][0]
    path=ROOT/call['request'];saved=read(path.parent.parent/'validated.json')
    receipt=read(path.parent/'receipt.json')
    assert receipt['raw_sha256']==sha(path.parent/'response.raw.json') and saved['raw']==read(path.parent/'response.raw.json')
    return saved


def check(historical=False):
    p=read(BASE/'protocol.json');assert p['protocol_sha256']==digest({a:b for a,b in p.items() if a!='protocol_sha256'})
    assert p['parameters']==P and p['seeds']==list(SEEDS) and p['arms']==list(ARMS)
    for group in ('sources_sha256','inputs_sha256')+(('protected_sha256',) if historical else ()):
        for name,h in p[group].items():
            if sha(ROOT/name)!=h:raise ValueError('Frozen artifact changed: '+name)
    return p


def copy_numeric_cache():
    copied=0
    for task in TASKS:
        source=frozen.BASE/'cache'/SHORT[task];target=BASE/'cache'/SHORT[task];target.mkdir(parents=True,exist_ok=True)
        for path in source.iterdir():
            if path.suffix not in ('.pt','.npz','.json'):continue
            dest=target/path.name
            if dest.exists():assert sha(dest)==sha(path)
            else:shutil.copyfile(path,dest);copied+=1
    return copied


def prepare():
    if (BASE/'protocol.json').exists():return check(True)
    assert not (BASE/'runs').exists() and not (BASE/'requests').exists()
    old=original_check(True);qa=read(BASE/'preflight.json');assert qa['status']=='PASS_MULTISEED_ADAPTER_PREFLIGHT'
    frozen_manifest=read(frozen.BASE/'completed_artifacts_sha256.json')['files']
    for name,h in frozen_manifest.items():assert sha(ROOT/name)==h,name
    protected=dict(old['protected_sha256']);protected.update(old['sources_sha256']);protected.update(frozen_manifest)
    protected[(frozen.BASE/'completed_artifacts_sha256.json').relative_to(ROOT).as_posix()]=sha(frozen.BASE/'completed_artifacts_sha256.json')
    sources={path.relative_to(ROOT).as_posix():sha(path) for path in list(Path(__file__).parent.glob('*.py'))+
        [ROOT/'code/experiment_tools/run_hypothesis_scope_multiseed.py',ROOT/'docs/hypothesis_scope_multiseed_20261105.txt']}
    sources.update(old['sources_sha256'])
    priors={task:dict(request=initial_prior(task)['request'],raw_sha256=digest(initial_prior(task)['raw']),
        first_input=frozen.locked_first(task)) for task in TASKS}
    p=dict(variant=VARIANT,registered_UTC=datetime.now(timezone.utc).isoformat(),parameters=deepcopy(P),
        seeds=list(SEEDS),seed0_reused=True,arms=list(ARMS),tasks=list(TASKS),jobs=[list(j) for j in JOBS],
        initial=0,queries=20,model=MODEL,effort=EFFORT,fork_turns='none',numerical_seed=0,
        sources_sha256=sources,inputs_sha256=old['inputs_sha256'],protected_sha256=protected,
        frozen_protocol_sha256=old['protocol_sha256'],seed0_controls=old['controls'],initial_priors=priors,
        maximum_new_trajectories=72,maximum_new_experimental_queries=1440,
        maximum_new_logical_LLM_requests=72,maximum_new_LLM_attempts=144,
        tie_semantics='PCG64 feature-keyed stream [seed,query] changes at every query including the pure-BO zero-data first choice; exact maximum only',
        GP_semantics='Frozen native fit and MC512 streams remain numerical seed0; only tie order and data induced by selection change',
        knowledge_semantics='Same frozen first point and BLINDED initial hypotheses for all seeds; initial prior is explicitly reused, not a fresh response',
        refresh_semantics='Fresh independent subagent for each own-history refresh; caches separated by task/seed; no cross-seed response replay',
        transport_adaptation='Seal only after actual agent completion is acknowledged; scientific prompt/schema unchanged',
        old_multiseed_controls_not_reused='Older cold_s123 changes query1 only, whereas this study changes all-query tie streams',
        aggregation='Per-task seed0..3 mean/sample SD, paired seed differences and win/tie/loss; also seeds1..3 separately; no averaging raw task units',
        inference_limit='Conditional on fixed LLM first point/initial hypotheses; later LLM-response and tie-stream variability are joint, not separately identified',
        preflight_sha256=sha(BASE/'preflight.json'),precompiled_python=True,persistent_CPU_workers=8,BLAS_threads_per_worker=1)
    p['protocol_sha256']=digest(p);write(BASE/'protocol.json',p)
    for name in sources:
        dest=BASE/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    copy_numeric_cache();REPORT.mkdir(parents=True,exist_ok=True)
    return check(True)


def initialize():frozen.initialize()


def apply_control(pool,folder,state,commit):
    assert commit['prestate_sha256']==digest(state) and commit['commit_sha256']==digest({a:b for a,b in commit.items() if a!='commit_sha256'})
    d=commit['decision'];q=commit['query'];assert q==len(state['history'])+1 and d['selected_id'] not in {h['candidate_id'] for h in state['history']}
    observed=frozen.native.read_observation(ROOT,pool,d['selected_id'])
    write(folder/'rounds'/f'{q:02d}_feedback.json',dict(commit_sha256=commit['commit_sha256'],observation=observed))
    state['history'].append(observed);state['decisions'].append(dict(query=q,**observed,decision=d,best=max(h['raw_value'] for h in state['history'])))
    state['status']='COMPLETE' if q==20 else 'RUNNING';write(folder/'state.json',state,replace=True)
    print(json.dumps(dict(task=SHORT[pool.task],seed=state['seed'],arm=state['arm'],query=q,value=observed['raw_value'],best=state['decisions'][-1]['best'])),flush=True)


def run_control(task,seed,arm,p):
    pool=frozen.pool_for(task);folder=folder_for(task,seed,arm);folder.mkdir(parents=True,exist_ok=True)
    with FileLock(str(folder/'.worker.lock'),timeout=1):
        sf=folder/'state.json';state=read(sf) if sf.exists() else initial_state(task,seed,arm,p)
        assert state['protocol_sha256']==p['protocol_sha256'] and state['seed']==seed
        write(folder/'initial_commit.json',dict(task=task,seed=seed,arm=arm,protocol_sha256=p['protocol_sha256'],initial_ids=[],history=[]))
        gp=None
        while len(state['history'])<20:
            q=len(state['history'])+1;cp=folder/'rounds'/f'{q:02d}_commit.json'
            if cp.exists():commit=read(cp)
            else:
                metadata={}
                if q==1:
                    j=int(np.argmin(frozen.native.semantic_priorities(pool,1,seed))) if arm=='bo' else pool.ix[frozen.locked_first(task)['candidate_id']]
                    d=dict(query=q,selected_id=pool.ids[j],conditions=deepcopy(pool.points[j]),
                        reason='zero_data_feature_prior' if arm=='bo' else 'reused_frozen_independent_LLM_first',GP_fitted=False,qLogEI_defined=False,
                        first_source=None if arm=='bo' else p['initial_priors'][task]['first_input'])
                else:
                    if gp is None:gp=SeededGP(pool,BASE/'cache'/SHORT[task],tie_seed=seed)
                    ctx=gp.context(state['history'],q);j=ctx['bo_index'];stats=ctx['stats'];ap=folder/'rounds'/f'{q:02d}_GP.npz'
                    if not ap.exists():
                        with ap.open('xb') as f:np.savez_compressed(f,**{a:stats[a] for a in ('mu_y','sd_y','sd_predictive_y')},
                            **{a:ctx[a] for a in ('base_logei','available','tie_priority')})
                    d=dict(query=q,selected_id=pool.ids[j],conditions=deepcopy(pool.points[j]),reason='native_full_pool_qLogEI',
                        GP_fitted=True,qLogEI_defined=True,base_logEI=float(ctx['base_logei'][j]),GP_stage=ctx['stage'],
                        GP_mean_raw=float(stats['mu_y'][j]*pool.scale),GP_SD_raw=float(stats['sd_y'][j]*pool.scale),
                        fit_key=stats['fit_key'],operator_compiles_in_loop=gp.counters['operator_compiles_in_loop'])
                    metadata=dict(GP_arrays_sha256=sha(ap),GP_parameters=stats['fit_parameters'],acquisition=ctx['acquisition_meta'],output_scale_convention=stats.get('cold_scale_convention'))
                commit=dict(query=q,prestate_sha256=digest(state),decision=d,**metadata);commit['commit_sha256']=digest(commit);write(cp,commit)
            apply_control(pool,folder,state,commit)
            if gp is not None:gp.release()
    return dict(task=SHORT[task],seed=seed,arm=arm,status='COMPLETE')


def run_hyp(task,seed,p):
    folder=folder_for(task,seed,'hyp');folder.mkdir(parents=True,exist_ok=True)
    with FileLock(str(folder/'.adapter.lock'),timeout=1):
        sf=folder/'state.json'
        if not sf.exists():write(sf,initial_state(task,seed,'hyp',p))
        prior=initial_prior(task)
        write(folder/'initial_knowledge_reuse.json',dict(request=prior['request'],raw_sha256=digest(prior['raw']),
            semantics='Fixed blinded initial knowledge, explicitly reused; no new initial LLM request'))
        def request(short,stage,system,public,validator):
            assert short==SHORT[task]
            if stage=='query01':
                assert public['own_observations']==[] and system==prompts.SYSTEM
                assert validator(prior['raw'])==prior['plan'];return deepcopy(prior)
            return transport.request(f's{seed}_{short}',stage,system,public,validator)
        def gp(pool,path,seed=0):
            assert seed==0 and Path(path)==BASE/'cache'/SHORT[task]
            return SeededGP(pool,path,tie_seed=seed_for_job)
        seed_for_job=seed
        def scoped_print(value,**kwargs):
            try:value=json.dumps(dict(json.loads(value),seed=seed_for_job,arm='hyp'))
            except (ValueError,TypeError):pass
            return builtins.print(value,**kwargs)
        with ExitStack() as stack:
            for obj,name,value in ((frozen,'check',lambda *a,**kw:p),(frozen,'BASE',BASE),
                (frozen,'folder_for',lambda t:folder_for(t,seed_for_job,'hyp')),(frozen,'ColdGP',gp),
                (frozen.transport,'request',request)):
                stack.enter_context(patch.object(obj,name,value))
            stack.enter_context(patch.object(frozen,'print',scoped_print,create=True))
            result=frozen.run_one(task)
        return dict(result,seed=seed,arm='hyp')


def run_one(task,seed,arm):
    assert (task,seed,arm) in JOBS;p=check()
    return run_hyp(task,seed,p) if arm=='hyp' else run_control(task,seed,arm,p)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','preflight','run','pending','dispatch'))
    parser.add_argument('--workers',type=int,default=8);parser.add_argument('--spawned',nargs='*',default=[]);parser.add_argument('--completed',nargs='*',default=[])
    parser.add_argument('--arms',nargs='*',choices=ARMS,default=list(ARMS));args=parser.parse_args()
    if args.action=='prepare':value=dict(protocol_sha256=prepare()['protocol_sha256'])
    elif args.action=='preflight':
        from .validation import preflight
        value=preflight()
    elif args.action=='run':
        check(True)
        with ProcessPoolExecutor(max_workers=min(8,args.workers),initializer=initialize) as executor:
            jobs=[j for j in JOBS if j[2] in args.arms]
            for future in as_completed([executor.submit(run_one,*j) for j in jobs]):print(json.dumps(future.result()),flush=True)
        value=dict(status='COMPLETE',trajectories=len(jobs))
    else:
        for item in transport.pending():
            name=item['invocation']['task_name']
            if name in args.spawned and not item['assigned']:transport.assign(item['request'],dict(task_name='/root/'+name))
            if name in args.completed:transport.complete(item['request']);transport.seal(item['request'])
        value=[dict(name=r['invocation']['task_name'],assigned=r['assigned'],ready=r['output_exists'],export=r['export']) for r in transport.pending()]
    print(json.dumps(value,ensure_ascii=True))


if __name__=='__main__':main()
