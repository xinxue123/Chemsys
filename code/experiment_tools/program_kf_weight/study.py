"""Pre-registered fixed-prefix diagnostics followed by four independent traces."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
import argparse
import csv
import json
import os
import shutil

import numpy as np
from cold_start_v13 import study as frozen, policy as notes_policy, control
from cold_start_v13.gp import ColdGP
from chem.benchmarks import knowledge_gp
from reasoning_memory_v12.llm import compiled_extractor, validate_insight
from reasoning_memory_v13 import calibration
from knowledge_rank_v9 import engine
from v13_origin_ablation import queue
from . import VARIANT, MODEL, EFFORT, CELLS, ARMS, QUERIES, policy, prompts

ROOT=Path(__file__).resolve().parents[3]
DESIGN='docs/program_kf_weight_20261103.txt'
PRIORITIES=knowledge_gp.semantic_priorities
SHORT={'Perera_Suzuki':'perera','HPLC':'hplc'}
_CHECKED=None


def base():return ROOT/'experiments/runs'/VARIANT
def rel(path):return Path(path).resolve().relative_to(ROOT).as_posix()
def read(path):return queue.read(path)
def write(path,value):return queue.write_once(path,value)


def jobs():
    return [dict(id=f'prefix_{SHORT[t]}{s}',cell=f'{SHORT[t]}{s}',task=t,seed=s,arm='diagnostic',phase='prefix')
            for t,s in CELLS]+[
        dict(id=f'trace_{SHORT[t]}{s}_{a}',cell=f'{SHORT[t]}{s}',task=t,seed=s,arm=a,phase='traces')
        for t,s in CELLS for a in ARMS]


def prepare():
    if (base()/'protocol.json').exists():return check(True)
    prior=read(ROOT/'experiments/runs/v13_origin_ablation_20261103/targeted/protocol.json')
    origin=read(ROOT/'experiments/runs/v13_origin_ablation_20261103/protocol.json')
    sources=dict(prior['sources_sha256']);inputs=dict(origin['inputs_sha256'])
    protected=dict(origin['protected_sha256'])
    for path in (Path(__file__).parent).glob('*.py'):sources[rel(path)]=queue.file_sha(path)
    for name in (DESIGN,'code/experiment_tools/run_program_kf_weight.py','code/tools/program_kf_dispatch.py'):
        sources[name]=queue.file_sha(ROOT/name)
    references={}
    with (ROOT/'paper/v13_late/data/metrics.csv').open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            if row['regime']!='cold':continue
            path=ROOT/row['state_path']
            if queue.file_sha(path)!=row['state_sha256']:raise ValueError('Historical state changed')
            protected[rel(path)]=queue.file_sha(path)
            if (row['task'],int(row['seed'])) in {(SHORT[t],s) for t,s in CELLS}:
                references[f"{row['task']}{row['seed']}_{row['method']}"]=dict(state_path=rel(path),state_sha256=queue.file_sha(path))
                if row['method']=='v13':
                    for p in (path.parent/'rounds').glob('*'):
                        if p.is_file():protected[rel(p)]=queue.file_sha(p)
    for milestone in ('v13_targeted_ablation_20261103',):
        folder=ROOT/'experiments/milestones'/milestone
        for name in ('manifest.json','receipt.json','evidence.zip'):
            protected[rel(folder/name)]=queue.file_sha(folder/name)
    for path in (ROOT/'experiments/runs/v13_origin_ablation_20261103/targeted/runs').glob('*/state.json'):
        protected[rel(path)]=queue.file_sha(path)
    p=dict(variant=VARIANT,registered_utc=datetime.now(timezone.utc).isoformat(),jobs=jobs(),
        model=MODEL,effort=EFFORT,backend='isolated_fresh_subagents_exact_input_coupling',fork_turns='none',
        initial=0,queries=20,new_trajectories=4,new_experiment_queries=80,
        historical_prefixes=40,historical_prefix_experiment_queries=0,numerical_seed=0,
        parameters=policy.PARAMETERS,note_parameters=notes_policy.PARAMETERS,
        sources_sha256=sources,inputs_sha256=inputs,protected_sha256=protected,references=references,
        design=(ROOT/DESIGN).read_text(encoding='utf-8'),new_outcomes_at_registration=0)
    p['protocol_sha256']=control.digest(p);write(base()/'protocol.json',p)
    for name,h in sources.items():
        dst=base()/'frozen_source'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dst)
        if queue.file_sha(dst)!=h:raise ValueError('Scientific source copy differs')
    for task,seed in CELLS:
        src=ROOT/'experiments/runs/v13_origin_ablation_20261103/targeted/cache'/task
        dst=base()/'cache'/task;dst.mkdir(parents=True,exist_ok=True)
        for path in src.glob('*'):
            if path.suffix in ('.pt','.npz'):
                target=dst/path.name
                if not target.exists():shutil.copy2(path,target)
                if queue.file_sha(path)!=queue.file_sha(target):raise ValueError('Copied numerical cache differs')
    return check(True)


def check(protected=False):
    p=read(base()/'protocol.json')
    if p['protocol_sha256']!=control.digest({k:v for k,v in p.items() if k!='protocol_sha256'}):
        raise ValueError('Registered protocol changed')
    for group in ('sources_sha256','inputs_sha256')+ (('protected_sha256',) if protected else ()):
        for name,h in p[group].items():
            if queue.file_sha(ROOT/name)!=h:raise ValueError('Frozen evidence changed: '+name)
    return p


@contextmanager
def transport_scope(job=None):
    previous=queue.CURRENT
    with patch.object(queue,'base',return_value=base().resolve()),patch.object(queue,'VARIANT',VARIANT):
        queue.CURRENT=job
        try:yield
        finally:queue.CURRENT=previous


class GP(ColdGP):
    def __init__(self,pool,directory,start_seed):
        super().__init__(pool,directory,seed=0);self.start_seed=start_seed
    def context(self,history,q):
        def prior(pool,q,seed=0):
            if seed!=0:raise ValueError('Numerical seed must be zero')
            return PRIORITIES(pool,q,self.start_seed if q==1 else 0)
        with patch.object(knowledge_gp,'semantic_priorities',side_effect=prior):
            return super().context(history,q)


def saved_context(folder,q):
    fit=read(folder/'rounds'/f'{q:02}_fit.json')
    with np.load(folder/'rounds'/f'{q:02}_GP.npz',allow_pickle=False) as z:
        ctx=dict(stats={k:z[k].copy() for k in ('mu_y','sd_y','sd_predictive_y')},
            **{k:z[k].copy() for k in ('base_logei','available','tie_priority')})
    ctx.update(stage=fit['stage'],fitted_GP=fit['fitted_GP'])
    return ctx


def reference(p,task,seed,method='v13'):
    return ROOT/p['references'][f'{SHORT[task]}{seed}_{method}']['state_path']


def held_state(old,q):
    return dict(task=old['task'],history=deepcopy(old['history'][:q-1]),
        insights=deepcopy(old['insights'][:q-1]),decisions=deepcopy(old['decisions'][:q-1]),
        notes=[],review_queries=[x for x in old['review_queries'] if x<q],
        edit_calls=[x for x in old['edit_calls'] if x<q])


def prefix_job(job,p):
    path=reference(p,job['task'],job['seed']);old=read(path);folder=base()/'prefix'/job['cell']
    sf=folder/'state.json';state=read(sf) if sf.exists() else dict(query=0,ranker_KF=calibration.initial(),status='READY')
    pool=frozen.pool_for(ROOT,job['task']);compiled_extractor(str(ROOT))
    with transport_scope(job):
        for q in range(state['query']+1,21):
            c=read(path.parent/'rounds'/f'{q:02}_commit.json');ctx=saved_context(path.parent,q)
            if engine.gp_hash(ctx)!=c['GP_sha256']:raise ValueError('Held GP differs')
            hist=held_state(old,q);hist['ranker_KF']=state['ranker_KF']
            notes=c['edited_notes'];ids=c['display_ids']
            if [pool.ids[int(j)] for j in engine.candidates(pool,ctx,q)[1]]!=ids:raise ValueError('Held menu differs')
            cp=folder/'rounds'/f'{q:02}_commit.json'
            if cp.exists():commit=read(cp)
            else:
                public=prompts.reasoning_public(ROOT,pool,hist,notes,ids,q)
                response=queue.request(ROOT,job['id'],'choose',q,prompts.REASONING_SYSTEM,public,
                    lambda raw:policy.plan(ROOT,pool,notes,hist,ctx,ids,q,raw,'fixed'),
                    selected_validator=lambda raw:validate_insight(ROOT,raw['insight'],ids))
                fixed=response['plan'];dynamic=policy.plan(ROOT,pool,notes,hist,ctx,ids,q,response['raw'],'dynamic')
                commit=dict(query=q,source_commit_path=rel(path.parent/'rounds'/f'{q:02}_commit.json'),
                    source_commit_sha256=queue.file_sha(path.parent/'rounds'/f'{q:02}_commit.json'),
                    GP_sha256=engine.gp_hash(ctx),display_ids=ids,ranker_before=deepcopy(state['ranker_KF']),
                    reasoning=response,fixed=fixed,dynamic=dynamic,
                    held_behavior_id=old['history'][q-1]['candidate_id'],new_experiment_queries=0)
                commit['commit_sha256']=control.digest(commit);write(cp,commit)
            if commit['ranker_before']!=state['ranker_KF']:raise ValueError('Held-prefix KF state changed')
            observed=deepcopy(old['history'][q-1])
            if q==1:event=dict(kind='no_incumbent_no_update');after=state['ranker_KF']
            else:
                cid=observed['candidate_id'];p_llm=commit['fixed']['probabilities'][cid]
                p_gp=engine.gp_probability(ctx,pool.ix[cid],max(h['y'] for h in hist['history']))
                after,event=calibration.update(state['ranker_KF'],p_llm,p_gp,observed['y']>max(h['y'] for h in hist['history']))
            write(folder/'rounds'/f'{q:02}_feedback.json',dict(observed_archived_point=observed,calibration=event,after=after,
                not_a_new_experiment=True,alternative_selected_candidates_not_evaluated=True))
            state=dict(query=q,ranker_KF=after,status='COMPLETE' if q==20 else 'READY')
            frozen.warm.write(sf,state)
            print(json.dumps(dict(phase='prefix',cell=job['cell'],query=q,
                fixed=commit['fixed']['selected_id'],dynamic=commit['dynamic']['selected_id'])),flush=True)
    return dict(id=job['id'],status='COMPLETE',historical_prefixes=20,new_queries=0)


def trace_job(job,p):
    pool=frozen.pool_for(ROOT,job['task']);folder=base()/'runs'/job['id'];sf=folder/'state.json'
    initial=dict(task=job['task'],arm=job['arm'],starting_seed=job['seed'],initial_ids=[],initial=0,queries=20,
                 protocol_sha256=p['protocol_sha256'],input_sha256=queue.file_sha(pool.source_path))
    initial['commit_sha256']=control.digest(initial);write(folder/'initial_commit.json',initial)
    if sf.exists():state=read(sf)
    else:
        state=frozen.initial_state(job['task'],job['arm'],p,initial)
        state.update(seed=job['seed'],starting_seed=job['seed'],numerical_seed=0,ranker_KF=calibration.initial())
        frozen.warm.write(sf,state)
    if state['status']=='COMPLETE':return dict(id=job['id'],status='COMPLETE',queries=20)
    gp=GP(pool,base()/'cache'/job['task'],job['seed']);compiled_extractor(str(ROOT))
    try:
        with transport_scope(job):
            while len(state['history'])<20:
                q=len(state['history'])+1;cp=folder/'rounds'/f'{q:02}_commit.json'
                if cp.exists():commit=read(cp)
                else:
                    ctx=gp.context(state['history'],q);ordered,display=engine.candidates(pool,ctx,q)
                    ids=[pool.ids[int(j)] for j in display];notes=state['notes'];memory=None
                    refresh=notes_policy.edit_event(pool,state,ordered,q)
                    if refresh['allowed']:
                        memory=queue.request(ROOT,job['id'],'edit',q,prompts.MEMORY_SYSTEM,
                            prompts.memory_public(pool,state,ids,q,refresh),lambda raw:notes_policy.apply_edits(raw,pool,state,q))
                        notes=memory['plan']['notes']
                    response=queue.request(ROOT,job['id'],'choose',q,prompts.REASONING_SYSTEM,
                        prompts.reasoning_public(ROOT,pool,state,notes,ids,q),
                        lambda raw:policy.plan(ROOT,pool,notes,state,ctx,ids,q,raw,job['arm']),
                        selected_validator=lambda raw:validate_insight(ROOT,raw['insight'],ids))
                    packet=response['plan']
                    commit=dict(query=q,selected_id=packet['selected_id'],prestate_sha256=control.digest(state),
                        protocol_sha256=p['protocol_sha256'],GP_sha256=engine.gp_hash(ctx),
                        ordered_ids=[pool.ids[int(j)] for j in ordered],display_ids=ids,
                        stage=ctx['stage'],qLogEI_defined=ctx['fitted_GP'],refresh=refresh,memory=memory,
                        edited_notes=notes,reasoning=response,decision=packet)
                    commit['commit_sha256']=control.digest(commit)
                    frozen._save_context(folder,q,ctx);write(cp,commit)
                if commit['prestate_sha256']!=control.digest(state):raise ValueError('State differs before oracle')
                observation=knowledge_gp.read_observation(ROOT,pool,commit['selected_id'])
                state=policy.feedback(state,commit,observation)
                frozen.warm.write(sf,state);frozen.warm.write(folder/'runtime_counters.json',gp.counters);gp.release()
                print(json.dumps(dict(phase='trace',cell=job['cell'],arm=job['arm'],query=q,
                    objective=observation['raw_value'],best=max(h['raw_value'] for h in state['history']))),flush=True)
    except Exception as error:
        frozen.warm.write(folder/'execution_error.json',dict(type=type(error).__name__,message=str(error),query=len(state['history'])+1));raise
    finally:gp.release()
    write(folder/'completion.json',dict(status='COMPLETE',queries=20))
    return dict(id=job['id'],status='COMPLETE',queries=20)


def run_job(job):
    global _CHECKED
    if _CHECKED is None:_CHECKED=check()
    os.chdir(ROOT/'experiments/chemist_system')
    return prefix_job(job,_CHECKED) if job['phase']=='prefix' else trace_job(job,_CHECKED)


def preflight():
    p=check(True);rows=[]
    compiled_extractor(str(ROOT))
    for task,seed in CELLS:
        path=reference(p,task,seed);old=read(path);pool=frozen.pool_for(ROOT,task)
        gp=GP(pool,base()/'cache'/task,seed)
        for n in (0,1,4,7):
            ctx=gp.context(old['history'][:n],n+1);expected=saved_context(path.parent,n+1)
            if engine.gp_hash(ctx)!=engine.gp_hash(expected):raise ValueError('Native GP preflight differs')
            rows.append(dict(task=task,seed=seed,prefix=n,GP_and_menu_exact=True));gp.release()
        if gp.counters['operator_compiles_in_loop']!=0:raise ValueError('Operator compilation in query loop')
    value=dict(status='PASS',checks=rows,LLM_calls=0,oracle_queries=0,operator_compiles_in_loop=0)
    write(base()/'preflight.json',value);return value


def batch(phase,workers=4):
    p=check(True)
    if phase=='traces' and read(base()/'prefix_complete.json')['status']!='COMPLETE':
        raise ValueError('Finish and inspect fixed-prefix diagnostics before complete trajectories')
    if read(base()/'preflight.json')['status']!='PASS':raise ValueError('Preflight required')
    todo=[j for j in p['jobs'] if j['phase']==phase];results=[];errors=[]
    with ProcessPoolExecutor(max_workers=min(workers,len(todo))) as executor:
        fs={executor.submit(run_job,j):j for j in todo}
        for f in as_completed(fs):
            try:results.append(f.result())
            except Exception as error:
                e=dict(job=fs[f]['id'],type=type(error).__name__,message=str(error));errors.append(e);print('JOB_ERROR '+json.dumps(e),flush=True)
    check(True);result=dict(status='COMPLETE' if not errors else 'FAILED',phase=phase,results=results,errors=errors)
    write(base()/f'{phase}_complete.json',result)
    if errors:raise RuntimeError(str(errors))
    return result


def main():
    a=argparse.ArgumentParser();a.add_argument('action',choices=('prepare','preflight','prefix','traces'));a.add_argument('--workers',type=int,default=4)
    args=a.parse_args();r=prepare() if args.action=='prepare' else preflight() if args.action=='preflight' else batch(args.action,args.workers)
    print(json.dumps(r if args.action!='prepare' else dict(variant=VARIANT,protocol_sha256=r['protocol_sha256'],new_queries=80)))


if __name__=='__main__':main()
