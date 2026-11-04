"""Frozen eight-task study; historical evidence read-only; commit before oracle."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import json
import shutil
import subprocess
import numpy as np
from chem.benchmarks import knowledge_gp as gpmod,knowledge_study as old
from chem.benchmarks import knowledge_openai_client as api
from chem.benchmarks.knowledge_llm import digest,write
from . import engine,llm

VARIANT='v9_kf_rank'
ARMS=('bo','reasoning_bo','kf_rank')
DESIGN='docs/knowledge_rank_v9_design_20261101.txt'


def directory(root):return Path(root)/'experiments/runs/knowledge_loop_20261101'/VARIANT


def sources(root):
    root=Path(root);files=old.source_hashes(root)
    extra=list((root/'code/experiment_tools/knowledge_rank_v9').glob('*.py'))
    extra+=[root/'code/experiment_tools/run_knowledge_rank_v9.py',
        root/'code/chemist_system/third_party/reasoning_bo_official/src/bo/reasoner/base.py',
        root/'code/chemist_system/third_party/reasoning_bo_official/src/prompts/test_prompts.json']
    files.update({p.relative_to(root).as_posix():old.sha(p) for p in extra})
    return files


def protected(root):
    base=old.paths(root,'v8');paths=[base/'protocol.json']
    for task in gpmod.TASKS:
        for arm in ('bo','full'):
            run=base/'runs'/f'{task}_{arm}_agent_initial_terra'
            paths.extend(p for p in run.rglob('*') if p.is_file())
    return {p.relative_to(root).as_posix():old.sha(p) for p in sorted(paths)}


def prepare(root):
    root=Path(root).resolve();folder=directory(root).resolve()
    if not folder.is_relative_to(root):raise ValueError('Run path redirects outside workspace')
    dest=folder/'protocol.json'
    if dest.exists():return check(root)
    if (folder/'runs').exists() or (folder/'requests').exists():raise ValueError('Cannot register after activity')
    v8=old.check(root,'v8')
    p=dict(variant=VARIANT,created_utc=datetime.now(timezone.utc).isoformat(),
        tasks=list(gpmod.TASKS),arms=list(ARMS),seed=0,trial_seed=0,tie_seed=0,numerical_seed=0,menu_seed=0,
        initial=4,queries=16,knowledge_condition='real_agent_terra; no manual pressure priors',
        model=api.MODEL,reasoning_effort=api.REASONING_EFFORT,parameters=engine.PARAMETERS,
        source_sha256=sources(root),inputs=deepcopy(v8['inputs']),V8_protocol_sha256=v8['protocol_sha256'],
        protected_V8_sha256=protected(root),design_path=DESIGN,design_sha256=old.sha(root/DESIGN),
        prompt_sha256=dict(kf_rank=digest(llm.KF_SYSTEM),reasoning_bo=digest(llm.RBO_SYSTEM)),
        code_milestone=subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True,check=True).stdout.strip(),
        action_space='same measured pool; both LLM arms share base q1 top5, physical neutral display order; no V8 epsilon gate',
        RBO_adaptations=['official confidence extraction AST n=1','candidate_id transport resolved to supplied full conditions',
            'shared4initial16sequential','same Terra medium and native per-task GP','no warmstart/Notes/posttraining',
            'current exact own prefix and visible Insight history; no private reasoning'],
        acceptance='at least5of8strictAUCwins vs BO (difference>1e-9); ties not wins; disclose final-best losses; seed0 development only')
    p['protocol_sha256']=digest(p);write(dest,p)
    snapshot=root/'runtime/knowledge_loop/snapshots'/VARIANT
    for name in p['source_sha256']:
        target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,target)
        if old.sha(target)!=p['source_sha256'][name]:raise ValueError('Snapshot mismatch')
    (snapshot/DESIGN).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/DESIGN,snapshot/DESIGN)
    shutil.copy2(dest,snapshot/'protocol.json')
    files=size=0
    for source in (old.paths(root,'v8')/'cache').rglob('*'):
        if source.is_file() and source.suffix in ('.pt','.npz','.json'):
            target=folder/'cache'/source.relative_to(old.paths(root,'v8')/'cache');target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and old.sha(target)!=old.sha(source):raise ValueError('Conflicting cache prefix')
            if not target.exists():shutil.copy2(source,target)
            files+=1;size+=source.stat().st_size
    receipt=dict(protocol_sha256=p['protocol_sha256'],before_new_queries=True,LLM_requests_copied=0,
        GP_cache_files=files,GP_cache_bytes=size,snapshot=str(snapshot),protected_V8_files=len(p['protected_V8_sha256']))
    write(root/'experiments/reports/knowledge_loop_20261101/v9_freeze_receipt.json',receipt)
    return check(root)


def check(root):
    root=Path(root);p=old.read(directory(root)/'protocol.json')
    if p['protocol_sha256']!=digest({k:v for k,v in p.items() if k!='protocol_sha256'}):raise ValueError('Protocol changed')
    if p['source_sha256']!=sources(root) or p['design_sha256']!=old.sha(root/p['design_path']):raise ValueError('Frozen source/design drift')
    for m in p['inputs']:
        for path,sha in m['files_sha256'].items():
            if old.sha(root/path)!=sha:raise ValueError('Candidate pool drift')
    if p['tasks']!=list(gpmod.TASKS) or p['arms']!=list(ARMS) or p['seed']!=0 or p['initial']!=4 or p['queries']!=16:
        raise ValueError('Registered scope changed')
    return p


def run_one(root,task,arm):
    root=Path(root).resolve();p=check(root)
    if task not in p['tasks'] or arm not in ARMS:raise ValueError('Unregistered task/arm')
    pool=gpmod.load_task(root,task,seed=0);folder=directory(root);run=folder/'runs'/f'{task}_{arm}'
    statefile=run/'state.json'
    if statefile.exists():
        state=old.read(statefile)
        init=old.initial_commit(run,pool,p,arm,'agent_initial_terra',create=False)
        if state['initial_commit_sha256']!=init['commit_sha256'] or state['protocol_sha256']!=p['protocol_sha256']:
            raise ValueError('Resumed registration/initial commitment changed')
    else:
        init=old.initial_commit(run,pool,p,arm,'agent_initial_terra',create=True)
        state=dict(task=task,arm=arm,seed=0,protocol_sha256=p['protocol_sha256'],
            initial_commit_sha256=init['commit_sha256'],history=[gpmod.read_observation(root,pool,key) for key in pool.initial_ids],
            hypotheses=[],decisions=[],insights=[],edit_calls=[],status='READY')
        write(statefile,state)
    if state['status']=='COMPLETE':return summary(state)
    gp=gpmod.GPAdapter(pool,folder/'cache'/task,seed=0)
    client=api.APIClient(api.load_config(root/'openai-key.txt')) if arm!='bo' else None
    try:
        while len(state['decisions'])<16:
            t=len(state['decisions'])+1;cp=run/'rounds'/f'{t:02d}_commit.json'
            if cp.exists():commit=old.read(cp)
            else:
                context=gp.context(state['history'],t);ordered,display=engine.candidates(pool,context,t)
                ids=[pool.ids[int(j)] for j in display];receipt=refresh=None
                if arm=='kf_rank':
                    refresh=engine.edit_event(pool,state,ordered,t)
                    public=llm.kf_public(pool,state,ids,t,refresh)
                    receipt=llm.request(client,llm.KF_SYSTEM,public,folder/'requests',
                        lambda raw:llm.validate_kf(raw,pool,state,ids,t,refresh))
                elif arm=='reasoning_bo':
                    public=llm.rbo_public(root,pool,state,ids,t)
                    receipt=llm.request(client,llm.RBO_SYSTEM,public,folder/'requests',lambda raw:llm.validate_rbo(root,raw,ids))
                commit=engine.plan(pool,state,context,t,arm,receipt,refresh)
                cp.parent.mkdir(parents=True,exist_ok=True)
                with (cp.parent/f'{t:02d}_GP.npz').open('wb') as stream:
                    np.savez_compressed(stream,mu_y=context['stats']['mu_y'],sd_y=context['stats']['sd_y'],
                        sd_predictive_y=context['stats']['sd_predictive_y'],base_logei=context['base_logei'],
                        available=context['available'],tie_priority=context['tie_priority'],eligible_V8=context['eligible'])
                write(cp.parent/f'{t:02d}_fit.json',context['stats']['fit_parameters'])
                write(cp,commit)
            if commit['prestate_sha256']!=digest(state) or not cp.exists():raise ValueError('Committed state mismatch')
            observation=gpmod.read_observation(root,pool,commit['selected_id'])
            engine.feedback(state,commit,observation)
            state['cache_counters']=gp.counters;write(statefile,state);gp.release()
            print(json.dumps(dict(task=task,arm=arm,query=t,selection=commit['mode'],objective=observation['raw_value'],
                KF_events=[e['reason'] for e in state['decisions'][-1]['KF_feedback']]),ensure_ascii=False),flush=True)
    except Exception as error:
        write(run/'execution_error.json',dict(kind=type(error).__name__,safe_message=str(error) if isinstance(error,api.OpenAIClientError)
            else 'Inspect frozen local traceback; original pre-feedback state preserved',query=len(state['decisions'])+1))
        raise
    finally:
        if client:client.close()
    check(root);return summary(state)


def summary(state):
    values=np.asarray([r['raw_value'] for r in state['history']]);curve=np.maximum.accumulate(values)
    events=[e for d in state['decisions'] for e in d['KF_feedback']]
    return dict(task=state['task'],arm=state['arm'],status=state['status'],queries=len(state['decisions']),
        SearchAUC=float(curve[4:].mean()) if len(values)>4 else None,final_best=float(curve[-1]),
        first_best_total=int(np.flatnonzero(values==curve[-1])[0])+1,
        non_BO_choices=sum(d['mode']!='base_bo' for d in state['decisions']),
        KF_weight_selection_changes=sum(d['KF_changes_selection_from_fixed_weights'] for d in state['decisions']),
        forecast_updates=len(events),downgrades=sum(e['after']['weight']<e['before']['weight'] for e in events),
        automatic_pauses=sum(e['reason']=='automatic_pause' for e in events),automatic_retractions=sum(e['reason']=='automatic_retract' for e in events),
        automatic_resumes=sum(e['reason']=='automatic_resume' for e in events),edit_calls=len(state['edit_calls']),LLM_calls=len(state['insights']))


def batch(root,arm,workers=8):
    p=check(root)
    if arm!='bo':
        for task in p['tasks']:
            if old.read(directory(root)/'runs'/f'{task}_bo/state.json')['status']!='COMPLETE':raise ValueError('Run matched BO first')
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures=[executor.submit(run_one,str(root),task,arm) for task in p['tasks']]
        for future in as_completed(futures):print('FINISH '+json.dumps(future.result(),ensure_ascii=False),flush=True)
