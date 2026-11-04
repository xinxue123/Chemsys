"""Eight new trajectories; frozen V9/RBO/BO controls are read-only."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import json,shutil,subprocess
import numpy as np
from chem.benchmarks import knowledge_gp as gpmod,knowledge_study as old,knowledge_openai_client as api
from chem.benchmarks.knowledge_llm import digest,write
from knowledge_rank_v9 import study as v9,engine as previous
from . import engine,llm,information

VARIANT='v10_information'
DESIGN='docs/knowledge_information_v10_protocol_20261101.txt'


def directory(root):return Path(root)/'experiments/runs/knowledge_loop_20261101'/VARIANT


def sources(root):
    root=Path(root);files=v9.sources(root)
    extra=list((root/'code/experiment_tools/knowledge_information_v10').rglob('*.py'))
    extra+=list((root/'code/experiment_tools/knowledge_design_v10').rglob('*.py'))
    extra+=[root/'code/experiment_tools/run_knowledge_information_v10.py']
    files.update({p.relative_to(root).as_posix():old.sha(p) for p in extra})
    return files


def protected(root):
    root=Path(root);base=v9.directory(root)
    paths=[base/'protocol.json']
    paths += [p for kind in ('runs','requests') for p in (base/kind).rglob('*') if p.is_file() and p.suffix!='.lock']
    paths += [p for p in (root/'experiments/reports/knowledge_loop_20261101').glob('v9*') if p.is_file()]
    return {p.relative_to(root).as_posix():old.sha(p) for p in paths}


def copy_cache(root):
    root=Path(root);count=0
    for source in (v9.directory(root)/'cache').rglob('*'):
        if not source.is_file() or source.suffix not in ('.pt','.npz','.json'):continue
        dest=directory(root)/'cache'/source.relative_to(v9.directory(root)/'cache')
        if not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest);count+=1
        elif old.sha(dest)!=old.sha(source):raise ValueError('Native cached prefix conflict')
    return count


def preflight_one(root,task):
    root=Path(root);pool=gpmod.load_task(root,task,seed=0)
    history=old.read(v9.directory(root)/'runs'/f'{task}_bo/state.json')['history'][:4]
    gp=gpmod.GPAdapter(pool,directory(root)/'cache'/task,seed=0)
    context=gp.context(history,1);before=previous.gp_hash(context)
    ordered,_=previous.candidates(pool,context,1)
    values,arrays=information.evaluate(gp,context,[int(ordered[0]),int(ordered[-1]),pool.ix[pool.initial_ids[0]]])
    if before!=previous.gp_hash(context):raise ValueError('GP preflight changed original predictions')
    with np.load(v9.directory(root)/'runs'/f'{task}_bo/rounds/01_GP.npz',allow_pickle=False) as archive:
        for name,actual in [('mu_y',context['stats']['mu_y']),('sd_y',context['stats']['sd_y']),('base_logei',context['base_logei'])]:
            np.testing.assert_array_equal(archive[name],actual)
    value=dict(task=task,status='PASS',max_mean_error=float(arrays['max_mean_error']),
        max_variance_error=float(arrays['max_variance_error']),KG=list(values.values()),counters=gp.counters)
    gp.release();return value


def preflight(root,workers=8):
    copy_cache(root)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        rows=list(executor.map(preflight_one,[str(root)]*8,gpmod.TASKS))
    write(Path(root)/'experiments/reports/knowledge_loop_20261101/v10_GP_preflight.json',
        dict(status='PASS',rows=rows,oracle_queries=0,LLM_calls=0))
    return rows


def prepare(root):
    root=Path(root).resolve();folder=directory(root).resolve()
    if not folder.is_relative_to(root):raise ValueError('Run path escapes workspace')
    if (folder/'protocol.json').exists():return check(root)
    if (folder/'runs').exists() or (folder/'requests').exists():raise ValueError('Register before online activity')
    prior=v9.check(root)
    pf=old.read(root/'experiments/reports/knowledge_loop_20261101/v10_GP_preflight.json')
    if pf['status']!='PASS' or {r['task'] for r in pf['rows']}!=set(gpmod.TASKS):raise ValueError('Native GP preflight required')
    p=dict(variant=VARIANT,created_utc=datetime.now(timezone.utc).isoformat(),tasks=list(gpmod.TASKS),
        arms=['information'],readonly_controls=['V9','reasoning_bo','bo'],seed=0,trial_seed=0,tie_seed=0,
        numerical_seed=0,initial=4,queries=16,model=api.MODEL,reasoning_effort=api.REASONING_EFFORT,
        knowledge_condition='real_agent_terra; no manual pressure priors',parameters=engine.PARAMETERS,
        source_sha256=sources(root),inputs=deepcopy(prior['inputs']),protected_V9_sha256=protected(root),
        V9_protocol_sha256=prior['protocol_sha256'],design_path=DESIGN,design_sha256=old.sha(root/DESIGN),
        prompts_sha256=dict(proposal=digest(llm.PROPOSAL_SYSTEM),forecast=digest(llm.FORECAST_SYSTEM)),
        code_milestone=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        action_space='V9 top5 fallback plus up to4 independently mapped full-pool probes; neutral union display',
        controls_policy='Reuse24 sealed trajectories. User explicitly declined additional information-only controls.',
        acceptance='at least5/8strictSearchAUCwins vs pureBO; ties not wins; all final-best losses disclosed; seed0development')
    p['protocol_sha256']=digest(p);write(folder/'protocol.json',p)
    snapshot=root/'runtime/knowledge_loop/snapshots'/VARIANT
    for name,expected in p['source_sha256'].items():
        target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,target)
        if old.sha(target)!=expected:raise ValueError('Source snapshot mismatch')
    (snapshot/DESIGN).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/DESIGN,snapshot/DESIGN)
    shutil.copy2(folder/'protocol.json',snapshot/'protocol.json');copy_cache(root)
    return check(root)


def check(root):
    root=Path(root);p=old.read(directory(root)/'protocol.json')
    if p['protocol_sha256']!=digest({k:v for k,v in p.items() if k!='protocol_sha256'}):raise ValueError('Protocol changed')
    if p['source_sha256']!=sources(root) or p['design_sha256']!=old.sha(root/p['design_path']):raise ValueError('Frozen code/design drift')
    for row in p['inputs']:
        for name,expected in row['files_sha256'].items():
            if old.sha(root/name)!=expected:raise ValueError('Original pool drift')
    if p['tasks']!=list(gpmod.TASKS) or p['initial']!=4 or p['queries']!=16 or p['seed']!=0:raise ValueError('Registered scope changed')
    return p


def run_one(root,task):
    root=Path(root).resolve();p=check(root)
    if task not in p['tasks']:raise ValueError('Unregistered task')
    pool=gpmod.load_task(root,task,seed=0);folder=directory(root);run=folder/'runs'/f'{task}_information'
    statefile=run/'state.json'
    if statefile.exists():
        state=old.read(statefile);init=old.initial_commit(run,pool,p,'information','agent_initial_terra',create=False)
        if state['initial_commit_sha256']!=init['commit_sha256'] or state['protocol_sha256']!=p['protocol_sha256']:raise ValueError('Resume prefix changed')
    else:
        init=old.initial_commit(run,pool,p,'information','agent_initial_terra',create=True)
        state=dict(task=task,arm='information',seed=0,protocol_sha256=p['protocol_sha256'],initial_commit_sha256=init['commit_sha256'],
            history=[gpmod.read_observation(root,pool,key) for key in pool.initial_ids],hypotheses=[],decisions=[],insights=[],edit_calls=[],
            information_used=0,dissent_used=False,tested_probe_signatures=[],status='READY')
        write(statefile,state)
    if state['status']=='COMPLETE':return summary(state)
    gp=gpmod.GPAdapter(pool,folder/'cache'/task,seed=0);client=api.APIClient(api.load_config(root/'openai-key.txt'))
    try:
        while len(state['decisions'])<16:
            t=len(state['decisions'])+1;cp=run/'rounds'/f'{t:02d}_commit.json'
            if cp.exists():commit=old.read(cp)
            else:
                context=gp.context(state['history'],t);ordered,_=previous.candidates(pool,context,t)
                refresh=previous.edit_event(pool,state,ordered,t);proposal=None
                if refresh['allowed']:
                    public=llm.proposal_public(pool,state,t,refresh,context['tie_priority'])
                    proposal=llm.request(client,llm.PROPOSAL_SYSTEM,public,folder/'requests',
                        lambda raw:llm.validate_proposal(raw,pool,state,t,context['available'],context['tie_priority']))
                hs=proposal['plan']['hypotheses'] if proposal else state['hypotheses']
                _,display,_=engine.menu(pool,state,hs,context,t);ids=[pool.ids[j] for j in display]
                public=llm.forecast_public(pool,state,hs,ids,t)
                forecast=llm.request(client,llm.FORECAST_SYSTEM,public,folder/'requests',lambda raw:llm.validate_forecast(raw,hs,ids))
                commit,kg=engine.plan(pool,state,context,t,proposal,forecast,refresh,gp)
                cp.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(cp.parent/f'{t:02d}_GP.npz',mu_y=context['stats']['mu_y'],sd_y=context['stats']['sd_y'],
                    sd_predictive_y=context['stats']['sd_predictive_y'],base_logei=context['base_logei'],available=context['available'],
                    tie_priority=context['tie_priority'],eligible_V8=context['eligible'])
                if kg is not None:np.savez_compressed(cp.parent/f'{t:02d}_KG.npz',**kg)
                write(cp.parent/f'{t:02d}_fit.json',context['stats']['fit_parameters']);write(cp,commit)
            if commit['prestate_sha256']!=digest(state):raise ValueError('Committed prestate mismatch')
            observation=gpmod.read_observation(root,pool,commit['selected_id']);engine.feedback(state,commit,observation)
            state['cache_counters']=gp.counters;write(statefile,state);gp.release()
            print(json.dumps(dict(task=task,query=t,mode=commit['mode'],objective=observation['raw_value'],
                info_used=state['information_used'],outside_top5=commit['selected_id'] not in commit['ordered_ids'],
                KF_events=[e['reason'] for e in state['decisions'][-1]['KF_feedback']]),ensure_ascii=False),flush=True)
    except Exception as error:
        write(run/'execution_error.json',dict(kind=type(error).__name__,query=len(state['decisions'])+1,
            safe_message=str(error) if isinstance(error,api.OpenAIClientError) else 'Inspect local traceback; frozen prefix retained'))
        raise
    finally:client.close()
    check(root);return summary(state)


def summary(state):
    row=v9.summary(state)
    row.update(non_BO_choices=sum(d['selected_id']!=d['BO_reference_id'] for d in state['decisions']),
        information_used=state['information_used'],dissent_used=state['dissent_used'],
        LLM_calls=len(state['insights'])+len(state['edit_calls']))
    return row


def batch(root,workers=8):
    p=check(root)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        fs=[executor.submit(run_one,str(root),task) for task in p['tasks']]
        for f in as_completed(fs):print('FINISH '+json.dumps(f.result(),ensure_ascii=False),flush=True)
