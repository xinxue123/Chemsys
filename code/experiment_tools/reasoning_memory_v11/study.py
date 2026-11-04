"""One new Reasoning-BO-policy arm and read-only matched historical controls."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import json,shutil,subprocess
import numpy as np
from chem.benchmarks import knowledge_gp as gpmod,knowledge_study as old,knowledge_openai_client as api
from chem.benchmarks.knowledge_llm import digest,write
from knowledge_rank_v9 import study as v9,engine as previous
from knowledge_information_v10 import study as v10
from . import llm,engine

VARIANT='v11_reasoning_memory'
DESIGN='docs/reasoning_memory_v11_protocol_20261101.txt'


def directory(root):return Path(root)/'experiments/runs/knowledge_loop_20261101'/VARIANT


def sources(root):
    root=Path(root);files=v10.sources(root)
    extra=list((root/'code/experiment_tools/reasoning_memory_v11').rglob('*.py'))
    extra += [root/'code/experiment_tools/run_reasoning_memory_v11.py']
    files.update({p.relative_to(root).as_posix():old.sha(p) for p in extra});return files


def protected(root):
    root=Path(root);files=v10.protected(root);base=v10.directory(root)
    paths=[base/'protocol.json']+[p for kind in ('runs','requests') for p in (base/kind).rglob('*') if p.is_file() and p.suffix!='.lock']
    paths += [p for p in (root/'experiments/reports/knowledge_loop_20261101').glob('v10*') if p.is_file()]
    files.update({p.relative_to(root).as_posix():old.sha(p) for p in paths});return files


def copy_cache(root):
    root=Path(root);count=0
    for source in (v10.directory(root)/'cache').rglob('*'):
        if not source.is_file() or source.suffix not in ('.pt','.npz','.json'):continue
        dest=directory(root)/'cache'/source.relative_to(v10.directory(root)/'cache')
        if not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest);count+=1
        elif old.sha(dest)!=old.sha(source):raise ValueError('Native prefix cache conflict')
    return count


def preflight_one(root,task):
    root=Path(root);pool=gpmod.load_task(root,task,seed=0)
    history=old.read(v9.directory(root)/'runs'/f'{task}_bo/state.json')['history'][:4]
    gp=gpmod.GPAdapter(pool,directory(root)/'cache'/task,seed=0);c=gp.context(history,1)
    with np.load(v9.directory(root)/'runs'/f'{task}_bo/rounds/01_GP.npz',allow_pickle=False) as z:
        for key in ('mu_y','sd_y','sd_predictive_y'):np.testing.assert_array_equal(z[key],c['stats'][key])
        np.testing.assert_array_equal(z['base_logei'],c['base_logei'])
    value=dict(task=task,status='PASS_EXACT_ORIGINAL_GP',counters=gp.counters);gp.release();return value


def preflight(root,workers=8):
    copy_cache(root)
    with ProcessPoolExecutor(max_workers=workers) as ex:rows=list(ex.map(preflight_one,[str(root)]*8,gpmod.TASKS))
    write(Path(root)/'experiments/reports/knowledge_loop_20261101/v11_GP_preflight.json',dict(status='PASS',rows=rows,LLM_calls=0,new_queries=0))
    return rows


def prepare(root):
    root=Path(root).resolve();folder=directory(root).resolve()
    if not folder.is_relative_to(root):raise ValueError('Run path escapes workspace')
    if (folder/'protocol.json').exists():return check(root)
    if (folder/'runs').exists() or (folder/'requests').exists():raise ValueError('Register before new activity')
    prior=v10.check(root);pf=old.read(root/'experiments/reports/knowledge_loop_20261101/v11_GP_preflight.json')
    if pf['status']!='PASS' or {r['task'] for r in pf['rows']}!=set(gpmod.TASKS):raise ValueError('All8 native GP checks required')
    p=dict(variant=VARIANT,created_utc=datetime.now(timezone.utc).isoformat(),tasks=list(gpmod.TASKS),arms=['rbo_memory'],
        readonly_controls=['bo','reasoning_bo','V9','V10'],seed=0,trial_seed=0,tie_seed=0,numerical_seed=0,initial=4,queries=16,
        model=api.MODEL,reasoning_effort=api.REASONING_EFFORT,knowledge_condition='own_data_real_agent_no_manual_pressure_priors',
        parameters=engine.PARAMETERS,inputs=deepcopy(prior['inputs']),source_sha256=sources(root),protected_controls_sha256=protected(root),
        design_path=DESIGN,design_sha256=old.sha(root/DESIGN),prompt_sha256=dict(memory=digest(llm.MEMORY_SYSTEM),reasoning=digest(llm.REASONING_SYSTEM)),
        code_milestone=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        official_commit='4df3cd64abd1be1aab533627578523610f89788c',official_selection='unaltered BaseReasoner._extract_candidates_from_insight AST n=1; compile once per process',
        adaptations=['shared4initial,16sequential5-to1 vs paper LLM warmstart/batch5-to3',
            'same native GP/qLogEI instead of changing baseline to official Ax backend',
            'candidate_id I/O resolves to supplied full inputs; official confidence selection unchanged',
            'local evidence Notes retrieved by explicit scopes rather than external vector/graph services',
            'event Notes semantic edits max4, objective feedback every round',
            'no posttraining, outside-pool proposals, or raw private reasoning'],
        acceptance='at least5of8strictSearchAUCwins vs BO; ties not wins; report all final-best losses; seed0 development')
    p['protocol_sha256']=digest(p);write(folder/'protocol.json',p)
    snapshot=root/'runtime/knowledge_loop/snapshots'/VARIANT
    for name,sha in p['source_sha256'].items():
        dest=snapshot/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,dest)
        if old.sha(dest)!=sha:raise ValueError('Snapshot drift')
    (snapshot/DESIGN).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/DESIGN,snapshot/DESIGN)
    shutil.copy2(folder/'protocol.json',snapshot/'protocol.json');copy_cache(root);return check(root)


def check(root):
    root=Path(root);p=old.read(directory(root)/'protocol.json')
    if p['protocol_sha256']!=digest({k:v for k,v in p.items() if k!='protocol_sha256'}):raise ValueError('Protocol changed')
    current=sources(root)
    if p['source_sha256']!=current:
        amendment=old.read(directory(root)/'technical_amendment_01.json')
        if (amendment['amendment_sha256']!=digest({k:v for k,v in amendment.items() if k!='amendment_sha256'})
            or amendment['protocol_sha256']!=p['protocol_sha256'] or amendment['revised_source_sha256']!=current
            or amendment['kind']!='NON_SELECTION_CITATION_TRANSPORT_FIX'):
            raise ValueError('Frozen source/amendment drift')
    if p['design_sha256']!=old.sha(root/p['design_path']):raise ValueError('Frozen design drift')
    for item in p['inputs']:
        for name,sha in item['files_sha256'].items():
            if old.sha(root/name)!=sha:raise ValueError('Candidate input drift')
    if p['tasks']!=list(gpmod.TASKS) or p['seed']!=0 or p['initial']!=4 or p['queries']!=16:raise ValueError('Registered scope changed')
    return p


def run_one(root,task):
    root=Path(root).resolve();p=check(root)
    if task not in p['tasks']:raise ValueError('Unregistered task')
    pool=gpmod.load_task(root,task,seed=0);base=directory(root);run=base/'runs'/f'{task}_rbo_memory';sf=run/'state.json'
    if sf.exists():
        state=old.read(sf);init=old.initial_commit(run,pool,p,'rbo_memory','agent_initial_terra',create=False)
        if init['commit_sha256']!=state['initial_commit_sha256'] or state['protocol_sha256']!=p['protocol_sha256']:raise ValueError('Resume commitment changed')
    else:
        init=old.initial_commit(run,pool,p,'rbo_memory','agent_initial_terra',create=True)
        state=dict(task=task,arm='rbo_memory',seed=0,protocol_sha256=p['protocol_sha256'],initial_commit_sha256=init['commit_sha256'],
            history=[gpmod.read_observation(root,pool,key) for key in pool.initial_ids],hypotheses=[],decisions=[],insights=[],edit_calls=[],status='READY')
        write(sf,state)
    if state['status']=='COMPLETE':return summary(state)
    gp=gpmod.GPAdapter(pool,base/'cache'/task,seed=0);client=api.APIClient(api.load_config(root/'openai-key.txt'))
    llm.compiled_extractor(str(root))
    try:
        while len(state['decisions'])<16:
            t=len(state['decisions'])+1;cp=run/'rounds'/f'{t:02d}_commit.json'
            if cp.exists():commit=old.read(cp)
            else:
                context=gp.context(state['history'],t);ordered,display=previous.candidates(pool,context,t);ids=[pool.ids[j] for j in display]
                refresh=previous.edit_event(pool,state,ordered,t);memory=None
                if refresh['allowed']:
                    public=llm.memory_public(pool,state,ids,t,refresh)
                    memory=llm.request(client,llm.MEMORY_SYSTEM,public,base/'requests',lambda raw:llm.validate_memory(raw,pool,state,ids,t))
                hs=memory['plan']['hypotheses'] if memory else state['hypotheses']
                public=llm.reasoning_public(root,pool,state,hs,ids,t)
                response=llm.request(client,llm.REASONING_SYSTEM,public,base/'requests',
                    lambda raw:llm.validate_reasoning(root,raw,hs,ids,public['retrieved_notes']))
                commit=engine.plan(root,pool,state,context,t,memory,response,refresh);cp.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(cp.parent/f'{t:02d}_GP.npz',mu_y=context['stats']['mu_y'],sd_y=context['stats']['sd_y'],
                    sd_predictive_y=context['stats']['sd_predictive_y'],base_logei=context['base_logei'],available=context['available'],tie_priority=context['tie_priority'])
                write(cp.parent/f'{t:02d}_fit.json',context['stats']['fit_parameters']);write(cp,commit)
            if commit['prestate_sha256']!=digest(state):raise ValueError('Committed prestate changed')
            obs=gpmod.read_observation(root,pool,commit['selected_id']);engine.feedback(state,commit,obs)
            state['cache_counters']=gp.counters;write(sf,state);gp.release()
            print(json.dumps(dict(task=task,query=t,mode=commit['mode'],objective=obs['raw_value'],
                cited_notes=len(commit['memory_references']),KF_events=[e['reason'] for e in state['decisions'][-1]['KF_feedback']]),ensure_ascii=False),flush=True)
    except Exception as error:
        write(run/'execution_error.json',dict(kind=type(error).__name__,query=len(state['decisions'])+1,
            safe_message=str(error) if isinstance(error,api.OpenAIClientError) else 'Inspect local traceback; frozen prefix preserved'));raise
    finally:client.close()
    check(root);return summary(state)


def summary(state):
    result=v9.summary(state)
    result.update(LLM_calls=len(state['insights'])+len(state['edit_calls']),
        decisions_citing_notes=sum(bool(d['memory_references']) for d in state['decisions']),
        counterevidence_citations=sum(r['use']=='counterevidence' for d in state['decisions'] for r in d['memory_references']))
    return result


def batch(root,workers=8):
    p=check(root)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        fs=[ex.submit(run_one,str(root),t) for t in p['tasks']]
        for f in as_completed(fs):print('FINISH '+json.dumps(f.result(),ensure_ascii=False),flush=True)
