"""Registered six-trajectory V14 validation; frozen controls are read-only."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
from unittest.mock import patch
import json,os,shutil
import numpy as np
from filelock import FileLock
from cold_start_v13 import study as frozen,control as old,validation as old_validation
from cold_start_v13.gp import ColdGP
from cold_start_v13_seeds import study as ms
from chem.benchmarks import knowledge_gp
from knowledge_rank_v9 import engine
from reasoning_memory_v12.llm import compiled_extractor,validate_insight
from . import VARIANT,MODEL,EFFORT,QUERIES,transport,controller,policy,prompts

DESIGN='docs/reason_cards_v14_20261103.txt'
JOBS=(('Perera_Suzuki',2),('HPLC',0),('Reizman_Suzuki_Case_1',2),
      ('Doyle_Buchwald_Hartwig',3),('SnAr',0),('HPLC',2))
PRIORITIES=knowledge_gp.semantic_priorities


def tag(task,seed):
    return f'v14_{frozen.SHORT[task]}_s{seed}'


def run_path(root,task,seed):
    return transport.base(root)/'runs'/tag(task,seed)


def baseline_path(root,task,seed,arm):
    return frozen.run_path(root,task,arm) if seed==0 else ms.run_path(root,task,seed,arm)


def sources(root):
    root=Path(root)
    result=ms.sources(root)
    files=list((root/'code/experiment_tools/reason_cards_v14').glob('*.py'))
    files += [root/'code/experiment_tools/run_reason_cards_v14.py',root/DESIGN]
    result.update({p.relative_to(root).as_posix():transport.file_sha(p) for p in files})
    return result


def check(root,historical=False):
    root=Path(root).resolve();p=transport.read(transport.base(root)/'protocol.json')
    if p['protocol_sha256']!=old.digest({k:v for k,v in p.items() if k!='protocol_sha256'}):
        raise ValueError('V14 protocol changed')
    if p['sources_sha256']!=sources(root) or p['jobs']!=[list(j) for j in JOBS]:
        raise ValueError('Registered implementation or scope changed')
    if p['parameters']!=policy.PARAMETERS or (p['initial'],p['queries'])!=(0,20):
        raise ValueError('Registered parameters or budget changed')
    for rel,expected in p['inputs_sha256'].items():
        if transport.file_sha(root/rel)!=expected:raise ValueError('Frozen input changed: '+rel)
    if historical:
        for rel,expected in p['protected_sha256'].items():
            if transport.file_sha(root/rel)!=expected:raise ValueError('Frozen evidence changed: '+rel)
    return p


def prepare(root):
    root=Path(root).resolve();base=transport.base(root)
    if (base/'protocol.json').exists():return check(root,True)
    if (base/'runs').exists() or (base/'requests').exists():raise ValueError('Register before any new decisions')
    prior=ms.check(root,True)
    protected=dict(prior['protected_seed0_sha256'])
    for rel in ('experiments/runs/cold_s123_20261102','experiments/reports/cold_s123_20261102'):
        directory=root/rel
        for p in directory.rglob('*'):
            if p.is_file() and '.lock' not in p.name and not any(a in ('cache','preflight_cache') for a in p.relative_to(directory).parts):
                protected[p.relative_to(root).as_posix()]=transport.file_sha(p)
    baselines=[]
    for task,seed in JOBS:
        for arm in ('bo','v13','reasoning_bo'):
            path=baseline_path(root,task,seed,arm)/'state.json';state=transport.read(path)
            baselines.append(dict(task=task,seed=seed,method=arm,**old_validation.metrics(state),
                state_sha256=transport.file_sha(path),path=path.relative_to(root).as_posix()))
    protocol=dict(variant=VARIANT,created_utc=datetime.now(timezone.utc).isoformat(),
        jobs=[list(j) for j in JOBS],initial=0,queries=20,numerical_seed=0,
        maximum_new_experiments=120,maximum_logical_LLM_calls=144,
        model=MODEL,effort=EFFORT,LLM_backend='fresh_independent_subagent',
        parameters=policy.PARAMETERS,sources_sha256=sources(root),protected_sha256=protected,
        inputs_sha256=frozen.check(root)['inputs_sha256'],
        frozen_multiseed_protocol_sha256=prior['protocol_sha256'],baselines=baselines,
        primary_metric='mean of all20 post-query cumulative best values',
        study_scope='Targeted diagnostic validation: two known failures and four known gains. Not a fresh unbiased benchmark or stability estimate.',
        algorithm_differences=['always-on candidate forecast cards','matched/local direction credit',
            'KF-weighted logEI opportunity guard','physical local adverse-evidence guard',
            'two combined review experiments','executable-note refresh'],
        official_algorithm_difference='Official confidence extractor supplies the proposal. V14 may execute own GP reference instead; never call this an unmodified official Reasoning BO run.',
        no_GP_change=True,no_baseline_rerun=True,no_oracle_values_read_at_registration=True,
        no_LLM_calls_at_registration=True,no_midrun_parameter_changes=True)
    protocol['protocol_sha256']=old.digest(protocol)
    transport.write_once(base/'protocol.json',protocol)
    snapshot=root/'runtime/knowledge_loop/snapshots'/VARIANT
    for name,digest in protocol['sources_sha256'].items():
        target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/name,target)
        assert transport.file_sha(target)==digest
    transport.write_once(snapshot/'protocol.json',protocol)
    return check(root,True)


class GP(ColdGP):
    def __init__(self,pool,directory,start_seed):
        super().__init__(pool,directory,seed=0);self.start_seed=start_seed

    def context(self,history,query):
        def prior(pool,q,seed=0):
            if seed!=0:raise ValueError('GP numerical seed must remain0')
            return PRIORITIES(pool,q,self.start_seed if q==1 else 0)
        with patch.object(knowledge_gp,'semantic_priorities',side_effect=prior):
            return super().context(history,query)


def initial_state(task,seed,protocol,initial):
    state=frozen.initial_state(task,'v13',protocol,initial)
    state.update(seed=seed,starting_seed=seed,numerical_seed=0,variant=VARIANT,
                 decision_channel=controller.initial_channel())
    return state


def feedback(pool,state,commit,observation):
    if (commit['commit_sha256']!=old.digest({k:v for k,v in commit.items() if k!='commit_sha256'})
        or commit['prestate_sha256']!=old.digest(state) or commit['query']!=len(state['history'])+1
        or observation['candidate_id']!=commit['selected_id']
        or observation['candidate_id'] in {r['candidate_id'] for r in state['history']}):
        raise ValueError('Committed experiment mismatch')
    out=deepcopy(state);packet=commit['decision']
    notes,channel,events,card_event=controller.feedback(pool,commit['edited_notes'],state['decision_channel'],
        state['history'],packet,observation)
    out.update(notes=notes,decision_channel=channel)
    if commit['memory']:out['edit_calls'].append(commit['query'])
    uses=packet['knowledge_uses']
    if any(r['use']=='review_test' for r in uses):out['review_queries'].append(commit['query'])
    out['insights'].append(dict(query=commit['query'],insight=deepcopy(commit['reasoning']['raw']['insight']),
        proposed_id=packet['proposed_id'],executed_id=packet['selected_id'],deployment_reason=packet['deployment']['reason']))
    before=max((r['y'] for r in state['history']),default=None)
    out['history'].append(observation)
    out['decisions'].append(dict(query=commit['query'],selected_id=packet['selected_id'],proposed_id=packet['proposed_id'],
        BO_reference_id=packet['BO_reference_id'],raw_value=observation['raw_value'],y=observation['y'],
        incumbent_before_y=before,incumbent_after_y=max(r['y'] for r in out['history']),
        feedback=events,card_feedback=card_event,knowledge_uses=uses,
        decision_card=packet['decision_card'],deployment=packet['deployment'],
        note_coverage_count=len(commit['reasoning_public_coverage']),stage=commit['stage'],commit_sha256=commit['commit_sha256']))
    out['status']='COMPLETE' if len(out['history'])==20 else 'RUNNING'
    return out


def run_one(root,task,seed):
    root=Path(root).resolve();os.chdir(root/'experiments/chemist_system')
    protocol=check(root);pool=frozen.pool_for(root,task);folder=run_path(root,task,seed)
    folder.mkdir(parents=True,exist_ok=True)
    with FileLock(str(folder/'.worker.lock'),timeout=1):
        initial=dict(task=task,starting_seed=seed,initial_ids=[],history=[],initial=0,queries=20,
            protocol_sha256=protocol['protocol_sha256'],input_sha256=transport.file_sha(pool.source_path))
        initial['commit_sha256']=old.digest(initial);transport.write_once(folder/'initial_commit.json',initial)
        sf=folder/'state.json';state=transport.read(sf) if sf.exists() else initial_state(task,seed,protocol,initial)
        if state['protocol_sha256']!=protocol['protocol_sha256']:raise ValueError('Resume protocol mismatch')
        if state['status']=='COMPLETE':return
        gp=GP(pool,transport.base(root)/'cache'/task,seed)
        compiled_extractor(str(root))
        try:
            while len(state['history'])<20:
                q=len(state['history'])+1;cp=folder/'rounds'/f'{q:02d}_commit.json'
                if cp.exists():commit=transport.read(cp)
                else:
                    context=gp.context(state['history'],q);ordered,display=engine.candidates(pool,context,q)
                    ids=[pool.ids[int(j)] for j in display]
                    refresh=policy.edit_event(pool,state,ordered,q);memory=None;notes=state['notes']
                    if refresh['allowed']:
                        memory=transport.request(root,tag(task,seed),'edit',q,prompts.MEMORY_SYSTEM,
                            prompts.memory_public(pool,state,ids,q,refresh),lambda raw:policy.apply_edits(raw,pool,state,q))
                        notes=memory['plan']['notes']
                    public=prompts.reasoning_public(root,pool,state,notes,ids,q)
                    reasoning=transport.request(root,tag(task,seed),'choose',q,prompts.REASONING_SYSTEM,public,
                        lambda raw:policy.validate_decision(root,pool,notes,state,context,ids,q,raw),
                        selected_validator=lambda raw:validate_insight(root,raw['insight'],ids))
                    packet=reasoning['plan']
                    commit=dict(query=q,selected_id=packet['selected_id'],prestate_sha256=old.digest(state),
                        protocol_sha256=protocol['protocol_sha256'],BO_reference_id=packet['BO_reference_id'],
                        GP_fit_key=context['stats']['fit_key'],GP_sha256=engine.gp_hash(context),
                        ordered_ids=[pool.ids[int(j)] for j in ordered],display_ids=ids,
                        stage=context['stage'],qLogEI_defined=context['fitted_GP'],
                        edited_notes=notes,decision=packet,refresh=refresh,memory=memory,reasoning=reasoning,
                        reasoning_public_coverage=[(n['id'],n['version']) for n in public['retrieved_notes']['knowledge']])
                    commit['commit_sha256']=old.digest(commit)
                    frozen._save_context(folder,q,context);transport.write_once(cp,commit)
                if commit['prestate_sha256']!=old.digest(state):raise ValueError('State changed before oracle')
                observation=knowledge_gp.read_observation(root,pool,commit['selected_id'])
                state=feedback(pool,state,commit,observation);frozen.warm.write(sf,state)
                frozen.warm.write(folder/'runtime_counters.json',gp.counters);gp.release()
                print(json.dumps(dict(task=tag(task,seed),query=q,objective=observation['raw_value'],
                    best=max(r['raw_value'] for r in state['history']),proposal=commit['decision']['proposed_id'],
                    deployed=commit['decision']['deployment']['accepted'],reason=commit['decision']['deployment']['reason'])),flush=True)
        except Exception as exc:
            frozen.warm.write(folder/'execution_error.json',dict(type=type(exc).__name__,message=str(exc),query=len(state['history'])+1));raise
        finally:gp.release()
    transport.write_once(folder/'completion.json',dict(status='COMPLETE',queries=20))


def _preflight(root,job):
    root=Path(root);task,seed=job;pool=frozen.pool_for(root,task)
    gp=GP(pool,transport.base(root)/'preflight_cache'/task,seed)
    reference=ColdGP(pool,transport.base(root)/'preflight_cache'/task,seed=0)
    zero=gp.context([],1);ordered,display=engine.candidates(pool,zero,1)
    old_commit=transport.read(baseline_path(root,task,seed,'bo')/'rounds/01_commit.json')
    assert old_commit['display_ids']==[pool.ids[int(j)] for j in display]
    first=np.argsort(zero['tie_priority'])[:4]
    history=[dict(candidate_id=pool.ids[int(j)],y=y,raw_value=y*pool.scale) for j,y in zip(first,(.2,.4,.6,.3))]
    rows=[]
    for n in (1,2,4):
        candidate=gp.context(history[:n],n+1);expected=reference.context(history[:n],n+1)
        assert engine.gp_hash(candidate)==engine.gp_hash(expected)
        rows.append(dict(prefix=n,GP_and_acquisition_exact=True));gp.release();reference.release()
    assert gp.counters['operator_compiles_in_loop']==0
    result=dict(task=task,starting_seed=seed,status='PASS',first_menu_exact=True,synthetic_prefixes=rows,
        oracle_values_read=0,LLM_calls=0,operator_compiles_in_loop=0)
    transport.write_once(transport.base(root)/'preflight'/f'{tag(task,seed)}.json',result)
    return result


def preflight(root,workers=3):
    check(root,True)
    with ProcessPoolExecutor(max_workers=min(3,workers)) as executor:
        rows=list(executor.map(_preflight,[str(root)]*len(JOBS),JOBS))
    transport.write_once(transport.base(root)/'preflight.json',dict(status='PASS',rows=rows))
    return dict(status='PASS',trajectories=len(rows),synthetic_prefixes=[0,1,2,4])


def batch(root,workers=3):
    check(root,True)
    assert transport.read(transport.base(root)/'preflight.json')['status']=='PASS'
    errors=[]
    with ProcessPoolExecutor(max_workers=min(3,workers)) as executor:
        futures={executor.submit(run_one,str(root),*job):job for job in JOBS}
        for f in as_completed(futures):
            try:f.result()
            except Exception as error:
                errors.append(dict(job=futures[f],type=type(error).__name__,message=str(error)))
                print('JOB_ERROR '+json.dumps(errors[-1]),flush=True)
    check(root,True);transport.write_once(transport.base(root)/'batch_receipt.json',dict(status='FAILED' if errors else 'COMPLETE',errors=errors))
    if errors:raise RuntimeError(str(errors))


def status(root):
    rows=[]
    for task,seed in JOBS:
        path=run_path(root,task,seed)/'state.json';state=transport.read(path) if path.exists() else {}
        rows.append(dict(task=task,seed=seed,status=state.get('status','NOT_STARTED'),queries=len(state.get('history',[]))))
    pending=[]
    for r in transport.pending(root):
        export=transport.read(Path(r['request']).parent/'export.json')
        pending.append(dict(task=r['invocation']['task_name'],input=(Path(root)/export['input']).as_posix(),
                            assigned=r['assigned'],output_exists=r['output_exists']))
    return dict(complete=sum(r['status']=='COMPLETE' for r in rows),queries=sum(r['queries'] for r in rows),rows=rows,pending=pending)
