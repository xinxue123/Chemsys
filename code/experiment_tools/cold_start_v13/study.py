"""Twenty paid decisions from an empty history, with prospective commitments."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import shutil

import numpy as np
from filelock import FileLock
from chem.benchmarks import knowledge_gp as native_gp
from knowledge_rank_v9 import engine as native
from reasoning_memory_v12.llm import validate_insight, compiled_extractor
from state_split_v13_experiment import study as warm
from v13_initialization_seeds import study as multi
from . import ARMS, BACKEND, EFFORT, MODEL, QUERIES, VARIANT, adapter, control, policy, prompts, transport
from .gp import ColdGP

SHORT = dict(warm.SHORT)
JOBS = tuple((task,arm) for task in SHORT for arm in ARMS)
DESIGN = 'docs/cold_start_v13_20261102.txt'
MILESTONE = 'experiments/milestones/v13_warmstart_20261102'


def directory(root):
    return transport.base(root)


def tag(task, arm):
    if (task,arm) not in JOBS:
        raise ValueError('Unregistered task/arm')
    return SHORT[task] + '_' + dict(bo='bo',v13='v13',reasoning_bo='rbo')[arm]


def run_path(root, task, arm):
    return directory(root) / 'runs' / tag(task,arm)


def pool_for(root, task):
    pool = native_gp.load_task(root,task,seed=0)
    # The legacy input-only loader validates four historical input labels. They
    # have no role in this study, and none of their objectives is read here.
    pool.initial_ids = ()
    return pool


def sources(root):
    root = Path(root).resolve()
    result = multi.sources(root)
    paths = list((root / 'code/experiment_tools/cold_start_v13').glob('*.py'))
    paths += [root / 'code/experiment_tools/run_cold_start_v13.py',root / DESIGN]
    result.update({p.relative_to(root).as_posix():transport.file_sha(p) for p in paths})
    return result


def check(root, historical=False):
    root = Path(root).resolve()
    p = transport.read(directory(root) / 'protocol.json')
    if p['protocol_sha256'] != control.digest({k:v for k,v in p.items() if k!='protocol_sha256'}):
        raise ValueError('Cold-start protocol changed')
    if p['sources_sha256'] != sources(root) or p['jobs'] != [list(j) for j in JOBS]:
        raise ValueError('Registered source or scope changed')
    if (p['initial'],p['queries'],p['seed']) != (0,QUERIES,0) or p['parameters'] != policy.PARAMETERS:
        raise ValueError('Budget, randomness or V13 parameters changed')
    for name,sha in p['inputs_sha256'].items():
        if transport.file_sha(root / name) != sha:
            raise ValueError('Frozen candidate/oracle input changed')
    if historical:
        receipt = transport.read(root / MILESTONE / 'receipt.json')
        if receipt['status'] != 'SEALED_AND_VERIFIED':
            raise ValueError('Historical evidence was not sealed')
        for name,sha in receipt['artifact_sha256'].items():
            if transport.file_sha(root / MILESTONE / name) != sha:
                raise ValueError('Historical milestone changed')
        manifest = transport.read(root / MILESTONE / 'manifest.json')
        for name,sha in manifest['files_sha256'].items():
            if transport.file_sha(root / name) != sha:
                raise ValueError('Warm-start evidence changed: '+name)
    return p


def prepare(root):
    root = Path(root).resolve()
    d = directory(root)
    if (d / 'protocol.json').exists():
        return check(root,True)
    if (d / 'runs').exists() or (d / 'requests').exists():
        raise ValueError('Registration must precede every paid outcome and LLM call')
    prior = multi.check(root,True)
    inputs = {}
    pools = []
    for task in SHORT:
        pool = pool_for(root,task)
        for path in (pool.source_path,pool.source_path.with_name('oracle.npz')):
            inputs[path.relative_to(root).as_posix()] = transport.file_sha(path)
        pools.append(dict(task=task,candidate_count=len(pool.ids),axes=list(pool.axes),
            kinds=list(pool.kinds),initial_ids=[],sorted_features_sha256=control.digest(sorted(pool.features))))
    p = dict(variant=VARIANT,created_utc=datetime.now(timezone.utc).isoformat(),
        jobs=[list(j) for j in JOBS],initial=0,queries=QUERIES,seed=0,numerical_seed=0,
        tie_seed=0,menu_seed=0,model=MODEL,effort=EFFORT,backend=BACKEND,fork_turns='none',
        maximum_new_queries=len(JOBS)*QUERIES,maximum_logical_LLM_calls=8*(2*QUERIES+4),
        sources_sha256=sources(root),inputs_sha256=inputs,pools=pools,
        parameters=deepcopy(policy.PARAMETERS),frozen_warm_protocol_sha256=prior['protocol_sha256'],
        milestone_manifest_sha256=transport.file_sha(root / MILESTONE / 'manifest.json'),
        first_round='No GP/EI/incumbent; same feature-keyed random five-input menu, BO picks its first prior priority',
        singleton_GP='Same native model/acquisition; posterior output scale1 for singleton or constant-value prefixes only',
        primary_metric='Mean of20 post-query cumulative best objectives; all paid queries count',
        inference_limit='One seed and one independently realized LLM trajectory per method/task; no stability claim',
        accepted_differences=['empty_initial_history','20_paid_queries','no_data_prior_round1',
            'singleton_and_constant_prefix_output_units','absolute_query_index','first_note_after_observation',
            'late_refresh_reserved_until_paid_query15','correct_remaining_query_count'],
        no_outcomes_read_at_registration=True,no_LLM_calls_at_registration=True)
    p['protocol_sha256'] = control.digest(p)
    transport.write_once(d / 'protocol.json',p)
    snapshot = root / 'runtime/knowledge_loop/snapshots' / VARIANT
    for name,sha in p['sources_sha256'].items():
        target = snapshot / name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root / name,target)
        if transport.file_sha(target) != sha:
            raise ValueError('Source snapshot mismatch')
    transport.write_once(snapshot / 'protocol.json',p)
    return check(root,True)


def _save_context(folder, query, context):
    stem = folder / 'rounds' / f'{query:02d}'
    stem.parent.mkdir(parents=True,exist_ok=True)
    target = stem.with_name(stem.name+'_GP.npz')
    arrays = {k:context['stats'][k] for k in ('mu_y','sd_y','sd_predictive_y')}
    arrays.update({k:context[k] for k in ('base_logei','available','tie_priority')})
    temporary = target.with_suffix('.partial')
    with temporary.open('xb') as f:
        np.savez_compressed(f,**arrays)
    temporary.replace(target)
    transport.write_once(stem.with_name(stem.name+'_fit.json'),dict(
        stage=context['stage'],fitted_GP=context['fitted_GP'],
        acquisition=context['acquisition_meta'],parameters=context['stats']['fit_parameters'],
        cold_scale_convention=context['stats'].get('cold_scale_convention')))


def initial_state(task, arm, protocol, initial):
    return dict(task=task,arm=arm,seed=0,protocol_sha256=protocol['protocol_sha256'],
        initial_commit_sha256=initial['commit_sha256'],status='READY',history=[],
        notes=[],decisions=[],insights=[],edit_calls=[],review_queries=[])


def feedback(state, commit, observation):
    if (commit['commit_sha256'] != control.digest({k:v for k,v in commit.items() if k!='commit_sha256'})
        or commit['prestate_sha256'] != control.digest(state)
        or commit['query'] != len(state['history'])+1
        or observation['candidate_id'] != commit['selected_id']
        or observation['candidate_id'] in {r['candidate_id'] for r in state['history']}):
        raise ValueError('Pre-observation commitment mismatch')
    if any(type(observation.get(k)) not in (int,float) or not np.isfinite(observation[k]) for k in ('y','raw_value')):
        raise ValueError('Nonfinite paid outcome')
    out = deepcopy(state)
    before = max((r['y'] for r in state['history']),default=None)
    if state['arm']=='v13':
        notes,events = adapter.apply_feedback(commit['edited_notes'],state['history'],commit['decision'],observation)
        out['notes'] = notes
    else:
        events = []
    if commit['reasoning'] is not None:
        raw = commit['reasoning']['raw']
        out['insights'].append(dict(query=commit['query'],insight=deepcopy(raw['insight'] if state['arm']=='v13' else raw)))
    if commit['memory'] is not None:
        out['edit_calls'].append(commit['query'])
    uses = commit['decision']['knowledge_uses'] if state['arm']=='v13' else []
    if any(r['use']=='review_test' for r in uses):
        out['review_queries'].append(commit['query'])
    out['history'].append(observation)
    out['decisions'].append(dict(query=commit['query'],selected_id=observation['candidate_id'],
        BO_reference_id=commit['BO_reference_id'],raw_value=observation['raw_value'],y=observation['y'],
        incumbent_before_y=before,incumbent_after_y=max(r['y'] for r in out['history']),
        feedback=events,knowledge_uses=deepcopy(uses),stage=commit['stage'],
        commit_sha256=commit['commit_sha256']))
    out['status'] = 'COMPLETE' if len(out['history'])==QUERIES else 'RUNNING'
    return out


def run_one(root, task, arm):
    root = Path(root).resolve()
    os.chdir(root / 'experiments/chemist_system')
    protocol = check(root)
    pool = pool_for(root,task)
    folder = run_path(root,task,arm)
    folder.mkdir(parents=True,exist_ok=True)
    with FileLock(str(folder / '.worker.lock'),timeout=1):
        sf = folder / 'state.json'
        initial = dict(task=task,arm=arm,initial_ids=[],history=[],initial=0,queries=QUERIES,
            protocol_sha256=protocol['protocol_sha256'],input_sha256=transport.file_sha(pool.source_path))
        initial['commit_sha256'] = control.digest(initial)
        transport.write_once(folder / 'initial_commit.json',initial)
        state = transport.read(sf) if sf.exists() else initial_state(task,arm,protocol,initial)
        if state['protocol_sha256']!=protocol['protocol_sha256'] or state['initial_commit_sha256']!=initial['commit_sha256']:
            raise ValueError('Resume mismatch')
        if not sf.exists():
            warm.write(sf,state)
        if state['status']=='COMPLETE':
            return dict(task=task,arm=arm,status='COMPLETE',queries=QUERIES)
        gp = ColdGP(pool,directory(root) / 'cache' / task,seed=0)
        compiled_extractor(str(root))
        try:
            while len(state['history']) < QUERIES:
                query = len(state['history'])+1
                cp = folder / 'rounds' / f'{query:02d}_commit.json'
                if cp.exists():
                    commit = transport.read(cp)
                else:
                    context = gp.context(state['history'],query)
                    original,display = native.candidates(pool,context,query)
                    ids = [pool.ids[int(j)] for j in display]
                    refresh = policy.edit_event(pool,state,original,query) if arm=='v13' else dict(allowed=False,reasons=['no_versioned_memory'])
                    memory = reasoning = decision = None
                    notes = state['notes']
                    if arm=='v13':
                        if refresh['allowed']:
                            memory = transport.request(root,tag(task,arm),'edit',query,prompts.MEMORY_SYSTEM,
                                prompts.memory_public(pool,state,ids,query,refresh),
                                lambda raw:policy.apply_edits(raw,pool,state,query))
                            notes = memory['plan']['notes']
                        reasoning = transport.request(root,tag(task,arm),'choose',query,prompts.REASONING_SYSTEM,
                            prompts.reasoning_public(root,pool,state,notes,ids,query),
                            lambda raw:policy.validate_decision(root,pool,notes,state,context,ids,query,raw),
                            selected_validator=lambda raw:validate_insight(root,raw['insight'],ids))
                        decision = reasoning['plan']
                        selected = decision['selected_id']
                    elif arm=='reasoning_bo':
                        reasoning = transport.request(root,tag(task,arm),'choose',query,prompts.RBO_SYSTEM,
                            prompts.rbo_public(root,pool,state,ids,query),
                            lambda raw:dict(selected_id=validate_insight(root,raw,ids)),
                            selected_validator=lambda raw:validate_insight(root,raw,ids))
                        selected = reasoning['plan']['selected_id']
                    else:
                        selected = pool.ids[int(original[0])]
                    commit = dict(query=query,selected_id=selected,prestate_sha256=control.digest(state),
                        protocol_sha256=protocol['protocol_sha256'],BO_reference_id=pool.ids[int(original[0])],
                        GP_fit_key=context['stats']['fit_key'],GP_sha256=native.gp_hash(context),
                        ordered_ids=[pool.ids[int(j)] for j in original],display_ids=ids,
                        stage=context['stage'],qLogEI_defined=context['fitted_GP'],
                        opportunity_gap=float(context['base_logei'][original[0]]-context['base_logei'][pool.ix[selected]]) if context['fitted_GP'] else None,
                        edited_notes=notes,decision=decision,refresh=refresh,memory=memory,reasoning=reasoning)
                    commit['commit_sha256'] = control.digest(commit)
                    _save_context(folder,query,context)
                    transport.write_once(cp,commit)
                if commit['prestate_sha256']!=control.digest(state):
                    raise ValueError('Committed state changed before oracle')
                observation = native_gp.read_observation(root,pool,commit['selected_id'])
                state = feedback(state,commit,observation)
                warm.write(sf,state)
                warm.write(folder / 'runtime_counters.json',gp.counters)
                gp.release()
                print(json.dumps(dict(task=tag(task,arm),query=query,objective=observation['raw_value'],
                    best=max(r['raw_value'] for r in state['history'])),ensure_ascii=False),flush=True)
        except Exception as exc:
            warm.write(folder / 'execution_error.json',dict(type=type(exc).__name__,message=str(exc),query=len(state['history'])+1))
            raise
        finally:
            gp.release()
    result = dict(task=task,arm=arm,status='COMPLETE',queries=QUERIES)
    transport.write_once(folder / 'completion.json',result)
    print('FINISH '+json.dumps(result),flush=True)
    return result


def batch(root, workers=3):
    check(root,True)
    if transport.read(directory(root) / 'preflight.json')['status']!='PASS':
        raise ValueError('Cold-start preflight required')
    errors = []
    with ProcessPoolExecutor(max_workers=min(workers,3)) as executor:
        futures = {executor.submit(run_one,str(root),*job):job for job in JOBS}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                row = dict(job=futures[future],type=type(exc).__name__,message=str(exc))
                errors.append(row)
                print('JOB_ERROR '+json.dumps(row),flush=True)
    check(root,True)
    transport.write_once(directory(root) / 'batch_receipt.json',dict(status='INCOMPLETE' if errors else 'COMPLETE',errors=errors))
    if errors:
        raise RuntimeError('Preserved failed prefixes: '+json.dumps(errors))


def status(root):
    rows = []
    for task,arm in JOBS:
        path = run_path(root,task,arm) / 'state.json'
        state = transport.read(path) if path.exists() else {}
        rows.append(dict(task=SHORT[task],arm=arm,status=state.get('status','NOT_STARTED'),queries=len(state.get('history',[]))))
    return dict(complete=sum(r['status']=='COMPLETE' for r in rows),queries=sum(r['queries'] for r in rows),rows=rows)
