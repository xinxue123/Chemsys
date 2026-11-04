"""Seed-zero, eight-task isolated experiment. Commit before every oracle read."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import shutil

import numpy as np
from filelock import FileLock
from chem.benchmarks import knowledge_gp as gpmod, knowledge_study as old
from chem.benchmarks.knowledge_llm import write
from knowledge_rank_v9 import study as v9, engine as native
from reasoning_memory_v12 import study as v12
from reasoning_memory_v12.llm import validate_insight, compiled_extractor
from reasoning_memory_v13 import control, adapter
from . import VARIANT, MODEL, EFFORT, BACKEND, policy, prompts, transport

SHORT = dict(zip(gpmod.TASKS, ('c1', 'c2', 'c3', 'c4', 'perera', 'doyle', 'snar', 'hplc')))
DESIGN = 'docs/v13_state_split_experiment_20261102.txt'


def sources(root):
    root = Path(root)
    files = v12.sources(root)
    for folder in ('reasoning_memory_v13', 'state_split_v13_experiment'):
        files.update({p.relative_to(root).as_posix(): old.sha(p)
                      for p in (root / 'code/experiment_tools' / folder).rglob('*.py')})
    for name in ('code/experiment_tools/run_state_split_v13.py', 'code/experiment_tools/paper_v12_subagent/transport.py',
                 'code/experiment_tools/paper_v12_subagent/__init__.py', DESIGN):
        files[name] = old.sha(root / name)
    return files


def controls(root):
    root = Path(root)
    result = {}
    for task in gpmod.TASKS:
        paths = {arm: v9.directory(root) / 'runs' / f'{task}_{arm}/state.json' for arm in ('bo', 'reasoning_bo')}
        paths['v12_subagent'] = root / 'experiments/runs/paper_v12_subagent/s0' / SHORT[task] / 'memory/runs' / f'{task}_memory/state.json'
        api = v12.directory(root) / 'runs' / f'{task}_memory/state.json'
        if api.exists():
            paths['v12_api'] = api
        result[task] = {name: dict(path=p.relative_to(root).as_posix(), sha256=old.sha(p)) for name, p in paths.items()}
    return result


def check(root, verify_controls=False):
    root = Path(root)
    protocol = old.read(transport.base(root) / 'protocol.json')
    if protocol['protocol_sha256'] != control.digest({k: v for k, v in protocol.items() if k != 'protocol_sha256'}):
        raise ValueError('Protocol changed')
    if protocol['sources_sha256'] != sources(root):
        raise ValueError('Registered source changed')
    if protocol['tasks'] != list(gpmod.TASKS) or (protocol['seed'], protocol['initial'], protocol['queries']) != (0, 4, 16):
        raise ValueError('Experiment scope changed')
    for row in protocol['inputs']:
        for path, sha in row['files_sha256'].items():
            if old.sha(root / path) != sha:
                raise ValueError('Input data changed')
    if verify_controls and protocol['controls'] != controls(root):
        raise ValueError('Frozen control states changed')
    return protocol


def prepare(root):
    root = Path(root).resolve()
    base = transport.base(root)
    if (base / 'protocol.json').exists():
        return check(root, True)
    if (base / 'runs').exists() or (base / 'requests').exists():
        raise ValueError('Registration must precede experimental queries')
    prior = old.read(v9.directory(root) / 'protocol.json')
    p = dict(variant=VARIANT, tasks=list(gpmod.TASKS), seed=0, trial_seed=0, tie_seed=0,
             numerical_seed=0, menu_seed=0, initial=4, queries=16, model=MODEL, effort=EFFORT,
             backend=BACKEND, fork_turns='none', created_utc=datetime.now(timezone.utc).isoformat(),
             parameters=policy.PARAMETERS, sources_sha256=sources(root), inputs=prior['inputs'],
             controls=controls(root), knowledge_condition='own_data_agent_initial_no_artificial_priors',
             prompt_sha256=dict(editor=control.digest(prompts.MEMORY_SYSTEM), reasoning=control.digest(prompts.REASONING_SYSTEM)),
             acceptance='Report all eight task SearchAUC differences; no post-hoc trajectory selection or promotion',
             inference_limit='One initialization and one fresh LLM realization per task; controls have separately labeled backends')
    p['protocol_sha256'] = control.digest(p)
    transport.write_once(base / 'protocol.json', p)
    snapshot = root / 'runtime/knowledge_loop/snapshots' / VARIANT
    for name, sha in p['sources_sha256'].items():
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target)
        if old.sha(target) != sha:
            raise ValueError('Snapshot drift')
    transport.write_once(snapshot / 'protocol.json', p)
    return check(root, True)


def copy_cache(root, task):
    # Only numerical caches are reused; no historical response is consulted.
    source = v9.directory(root) / 'cache' / task
    target = transport.base(root) / 'cache' / task
    count = 0
    for path in source.rglob('*'):
        if path.is_file() and path.suffix in ('.pt', '.npz', '.json'):
            dest = target / path.relative_to(source)
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
                count += 1
            elif old.sha(dest) != old.sha(path):
                raise ValueError('Native cache conflict')
    return count


def feedback(state, commit, observation):
    if (commit['commit_sha256'] != control.digest({k: v for k, v in commit.items() if k != 'commit_sha256'}) or
            commit['prestate_sha256'] != control.digest(state)):
        raise ValueError('Pre-observation commitment mismatch')
    notes, events = adapter.apply_feedback(commit['edited_notes'], state['history'], commit['decision'], observation)
    out = deepcopy(state)
    before = max(r['y'] for r in state['history'])
    out['notes'] = notes
    out['history'].append(observation)
    out['insights'].append(dict(query=commit['query'], insight=commit['decision']['insight']))
    if commit['refresh']['allowed']:
        out['edit_calls'].append(commit['query'])
    if any(r['use'] == 'review_test' for r in commit['decision']['knowledge_uses']):
        out['review_queries'].append(commit['query'])
    out['decisions'].append(dict(query=commit['query'], selected_id=observation['candidate_id'],
        BO_reference_id=commit['BO_reference_id'], raw_value=observation['raw_value'], y=observation['y'],
        incumbent_before_y=before, incumbent_after_y=max(before, observation['y']),
        feedback=events, knowledge_uses=commit['decision']['knowledge_uses'],
        commit_sha256=commit['commit_sha256'], mode='base_bo' if observation['candidate_id'] == commit['BO_reference_id'] else 'ReasoningBO_split_state'))
    out['status'] = 'COMPLETE' if len(out['decisions']) == 16 else 'RUNNING'
    return out


def run_one(root, task):
    root = Path(root).resolve()
    os.chdir(root / 'experiments/chemist_system')
    p = check(root)
    pool = gpmod.load_task(root, task, seed=0)
    if task not in SHORT:
        raise ValueError('Unregistered task')
    run = transport.base(root) / 'runs' / SHORT[task]
    run.mkdir(parents=True, exist_ok=True)
    with FileLock(str(run / '.worker.lock'), timeout=1):
        sf = run / 'state.json'
        init = old.initial_commit(run, pool, p, 'state_split', 'agent_initial_terra', create=not sf.exists())
        if sf.exists():
            state = old.read(sf)
            if state['protocol_sha256'] != p['protocol_sha256'] or state['initial_commit_sha256'] != init['commit_sha256']:
                raise ValueError('Resume mismatch')
        else:
            state = dict(task=task, arm='state_split', seed=0, protocol_sha256=p['protocol_sha256'],
                         initial_commit_sha256=init['commit_sha256'], status='READY',
                         history=[gpmod.read_observation(root, pool, key) for key in pool.initial_ids],
                         notes=[], decisions=[], insights=[], edit_calls=[], review_queries=[])
            reference = old.read(root / p['controls'][task]['bo']['path'])
            if state['history'] != reference['history'][:4]:
                raise ValueError('Initialization differs from frozen pure BO')
            write(sf, state)
        if state['status'] == 'COMPLETE':
            return dict(task=task, status='COMPLETE', queries=16)
        copy_cache(root, task)
        gp = gpmod.GPAdapter(pool, transport.base(root) / 'cache' / task, seed=0)
        compiled_extractor(str(root))  # one compile per worker, before round loop
        try:
            while len(state['decisions']) < 16:
                query = len(state['decisions']) + 1
                cp = run / 'rounds' / f'{query:02d}_commit.json'
                if cp.exists():
                    commit = old.read(cp)
                else:
                    context = gp.context(state['history'], query)
                    if query == 1:
                        path = (root / p['controls'][task]['bo']['path']).parent / 'rounds/01_GP.npz'
                        with np.load(path, allow_pickle=False) as saved:
                            for key in ('mu_y', 'sd_y', 'sd_predictive_y'):
                                np.testing.assert_array_equal(saved[key], context['stats'][key])
                            np.testing.assert_array_equal(saved['base_logei'], context['base_logei'])
                        transport.write_once(run / 'initial_GP_check.json', dict(status='PASS_EXACT', original_file=str(path)))
                    original, display = native.candidates(pool, context, query)
                    ids = [pool.ids[int(j)] for j in display]
                    refresh = policy.edit_event(pool, state, original, query)
                    memory = None
                    if refresh['allowed']:
                        memory = transport.request(root, SHORT[task], 'edit', query, prompts.MEMORY_SYSTEM,
                            prompts.memory_public(pool, state, ids, query, refresh),
                            lambda raw: policy.apply_edits(raw, pool, state, query))
                    notes = memory['plan']['notes'] if memory else state['notes']
                    response = transport.request(root, SHORT[task], 'choose', query, prompts.REASONING_SYSTEM,
                        prompts.reasoning_public(root, pool, state, notes, ids, query),
                        lambda raw: policy.validate_decision(root, pool, notes, state, context, ids, query, raw),
                        selected_validator=lambda raw: validate_insight(root, raw['insight'], ids))
                    packet = response['plan']
                    commit = dict(query=query, selected_id=packet['selected_id'], prestate_sha256=control.digest(state),
                                  protocol_sha256=p['protocol_sha256'], BO_reference_id=pool.ids[int(original[0])],
                                  GP_fit_key=context['stats']['fit_key'], GP_sha256=native.gp_hash(context),
                                  ordered_ids=[pool.ids[int(j)] for j in original], display_ids=ids,
                                  opportunity_gap=float(context['base_logei'][original[0]]-context['base_logei'][pool.ix[packet['selected_id']]]),
                                  edited_notes=notes, decision=packet, refresh=refresh,
                                  memory=memory, reasoning=response)
                    commit['commit_sha256'] = control.digest(commit)
                    cp.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(cp.parent / f'{query:02d}_GP.npz',
                        **{k: context['stats'][k] for k in ('mu_y', 'sd_y', 'sd_predictive_y')},
                        **{k: context[k] for k in ('base_logei', 'available', 'tie_priority')})
                    transport.write_once(cp.parent / f'{query:02d}_fit.json', context['stats']['fit_parameters'])
                    transport.write_once(cp, commit)
                if commit['prestate_sha256'] != control.digest(state):
                    raise ValueError('Committed prestate changed before oracle')
                observation = gpmod.read_observation(root, pool, commit['selected_id'])
                state = feedback(state, commit, observation)
                write(sf, state)
                write(run / 'runtime_counters.json', gp.counters)
                gp.release()
                print(json.dumps(dict(task=SHORT[task], query=query, objective=observation['raw_value'],
                    best=max(r['raw_value'] for r in state['history']), feedback=len(state['decisions'][-1]['feedback'])), ensure_ascii=False), flush=True)
        except Exception as error:
            write(run / 'execution_error.json', dict(type=type(error).__name__, message=str(error), query=len(state['decisions'])+1))
            raise
        finally:
            gp.release()
        check(root, True)
        return dict(task=task, status='COMPLETE', queries=16)


def batch(root, workers=3):
    check(root, True)
    with ProcessPoolExecutor(max_workers=min(workers, 8)) as executor:
        futures = [executor.submit(run_one, str(root), task) for task in gpmod.TASKS]
        for f in as_completed(futures):
            print('FINISH ' + json.dumps(f.result()), flush=True)


def status(root):
    rows = []
    for task, short in SHORT.items():
        sf = transport.base(root) / 'runs' / short / 'state.json'
        s = old.read(sf) if sf.exists() else {}
        rows.append(dict(task=short, status=s.get('status', 'NOT_STARTED'), queries=len(s.get('decisions', []))))
    return dict(complete=sum(r['status'] == 'COMPLETE' for r in rows), queries=sum(r['queries'] for r in rows),
                rows=rows, pending=transport.pending(root))
