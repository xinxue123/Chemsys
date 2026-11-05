"""Durable, isolated knowledge-loop experiments with exact shared BO controls."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import os
import time
import numpy as np

from . import knowledge_gp as gpmod
from . import knowledge_engine as engine
from . import knowledge_llm as llm
from . import knowledge_openai_client as api
from . import knowledge_priors as priors
from . import knowledge_policy as policy


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def paths(root, variant):
    if not variant or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in variant):
        raise ValueError('simple variant name required')
    return Path(root) / 'experiments/runs/knowledge_loop_20261101' / variant


def source_hashes(root):
    code = Path(root) / 'code/chemist_system'
    files = list((code / 'chem').rglob('*.py')) + [code / 'knowledge_loop.py']
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(files)}


def _trial_seed(protocol, requested=None):
    seed = gpmod.validate_trial_seed(protocol.get('trial_seed', protocol['seed']))
    if protocol['seed'] != seed or requested is not None and gpmod.validate_trial_seed(requested) != seed:
        raise ValueError('Requested seed differs from frozen protocol')
    if (protocol.get('tie_seed', seed) != seed or protocol.get('numerical_seed', 0) != 0 or
            protocol.get('menu_seed', 0) != 0):
        raise ValueError('Unregistered random-stream policy')
    return seed


def _confirmation_inputs(root):
    registry, rule = gpmod.seed1_registration(root)
    # Only the adapter and runner may change; the V3 mechanism/prompt stay fixed.
    for rel, expected in rule['core_source_sha256'].items():
        if rel.endswith('/knowledge_gp.py'):
            continue
        if sha(Path(root) / rel) != expected:
            raise ValueError('V3 policy or prompt changed after seed1 registration')
    if llm.digest(llm.SYSTEM) != rule['prompt_sha256']:
        raise ValueError('Registered V3 system prompt changed')
    rows = []
    for row in registry['tasks']:
        pool = gpmod.load_task(root, row['task'], seed=1)
        rows.append(dict(task=pool.task, group=pool.group, candidate_count=len(pool.ids),
            initial_ids=list(pool.initial_ids), initial_features_sha256=row['initial_features_sha256'],
            unit=pool.unit, knowledge_conditions=['agent_initial_terra'],
            files_sha256={row['input_pool']: row['input_pool_sha256']}))
    return rows


def prepare(root, variant, seed=0):
    root = Path(root).resolve()
    gpmod.validate_trial_seed(seed)
    if (seed == 1) != (variant == 'v3_seed1_confirm'):
        raise ValueError('Seed1 requires the reserved v3_seed1_confirm variant')
    # The workspace deliberately maps experiments/runs to the archive tree.
    # Resolve that existing mapping, then reject escapes from the workspace.
    directory = paths(root, variant).resolve()
    if not directory.is_relative_to(root):
        raise ValueError('Protocol path redirects outside the workspace')
    if seed == 1:
        inputs = _confirmation_inputs(root)
    else:
        manifest = Path(root) / 'experiments/reports/knowledge_loop_20261101/baseline_inventory.json'
        inputs = read(manifest)['tasks']
    source = source_hashes(root)
    protocol = dict(variant=variant, source_sha256=source, inputs=inputs,
        goal_config_path='code/resources/knowledge_loop_goal_20261101_majority.json',
        goal_config_sha256=sha(Path(root) / 'code/resources/knowledge_loop_goal_20261101_majority.json'),
        parameters=engine.PARAMETERS, model=api.MODEL, reasoning_effort=api.REASONING_EFFORT,
        seed=seed, trial_seed=seed, tie_seed=seed, numerical_seed=0, menu_seed=0,
        numerical_seed_policy='P0 native cache seed0; P1/P2 existing fixed formulas',
        initial_registry_path=gpmod.SEED1_REGISTRY_PATH if seed else None,
        initial_registry_sha256=gpmod.SEED1_REGISTRY_SHA256 if seed else None,
        initial_registry_file_sha256=sha(Path(root)/gpmod.SEED1_REGISTRY_PATH) if seed else None,
        initial_rule_path=gpmod.SEED1_RULE_PATH if seed else None,
        initial_rule_sha256=gpmod.SEED1_RULE_SHA256 if seed else None,
        initial=4, queries=16, tasks=list(gpmod.TASKS),
        GP='archived numerical recipes per task; identical BO/agent; no structural-kernel rescue',
        tie='exact qLogEI then PCG64 mapped through sorted physical features',
        covariance='joint latent GP in engineering y units; predictive noise for evidence',
        stage='development; seed0 is not independent confirmation',
        supported_conditions=['agent_initial_terra', *priors.CONDITIONS],
        synthetic_prior_scope='P0 only; each historical region clause instantiated as one real unqueried pair; zero initial evidence weight',
        pending_conditions=[],
        baseline='new matched BO per task; historical values retained separately',
        no_protected_prefix=True, prompt_sha256=llm.digest(llm.SYSTEM))
    protocol.update(prompt_state='current_round_coverage', domain_context='public_process_units_v1',
        design_change='Public domain/units plus explicit mechanistic-prior permission and current coverage state; compared with v2 deployment, no new field mechanism',
        development_stage='Eight real-agent tasks first, full versus deployment-only; expand the globally fixed candidate to pressure conditions if informative',
        acceptance='User revised: majority superiority (at least5 of8 real tasks strictly better SearchAUC); final best and all losses disclosed; stress diagnostics separate')
    if seed == 1:
        protocol.update(stage='preregistered additional paired seed1 confirmation; seed0 primary unchanged',
            supported_conditions=['agent_initial_terra'], supported_modes=['bo', 'full'],
            synthetic_prior_scope='not included in the independent paired confirmation',
            initial_commit='input-only initial_commit.json flushed and fsynced before initial outcome reads',
            design_change='Only registered initial4 and tie stream change; V3 full policy, prompt, menu RNG and numerical GP fixed',
            development_stage='frozen V3 full versus matched BO; one registered seed; no selection after outcomes')
    protocol['protocol_sha256'] = llm.digest(protocol)
    dest = directory / 'protocol.json'
    if dest.exists():
        assert read(dest) == protocol, 'Frozen variant changed; create a new variant instead'
    else:
        llm.write(dest, protocol)
    return protocol


def check(root, variant):
    p = read(paths(root, variant) / 'protocol.json')
    assert p['protocol_sha256'] == llm.digest({k:v for k,v in p.items() if k != 'protocol_sha256'})
    assert p['source_sha256'] == source_hashes(root), 'Frozen code changed'
    assert p['goal_config_sha256'] == sha(Path(root)/p.get('goal_config_path','code/resources/knowledge_loop_goal_20261101.json')), 'Goal configuration changed'
    seed = _trial_seed(p)
    if seed == 1:
        if (variant != 'v3_seed1_confirm' or p['initial_registry_path'] != gpmod.SEED1_REGISTRY_PATH or
                p['initial_registry_sha256'] != gpmod.SEED1_REGISTRY_SHA256 or
                p['initial_rule_path'] != gpmod.SEED1_RULE_PATH or p['initial_rule_sha256'] != gpmod.SEED1_RULE_SHA256 or
                p['initial_registry_file_sha256'] != sha(Path(root)/gpmod.SEED1_REGISTRY_PATH) or
                p['supported_modes'] != ['bo', 'full'] or p['supported_conditions'] != ['agent_initial_terra'] or
                p['inputs'] != _confirmation_inputs(root)):
            raise ValueError('Independent confirmation registration changed')
    for task in p['inputs']:
        for rel, expected in task['files_sha256'].items():
            assert sha(Path(root)/rel) == expected, rel
    return p


def initial_commit(directory, pool, protocol, mode, condition, *, create):
    """Atomically persist the input-only four-point commitment before revelation."""
    commit = dict(task=pool.task, mode=mode, condition=condition, seed=_trial_seed(protocol),
        protocol_sha256=protocol['protocol_sha256'],
        initial_registry_sha256=protocol.get('initial_registry_sha256'),
        initial_ids=list(pool.initial_ids),
        initial_points=[pool.points[pool.ix[key]] for key in pool.initial_ids],
        initial_features=[list(pool.features[pool.ix[key]]) for key in pool.initial_ids],
        input_pool_sha256=sha(pool.source_path))
    commit['commit_sha256'] = llm.digest(commit)
    path = Path(directory) / 'initial_commit.json'
    if path.exists():
        if read(path) != commit:
            raise ValueError('Initial commitment changed')
    elif not create:
        raise ValueError('Initial commitment missing from resumed run')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.json.tmp')
        with temporary.open('w', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(commit, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    return commit


def prepare_prompt_state(pool, state, context, t):
    """Synchronize current coverage without mutating the durable pre-query state."""
    prepared = deepcopy(state)
    prepared['hypotheses'], events = policy.refresh_coverage(pool, state['hypotheses'],
        np.flatnonzero(context['eligible']), t)
    return prepared, events


def run_one(root, variant, task, mode='full', condition='agent_initial_terra', seed=None):
    root = Path(root).resolve()
    p = check(root, variant)
    trial = _trial_seed(p, seed)
    pool = gpmod.load_task(root, task, seed=trial)
    if mode not in p.get('supported_modes', ['bo', 'shadow', 'deployment', 'full']):
        raise ValueError('Mode is not registered for this variant')
    if condition not in p['supported_conditions'] or (condition != 'agent_initial_terra' and pool.group != 'P0'):
        raise ValueError('Unknown condition or synthetic prior outside P0')
    if mode == 'bo' and condition != 'agent_initial_terra':
        raise ValueError('Use the single common BO control for every knowledge condition')
    variant_root = paths(root, variant).resolve()
    if not variant_root.is_relative_to(root):
        raise ValueError('Variant path redirects outside the workspace')
    directory = variant_root / 'runs' / f'{pool.task}_{mode}_{condition}'
    cache_directory = variant_root / 'cache' / pool.task
    if (not directory.resolve().is_relative_to(variant_root / 'runs') or
            not cache_directory.resolve().is_relative_to(variant_root / 'cache')):
        raise ValueError('Run/cache path escapes registered variant')
    statefile = directory / 'state.json'
    if statefile.exists():
        state = read(statefile)
        assert state['protocol_sha256'] == p['protocol_sha256']
        if state['seed'] != trial or [h['candidate_id'] for h in state['history'][:4]] != list(pool.initial_ids):
            raise ValueError('Resumed initialization/seed differs from registration')
        if trial == 1 or 'initial_commit_sha256' in state:
            init = initial_commit(directory, pool, p, mode, condition, create=False)
            if state.get('initial_commit_sha256') != init['commit_sha256']:
                raise ValueError('Resumed initial commitment hash mismatch')
    else:
        init = initial_commit(directory, pool, p, mode, condition, create=True)
        state = dict(task=pool.task, mode=mode, condition=condition, seed=trial,
            initial_commit_sha256=init['commit_sha256'],
            history=[gpmod.read_observation(root, pool, key) for key in pool.initial_ids],
            hypotheses=[], calls=[], decisions=[], verification_used=0, status='READY',
            protocol_sha256=p['protocol_sha256'])
        if condition != 'agent_initial_terra':
            state['hypotheses'], prior_receipt = priors.build_priors(pool, condition, pool.initial_ids)
            llm.write(directory/'prior_receipt.json', prior_receipt)
            state['prior_receipt_sha256'] = llm.digest(prior_receipt)
        llm.write(statefile, state)
    if state['status'] == 'COMPLETE':
        return summarize(state)
    cache = gpmod.GPAdapter(pool, cache_directory, seed=trial)
    engine.warm()
    client = api.APIClient(api.load_config(root/'openai-key.txt')) if mode != 'bo' else None
    try:
        while len(state['decisions']) < 16:
            t = len(state['decisions']) + 1
            cp = directory / 'rounds' / f'{t:02d}_commit.json'
            if cp.exists():
                commit = read(cp)
                assert commit['history_sha256'] == llm.digest(state['history'])
                assert commit['protocol_sha256'] == p['protocol_sha256']
            else:
                context = cache.context(state['history'], t)
                prepared, pre_call_events = prepare_prompt_state(pool, state, context, t)
                rows, _ = engine.coverage(pool, cache, context, prepared['hypotheses'], prepared['history'])
                proposal, receipt, menu = None, None, []
                if mode != 'bo' and engine.should_call(prepared, t, rows):
                    menu = llm.make_menu(pool, context, t, seed=0)
                    if len(menu) >= 2:
                        public = llm.public_payload(pool, prepared, t, menu, rows)
                        receipt = llm.request(client, public, paths(root, variant)/'requests')
                        proposal = receipt['proposal']
                commit = engine.plan(pool, cache, prepared, t, context, proposal, menu, mode)
                calls = deepcopy(state['calls'])
                if receipt:
                    calls.append(dict(query=t, context_sha256=receipt['context_sha256'], receipt=receipt))
                commit.update(protocol_sha256=p['protocol_sha256'], calls_at_commit=calls,
                    pre_call_coverage_events=pre_call_events)
                cp.parent.mkdir(parents=True, exist_ok=True)
                with (cp.parent / f'{t:02d}_GP.npz').open('wb') as f:
                    np.savez_compressed(f, mu_y=context['stats']['mu_y'], sd_y=context['stats']['sd_y'],
                        sd_predictive_y=context['stats']['sd_predictive_y'], base_logei=context['base_logei'],
                        available=context['available'], eligible=context['eligible'], tie_priority=context['tie_priority'])
                llm.write(cp.parent / f'{t:02d}_fit.json', context['stats']['fit_parameters'])
                llm.write(cp, commit)
            # The durable commitment is the sole authorization to reveal a new y.
            assert cp.exists() and commit['selected_id'] not in {h['candidate_id'] for h in state['history']}
            observation = gpmod.read_observation(root, pool, commit['selected_id'])
            state['calls'] = deepcopy(commit['calls_at_commit'])
            engine.feedback(state, commit, observation)
            state['cache_counters'] = cache.counters
            llm.write(statefile, state)
            print(json.dumps(dict(task=pool.task, mode=mode, condition=condition, query=t, selection=commit['mode'],
                objective=observation['raw_value'], calls=len(state['calls']),
                supported=sum(r['eligible_inside_scope'] > 0 for r in commit['coverage'])), ensure_ascii=False), flush=True)
            cache.release()
    except Exception as error:
        state['status'] = 'INTERRUPTED'
        # API exceptions are already safe; never include arbitrary response/credential content.
        state['error'] = dict(kind=type(error).__name__, message=str(error) if isinstance(error, api.OpenAIClientError) else 'See local exception type and verify frozen runner')
        llm.write(statefile, state)
        raise
    finally:
        if client:
            client.close()
    check(root, variant)
    return summarize(state)


def summarize(state):
    values = np.array([h['raw_value'] for h in state['history']])
    curve = np.maximum.accumulate(values)
    return dict(task=state['task'], mode=state['mode'], condition=state['condition'], status=state['status'],
        queries=len(state['decisions']), SearchAUC=float(curve[4:].mean()) if len(curve)>4 else None,
        final_best=float(curve[-1]), first_best_total=int(np.flatnonzero(values==curve[-1])[0])+1,
        knowledge_choices=sum(d['mode']=='knowledge_guided' for d in state['decisions']),
        optional_verifications=state['verification_used'], calls=len(state['calls']))


def batch(root, variant, tasks, mode, workers=3, condition='agent_initial_terra', seed=None):
    _trial_seed(check(root, variant), seed)
    if mode != 'bo':
        for task in tasks:
            baseline = paths(root, variant)/'runs'/f'{task}_bo_agent_initial_terra'/'state.json'
            assert baseline.exists() and read(baseline)['status']=='COMPLETE', 'Run matched BO first'
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for future in as_completed([executor.submit(run_one, str(root), variant, task, mode, condition, seed) for task in tasks]):
            print('FINISH ' + json.dumps(future.result(), ensure_ascii=False), flush=True)


def status(root, variant):
    rows = [summarize(read(p)) for p in sorted((paths(root, variant)/'runs').glob('*/state.json'))]
    return dict(runs=rows, complete=sum(r['status']=='COMPLETE' for r in rows))
