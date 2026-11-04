"""Frozen follow-up for the program-side KF experiment."""
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
import csv
import json
import os
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/chemist_system'), str(ROOT / 'code/experiment_tools')]
from program_kf_weight import study as previous, policy, prompts, MODEL, EFFORT
from v13_late_validation import study as late
from v13_origin_ablation import queue
from cold_start_v13 import control
from cold_start_v13 import prompts as native_prompts
from reasoning_memory_v12.llm import validate_insight
from reasoning_memory_v13 import calibration
from knowledge_rank_v9 import engine
from chem.benchmarks import knowledge_gp

VARIANT = 'program_kf_confirmation_20261104'
BASE = ROOT / 'experiments/runs' / VARIANT
DOC = ROOT / 'docs' / (VARIANT + '.txt')
CELLS = (('Perera_Suzuki', 'perera', 1), ('HPLC', 'hplc', 1), ('Direct_Arylation', 'aryl', 0))
END = 10
read, write, sha = queue.read, queue.write_once, queue.file_sha
native_request = queue.request
native_pool = previous.frozen.pool_for
PARENT = '/root/kf_confirmation_dispatch'


def scope(job=None):
    stack = ExitStack()
    stack.enter_context(patch.object(previous, 'base', return_value=BASE))
    stack.enter_context(patch.object(previous, 'VARIANT', VARIANT))
    stack.enter_context(patch.object(previous.frozen, 'pool_for', side_effect=lambda root, task: pool_for(task)))
    stack.enter_context(patch.object(native_prompts, 'task_context', side_effect=late.domain))
    stack.enter_context(patch.object(queue, 'request', side_effect=request))
    stack.enter_context(previous.transport_scope(job))
    return stack


def pool_for(task):
    return late.pool_for(task) if task == 'Direct_Arylation' else native_pool(ROOT, task)


def rel(path):
    return previous.rel(path)


def register():
    if (BASE / 'protocol.json').exists():
        return verify()
    old = previous.check(True)
    protected = dict(old['protected_sha256'])
    source = dict(old['sources_sha256'])
    inputs = dict(old['inputs_sha256'])
    for p in (Path(__file__), DOC, Path(late.__file__)):
        source[rel(p)] = sha(p)
    for name in ('protocol.json', 'prefix_complete.json', 'traces_complete.json'):
        p = previous.base() / name
        protected[rel(p)] = sha(p)
    aryl = late.base() / 'holdout'
    for p in (aryl / 'pool.json', aryl / 'oracle.npz', aryl / 'receipt.json'):
        inputs[rel(p)] = sha(p)
    with (ROOT / 'paper/v13_late/data/metrics.csv').open(encoding='utf-8-sig', newline='') as f:
        historical = list(csv.DictReader(f))
    with (ROOT / 'paper/v13_late/data/final_metrics.csv').open(encoding='utf-8-sig', newline='') as f:
        transfer = list(csv.DictReader(f))
    jobs, refs = [], {}
    for task, short, seed in CELLS:
        cell = f'{short}{seed}'
        for arm in ('v13', 'bo', 'reasoning_bo'):
            if short == 'aryl':
                legacy_arm = {'v13': 'full', 'bo': 'bo', 'reasoning_bo': 'rbo'}[arm]
                row = next(r for r in transfer if r['job'] == f'h_aryl{seed}_{legacy_arm}')
            else:
                row = next(r for r in historical if r['regime'] == 'cold' and r['task'] == short
                           and int(r['seed']) == seed and r['method'] == arm)
            path = ROOT / row['state_path']
            if sha(path) != row['state_sha256']:
                raise ValueError(f'Historical state differs: {path}')
            refs[f'{cell}_{arm}'] = rel(path)
            protected[rel(path)] = sha(path)
            if arm == 'v13':
                for q in range(1, END + 1):
                    for suffix in ('commit.json', 'fit.json', 'GP.npz'):
                        p = path.parent / 'rounds' / f'{q:02}_{suffix}'
                        protected[rel(p)] = sha(p)
        for arm, phase in (('diagnostic', 'prefix'), ('fixed', 'traces'), ('dynamic', 'traces')):
            jobs.append(dict(id=f'{phase}_{cell}_{arm}', cell=cell, task=task, seed=seed, arm=arm, phase=phase))
    p = dict(variant=VARIANT, timestamp_utc=datetime.now(timezone.utc).isoformat(), jobs=jobs,
             references=refs, sources_sha256=source, inputs_sha256=inputs,
             protected_sha256=protected, initial=0, queries=20, new_trajectories=6,
             prefix_rounds=list(range(2, END + 1)), new_online_queries=120,
             model=MODEL, effort=EFFORT, responder_parent=PARENT,
             prior_protocol_sha256=old['protocol_sha256'], new_llm_calls_at_registration=0,
             design=DOC.read_text(encoding='utf-8-sig'))
    p['protocol_sha256'] = control.digest(p)
    write(BASE / 'protocol.json', p)
    return verify()


def verify():
    with scope():
        return previous.check(True)


def request(root, tag, kind, q, system, public, validator, **kw):
    """Normalize only non-applicable Note forecast metadata; keep raw evidence."""
    if kind != 'choose':
        return native_request(root, tag, kind, q, system, public, validator, **kw)
    ids = [row['candidate_id'] for row in public['candidate_menu']]
    normalize_log = BASE / 'normalization' / tag / f'{q:02}.json'

    def validate(raw):
        chosen = validate_insight(root, raw['insight'], ids)
        active = {(n['id'], n['version']) for n in public['retrieved_notes']['knowledge']
                  if chosen in n['covered_candidate_ids'] and n['recommendation']['permission'] != 'withdrawn'}
        assessments = raw['assessments']
        if not isinstance(assessments, list):
            return validator(raw)
        kept = [a for a in assessments if not isinstance(a, dict)
                or type(a.get('id')) is not str or type(a.get('version')) is not int
                or (a['id'], a['version']) in active]
        plan = validator(dict(raw, assessments=kept))
        if kept != assessments:
            write(normalize_log, dict(original_sha256=control.digest(raw), chosen=chosen,
                                      removed=[a for a in assessments if a not in kept],
                                      forecasts_preserved=True))
        return plan

    return native_request(root, tag, kind, q, system, public, validate, **kw)


def rank_opportunity(fixed, pool, ctx, ids, state):
    g = fixed['ranking']['GP_rank_scores']
    llm = fixed['ranking']['LLM_rank_scores']
    if fixed['ranking']['LLM_all_tied']:
        return False
    # An affine upper envelope changes winner only at intersections.
    slopes = [b - a for a, b in zip(g, llm)]
    ends = {0., .8}
    for i in range(len(g)):
        for j in range(i):
            if slopes[i] != slopes[j]:
                w = (g[j] - g[i]) / (slopes[i] - slopes[j])
                if 0 <= w <= .8:
                    ends.add(w)
    ends = sorted(ends)
    test = ends + [(a + b) / 2 for a, b in zip(ends, ends[1:])]
    return any(policy.rank(pool, ctx, ids, fixed['probabilities'], w,
                           fixed['proposed_id'], state['history'])['selected_id']
               != fixed['selected_id'] for w in test)


def prefix(job, p):
    path = ROOT / p['references'][job['cell'] + '_v13']
    old = read(path)
    pool = pool_for(job['task'])
    folder = BASE / 'prefix' / job['cell']
    previous.compiled_extractor(str(ROOT))
    kf = calibration.initial()
    for q in p['prefix_rounds']:
        ctx = previous.saved_context(path.parent, q)
        saved = read(path.parent / 'rounds' / f'{q:02}_commit.json')
        if engine.gp_hash(ctx) != saved['GP_sha256']:
            raise ValueError('GP hash differs')
        ids, notes = saved['display_ids'], saved['edited_notes']
        state = previous.held_state(old, q)
        state['ranker_KF'] = kf
        incumbent = max(h['raw_value'] for h in state['history'])
        cp = folder / f'{q:02}_commit.json'
        if cp.exists():
            record = read(cp)
            if record['fixed']['ranker_before'] != kf:
                raise ValueError('Changed KF replay state')
        else:
            public = prompts.reasoning_public(ROOT, pool, state, notes, ids, q)
            # Reuse exactly the normalized packet for both program choices.
            def both(raw):
                return {a: policy.plan(ROOT, pool, notes, state, ctx, ids, q, raw, a)
                        for a in ('fixed', 'dynamic')}
            response = queue.request(ROOT, job['id'], 'choose', q, prompts.REASONING_SYSTEM, public, both,
                                     selected_validator=lambda raw: validate_insight(ROOT, raw['insight'], ids))
            fixed, dynamic = response['plan']['fixed'], response['plan']['dynamic']
            record = dict(query=q, fixed=fixed, dynamic=dynamic, incumbent=incumbent,
                          source_commit_sha256=sha(path.parent / 'rounds' / f'{q:02}_commit.json'),
                          GP_sha256=engine.gp_hash(ctx), response=response,
                          history_sha256=control.digest(state['history']),
                          envelope_switch_possible=rank_opportunity(fixed, pool, ctx, ids, state))
            record['commit_sha256'] = control.digest(record)
            write(cp, record)
        obs = old['history'][q-1]
        gp = engine.gp_probability(ctx, pool.ix[obs['candidate_id']], incumbent / pool.scale)
        kf, feedback = calibration.update(kf, record['fixed']['probabilities'][obs['candidate_id']],
                                          gp, obs['raw_value'] > incumbent)
        write(folder / f'{q:02}_archived_feedback.json', dict(archived_observation=obs, feedback=feedback))
        print(f"{job['cell']} q{q:02} committed", flush=True)
    return dict(job=job['id'], committed=len(p['prefix_rounds']))


def evaluate_prefix(p):
    rows = []
    for job in p['jobs']:
        if job['phase'] != 'prefix':
            continue
        for q in p['prefix_rounds']:
            path = BASE / 'prefix' / job['cell'] / f'{q:02}_commit.json'
            c = read(path)
            if control.digest({k:v for k,v in c.items() if k != 'commit_sha256'}) != c['commit_sha256']:
                raise ValueError(f'Bad commitment {path}')
            rows.append((job, path, c))
    write(BASE / 'prefix_lock.json', dict(protocol=p['protocol_sha256'],
          decisions_sha256={rel(path): sha(path) for _, path, _ in rows}, all_prefixes_locked=True))
    results = []
    for job, _, c in rows:
        f, d = c['fixed'], c['dynamic']
        switch = f['selected_id'] != d['selected_id']
        values = [None, None]
        uplift = 0.
        if switch:
            pool = pool_for(job['task'])
            values = [knowledge_gp.read_observation(ROOT, pool, x['selected_id'])['raw_value'] for x in (f, d)]
            uplift = max(0., values[1]-c['incumbent']) - max(0., values[0]-c['incumbent'])
        results.append(dict(cell=job['cell'], query=c['query'], incumbent=c['incumbent'],
            gp_all_tied=f['ranking']['GP_all_tied'], llm_all_tied=f['ranking']['LLM_all_tied'],
            envelope_switch_possible=c['envelope_switch_possible'], switched=switch,
            fixed_id=f['selected_id'], dynamic_id=d['selected_id'], fixed_weight=f['ranking']['weight'],
            dynamic_weight=d['ranking']['weight'], fixed_outcome=values[0], dynamic_outcome=values[1],
            delta_improvement=uplift, offline_lookups=2*int(switch)))
    out = dict(status='COMPLETE', prefixes=len(results), offline_lookups=sum(r['offline_lookups'] for r in results),
               new_online_queries=0, rows=results)
    write(BASE / 'prefix_complete.json', out)
    return out


def job_entry(job, p):
    os.chdir(ROOT / 'experiments/chemist_system')
    with scope(job):
        return prefix(job, p) if job['phase'] == 'prefix' else previous.trace_job(job, p)


def batch(phase):
    p = verify()
    if phase == 'traces' and read(BASE / 'prefix_complete.json')['status'] != 'COMPLETE':
        raise ValueError('Evaluate frozen prefixes first')
    todo = [j for j in p['jobs'] if j['phase'] == phase]
    with ProcessPoolExecutor(max_workers=len(todo)) as pool:
        result = list(pool.map(job_entry, todo, [p]*len(todo)))
    verify()
    if phase == 'prefix':
        return evaluate_prefix(p)
    write(BASE / 'traces_complete.json', dict(status='COMPLETE', jobs=result))
    return result


def preflight():
    p = verify()
    previous.compiled_extractor(str(ROOT))
    rows = []
    for task, short, seed in CELLS:
        pool = pool_for(task)
        for q in (1,2,5,10):
            path = ROOT / p['references'][f'{short}{seed}_v13']
            ctx = previous.saved_context(path.parent, q)
            commit = read(path.parent / 'rounds' / f'{q:02}_commit.json')
            ids = [pool.ids[int(j)] for j in engine.candidates(pool, ctx, q)[1]]
            if ids != commit['display_ids'] or engine.gp_hash(ctx) != commit['GP_sha256']:
                raise ValueError('Archived GP or menu mismatched')
            if q == 1:
                gp = previous.GP(pool, BASE / 'cache' / task, seed)
                native = gp.context([], 1)
                if engine.gp_hash(native) != engine.gp_hash(ctx) or gp.counters['operator_compiles_in_loop']:
                    raise ValueError('No-data menu or compiler differs')
                gp.release()
            rows.append(dict(task=task, query=q, GP_and_menu_exact=True))
    value = dict(status='PASS', checks=rows, new_oracle_queries=0)
    write(BASE / 'preflight.json', value)
    return value


def dispatch(action):
    with scope():
        if action == 'pending':
            return queue.pending(ROOT)
        rows = json.loads(sys.stdin.read())
        for row in rows:
            path = Path(row['request']).resolve()
            if not path.is_relative_to(BASE.resolve()):
                raise ValueError('Wrong request namespace')
            if action == 'assign':
                request = read(path)
                invocation = read(path.parent / 'invocation.json')
                result = row['spawn_result']
                actual = result.get('task_name', result.get('agent_id'))
                if actual != PARENT + '/' + invocation['task_name']:
                    raise ValueError('Wrong responder identity')
                write(path.parent / 'assignment.json', dict(spawn_request=invocation,
                      spawn_result=result, agent_id=actual, request_sha256=request['request_sha256']))
            elif action == 'seal':
                queue.seal(ROOT, path)
        return dict(processed=len(rows))


if __name__ == '__main__':
    action = sys.argv[1]
    if action == 'register':
        result = dict(protocol_sha256=register()['protocol_sha256'])
    elif action == 'preflight':
        result = preflight()
    elif action in ('prefix', 'traces'):
        if read(BASE / 'preflight.json')['status'] != 'PASS':
            raise ValueError('Preflight required')
        result = batch(action)
    else:
        result = dispatch(action)
    print(json.dumps(result, ensure_ascii=False))
