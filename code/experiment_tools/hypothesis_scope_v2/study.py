"""Run the registered v2 arms in a separate materialized workspace.

Frozen code supplies the GP, selection, knowledge state and prompts. This module
changes only what the registration names: whether the first evidence-informed
preferred experiment is executed, whether scopes start before data, and whether
later refreshes are live.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import json
import shutil
import numpy as np
from filelock import FileLock
from hypothesis_scope_loop import study as frozen, policy, knowledge as k, prompts
from hypothesis_scope_multiseed import study as multi
from . import VARIANT, SEEDS, ARMS, EXECUTE_ARMS, LIVE_ARMS, EVIDENCE_FIRST_ARMS, transport

ROOT = frozen.ROOT
BASE = ROOT/'experiments/runs'/VARIANT
REPORT = ROOT/'experiments/reports'/VARIANT
SHORT = frozen.SHORT
TASKS = tuple(frozen.native.TASKS)
read, write, sha, digest = frozen.read, frozen.write, frozen.sha, frozen.digest
FROZEN_MULTI = ROOT/'experiments/runs/hypothesis_scope_multiseed_20261105'


def folder_for(task, seed, arm):
    return BASE/'runs'/arm/f's{seed}'/SHORT[task]


def source_folder(task, seed, kind):
    if seed == 0:
        return frozen.folder_for(task) if kind == 'hyp' else frozen.first.folder_for(task)
    return FROZEN_MULTI/'runs'/f's{seed}'/(SHORT[task] + '_' + kind)


def header(task, seed, arm):
    return dict(task=task, seed=seed, tie_seed=seed, numerical_seed=0, arm=arm, initial_ids=[], initial=0,
                queries=20, status='RUNNING', preferred_executed=False)


def prefix_from_controller(task, seed, arm):
    """Frozen controller state after query 3: history, knowledge and feedback."""
    src = source_folder(task, seed, 'hyp')
    final = read(src/'state.json')
    commit = read(src/'rounds/03_commit.json')
    feedback = read(src/'rounds/03_feedback.json')
    state = header(task, seed, arm)
    state.update(history=deepcopy(final['history'][:3]), decisions=deepcopy(final['decisions'][:3]),
                 **deepcopy(commit['working_knowledge']))
    state['notes'] = deepcopy(feedback['notes_after_feedback'])
    return state


def prefix_from_first(task, seed, arm):
    """Same-first BO state after query 3, with no knowledge installed."""
    final = read(source_folder(task, seed, 'first')/'state.json')
    state = header(task, seed, arm)
    state.update(history=deepcopy(final['history'][:3]), decisions=deepcopy(final['decisions'][:3]),
                 notes=[], archive=[], calls=[], permission_events=[])
    best = None
    for row in state['decisions']:
        row['decision'].setdefault('incumbent_before', best)
        row['decision'].setdefault('active_hypothesis_ids', [])
        row.setdefault('knowledge_feedback', [])
        row['decision'].setdefault('locked_forecasts', [])
        best = row['raw_value'] if best is None else max(best, row['raw_value'])
        row.setdefault('best', best)
    return state


def run_arm(task, seed, arm):
    pool = frozen.pool_for(task)
    scopes = k.Scopes(pool)
    folder = folder_for(task, seed, arm)
    folder.mkdir(parents=True, exist_ok=True)
    with FileLock(str(folder/'.worker.lock'), timeout=1):
        sf = folder/'state.json'
        if sf.exists():
            state = read(sf)
        else:
            state = prefix_from_first(task, seed, arm) if arm in EVIDENCE_FIRST_ARMS else prefix_from_controller(task, seed, arm)
        if state['status'] == 'COMPLETE':
            return dict(task=SHORT[task], seed=seed, arm=arm, status='COMPLETE')
        sealed4 = None if arm in EVIDENCE_FIRST_ARMS else read(source_folder(task, seed, 'hyp')/'rounds/04_commit.json')
        gp = multi.SeededGP(pool, BASE/'cache'/SHORT[task], tie_seed=seed)
        while len(state['history']) < 20:
            q = len(state['history']) + 1
            cp = folder/'rounds'/f'{q:02d}_commit.json'
            if cp.exists():
                commit = read(cp)
            else:
                commit = decide(task, seed, arm, pool, scopes, gp, state, q, sealed4, folder)
                commit['commit_sha256'] = digest(commit)
                write(cp, commit)
            frozen.apply_outcome(pool, folder, state, commit)
            if commit['decision']['reason'] == 'executed_LLM_preferred_test':
                state['preferred_executed'] = True
                write(sf, state, replace=True)
            gp.release()
    return dict(task=SHORT[task], seed=seed, arm=arm, status='COMPLETE')


def decide(task, seed, arm, pool, scopes, gp, state, q, sealed4, folder):
    work = deepcopy(state)
    inc = max(r['raw_value'] for r in state['history'])
    available = ~np.isin(pool.ids, [r['candidate_id'] for r in state['history']])
    preferred = projection = None
    if q == 4 and sealed4 is not None:
        # The prefix is the frozen controller's, so its sealed query-4 response is reused.
        event = sealed4['decision']['refresh_event']
        work.update(deepcopy(sealed4['working_knowledge']))
        preferred, projection = sealed4['decision']['LLM_preferred'], sealed4['decision']['LLM_projection']
    else:
        work['notes'], events = k.pause_for_recommendation(work['notes'], inc)
        work['permission_events'] += events
        event = policy.event(work, scopes, available, inc, q)
        if event['call'] and arm in LIVE_ARMS:
            public = prompts.public(pool, work, event, frozen.locked_first(task))
            proposal = transport.request(f'{arm}/s{seed}_{SHORT[task]}', f'query{q:02d}', prompts.SYSTEM, public,
                                         lambda raw: prompts.validate(raw, pool, work, frozen.locked_first(task)))
            previous = deepcopy(work['notes'])
            work['notes'], edits = k.install(proposal['plan']['drafts'], previous, work['archive'], scopes, q, event['stagnated'])
            work['archive'] += previous
            work['calls'].append(dict(query=q, event=event, request=proposal['request'], preferred=proposal['plan']['preferred'],
                                      projection=proposal['plan']['projection'], edits=edits))
            work['notes'], events = k.pause_for_recommendation(work['notes'], inc)
            work['permission_events'] += events
            preferred, projection = proposal['plan']['preferred'], proposal['plan']['projection']
        elif event['call']:
            event = dict(event, call=False, hold='no_later_refresh_arm')
    context = gp.context(state['history'], q)
    choice = policy.select(context, scopes, work['notes'], inc)
    j, causes, reason = choice['index'], choice['causes'], choice['reason']
    check = None
    if q == 4 and sealed4 is not None:
        d = sealed4['decision']
        check = dict(BO_matches_frozen=pool.ids[context['bo_index']] == d['unrestricted_BO_id'],
                     selection_matches_frozen=pool.ids[j] == d['selected_id'])
        write(folder/'q4_reference_check.json', check)
    execute = arm in EXECUTE_ARMS and projection is not None and not state.get('preferred_executed', False)
    if execute:
        j = pool.ix[projection['candidate_id']]
        assert context['available'][j], 'Preferred experiment was already measured'
        causes = {n['semantic_key']: n['id'] in preferred['basis_ids'] for n in work['notes']}
        reason = 'executed_LLM_preferred_test'
    stats = context['stats']
    arrays = {a: stats[a] for a in ('mu_y', 'sd_y', 'sd_predictive_y')}
    arrays.update({a: context[a] for a in ('base_logei', 'available', 'tie_priority')})
    arrays.update(final_mask=choice['mask'], proposed_scope_mask=choice['proposed_scope_mask'])
    for n in work['notes']:
        arrays['scope_' + n['semantic_key']] = scopes.mask(n['scope'])
    ap = folder/'rounds'/f'{q:02d}_GP.npz'
    ap.parent.mkdir(parents=True, exist_ok=True)
    if not ap.exists():
        with ap.open('xb') as handle:
            np.savez_compressed(handle, **arrays)
    decision = dict(query=q, conditions=deepcopy(pool.points[j]), incumbent_before=inc, refresh_event=event,
                    active_hypothesis_ids=[n['id'] for n in choice['active']], LLM_preferred=preferred, LLM_projection=projection,
                    selection_is_native_qLogEI_after_first=reason != 'executed_LLM_preferred_test', global_qLogEI_gap_gate=False,
                    reason=reason, GP_fitted=True, qLogEI_defined=True, scope_count=int(choice['mask'].sum()),
                    proposed_scope_count=int(choice['proposed_scope_mask'].sum()), fallback=choice['fallback'],
                    unrestricted_BO_id=pool.ids[context['bo_index']], changed_from_unrestricted_BO=bool(j != context['bo_index']),
                    scope_choice_id=pool.ids[choice['index']], base_logEI=float(context['base_logei'][j]),
                    qLogEI_opportunity_gap=float(context['base_logei'][context['bo_index']] - context['base_logei'][j]),
                    GP_mean_raw=float(stats['mu_y'][j]*pool.scale), GP_SD_raw=float(stats['sd_y'][j]*pool.scale),
                    GP_predictive_SD_raw=float(stats['sd_predictive_y'][j]*pool.scale), GP_stage=context['stage'],
                    causal_diagnostics=choice['causal_diagnostics'], fit_key=stats['fit_key'],
                    operator_compiles_in_loop=gp.counters['operator_compiles_in_loop'], selected_id=pool.ids[j],
                    locked_forecasts=k.freeze_forecasts(pool, scopes, work['notes'], j, q, inc, context, causes))
    return dict(query=q, prestate_sha256=digest(state), decision=decision,
                working_knowledge={a: work[a] for a in ('notes', 'archive', 'calls', 'permission_events')},
                GP_arrays_sha256=sha(ap), GP_parameters=stats['fit_parameters'], acquisition=context['acquisition_meta'],
                output_scale_convention=stats.get('cold_scale_convention'))


def run_reproduction(task, seed, arm):
    """Registered gate: rerun frozen BO controls with an empty cache in this namespace."""
    multi.BASE = BASE/'reproduction'
    multi.folder_for = lambda t, s, a: BASE/'reproduction'/'runs'/f's{s}'/(SHORT[t] + '_' + a)
    protocol = read(FROZEN_MULTI/'protocol.json')
    return multi.run_control(task, seed, arm, protocol)


def compare_reproduction():
    rows = []
    for task in TASKS:
        for seed in (1, 2):
            for arm in ('bo', 'first'):
                new = read(BASE/'reproduction'/'runs'/f's{seed}'/(SHORT[task] + '_' + arm)/'state.json')
                old = read(FROZEN_MULTI/'runs'/f's{seed}'/(SHORT[task] + '_' + arm)/'state.json')
                same = [a['candidate_id'] == b['candidate_id'] and a['raw_value'] == b['raw_value']
                        for a, b in zip(new['history'], old['history'])]
                rows.append(dict(task=SHORT[task], seed=seed, arm=arm, matched=sum(same), of=len(same),
                                 first_mismatch=next((i + 1 for i, s in enumerate(same) if not s), None)))
    passed = all(r['matched'] == 20 for r in rows)
    result = dict(status='PASS_REPRODUCTION' if passed else 'FAIL_REPRODUCTION', trajectories=len(rows),
                  exact=sum(r['matched'] == 20 for r in rows), rows=rows)
    REPORT.mkdir(parents=True, exist_ok=True)
    write(REPORT/'reproduction_gate.json', result, replace=True)
    return result


def report():
    REPORT.mkdir(parents=True, exist_ok=True)
    metrics, traces, usage = [], [], {}
    for arm in ARMS:
        for task in TASKS:
            for seed in SEEDS:
                path = folder_for(task, seed, arm)/'state.json'
                if not path.exists():
                    continue
                state = read(path)
                if state['status'] != 'COMPLETE':
                    continue
                best = np.maximum.accumulate([r['raw_value'] for r in state['history']])
                executed = next((r['query'] for r in state['decisions'] if r['decision'].get('reason') == 'executed_LLM_preferred_test'), '')
                metrics.append(dict(arm=arm, task=SHORT[task], seed=seed, SearchAUC=float(best.mean()), final_best=float(best[-1]),
                                    calls_installed=len(state['calls']), executed_query=executed))
                for i, r in enumerate(state['history']):
                    traces.append(dict(arm=arm, task=SHORT[task], seed=seed, query=i + 1, candidate_id=r['candidate_id'],
                                       raw_value=r['raw_value'], cumulative_best=float(best[i])))
    for name, rows in (('per_seed_results.csv', metrics), ('trajectories.csv', traces)):
        with open(REPORT/name, 'w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    for path in BASE.glob('requests/*/*/*/validated.json'):
        saved = read(path)
        arm = path.relative_to(BASE/'requests').parts[0]
        row = usage.setdefault(arm, dict(requests=0, input_tokens=0, output_tokens=0, total_tokens=0, models=[]))
        row['requests'] += 1
        for key in ('input_tokens', 'output_tokens', 'total_tokens'):
            row[key] += saved['usage'].get(key, 0)
        if saved['resolved_model'] not in row['models']:
            row['models'].append(saved['resolved_model'])
    checks = [read(p) for p in BASE.glob('runs/*/*/*/q4_reference_check.json')]
    # Same-environment references: the reproduction gate failed on floating-point near-ties.
    with open(REPORT/'per_seed_results.csv', 'a', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]), lineterminator='\n')
        for task in TASKS:
            for seed in SEEDS:
                for arm in ('bo', 'first'):
                    path = BASE/'reproduction'/'runs'/f's{seed}'/(SHORT[task] + '_' + arm)/'state.json'
                    if path.exists() and read(path)['status'] == 'COMPLETE':
                        best = np.maximum.accumulate([r['raw_value'] for r in read(path)['history']])
                        writer.writerow(dict(arm=arm + '_env', task=SHORT[task], seed=seed, SearchAUC=float(best.mean()),
                                             final_best=float(best[-1]), calls_installed=0, executed_query=''))
    summary = dict(generated_UTC=datetime.now(timezone.utc).isoformat(), completed=len(metrics), api_usage=usage,
                   q4_checks=len(checks), q4_BO_matches=sum(c['BO_matches_frozen'] for c in checks),
                   q4_selection_matches=sum(c['selection_matches_frozen'] for c in checks))
    write(REPORT/'summary.json', summary, replace=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('reproduce', 'references', 'run', 'report'))
    parser.add_argument('--arms', nargs='*', choices=ARMS, default=[])
    parser.add_argument('--tasks', nargs='*', default=[])
    parser.add_argument('--seeds', nargs='*', type=int, default=list(SEEDS))
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    tasks = [t for t in TASKS if not args.tasks or SHORT[t] in args.tasks]
    if args.action == 'report':
        print(json.dumps(report(), indent=1))
        return
    if args.action == 'reproduce':
        jobs = [(run_reproduction, t, s, a) for s in (1, 2) for a in ('bo', 'first') for t in tasks]
    elif args.action == 'references':
        jobs = [(run_reproduction, t, s, a) for s in (0, 3) for a in ('bo', 'first') for t in tasks]
    else:
        jobs = [(run_arm, t, s, a) for a in args.arms for s in args.seeds for t in tasks]
    BASE.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=min(8, args.workers), initializer=frozen.initialize) as executor:
        futures = [executor.submit(*job) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            print(json.dumps(dict(done=done, of=len(jobs), **future.result())), flush=True)
    if args.action == 'reproduce':
        result = compare_reproduction()
        print(json.dumps({key: result[key] for key in ('status', 'trajectories', 'exact')}))
