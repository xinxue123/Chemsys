"""Freeze then replay all eight v2 deployment trajectories as DEVELOPMENT diagnostics.

No oracle reads, LLM requests, GP fitting, new choices, or new SearchAUC.
Run --freeze first; a separate --replay invocation verifies the frozen inputs.
"""
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import argparse
import json
import math
import os
import sys

import numpy as np
from scipy.special import log_ndtr, logsumexp

import knowledge_shadow as shadow


ROOT = Path(__file__).resolve().parents[3]
VARIANT = ROOT / 'experiments/runs/knowledge_loop_20261101/v2'
OUT = ROOT / 'experiments/reports/knowledge_loop_20261101/shadow_fields_v2_deployment'
TASKS = [f'Reizman_Suzuki_Case_{i}' for i in range(1, 5)] + [
    'Perera_Suzuki', 'Doyle_Buchwald_Hartwig', 'SnAr', 'HPLC']
ARMS = ['pair', 'raw_field', 'gp_conditioned_field']
PARAMETERS = dict(assessment=shadow.LABEL, mapping=shadow.MAPPING,
    prior_slope_mean=0., prior_slope_variance=.06**2,
    prior_choice='explicit development freeze; shared engineering-scale SD .06, not an old module default',
    process_noise=0., coefficient_sign_constraint='none; negative slopes must be reported, not called supported direction',
    redundancy_probability=.99, nonzero_atol=1e-12,
    task_selection='all eight completed v2 deployment_agent_initial_terra',
    aggregation='each queried point once: equal-density mixture of all live versions covering that selected point; no coverage means GP',
    pair_scoring='original exact conditional density, scope-masked; MI informative flag separately reported',
    calibration='independent zero-mean version states; every consecutive live round; score before KF',
    supported_mean_evaluation='nonzero basis with at least one earlier distinct-point update in that version',
    conditioning='matching saved GP prior K+noise and cross; original scope reapplied; no refit',
    candidate_history='frozen actual deployment selections; no off-policy selection claim',
    output_restrictions=['no new SearchAUC', 'no new strategy trajectory', 'no prospective experiment claim'])
SOURCE_FILES = [Path(__file__).resolve(), Path(shadow.__file__).resolve(),
                Path(__file__).with_name('knowledge_saved_gp.py').resolve()]


def read(path):
    if 'oracle' in str(path).lower():
        raise ValueError('Oracle paths are prohibited, including for hashing')
    return json.loads(Path(path).read_text(encoding='utf-8'))


def filehash(path):
    if 'oracle' in str(path).lower():
        raise ValueError('Oracle paths are prohibited, including for hashing')
    return sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def run_path(task):
    return VARIANT / 'runs' / f'{task}_deployment_agent_initial_terra'


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'frozen_inputs.json'
    if target.exists():
        raise ValueError('Existing freeze is immutable; do not overwrite it')
    protocol = read(VARIANT / 'protocol.json')
    if protocol['tasks'] != TASKS:
        raise ValueError('Task inventory differs from the declared eight tasks')
    files = {VARIANT / 'protocol.json', *SOURCE_FILES}
    manifest = {m['task']: m for m in protocol['inputs']}
    tasks = []
    for task in TASKS:
        folder = run_path(task)
        state = read(folder / 'state.json')
        if (state['status'] != 'COMPLETE' or len(state['history']) != 20 or
                len(state['decisions']) != 16 or state['mode'] != 'deployment'):
            raise ValueError(f'{task}: need its complete unmodified deployment trajectory')
        pool = ROOT / next(p for p in manifest[task]['files_sha256'] if p.endswith('/pool.json'))
        files.update([pool, folder / 'state.json'])
        for t in range(1, 17):
            commit_path = folder / 'rounds' / f'{t:02d}_commit.json'
            commit = read(commit_path)
            files.update([commit_path, folder / 'rounds' / f'{t:02d}_fit.json',
                folder / 'rounds' / f'{t:02d}_GP.npz',
                VARIANT / 'cache' / task / (commit['GP_fit_key'] + '.pt')])
        tasks.append(dict(task=task, group=manifest[task]['group'], public_pool=rel(pool),
                          run=rel(folder), queries=16))
    receipt = dict(assessment=shadow.LABEL, frozen_utc=datetime.now(timezone.utc).isoformat(),
                   protocol_sha256=protocol['protocol_sha256'], parameters=PARAMETERS,
                   tasks=tasks, files_sha256={rel(p): filehash(p) for p in sorted(files)},
                   frozen_before_any_shadow_replay=True, outcome_access='only already queried state/history',
                   prohibited='oracle files are neither read nor hashed')
    receipt['freeze_sha256'] = shadow.digest(receipt)
    write(target, receipt)
    print('FROZEN', receipt['freeze_sha256'], len(receipt['files_sha256']), 'files', flush=True)


def pool_from_public(item):
    raw = read(ROOT / item['public_pool'])
    group = item['group']
    if group == 'P0':
        axes = ['t_res', 'temperature', 'catalyst_loading', 'catalyst']
        features = [tuple(map(float, z[:3])) + (p['catalyst'],) for z, p in zip(raw['z'], raw['points'])]
        scale = 100.
    elif group == 'P1':
        axes = raw['axes']
        features = [tuple(p[a] for a in axes) for p in raw['points']]
        scale = 100.
    else:
        axes = raw['axes']
        features = [tuple(map(float, z)) for z in raw['z']]
        scale = float(raw['objective_scale'])
    if any(set(point) - set(axes) for point in raw['points']):
        raise ValueError('Public pool contains a non-input field')
    return dict(raw=raw, group=group, task=item['task'], axes=axes, features=features,
                ids=raw['ids'], ix={v: i for i, v in enumerate(raw['ids'])}, scale=scale)


def io_guard(allowed):
    """Runtime guard: no benchmark read outside the frozen input allowlist."""
    allowed = {str((ROOT / p).resolve()).casefold() for p in allowed}
    data_root = str((ROOT / 'experiments').resolve()).casefold()
    output_root = str(OUT.resolve()).casefold()
    def audit(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            path = str(Path(os.fsdecode(args[0])).resolve()).casefold()
            if 'oracle' in path:
                raise PermissionError('Oracle file access is prohibited by this diagnostic')
            if path.startswith(data_root) and not path.startswith(output_root) and path not in allowed:
                raise PermissionError('Unfrozen experiment file access is prohibited')
        elif event in ('socket.connect', 'socket.getaddrinfo'):
            raise PermissionError('Network access is prohibited by this diagnostic')
    sys.addaudithook(audit)


def replay_task(item, freeze_hash):
    from knowledge_saved_gp import SavedJointGP
    task = item['task']
    folder, pool = ROOT / item['run'], pool_from_public(item)
    state = read(folder / 'state.json')
    fields, calibrators, traces, per_round = {}, {}, [], []
    counts = Counter()
    coverage_counts = Counter()
    independent = {arm: set() for arm in ARMS}
    supported = {arm: set() for arm in ARMS}
    totals = {arm: Counter() for arm in ARMS}
    max_mean_error = max_var_error = 0.
    for t in range(1, 17):
        commit = read(folder / 'rounds' / f'{t:02d}_commit.json')
        prefix = state['history'][:t + 3]
        observed = state['history'][t + 3]
        if observed['candidate_id'] != commit['selected_id']:
            raise ValueError('Stored feedback does not match the frozen selected point')
        fit = read(folder / 'rounds' / f'{t:02d}_fit.json')
        gp = SavedJointGP(VARIANT / 'cache' / task / (commit['GP_fit_key'] + '.pt'),
                          pool, prefix, commit, fit)
        conditioning = gp.conditioning(commit)
        point_index = pool['ix'][commit['selected_id']]
        point = dict(zip(pool['axes'], pool['features'][point_index]))
        with np.load(folder / 'rounds' / f'{t:02d}_GP.npz', allow_pickle=False) as stored:
            mu = float(stored['mu_y'][point_index])
            var = float(stored['sd_predictive_y'][point_index])**2
        np.testing.assert_allclose(gp.mean, mu, atol=1e-9, rtol=1e-7)
        np.testing.assert_allclose(gp.predictive_variance, var, atol=1e-9, rtol=1e-6)
        max_mean_error = max(max_mean_error, abs(gp.mean - mu))
        max_var_error = max(max_var_error, abs(gp.predictive_variance - var))
        included = []
        for h in commit['hypotheses_at_commit']:
            if h['status'] == 'retracted':
                continue
            key = (h['id'], h['version'])
            if key not in fields:
                fields[key] = shadow.field_from_commit(commit, *key, pool['axes'])
                calibrators[key] = shadow.initial_state(fields[key], PARAMETERS['prior_slope_variance'])
            before = calibrators[key]
            frozen = shadow.prepare_prediction(commit, fields[key], before, point, prefix, conditioning)
            after, receipt = shadow.score_and_update(before, frozen, dict(
                candidate_id=observed['candidate_id'], query_round=t, y=observed['y']))
            calibrators[key] = after
            pred = frozen['pair_prediction']
            alpha = pred['contrast_mean'] / math.sqrt(pred['contrast_var'] + pred['preference_noise_sd']**2)
            log_probability = float(log_ndtr(alpha))
            probability = math.exp(log_probability)
            cap_upper = float(np.logaddexp(math.log(.5), math.log(.5) + min(math.log(4), -log_probability)))
            coverage = next(c for c in commit['coverage'] if (c['id'], c['version']) == key)
            redundant = probability >= PARAMETERS['redundancy_probability']
            counts['live_claim_rounds'] += 1
            counts['pair_P_ge_099_all_live'] += redundant
            coverage_counts['eligible_scope_candidate_exposures'] += coverage['eligible_inside_scope']
            coverage_counts['live_claim_rounds_with_eligible_scope'] += bool(coverage['eligible_inside_scope'])
            coverage_counts['live_claim_rounds_with_cap_rank_effect'] += bool(coverage['can_change_rank_at_cap'])
            prior_updates = {arm: sum(e['calibration_updates'][arm]['updated'] for e in before['events'])
                             for arm in ARMS[1:]}
            if frozen['in_frozen_scope']:
                counts['covered_claim_point_exposures'] += 1
                counts['pair_P_ge_099_covered'] += redundant
                counts['original_informative_covered'] += receipt['original_pair_informative']
                included.append((frozen, receipt))
                independent['pair'].add(t)
                supported['pair'].add(t)
                for arm in ARMS[1:]:
                    value = frozen['arms'][arm]
                    if abs(value['basis']) > PARAMETERS['nonzero_atol']:
                        independent[arm].add(t)
                        if prior_updates[arm] > 0:
                            supported[arm].add(t)
                    if before['calibrators'][arm]['mean'] < -PARAMETERS['nonzero_atol']:
                        counts[arm + '_negative_slope_covered_exposures'] += 1
            traces.append(dict(task=task, query=t, hypothesis_id=h['id'], version=h['version'],
                frozen=frozen, feedback=receipt, coverage=coverage,
                original_pair_probability=probability, original_pair_redundant_P_ge_099=redundant,
                pair_cap_logei_boost_upper=cap_upper, prior_distinct_updates=prior_updates))
        y = float(observed['y'])
        row = dict(task=task, query=t, point_id=observed['candidate_id'],
                   included_live_scoped_versions=len(included), gp_mean=mu, gp_variance=var,
                   observed_y=y, arms={})
        for arm in ARMS:
            if included:
                mean = float(np.mean([f['arms'][arm]['mean'] for f, r in included]))
                gains = [r['metrics'][arm]['predictive_logscore_gain'] for f, r in included]
                log_gain = float(logsumexp(gains) - math.log(len(gains)))
                mse_gain = (y - mu)**2 - (y - mean)**2
            else:
                mean, log_gain, mse_gain = mu, 0., 0.
            metrics = dict(mean_squared_error_gain=mse_gain,
                           same_variance_mean_logscore_gain=mse_gain / (2 * var),
                           predictive_logscore_gain=log_gain,
                           mixture_mean=mean, changed_mean=abs(mean - mu) > PARAMETERS['nonzero_atol'])
            row['arms'][arm] = metrics
            for name in ('mean_squared_error_gain', 'same_variance_mean_logscore_gain', 'predictive_logscore_gain'):
                totals[arm][name] += metrics[name]
            totals[arm]['unique_mean_changed_points'] += metrics['changed_mean']
            totals[arm]['mean_error_improved_points'] += mse_gain > PARAMETERS['nonzero_atol']
            totals[arm]['mean_error_worsened_points'] += mse_gain < -PARAMETERS['nonzero_atol']
        per_round.append(row)
    summary = dict(task=task, queries=16, assessment=shadow.LABEL, hypothesis_versions=len(fields),
                   unique_selected_points_in_any_scope=sum(r['included_live_scoped_versions'] > 0 for r in per_round),
                   engineering_y_to_native_scale=pool['scale'], counts=dict(counts), coverage=dict(coverage_counts),
                   arms={}, GP_selected_reconstruction_max_abs_mean_error=max_mean_error,
                   GP_selected_reconstruction_max_abs_predictive_variance_error=max_var_error)
    for arm in ARMS:
        summary['arms'][arm] = dict(totals[arm], unique_nonzero_or_pair_supported_points=len(independent[arm]),
                                   unique_points_after_prior_distinct_calibration=len(supported[arm]))
    write(OUT / (task + '.json'), dict(assessment=shadow.LABEL, freeze_sha256=freeze_hash,
                                     summary=summary, per_round=per_round, claim_traces=traces))
    print('DONE', task, 'covered points', summary['unique_selected_points_in_any_scope'], flush=True)
    return summary


def replay():
    frozen = read(OUT / 'frozen_inputs.json')
    if frozen['freeze_sha256'] != shadow.digest({k: v for k, v in frozen.items() if k != 'freeze_sha256'}):
        raise ValueError('Freeze receipt was modified')
    if frozen['parameters'] != PARAMETERS or [x['task'] for x in frozen['tasks']] != TASKS:
        raise ValueError('Diagnostic parameters/tasks differ from the prior freeze')
    effective_hashes = dict(frozen['files_sha256'])
    amendments = []
    for path in sorted(OUT.glob('source_fix_*.json')):
        fix = read(path)
        if (fix['original_freeze_sha256'] != frozen['freeze_sha256'] or
                fix['parameters_sha256'] != shadow.digest(PARAMETERS) or
                fix['source_file'] not in {rel(Path(__file__)), rel(Path(__file__).with_name('knowledge_saved_gp.py'))} or
                fix['old_source_sha256'] != effective_hashes[fix['source_file']]):
            raise ValueError('Invalid declared source compatibility repair')
        effective_hashes[fix['source_file']] = fix['new_source_sha256']
        amendments.append(fix)
    for name, expected in effective_hashes.items():
        if filehash(ROOT / name) != expected:
            raise ValueError(f'Frozen source/input changed: {name}')
    if (OUT / 'summary.json').exists():
        raise ValueError('Completed replay output exists; do not silently replace it')
    io_guard(frozen['files_sha256'])
    started = datetime.now(timezone.utc).isoformat()
    summaries = [replay_task(item, frozen['freeze_sha256']) for item in frozen['tasks']]
    result = dict(assessment=shadow.LABEL, freeze_sha256=frozen['freeze_sha256'],
                  started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
                  parameters=PARAMETERS, tasks=summaries,
                  source_compatibility_repairs=amendments,
                  oracle_reads=0, llm_calls=0, gp_fits=0, new_selected_points=0,
                  independent_unit='one already-queried point per round; claim exposure counts are not independent observations',
                  caution='Symmetric a0=0 prior learns signed residual geometry; field wins do not establish the original preference direction.')
    write(OUT / 'summary.json', result)
    print('SUMMARY', OUT / 'summary.json', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze', action='store_true')
    group.add_argument('--replay', action='store_true')
    args = ap.parse_args()
    freeze() if args.freeze else replay()
