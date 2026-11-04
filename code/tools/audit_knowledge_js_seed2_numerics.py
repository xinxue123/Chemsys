"""Independent numerical recheck of the fixed seed2 prefix JS proof.

Reuses independent saved-array/geometry audit helpers, not the JS diagnostic
integrator or any production/staging model. Only already listed B candidates
and BO predictions are integrated. No oracle, fits, API or hypothetical rollout.
"""
from collections import Counter
from functools import lru_cache
import json
from math import exp, log, sqrt, pi
from pathlib import Path
import warnings

import numpy as np
from scipy.integrate import quad, IntegrationWarning
from scipy.special import expit, log_ndtr

import audit_knowledge_transfer_loop as a

ROOT = a.ROOT
OUT = ROOT/'experiments/reports/knowledge_loop_20261101'
LN2, LN2PI = log(2.), log(2*pi)
EPSABS, EPSREL = 2e-14, 2e-12


def entropy_deficit(log_likelihood_ratio):
    """ln2-H(sigmoid(log L)), nonnegative; stable around identical densities."""
    z = abs(float(log_likelihood_ratio))
    if z < 1e-3:
        z2 = z*z
        return z2*(1/8-z2/64+z2*z2/576)
    return LN2 - (float(np.logaddexp(0., -z)) + z*float(expit(-z)))


@lru_cache(maxsize=None)
def independent_js(mu, vy, ref, log_t, tau=.06):
    if vy == 0 or np.isneginf(log_t):
        return dict(JS=0., quadrature_error_estimate=0., evaluations=0,
                    q_normalization=1., expert_normalization=1., warnings=[], no_effect_exact=True)
    assert vy > 0 and log_t <= 0 and np.isfinite(log_t)
    sd, vz = sqrt(vy), vy+tau*tau
    alpha = (mu-ref)/sqrt(vz)
    log_p = float(log_ndtr(alpha))
    log_complement = float(np.log(-np.expm1(log_t))) if log_t < 0 else -np.inf
    mills = exp(-alpha*alpha/2-LN2PI/2-log_p)
    center = sqrt(vy/vz)*mills
    reduction = min(1., max(0., mills*(mills+alpha)))
    conditional_sd_z = sqrt(max(0., 1.-vy/vz*reduction))
    # Full real-line integration, with splits at both density centers and the
    # probit transition. This does not reuse the old finite +/-16 windows.
    splits = sorted(set([-12., 0., 12., (ref-mu)/sd, center] +
                        [center+k*conditional_sd_z for k in (-12., -4., -1., 1., 4., 12.)]))
    pieces = list(zip([-np.inf]+splits, splits+[np.inf]))
    calls = 0

    def values(z):
        nonlocal calls
        calls += 1
        lg = -z*z/2-LN2PI/2
        ell = float(np.logaddexp(log_complement,
            log_t+float(log_ndtr((mu+sd*z-ref)/tau))-log_p))
        log_q = lg + float(np.logaddexp(0., ell))-LN2
        q = exp(log_q)
        return q*entropy_deficit(ell), q, exp(lg+ell)

    totals, errors, caught = [], [], []
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always', IntegrationWarning)
        for component in range(3):
            results = [quad(lambda z: values(z)[component], lo, hi,
                            epsabs=EPSABS/len(pieces), epsrel=EPSREL, limit=300)
                       for lo, hi in pieces]
            totals.append(sum(x[0] for x in results))
            errors.append(sum(x[1] for x in results))
        caught = [str(w.message) for w in ws]
    assert 0 <= totals[0] <= LN2+1e-13
    assert abs(totals[1]-1.) < 5e-12 and abs(totals[2]-1.) < 5e-12
    return dict(JS=totals[0], quadrature_error_estimate=errors[0], evaluations=calls,
                q_normalization=totals[1], expert_normalization=totals[2],
                normalization_errors=errors[1:], warnings=caught, no_effect_exact=False)


def prediction_receipt(arrays, data, j):
    if j not in data['positions']:
        return dict(JS=0., quadrature_error_estimate=0., evaluations=0,
            no_effect_exact=True, reason='No executable expert at BO; e=g exactly.', inputs=None,
            q_normalization=1., expert_normalization=1., warnings=[])
    position = data['positions'][j]
    args = (float(arrays['mu_y'][j]), float(arrays['sd_predictive_y'][j]**2),
            float(data['probes'][j]['reference_y']), float(data['values']['log_tempering'][position]))
    result = dict(independent_js(*args))
    result['inputs'] = dict(mu=args[0], vy=args[1], reference_y=args[2],
                           log_tempering=args[3] if np.isfinite(args[3]) else None, tau=.06)
    return result


def winner(rows, value='independent_delta_JS', gate='independent_error_gate'):
    allowed = [r for r in rows if r[gate]]
    return max(allowed, key=lambda r:(r[value], -r['gap'], -r['tie_priority'])) if allowed else None


def identity(row):
    return (row['point_id'], tuple(row['rule'])) if row else None


def main():
    proof_path = OUT/'seed2_JS_prefix_induction_review.json'
    proof = a.read(proof_path)
    folder = ROOT/'experiments/runs/knowledge_loop_20261101/v5_seed2_confirm'
    protocol = a.read(folder/'protocol.json')
    assert protocol['protocol_sha256'] == proof['protocol_sha256']
    manifests = {x['task']:x for x in protocol['inputs']}
    pools = {name:a.common.public_pool(ROOT, item) for name,item in manifests.items()}
    candidates, references, rounds, inputs = [], [], [], []
    all_class_changes, all_winner_changes = [], []
    for entry in proof['rounds']:
        if 'B_all_common_candidates' not in entry:
            continue
        task, t = entry['task'], entry['query']
        pool = pools[task]
        run = folder/'runs'/f'{task}_full_agent_initial_terra'
        cp = run/'rounds'/f'{t:02d}_commit.json'
        gp = cp.with_name(f'{t:02d}_GP.npz')
        commit = a.read(cp)
        # Only prequery history entries are used; no selected/future outcome
        # is accessed or needed by a gate, quadrature, ranking or this report.
        prefix = a.read(run/'state.json')['history'][:t+3]
        assert a.digest(prefix) == entry['prefix_sha256'] == commit['history_sha256']
        assert a.digest(commit) == entry['commit_sha256']
        with np.load(gp, allow_pickle=False) as saved:
            arrays = {k:saved[k].copy() for k in saved.files}
        coverage, calc = a.calculations(pool, commit['hypotheses_at_commit'], prefix, arrays)
        inputs.append(dict(task=task, query=t, commit_sha256=a.filehash(cp),
                           GP_sha256=a.filehash(gp), prefix_sha256=entry['prefix_sha256']))
        bo = pool['ix'][commit['BO_reference_id']]
        bo_by_rule = {}
        for row in coverage:
            key = a.hkey(row)
            if row['weight'] > 0 or not row['can_change_rank_at_cap'] or key not in calc:
                continue
            h = next(h for h in commit['hypotheses_at_commit'] if a.hkey(h) == key)
            ev = a.evidence(h)
            if ev['effective_prediction_score'] < 0 or ev['failures'] > ev['successes']:
                continue
            check = prediction_receipt(arrays, calc[key], bo)
            bo_by_rule[key] = check
            references.append(dict(task=task, query=t, rule=list(key), BO_id=pool['ids'][bo], **check))
        checked = []
        for old in entry['B_all_common_candidates']:
            key, j = tuple(old['rule']), pool['ix'][old['point_id']]
            data = calc[key]
            assert data['probes'][j]['reference_id'] == old['reference_id']
            current, reference = prediction_receipt(arrays, data, j), bo_by_rule[key]
            delta = current['JS']-reference['JS']
            error = current['quadrature_error_estimate']+reference['quadrature_error_estimate']
            threshold = max(1e-12, error)
            old_margin = delta-old['numerical_threshold']
            new_margin = delta-threshold
            combined_errors = error+old['JS_error_estimate']+old['BO_JS_error_estimate']
            # Audit-only reporting window; this never changes the hypothetical
            # gate, candidate membership or ranking.
            near = min(abs(old_margin), abs(new_margin)) <= max(1e-10, 10*combined_errors)
            row = dict(old, task=task, query=t, independent_JS=current['JS'],
                independent_BO_JS=reference['JS'], independent_delta_JS=delta,
                independent_threshold=threshold, independent_error_sum=error,
                fixed_original_threshold_gate=delta>old['numerical_threshold'],
                independent_error_gate=delta>threshold,
                delta_difference=delta-old['delta_JS'],
                fixed_original_threshold_margin=old_margin, independent_threshold_margin=new_margin,
                near_numerical_threshold=near, candidate_density_check=current)
            checked.append(row)
            candidates.append(row)
            if (row['fixed_original_threshold_gate'] != old['JS_numerical_gate'] or
                    row['independent_error_gate'] != old['JS_numerical_gate']):
                all_class_changes.append(row)
        old_winner = entry.get('B_winner')
        new_winner = winner(checked)
        fixed_winner = winner(checked, gate='fixed_original_threshold_gate')
        ordered = sorted([x for x in checked if x['independent_error_gate']],
                         key=lambda x:(x['independent_delta_JS'], -x['gap'], -x['tie_priority']), reverse=True)
        top_gap = (ordered[0]['independent_delta_JS']-ordered[1]['independent_delta_JS']) if len(ordered)>1 else None
        other_point = next((r for r in ordered[1:] if r['point_id'] != ordered[0]['point_id']), None) if ordered else None
        point_gap = ordered[0]['independent_delta_JS']-other_point['independent_delta_JS'] if other_point else None
        result = dict(task=task, query=t, candidate_count=len(checked),
            old_winner=identity(old_winner), strict_winner=identity(new_winner), fixed_threshold_winner=identity(fixed_winner),
            winner_unchanged=identity(old_winner)==identity(new_winner)==identity(fixed_winner),
            top_two_delta_JS_gap=top_gap,
            top_distinct_point_delta_JS_gap=point_gap,
            top_tie_break='original gap then physical tie priority; exact duplicated predictions remain tied')
        rounds.append(result)
        if not result['winner_unchanged']:
            all_winner_changes.append(result)
    near_rows = [r for r in candidates if r['near_numerical_threshold']]
    closest = sorted(candidates, key=lambda r:abs(r['independent_threshold_margin']))[:8]
    warnings_found = [r for r in candidates if r['candidate_density_check']['warnings']]
    changed_tasks = [x['task'] for x in proof['B_first_divergences']]
    report = dict(audit='independent_JS_full_real_line_and_entropy_expression',
        source_sha256=a.filehash(__file__), input_proof_sha256=a.filehash(proof_path),
        independent_geometry_helper_sha256=a.filehash(a.__file__), protocol_sha256=proof['protocol_sha256'],
        old_quad_tolerances=dict(epsabs=2e-11, epsrel=2e-9),
        new_quad_tolerances=dict(epsabs=EPSABS, epsrel=EPSREL, limit=300,
            absolute_error_budget='epsabs split across all full-real-line intervals'),
        expression='JS=E_q[ln(2)-H_Bernoulli(sigmoid(log(e/g)))], q=.5g+.5e',
        numerical_result='STABLE_UNDER_LISTED_INDEPENDENT_CHECKS' if not all_class_changes and not all_winner_changes else 'CLASSIFICATION_OR_RANK_UNSTABLE',
        totals=dict(B_evaluated_rounds=len(rounds), nonempty_rounds=sum(r['candidate_count']>0 for r in rounds),
            candidate_rule_pairs=len(candidates), common_rule_BO_checks=len(references),
            no_effect_BO_checks=sum(r['no_effect_exact'] for r in references),
            unique_density_integrals=independent_js.cache_info().currsize,
            candidate_classification_changes=len(all_class_changes), round_winner_changes=len(all_winner_changes),
            near_threshold_candidates=len(near_rows), quadrature_warning_candidate_records=len(warnings_found)),
        max_absolute_delta_JS_difference=max(abs(r['delta_difference']) for r in candidates),
        minimum_independent_gate_margin_abs=min(abs(r['independent_threshold_margin']) for r in candidates),
        near_threshold_definition='Audit flag only: distance to either gate <= max(1e-10,10*(old and new combined error estimates))',
        near_threshold_candidates=near_rows, closest_threshold_candidates=closest,
        changed_classifications=all_class_changes, changed_winners=all_winner_changes,
        candidates=candidates, rule_BO_checks=references, rounds=rounds, input_receipts=inputs,
        scientific_scope=dict(initialization='registered seed2 only', stated_B_first_divergence_tasks=changed_tasks,
            conditional_win_upper_bound=4, majority_required=5,
            meaning='Supports the fixed seed2 prefix coupling argument; not a theorem for other initializations or policies.'),
        limitations=['QUADPACK errors remain estimates, not rigorous mathematical certificates.',
            'Independent expression, stricter tolerance and full-real-line normalization checks test numerical robustness, not optimization benefit.',
            'No post-divergence candidate, outcome, policy or threshold was explored.',
            'Existing same-context response reuse and unchanged GP/feedback/prompt assumptions remain necessary for trajectory induction.'],
        oracle_files_opened=0, new_outcome_queries=0, GP_fits=0, LLM_calls=0, production_or_staging_math_used=False)
    destination = OUT/'seed2_JS_numerical_robustness_review.json'
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    lines = ['Seed2固定JS归纳：独立数值稳健性复核', report['numerical_result'],
        json.dumps(report['totals'], ensure_ascii=False),
        f"最大DeltaJS重算差={report['max_absolute_delta_JS_difference']:.12g}; 最小门距离={report['minimum_independent_gate_margin_abs']:.12g}",
        '不同表达：q乘模型标签后验的Bernoulli熵亏损；积分覆盖整个实数轴，不复用原有限窗口；更严格quad绝对/相对容差2e-14/2e-12。',
        '同时比较保留原数值门与按严格积分误差重算门；两者均检查。零tempering或不可执行BO专家直接定义JS=0。',
        f'接近门限的候选共{len(near_rows)}个；全部具体候选、BO及积分正规化结果保存在JSON。', '',
        'task | q | count | old winner | strict winner | top2 rule-pair / distinct-point deltaJS gap']
    for row in rounds:
        if row['candidate_count']:
            lines.append(f"{row['task']} | {row['query']} | {row['candidate_count']} | {row['old_winner']} | {row['strict_winner']} | {row['top_two_delta_JS_gap']} / {row['top_distinct_point_delta_JS_gap']}")
    lines += ['', '距离门限最近的8个候选（非新增准入阈值）：', 'task | q | point/rule | DeltaJS | strict threshold | margin']
    for row in closest:
        lines.append(f"{row['task']} | {row['query']} | {row['point_id']}/{row['rule']} | {row['independent_delta_JS']:.12g} | {row['independent_threshold']:.12g} | {row['independent_threshold_margin']:.12g}")
    lines += ['', '未发现近零分类不稳定不等于获得严格数学误差证书；QUADPACK误差仍是估计。',
        '结论只支持固定seed2及相同context缓存耦合下：B仅原持平Case2可能分歧，即使该任务获胜最多4/8；不意味着其他初始化不可能多数获益。',
        '没有读取未查询目标、调用LLM、拟合GP或执行新优化。']
    (OUT/'seed2_JS_numerical_robustness_review.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('numerical_result','totals','max_absolute_delta_JS_difference',
                                         'minimum_independent_gate_margin_abs')}, ensure_ascii=False, indent=2))
    return int(bool(all_class_changes or all_winner_changes))


if __name__ == '__main__':
    raise SystemExit(main())
