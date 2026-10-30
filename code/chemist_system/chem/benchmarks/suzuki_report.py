"""Population-complete metrics and independent numerical decision reconstruction."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import inspect
import json
from pathlib import Path
import time
import warnings

import numpy as np
from scipy.special import ndtr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from chem.decision.replay import write_json_atomic
from .suzuki_domain import (digest, encode, candidate_grid, region_grid, same_point,
                            validate_point, FrozenSuzukiOracle)
from .suzuki_runner import read_json, check_protocol, installed_sources, ROOT, RESEARCH, METHODS


def _independent_selection_audit(job):
    """Re-fit sklearn directly; use dense linear solves and direct Gaussian EI.

    Does not call MixedGP, justify_select, lgbo_select or production log EI.
    Candidate/region coordinate construction is shared and checked by domain tests.
    """
    root, spec = job
    state = read_json(Path(root) / "runs" / spec["run_id"] / "state.json")
    results = []
    psi0 = None
    with threadpool_limits(1):
        for decision, answer in zip(state['decisions'], state['responses']):
            t = decision['iteration']
            history = state['history'][:t + 3]
            rows = [r['point'] for r in history]
            y = np.array([r['yield_percent'] / 100 for r in history])
            x = encode(rows)
            seed = 20261026 + spec['case'] * 10000 + spec['seed'] * 100 + t
            gp = GaussianProcessRegressor(
                kernel=ConstantKernel(1., (1e-3, 1e3)) * Matern(np.ones(11), (.01, 100), nu=2.5),
                alpha=1e-6, normalize_y=True, n_restarts_optimizer=5, random_state=seed)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', ConvergenceWarning)
                gp.fit(x, y)
            mean, scale = float(y.mean()), float(y.std()) or 1.
            k_train = gp.kernel_(x) + np.eye(len(x)) * 1e-6
            alpha = np.linalg.solve(k_train, (y - mean) / scale)

            def moments(points):
                z = encode(points)
                cross = gp.kernel_(z, x)
                mu = (cross @ alpha) * scale + mean
                var = gp.kernel_.diag(z) - np.einsum('ij,ji->i', cross, np.linalg.solve(k_train, cross.T))
                return mu, np.sqrt(np.maximum(var, 0)) * scale

            candidates = [p for p in candidate_grid(spec['case'], spec['seed'], t)
                          if not any(same_point(p, row) for row in rows)]
            if spec['method'] == 'LLINBO':
                if psi0 is None:
                    fixed = [p for p in candidate_grid(spec['case'], spec['seed'], 0)
                             if not any(same_point(p, row) for row in rows)]
                    psi0 = float(np.max(moments(fixed)[1] ** 2))
                pool = [*candidates, answer['point']]
                mu, sigma = moments(pool)
                multiplier = np.sqrt(2 * np.log(t * 4 * np.pi**2 / .1 * 6))
                acq = mu + multiplier * sigma
                best = int(np.argmax(acq))
                gap = float(acq[best] - acq[-1])
                accepted = gap <= psi0 / t
                selected = pool[-1] if accepted else pool[best]
                assert abs(gap - decision['ucb_gap']) < 1e-8, 'UCB gap reconstruction mismatch'
                assert abs(psi0 / t - decision['psi']) < 1e-10, 'initial variance schedule mismatch'
                assert accepted == decision['accepted_llm'], 'GPR acceptance mismatch'
                result = {'run_id': spec['run_id'], 'iteration': t, 'accepted': accepted}
                audit_pool, audit_scores = pool, acq
            else:
                grid, a = region_grid(answer['region'], seed, 512)
                g = encode(grid)
                train_g = gp.kernel_(x, g)
                post_g = (gp.kernel_(g) - train_g.T @ np.linalg.solve(k_train, train_g)) * scale**2
                regional_var = max(float(a @ post_g @ a), 0.)
                lam = answer['confidence'] / np.sqrt(max(regional_var, 1e-12))
                mu, sigma = moments(candidates)
                shifts = []
                for j in range(0, len(candidates), 256):
                    z = encode(candidates[j:j+256])
                    cross = (gp.kernel_(z, g) - gp.kernel_(z, x) @ np.linalg.solve(k_train, train_g)) * scale**2
                    shifts.extend(lam * (cross @ a))
                shifts = np.array(shifts)
                delta = mu + shifts - y.max()
                z = delta / np.maximum(sigma, 1e-300)
                ei = np.maximum(delta * ndtr(z) + sigma * np.exp(-z**2 / 2) / np.sqrt(2*np.pi), 0)
                assert ei.max() > 0, 'direct EI underflow; independent high precision audit required'
                i = int(np.argmax(ei))
                selected = candidates[i]
                actual_i = next(j for j,p in enumerate(candidates) if same_point(p,decision['point'],1e-10))
                assert abs(lam - decision['lambda']) <= 1e-7 * max(1, abs(lam)), 'lambda reconstruction mismatch'
                assert abs(shifts[actual_i] - decision['selected_mean_shift']) < 1e-8, 'posterior mean shift mismatch'
                assert abs(sigma[actual_i] - decision['selected_sigma']) < 1e-8, 'unchanged sigma mismatch'
                result = {'run_id': spec['run_id'], 'iteration': t, 'mean_shift': float(shifts[actual_i]),
                          'independent_ei_at_saved_choice': float(ei[actual_i])}
                audit_pool, audit_scores = candidates, ei
            result['exact_point_match'] = same_point(selected,decision['point'],1e-10)
            result['machine_precision_tie'] = False
            if not same_point(selected, decision['point'], 1e-10):
                actual_i = next(j for j,p in enumerate(audit_pool) if same_point(p,decision['point'],1e-10))
                diagnostic = {'run_id':spec['run_id'],'iteration':t,'method':spec['method'],
                              'independent_argmax':selected,'saved_choice':decision['point'],
                              'maximum_independent_score':float(np.max(audit_scores)),
                              'saved_choice_independent_score':float(audit_scores[actual_i]),
                              'absolute_score_gap':float(np.max(audit_scores)-audit_scores[actual_i])}
                # Dense solves and Cholesky prediction can order multiple equal
                # maxima differently. Accept only a 64-ULP score equivalence,
                # never a broad absolute EI threshold or a changed trajectory.
                bound = float(64 * np.spacing(abs(float(np.max(audit_scores)))))
                diagnostic['score_equivalence_bound_64_ulp'] = bound
                if diagnostic['absolute_score_gap'] > bound:
                    write_json_atomic(Path(root)/'audit_mismatches'/f"{spec['run_id']}_t{t}_substantive.json",diagnostic)
                    raise AssertionError(f"independent choice mismatch: {diagnostic}")
                result.update(machine_precision_tie=True, tie_diagnostic=diagnostic)
            result['status'] = 'PASS'
            results.append(result)
    return results


def _stats(values):
    a = np.array(values, dtype=float)
    return {'n': len(a), 'mean': float(a.mean()), 'sd': float(a.std(ddof=1)) if len(a) > 1 else 0.,
            'values': a.tolist()}


def _curves(history):
    values = [r['yield_percent'] / 100 for r in history]
    assert len(values) <= 20
    # Carry the last incumbent across an unfilled planned budget. Never drop a run.
    curve = np.maximum.accumulate(values).tolist() if values else [0.]
    curve.extend([curve[-1]] * (20 - len(curve)))
    return np.array(curve)


def report(root=ROOT, verify_selections=False):
    root = Path(root)
    protocol = check_protocol(root)
    manifest = read_json(root / 'manifest.json')
    assert len(manifest) == 60 and set(s['case'] for s in manifest) == {1, 2, 3, 4}
    assert installed_sources() == protocol['hebo_sources'], 'official HEBO package sources changed'
    for path, expected in protocol['oracle_files_sha256'].items():
        assert sha256((root / path).read_bytes()).hexdigest() == expected, 'frozen oracle asset changed'
    states, metrics = {}, []
    counts = {'planned_runs': 60, 'completed_runs': 0, 'failed_runs': 0, 'waiting_runs': 0,
              'queries': 0, 'additional_queries': 0, 'llm_responses': 0, 'hard_domain_violations': 0,
              'duplicate_queries': 0, 'official_hebo_evolution_calls': 0, 'operator_compiles_in_loop': 0}
    max_error = 0.
    for spec in manifest:
        state = read_json(root / 'runs' / spec['run_id'] / 'state.json')
        states[spec['run_id']] = state
        h = state['history']
        assert state['protocol_sha256'] == protocol['protocol_sha256']
        assert [r['point'] for r in h[:4]] == spec['initial_points'], 'paired initialization changed'
        assert len(h) <= 20 and len(state['decisions']) == len(h) - 4
        assert [r['query'] for r in h] == list(range(1, len(h)+1))
        for i, row in enumerate(h):
            validate_point(row['point'])
            assert not any(same_point(row['point'], prev['point']) for prev in h[:i]), 'repeated query'
        oracle = FrozenSuzukiOracle(root / 'oracles', spec['case'])
        actual = oracle.predict([r['point'] for r in h]) * 100
        error = float(np.max(np.abs(actual - np.array([r['yield_percent'] for r in h]))))
        assert error <= 1e-4, 'saved oracle outcomes changed'
        max_error = max(max_error, error)
        if spec['method'] != 'HEBO':
            assert len(state['responses']) == len(state['decisions'])
            for d, a in zip(state['decisions'], state['responses']):
                assert d['response_sha256'] == digest(a)
                t = d['iteration']
                request = read_json(root / 'requests' / a['context_sha256'] / 'request.json')
                saved_hash = request.pop('context_sha256')
                assert saved_hash == digest(request), 'request context hash mismatch'
                assert request['history'] == h[:t+3], 'future or cross-trajectory response leakage'
                assert request['run_id'] == spec['run_id'] and request['iteration'] == t
                assert all(1 <= q <= len(request['history']) for q in a['supporting_queries'])
                raw = read_json(root / 'requests' / saved_hash / 'response.json')
                assert raw == a and raw['provider']['origin'] == 'current_assistant'
        else:
            for d in state['decisions']:
                assert d['official_hebo'] == '0.3.6' and d['native_continuous_suggest']
                assert d['evolution_calls'] == 1 and d['evolution_population'] == 100 and d['evolution_generations'] == 100
                assert d['acquisition_objectives'] == 3 and d['gp_epochs'] == 100
                counts['official_hebo_evolution_calls'] += d['evolution_calls']
        for d in state['decisions']:
            assert d['engine_files_sha256'] == protocol['engine_files_sha256'] and d['operator_compiles_in_loop'] == 0
        counts['queries'] += len(h)
        counts['additional_queries'] += len(state['decisions'])
        counts['llm_responses'] += len(state['responses'])
        counts['completed_runs'] += state['status'] == 'COMPLETED'
        counts['failed_runs'] += state['status'] == 'FAILED'
        counts['waiting_runs'] += state['status'] == 'WAITING_LLM'
        curve = _curves(h)
        metrics.append({**{k: spec[k] for k in ('run_id', 'case', 'seed', 'method')},
                        'status': state['status'], 'queries': len(h), 'best_so_far': curve.tolist(),
                        'search_auc': float(curve[4:20].mean()), 'final_best_yield': float(curve[-1]),
                        'first_query_ge_80': next((i+1 for i, v in enumerate(curve[:len(h)]) if v >= .8), None),
                        'first_query_ge_90': next((i+1 for i, v in enumerate(curve[:len(h)]) if v >= .9), None),
                        'optimization_seconds': state['optimization_seconds'], 'oracle_seconds': state['oracle_seconds']})
    cases = {}
    for case in range(1, 5):
        cases[str(case)] = {}
        for method in METHODS:
            group = [r for r in metrics if r['case'] == case and r['method'] == method]
            group.sort(key=lambda r: r['seed'])
            cases[str(case)][method] = {k: _stats([r[k] for r in group]) for k in ('search_auc', 'final_best_yield')}
            cases[str(case)][method].update(reached_80=sum(r['first_query_ge_80'] is not None for r in group),
                                           reached_90=sum(r['first_query_ge_90'] is not None for r in group))
    overall = {m: {k: _stats([r[k] for r in metrics if r['method'] == m]) for k in ('search_auc', 'final_best_yield')}
               for m in METHODS}
    paired = {}
    for method in ('LLINBO', 'LGBO'):
        differences = np.array([[cases[str(c)][method]['search_auc']['values'][s] - cases[str(c)]['HEBO']['search_auc']['values'][s]
                                 for s in range(5)] for c in range(1,5)])
        rng = np.random.default_rng(20261027)
        means = np.array([np.mean([rng.choice(row, 5, replace=True).mean() for row in differences]) for _ in range(10000)])
        paired[method + '_minus_HEBO'] = {'mean_search_auc_difference': float(differences.mean()),
            'fixed_four_cases_stratified_seed_bootstrap_95_ci': np.quantile(means, [.025, .975]).tolist(),
            'per_case_mean': differences.mean(axis=1).tolist(), 'all_20_paired_differences': differences.ravel().tolist(),
            'scope': 'exploratory resampling interval over 5 seeds in each fixed case; shared LLM context limits independence; not an interval over new chemistry tasks or independent LLM instances'}
    ll = [d for s in states.values() if s['method'] == 'LLINBO' for d in s['decisions']]
    lg = [d for s in states.values() if s['method'] == 'LGBO' for d in s['decisions']]
    mechanism = {'LLINBO_decisions': len(ll), 'LLINBO_accepted': sum(d['accepted_llm'] for d in ll),
                 'LLINBO_rejected': sum(not d['accepted_llm'] for d in ll),
                 'LGBO_decisions': len(lg), 'LGBO_choice_changed_relative_to_same_prefix_untilted_EI': sum(d['guidance_changed_choice'] for d in lg),
                 'counterfactual_boundary': 'same-prefix acquisition diagnostic only, not an additional experimental BO baseline'}
    selection_audit = None
    audit_inputs = {s['run_id']:sha256((root/'runs'/s['run_id']/'state.json').read_bytes()).hexdigest() for s in manifest}
    audit_source = sha256(inspect.getsource(_independent_selection_audit).encode('utf8')).hexdigest()
    if verify_selections:
        started = time.perf_counter()
        jobs = [(str(root), spec) for spec in manifest if spec['method'] != 'HEBO']
        with ProcessPoolExecutor(max_workers=4) as executor:
            rows = [r for batch in executor.map(_independent_selection_audit, jobs) for r in batch]
        ties = sum(r['machine_precision_tie'] for r in rows)
        selection_audit = {'status': 'PASS_WITH_MACHINE_PRECISION_TIES' if ties else 'PASS',
                           'protocol_sha256':protocol['protocol_sha256'], 'input_state_sha256':audit_inputs,
                           'reconstruction_source_sha256':audit_source,
                           'decisions': len(rows), 'independent_gp_refits': len(rows),
                           'exact_point_matches':sum(r['exact_point_match'] for r in rows), 'machine_precision_ties':ties,
                           'seconds': time.perf_counter() - started, 'rows': rows,
                           'boundary': 'independent dense posterior/UCB/EI reconstruction; 64-ULP score-equivalent maxima recorded separately from exact point identity; all original trajectories unchanged; HEBO source/path checked separately'}
        write_json_atomic(root / 'independent_selection_audit.json', selection_audit)
    elif (root / 'independent_selection_audit.json').exists():
        cached = read_json(root / 'independent_selection_audit.json')
        if (cached.get('protocol_sha256') != protocol['protocol_sha256'] or cached.get('input_state_sha256') != audit_inputs
                or cached.get('reconstruction_source_sha256') != audit_source):
            raise ValueError('cached independent audit is stale; use --verify-selections')
        selection_audit = cached
    hebo_audit_file = root / 'hebo_numerical_audit.json'
    hebo_numerics = None
    if hebo_audit_file.exists():
        hebo_audit = read_json(hebo_audit_file)
        hebo_numerics = {k:v for k,v in hebo_audit.items() if k != 'runs'}
        hebo_numerics['affected_runs'] = [r['run_id'] for r in hebo_audit['runs'] if any(
            d['fit_epochs_abandoned'] or d['random_prediction_fallback_calls'] for d in r['iterations'])]
    result = {'schema_version': 1, 'protocol_sha256': protocol['protocol_sha256'], 'counts': counts,
              'max_saved_oracle_error_percentage_points': max_error, 'per_case': cases, 'overall': overall,
              'paired_search_auc': paired, 'mechanism': mechanism, 'runs': metrics, 'hebo_numerical_audit': hebo_numerics,
              'selection_audit': None if selection_audit is None else {k:v for k,v in selection_audit.items() if k != 'rows'},
              'limitations': [protocol['llm']['contamination_disclosure'],
                  'Five seeds per case; no fresh held-out chemistry task or physical experiment.',
                  'LLINBO and LGBO are documented mixed-domain adaptations, not exact original-paper reproductions.',
                  'HEBO uses native continuous evolution while LLM methods approximate acquisition maxima using Sobol candidates.',
                  'UCB/EI and surrogate differences confound attribution to chemical knowledge alone.',
                  'Optimization timings exclude interactive assistant reasoning and waiting; full LLM token/cost measurements are unavailable.',
                  'Software qualification does not establish scientific superiority.',
                  'Native HEBO numerical fallbacks are retained and reported; this comparator is not a numerically clean HEBO qualification.']}
    repaired_root = root / 'hebo_default64'
    if (repaired_root / 'status.json').exists():
        result['hebo_precision_check'] = _precision_summary(root, protocol, metrics)
    write_json_atomic(root / 'results.json', result)
    write_json_atomic(RESEARCH / 'results.json', result)
    _write_table(result)
    _figures(result)
    return {k:v for k,v in result.items() if k not in ('runs', 'per_case')}


def _write_table(result):
    lines = ['# Suzuki Case 1–4：LLINBO、LGBO、HEBO', '',
             '每个 Case 5 个配对种子；4 次相同初始实验 + 16 次新增查询。数值为均值 ± 种子标准差。SearchAUC 为新增预算位置上的 best-so-far 产率均值，表中以百分数显示。', '',
             '| Case | 方法 | 最终最好产率 % | SearchAUC % | ≥80% | ≥90% |',
             '|---|---|---:|---:|---:|---:|']
    for case, methods in result['per_case'].items():
        for method, data in methods.items():
            final, auc = data['final_best_yield'], data['search_auc']
            lines.append(f"| {case} | {method} | {final['mean']*100:.2f} ± {final['sd']*100:.2f} | {auc['mean']*100:.2f} ± {auc['sd']*100:.2f} | {data['reached_80']}/5 | {data['reached_90']}/5 |")
    lines += ['', 'LLM 回答来自当前共享聊天，已接触历史 Suzuki 工作；本比较不是盲测。LLINBO、LGBO 的适配和论文/代码差异见 protocol.json。连续域全局最优值未知，不报告真实全局后悔值。', '']
    a = result['hebo_numerical_audit']
    if a:
        lines += [f"官方 HEBO 的 {a['decisions']} 次选点均重放核对，其中 {a['steps_with_random_prediction_fallback']} 轮触发随机预测回退，"
                  f"涉及 {len(a['affected_runs'])}/20 条轨迹；另有 {a['abandoned_fit_epochs']} 次 GP 训练更新因数值异常放弃。"
                  '这些输出均保留，不能将与本组 HEBO 的差异解释为优于数值完全正常的 HEBO。', '']
    (RESEARCH / 'results_tables.md').write_text('\n'.join(lines), encoding='utf-8')


def _precision_summary(root, parent, primary_metrics):
    freeze = read_json(RESEARCH / 'hebo_precision_protocol.json')
    assert freeze['parent_protocol_sha256'] == parent['protocol_sha256']
    assert freeze['script_sha256'] == sha256(Path('scripts/suzuki_hebo_precision.py').read_bytes()).hexdigest()
    assert freeze['hebo_sources'] == installed_sources()
    rows = []
    failed = 0
    max_error = 0.
    total_decisions = 0
    for spec in read_json(root / 'manifest.json'):
        if spec['method'] != 'HEBO': continue
        s = read_json(root / 'hebo_default64' / spec['run_id'] / 'state.json')
        assert s['precision_protocol_sha256'] == freeze['precision_protocol_sha256']
        assert [h['point'] for h in s['history'][:4]] == spec['initial_points']
        assert len(s['history']) <= 20 and len(s['decisions']) == len(s['history']) - 4
        failed += s['status'] == 'FAILED'
        for i,h in enumerate(s['history']):
            validate_point(h['point'])
            assert not any(same_point(h['point'], r['point']) for r in s['history'][:i])
        oracle = FrozenSuzukiOracle(root / 'oracles',spec['case'])
        y = oracle.predict([h['point'] for h in s['history']]) * 100
        error = float(np.max(np.abs(y - np.array([h['yield_percent'] for h in s['history']]))))
        assert error <= 1e-4
        max_error = max(max_error,error)
        for d in s['decisions']:
            assert d['fit_epochs_abandoned'] == d['random_prediction_fallback_calls'] == 0
            assert d['measured_gp_prediction_dtypes'] == [['torch.float64','torch.float64']]
            assert d['evolution_population'] == d['evolution_generations'] == d['gp_epochs'] == 100
            assert d['acquisition_objectives'] == 3
            total_decisions += 1
        curve = _curves(s['history'])
        rows.append({'case':spec['case'],'seed':spec['seed'],'run_id':spec['run_id'], 'status':s['status'],
                     'best_so_far':curve.tolist(),'search_auc':float(curve[4:].mean()),
                     'final_best_yield':float(curve[-1]),'queries':len(s['history'])})
    cases = {str(c):{k:_stats([r[k] for r in rows if r['case']==c]) for k in ('search_auc','final_best_yield')}
             for c in range(1,5)}
    summary = {'protocol_sha256':freeze['precision_protocol_sha256'],'planned_runs':20,
               'completed_runs':sum(r['status']=='COMPLETED' for r in rows),'failed_runs':failed,
               'oracle_queries':sum(r['queries'] for r in rows),'measured_float64_gp_decisions':total_decisions,
               'accepted_decisions_with_upstream_numerical_fallbacks':0,
               'max_saved_oracle_error_percentage_points':max_error,'per_case':cases,'runs':rows,
               'role':'secondary numerical repair; original native HEBO retained; not all-float64 input arithmetic',
               'overall':{k:_stats([r[k] for r in rows]) for k in ('search_auc','final_best_yield')}}
    write_json_atomic(RESEARCH / 'hebo_precision_results.json',summary)
    table = ['# HEBO 默认参数双精度核验', '',
             '主实验三算法不变。HEBO64 是数值故障发现后单独冻结的精度修复，原生 HEBO 全部保留。每列仍为同一 Case 的5个配对种子。数值为 SearchAUC %（均值 ± 种子标准差）。', '',
             '| Case | LLINBO | LGBO | HEBO 原生 | HEBO 默认参数双精度 |', '|---|---:|---:|---:|---:|']
    for c in range(1,5):
        cells = []
        for m in METHODS:
            v = _stats([r['search_auc'] for r in primary_metrics if r['case']==c and r['method']==m])
            cells.append(f"{v['mean']*100:.2f} ± {v['sd']*100:.2f}")
        v=cases[str(c)]['search_auc']
        table.append(f"| {c} | " + ' | '.join(cells) + f" | {v['mean']*100:.2f} ± {v['sd']*100:.2f} |")
    table += ['', f"修复核验完成 {summary['completed_runs']}/20 条轨迹，实际 float64 GP 决策 {total_decisions} 次。失败轨迹保留在分母中。", '',
              'UCB/EI、代理模型和采集函数搜索实现存在差异；本表不能单独识别化学知识注入的净收益。共享聊天也限制了 LLM 独立性。', '']
    (RESEARCH / 'hebo_precision_comparison.md').write_text('\n'.join(table),encoding='utf8')
    return {k:v for k,v in summary.items() if k != 'runs'}


def _figures(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), sharex=True, sharey=True, layout='constrained')
    styles = {'LLINBO': ('#2878B5', '-'), 'LGBO': ('#D25B38', '--'), 'HEBO': ('#3A8D5D', '-.')}
    for case, ax in zip(range(1,5), axes.flat):
        for method in METHODS:
            curves = np.array([r['best_so_far'] for r in result['runs'] if r['case'] == case and r['method'] == method]) * 100
            mean, sd = curves.mean(axis=0), curves.std(axis=0, ddof=1)
            color, line = styles[method]
            ax.plot(np.arange(1,21), mean, label=method, color=color, linestyle=line, linewidth=2)
            ax.fill_between(np.arange(1,21), np.maximum(0,mean-sd), np.minimum(100,mean+sd), color=color, alpha=.11)
        ax.axvline(4.5, color='gray', linewidth=.8, linestyle=':')
        ax.set(title=f'Reizman Suzuki Case {case}', xlim=(1,20), ylim=(0,100), xticks=[1,4,8,12,16,20])
        ax.grid(alpha=.15)
    for ax in axes[-1]: ax.set_xlabel('Total oracle queries (4 shared initial + 16 sequential)')
    for ax in axes[:,0]: ax.set_ylabel('Best-so-far yield (%)')
    axes[0,0].legend(loc='upper left', fontsize=8)
    fig.suptitle('5 paired seeds/case; bands = seed SD (clipped to 0–100%)\n'
                 'Native HEBO: 5/320 fallback steps; LLM answers share one chat', fontsize=11)
    folder = RESEARCH / 'figures'
    folder.mkdir(parents=True, exist_ok=True)
    for ext in ('png','pdf','svg'):
        fig.savefig(folder / f'suzuki_learning_curves.{ext}', dpi=180)
    plt.close(fig)
