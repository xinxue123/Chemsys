"""Complete denominators, frozen comparators, and independent algebra audit."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path
import csv
import json
import math
import statistics
import xml.etree.ElementTree as ET

import numpy as np
import torch

from . import grounded_regional_suzuki_study as s
from . import grounded_regional_suzuki_engine as e
from . import regional_ablation_suzuki_report as original_audit
from . import agent_score_suzuki_report as metrics
from .suzuki_domain import digest

COMPARATORS = ('G3_original', 'Gate_only', 'LGBO', 'LLINBO_Justify', 'Pure_BO')
ALL_METHODS = ('G3_original', *s.METHODS, *COMPARATORS[1:])
LABELS = {'G3_original': 'Original G3', 'R1_grounded': 'R1: grounded',
          'R2_conflict_fallback': 'R2: + conflict fallback', 'R3_conservative_region': 'R3: + conservative region',
          'Gate_only': 'Gate only', 'LGBO': 'Regional LGBO', 'LLINBO_Justify': 'LLINBO-Justify', 'Pure_BO': 'Pure BO'}


def rows():
    return [s.read(s.ROOT / 'runs' / spec['run_id'] / 'state.json') for spec in s.read(s.ROOT / 'manifest.json')]


def controls():
    result = []
    for c in (1, 2, 3, 4):
        for regime in s.REGIMES:
            paths = {'G3_original': s.previous.ROOT / 'runs' / f'c{c}_s0_{regime}_G3_gate_KF_B4_event' / 'state.json',
                'Gate_only': s.previous.ROOT / 'runs' / f'c{c}_s0_{regime}_G0_gate' / 'state.json',
                'LGBO': s.previous.prior.ROOT / 'runs' / f'c{c}_s0_{regime}_LGBO_region' / 'state.json',
                'LLINBO_Justify': s.previous.prior.ROOT / 'runs' / f'c{c}_s0_{regime}_LLINBO_matched' / 'state.json',
                'Pure_BO': s.previous.prior.previous.ROOT / 'base' / f'c{c}_s0' / 'state.json'}
            for method, path in paths.items():
                row = s.read(path)
                result.append({**row, 'method': method, 'regime': regime, 'source_path': path.as_posix()})
    return result


def raw_metric(row):
    y = np.asarray([h['y'] for h in row['history']]) * 100
    first = int(np.flatnonzero(y == y.max())[0]) + 1
    return {'case': row['case'], 'regime': row['regime'], 'method': row['method'],
        'status': row['status'], 'SearchAUC': metrics.metric(row)['AUC'], 'final_best': float(y.max()),
        'first_own_best_total': first, 'first_own_best_added': max(0, first - 4)}


def usage():
    responses = [s.read(p) for p in (s.ROOT / 'requests').glob('*/attempt*/response.json')]
    states = rows()
    return {'logical_refreshes': sum(len(v['calls']) for v in states),
        'accepted_refreshes': sum(c['status'] == 'ACCEPTED' for v in states for c in v['calls']),
        'failed_refreshes': sum(c['status'] != 'ACCEPTED' for v in states for c in v['calls']),
        'received_responses': len(responses),
        'HTTP_errors': len(list((s.ROOT / 'requests').glob('*/attempt*/transport_error.json'))),
        'factual_or_schema_repairs': len(list((s.ROOT / 'requests').glob('*/attempt*/validation_error.json'))),
        'checked_facts': sum(len(c['plan']['checked_facts']) for v in states for c in v['calls'] if c['status'] == 'ACCEPTED'),
        'models': sorted({v['model'] for v in responses}),
        'fingerprints': sorted({v['system_fingerprint'] for v in responses if v.get('system_fingerprint')}),
        **{k: sum(v['usage'].get(k, 0) for v in responses) for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')}}


def export_csv(path, values):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def report():
    protocol = s.check_protocol()
    states = rows()
    assert len(states) == 48 and all(r['status'] in ('COMPLETE', 'FAILED') for r in states)
    allrows = controls() + states
    raw = [raw_metric(r) for r in allrows]
    summary = {}
    for name, cases in [('primary', (2, 3, 4)), ('all_four', (1, 2, 3, 4))]:
        summary[name] = {}
        for regime in (*s.REGIMES, 'all_conditions'):
            group = [v for v in raw if v['case'] in cases and (regime == 'all_conditions' or v['regime'] == regime)]
            summary[name][regime] = {method: {
                'n_case_conditions': len([v for v in group if v['method'] == method]),
                'SearchAUC_mean': statistics.mean(v['SearchAUC'] for v in group if v['method'] == method),
                'final_best_mean': statistics.mean(v['final_best'] for v in group if v['method'] == method)}
                for method in ALL_METHODS}
    paired = []
    for case in (1, 2, 3, 4):
        for regime in s.REGIMES:
            group = {v['method']: v for v in raw if v['case'] == case and v['regime'] == regime}
            for method, reference in [(s.METHODS[0], 'G3_original'), (s.METHODS[1], s.METHODS[0]),
                                      (s.METHODS[2], s.METHODS[1]), (s.METHODS[2], 'G3_original'),
                                      (s.METHODS[2], 'Gate_only'), (s.METHODS[2], 'Pure_BO'),
                                      (s.METHODS[2], 'LLINBO_Justify')]:
                paired.append({'case': case, 'regime': regime, 'method': method, 'reference': reference,
                    'SearchAUC_delta_pp': group[method]['SearchAUC'] - group[reference]['SearchAUC'],
                    'final_best_delta_pp': group[method]['final_best'] - group[reference]['final_best']})
    rounds = []
    for state in states:
        incumbent = max(h['y'] for h in state['history'][:4])
        for ev, h in zip(state['decisions'], state['history'][4:]):
            incumbent = max(incumbent, h['y'])
            rounds.append({'case': state['case'], 'regime': state['regime'], 'method': state['method'],
                'query': ev['iteration'], 'selected_id': ev['selected_id'], 'base_reference_id': ev['reference_id'],
                'yield_percent': h['y'] * 100, 'incumbent_percent': incumbent * 100,
                'GP_mean_percent': ev['measurement']['prequery_mu_y'] * 100,
                'GP_observation_SD_pp': ev['measurement']['prequery_predictive_sd_y'] * 100,
                'raw_region_shift_pp': ev['raw_shift_y'] * 100, 'deployed_shift_pp': ev['deployed_shift_y'] * 100,
                'amplitude': ev['amplitude'], 'amplitude_SD': ev['deployment']['amplitude_SD'],
                'neutral_fallback': ev['deployment']['neutral_fallback'],
                'conservative_penalty': ev['deployment']['conservative_penalty_applied'],
                'B4_revoked_ids': ','.join(ev['diagnosis']['revoked_ids']),
                'eligible_candidates': ev['eligible_candidates'], 'base_logei_loss': ev['base_logei_loss'],
                'budget_logei': ev['budget_logei'], 'decision_reason': ev['deployment']['reason'],
                'LLM_refresh': any(c['iteration'] == ev['iteration'] for c in state['calls'])})
    mechanisms = {m: {'fallback_rounds': sum(v['neutral_fallback'] for v in rounds if v['method'] == m),
        'new_region_penalty_rounds': sum(v['conservative_penalty'] for v in rounds if v['method'] == m),
        'base_argmax_rounds': sum(v['selected_id'] == v['base_reference_id'] for v in rounds if v['method'] == m),
        'singleton_budget_rounds': sum(v['eligible_candidates'] == 1 for v in rounds if v['method'] == m)} for m in s.METHODS}
    result = {'study': protocol['study'], 'protocol_sha256': protocol['protocol_sha256'],
        'planned': 48, 'completed': sum(v['status'] == 'COMPLETE' for v in states),
        'additional_queries': sum(len(v['decisions']) for v in states), 'raw_metrics': raw,
        'summary': summary, 'paired_deltas': paired, 'usage': usage(), 'mechanisms': mechanisms,
        'limits': ['One seed; fresh LLM realizations and changed prompts are bundled in R1.',
            'R2-R1 and R3-R2 are cumulative full-trajectory comparisons conditional on earlier changes; LLM feedback differs on diverged paths.',
            'Pure BO is the same frozen trajectory repeated in four condition tables, not four independent repetitions.',
            'Three artificial knowledge conditions are outcome-aware stress priors.',
            'Typed fact checks do not validate causal chemistry; one-SD deployment is a heuristic, not a guaranteed confidence bound.',
            'No post-result parameter tuning; all planned trajectories retained, failed incomplete histories carry their incumbent.']}
    s.write(s.OUT / 'results.json', result)
    export_csv(s.OUT / 'metrics.csv', raw)
    export_csv(s.OUT / 'paired_deltas.csv', paired)
    export_csv(s.OUT / 'rounds.csv', rounds)
    figures(allrows)
    write_report(result)
    return {'completed': result['completed'], 'additional_queries': result['additional_queries'],
            'primary': summary['primary'], 'usage': result['usage']}


def figures(allrows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods = ('G3_original', *s.METHODS, 'Pure_BO')
    colors = ('#777777', '#CC79A7', '#0072B2', '#D55E00', '#111111')
    fig, axes = plt.subplots(4, 4, figsize=(17, 11), layout='constrained')
    for i, case in enumerate((1, 2, 3, 4)):
        for j, regime in enumerate(s.REGIMES):
            ax = axes[i, j]
            for method, color in zip(methods, colors):
                row = next(v for v in allrows if v['case'] == case and v['regime'] == regime and v['method'] == method)
                ax.plot(range(1, 17), metrics.metric(row)['curve'], color=color, lw=1.7,
                        linestyle='--' if method in ('G3_original', 'Pure_BO') else '-', label=LABELS[method])
            ax.set(title=f'Case {case} | {regime}', xlabel='Additional query', ylabel='Best yield (%)', xlim=(1, 16), ylim=(0, 100))
            ax.grid(alpha=.18)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='outside lower center', ncol=5)
    fig.savefig(s.OUT / 'all_conditions.png', dpi=160)
    plt.close(fig)
    methods = ('G3_original', s.METHODS[1], s.METHODS[2], 'LLINBO_Justify', 'Pure_BO')
    colors = ('#777777', '#0072B2', '#D55E00', '#009E73', '#111111')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for i, case in enumerate((3, 4)):
        for method, color in zip(methods, colors):
            row = next(v for v in allrows if v['case'] == case and v['regime'] == 'assistant_initial' and v['method'] == method)
            y = [h['y'] * 100 for h in row['history']]
            axes[i, 0].plot(range(1, 21), y, color=color, marker='.', ms=4, lw=1., alpha=.85, label=LABELS[method])
            axes[i, 1].plot(range(1, 21), np.maximum.accumulate(y), color=color, marker='.', ms=4, lw=1.6, label=LABELS[method])
        for j, name in enumerate(('Observed yield', 'Cumulative best')):
            axes[i, j].axvspan(.5, 4.5, color='#dddddd', alpha=.3)
            axes[i, j].set(title=f'Case {case} | {name}', xlabel='Experiment (4 initial + 16 queries)', ylabel='Yield (%)', ylim=(0, 100), xlim=(1, 20))
            axes[i, j].set_xticks([1, 4, 8, 12, 16, 20])
            axes[i, j].grid(alpha=.18)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='outside lower center', ncol=3)
    fig.savefig(s.OUT / 'case3_case4_assistant_initial.png', dpi=180)
    plt.close(fig)


def write_report(result):
    summary = result['summary']['primary']['all_conditions']
    lines = ['# G3 修复后完整轨迹验证', '',
        'Mixed GP、4692 个候选、seed=0、4 次初始化＋16 次查询保持一致。新跑 48 条完整轨迹；原 G3、仅预算门限、LGBO、LLINBO-Justify 和纯 BO 均为只读历史对照。', '',
        'R1：具名归一化/物理变量，逐条验证 LLM 引用的已观测数值。R2：再加入 B4 冲突时部署幅度为零，回退基础 qLogEI。R3：再对初始区域以外的几何使用 α=clip(a−sqrt(Paa),0,1)。初始几何不减幅度；冲突优先回退；KF 仍用未缩放区域场进行影子校准。', '',
        '规则与参数在查询前冻结；本次没有根据结果调参。事实校验不证明化学因果，单个 seed 不支持显著性结论。', '',
        '## 主测试 Case 2–4，四种知识条件平均', '',
        '| 方法 | SearchAUC | 最终最佳产率 |', '|---|---:|---:|']
    for method in ALL_METHODS:
        v = summary[method]
        lines.append(f"| {LABELS[method]} | {v['SearchAUC_mean']:.4f} | {v['final_best_mean']:.4f}% |")
    lines += ['', 'SearchAUC 是新增查询第 1–16 次的累计最佳产率平均值；单位为百分点。下表首次最佳以全部实验序号计，括号为新增查询序号；初始化已到最佳时新增序号记 0。', '']
    for case in (1, 2, 3, 4):
        lines += [f'## Case {case}' + ('（参考案例）' if case == 1 else ''), '',
                  '| 知识条件 | 方法 | SearchAUC | 最终最佳 | 首次最佳：全部（新增） |', '|---|---|---:|---:|---:|']
        for regime in s.REGIMES:
            for method in ALL_METHODS:
                v = next(v for v in result['raw_metrics'] if v['case'] == case and v['regime'] == regime and v['method'] == method)
                lines.append(f"| {regime} | {LABELS[method]} | {v['SearchAUC']:.4f} | {v['final_best']:.4f}% | {v['first_own_best_total']}（{v['first_own_best_added']}） |")
        lines.append('')
    lines += ['## 使用与证据范围', '', '```json', json.dumps(result['usage'], ensure_ascii=False, indent=2), '```', '',
        '逐轮表 `rounds.csv` 记录候选、基础 BO 参考点、GP 均值/观测标准差、原始/部署区域偏移、幅度/标准差、B4 删除条款、预算、刷新与决策原因。每次请求的原始提示词、响应、事实检查和修复均保存在独立 requests 目录。', '',
        'R1 同时改变提示词和 LLM 实现，不是只改变一个数值开关；R2/R3 轨迹分叉后各自接收新的实验反馈。累计档位差值反映这些完整策略的效果，不能直接解释成模块的独立因果贡献。纯 BO 在四张知识条件表中引用同一条轨迹。', '']
    (s.OUT / 'report.md').write_text('\n'.join(lines), encoding='utf8')


def audit_case(case):
    protocol = s.check_protocol()
    pool = s.previous.inputs(case)
    ids = pool['ids']
    ix = {key: i for i, key in enumerate(ids)}
    sources = [s.ROOT / 'cache' / f'c{case}_s0']
    cache = e.GroundedRegionalCache(pool, s.ROOT / 'audit_cache' / f'c{case}_s0', case, sources=sources)
    values = np.load(s.previous.prior.original.ROOT / 'inputs' / f'case{case}' / 'oracle.npz')['values']
    choices = calls = score_count = facts = 0
    max_score = max_shift = max_filter = max_perm = 0.
    roundoff = 0
    fits, scores, fields = set(), set(), {}
    specs = [v for v in s.read(s.ROOT / 'manifest.json') if v['case'] == case]
    for spec in specs:
        actual = s.read(s.ROOT / 'runs' / spec['run_id'] / 'state.json')
        assert actual['status'] == 'COMPLETE' and actual['protocol_sha256'] == protocol['protocol_sha256']
        knowledge = s.previous.initial_knowledge(case, spec['regime'])
        state = {**spec, 'history': s.previous.prior.initial(case, pool), 'knowledge': knowledge,
            'initial_knowledge': deepcopy(knowledge), 'slots': {h['id']: h['coordinate'] for h in knowledge['clauses']},
            'banks': {}, 'decisions': [], 'calls': []}
        assert actual['history'][:4] == state['history'] and actual['initial_knowledge'] == knowledge
        assert len({h['candidate_id'] for h in actual['history']}) == 20
        for t, ev in enumerate(actual['decisions'], 1):
            before = e.setup(cache, state, pool, t)
            reasons = e.original.event_reasons(state, before, t)
            call = next((c for c in actual['calls'] if c['iteration'] == t), None)
            assert bool(reasons) == bool(call)
            expected_before = {'diagnosis': before['diagnosis'].to_dict(), 'deployment': before['deployment'],
                               'selected_id': ids[before['selected']]}
            assert json.loads(json.dumps(expected_before)) == ev['pre_refresh']
            if call:
                assert call['reasons'] == reasons and call['knowledge_before'] == state['knowledge']
                folder = s.ROOT / 'requests' / call['context_sha256']
                context = s.read(folder / 'context.json')
                public = s.public_request(state, pool, before, reasons)
                assert context['public'] == json.loads(json.dumps(public)) and context['history'] == state['history']
                raw_context = deepcopy(context)
                key = raw_context.pop('context_sha256')
                assert digest(raw_context) == key == call['context_sha256']
                if call['status'] == 'ACCEPTED':
                    valid = s.read(folder / 'validated.json')
                    answer_path = folder / f"attempt{valid['accepted_attempt']:02d}" / 'response.json'
                    answer = s.read(answer_path)
                    assert s.filehash(answer_path) == valid['response_sha256'] and answer['model'] == s.api.MODEL
                    assert answer['choices'][0]['finish_reason'] == 'stop'
                    plan = e.validate_refresh(json.loads(answer['choices'][0]['message']['content']),
                        state['knowledge'], state['slots'], context['public']['observations'], pool)
                    assert plan == valid['plan'] == call['plan']
                    first_messages = [{'role': 'system', 'content': e.SYSTEM_PROMPT},
                        {'role': 'user', 'content': json.dumps(context['public'], ensure_ascii=False, separators=(',', ':'))}]
                    received = 0
                    for n in range(1, valid['accepted_attempt'] + 1):
                        attempt = folder / f'attempt{n:02d}'
                        body = s.read(attempt / 'request.json')
                        assert {k: v for k, v in body.items() if k != 'messages'} == s.PARAMETERS
                        assert body['messages'][:2] == first_messages and len(body['messages']) in (2, 4)
                        if (attempt / 'response.json').exists():
                            received += 1
                            if n < valid['accepted_attempt']:
                                old = s.read(attempt / 'response.json')
                                try:
                                    assert old['model'] == s.api.MODEL and old['choices'][0]['finish_reason'] == 'stop'
                                    e.validate_refresh(json.loads(old['choices'][0]['message']['content']),
                                        state['knowledge'], state['slots'], context['public']['observations'], pool)
                                except (ValueError, TypeError, KeyError, IndexError, AssertionError):
                                    pass
                                else:
                                    raise AssertionError('valid earlier response skipped')
                    assert received == valid['received_responses'] <= 3
                    facts += len(plan['checked_facts'])
                    state['knowledge'] = plan['knowledge']
                state['calls'].append(call)
                calls += 1
            d = e.setup(cache, state, pool, t)
            assert d['base_meta'] == ev['base_acquisition'] and d['guided_meta'] == ev['guided_acquisition']
            assert d['field_meta'] == ev['field_acquisition'] and state['knowledge'] == ev['knowledge']
            assert json.loads(json.dumps(d['diagnosis'].to_dict())) == ev['diagnosis']
            assert d['retained'] == ev['retained_ids'] and ids[d['reference']] == ev['reference_id']
            assert d['bank_key'] == ev['bank_key'] and d['bank'] == ev['bank_before'] and d['prior'] == ev['bank_prior']
            assert d['deployment'] == ev['deployment']
            keep, _, _ = original_audit.subsets.independent_subset(d['base'], e.original.membership(pool, state['knowledge']),
                d['budget'], ~d['available'], ids, [h['id'] for h in state['knowledge']['clauses']])
            assert keep == tuple(d['retained'])
            # Recompute the new deployment rule without calling deployment().
            initial_key = e.original.region_key(state['initial_knowledge'], [h['id'] for h in state['initial_knowledge']['clauses']])
            alpha = float(np.clip(d['prior']['mean'][1], 0., 1.)) if d['prior'] else 1.
            if spec['method'] == s.METHODS[2] and d['bank_key'] and d['bank_key'] != initial_key:
                alpha = float(np.clip(d['prior']['mean'][1] - math.sqrt(d['prior']['covariance'][1][1]), 0., 1.))
            conflict = bool(d['diagnosis'].revoked_ids) or d['diagnosis'].conflict_kind != 'none'
            if spec['method'] in s.METHODS[1:] and conflict:
                alpha = 0.
            assert alpha == d['amplitude'] == ev['amplitude']
            j = min(np.flatnonzero(d['eligible']), key=lambda k: (-float(d['score'][k]), ids[k]))
            assert ids[j] == ev['selected_id'] and d['budget'] == ev['budget_logei']
            assert int(d['eligible'].sum()) == ev['eligible_candidates']
            assert float(d['base'][d['reference']] - d['base'][j]) == ev['base_logei_loss'] <= d['budget']
            if spec['method'] in s.METHODS[1:] and conflict:
                assert j == d['reference']
            fit, value = cache.model(state['history'])
            stats = d['stats']
            if fit not in fits:
                saved = torch.load(cache.directory / (fit + '.pt'), map_location='cpu', weights_only=False)
                expected_X = e.original.common.mixed.encode_mixed(pool)[[ix[h['candidate_id']] for h in sorted(state['history'], key=lambda h: h['candidate_id'])]]
                assert np.array_equal(saved['train_X'], expected_X)
                pm, ps = original_audit.mixed_report.permutation_posterior(cache, state['history'], value['model'], value)
                error = max(float(np.max(np.abs(pm - stats['mu_std']))), float(np.max(np.abs(ps - stats['sd_std']))))
                max_perm = max(max_perm, error)
                assert error < 1e-9
                fits.add(fit)
            field_key = (fit, d['bank_key'])
            if field_key not in fields:
                field = np.zeros(len(ids)) if not d['retained'] else original_audit.independent_shift(
                    value['model'], e.original.region(state['knowledge'], d['retained']), cache.z)
                error = float(np.max(np.abs(field - d['raw_shift_std'])))
                assert error < 1e-10
                max_shift = max(max_shift, error)
                fields[field_key] = field
            score_key = d['guided_meta']['key']
            if score_key not in scores:
                applied = fields[field_key] * alpha
                replay = original_audit.algebra.marginal_scores(stats['mu_std'], stats['sd_std'], applied,
                    value['best_std'], d['base_meta']['mc_seed'], cache.z)
                error = float(np.max(np.abs(replay - d['score'])))
                max_score = max(max_score, error)
                if error >= 1e-8:
                    exact_shift = original_audit.algebra.marginal_scores(stats['mu_std'], stats['sd_std'], d['shift_std'],
                        value['best_std'], d['base_meta']['mc_seed'], cache.z)
                    control_error = float(np.max(np.abs(exact_shift - d['score'])))
                    bound = float(np.max(np.abs(applied - d['shift_std']))) / 1e-6 + control_error + 1e-10
                    assert control_error < 1e-8 and error <= bound
                    roundoff += 1
                scores.add(score_key)
                score_count += len(ids)
            y = float(values[j])
            assert actual['history'][t + 3] == {'candidate_id': ids[j], 'y': y}
            raw_y = float(d['raw_shift_std'][j] * float(value['std']))
            mu, sd = float(stats['mu_y'][j]), float(stats['sd_predictive_y'][j])
            shift = float(d['shift_std'][j] * float(value['std']))
            m = ev['measurement']
            assert abs(raw_y - ev['raw_shift_y']) < 1e-12 and abs(shift - ev['deployed_shift_y']) < 1e-12
            assert abs(mu - m['prequery_mu_y']) < 1e-12 and abs(sd - m['prequery_predictive_sd_y']) < 1e-12
            predictions = {'GP': mu, 'regional_unscaled': mu + raw_y, 'deployed': mu + shift}
            if d['bank'] is not None:
                theta, cov = np.asarray(d['prior']['mean']), np.asarray(d['prior']['covariance'])
                feature = np.array([1., raw_y])
                variance = float(feature @ cov @ feature + sd * sd + .0009)
                innovation = float(y - mu - feature @ theta)
                gain = cov @ feature / variance
                updated = theta + gain * innovation
                pc = cov - np.outer(cov @ feature, cov @ feature) / variance
                after = ev['bank_after']
                error = max(float(np.max(np.abs(updated - after['mean']))), float(np.max(np.abs(pc - after['covariance']))))
                assert error < 1e-11 and abs(innovation - m['innovation']) < 1e-12
                assert abs(variance - m['innovation_variance']) < 1e-12
                assert abs(innovation / math.sqrt(variance) - m['innovation_z']) < 1e-12
                assert after['last_round'] == t and after['updates'] == d['bank']['updates'] + 1
                assert np.linalg.eigvalsh(pc).min() >= -1e-12
                max_filter = max(max_filter, error)
                state['banks'][d['bank_key']] = deepcopy(after)
                predictions['KF_joint'] = mu + theta[0] + theta[1] * raw_y
            else:
                assert ev['bank_after'] is None
            for key, pred in predictions.items():
                assert abs(pred - m['predictions'][key]) < 1e-12
                assert abs((y - pred) ** 2 - m['squared_errors'][key]) < 1e-12
            state['decisions'].append(ev)
            state['history'].append(actual['history'][t + 3])
            choices += 1
        assert state['history'] == actual['history'] and state['banks'] == actual['banks']
    assert cache.counters['fits'] == 0
    return {'case': case, 'choices': choices, 'LLM_contexts': calls, 'facts_checked': facts,
        'category_permuted_prefixes': len(fits), 'max_label_error': max_perm,
        'independent_qLogEI_values': score_count, 'max_qLogEI_error': max_score,
        'max_independent_region_error': max_shift, 'max_independent_KF_error': max_filter,
        'roundoff_amplified_score_tables': roundoff, 'new_audit_GP_fits': cache.counters['fits'],
        'operator_compiles_in_loop': sum(s.previous.prior.previous.JIT.values())}


def audit_checkpoint(case):
    """May run as soon as one case completes, overlapping other API workers."""
    protocol = s.check_protocol()
    specs = [v for v in s.read(s.ROOT / 'manifest.json') if v['case'] == case]
    paths = [s.ROOT / 'runs' / v['run_id'] / 'state.json' for v in specs]
    assert all(s.read(p)['status'] == 'COMPLETE' for p in paths)
    states = {p.as_posix(): s.filehash(p) for p in paths}
    path = s.OUT / 'audit_checkpoints' / f'case{case}.json'
    source_hash = s.filehash(__file__)
    if path.exists():
        saved = s.read(path)
        if (saved['source_sha256'] == source_hash and saved['state_sha256'] == states
                and saved['protocol_sha256'] == protocol['protocol_sha256']):
            return saved['audit']
    audit = audit_case(case)
    s.write(path, {'source_sha256': source_hash, 'state_sha256': states,
        'protocol_sha256': protocol['protocol_sha256'], 'audit': audit})
    return audit


def verify():
    protocol = s.check_protocol()
    result = s.read(s.OUT / 'results.json')
    assert result['protocol_sha256'] == protocol['protocol_sha256']
    assert result['completed'] == 48 and result['additional_queries'] == 768
    with ProcessPoolExecutor(max_workers=4, initializer=s.worker_init) as executor:
        cases = list(executor.map(audit_checkpoint, (1, 2, 3, 4)))
    s.write(s.OUT / 'audit_cases.json', cases)
    allrows = controls() + rows()
    assert [raw_metric(v) for v in allrows] == result['raw_metrics']
    for scope, case_set in [('primary', (2, 3, 4)), ('all_four', (1, 2, 3, 4))]:
        for regime, group in result['summary'][scope].items():
            for method, values in group.items():
                selected = [v for v in result['raw_metrics'] if v['case'] in case_set and v['method'] == method
                            and (regime == 'all_conditions' or v['regime'] == regime)]
                assert values == {'n_case_conditions': len(selected),
                    'SearchAUC_mean': statistics.mean(v['SearchAUC'] for v in selected),
                    'final_best_mean': statistics.mean(v['final_best'] for v in selected)}
    by_key = {(v['case'], v['regime'], v['method']): v for v in result['raw_metrics']}
    for delta in result['paired_deltas']:
        method = by_key[delta['case'], delta['regime'], delta['method']]
        reference = by_key[delta['case'], delta['regime'], delta['reference']]
        assert delta['SearchAUC_delta_pp'] == method['SearchAUC'] - reference['SearchAUC']
        assert delta['final_best_delta_pp'] == method['final_best'] - reference['final_best']
    assert result['usage'] == usage()
    for f, expected in protocol['protected_files'].items():
        assert s.filehash(f) == expected
    # Traverse the parent's already pinned inventories as well: G3/G0 are in
    # its own release, whereas LGBO, LLINBO, and pure BO live in older releases.
    parent = s.previous.check_protocol()
    pinned_paths = {str(Path(f).resolve()).casefold() for f in protocol['protected_files']}
    prior_inventories = {}
    for name, entry in parent['protected_inventories'].items():
        assert s.filehash(entry['path']) == entry['sha256']
        inventory = s.read(entry['path'])
        for file in inventory['files']:
            assert s.filehash(file['path']) == file['sha256'], file['path']
            pinned_paths.add(str(Path(file['path']).resolve()).casefold())
        prior_inventories[name] = len(inventory['files'])
    for control in controls():
        assert str(Path(control['source_path']).resolve()).casefold() in pinned_paths
    config = s.api.load_config('../config.txt')
    credential_scan = [p for root in (s.ROOT, s.OUT) for p in root.rglob('*') if p.is_file() and p.suffix in ('.json', '.csv', '.md', '.log', '.xml')]
    for path in credential_scan:
        assert config.api_key not in path.read_text(encoding='utf-8-sig')
    tests = list(ET.parse(s.OUT / 'component_tests.xml').getroot().iter('testsuite'))
    assert all(int(v.attrib.get('failures', 0)) == int(v.attrib.get('errors', 0)) == 0 for v in tests)
    assert sum(v['choices'] for v in cases) == 768
    assert all(v['new_audit_GP_fits'] == v['operator_compiles_in_loop'] == 0 for v in cases)
    receipt = {'integrity_status': 'PASS', 'planned_new_trajectories': 48, 'completed': 48,
        'decisions_verified': 768, 'LLM_contexts_verified': sum(v['LLM_contexts'] for v in cases),
        'facts_checked': sum(v['facts_checked'] for v in cases),
        'category_permuted_prefixes': sum(v['category_permuted_prefixes'] for v in cases),
        'independent_qLogEI_values': sum(v['independent_qLogEI_values'] for v in cases),
        'max_qLogEI_error': max(v['max_qLogEI_error'] for v in cases),
        'max_independent_region_error': max(v['max_independent_region_error'] for v in cases),
        'max_independent_KF_error': max(v['max_independent_KF_error'] for v in cases),
        'protected_files_unchanged': len(protocol['protected_files']), 'credential_leaks': 0,
        'older_release_files_unchanged': prior_inventories,
        'component_tests': sum(int(v.attrib.get('tests', 0)) for v in tests),
        'operator_compiles_in_loop': 0, 'results_sha256': s.filehash(s.OUT / 'results.json'),
        'report_source_sha256': s.filehash(__file__),
        'boundary': 'Implementation integrity and complete matched-budget trajectories; no significance or chemistry-truth claim.'}
    s.write(s.OUT / 'validation_receipt.json', receipt)
    return receipt
