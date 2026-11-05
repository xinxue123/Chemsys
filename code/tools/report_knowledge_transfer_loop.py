"""Metrics and forward-evidence summary for the frozen v5 transfer-scope study."""
from collections import Counter
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code/chemist_system'))
from chem.benchmarks import knowledge_study as study
from chem.benchmarks import knowledge_rule_math as math
from chem.benchmarks.knowledge_llm import digest


def main(variant='v5'):
    if variant not in ('v5', 'v5_seed2_confirm'):
        raise ValueError('Only registered V5 development/confirmation reports are supported')
    confirmation = variant == 'v5_seed2_confirm'
    protocol = study.check(ROOT, variant)
    folder = study.paths(ROOT, variant)
    rows = []
    for path in sorted((folder/'runs').glob('*/state.json')):
        state = study.read(path)
        row = study.summarize(state)
        assert len(state['history']) == 4 + len(state['decisions'])
        assert len({h['candidate_id'] for h in state['history']}) == len(state['history'])
        roles, predictions, facts, changes, mixture_gains = Counter(), 0, {}, [], []
        for t, decision in enumerate(state['decisions'], 1):
            c = study.read(path.parent/'rounds'/f'{t:02d}_commit.json')
            assert digest(c) == decision['commit_sha256']
            assert c['history_sha256'] == digest(state['history'][:t+3])
            assert c['protocol_sha256'] == protocol['protocol_sha256']
            assert c['selected_id'] == state['history'][t+3]['candidate_id']
            assert c['opportunity_gap'] <= c['epsilon'] + 1e-12
            roles[c['mode']] += 1
            prefix = {h['candidate_id']: h for h in state['history'][:t+3]}
            outcome = state['history'][t+3]
            assert len(c['shadows']) == len(decision['evidence'])
            for sh, e in zip(c['shadows'], decision['evidence']):
                pred = sh['prediction']
                assert pred['reference_id'] in prefix and pred['point_id'] not in prefix
                assert pred['reference_y'] == prefix[pred['reference_id']]['y']
                assert abs(math.score_prediction(pred, outcome['y']) - e['delta_log_score']) <= 1e-9
                success = outcome['y'] > pred['reference_y']
                assert e['fact_success'] == success
                if not success or e['delta_log_score'] <= 0:
                    assert e['trusted_weight_after'] <= e['trusted_weight_before'] + 1e-12
                else:
                    assert e['trusted_weight_after'] <= e['trusted_weight_before'] + np.tanh(e['delta_log_score']/2) + 1e-12
                facts[(t, pred['point_id'], pred['reference_id'])] = success
                predictions += 1
            if c['mode'] == 'knowledge_guided':
                assert c['combined_logei'] > c['base_logei']
                assert c['selected_mixture']
                changes.append(t)
                mixture_gains.append(decision['deployed_mixture_log_score_gain'])
        complete = state['status'] == 'COMPLETE'
        assert not complete or len(state['decisions']) == 16
        assert len(state['calls']) <= 4 and state['verification_used'] <= 1
        assert state.get('cache_counters', {}).get('operator_compiles_in_loop', 0) == 0
        row.update(selection_modes=dict(roles), forward_predictions=predictions,
            physical_comparisons=len(facts), successful_comparisons=sum(facts.values()),
            failed_comparisons=len(facts)-sum(facts.values()), weighted_rounds=changes,
            weighted_mixture_logscore_gains=mixture_gains,
            reusable_rule_versions=sum(len(h['ledger']['events']) >= 2 for h in state['hypotheses']),
            comparison_check='PASS_LISTED_TRACE_CHECKS', run=str(path.relative_to(ROOT)))
        rows.append(row)
    baseline = {r['task']:r for r in rows if r['mode'] == 'bo' and r['status'] == 'COMPLETE'}
    arms = {}
    for mode in protocol['development_arms']:
        arm = [r for r in rows if r['mode'] == mode and r['status'] == 'COMPLETE']
        for r in arm:
            b = baseline[r['task']]
            r['AUC_delta'] = r['SearchAUC'] - b['SearchAUC']
            r['best_delta'] = r['final_best'] - b['final_best']
            r['AUC_relative_percent'] = 100 * r['AUC_delta'] / b['SearchAUC']
        wins = sum(r['AUC_delta'] > 1e-9 for r in arm)
        losses = sum(r['AUC_delta'] < -1e-9 for r in arm)
        weighted = sum(r['knowledge_choices'] for r in arm)
        arms[mode] = dict(completed=len(arm), wins=wins, ties=len(arm)-wins-losses, losses=losses,
            final_best_losses=sum(r['best_delta'] < -1e-9 for r in arm),
            weighted_choices=weighted, optional_verifications=sum(r['optional_verifications'] for r in arm),
            majority_development_pass=len(arm)==8 and wins>=5,
            candidate_gate=len(arm)==8 and wins>=5 and weighted>0)
    eligible = [m for m,a in arms.items() if m == 'full' and a['candidate_gate']
                and all(x['completed'] == 8 for x in arms.values())]
    selected = max(eligible, key=lambda m:(arms[m]['wins'],m=='deployment')) if eligible else None
    if confirmation:
        selected = 'full'  # Selected before this initialization, regardless of its result.
    for r in rows:
        if r['mode'] == 'full' and r['status'] == 'COMPLETE':
            controls = [x for x in rows if x['task'] == r['task'] and x['mode'] == 'verification' and x['status'] == 'COMPLETE']
            if controls:
                r['AUC_delta_vs_verification'] = r['SearchAUC'] - controls[0]['SearchAUC']
                r['best_delta_vs_verification'] = r['final_best'] - controls[0]['final_best']
                full_state, control_state = study.read(ROOT/r['run']), study.read(ROOT/controls[0]['run'])
                assert full_state['history'][:4] == control_state['history'][:4]
                paired = list(zip(full_state['decisions'], control_state['decisions']))
                r['first_query_selection_diff_vs_verification'] = next((a['query'] for a,b in paired if a['selected_id'] != b['selected_id']), None)
                r['first_query_mechanism_diff_vs_verification'] = next((a['query'] for a,b in paired if a['mode'] != b['mode']), None)
                r['same_point_different_mechanism_rounds'] = [a['query'] for a,b in paired if a['selected_id'] == b['selected_id'] and a['mode'] != b['mode']]
    audit_path = ROOT/'experiments/reports/knowledge_loop_20261101'/f'{variant}_joint_audit.json'
    audited = False
    if confirmation and audit_path.exists():
        audit = study.read(audit_path)
        audited = (audit['technical_result'] == 'PASS_LISTED_CHECKS' and not audit['failures']
                   and not audit['pending'] and not audit['source_drift']
                   and audit['totals']['runs'] == 16 and audit['totals']['rounds'] == 256
                   and audit['protocol_sha256'] == protocol['protocol_sha256'])
    primary = confirmation and arms['full']['majority_development_pass']
    result = dict(variant=variant, protocol_sha256=protocol['protocol_sha256'], rows=rows, arms=arms,
        preregistered_global_selection=selected, independent_confirmation=confirmation,
        confirmation_primary_pass=primary if confirmation else None,
        independent_confirmation_audited=audited, goal_achieved=bool(primary and audited),
        note=('One preregistered paired initialization of the same8domains. Policy fixed before outcomes; no posthoc selection or independent-dataset/multi-seed guarantee.' if confirmation else
              'Trace checks do not replace independent full-candidate/GP/ledger audit. No per-task policy selection; seed1 is already viewed and cannot confirm this candidate.'))
    output = ROOT/'experiments/reports/knowledge_loop_20261101'/f'{variant}_results.json'
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines = [('v5：冻结方案的预登记seed2配对确认' if confirmation else 'v5：事前可迁移背景范围＋渐进信用恢复（seed0开发结果）'),' ',
        ('候选池、GP与native qLogEI、4＋16预算、模型gpt-6.1-terra medium保持；仅预登记初始化和同分随机流使用seed2，数值GP与菜单仍用seed0。' if confirmation else
         '候选池、初始4点＋16查询、GP与native qLogEI、seed0保持；模型gpt-6.1-terra medium。'),
        '每次知识影响的新plus均有事先冻结的已观测参照和正规化预测。预测增量与实测是否超过参照分账。',
        ('全局v5_full已在本次结果揭示前选定，只与匹配BO比较，不按任务选赢家。' if confirmation else
         'full与verification均最多一次主动验证，后者只禁信用加权采集，保留反馈和刷新。只有full是候选最终策略。'),
        '任务\t方法\tSearchAUC\t最终最佳\t首次最佳总轮次\tAUC较BO\t最佳较BO\t加权改点\t主动验证']
    for task in protocol['tasks']:
        for mode in ('bo','verification','full'):
            for r in rows:
                if r['task']==task and r['mode']==mode and r['status']=='COMPLETE':
                    lines.append(f"{task}\t{mode}\t{r['SearchAUC']:.6f}\t{r['final_best']:.6f}\t{r['first_best_total']}\t{r.get('AUC_delta',0):+.6f}\t{r.get('best_delta',0):+.6f}\t{r['knowledge_choices']}\t{r['optional_verifications']}")
    lines += ['','P0/Doyle产率、Perera UV产品面积%、SnAr STY、HPLC峰面积分别报告，不跨单位平均。']
    for mode,a in arms.items():
        lines.append(f"{mode}：完成{a['completed']}/8，AUC胜/平/负={a['wins']}/{a['ties']}/{a['losses']}，最终最佳下降{a['final_best_losses']}个任务，加权改点{a['weighted_choices']}次，主动验证{a['optional_verifications']}次。")
    lines += [f'事先选定策略：{selected or "尚无合格候选"}。',
        '事实成功、预测比GP更准、实际改变选择、搜索收益是四个不同层次。',
        (f'预登记多数优势性能通过：{primary}；独立全量审计通过：{audited}。这只是相同8任务的一次新初始化确认，不是新数据集或多种子统计保证。' if confirmation else
         '这是开发结果；需独立完整审计及新初始化确认，不将已查看seed1当成独立验证。')]
    if not confirmation:
        lines += ['', 'Full相对verification为整个学习部署政策的轨迹总效应，含后续验证额度与刷新差异，不是单轮均值修正因果效应。']
    for r in rows:
        if 'AUC_delta_vs_verification' in r:
            lines.append(f"{r['task']}：full-verification AUC {r['AUC_delta_vs_verification']:+.6f}；最终最佳 {r['best_delta_vs_verification']:+.6f}")
    output.with_suffix('.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(arms=arms,selected_global_candidate=selected,report=str(output)),ensure_ascii=False))


if __name__ == '__main__':
    main()
