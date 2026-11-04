"""Compare the fixed v3 policy's development and preregistered paired evidence."""
from hashlib import sha256
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / 'experiments/reports/knowledge_loop_20261101'
TASKS = [f'Reizman_Suzuki_Case_{i}' for i in range(1, 5)] + [
    'Perera_Suzuki', 'Doyle_Buchwald_Hartwig', 'SnAr', 'HPLC']


def load(name, sources):
    path = REPORTS / name
    sources[name] = sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    sources, stages = {}, []
    for variant, seed, audit_name in [
        ('v3', 0, 'v3_joint_independent_audit.json'),
        ('v3_seed1_confirm', 1, 'v3_seed1_confirm_joint_audit.json')]:
        report = load(f'{variant}_results.json', sources)
        audit = load(audit_name, sources)
        assert audit['technical_status'] == 'PASS_LISTED_CHECKS'
        assert not audit['failures'] and not audit['source_drift']
        rows = []
        for task in TASKS:
            pair = {}
            for mode in ('bo', 'full'):
                matches = [r for r in report['rows'] if r['task'] == task and r['mode'] == mode
                           and r['condition'] == 'agent_initial_terra']
                assert len(matches) == 1
                row = matches[0]
                assert row['status'] == 'COMPLETE' and row['queries'] == 16
                checked = next(r for r in audit['rows'] if r['task'] == task and r['mode'] == mode
                               and r['condition'] == 'agent_initial_terra')
                for key in ('SearchAUC', 'final_best', 'first_best_total'):
                    assert abs(row[key] - checked[key]) <= 1e-9, (variant, task, mode, key)
                pair[mode] = {key: row[key] for key in ('SearchAUC', 'final_best', 'first_best_total')}
                if mode == 'full':
                    pair[mode].update(knowledge_choices=row['knowledge_choices'],
                                      optional_verifications=row['optional_verifications'])
            delta = pair['full']['SearchAUC'] - pair['bo']['SearchAUC']
            rows.append(dict(task=task, **pair, AUC_delta=delta,
                AUC_relative_percent=100 * delta / pair['bo']['SearchAUC'],
                best_delta=pair['full']['final_best'] - pair['bo']['final_best'],
                outcome='win' if delta > 1e-9 else 'loss' if delta < -1e-9 else 'tie'))
        counts = {key: sum(r['outcome'] == key for r in rows) for key in ('win', 'tie', 'loss')}
        stages.append(dict(variant=variant, seed=seed, rows=rows, counts=counts,
            majority_performance_pass=counts['win'] >= 5,
            weighted_choices=sum(r['full']['knowledge_choices'] for r in rows),
            optional_verifications=sum(r['full']['optional_verifications'] for r in rows),
            audit=audit_name))
    assessment = dict(selected_global_policy='v3 full', primary_metric='SearchAUC',
        primary_rule='At least5of8 real tasks strictly exceed matched BO; ties are not wins',
        source_sha256=sources, stages=stages,
        user_primary_development_target_met=stages[0]['majority_performance_pass'],
        majority_replicated_in_registered_confirmation=stages[1]['majority_performance_pass'],
        stable_majority_advantage_supported=False, overall_research_goal_status='ACTIVE',
        main_promotion=False,
        mechanism_limit='Development:7 optional verifications and0 weighted choices. Confirmation:5 optional verifications and2 weighted choices; both weighted-selected points lie outside the deployed claim scope and produce no informative update for that claim.',
        next_question='Require that the point selected by knowledge and the future winner justified by a verification remain within an informative, executable claim scope; evaluate this as a new mechanism, not as a passed confirmation on seed1.',
        independence_limit='Same8domains, one registered additional initialization. No significance or universal noninferiority claim; failed seed1 may be diagnosed but cannot later be reused as untouched confirmation.')
    lines = ['多数任务优势：开发比较与预注册配对确认', '',
        '目标按用户修订：固定一套策略，在8个真实任务中至少5个SearchAUC严格优于匹配纯BO；不要求所有case超越。持平不计胜。',
        '统一候选池、每任务相同初始4点和16次查询、相同GP/qLogEI；模型为gpt-6.1-terra medium。',
        'seed1只更换预登记初始点和同分顺序；GP/MC数值配方及LLM菜单随机流保持seed0。',
        'P0/Doyle是产率，Perera是UV产物面积百分比，SnAr是STY，HPLC是峰面积；不跨单位平均原始指标。', '',
        '结论：seed0开发比较5胜1平2负，达到用户的多数优势性能目标；seed1确认1胜3平4负，优势未复现。',
        '保留开发结果，但不能将v3称为稳定优于BO的最终策略。所有对照、失败轨迹和源码快照均保留。']
    for stage in stages:
        c = stage['counts']
        lines += ['', f"{stage['variant']}（seed={stage['seed']}）：{c['win']}胜{c['tie']}平{c['loss']}负",
            '任务\tBO SearchAUC\tv3 SearchAUC\t相对变化%\tBO最终最佳\tv3最终最佳\t首次最佳总轮次 BO/v3']
        for r in stage['rows']:
            b, a = r['bo'], r['full']
            lines.append(f"{r['task']}\t{b['SearchAUC']:.6f}\t{a['SearchAUC']:.6f}\t{r['AUC_relative_percent']:+.3f}\t{b['final_best']:.6f}\t{a['final_best']:.6f}\t{b['first_best_total']}/{a['first_best_total']}")
        lines += [f"实际加权改点{stage['weighted_choices']}次；受控主动验证{stage['optional_verifications']}次。"]
    lines += ['', '机制与执行边界：',
        '当前闭环是局部候选偏好假设、GP条件预测、预提交预测、真实反馈log-score、版本化权重和受控采集混合；不把它称作旧KF机制。',
        '开发集的7次首次非BO分叉来自主动验证，没有加权知识直接改点；19轮进入加权计算不等于19轮知识改变选择。',
        '确认集Doyle q10、Perera q15：知识压低原BO首选，选出scope之外的同分候选；当轮反馈又因out-of-scope而不更新该知识。属于决策与反馈衔接不足，不能简单断言化学命题错误。',
        'SnAr q1验证的当轮产出比同前缀BO更高，但后续轨迹慢一步到高值、最终未找到BO最高点；不能将总损失全部算作首轮知识错误。',
        'Case3 q1并未立即创造更高incumbent，收益主要发生在之后搜索更快上升。轨迹分歧后，逐轮差值不能独立归因为一个模块。',
        'seed0主比较与消融24条/384轮，以及压力12条/192轮均通过独立joint审计。seed1全部16条/256轮、459条预测也通过，source drift=0；技术通过不代表性能通过。',
        '12个人工压力条件仅作稳健性诊断，不并入8任务胜率。确认集未调参或挑选轨迹。', '',
        '下一步研究：检查知识影响的实际选中点能否提供该知识的有效反馈，并检查验证以后是否确有可执行的局部候选；修复闭环缺口后再冻结新候选，不能用已查看的seed1冒充独立确认。',
        '代码里程碑：3793bb3；main未合并。报告源文件SHA见同名JSON。']
    output = REPORTS / 'majority_assessment.json'
    output.write_text(json.dumps(assessment, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    output.with_suffix('.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(dict(report=str(output), development=stages[0]['counts'], confirmation=stages[1]['counts']), ensure_ascii=False))


if __name__ == '__main__':
    main()
