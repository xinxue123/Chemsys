"""Read-only replay and paired multi-initialization analysis for frozen V13."""
from collections import Counter
from pathlib import Path
import json
import sys
import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code/chemist_system'), str(ROOT / 'code/experiment_tools'), str(ROOT / 'code/tools')]
from v13_initialization_seeds import study
from state_split_v13_experiment import transport
from reasoning_memory_v12.llm import validate_insight
from v13_results import replay, metrics, dump_csv
from v13_report_zh import NAMES

OUT = ROOT / 'experiments/reports/v13_seed123_20261102'


def request_audit(folder):
    ids, attempts, repaired = [], Counter(), 0
    for path in (folder / 'requests').glob('*/*/attempt*/response_receipt.json'):
        af = path.parent
        receipt = transport.read(path)
        req = transport.verify_request(af / 'request.json')
        assignment = transport.read(af / 'assignment.json')
        invocation = transport.read(af / 'invocation.json')
        if (receipt['request_sha256'] != req['request_sha256'] or
                receipt['raw_sha256'] != transport.file_sha(af / 'response.raw.json') or
                receipt['assignment_sha256'] != transport.file_sha(af / 'assignment.json') or
                assignment['spawn_request'] != invocation or
                assignment['spawn_result'] != {'task_name': assignment['agent_id']} or
                assignment['agent_id'] != '/root/' + invocation['task_name'] or
                invocation['model'] != 'gpt-6.1-terra' or invocation['reasoning_effort'] != 'medium' or
                invocation['fork_turns'] != 'none'):
            raise ValueError('Actual fresh-agent receipt mismatch')
        export = transport.read(af / 'export.json')
        if (transport.file_sha(ROOT / export['input']) != export['input_sha256'] or
                transport.read(ROOT / export['input']) != {'system': req['system'], 'public': req['public']}):
            raise ValueError('Public packet drift')
        validation = transport.read(af / 'validation.json')
        if req['kind'] == 'choose' and req['attempt'] > 1 and validation['status'] == 'VALIDATED':
            original = req['public']['original']
            locked = req['public']['selected_id_must_remain']
            chosen = validate_insight(ROOT, transport.read(af / 'response.raw.json')['insight'],
                                      [r['candidate_id'] for r in original['candidate_menu']])
            if locked is not None:
                if chosen != locked:
                    raise ValueError('Schema repair changed an already valid point')
                repaired += 1
        ids.append(receipt['agent_id'])
        attempts[validation['status']] += 1
    return ids, attempts, repaired


def stats(values):
    values = np.asarray(values, float)
    return dict(mean=float(values.mean()), sd=float(values.std(ddof=1)),
                minimum=float(values.min()), maximum=float(values.max()))


def main():
    protocol = study.check(ROOT, True)
    metadata = transport.read(study.directory(ROOT) / 'metadata_writer_adapter.json')
    if metadata['helper_sha256'] != transport.file_sha(ROOT / 'code/tools/v13_multiseed_dispatch.py'):
        raise ValueError('Metadata writer compatibility helper changed')
    path_adapter = transport.read(study.directory(ROOT) / 'transport_path_adapter.json')
    if path_adapter['helper_sha256'] != transport.file_sha(ROOT / 'code/tools/v13_multiseed_seal_windows.py'):
        raise ValueError('Windows path-spelling helper changed')
    preflight = transport.read(study.directory(ROOT) / 'preflight.json')
    if preflight['status'] != 'PASS' or len(preflight['rows']) != 24:
        raise ValueError('Incomplete matched numerical preflight')
    delivery_incidents = []
    for incident_path in study.directory(ROOT).glob('transport_delivery_incident_*.json'):
        incident = transport.read(incident_path)
        delivered = ROOT / incident['response_path']
        if transport.read(delivered) != incident['original_payload'] or incident['scientific_content_changed']:
            raise ValueError('Transport recovery altered authored response')
        delivery_incidents.append(dict(path=str(incident_path.relative_to(ROOT)),
            agent_id=incident['agent_id'], incident_sha256=transport.file_sha(incident_path),
            delivered_sha256=transport.file_sha(delivered), original_payload_verified=True))
    execution_incidents = []
    for incident_path in study.directory(ROOT).glob('execution_availability_incident_*.json'):
        incident = transport.read(incident_path)
        request = transport.verify_request(ROOT / incident['request_path'])
        response = ROOT / incident['response_path']
        assignment = transport.read(response.parent / 'assignment.json')
        validation = transport.read(response.parent / 'validation.json')
        if (incident['request_sha256'] != request['request_sha256'] or
                incident['request_attempt'] != request['attempt'] or
                incident['agent_id'] != assignment['agent_id'] or
                incident['response_sha256'] != transport.file_sha(response) or
                validation['status'] != 'VALIDATED' or
                not incident['resumed_same_original_agent'] or
                not incident['output_absent_before_resume'] or
                any(incident[k] for k in ('new_scientific_request', 'response_edited',
                    'model_changed', 'oracle_released_before_validated_response'))):
            raise ValueError('Execution availability recovery custody mismatch')
        execution_incidents.append(dict(path=str(incident_path.relative_to(ROOT)),
            agent_id=incident['agent_id'], incident_sha256=transport.file_sha(incident_path),
            response_custody_verified=True,
            execution_failure_and_resumption='operator_recorded_tool_events'))
    OUT.mkdir(parents=True, exist_ok=True)
    rows, pairs, rounds, missing, summaries, agent_ids = [], [], [], [], [], []
    attempts, note_counts, use_counts, calibration, edits = (Counter() for _ in range(5))
    repairs = 0
    for task, seed in study.JOBS:
        folder = study.base(ROOT, task, seed)
        sf = study.run_path(ROOT, task, seed) / 'state.json'
        state = transport.read(sf) if sf.exists() else {}
        if state.get('status') != 'COMPLETE':
            missing.append(dict(task=study.SHORT[task], seed=seed, queries=len(state.get('decisions', []))))
            continue
        counters = transport.read(sf.parent / 'runtime_counters.json')
        if counters['operator_compiles_in_loop'] != 0 or len(state['review_queries']) > 2:
            raise ValueError('Operator/review budget violation')
        launch = transport.read(folder / 'launch.json')
        if launch['initialization_seed'] != seed or launch['protocol_sha256'] != protocol['protocol_sha256']:
            raise ValueError('Seed launch mismatch')
        with study.redirected(ROOT, task, seed, protocol) as (_, env):
            trajectory = replay(task, state, env)
        rounds += [dict(r, task=study.SHORT[task], seed=seed) for r in trajectory]
        ids, counts, repaired = request_audit(folder)
        agent_ids += ids
        attempts.update(counts)
        repairs += repaired
        note_counts.update(n['recommendation']['permission'] for n in state['notes'])
        for d in state['decisions']:
            use_counts.update(r['use'] for r in d['knowledge_uses'])
            for e in d['feedback']:
                before, after = e['calibration']['before']['reliability'], e['calibration']['after']['reliability']
                calibration['increased' if after > before + 1e-12 else 'decreased' if after < before - 1e-12 else 'unchanged'] += 1
                if e['recommendation_after'] == 'allowed' and e['recommendation_before'] != 'allowed':
                    if not e['improved']:
                        raise ValueError('Failure promoted recommendation permission')
                    if e['recommendation_before'] == 'reviewing':
                        calibration['successful_review_restorations'] += 1
            commit = transport.read(sf.parent / 'rounds' / f"{d['query']:02d}_commit.json")
            if commit['memory']:
                edits.update(r.get('outcome', 'rejected') if r['accepted'] else 'rejected' for r in commit['memory']['plan']['reviews'])
        m = metrics(state)
        short = study.SHORT[task]
        rows.append(dict(task=short, seed=seed, method='V13', backend='fresh_subagent', **m))
        pair = dict(task=short, seed=seed, V13_SearchAUC=m['SearchAUC'], V13_final_best=m['final_best'])
        refs = study.envelope(protocol, task, seed)['controls'][task]
        for arm in ('bo', 'reasoning_bo', 'v12_subagent'):
            reference = transport.read(ROOT / refs[arm]['path'])
            if state['history'][:4] != reference['history'][:4]:
                raise ValueError('Control initialization mismatch')
            baseline = metrics(reference)
            if baseline['SearchAUC'] <= 0:
                raise ValueError('A positive reference is required for relative differences')
            rows.append(dict(task=short, seed=seed, method=arm, backend='none' if arm == 'bo' else 'historical_fresh_subagent', **baseline))
            pair[arm + '_SearchAUC'] = baseline['SearchAUC']
            pair[arm + '_final_best'] = baseline['final_best']
            pair['delta_vs_' + arm] = m['SearchAUC'] - baseline['SearchAUC']
            pair['delta_vs_' + arm + '_pct'] = 100 * (m['SearchAUC'] / baseline['SearchAUC'] - 1)
        pairs.append(pair)
    if len(agent_ids) != len(set(agent_ids)):
        raise ValueError('An agent was reused between experimental requests')
    for task in study.SHORT.values():
        subset = [p for p in pairs if p['task'] == task]
        if len(subset) != 3:
            continue
        for arm in ('bo', 'reasoning_bo', 'v12_subagent'):
            values = np.array([p['delta_vs_' + arm] for p in subset])
            relative = np.array([p['delta_vs_' + arm + '_pct'] for p in subset])
            new = stats([p['V13_SearchAUC'] for p in subset])
            old = stats([p[arm + '_SearchAUC'] for p in subset])
            d, r = stats(values), stats(relative)
            interval = float(student_t.ppf(.975, 2) * d['sd'] / np.sqrt(3))
            summaries.append(dict(task=task, reference=arm, n=3,
                V13_mean=new['mean'], V13_sd=new['sd'], reference_mean=old['mean'], reference_sd=old['sd'],
                paired_delta_mean=d['mean'], paired_delta_sd=d['sd'], delta_pct_mean=r['mean'],
                delta_pct_min=r['minimum'], delta_pct_max=r['maximum'],
                exploratory_paired_t95_low=d['mean'] - interval, exploratory_paired_t95_high=d['mean'] + interval,
                wins=int((values > 1e-8).sum()), ties=int((abs(values) <= 1e-8).sum()), losses=int((values < -1e-8).sum()),
                all_three_positive=bool((values > 1e-8).all())))
    audit = dict(status='PASS_COMPLETE' if not missing else 'PARTIAL_NO_STABILITY_CONCLUSION',
        complete_trajectories=len(pairs), missing=missing, queries_replayed=len(rounds),
        protocol_sha256=protocol['protocol_sha256'], source_and_controls_unchanged=True,
        unique_fresh_agents=len(agent_ids), schema_attempts=dict(attempts), locked_point_repairs_verified=repairs,
        operator_compiles_in_loop=0, oracle_calls_in_audit=0,
        baseline_queries_preflight_audited=sum(r['baseline_queries_audited'] for r in preflight['rows']),
        metadata_writer_compatibility=metadata,
        windows_sealing_path_compatibility=path_adapter,
        transport_only_delivery_recoveries=delivery_incidents,
        agent_execution_availability_recoveries=execution_incidents,
        final_note_permissions=dict(note_counts), knowledge_use_counts=dict(use_counts),
        calibration_changes=dict(calibration), memory_edit_counts=dict(edits),
        failure_based_positive_restorations=0,
        trajectory_counts={k: sum(r[k] for r in rounds) for k in ('forecast_count', 'accurate_failure_events', 'new_suppressions', 'newly_allowed', 'changed_from_GP')},
        analysis_source_sha256=transport.file_sha(__file__))
    dump_csv(OUT / 'metrics.csv', rows)
    dump_csv(OUT / 'paired_results.csv', pairs)
    dump_csv(OUT / 'task_summary.csv', summaries)
    dump_csv(OUT / 'rounds.csv', rounds)
    (OUT / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['冻结V13多初始化验证：seed=1、2、3', '', audit['status'],
        f"已完成{len(pairs)}/24条新V13轨迹，重放{len(rounds)}/384次查询。",
        '保持候选池、原GP、4+16预算与全部V13机制；仅改变4个初始化点。GP数值、并列及菜单随机流保持0。',
        '复用匹配的冻结纯BO、V12子智能体、Reasoning BO子智能体对照。新响应逐次独立生成，不复用LLM缓存。',
        'seed0作为开发结果单列于前一报告，不混入下表。seed1/2/3此前用于V12研究，并非整个研究项目从未见过的测试初始化。',
        'SearchAUC为16个新查询后累计最佳目标的均值，包含初始化最好值，全部查询计数。不同任务原始单位不可直接平均。', '',
        '任务 | V13均值±样本SD | BO均值±样本SD | 平均配对相对变化 | seed1/2/3变化 | 胜/平/负']
    for row in summaries:
        if row['reference'] != 'bo':
            continue
        subset = sorted((p for p in pairs if p['task'] == row['task']), key=lambda p: p['seed'])
        ds = '/'.join(f"{p['delta_vs_bo_pct']:+.2f}%" for p in subset)
        lines.append(f"{NAMES[row['task']]} | {row['V13_mean']:.4f}±{row['V13_sd']:.4f} | "
            f"{row['reference_mean']:.4f}±{row['reference_sd']:.4f} | {row['delta_pct_mean']:+.2f}% | {ds} | "
            f"{row['wins']}/{row['ties']}/{row['losses']}")
    if not missing:
        for arm in ('bo', 'reasoning_bo', 'v12_subagent'):
            selected = [r for r in summaries if r['reference'] == arm]
            lines.append(f"相对{arm}：24组配对{sum(r['wins'] for r in selected)}胜/{sum(r['ties'] for r in selected)}平/"
                f"{sum(r['losses'] for r in selected)}负；8个任务中{sum(r['paired_delta_mean'] > 0 for r in selected)}个均值为正，"
                f"{sum(r['all_three_positive'] for r in selected)}个三个seed均为正。")
        lines += ['', '与两种LLM对照的逐任务比较（配对相对SearchAUC变化均值；胜/平/负）',
            '任务 | 相对Reasoning BO | 胜/平/负 | 相对V12 | 胜/平/负']
        for task in study.SHORT.values():
            refs = {r['reference']: r for r in summaries if r['task'] == task}
            a, b = refs['reasoning_bo'], refs['v12_subagent']
            lines.append(f"{NAMES[task]} | {a['delta_pct_mean']:+.2f}% | "
                f"{a['wins']}/{a['ties']}/{a['losses']} | {b['delta_pct_mean']:+.2f}% | "
                f"{b['wins']}/{b['ties']}/{b['losses']}")
        bo_summary = [r for r in summaries if r['reference'] == 'bo']
        consistent = [NAMES[r['task']] for r in bo_summary if r['all_three_positive']]
        crossing = sum(r['exploratory_paired_t95_low'] <= 0 <= r['exploratory_paired_t95_high']
                       for r in bo_summary)
        lines += ['', '稳定性解读：',
            '三个新seed的SearchAUC均优于BO的任务：' + ('、'.join(consistent) or '无') + '。',
            f'相对BO的探索性配对95%区间覆盖0：{crossing}/8个任务。不能据此宣称统计上已建立稳定优势。',
            'HPLC的三个seed有一胜两负，正的平均相对变化由seed1的大幅收益主导，不能视为跨初始化稳定收益。',
            '本轮V13未建立整体稳定优势，也未显示相对Reasoning BO的稳定增量。原seed0属于开发结果，不能用新seed的Case2全胜覆盖原seed0的损失。',
            'KF降权、知识暂停与事件刷新在日志中实际发生；执行正确和状态分离生效，不等于这些模块已改善搜索效率。']
    lines += ['', '解释边界：每个初始化只有一次新LLM轨迹，因此测量初始化与LLM响应的联合波动，不能分离两类随机性。'
        '只有3个seed；全胜是本样本方向一致，不等于一般稳定性证明。task_summary.csv提供配对差值的探索性t95%区间（df=2，未做多重比较校正）。'
        '24个任务×seed不是24个独立任务。均值、胜率与区间须共同阅读。',
        '机制计数：' + json.dumps(audit['trajectory_counts'], ensure_ascii=False),
        'KF变化：' + json.dumps(audit['calibration_changes'], ensure_ascii=False),
        '知识编辑：' + json.dumps(audit['memory_edit_counts'], ensure_ascii=False),
        f'文件交付恢复：{len(delivery_incidents)}次；原始已生成JSON未修改，记录见audit.json及对应异常文件。',
        f'子智能体执行恢复：{len(execution_incidents)}次；额度错误发生后继续同一原始请求，未更换模型、编辑响应或提前读取实验结果；工具事件记录与响应凭证分别保存。',
        'Windows路径适配：Perera响应凭证临时路径长260字符；仅封存工具在文件写入边界加长路径前缀，保持原子写入与不可覆盖校验，未修改实验进程。',
        'metrics.csv含各方法最终最佳与首次达到轮次；paired_results.csv为逐seed对照；rounds.csv为全部逐轮记录；audit.json为核验结果。']
    (OUT / 'report_zh.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(audit, ensure_ascii=False))
    print('\n'.join(lines[9:]))


if __name__ == '__main__':
    main()
