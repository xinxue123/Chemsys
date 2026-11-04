"""Replay frozen loops, then aggregate paired starts without mixing task units."""
from collections import defaultdict
from pathlib import Path
import csv
import json

import numpy as np
from . import study


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def metric_rows(path, seed):
    result = []
    for row in read_csv(path):
        result.append(dict(task=row['task'],seed=seed,method=row['method'],
            **{k:float(row[k]) for k in ('SearchAUC','final_best','first_query_objective')},
            **{k:int(row[k]) for k in ('first_final_best_query','queries','initial')}))
    return result


def aggregate(rows):
    index = {(r['task'],r['seed'],r['method']):r for r in rows}
    if len(index) != len(rows):
        raise ValueError('Duplicate task/seed/method metrics')
    paired, summary = [], []
    for task in study.SHORT.values():
        for seed in (0,*study.SEEDS):
            arms = [index.get((task,seed,a)) for a in study.ARMS]
            if any(r is None for r in arms):
                continue
            bo, v13, rbo = arms
            paired.append(dict(task=task,seed=seed,BO_SearchAUC=bo['SearchAUC'],
                V13_SearchAUC=v13['SearchAUC'],ReasoningBO_SearchAUC=rbo['SearchAUC'],
                V13_minus_BO=v13['SearchAUC']-bo['SearchAUC'],
                ReasoningBO_minus_BO=rbo['SearchAUC']-bo['SearchAUC'],
                V13_minus_ReasoningBO=v13['SearchAUC']-rbo['SearchAUC'],
                V13_delta_pct=100*(v13['SearchAUC']/bo['SearchAUC']-1) if bo['SearchAUC'] else None,
                ReasoningBO_delta_pct=100*(rbo['SearchAUC']/bo['SearchAUC']-1) if bo['SearchAUC'] else None))
        for arm in study.ARMS:
            values = [index[(task,s,arm)] for s in (0,*study.SEEDS) if (task,s,arm) in index]
            if not values:
                continue
            row = dict(task=task,method=arm,n_seeds=len(values),seeds=','.join(str(r['seed']) for r in values))
            for key in ('SearchAUC','final_best','first_final_best_query'):
                data = np.array([r[key] for r in values],float)
                row[key+'_mean'] = float(data.mean())
                row[key+'_sample_sd'] = float(data.std(ddof=1)) if len(data)>1 else None
            differences = [r['SearchAUC']-index[(task,r['seed'],'bo')]['SearchAUC'] for r in values
                           if (task,r['seed'],'bo') in index]
            row.update(n_paired=len(differences),paired_delta_mean=float(np.mean(differences)) if differences else None,
                paired_delta_sample_sd=float(np.std(differences,ddof=1)) if len(differences)>1 else None,
                paired_delta_min=min(differences) if differences else None,
                paired_delta_max=max(differences) if differences else None,
                wins_vs_BO=sum(x>1e-8 for x in differences),ties_vs_BO=sum(abs(x)<=1e-8 for x in differences),
                losses_vs_BO=sum(x < -1e-8 for x in differences))
            summary.append(row)
    return paired,summary


def audit(root):
    root = Path(root).resolve()
    protocol = study.check(root,True)
    out = root / 'experiments/reports' / study.VARIANT
    out.mkdir(parents=True,exist_ok=True)
    metrics,rounds,checks,clarifiers,agents = [],[],[],[],[]
    for seed in study.SEEDS:
        with study.redirected(root,seed,protocol):
            check = study.validation.audit(root)
        seed_out = out / f's{seed}'
        check['starting_seed'] = seed
        checks.append(check)
        if (seed_out / 'metrics.csv').exists():
            metrics.extend(metric_rows(seed_out / 'metrics.csv',seed))
        if (seed_out / 'rounds.csv').exists():
            rounds.extend(dict(r,seed=seed) for r in read_csv(seed_out / 'rounds.csv'))
        for task in study.SHORT:
            for arm in study.ARMS:
                path = study.run_path(root,task,seed,arm) / 'state.json'
                if path.exists():
                    state = study.transport.read(path)
                    if (state['seed'],state['starting_seed'],state['numerical_seed']) != (seed,seed,0):
                        raise ValueError('Starting versus numerical seed metadata drift')
        for path in (study.base(root,seed) / 'requests').glob('*/*/attempt*/response_receipt.json'):
            agents.append(study.transport.read(path)['agent_id'])
        for path in (study.base(root,seed) / 'requests').glob('*/*/attempt*/format_clarification.json'):
            note = study.transport.read(path)
            req = study.transport.verify_request(path.parent / 'request.json')
            receipt_path = path.parent / 'response_receipt.json'
            if not receipt_path.exists():
                continue
            receipt = study.transport.read(receipt_path)
            if (note['message']!=study.SCHEMA_MESSAGE or note['request_sha256']!=req['request_sha256']
                or note['agent_id']!=receipt['agent_id'] or note['response_bytes_edited_by_parent']):
                raise ValueError('Schema-only clarification custody mismatch')
            clarifiers.append(dict(seed=seed,path=path.relative_to(root).as_posix(),
                sha256=study.transport.file_sha(path),agent_id=note['agent_id']))
        report = seed_out / 'report_zh.txt'
        text = report.read_text(encoding='utf-8')
        text = text.replace('seed=0，0+20预算',f'起始seed={seed}，数值seed=0，0+20预算',1)
        text += '\n仅首轮先验随机流随起始seed改变；后续数值与排序规则冻结。LLM为独立子智能体响应。\n'
        report.write_text(text,encoding='utf-8')
    if len(agents)!=len(set(agents)):
        raise ValueError('Agent reused across starting seeds')
    prior_agents = {study.transport.read(p)['agent_id'] for p in
                    (study.NATIVE_DIRECTORY(root) / 'requests').glob('*/*/attempt*/response_receipt.json')}
    if set(agents) & prior_agents:
        raise ValueError('Seed0 response agent reused')
    prior = metric_rows(root / 'experiments/reports/cold0_20261102/metrics.csv',0)
    all_metrics = prior+metrics
    paired,summary = aggregate(all_metrics)
    study.validation._dump_csv(out / 'metrics_new.csv',metrics)
    study.validation._dump_csv(out / 'metrics_all.csv',all_metrics)
    study.validation._dump_csv(out / 'paired.csv',paired)
    study.validation._dump_csv(out / 'summary.csv',summary)
    study.validation._dump_csv(out / 'rounds_new.csv',rounds)
    complete = all(c['status']=='PASS_COMPLETE' for c in checks)
    result = dict(status='PASS_COMPLETE' if complete else 'PARTIAL',new_trajectories=len(metrics),
        new_queries_replayed=len(rounds),total_trajectories_with_seed0=len(all_metrics),
        total_queries_with_seed0=len(rounds)+480,protocol_sha256=protocol['protocol_sha256'],
        seed_audits=checks,protected_seed0_files=len(protocol['protected_seed0_sha256']),
        historical_seed0_and_warmstart_unchanged=True,unique_fresh_agents=len(agents),
        first_round_schema_clarifications=clarifiers,initial=0,queries_per_trajectory=20,
        starting_seeds=[0,*study.SEEDS],numerical_seed=0,post_first_tie_seed=0,display_seed=0,
        oracle_calls_in_audit=0,operator_compiles_in_loop=0)
    if complete and (len(metrics),len(rounds),len(paired),len(clarifiers)) != (72,1440,32,24):
        raise ValueError('Complete multi-start accounting mismatch')
    study.check(root,True)
    (out / 'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines = ['V13零起始点多种子对照',result['status'],
        f'新增{len(metrics)}/72条轨迹，重放{len(rounds)}/1440次查询；包含封存seed0共{len(all_metrics)}/96条轨迹。',
        '设置：0条免费观测，每条20次查询；首轮共享菜单随起始seed变化，GP/qLogEI及后续排序固定数值seed0。',
        'LLM：每次请求独立gpt-6.1-terra medium子智能体，无跨seed/方法响应缓存。',
        '原冷启动seed0与更早4+16结果保持原样；本表不混入4+16轨迹。',
        'SearchAUC为全部20次查询后累计最佳的均值；标准差为样本标准差（ddof=1）。',
        '四个起始seed各一次LLM轨迹，联合反映开局与响应波动，不能单独分解二者方差。','',
        '任务 | 种子数 | BO SearchAUC均值±SD | V13均值±SD | Reasoning BO均值±SD | V13对BO胜/平/负 | V13最差配对差']
    for task in study.SHORT.values():
        groups = {r['method']:r for r in summary if r['task']==task}
        if len(groups)!=3:
            continue
        def fmt(arm):
            r = groups[arm]
            return f"{r['SearchAUC_mean']:.4f}±{r['SearchAUC_sample_sd']:.4f}" if r['SearchAUC_sample_sd'] is not None else f"{r['SearchAUC_mean']:.4f} (n=1)"
        v = groups['v13']
        lines.append(f"{task} | {v['n_seeds']} | {fmt('bo')} | {fmt('v13')} | {fmt('reasoning_bo')} | {v['wins_vs_BO']}/{v['ties_vs_BO']}/{v['losses_vs_BO']} | {v['paired_delta_min']:+.4f}")
    lines += ['', '逐种子SearchAUC配对差', '任务 | seed | V13-BO | Reasoning BO-BO | V13-Reasoning BO']
    lines += [f"{r['task']} | {r['seed']} | {r['V13_minus_BO']:+.4f} | {r['ReasoningBO_minus_BO']:+.4f} | {r['V13_minus_ReasoningBO']:+.4f}" for r in paired]
    lines += ['', 'metrics_all.csv：全部方法最终最佳与首次到达次数；paired.csv：逐种子差；summary.csv：均值、SD及胜负；rounds_new.csv：逐轮记录。',
        '不跨不同任务单位平均原始目标。seed0非新重复，独立列明；格式失败与修复原文均保留。']
    (out / 'report_zh.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return {k:result[k] for k in ('status','new_trajectories','new_queries_replayed','total_trajectories_with_seed0','total_queries_with_seed0','historical_seed0_and_warmstart_unchanged','unique_fresh_agents')}
