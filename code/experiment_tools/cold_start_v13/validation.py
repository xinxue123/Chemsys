"""Input-only numerical preflight and read-only committed-trajectory replay."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path
import csv
import json
import os

import numpy as np
from chem.benchmarks import knowledge_gp as native_gp
from knowledge_rank_v9 import engine as native
from reasoning_memory_v12.llm import validate_insight
from . import QUERIES, adapter, control, policy, prompts, study, transport
from .gp import ColdGP


def _preflight_task(root, task):
    root = Path(root).resolve()
    os.chdir(root / 'experiments/chemist_system')
    pool = study.pool_for(root,task)
    folder = study.directory(root) / 'preflight_cache' / task
    gp = ColdGP(pool,folder,seed=0)
    reference = native_gp.GPAdapter(pool,folder,seed=0)
    first = gp.context([],1)
    assert not first['fitted_GP'] and not first['acquisition_meta']['qLogEI_defined']
    assert np.isnan(first['stats']['mu_y']).all() and np.isnan(first['stats']['sd_y']).all()
    assert np.array_equal(first['base_logei'],np.zeros(len(pool.ids)))
    original,display = native.candidates(pool,first,1)
    assert int(original[0])==first['bo_index']
    assert list(original)==list(np.argsort(first['tie_priority'])[:5])
    # Only synthetic values are used. They are never passed to scientific runs.
    indices = np.argsort(first['tie_priority'])[:4]
    history = [dict(candidate_id=pool.ids[int(j)],y=y,raw_value=y*pool.scale)
               for j,y in zip(indices,(.2,.4,.6,.3))]
    rows = []
    for n in (1,2,4):
        prefix = history[:n]
        context = gp.context(prefix,n+1)
        native_context = reference._context([dict(candidate_id=r['candidate_id'],y=r['y']) for r in prefix],n+1)
        np.testing.assert_array_equal(context['base_logei'],native_context['base_logei'])
        if n==1:
            assert context['stats']['std']==1.
            assert np.isfinite(context['stats']['sd_y']).all()
            assert float(context['stats']['sd_y'].max())>1e-6
        else:
            for key in ('mu_y','sd_y','sd_predictive_y'):
                np.testing.assert_array_equal(context['stats'][key],native_context['stats'][key])
        if n==4:
            warm = reference.context(prefix,1)
            for key in ('mu_y','sd_y','sd_predictive_y'):
                np.testing.assert_array_equal(context['stats'][key],warm['stats'][key])
            np.testing.assert_array_equal(context['base_logei'],warm['base_logei'])
        rows.append(dict(synthetic_observations=n,native_acquisition_exact=True,
            native_posterior_exact=n!=1,singleton_unit_convention=n==1))
        gp.release()
        reference.release()
    constant_prefix = [dict(r,y=0.,raw_value=0.) for r in history[:2]]
    constant = gp.context(constant_prefix,3)
    native_constant = reference._context([dict(candidate_id=r['candidate_id'],y=r['y']) for r in constant_prefix],3)
    np.testing.assert_array_equal(constant['base_logei'],native_constant['base_logei'])
    assert constant['stats']['std']==1. and constant['stats']['sd_y'].max()>1e-6
    rows.append(dict(synthetic_observations=2,constant_values=True,native_acquisition_exact=True,
        constant_unit_convention=True))
    gp.release()
    reference.release()
    assert gp.counters['operator_compiles_in_loop']==0
    result = dict(task=task,status='PASS',initial_ids=[],first_menu=[pool.ids[int(j)] for j in display],
        BO_first_id=pool.ids[int(original[0])],numerical_checks=rows,oracle_values_read=0,
        LLM_calls=0,synthetic_cache_separate=True,counters=gp.counters)
    transport.write_once(study.directory(root) / 'preflight' / (study.SHORT[task]+'.json'),result)
    return result


def preflight(root, workers=3):
    study.check(root,True)
    with ProcessPoolExecutor(max_workers=min(workers,3)) as executor:
        rows = list(executor.map(_preflight_task,[str(root)]*len(study.SHORT),study.SHORT))
    result = dict(status='PASS',tasks=len(rows),rows=rows,oracle_values_read=0,LLM_calls=0)
    transport.write_once(study.directory(root) / 'preflight.json',result)
    return dict(status='PASS',tasks=len(rows),synthetic_prefixes=[0,1,2,4],oracle_values_read=0,LLM_calls=0)


def metrics(state):
    history = state['history']
    if len(history)!=QUERIES or len(state['decisions'])!=QUERIES or len({r['candidate_id'] for r in history})!=QUERIES:
        raise ValueError('A complete, unrepeated zero+20 trajectory is required')
    values = np.asarray([r['raw_value'] for r in history],float)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite objective')
    best = np.maximum.accumulate(values)
    return dict(SearchAUC=float(best.mean()),final_best=float(best[-1]),
        first_final_best_query=int(np.flatnonzero(values==best[-1])[0])+1,
        first_query_objective=float(values[0]),queries=QUERIES,initial=0)


def _request_result(root, receipt, expected_system, expected_public):
    request_path = Path(root) / receipt['request_relpath']
    context = transport.read(request_path.parent.parent / 'context.json')
    if context['system']!=expected_system or control.digest(context['public'])!=control.digest(expected_public):
        raise ValueError('Own-prefix public prompt changed')
    if receipt['context_sha256']!=control.digest(context):
        raise ValueError('LLM context digest mismatch')
    raw = transport.read(request_path.parent / 'response.raw.json')
    if raw!=receipt['raw']:
        raise ValueError('Authored response changed')
    return raw


def replay(root, task, arm, state, protocol):
    folder = study.run_path(root,task,arm)
    pool = study.pool_for(root,task)
    initial = transport.read(folder / 'initial_commit.json')
    if initial['initial_ids'] or initial['history'] or initial['initial']!=0:
        raise ValueError('Nonempty warm start in cold study')
    if initial['commit_sha256']!=control.digest({k:v for k,v in initial.items() if k!='commit_sha256'}):
        raise ValueError('Initial commitment changed')
    current = study.initial_state(task,arm,protocol,initial)
    rows = []
    for query,observation in enumerate(state['history'],1):
        commit = transport.read(folder / 'rounds' / f'{query:02d}_commit.json')
        fit = transport.read(folder / 'rounds' / f'{query:02d}_fit.json')
        with np.load(folder / 'rounds' / f'{query:02d}_GP.npz',allow_pickle=False) as z:
            context = dict(stats={k:z[k].copy() for k in ('mu_y','sd_y','sd_predictive_y')},
                **{k:z[k].copy() for k in ('base_logei','available','tie_priority')})
        if native.gp_hash(context)!=commit['GP_sha256']:
            raise ValueError('Committed GP arrays changed')
        np.testing.assert_array_equal(context['available'],~np.isin(pool.ids,[r['candidate_id'] for r in current['history']]))
        np.testing.assert_array_equal(context['tie_priority'],native_gp.semantic_priorities(pool,query,0))
        if query==1:
            assert commit['stage']=='no_data' and not commit['qLogEI_defined']
            assert not fit['fitted_GP'] and not fit['acquisition']['qLogEI_defined']
            for key in ('mu_y','sd_y','sd_predictive_y'):
                assert np.isnan(context['stats'][key]).all()
            assert np.array_equal(context['base_logei'],np.zeros(len(pool.ids)))
        else:
            assert commit['qLogEI_defined'] and fit['fitted_GP']
            assert all(np.isfinite(v).all() for v in context['stats'].values())
            assert np.isfinite(context['base_logei']).all()
        ordered,display = native.candidates(pool,context,query)
        ids = [pool.ids[int(j)] for j in display]
        if commit['ordered_ids']!=[pool.ids[int(j)] for j in ordered] or commit['display_ids']!=ids:
            raise ValueError('Menu mismatch')
        if commit['BO_reference_id']!=pool.ids[int(ordered[0])]:
            raise ValueError('BO reference mismatch')
        if arm=='v13':
            refresh = policy.edit_event(pool,current,ordered,query)
            if commit['refresh']!=refresh or bool(commit['memory'])!=refresh['allowed']:
                raise ValueError('Event or memory-call schedule mismatch')
            memory = commit['memory']
            if memory:
                raw_memory = _request_result(root,memory,prompts.MEMORY_SYSTEM,
                    prompts.memory_public(pool,current,ids,query,refresh))
                edited = policy.apply_edits(raw_memory,pool,current,query)
                if edited!=memory['plan']:
                    raise ValueError('Memory edit replay mismatch')
                notes = edited['notes']
            else:
                notes = current['notes']
            if notes!=commit['edited_notes']:
                raise ValueError('Committed note versions changed')
            raw = _request_result(root,commit['reasoning'],prompts.REASONING_SYSTEM,
                prompts.reasoning_public(root,pool,current,notes,ids,query))
            packet = policy.validate_decision(root,pool,notes,current,context,ids,query,raw)
            if packet!=commit['decision']:
                raise ValueError('Separated controller replay mismatch')
            commit['decision'],commit['edited_notes'] = packet,notes
            commit['reasoning']['raw'] = raw
        elif arm=='reasoning_bo':
            raw = _request_result(root,commit['reasoning'],prompts.RBO_SYSTEM,
                prompts.rbo_public(root,pool,current,ids,query))
            if validate_insight(root,raw,ids)!=commit['selected_id']:
                raise ValueError('Official Reasoning BO selection mismatch')
            commit['reasoning']['raw'] = raw
        elif commit['selected_id']!=pool.ids[int(ordered[0])]:
            raise ValueError('Pure BO did not select its base acquisition/prior argmax')
        current = study.feedback(current,commit,observation)
        if arm=='v13' and (len(current['review_queries'])>2 or len(current['edit_calls'])>4):
            raise ValueError('Memory/review budget exceeded')
        events = current['decisions'][-1]['feedback']
        if any(e['recommendation_after']=='allowed' and e['recommendation_before']!='allowed' and not e['improved'] for e in events):
            raise ValueError('Failure restored positive permission')
        j = pool.ix[observation['candidate_id']]
        rows.append(dict(task=study.SHORT[task],method=arm,query=query,candidate_id=observation['candidate_id'],
            objective=observation['raw_value'],incumbent=max(r['raw_value'] for r in current['history']),
            GP_top1=commit['BO_reference_id'],stage=commit['stage'],
            selected_GP_mean=float(context['stats']['mu_y'][j])*pool.scale if query>1 else None,
            selected_GP_SD=float(context['stats']['sd_y'][j])*pool.scale if query>1 else None,
            selected_qLogEI=float(context['base_logei'][j]) if query>1 else None,
            changed_from_base=observation['candidate_id']!=commit['BO_reference_id'],
            forecast_count=len(commit['decision']['forecasts']) if arm=='v13' else 0,
            memory_refresh=bool(commit['memory']),refresh_reasons=';'.join(commit['refresh']['reasons']),
            knowledge_uses=json.dumps(current['decisions'][-1]['knowledge_uses']),
            accurate_failure_events=sum(e['accurate_failure_forecast'] for e in events),
            new_suppressions=sum(e['recommendation_before']!='suppressed' and e['recommendation_after']=='suppressed' for e in events)))
    if current!=state:
        raise ValueError('Full state replay mismatch')
    return rows


def _dump_csv(path, rows):
    if rows:
        with path.open('w',encoding='utf-8-sig',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def audit(root):
    root = Path(root).resolve()
    protocol = study.check(root,True)
    out = root / 'experiments/reports' / study.VARIANT
    out.mkdir(parents=True,exist_ok=True)
    metrics_rows,rounds,missing = [],[],[]
    for task,arm in study.JOBS:
        path = study.run_path(root,task,arm) / 'state.json'
        state = transport.read(path) if path.exists() else {}
        if state.get('status')!='COMPLETE':
            missing.append(dict(task=study.SHORT[task],arm=arm,queries=len(state.get('history',[]))))
            continue
        rounds.extend(replay(root,task,arm,state,protocol))
        metrics_rows.append(dict(task=study.SHORT[task],method=arm,**metrics(state)))
        if transport.read(path.parent / 'runtime_counters.json')['operator_compiles_in_loop']!=0:
            raise ValueError('Operator compiled inside the loop')
    agent_ids,attempts,locked = [],Counter(),0
    for path in (study.directory(root) / 'requests').glob('*/*/attempt*/response_receipt.json'):
        af = path.parent
        receipt,assignment = transport.read(path),transport.read(af / 'assignment.json')
        request,invocation = transport.verify_request(af / 'request.json'),transport.read(af / 'invocation.json')
        if (receipt['request_sha256']!=request['request_sha256'] or receipt['raw_sha256']!=transport.file_sha(af / 'response.raw.json')
            or receipt['assignment_sha256']!=transport.file_sha(af / 'assignment.json')
            or assignment['spawn_request']!=invocation or assignment['spawn_result']!={'task_name':assignment['agent_id']}
            or assignment['agent_id']!='/root/'+invocation['task_name'] or invocation['fork_turns']!='none'
            or invocation['model']!='gpt-6.1-terra' or invocation['reasoning_effort']!='medium'):
            raise ValueError('Fresh-agent response custody mismatch')
        export = transport.read(af / 'export.json')
        if (transport.file_sha(root / export['input'])!=export['input_sha256'] or
            transport.read(root / export['input'])!={'system':request['system'],'public':request['public']}):
            raise ValueError('Public packet drift')
        validation = transport.read(af / 'validation.json')
        if request['attempt']>1 and request['kind']=='choose' and validation['status']=='VALIDATED':
            selected = request['public']['selected_id_must_remain']
            if selected is not None:
                raw = transport.read(af / 'response.raw.json')
                insight = raw['insight'] if 'insight' in raw else raw
                ids = [r['candidate_id'] for r in request['public']['original']['candidate_menu']]
                if validate_insight(root,insight,ids)!=selected:
                    raise ValueError('Schema repair changed a valid selected point')
                locked += 1
        agent_ids.append(assignment['agent_id'])
        attempts[validation['status']] += 1
    if len(agent_ids)!=len(set(agent_ids)):
        raise ValueError('Agent reused between independent requests')
    comparisons = []
    names = {'bo':'Pure BO','v13':'V13','reasoning_bo':'Reasoning BO'}
    for task in study.SHORT.values():
        items = {r['method']:r for r in metrics_rows if r['task']==task}
        if len(items)!=3:
            continue
        firsts = []
        full = next(k for k,v in study.SHORT.items() if v==task)
        for arm in study.ARMS:
            firsts.append(transport.read(study.run_path(root,full,arm) / 'rounds/01_commit.json'))
        if any(c['ordered_ids']!=firsts[0]['ordered_ids'] or c['display_ids']!=firsts[0]['display_ids'] for c in firsts):
            raise ValueError('First-round candidate menu differs between arms')
        bo = items['bo']['SearchAUC']
        row = dict(task=task,BO_SearchAUC=bo,V13_SearchAUC=items['v13']['SearchAUC'],
            ReasoningBO_SearchAUC=items['reasoning_bo']['SearchAUC'],
            V13_delta_pct=100*(items['v13']['SearchAUC']/bo-1) if bo else None,
            ReasoningBO_delta_pct=100*(items['reasoning_bo']['SearchAUC']/bo-1) if bo else None,
            BO_final_best=items['bo']['final_best'],V13_final_best=items['v13']['final_best'],
            ReasoningBO_final_best=items['reasoning_bo']['final_best'])
        comparisons.append(row)
    result = dict(status='PASS_COMPLETE' if not missing else 'PARTIAL',
        trajectories=len(metrics_rows),queries_replayed=len(rounds),missing=missing,
        protocol_sha256=protocol['protocol_sha256'],historical_evidence_unchanged=True,
        initial_observations=0,paid_query_budget=20,unique_fresh_agents=len(agent_ids),
        schema_attempts=dict(attempts),locked_point_repairs_verified=locked,
        oracle_calls_in_audit=0,operator_compiles_in_loop=0,
        failure_based_positive_restorations=0)
    _dump_csv(out / 'metrics.csv',metrics_rows)
    _dump_csv(out / 'rounds.csv',rounds)
    _dump_csv(out / 'comparison.csv',comparisons)
    (out / 'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines = ['V13零起始点对比：8个任务，seed=0，0+20预算',result['status'],
        f'已完成{len(metrics_rows)}/24条轨迹，核验{len(rounds)}/480次查询。',
        '所有方法从空历史开始。首轮无GP/EI，使用共享随机5点菜单；所有20次实验计入SearchAUC。',
        '单点或全同反馈仅将后验输出尺度设为1，GP模型与采集值仍沿用原算法；其余前缀数值不变。V13首次知识编辑在q2，后期额度保留至q15。',
        '历史4+16结果单独封存，不混入本表。单seed与单次独立LLM响应不能证明稳定性。','',
        '任务 | BO SearchAUC | V13 SearchAUC | Reasoning BO SearchAUC | V13相对BO | Reasoning BO相对BO']
    for r in comparisons:
        a = f"{r['V13_delta_pct']:+.2f}%" if r['V13_delta_pct'] is not None else 'NA'
        b = f"{r['ReasoningBO_delta_pct']:+.2f}%" if r['ReasoningBO_delta_pct'] is not None else 'NA'
        lines.append(f"{r['task']} | {r['BO_SearchAUC']:.4f} | {r['V13_SearchAUC']:.4f} | {r['ReasoningBO_SearchAUC']:.4f} | {a} | {b}")
    lines += ['', '最终最佳值与首次到达轮次', '任务 | 方法 | 最终最佳 | 首次到达查询']
    lines += [f"{r['task']} | {names[r['method']]} | {r['final_best']:.5f} | {r['first_final_best_query']}" for r in metrics_rows]
    if not missing:
        for arm in ('v13','reasoning_bo'):
            diffs = [r['SearchAUC']-next(b['SearchAUC'] for b in metrics_rows if b['task']==r['task'] and b['method']=='bo') for r in metrics_rows if r['method']==arm]
            lines.append(f"{names[arm]}相对BO：{sum(x>1e-8 for x in diffs)}胜/{sum(abs(x)<=1e-8 for x in diffs)}平/{sum(x < -1e-8 for x in diffs)}负。不同任务单位不同，不汇总原始目标均值。")
    lines += ['', '指标为全部20次查询后的累计最佳值之均值，无免费初始化点；零值与失败查询均计数。',
        'raw response、实验前承诺和GP数组位于独立cold0命名空间。metrics.csv含首次最佳轮次，rounds.csv含逐轮选点/预测/知识事件。']
    (out / 'report_zh.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result
