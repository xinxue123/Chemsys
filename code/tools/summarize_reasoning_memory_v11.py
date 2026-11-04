"""Export cost, exact decision explanations and a concise V11 research handoff."""
from collections import Counter
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/chemist_system'));sys.path.insert(0,str(ROOT/'code/experiment_tools'))
from reasoning_memory_v11 import study
from knowledge_information_v10.report import table
from audit_knowledge_rank_v9 import scope


def build():
    p=study.check(ROOT);base=study.directory(ROOT);out=ROOT/'experiments/reports/knowledge_loop_20261101'
    result=study.old.read(out/'v11_results.json');audit=study.old.read(out/'v11_audit.json')
    if audit['status']!='PASS_LISTED_CHECKS':raise ValueError('Complete audit required before handoff')
    traces=[];refs=[];keys=set();models=Counter();usage=Counter();actions=Counter();events=Counter();reasons=Counter()
    for task in p['tasks']:
        run=base/'runs'/f'{task}_rbo_memory';state=study.old.read(run/'state.json');pool=study.gpmod.load_task(ROOT,task,seed=0)
        if state['status']!='COMPLETE':raise ValueError('All8 trajectories required')
        for t,d in enumerate(state['decisions'],1):
            c=study.old.read(run/'rounds'/f'{t:02d}_commit.json');keys.add(c['LLM_context_sha256'])
            if c['memory_context_sha256']:keys.add(c['memory_context_sha256'])
            public=study.old.read(base/'requests'/c['LLM_context_sha256']/'context.json')['public']
            retrieved={(h['id'],h['version']) for group in ('supporting_notes','cautionary_notes') for h in public['retrieved_notes'][group]}
            selected=pool.ix[c['selected_id']];winning=c['LLM_raw']['hypotheses'][c['winning_hypothesis_index']]
            traces.append(dict(task=task,query=t,selected_id=c['selected_id'],original_qLogEI_rank=c['ordered_ids'].index(c['selected_id'])+1,
                objective=d['raw_value'],incumbent_improvement=(d['incumbent_after_y']-d['incumbent_before_y'])*pool.scale,
                official_confidence=winning['confidence'],strategy=winning['strategy'],rationale=winning['rationale'],
                cited_notes=len(c['memory_references']),scored_forecasts=len(c['predictions']),refresh=c['refresh']['allowed'],
                KF_events=';'.join(e['reason'] for e in d['KF_feedback'])))
            for ref in c['memory_references']:
                h=next(h for h in c['hypotheses_at_commit'] if (h['id'],h['version'])==(ref['id'],ref['version']))
                refs.append(dict(task=task,query=t,**ref,in_retrieved_scope=(ref['id'],ref['version']) in retrieved,
                    selected_point_in_scope=scope(pool,h,selected),lifecycle=h['kf']['status'],KF_weight=h['kf']['weight'],
                    local_feedback=h['kf']['local_updates'],claim=h['claim']))
            actions.update(r.get('outcome','rejected') for r in c['reviews'])
            events.update(e['reason'] for e in d['KF_feedback'])
            if c['refresh']['allowed']:reasons.update(c['refresh']['reasons'])
    responses=list((base/'requests').glob('*/attempt*/response.json'))
    for file in responses:
        r=study.old.read(file);models[r.get('resolved_model',r['model'])]+=1
        for name in ('input_tokens','output_tokens','total_tokens'):usage[name]+=r['usage'].get(name,0)
    costs=dict(logical_calls=len(keys),validated_calls=sum((base/'requests'/key/'validated.json').exists() for key in keys),
        successful_HTTP_responses=len(responses),models=dict(models),**dict(usage),
        schema_repair_HTTP_responses=sum(f.parent.name!='attempt1' for f in responses),
        historical_schema_rejection_receipts=len(list((base/'requests').glob('*/attempt*/schema_error.json'))),
        resolved_technical_amendment='technical_amendment_01.json',
        note='Historical rejected citation sidecars are retained. First saved official Insight was reused after the annotation-only fix.')
    study.write(out/'v11_API_usage.json',costs);table(out/'v11_decision_explanations.csv',traces);table(out/'v11_note_citations.csv',refs)
    summary=dict(non_BO_choices=sum(r['original_qLogEI_rank']>1 for r in traces),
        qLogEI_selected_rank_counts=dict(Counter(r['original_qLogEI_rank'] for r in traces)),
        decisions_citing_notes=sum(r['cited_notes']>0 for r in traces),citations=len(refs),
        citations_outside_retrieval=sum(not r['in_retrieved_scope'] for r in refs),
        citations_outside_selected_scope=sum(not r['selected_point_in_scope'] for r in refs),
        active_support_citations=sum(r['use']=='support' and r['lifecycle']=='active' for r in refs),
        counterevidence_citations=sum(r['use']=='counterevidence' for r in refs),
        memory_actions=dict(actions),refresh_reasons=dict(reasons),KF_events=dict(events),
        causal_module_attribution=False,interpretation='Citations describe public rationales; no no-memory ablation, so they do not prove causal benefit.')
    study.write(out/'v11_mechanisms.json',summary)
    index={(r['task'],r['method']):r for r in result['rows']}
    win=result['V11_vs']['Pure BO'];reasoning=result['V11_vs']['Reasoning BO']
    lines=['V11：改用Reasoning BO主线后的实验结果','',
        f"已完成8任务×16查询。SearchAUC相对纯BO：{win['wins']}胜/{win['ties']}平/{win['losses']}负；相对Reasoning BO：{reasoning['wins']}胜/{reasoning['ties']}平/{reasoning['losses']}负。",
        f"多数优势门限（至少5/8严格胜BO）：{result['majority_AUC_gate']}。同一seed0开发比较，未进行新的独立种子确认。",'',
        '结论边界：相对原Reasoning BO，新增记忆组合只在Case4和HPLC提高SearchAUC；Case1、Case2、Perera下降，Case3、Doyle和SnAr持平。不能宣称该组合整体稳定优于Reasoning BO。',
        'Case2相对纯BO的SearchAUC下降30.16%；Perera和Doyle的最终最佳也低于纯BO。保留这些负结果，不逐任务挑选版本。','',
        '实际方案','GP在原候选池上计算原qLogEI并给出5点；LLM使用本轨迹结果、官方Insight历史与Notes，形成3–5个带置信等级的假设。官方稳定置信排序直接选1点。',
        '实验前附带预测固定参照的超越概率；实验后计算相对原GP的Brier增益，KF更新知识可靠性。可靠性、暂停/撤回、旧负证据作为下一轮LLM的记忆。事件触发最多4次Notes修订。',
        'KF不改GP、不改qLogEI、不直接给候选加权，也不否决官方选择。记忆引用并非因果增益证明。','',
        '统一设置与官方差异','相同候选池、原GP/Mixed GP、seed0、4初始化+16查询。新方案真实调用gpt-6.1-terra/medium；其它四组直接读封存结果。',
        '使用官方提示词与原始置信提取方法，固定官方commit4df3cd64abd1be1aab533627578523610f89788c。为与已有对照公平，保留5→1、共享初始化及本地GP后端；未复现论文5→3批量、LLM热启动、外部图检索或后训练。',
        '官方论文：https://arxiv.org/html/2505.12833v2','',
        'SearchAUC（16次查询后累计最佳的平均；保留各任务原单位）',
        '任务\t纯BO\tReasoning BO\tV9\tV10\tV11\tV11较BO/%']
    for task in p['tasks']:
        r=index[task,'V11'];values=[index[task,m]['SearchAUC'] for m in ('Pure BO','Reasoning BO','V9','V10','V11')]
        lines.append(task+'\t'+'\t'.join(f'{v:.4f}' for v in values)+f"\t{r['AUC_relative_percent']:+.3f}")
    lines+=['','最终最佳与首达（总实验序号，含4次初始化）','任务\t纯BO最佳\tRBO最佳\tV11最佳\tV11首达\tV11最佳较BO']
    for task in p['tasks']:
        r=index[task,'V11'];lines.append(f"{task}\t{index[task,'Pure BO']['final_best']:.6f}\t{index[task,'Reasoning BO']['final_best']:.6f}\t{r['final_best']:.6f}\t{r['first_best_total']}\t{r['final_best_delta_vs_BO']:+.6f}")
    lines+=['','机制执行证据',json.dumps(summary,ensure_ascii=False,indent=2),'',
        '这些计数说明模块已运行；没有本版无记忆或无KF完整消融，不能据此断言哪个模块导致收益。',
        '实际出现4次KF自动暂停、0次KF自动撤回、0次自动恢复；6次撤回来自LLM事件动作。4次版本修订的适用负证据继承数量为0，继承逻辑通过回归测试，但本轮没有实际触发的继承案例。',
        '引用位于预测Notes但不属于当前检索范围时已独立标注；范围外引用不等于范围内证据，也不用于KF更新。','',
        '执行问题与修复','初次运行中7个任务的引用侧表校验失败：模型引用了提示内forecast_notes中的知识，但原校验只接受当前retrieved_notes。所有失败均发生在下一次oracle读取前。',
        '技术修订只放宽为可引用提示中明确提供的准确ID/version；官方Insight、系统提示、GP、选点、KF与事件规则未改。修订前32次已完成选择和378份记录逐项核验保留；原始失败响应及多余格式修复成本均保留。',
        '原协议SHA：'+p['protocol_sha256'],'修订SHA：'+audit['technical_amendment']['amendment_sha256'],'',
        '验证与成本','10项流程测试、3项审计回归测试通过；8任务首轮GP数组完全一致；128轮官方排序、事前预测和KF审计通过。',
        '原生GP预热每进程一次并复用缓存，循环中重复编译计数为0；小数据GP使用8路CPU。',json.dumps(costs,ensure_ascii=False,indent=2),
        'V11/V10最多20次逻辑LLM调用，RBO/V9为16次；只保证实验查询预算一致。',
        '审计独立重算选点、CDF、KF、生命周期和继承；提示/动作结构复核复用生产函数，未独立重新拟合GP，也没有外部可信时间戳。','',
        '复现入口：code/experiment_tools/run_reasoning_memory_v11.py；check检查冻结；status查看进度；report重建汇总。',
        '已完成run会读取现有结果；只读审计：code/tools/audit_reasoning_memory_v11.py。',
        '逐轮决策理由：experiments/reports/knowledge_loop_20261101/v11_decision_explanations.csv',
        '引用范围检查：experiments/reports/knowledge_loop_20261101/v11_note_citations.csv']
    (ROOT/'docs/reasoning_memory_v11_results_20261101.txt').write_bytes(('\n'.join(lines)+'\n').encode('utf-8'))
    names=['v11_results.json','v11_comparison.csv','v11_rounds.csv','v11_note_decisions.csv','v11_KF_updates.csv',
        'v11_decision_explanations.csv','v11_note_citations.csv','v11_mechanisms.json','v11_API_usage.json','v11_audit.json',
        'v11_GP_preflight.json','v11_observed_comparison.png','v11_observed_comparison.pdf',
        'v11_incumbent_comparison.png','v11_incumbent_comparison.pdf','v11_run.log','v11_resume01.log']
    delivery_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    receipt=dict(request_status='COMPLETE_IMPLEMENTATION_AND_EIGHT_TASK_COMPARISON',complete_new_runs=8,new_queries=128,
        reused_control_runs=32,new_control_runs=0,seed=0,initial=4,queries_per_task=16,
        protocol_sha256=p['protocol_sha256'],original_core_commit=p['code_milestone'],
        amendment_sha256=audit['technical_amendment']['amendment_sha256'],delivery_commit=delivery_commit,
        majority_AUC_gate=result['majority_AUC_gate'],V11_vs=result['V11_vs'],independent_confirmation=False,
        stable_majority_advantage_supported=False,causal_module_increment_demonstrated=False,
        audit_status=audit['status'],audited_queries=128,audited_forecasts=audit['forecasts'],
        pipeline_tests_passed=10,audit_regression_tests_passed=3,native_GP_initial_checks_passed=8,
        protected_control_files=audit['protected_control_files'],
        figures_reviewed=['v11_observed_comparison.png','v11_incumbent_comparison.png'],
        figure_review='Rendered PNGs inspected for complete8task panels, legible axes/legend, correct series, no clipping. Scientific static figures, independent task scales.',
        API_usage=costs,files_sha256={name:study.old.sha(out/name) for name in names},
        created_utc=datetime.now(timezone.utc).isoformat())
    study.write(out/'v11_completion_receipt.json',receipt)
    status=study.old.read(out/'STATUS.json');already=status.get('v11',{}).get('request_status')==receipt['request_status']
    if not already:
        status['new_optimization_runs']+=8;status['new_experimental_queries']+=128
    status.update(current_phase='V11_COMPLETE_SEED0_MAJORITY_GATE_MET',variant=study.VARIANT,protocol_sha256=p['protocol_sha256'],
        current_sessions={},active_optimization_processes=0,code_milestone=p['code_milestone'],delivery_commit=delivery_commit,
        user_primary_development_target_met=result['majority_AUC_gate'],stable_majority_advantage_supported=False,
        updated_utc=datetime.now(timezone.utc).isoformat(),v11=receipt)
    status['next_steps']=['Requested V11 implementation and all8task comparison completed.',
        'Retain5wins3losses versus BO and2wins3ties3losses versus Reasoning BO; no causal memory-increment claim.',
        'Keep all prior controls and original citation-failure evidence. No further experiments run.']
    study.write(out/'STATUS.json',status)
    print(json.dumps(dict(costs=costs,mechanisms=summary),ensure_ascii=False,indent=2))


if __name__=='__main__':build()
