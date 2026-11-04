"""V7 proposal-interface/trajectory diagnosis after all eight full runs finish.

Uses only candidate inputs, own queried outcomes, durable commits and saved GP
arrays. No oracle/API/fitting. This is not a replacement for the joint audit.
V7 changes prompt side information; observed differences are not pure gate effects.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
from math import log
from pathlib import Path
import json
import numpy as np
import review_knowledge_anchor_mechanism as prior

ROOT,REPORTS,a=prior.ROOT,prior.REPORTS,prior.audit


def rate(numerator,denominator):
    return numerator/denominator if denominator else None


def describe_example(value):
    """Reuse the numerical V5 predictor review without implying V6's gate."""
    if isinstance(value,list):return [describe_example(v) for v in value]
    if not isinstance(value,dict):return value
    out={}
    for key,item in value.items():
        if key=='all_V6_gates_pass':continue
        if key=='V6_proposal_membership':key='is_accepted_live_example_descriptive_only'
        out[key]=describe_example(item)
    return out


def summarize_calls(calls):
    accepted=[p for c in calls for p in c['accepted_examples']]
    opened=[p for c in calls if c['optional_window_open'] for p in c['accepted_examples']]
    inside=sum(p['passes_narrow_cost_gate'] for p in opened)
    return dict(LLM_calls=len(calls),open_window_calls=sum(c['optional_window_open'] for c in calls),
        accepted_add_revise=len(accepted),rejected_actions=sum(len(c['rejected_actions']) for c in calls),
        accepted_origins_not_executable=sum(not p['currently_executable'] for p in accepted),
        all_accepted_narrow_inside=sum(p['passes_narrow_cost_gate'] for p in accepted),
        open_window_accepted=len(opened),open_window_narrow_inside=inside,
        open_window_narrow_outside=len(opened)-inside,open_window_narrow_inside_rate=rate(inside,len(opened)),
        open_window_narrow_outside_rate=rate(len(opened)-inside,len(opened)),
        displayed_reference_used=sum(p['proposed_reference_was_displayed'] for p in accepted),
        accepted_reference_hint_truncated=sum(p['reference_hints_truncated'] for p in accepted))


def call_review(folder,pool,run,state,commit):
    t=commit['query'];call=next(c for c in state['calls'] if c['query']==t)
    path=folder/'requests'/call['context_sha256']/'context.json'
    prior.INPUT_HASHES[path.relative_to(ROOT).as_posix()]=a.filehash(path)
    context=a.read(path);assert a.digest(context)==call['context_sha256']
    public=context['public'];support=public['proposal_support'];arrays=prior.arrays_for(run,t)
    assert support['schema']=='known_reference_cost_flags_v1'
    assert [r['point_id'] for r in support['candidate_support']]==commit['menu_ids']
    prefix=state['history'][:t+3]
    assert [r['id'] for r in public['own_observations']]==[r['candidate_id'] for r in prefix]
    bo=pool['ix'][commit['BO_reference_id']]
    window=t<=10 and sum(d['mode']=='optional_verification' for d in state['decisions'][:t-1])<1
    assert support['optional_window_open']==window
    by_id={r['point_id']:r for r in support['candidate_support']}
    for point,r in by_id.items():
        j=pool['ix'][point]
        expected=bool(arrays['eligible'][j] and arrays['base_logei'][bo]-arrays['base_logei'][j]<=min(commit['epsilon'],log(1.1)))
        assert expected==r['passes_narrow_cost_gate']
    _,calc=a.calculations(pool,commit['hypotheses_at_commit'],prefix,arrays)
    examples=[]
    for review in commit['reviews']:
        if not review.get('accepted') or review.get('op') not in ('add','revise'):continue
        action=commit['proposal']['actions'][review['action_index']]
        key=(review['id'],review['version']);candidate=by_id[action['plus_id']]
        references={r['reference_id']:r for r in candidate['references']}
        ref=references.get(action['minus_id']);j=pool['ix'][action['plus_id']]
        if ref is not None:
            required=set(ref['required_focus_axes']);possible=required|set(ref['additional_changed_axes']);focus=set(action['focus'])
            assert 1<=len(focus)<=2 and required<=focus<=possible
        examples.append(dict(action_index=review['action_index'],op=action['op'],outcome=review['outcome'],
            rule=list(key),point_id=action['plus_id'],proposed_reference_id=action['minus_id'],
            proposed_reference_was_displayed=ref is not None,
            reference_hints_truncated=candidate['references_truncated'],
            compatible_reference_count=candidate['compatible_reference_count'],
            passes_narrow_cost_gate=candidate['passes_narrow_cost_gate'],
            gap=float(arrays['base_logei'][bo]-arrays['base_logei'][j]),
            currently_executable=key in calc and j in calc[key]['positions'],
            is_BO_example=j==bo,action_sha256=a.digest(action)))
    return dict(task=state['task'],query=t,context_sha256=call['context_sha256'],commit_sha256=a.digest(commit),
        optional_window_open=window,menu_count=len(by_id),
        menu_narrow_inside=sum(r['passes_narrow_cost_gate'] for r in by_id.values()),
        menu_without_geometric_references=sum(r['compatible_reference_count']==0 for r in by_id.values()),
        accepted_examples=examples,rejected_actions=[r for r in commit['reviews'] if r.get('accepted') is False])


def first_divergence(states,commits,dirs):
    first_proposal=next((t for t in range(1,17) if a.digest(commits['v5'][t-1]['proposal'])!=a.digest(commits['v7'][t-1]['proposal'])),None)
    for t,(old,new) in enumerate(zip(commits['v5'],commits['v7']),1):
        if old['selected_id']==new['selected_id']:continue
        same=a.digest(states['v5']['history'][:t+3])==a.digest(states['v7']['history'][:t+3])
        gp=prior.compare_gp(prior.arrays_for(dirs['v5'],t),prior.arrays_for(dirs['v7'],t))
        return dict(query=t,same_observed_prefix=same,GP_comparison=gp,
            v5_selected=old['selected_id'],v7_selected=new['selected_id'],v5_mode=old['mode'],v7_mode=new['mode'],
            v5_actual_queried_value=states['v5']['decisions'][t-1]['raw_value'],
            v7_actual_queried_value=states['v7']['decisions'][t-1]['raw_value'],
            same_menu=old['menu_ids']==new['menu_ids'],same_current_proposal=a.digest(old['proposal'])==a.digest(new['proposal']),
            same_hypotheses=a.digest(old['hypotheses_at_commit'])==a.digest(new['hypotheses_at_commit']),
            first_proposal_difference_query=first_proposal,
            attribution='V7 has a different system prompt and public support. Even at the same observed prefix/GP, this is an interface-induced proposal/decision difference, not a pure gate effect.')
    return None


def credit_round(pool, state, commit, run):
    """All-round prospective credit and actual weighted ranking, no refit."""
    t=commit['query'];arrays=prior.arrays_for(run,t)
    rows,calc=a.calculations(pool,commit['hypotheses_at_commit'],state['history'][:t+3],arrays)
    a.exact(rows,commit['coverage'],'diagnostic independent V7 coverage')
    base=arrays['base_logei'];combined=base.copy();deployed=commit['deployed_versions']
    if deployed:
        weights=[r['effective_mixture_weight'] for r in deployed]
        offsets=np.stack([calc[a.hkey(r)]['offsets'] for r in deployed])
        correction=a.logsumexp(np.vstack([np.full_like(base,log(1-sum(weights)))]+
            [log(w)+o for w,o in zip(weights,offsets)]),axis=0)
        correction[np.all(offsets<=a.ATOL,axis=0)]=0.
        combined+=correction
    winner=a.select(combined,arrays['eligible'],arrays['tie_priority'])
    bo=pool['ix'][commit['BO_reference_id']]
    assert pool['ids'][a.select(base,arrays['eligible'],arrays['tie_priority'])]==commit['BO_reference_id']
    assert winner==bo, 'V7 diagnostic found weighted ranking changed; revise zero-guided conclusion'
    details=[]
    for row in rows:
        h=next(h for h in commit['hypotheses_at_commit'] if a.hkey(h)==a.hkey(row))
        if h['ledger']['trusted_weight']<=0:continue
        details.append(dict(**row,status=h['status'],latent_trusted_weight=h['ledger']['trusted_weight']))
    feedback=state['decisions'][t-1]['evidence'];bykey={(x['hypothesis_id'],x['version']):x for x in feedback}
    errors=[]
    for sh in commit['shadows']:
        pred=sh['prediction'];event=bykey[a.hkey(sh)]
        assert pred['prediction_sha256']==a.digest({k:v for k,v in pred.items() if k!='prediction_sha256'})
        independently=dict(pred)
        independently['event_log_probability']=float(a.log_ndtr((pred['gp_mean_y']-pred['reference_y'])/
            np.sqrt(pred['gp_var_y']+pred['preference_noise_sd']**2)))
        expected=a.score(independently,state['history'][t+3]['y'])
        errors.append(abs(expected-event['delta_log_score']))
        a.close(expected,event['delta_log_score'],'independent frozen-prediction logscore',atol=1e-10,rtol=1e-10)
    return dict(task=state['task'],query=t,mode=commit['mode'],weighted_winner=pool['ids'][winner],
        BO_reference_id=commit['BO_reference_id'],deployed_versions=deployed,
        positive_credit_rules=details,selected_mixture=commit['selected_mixture'],
        prospective_shadows=commit['shadows'],feedback=feedback,
        maximum_independent_feedback_logscore_error=max(errors,default=0.))


def self_test():
    assert summarize_calls([])['open_window_narrow_inside_rate'] is None
    calls=[dict(optional_window_open=True,rejected_actions=[],accepted_examples=[
        dict(passes_narrow_cost_gate=True,currently_executable=True,proposed_reference_was_displayed=True,reference_hints_truncated=False),
        dict(passes_narrow_cost_gate=False,currently_executable=True,proposed_reference_was_displayed=False,reference_hints_truncated=True)]),
        dict(optional_window_open=False,rejected_actions=[],accepted_examples=[
        dict(passes_narrow_cost_gate=True,currently_executable=True,proposed_reference_was_displayed=True,reference_hints_truncated=False)])]
    s=summarize_calls(calls)
    assert (s['accepted_add_revise'],s['open_window_accepted'],s['open_window_narrow_inside_rate'])==(3,2,.5)
    clean=describe_example(dict(V6_proposal_membership=True,all_V6_gates_pass=True))
    assert clean=={'is_accepted_live_example_descriptive_only':True}
    print('PASS: synthetic denominator/window and descriptive-membership checks; no experimental data opened.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test:self_test();return
    folders={v:ROOT/'experiments/runs/knowledge_loop_20261101'/v for v in ('v5','v7')}
    protocols={v:prior.read(folder/'protocol.json') for v,folder in folders.items()}
    assert protocols['v5']['tasks']==protocols['v7']['tasks'] and len(protocols['v7']['tasks'])==8
    assert protocols['v5']['prompt_sha256']!=protocols['v7']['prompt_sha256']
    tasks=protocols['v7']['tasks'];states={};dirs={}
    # Full-result diagnosis is blocked until all arms/tasks are COMPLETE.
    for variant,folder in folders.items():
        p=protocols[variant];assert p['protocol_sha256']==a.digest({k:v for k,v in p.items() if k!='protocol_sha256'})
        states[variant]={};dirs[variant]={}
        for task in tasks:
            for mode in ('bo','full'):
                run=folder/'runs'/f'{task}_{mode}_agent_initial_terra';state=prior.read(run/'state.json')
                if state['status']!='COMPLETE' or len(state['decisions'])!=16 or len(state['history'])!=20:
                    raise RuntimeError(f'Await all completed full/BO trajectories: {variant}/{task}/{mode}')
                states[variant][task,mode]=state;dirs[variant][task,mode]=run
    baseline_path=REPORTS/'v7_existing_proposal_counts.json';baseline=a.read(baseline_path)
    prior.INPUT_HASHES[baseline_path.relative_to(ROOT).as_posix()]=a.filehash(baseline_path)
    old=baseline['totals'];seed0=next(r for r in baseline['rows'] if r['variant']=='v5')
    baseline_rates=dict(pooled_seed0_seed2=dict(open_accepted=46,narrow_outside=21,narrow_inside=25,inside_rate=25/46,outside_rate=21/46),
        matched_seed0=dict(open_accepted=20,narrow_outside=9,narrow_inside=11,inside_rate=11/20,outside_rate=9/20),
        note='Pooled21/46 outside mixes two initializations. Matched seed0 is9/20 outside. Window/quota availability itself can change after earlier decisions; rates are descriptive, not causal estimates.')
    assert old['window_and_quota_open_accepted_proposals']==46 and old['window_and_quota_open_above_optional_budget']==21
    assert seed0['window_and_quota_open_accepted_proposals']==20 and seed0['window_and_quota_open_above_optional_budget']==9
    pools={m['task']:a.common.public_pool(ROOT,m) for m in protocols['v7']['inputs']}
    comparisons,events,calls,credit_rounds=[],[],[],[]
    for task in tasks:
        pool=pools[task];state={v:states[v][task,'full'] for v in folders};commits={}
        assert a.digest(state['v5']['history'][:4])==a.digest(state['v7']['history'][:4])
        assert a.digest(states['v5'][task,'bo']['history'])==a.digest(states['v7'][task,'bo']['history'])
        for variant in folders:
            run=dirs[variant][task,'full'];commits[variant]=[];records=[]
            for t in range(1,17):
                c=prior.read(run/'rounds'/f'{t:02d}_commit.json')
                assert c['history_sha256']==a.digest(state[variant]['history'][:t+3])
                assert a.digest(c)==state[variant]['decisions'][t-1]['commit_sha256']
                records=prior.advance_proposals(records,c);commits[variant].append(c)
                if c['mode'] in ('optional_verification','knowledge_guided') or c['selected_mixture']:
                    events.append(describe_example(prior.special_event(variant,pool,run,state[variant],c,records)))
                if variant=='v7' and any(x['query']==t for x in state[variant]['calls']):
                    calls.append(call_review(folders[variant],pool,run,state[variant],c))
                if variant=='v7':credit_rounds.append(credit_round(pool,state[variant],c,run))
        metrics={v:prior.measures(state[v]) for v in folders};metrics['BO']=prior.measures(states['v7'][task,'bo'])
        first=first_divergence(state,commits,{v:dirs[v][task,'full'] for v in folders})
        curves={v:np.maximum.accumulate([r['raw_value'] for r in state[v]['history']])[4:] for v in folders}
        trajectory=[]
        for t in range(1,17):
            trajectory.append(dict(query=t,v5_selected=commits['v5'][t-1]['selected_id'],v7_selected=commits['v7'][t-1]['selected_id'],
                same_prefix=commits['v5'][t-1]['history_sha256']==commits['v7'][t-1]['history_sha256'],
                v5_actual_value=state['v5']['decisions'][t-1]['raw_value'],v7_actual_value=state['v7']['decisions'][t-1]['raw_value'],
                v5_best=float(curves['v5'][t-1]),v7_best=float(curves['v7'][t-1]),best_delta_v7_minus_v5=float(curves['v7'][t-1]-curves['v5'][t-1])))
        comparisons.append(dict(task=task,unit=pool['unit'],metrics=metrics,
            AUC_delta_v7_minus_v5=metrics['v7']['SearchAUC']-metrics['v5']['SearchAUC'],
            AUC_delta_v7_minus_BO=metrics['v7']['SearchAUC']-metrics['BO']['SearchAUC'],
            final_delta_v7_minus_v5=metrics['v7']['final_best']-metrics['v5']['final_best'],
            final_delta_v7_minus_BO=metrics['v7']['final_best']-metrics['BO']['final_best'],
            first_selected_point_divergence=first,proposal_support_counts=summarize_calls([x for x in calls if x['task']==task]),trajectory=trajectory))
    proposal_summary=summarize_calls(calls)
    feedback=[e for r in credit_rounds for e in r['feedback']]
    credit_summary=dict(prospective_predictions=len(feedback),
        positive_score_and_fact_success=sum(e['delta_log_score']>0 and e['fact_success'] for e in feedback),
        positive_score_but_fact_failure=sum(e['delta_log_score']>0 and not e['fact_success'] for e in feedback),
        trusted_weight_increases=sum(e['trusted_weight_after']>e['trusted_weight_before']+1e-12 for e in feedback),
        trusted_weight_decreases=sum(e['trusted_weight_after']<e['trusted_weight_before']-1e-12 for e in feedback),
        rounds_with_positive_latent_credit=sum(bool(r['positive_credit_rules']) for r in credit_rounds),
        rounds_with_deployed_positive_weight=sum(bool(r['deployed_versions']) for r in credit_rounds),
        selected_points_with_positive_weight=sum(bool(r['selected_mixture']) for r in credit_rounds),
        positive_credit_rule_rounds_without_executable_probe=sum(not h['executable_probes'] for r in credit_rounds for h in r['positive_credit_rules']),
        positive_credit_rule_rounds_with_executable_probe=sum(bool(h['executable_probes']) for r in credit_rounds for h in r['positive_credit_rules']),
        positive_credit_rule_rounds_with_rank_effect_at_cap=sum(h['can_change_rank_at_cap'] for r in credit_rounds for h in r['positive_credit_rules']),
        maximum_independent_feedback_logscore_error=max(r['maximum_independent_feedback_logscore_error'] for r in credit_rounds),
        independently_checked_weighted_winner_equals_BO=128,
        interpretation='Evidence and trust updates occur, but no learned positive weight changes an actual selected point. All eight non-BO selections are cold-start verification, not deployment. Feedback may still affect refresh/eligibility; this does not prove an entirely non-learning trajectory.')
    audit_path=REPORTS/'v7_joint_audit.json';audit_receipt=a.read(audit_path)
    prior.INPUT_HASHES[audit_path.relative_to(ROOT).as_posix()]=a.filehash(audit_path)
    audit_summary={k:audit_receipt[k] for k in ('technical_result','failures','pending','totals')}
    findings=[
        '接口合规改善：开放窗口接受的15/15示例均在窄预算内，但原seed0是11/20，分母随提前使用验证额度和调用状态变化；这不是独立因果估计。',
        '有前瞻学习账本，但未展示正信用部署收益：24个冻结预测、14次信任增加；128轮的独立混合排序均仍选择BO。只有5轮有可部署正权重、2轮选中点承载正权重，均未改点。反馈仍可影响后续刷新与验证资格，不能把零weighted改点扩大为完全没有学习行为。',
        'C1：q1验证24.6667低于参照26.4827，虽logscore+0.132949但事实失败，信用保持0。q3/q4/q12各有正信用，下一轮都无可执行点。相较V5 AUC+4.99639但仍比BO低0.547387，后续改善不能归于正权重部署。',
        'C2：q1提前验证得到4.28368，低于参照6.51460且logscore负；原V5 q1是BO，q4才验证。q4/q5新规则信用0→0.124299→0.257457，但q5修正未改BO，q6已无可执行点。最终AUC相较V5下降3.87165；这是不同GP轨迹的总效果，不能逐轮归为同一门限即时损失。',
        'C3：与V5全部选点相同；q1正logscore但事实失败，不增信用。q11获得很小正信用后无可执行点。超过BO的轨迹没有正权重改点贡献。',
        'C4：q1与V5同点、同反馈、同信用1/3；q2同GP却失去V5 guided点p02278。V7预声明catalyst_loading范围[0.44,0.53]排除了该点的0.695546；仅剩2可执行点，即使cap=.5也不改排序，q3归为dormant。AUC较V5少0.721339但终值更高，不能单凭AUC判为全面退化。',
        'Perera：q7验证p03015获91.14，高于参照67.88，信用增至0.119415；验证后剩余可执行点0，所以不能形成后续部署。AUC仅较BO高0.011875，终值相同，不应把这点微差写成有力优势。',
        'Doyle：q1新H001对BO点p01450作预测，62.6575低于参照77.6669，logscore−3.898097，信用0。q3与V5同GP却无可执行规则，选择BO80.0744而V5验证60.5253，第一处分歧即时值反而更高，最终损失不能归为q3即时错误。q4 H002验证27.3348超过其低参照15.1458，获信用0.125222但剩余点0，不能再部署；AUC较BO−4.43322、最终−9.80885。另q4条件EI的GH64独立积分误差超阈值，正式审计FAIL尚未解除，应与机制限制分别记录。',
        'SnAr：与V5全部选点相同；q1正信用形成，q2仅余4/3可执行点且均不能在cap下改排序；q3休眠。q11/q13微小信用作用于BO原赢家，无改点。HPLC也与V5全部选点相同，但q1事实失败、logscore−4.489741且余点0；其优势不是学得正确知识后的部署。',
        '结论边界：6胜2负是开发结果，不是确认结论；技术审计当前未PASS。主要机制缺口是获信用后缺少仍可执行且能改变排序的决策，不宜直接以扩scope或硬过滤singleton修复：历史已存在该过滤删除获益验证的反例。保持冻结，不新试验。']
    summary=dict(tasks=8,v7_AUC_wins_BO=sum(c['AUC_delta_v7_minus_BO']>1e-9 for c in comparisons),
        v7_AUC_ties_BO=sum(abs(c['AUC_delta_v7_minus_BO'])<=1e-9 for c in comparisons),
        v7_AUC_losses_BO=sum(c['AUC_delta_v7_minus_BO']<-1e-9 for c in comparisons),
        v7_AUC_wins_V5=sum(c['AUC_delta_v7_minus_v5']>1e-9 for c in comparisons),
        v7_AUC_ties_V5=sum(abs(c['AUC_delta_v7_minus_v5'])<=1e-9 for c in comparisons),
        v7_AUC_losses_V5=sum(c['AUC_delta_v7_minus_v5']<-1e-9 for c in comparisons),
        all_128_query_modes={v:dict(Counter(d['mode'] for task in tasks for d in states[v][task,'full']['decisions'])) for v in folders},
        listed_event_subset_modes={v:dict(Counter(e['mode'] for e in events if e['variant']==v)) for v in folders},
        listed_event_scope='Only optional, guided, or selected point with positive deployed mixture; not all128queries.')
    report=dict(created_at=datetime.now(timezone.utc).isoformat(),development_only=True,
        script_sha256=a.filehash(__file__),reused_review_helper_sha256=a.filehash(prior.__file__),
        protocol_sha256={v:p['protocol_sha256'] for v,p in protocols.items()},input_sha256=prior.INPUT_HASHES,
        source_constraints='No production changes; V7 differs in the LLM prompt/interface, not in V5 decision/feedback gates.',
        LLM_calls=0,GP_fits=0,new_experiments=0,oracle_files_opened=0,unqueried_targets_read=0,
        technical_audit_result='NOT_A_FULL_AUDIT; independent V7 audit status is copied below without overriding its failures',
        independent_joint_audit_snapshot=audit_summary,mechanism_findings=findings,
        summary=summary,proposal_support_summary=proposal_summary,prior_proposal_baselines=baseline_rates,
        credit_summary=credit_summary,all_round_credit_review=credit_rounds,
        comparisons=comparisons,calls=calls,special_events=events,
        limitations=['Changing instruction text, geometric hints and narrow-cost flags is one combined interface change; no component-specific causal attribution.',
            'V7 prompts differ even when the observed prefix and GP are identical. LLM proposals and refresh state may differ before selected points diverge.',
            'Narrow-cost compliance is necessary, not sufficient for informative/useful validation or successful chemistry.',
            'Open-window proposal rates have policy-dependent denominators; compare counts and per-task windows, not only percentages.',
            'After chosen-point divergence the GP histories differ. Later per-round outcomes do not isolate a gate, hypothesis or weight effect.',
            'All outcomes are already queried; this report neither requests counterfactual targets nor refits GPs.'])
    (REPORTS/'v7_mechanism_review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['V7提案接口与完整轨迹：只读机制诊断','全部8任务完成后生成；无oracle/API/拟合或新试验。',
        'V7提示输入明确不同，任何proposal/choice差异都不能解释为仅改变门限的效果。',
        '独立正式审计状态：'+json.dumps(audit_summary,ensure_ascii=False),'',
        *findings,'',json.dumps(summary,ensure_ascii=False),'',
        '提议统计：'+json.dumps(proposal_summary,ensure_ascii=False),
        '学习信用统计：'+json.dumps(credit_summary,ensure_ascii=False),
        '既有 pooled baseline：窗口开放时窄门外21/46（内25/46）；同seed0 baseline：外9/20（内11/20）。',
        '分母受提前使用验证额度、LLM事件调用和轨迹影响，以上为描述性接口指标，不是独立因果效果。','',
        'task | V5 AUC | V7 AUC | BO AUC | V7-V5 | V7-BO | final-BO | first choice divergence | open accepted inside/total']
    for c in comparisons:
        m=c['metrics'];f=c['first_selected_point_divergence'];p=c['proposal_support_counts']
        lines.append(f"{c['task']} | {m['v5']['SearchAUC']:.8g} | {m['v7']['SearchAUC']:.8g} | {m['BO']['SearchAUC']:.8g} | {c['AUC_delta_v7_minus_v5']:.8g} | {c['AUC_delta_v7_minus_BO']:.8g} | {c['final_delta_v7_minus_BO']:.8g} | {f['query'] if f else '无'} | {p['open_window_narrow_inside']}/{p['open_window_accepted']}")
        if f:lines.append(f"  q{f['query']}: {f['v5_selected']}({f['v5_mode']})->{f['v7_selected']}({f['v7_mode']}), same_prefix={f['same_observed_prefix']}, GP_exact={f['GP_comparison']['all_exact']}, same_current_proposal={f['same_current_proposal']}, first_proposal_difference={f['first_proposal_difference_query']}。这是接口差异，不是纯gate效应。")
    lines.extend(['','逐次LLM调用：task/q | window open | menu narrow/total | accepted narrow/total | rejected'])
    for c in calls:
        examples=c['accepted_examples'];inside=sum(p['passes_narrow_cost_gate'] for p in examples)
        lines.append(f"{c['task']}/{c['query']} | {c['optional_window_open']} | {c['menu_narrow_inside']}/{c['menu_count']} | {inside}/{len(examples)} | {len(c['rejected_actions'])}")
    lines.extend(['','全部optional/guided及选中点有正混合权重事件（子集，非128轮总数）：'])
    for e in events:
        lines.append(f"{e['variant']} {e['task']} q{e['query']} {e['mode']}: {e['selected_id']}, observed={e['selected_raw_value']:.8g}, accepted_example={e['selected_is_any_accepted_live_plus']}, mixture_gain={e['actual_mixture_logscore']:.8g}")
        if e['verification']:lines.append('  verification='+json.dumps(e['verification'],ensure_ascii=False))
        for p in e['predictors']:
            pred,f=p['prediction'],p['feedback']
            lines.append(f"  {p['id']}v{p['version']}: ref={pred['reference_id']} (origin_changed={p['actual_reference_differs_from_origin']}), mu={pred['gp_mean_y']:.7g}, var={pred['gp_var_y']:.7g}, t={pred['tempering']:.7g}, w={p['external_mixture_weight']:.7g}; score={f['delta_log_score']:.7g}, fact={f['fact_success']}, trust={f['trusted_weight_before']:.7g}->{f['trusted_weight_after']:.7g}")
            if p['external_mixture_weight']>0:lines.append('    prior_credit='+json.dumps(p['prior_feedback_chain'],ensure_ascii=False))
    lines.extend(['','全部冻结预测/实际反馈/信用链、逐轮曲线与输入hash见JSON。性能与独立技术审计分别报告。'])
    (REPORTS/'v7_mechanism_review.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(report=str(REPORTS/'v7_mechanism_review.json'),summary=summary,proposals=proposal_summary),ensure_ascii=False))


if __name__=='__main__':main()
