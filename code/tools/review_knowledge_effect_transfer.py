"""Frozen prequential evidence transfer in V5/V7 seed0, without refitting.

Only full-arm queried histories, committed forecasts, candidate-only pools and
saved GP marginals are read. A 'new point' is an unqueried candidate, not a claim
of statistical independence. Software-derived conditional forecasts are never
described as numerical predictions supplied by the LLM.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from math import exp, log, pi, sqrt
from pathlib import Path
import numpy as np
from scipy.special import log_ndtr
import audit_knowledge_transfer_loop as audit

ROOT=audit.ROOT
REPORTS=ROOT/'experiments/reports/knowledge_loop_20261101'
INPUT_HASHES={}
VARIANTS=('v5','v7')


def read(path):
    path=Path(path)
    if path.name not in ('state.json','protocol.json') and not path.name.endswith('_commit.json'):
        raise ValueError('Unregistered diagnostic JSON input: '+str(path))
    INPUT_HASHES[path.relative_to(ROOT).as_posix()]=audit.filehash(path)
    return audit.read(path)


def gp_arrays(path):
    INPUT_HASHES[path.relative_to(ROOT).as_posix()]=audit.filehash(path)
    with np.load(path,allow_pickle=False) as saved:
        assert set(saved.files)=={'mu_y','sd_y','sd_predictive_y','base_logei','available','eligible','tie_priority'}
        return {k:saved[k].copy() for k in saved.files}


def sign(x):
    return 'positive' if x>0 else 'negative' if x<0 else 'zero'


def frozen_score(pred,y):
    """Independent observation density ratio, using the recorded tempering."""
    mu,vy,r,tau=map(float,(pred['gp_mean_y'],pred['gp_var_y'],pred['reference_y'],pred['preference_noise_sd']))
    alpha=(mu-r)/sqrt(vy+tau*tau)
    lp=float(log_ndtr(alpha))
    audit.close(lp,pred['event_log_probability'],'prequential event probability')
    if pred['log_tempering'] is None:
        return 0.,0.,lp
    pure=float(log_ndtr((y-r)/tau))-lp
    t=pred['tempering']
    gain=pure if t==1 else float(np.logaddexp(np.log1p(-t),pred['log_tempering']+pure))
    mean_shift=t*vy/sqrt(vy+tau*tau)*exp(-alpha*alpha/2-log(2*pi)/2-lp)
    return gain,mean_shift,lp


def subsequent_comparison(first,later):
    """Different physical points; differences are descriptive, not causal."""
    assert first['query']<later['query'] and first['point_id']!=later['point_id']
    return dict(first_query=first['query'],later_query=later['query'],
        first_point_id=first['point_id'],later_point_id=later['point_id'],
        same_scope=first['scope_sha256']==later['scope_sha256'],
        same_reference=first['reference_id']==later['reference_id'],
        first_reference_id=first['reference_id'],later_reference_id=later['reference_id'],
        reference_source=later['reference_source'],
        GP_residual_first=first['GP_residual_y'],GP_residual_later=later['GP_residual_y'],
        GP_residual_change=later['GP_residual_y']-first['GP_residual_y'],
        GP_residual_sign_first=first['GP_residual_sign'],GP_residual_sign_later=later['GP_residual_sign'],
        GP_standardized_residual_first=first['GP_standardized_residual'],
        GP_standardized_residual_later=later['GP_standardized_residual'],
        expert_mean_increment_first=first['expert_mean_increment_y'],
        expert_mean_increment_later=later['expert_mean_increment_y'],
        expert_mean_increment_change=later['expert_mean_increment_y']-first['expert_mean_increment_y'],
        event_probability_first=first['GP_soft_fact_probability'],event_probability_later=later['GP_soft_fact_probability'],
        logscore_first=first['delta_log_score'],logscore_later=later['delta_log_score'],
        logscore_change=later['delta_log_score']-first['delta_log_score'],
        fact_first=first['fact_success'],fact_later=later['fact_success'],
        smaller_expert_increment=later['expert_mean_increment_y']<first['expert_mean_increment_y'],
        smaller_absolute_logscore=abs(later['delta_log_score'])<abs(first['delta_log_score']),
        first_residual_direction_repeats=first['GP_residual_sign']==later['GP_residual_sign'],
        later_score_positive=later['delta_log_score']>0,
        interpretation='Same frozen rule, but different input and adaptive GP. Shrinking increments/scores can indicate local redundancy; they do not identify causal GP absorption.')


def forecast(pool,state,commit,arrays,shadow,h):
    pred=shadow['prediction'];t=commit['query'];j=pool['ix'][commit['selected_id']]
    prefix=state['history'][:t+3];known={r['candidate_id']:r for r in prefix}
    observed=state['history'][t+3]
    assert pred['point_id']==commit['selected_id']==observed['candidate_id'] and pred['point_id'] not in known
    assert pred['query_round']==t and pred['semantic_key']==h['semantic_key']
    assert audit.hkey(shadow)==(pred['hypothesis_id'],pred['version'])==audit.hkey(h)
    assert pred['prediction_sha256']==audit.digest({k:v for k,v in pred.items() if k!='prediction_sha256'})
    assert pred['reference_id'] in known and known[pred['reference_id']]['y']==pred['reference_y']
    audit.close(pred['gp_mean_y'],arrays['mu_y'][j],'frozen selected GP mean')
    audit.close(pred['gp_var_y'],arrays['sd_predictive_y'][j]**2,'frozen selected observation variance')
    audit.close(pred['gp_var_latent_y'],arrays['sd_y'][j]**2,'frozen selected latent variance')
    event=next(e for e in state['decisions'][t-1]['evidence'] if (e['hypothesis_id'],e['version'])==audit.hkey(h))
    y=float(observed['y']);gain,increment,lp=frozen_score(pred,y)
    audit.close(gain,event['delta_log_score'],'frozen prospective feedback score')
    audit.close(pred['knowledge_mean_y']-pred['gp_mean_y'],increment,'software-derived expert mean increment')
    assert event['fact_success']==(y>pred['reference_y'])
    reference_index=next(k for k,r in enumerate(prefix) if r['candidate_id']==pred['reference_id'])
    reference_query=reference_index-3 if reference_index>=4 else None
    residual=y-pred['gp_mean_y'];sd=sqrt(pred['gp_var_y'])
    if reference_index<4:source='initial_observation'
    elif reference_query<h['created_round']:source='queried_before_rule_creation'
    else:source='queried_after_rule_creation'
    return dict(query=t,point_id=pred['point_id'],mode=commit['mode'],
        reference_id=pred['reference_id'],reference_y=pred['reference_y'],reference_raw_value=known[pred['reference_id']]['raw_value'],
        reference_observed_total_index=reference_index+1,reference_query=reference_query,reference_source=source,
        reference_is_origin_minus=pred['reference_id']==h['minus_id'],
        scope_sha256=audit.digest(h['scope']),semantic_key=h['semantic_key'],
        GP_mean_y=pred['gp_mean_y'],GP_var_latent_y=pred['gp_var_latent_y'],GP_var_observation_y=pred['gp_var_y'],
        observed_y=y,observed_raw_value=observed['raw_value'],
        GP_residual_y=residual,GP_residual_raw=residual*pool['scale'],GP_residual_sign=sign(residual),
        GP_standardized_residual=residual/sd if sd>0 else None,
        software_expert_mean_y=pred['knowledge_mean_y'],expert_mean_increment_y=increment,
        software_expert_residual_y=y-pred['knowledge_mean_y'],
        squared_error_gain_y=residual**2-(y-pred['knowledge_mean_y'])**2,
        GP_soft_fact_probability=exp(lp),fact_success=event['fact_success'],delta_log_score=gain,
        tempering=pred['tempering'],log_tempering=pred['log_tempering'],zero_effect_policy=pred['zero_effect_policy'],
        trusted_weight_before=event['trusted_weight_before'],trusted_weight_after=event['trusted_weight_after'],
        deployed_weight=next((x['weight'] for x in commit['selected_mixture'] if audit.hkey(x)==audit.hkey(h)),0.),
        outside_original_background_axes=audit.outside_origin_background(pool,h,pred['point_id']),
        history_sha256=commit['history_sha256'],commit_sha256=audit.digest(commit),
        prediction_sha256=pred['prediction_sha256'],
        GP_residual_source='query-time frozen GP mean; never a hindsight refit',
        numerical_prediction_source='software normalized conditional expert; LLM supplied no numerical effect size')


def availability(pool,state,commit,arrays,h,had_forecast):
    row=next((r for r in commit['coverage'] if audit.hkey(r)==audit.hkey(h)),None)
    available=np.flatnonzero(arrays['available'])
    raw_scope=int(audit.scope_mask(pool,h,available).sum())
    if h['status']=='retracted':raw_probes=0
    else:raw_probes=len(audit.probes(pool,h,state['history'][:commit['query']+3],arrays['available']))
    probes=0 if row is None else row['executable_probes']
    if h['status']=='retracted':reason='rule_retracted'
    elif probes:
        reason='eligible_but_no_rank_effect_at_cap' if not row['can_change_rank_at_cap'] else 'eligible_with_rank_effect_at_cap'
    elif not raw_scope:reason='scope_has_no_unqueried_candidate'
    elif not raw_probes:reason='no_known_comparable_reference_or_preferred_plus'
    else:reason='comparable_probes_outside_current_budget'
    return dict(query=commit['query'],status=h['status'],
        unqueried_candidates_in_scope=raw_scope,all_available_comparable_probes=raw_probes,
        eligible_inside_scope=0 if row is None else row['eligible_inside_scope'],
        executable_probes=probes,can_change_rank_at_cap=False if row is None else row['can_change_rank_at_cap'],
        has_new_point_forecast=had_forecast,availability_reason=reason,
        selected_point=commit['selected_id'],mode=commit['mode'],
        latent_trusted_weight=h['ledger']['trusted_weight'],deployed_weight=0 if row is None else row['weight'])


def review_run(variant,task,pool,folder):
    run=folder/'runs'/f'{task}_full_agent_initial_terra';state=read(run/'state.json')
    assert state['status']=='COMPLETE' and len(state['history'])==20 and len(state['decisions'])==16
    assert len({r['candidate_id'] for r in state['history']})==20
    groups={};commits=[];arrays=[]
    for t in range(1,17):
        c=read(run/'rounds'/f'{t:02d}_commit.json');g=gp_arrays(run/'rounds'/f'{t:02d}_GP.npz')
        assert c['query']==t and c['history_sha256']==audit.digest(state['history'][:t+3])
        assert audit.digest(c)==state['decisions'][t-1]['commit_sha256']
        commits.append(c);arrays.append(g)
        for h in c['hypotheses_at_commit']:
            key=audit.hkey(h)
            if key not in groups:
                groups[key]=dict(variant=variant,task=task,unit=pool['unit'],scale=pool['scale'],id=key[0],version=key[1],
                    created_round=h['created_round'],semantic_key=h['semantic_key'],scope=h['scope'],focus=h['focus'],
                    background=h['background'],origin_plus=h['plus_id'],origin_minus=h['minus_id'],reason=h['reason'],
                    parent=h['parent'],forecasts=[],post_first_forecast_rounds=[])
            assert groups[key]['semantic_key']==h['semantic_key'] and audit.digest(groups[key]['scope'])==audit.digest(h['scope'])
        for sh in c['shadows']:
            h=next(h for h in c['hypotheses_at_commit'] if audit.hkey(h)==audit.hkey(sh))
            groups[audit.hkey(h)]['forecasts'].append(forecast(pool,state,c,g,sh,h))
    for key,group in groups.items():
        fs=group['forecasts'];assert len({f['point_id'] for f in fs})==len(fs)
        group['frozen_forecast_count']=len(fs)
        group['first_to_later_comparisons']=[subsequent_comparison(fs[0],f) for f in fs[1:]] if fs else []
        group['unique_references']=list(dict.fromkeys(f['reference_id'] for f in fs))
        last=next(h for h in commits[-1]['hypotheses_at_commit'] if audit.hkey(h)==key)
        group['status_at_last_commit']=last['status']
        group['all_rule_rounds']=[]
        for c,g in zip(commits,arrays):
            h=next((h for h in c['hypotheses_at_commit'] if audit.hkey(h)==key),None)
            if h is None:continue
            detail=availability(pool,state,c,g,h,any(f['query']==c['query'] for f in fs))
            group['all_rule_rounds'].append(detail)
            if fs and c['query']>fs[0]['query']:group['post_first_forecast_rounds'].append(detail)
        group['availability_reason_counts']=dict(Counter(r['availability_reason'] for r in group['post_first_forecast_rounds']))
        group['first_next_round']=group['post_first_forecast_rounds'][0] if group['post_first_forecast_rounds'] else None
    return list(groups.values())


def summarize(groups):
    fs=[dict(variant=g['variant'],task=g['task'],rule_id=g['id'],version=g['version'],**f) for g in groups for f in g['forecasts']]
    repeats=[g for g in groups if len(g['forecasts'])>=2]
    pairs=[p for g in groups for p in g['first_to_later_comparisons']]
    return dict(rule_versions=len(groups),with_zero_forecasts=sum(not g['forecasts'] for g in groups),
        with_one_forecast=sum(len(g['forecasts'])==1 for g in groups),with_at_least_two_forecasts=len(repeats),
        with_at_least_three_forecasts=sum(len(g['forecasts'])>=3 for g in groups),
        with_at_least_two_non_noop_forecasts=sum(sum(not f['zero_effect_policy'] for f in g['forecasts'])>=2 for g in groups),
        repeated_rule_task_clusters=len({g['task'] for g in repeats}),
        total_frozen_forecasts=len(fs),
        distinct_variant_task_query_points=len({(f['variant'],f['task'],f['query'],f['point_id']) for f in fs}),
        distinct_task_data_rows_across_variants=len({(f['task'],f['point_id']) for f in fs}),
        first_to_later_pairs=len(pairs),same_reference_pairs=sum(p['same_reference'] for p in pairs),
        different_reference_pairs=sum(not p['same_reference'] for p in pairs),
        same_GP_residual_sign_pairs=sum(p['first_residual_direction_repeats'] for p in pairs),
        later_positive_logscore_pairs=sum(p['later_score_positive'] for p in pairs),
        later_positive_logscore_but_negative_GP_residual_pairs=sum(p['later_score_positive'] and p['GP_residual_later']<0 for p in pairs),
        smaller_expert_increment_and_absolute_score_pairs=sum(p['smaller_expert_increment'] and p['smaller_absolute_logscore'] for p in pairs),
        positive_GP_residuals=sum(f['GP_residual_y']>0 for f in fs),negative_GP_residuals=sum(f['GP_residual_y']<0 for f in fs),
        first_score_positive_rules=sum(bool(g['forecasts']) and g['forecasts'][0]['delta_log_score']>0 for g in groups),
        first_fact_and_score_positive_rules=sum(bool(g['forecasts']) and g['forecasts'][0]['delta_log_score']>0 and g['forecasts'][0]['fact_success'] for g in groups),
        repeated_rule_ids=[f"{g['variant']}/{g['task']}/{g['id']}v{g['version']}" for g in repeats],
        one_forecast_rule_first_next_reasons=dict(Counter(g['first_next_round']['availability_reason'] if g['first_next_round'] else 'no_future_budget' for g in groups if len(g['forecasts'])==1)),
        post_first_forecast_rule_round_reasons=dict(Counter(r['availability_reason'] for g in groups for r in g['post_first_forecast_rounds'])),
        numerical_effect_predictions_supplied_by_LLM=0,
        statistical_effective_sample_size=None,
        ESS_limitation='These are design/cluster counts, not independent effective N: adaptive prefixes, shared references, overlapping hypotheses and duplicate dataset rows are correlated.')


def self_test():
    p=dict(gp_mean_y=.3,gp_var_y=.04,reference_y=.4,preference_noise_sd=.06,tempering=1.,log_tempering=0.)
    p['event_log_probability']=float(log_ndtr((.3-.4)/sqrt(.04+.06**2)))
    low,shift,_=frozen_score(p,.1);high,_,_=frozen_score(p,.6)
    assert low<0<high and shift>0
    p.update(tempering=0.,log_tempering=None);assert frozen_score(p,.6)[:2]==(0.,0.)
    assert summarize([])['statistical_effective_sample_size'] is None
    pool=dict(ids=['ref','new'],ix={'ref':0,'new':1},axes=['x'],kinds=['continuous'],features=[[0.],[.3]])
    h=dict(id='h',version=1,status='dormant',focus={'x':{'direction':'increase','bounds':[0.,.5]}},
        scope={'axes':{'x':{'kind':'continuous','bounds':[0.,.5]}},'background_radius':.25},
        categorical_preference={},ledger={'trusted_weight':0.})
    h['semantic_key']=audit.digest(audit.contract(h))
    state={'history':[{'candidate_id':'ref','y':.2}]}
    c=dict(query=1,selected_id='new',mode='base_bo',coverage=[dict(id='h',version=1,
        executable_probes=0,eligible_inside_scope=0,can_change_rank_at_cap=False,weight=0.)])
    assert availability(pool,state,c,{'available':np.array([False,True])},h,False)['availability_reason']=='comparable_probes_outside_current_budget'
    assert availability(pool,state,c,{'available':np.array([False,False])},h,False)['availability_reason']=='scope_has_no_unqueried_candidate'
    h['status']='retracted'
    assert availability(pool,state,c,{'available':np.array([False,True])},h,False)['availability_reason']=='rule_retracted'
    print('PASS: frozen density/no-effect, empty sample counts, budget versus scope versus retraction; no experiment files read.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test:self_test();return
    protocols={v:read(ROOT/'experiments/runs/knowledge_loop_20261101'/v/'protocol.json') for v in VARIANTS}
    assert protocols['v5']['tasks']==protocols['v7']['tasks'] and len(protocols['v5']['tasks'])==8
    all_groups=[]
    for variant,p in protocols.items():
        assert p['seed']==0 and p['initial']==4 and p['queries']==16
        assert p['protocol_sha256']==audit.digest({k:v for k,v in p.items() if k!='protocol_sha256'})
        folder=ROOT/'experiments/runs/knowledge_loop_20261101'/variant
        for manifest in p['inputs']:
            pool=audit.common.public_pool(ROOT,manifest)
            for rel in manifest['files_sha256']:
                if rel.endswith('/pool.json'):INPUT_HASHES[rel]=audit.filehash(ROOT/rel)
            all_groups.extend(review_run(variant,manifest['task'],pool,folder))
    summaries={v:summarize([g for g in all_groups if g['variant']==v]) for v in VARIANTS}
    summaries['pooled_descriptive_only']=summarize(all_groups)
    task_counts=[dict(variant=v,task=task,**summarize([g for g in all_groups if g['variant']==v and g['task']==task])) for v in VARIANTS for task in protocols[v]['tasks']]
    findings=[
        '样本不足以作稳定幅度校准结论。V5仅6/46规则版本有至少2次冻结预测，V7仅3/28；合计9规则、10个首点到后续新点对照，覆盖4种任务。只有V5 Case2 H005达到3次；Case3、Perera、Doyle、HPLC两方案均没有重复预测规则。',
        '56个前瞻预测不是56个独立实验样本：对应52个variant/task/query/point，但跨方案只有40个不同task/data行。同一选点可同时评估多条规则，同一参照反复使用；无法从这些日志给出可靠的统计有效样本量。',
        '存在局部正残差持续的开发性例子：V5 Case1 H001 q1→q4，残差+.04560→+.12958、score+.4383→+.9587；V7 Case2 H002 q4→q5，残差+.02165→+.04081、score+.2499→+.2769。每例仍只有一个后续新点，不能据此估计稳定效应或选择调参赢家。',
        '与GP增量趋于冗余相符的例子：V5 Case2 H005 q7残差+.11436，q9变−.003766，q16仅+.000995；软件专家均值增量由.03536降为.000583/.000912。同ref且后续score很小，但点位与GP均变，不能单独确认是GP吸收。SnAr两方案q11/q13的同一物理点也表现+.03197→−.005078；它们是同一轨迹，不能当作独立复现。',
        '正logscore不等于正的可迁移GP残差：10个后续对照中3个真实残差为负但专家score为正。V7 Case2 H004 q13残差−.001947、score+3.61e−5；其q11残差已经为负。旧规则是软件定义的单向条件分布，LLM没有提供数字幅度，所以这些数据不能证明LLM预测过某个数值效应。',
        '缺少第二次预测的原因不能统称scope不足或GP吸收。单次预测规则的紧接下一轮：V5 19条中11条仍有全池可比probe、只是被当轮机会预算排除，3条scope确实没有未查点；V7 18条中6条预算外、3条scope耗尽、3条无可比preferred-plus，另外6条仍有预算内可执行点但未再选中。后续撤回另按每轮记录，不能当成负面幅度估计。',
        '能够检验的是数据管线和极小规模的事前残差方向复现；无法用这些旧日志确认一个新数值专家的幅度校准、跨背景迁移、相对current-GP的增量或多数任务优势。任何后续模型需要另行事前注册并保留独立数据；本报告不重拟合、不选幅度/门限、不产生赢家。']
    limitations=[
        'Only V5/V7 seed0 development logs. No seed2 inspection, no new experiment, API, oracle access or GP fitting.',
        'A later independent new point means a different previously unqueried candidate. It is not an independent statistical replicate.',
        'Every residual is observed_y minus its own query-time committed GP mean, checked against saved GP marginals. No later GP refit residual is substituted.',
        'LLM supplied rule direction/scope/reason, not numerical effect size. Expert means, variances and logscore come from the software conditional model; these logs cannot directly calibrate an LLM numerical prediction that never existed.',
        'Repeated rule forecasts share an unchanged semantic/scope but may use a new highest compatible reference. Shared references and overlapping rules correlate observations; pooled forecast count is not an effective independent sample size.',
        'Rules with one forecast have no within-rule out-of-sample calibration test. Two forecasts provide at most one first-evidence/one later-point contrast, adaptively selected; no train/test hyperparameter search is justified.',
        'A decreasing software expert mean increment or logscore can be consistent with redundancy to the current GP, but distinct inputs, changed references and refitted GP hyperparameters prevent identifying causal absorption.',
        'Scope exhaustion, incompatible references, budget exclusion and explicit retraction are separately recorded; absence of a repeated prediction does not demonstrate GP absorption or false chemistry.',
        'V7 independent audit retains its Doyle q4 conditional-EI quadrature failure. Frozen density/feedback arithmetic is checkable, but this diagnosis does not clear the acquisition numerical defect or promote V7.',
        'No posterior counterfactual labels, fabricated LLM residual predictions, calibrated effect winner, optimized threshold or new strategy is produced.']
    report=dict(created_at=datetime.now(timezone.utc).isoformat(),script_sha256=audit.filehash(__file__),
        helper_sha256={Path(audit.__file__).name:audit.filehash(audit.__file__),Path(audit.v4.__file__).name:audit.filehash(audit.v4.__file__)},
        protocol_sha256={v:p['protocol_sha256'] for v,p in protocols.items()},input_sha256=INPUT_HASHES,
        development_only=True,GP_fits=0,LLM_calls=0,oracle_files_opened=0,unqueried_targets_read=0,new_experiments=0,
        technical_scope='Prospective prediction/feedback and geometry review only; not a full GP/acquisition audit.',
        summaries=summaries,task_counts=task_counts,rules=all_groups,findings=findings,limitations=limitations)
    out=REPORTS/'knowledge_effect_transfer_review.json';out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['V5/V7：冻结规则的前瞻残差与证据迁移诊断','仅已查询的开发轨迹；所有残差使用当时事前GP均值。','',*findings,'',
        'variant | rules | forecasts | >=2 forecast rules | >=3 | repeated-task clusters | first→later | same ref',
        *[f"{v} | {s['rule_versions']} | {s['total_frozen_forecasts']} | {s['with_at_least_two_forecasts']} | {s['with_at_least_three_forecasts']} | {s['repeated_rule_task_clusters']} | {s['first_to_later_pairs']} | {s['same_reference_pairs']}" for v,s in summaries.items()],
        '', '统计有效样本量不能从这些相关的自适应记录直接给出；下表为实际可用规则簇与前瞻观测数，不当作独立重复。',
        '全部任务：variant/task | rules | 0/1/>=2 forecasts | first→later']
    for s in task_counts:lines.append(f"{s['variant']}/{s['task']} | {s['rule_versions']} | {s['with_zero_forecasts']}/{s['with_one_forecast']}/{s['with_at_least_two_forecasts']} | {s['first_to_later_pairs']}")
    lines.extend(['','重复预测的规则（统一内部目标尺度；GP residual不是后来重拟合值）：'])
    for g in all_groups:
        if len(g['forecasts'])<2:continue
        lines.append(f"{g['variant']}/{g['task']}/{g['id']}v{g['version']} fixed_scope={audit.digest(g['scope'])}; final={g['status_at_last_commit']}")
        for f in g['forecasts']:
            z='NA' if f['GP_standardized_residual'] is None else f"{f['GP_standardized_residual']:.6g}"
            lines.append(f"  q{f['query']} {f['point_id']} ref={f['reference_id']}({f['reference_source']}): mu={f['GP_mean_y']:.7g}, y={f['observed_y']:.7g}, residual={f['GP_residual_y']:.7g}, z={z}, expert_increment={f['expert_mean_increment_y']:.7g}, score={f['delta_log_score']:.7g}, fact={f['fact_success']}, trust={f['trusted_weight_before']:.6g}->{f['trusted_weight_after']:.6g}")
        for pair in g['first_to_later_comparisons']:
            lines.append(f"  first→q{pair['later_query']}: same_ref={pair['same_reference']}, residual_sign={pair['GP_residual_sign_first']}→{pair['GP_residual_sign_later']}, increment_shrinks={pair['smaller_expert_increment']}, abs_score_shrinks={pair['smaller_absolute_logscore']}; 不能据此单独认定GP吸收。")
    lines.extend(['','所有取得首次证据规则的后续可用性：rule | N forecasts | first-next reason | subsequent reason counts'])
    for g in all_groups:
        if not g['forecasts']:continue
        first_next=g['first_next_round']
        lines.append(f"{g['variant']}/{g['task']}/{g['id']}v{g['version']} | {len(g['forecasts'])} | {first_next['availability_reason'] if first_next else 'no_future_budget'} | {json.dumps(g['availability_reason_counts'],ensure_ascii=False)}")
    lines.extend(['','适用边界：',*limitations])
    (REPORTS/'knowledge_effect_transfer_review.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(report=str(out),summaries=summaries),ensure_ascii=False))


if __name__=='__main__':main()
