"""Read-only V6/V5 trajectory diagnosis after all eight V6 full runs finish.

No production imports, APIs, model fits, oracle files or unqueried targets.
Same-prefix marginal calculations use the already audited saved GP arrays.
This is a mechanism review, not a substitute for the independent V6 audit.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from math import log
from pathlib import Path
import json
import sys

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p/'code/tools/audit_knowledge_transfer_loop.py').is_file())
sys.path.insert(0, str(ROOT/'code/tools'))
import numpy as np
import audit_knowledge_transfer_loop as audit

REPORTS = ROOT/'experiments/reports/knowledge_loop_20261101'
INPUT_HASHES = {}


def read(path):
    path = Path(path)
    if path.name not in ('state.json', 'protocol.json') and not path.name.endswith('_commit.json'):
        raise ValueError('Unregistered diagnostic input: '+str(path))
    INPUT_HASHES[path.relative_to(ROOT).as_posix()] = audit.filehash(path)
    return audit.read(path)


def arrays_for(run, t):
    path = run/'rounds'/f'{t:02d}_GP.npz'
    INPUT_HASHES[path.relative_to(ROOT).as_posix()] = audit.filehash(path)
    with np.load(path, allow_pickle=False) as saved:
        values = {k: saved[k].copy() for k in saved.files}
    assert set(values) == {'mu_y', 'sd_y', 'sd_predictive_y', 'base_logei', 'available', 'eligible', 'tie_priority'}
    return values


def measures(state):
    raw = np.asarray([row['raw_value'] for row in state['history']], float)
    assert len(raw) == 20 and np.isfinite(raw).all()
    best = np.maximum.accumulate(raw)
    return dict(SearchAUC=float(best[4:].mean()), final_best=float(best[-1]),
        first_best_total=int(np.flatnonzero(raw == best[-1])[0])+1,
        optional_verifications=sum(d['mode'] == 'optional_verification' for d in state['decisions']),
        knowledge_choices=sum(d['mode'] == 'knowledge_guided' for d in state['decisions']))


def advance_proposals(previous, commit):
    records = deepcopy(previous)
    for event in commit['reviews']:
        if not event.get('accepted') or event.get('op') not in ('add', 'revise'):
            continue
        if event.get('outcome') not in ('add', 'revise', 'duplicate_semantics_preserved'):
            continue
        action = commit['proposal']['actions'][event['action_index']]
        h = next(h for h in commit['hypotheses_at_commit']
                 if audit.hkey(h) == (event['id'], event['version']))
        record = dict(id=h['id'], version=h['version'], semantic_key=h['semantic_key'],
            point_id=action['plus_id'], proposed_reference_id=action['minus_id'],
            proposal_round=commit['query'], action_index=event['action_index'], action_sha256=audit.digest(action))
        if record not in records:
            records.append(record)
    return records


def nominations(commit, records, point, key=None):
    live = {(h['id'], h['version'], h['semantic_key']) for h in commit['hypotheses_at_commit']
            if h['status'] != 'retracted'}
    return [deepcopy(r) for r in records if r['point_id'] == point
            and (r['id'], r['version'], r['semantic_key']) in live
            and (key is None or audit.hkey(r) == key)]


def point_gate(pool, state, commit, arrays, point, key, records):
    """Old cold-start gates and V6's separate accepted-example membership."""
    t = commit['query']; prefix = state['history'][:t+3]
    h = next((h for h in commit['hypotheses_at_commit'] if audit.hkey(h) == key), None)
    if h is None:
        return dict(point_id=point, rule=list(key), rule_present=False)
    rows, calculated = audit.calculations(pool, commit['hypotheses_at_commit'], prefix, arrays)
    row = next((r for r in rows if audit.hkey(r) == key), None)
    data = calculated.get(key)
    j, bo = pool['ix'][point], pool['ix'][commit['BO_reference_id']]
    evidence = audit.evidence(h)
    gap = float(arrays['base_logei'][bo]-arrays['base_logei'][j])
    matches = nominations(commit, records, point, key)
    gates = dict(rule_live=h['status'] != 'retracted', new_point=bool(arrays['available'][j]),
        in_budget=bool(arrays['eligible'][j]), within_verification_window=t<=10,
        quota_available=sum(d['mode']=='optional_verification' for d in state['decisions'][:t-1])<1,
        zero_weight=bool(row is not None and row['weight']==0),
        rank_effect_at_cap=bool(row and row['can_change_rank_at_cap']),
        no_net_counterevidence=evidence['effective_prediction_score']>=0 and evidence['failures']<=evidence['successes'],
        non_BO=j!=bo, narrow_gap=gap<=min(commit['epsilon'],log(1.1)),
        executable=False, incremental_MI=False, positive_effect=False)
    out = dict(point_id=point, rule=list(key), rule_present=True, origin_plus=h['plus_id'],
        origin_minus=h['minus_id'], gap=gap, accepted_live_proposals=matches,
        V6_proposal_membership=bool(matches), old_gates=gates,
        prior_effective_prediction_score=evidence['effective_prediction_score'],
        prior_successes=evidence['successes'], prior_failures=evidence['failures'])
    if data and j in data['positions']:
        p=data['positions'][j]; bp=data['positions'].get(bo)
        info=float(data['values']['sign_information'][p])
        binfo=float(data['values']['sign_information'][bp]) if bp is not None else 0.
        ref=data['probes'][j]
        out.update(sign_information=info, same_rule_BO_information=binfo, incremental_information=info-binfo,
            offset=float(data['values']['offset'][p]), reference=deepcopy(ref),
            reference_changed_from_origin=ref['reference_id']!=h['minus_id'])
        gates.update(executable=True, incremental_MI=info-binfo>=.01,
                     positive_effect=bool(data['values']['offset'][p]>audit.ATOL))
    out['failed_old_gates']=[k for k,v in gates.items() if not v]
    out['all_old_gates_pass']=all(gates.values())
    out['all_V6_gates_pass']=out['all_old_gates_pass'] and bool(matches)
    return out


def special_event(variant, pool, run, state, commit, records):
    t=commit['query']; decision=state['decisions'][t-1]
    arrays=arrays_for(run,t)
    key=audit.hkey(commit['verification']) if commit['verification'] else None
    weights={audit.hkey(x):x['weight'] for x in commit['selected_mixture']}
    feedback={(e['hypothesis_id'],e['version']):e for e in decision['evidence']}
    predictors=[]
    for shadow in commit['shadows']:
        k=audit.hkey(shadow); p=shadow['prediction']; h=next(h for h in commit['hypotheses_at_commit'] if audit.hkey(h)==k)
        predictors.append(dict(id=k[0],version=k[1],is_validation_target=k==key,
            external_mixture_weight=weights.get(k,0.), conditional_density_weight=weights.get(k,0.)*p['tempering'],
            origin_plus=h['plus_id'],origin_minus=h['minus_id'],point_is_rule_origin=commit['selected_id']==h['plus_id'],
            actual_reference_differs_from_origin=p['reference_id']!=h['minus_id'],
            accepted_live_proposals=nominations(commit,records,commit['selected_id'],k),
            prior_feedback_chain=[{name:deepcopy(row[name]) for name in
                ('query_round','point_id','reference_id','reference_y','observed_y','fact_success','delta_log_score')}
                for row in h['ledger']['events']],
            information_nats=shadow['information_nats'],prediction=deepcopy(p),feedback=deepcopy(feedback[k])))
    event=dict(variant=variant,task=state['task'],query=t,mode=commit['mode'],selected_id=commit['selected_id'],
        BO_reference_id=commit['BO_reference_id'],opportunity_gap=commit['opportunity_gap'],
        selected_raw_value=decision['raw_value'],incumbent_before_raw=decision['incumbent_before_y']*pool['scale'],
        incumbent_after_raw=decision['incumbent_after_y']*pool['scale'],
        selected_is_any_accepted_live_plus=bool(nominations(commit,records,commit['selected_id'])),
        verification=deepcopy(commit['verification']),deployed_versions=deepcopy(commit['deployed_versions']),
        selected_mixture=deepcopy(commit['selected_mixture']),actual_mixture_logscore=decision['deployed_mixture_log_score_gain'],
        predictors=predictors,history_sha256=commit['history_sha256'],commit_sha256=audit.digest(commit))
    if key is not None:
        h=next(h for h in commit['hypotheses_at_commit'] if audit.hkey(h)==key)
        event['target_origin_gates']=point_gate(pool,state,commit,arrays,h['plus_id'],key,records)
        event['actual_selected_gates']=point_gate(pool,state,commit,arrays,commit['selected_id'],key,records)
    return event


def compare_gp(a,b):
    assert set(a)==set(b)
    fields={}
    for key in a:
        fields[key]=dict(exact=bool(np.array_equal(a[key],b[key])),
            maximum_absolute_difference=float(np.max(np.abs(a[key].astype(float)-b[key].astype(float)))))
    return dict(all_exact=all(v['exact'] for v in fields.values()),fields=fields)


def first_difference(pool, left, right, commits, records, dirs):
    for t,(old,new) in enumerate(zip(commits['v5'],commits['v6']),1):
        if old['selected_id']==new['selected_id']:
            continue
        prefixes={v:s['history'][:t+3] for v,s in [('v5',left),('v6',right)]}
        same=audit.digest(prefixes['v5'])==audit.digest(prefixes['v6'])
        arrays={v:arrays_for(dirs[v],t) for v in ('v5','v6')}
        result=dict(query=t,same_observed_prefix=same,
            GP_comparison=compare_gp(arrays['v5'],arrays['v6']),
            old_selected=old['selected_id'],new_selected=new['selected_id'],old_mode=old['mode'],new_mode=new['mode'],
            same_BO_reference=old['BO_reference_id']==new['BO_reference_id'],
            same_menu=old['menu_ids']==new['menu_ids'],same_proposal=audit.digest(old['proposal'])==audit.digest(new['proposal']),
            same_hypotheses=audit.digest(old['hypotheses_at_commit'])==audit.digest(new['hypotheses_at_commit']),
            old_was_any_accepted_live_plus=bool(nominations(old,records['v5'][t-1],old['selected_id'])),
            v6_has_old_point_as_any_live_proposal=bool(nominations(new,records['v6'][t-1],old['selected_id'])),
            observed_outcomes=dict(v5=left['decisions'][t-1]['raw_value'],v6=right['decisions'][t-1]['raw_value']))
        if old['verification']:
            key=audit.hkey(old['verification']);h=next(h for h in old['hypotheses_at_commit'] if audit.hkey(h)==key)
            result['old_validation_origin']=point_gate(pool,left,old,arrays['v5'],h['plus_id'],key,records['v5'][t-1])
            result['old_selected_under_V5']=point_gate(pool,left,old,arrays['v5'],old['selected_id'],key,records['v5'][t-1])
            result['old_selected_under_V6']=point_gate(pool,right,new,arrays['v6'],old['selected_id'],key,records['v6'][t-1])
            q=result['old_selected_under_V6']
            result['membership_only_exclusion']=bool(q.get('all_old_gates_pass') and not q['V6_proposal_membership'])
        result['interpretation']='Same-prefix immediate selection-policy comparison only; subsequent prefixes diverge.' if same else 'Prefixes already differ; cannot assign an immediate causal effect to the selection restriction.'
        return result
    return None


def self_test():
    d=compare_gp({'x':np.array([1.,2.])},{'x':np.array([1.,2.])})
    assert d['all_exact']
    assert not compare_gp({'x':np.array([1.])},{'x':np.array([1.1])})['all_exact']
    c=dict(hypotheses_at_commit=[dict(id='h',version=2,semantic_key='s',status='active')])
    rs=[dict(id='h',version=1,semantic_key='s',point_id='x'),dict(id='h',version=2,semantic_key='s',point_id='x')]
    assert len(nominations(c,rs,'x'))==1
    c['hypotheses_at_commit'][0]['status']='retracted'
    assert not nominations(c,rs,'x')
    print('PASS: synthetic GP comparison and live proposal identity checks; no experiment files opened.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        self_test();return
    folders={v:ROOT/'experiments/runs/knowledge_loop_20261101'/v for v in ('v5','v6')}
    protocols={v:read(folder/'protocol.json') for v,folder in folders.items()}
    for p in protocols.values():
        assert p['protocol_sha256']==audit.digest({k:v for k,v in p.items() if k!='protocol_sha256'})
    assert protocols['v5']['tasks']==protocols['v6']['tasks'] and len(protocols['v6']['tasks'])==8
    tasks=protocols['v6']['tasks']; states={}; dirs={}
    # All-complete gate before any mechanism calculation or report write.
    for variant in folders:
        states[variant]={};dirs[variant]={}
        for task in tasks:
            for mode in ('bo','full'):
                run=folders[variant]/'runs'/f'{task}_{mode}_agent_initial_terra'
                state=read(run/'state.json')
                if state['status']!='COMPLETE' or len(state['decisions'])!=16 or len(state['history'])!=20:
                    raise RuntimeError(f'Await all complete: {variant}/{task}/{mode}')
                states[variant][task,mode]=state;dirs[variant][task,mode]=run
    pools={m['task']:audit.common.public_pool(ROOT,m) for m in protocols['v6']['inputs']}
    # public_pool reads candidate-only pool.json; hash only those input files.
    for m in protocols['v6']['inputs']:
        for rel in m['files_sha256']:
            if rel.endswith('/pool.json'):
                INPUT_HASHES[rel]=audit.filehash(ROOT/rel)
    comparisons,events=[],[]
    for task in tasks:
        pool=pools[task]; state={v:states[v][task,'full'] for v in folders}
        assert audit.digest(state['v5']['history'][:4])==audit.digest(state['v6']['history'][:4])
        assert audit.digest(states['v5'][task,'bo']['history'])==audit.digest(states['v6'][task,'bo']['history'])
        commits,record_history={},{}
        for variant in folders:
            run=dirs[variant][task,'full'];commits[variant]=[];record_history[variant]=[];records=[]
            for t in range(1,17):
                c=read(run/'rounds'/f'{t:02d}_commit.json')
                assert c['history_sha256']==audit.digest(state[variant]['history'][:t+3])
                assert audit.digest(c)==state[variant]['decisions'][t-1]['commit_sha256']
                records=advance_proposals(records,c)
                if variant=='v6':
                    assert audit.digest(records)==audit.digest(c['validation_proposals'])
                commits[variant].append(c);record_history[variant].append(deepcopy(records))
                if c['mode'] in ('optional_verification','knowledge_guided') or c['selected_mixture']:
                    events.append(special_event(variant,pool,run,state[variant],c,records))
        first=first_difference(pool,state['v5'],state['v6'],commits,record_history,
                              {v:dirs[v][task,'full'] for v in folders})
        stats={v:measures(state[v]) for v in folders};stats['BO']=measures(states['v6'][task,'bo'])
        curves={v:np.maximum.accumulate([r['raw_value'] for r in state[v]['history']])[4:] for v in folders}
        curve_rows=[]
        previous_delta=0.
        for t in range(1,17):
            delta=float(curves['v6'][t-1]-curves['v5'][t-1])
            curve_rows.append(dict(query=t,v5_selected=commits['v5'][t-1]['selected_id'],v6_selected=commits['v6'][t-1]['selected_id'],
                same_prequery_prefix=commits['v5'][t-1]['history_sha256']==commits['v6'][t-1]['history_sha256'],
                v5_observed=state['v5']['decisions'][t-1]['raw_value'],v6_observed=state['v6']['decisions'][t-1]['raw_value'],
                v5_best=float(curves['v5'][t-1]),v6_best=float(curves['v6'][t-1]),
                incumbent_delta=delta,delta_change=delta-previous_delta))
            previous_delta=delta
        a5,a6,b=stats['v5']['SearchAUC'],stats['v6']['SearchAUC'],stats['BO']['SearchAUC']
        comparisons.append(dict(task=task,unit=pool['unit'],metrics=stats,first_selected_point_divergence=first,
            AUC_delta_v6_minus_v5=a6-a5,AUC_delta_v6_minus_BO=a6-b,AUC_delta_v5_minus_BO=a5-b,
            final_delta_v6_minus_v5=stats['v6']['final_best']-stats['v5']['final_best'],
            final_delta_v6_minus_BO=stats['v6']['final_best']-stats['BO']['final_best'],
            prior_AUC_benefit_retained=bool(a5>b+1e-9 and a6>b+1e-9),
            prior_AUC_benefit_lost=bool(a5>b+1e-9 and a6<=b+1e-9),
            curve=curve_rows,causal_boundary='After first selected-point divergence, per-round curve differences are descriptive, not isolated immediate effects.'))
    findings=[
        '5个首次分歧全部同prefix、同GP数组、同menu/提案/规则；V5被选点在V6目标规则下仍通过原门限，仅因不是已接受示例plus被排除。没有据此发现GP漂移或阈值计算错误。',
        'Case1：q1从V5 H001的scope点p01236转为已提议H002示例p01264；后者真实产率高4.889874点，且同时给两条覆盖规则正事实/正分数。q2的H001带权选点p01236失败，score=-.5810745使trust从1/3降至0，纠错生效。后续轨迹AUC增加9.880470，但终值与V5相同；不能将所有后期差值只归因于q1。',
        'Case3：q1原示例本已过门，新限制用p02278代替scope内MI更高的p02094。两者都未超过已知ref；正预测分数不等于事实成功，双账均让trust保持0。之后没有知识加权改点；AUC下降.684590、终值下降1.064926来自不同实验输入后的搜索轨迹，非错误信用增长。',
        'Case4：原示例gap=.539310>log1.1，scope中的合格替代点被禁，q1退BO。V5 q1正证据曾让H001获得1/3信用并在q2带权改点；V6整条轨迹回到BO，失去AUC优势2.184374，但终值反而比V5高.727646。不能把这项限制称为全面改善。',
        'Doyle：q3旧scope点p00462换为另一已提议规则H002示例p01510，真实值60.525329变23.041344；两者都低于各自既定ref。V5 score正而fact假，V6 score=-.359855且fact假，均无信用，后续没有加权改点。AUC下降4.842061、终值下降11.305784；门限只约束GP机会代价和信息，不保证真实实验产率。',
        'SnAr：原示例DeltaMI=.00232754<.01，scope替代点被禁后q1用BO。初始incumbent相对V5少1074.067；BO点仍为H001提供事前score=.7010703和正事实，合法获得1/3信用，q2因此仍有加权改点（不是LLM示例点）。q6另条规则负score=-1.517006/事实失败使其trust归零。AUC下降735.261613但最终追平；有信用更新/改点不等于提升搜索。',
        'Case2与HPLC主动验证本来就是已接受示例，全部16点保持V5；对应AUC收益保留。Perera两版均与BO同轨。',
        '总体：仅Case1改善，3任务同轨、4任务AUC降低；V6对BO仅4/8胜，不满足多数门。真实示例保真是可解释性限制，不能替代性能验证；本轮不支持推广该限制。',
        '边界：V6只限制主动验证，不限制已学规则在scope中的加权部署；两次guided仍可选未被LLM列为示例的点。所有后期曲线差只作描述，不能用不同prefix比较证明单轮即时因果。']
    result=dict(created_at=datetime.now(timezone.utc).isoformat(),development_only=True,
        technical_audit_result='NOT_A_FULL_AUDIT; use independent V6 joint receipt',
        script_sha256=audit.filehash(Path(__file__)),protocol_sha256={v:p['protocol_sha256'] for v,p in protocols.items()},
        input_sha256=INPUT_HASHES,oracle_files_opened=0,unqueried_targets_read=0,GP_fits=0,LLM_calls=0,new_experiments=0,
        comparisons=comparisons,special_events=events,diagnostic_findings=findings,
        summary=dict(tasks=len(comparisons),v6_AUC_wins_BO=sum(x['AUC_delta_v6_minus_BO']>1e-9 for x in comparisons),
            v6_AUC_ties_BO=sum(abs(x['AUC_delta_v6_minus_BO'])<=1e-9 for x in comparisons),
            v6_AUC_losses_BO=sum(x['AUC_delta_v6_minus_BO']<-1e-9 for x in comparisons),
            v6_AUC_wins_V5=sum(x['AUC_delta_v6_minus_v5']>1e-9 for x in comparisons),
            v6_AUC_ties_V5=sum(abs(x['AUC_delta_v6_minus_v5'])<=1e-9 for x in comparisons),
            v6_AUC_losses_V5=sum(x['AUC_delta_v6_minus_v5']<-1e-9 for x in comparisons),
            event_modes_scope='Only the diagnostic event subset: optional verification, knowledge-guided choices, or a selected point with a positive deployed mixture; not all 128 queries.',
            event_modes={v:dict(Counter(e['mode'] for e in events if e['variant']==v)) for v in folders},
            all_128_query_modes={v:dict(base_bo=128-sum(c['metrics'][v]['optional_verifications']+c['metrics'][v]['knowledge_choices'] for c in comparisons),
                optional_verification=sum(c['metrics'][v]['optional_verifications'] for c in comparisons),
                knowledge_guided=sum(c['metrics'][v]['knowledge_choices'] for c in comparisons)) for v in folders}),
        limitations=['Already inspected seed0 development comparison; not an independent confirmation.',
            'Only first divergence with the same own-observed prefix supports a direct selection-mechanism comparison.',
            'Later outcomes use different histories; no counterfactual unqueried outcomes are obtained.',
            'Proposals are original prompt examples, not explicitly ranked independent LLM recommendations.',
            'All existing numerical, information, ref-selection and deployment rules remain those of frozen V5.'])
    (REPORTS/'v6_mechanism_review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['V6 示例点验证：只读机制诊断',
        '只读取完成轨迹中的已查询结果、事前commit及已存GP；无oracle、拟合、API或新试验。',
        '机制分析不替代独立全量审计；首次选点分歧以后，不把不同历史中的轮次差异称为即时因果。','',
        'event_modes仅统计本诊断列出的事件子集（主动验证、知识改点、选中点带有正混合权重），不是全部128轮；完整模式数见all_128_query_modes。',
        json.dumps(result['summary'],ensure_ascii=False),'',*findings,'',
        'task | V5 AUC | V6 AUC | BO AUC | V6-V5 | V6-BO | V6 final | final-BO | first divergence']
    for c in comparisons:
        m=c['metrics'];f=c['first_selected_point_divergence']
        lines.append(f"{c['task']} | {m['v5']['SearchAUC']:.8g} | {m['v6']['SearchAUC']:.8g} | {m['BO']['SearchAUC']:.8g} | {c['AUC_delta_v6_minus_v5']:.8g} | {c['AUC_delta_v6_minus_BO']:.8g} | {m['v6']['final_best']:.8g} | {c['final_delta_v6_minus_BO']:.8g} | {f['query'] if f else '无'}")
    for c in comparisons:
        f=c['first_selected_point_divergence'];lines.extend(['',c['task']])
        if f:
            lines.append(f"首次分歧q{f['query']}：{f['old_selected']}({f['old_mode']})->{f['new_selected']}({f['new_mode']})；同前缀={f['same_observed_prefix']}，GP数组完全相同={f['GP_comparison']['all_exact']}，同提案={f['same_proposal']}。")
            if 'old_validation_origin' in f:
                o=f['old_validation_origin'];s=f['old_selected_under_V6']
                lines.append(f"V5目标规则原plus={o['point_id']}，gap={o['gap']:.7g}，DeltaMI={o.get('incremental_information')}，失败门限={o.get('failed_old_gates')}。")
                lines.append(f"V5实际选点是当前被接受示例={f['old_was_any_accepted_live_plus']}；在V6相同规则下示例membership={s.get('V6_proposal_membership')}；只因membership排除={f.get('membership_only_exclusion')}。")
        else:
            lines.append('16次追加查询完全相同，无选点分歧。')
        changed=[r for r in c['curve'] if abs(r['delta_change'])>1e-9]
        lines.append('累计最佳差值变化（描述性）：'+('; '.join(f"q{r['query']} 差值{r['incumbent_delta']:.7g} (变化{r['delta_change']:+.7g})" for r in changed) or '无'))
    lines.extend(['','全部主动验证，以及实际选中点带有正混合权重的轮次：'])
    for e in events:
        lines.append(f"{e['variant']} {e['task']} q{e['query']} {e['mode']}: {e['selected_id']}, observed={e['selected_raw_value']:.8g}, gap={e['opportunity_gap']:.7g}, accepted_example={e['selected_is_any_accepted_live_plus']}, deployed_gain={e['actual_mixture_logscore']:.7g}")
        if e['verification']:
            lines.append('  validation='+json.dumps(e['verification'],ensure_ascii=False))
        for p in e['predictors']:
            pred,f=p['prediction'],p['feedback']
            lines.append(f"  {p['id']}v{p['version']}: ref={pred['reference_id']} (origin_ref_changed={p['actual_reference_differs_from_origin']}), GPmu={pred['gp_mean_y']:.7g}, GPvar={pred['gp_var_y']:.7g}, expert_t={pred['tempering']:.7g}, mixture_w={p['external_mixture_weight']:.7g}; score={f['delta_log_score']:.7g}, fact={f['fact_success']}, trust={f['trusted_weight_before']:.7g}->{f['trusted_weight_after']:.7g}")
            if p['external_mixture_weight']>0:
                lines.append('    prior_credit='+json.dumps(p['prior_feedback_chain'],ensure_ascii=False))
    lines.extend(['','完整冻结预测、反馈、逐轮曲线及同前缀门限复算见JSON。未查点的真实值从未被读取；不同单位的AUC不可跨任务直接加总。'])
    (REPORTS/'v6_mechanism_review.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(report=str(REPORTS/'v6_mechanism_review.json'),summary=result['summary']),ensure_ascii=False))


if __name__=='__main__':
    main()
