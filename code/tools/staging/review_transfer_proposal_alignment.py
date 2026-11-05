"""Development-only, prefix-only menu/proposal alignment review of frozen V5.

No oracle files, new evaluations, API calls or fitting. Counts use audited saved
GP marginals and independent rule geometry/math, never a future outcome.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
from math import log
from collections import Counter
import numpy as np
import audit_knowledge_transfer_loop as a

ROOT = a.ROOT
OUT = ROOT / 'experiments/reports/knowledge_loop_20261101'


def possible_references(pool, indices, prefix):
    """Existence of SOME legal 1-2-axis hypothesis, not a chemical judgment.

    Every changed category and continuous separation >.25 must be a focus;
    at most two such axes are allowed. If zero are required, at least one axis
    must still differ so a directed one-axis rule can be stated.
    """
    features = np.asarray(pool['features'], dtype=object)
    cand = features[indices]
    refs = [[] for _ in indices]
    for row in sorted(prefix, key=lambda r: (-r['y'], pool['features'][pool['ix'][r['candidate_id']]])):
        ref = features[pool['ix'][row['candidate_id']]]
        mandatory = np.zeros(len(indices), int)
        different = np.zeros(len(indices), bool)
        for k, kind in enumerate(pool['kinds']):
            if kind == 'categorical':
                delta = cand[:, k] != ref[k]
                mandatory += delta
                different |= delta
            else:
                delta = np.abs(np.asarray(cand[:, k], float)-float(ref[k]))
                mandatory += delta > .25+a.ATOL
                different |= delta > 0
        for pos in np.flatnonzero((mandatory <= 2) & different):
            refs[int(pos)].append(row)
    return refs


def candidate_counts(pool, arrays, prefix, menu_ids, bo, epsilon):
    all_js = np.flatnonzero(arrays['eligible'])
    legal = possible_references(pool, all_js, prefix)
    compatible = {int(j): refs for j, refs in zip(all_js, legal)}
    menu = [pool['ix'][i] for i in menu_ids]
    narrow = {int(j) for j in all_js if arrays['base_logei'][bo]-arrays['base_logei'][j] <= min(epsilon, log(1.1))}
    viable = {j for j in narrow if compatible[j]}
    detail = []
    for j in menu:
        refs = compatible[j]
        # Absolute MI is only a necessary upper-screen. The real incremental
        # gate subtracts the BO point's information under the SAME new rule.
        vals = a.math_fields(np.full(len(refs), arrays['mu_y'][j]),
            np.full(len(refs), arrays['sd_y'][j]**2),
            np.full(len(refs), arrays['sd_predictive_y'][j]**2),
            np.array([r['y'] for r in refs]), max(x['y'] for x in prefix)) if refs else None
        info = vals['sign_information'] if vals is not None else []
        detail.append(dict(point_id=pool['ids'][j], is_BO=j==bo,
            gap=float(arrays['base_logei'][bo]-arrays['base_logei'][j]), narrow_gap=j in narrow,
            reference_ids=[r['candidate_id'] for r in refs], legal_reference_count=len(refs),
            highest_geometrically_compatible_reference=refs[0]['candidate_id'] if refs else None,
            highest_reference_sign_information=float(info[0]) if refs else None,
            maximum_reference_sign_information=float(max(info)) if refs else None,
            any_reference_absolute_MI_at_least_001=bool(len(info) and max(info)>=.01)))
    return dict(menu_count=len(menu), menu_narrow_gap=sum(j in narrow for j in menu),
        menu_narrow_comparable=sum(j in viable for j in menu),
        menu_narrow_comparable_nonBO=sum(j in viable and j != bo for j in menu),
        pool_narrow_comparable=len(viable), pool_narrow_comparable_nonBO=len(viable-{bo}),
        menu_narrow_comparable_nonBO_absolute_MI=sum(d['narrow_gap'] and not d['is_BO'] and d['any_reference_absolute_MI_at_least_001'] for d in detail),
        menu_candidates=detail)


def origin_review(pool, h, row, data, prefix, arrays, commit, verification_used):
    bo = pool['ix'][commit['BO_reference_id']]
    j = pool['ix'][h['plus_id']]
    gap = float(arrays['base_logei'][bo]-arrays['base_logei'][j])
    ev = a.evidence(h)
    result = dict(id=h['id'], version=h['version'], created_round=h['created_round'],
        origin_plus=h['plus_id'], origin_minus=h['minus_id'], gap=gap, origin_is_BO=j==bo,
        can_change_rank_at_cap=row['can_change_rank_at_cap'],
        effective_prediction_score=ev['effective_prediction_score'], successes=ev['successes'], failures=ev['failures'],
        weight=row['weight'], origin_executable=bool(data and j in data['probes']),
        selected_is_origin=commit['selected_id']==h['plus_id'],
        gates=dict(before_cutoff=commit['query']<=10, quota_available=verification_used<1,
            zero_weight=row['weight']==0, rank_at_cap=row['can_change_rank_at_cap'],
            counterevidence_acceptable=ev['effective_prediction_score']>=0 and ev['failures']<=ev['successes'],
            not_BO=j!=bo, gap=gap<=min(commit['epsilon'],log(1.1)),
            incremental_information=False, positive_offset=False, executable=False))
    if data and j in data['probes']:
        pos=data['positions'][j]
        bp=data['positions'].get(bo)
        info=float(data['values']['sign_information'][pos])
        bo_info=float(data['values']['sign_information'][bp]) if bp is not None else 0.
        result.update(actual_reference_id=data['probes'][j]['reference_id'],
            reference_changed_from_example=data['probes'][j]['reference_id']!=h['minus_id'],
            sign_information=info, same_rule_BO_information=bo_info, incremental_information=info-bo_info,
            offset=float(data['values']['offset'][pos]))
        result['gates'].update(executable=True, incremental_information=info-bo_info>=.01,
            positive_offset=bool(data['values']['offset'][pos]>a.ATOL))
    result['all_origin_cold_start_gates']=bool(all(result['gates'].values()))
    result['failed_gates']=[k for k,v in result['gates'].items() if not v]
    return result


def main():
    calls, optional = [], []
    source_hashes = {}
    for variant in ('v5', 'v5_seed2_confirm'):
        folder=ROOT/'experiments/runs/knowledge_loop_20261101'/variant
        protocol=a.read(folder/'protocol.json')
        source_hashes[variant]=dict(protocol=protocol['protocol_sha256'])
        pools={m['task']: a.common.public_pool(ROOT,m) for m in protocol['inputs']}
        for path in sorted((folder/'runs').glob('*_full_agent_initial_terra/state.json')):
            state=a.read(path); pool=pools[state['task']]
            for t in sorted(set(c['query'] for c in state['calls']) | set(d['query'] for d in state['decisions'] if d['mode']=='optional_verification')):
                cp=path.parent/'rounds'/f'{t:02d}_commit.json'
                c=a.read(cp);prefix=state['history'][:t+3]
                assert a.digest(prefix)==c['history_sha256']
                with np.load(cp.with_name(f'{t:02d}_GP.npz'),allow_pickle=False) as z:
                    arrays={k:z[k].copy() for k in z.files}
                assert set(arrays)=={'mu_y','sd_y','sd_predictive_y','base_logei','available','eligible','tie_priority'}
                bo=pool['ix'][c['BO_reference_id']]
                used=sum(d['mode']=='optional_verification' for d in state['decisions'][:t-1])
                rows,calc=a.calculations(pool,c['hypotheses_at_commit'],prefix,arrays)
                rowmap={a.hkey(r):r for r in rows}
                new=[]
                for review in c['reviews']:
                    if review.get('accepted') and review.get('op') in ('add','revise'):
                        key=(review['id'],review['version'])
                        h=next(h for h in c['hypotheses_at_commit'] if a.hkey(h)==key)
                        new.append(origin_review(pool,h,rowmap[key],calc.get(key),prefix,arrays,c,used))
                if any(call['query']==t for call in state['calls']):
                    call=next(call for call in state['calls'] if call['query']==t)
                    public=a.read(folder/'requests'/call['context_sha256']/'context.json')['public']
                    assert [r['id'] for r in public['candidate_menu']]==c['menu_ids']
                    counts=candidate_counts(pool,arrays,prefix,c['menu_ids'],bo,c['epsilon'])
                    calls.append(dict(variant=variant,task=state['task'],query=t,
                        verification_used=used,verification_round_open=t<=10 and used<1,
                        selected_mode=c['mode'],selected_id=c['selected_id'],BO_reference_id=c['BO_reference_id'],
                        context_sha256=call['context_sha256'],commit_sha256=a.digest(c),
                        action_count=len(c['proposal']['actions']),new_accepted_rules=new,
                        rejected_actions=[r for r in c['reviews'] if r.get('accepted') is False],**counts))
                if c['mode']=='optional_verification':
                    key=a.hkey(c['verification']);h=next(h for h in c['hypotheses_at_commit'] if a.hkey(h)==key)
                    origin=origin_review(pool,h,rowmap[key],calc.get(key),prefix,arrays,c,used)
                    js=pool['ix'][c['selected_id']];p=calc[key]['positions'][js]
                    optional.append(dict(variant=variant,task=state['task'],query=t,selected_id=c['selected_id'],
                        origin=origin,selected_in_current_menu=c['selected_id'] in c['menu_ids'],
                        LLM_called_this_round=bool(c['menu_ids']),
                        selected_in_rule_creation_menu=c['selected_id'] in a.read(path.parent/'rounds'/f"{h['created_round']:02d}_commit.json")['menu_ids'],
                        selected_reference_id=calc[key]['probes'][js]['reference_id'],
                        selected_information=float(calc[key]['values']['sign_information'][p]),
                        selected_incremental_information=c['verification']['incremental_information']))
    summary=[]
    for variant in ('v5','v5_seed2_confirm'):
        cc=[r for r in calls if r['variant']==variant]
        oo=[r for r in optional if r['variant']==variant]
        nn=[n for r in cc for n in r['new_accepted_rules']]
        early=[r for r in cc if r['verification_round_open']]
        early_new=[n for r in early for n in r['new_accepted_rules']]
        summary.append(dict(variant=variant,calls=len(cc),menus_total=sum(c['menu_count'] for c in cc),
            calls_without_narrow_comparable_nonBO=sum(c['menu_narrow_comparable_nonBO']==0 for c in cc),
            early_unused_calls=len(early),early_without_narrow_comparable_nonBO=sum(c['menu_narrow_comparable_nonBO']==0 for c in early),
            early_menu_miss_but_pool_has_candidate=sum(c['menu_narrow_comparable_nonBO']==0 and c['pool_narrow_comparable_nonBO']>0 for c in early),
            early_without_absolute_MI_candidate=sum(c['menu_narrow_comparable_nonBO_absolute_MI']==0 for c in early),
            early_accepted_rules=len(early_new),
            early_origin_failure_gate_counts=dict(Counter(k for n in early_new for k in n['failed_gates'])),
            accepted_new_rules=len(nn),origin_meets_all_gates=sum(n['all_origin_cold_start_gates'] for n in nn),
            origin_reference_reanchored=sum(n.get('reference_changed_from_example',False) for n in nn),
            origin_failure_gate_counts=dict(Counter(k for n in nn for k in n['failed_gates'])),
            optional_count=len(oo),optional_selected_origin=sum(o['origin']['selected_is_origin'] for o in oo),
            optional_selected_in_creation_menu=sum(o['selected_in_rule_creation_menu'] for o in oo)))
    result=dict(development_only=True,source_protocols=source_hashes,oracle_reads=0,new_experiments=0,fits=0,LLM_calls=0,
        candidate_geometry_definition='exists legal one-or-two focus axes relative to own observed reference; not a chemistry endorsement or a rule-specific MI gate',
        limitations=['Each call uses only its own committed prefix and saved GP. Counts do not forecast counterfactual trajectory performance.',
            'Absolute sign MI >=.01 is necessary, not sufficient for the same-rule incremental MI >=.01 requirement.',
            'After q10 or after optional quota is consumed, a narrow menu cannot trigger current cold-start verification.',
            'This is postconfirmation development diagnosis, not a new independent confirmation.'],
        summary=summary,calls=calls,optional=optional)
    target=OUT/'v5_menu_proposal_alignment_review.json'
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['V5 主方案 full：LLM 菜单与主动验证目标对齐的只读诊断',
        '范围：seed0 开发与 seed2 确认各8任务、各32次LLM调用；未纳入seed0 verification消融重复菜单。',
        '仅读取已查询history的本轮前缀、候选物理特征、已审计GP数组和commit；未拟合、未调用API、未查询新产率。',
        '',
        '定义与边界：',
        'N = 菜单中非BO候选，满足 gap<=min(epsilon,log1.1)，并存在一个自身已观测reference，能构成某个合法1–2 focus轴比较。',
        '几何可比：所有改变的类别及连续差>.25的维度必须属于focus，至多两个；至少一个维度不同。方向/范围尚待LLM明确。',
        'I = N中至少有一个合法reference使绝对sign MI>=.01的候选数；这只是必要条件，不能替代该规则的MI(plus)-MI(BO)>=.01。',
        'P = 全部当前eligible候选中的N计数；M=菜单总点数；O=时间<=10且主动验证额度未用。',
        'H/G = 本次接受新规则或修订的数量 / 原plus通过所有既有冷启动门限的数量。',
        '不以未知产率比较候选，不把菜单覆盖或原plus合法性当作效能改善证据。',
        '',
        '核心结论：',
        '1. 冷启动仍开放时，seed0 10/11次、seed2 12/16次菜单已有N>0。菜单普遍完全缺可验证候选的解释不成立。',
        '2. 只有seed0 Doyle q1是菜单N=0但全池P>0。seed2四次N=0时P也为0，改菜单不能在不放宽门限时造出机会。',
        '3. seed2 Case2 q1 N=11但I=1；q4 N=1但I=0。仅把ref列给LLM不能保证MI门限能过。',
        '4. 12次主动验证有6次最终不是LLM规则origin plus；其中4次origin本来通过全部门限，软件只是更偏好其他点的MI。',
        '5. 另外2次origin确有门限问题：seed0 Case4 q1 gap不通过，seed0 SnAr q1增量MI不通过。原点保真会失去这两条旧轨迹的现有主动验证，不能宣称必然提升。',
        '',
        '本轮选定的最小V6隔离：只限制optional的候选为已接受LLM真实示例plus；菜单、SYSTEM、门限、同信息量排序、最高可比ref及scope加权部署保持V5。新模块已通过12项合成检查，尚无新实验性能结论。',
        '这属于可归因性的机制限制，不是修复已证明算错的V5算法。duplicate_semantics新plus应写独立validation_proposals，不改旧rule origin/ledger；已查询、撤回、语义不匹配或当前不可执行的提议仅失效，不暗中换点。',
        '',
        '较大接口修法建议（未纳入本轮V6隔离、尚未实施或实验）：',
        'A. 在固定原门限内，展示可验证的未查plus及自身已测可比reference/可比较轴，不展示GP数值、MI排序或BO标签。候选菜单可从全eligible集合构造，避免仅过滤旧菜单导致Doyle q1漏点；固定物理排序/随机种子并记录选择规则。',
        'B. LLM显式给proposal_rank和预期plus>ref的局部命题；rank是化学试验优先级，不是后验EI。先记录提议，再由GP按原gap、同rule增量MI、机会门、反证和预算审查实际plus。',
        'C. 审查依LLM顺序选择首个通过者，全部失败即BO；不再在scope中用全局maxMI替换为LLM未提议点。保留plus/ref、被拒门限、审查顺序和原BO。',
        'D. 参考值仍须是该实际scope/focus内最高已知可比reference。不能允许LLM挑低产率ref以人为抬高事实成功率；若系统需重锚必须在预测冻结前明确记录，不可静默更换。',
        'E. rule/version与probe分离、负事实与负预测分数的适用继承、正信用不跨新语义继承、范围事前冻结、双账权重/平滑恢复/最多一次验证均保持。新plus不等于新rule。',
        '',
        '保真和归因风险：',
        '菜单即GP筛选，所以LLM仅在GP风险约束内独立排序，不能声称独立于GP的全空间提议；未显示数值不等于无GP选择偏差。',
        'focus/scope决定可比reference及同rule BO信息，必须先冻结再审查；用宽泛几何预筛的MI不能替代最终规则信息。',
        'MI优先目标是辨认规则而非最大化当轮收益；换成LLM排序只是换偏好，不能先验保证SearchAUC提升。',
        '候选菜单与排序执行同时变化的全轨迹比较只能证明组合效应。若要独立归因，至少在同菜单比较旧maxMI选择与忠实proposal选择；同前缀只读计数不是性能反事实。',
        'seed0与seed2现均已用于诊断，任何新版本都不能把它们再称为未见确认；禁止根据case产率选不同规则或放宽门限。',
        '',
        '全部64次调用：variant | task | q | O | M | N | I | P | H/G']
    for c in calls:
        passes=sum(n['all_origin_cold_start_gates'] for n in c['new_accepted_rules'])
        lines.append(f"{c['variant']} | {c['task']} | {c['query']} | {int(c['verification_round_open'])} | {c['menu_count']} | {c['menu_narrow_comparable_nonBO']} | {c['menu_narrow_comparable_nonBO_absolute_MI']} | {c['pool_narrow_comparable_nonBO']} | {len(c['new_accepted_rules'])}/{passes}")
    lines.extend(['','实际主动验证：variant | task/q | origin->selected | origin DeltaMI | selected DeltaMI | origin失败门限'])
    for o in optional:
        z=o['origin']
        lines.append(f"{o['variant']} | {o['task']}/{o['query']} | {z['origin_plus']}->{o['selected_id']} | {z.get('incremental_information')} | {o['selected_incremental_information']} | {','.join(z['failed_gates']) or '无'}")
    lines.extend(['','逐候选ref列表、全部新规则原plus门限与哈希见同名JSON。脚本仅为开发性诊断，不修改冻结源码。'])
    (OUT/'v5_menu_proposal_alignment_review.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(report=str(target),summary=summary),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
