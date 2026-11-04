"""No-outcome seed2 induction for JS ranking, and first divergence of JS gate.

A keeps every V5 gate and changes only optional ranking to delta-JS.
B additionally replaces delta-sign-MI>=.01 by delta-JS>max(1e-12,errx+errBO).
Both are fixed hypothetical policy definitions. No new trajectory is executed.
"""
from collections import Counter
import json
from math import log
from pathlib import Path
import numpy as np
import knowledge_optional_density_review as d

a=d.a;ROOT=d.ROOT;OUT=d.OUT


def density(arrays,data,j):
    if j not in data['positions']:return dict(JS=0.,quadrature_error_estimate=0.)
    p=data['positions'][j]
    return d.density_js_receipt(float(arrays['mu_y'][j]),float(arrays['sd_predictive_y'][j]**2),
        float(data['probes'][j]['reference_y']),float(data['values']['log_tempering'][p]))


def candidates(pool,commit,prefix,arrays,rows,calc,include_B):
    bo=pool['ix'][commit['BO_reference_id']];base=arrays['base_logei'];ties=arrays['tie_priority']
    original=[];broader=[]
    for row in rows:
        key=a.hkey(row)
        if row['weight']>0 or not row['can_change_rank_at_cap'] or key not in calc:continue
        h=next(h for h in commit['hypotheses_at_commit'] if a.hkey(h)==key);e=a.evidence(h)
        if e['effective_prediction_score']<0 or e['failures']>e['successes']:continue
        data=calc[key];bp=data['positions'].get(bo)
        bmi=float(data['values']['sign_information'][bp]) if bp is not None else 0.
        bjs=density(arrays,data,bo)
        for pos,j in enumerate(data['indices']):
            j=int(j);gap=float(base[bo]-base[j]);mi=float(data['values']['sign_information'][pos]);delta=mi-bmi
            if j==bo or gap>min(commit['epsilon'],log(1.1)) or data['values']['offset'][pos]<=a.ATOL:continue
            if not include_B and delta<.01:continue
            js=density(arrays,data,j);difference=js['JS']-bjs['JS']
            error=js['quadrature_error_estimate']+bjs['quadrature_error_estimate'];threshold=max(1e-12,error)
            row_data=dict(point_id=pool['ids'][j],rule=list(key),reference_id=data['probes'][j]['reference_id'],
                gap=gap,sign_MI=mi,BO_sign_MI=bmi,delta_sign_MI=delta,JS=js['JS'],BO_JS=bjs['JS'],
                JS_error_estimate=js['quadrature_error_estimate'],BO_JS_error_estimate=bjs['quadrature_error_estimate'],
                delta_JS=difference,numerical_threshold=threshold,JS_numerical_gate=difference>threshold,
                original_MI_gate=delta>=.01,tie_priority=int(ties[j]))
            if delta>=.01:original.append(row_data)
            if include_B:broader.append(row_data)
    return original,broader


def winner(rows,metric):
    return max(rows,key=lambda x:(x[metric],-x['gap'],-x['tie_priority'])) if rows else None


def main():
    folder=ROOT/'experiments/runs/knowledge_loop_20261101/v5_seed2_confirm';protocol=a.read(folder/'protocol.json')
    assert protocol['protocol_sha256']==a.digest({k:v for k,v in protocol.items() if k!='protocol_sha256'})
    rounds=[];tasks=[];first_B=[]
    for item in protocol['inputs']:
        pool=a.common.public_pool(ROOT,item);run=folder/'runs'/f"{item['task']}_full_agent_initial_terra"
        state=a.read(run/'state.json');assert state['status']=='COMPLETE' and len(state['decisions'])==16
        prior_equal=True;B_equal=True;optional_used=0;B_diff=None
        for t in range(1,17):
            c=a.read(run/'rounds'/f'{t:02d}_commit.json');prefix=state['history'][:t+3]
            assert c['query']==t and c['history_sha256']==a.digest(prefix)
            with np.load(run/'rounds'/f'{t:02d}_GP.npz',allow_pickle=False) as f:arrays={k:f[k].copy() for k in f.files}
            rows,calc=a.calculations(pool,c['hypotheses_at_commit'],prefix,arrays)
            bo=pool['ix'][c['BO_reference_id']]
            pre=d.score_rank(arrays['base_logei'],calc,rows,None,0.,arrays['eligible'],arrays['tie_priority'])
            record=dict(task=item['task'],query=t,prior_A_prefix_equal=prior_equal,prefix_sha256=c['history_sha256'],
                commit_sha256=a.digest(c),GP_fit_key=c['GP_fit_key'],original_point=c['selected_id'],original_mode=c['mode'],
                weighted_preselection=pool['ids'][pre],optional_used_before=optional_used,
                exact_context_hashes=[r['context_sha256'] for r in c['calls_at_commit'] if r['query']==t])
            old=[];broader=[]
            if pre!=bo:
                reason='weighted_choice_precedes_optional';A_point=pool['ids'][pre];A_mode='knowledge_guided'
            elif t>10 or optional_used>=1:
                reason='optional_time_or_quota_closed';A_point=pool['ids'][bo];A_mode='base_bo'
            else:
                old,broader=candidates(pool,c,prefix,arrays,rows,calc,B_equal)
                mi=winner(old,'delta_sign_MI');js=winner(old,'delta_JS')
                if mi:
                    assert mi['point_id']==c['selected_id'] and tuple(mi['rule'])==a.hkey(c['verification'])
                    reason='optional_ranking_active';A_point=js['point_id'];A_mode='optional_verification'
                else:
                    reason='original_optional_candidate_set_empty';A_point=pool['ids'][bo];A_mode='base_bo'
                record.update(original_gated_count=len(old),A_MI_winner=mi,A_JS_winner=js)
            equal=A_point==c['selected_id'] and A_mode==c['mode']
            record.update(A_point=A_point,A_mode=A_mode,A_equal=equal,ranking_reason=reason)
            assert equal,(item['task'],t,record)
            prior_equal=prior_equal and equal
            if B_equal:
                if reason in ('weighted_choice_precedes_optional','optional_time_or_quota_closed'):
                    B_point=A_point;B_mode=A_mode;bj=None
                else:
                    allowed=[r for r in broader if r['JS_numerical_gate']]
                    bj=winner(allowed,'delta_JS')
                    B_point=bj['point_id'] if bj else pool['ids'][bo]
                    B_mode='optional_verification' if bj else 'base_bo'
                    record.update(B_common_candidate_count=len(broader),B_JS_gated_count=len(allowed),
                        B_added_candidates=[r for r in allowed if not r['original_MI_gate']],B_all_common_candidates=broader)
                record.update(B_point=B_point,B_mode=B_mode,B_winner=bj,
                    B_equal=B_point==c['selected_id'] and B_mode==c['mode'])
                if not record['B_equal']:
                    B_diff=dict(task=item['task'],query=t,old_point=c['selected_id'],old_mode=c['mode'],
                        new_point=B_point,new_mode=B_mode,old_winner=record.get('A_MI_winner'),new_winner=bj,
                        prefix_sha256=c['history_sha256'],GP_fit_key=c['GP_fit_key'],
                        new_candidate_failed_only_original_MI_gate=bool(bj and not bj['original_MI_gate']),
                        interpretation='Only this first same-prefix policy divergence is evaluated; no later hypothetical trajectory or outcomes.')
                    B_equal=False;first_B.append(B_diff)
            else:record['B_not_evaluated']='Earlier B divergence; saved original prefix is not a hypothetical B prefix.'
            optional_used+=int(c['mode']=='optional_verification')
            rounds.append(record)
        tasks.append(dict(task=item['task'],A_all16_same=prior_equal,A_same_learning_state_and_observed_trajectory_conditional_on_same_cached_responses=prior_equal,
            B_first_divergence=B_diff,B_all16_same=B_equal))
    # This already published aggregate classification is used only for the
    # task-count upper bound, never to choose a point, gate, threshold or rank.
    published_path=OUT/'v5_seed2_confirm_results.json';published=a.read(published_path)
    assert published['protocol_sha256']==protocol['protocol_sha256']
    published_rows={(r['task'],r['mode']):r for r in published['rows']}
    for task in tasks:
        name=task['task'];old_auc=published_rows[name,'full']['SearchAUC'];bo_auc=published_rows[name,'bo']['SearchAUC']
        delta=old_auc-bo_auc
        task['published_original_AUC']=old_auc;task['published_BO_AUC']=bo_auc
        task['published_original_AUC_delta']=delta
        task['published_original_class']='win' if delta>1e-9 else ('loss' if delta<-1e-9 else 'tie')
    known_counts=dict(Counter(t['published_original_class'] for t in tasks))
    unchanged_wins=sum(t['B_all16_same'] and t['published_original_class']=='win' for t in tasks)
    upper=unchanged_wins+len(first_B)
    assert known_counts=={'win':3,'tie':3,'loss':2} and upper==4
    report=dict(development_diagnostic=True,script_sha256=a.filehash(__file__),math_script_sha256=a.filehash(d.__file__),
        protocol_sha256=protocol['protocol_sha256'],oracle_files_opened=0,GP_fits=0,LLM_calls=0,new_optimization_runs=0,
        current_round_outcomes_used=False,
        A_definition='Keep original delta-sign-MI>=.01 gate and all other V5 gates; rank original optional candidates by delta-JS, then original gap/tie.',
        A_proof_assumptions=['Same initialization and numerical GP recipe; same prior observed points imply the frozen stored GP and rule ledger.',
            'All non-ranking math, feedback, reference selection, event scheduling and menu are unchanged.',
            'No JS metadata enters the public payload or learning ledger; identical full request contexts reuse the same frozen response cache.',
            'Same chosen point plus unchanged frozen predictor yields the same observation/feedback/next learning state; observations need not be reread for this induction.',
            'Protocol hashes, diagnostic scores and bookkeeping receipts are not claimed byte-identical across hypothetical variants.'],
        A_result='ALL_8_TASKS_128_QUERIES_IDENTICAL_UNDER_STATED_ISOLATED_CHANGE',
        A_skip_reasons=dict(Counter(r['ranking_reason'] for r in rounds)),
        B_definition='Same as A except replace delta-sign-MI>=.01 with delta-JS>max(1e-12,err_point+err_BO); rank by delta-JS, then original gap/tie.',
        B_numerical_error_note='QUADPACK error estimates, not formally certified absolute bounds; fixed numerical floor/tolerances are independent of outcomes.',
        B_tasks_with_first_divergence=len(first_B),B_first_divergences=first_B,tasks=tasks,rounds=rounds,
        published_seed2_classification=dict(source_path=published_path.relative_to(ROOT).as_posix(),source_sha256=a.filehash(published_path),
            counts=known_counts,winners=[t['task'] for t in tasks if t['published_original_class']=='win'],
            role='Already known aggregate metrics used only to bound possible winning-task count, never candidate selection.'),
        B_conditional_winning_task_upper_bound=dict(unchanged_winning_tasks=unchanged_wins,changed_tasks=len(first_B),upper_bound=upper,
            required_majority=5,can_meet_majority=False,
            conditions='Same frozen own-prefix LLM full-context cache, unchanged GP/feedback/prompt/event scheduling, and isolated fixed B gate+rank. Seven tasks provably keep their existing point sequence; only originally tied Case2 may diverge.',
            interpretation='Even granting changed Case2 a hypothetical win gives at most4/8. No Case2 outcome, new trajectory AUC or improvement probability is predicted.'),
        conclusions=['A cannot fix the already observed seed2 performance failure: every selected point is identical under the explicit coupling/cache assumptions.',
            'B first divergences only establish a nontrivial mechanism change. No selected hypothetical point outcome or AUC is consulted.',
            'With seven task histories fixed and only originally tied Case2 changing, B has a conditional upper bound of4/8 wins and cannot reach the5/8 target.',
            'No outcome-driven threshold fitting, posterior GP fitting, or post-divergence trajectory continuation was performed.'])
    (OUT/'seed2_JS_prefix_induction_review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['Seed2 JS-only归纳与JS门首分歧（prefix-only）',report['A_result'],
        'A仅换排序：128轮全部原选点/模式一致；依赖完全相同context复用原LLM响应、JS元数据不进入prompt/信用账本。',
        'B固定为JS门+JS排序；门限DeltaJS>max(1e-12,errx+errBO)，其余门原样。不计算分歧后轨迹，不读取本轮真实结果。',
        'A各跳过/排名原因：'+json.dumps(report['A_skip_reasons'],ensure_ascii=False),'',
        '已发表seed2原分类：'+json.dumps(known_counts,ensure_ascii=False),
        '在同context复用冻结LLM缓存及其他机制不变条件下，B仅原持平的Case2可变，最多4/8胜；这不是对Case2新轨迹AUC的预测。','',
        'task | original class | A all16 identical | B first divergence | old->new | new DeltaMI/DeltaJS | numerical threshold']
    for task in tasks:
        b=task['B_first_divergence'];v=b['new_winner'] if b else None
        lines.append(f"{task['task']} | {task['published_original_class']} | {task['A_all16_same']} | {b['query'] if b else '无'} | {(b['old_point']+'->'+b['new_point']) if b else '全部同轨'} | {(str(v['delta_sign_MI'])+'/'+str(v['delta_JS'])) if v else '-'} | {v['numerical_threshold'] if v else '-'}")
    lines.extend(['','本报告无任何新增真实查询/拟合/API，B改点不代表改进；不得据未查候选的真实值筛选策略。'])
    (OUT/'seed2_JS_prefix_induction_review.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(A=report['A_result'],reasons=report['A_skip_reasons'],B_tasks=len(first_B),first_B=first_B),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
