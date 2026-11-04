"""Report the prespecified V6 example-validation experiment; no oracle reads."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code/chemist_system'))
from chem.benchmarks import knowledge_study as study
from chem.benchmarks.knowledge_llm import digest


def metrics(state):
    assert state['status'] == 'COMPLETE' and len(state['history']) == 20
    assert len(state['decisions']) == 16
    assert len({r['candidate_id'] for r in state['history']}) == 20
    return study.summarize(state)


def main():
    p = study.check(ROOT, 'v6')
    folder, old = study.paths(ROOT, 'v6'), study.paths(ROOT, 'v5')
    rows, traces, equivalence = [], [], []
    for task in p['tasks']:
        def load(base, mode):
            return study.read(base/'runs'/f'{task}_{mode}_agent_initial_terra'/'state.json')
        bo, full, previous, old_bo = load(folder,'bo'), load(folder,'full'), load(old,'full'), load(old,'bo')
        assert bo['history'] == old_bo['history'], 'Matched BO changed'
        assert full['history'][:4] == bo['history'][:4] == previous['history'][:4]
        equivalence.append(dict(task=task, history_exact=True, cache_counters=bo['cache_counters']))
        a, b, prior = metrics(full), metrics(bo), metrics(previous)
        row = dict(task=task, BO_SearchAUC=b['SearchAUC'], V5_SearchAUC=prior['SearchAUC'],
            V6_SearchAUC=a['SearchAUC'], AUC_delta_vs_BO=a['SearchAUC']-b['SearchAUC'],
            AUC_delta_vs_V5=a['SearchAUC']-prior['SearchAUC'],
            AUC_relative_percent_vs_BO=100*(a['SearchAUC']-b['SearchAUC'])/b['SearchAUC'],
            BO_final_best=b['final_best'], V5_final_best=prior['final_best'], V6_final_best=a['final_best'],
            final_delta_vs_BO=a['final_best']-b['final_best'],
            final_delta_vs_V5=a['final_best']-prior['final_best'],
            BO_first_best_total=b['first_best_total'], V5_first_best_total=prior['first_best_total'],
            V6_first_best_total=a['first_best_total'], weighted_choices=a['knowledge_choices'],
            optional_verifications=a['optional_verifications'], calls=a['calls'])
        rows.append(row)
        for mode, state in [('bo',bo),('full',full)]:
            for t,d in enumerate(state['decisions'],1):
                c = study.read(folder/'runs'/f'{task}_{mode}_agent_initial_terra'/'rounds'/f'{t:02d}_commit.json')
                assert digest(c) == d['commit_sha256'] and c['history_sha256'] == digest(state['history'][:t+3])
                assert c['selected_id'] == state['history'][t+3]['candidate_id']
                assert c['opportunity_gap'] <= c['epsilon']+1e-12
                if c['mode'] == 'optional_verification':
                    proposal = c['verification']['proposal']
                    assert proposal['point_id'] == c['selected_id']
                    assert proposal in c['validation_proposals']
                    assert proposal['proposal_round'] <= t
                if mode == 'full' and c['mode'] != 'base_bo':
                    traces.append(dict(task=task,query=t,mode=c['mode'],point_id=c['selected_id'],
                        BO_reference_id=c['BO_reference_id'],opportunity_gap=c['opportunity_gap'],
                        raw_value=d['raw_value'],incumbent_before_y=d['incumbent_before_y'],
                        incumbent_after_y=d['incumbent_after_y'],verification=c['verification'],
                        evidence=d['evidence'],selected_mixture=c['selected_mixture']))
    def comparison(field):
        wins = sum(r[field] > 1e-9 for r in rows)
        losses = sum(r[field] < -1e-9 for r in rows)
        return dict(wins=wins,ties=len(rows)-wins-losses,losses=losses)
    against_bo, against_v5 = comparison('AUC_delta_vs_BO'), comparison('AUC_delta_vs_V5')
    weighted = sum(r['weighted_choices'] for r in rows)
    reports = ROOT/'experiments/reports/knowledge_loop_20261101'
    audit_path = reports/'v6_joint_audit.json'
    audited = False
    if audit_path.exists():
        audit = study.read(audit_path)
        audited = (audit['technical_result'] == 'PASS_LISTED_CHECKS' and not audit['failures']
            and not audit['pending'] and not audit['source_drift']
            and audit['totals']['runs'] == 16 and audit['totals']['rounds'] == 256
            and audit['protocol_sha256'] == p['protocol_sha256'])
    result = dict(variant='v6',protocol_sha256=p['protocol_sha256'],rows=rows,
        SearchAUC_vs_BO=against_bo,SearchAUC_vs_V5=against_v5,
        weighted_choices=weighted,optional_verifications=sum(r['optional_verifications'] for r in rows),
        final_best_losses_vs_BO=sum(r['final_delta_vs_BO'] < -1e-9 for r in rows),
        final_best_losses_vs_V5=sum(r['final_delta_vs_V5'] < -1e-9 for r in rows),
        majority_development_pass=against_bo['wins']>=5,
        candidate_gate=against_bo['wins']>=5 and weighted>0,
        independently_audited=audited,independent_confirmation=False,goal_achieved=False,
        changed_decisions=traces,BO_equivalence=equivalence,
        note='Frozen seed0 development. No oracle access. This report checks receipts but does not replace independent full-candidate/GP/ledger reconstruction.')
    path = reports/'v6_results.json'
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['V6：示例点主动验证，seed0开发结果',
        '同一候选池、初始化4点、追加16次、GP/qLogEI、Terra medium。仅主动验证候选集合改变。',
        '任务\tBO AUC\tV5 AUC\tV6 AUC\tV6-BO\tV6-V5\tBO最佳\tV5最佳\tV6最佳\t首达总轮(BO/V5/V6)\t加权/验证']
    for r in rows:
        lines.append(f"{r['task']}\t{r['BO_SearchAUC']:.6f}\t{r['V5_SearchAUC']:.6f}\t{r['V6_SearchAUC']:.6f}\t{r['AUC_delta_vs_BO']:+.6f}\t{r['AUC_delta_vs_V5']:+.6f}\t{r['BO_final_best']:.6f}\t{r['V5_final_best']:.6f}\t{r['V6_final_best']:.6f}\t{r['BO_first_best_total']}/{r['V5_first_best_total']}/{r['V6_first_best_total']}\t{r['weighted_choices']}/{r['optional_verifications']}")
    lines += ['',f'相对BO胜/平/负：{against_bo}；相对V5：{against_v5}。',
        f'加权改点{weighted}次；主动验证{result["optional_verifications"]}次；最终最佳较BO下降{result["final_best_losses_vs_BO"]}个任务。',
        f'开发多数性能通过：{result["majority_development_pass"]}；含实际加权的候选门：{result["candidate_gate"]}；全量独立审计：{audited}。',
        '不同任务目标单位不同，不跨单位平均原始AUC。首达轮次含4个初始化。',
        '这是开发对照，不是未见初始化确认；允许个别任务退化，完整保留失败。',
        'V6验证选择与提案plus一致，但实际最高可比已知reference仍可能不同；不是LLM独立优先级排序实验。']
    path.with_suffix('.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('SearchAUC_vs_BO','SearchAUC_vs_V5','weighted_choices','optional_verifications','candidate_gate','independently_audited')},ensure_ascii=False))


if __name__ == '__main__':
    main()
