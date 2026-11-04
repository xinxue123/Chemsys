"""V7 fixed-policy metrics, proposal support, and staged learning ablation."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/chemist_system'))
from chem.benchmarks import knowledge_study as study
from chem.benchmarks.knowledge_llm import digest


def main():
    p=study.check(ROOT,'v7');folder=study.paths(ROOT,'v7')
    reports=ROOT/'experiments/reports/knowledge_loop_20261101'
    states={};rows=[];interfaces=[];changed=[]
    for path in sorted((folder/'runs').glob('*/state.json')):
        state=study.read(path)
        assert state['status']=='COMPLETE' and len(state['history'])==20 and len(state['decisions'])==16
        assert len({x['candidate_id'] for x in state['history']})==20
        assert state['protocol_sha256']==p['protocol_sha256']
        states[state['task'],state['mode']]=state
        row=study.summarize(state);row['run']=path.relative_to(ROOT).as_posix()
        counts=Counter()
        for t,d in enumerate(state['decisions'],1):
            c=study.read(path.parent/'rounds'/f'{t:02d}_commit.json')
            assert d['commit_sha256']==digest(c) and c['history_sha256']==digest(state['history'][:t+3])
            assert c['selected_id']==d['selected_id']==state['history'][t+3]['candidate_id']
            assert c['opportunity_gap']<=c['epsilon']+1e-12
            counts[c['mode']]+=1
            if c['proposal'] is not None:
                call=c['calls_at_commit'][-1]
                context=study.read(folder/'requests'/call['context_sha256']/'context.json')
                assert digest(context)==call['context_sha256']
                support=context['public']['proposal_support']
                by_id={r['point_id']:r for r in support['candidate_support']}
                accepted=[r for r in c['reviews'] if r.get('accepted') and r.get('op') in ('add','revise')]
                flags=[by_id[c['proposal']['actions'][r['action_index']]['plus_id']]['passes_narrow_cost_gate'] for r in accepted]
                interfaces.append(dict(task=state['task'],mode=state['mode'],query=t,
                    optional_window_open=support['optional_window_open'],accepted_add_revise=len(accepted),
                    accepted_inside_narrow_gate=sum(flags),accepted_outside_narrow_gate=len(flags)-sum(flags),
                    displayed_candidates=len(by_id),with_known_reference=sum(bool(x['references']) for x in by_id.values()),
                    context_sha256=call['context_sha256']))
            if state['mode']=='full' and c['mode']!='base_bo':
                changed.append(dict(task=state['task'],query=t,mode=c['mode'],selected_id=c['selected_id'],
                    BO_reference_id=c['BO_reference_id'],opportunity_gap=c['opportunity_gap'],
                    raw_value=d['raw_value'],verification=c['verification'],evidence=d['evidence'],
                    mixture_logscore=d['deployed_mixture_log_score_gain']))
        row['selection_modes']=dict(counts);rows.append(row)
    assert all((task,mode) in states for task in p['tasks'] for mode in ('bo','full'))
    mapping={(r['task'],r['mode']):r for r in rows}
    for task in p['tasks']:
        b=mapping[task,'bo'];old=study.read(study.paths(ROOT,'v5')/'runs'/f'{task}_bo_agent_initial_terra'/'state.json')
        assert states[task,'bo']['history']==old['history']
        prior=study.summarize(study.read(study.paths(ROOT,'v5')/'runs'/f'{task}_full_agent_initial_terra'/'state.json'))
        for mode in ('full','verification'):
            if (task,mode) not in states:continue
            s=states[task,mode];r=mapping[task,mode]
            assert s['history'][:4]==states[task,'bo']['history'][:4]
            r.update(AUC_delta_vs_BO=r['SearchAUC']-b['SearchAUC'],best_delta_vs_BO=r['final_best']-b['final_best'],
                V5_SearchAUC=prior['SearchAUC'],AUC_delta_vs_V5=r['SearchAUC']-prior['SearchAUC'])
        if (task,'verification') in states:
            f,v=states[task,'full'],states[task,'verification'];r=mapping[task,'full']
            fc={x['query']:x['context_sha256'] for x in f['calls']};vc={x['query']:x['context_sha256'] for x in v['calls']}
            r.update(AUC_delta_vs_verification=r['SearchAUC']-mapping[task,'verification']['SearchAUC'],
                best_delta_vs_verification=r['final_best']-mapping[task,'verification']['final_best'],
                first_query_selection_difference=next((x['query'] for x,y in zip(f['decisions'],v['decisions']) if x['selected_id']!=y['selected_id']),None),
                first_query_mechanism_difference=next((x['query'] for x,y in zip(f['decisions'],v['decisions']) if x['mode']!=y['mode']),None),
                first_call_context_difference=next((t for t in sorted(fc.keys()|vc.keys()) if fc.get(t)!=vc.get(t)),None))
    arms={}
    for mode in ('full','verification'):
        arm=[r for r in rows if r['mode']==mode]
        wins=sum(r['AUC_delta_vs_BO']>1e-9 for r in arm);losses=sum(r['AUC_delta_vs_BO']<-1e-9 for r in arm)
        arms[mode]=dict(completed=len(arm),wins=wins,ties=len(arm)-wins-losses,losses=losses,
            final_best_losses=sum(r['best_delta_vs_BO']<-1e-9 for r in arm),
            weighted_choices=sum(r['knowledge_choices'] for r in arm),
            optional_verifications=sum(r['optional_verifications'] for r in arm))
    stage1=arms['full']['wins']>=5 and arms['full']['weighted_choices']>0
    learned=sum(r.get('AUC_delta_vs_verification',0)>1e-9 for r in rows if r['mode']=='full')
    stage2=arms['verification']['completed']==8
    ready=stage1 and stage2 and learned>0
    stage=('READY_FOR_FRESH_REGISTRATION' if ready else 'STAGE2_NO_LEARNING_ADVANTAGE' if stage2
           else 'STAGE1_PASS_VERIFICATION_DUE' if stage1 else 'STAGE1_REJECTED_STOP')
    audited=False;ap=reports/'v7_joint_audit.json'
    if ap.exists():
        a=study.read(ap)
        audited=(a['technical_result']=='PASS_LISTED_CHECKS' and not a['failures'] and not a['pending']
            and not a['source_drift'] and a['protocol_sha256']==p['protocol_sha256']
            and a['totals']['runs']==len(rows) and a['totals']['rounds']==16*len(rows))
    support_counts={}
    for mode in ('full','verification'):
        calls=[x for x in interfaces if x['mode']==mode];open_calls=[x for x in calls if x['optional_window_open']]
        support_counts[mode]=dict(calls=len(calls),accepted_add_revise=sum(x['accepted_add_revise'] for x in calls),
            accepted_outside_narrow_gate=sum(x['accepted_outside_narrow_gate'] for x in calls),
            open_window_accepted=sum(x['accepted_add_revise'] for x in open_calls),
            open_window_outside_narrow=sum(x['accepted_outside_narrow_gate'] for x in open_calls))
    result=dict(variant='v7',protocol_sha256=p['protocol_sha256'],rows=rows,arms=arms,stage=stage,
        stage1_pass=stage1,stage2_complete=stage2,full_AUC_wins_vs_verification=learned,
        ready_for_fresh_registration=ready and audited,independently_audited=audited,
        independent_confirmation=False,goal_achieved=False,proposal_support=support_counts,
        interfaces=interfaces,changed_decisions=changed,
        note='Constraint-informed proposal interface, not GP-independent recommendations. All seed0 development; old seed1/2 are already viewed. No per-case policy selection.')
    output=reports/'v7_results.json';output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['V7执行约束提示接口：seed0开发结果',
        '同一候选集、初始化、GP/native qLogEI、4+16；Terra medium。仅LLM输入/提示改变。',
        '任务\t方法\tSearchAUC\t最终最佳\t首达总轮次\tAUC较BO\t最佳较BO\tAUC较V5\t加权/验证']
    for task in p['tasks']:
        for mode in ('bo','full','verification'):
            if (task,mode) not in mapping:continue
            r=mapping[task,mode]
            lines.append(f"{task}\t{mode}\t{r['SearchAUC']:.6f}\t{r['final_best']:.6f}\t{r['first_best_total']}\t{r.get('AUC_delta_vs_BO',0):+.6f}\t{r.get('best_delta_vs_BO',0):+.6f}\t{r.get('AUC_delta_vs_V5',0):+.6f}\t{r['knowledge_choices']}/{r['optional_verifications']}")
    lines+=['',f'阶段：{stage}；独立全量审计：{audited}。',
        'full统计：'+json.dumps(arms['full'],ensure_ascii=False),
        'verification统计：'+json.dumps(arms['verification'],ensure_ascii=False),
        '提案接口统计：'+json.dumps(support_counts,ensure_ascii=False),
        f'full较verification的AUC获益任务数：{learned}；未完成该臂时不可解释为零学习效应。',
        '主门不足5/8严格胜出或无加权改点时，按预声明停止本候选，不补验证控制。',
        '不同任务目标单位不同，不跨单位平均。首达轮次含4个初始化；全部结果保留。',
        '同观测历史但验证额度不同可能导致窗口flag和后续LLM提案不同，比较的是整个学习政策的轨迹效应。']
    output.with_suffix('.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(stage=stage,arms=arms,ready_for_registration=result['ready_for_fresh_registration'],audit=audited),ensure_ascii=False))


if __name__=='__main__':
    main()
