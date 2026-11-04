"""Complete V8 trajectory results; never select the best version per task."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/chemist_system'))
from chem.benchmarks import knowledge_study as study
from chem.benchmarks.knowledge_llm import digest


def registered_run(state,path,folder,protocol,seen):
    key=(state['task'],state['mode'])
    expected=folder/'runs'/f"{state['task']}_{state['mode']}_agent_initial_terra"/'state.json'
    if (key in seen or state['task'] not in protocol['tasks'] or state['mode'] not in protocol['supported_modes']
            or state.get('condition')!='agent_initial_terra' or state.get('seed')!=0
            or path.resolve()!=expected.resolve()):raise ValueError('Duplicate or unregistered trajectory')
    return key


def main():
    protocol=study.check(ROOT,'v8');folder=study.paths(ROOT,'v8')
    reports=ROOT/'experiments/reports/knowledge_loop_20261101'
    states={};rows=[];changes=[];diagnostics=[]
    for path in sorted((folder/'runs').glob('*/state.json')):
        state=study.read(path);key=registered_run(state,path,folder,protocol,states);states[key]=state
        if (state['status']!='COMPLETE' or len(state['history'])!=20 or len(state['decisions'])!=16
                or len({h['candidate_id'] for h in state['history']})!=20):raise ValueError('Incomplete trajectory: '+str(key))
        if state['protocol_sha256']!=protocol['protocol_sha256']:raise ValueError('Wrong protocol')
        row=study.summarize(state);row['run']=path.relative_to(ROOT).as_posix()
        counter=Counter();forecast_gains=[];weights=[]
        for t,d in enumerate(state['decisions'],1):
            c=study.read(path.parent/'rounds'/f'{t:02d}_commit.json')
            if (d['commit_sha256']!=digest(c) or c['history_sha256']!=digest(state['history'][:t+3])
                    or c['selected_id']!=d['selected_id'] or c['selected_id']!=state['history'][t+3]['candidate_id']
                    or c['mode']!=d['mode'] or c['mode']=='knowledge_guided' and c['selected_id']==c['BO_reference_id']):
                raise ValueError('Commit/prefix mismatch')
            if c['opportunity_gap']>c['epsilon']+1e-12:raise ValueError('Budget exceeded')
            counter[c['mode']]+=1
            counter['forecasts']+=len(c['shadows'])
            for review in c['reviews']:
                if review.get('created_expert'):counter['created_experts']+=1
                if review.get('outcome')=='numeric_abstention':counter['numeric_abstentions']+=1
            counter['versions_deployed']+=len(c['deployed_versions'])
            counter['positive_feedback']+=sum(all(e[k]>0 for k in ('gp_gain','null_gain','mean_gain','null_mean_gain')) for e in d['evidence'])
            counter['weight_increases']+=sum(e['trusted_weight_after']>e['trusted_weight_before'] for e in d['evidence'])
            weights.extend(e['trusted_weight_after'] for e in d['evidence'])
            forecast_gains.extend(d['evidence'])
            diagnostics.append(dict(task=state['task'],mode=state['mode'],query=t,
                selection=c['mode'],coverage=c['coverage'],verification=c['verification'],
                beta=[dict(id=n['id'],version=n['version'],mean=n['beta_mean'],variance=n['beta_variance']) for n in c['numeric_candidates']],
                evidence=d['evidence'],fallback=c['fallback_reason']))
            if state['mode']=='full' and c['mode']!='base_bo':
                changes.append(dict(task=state['task'],query=t,mode=c['mode'],selected_id=c['selected_id'],
                    BO_reference_id=c['BO_reference_id'],opportunity_gap=c['opportunity_gap'],
                    raw_value=d['raw_value'],verification=c['verification'],evidence=d['evidence'],
                    mixture_logscore=d['deployed_mixture_log_score_gain']))
        row.update(diagnostics=dict(counter),maximum_trust=max(weights,default=0.),
            cumulative_prospective_gains={k:sum(e[k] for e in forecast_gains) for k in ('gp_gain','null_gain','mean_gain','null_mean_gain')})
        if row['knowledge_choices']!=counter['knowledge_guided']:raise ValueError('Weighted choice counter differs from commitments')
        rows.append(row)
    mapping={(r['task'],r['mode']):r for r in rows}
    if any((task,mode) not in states for task in protocol['tasks'] for mode in ('bo','full')):
        raise ValueError('All8 BO/full trajectories are required')
    for task in protocol['tasks']:
        baseline=mapping[task,'bo'];old=study.read(study.paths(ROOT,'v7')/'runs'/f'{task}_bo_agent_initial_terra'/'state.json')
        if states[task,'bo']['history']!=old['history']:raise ValueError('Full20-point BO history differs from frozen control')
        for mode in ('full','verification','no_calibration'):
            if (task,mode) not in states:continue
            r=mapping[task,mode]
            if states[task,mode]['history'][:4]!=states[task,'bo']['history'][:4]:raise ValueError('Initialization mismatch')
            r.update(AUC_delta_vs_BO=r['SearchAUC']-baseline['SearchAUC'],best_delta_vs_BO=r['final_best']-baseline['final_best'])
        for mode in ('verification','no_calibration'):
            if (task,mode) in states:
                mapping[task,'full']['AUC_delta_vs_'+mode]=mapping[task,'full']['SearchAUC']-mapping[task,mode]['SearchAUC']
    arms={}
    for mode in ('full','verification','no_calibration'):
        arm=[r for r in rows if r['mode']==mode]
        wins=sum(r['AUC_delta_vs_BO']>1e-9 for r in arm);losses=sum(r['AUC_delta_vs_BO']<-1e-9 for r in arm)
        arms[mode]=dict(completed=len(arm),wins=wins,ties=len(arm)-wins-losses,losses=losses,
            final_best_losses=sum(r['best_delta_vs_BO']<-1e-9 for r in arm),
            weighted_choices=sum(r['knowledge_choices'] for r in arm),optional_verifications=sum(r['optional_verifications'] for r in arm))
    stage1=arms['full']['wins']>=5 and arms['full']['weighted_choices']>0
    stage2=all(arms[mode]['completed']==8 for mode in ('verification','no_calibration'))
    gains={mode:sum(r.get('AUC_delta_vs_'+mode,0)>1e-9 for r in rows if r['mode']=='full') for mode in ('verification','no_calibration')}
    audit_path=reports/'v8_joint_audit.json';audit=study.read(audit_path) if audit_path.exists() else {}
    audited=(audit.get('technical_result')=='PASS_LISTED_CHECKS' and not audit.get('pending')
        and not audit.get('failures') and audit.get('protocol_sha256')==protocol['protocol_sha256']
        and audit.get('totals',{}).get('runs')==len(rows))
    ready=stage1 and stage2 and all(gains.values()) and audited
    stage=('READY_FOR_FRESH_REGISTRATION' if ready else 'STAGE2_AUDIT_OR_INCREMENT_PENDING' if stage2
           else 'STAGE1_PASS_CONTROLS_DUE' if stage1 else 'STAGE1_REJECTED_STOP')
    result=dict(variant='v8',protocol_sha256=protocol['protocol_sha256'],rows=rows,arms=arms,
        stage=stage,stage1_pass=stage1,stage2_complete=stage2,learning_control_AUC_wins=gains,
        independently_audited=audited,ready_for_fresh_registration=ready,
        independent_confirmation=False,goal_achieved=False,changed_decisions=changes,round_diagnostics=diagnostics,
        note='Seed0 development on all8tasks. Same policy everywhere. Historical failures preserved; no tuning from unqueried objectives.')
    (reports/'v8_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['V8：冻结GP数值知识闭环；8任务 seed0 开发结果',
        '同一候选集、4初始化＋16查询、GP与native qLogEI；gpt-6.1-terra medium。',
        '任务\t方法\tSearchAUC\t最终最佳\t首达总轮次\tAUC较BO\t终值较BO\t加权/验证']
    for task in protocol['tasks']:
        for mode in ('bo','full','verification','no_calibration'):
            if (task,mode) not in mapping:continue
            r=mapping[task,mode]
            lines.append(f"{task}\t{mode}\t{r['SearchAUC']:.9f}\t{r['final_best']:.9f}\t{r['first_best_total']}\t"
                f"{r.get('AUC_delta_vs_BO',0):+.9f}\t{r.get('best_delta_vs_BO',0):+.9f}\t{r['knowledge_choices']}/{r['optional_verifications']}")
    lines.extend(['',json.dumps(arms,ensure_ascii=False),f'阶段判定：{stage}',f'完整独立审计通过：{audited}',
        '多数任务门：至少5/8 AUC严格胜出；差异<=1e-9视为数值持平。最终最佳值损失全部保留。',
        '加权/验证为同轨迹相对当轮GP的实际改点次数，不能把分歧后所有AUC差异归因于单一模块。',
        '本报告不宣称独立初始化确认或最终目标完成。'])
    (reports/'v8_results.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(stage=stage,arms=arms,independently_audited=audited),ensure_ascii=False))


if __name__=='__main__':main()
