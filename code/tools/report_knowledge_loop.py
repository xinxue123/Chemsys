"""Read-only metric, commitment and prospective-score audit for a frozen run."""
from collections import Counter
from pathlib import Path
from math import sqrt
import argparse
import json
import sys
import numpy as np
from scipy.special import log_ndtr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code/chemist_system'))
from chem.benchmarks import knowledge_study as study
from chem.benchmarks import knowledge_llm as llm


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--variant',default='v2');args=ap.parse_args()
    protocol=study.check(ROOT,args.variant)
    confirmation=protocol.get('trial_seed',protocol['seed']) == 1
    stage='registered_paired_confirmation' if confirmation else 'development'
    directory=study.paths(ROOT,args.variant)
    rows=[]
    for path in sorted((directory/'runs').glob('*/state.json')):
        state=study.read(path); row=study.summarize(state)
        assert len(state['history'])==4+len(state['decisions'])
        assert len(set(h['candidate_id'] for h in state['history']))==len(state['history'])
        modes=Counter(); positive=0; predictions=0; deployed_rounds=0
        for t,d in enumerate(state['decisions'],1):
            cp=path.parent/'rounds'/f'{t:02d}_commit.json';c=study.read(cp)
            assert c['history_sha256']==llm.digest(state['history'][:t+3])
            assert c['protocol_sha256']==protocol['protocol_sha256']
            assert c['query']==t and c['selected_id']==state['history'][t+3]['candidate_id']==d['selected_id']
            assert llm.digest(c)==d['commit_sha256']
            assert c['opportunity_gap'] <= c['epsilon']+1e-12
            with np.load(cp.parent/f'{t:02d}_GP.npz') as z:
                assert np.isfinite(z['base_logei']).all()
                assert int(z['available'].sum())==len(z['available'])-t-3
            if d['mode']=='optional_verification':
                assert t<=10 and c['opportunity_gap']<=np.log(1.1)+1e-12
            modes[d['mode']]+=1
            deployed_rounds+=bool(c['deployed_versions'])
            assert len(c['shadows'])==len(d['evidence'])
            for sh,r in zip(c['shadows'],d['evidence']):
                pred=sh['prediction'];predictions+=1
                assert pred['point_id']==c['selected_id'] and pred['query_round']==t
                assert pred['version']==r['version'] and pred['hypothesis_id']==r['hypothesis_id']
                if sh['informative']:
                    vz=pred['contrast_var']+pred['preference_noise_sd']**2
                    rem=max(vz-pred['cov_y_contrast']**2/pred['gp_var_y'],1e-15*vz)
                    alpha=pred['contrast_mean']/sqrt(vz)
                    beta=(pred['contrast_mean']+pred['cov_y_contrast']/pred['gp_var_y']*(d['y']-pred['gp_mean_y']))/sqrt(rem)
                    expected=float(log_ndtr(beta)-log_ndtr(alpha))
                    assert np.isclose(expected,r['delta_log_score'],rtol=1e-10,atol=1e-11)
                    positive+=expected>0
                else:
                    assert r['delta_log_score'] is None
        assert len(state['calls'])<=4 and state['verification_used']<=1
        assert state.get('cache_counters',{}).get('operator_compiles_in_loop',0)==0
        row.update(selection_modes=dict(modes),forecast_count=predictions,positive_evidence=positive,
                   deployment_calculation_rounds=deployed_rounds,trace_audit='PASS',run=str(path.relative_to(ROOT)))
        rows.append(row)
    bo={r['task']:r for r in rows if r['mode']=='bo' and r['status']=='COMPLETE'}
    for row in rows:
        if row['mode']!='bo' and row['status']=='COMPLETE' and row['task'] in bo:
            b=bo[row['task']]
            row['AUC_delta_vs_matched_BO']=row['SearchAUC']-b['SearchAUC']
            row['best_delta_vs_matched_BO']=row['final_best']-b['final_best']
            row['performance_pass']=row['AUC_delta_vs_matched_BO']>=-1e-9 and row['best_delta_vs_matched_BO']>=-1e-9
    completed=[r for r in rows if r['mode']=='full' and r['status']=='COMPLETE']
    majority = 'majority' in protocol.get('goal_config_path','')
    arms = {}
    for mode in ('full','deployment','shadow'):
        arm=[r for r in rows if r['mode']==mode and r['condition']=='agent_initial_terra' and r['status']=='COMPLETE']
        wins=sum(r.get('AUC_delta_vs_matched_BO',0)>1e-9 for r in arm)
        losses=sum(r.get('AUC_delta_vs_matched_BO',0)<-1e-9 for r in arm)
        arms[mode]=dict(completed=len(arm),AUC_wins=wins,AUC_losses=losses,AUC_ties=len(arm)-wins-losses,
            majority_development_pass=len(arm)==8 and wins>=5,
            knowledge_choices=sum(r['knowledge_choices'] for r in arm),
            knowledge_tasks=[r['task'] for r in arm if r['knowledge_choices']>0])
    report=dict(variant=args.variant,stage=stage,seed=protocol['seed'],rows=rows,completed_full_scenarios=len(completed),target_full_scenarios=8 if confirmation else 20,
        current_acceptance='majority5of8realAUCwins' if majority else 'original20scenario_noninferiority',
        real_agent_arm_comparisons=arms,
        performance_complete=any(x['majority_development_pass'] for x in arms.values()) if majority else len(completed)==20 and all(r.get('performance_pass',False) for r in completed),
        independent_validation=False,independent_trajectory_complete=confirmation and len(completed)==8,goal_achieved=False,
        warning='Same eight benchmark domains; the registered seed1 arm is a new paired trajectory, not unseen-dataset generalization. Metrics alone do not certify joint audit or causal module attribution.')
    out=ROOT/'experiments/reports/knowledge_loop_20261101'/f'{args.variant}_results.json'
    llm.write(out,report)
    lines=[f"知识闭环 {args.variant} 实验结果（seed={protocol['seed']}，4+16）",
        '模型 gpt-6.1-terra，reasoning.effort=medium。'+('这是预先登记的新初始轨迹配对确认，仍是同8个数据域；不代表未见数据集泛化。' if confirmation else '以下是开发结果，不是独立确认。'),
        '纯BO与Agent共用候选池、四点初始化、GP和qLogEI；精确同分按物理特征独立随机优先级裁决。',
        'P0和Doyle为产率百分比，Perera为UV产物面积百分比；SnAr为STY，HPLC为峰面积，不跨单位平均。', '',
        '真实LLM初始知识：full=包含最多一次主动验证；deployment=仅前瞻反馈后受控部署；shadow=知识仅预测、不影响选点。',
        '任务\t方法\tSearchAUC\t最终最佳\t首次最佳总轮次\tAUC较BO\t最佳较BO\t知识选点次数\t主动验证次数\t双指标不退化(参考)']
    tasks=protocol['tasks']
    for task in tasks:
        for mode in ('bo','shadow','deployment','full'):
            matches=[r for r in rows if r['task']==task and r['mode']==mode and r['condition']=='agent_initial_terra' and r['status']=='COMPLETE']
            for r in matches:
                lines.append('\t'.join(map(str,[task,mode,f"{r['SearchAUC']:.6f}",f"{r['final_best']:.6f}",r['first_best_total'],
                    f"{r.get('AUC_delta_vs_matched_BO',0):+.6f}",f"{r.get('best_delta_vs_matched_BO',0):+.6f}",
                    r['knowledge_choices'],r['optional_verifications'],r.get('performance_pass','baseline') ])))
    lines += ['', 'P0人工压力条件：原区域先验转为真实候选对，不是LLM自行发现的知识。',
        '任务\t条件\tSearchAUC\t最终最佳\tAUC较BO\t最佳较BO\t知识选点次数\t双指标通过']
    for r in rows:
        if r['mode']=='full' and r['condition']!='agent_initial_terra' and r['status']=='COMPLETE':
            lines.append('\t'.join(map(str,[r['task'],r['condition'],f"{r['SearchAUC']:.6f}",f"{r['final_best']:.6f}",
                f"{r['AUC_delta_vs_matched_BO']:+.6f}",f"{r['best_delta_vs_matched_BO']:+.6f}",r['knowledge_choices'],r['performance_pass']])))
    completion_line = (f"预先登记的配对确认：full完成{arms['full']['completed']}/8；不包含人工压力条件。" if confirmation else f"full真实Agent完成{arms['full']['completed']}/8；人工压力条件完成{sum(r['condition']!='agent_initial_terra' for r in completed)}/12（稳健性诊断）。" if majority else
        f"full完整场景：{len(completed)}/20；通过双指标：{sum(r.get('performance_pass',False) for r in completed)}/{len(completed)}。")
    lines += ['',completion_line,
        '技术审计单独见同目录joint_independent_audit.json；技术PASS不替代性能门。',
        '正向预测证据、进入采集计算、实际改变选点是三个不同事件。单次主动验证引发的后续轨迹变化不能宣称为加权知识的因果收益。',
        '完整目标尚未确认达成；需前瞻知识贡献与固定策略的独立验证，未合并main。']
    if majority:
        lines += ['', '用户修订后的主标准：8个真实Agent任务中至少5个SearchAUC严格优于BO。持平不计优势；终值和损失完整报告。',
            '原20场景全部不退化已不再是要求；人工条件为稳健性诊断。']
        for mode,a in arms.items():
            lines.append(f"{mode}: 完成{a['completed']}/8；AUC胜/平/负={a['AUC_wins']}/{a['AUC_ties']}/{a['AUC_losses']}；知识改点{a['knowledge_choices']}次；开发多数优势={a['majority_development_pass']}。")
    else:
        lines.append('历史原验收为20场景不弱BO；后续用户修订验收见knowledge_loop_goal_20261101_majority.json，不追溯改写此历史协议。')
    out.with_suffix('.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
