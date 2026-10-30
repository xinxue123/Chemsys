"""Prequential GP calibration on committed choices, never unqueried outcomes."""
from pathlib import Path
import json
import math
import sys
import numpy as np
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from chem.benchmarks import reasoning_bo_data as data
OUT=Path(__file__).resolve().parent
VARIANTS=['robust_fit_t0','robust_anchor_t0','robust_agent_t0','additive_t0','tie_agent_t0']


def main():
    rows=[];summaries=[]
    for task in data.SPECS:
        pool=data.load_pool(task);lookup={k:i for i,k in enumerate(pool['ids'])}
        for variant in VARIANTS:
            root=Path('data/runs/stable_r2_20261029')/variant/'runs'/(task+'_s0_'+variant)
            state=data.read(root/'state.json')
            for event in state['decisions']:
                t=event['iteration'];j=lookup[event['selected_id']];prefix=state['history'][:t+3]
                observed=[pool['points'][lookup[h['candidate_id']]] for h in prefix]
                point=pool['points'][j]
                nearest=min(sum(point[a]!=p[a] for a in pool['axes']) for p in observed)
                contrasts={a:sum([b for b in pool['axes'] if observed[u][b]!=observed[v][b]]==[a]
                    for u in range(len(prefix)) for v in range(u)) for a in pool['axes']}
                with np.load(root/'rounds'/f'{t:02d}_candidates.npz') as x:
                    mu=float(x['mu_y'][j]);sd=float(x['sd_observation_y'][j]);error=event['y']-mu;z=error/sd
                rows.append({'task':task,'variant':variant,'query':t,'selected_id':event['selected_id'],
                    'mode':event['execution_mode'],'GP_mu_percent':mu*100,'GP_observation_SD_pp':sd*100,
                    'observed_percent':event['y']*100,'innovation_z':z,'covered_95pct':abs(z)<=1.95996398454,
                    'Gaussian_NLL_y_units':.5*z*z+math.log(sd)+.5*math.log(2*math.pi),
                    'nearest_observed_edit_count':nearest,'own_prefix_isolated_axis_contrasts':contrasts,
                    'retained':event['retained'],'knowledge_amplitude':event['amplitude']})
            own=[r for r in rows if r['task']==task and r['variant']==variant]
            summaries.append({'task':task,'variant':variant,'queries':len(own),
                '95pct_coverage':sum(r['covered_95pct'] for r in own)/len(own),
                'abs_z_gt3':sum(abs(r['innovation_z'])>3 for r in own),
                'max_abs_z':max(abs(r['innovation_z']) for r in own),
                'mean_NLL':float(np.mean([r['Gaussian_NLL_y_units'] for r in own])),
                'mean_absolute_error_pp':float(np.mean([abs(r['observed_percent']-r['GP_mu_percent']) for r in own]))})
    result={'status':'READ_ONLY_AUDIT_COMPLETE','scope':'160 actual committed choices from five development strategies; adaptive selected-point calibration, not full-domain or independent calibration.',
        'summary':summaries,'rows':rows,'no_oracle_file_read':True,'LLM_calls':0}
    data.write(OUT/'calibration_audit.json',result)
    text=['**稳定策略：已选择点的逐轮GP校准审计**','',
        '仅对五个失败开发版本实际选择并观测的160个点计算，使用选择前日志中的GP预测。它不是全候选覆盖率或独立校准测试；不能据此选择未来高产候选。','',
        '| 数据集 | 版本 | 95%预测区间覆盖率 | abs(z)>3轮数 | 最大abs(z) | 平均绝对误差/百分点 |','|---|---|---:|---:|---:|---:|']
    for r in summaries:text.append(f"| {r['task']} | {r['variant']} | {r['95pct_coverage']:.1%} | {r['abs_z_gt3']} | {r['max_abs_z']:.2f} | {r['mean_absolute_error_pp']:.2f} |")
    text += ['', 'z使用GP含观测噪声标准差。零单轴对照表示没有一对仅改变该变量的已有实验，不能把跨多条件的边际差解释为该变量的确定因果效应。所有逐轮数值及对照计数见calibration_audit.json。']
    (OUT/'calibration_audit.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    print(json.dumps(summaries,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
