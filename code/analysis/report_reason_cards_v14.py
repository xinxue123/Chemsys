"""Readable handoff from completed V14 receipts; no experimental execution."""
from collections import Counter
from pathlib import Path
import csv,json

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/reports/reason_cards_v14_20261103'
RUNS=ROOT/'experiments/runs/reason_cards_v14_20261103/runs'


def main():
    audit=json.loads((OUT/'audit.json').read_text(encoding='utf-8'))
    assert audit['status']=='PASS_COMPLETE' and audit['queries_replayed']==120
    with (OUT/'comparison.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    names={'Perera_Suzuki':'Perera Suzuki','HPLC':'HPLC','Reizman_Suzuki_Case_1':'Suzuki Case 1',
           'Doyle_Buchwald_Hartwig':'Doyle Buchwald-Hartwig','SnAr':'SnAr'}
    counts=[]
    for folder in sorted(RUNS.iterdir()):
        if not folder.is_dir():continue
        s=json.loads((folder/'state.json').read_text(encoding='utf-8'));ch=s['decision_channel']
        events=[e for d in s['decisions'] for e in d['feedback']]
        counts.append(dict(run=folder.name,card_KF_updates=ch['calibration']['updates'],
            memory_calls=s['edit_calls'],gate_rejections=sum(not d['deployment']['accepted'] for d in s['decisions']),
            local_facts=len(ch['facts']),local_statuses=dict(Counter(f['status'] for f in ch['facts'])),
            note_reviews=s['review_queries'],card_reviews=ch['review_queries'],
            matched_direction_tests=sum(e['direction_test'] for e in events),
            calibration_only_direction_tests=sum(e['declared_use'] in ('positive_support','tentative_test','review_test')
                and not e['direction_test'] for e in events),
            note_versions=len(s['notes'])))
    (OUT/'mechanisms.json').write_text(json.dumps(counts,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    statuses=Counter()
    for c in counts:statuses.update(c['local_statuses'])
    assert sum(c['card_KF_updates'] for c in counts)==114
    lines=['V14 实现与首次针对性验证','2026-11-03','',
        '范围：两条已知失利轨迹＋四条既有获益轨迹，共6条、120次查询。每条0条初始观测＋20次付费查询。',
        '候选池、原GP实现、数值seed=0及起始菜单规则保持原样；V13、Reasoning BO、BO对照复用封存记录。',
        'LLM为每次请求独立的gpt-6.1-terra medium子智能体；无跨请求响应缓存。','',
        '实现内容：',
        '1. 每轮给五候选分别提交改善概率；无区域Notes时也有预测卡。q1没有incumbent，概率为null。',
        '2. GP审查官方提议；被拒后只执行GP参照，按实际执行点的预测更新KF，不给未执行点造标签。',
        '3. 知识方向信用要求真实focus变化与受控背景；预测可信度不等于正向推荐许可。',
        '4. KF数值参与采集机会预算；物理局部反证不因改名消失，复核共享两次实验预算。',
        '5. 刷新检查可执行候选覆盖，空操作仍计调用；新增知识至少覆盖两个未查询输入。','',
        'SearchAUC（20次查询后的累计最佳均值；任务单位各自保留）：',
        '任务 | seed | BO | V13 | Reasoning BO | V14 | V14-V13']
    for r in rows:
        lines.append(f"{names[r['task']]} | {r['seed']} | {float(r['bo_SearchAUC']):.5f} | {float(r['v13_SearchAUC']):.5f} | {float(r['reasoning_bo_SearchAUC']):.5f} | {float(r['v14_SearchAUC']):.5f} | {float(r['V14_minus_V13']):+.5f}")
    lines+=['','V13 → V14 的最终最佳值及首次到达本轨迹最终最佳的查询次数：']
    for r in rows:
        lines.append(f"{names[r['task']]} seed={r['seed']}: {float(r['v13_final_best']):.5f}(q{r['v13_first_final_best_query']}) → {float(r['v14_final_best']):.5f}(q{r['v14_first_final_best_query']})")
    wins=sum(float(r['V14_minus_V13'])>1e-8 for r in rows)
    bo_wins=sum(float(r['V14_minus_BO'])>1e-8 for r in rows)
    lines+=['',f'结果：相对V13，{wins}/6条SearchAUC提高；相对BO，{bo_wins}/6条更高。',
        'Perera损失减少但仍不如BO；Case1、SnAr、HPLC seed2提高；Doyle较V13下降但仍高于BO；HPLC seed0显著回退。',
        '因此尚未完成同时改善两条失利轨迹的目标，V14保留为研究候选，V13继续封存作对照。',
        '此样本按已知失败/成功选取，且每条仅一份新LLM响应轨迹，不能声称多种子稳定或某模块的独立因果增量。','',
        '实际触发的机制：',
        f"候选预测KF更新114次；GP审查拒绝17次；局部事实{sum(c['local_facts'] for c in counts)}条，最终状态{dict(statuses)}。",
        f"局部复核{sum(len(c['card_reviews']) for c in counts)}次；无局部事实恢复；一次局部事实在两次失败复核后转withdrawn。",
        '区域Notes在五条轨迹中为空，Perera仅有两个tentative版本；本轮没有匹配方向测试获得正向信用。',
        '这表明候选级闭环已运行，但不能将结果说成成熟区域知识的稳定收益。',
        '暂停/撤回的事实级通路有实测触发；恢复通路仅通过合成机制测试，本轮没有成功恢复实例。','',
        '验证：11项机制测试通过；合成0/1/2/4前缀GP与采集一致；120次决策完整重放通过。',
        '144份独立子智能体响应全部通过结构校验，无格式重试；查询循环内算子编译为0；旧对照哈希未变。',
        '审计适配：按response.raw.json原字节恢复字典顺序，避免嵌入提示中的JSON字符串键顺序不同；每次均断言canonical承诺完全相同。',
        '科学代码、参数、输出与原协议未修改。审计适配器的SHA256记录在audit.json。','',
        '代码：code/experiment_tools/reason_cards_v14/；协议：docs/reason_cards_v14_20261103.txt。',
        '主循环：study.run_one；GP审查：controller.gate；知识信用：controller.credit_eligibility；反馈：controller.feedback。',
        '逐轮表：rounds.csv（提议、实际执行、GP参照、预算、KF、局部抑制、编辑）；细节见runs/*/rounds/*_commit.json。',
        '复核命令（不发起新实验）：',
        'runtime/environments/.venv-official/Scripts/python.exe code/analysis/audit_reason_cards_v14_receipts.py',
        '不要用已冻结CLI中的原audit入口重建排序后的提示副本；使用上述原始响应审计适配器。']
    (OUT/'summary_zh.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(V14_wins_vs_V13=wins,V14_wins_vs_BO=bo_wins,local_fact_statuses=dict(statuses),
        summary=(OUT/'summary_zh.txt').as_posix()),ensure_ascii=False))


if __name__=='__main__':main()
