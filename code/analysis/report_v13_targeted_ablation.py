"""Report only the four user-retained cells; cancelled arms stay excluded."""
from collections import Counter
from pathlib import Path
from statistics import mean
import csv
import hashlib
import json
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools'),str(ROOT/'code/analysis')]
import v13_targeted_ablation as study
from v13_origin_ablation import queue
from report_v13_origin_ablation import measure,wtl

OUT=ROOT/'experiments/reports'/study.VARIANT
NAMES=dict(full='V13',fixed='Fixed KF',no_refresh='No later editing',neither='Neither',bo='BO',reasoning_bo='Reasoning BO')


def write(name,values):
    values=list(values)
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader()
        w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in values)


def main():
    if queue.read(OUT/'audit.json')['status']!='PASS_COMPLETE':raise ValueError('Full targeted audit required')
    p=study.check(True);metrics=[];rounds=[];states={};origins=Counter();editor_actions=[];note_visibility=[]
    for job in p['jobs']:
        folder=study.base()/'runs'/job['id'];s=queue.read(folder/'state.json')
        task=study.original.SHORT[job['task']];key=(task,job['seed'],job['arm']);states[key]=s
        m,rr=measure(*key,s,study.rel(folder/'state.json'))
        first_note=None;initial_q2=0;coverage=0;changed=0
        for row,d in zip(rr,s['decisions']):
            q=row['query'];c=queue.read(folder/'rounds'/f'{q:02}_commit.json')
            meta=ROOT/c['reasoning']['request_relpath'];ctx=queue.read(meta.parent/'context.json')
            visible=ctx['public']['retrieved_notes']['knowledge'];coverage+=bool(visible)
            for n in c['edited_notes']:
                note_visibility.append(dict(task=task,seed=job['seed'],arm=job['arm'],query=q,note_id=n['id'],version=n['version'],
                    effect=n['effect'],permission=n['recommendation']['permission'],
                    menu_candidates_covered=sum(study.original.control.in_scope(n['scope'],study.original.adapter.conditions(
                        study.original.POOL(ROOT,job['task']),cid)) for cid in c['display_ids'])))
            if c['memory']:
                reviews=c['memory']['plan']['reviews']
                for action in reviews or [dict(op='empty',accepted=False,outcome='empty')]:
                    editor_actions.append(dict(task=task,seed=job['seed'],arm=job['arm'],query=q,
                        op=action.get('op',''),accepted=action.get('accepted',False),outcome=action.get('outcome',''),
                        note_id=action.get('id',''),reason=action.get('reason','')))
            if q==2:initial_q2=len(c['edited_notes'])
            if first_note is None and c['edited_notes']:first_note=q
            changed+=sum(abs(e['calibration']['before']['reliability']-e['calibration']['after']['reliability'])>1e-12 for e in d['feedback'])
            origins[queue.read(meta)['origin']]+=1
            row.update(selected_id=c['selected_id'],GP_reference_id=c['BO_reference_id'],
                differs_from_GP=int(c['selected_id']!=c['BO_reference_id']),editor=int(c['memory'] is not None),
                menu_note_count=len(visible),forecasts=len(d['feedback']),knowledge_uses=d['knowledge_uses'])
        y=np.array([h['raw_value'] for h in s['history']]);best=np.maximum.accumulate(y)
        increments=np.diff(np.r_[0.,best]);credit=np.arange(20,0,-1)/20*increments
        m.update(origin='frozen_response_replay' if job['arm']=='full' else 'conditional_control',
            editor_calls=len(s['edit_calls']),editor_queries=s['edit_calls'],initial_q2_notes=initial_q2,
            first_accepted_Note_query='' if first_note is None else first_note,menu_coverage=coverage,
            forecast_events=sum(len(d['feedback']) for d in s['decisions']),reliability_changes=changed,
            early_origin_SearchAUC=float(credit[:10].sum()),late_origin_SearchAUC=float(credit[10:].sum()))
        assert np.isclose(m['SearchAUC'],m['early_origin_SearchAUC']+m['late_origin_SearchAUC'])
        metrics.append(m);rounds.extend(rr)
    kept={(study.original.SHORT[t],s) for t,s in study.CELLS}
    for old in study.original.historical():
        if (old['task'],int(old['seed'])) not in kept or old['method'] not in ('bo','reasoning_bo'):continue
        s=queue.read(ROOT/old['state_path']);key=(old['task'],int(old['seed']),old['method']);states[key]=s
        m,rr=measure(*key,s,old['state_path']);best=np.maximum.accumulate([h['raw_value'] for h in s['history']])
        credit=np.arange(20,0,-1)/20*np.diff(np.r_[0.,best])
        m.update(origin='archived_uncoupled_reference',editor_calls=0,editor_queries=[],initial_q2_notes=0,
            first_accepted_Note_query='',menu_coverage=0,forecast_events=0,reliability_changes=0,
            early_origin_SearchAUC=float(credit[:10].sum()),late_origin_SearchAUC=float(credit[10:].sum()))
        metrics.append(m)
        for row,d in zip(rr,s['decisions']):
            row.update(selected_id=d['selected_id'],GP_reference_id=d['BO_reference_id'],
                differs_from_GP=int(d['selected_id']!=d['BO_reference_id']),editor=0,menu_note_count=0,forecasts=0,knowledge_uses=[])
        rounds.extend(rr)
    lookup={(m['task'],m['seed'],m['arm']):m for m in metrics};effects=[]
    comparisons=(('KF_with_editing','full','fixed'),('editing_with_KF','full','no_refresh'),
        ('KF_without_later_editing','no_refresh','neither'),('editing_with_fixed_KF','fixed','neither'),
        ('joint_modules','full','neither'),('full_vs_BO','full','bo'),('full_vs_RBO','full','reasoning_bo'))
    for task,seed in sorted(kept):
        for name,a,b in comparisons:
            left=lookup[task,seed,a];right=lookup[task,seed,b]
            effects.append(dict(task=task,seed=seed,contrast=name,left=a,right=b,
                delta_SearchAUC=left['SearchAUC']-right['SearchAUC'],delta_final_best=left['final_best']-right['final_best'],
                delta_early_origin=left['early_origin_SearchAUC']-right['early_origin_SearchAUC'],
                delta_late_origin=left['late_origin_SearchAUC']-right['late_origin_SearchAUC'],
                identical_all_choices=int([h['candidate_id'] for h in states[task,seed,a]['history']]==
                    [h['candidate_id'] for h in states[task,seed,b]['history']]),
                coupled_response_map=int(a not in ('bo','reasoning_bo') and b not in ('bo','reasoning_bo'))))
        value=lambda a,k:lookup[task,seed,a][k]
        effects.append(dict(task=task,seed=seed,contrast='interaction',left='factorial',right='factorial',
            delta_SearchAUC=value('full','SearchAUC')-value('fixed','SearchAUC')-value('no_refresh','SearchAUC')+value('neither','SearchAUC'),
            delta_final_best=value('full','final_best')-value('fixed','final_best')-value('no_refresh','final_best')+value('neither','final_best'),
            delta_early_origin=value('full','early_origin_SearchAUC')-value('fixed','early_origin_SearchAUC')-value('no_refresh','early_origin_SearchAUC')+value('neither','early_origin_SearchAUC'),
            delta_late_origin=value('full','late_origin_SearchAUC')-value('fixed','late_origin_SearchAUC')-value('no_refresh','late_origin_SearchAUC')+value('neither','late_origin_SearchAUC'),
            identical_all_choices='',coupled_response_map=1))
    for r in effects:assert np.isclose(r['delta_SearchAUC'],r['delta_early_origin']+r['delta_late_origin'])
    divergences=[]
    for task,seed in sorted(kept):
        for arm in ('fixed','no_refresh','neither'):
            first_prompt=first_choice='';fn=an=[];details={}
            for q in range(1,21):
                fc=queue.read((ROOT/lookup[task,seed,'full']['state_path']).parent/'rounds'/f'{q:02}_commit.json')
                ac=queue.read((ROOT/lookup[task,seed,arm]['state_path']).parent/'rounds'/f'{q:02}_commit.json')
                if first_prompt=='' and fc['reasoning']['context_sha256']!=ac['reasoning']['context_sha256']:
                    first_prompt=q
                    fp=ROOT/fc['reasoning']['request_relpath'];ap=ROOT/ac['reasoning']['request_relpath']
                    fn=queue.read(fp.parent/'context.json')['public']['retrieved_notes']['knowledge']
                    an=queue.read(ap.parent/'context.json')['public']['retrieved_notes']['knowledge']
                if first_choice=='' and fc['selected_id']!=ac['selected_id']:
                    first_choice=q
                    if fc['GP_sha256']!=ac['GP_sha256'] or fc['display_ids']!=ac['display_ids']:
                        raise ValueError('First choice divergence has inconsistent GP/menu')
                    details=dict(full_selected=fc['selected_id'],ablation_selected=ac['selected_id'],
                        full_objective=states[task,seed,'full']['history'][q-1]['raw_value'],
                        ablation_objective=states[task,seed,arm]['history'][q-1]['raw_value'])
            divergences.append(dict(task=task,seed=seed,arm=arm,first_prompt_difference=first_prompt,
                first_choice_difference=first_choice,same_GP_and_menu_at_first_choice_difference=bool(first_choice),
                full_selected=details.get('full_selected',''),ablation_selected=details.get('ablation_selected',''),
                full_objective=details.get('full_objective',''),ablation_objective=details.get('ablation_objective',''),
                full_Note_context_at_first_prompt_difference=fn,ablation_Note_context_at_first_prompt_difference=an))
    # The already-completed, fixed-prefix probes remain separate from new runs.
    probes=[]
    for j in queue.read(study.original.base()/'protocol.json')['probes']:
        folder=study.original.base()/'probes'/j['id'];c=queue.read(folder/'commit.json');o=queue.read(folder/'outcome.json')
        probes.append(dict(seed=j['seed'],query=j['query'],template=j['template'],repetition=j['rep'],
            selected_id=c['selected_id'],objective=o['observation']['raw_value'],incumbent=o['incumbent']))
    broad=[]
    for name in p['cancelled_phase_states_sha256']:
        s=queue.read(ROOT/name)
        if name.split('/')[-2].endswith('_full'):continue
        broad.append(s)
    actual_requests=[]
    allowed_ids={j['id'] for j in p['jobs']}
    for path in (study.original.base()/'responses').glob('*/*/a*/request.json'):
        meta=queue.read(path)
        if meta['job'] in allowed_ids and (path.parent/'response_receipt.json').exists():actual_requests.append(path)
    removed_editing=[];active_KF=[]
    for task,seed in sorted(kept):
        if lookup[task,seed,'full']['forecast_events']>0:active_KF.append(dict(task=task,seed=seed))
        saved=lookup[task,seed,'full']['editor_calls']-lookup[task,seed,'no_refresh']['editor_calls']
        if saved:removed_editing.append(dict(task=task,seed=seed,editor_calls_saved=saved))
    facts=dict(status='COMPLETE',primary_cells=4,full_replays=4,control_trajectories=12,control_queries=240,
        no_notes_cancelled=True,old_broad_phase_control_queries=sum(len(s['history']) for s in broad),
        old_broad_phase_completed_controls=sum(s['status']=='COMPLETE' for s in broad),
        chooser_call_origins=dict(origins),contrasts={name:dict(SearchAUC=wtl(r['delta_SearchAUC'] for r in effects if r['contrast']==name),
            final_best=wtl(r['delta_final_best'] for r in effects if r['contrast']==name)) for name in dict.fromkeys(r['contrast'] for r in effects)},
        actual_new_responder_requests=len(actual_requests),
        schema_rejected_requests=sum(queue.read(path.parent/'validation.json')['status']=='SCHEMA_REJECTED' for path in actual_requests),
        KF_exercised_cells=active_KF,editing_actually_removed_cells=removed_editing,
        full_editor_accepted_actions=dict(Counter(r['op'] for r in editor_actions if r['arm']=='full' and r['accepted'])),
        Perera_q15_H002_remaining_menu_coverage=sum(r['menu_candidates_covered'] for r in note_visibility if r['task']=='perera' and r['seed']==2 and r['arm']=='full' and r['note_id']=='H002' and r['query']>=15),
        targeted_not_general_performance_claim=True,full_replay_is_not_independent_LLM_replication=True,
        first_nonempty_note_not_first_empty_editor=True,
        altered_prompt_sampling_not_independently_paired=True)
    for name,data in [('metrics.csv',metrics),('rounds.csv',rounds),('paired_effects.csv',effects),
        ('prompt_probes.csv',probes),('first_divergence.csv',divergences),
        ('editor_actions.csv',editor_actions),('note_visibility.csv',note_visibility)]:write(name,data)
    (OUT/'facts.json').write_text(json.dumps(facts,indent=2)+'\n',encoding='utf-8')
    render(metrics,effects,facts,probes,divergences)
    figures(rounds,effects)
    print(json.dumps(facts))


def render(metrics,effects,facts,probes,divergences):
    f=lambda d:f"{d['wins']}胜/{d['ties']}平/{d['losses']}负"
    lines=['V13代表案例消融结果','',
        '主范围：Doyle seed0、1；HPLC seed3；Perera seed2。12条新控制，每条0＋20，共240次预算查询。',
        '完整V13为冻结响应精确重放；BO、Reasoning BO沿用冻结历史结果。主消融不包含Note全移除组，旧相关轨迹保留为取消阶段。',
        '同一起始编号下完全相同的科学提示共用响应；改变的提示重新调用独立gpt-6.1-terra medium子智能体。',
        '这是选定机制案例的条件响应映射诊断，不是8任务总体优势或新增独立重复。','',
        '核心发现：',
        '1. Doyle两个起始编号的完整、固定KF、停后续编辑和两者关闭，20次选点全部相同；这些优势不需要动态KF或后续Note编辑。',
        '2. HPLC seed3和Perera seed2确有KF反馈。关闭数值自适应后两者SearchAUC数值都更高；HPLC最终最佳则由完整KF更高。',
        '3. 真正减少后续编辑的组合有'+str(len(facts['editing_actually_removed_cells']))+'个，均未改变选点；Doyle seed0从未形成Note，停更开关未触发，不能把它算作有力的刷新检验。',
        '4. 本轮没有证明KF带来SearchAUC增量，也没有证明首次登记Note后的独立语义编辑带来增量。改变提示后的LLM抽样差异仍存在，不能外推为普遍有害。',
        f"5. 240次预算查询对应新增{facts['actual_new_responder_requests']}个独立响应/格式修正子智能体请求；两者不是同一计数。",'',
        '后续编辑为何没有改点：Doyle seed1和HPLC seed3均保留H001；Perera seed2在q9新增H002被结构检查拒绝，q15新增H002虽被接受，但在q15～20的候选菜单覆盖数为'+str(facts['Perera_q15_H002_remaining_menu_coverage'])+'。',
        '这定位了本轮独立编辑没有形成新增选点信息的原因，不能据此断言任何知识更新方式都无用。',
        '本轮保留初始Note、事实反馈和权限控制，因而没有检验这些机制整体的价值。','',
        '方法 | SearchAUC | 最终最佳 | 首次自身最佳轮次 | 编辑轮次 | 首次登记Note轮次 | KF权重变化次数']
    for task,seed in dict.fromkeys((r['task'],r['seed']) for r in metrics):
        lines.append(f'\n{task} seed={seed}')
        for arm in ('bo','reasoning_bo','full','fixed','no_refresh','neither'):
            m=next(m for m in metrics if (m['task'],m['seed'],m['arm'])==(task,seed,arm))
            lines.append(f"{NAMES[arm]} | {m['SearchAUC']:.4f} | {m['final_best']:.4f} | {m['first_final_best_query']} | {m['editor_queries']} | {m['first_accepted_Note_query']} | {m['reliability_changes']}")
    lines+=['','配对模块增量（正值有利于开启模块；不同任务原单位不能相加）：']
    for r in effects:lines.append(f"{r['task']} seed{r['seed']} {r['contrast']}: SearchAUC {r['delta_SearchAUC']:+.4f}, best {r['delta_final_best']:+.4f}, identical choices {r['identical_all_choices']}")
    lines+=['','四个目标组合的方向计数（描述性，不是显著性检验）：']
    for name,d in facts['contrasts'].items():lines.append(f"{name}: SearchAUC {f(d['SearchAUC'])}; final best {f(d['final_best'])}")
    lines+=['','首次分化位置（首次不同选点之前历史、GP和候选菜单相同）：']
    for r in divergences:
        lines.append(f"{r['task']} seed{r['seed']} {r['arm']}: prompt q{r['first_prompt_difference']}, choice q{r['first_choice_difference']}, "
            f"full {r['full_selected']} -> {r['full_objective']}, ablation {r['ablation_selected']} -> {r['ablation_objective']}")
    lines+=['','Doyle固定前缀提示探针（此前已完成；每模板三次独立响应；不计入主轨迹SearchAUC）：']
    for seed,q in ((0,2),(1,9)):
        for template in ('v13','rbo'):
            rs=[r for r in probes if r['seed']==seed and r['template']==template]
            lines.append(f"seed{seed} q{q} {template}: "+'; '.join(f"{r['selected_id']} -> {r['objective']:.4f}" for r in rs))
    lines+=['','解释边界：',
        '关闭刷新：首次登记非空Note后停止后续语义编辑；此前空编辑不算形成初始知识。登记成功不等于化学结论已被实验证实。',
        'KF关闭仍保留预测、实际反馈、推荐权限、暂停和复核；它只冻结数值校正。',
        '关闭独立编辑不等于LLM不再读取新结果；每轮Insight推理仍保留。',
        '完全相同输入共享响应，控制了相同提示下的重复抽样；改变提示后的LLM波动仍未单独消除。',
        '固定前缀探针只说明当轮选择分布；低产点也可能带来未来信息，不能据当轮产率推断整条搜索。',
        f"原广泛阶段已按用户指令停止，已执行{facts['old_broad_phase_control_queries']}次对照查询保留为取消阶段，退出主分析。",
        '旧代码、原模型、旧结果、已取消阶段均经哈希保护；未覆盖原论文。']
    (OUT/'代表案例消融结果.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8-sig')


def figures(rounds,effects):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cells=(('doyle',0),('doyle',1),('hplc',3),('perera',2))
    colors=dict(full='#1F659E',fixed='#D59B38',no_refresh='#7C657F',neither='#647A38',bo='#4E5257',reasoning_bo='#C16B8D')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,(task,seed) in zip(axes.flat,cells):
        for arm in ('bo','reasoning_bo','full','fixed','no_refresh','neither'):
            rs=sorted((r for r in rounds if (r['task'],r['seed'],r['arm'])==(task,seed,arm)),key=lambda r:r['query'])
            ax.plot([r['query'] for r in rs],[r['cumulative_best'] for r in rs],label=NAMES[arm],color=colors[arm],
                linestyle='--' if arm in ('bo','fixed','neither') else '-',marker='o' if arm in ('fixed','no_refresh') else None,
                markersize=3,linewidth=1.5,alpha=.85)
        title={'doyle':'Doyle','hplc':'HPLC','perera':'Perera'}[task]
        ax.set_title(f'{title} start {seed}',loc='left');ax.set_xlabel('Paid query')
        ax.set_ylabel('Best yield (%)' if task!='hplc' else 'Best peak area');ax.set_xticks([1,5,10,15,20]);ax.grid(alpha=.18)
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False)
    fig.text(.5,.012,'Selected mechanism cases; unchanged 0+20 budget. Full V13 is replay; controls use coupled exact-input responses.\n'
        'Overlapping curves are retained; archived BO/RBO responses are not coupled to these controls.',ha='center',fontsize=9)
    fig.subplots_adjust(top=.85,bottom=.13,hspace=.4,wspace=.3)
    fig.savefig(OUT/'targeted_curves.png',dpi=180,facecolor='white');fig.savefig(OUT/'targeted_curves.pdf');plt.close(fig)


if __name__=='__main__':main()
