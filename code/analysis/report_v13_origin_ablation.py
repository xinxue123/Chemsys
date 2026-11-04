"""Report completed factorial and targeted probes; never generates decisions."""
from collections import Counter,defaultdict
from pathlib import Path
from statistics import mean,stdev
import csv
import hashlib
import json
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from v13_origin_ablation import study,queue

OUT=ROOT/'experiments/reports'/study.VARIANT
LABELS=dict(c1='Suzuki 1',c2='Suzuki 2',c3='Suzuki 3',c4='Suzuki 4',perera='Perera',
            doyle='Doyle',snar='SnAr',hplc='HPLC')
ARMS=('bo','reasoning_bo','full','fixed','no_refresh','neither','no_notes')
NAMES=dict(bo='BO',reasoning_bo='Reasoning BO',full='V13',fixed='Fixed KF',
           no_refresh='No refresh',neither='Neither',no_notes='No Notes')
EPS=1e-8


def read(p):return queue.read(p)
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(name,values):
    values=list(values)
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader()
        w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in values)
def wtl(values):
    c=Counter('win' if x>EPS else 'loss' if x < -EPS else 'tie' for x in values)
    return dict(wins=c['win'],ties=c['tie'],losses=c['loss'])


def main():
    audit=read(OUT/'audit.json')
    if audit['status']!='PASS_COMPLETE':raise ValueError('Complete native replay audit required')
    p=study.check(True);metrics=[];rounds=[];states={};call_origins=Counter()
    for job in p['jobs']:
        folder=study.base()/'runs'/job['id'];s=read(folder/'state.json')
        task=study.SHORT[job['task']];key=(task,job['seed'],job['arm']);states[key]=s
        metric,rr=measure(task,job['seed'],job['arm'],s,study.rel(folder/'state.json'))
        metric['origin']='frozen_full_replay' if job['arm']=='full' else 'new_control_conditional_response_map'
        note_menus=0;event_changes=0;selected_note_queries=0
        initial_notes=0;first_live=first_covered=first_scored=''
        for row,d in zip(rr,s['decisions']):
            c=read(folder/'rounds'/f"{row['query']:02}_commit.json")
            rq=ROOT/c['reasoning']['request_relpath'];public=read(rq.parent/'context.json')['public']
            covered=public['retrieved_notes']['knowledge'];note_menus+=bool(covered)
            live=sum(n['recommendation']['permission']!='withdrawn' for n in c['edited_notes'])
            if row['query']==2:initial_notes=live
            if live and first_live=='':first_live=row['query']
            if covered and first_covered=='':first_covered=row['query']
            if d['feedback'] and first_scored=='':first_scored=row['query']
            event_changes+=sum(abs(e['calibration']['after']['reliability']-e['calibration']['before']['reliability'])>1e-12 for e in d['feedback'])
            selected_note_queries+=bool(d['feedback'])
            row.update(selected_id=c['selected_id'],GP_reference=c['BO_reference_id'],
                changed_from_GP=int(c['selected_id']!=c['BO_reference_id']),menu_note_count=len(covered),
                forecast_count=len(d['feedback']),knowledge_uses=d['knowledge_uses'],refresh=int(c['memory'] is not None),
                choose_origin=read(rq)['origin'])
            call_origins[row['choose_origin']]+=1
        metric.update(editor_calls=len(s['edit_calls']),forecast_events=sum(len(d['feedback']) for d in s['decisions']),
            selected_forecast_queries=selected_note_queries,menu_covered_queries=note_menus,
            actual_reliability_changes=event_changes,initial_q2_live_notes=initial_notes,
            first_live_note_query=first_live,first_menu_covered_query=first_covered,first_scored_query=first_scored)
        metrics.append(metric);rounds.extend(rr)
    for m in study.historical():
        if m['method'] not in ('bo','reasoning_bo'):continue
        s=read(ROOT/m['state_path']);key=(m['task'],int(m['seed']),m['method']);states[key]=s
        metric,rr=measure(*key,s,m['state_path'])
        metric.update(origin='archived_independent_response_reference',editor_calls=0,forecast_events=0,
            selected_forecast_queries=0,menu_covered_queries=0,actual_reliability_changes=0,
            initial_q2_live_notes=0,first_live_note_query='',first_menu_covered_query='',first_scored_query='')
        metrics.append(metric)
        for row,d in zip(rr,s['decisions']):
            row.update(selected_id=d['selected_id'],GP_reference=d['BO_reference_id'],
                changed_from_GP=int(d['selected_id']!=d['BO_reference_id']),menu_note_count=0,
                forecast_count=0,knowledge_uses=[],refresh=0,choose_origin='archived_reference')
        rounds.extend(rr)
    by={(m['task'],m['seed'],m['arm']):m for m in metrics}
    contrasts=[]
    comparisons=(('KF_with_refresh','full','fixed'),('KF_without_refresh','no_refresh','neither'),
                 ('refresh_with_KF','full','no_refresh'),('refresh_without_KF','fixed','neither'),
                 ('both_vs_neither','full','neither'),('structured_bundle','full','no_notes'),
                 ('full_vs_BO','full','bo'),('full_vs_RBO','full','reasoning_bo'),
                 ('chooser_vs_RBO','no_notes','reasoning_bo'))
    for task in LABELS:
        for seed in range(4):
            for name,left,right in comparisons:
                a=by[task,seed,left];b=by[task,seed,right]
                contrasts.append(dict(task=task,seed=seed,contrast=name,left=left,right=right,
                    delta_SearchAUC=a['SearchAUC']-b['SearchAUC'],delta_final_best=a['final_best']-b['final_best'],
                    identical_all_choices=int([h['candidate_id'] for h in states[task,seed,left]['history']]==
                                               [h['candidate_id'] for h in states[task,seed,right]['history']]),
                    coupled_response_map=int(left not in ('bo','reasoning_bo') and right not in ('bo','reasoning_bo'))))
            def value(arm,key):return by[task,seed,arm][key]
            contrasts.append(dict(task=task,seed=seed,contrast='KF_refresh_interaction',left='factorial',right='factorial',
                delta_SearchAUC=value('full','SearchAUC')-value('fixed','SearchAUC')-value('no_refresh','SearchAUC')+value('neither','SearchAUC'),
                delta_final_best=value('full','final_best')-value('fixed','final_best')-value('no_refresh','final_best')+value('neither','final_best'),
                identical_all_choices='',coupled_response_map=1))
    summary=[]
    for task in LABELS:
        for arm in ARMS:
            rs=[m for m in metrics if m['task']==task and m['arm']==arm]
            summary.append(dict(task=task,arm=arm,n=len(rs),SearchAUC_mean=mean(r['SearchAUC'] for r in rs),
                SearchAUC_sample_sd=stdev(r['SearchAUC'] for r in rs),final_best_mean=mean(r['final_best'] for r in rs),
                final_best_sample_sd=stdev(r['final_best'] for r in rs)))
    effect_summary=[]
    for task in LABELS:
        for name in dict.fromkeys(r['contrast'] for r in contrasts):
            rs=[r for r in contrasts if r['task']==task and r['contrast']==name]
            effect_summary.append(dict(task=task,contrast=name,n=4,delta_SearchAUC_mean=mean(r['delta_SearchAUC'] for r in rs),
                delta_final_best_mean=mean(r['delta_final_best'] for r in rs),
                **wtl(r['delta_SearchAUC'] for r in rs),identical_choice_trajectories=sum(r['identical_all_choices']==1 for r in rs)))
    inactive=[]
    for task in LABELS:
        for seed in range(4):
            full=by[task,seed,'full']
            kf_same=next(r for r in contrasts if r['task']==task and r['seed']==seed and r['contrast']=='KF_with_refresh')['identical_all_choices']
            bundle_same=next(r for r in contrasts if r['task']==task and r['seed']==seed and r['contrast']=='structured_bundle')['identical_all_choices']
            if full['forecast_events']==0 and not kf_same:raise ValueError('Inactive KF changed trajectory under coupled response map')
            if full['menu_covered_queries']==0 and not bundle_same:raise ValueError('Invisible Note changed chooser trajectory')
            no_refresh_same=next(r for r in contrasts if r['task']==task and r['seed']==seed and r['contrast']=='refresh_without_KF')
            if full['initial_q2_live_notes']==0:
                def ids(arm):return [h['candidate_id'] for h in states[task,seed,arm]['history']]
                if not ids('no_refresh')==ids('neither')==ids('no_notes'):
                    raise ValueError('Empty initial knowledge did not yield equivalent no-editor controls')
            inactive.append(dict(task=task,seed=seed,full_forecast_events=full['forecast_events'],
                full_menu_covered_queries=full['menu_covered_queries'],full_fixed_identical=kf_same,full_no_notes_identical=bundle_same,
                initial_q2_live_notes=full['initial_q2_live_notes']))
    probes=[]
    for job in p['probes']:
        folder=study.base()/'probes'/job['id'];c=read(folder/'commit.json');o=read(folder/'outcome.json')
        sf=ROOT/study.original(job['task'],job['seed'])['state_path'];original=read(sf.parent/'rounds'/f"{job['query']:02}_commit.json")
        rf=ROOT/study.original(job['task'],job['seed'],'reasoning_bo')['state_path'];original_rbo=read(rf.parent/'rounds'/f"{job['query']:02}_commit.json")
        probes.append(dict(seed=job['seed'],query=job['query'],template=job['template'],repetition=job['rep'],
            selected_id=c['selected_id'],objective=o['observation']['raw_value'],incumbent=o['incumbent'],
            immediate_record_gain=max(0.,o['observation']['raw_value']-o['incumbent']),
            matches_original_V13_point=int(c['selected_id']==original['selected_id']),
            matches_original_RBO_point=int(c['selected_id']==original_rbo['selected_id'])))
    facts=dict(status='COMPLETE',metrics=len(metrics),new_control_trajectories=128,new_control_queries=2560,
        targeted_one_step_probes=len(probes),call_origins=dict(call_origins),
        contrasts={name:dict(SearchAUC=wtl(r['delta_SearchAUC'] for r in contrasts if r['contrast']==name),
            final_best=wtl(r['delta_final_best'] for r in contrasts if r['contrast']==name))
            for name in dict.fromkeys(r['contrast'] for r in contrasts)},
        no_KF_feedback_full_cells=sum(r['full_forecast_events']==0 for r in inactive),
        zero_Note_menu_coverage_full_cells=sum(r['full_menu_covered_queries']==0 for r in inactive),
        empty_initial_q2_knowledge_cells=sum(r['initial_q2_live_notes']==0 for r in inactive),
        limitations=['Frozen-response-map conditional diagnostic; full baseline is replay, not fresh independent evidence',
            'Identical inputs are coupled; language sampling at altered inputs is not separately controlled',
            'Archived BO/RBO references are not response-coupled','Four starts, descriptive estimates',
            'Stopping editing after an empty q2 editor also prevents later first Note creation; it is not a pure existing-Note revision effect',
            'One-step prompt probes measure choice variability, not future optimization value'])
    for name,data in [('metrics.csv',metrics),('rounds.csv',rounds),('paired_effects.csv',contrasts),
        ('task_summary.csv',summary),('module_summary.csv',effect_summary),('inactive_cells.csv',inactive),('prompt_probes.csv',probes)]:write(name,data)
    (OUT/'facts.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    figures(effect_summary);text(summary,facts,probes)
    (OUT/'artifact_manifest.json').write_text(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest()
        for f in OUT.iterdir() if f.is_file() and f.name!='artifact_manifest.json'},indent=2)+'\n',encoding='utf-8')
    print(json.dumps(facts))


def measure(task,seed,arm,s,path):
    y=np.array([h['raw_value'] for h in s['history']],float);best=np.maximum.accumulate(y)
    if len(y)!=20 or s['status']!='COMPLETE':raise ValueError('Incomplete 0+20 trajectory')
    metric=dict(task=task,seed=seed,arm=arm,SearchAUC=float(best.mean()),final_best=float(best[-1]),
        first_final_best_query=int(np.flatnonzero(best>=best[-1]-EPS)[0])+1,
        M10=float(best[9]),late_gain=float(best[-1]-best[9]),state_path=path)
    rr=[dict(task=task,seed=seed,arm=arm,query=q,objective=float(v),cumulative_best=float(b)) for q,(v,b) in enumerate(zip(y,best),1)]
    return metric,rr


def figures(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    effects=('KF_with_refresh','refresh_with_KF','both_vs_neither','structured_bundle')
    labels=('Adaptive KF','Later refresh','KF + refresh','Structured bundle')
    fig,axes=plt.subplots(4,2,figsize=(12,12))
    for ax,task in zip(axes.flat,LABELS):
        for i,e in enumerate(effects):
            row=next(r for r in summary if r['task']==task and r['contrast']==e)
            val=row['delta_SearchAUC_mean']
            ax.barh(i,val,color='#386FAD' if val>=0 else '#D9AE53',edgecolor='#263A4D',
                    hatch=None if val>=0 else '///',linewidth=.6)
            ax.annotate(f'{val:+.2f}',(val,i),xytext=(5 if val>=0 else -5,0),textcoords='offset points',
                        ha='left' if val>=0 else 'right',va='center',fontsize=9)
        ax.axvline(0,color='#333333',linewidth=.8);ax.set_yticks(range(4),labels);ax.invert_yaxis()
        ax.set_title(LABELS[task],loc='left',fontweight='bold');ax.set_xlabel('SearchAUC difference (native units)')
        ax.margins(x=.4);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('From-zero V13 ablations: conditional module contrasts',fontsize=14,fontweight='bold')
    fig.text(.5,.015,'Four starts per task; positive favors the enabled module. Independent scales and task units.\n'
        'Identical scientific inputs share frozen responses; altered-input response variability remains.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.16,right=.97,top=.94,bottom=.08,hspace=.65,wspace=.65)
    fig.savefig(OUT/'module_effects.png',dpi=180,facecolor='white');fig.savefig(OUT/'module_effects.pdf');plt.close(fig)


def text(summary,facts,probes):
    f=lambda x:f"{x['wins']}胜/{x['ties']}平/{x['losses']}负"
    lines=['V13从零初始化收益来源消融','',
        '8任务×4起始编号；每条0＋20；冻结GP/候选池/数值种子。',
        '完整V13的32条是冻结响应精确重放。4种控制共128条、2560次预算内查询。',
        '同一起始编号的完全相同科学输入共用响应；改变输入后使用新的独立子智能体。',
        '这是条件响应映射下的机制诊断，不是完整V13的新增独立重复。','',
        '完整V13与消融的配对结果（32组，SearchAUC / 最终最佳）：']
    for name,x in facts['contrasts'].items():lines.append(f"{name}: {f(x['SearchAUC'])} / {f(x['final_best'])}")
    lines+=['','任务均值，单元格SearchAUC / 最终最佳；各任务单位不同，禁止跨任务平均原值。']
    for task in LABELS:
        lines.append(LABELS[task])
        for r in summary:
            if r['task']==task:lines.append(f"  {NAMES[r['arm']]}: {r['SearchAUC_mean']:.4f} / {r['final_best_mean']:.4f}")
    lines+=['','Doyle固定前缀提示/接口探针：每种提示3次独立响应，观察历史及Insight相同。']
    for seed,q in ((0,2),(1,9)):
        for template in ('v13','rbo'):
            rs=[r for r in probes if r['seed']==seed and r['query']==q and r['template']==template]
            lines.append(f"seed{seed} q{q} {template}: "+'; '.join(f"{r['selected_id']} -> {r['objective']:.4f}" for r in rs))
    lines+=['',f"32组中q2初始知识为空：{facts['empty_initial_q2_knowledge_cells']}组。",
        '在这些组内，停止后续编辑也阻止了之后首次创建有效Note，且no_refresh/neither/no_notes三条轨迹应完全一致。',
        '因此对应刷新差值包含后续知识创建的价值，不能全部称为修订已有知识的收益。',
        '','解释限制：',*facts['limitations'],
        '第一步没有GP；每一步都收费。固定KF保留事实和权限反馈，停止刷新保留q2初始编辑（允许空知识）。',
        'no_notes保留V13选点提示及Insight，因此它与Reasoning BO不是同一个算法。',
        '选点探针的当轮产率不能代表后续信息价值，尤其Doyle seed0最早分化点本身产率较低。',
        'audit.json为全部GP数组/候选菜单/LLM输入/原始响应/状态转移审计记录；原结果均未覆盖。']
    (OUT/'消融结果说明.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8-sig')


if __name__=='__main__':main()
