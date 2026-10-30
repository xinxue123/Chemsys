"""Validate both R1 trajectories and compare with read-only sealed controls."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import csv
import json
import numpy as np
from . import revised_continuous_study as s
from . import revised_continuous_engine as e
from . import revised_continuous_llm as llm
from . import continuous_benchmark_data as data
from .suzuki_domain import digest

NAMES={'Pure_BO':'纯BO（qLogEI）','LGBO_region':'区域LGBO','Gate_only':'仅预算门限','B4_hard':'硬B4',
       'B4_soft':'软B4＋区域注入','LLINBO_Justify':'LLINBO-Justify','ReasoningBO':'Reasoning BO',
       'New_Agent':'旧新方案','New_Agent_R1':'修正版新方案R1'}

def csv_write(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def prefix_state(state,t):
    prev=state['decisions'][:t-1];banks={}
    for event in prev:
        if event['bank_key'] is not None and event['bank_after'] is not None:banks[event['bank_key']]=deepcopy(event['bank_after'])
    call=next((c for c in state['calls'] if c['iteration']==t),None)
    return {**deepcopy(state),'history':deepcopy(state['history'][:t+3]),'decisions':deepcopy(prev),
        'knowledge':deepcopy(call['knowledge_before'] if call else prev[-1]['knowledge'] if prev else state['initial_knowledge']),
        'banks':banks,'region_trust':deepcopy(prev[-1]['region_trust_after']) if prev else {},
        'calls':[deepcopy(c) for c in state['calls'] if c['iteration']<t],
        'interaction_tests':[deepcopy(c) for c in state['interaction_tests'] if c['query']<t]}

def no_refit(*args,**kwargs):raise RuntimeError('Audit refuses new GP fits.')

def audit_task(task):
    e.old.warm();pool=s.frozen.pool(task);cache=e.Cache(pool);state=data.read(s.ROOT/'runs'/(task+'_New_Agent_R1')/'state.json')
    assert state['status']=='COMPLETE' and len(state['history'])==20 and len(state['decisions'])==16
    ready=data.read(s.frozen.ROOT/'initialization'/task/'ready.json')
    assert state['history'][:4]==ready['history'] and state['initial_knowledge']==ready['knowledge']
    assert len({h['candidate_id'] for h in state['history']})==20
    assert len(state['calls'])<=3 and sum(c['iteration']<=8 for c in state['calls'])<=2
    assert all(c['iteration']<=14 for c in state['calls'])
    assert all(b['iteration']-a['iteration']>=2 for a,b in zip(state['calls'],state['calls'][1:]))
    assert len(state['interaction_tests'])<=2
    errors=[];filter_errors=[];negative=0;contexts=set();count=0
    with patch.object(e.old,'fit_gpytorch_mll',no_refit):
        for t,event in enumerate(state['decisions'],1):
            own=prefix_state(state,t);folder=s.ROOT/'runs'/state['run_id']/'rounds';cp=folder/f'{t:02d}_committed.json';c=data.read(cp)
            assert data.filehash(cp)==event['commit_sha256'] and c['history']==own['history'] and c['history_sha256']==digest(own['history'])
            table=folder/f'{t:02d}_candidates.npz';assert data.filehash(table)==c['candidate_table_sha256']
            with np.load(table) as f:arrays={k:f[k].copy() for k in f.files}
            d=e.setup(cache,own,t);reasons=e.event_reasons(own,d,t);quota=e.refresh_permission(own,t)
            call=next((v for v in state['calls'] if v['iteration']==t),None)
            if call:
                assert quota['allowed'] and quota==call['quota_before'] and reasons==call['reasons']
                plan=llm.validate_region(call['answer']['raw'],pool,own);assert plan==call['answer']['plan']
                context=data.read(s.ROOT/'requests'/call['context_sha256']/'context.json');assert context['observed_prefix']==own['history']
                assert context['public']['trust_gate']==d['trust']
                own['knowledge']=deepcopy(plan['knowledge']);own['calls'].append(deepcopy(call));d=e.setup(cache,own,t);contexts.add(call['context_sha256'])
            else:assert not reasons or not quota['allowed'],(task,t,reasons,quota)
            assert own['region_trust']==c['region_trust_at_commit'] and own['knowledge']==c['knowledge']
            assert d['admission']==c['admission'] and d['bank']==c['bank_before'] and d['prior']==c['bank_prior']
            assert d['amplitude']==c['amplitude'] and d['gate_open']==c['gate_open'] and d['trust']==c['trust']
            assert pool['ids'][d['normal_proposal']]==c['normal_proposal_id']
            for name,values in [('base_qLogEI',d['base']),('guided_qLogEI',d['score']),('raw_shift_std',d['raw_shift_std']),
                ('deployed_shift_std',d['shift_std']),('mu_y',d['stats']['mu_y']),('sd_y',d['stats']['sd_y']),('sd_predictive_y',d['stats']['sd_predictive_y'])]:
                err=float(np.max(np.abs(values-arrays[name])));assert err<1e-8,(task,t,name,err);errors.append(err)
            assert np.array_equal(arrays['eligible'],d['eligible']) and np.array_equal(arrays['available'],d['available'])
            assert np.array_equal(arrays['full_region_unqueried'],e.old.membership(pool,c['knowledge']).all(0)&d['available'])
            j=cache.ix[c['selected_id']];assert d['available'][j] and d['eligible'][j] and not c['normal_gate_exception']
            if c['interaction']:
                assert t in (2,6);joint=c['interaction'];prepared=e.interaction_candidates(cache,own,d,t);assert prepared==joint['prepared']
                checked=llm.validate_interaction(joint['answer']['raw'],prepared,own);assert checked==joint['answer']['plan'] and checked['candidate_id']==c['selected_id']
                context=data.read(s.ROOT/'requests'/joint['context_sha256']/'context.json');assert context['observed_prefix']==own['history'];contexts.add(joint['context_sha256'])
                mask=np.zeros(len(pool['ids']),dtype=bool)
                for row in prepared['candidates']:mask[cache.ix[row['candidate_id']]]=True
                assert np.array_equal(mask,arrays['interaction_pool']) and np.all(d['eligible'][mask])
            else:assert j==d['selected'] and not arrays['interaction_pool'].any()
            observed=data.reveal(task,c['selected_id']);assert observed==state['history'][t+3] and observed['y']==event['y']
            m=event['measurement'];mu=float(d['stats']['mu_y'][j]);raw=float(d['raw_shift_std'][j]*d['stats']['std']);R=float(d['stats']['sd_predictive_y'][j]**2+.0009)
            assert abs(m['actual_deployed_mean_y']-(mu+d['amplitude']*raw))<1e-12
            assert abs(m['innovation']-(event['y']-m['actual_deployed_mean_y']))<1e-12
            gain=(event['y']-mu)**2-(event['y']-m['actual_deployed_mean_y'])**2;assert abs(gain-m['actual_paired_SE_gain'])<1e-12
            if d['bank'] is not None:
                prior=d['prior'];H=raw if d['gate_open'] else 0.;P=prior['variance'];a=prior['mean'];innovation=event['y']-mu-a*H
                expected=float(np.clip(a+(P*H/(P*H*H+R))*innovation,0,1));variance=P*R/(P*H*H+R)
                err=max(abs(expected-event['bank_after']['mean']),abs(variance-event['bank_after']['variance']));assert err<1e-12;filter_errors.append(err)
                if gain < -1e-12:assert event['bank_after']['mean']<=a+1e-12;negative+=1
                d['_measurement']=m;updated=deepcopy(event['bank_after']);feedback=e.trust_feedback(own,{**d,'selected':j},t,event['y'],updated)
                assert feedback==event['trust_feedback'] and own['region_trust']==event['region_trust_after']
            assert 0<=event['amplitude']<=1
            count+=1;cache.release()
    assert state['operator_compiles_in_loop']==0 and cache.counters['fits']==0
    return {'task':task,'choices_checked':count,'KF_updates_checked':len(filter_errors),'negative_gain_nonincrease_checks':negative,
        'max_prediction_or_acquisition_error':max(errors),'max_KF_error':max(filter_errors,default=0.),
        'LLM_contexts_checked':len(contexts),'refresh_rounds':[c['iteration'] for c in state['calls']],
        'interaction_rounds':[v['query'] for v in state['interaction_tests']],'GP_refits_in_audit':0,'operator_compiles_in_loop':0}

def audit():
    p=s.check();assert s.status()['complete']==2
    with ProcessPoolExecutor(max_workers=2) as ex:checked=list(ex.map(audit_task,s.TASKS))
    keep=data.read(s.OUT/'protected_snapshot.json');assert digest(keep)==p['protected_snapshot_sha256']
    for path,h in keep.items():assert data.filehash(path)==h,path
    receipt={'status':'PASS','completed_trajectories':2,'added_choices_verified':32,'baselines_rerun':0,
        'protected_files_unchanged':len(keep),'checks':checked,'scope':'technical execution; not a claim that the revised strategy wins'}
    data.write(s.OUT/'validation_receipt.json',receipt);return receipt

def metric(state,pool):
    y=np.array([h['raw_value'] for h in state['history']]);best=np.maximum.accumulate(y)
    return {'task':state['task'],'method':'New_Agent_R1','SearchAUC':float(best[4:].mean()),'final_best':float(best[-1]),
        'first_final_best_total':int(np.flatnonzero(y==best[-1])[0]+1),'initial_best':float(best[3]),
        'refreshes':len(state['calls']),'controlled_tests':len(state['interaction_tests']),'gate_exceptions':0,
        'nonbase_choices':sum(v['selected_id']!=v['reference_id'] for v in state['decisions']),
        'unit':pool['objective_unit'],'result_source':'R1_only_new_rerun'}

def report():
    s.check();assert data.read(s.OUT/'validation_receipt.json')['status']=='PASS'
    ss=s.states();assert len(ss)==2 and all(r['status']=='COMPLETE' for r in ss)
    controls=deepcopy(data.read(s.frozen.OUT/'results.json')['metrics'])
    for r in controls:r['result_source']='frozen_original_no_rerun'
    current=[metric(state,s.frozen.pool(state['task'])) for state in ss];combined=controls+current
    csv_write(s.OUT/'comparison.csv',combined);data.write(s.OUT/'results.json',{'metrics':combined,'new_metrics':current,'protocol_sha256':s.check()['protocol_sha256']})
    text=['# SnAr与HPLC：修正版新方案R1','',
        '仅重跑2条新方案轨迹，其他16条原对照及旧新方案结果全部只读。相同候选集、GP、seed=0初始化和4＋16预算。规则在本轮成绩之前固定，没有择优筛选。','',
        '修正：未查询候选知识准入；无偏移项、与实际部署预测一致的标量KF及配对误差增益；前8轮最多2次刷新，保留第9–14轮额度；第2/6轮最多2次交互检验，严格满足原qLogEI门限。','']
    curves={}
    for task in s.TASKS:
        pool=s.frozen.pool(task);state=next(r for r in ss if r['task']==task);group={r['method']:r for r in combined if r['task']==task}
        text+=['## '+task,'',f"候选数{len(pool['ids'])}；目标单位：{pool['objective_unit']}。",'',
            '| 方法 | SearchAUC | 最终最佳 | 首次最佳/总次序 |','|---|---:|---:|---:|']
        for method in NAMES:
            r=group[method];text.append(f"| {NAMES[method]} | {r['SearchAUC']:.3f} | {r['final_best']:.3f} | {r['first_final_best_total']} |")
        n=group['New_Agent_R1'];b=group['New_Agent'];bo=group['Pure_BO'];li=group['LLINBO_Justify']
        text+=['',f"R1的AUC较旧新方案{n['SearchAUC']-b['SearchAUC']:+.3f}，较纯BO{n['SearchAUC']-bo['SearchAUC']:+.3f}，较LLINBO{n['SearchAUC']-li['SearchAUC']:+.3f}。",'',
            '刷新轮次：'+str([c['iteration'] for c in state['calls']])+'；交互检验：'+str([c['query'] for c in state['interaction_tests']])+'；越过采集门限次数：0。','',
            f'![逐轮与累计最佳]({task}_comparison.png)','']
        trajectories=[];scale=pool['objective_scale'];folder=s.OUT/task;folder.mkdir(parents=True,exist_ok=True)
        with np.load(data.ROOT/task/'oracle.npz') as f:truth=f['raw_values'].copy()
        for t,ev in enumerate(state['decisions'],1):
            j=pool['ids'].index(ev['selected_id']);m=ev['measurement']
            trajectories.append({'query_round':t,'total_experiment':t+4,'candidate_id':ev['selected_id'],**pool['points'][j],
                'objective':ev['raw_value'],'cumulative_best':ev['incumbent_after']*scale,'mode':ev['execution_mode'],
                'GP_mean':m['prequery_mu_y']*scale,'GP_predictive_SD':m['prequery_predictive_sd_y']*scale,
                'deployed_correction':m['actual_deployed_correction_y']*scale,'deployed_mean':m['actual_deployed_mean_y']*scale,
                'paired_SE_gain_raw_squared':m['actual_paired_SE_gain']*scale*scale,'alpha':ev['amplitude'],
                'coefficient_after':m['coefficient_after'],'gate_open':ev['gate_open'],
                'knowledge_unqueried_witnesses':ev['admission']['unqueried_witness_count'],
                'B4_revoked_ids':','.join(ev['diagnosis']['revoked_ids']),'base_opportunity_loss':ev['base_opportunity_loss'],'budget':ev['budget']})
            path=s.ROOT/'runs'/state['run_id']/'rounds'/f'{t:02d}_candidates.npz';c=data.read(path.with_name(f'{t:02d}_committed.json'))
            with np.load(path) as f:z={k:f[k].copy() for k in f.files}
            # Decode each stored array ONCE; no per-cell NPZ decompression.
            allrows=[{'candidate_id':key,**pool['points'][j],'objective_for_analysis_only':float(truth[j]),
                'queried_before':not bool(z['available'][j]),'GP_mean':float(z['mu_y'][j])*scale,'GP_latent_SD':float(z['sd_y'][j])*scale,
                'GP_predictive_SD':float(z['sd_predictive_y'][j])*scale,'base_qLogEI':float(z['base_qLogEI'][j]),
                'budget_eligible':bool(z['eligible'][j]),'raw_region_field':float(z['raw_shift_std'][j])*c['GP_scale_y']*scale,
                'deployed_correction':float(z['deployed_shift_std'][j])*c['GP_scale_y']*scale,'guided_qLogEI':float(z['guided_qLogEI'][j]),
                'knowledge_unqueried_witness':bool(z['full_region_unqueried'][j]),'interaction_pool':bool(z['interaction_pool'][j]),
                'selected':key==ev['selected_id']} for j,key in enumerate(pool['ids'])]
            csv_write(folder/'candidate_tables'/f'{t:02d}_candidates.csv',allrows)
        csv_write(folder/'trajectory.csv',trajectories)
        csv_write(folder/'interaction_tests.csv',state['interaction_tests'])
        curves[task]={}
        for method in ('Pure_BO','LLINBO_Justify','ReasoningBO','New_Agent','New_Agent_R1'):
            source=state if method=='New_Agent_R1' else data.read(s.frozen.ROOT/'runs'/(task+'_'+method)/'state.json')
            y=np.array([h['raw_value'] for h in source['history']]);curves[task][method]={'objective':y.tolist(),'incumbent':np.maximum.accumulate(y).tolist()}
    data.write(s.OUT/'curves.json',curves)
    text+=['## 口径与必要适配','',
        '- SearchAUC仍为第5–20个实验位置累计最佳的均值，保留各任务原单位，不跨任务合并。首次最佳针对各轨迹自身最终最佳。',
        '- 原始初始建议和初始4点复用；HPLC初始建议无新候选见证，准入修复占用新的刷新额度，已明确记录。',
        '- SnAr优化STY，E-factor仅记录；HPLC仍按相同1007组输入的首条实测值回放，没有改用重复均值或额外插值。',
        '- KF实际部署增益与影子信任恢复证据分开记录；门关闭时不伪称知识已经带来预测收益。',
        '- HPLC无精确双轴邻居时采用匹配背景的现有联合条件；其余变量变化公开，不宣称独立因果交互。',
        '- 本轮只有每任务一个seed=0初始化；即使结果改善，也不能据此主张统计显著性或普遍优势。','']
    (s.OUT/'REPORT.md').write_text('\n'.join(text),encoding='utf-8');plot(curves)
    return {'status':'SAVED','new_metrics':current,'report':str((s.OUT/'REPORT.md').resolve())}

def plot(curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'Pure_BO':'#666666','LLINBO_Justify':'#2676ad','ReasoningBO':'#ae6a37','New_Agent':'#b56a83','New_Agent_R1':'#d02b2b'}
    for task in s.TASKS:
        with plt.rc_context({'pdf.fonttype':42}):
            fig,axes=plt.subplots(1,2,figsize=(12.4,4.2),layout='constrained')
            for method,series in curves[task].items():
                for ax,key in zip(axes,('objective','incumbent')):
                    ax.plot(range(1,21),series[key],label=method,color=colors[method],linewidth=2.2 if method=='New_Agent_R1' else 1.2,
                        marker='o' if method=='New_Agent_R1' else None,markersize=3,linestyle='--' if method=='New_Agent' else '-')
            for ax,title in zip(axes,('Observed objective','Cumulative best')):
                ax.set_title(task+' | '+title);ax.set_xlabel('Total experiment (4 + 16)');ax.set_ylabel('STY (kg m^-3 h^-1)' if task=='SnAr' else 'Peak area (source units)');ax.grid(alpha=.2)
            handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=8)
            fig.savefig(s.OUT/f'{task}_comparison.png',dpi=180);fig.savefig(s.OUT/f'{task}_comparison.pdf');plt.close(fig)

def inspect(task,t):
    assert 1<=t<=16;state=data.read(s.ROOT/'runs'/(task+'_New_Agent_R1')/'state.json');path=s.ROOT/'runs'/state['run_id']/'rounds'/f'{t:02d}_committed.json'
    return {'decision':state['decisions'][t-1],'commit':str(path.resolve()),'candidate_arrays':str(path.with_name(f'{t:02d}_candidates.npz').resolve()),
        'candidate_CSV':str((s.OUT/task/'candidate_tables'/f'{t:02d}_candidates.csv').resolve())}
