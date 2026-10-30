"""Independent trajectory checks, concise comparison and inspectable exports."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path
import csv
import json
import numpy as np
from . import continuous_benchmark_study as s
from . import continuous_benchmark_data as data
from . import continuous_benchmark_engine as e
from .suzuki_domain import digest

NAMES={'Pure_BO':'纯BO（qLogEI）','LGBO_region':'区域LGBO','Gate_only':'仅预算门限','B4_hard':'硬B4',
       'B4_soft':'软B4＋区域注入','LLINBO_Justify':'LLINBO-Justify','ReasoningBO':'Reasoning BO','New_Agent':'新方案Agent'}

def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def audit_one(item):
    task,method=item;e.warm();p=s.pool(task);cache=e.Cache(p);state=data.read(s.ROOT/'runs'/(task+'_'+method)/'state.json')
    assert state['status']=='COMPLETE' and len(state['history'])==20 and len(state['decisions'])==16
    assert len({h['candidate_id'] for h in state['history']})==20 and state['history'][:4]==s.initial(task)
    ready=data.read(s.ROOT/'initialization'/task/'ready.json');assert state['initial_knowledge']==ready['knowledge']
    if method!='New_Agent':assert state['knowledge']==ready['knowledge'] and not state['calls']
    choices=kfs=0;max_gp=max_acq=max_kf=0.;exceptions=0
    for t,event in enumerate(state['decisions'],1):
        prefix=state['history'][:t+3];folder=s.ROOT/'runs'/state['run_id']/'rounds'
        cp=folder/f'{t:02d}_committed.json';c=data.read(cp);assert data.filehash(cp)==event['commit_sha256']
        assert c['history']==prefix and c['history_sha256']==digest(prefix)
        table=folder/f'{t:02d}_candidates.npz';assert data.filehash(table)==c['table_sha256']
        with np.load(table) as f:a={k:f[k].copy() for k in f.files}
        available=~np.isin(p['ids'],[h['candidate_id'] for h in prefix]);assert np.array_equal(available,a['available'])
        v=cache.model(prefix)
        for name in ('mu_y','sd_y','sd_predictive_y','mu_std','sd_std'):
            err=float(np.max(np.abs(v[name]-a[name])));max_gp=max(max_gp,err);assert err<1e-9,(task,method,t,name,err)
        base=e.acquisition(a['mu_std'],a['sd_std'],np.zeros(len(p['ids'])),v['best_std'],c['base_meta']['mc_seed'])
        guided=e.acquisition(a['mu_std'],a['sd_std'],a['deployed_shift_std'],v['best_std'],c['base_meta']['mc_seed'])
        err=max(float(np.max(abs(base-a['base_qLogEI']))),float(np.max(abs(guided-a['guided_qLogEI']))));max_acq=max(max_acq,err);assert err<1e-8
        budget=float(np.log(4)*(16-t)/15);eligible=available&(base[available].max()-base<=budget)
        assert budget==c['budget'] and np.array_equal(eligible,a['eligible'])
        proxy,_,_=e.information_proxy(cache.X.numpy(),[cache.ix[h['candidate_id']] for h in prefix]);ranks=e.priority(len(base),t,0)
        assert np.array_equal(proxy,a['information_proxy']) and np.array_equal(ranks,a['tie_priority'])
        def choose(scores,mask):
            js=np.flatnonzero(mask);js=js[scores[js]==scores[js].max()];js=js[proxy[js]==proxy[js].max()]
            js=js[a['sd_y'][js]==a['sd_y'][js].max()];return int(js[np.argmin(ranks[js])])
        ref=choose(base,available);assert p['ids'][ref]==c['reference_id']
        mem=e.membership(p,c['knowledge']);dg=e.select_candidate(base,mem,budget,candidate_ids=p['ids'],hypothesis_ids=[h['id'] for h in c['knowledge']['clauses']],queried_mask=~available)
        assert list(dg.retained_ids)==c['diagnosis']['retained_ids'] and list(dg.revoked_ids)==c['diagnosis']['revoked_ids']
        positions=[i for i,h in enumerate(c['knowledge']['clauses']) if h['id'] in dg.retained_ids]
        feasible=eligible&(mem[positions].all(0) if positions else True)
        mask=available if method=='LGBO_region' else feasible if method=='B4_hard' else eligible
        assert p['ids'][choose(guided,mask)]==c['normal_proposal_id']
        j=cache.ix[c['selected_id']];assert available[j]
        exception=not bool(eligible[j]);assert exception==c['normal_gate_exception'];exceptions+=exception
        if c['execution_mode']=='normal':assert p['ids'][choose(guided,mask)]==c['selected_id']
        if c['validation']:
            assert t in (2,4,6) and method=='New_Agent'
            anchor=cache.ix[c['validation']['prepared']['anchor_candidate_id']]
            assert np.sum(cache.X[j].numpy()!=cache.X[anchor].numpy())==1
            val=c['validation'];own={**state,'history':prefix,'decisions':state['decisions'][:t-1],'knowledge':c['knowledge'],
                'validations':[row for row in state['validations'] if row['query']<t]}
            prepared=e.validation_candidates(cache,own,{'available':available,'eligible':eligible,'stats':v,'base':base,'tie_priority':ranks})
            assert prepared==val['prepared'] and val['answer']['plan']['candidate_id']==c['selected_id']
        elif method not in ('LGBO_region','ReasoningBO'):assert eligible[j]
        if method=='LLINBO_Justify':
            record=c['LLM'];prop=cache.ix[record['proposed_id']]
            assert record['accepted']==bool(base[ref]<=base[prop]+budget)
            assert j==(prop if record['accepted'] else ref)
            projected=s.llm.project(p,{'history':prefix},t,record['plan']['point']);assert projected==record['projection']
        if method=='ReasoningBO':
            top=sorted(np.flatnonzero(available),key=lambda j:(-base[j],ranks[j]))[:5]
            assert j in top
            checked=s.llm.validate_rbo(c['LLM']['raw'],p,{'history':prefix},top);assert checked['selected_id']==c['selected_id']
        raw,field_meta=cache.field(prefix,c['region']);assert np.max(abs(raw-a['raw_shift_std']))<1e-9
        assert np.max(abs(a['deployed_shift_std']-c['amplitude']*raw))<1e-10
        if c['bank_before'] is not None:
            prior=c['bank_prior'];theta=np.array(prior['mean']);P=np.array(prior['covariance']);H=np.array([1.,raw[j]*v['std']])
            y=state['history'][t+3]['y'];R=a['sd_predictive_y'][j]**2+.0009;innovation=y-a['mu_y'][j]-H@theta
            gain=P@H/(H@P@H+R);expected=theta+gain*innovation;cov=P-np.outer(P@H,H@P)/(H@P@H+R)
            err=max(float(np.max(abs(expected-np.array(event['bank_after']['mean'])))),float(np.max(abs(cov-np.array(event['bank_after']['covariance'])))))
            assert err<1e-12;max_kf=max(max_kf,err);kfs+=1
            if c['trust']['trust_blocked']:assert c['amplitude']==0. and np.array_equal(base,guided)
            if c['trust']['recovered']:assert c['trust']['entry_after']['recovery_evidence']['positive_evidence']
        observed=data.reveal(task,c['selected_id']);assert observed==state['history'][t+3] and observed['y']==event['y']
        choices+=1;cache.release()
    assert state['operator_compiles_in_loop']==0
    return {'task':task,'method':method,'choices_checked':choices,'KF_updates_checked':kfs,'gate_exceptions_declared':exceptions,
        'max_GP_replay_error':max_gp,'max_qLogEI_replay_error':max_acq,'max_KF_error':max_kf,'operator_compiles_in_loop':state['operator_compiles_in_loop']}

def audit():
    p=s.check();assert s.status()['complete']==16
    with ProcessPoolExecutor(max_workers=8) as ex:rows=list(ex.map(audit_one,[(t,m) for t in s.TASKS for m in e.METHODS]))
    contexts=0
    for path in (s.ROOT/'requests').glob('*/context.json'):
        c=data.read(path);answer=data.read(path.parent/'validated.json');assert answer['context_sha256']==c['context_sha256']
        q=deepcopy(c);key=q.pop('context_sha256');assert digest(q)==key
        hist=c['observed_prefix'];assert len(hist)==len(c['public']['observations'])
        for row,h in zip(c['public']['observations'],hist):assert abs(row['objective']-h['y']*s.pool(c['task'])['objective_scale'])<1e-6
        contexts+=1
    for path,h in p['protected_files'].items():assert data.filehash(path)==h,path
    receipt={'status':'PASS','trajectories':16,'queries_checked':sum(r['choices_checked'] for r in rows),'LLM_contexts':contexts,
        'protected_files_unchanged':len(p['protected_files']),'audits':rows,'scope':'technical execution; single initialization does not prove statistical superiority'}
    data.write(s.OUT/'validation_receipt.json',receipt);return receipt

def metrics(state,p):
    values=np.array([h['y']*p['objective_scale'] for h in state['history']]);curve=np.maximum.accumulate(values)
    return {'task':state['task'],'method':state['method'],'SearchAUC':float(curve[4:].mean()),'final_best':float(curve[-1]),
        'first_final_best_total':int(np.flatnonzero(values==curve[-1])[0]+1),'initial_best':float(curve[3]),
        'refreshes':len(state['calls']),'controlled_tests':len(state['validations']),
        'gate_exceptions':sum(x['normal_gate_exception'] for x in state['decisions']),
        'nonbase_choices':sum(x['selected_id']!=x['reference_id'] for x in state['decisions']),
        'LLINBO_accepted':sum(x.get('LLM',{}).get('accepted',False) for x in state['decisions'] if x.get('LLM')),
        'unit':p['objective_unit']}

def report():
    s.check();ss=s.states();assert len(ss)==16 and all(row['status']=='COMPLETE' for row in ss)
    rows=[metrics(row,s.pool(row['task'])) for row in ss];write_csv(s.OUT/'comparison.csv',rows)
    data.write(s.OUT/'results.json',{'metrics':rows,'protocol_sha256':s.check()['protocol_sha256']})
    lines=['# SnAr与HPLC实验对比','',
        '统一候选池、初始化seed=0、共同4次初始化＋16次逐点查询；每个任务一组共同初始LLM知识，模型deepseek-flash。没有人工正确/错误知识条件，也没有多种子调优。','',
        '新方案沿用区域协方差引导、KF动态校正、B4冲突回退、同区域证据恢复、事件刷新和预算内反例检验。全部变量为连续量，统一使用SingleTaskGP；不启用分子核或多类别联合分支。','',
        'SearchAUC为16次追加位置累计最佳目标值的均值；首次最佳为达到本轨迹最终最佳的总实验次序，包含初始化。不同任务单位不同，不合并排名。','']
    for task in s.TASKS:
        p=s.pool(task);group={r['method']:r for r in rows if r['task']==task}
        lines+=['## '+task,'',p['description'],'',f"候选数：{len(p['ids'])}。目标单位：{p['objective_unit']}。",'',
            '| 方法 | SearchAUC | 最终最佳 | 首次最佳/总次序 | AUC较纯BO |','|---|---:|---:|---:|---:|']
        for m in e.METHODS:
            r=group[m];lines.append(f"| {NAMES[m]} | {r['SearchAUC']:.3f} | {r['final_best']:.3f} | {r['first_final_best_total']} | {r['SearchAUC']-group['Pure_BO']['SearchAUC']:+.3f} |")
        lines+=['',f'![逐轮与累计最佳对比]({task}_comparison.png)','']
        # Decision-level tables include future objectives for debugging ONLY.
        # These exported oracle columns never feed a model or LLM request.
        with np.load(data.ROOT/task/'oracle.npz') as f:truth=f['raw_values'].copy()
        candidate_truth=[{'candidate_id':key,**p['points'][i],'simulated_or_measured_objective_for_analysis_only':float(truth[i])} for i,key in enumerate(p['ids'])]
        write_csv(s.OUT/task/'all_candidate_objectives.csv',candidate_truth)
        for state in [r for r in ss if r['task']==task]:
            trace=[]
            for t,event in enumerate(state['decisions'],1):
                j=p['ids'].index(event['selected_id']);trace.append({'query_round':t,'total_experiment':t+4,'candidate_id':event['selected_id'],**p['points'][j],
                    'objective':event['raw_value'],'cumulative_best':event['incumbent_after']*p['objective_scale'],'mode':event['execution_mode'],
                    'GP_mean':event['measurement']['prequery_mu_y']*p['objective_scale'],'GP_predictive_SD':event['measurement']['prequery_predictive_sd_y']*p['objective_scale'],
                    'KF_amplitude':event['amplitude'],'B4_revoked_ids':','.join(event['diagnosis']['revoked_ids']),
                    'trust_blocked':bool(event['trust'] and event['trust']['trust_blocked']),'base_opportunity_loss':event['base_opportunity_loss'],
                    'budget':event['budget'],'gate_exception':event['normal_gate_exception']})
            write_csv(s.OUT/task/(state['method']+'_trajectory.csv'),trace)
            if state['method']=='New_Agent':
                for t,event in enumerate(state['decisions'],1):
                    path=s.ROOT/'runs'/state['run_id']/'rounds'/f'{t:02d}_candidates.npz'
                    scale=data.read(path.with_name(f'{t:02d}_committed.json'))['GP_scale_y']*p['objective_scale']
                    with np.load(path) as z:
                        allrows=[{'candidate_id':key,**p['points'][j],'objective_for_analysis_only':float(truth[j]),
                            'queried_before':not bool(z['available'][j]),'GP_mean':float(z['mu_y'][j])*p['objective_scale'],
                            'GP_latent_SD':float(z['sd_y'][j])*p['objective_scale'],'GP_predictive_SD':float(z['sd_predictive_y'][j])*p['objective_scale'],
                            'base_qLogEI':float(z['base_qLogEI'][j]),'normal_budget_eligible':bool(z['eligible'][j]),
                            'raw_region_shift':float(z['raw_shift_std'][j])*scale,
                            'guided_qLogEI':float(z['guided_qLogEI'][j]),'selected':key==event['selected_id']} for j,key in enumerate(p['ids'])]
                    write_csv(s.OUT/task/'New_Agent_candidate_tables'/f'{t:02d}_candidates.csv',allrows)
    quality=data.read(s.OUT/'data_quality.json')['HPLC']
    lines+=['## 必要适配与范围','',
        '- SnAr是官方动力学模拟器，噪声设0；本轮只优化STY，同时记录E-factor，未执行双目标Pareto优化。',
        f"- HPLC原始1386条记录包含{quality['unique_inputs']}组不同输入，{quality['conflicting_replicate_groups']}组重复输入的目标不一致。按官方LGBO精确匹配取首条规则回放；原始记录全部保留，没有挑选最高重复值，没有使用IDW插值。",
        '- HPLC来源description的Targets行与config存在描述不一致；以config的peak_area、原始数值列和官方LGBO目标定义为准；不解释为反应产率。',
        '- 使用官方LGBO均值倾斜组件、官方LLINBO Justify分支和Reasoning BO置信排序组件；统一底座与5选1等必要适配详见protocol.json，不能等同于论文默认协议。',
        '- STY和峰面积使用固定工程尺度13000及10000归一化。KF噪声地板、事件阈值沿用既定归一化常数，没有根据本轮成绩调参；95%产率停止检验阈值不适用于这两项指标，故关闭。',
        '- 完全同分使用对照信息、潜在SD和独立随机优先级；候选编号不参与数值排序。普通反例仅在固定池存在精确单轴邻居时执行，计入16次预算，越过普通门限的情况显式记录。',
        '- 单初始化结果只说明这两条任务上的有限池表现，不提供统计显著性或普遍优势证据。','',
        '来源：[Summit SnAr](https://gosummit.readthedocs.io/en/latest/experiments_benchmarks/implemented_benchmarks.html)、[Olympus HPLC固定版本](https://github.com/aspuru-guzik-group/olympus/tree/440b6b58ebfcaa2391cff7e94b570fb4fda98d68/src/olympus/datasets/dataset_hplc)、[LGBO数据协议](https://github.com/Xinzhe309/LGBO/blob/main/docs/benchmark_data.md)。','']
    (s.OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    plot(ss)
    return {'status':'SAVED','report':str((s.OUT/'REPORT.md').resolve()),'metrics':rows}

def plot(ss):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#777777','#d28b26','#64a191','#8c6bb1','#b78ac6','#2876b8','#a75151','#d32f2f']
    for task in s.TASKS:
        fig,axs=plt.subplots(1,2,figsize=(13.2,4.6),layout='constrained');p=s.pool(task)
        for m,color in zip(e.METHODS,colors):
            state=next(r for r in ss if r['task']==task and r['method']==m);y=np.array([h['y']*p['objective_scale'] for h in state['history']]);lw=2.6 if m=='New_Agent' else 1.3
            for ax,vals in zip(axs,(y,np.maximum.accumulate(y))):
                ax.plot(np.arange(1,21),vals,label=m,color=color,linewidth=lw,marker='o' if m=='New_Agent' else None,markersize=3,alpha=.95 if m=='New_Agent' else .8)
                ax.axvspan(.5,4.5,color='#dddddd',alpha=.12);ax.set_xlabel('Total experiment (4 initialization + 16 queries)');ax.set_ylabel(p['objective_unit']);ax.set_xticks([1,4,8,12,16,20]);ax.grid(alpha=.2)
        axs[0].set_title(task+' | Observed objective');axs[1].set_title(task+' | Cumulative best')
        handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,fontsize=8)
        fig.savefig(s.OUT/f'{task}_comparison.png',dpi=180);fig.savefig(s.OUT/f'{task}_comparison.pdf');plt.close(fig)

def inspect(task,method,t):
    assert 1<=t<=16
    state=data.read(s.ROOT/'runs'/(task+'_'+method)/'state.json');event=state['decisions'][t-1]
    path=s.ROOT/'runs'/state['run_id']/'rounds'/f'{t:02d}_committed.json'
    return {'decision':event,'committed_prequery':str(path.resolve()),'all_candidates':str(path.with_name(f'{t:02d}_candidates.npz').resolve()),
            'debug_CSV':str((s.OUT/task/'New_Agent_candidate_tables'/f'{t:02d}_candidates.csv').resolve()) if method=='New_Agent' else None}
