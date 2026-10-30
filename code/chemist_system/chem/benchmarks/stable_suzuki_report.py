"""Read-only archived controls; all new choices, KF steps and contexts audited."""
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import csv
import json
import numpy as np
from . import stable_suzuki_study as s
from . import stable_suzuki_engine as e
from . import stable_suzuki_policy as policy
from . import kalman_knowledge_suzuki_report as marginal
from .suzuki_domain import digest


def states():return [s.read(p) for p in sorted((s.ROOT/'runs').glob('*/state.json'))]


def metrics(state):
    y=np.array([h['y']*100 for h in state['history']]);best=np.maximum.accumulate(y)
    return {'case':state['case'],'regime':state['regime'],'method':'Stable_revised','status':state['status'],
        'SearchAUC':float(best[4:].mean()),'final_best':float(best[-1]),
        'first_own_best_total':int(np.flatnonzero(y==best[-1])[0]+1),
        'first_own_best_added':max(0,int(np.flatnonzero(y==best[-1])[0]-3)),
        'refreshes':len(state.get('calls',[])),'controlled_tests':len(state.get('validations',[])),
        'B4_conflicts':sum(d.get('trust',{}).get('b4_fallback',False) for d in state['decisions']),
        'trust_blocked_rounds':sum(d.get('trust',{}).get('trust_blocked',False) for d in state['decisions']),
        'gate_exceptions':sum(d.get('acquisition_gate_exception',False) for d in state['decisions'])}


def write_csv(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def audit_case(case):
    s.warm();pool=s.prior.inputs(case);ix={k:i for i,k in enumerate(pool['ids'])};cache=s.cache_for(case,pool)
    choices=kfs=contexts=score_values=0;max_score_error=max_filter_error=0.;checked_contexts=set()
    max_relabel_error=0.
    for state in [r for r in states() if r['case']==case]:
        assert state['status']=='COMPLETE' and len(state['decisions'])==16 and len(state['history'])==20
        assert len({h['candidate_id'] for h in state['history']})==20
        assert state['history'][:4]==s.prior.initial(case,pool)
        assert state['initial_knowledge']==s.grounded.previous.initial_knowledge(case,state['regime'])
        assert len(state['calls'])<=3
        for t,event in enumerate(state['decisions'],1):
            prefix=state['history'][:t+3];folder=s.ROOT/'runs'/state['run_id']/'rounds'
            commit_path=folder/f'{t:02d}_committed.json';c=s.read(commit_path)
            assert s.filehash(commit_path)==event['commit_sha256'] and c['history']==prefix and digest(prefix)==c['history_sha256']
            table=folder/f'{t:02d}_candidates.npz';assert s.filehash(table)==c['candidate_table_sha256']
            with np.load(table) as f:d={k:f[k].copy() for k in f.files}
            base,score=d['base_qLogEI'],d['guided_qLogEI'];j=ix[event['selected_id']]
            available=~np.isin(pool['ids'],[h['candidate_id'] for h in prefix]);assert np.array_equal(available,d['available'])
            budget=float(np.log(4)*(16-t)/15);eligible=available&(base[available].max()-base<=budget)
            assert event['budget']==budget and np.array_equal(eligible,d['eligible'])
            assert available[j] and event['acquisition_gate_exception']==(not bool(eligible[j]))
            proxy,_,_=e.information_proxy(cache.z.numpy(),[ix[h['candidate_id']] for h in prefix])
            assert np.array_equal(proxy,d['information_proxy'])
            ranks=e.priority(len(base),t,0);assert np.array_equal(ranks,d['tie_priority'])
            def choose(scores,mask):
                js=np.flatnonzero(mask);js=js[scores[js]==scores[js].max()]
                js=js[proxy[js]==proxy[js].max()];sd=d['sd_y'][js];js=js[sd==sd.max()]
                return int(js[np.argmin(ranks[js])])
            ref=choose(base,available);assert pool['ids'][ref]==event['reference_id']
            assert pool['ids'][choose(score,eligible)]==event['normal_proposal_id']
            dg=e.original.diagnose(base,pool,c['knowledge'],budget,~available)
            assert list(dg.retained_ids)==event['diagnosis']['retained_ids']
            assert list(dg.revoked_ids)==event['diagnosis']['revoked_ids']
            assert dg.conflict_kind==event['diagnosis']['conflict_kind']
            if c['validation']:
                val=c['validation'];own={**state,'history':prefix,'knowledge':c['knowledge'],'decisions':state['decisions'][:t-1],
                    'validations':[v for v in state['validations'] if v['query']<t]}
                prepared=policy.build_candidates(pool,own,{'base':base,'available':available,'eligible':eligible,
                    'reference':ref,'stats':{'mu_y':d['mu_y'],'sd_y':d['sd_y']}})
                assert prepared==val['prepared']
                assert policy.validate(val['validated']['raw'],prepared,prefix)==val['validated']['plan']
                assert val['validated']['plan']['candidate_id']==event['selected_id']
            else:assert j==choose(score,eligible) and eligible[j]
            _,v=cache.model(prefix)
            rb=marginal.marginal_scores(d['mu_std'],d['sd_std'],np.zeros(len(base)),v['best_std'],c['base_meta']['mc_seed'],cache.z)
            rg=marginal.marginal_scores(d['mu_std'],d['sd_std'],d['deployed_shift_std'],v['best_std'],c['base_meta']['mc_seed'],cache.z)
            err=max(float(np.max(np.abs(rb-base))),float(np.max(np.abs(rg-score))));max_score_error=max(max_score_error,err)
            assert err<1e-7,(case,state['regime'],t,err)
            # Full-pool nominal relabeling at every observed prefix, no re-fit.
            from .mixed_kalman_suzuki_report import permutation_posterior
            pm,ps=permutation_posterior(cache,prefix,v['model'],v)
            pe=max(float(np.max(np.abs(pm-d['mu_std']))),float(np.max(np.abs(ps-d['sd_std']))))
            max_relabel_error=max(max_relabel_error,pe);assert pe<1e-9
            if c['bank_before'] is not None:
                prior=c['bank_prior'];theta=np.array(prior['mean']);P=np.array(prior['covariance']);H=np.array([1.,float(d['raw_shift_std'][j])*c['scale_y']])
                y=state['history'][t+3]['y'];R=float(d['sd_predictive_y'][j]**2+.0009);innovation=y-d['mu_y'][j]-H@theta
                gain=P@H/(H@P@H+R);mean=theta+gain*innovation;cov=P-np.outer(P@H,H@P)/(H@P@H+R)
                fe=max(float(np.max(np.abs(mean-np.array(event['bank_after']['mean'])))),float(np.max(np.abs(cov-np.array(event['bank_after']['covariance'])))))
                max_filter_error=max(max_filter_error,fe);assert fe<1e-12;kfs+=1
            assert c['trust']['amplitude_after_trust']==event['amplitude']
            if event['trust']['trust_blocked']:assert event['amplitude']==0. and np.array_equal(score,base)
            if event['trust']['recovered']:
                ev=event['trust']['entry_after']['recovery_evidence']
                assert ev['positive_evidence'] and ev['before_observation_count']>=event['trust']['entry_before']['blocked_observation_count']
            choices+=1;score_values+=2*len(base);cache.release()
        for call in state['calls']:
            key=call['context_sha256'];folder=s.ROOT/'requests'/key;c=s.read(folder/'context.json');answer=s.read(folder/'validated.json')
            assert c['observed_prefix']==state['history'][:call['iteration']+3]
            assert len(c['public']['observations'])==len(c['observed_prefix'])
            plan=s.grounded.e.validate_refresh(answer['raw'],call['knowledge_before'],state['slots'],c['public']['observations'],pool)
            assert json.loads(json.dumps(plan))==answer['plan'];checked_contexts.add(key);contexts+=1
    return {'case':case,'choices_verified':choices,'KF_updates_verified':kfs,'refresh_contexts_verified':contexts,
        'qLogEI_scores_recomputed_from_moments':score_values,'max_qLogEI_error':max_score_error,
        'max_KF_error':max_filter_error,'max_category_relabel_error':max_relabel_error,
        'operator_compiles_in_loop':sum(s.prior.previous.JIT.values())}


def validate():
    p=s.check();assert s.status()['complete']==16
    with ProcessPoolExecutor(max_workers=4) as executor:audits=list(executor.map(audit_case,(1,2,3,4)))
    protected=s.read(s.OUT/'protected_snapshot.json');assert digest(protected)==p['protected_snapshot_sha256']
    for f,h in protected.items():assert s.filehash(f)==h,f
    assert sum(a['choices_verified'] for a in audits)==256
    receipt={'status':'PASS','protocol_sha256':p['protocol_sha256'],'completed':16,'additional_queries':256,
        'protected_files_unchanged':len(protected),'audits':audits,'scope':'technical execution validation; no performance acceptance claim'}
    s.write(s.OUT/'validation_receipt.json',receipt);return receipt


def report():
    p=s.check();ss=states();assert len(ss)==16 and all(r['status']=='COMPLETE' for r in ss)
    current=[metrics(r) for r in ss]
    archived=s.read('research/suzuki_grounded_regional/results.json')['raw_metrics']
    rows=[*deepcopy(archived),*current]
    # Add both historical B4 arms, omitted by the previous narrative comparison.
    for c in (1,2,3,4):
        for regime in s.REGIMES:
            for name,label in [('B4_base','B4_hard'),('B4_LGBO','B4_soft')]:
                st=s.read(s.prior.ROOT/'runs'/f'c{c}_s0_{regime}_{name}'/'state.json')
                row=metrics(st);row.update(method=label,refreshes=0,controlled_tests=0,B4_conflicts=None,trust_blocked_rounds=None,gate_exceptions=0)
                rows.append(row)
    write_csv(s.OUT/'comparison_all.csv',rows);write_csv(s.OUT/'metrics.csv',current)
    traces=[]
    for state in ss:
        pool=s.prior.inputs(state['case']);ix={k:i for i,k in enumerate(pool['ids'])}
        for event in state['decisions']:
            j=ix[event['selected_id']];t=event['iteration']
            with np.load(s.ROOT/'runs'/state['run_id']/'rounds'/f'{t:02d}_candidates.npz') as d:
                gp=float(d['mu_y'][j]*100);sd=float(d['sd_y'][j]*100)
            traces.append({'case':state['case'],'regime':state['regime'],'query_round':t,'total_experiment':t+4,
                'candidate_id':event['selected_id'],**pool['points'][j],'yield_percent':event['y']*100,
                'incumbent_percent':event['incumbent_after']*100,'GP_mean_percent':gp,'GP_latent_SD_pp':sd,
                'base_qLogEI':event['base_qLogEI'],'guided_qLogEI':event['guided_qLogEI'],
                'raw_region_shift_pp':event['raw_shift_y']*100,'deployed_shift_pp':event['deployed_shift_y']*100,
                'KF_amplitude':event['amplitude'],'trust_blocked':event['trust']['trust_blocked'],
                'B4_conflict':event['trust']['b4_fallback'],'gate_eligible_count':event['eligible_count'],
                'gate_budget':event['budget'],'base_opportunity_loss':event['base_opportunity_loss'],
                'gate_exception':event['acquisition_gate_exception'],'execution_mode':event['execution_mode'],
                'refresh_before_choice':any(c['iteration']==t for c in state['calls'])})
    write_csv(s.OUT/'query_trace.csv',traces)
    api_usage={'received_responses':0,'prompt_tokens':0,'completion_tokens':0,'response_models':set(),'fingerprints':set()}
    for f in (s.ROOT/'requests').glob('*/attempt*/response.json'):
        a=s.read(f);u=a.get('usage',{});api_usage['received_responses']+=1
        for k in ('prompt_tokens','completion_tokens'):api_usage[k]+=u.get(k,0)
        api_usage['response_models'].add(a.get('model'));api_usage['fingerprints'].add(a.get('system_fingerprint'))
    for k in ('response_models','fingerprints'):api_usage[k]=sorted(v for v in api_usage[k] if v is not None)
    summary={}
    for regime in s.REGIMES:
        selected=[r for r in current if r['regime']==regime]
        summary[regime]={'SearchAUC_mean':float(np.mean([r['SearchAUC'] for r in selected])),
            'best_mean':float(np.mean([r['final_best'] for r in selected])),
            'delta_AUC_vs_Pure_BO':float(np.mean([r['SearchAUC']-next(a['SearchAUC'] for a in archived if a['case']==r['case'] and a['regime']==regime and a['method']=='Pure_BO') for r in selected]))}
    result={'protocol_sha256':p['protocol_sha256'],'planned':16,'completed':16,'raw_metrics':current,
        'comparison_rows':rows,'by_knowledge_condition':summary,'API_usage':api_usage,
        'refreshes':sum(len(st['calls']) for st in ss),'controlled_tests':sum(len(st['validations']) for st in ss),
        'validation_checks':[{'case':st['case'],'regime':st['regime'],**c} for st in ss for c in st['validation_checks']],
        'gate_exceptions':sum(r['gate_exceptions'] for r in current),
        'limits':['single seed0 initialization','archived baseline tie rule retained','covariance-unit repair and policy transfer are joint changes',
                  'anonymous catalysts prohibit molecular recovery and two-category joint tests','three constructed stress priors are reference-outcome-aware']}
    s.write(s.OUT/'results.json',result)
    lines=['# 修订版策略：Reizman Suzuki Case 1–4','',
        '已完成16/16条轨迹：四案例×四知识条件，seed=0，原4692候选，4次初始化＋16次查询。旧方案均读取冻结结果，没有重跑。',
        '', 'SearchAUC为16次追加实验后累计最佳产率的均值；首次最佳按包含初始化的第1–20次实验计。', '']
    names={'correct':'正确知识（人工压力测试）','wrong_category':'错误类别（人工压力测试）','mixed':'混合错误（人工压力测试）','assistant_initial':'助手初始建议'}
    methods=['Stable_revised','R2_conflict_fallback','R1_grounded','G3_original','Gate_only','LGBO','LLINBO_Justify','Pure_BO','B4_hard','B4_soft']
    for case in (1,2,3,4):
        lines+=['',f'## Case {case}','', '| 知识条件 | 方法 | SearchAUC | 最佳产率/% | 首次最佳/实验次序 |','|---|---|---:|---:|---:|']
        for regime in s.REGIMES:
            for method in methods:
                r=next(v for v in rows if v['case']==case and v['regime']==regime and v['method']==method)
                lines.append(f"| {names[regime]} | {method} | {r['SearchAUC']:.3f} | {r['final_best']:.3f} | {r['first_own_best_total']} |")
    lines+=['','## 实施差异与证据边界','',
        '沿用修订版的同区域信任闩锁、冻结预测误差改善后恢复、KF、B4、事件刷新、qLogEI预算门限及同分随机规则。区域协方差缓存统一到GP后验的输出单位；这与旧Suzuki区域代码不同，因此不是单一信任模块消融。',
        '连续变量保持连续核，匿名催化剂只做相等性比较。没有可用配体结构，且只有一个类别轴，故分子结构核、双类别联合反例与其局部跟进不适用。保留原候选集，不生成新条件，不把近邻冒充单轴对照。',
        f"实际刷新{result['refreshes']}次，精确单轴受控试验{result['controlled_tests']}次，明确记录的门限外试验{result['gate_exceptions']}次；机会预算仍共16次。完整跳过原因见results.json。",
        '四种初始知识逐字复用旧实验。正确、错误类别与混合错误是参考结果知情的人工压力条件，不能当作独立化学知识发现证据。',
        '所有性能结论限定为一个固定初始化。本轮没有检验初始化或LLM随机性的稳定性，未在Suzuki结果上继续调参。',
        '', '复现：` .venv-official/Scripts/python.exe stable_suzuki.py run `；状态：`stable_suzuki.py status`；校验：`stable_suzuki.py validate`。',
        '逐轮汇总：query_trace.csv；逐候选完整GP均值/潜在SD、基础与修正qLogEI、预算掩码：data/runs/suzuki_stable_20261029/runs/*/rounds/*_candidates.npz；提示词与真实响应：requests/*/attempt*/。']
    (s.OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return {'completed':16,'summary':summary,'refreshes':result['refreshes'],'controlled_tests':result['controlled_tests'],'API_usage':api_usage}


def release():
    p=s.check();receipt=s.read(s.OUT/'validation_receipt.json');assert receipt['status']=='PASS' and receipt['protocol_sha256']==p['protocol_sha256']
    files=set(p['sources'])|set(p['input_sha256'])
    for root in (s.ROOT,s.OUT):
        files.update(f.as_posix() for f in root.rglob('*') if f.is_file() and f.name not in ('release_inventory.json','release_receipt.json') and not f.name.endswith('.lock'))
    inventory={'study':p['study'],'files':[{'path':f,'bytes':Path(f).stat().st_size,'sha256':s.filehash(f)} for f in sorted(files)]}
    s.write(s.OUT/'release_inventory.json',inventory)
    result={'status':'SAVED_AND_VERIFIED','files':len(inventory['files']),'inventory_sha256':s.filehash(s.OUT/'release_inventory.json'),
        'protocol_sha256':p['protocol_sha256'],'completed':16,'additional_queries':256}
    s.write(s.OUT/'release_receipt.json',result);return result
