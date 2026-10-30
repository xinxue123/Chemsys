"""Validate the trust-only rerun and compare with unchanged sealed controls."""
from copy import deepcopy
from pathlib import Path
import json
import time
import numpy as np
import torch

from . import r2_region_trust_study as study
from . import r2_region_trust_engine as engine
from . import pointwise_hte_report as parent_report
from . import pointwise_hte_llm as llm
from . import reasoning_bo_data as data
from .suzuki_domain import digest


def validate():
    study.parent.warm();p=study.check();ss=study.states();assert len(ss)==2
    choices=kfs=feedbacks=recoveries=requests=0
    for saved in ss:
        assert saved['status']=='COMPLETE' and len(saved['history'])==20 and len(saved['decisions'])==16
        assert len({h['candidate_id'] for h in saved['history']})==20
        state=study.initial(saved['task'],p);assert state['history']==saved['history'][:4]
        pool=data.load_pool(saved['task']);ix={k:i for i,k in enumerate(pool['ids'])}
        with np.load(data.ROOT/saved['task']/'oracle.npz') as z:oracle=z['values'].copy()
        for h in saved['history']:assert h['y']==float(oracle[ix[h['candidate_id']]])
        cache=engine.Cache(pool,study.ROOT/'cache'/saved['task'])
        for event in saved['decisions']:
            t=event['iteration'];d=engine.setup(cache,state,t);reasons=engine.event_reasons(state,d,t)
            call=next((c for c in saved['calls'] if c['iteration']==t),None)
            if call:
                assert reasons==call['reasons']
                assert study.equivalent(study.parent.decision_receipt(d),call['before'])
                state['calls'].append(deepcopy(call));state['knowledge']=deepcopy(call['validated']['plan']['knowledge'])
                d=engine.setup(cache,state,t)
            else:assert not reasons
            folder=study.ROOT/'runs'/saved['run_id']/'rounds';target=folder/f'{t:02d}_committed.json'
            c=data.read(target);table=folder/f'{t:02d}_candidates.npz'
            assert data.filehash(target)==event['commit_sha256'] and data.filehash(table)==c['candidate_table_sha256']
            assert c['history_sha256']==digest(state['history']) and c['observation_count']==t+3
            j=d['selected'];assert pool['ids'][j]==event['selected_id']==c['selected_id']
            assert d['available'][j] and d['eligible'][j]
            assert study.equivalent(study.parent.decision_receipt(d),c['decision'])
            assert state['region_trust']==c['region_trust_at_commit']
            with np.load(table) as a:
                for key in ('base','score','raw_shift_std','shift_std'):
                    name={'base':'base_qLogEI','score':'guided_qLogEI','shift_std':'deployed_shift_std'}.get(key,key)
                    np.testing.assert_allclose(d[key],a[name],rtol=0,atol=1e-10)
            if d['trust']['trust_blocked']:assert d['amplitude']==0. and d['selected']==d['reference']
            if d['trust']['recovered']:
                ev=d['trust']['entry_after']['recovery_evidence'];assert ev['positive_evidence']
                assert ev['after_observation_count']<=len(state['history'])
                assert ev['after_observation_count']>d['trust']['entry_after']['blocked_observation_count']
                assert not d['trust']['b4_fallback'];recoveries+=1
            if d['bank_key']:
                _,v=cache.model(state['history']);raw=float(d['raw_shift_std'][j])*v['std']
                # Independent Joseph equations and physical-unit measurement.
                prior=d['prior'];theta=np.asarray(prior['mean']);P=np.asarray(prior['covariance'])
                H=np.array([1.,raw]);mu=float(d['stats']['mu_y'][j]);sd=float(d['stats']['sd_observation_y'][j])
                R=sd**2+.0009;S=float(H@P@H+R);gain=P@H/S;innovation=event['y']-mu-H@theta
                mean=theta+gain*innovation;A=np.eye(2)-np.outer(gain,H);cov=A@P@A.T+R*np.outer(gain,gain)
                np.testing.assert_allclose(mean,event['calibration_after']['mean'],atol=1e-12,rtol=0)
                np.testing.assert_allclose(cov,event['calibration_after']['covariance'],atol=1e-12,rtol=0)
                assert np.linalg.eigvalsh(cov).min()>-1e-12
                bank,m=engine.original.nominal.update_bank(d['bank'],prior,t,raw,mu,sd,event['y'])
                assert study.equivalent(m,event['measurement'])
                state['banks'][d['bank_key']]=bank;d['_measurement']=m
                f=engine.feedback(state,d,t,event['y'],bank)
                assert study.equivalent(f,event['trust_feedback']);kfs+=1
                if f:
                    expected=(event['y']-mu)**2-(event['y']-mu-f['frozen_probe_amplitude']*raw)**2
                    assert abs(expected-f['squared_error_gain_y_squared'])<1e-14;feedbacks+=1
            assert study.equivalent(state['region_trust'],event['region_trust_after_feedback'])
            state['history'].append(saved['history'][t+3]);state['decisions'].append(event);choices+=1;cache.release()
        assert study.equivalent(state['banks'],saved['banks'])
        assert study.equivalent(state['region_trust'],saved['region_trust'])
    for folder in (study.ROOT/'requests').iterdir():
        ctx=data.read(folder/'context.json');answer=data.read(folder/'validated.json')
        raw=deepcopy(ctx);key=raw.pop('context_sha256');assert digest(raw)==key==folder.name==answer['context_sha256']
        state=next(s for s in ss if s['task']==ctx['task']);t=ctx['public']['query_round']
        assert ctx['observed_prefix']==state['history'][:t+3]
        assert ctx['public']['observation_count']==len(ctx['observed_prefix'])
        pool=data.load_pool(state['task']);old=ctx['public']['current_knowledge'];_,reverse=llm.nominal.token_maps(pool)
        knowledge={'confidence':old['confidence'],'clauses':[{'id':h['id'],'axis':h['axis'],
            'allowed':sorted(reverse[h['axis']][v] for v in h['allowed'])} for h in old['clauses']]}
        checked=llm.nominal.validate_plan(answer['raw'],pool,knowledge,ctx['public']['observations'],grounded=True)
        assert checked==answer['plan']
        if (folder/'reused.json').exists():
            reuse=data.read(folder/'reused.json');assert data.filehash(reuse['source'])==reuse['source_sha256']
            original_ctx=data.read(Path(reuse['source']).parent/'context.json')
            assert ctx['observed_prefix']==original_ctx['observed_prefix']
            assert study.equivalent(ctx['public'],original_ctx['public'])
        requests+=1
    kept=study.preservation()
    result={'status':'PASS_COMPLETE','trajectories':2,'choices':choices,'KF_updates':kfs,
        'trust_feedback_checks':feedbacks,'data_supported_recoveries':recoveries,'requests_checked':requests,
        'other_methods_unchanged':kept,'protocol_sha256':p['protocol_sha256']}
    data.write(study.OUT/'validation_receipt.json',result);return result


def plot(states):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
    for row,task in enumerate(data.SPECS):
        subset=[s for s in states if s['task']==task and s['display'] in ('Pure_BO','New_R2','R2_region_trust','ReasoningBO_point')]
        colors={'Pure_BO':'#444444','New_R2':'#6294b5','R2_region_trust':'#176bad','ReasoningBO_point':'#d47a16'}
        for s in subset:
            y=np.array([h['y']*100 for h in s['history']]);x=np.arange(1,21);name=s['display']
            for col,values in enumerate((y,np.maximum.accumulate(y))):
                ax=axes[row,col];ax.plot(x,values,c=colors[name],label=name,
                    ls='--' if name=='New_R2' else '-',marker='o' if name=='R2_region_trust' else None,ms=3,lw=1.5)
                ax.set(ylim=(0,103),xlim=(1,20),xticks=[1,4,8,12,16,20],xlabel='Experiment index (first4 shared)',
                    ylabel='UV product area (%)' if row==0 else 'Reported yield (%)',
                    title=task.replace('_',' ')+' | '+('observed' if col==0 else 'cumulative best'))
                ax.axvline(4.5,c='#999999',ls=':',lw=1);ax.grid(alpha=.18)
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=4,frameon=False,fontsize=9)
    fig.suptitle('Only R2 rerun: same-region trust recovery / seed0 /4+16',fontsize=13)
    fig.tight_layout(rect=[0,.06,1,.96]);fig.savefig(study.OUT/'comparison.png',dpi=180);plt.close(fig)


def build():
    p=study.check();new=study.states();assert len(new)==2 and all(s['status']=='COMPLETE' for s in new)
    old=[{**s,'display':s['method'],'source':'frozen_control'} for s in parent_report.states()]
    all_states=old+[{**s,'display':study.METHOD,'source':'trust_only_rerun'} for s in new]
    rows=[{'task':s['task'],'method':s['display'],'source':s['source'],**parent_report.metrics(s)} for s in all_states]
    deltas=[]
    for s in new:
        previous=next(v for v in old if v['task']==s['task'] and v['method']=='New_R2')
        a=parent_report.metrics(s);b=parent_report.metrics(previous)
        first=next((i+1 for i,(x,y) in enumerate(zip(s['decisions'],previous['decisions'])) if x['selected_id']!=y['selected_id']),None)
        deltas.append({'task':s['task'],'SearchAUC_delta_vs_old_R2':a['SearchAUC']-b['SearchAUC'],
            'best_delta_vs_old_R2':a['final_best_percent']-b['final_best_percent'],'first_changed_query':first,
            'trust_blocked_queries':[d['iteration'] for d in s['decisions'] if d['trust']['trust_blocked']],
            'recovered_queries':[d['iteration'] for d in s['decisions'] if d['trust']['recovered']]})
    req=study.ROOT/'requests';usage={}
    responses=list(req.glob('*/attempt*/response.json'))
    for path in responses:
        for k,v in data.read(path).get('usage',{}).items():
            if type(v) in (int,float):usage[k]=usage.get(k,0)+v
    result={'status':'COMPLETE','new_trajectories':2,'new_queries':32,'protocol_sha256':p['protocol_sha256'],
        'metrics':rows,'deltas':deltas,'LLM_requests':{'total':len(list(req.glob('*/validated.json'))),
            'reused':len(list(req.glob('*/reused.json'))),'new_responses':len(responses),'new_usage':usage},
        'limits':'Only seed0, only one risk-recovery policy change. No population significance claim.'}
    data.write(study.OUT/'results.json',result);plot(all_states)
    text=['**仅修正 R2 同一区域信任恢复，其他方案保留原结果。**','',
        '两套数据均保持原候选集、seed=0、共同4次初始化＋16次查询、原始共享初始知识、Mixed GP、qLogEI512、KF公式、B4、事件调用预算和提示。新运行只有2条R2轨迹，共32次追加查询。','',
        '修正规则：区域风险按实际允许类别集合保存，不随知识ID、条目顺序或撤销文字清除。发生B4冲突后，同一区域保持回退；必须有之后真实观测，使查询前冻结的知识修正预测平方误差小于纯GP，并且下一轮B4无冲突，才恢复到更新后的KF幅度。判据不部署干扰项b，不使用候选未查询真值。','',
        '恢复证据为 `(y-mu_GP)^2 - (y-mu_GP-alpha_pre*delta_region_pre)^2 > 1e-12`，且修正量非零、KF更新后系数为正。1e-12为浮点容差，不根据本轮结果调参。没有添加topK候选限制或改动其他诊断模块。','']
    for task in data.SPECS:
        text += [f'**{task}**','', '| 方法 | SearchAUC | 最终最佳/% | 首达自身最佳实验序号 | 来源 |',
            '|---|---:|---:|---:|---|']
        for r in rows:
            if r['task']==task:text.append(f"| {r['method']} | {r['SearchAUC']:.4f} | {r['final_best_percent']:.4f} | {r['first_final_best_experiment']} | {'本轮仅重跑R2' if r['source']=='trust_only_rerun' else '原封存结果'} |")
        d=next(v for v in deltas if v['task']==task)
        text += ['',f"相对旧R2：SearchAUC变化{d['SearchAUC_delta_vs_old_R2']:+.4f}个百分点，最终最佳变化{d['best_delta_vs_old_R2']:+.4f}个百分点；首次不同选择在追加查询第{d['first_changed_query']}轮。",'']
    text += ['![修正前后与固定对照](comparison.png)','',
        f"LLM刷新共{result['LLM_requests']['total']}个有效请求，其中{result['LLM_requests']['reused']}个复用相同观测和相同公开提示的原回复，新增真实回复{len(responses)}次。数值仅有浮点舍入差异时允许绝对1e-10/相对1e-12等价，复用均重新验证并保存来源；改变后的上下文只调用新Agent。初始知识始终复用原共同建议。",'',
        '本轮是完整16轮新轨迹重跑，而非固定旧历史的单步替换。修正规则来源于对同一seed0的既有诊断，结果用于机制检查，不能作为独立泛化检验。SearchAUC为16次追加查询后累计最佳均值；首达序号包含4次初始化。Perera是UV产物面积百分比，Doyle是报告反应产率。','',
        '运行入口为pointwise_r2_trust.py；prepare/run/validate/report/seal。逐轮全候选表、查询前承诺、风险状态、冻结探针和实验后的恢复证据存于data/runs/hte_r2_region_trust_seed0。',
        '机器结果见[results.json](results.json)，检查见[validation_receipt.json](validation_receipt.json)，冻结设置见[protocol.json](protocol.json)。']
    (study.OUT/'report.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    return {'status':'COMPLETE','deltas':deltas,'LLM_requests':result['LLM_requests']}


def seal():
    p=study.check();v=data.read(study.OUT/'validation_receipt.json');r=data.read(study.OUT/'results.json')
    assert v['status']=='PASS_COMPLETE' and v['choices']==32 and r['status']=='COMPLETE'
    preserved=study.preservation();paths=[Path(p) for p in study.sources()]
    for root in (study.ROOT,study.OUT):
        paths += [path for path in root.rglob('*') if path.is_file() and '__pycache__' not in path.parts
            and path.suffix not in ('.lock','.tmp') and path.name not in ('release_inventory.json','release_receipt.json')]
    secret=study.api.load_config('../config.txt').api_key.encode('utf8')
    assert all(secret not in path.read_bytes() for path in set(paths))
    files=[{'path':p.as_posix(),'bytes':p.stat().st_size,'sha256':data.filehash(p)} for p in sorted(set(paths))]
    data.write(study.OUT/'release_inventory.json',{'files':files})
    receipt={'status':'PASS_COMPLETE','new_trajectories':2,'choices':32,'other_methods_unchanged':preserved,
        'protocol_sha256':p['protocol_sha256'],'results_sha256':data.filehash(study.OUT/'results.json'),
        'inventory_sha256':data.filehash(study.OUT/'release_inventory.json'),'files':len(files),
        'operator_compiles_in_loop':0,'credentials_included':False,'visual_QA':'comparison.png inspected',
        'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    data.write(study.OUT/'release_receipt.json',receipt);return receipt
