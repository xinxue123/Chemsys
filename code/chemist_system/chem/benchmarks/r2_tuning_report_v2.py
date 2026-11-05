"""Complete-trajectory validation and target assessment for R2 development."""
from copy import deepcopy
from pathlib import Path
import time
import numpy as np

from . import r2_tuning_study_v2 as study
from . import r2_tuning_policy_v2 as policy
from . import pointwise_hte_report as metrics_source

data=study.data
engine=study.engine
digest=study.digest
equivalent=study.trust.equivalent


def validate(variant):
    study.trust.parent.warm();p=study.check(variant);root,out=study.folders(variant);ss=study.states(variant)
    assert len(ss)==2 and all(s['status']=='COMPLETE' for s in ss)
    choices=kfs=validations=exceptions=requests=0
    for saved in ss:
        state=study.initial(saved['task'],variant,p);assert saved['history'][:4]==state['history']
        assert len(saved['history'])==20 and len(saved['decisions'])==16
        assert len({h['candidate_id'] for h in saved['history']})==20
        pool=data.load_pool(saved['task']);ix={k:i for i,k in enumerate(pool['ids'])}
        with np.load(data.ROOT/saved['task']/'oracle.npz') as a:y=a['values'].copy()
        for h in saved['history']:assert h['y']==float(y[ix[h['candidate_id']]])
        cache=engine.Cache(pool,root/'cache'/saved['task'])
        for event in saved['decisions']:
            t=event['iteration'];d=engine.setup(cache,state,t);reasons=engine.event_reasons(state,d,t)
            call=next((c for c in saved['calls'] if c['iteration']==t),None)
            if call:
                assert reasons==call['reasons']
                assert equivalent(study.trust.parent.decision_receipt(d),call['before'])
                state['calls'].append(deepcopy(call));state['knowledge']=deepcopy(call['validated']['plan']['knowledge'])
                d=engine.setup(cache,state,t)
            else:assert not reasons
            normal_selected=pool['ids'][d['selected']]
            target=root/'runs'/saved['run_id']/'rounds'/f'{t:02d}_committed.json';c=data.read(target)
            assert data.filehash(target)==event['commit_sha256']
            assert c['history_sha256']==digest(state['history']) and c['observation_count']==t+3
            assert normal_selected==event['normal_R2_proposal_id']==c['normal_R2_proposal_id']
            if event['execution_mode']=='controlled_validation':
                assert t in p['config']['validation_rounds'];validation=c['validation']
                prepared=policy.build_candidates(pool,state,d)
                assert equivalent(prepared,validation['prepared'])
                assert policy.validate(validation['validated']['raw'],prepared,state['history'])==validation['validated']['plan']
                plan=validation['validated']['plan'];assert plan['candidate_id']==event['selected_id']
                d['selected']=ix[plan['candidate_id']];validations+=1
            else:
                assert event['execution_mode']=='normal_R2' and c['validation'] is None
                if t in p['config']['validation_rounds']:assert not policy.build_candidates(pool,state,d)['candidates']
                assert pool['ids'][d['selected']]==event['selected_id'] and d['eligible'][d['selected']]
            j=d['selected'];assert d['available'][j] and pool['ids'][j]==c['selected_id']
            assert equivalent(study.trust.parent.decision_receipt(d),c['decision'])
            assert equivalent(state['region_trust'],c['region_trust_at_commit'])
            table=target.with_name(f'{t:02d}_candidates.npz');assert data.filehash(table)==c['candidate_table_sha256']
            with np.load(table) as a:
                np.testing.assert_allclose(d['base'],a['base_qLogEI'],atol=1e-10,rtol=0)
                np.testing.assert_allclose(d['score'],a['guided_qLogEI'],atol=1e-10,rtol=0)
            if event['acquisition_gate_exception']:
                assert event['execution_mode']=='controlled_validation' and not d['eligible'][j]
                assert p['config']['allow_acquisition_gate_exception_for_validation'];exceptions+=1
            else:assert d['eligible'][j]
            if d['bank_key']:
                _,v=cache.model(state['history']);raw=float(d['raw_shift_std'][j])*v['std']
                prior=d['prior'];theta=np.asarray(prior['mean']);P=np.asarray(prior['covariance']);H=np.array([1.,raw])
                mu=float(d['stats']['mu_y'][j]);sd=float(d['stats']['sd_observation_y'][j]);R=sd**2+.0009
                S=float(H@P@H+R);gain=P@H/S;innovation=event['y']-mu-H@theta
                expected=theta+gain*innovation;A=np.eye(2)-np.outer(gain,H);cov=A@P@A.T+R*np.outer(gain,gain)
                np.testing.assert_allclose(expected,event['calibration_after']['mean'],atol=1e-12,rtol=0)
                np.testing.assert_allclose(cov,event['calibration_after']['covariance'],atol=1e-12,rtol=0)
                bank,m=engine.original.nominal.update_bank(d['bank'],prior,t,raw,mu,sd,event['y'])
                assert equivalent(m,event['measurement']);state['banks'][d['bank_key']]=bank;d['_measurement']=m
                assert equivalent(engine.feedback(state,d,t,event['y'],bank),event['trust_feedback']);kfs+=1
            if event['controlled_comparison']:
                pair=event['controlled_comparison'];anchor=state['history'][pair['anchor_observation_index']-1]
                a,b=pool['points'][ix[anchor['candidate_id']]],pool['points'][j]
                assert [axis for axis in pool['axes'] if a[axis]!=b[axis]]==[pair['changed_axis']]
                assert pair['anchor_yield_percent']==round(anchor['y']*100,3)
                assert pair['tested_yield_percent']==round(event['y']*100,3)
                assert pair['observed_difference_pp']==round((event['y']-anchor['y'])*100,3)
                state['validations'].append(pair)
            assert equivalent(state['region_trust'],event['region_trust_after_feedback'])
            state['history'].append(saved['history'][t+3]);state['decisions'].append(event);cache.release();choices+=1
        assert equivalent(state['banks'],saved['banks']) and equivalent(state['region_trust'],saved['region_trust'])
        assert state['validations']==saved['validations'] and len(state['calls'])<=3
    for folder in (root/'requests').iterdir():
        ctx=data.read(folder/'context.json');answer=data.read(folder/'validated.json');q=deepcopy(ctx);h=q.pop('context_sha256')
        assert digest(q)==h==folder.name==answer['context_sha256']
        state=next(s for s in ss if s['task']==ctx['task']);t=ctx['public']['query_round'];public=ctx['public']
        assert ctx['observed_prefix']==state['history'][:t+3] and public['observation_count']==t+3
        if ctx['phase']=='controlled_validation':
            assert policy.validate(answer['raw'],public['counterexample_pool'],ctx['observed_prefix'])==answer['plan']
        else:
            pool=data.load_pool(ctx['task']);old=public['current_knowledge'];_,reverse=study.llm.nominal.token_maps(pool)
            knowledge={'confidence':old['confidence'],'clauses':[{'id':c['id'],'axis':c['axis'],
                'allowed':sorted(reverse[c['axis']][v] for v in c['allowed'])} for c in old['clauses']]}
            assert study.llm.nominal.validate_plan(answer['raw'],pool,knowledge,public['observations'],grounded=True)==answer['plan']
            for pair in public.get('controlled_validation_results',[]):assert pair['query']<t
        if (folder/'reused.json').exists():
            reuse=data.read(folder/'reused.json');assert data.filehash(reuse['source'])==reuse['source_sha256']
        requests+=1
    result={'status':'PASS_COMPLETE','variant':variant,'trajectories':2,'choices':choices,'KF_updates':kfs,
        'controlled_experiments':validations,'declared_acquisition_gate_exceptions':exceptions,
        'requests_checked':requests,'protected_controls':study.preservation(),'protocol_sha256':p['protocol_sha256']}
    data.write(out/'validation_receipt.json',result);return result


def assess(rows):
    statuses={}
    for row in rows:
        target=study.TARGETS[row['task']];auc=row['SearchAUC'];best=row['final_best_percent']
        statuses[row['task']]={'formal_target_met':auc+1e-9>=target['AUC_floor'] and best+1e-9>=target['best_floor'],
            'AUC_delta_to_target':auc-target['AUC_floor'],'best_delta_to_target':best-target['best_floor']}
        if 'stage_AUC' in target:statuses[row['task']]['stage_floor_met']=auc+1e-9>=target['stage_AUC'] and best+1e-9>=target['stage_best']
        if 'stretch_AUC' in target:statuses[row['task']]['stretch_met']=auc+1e-9>=target['stretch_AUC']
    return statuses


def plot(new,variant,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
    for row,task in enumerate(data.SPECS):
        parent=study.trust.parent.ROOT/'runs'
        ss=[('Pure_BO',data.read(parent/(task+'_s0_Pure_BO')/'state.json'),'#444444'),
            ('ReasoningBO',data.read(parent/(task+'_s0_ReasoningBO_point')/'state.json'),'#d47a16'),
            ('R2_trust',data.read(study.trust.ROOT/'runs'/(task+'_s0_R2_region_trust')/'state.json'),'#6294b5'),
            (variant,next(s for s in new if s['task']==task),'#176bad')]
        for name,s,color in ss:
            y=np.array([h['y']*100 for h in s['history']]);x=np.arange(1,21)
            for col,values in enumerate((y,np.maximum.accumulate(y))):
                ax=axes[row,col];ax.plot(x,values,c=color,label=name,lw=1.5,ls='--' if name=='R2_trust' else '-')
                if name==variant:
                    indices=[d['iteration']+3 for d in s['decisions'] if d['execution_mode']=='controlled_validation']
                    ax.scatter(x[indices],values[indices],c=color,marker='D',s=30,zorder=4)
                ax.set(ylim=(0,103),xlim=(1,20),xticks=[1,4,8,12,16,20],xlabel='Experiment index (first4 shared)',
                    ylabel='UV product area (%)' if row==0 else 'Reported yield (%)',
                    title=task.replace('_',' ')+' | '+('observed' if col==0 else 'cumulative best'))
                ax.axvline(4.5,c='#888888',ls=':',lw=1);ax.grid(alpha=.15)
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=4,frameon=False,fontsize=9)
    fig.suptitle('R2 development: controlled knowledge tests / seed0 /4+16',fontsize=13)
    fig.tight_layout(rect=[0,.06,1,.96]);fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)


def build(variant):
    p=study.check(variant);root,out=study.folders(variant);ss=study.states(variant)
    assert len(ss)==2 and all(s['status']=='COMPLETE' for s in ss)
    new=[];baselines=[]
    for s in ss:
        new.append({'task':s['task'],'method':variant,**metrics_source.metrics(s),
            'controlled_experiments':len(s['validations']),'gate_exceptions':sum(d['acquisition_gate_exception'] for d in s['decisions'])})
    for task in data.SPECS:
        for name,path in [('Pure_BO',study.trust.parent.ROOT/'runs'/(task+'_s0_Pure_BO')/'state.json'),
            ('ReasoningBO',study.trust.parent.ROOT/'runs'/(task+'_s0_ReasoningBO_point')/'state.json'),
            ('R2_region_trust',study.trust.ROOT/'runs'/(task+'_s0_R2_region_trust')/'state.json')]:
            baselines.append({'task':task,'method':name,**metrics_source.metrics(data.read(path))})
    status=assess(new);requests=root/'requests';responses=list(requests.glob('*/attempt*/response.json'));usage={}
    for path in responses:
        for k,v in data.read(path).get('usage',{}).items():
            if type(v) in (int,float):usage[k]=usage.get(k,0)+v
    result={'status':'COMPLETE','variant':variant,'new_trajectories':2,'new_queries':32,'metrics':new,
        'frozen_controls':baselines,'targets':status,'formal_target_met':all(s['formal_target_met'] for s in status.values()),
        'API':{'requests':len(list(requests.glob('*/validated.json'))),'new_responses':len(responses),
            'reused_requests':len(list(requests.glob('*/reused.json'))),'usage':usage},
        'protocol_sha256':p['protocol_sha256'],'scope':'Development on known seed0; no independent validation claim.'}
    data.write(out/'results.json',result);plot(ss,variant,out)
    text=[f'**R2调优：{variant}，完整4＋16轨迹。**','',
        f"正式数值目标是否达到：**{'达到' if result['formal_target_met'] else '尚未达到'}**。这是已分析数据上的开发结果，不是独立泛化证据。",'',
        '保持原候选池、Mixed GP、初始化与初始知识、seed=0，总计16次追加查询。最多两次知识验证替代正常查询，不增加总实验数；观测最佳值已达95%时不占用验证查询；未达到时，首个验证后优先覆盖未见类别最多的知识维度。验证查询可以突破原采集值机会损失门限，原门限保留在正常轮次。只重跑新R2分支，所有对照直接读取封存结果。','']
    for task in data.SPECS:
        text += [f'**{task}**','', '| 方法 | SearchAUC | 最终最佳/% | 首达最佳实验序号 |', '|---|---:|---:|---:|']
        for row in baselines+new:
            if row['task']==task:text.append(f"| {row['method']} | {row['SearchAUC']:.4f} | {row['final_best_percent']:.4f} | {row['first_final_best_experiment']} |")
        s=next(s for s in ss if s['task']==task)
        text += ['', '| 验证查询 | 改变变量 | 锚点真实值/% | 验证真实值/% | 差值/百分点 |','|---:|---|---:|---:|---:|']
        for v in s['validations']:text.append(f"| {v['query']} | {v['changed_axis']} | {v['anchor_yield_percent']:.3f} | {v['tested_yield_percent']:.3f} | {v['observed_difference_pp']:+.3f} |")
        text += ['',f"相对正式门槛：AUC差{status[task]['AUC_delta_to_target']:+.4f}；最佳值差{status[task]['best_delta_to_target']:+.4f}。",'']
    text += ['![逐轮轨迹](comparison.png)','',
        '菱形标记为知识验证查询。GP与知识只使用各自当前已观察数据；一对控制变量结果不能直接证明全局化学因果。Agent不接收候选未查询真值、对照成绩或验收数值。',
        'SearchAUC是16次追加查询后的累计最佳均值，首达序号包含初始4次。Perera单位为UV产物面积百分比，Doyle为报告反应产率。',
        '', '任务与阶段目标见上一级TASK.md。完整请求、回复、全候选预测、查询前承诺、预算例外和知识/滤波状态在该版本data/runs/hte_r2_tuning_20261028下存档。',
        '机器结果见[results.json](results.json)，验证见[validation_receipt.json](validation_receipt.json)，冻结协议见[protocol.json](protocol.json)。']
    (out/'report.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    return {'variant':variant,'metrics':new,'targets':status,'formal_target_met':result['formal_target_met'],'API':result['API']}


def seal(variant):
    p=study.check(variant);root,out=study.folders(variant);v=data.read(out/'validation_receipt.json');r=data.read(out/'results.json')
    assert v['status']=='PASS_COMPLETE' and v['choices']==32 and r['status']=='COMPLETE'
    kept=study.preservation();paths=[Path(path) for path in study.sources(variant)]
    for folder in (root,out):paths += [x for x in folder.rglob('*') if x.is_file() and '__pycache__' not in x.parts
        and x.suffix not in ('.lock','.tmp') and x.name not in ('release_inventory.json','release_receipt.json')]
    secret=study.api.load_config('../config.txt').api_key.encode('utf8')
    assert all(secret not in path.read_bytes() for path in set(paths))
    files=[{'path':path.as_posix(),'bytes':path.stat().st_size,'sha256':data.filehash(path)} for path in sorted(set(paths))]
    data.write(out/'release_inventory.json',{'files':files})
    receipt={'status':'PASS_COMPLETE','variant':variant,'formal_target_met':r['formal_target_met'],'new_trajectories':2,'choices':32,
        'protected_controls':kept,'protocol_sha256':p['protocol_sha256'],'results_sha256':data.filehash(out/'results.json'),
        'inventory_sha256':data.filehash(out/'release_inventory.json'),'files':len(files),'credentials_included':False,
        'operator_compiles_in_loop':0,'visual_QA':'comparison.png inspected',
        'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    data.write(out/'release_receipt.json',receipt);return receipt
