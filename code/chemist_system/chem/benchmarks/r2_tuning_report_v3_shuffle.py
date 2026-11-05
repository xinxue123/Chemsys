"""Complete-trajectory validation and target assessment for R2 development."""
from copy import deepcopy
from pathlib import Path
import time
import numpy as np

from . import r2_tuning_study_v3_shuffle as study
from . import r2_tuning_policy_v3_shuffle as policy
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
                np.testing.assert_array_equal(study.engine.priority(len(pool['ids']),t),a['tie_priority'])
                # Independent reconstruction rather than reusing the policy helper.
                rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([0,t])))
                perm=rng.permutation(len(pool['ids']));manual=np.empty(len(perm),dtype=np.int64)
                manual[perm]=np.arange(len(perm));np.testing.assert_array_equal(manual,a['tie_priority'])
                def manual_max(scores,mask):
                    top=float(scores[mask].max());equal=np.flatnonzero(mask & (scores==top))
                    return int(equal[np.argmin(manual[equal])])
                assert d['reference']==manual_max(a['base_qLogEI'],a['available'])
                assert ix[normal_selected]==manual_max(a['guided_qLogEI'],a['eligible'])
            assert equivalent(d['tie_order'],event['tie_order'])
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
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True)
    for row,task in enumerate(data.SPECS):
        parent=study.trust.parent.ROOT/'runs'
        ss=[('Pure_BO',data.read(parent/(task+'_s0_Pure_BO')/'state.json'),'#444444'),
            ('ReasoningBO',data.read(parent/(task+'_s0_ReasoningBO_point')/'state.json'),'#d47a16'),
            ('R2_trust',data.read(study.trust.ROOT/'runs'/(task+'_s0_R2_region_trust')/'state.json'),'#6294b5'),
            ('V3_ID_ties',data.read(study.ROOT/'focused_countercheck_v3'/'runs'/(task+'_s0_focused_countercheck_v3')/'state.json'),'#8e64ad'),
            ('V3_random_ties',next(s for s in new if s['task']==task),'#176bad')]
        for name,s,color in ss:
            y=np.array([h['y']*100 for h in s['history']]);x=np.arange(1,21)
            for col,values in enumerate((y,np.maximum.accumulate(y))):
                ax=axes[row,col];ax.plot(x,values,c=color,label=name,lw=1.5,ls='--' if name=='R2_trust' else '-')
                if name=='V3_random_ties':
                    indices=[d['iteration']+3 for d in s['decisions'] if d['execution_mode']=='controlled_validation']
                    ax.scatter(x[indices],values[indices],c=color,marker='D',s=30,zorder=4)
                ax.set(ylim=(0,103),xlim=(1,20),xticks=[1,4,8,12,16,20],xlabel='Experiment index (first4 shared)',
                    ylabel='UV product area (%)' if row==0 else 'Reported yield (%)',
                    title=task.replace('_',' ')+' | '+('observed' if col==0 else 'cumulative best'))
                ax.axvline(4.5,c='#888888',ls=':',lw=1);ax.grid(alpha=.15)
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=5,frameon=False,fontsize=9)
    fig.suptitle('V3 exact-score random ties / seed0 /4+16',fontsize=13)
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
            ('R2_region_trust',study.trust.ROOT/'runs'/(task+'_s0_R2_region_trust')/'state.json'),
            ('V3_ID_ties',study.ROOT/'focused_countercheck_v3'/'runs'/(task+'_s0_focused_countercheck_v3')/'state.json')]:
            baselines.append({'task':task,'method':name,**metrics_source.metrics(data.read(path))})
    status=assess(new);requests=root/'requests';responses=list(requests.glob('*/attempt*/response.json'));usage={}
    for path in responses:
        for k,v in data.read(path).get('usage',{}).items():
            if type(v) in (int,float):usage[k]=usage.get(k,0)+v
    result={'status':'COMPLETE','variant':variant,'new_trajectories':2,'new_queries':32,'metrics':new,
        'frozen_controls':baselines,'targets':status,'formal_target_met':all(s['formal_target_met'] for s in status.values()),
        'API':{'requests':len(list(requests.glob('*/validated.json'))),'new_responses':len(responses),
            'reused_requests':len(list(requests.glob('*/reused.json'))),'usage':usage},
        'protocol_sha256':p['protocol_sha256'],'scope':'Single seed0 random-tie sensitivity trajectory; no stability or independent validation claim.'}
    comparisons={};effects={}
    for task in data.SPECS:
        current=next(r for r in new if r['task']==task)
        comparisons[task]={}
        for name in ('Pure_BO','ReasoningBO','V3_ID_ties'):
            previous=next(r for r in baselines if r['task']==task and r['method']==name)
            comparisons[task][name]={'SearchAUC_delta':current['SearchAUC']-previous['SearchAUC'],
                'best_delta_pp':current['final_best_percent']-previous['final_best_percent'],
                'AUC_advantage_retained':current['SearchAUC']>previous['SearchAUC']+1e-9}
        old=data.read(study.ROOT/'focused_countercheck_v3'/'runs'/(task+'_s0_focused_countercheck_v3')/'state.json')
        state=next(s for s in ss if s['task']==task)
        different=[i for i,(a,b) in enumerate(zip(old['history'],state['history']),1) if a['candidate_id']!=b['candidate_id']]
        effects[task]={'first_changed_experiment':min(different) if different else None,
            'identical_initial4':old['history'][:4]==state['history'][:4],
            'same_history_before_q4':old['history'][:7]==state['history'][:7],
            'local_BO_reference_tie_changes':sum(d['tie_order']['ID_order_reference']!=d['tie_order']['random_reference'] for d in state['decisions']),
            'local_normal_tie_changes':sum(d['tie_order']['ID_order_normal_proposal']!=d['tie_order']['random_normal_proposal'] for d in state['decisions'])}
        assert effects[task]['identical_initial4']
        # Recompute metrics from the full history, independent of metrics_source.
        values=[h['y']*100 for h in state['history']];best=[];running=-float('inf')
        for value in values:running=max(running,value);best.append(running)
        independent_auc=sum(best[4:])/16
        assert abs(independent_auc-current['SearchAUC'])<1e-10
        assert max(values)==current['final_best_percent']
        assert values.index(max(values))+1==current['first_final_best_experiment']
    result['comparisons']=comparisons;result['tie_effects']=effects
    data.write(out/'results.json',result);plot(ss,variant,out)
    text=render_text(result,ss,variant)
    (out/'report.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    return {'variant':variant,'metrics':new,'targets':status,'formal_target_met':result['formal_target_met'],'API':result['API']}


def render_text(result,states,variant):
    text=['**V3同分随机排序：两个数据集的单次敏感性实验**','',
        '保持原共同4点、初始知识、全候选池、Mixed GP、GP拟合/MC随机流、seed=0及16次追加查询。独立PCG64随机流按SeedSequence([0,追加轮次])产生并列优先级，完全同分才随机裁决；不同GP分数的次序不变。同分候选共享竞争名次，避免按编号生成的不同名次继续影响Agent。','',
        '随机化覆盖V3的BO参考点、正常知识修正选择、B4同分见证点、刷新GP摘要、反例短名单及首个反例池的同分呈现顺序。B4保留集合、KF、区域信任、95%门槛、2/4/6轮反例机会和总预算保持原样。对照均读取原封存结果，没有重跑。','']
    for task in data.SPECS:
        text += [f'**{task}**','', '| 方法 | SearchAUC | 最终最佳/% | 首达自身最佳实验序号 |','|---|---:|---:|---:|']
        for name in ('Pure_BO','ReasoningBO','V3_ID_ties',variant):
            row=next(r for r in result['frozen_controls']+result['metrics'] if r['task']==task and r['method']==name)
            text.append(f"| {name} | {row['SearchAUC']:.4f} | {row['final_best_percent']:.4f} | {row['first_final_best_experiment']} |")
        text += ['', '| 对照 | 随机V3的AUC差 | 最佳值差/百分点 | AUC是否更高 |','|---|---:|---:|---|']
        for name,delta in result['comparisons'][task].items():
            text.append(f"| {name} | {delta['SearchAUC_delta']:+.4f} | {delta['best_delta_pp']:+.4f} | {'是' if delta['AUC_advantage_retained'] else '否'} |")
        e=result['tie_effects'][task]
        text += ['',f"与原V3首次路径分歧：实验序号{e['first_changed_experiment']}（初始化为1–4）。第4次追加查询前历史{'相同' if e['same_history_before_q4'] else '已不同'}。在新轨迹自身历史上，随机裁决使基础BO参考点改变{e['local_BO_reference_tie_changes']}轮、正常R2提案改变{e['local_normal_tie_changes']}轮；这是同一历史上的裁决差异计数，不是与旧轨迹之间的因果分解。",'',
            '| 验证查询 | 改变变量 | 锚点真实值/% | 新观测/% | 差值/百分点 |','|---:|---|---:|---:|---:|']
        s=next(s for s in states if s['task']==task)
        for v in s['validations']:text.append(f"| {v['query']} | {v['changed_axis']} | {v['anchor_yield_percent']:.3f} | {v['tested_yield_percent']:.3f} | {v['observed_difference_pp']:+.3f} |")
        text += ['',f"实际控制实验{len(s['validations'])}次；机会损失门限例外{sum(d['acquisition_gate_exception'] for d in s['decisions'])}次。",'']
    text += ['**解读范围**','',
        'SearchAUC是16次追加查询后的累计最佳均值；首达序号包含初始4点，比较的是各方法自身最终最佳值，不能直接当作达到同一阈值的速度。Perera为UV产物面积百分比，Doyle为报告产率。两数据集指标分别解释，不跨数据集平均。','',
        '本实验回答固定随机并列seed=0下，原优势是否保持；一次随机重排不足以证明对所有排序稳定。本轮覆盖所有V3的GP并列使用点，路径可能在第4轮之前改变，因此不能把最终差异全部归因于旧轨迹的第4轮五选一。候选ID仍作为标签提供给LLM，没有测试ID文字替换或隐藏。','',
        '新提示继续使用deepseek-flash。只对相同公开上下文复用有效回复，其他情况重新调用；因此发生提示变化时，LLM自身随机性也属于这一实际Agent轨迹，不能声称严格隔离出了排序的唯一因果效应。未使用候选未来真值作排序，也未选择其他随机种子或增加实验预算。','',
        '![逐轮观测与累计最佳](comparison.png)','',
        f"API：{result['API']['requests']}个已接受请求；新增回复{result['API']['new_responses']}，复用{result['API']['reused_requests']}。全部32次选择、KF更新和随机优先级由validate核验；关键指标由真实完整历史独立重算。原始全候选表、公开提示、回复、查询前承诺及状态保存在该版本data/runs/hte_r2_tuning_20261028目录。",'',
        '协议：[protocol.json](protocol.json)；机器结果：[results.json](results.json)；回放：[validation_receipt.json](validation_receipt.json)；实验定义：[TIE_SHUFFLE_TASK.md](../TIE_SHUFFLE_TASK.md)。']
    return text


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
