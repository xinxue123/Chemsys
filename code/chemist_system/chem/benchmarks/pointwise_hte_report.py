"""Sequential4+16 single-seed results, independent algebra and trace checks."""
from pathlib import Path
import csv
import json

import numpy as np
import torch

from . import pointwise_hte_study as study
from . import pointwise_hte_engine as engine
from . import pointwise_hte_llm as llm
from . import reasoning_bo_data as data
from .suzuki_domain import digest


def states():
    result=[]
    for spec in data.read(study.ROOT/'manifest.json'):
        path=study.ROOT/'runs'/spec['run_id']/'state.json'
        result.append(data.read(path) if path.exists() else {**spec,'status':'NOT_STARTED','decisions':[],'history':[]})
    return result


def metrics(s):
    if s['status']!='COMPLETE':return {}
    curve=np.maximum.accumulate([h['y']*100 for h in s['history']])
    return {'SearchAUC':float(curve[4:].mean()),'final_best_percent':float(curve[-1]),
        'first_final_best_experiment':int(np.flatnonzero(curve>=curve[-1]-1e-10)[0])+1,
        'initial_best_percent':float(curve[3]),'mean_queried_yield_percent':float(np.mean([h['y'] for h in s['history'][4:]])*100),
        'refreshes':len(s['calls']),'nonbase_choices':sum(d['selected_id']!=d['reference_id'] for d in s['decisions'])}


def write_csv(path,rows,fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)


def build():
    p=study.check();ss=states();rows=[];rounds=[]
    for s in ss:
        rows.append({k:s[k] for k in ('task','method','seed','run_id','status')}|metrics(s))
        for d in s['decisions']:
            rounds.append({'task':s['task'],'method':s['method'],**{k:d[k] for k in ('iteration','selected_id','reference_id','y',
                'incumbent_after','base_qLogEI','guided_qLogEI','base_opportunity_loss','budget','eligible_count','amplitude','fallback')}})
    write_csv(study.OUT/'metrics.csv',rows,['task','method','seed','run_id','status','SearchAUC','final_best_percent',
        'first_final_best_experiment','initial_best_percent','mean_queried_yield_percent','refreshes','nonbase_choices'])
    write_csv(study.OUT/'rounds.csv',rounds,['task','method','iteration','selected_id','reference_id','y','incumbent_after',
        'base_qLogEI','guided_qLogEI','base_opportunity_loss','budget','eligible_count','amplitude','fallback'])
    complete=all(s['status']=='COMPLETE' for s in ss)
    usage={};responses=list((study.ROOT/'requests').glob('*/attempt*/response.json'))
    for path in responses:
        for k,v in data.read(path).get('usage',{}).items():
            if type(v) in (int,float):usage[k]=usage.get(k,0)+v
    result={'status':'COMPLETE' if complete else 'INCOMPLETE','completed':sum(s['status']=='COMPLETE' for s in ss),
        'planned':16,'protocol_sha256':p['protocol_sha256'],'metrics':rows,'API':{'validated':len(list((study.ROOT/'requests').glob('*/validated.json'))),
            'responses':len(responses),'usage':usage},'limits':'Single seed, one initial knowledge condition. No population SD or significance claim.'}
    data.write(study.OUT/'results.json',result)
    text=['# 修正后：逐点4＋16对照，seed=0','',f"完成 {result['completed']}/16 条轨迹；状态 **{result['status']}**。",'',
        '各数据集使用共同Sobol4个初始实验，再逐点查询16次。统一全类别Mixed GP、qLogEI(q=1)、512个MC样本与原Suzuki Case预算门限。',
        '两套数据仍使用作者完整实测池（3696/3955）；底物属于可选变量。知识条件仅为真实Agent生成并共享的初始建议。',
        '区域LGBO、门限、两种B4和新方案对全池打分；LLINBO使用官方Justify分支与原统一底座接口；ReasoningBO为明确标记的5选1适配。',
        '已修复区域缓存输出单位、RBO观测计数和空KF测量事件检查。旧M1/M2结果未覆盖。','',
        'SearchAUC为16次追加查询后的累计最佳产率均值；首达最佳为本轨迹自己的最终最佳实验序号，包含4次初始化。单种子结果只作描述。']
    for task in data.SPECS:
        text+=['',f'## {task}','','| 方法 | SearchAUC | 最终最佳/% | 首达自身最佳序号 |','|---|---:|---:|---:|']
        for row in rows:
            if row['task']!=task:continue
            if row['status']=='COMPLETE':text.append(f"| {row['method']} | {row['SearchAUC']:.4f} | {row['final_best_percent']:.4f} | {row['first_final_best_experiment']} |")
            else:text.append(f"| {row['method']} | {row['status']} | — | — |")
    if complete:
        plot(ss)
        text+=['','![逐点产率与累计最佳](comparison.png)','',
            '图中第1–4次为共同预定初始化，竖线之后才是逐点选择。没有跨种子误差带。',
            '', '逐轮候选表为各run下rounds/*_candidates.npz，包含所有点的GP均值/不确定性、基础及修正qLogEI、可用与预算掩码；选择和依据见*_committed.json。',
            '所有真实LLM提示、可见回复及格式修复均位于data/runs/hte_pointwise_seed0/requests。',
            '', '必要适配、源文件和输入哈希见[protocol.json](protocol.json)；[validation_receipt.json](validation_receipt.json)给出校验边界。']
    study.OUT.mkdir(parents=True,exist_ok=True);(study.OUT/'report.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    return {'status':result['status'],'completed':result['completed'],'planned':16,'API':result['API']}


def plot(ss):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'Pure_BO':'#333333','LGBO_region':'#25826b','Gate_only':'#9061ab','B4_hard':'#ad9350',
        'B4_soft':'#bd718f','New_R2':'#176bad','LLINBO_Justify':'#b34e4e','ReasoningBO_point':'#d47a16'}
    markers=dict(zip(engine.METHODS,['v','D','^','<','>','o','P','s']))
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True)
    for row,task in enumerate(data.SPECS):
        for s in ss:
            if s['task']!=task:continue
            y=np.array([h['y']*100 for h in s['history']]);x=np.arange(1,21);method=s['method']
            for col,v in enumerate((y,np.maximum.accumulate(y))):
                ax=axes[row,col];ax.plot(x,v,color=colors[method],label=method,marker=markers[method],
                    markevery=3,ms=3,lw=1.4,ls='--' if method in ('B4_hard','B4_soft') else '-')
                ax.set_ylim(0,103);ax.set_xlim(1,20);ax.set_xticks([1,4,8,12,16,20]);ax.axvline(4.5,color='#888888',ls=':',lw=1)
                ax.grid(alpha=.17);ax.set_ylabel('Reported yield (%)');ax.set_xlabel('Experiment index (first4 = common initialization)')
                ax.set_title(task.replace('_',' ')+' | '+('observed yield' if col==0 else 'cumulative best'))
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=4,fontsize=9,frameon=False)
    fig.suptitle('Corrected sequential replay: seed0 / Mixed GP / qLogEI /4+16',fontsize=13)
    fig.tight_layout(rect=[0,.10,1,.96]);fig.savefig(study.OUT/'comparison.png',dpi=180);plt.close(fig)


@torch.no_grad()
def validate():
    study.warm();p=study.check();ss=states();count=0;kfcount=0;numeric=[];maxerr=0.
    for s in ss:
        if not s['history']:continue
        initial=data.read(study.ROOT/'initialization'/s['task']/'ready.json');assert s['history'][:4]==initial['history']
        assert len(s['history'])==4+len(s['decisions'])
        assert len({h['candidate_id'] for h in s['history']})==len(s['history'])
        pool=data.load_pool(s['task']);lookup={key:i for i,key in enumerate(pool['ids'])}
        oracle=np.load(data.ROOT/s['task']/'oracle.npz')['values']
        for h in s['history']:assert h['y']==float(oracle[lookup[h['candidate_id']]])
        cache=engine.Cache(pool,study.ROOT/'cache'/s['task'])
        for d in s['decisions']:
            t=d['iteration'];folder=study.ROOT/'runs'/s['run_id']/'rounds'
            target=folder/f'{t:02d}_committed.json';assert data.filehash(target)==d['commit_sha256']
            c=data.read(target);assert c['history_sha256']==digest(s['history'][:3+t])
            assert c['observation_count']==3+t and c['selected_id']==d['selected_id']
            table=folder/f'{t:02d}_candidates.npz';assert data.filehash(table)==c['candidate_table_sha256']
            with np.load(table) as a:
                j=lookup[d['selected_id']];ref=lookup[d['reference_id']]
                assert a['available'][j] and a['base_qLogEI'][ref]==a['base_qLogEI'][a['available']].max()
                if s['method'] not in ('LGBO_region','ReasoningBO_point'):assert a['eligible'][j]
                assert d['base_opportunity_loss']==float(a['base_qLogEI'][ref]-a['base_qLogEI'][j])
                expected=engine.nominal.argmax(a['guided_qLogEI'],
                    a['available'] if s['method']=='LGBO_region' else a['eligible'],pool['ids'])
                if s['method'] in ('Pure_BO','LGBO_region','Gate_only','B4_soft','New_R2'):assert j==expected
                if s['method']=='B4_hard':
                    mask=a['eligible'].copy();h=engine.nominal.membership(pool,d['knowledge'])
                    for k,clause in enumerate(d['knowledge']['clauses']):
                        if clause['id'] in d['retained']:mask&=h[k]
                    assert j==engine.nominal.argmax(a['base_qLogEI'],mask,pool['ids'])
            if d['fallback']:assert d['selected_id']==d['reference_id'] and d['amplitude']==0
            if d['measurement']:
                m=d['measurement'];pre=c['decision']['prior'];P=np.array(pre['covariance']);H=np.array(m['feature'])
                theta=np.array(pre['mean']);R=m['measurement_variance'];S=float(H@P@H+R)
                after=theta+P@H*(m['residual']-H@theta)/S
                gain=P@H/S;A=np.eye(2)-np.outer(gain,H);cov=A@P@A.T+np.outer(gain,gain)*R
                post=d['calibration_after'];assert np.isfinite(after).all() and S>0
                np.testing.assert_allclose(after,post['mean'],rtol=0,atol=1e-12)
                np.testing.assert_allclose(cov,post['covariance'],rtol=0,atol=1e-12)
                assert post['last_round']==t;kfcount+=1
            # Independent direct JOINT posterior covariance, not cached kernel algebra.
            if t in (1,8,16) and c['decision']['region'] is not None:
                hist=s['history'][:3+t];_,v=cache.model(hist);spec=c['decision']['region']
                tilt,_,_=cache.tilt(hist,spec);ix=[lookup[d['selected_id']]];xx=cache.X[ix]
                joint=v['model'].posterior(torch.cat([xx,tilt.Xg])).mvn.covariance_matrix
                direct=tilt._lam*(joint[:1,1:]@tilt._a)*v['std']
                with np.load(table) as a:recorded=a['raw_shift_std'][ix]*v['std']
                err=float(np.max(np.abs(recorded-direct.numpy())));maxerr=max(maxerr,err);assert err<1e-9
                numeric.append({'run_id':s['run_id'],'query':t,'direct_posterior_field_error':err})
            count+=1;cache.release()
        if s['status']=='COMPLETE':assert len(s['history'])==20 and len(s['decisions'])==16
    # Validate every accepted pointwise RBO count/subset and every regional citation.
    requests=0;facts=0
    for folder in (study.ROOT/'requests').iterdir():
        if not (folder/'validated.json').exists():continue
        ctx=data.read(folder/'context.json');answer=data.read(folder/'validated.json');raw=dict(ctx);key=raw.pop('context_sha256')
        assert digest(raw)==key==folder.name==answer['context_sha256'];public=ctx['public'];pool=data.load_pool(ctx['task'])
        assert public['observation_count']==len(ctx['observed_prefix'])
        if ctx['phase']=='ReasoningBO_point':
            prompt=public['official_prompt_with_pointwise_count_fix']
            assert f"data from {len(ctx['observed_prefix'])} completed experiments" in prompt
            assert f"query round {public['query_round']}." in prompt
            s=next(s for s in ss if s['task']==ctx['task'] and s['method']=='ReasoningBO_point')
            d=s['decisions'][public['query_round']-1]
            assert llm.validate_rbo(answer['raw'],pool,ctx['observed_prefix'],d['extra']['shortlist_ids'])==answer['plan']
        elif ctx['phase'] in ('initial_knowledge','refresh'):
            old=public['current_knowledge'];_,reverse=llm.nominal.token_maps(pool)
            knowledge={'confidence':old['confidence'],'clauses':[
                {'id':h['id'],'axis':h['axis'],'allowed':sorted(reverse[h['axis']][x] for x in h['allowed'])} for h in old['clauses']]}
            checked=llm.nominal.validate_plan(answer['raw'],pool,knowledge,public['observations'],grounded=True)
            assert checked==answer['plan'];facts+=len(checked['checked_facts'])
        elif ctx['phase']=='LLINBO_proposal':
            assert llm.nominal.validate_point(answer['raw'],pool,ctx['observed_prefix'])==answer['plan']
            s=next(s for s in ss if s['task']==ctx['task'] and s['method']=='LLINBO_Justify')
            d=s['decisions'][public['query_round']-1];extra=d['extra']
            assert extra['projection']==engine.nominal.project_proposal(answer['plan']['decoded_point'],pool,ctx['observed_prefix'])
            accepted=extra['proposal_base_logei_loss']<=extra['budget']
            assert extra['accepted']==accepted
            assert d['selected_id']==(extra['proposed_id'] if accepted else extra['reference_id'])
        requests+=1
    result={'status':'PASS_COMPLETE' if all(s['status']=='COMPLETE' for s in ss) else 'PASS_AVAILABLE_INCOMPLETE',
        'protocol_sha256':p['protocol_sha256'],'trajectories':sum(s['status']=='COMPLETE' for s in ss),
        'choices_audited':count,'KF_updates':kfcount,'valid_requests_checked':requests,'typed_facts_checked':facts,
        'direct_joint_posterior_field_checks':numeric,'maximum_field_error_yield_fraction':maxerr,
        'operator_compiles_in_loop':0}
    data.write(study.OUT/'validation_receipt.json',result);return result
