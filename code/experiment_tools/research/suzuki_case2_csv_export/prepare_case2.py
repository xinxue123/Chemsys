"""Extract frozen Case2 data; calculations use only the matching saved prefix.

Oracle values are attached for post-hoc export only, never used for ranking.
The existing experiment, caches, and release inventories remain read-only.
"""
from hashlib import sha256
from pathlib import Path
import csv
import json
import numpy as np

WORK=Path('research/suzuki_case2_csv_export')
PREPARED=WORK/'prepared'
OUTPUT=Path('outputs/case2_kf_event_b4_seed0')
ROOT=Path('data/runs/suzuki_kalman_knowledge')
POOL=Path('data/runs/suzuki_guarded/inputs/case2')


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def hashfile(path):return sha256(Path(path).read_bytes()).hexdigest()
def dump(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n',encoding='utf8')


def load_npz(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}


def predicate(z,h):
    if h['variable']=='catalyst':return np.isin(np.floor(z[:,3]*8).astype(int).clip(0,7),h['categories'])
    k=('residence_time','temperature','catalyst_loading').index(h['variable']);lo,hi=h['bounds']
    return (z[:,k]>=lo)&(z[:,k]<=hi)


def ranks(scores,mask,ids):
    order=sorted(np.flatnonzero(mask),key=lambda j:(-float(scores[j]),ids[j]));result=[None]*len(ids)
    for rank,j in enumerate(order,1):result[j]=rank
    return result,order


def joined(value):return ';'.join(value)
def text_json(value):return json.dumps(value,ensure_ascii=False,separators=(',',':'))


def prepare():
    paths=[ROOT/'runs/c2_s0/state.json',POOL/'pool.json',POOL/'oracle.npz',Path('research/suzuki_kalman_knowledge/protocol.json'),
           Path('research/suzuki_kalman_knowledge/release_inventory.json')]
    state=read(paths[0]);pool=read(paths[1]);oracle=load_npz(paths[2])['values'];ids=pool['ids'];z=np.array(pool['z']);points=pool['points'];ix={k:j for j,k in enumerate(ids)}
    assert state['status']=='COMPLETE' and state['case']==2 and state['seed']==0 and len(ids)==4692 and len(state['decisions'])==16
    assert len(set(ids))==4692 and oracle.shape==(4692,) and np.isfinite(oracle).all()
    source_hashes={p.as_posix():hashfile(p) for p in paths};tables=[];event_rows=[];chosen={h['candidate_id']:i for i,h in enumerate(state['history'],1)}
    candidate_headers=['candidate_id','catalyst','temperature_C','residence_time_s','catalyst_loading_mol_pct','oracle_yield_pct_posthoc',
                       'selected_total_query','selected_bo_round','initial_design','evaluated_in_this_run']
    candidate_rows=[]
    for j,key in enumerate(ids):
        p=points[j];q=chosen.get(key)
        candidate_rows.append([key,p['catalyst'],p['temperature'],p['t_res'],p['catalyst_loading'],float(oracle[j]*100),q,q-4 if q and q>4 else None,int(q is not None and q<=4),int(q is not None)])
    dump(PREPARED/'candidates.json',{'headers':candidate_headers,'rows':candidate_rows})
    tables.append({'kind':'candidates','input':'candidates.json','output':'case2_candidates.csv','rows':4692,'columns':len(candidate_headers)})
    headers=['bo_round','total_query','observations_before','candidate_id','catalyst','temperature_C','residence_time_s','catalyst_loading_mol_pct',
             'oracle_yield_pct_posthoc','observed_before','observed_yield_pct_before','available_before',
             'gp_mean_pct','gp_epistemic_sd_pp','gp_observation_sd_pp','acquisition_sd_pp',
             'raw_knowledge_shift_pp','knowledge_cap_active','knowledge_shift_pp','guided_mean_pct',
             'base_mc_qlogei','guided_mc_qlogei','base_gap_logei','budget_logei','in_budget_set',
             'base_rank_available','guided_rank_in_budget','satisfies_retained_predicates','is_base_reference','is_selected']
    decision_headers=['total_query','bo_round','phase','candidate_id','catalyst','temperature_C','residence_time_s','catalyst_loading_mol_pct',
        'actual_yield_pct','incumbent_before_pct','incumbent_after_pct','gp_mean_pct','gp_epistemic_sd_pp','gp_observation_sd_pp',
        'guided_mean_pct','knowledge_shift_pp','joint_KF_forecast_pct','base_reference_id','base_rank_available','guided_rank_in_budget',
        'base_mc_qlogei','guided_mc_qlogei','base_gap_logei','budget_logei','eligible_candidates','base_first_second_gap_logei',
        'same_prefix_BO_choice_changed','retained_ids','removed_for_this_round_ids','B4_conflict_kind','satisfies_retained_predicates',
        'LLM_called_this_round','LLM_calls_used','LLM_trigger_reasons','LLM_actions_zh','knowledge_snapshot_json',
        'observed_GP_residual_pp','joint_innovation_z','decision_reason_zh','LLM_original_summary']
    for i,h in enumerate(state['history'][:4],1):
        j=ix[h['candidate_id']];p=points[j]
        row={k:None for k in decision_headers};row.update(total_query=i,phase='initial',candidate_id=ids[j],catalyst=p['catalyst'],temperature_C=p['temperature'],residence_time_s=p['t_res'],catalyst_loading_mol_pct=p['catalyst_loading'],
             actual_yield_pct=h['y']*100,incumbent_before_pct=max([v['y']*100 for v in state['history'][:i-1]],default=None),
             incumbent_after_pct=max(v['y']*100 for v in state['history'][:i]),decision_reason_zh=f'公共LHS初始化第{i}个条件；未经过BO采集排序或LLM知识选择。')
        event_rows.append([row[k] for k in decision_headers])
    for t,event in enumerate(state['decisions'],1):
        cache=ROOT/'cache/c2_s0';pfile=cache/(event['posterior_key']+'.npz');bfile=cache/(event['base_score_key']+'.npz');gfile=cache/(event['guided_score_key']+'.npz')
        for path in (pfile,bfile,gfile):source_hashes[path.as_posix()]=hashfile(path)
        stats=load_npz(pfile);base=load_npz(bfile)['scores'];guided=load_npz(gfile)['scores']
        prefix=state['history'][:t+3];seen={v['candidate_id']:v['y'] for v in prefix};available=np.array([key not in seen for key in ids])
        b_rank,b_order=ranks(base,available,ids);reference=b_order[0];gap=float(base[reference])-base;eligible=available&(gap<=event['budget_logei']);g_rank,g_order=ranks(guided,eligible,ids)
        selected=ix[event['selected_id']];retained=set(event['diagnosis']['retained_ids']);raw=np.zeros(4692);satisfies=np.ones(4692,dtype=bool)
        for h in event['hypotheses']:
            if h['id'] not in retained:continue
            m=predicate(z,h);assert float(m.mean())==h['pool_fraction']
            raw+=event['trust'][h['id']]*(h['effect_pp']/100)*(m.astype(float)-h['pool_fraction']);satisfies&=m
        cap=.5*stats['sd_y'];delta=np.clip(raw,-cap,cap);cap_active=np.abs(raw)>cap
        ystd=np.std(np.array([v['y'] for v in prefix]),ddof=0)
        acquisition_sd=np.sqrt(stats['sd_std']**2+1e-6)*ystd
        assert g_order[0]==selected and ids[reference]==event['reference_id'] and int(eligible.sum())==event['eligible_candidates']
        assert abs(float(stats['mu_y'][selected])-event['selected_gp_mu_y'])<1e-12
        assert abs(float(delta[selected])-event['mean_change_y'])<1e-12
        assert state['history'][t+3]['candidate_id']==ids[selected] and state['history'][t+3]['y']==float(oracle[selected])
        rows=[]
        for j,key in enumerate(ids):
            p=points[j]
            rows.append([t,t+4,len(prefix),key,p['catalyst'],p['temperature'],p['t_res'],p['catalyst_loading'],float(oracle[j]*100),
                int(key in seen),seen[key]*100 if key in seen else None,int(available[j]),
                float(stats['mu_y'][j]*100),float(stats['sd_y'][j]*100),float(stats['sd_predictive_y'][j]*100),float(acquisition_sd[j]*100),
                float(raw[j]*100),int(cap_active[j]),float(delta[j]*100),float((stats['mu_y'][j]+delta[j])*100),
                float(base[j]),float(guided[j]),float(gap[j]),event['budget_logei'],int(eligible[j]),b_rank[j],g_rank[j],int(satisfies[j]),int(j==reference),int(j==selected)])
        name=f'predictions_round{t:02d}.json';dump(PREPARED/name,{'headers':headers,'rows':rows})
        tables.append({'kind':'predictions','input':name,'output':'case2_bo_predictions.csv','round':t,'rows':4692,'columns':len(headers)})
        calls=[c for c in state['calls'] if c['iteration']<=t];call=next((c for c in calls if c['iteration']==t),None)
        actions=[]
        if call:
            for h in call['plan']['hypotheses']:
                op={'add':'新增','revise':'修订并重置校正状态','retain':'保留校正状态'}[h['operation']]
                condition=('催化剂编号'+str(h['categories'])) if h['variable']=='catalyst' else h['variable']+'归一化区间'+str(h['bounds'])
                actions.append(f"{op}{h['id']}：{condition}，效应{h['effect_pp']:g}个百分点")
            actions += [f"撤销{h['id']}" for h in call['plan']['revocations']]
        kept=joined(event['diagnosis']['retained_ids']) or '无';removed=joined(event['diagnosis']['revoked_ids']) or '无'
        if eligible.sum()==1:reason='预算集只有1个未查询候选，本轮被约束为基础qLogEI第一名；知识无法改变选点。'
        elif retained:reason=f"B4保留知识{kept}；在{eligible.sum()}个预算内未查询候选中选择知识修正后512路MC-qLogEI最高者。"
        else:reason='本轮没有可保留知识，执行基础512路MC-qLogEI最大值候选。'
        reason+=f" 基础采集机会损失{gap[selected]:.6f}，不超过门限{event['budget_logei']:.6f}。"
        if retained and eligible.sum()>1:reason+=' 知识改变了同前缀基础BO首选。' if selected!=reference else ' 知识参与评分，但选点与同前缀基础BO相同。'
        if retained and not satisfies[selected]:reason+=' 采用软知识引导，选中条件不满足全部保留谓词。'
        if len(calls)==4 and not call:reason+=' 本案例4次LLM调用额度已用完，继续GP与卡尔曼更新。'
        snapshot=[{'id':h['id'],'version':h['version'],'variable':h['variable'],'bounds':h['bounds'],'categories':h['categories'],
                   'effect_pp':h['effect_pp'],'trust':event['trust'][h['id']],'retained_this_round':h['id'] in retained} for h in event['hypotheses']]
        p=points[selected];measurement=event['measurement']
        row={'total_query':t+4,'bo_round':t,'phase':'BO','candidate_id':ids[selected],'catalyst':p['catalyst'],'temperature_C':p['temperature'],
             'residence_time_s':p['t_res'],'catalyst_loading_mol_pct':p['catalyst_loading'],'actual_yield_pct':float(oracle[selected]*100),
             'incumbent_before_pct':max(v['y'] for v in prefix)*100,'incumbent_after_pct':max(v['y'] for v in state['history'][:t+4])*100,
             'gp_mean_pct':float(stats['mu_y'][selected]*100),'gp_epistemic_sd_pp':float(stats['sd_y'][selected]*100),
             'gp_observation_sd_pp':float(stats['sd_predictive_y'][selected]*100),'guided_mean_pct':float((stats['mu_y'][selected]+delta[selected])*100),
             'knowledge_shift_pp':float(delta[selected]*100),'joint_KF_forecast_pct':measurement['predictions']['GP_joint_KF']*100,
             'base_reference_id':ids[reference],'base_rank_available':b_rank[selected],'guided_rank_in_budget':g_rank[selected],
             'base_mc_qlogei':float(base[selected]),'guided_mc_qlogei':float(guided[selected]),'base_gap_logei':float(gap[selected]),
             'budget_logei':event['budget_logei'],'eligible_candidates':int(eligible.sum()),'base_first_second_gap_logei':float(base[b_order[0]]-base[b_order[1]]),
             'same_prefix_BO_choice_changed':int(selected!=reference),'retained_ids':joined(event['diagnosis']['retained_ids']),
             'removed_for_this_round_ids':joined(event['diagnosis']['revoked_ids']),'B4_conflict_kind':event['diagnosis']['conflict_kind'],
             'satisfies_retained_predicates':int(satisfies[selected]),'LLM_called_this_round':int(call is not None),'LLM_calls_used':len(calls),
             'LLM_trigger_reasons':joined(call['reasons']) if call else '', 'LLM_actions_zh':'；'.join(actions),'knowledge_snapshot_json':text_json(snapshot),
             'observed_GP_residual_pp':measurement['residual']*100,'joint_innovation_z':measurement['innovation_z'],'decision_reason_zh':reason,
             'LLM_original_summary':call['plan']['summary'] if call else ''}
        assert set(row)==set(decision_headers);event_rows.append([row[k] for k in decision_headers])
    dump(PREPARED/'decisions.json',{'headers':decision_headers,'rows':event_rows})
    tables.append({'kind':'decisions','input':'decisions.json','output':'case2_decisions.csv','rows':20,'columns':len(decision_headers)})
    manifest={'case':2,'seed':0,'method':state['method'],'protocol_sha256':state['protocol_sha256'],'output_directory':OUTPUT.as_posix(),
        'tables':tables,'inputs':source_hashes,'source_hashes_unchanged':all(hashfile(p)==h for p,h in source_hashes.items()),
        'no_new_API_calls':True,'no_new_oracle_queries':True,'no_new_GP_fits':True,'prediction_rows':75072,'candidate_rows':4692,'decision_rows':20,
        'export_source_sha256':hashfile(Path(__file__))}
    assert manifest['source_hashes_unchanged'];dump(WORK/'prepared_manifest.json',manifest)
    print(json.dumps({k:manifest[k] for k in ('method','candidate_rows','prediction_rows','decision_rows','source_hashes_unchanged')}))


if __name__=='__main__':prepare()
