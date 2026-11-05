"""Matched observed branches and source-linked explanations of high-point entry."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import io
import json
import sys

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/analysis'))
import trace_v13_highpoint_origin as origin
OUT=origin.OUT


def load(name):return pd.read_csv(io.BytesIO(origin.tracked(OUT/name)),encoding='utf-8-sig')
def save(name,data):origin.save(name,data)


def novelty(pool,history,point):
    def changes(a,b):return origin.profile.helpers.axis_changes(pool,a,b)
    prior=[pool.points[pool.ix[h['candidate_id']]] for h in history]
    anchor=pool.points[pool.ix[max(history,key=lambda h:h['raw_value'])['candidate_id']]]
    ch=changes(anchor,point);cat=[i for i,k in enumerate(pool.kinds) if k=='categorical']
    newcat=sum(all(p[pool.axes[i]]!=point[pool.axes[i]] for p in prior) for i in cat)
    category_change=sum(ch[i] for i in cat);continuous_change=sum(ch)-category_change
    return dict(new_categorical_levels=newcat,category_axes_changed_vs_incumbent=category_change,
        continuous_axes_changed_vs_incumbent=continuous_change,
        categorical_only_control=category_change>0 and continuous_change==0,
        pure_single_category_control=category_change==1 and continuous_change==0,
        continuous_only_local=category_change==0 and continuous_change==1)


def gp_rank_target(f,js):
    valid=[j for j in js if f['available'][j]]
    if not valid:return None
    j=min(valid,key=lambda z:(-f['base_logei'][z],f['tie_priority'][z]))
    return origin.rank(f['base_logei'],f['available'],f['tie_priority'],j)['rank']


def branch(spec,d):
    a=d.loc[spec.path_id];b=d.loc[spec.BO_path_id];pool=origin.POOLS[a.task]
    ha=origin.STATES[a.path_id]['history'];hb=origin.STATES[b.path_id]['history']
    shared=next((i for i,(x,y) in enumerate(zip(ha,hb)) if x['candidate_id']!=y['candidate_id']),20)
    result=spec.to_dict();result.update(shared_observations=shared,first_divergence_total=shared+1 if shared<20 else None)
    if shared==20:return result
    assert shared>=4
    assert [(h['candidate_id'],h['raw_value']) for h in ha[:shared]]==[(h['candidate_id'],h['raw_value']) for h in hb[:shared]]
    total=shared+1;fa=origin.frame(a,total);fb=origin.frame(b,total)
    def signature(f):
        ctx=dict(stats={k:f[k] for k in ('mu_y','sd_y','sd_predictive_y')},
                 **{k:f[k] for k in ('base_logei','available','tie_priority')})
        return origin.engine.gp_hash(ctx)
    assert signature(fa)==signature(fb),(a.path_id,b.path_id,total)
    menu=lambda f:set(f['commit'].get('display_ids',f['commit'].get('ordered_ids',[])))
    assert menu(fa)==menu(fb)
    globaljs=np.flatnonzero(origin.TRUTH[a.task]==origin.TRUTH[a.task].max())
    result.update(same_predecision_GP=True,same_predecision_menu=True,global_best_rank_before=gp_rank_target(fa,globaljs))
    for label,r,h,f in [('agent',a,ha,fa),('BO',b,hb,fb)]:
        chosen=h[shared];j=pool.ix[chosen['candidate_id']];dec=f['commit'].get('decision') or {}
        result.update({label+'_'+k:v for k,v in novelty(pool,h[:shared],pool.points[j]).items()})
        result.update({label+'_selected_id':chosen['candidate_id'],label+'_selected_value':chosen['raw_value'],
            label+'_selected_predicted_mean':float(f['mu_y'][j]*pool.scale),
            label+'_selected_SD':float(f['sd_y'][j]*pool.scale),
            label+'_selected_native_rank':origin.rank(f['base_logei'],f['available'],f['tie_priority'],j)['rank'],
            label+'_knowledge_uses':json.dumps(dec.get('knowledge_uses',[])),
            label+'_reason':dec.get('insight',{}).get('comment',''),
            label+'_commit_path':(f['folder']/'rounds'/f"{f['q']:02}_commit.json").relative_to(ROOT).as_posix()})
        if total<20:result[label+'_global_rank_after_actual_observation']=gp_rank_target(origin.frame(r,total+1),globaljs)
    result['agent_immediate_minus_BO_normalized']=(result['agent_selected_value']-result['BO_selected_value'])/a.pool_range
    ar=result.get('agent_global_rank_after_actual_observation');br=result.get('BO_global_rank_after_actual_observation')
    result['agent_global_rank_better_after_observation']=ar<br if ar is not None and br is not None else None
    return result


def high_selection(event,d):
    r=d.loc[event.path_id];t=int(event.selection_total);pool=origin.POOLS[r.task]
    f=origin.frame(r,t);h=origin.STATES[r.path_id]['history'];selected=h[t-1];j=pool.ix[selected['candidate_id']]
    dec=f['commit'].get('decision') or {};uses=dec.get('knowledge_uses') or [];notes=dec.get('notes_at_commit') or []
    prior=0;reliabilities=[]
    for use in uses:
        match=next((n for n in notes if n.get('id')==use.get('id') and n.get('version')==use.get('version')),None)
        if match:
            cal=match.get('calibration',{})
            prior+=int(cal.get('updates',0)>0)
            reliabilities.append(cal.get('reliability'))
    result=dict(path_id=r.path_id,task=r.task,regime=r.regime,method=r.method,seed=r.seed,
        selection_total=t,selected_id=selected['candidate_id'],selected_value=selected['raw_value'],
        native_GP_rank=origin.rank(f['base_logei'],f['available'],f['tie_priority'],j)['rank'],
        selected_is_GP_reference=selected['candidate_id']==f['commit'].get('BO_reference_id'),
        forecasts=len(dec.get('forecasts',[])),knowledge_uses=len(uses),prior_feedback_knowledge_uses=prior,
        uses_detail=json.dumps(uses),prior_reliabilities=json.dumps(reliabilities),
        refresh_at_selection=f['commit'].get('memory') is not None,
        recorded_reason=dec.get('insight',{}).get('comment',''),
        conditions=json.dumps(pool.points[j]),
        commit_path=(f['folder']/'rounds'/f"{f['q']:02}_commit.json").relative_to(ROOT).as_posix(),
        **novelty(pool,h[:t-1],pool.points[j]))
    return result


def verify_conditioning(numerical,d):
    checks=[]
    for task in ('c2','doyle','hplc'):
        choices=numerical[(numerical.task==task)&(~numerical.initial)&(numerical.total>=6)]
        # Fixed task examples; no selection by the size/sign of the claimed effect.
        sample=choices.sort_values(['path_id','total','target_kind']).iloc[0]
        r=d.loc[sample.path_id];before=origin.frame(r,int(sample.total)-1)
        model,norm,_,_,_=origin.reconstruct(r,before)
        new=origin.STATES[r.path_id]['history'][int(sample.total)-2];pool=origin.POOLS[task]
        newX=origin.X[task][[pool.ix[new['candidate_id']]]]
        newY=torch.tensor([[(new['y']-norm['mean'])/norm['std']]],dtype=torch.double)
        with torch.no_grad():
            conditioned=model.condition_on_observations(newX,newY)
            p=conditioned.posterior(origin.X[task][[int(sample.target_index)]])
            mu=(float(p.mean.reshape(()))*norm['std']+norm['mean'])*pool.scale
            sd=float(p.variance.reshape(()).sqrt())*norm['std']*pool.scale
        np.testing.assert_allclose(mu,sample.target_mean_fixed_update,rtol=1e-6,atol=2e-5)
        np.testing.assert_allclose(sd,sample.target_SD_fixed_update,rtol=1e-6,atol=2e-5)
        checks.append(dict(task=task,path_id=sample.path_id,total=int(sample.total),
            conditioned_model_mu=mu,matrix_update_mu=sample.target_mean_fixed_update,
            conditioned_model_SD=sd,matrix_update_SD=sample.target_SD_fixed_update,status='PASS'))
    save('independent_conditioning_checks.csv',checks)


def main():
    origin.tracked(__file__);d=origin.initialize().set_index('path_id',drop=False)
    for pool in origin.POOLS.values():
        origin.profile.helpers.BOUNDS[pool.task]={axis:(min(p[axis] for p in pool.points),max(p[axis] for p in pool.points))
            for axis,kind in zip(pool.axes,pool.kinds) if kind=='continuous'}
    events=load('all_first_exposure_events.csv');numerical=load('one_step_numerical_mechanisms.csv')
    p=pd.read_csv(io.BytesIO(origin.tracked(origin.SOURCE/'all_main_BO_pairs_decomposed.csv')),encoding='utf-8-sig')
    warm=p[p.regime=='warm']
    with ThreadPoolExecutor(max_workers=8) as executor:
        branches=list(executor.map(lambda r:branch(r,d),[r for _,r in warm.iterrows()]))
    branchdata=pd.DataFrame(branches);save('all_warm_first_divergences.csv',branchdata)
    lowbranch=branchdata[(branchdata.start_fraction_max<.8)&(branchdata.BO_start_fraction_max<.8)]
    save('all_lower_start_warm_first_divergences.csv',lowbranch)
    ids=set(d[d.primary&(d.start_fraction_max<.8)].path_id)
    high=events[events.path_id.isin(ids)&(events.target_kind=='first90_hit')&events.selection_total.notna()]
    with ThreadPoolExecutor(max_workers=8) as executor:
        selections=list(executor.map(lambda r:high_selection(r,d),[r for _,r in high.iterrows()]))
    save('all_first_high_selections_with_reasons.csv',selections)
    verify_conditioning(numerical,d)
    receipts=dict(status='PASS_OBSERVED_BRANCH_AND_CONDITIONING_CHECKS',warm_pairs=len(branchdata),
        low_start_warm_pairs=len(lowbranch),first_high_selections=len(selections),
        independent_conditioning_check_families=3,new_GP_fits=0,new_LLM_calls=0,new_experiments=0)
    for path,h in origin.READS.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h,path
    (OUT/'branch_audit.json').write_text(json.dumps(receipts,indent=2)+'\n',encoding='utf-8')
    (OUT/'branch_source_hashes.json').write_text(json.dumps(origin.READS,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipts,indent=2))


if __name__=='__main__':main()
