"""Full-population archived trajectory features; no fitting or new decisions."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean
import csv
import hashlib
import io
import json
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/analysis'),str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
import audit_v13_path_quality as helpers
from chem.benchmarks import knowledge_gp
from knowledge_rank_v9 import engine
from v13_late_validation import study as late
import build_v13_path_census as census

OUT=census.OUT
READS={}
POOLS={};TRUTH={};STATES={}
FULL={**{f'c{i}':f'Reizman_Suzuki_Case_{i}' for i in range(1,5)},
      'perera':'Perera_Suzuki','doyle':'Doyle_Buchwald_Hartwig','snar':'SnAr','hplc':'HPLC','aryl':'Direct_Arylation'}


def tracked(path):
    p=Path(path).resolve();b=p.read_bytes()
    READS[p.relative_to(ROOT).as_posix()]=hashlib.sha256(b).hexdigest()
    return b


def read(path):return json.loads(tracked(path).decode('utf-8-sig'))


def rows(path):return list(csv.DictReader(io.StringIO(tracked(path).decode('utf-8-sig'))))


def save(name,records):
    if not records:return
    fields=list(dict.fromkeys(k for r in records for k in r))
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
        w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v
                     for k,v in r.items()} for r in records)


def score_feedback(decision):
    events=decision.get('feedback',decision.get('KF_feedback',[]))
    if not isinstance(events,list):events=[]
    result=dict(KF_updates=0,KF_increases=0,KF_decreases=0,new_suppressions=0,
                unparsed_feedback_events=0,numeric_ranker_updates=int(bool(decision.get('ranker_feedback'))),
                candidate_card_updates=int(bool(decision.get('card_feedback'))))
    for e in events:
        if e.get('prior_transport'):continue
        cal=e.get('calibration',e)
        if not all(k in cal for k in ('Brier_gain','before','after')):
            result['unparsed_feedback_events']+=1;continue
        before=cal['before'].get('reliability',cal['before'].get('weight'))
        after=cal['after'].get('reliability',cal['after'].get('weight'))
        if before is None or after is None:
            result['unparsed_feedback_events']+=1;continue
        result['KF_updates']+=1;result['KF_increases']+=after>before;result['KF_decreases']+=after<before
        pb=e.get('recommendation_before',cal['before'].get('status'))
        pa=e.get('recommendation_after',cal['after'].get('status'))
        result['new_suppressions']+=pa in ('paused','suppressed','retracted') and pa!=pb
    return result


def asset_folder(record,total):
    own=(ROOT/record['source_path']).parent
    if record['regime']=='continuation10+10' and total<=10:
        sf=late.source_state(FULL[record['task']],int(record['seed']))
        source=read(sf)
        assert source['history'][:10]==STATES[record['path_id']]['history'][:10]
        return sf.parent,total,True
    initial=0 if record['regime']=='continuation10+10' else int(record['initial'])
    return own,total-initial,False


def gp_profile(spec):
    record,n=spec
    pool=POOLS[record['task']];truth=TRUTH[record['task']];state=STATES[record['path_id']]
    folder,q,prefix_reuse=asset_folder(record,n+1)
    file=folder/'rounds'/f'{q:02}_GP.npz';fit=folder/'rounds'/f'{q:02}_fit.json'
    base=dict(path_id=record['path_id'],family=record['family'],primary=record['primary'],task=record['task'],
              method=record['method'],backend=record['backend'],regime=record['regime'],observed=n,
              prefix_asset_reused=prefix_reuse,GP_schema_available=file.exists() and fit.exists())
    if not base['GP_schema_available']:return base
    with np.load(io.BytesIO(tracked(file)),allow_pickle=False) as z:
        arrays={k:z[k].copy() for k in ('mu_y','sd_y','sd_predictive_y','base_logei','available','tie_priority')}
    available=~np.isin(pool.ids,[h['candidate_id'] for h in state['history'][:n]])
    np.testing.assert_array_equal(available,arrays['available'])
    c=read(folder/'rounds'/f'{q:02}_commit.json')
    ctx=dict(stats={k:arrays[k] for k in ('mu_y','sd_y','sd_predictive_y')},
             **{k:arrays[k] for k in ('base_logei','available','tie_priority')})
    base['GP_commit_hash_available']='GP_sha256' in c
    if base['GP_commit_hash_available']:
        assert engine.gp_hash(ctx)==c['GP_sha256'],str(file)
    params=read(fit);params=params.get('parameters',params)
    tag='perera' if pool.group=='P1' else record['task']
    cat=helpers.category_parameters(params,tag)
    mu=arrays['mu_y']*pool.scale;sd=arrays['sd_y']*pool.scale
    observed=np.array([h['raw_value'] for h in state['history'][:n]])
    span=float(np.ptp(truth));threshold=.95*float(truth.max())
    high=available&(truth>=threshold)
    base.update(mean_SD=float(sd[available].mean()),mean_SD_normalized=float(sd[available].mean()/span),
        coverage95=float(np.mean(np.abs(mu[available]-truth[available])<=1.96*sd[available])),
        RMSE_normalized=float(np.sqrt(np.mean((mu[available]-truth[available])**2))/span),
        observed_response_SD=float(observed.std(ddof=0)),
        high_underestimated_fraction=float(np.mean((mu+1.96*sd)[high]<threshold)) if high.any() else None,
        mixed_category_correlation=cat['category_correlation'],
        categorical_axis_correlations=cat['category_axis_correlations'],
        exact_acquisition_maxima=int(np.count_nonzero(available &
                                      (arrays['base_logei']==arrays['base_logei'][available].max()))))
    return base


def path_profile(record):
    path_id=record['path_id'];state=STATES[path_id];pool=POOLS[record['task']]
    truth=TRUTH[record['task']];y=np.array([h['raw_value'] for h in state['history']])
    best=np.maximum.accumulate(y);lo=float(truth.min());hi=float(truth.max());span=hi-lo
    for obs in state['history']:census.close(obs['raw_value'],truth[pool.ix[obs['candidate_id']]])
    init=0 if record['regime']=='continuation10+10' else int(record['initial'])
    result=dict(record,first_value=float(y[0]),best4=float(best[3]),best8=float(best[7]),best12=float(best[11]),
        best16=float(best[15]),best4_normalized=float((best[3]-lo)/span),best8_normalized=float((best[7]-lo)/span),
        final_normalized=float((best[-1]-lo)/span),final_fraction_max=float(best[-1]/hi),
        AUC_normalized=float((float(record['SearchAUC'])-lo)/span),
        future_AUC_after8_normalized=float((best[8:].mean()-lo)/span),
        remaining_best_gain_after8_normalized=float((best[-1]-best[7])/span),
        terminal_tier='at_least_95pct_max' if best[-1]>=.95*hi else
                      '80_to_95pct_max' if best[-1]>=.8*hi else 'below_80pct_max',
        **{f'first{pct}_total':next((q for q,v in enumerate(y,1) if v>=pct/100*hi),None) for pct in (80,90,95)})
    for n in (4,8,12,20):
        d=helpers.design(pool,state['history'][:n])
        result.update({f'Gower{n}':d['mean_Gower_distance'],f'one_factor_pairs{n}':d['one_factor_pairs'],
            f'category_contrast_pairs{n}':d['categorical_contrast_pairs'],
            f'response_SD{n}_normalized':float(y[:n].std(ddof=0)/span),
            f'category_support{n}':d['category_support']})
    rounds=[]
    decisions=state['decisions'];expected=len(y)-init
    assert len(decisions)==expected,(path_id,len(decisions),expected)
    for i,d in enumerate(decisions):
        total=init+i+1;obs=state['history'][total-1]
        folder,q,reused=asset_folder(record,total)
        cp=folder/'rounds'/f'{q:02}_commit.json'
        if not cp.exists():
            rounds.append(dict(path_id=path_id,total_observations=total,commit_available=False));continue
        c=read(cp)
        assert c['selected_id']==obs['candidate_id']
        # V8 stored a bare canonical commit and put its digest only in the
        # decision. Later serializers also embed that digest in the commit.
        computed=helpers.digest({k:v for k,v in c.items() if k!='commit_sha256'})
        assert computed==d['commit_sha256'],str(cp)
        if 'commit_sha256' in c:assert c['commit_sha256']==computed,str(cp)
        old=state['history'][:total-1]
        incumbent=max((h['raw_value'] for h in old),default=None)
        menu=c.get('display_ids',c.get('ordered_ids',c.get('menu_ids',[])))
        assert not set(menu)&{h['candidate_id'] for h in old}
        menu_values=[float(truth[pool.ix[cid]]) for cid in menu]
        reference=c.get('BO_reference_id',d.get('BO_reference_id'))
        rv=float(truth[pool.ix[reference]]) if reference in pool.ix else None
        improved=incumbent is not None and obs['raw_value']>incumbent
        changed=None
        if old:
            anchor=max(old,key=lambda h:h['raw_value'])
            changed=helpers.axis_changes(pool,pool.points[pool.ix[obs['candidate_id']]],pool.points[pool.ix[anchor['candidate_id']]])
        diag=score_feedback(d)
        row=dict(path_id=path_id,family=record['family'],task=record['task'],method=record['method'],
            regime=record['regime'],total_observations=total,query=q,commit_available=True,
            prefix_asset_reused=reused,selected_id=obs['candidate_id'],objective=obs['raw_value'],
            incumbent_before=incumbent,cumulative_best=float(best[total-1]),
            native_reference_id=reference,reference_objective=rv,menu_best=max(menu_values) if menu_values else None,
            selected_in_declared_menu=obs['candidate_id'] in menu,
            changed_from_reference=reference!=obs['candidate_id'] if reference else None,
            missed_reference_record=bool(incumbent is not None and rv is not None and rv>incumbent and not improved),
            local_record_advantage_normalized=(max(obs['raw_value']-incumbent,0)-max(rv-incumbent,0))/span
                if incumbent is not None and rv is not None else None,
            menu_offers95=bool(menu_values and max(menu_values)>=.95*hi),
            missed95_before_hit=bool(menu_values and max(menu_values)>=.95*hi and obs['raw_value']<.95*hi and
                                     (incumbent is None or incumbent<.95*hi)),
            low_incumbent_one_factor=bool(changed is not None and sum(changed)==1 and incumbent<.5*hi),
            one_factor_vs_incumbent=bool(changed is not None and sum(changed)==1),
            editor_call=bool(c.get('memory') is not None or (c.get('refresh') or {}).get('allowed',False)),**diag)
        rounds.append(row)
    for cut in (8,20):
        selected=[r for r in rounds if r.get('commit_available') and r['total_observations']<=cut]
        for field in ('changed_from_reference','missed_reference_record','missed95_before_hit',
                      'low_incumbent_one_factor','one_factor_vs_incumbent','editor_call','KF_updates',
                      'KF_increases','KF_decreases','new_suppressions','numeric_ranker_updates','candidate_card_updates'):
            result[f'{field}_through{cut}']=sum(r[field] or 0 for r in selected)
        result[f'local_record_advantage_through{cut}']=sum(r['local_record_advantage_normalized'] or 0 for r in selected)
    result['commit_rows_available']=sum(r.get('commit_available',False) for r in rounds)
    result['first_menu95_total']=next((r['total_observations'] for r in rounds if r.get('menu_offers95')),None)
    return result,rounds


def main():
    records=rows(OUT/'path_catalogue.csv')
    for r in records:
        for key in ('initial','queries','seed','observations'):r[key]=int(r[key])
        for key in ('primary','explicit_replay'):r[key]=r[key]=='True'
        r['SearchAUC']=float(r['SearchAUC']);r['final_best']=float(r['final_best'])
    tracked(__file__);tracked(helpers.__file__);tracked(ROOT/'docs/v13_all_paths_20261104.txt')
    for task,full in FULL.items():
        pool=late.pool_for(full) if task=='aryl' else knowledge_gp.load_task(ROOT,full,seed=0)
        POOLS[task]=pool;tracked(pool.source_path)
        with np.load(io.BytesIO(tracked(pool.source_path.with_name('oracle.npz')))) as z:
            TRUTH[task]=z['raw_values'].copy() if 'raw_values' in z else z['values'].copy()*pool.scale
        helpers.BOUNDS[pool.task]={a:(min(p[a] for p in pool.points),max(p[a] for p in pool.points))
            for a,k in zip(pool.axes,pool.kinds) if k=='continuous'}
    for r in records:
        b=tracked(ROOT/r['source_path']);assert hashlib.sha256(b).hexdigest()==r['source_sha256']
        STATES[r['path_id']]=json.loads(b.decode('utf-8-sig'))
    profiles=[];round_rows=[]
    with ThreadPoolExecutor(max_workers=8) as executor:
        for r,rr in executor.map(path_profile,records):profiles.append(r);round_rows.extend(rr)
    gp_rows=[]
    with ThreadPoolExecutor(max_workers=8) as executor:
        for item in executor.map(gp_profile,[(r,n) for r in records for n in (4,8,12,19)]):gp_rows.append(item)
    gp_index={(g['path_id'],g['observed']):g for g in gp_rows}
    for r in profiles:
        for n in (4,8,12,19):
            g=gp_index[r['path_id'],n]
            for key in ('mean_SD_normalized','coverage95','high_underestimated_fraction',
                        'mixed_category_correlation','exact_acquisition_maxima'):
                r[f'GP_{key}_{n}']=g.get(key)
    save('all_path_profiles.csv',profiles);save('all_decision_features.csv',round_rows)
    save('all_GP_prefixes.csv',gp_rows)
    failures=[r for r in gp_rows if not r['GP_schema_available']]
    fact=dict(status='PASS_ALL_PATH_PROFILES',paths=len(profiles),primary_paths=sum(r['primary'] for r in profiles),
        decision_rows=len(round_rows),commit_rows_available=sum(r.get('commit_available',False) for r in round_rows),
        GP_prefixes=len(gp_rows),GP_prefixes_available=sum(r['GP_schema_available'] for r in gp_rows),
        GP_prefixes_without_commit_hash=sum(r.get('GP_commit_hash_available') is False for r in gp_rows),
        missing_GP_schemas=failures,unparsed_feedback_events=sum(r.get('unparsed_feedback_events',0) for r in round_rows),
        new_GP_fits=0,new_LLM_calls=0,new_experiments=0)
    for name,h in READS.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
    (OUT/'profile_audit.json').write_text(json.dumps(fact,indent=2)+'\n',encoding='utf-8')
    (OUT/'profile_source_hashes.json').write_text(json.dumps(READS,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in fact.items() if k!='missing_GP_schemas'},indent=2))
    if failures:print('MISSING_SCHEMAS',len(failures))


if __name__=='__main__':main()
