"""Inspect archived V13 path quality without fitting, choosing or querying."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from statistics import mean
import csv
import hashlib
import io
import json
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/reports/v13_path_quality_audit_20261104'
DATA=ROOT/'paper/v13_late/data'
TASKS=('c1','c2','c3','c4','perera','doyle','snar','hplc')
READS={}
BOUNDS={}
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from chem.benchmarks import knowledge_gp
from cold_start_v13 import study as cold
from state_split_v13_experiment import study as warm
from reasoning_memory_v13 import control


def tracked(path):
    p=Path(path).resolve()
    b=p.read_bytes()
    READS[p.relative_to(ROOT).as_posix()]=hashlib.sha256(b).hexdigest()
    return b


def read(path):
    return json.loads(tracked(path).decode('utf-8-sig'))


def rows(path):
    return list(csv.DictReader(io.StringIO(tracked(path).decode('utf-8-sig'))))


def digest(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def close(a,b):
    assert np.isclose(float(a),float(b),rtol=1e-10,atol=1e-8),(a,b)


def save(name,records):
    if not records:
        return
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n')
        w.writeheader()
        w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v
                     for k,v in r.items()} for r in records)


def softplus(v):
    return float(np.logaddexp(0.,np.asarray(v,dtype=float).reshape(-1)[0]))


def category_parameters(parameters,task):
    if task.startswith('c'):
        prefix='covar_module.kernels.'
        a=softplus(parameters[prefix+'0.raw_outputscale'])
        b=softplus(parameters[prefix+'1.raw_outputscale'])
        c=softplus(parameters[prefix+'0.base_kernel.kernels.1.raw_outputscale'])
        l1=softplus(parameters[prefix+'0.base_kernel.kernels.1.base_kernel.raw_lengthscale'])+1e-6
        l2=softplus(parameters[prefix+'1.base_kernel.kernels.1.raw_lengthscale'])+1e-6
        total=a*(1+c)+b
        corr=(a*(1+c*np.exp(-1/l1))+b*np.exp(-1/l2))/total if total>1e-250 else None
        return dict(category_correlation=corr,category_axis_correlations=[],category_amplitude=c,
                    category_lengthscales=[l1,l2])
    if task in ('perera','doyle'):
        raw=np.asarray(parameters['covar_module.base_kernel.raw_lengthscale'],float).reshape(-1)
        ell=np.logaddexp(0.,raw)+1e-6
        return dict(category_correlation=None,
                    category_axis_correlations=np.exp(-1/(len(ell)*ell)).tolist(),
                    category_amplitude=softplus(parameters['covar_module.raw_outputscale']),
                    category_lengthscales=ell.tolist())
    return dict(category_correlation=None,category_axis_correlations=[],category_amplitude=None,
                category_lengthscales=[])


def verify_kernel_decode(parameters):
    """Two synthetic inputs, no fit or posterior; validate parameter transforms."""
    import torch
    from botorch.models.gp_regression_mixed import MixedSingleTaskGP
    torch.set_num_threads(1)
    x=torch.tensor([[.5,.5,.5,0.],[.5,.5,.5,1.]],dtype=torch.double)
    model=MixedSingleTaskGP(x,torch.tensor([[0.],[1.]],dtype=torch.double),cat_dims=[3])
    with torch.no_grad():
        for name,parameter in model.named_parameters():
            parameter.copy_(torch.tensor(parameters[name],dtype=torch.double))
        k=model.covar_module(x).to_dense().numpy()
    actual=float(k[0,1]/np.sqrt(k[0,0]*k[1,1]))
    expected=category_parameters(parameters,'c1')['category_correlation']
    close(actual,expected)
    return dict(native=actual,decoded=expected,new_GP_fits=0)


def axis_changes(pool,a,b):
    delta=[]
    for axis,kind in zip(pool.axes,pool.kinds):
        if kind=='categorical':
            delta.append(a[axis]!=b[axis])
        else:
            lo,hi=BOUNDS[pool.task][axis]
            delta.append(abs(a[axis]-b[axis])>1e-8*(hi-lo))
    return delta


def design(pool,history):
    n=len(history)
    points=[pool.points[pool.ix[h['candidate_id']]] for h in history]
    continuous=[a for a,k in zip(pool.axes,pool.kinds) if k=='continuous']
    categorical=[a for a,k in zip(pool.axes,pool.kinds) if k=='categorical']
    bounds=BOUNDS[pool.task]
    distances=[];one_factor=cat_contrasts=0
    for i in range(n):
        for j in range(i):
            changes=axis_changes(pool,points[i],points[j])
            one_factor+=int(sum(changes)==1)
            cat_contrasts+=int(sum(changes)==1 and any(
                changes[k] for k,kind in enumerate(pool.kinds) if kind=='categorical'))
            ds=[abs(points[i][a]-points[j][a])/(bounds[a][1]-bounds[a][0]) for a in continuous]
            ds += [float(points[i][a]!=points[j][a]) for a in categorical]
            distances.append(mean(ds))
    return dict(observations=n,mean_Gower_distance=mean(distances) if distances else None,
        one_factor_pairs=one_factor,categorical_contrast_pairs=cat_contrasts,
        category_support={a:len({p[a] for p in points}) for a in categorical})


def array_profile(spec):
    key,q,n,path,fit,scale,truth,highset,response_values,expected_hash,expected_available=spec
    b=tracked(path)
    assert hashlib.sha256(b).hexdigest()==expected_hash,str(path)
    with np.load(io.BytesIO(b),allow_pickle=False) as z:
        mu=z['mu_y']*scale;sd=z['sd_y']*scale;available=z['available'].copy()
        score=z['base_logei'].copy()
    np.testing.assert_array_equal(available,expected_available)
    params=read(fit)
    decoded=category_parameters(params.get('parameters',params),key[1])
    raw_error=np.abs(mu-truth)
    hs=highset & available
    result=dict(regime=key[0],task=key[1],seed=key[2],query=q,observed=n,
        observed_response_SD=float(np.std(response_values,ddof=0)),
        observed_response_range=float(np.ptp(response_values)),
        unqueried=int(available.sum()),mean_SD=float(sd[available].mean()),
        coverage95=float(np.mean(raw_error[available]<=1.96*sd[available])),
        RMSE=float(np.sqrt(np.mean(raw_error[available]**2))),
        high_candidates_unqueried=int(hs.sum()),
        high_candidates_underestimated_fraction=float(np.mean((mu+1.96*sd)[hs]<.95*float(truth.max()))) if hs.any() else None,
        exact_acquisition_maxima=int(np.count_nonzero(available&(score==score[available].max()))),
        **decoded)
    return key,n,result,dict(mu=mu,sd=sd,available=available)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    tracked(Path(__file__));tracked(ROOT/'docs/v13_path_quality_audit_20261104.txt')
    for module in (cold,warm,control,knowledge_gp):
        tracked(module.__file__)
    for name in ('mixed_kalman_suzuki_engine','reaction_transfer_engine','continuous_benchmark_engine'):
        tracked(ROOT/'code/chemist_system/chem/benchmarks'/f'{name}.py')
    catalogue=rows(DATA/'metrics.csv')
    selected=[r for r in catalogue if r['method'] in ('v13','bo','reasoning_bo')]
    assert len(selected)==192
    datasets={r['task']:r for r in rows(DATA/'datasets.csv')}
    pools={};truth={};highsets={}
    for task,d in datasets.items():
        pool=knowledge_gp.load_task(ROOT,d['full_task'],seed=0)
        pbytes=tracked(pool.source_path)
        assert hashlib.sha256(pbytes).hexdigest()==d['pool_sha256']
        b=tracked(ROOT/d['oracle_path'])
        assert hashlib.sha256(b).hexdigest()==d['oracle_sha256']
        with np.load(io.BytesIO(b),allow_pickle=False) as z:
            y=z['raw_values'].copy() if 'raw_values' in z else z['values'].copy()*pool.scale
        pools[task]=pool;truth[task]=y;highsets[task]=y>=.95*float(y.max())
        BOUNDS[pool.task]={axis:(min(pt[axis] for pt in pool.points),max(pt[axis] for pt in pool.points))
                          for axis,kind in zip(pool.axes,pool.kinds) if kind=='continuous'}
    states={};metadata={};base_metrics=[]
    for m in selected:
        key=m['regime'],m['task'],int(m['seed']),m['method']
        path=ROOT/m['state_path'];s=read(path)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==m['state_sha256']
        assert s['status']=='COMPLETE' and len(s['history'])==20
        assert len({h['candidate_id'] for h in s['history']})==20
        ini=int(m['initial']);budget=int(m['queries'])
        assert ini+budget==20 and len(s['decisions'])==budget
        vals=np.array([h['raw_value'] for h in s['history']]);best=np.maximum.accumulate(vals)
        for h in s['history']:
            close(h['raw_value'],truth[key[1]][pools[key[1]].ix[h['candidate_id']]])
        close(best[ini:].mean(),m['SearchAUC']);close(vals.max(),m['final_best'])
        record=dict(regime=key[0],task=key[1],seed=key[2],method=key[3],initial=ini,queries=budget,
            first_value=float(vals[0]),initial_best=float(best[ini-1]) if ini else None,
            SearchAUC=float(best[ini:].mean()),final_best=float(best[-1]),
            final_fraction_of_max=float(best[-1]/truth[key[1]].max()),
            first_final_total=int(np.flatnonzero(vals==best[-1])[0])+1,
            best4=float(best[3]),best8=float(best[7]),best12=float(best[11]),best16=float(best[15]),
            late_gain_after10=float(best[-1]-best[9]),raw_sequence=vals.tolist(),best_sequence=best.tolist(),
            **{f'first{pct}_total':next((q for q,y in enumerate(vals,1) if y>=pct/100*float(truth[key[1]].max())),None)
               for pct in (80,90,95)})
        base_metrics.append(record);states[key]=s;metadata[key]=m

    commit_specs=[]
    for key,m in metadata.items():
        if key[3]=='v13':
            commit_specs += [(key,q,(ROOT/m['state_path']).parent/'rounds'/f'{q:02}_commit.json')
                            for q in range(1,int(m['queries'])+1)]
    def load_commit(spec):
        key,q,path=spec
        return key,q,read(path)
    with ThreadPoolExecutor(max_workers=8) as executor:
        commits={(key,q):c for key,q,c in executor.map(load_commit,commit_specs)}
    assert len(commits)==1152
    round_rows=[];trajectory_rows=[];design_rows=[];profilespecs=[]
    fit_index={(r['regime'],r['task'],int(r['seed']),int(r['query'])):r
               for r in rows(DATA/'fit_inventory.csv')}
    kernel_check=None
    for key,s in states.items():
        if key[3]!='v13':continue
        pool=pools[key[1]];m=metadata[key];ini=int(m['initial']);budget=int(m['queries'])
        pre=deepcopy(s)
        for field in ('notes','decisions','insights','edit_calls','review_queries'):
            pre[field]=[]
        pre['history']=deepcopy(s['history'][:ini]);pre['status']='READY'
        summary=next(dict(r) for r in base_metrics if (r['regime'],r['task'],r['seed'],r['method'])==key)
        for q in range(1,budget+1):
            c=commits[key,q];h=s['history'][ini+q-1]
            assert c['commit_sha256']==digest({k:v for k,v in c.items() if k!='commit_sha256'})
            assert c['prestate_sha256']==control.digest(pre)
            assert c['selected_id']==h['candidate_id']
            assert c['BO_reference_id']==c['ordered_ids'][0]
            assert len(c['display_ids'])==5 and set(c['display_ids'])==set(c['ordered_ids'])
            assert h['candidate_id'] in c['display_ids']
            assert not set(c['display_ids']) & {o['candidate_id'] for o in pre['history']}
            incumbent=max((x['raw_value'] for x in pre['history']),default=None)
            point=pool.points[pool.ix[h['candidate_id']]]
            changes=None
            if pre['history']:
                anchor=max(pre['history'],key=lambda x:x['raw_value'])
                changes=axis_changes(pool,point,pool.points[pool.ix[anchor['candidate_id']]])
            menu_values=[float(truth[key[1]][pool.ix[cid]]) for cid in c['display_ids']]
            reference=float(truth[key[1]][pool.ix[c['BO_reference_id']]])
            improve=h['raw_value']>incumbent if incumbent is not None else True
            top95=.95*float(truth[key[1]].max())
            before_good=incumbent is not None and incumbent>=top95
            uses=c['decision']['knowledge_uses']
            feedback=s['decisions'][q-1]['feedback']
            row=dict(regime=key[0],task=key[1],seed=key[2],query=q,total_observations=ini+q,
                candidate_id=h['candidate_id'],objective=h['raw_value'],incumbent_before=incumbent,
                cumulative_best=max(h['raw_value'],incumbent) if incumbent is not None else h['raw_value'],
                own_GP_reference=c['BO_reference_id'],changed_from_GP=h['candidate_id']!=c['BO_reference_id'],
                reference_objective=reference,menu_best=max(menu_values),menu_good95=max(menu_values)>=top95,
                missed_good95_before_hit=not before_good and max(menu_values)>=top95 and h['raw_value']<top95,
                missed_reference_record=incumbent is not None and reference>incumbent and not improve,
                extra_record_vs_reference=incumbent is not None and improve and reference<=incumbent,
                changed_axes_vs_incumbent=sum(changes) if changes is not None else None,
                one_factor_vs_incumbent=sum(changes)==1 if changes is not None else False,
                low_incumbent_one_factor=bool(changes is not None and sum(changes)==1 and incumbent<.5*float(truth[key[1]].max())),
                forecast_count=len(c['decision']['forecasts']),KF_updates=len(feedback),
                KF_down=sum(f['calibration']['after']['reliability']<f['calibration']['before']['reliability'] for f in feedback),
                KF_up=sum(f['calibration']['after']['reliability']>f['calibration']['before']['reliability'] for f in feedback),
                new_suppressions=sum(f.get('recommendation_before')!='suppressed' and f.get('recommendation_after')=='suppressed' for f in feedback),
                knowledge_uses=uses,editor=c.get('memory') is not None,review=any(u['use']=='review_test' for u in uses),
                opportunity_gap=c['opportunity_gap'],comment=c['decision']['insight']['comment'],conditions=point)
            round_rows.append(row)
            pre=(cold.feedback if key[0]=='cold' else warm.feedback)(pre,c,h)
        assert pre==s
        own=[r for r in round_rows if (r['regime'],r['task'],r['seed'])==key[:3]]
        summary.update({name:sum(r[field] for r in own) for name,field in (
            ('changed_from_GP','changed_from_GP'),('KF_updates','KF_updates'),('KF_down','KF_down'),('KF_up','KF_up'),
            ('forecast_choices','forecast_count'),('new_suppressions','new_suppressions'),
            ('editor_calls','editor'),('review_queries','review'),('missed_good_menus','missed_good95_before_hit'),
            ('missed_GP_records','missed_reference_record'),('extra_GP_records','extra_record_vs_reference'),
            ('incumbent_one_factor_choices','one_factor_vs_incumbent'),('low_incumbent_one_factor_choices','low_incumbent_one_factor'))})
        summary['first_menu_good95']=next((r['total_observations'] for r in own if r['menu_good95']),None)
        summary['first5_one_factor']=sum(r['one_factor_vs_incumbent'] for r in own if r['query']<=5)
        summary['first5_low_one_factor']=sum(r['low_incumbent_one_factor'] for r in own if r['query']<=5)
        bo=next(r for r in base_metrics if (r['regime'],r['task'],r['seed'],r['method'])==key[:3]+('bo',))
        rbo=next(r for r in base_metrics if (r['regime'],r['task'],r['seed'],r['method'])==key[:3]+('reasoning_bo',))
        summary['AUC_vs_BO']=summary['SearchAUC']-bo['SearchAUC']
        summary['final_vs_BO']=summary['final_best']-bo['final_best']
        summary['AUC_vs_RBO']=summary['SearchAUC']-rbo['SearchAUC']
        summary['same_full_path_as_RBO']=[h['candidate_id'] for h in s['history']]==[h['candidate_id'] for h in states[key[:3]+('reasoning_bo',)]['history']]
        trajectory_rows.append(summary)
        for n in (4,8,12,20):
            design_rows.append(dict(regime=key[0],task=key[1],seed=key[2],**design(pool,s['history'][:n])))
        for n in (1,4,8,12,19):
            q=n-ini+1
            if not 1<=q<=budget:continue
            f=fit_index[key[:3]+(q,)]
            assert len(pre['history'])==20
            fb=tracked(ROOT/f['fit_record'])
            assert hashlib.sha256(fb).hexdigest()==f['fit_sha256']
            if kernel_check is None and key[1]=='c1' and n==4:
                pars=json.loads(fb.decode('utf-8'))
                kernel_check=verify_kernel_decode(pars.get('parameters',pars))
            profilespecs.append((key[:3],q,n,ROOT/f['GP_arrays'],ROOT/f['fit_record'],pool.scale,truth[key[1]],highsets[key[1]],
                                [h['raw_value'] for h in s['history'][:n]],f['GP_arrays_sha256'],
                                ~np.isin(pool.ids,[h['candidate_id'] for h in s['history'][:n]])))
    gp_rows=[];arrays={}
    with ThreadPoolExecutor(max_workers=8) as executor:
        for key,n,row,a in executor.map(array_profile,profilespecs):
            gp_rows.append(row);arrays[key+(n,)]=a
    extremes=[];common_profiles=[]
    for regime in ('cold','warm'):
        for task in TASKS:
            options=[r for r in trajectory_rows if r['regime']==regime and r['task']==task]
            order=sorted(options,key=lambda r:(r['final_best'],r['SearchAUC'],r['seed']))
            low,high=order[0],order[-1]
            tied=abs(high['final_best']-low['final_best'])<1e-8
            extremes.append(dict(regime=regime,task=task,comparison='speed_only_all_final_values_tied' if tied else 'terminal_quality',
                high_seed=high['seed'],low_seed=low['seed'],high_final=high['final_best'],low_final=low['final_best'],
                high_SearchAUC=high['SearchAUC'],low_SearchAUC=low['SearchAUC'],high_best4=high['best4'],low_best4=low['best4'],
                high_best8=high['best8'],low_best8=low['best8'],high_first90=high['first90_total'],low_first90=low['first90_total'],
                high_late_gain=high['late_gain_after10'],low_late_gain=low['late_gain_after10'],
                high_KF=high['KF_updates'],low_KF=low['KF_updates'],
                high_missed_menus=high['missed_good_menus'],low_missed_menus=low['missed_good_menus'],
                high_low_anchor_tests=high['first5_low_one_factor'],low_low_anchor_tests=low['first5_low_one_factor']))
            for n in (4,8,12,19):
                ah=arrays[regime,task,high['seed'],n];al=arrays[regime,task,low['seed'],n]
                common=ah['available']&al['available']
                item=dict(regime=regime,task=task,observed=n,high_seed=high['seed'],low_seed=low['seed'],common_candidates=int(common.sum()))
                for label,a,seed in [('high',ah,high['seed']),('low',al,low['seed'])]:
                    err=np.abs(a['mu']-truth[task])
                    g=next(r for r in gp_rows if (r['regime'],r['task'],r['seed'],r['observed'])==(regime,task,seed,n))
                    item[label+'_mean_SD']=float(a['sd'][common].mean())
                    item[label+'_coverage95']=float(np.mean(err[common]<=1.96*a['sd'][common]))
                    item[label+'_category_correlation']=g['category_correlation']
                    common_high=common&highsets[task]
                    item[label+'_high_underestimated']=float(np.mean(
                        (a['mu']+1.96*a['sd'])[common_high]<.95*float(truth[task].max()))) if common_high.any() else None
                    item[label+'_observed_response_SD']=g['observed_response_SD']
                common_profiles.append(item)
    save('all_method_metrics.csv',base_metrics);save('v13_trajectory_profiles.csv',trajectory_rows)
    save('v13_rounds.csv',round_rows);save('high_low_pairs.csv',extremes)
    save('design_prefixes.csv',design_rows);save('GP_prefixes.csv',gp_rows)
    save('paired_GP_common_pool.csv',common_profiles)
    facts=dict(status='PASS_READ_ONLY',method_trajectories_checked=len(states),
        measured_history_records_checked=sum(len(s['history']) for s in states.values()),
        V13_paths=64,V13_commit_feedback_replays=len(commits),GP_prefix_profiles=len(gp_rows),
        new_GP_fits=0,new_LLM_calls=0,new_sequential_experiments=0,
        native_kernel_decode_check=kernel_check,
        tied_terminal_comparisons=sum(r['comparison'].startswith('speed_only') for r in extremes),
        cold_AUC_wins_vs_BO=sum(r['AUC_vs_BO']>1e-8 for r in trajectory_rows if r['regime']=='cold'),
        cold_AUC_losses_vs_BO=sum(r['AUC_vs_BO']<-1e-8 for r in trajectory_rows if r['regime']=='cold'),
        source_files_read=len(READS))
    for name,h in READS.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
    (OUT/'source_hashes.json').write_text(json.dumps(READS,indent=2)+'\n',encoding='utf-8')
    (OUT/'audit.json').write_text(json.dumps(facts,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(facts,indent=2))
    for r in extremes:
        if r['regime']=='cold':print(json.dumps(r))


if __name__=='__main__':
    main()
