"""Independent uncertainty-policy audit and paired outcome summaries."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from itertools import combinations
from pathlib import Path
import json
import statistics as st

import numpy as np
import torch

from . import uncertain_suzuki_study as s
from . import uncertain_suzuki_engine as e
from . import guarded_suzuki_study as parent
from . import guarded_suzuki_engine as common
from . import llinbo_guarded_report as lr
from .official_suzuki_adapter import check_sources
from .suzuki_domain import digest

OLD=('B0','LGBO_region','LGBO_gate','B4_base','B4_LGBO','LLINBO_matched')
NEW=(*e.METHODS,*e.CONTROLS)

def states(stage):return [s.read(f) for f in sorted((s.ROOT/stage/'runs').glob('*/state.json'))]
def metric(row):return parent.state_metrics(row) if 'ratio' in row else lr.metric(row)
def auc(row):return metric(row)['auc']
def get_old(case,seed,regime,method):
    if method=='LLINBO_matched':return s.read(Path('data/runs/suzuki_llinbo_guarded/runs')/f'c{case}_s{seed}_{regime}_LLINBO_matched/state.json')
    return lr.baseline(case,seed,regime,method)

def summarize(rows):
    return {'n':len(rows),'completed':sum(r['status']=='COMPLETE' for r in rows),'AUC_mean':st.mean(auc(r) for r in rows),
        'AUC_sd':st.stdev(auc(r) for r in rows),'final_mean':st.mean(metric(r)['final'] for r in rows),
        'by_case':{str(c):st.mean(auc(r) for r in rows if r['case']==c) for c in sorted({r['case'] for r in rows})}}

def report():
    p=s.check_protocol();selection=s.read(s.OUT/'selected_parameters.json');rows=states('test')
    assert len(rows)==315 and all(r['status'] in ('COMPLETE','FAILED') for r in rows)
    tables={};paired={};mechanism={}
    for g in common.REGIMES:
        groups={m:[next(r for r in rows if r['case']==c and r['seed']==seed and r['method']==m and r['regime']==('none' if m=='sigma_BO' else g)) for c in (2,3,4) for seed in range(5)] for m in NEW}
        groups.update({m:[get_old(c,seed,g,m) for c in (2,3,4) for seed in range(5)] for m in OLD})
        tables[g]={m:summarize(v) for m,v in groups.items()};paired[g]={};mechanism[g]={}
        for method in NEW:
            mine=groups[method];events=[v for r in mine for v in r['decisions']]
            mechanism[g][method]={'decisions':len(events),'changed_from_same_history_base':sum(v['selected_id']!=v['reference_id'] for v in events),
                'probe_decisions':sum(v['probe_round'] for v in events),'probe_fallbacks':sum(v['boundary_probe_fallback'] for v in events),
                'mean_retained_clauses':st.mean(len(v['retained_ids']) for v in events),
                'epistemic_sd_mean_pp':st.mean(v['epistemic_sd_y']*100 for v in events),
                'predictive_sd_mean_pp':st.mean(v['predictive_sd_y']*100 for v in events),
                'predictive_1sd_coverage':st.mean(abs(v['predictive_z'])<=1 for v in events),
                'predictive_1_96sd_coverage':st.mean(abs(v['predictive_z'])<=1.96 for v in events),
                'prefix_inflated_1_96sd_coverage':st.mean(abs(v['predictive_z'])<=1.96*v['residual_scale'] for v in events),
                'mean_residual_scale':st.mean(v['residual_scale'] for v in events),
                'mean_clipped_AUC_downside_to_B0_pp':st.mean(max(auc(b)-auc(a),0.) for a,b in zip(mine,groups['B0']))}
            paired[g][method]={}
            for control in ('LGBO_region','LGBO_gate','B4_base','LLINBO_matched','sigma_probe','sigma_BO'):
                d=[auc(a)-auc(b) for a,b in zip(mine,groups[control])]
                paired[g][method][control]={'mean_AUC_delta_pp':st.mean(d),
                    'by_case':{str(c):st.mean(v for v,r in zip(d,mine) if r['case']==c) for c in (2,3,4)},
                    'wins':sum(v>1e-10 for v in d),'losses':sum(v<-1e-10 for v in d),'ties':sum(abs(v)<=1e-10 for v in d)}
    result={'study':p['study'],'protocol_sha256':p['protocol_sha256'],'selection':selection,'planned_test_runs':315,
        'completed_test_runs':sum(r['status']=='COMPLETE' for r in rows),'logical_test_queries':sum(len(r['history']) for r in rows),
        'tables':tables,'paired':paired,'mechanism':mechanism,
        'wrong_mixed_means':{m:st.mean(tables[g][m]['AUC_mean'] for g in ('wrong_category','mixed')) for m in (*NEW,*OLD)},
        'limits':['Three test chemistries, shared prior exposure, not blind.',
            'Residual variance inflation is a heuristic; observed-query coverage is not full-domain coverage.',
            'Independently selected k can differ between credible_B4 and residual_B4; aggregate comparison does not isolate calibration alone.',
            'No new LLM answers; dynamic LLINBO/projection is a system comparator, not identical knowledge cadence.']}
    s.write(s.OUT/'results.json',result)
    mechanism_diagnosis(rows)
    return {'completed':result['completed_test_runs'],'AUC':{g:{m:round(v['AUC_mean'],4) for m,v in t.items()} for g,t in tables.items()},'wrong_mixed_means':result['wrong_mixed_means']}

def mechanism_diagnosis(rows):
    """Post-hoc path comparisons and evaluator-only probe-domain ceilings."""
    result={'scope':'Post-hoc mechanism diagnostics; finite-pool ceilings do not measure information gain.',
        'B4_path_comparison':{},'residual_vs_credible':{},'boundary_domain':[]}
    for g in common.REGIMES:
        result['B4_path_comparison'][g]={}
        selected={m:sorted([r for r in rows if r['method']==m and r['regime']==g],key=lambda r:(r['case'],r['seed'])) for m in ('credible_B4','residual_B4')}
        for m,mine in selected.items():
            old=[get_old(r['case'],r['seed'],g,'B4_base') for r in mine]
            result['B4_path_comparison'][g][m]={'pairs':len(mine),
                'same_complete_history':sum(a['history']==b['history'] for a,b in zip(mine,old)),
                'same_incumbent_curve':sum(lr.metric(a)['curve']==lr.metric(b)['curve'] for a,b in zip(mine,old)),
                'same_AUC':sum(abs(auc(a)-auc(b))<1e-10 for a,b in zip(mine,old))}
        a,b=selected['residual_B4'],selected['credible_B4'];events=[v for r in a for v in r['decisions']]
        result['residual_vs_credible'][g]={'pairs':len(a),'same_complete_history':sum(x['history']==y['history'] for x,y in zip(a,b)),
            'same_incumbent_curve':sum(lr.metric(x)['curve']==lr.metric(y)['curve'] for x,y in zip(a,b)),
            'predictive_1_96sd_coverage':st.mean(abs(v['predictive_z'])<=1.96 for v in events),
            'inflated_coverage':st.mean(abs(v['predictive_z'])<=1.96*v['residual_scale'] for v in events),
            'mean_base_interval_width_pp':st.mean(2*1.96*v['predictive_sd_y']*100 for v in events),
            'mean_inflated_interval_width_pp':st.mean(2*1.96*v['predictive_sd_y']*v['residual_scale']*100 for v in events)}
    for case in (2,3,4):
        folder=parent.ROOT/'inputs'/f'case{case}';pool=s.read(folder/'pool.json');values=np.load(folder/'oracle.npz')['values']
        for g in common.REGIMES:
            # Artificial priors share geometry across seeds; actual priors need all5.
            for seed in ([0] if g!='assistant_initial' else range(5)):
                advice=s.read(folder/'advice.json')[str(seed)][g];h=common.clause_membership(pool['z'],advice);boundary=h.sum(0)==len(advice['clauses'])-1
                boundary_best=float(values[boundary].max()) if boundary.any() else None;maximum=float(values.max())
                result['boundary_domain'].append({'case':case,'seed':seed,'regime':g,'candidates':int(boundary.sum()),
                    'includes_any_pool_optimum':boundary_best is not None and maximum-boundary_best<1e-12,
                    'finite_pool_optimum_gap_pp':(maximum-boundary_best)*100 if boundary_best is not None else None})
    s.write(s.OUT/'mechanism_diagnosis.json',result)
    return result

def independent_subset(base,member,budget,queried,ids,hids):
    available=[i for i in range(len(ids)) if not queried[i]];best=max(float(base[i]) for i in available)
    eligible=[i for i in available if best-float(base[i])<=budget]
    for count in range(len(hids),-1,-1):
        for kept in combinations(sorted(hids),count):
            valid=[i for i in eligible if all(member[hids.index(h),i] for h in kept)]
            if valid:return tuple(kept),min(valid,key=lambda i:(-float(base[i]),ids[i]))
    raise AssertionError('empty subset must be feasible')

def verify():
    p=s.check_protocol();parent.check_protocol();check_sources();rows=states('test');dev=states('development')
    assert len(rows)==315 and len(dev)==210 and all(r['status']=='COMPLETE' for r in rows+dev)
    assert all(r['protocol_sha256']==p['protocol_sha256'] for r in rows)
    assert all(r['protocol_sha256'] in [p['protocol_sha256'],*p.get('accepted_development_protocols',[])] for r in dev)
    start=s.read(s.OUT/'test_start_receipt.json');assert start['test_states_before_start']==0
    assert start['selected_parameters_sha256']==s.filehash(s.OUT/'selected_parameters.json')
    selection=s.read(s.OUT/'selected_parameters.json');checks=0;subsets=0;initial_checks=0;nominal_changes=0;calibration_changes=0
    arrays={}
    def load(folder,key):
        identifier=str(folder/key)
        if identifier not in arrays:
            with np.load(folder/(key+'.npz'),allow_pickle=False) as d:arrays[identifier]={k:d[k].copy() for k in d.files}
        return arrays[identifier]
    for r in rows:
        case,seed,g=r['case'],r['seed'],r['regime'];folder=parent.ROOT/'inputs'/f'case{case}'
        pool=s.read(folder/'pool.json');ids=pool['ids'];ix={key:i for i,key in enumerate(ids)};y=np.load(folder/'oracle.npz')['values']
        advice=s.read(folder/'advice.json')[str(seed)].get(g,{'clauses':[]});hids=[v['id'] for v in advice['clauses']]
        member=common.clause_membership(pool['z'],advice);h=r['history'];cache=s.ROOT/'cache'/f'c{case}_s{seed}_{g}'
        assert len(h)==20 and len(r['decisions'])==16 and len({v['candidate_id'] for v in h})==20
        assert [v['candidate_id'] for v in h[:4]]==pool['initial_ids'][str(seed)]
        assert all(v['y']==float(y[ix[v['candidate_id']]]) for v in h)
        if r['method'] in e.METHODS:assert r['parameter']==selection['methods'][r['method']]['parameter']
        else:assert r['parameter']==selection['methods']['boundary_probe']['parameter']
        for t,event in enumerate(r['decisions'],1):
            scores=load(cache,event['base_acquisition']['key'])['scores'];post=load(cache,event['posterior_key'])
            prefix=h[:t+3]
            fit_key=digest({'history':sorted(prefix,key=lambda v:v['candidate_id']),'case':case,'seed':seed,'recipe':s.filehash(common.__file__)})
            mc_seed=38102026+case*10000+seed*100+len(prefix)-3
            extension_recipe=s.filehash(e.__file__)
            assert event['base_acquisition']['fit_key']==str(post['fit_key'])==fit_key
            assert event['base_acquisition']['mc_seed']==mc_seed
            assert event['posterior_key']==digest({'fit':fit_key,'posterior':'epistemic_and_predictive','recipe':extension_recipe})
            for kind,meta in event['guided_acquisitions'].items():
                assert meta['fit_key']==fit_key and meta['mc_seed']==mc_seed
                payload={'fit':fit_key,'box':common.region_for(advice,hids),'confidence':advice['confidence'],'mc_seed':mc_seed}
                if kind=='cap':payload.update(kappa=r['parameter'],recipe=extension_recipe)
                assert meta['key']==digest(payload)
            queried=np.array([key in {v['candidate_id'] for v in h[:t+3]} for key in ids]);available=~queried
            best=min(np.flatnonzero(available),key=lambda i:(-float(scores[i]),ids[i]));budget=float(np.log(4)*(16-t)/15)
            eligible=available&((scores[best]-scores)<=budget);chosen=ix[event['selected_id']]
            assert not queried[chosen] and float(scores[best]-scores[chosen])==event['base_logei_loss']<=budget
            assert budget==event['budget_logei'] and ids[best]==event['reference_id'] and event['selected_id']==h[t+3]['candidate_id']
            if t==1:
                old=get_old(case,seed,'correct','B0');key=old['decisions'][0]['base_acquisition']['key'];assert key==event['base_acquisition']['key']
                with np.load(parent.ROOT/'cache'/f'c{case}_s{seed}'/(key+'.npz')) as d:assert np.array_equal(scores,d['scores'])
                initial_checks+=1
            prior_events=r['decisions'][:t-1]
            scale=1. if len(prior_events)<3 else float(np.clip(np.sqrt(st.mean(min(float(v['predictive_z'])**2,100.) for v in prior_events)),1.,3.))
            assert abs(scale-event['residual_scale'])<1e-12
            def argmax(array,mask):return min(np.flatnonzero(mask),key=lambda i:(-float(array[i]),ids[i]))
            method=r['method'];parameter=r['parameter']
            if method in ('credible_B4','residual_B4'):
                k=parameter*(scale if method=='residual_B4' else 1.)
                potential=post['mu_std']+k*post['sd_std']>=np.max((post['mu_std']-k*post['sd_std'])[available])
                kept,expected=independent_subset(scores,member&potential[None,:],budget,queried,ids,hids)
                assert tuple(event['retained_ids'])==kept;subsets+=1
                _,nominal=independent_subset(scores,member,budget,queried,ids,hids)
                nominal_changes+=int(expected!=nominal)
                if method=='residual_B4':
                    mask=post['mu_std']+parameter*post['sd_std']>=np.max((post['mu_std']-parameter*post['sd_std'])[available])
                    _,no_scale=independent_subset(scores,member&mask[None,:],budget,queried,ids,hids)
                    calibration_changes+=int(expected!=no_scale)
            elif method=='sigma_cap':expected=argmax(load(cache,event['guided_acquisitions']['cap']['key'])['scores'],eligible)
            else:
                probe=t in ({3:(4,8,12),4:(3,6,9,12)}[int(parameter)])
                assert event['probe_round']==probe
                if probe:
                    mask=eligible.copy()
                    if method=='boundary_probe':
                        mask&=(member.sum(0)==len(hids)-1);fallback=not mask.any()
                        assert fallback==event['boundary_probe_fallback']
                        if fallback:mask=eligible.copy()
                    expected=argmax(post['sd_std'],mask)
                else:expected=best if method=='sigma_BO' else argmax(load(cache,event['guided_acquisitions']['full']['key'])['scores'],eligible)
            assert expected==chosen
            observed=h[t+3]['y'];assert float(post['mu_y'][chosen])==event['predicted_y']
            assert float(post['sd_y'][chosen])==event['epistemic_sd_y'] and float(post['sd_predictive_y'][chosen])==event['predictive_sd_y']
            assert (observed-event['predicted_y'])/max(event['predictive_sd_y'],1e-12)==event['predictive_z']
            checks+=1
        assert r['operator_compiles_in_loop']==0
    # Independent raw-prefix development selection and fallback.
    b0=st.mean(auc(get_old(1,seed,'correct','B0')) for seed in range(5));lg=st.mean(auc(get_old(1,seed,'correct','LGBO_region')) for seed in range(5));gain=lg-b0
    for m in e.METHODS:
        options=[]
        for parameter in e.PARAMETERS[m]:
            means={g:st.mean(auc(r) for r in dev if r['method']==m and r['parameter']==parameter and r['regime']==g) for g in ('correct','wrong_category','mixed')}
            good=means['correct'];qualified=good>=lg-1 and (gain<1 or (good-b0)/gain>=.8)
            if qualified:options.append((parameter,min(means['wrong_category'],means['mixed']),(means['wrong_category']+means['mixed'])/2))
        expected=min(options,key=lambda v:(-v[1],-v[2],v[0]))[0] if options else e.FALLBACK[m]
        assert selection['methods'][m]['parameter']==expected and selection['methods'][m]['development_qualified']==bool(options)
    protected={}
    for label,path in [('original','research/suzuki_official/release_inventory.json'),('matched','research/suzuki_guarded/release_inventory.json'),('LLINBO','research/suzuki_llinbo_guarded/release_inventory.json')]:
        inventory=s.read(path)
        for f in inventory['files']:assert s.filehash(f['path'])==f['sha256'],f['path']
        protected[label]=len(inventory['files'])
    result=s.read(s.OUT/'results.json')
    for g in common.REGIMES:
        for m in NEW:
            raw=[r for r in rows if r['method']==m and r['regime']==('none' if m=='sigma_BO' else g)]
            independent=st.mean(sum(max(v['y'] for v in r['history'][:i])*100 for i in range(5,21))/16 for r in raw)
            assert abs(independent-result['tables'][g][m]['AUC_mean'])<1e-10
    receipt={'integrity_status':'PASS','protocol_sha256':p['protocol_sha256'],'development_runs':210,'test_runs':315,
        'logical_queries':10500,'independent_policy_checks':checks,'independent_subset_checks':subsets,
        'exact_observation_prefix_cache_lineages_checked':checks,
        'initial_common_score_tables_exact':initial_checks,'uncertainty_B4_choices_changed_from_original_B4_same_prefix':nominal_changes,
        'residual_inflation_changes_from_same_k_uninflated_choice':calibration_changes,
        'budget_violations':0,'operator_compiles_in_loop':0,'protected_files_unchanged':protected,
        'development_selection_recomputed':True,'results_sha256':s.filehash(s.OUT/'results.json'),
        'boundary':'Implementation integrity only; predictive uncertainty and optimization advantage evaluated separately.'}
    s.write(s.OUT/'validation_receipt.json',receipt);return receipt

def audit_group(job):
    """Original LLINBO histories held fixed; no new optimization trajectory."""
    case,seed=job;folder=parent.ROOT/'inputs'/f'case{case}';pool=s.read(folder/'pool.json');ix={v:i for i,v in enumerate(pool['ids'])}
    y=np.load(folder/'oracle.npz')['values'];cache=s.cache_for(case,seed,'llinbo_audit','audit_cache');records=[]
    for g in common.REGIMES:
        row=get_old(case,seed,g,'LLINBO_matched');past=[]
        for t,event in enumerate(row['decisions'],1):
            history=row['history'][:t+3];post=cache.posterior_stats(history);_,value=cache.model(history)
            ib,ip=ix[event['reference_id']],ix[event['proposed_id']];selected=ix[event['selected_id']]
            with torch.no_grad():
                joint=value['model'].posterior(cache.z[[ib,ip]],observation_noise=False)
                cov=joint.mvn.covariance_matrix.numpy()
            diff_sd=float(np.sqrt(max(float(cov[0,0]+cov[1,1]-2*cov[0,1]),1e-18)))
            gap=float(post['mu_std'][ib]-post['mu_std'][ip]);scale=e.residual_scale(past)
            gates={'fixed_logEI':event['accepted'],'paired_sd_k1':gap<=diff_sd,
                'paired_sd_k2':gap<=2*diff_sd,'paired_sd_k2_residual':gap<=2*scale*diff_sd}
            incumbent=max(v['y'] for v in history);proposed_y=float(y[ip]);base_y=float(y[ib])
            actual=row['history'][t+3]['y'];z=(actual-float(post['mu_y'][selected]))/max(float(post['sd_predictive_y'][selected]),1e-12)
            records.append({'case':case,'seed':seed,'regime':g,'iteration':t,'gates':gates,'mean_gap_std':gap,
                'paired_epistemic_sd_std':diff_sd,'residual_scale':scale,'predictive_z':z,
                'proposal_improves_current_incumbent':proposed_y>incumbent+1e-10,
                'proposal_one_step_incumbent_advantage_vs_base_pp':(max(incumbent,proposed_y)-max(incumbent,base_y))*100})
            past.append({'predictive_z':z})
    assert cache.counters['fits']==0
    s.write(s.ROOT/'llinbo_audit'/f'c{case}_s{seed}.json',{'records':records,'cache_counters':cache.counters})
    return records

def audit_llinbo():
    s.check_protocol();jobs=[(c,seed) for c in (2,3,4) for seed in range(5)]
    with ProcessPoolExecutor(max_workers=4,initializer=s.worker_init) as pool:groups=list(pool.map(audit_group,jobs))
    records=[v for group in groups for v in group];result={}
    for g in common.REGIMES:
        subset=[v for v in records if v['regime']==g];result[g]={}
        for method in subset[0]['gates']:
            result[g][method]={'prefixes':len(subset),'accepted':sum(v['gates'][method] for v in subset),
                'rejected_proposal_would_improve_incumbent':sum(not v['gates'][method] and v['proposal_improves_current_incumbent'] for v in subset),
                'accepted_proposal_one_step_incumbent_worse_than_base':sum(v['gates'][method] and v['proposal_one_step_incumbent_advantage_vs_base_pp']<-1e-10 for v in subset)}
        result[g]['predictive_1_96sd_coverage']=st.mean(abs(v['predictive_z'])<=1.96 for v in subset)
    output={'scope':'960 frozen original LLINBO prefixes; offline diagnostics, not new policy trajectories or SearchAUC.',
        'paired_sd':'Variance of f(base)-f(proposal) includes their posterior covariance; no independent-point assumption.',
        'by_regime':result,'new_LLM_answers':0,'new_oracle_queries':0,'new_GP_fits':0,
        'caution':'Alternative gates change subsequent data and LLM inputs. These counts cannot establish improved sequential optimization.'}
    s.write(s.OUT/'llinbo_uncertainty_audit.json',output);return output
