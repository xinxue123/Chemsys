"""Qualification and full-population reporting for matched regional guidance."""
from __future__ import annotations
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import statistics as st
import numpy as np
import torch

from . import guarded_suzuki_engine as e
from . import guarded_suzuki_study as s


def independent_subset(base,member,budget,queried,ids,hids):
    available=[i for i in range(len(ids)) if not queried[i]]
    maximum=max(float(base[i]) for i in available)
    valid=[i for i in available if maximum-float(base[i])<=budget]
    for count in range(len(hids),-1,-1):
        for retained in sorted(combinations(sorted(hids),count)):
            witnesses=[i for i in valid if all(member[hids.index(h),i] for h in retained)]
            if witnesses:
                selected=min(witnesses,key=lambda i:(-float(base[i]),ids[i]))
                return tuple(retained),int(selected),witnesses
    raise AssertionError('empty subset must be available')


def qualify():
    """Interface and mathematical fixtures; these are not optimization results."""
    from threadpoolctl import threadpool_limits
    from botorch.acquisition.logei import qLogExpectedImprovement
    from botorch.sampling.normal import SobolQMCNormalSampler
    s.worker_init()
    pool=s.read(s.ROOT/'inputs/case1/pool.json');advice=s.read(s.ROOT/'inputs/case1/advice.json')['0']['correct']
    y=np.load(s.ROOT/'inputs/case1/oracle.npz')['values']
    lookup={key:i for i,key in enumerate(pool['ids'])}
    history=[{'candidate_id':key,'y':float(y[lookup[key]])} for key in pool['initial_ids']['0']]
    with threadpool_limits(1):
        cache=e.ScoreCache(pool['z'],pool['ids'],s.ROOT/'qualification_cache',1,0)
        base,base_meta=cache.score(history)
        full=tuple(c['id'] for c in advice['clauses']);box=e.region_for(advice,full)
        guided,guide_meta=cache.score(history,box,advice['confidence'])
        key,value=cache.model(history)
        # Independent original selector, using native regional construction.
        tilt=cache.boo.LinearExponentialRegionalMeanTiltPlugAndPlay(
            bounds=torch.tensor([[0.]*4,[1.]*4],dtype=torch.double),grid_size=512,smooth=.08,dtype=torch.double)
        e.seed_all(guide_meta['mc_seed']);tilt.set_box_region(*box)
        tilt.fit_lambda_by_delta(value['model'],guide_meta['delta'],observation_noise=False);tilt.prepare_cache(value['model'])
        model=cache.boo.TiltedModel(value['model'],tilt).eval()
        acq=qLogExpectedImprovement(model=model,best_f=value['best_std'],
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([512]),seed=guide_meta['mc_seed']))
        selected=cache.boo._greedy_select_with_min_dist(acq,cache.z,n=1,r_min=None)
        index=e.argmax(guided,np.ones(len(guided),dtype=bool),pool['ids'])
        assert torch.equal(selected[0],cache.z[index])
        zero=cache.boo.LinearExponentialRegionalMeanTiltPlugAndPlay(
            bounds=torch.tensor([[0.]*4,[1.]*4],dtype=torch.double),grid_size=512,smooth=.08,dtype=torch.double)
        zero.set_box_region([0.]*4,[1.]*4);zero.fit_lambda_by_delta(value['model'],0.);zero.prepare_cache(value['model'])
        x=cache.z[:32].unsqueeze(1)
        with torch.no_grad():
            p0=cache.boo.TiltedModel(value['model'],zero).posterior(x);p1=model.posterior(x)
        covariance_error=float((p0.variance-p1.variance).abs().max());assert covariance_error==0.
        restored=e.ScoreCache(pool['z'],pool['ids'],s.ROOT/'qualification_cache',1,0)
        _,restored_model=restored.model(history)
        assert all(torch.equal(v,restored_model['model'].state_dict()[k]) for k,v in value['model'].state_dict().items())
    # Independent subset enumeration for new score/soft-admission interface.
    rng=np.random.default_rng(531027);checks=0
    for n in (6,17,41):
        for m in range(5):
            for trial in range(8):
                b=rng.normal(size=n);h=rng.random((m,n))>.35;q=rng.random(n)<.15;q[0]=False
                ids=[f'p{i:03d}' for i in range(n)];hids=[f'H{i+1}' for i in range(m)]
                a={'confidence':.7,'clauses':[{'id':key,'coordinate':j%4,'interval':[0.,1.]} for j,key in enumerate(hids)]}
                budget=float(rng.uniform(.1,2.))
                kept,index,witnesses=independent_subset(b,h,budget,q,ids,hids)
                for method in ('B4_base','B4_LGBO'):
                    selected,event=e.choose_policy(method,b,a,h,budget,q,ids,lambda subset:b+.15*np.arange(n))
                    assert tuple(event['retained_ids'])==kept and not event['budget_violation']
                    if method=='B4_base':assert selected==index
                    else:assert event['retained_box_witness_count']==len(witnesses)
                    checks+=1
    qualities=s.read(s.OUT/'prior_quality_evaluation_only.json')
    assert all(abs(q['region_optimum_gap_pct'])<1e-10 for q in qualities if q['regime']=='correct')
    assert all(q['region_optimum_gap_pct']>0 for q in qualities if q['regime'] in ('wrong_category','mixed'))
    result={'status':'PASS','scope':'source, cache, covariance, selector and subset fixtures only',
        'original_regional_argmax_equal':True,'covariance_error':covariance_error,'cached_GP_state_equal':True,
        'independent_subset_decisions':checks,'common_pool_candidates':len(pool['ids']),
        'MC_paths':base_meta['mc_paths'],'native_GP_class':type(value['model']).__name__,
        'native_kernel':type(value['model'].covar_module).__name__,
        'operator_compiles_after_warmup':sum(s.JIT.values()),'controlled_correct_and_wrong_labels_verified':True}
    s.write(s.OUT/'qualification.json',result)
    return result


def _states(stage):return [s.read(f) for f in sorted((s.ROOT/stage/'runs').glob('*/state.json'))]


def aggregate(rows):
    return {'n':len(rows),'complete':sum(r['status']=='COMPLETE' for r in rows),
        'AUC_mean':st.mean(r['auc'] for r in rows),'AUC_sd':st.stdev(r['auc'] for r in rows) if len(rows)>1 else 0.,
        'final_mean':st.mean(r['final'] for r in rows),'initial_mean':st.mean(r['initial'] for r in rows),
        'queries':sum(r['queries'] for r in rows),
        'by_case':{str(c):{'n':sum(r['case']==c for r in rows),
            'AUC_mean':st.mean(r['auc'] for r in rows if r['case']==c),
            'final_mean':st.mean(r['final'] for r in rows if r['case']==c)} for c in sorted({r['case'] for r in rows})}}


def report():
    p=s.check_protocol();selection=s.read(s.OUT/'selected_budget.json');ratio=selection['ratio']
    states=_states('test');assert len(states)==p['test_unique_runs']
    assert all(r['status'] in ('COMPLETE','FAILED') for r in states)
    rows=[s.state_metrics(state) for state in states];tables={};paired={}
    def get(method,regime,case,seed):
        return next(r for r in rows if r['method']==method and r['case']==case and r['seed']==seed and (r['regime']==regime or method=='B0'))
    for regime in e.REGIMES:
        tables[regime]={}
        for method in e.METHODS:
            subset=[get(method,regime,c,seed) for c in (2,3,4) for seed in s.SEEDS]
            tables[regime][method]=aggregate(subset)
        for control in ('B0','LGBO_region','B3_gate','LGBO_gate','B4_base','A2_LGBO'):
            diffs=[{'case':c,'seed':seed,'AUC_delta':get('B4_LGBO',regime,c,seed)['auc']-get(control,regime,c,seed)['auc'],
                'final_delta':get('B4_LGBO',regime,c,seed)['final']-get(control,regime,c,seed)['final']}
                for c in (2,3,4) for seed in s.SEEDS]
            paired[regime+'_B4_LGBO_minus_'+control]={'mean_AUC_delta':st.mean(r['AUC_delta'] for r in diffs),
                'by_case':{str(c):st.mean(r['AUC_delta'] for r in diffs if r['case']==c) for c in (2,3,4)},'pairs':diffs}
    good=tables['correct'];base=good['B0']['AUC_mean'];guided=good['LGBO_region']['AUC_mean'];protected=good['B4_LGBO']['AUC_mean']
    gain=guided-base;retention=(protected-base)/gain if gain>=1 else None
    harmful=lambda method:st.mean(tables[g][method]['AUC_mean'] for g in ('wrong_category','mixed'))
    increments={method:harmful('B4_LGBO')-harmful(method) for method in ('LGBO_region','B3_gate','LGBO_gate','A2_LGBO')}
    gates={'correct_prior_mean_within_1pp':protected>=guided-1,
        'positive_guidance_gain_retained_80pct':retention is None or retention>=.8,
        'wrong_mixed_mean_improves_LGBO_at_least_2pp':increments['LGBO_region']>=2,
        'beats_each_simple_guard_on_wrong_mixed_mean':all(increments[m]>0 for m in ('B3_gate','LGBO_gate','A2_LGBO'))}
    mechanism={}
    for regime in e.REGIMES:
        mechanism[regime]={}
        for method in e.METHODS:
            sample=[r for r in states if r['method']==method and (r['regime']==regime or method=='B0')]
            events=[event for r in sample for event in r['decisions']]
            joint=0
            for event in events:
                detail=event.get('minimal_revocation_receipt')
                if detail:
                    checks=detail['subset_checks'];n=len(checks).bit_length()-1
                    joint+=int(not checks[-1]['eligible_count'] and all(checks[1<<j]['eligible_count'] for j in range(n)))
            mechanism[regime][method]={'decisions':len(events),
                'query_changed_from_same_history_base':sum(v['selected_id']!=v['reference_id'] for v in events),
                'mean_retained_clauses':st.mean(len(v['retained_ids']) for v in events) if events else None,
                'revocation_rounds':sum(bool(v['revoked_ids']) for v in events),
                'joint_only_admission_conflicts':joint,
                'queries_outside_retained_preference_box':sum(not v['selected_satisfies_retained_box'] for v in events),
                'maximum_base_logei_loss':max((v['base_logei_loss'] for v in events),default=None)}
    harmful_recovery={}
    for regime in ('wrong_category','mixed','assistant_initial'):
        harmful_recovery[regime]={}
        for method in e.METHODS:
            differences=[get(method,regime,c,seed)['auc']-get('B0',regime,c,seed)['auc'] for c in (2,3,4) for seed in s.SEEDS]
            harmful_recovery[regime][method]={'mean_difference_from_B0_pct':st.mean(differences),
                'mean_downside_loss_to_B0_pct':st.mean(max(-v,0.) for v in differences),
                'within_1pp_of_B0_or_better':sum(v>=-1 for v in differences),'n':15}
    result={'study':p['study'],'protocol_sha256':p['protocol_sha256'],'selected_budget':selection,
        'test_rows':len(rows),'tables':tables,'paired_comparisons':paired,'runs':rows,
        'mechanism':mechanism,'downside_and_recovery':harmful_recovery,
        'empirical_acceptance':{'all_gates_met':all(gates.values()),'gates':gates,'correct_gain_retention':retention,
            'correct_LGBO_gain_over_B0_pct':gain,'gain_retention_applicable':gain>=1,
            'wrong_mixed_AUC_deltas':increments,'statistical_claim':'descriptive exploratory gates, not proof of population noninferiority'},
        'limitations':['Three reaction test scenarios; same chat has prior outcome exposure.',
            'Correct/wrong stress priors use reference outcomes; this does not measure automatic generation of sound chemical knowledge.',
            'Regional-only finite-pool LGBO component; not a rerun of the official complete method.',
            'B4_LGBO admits soft preferences; its queries need not satisfy the retained preference box.',
            'Opportunity protection is on common MC logEI scores, not true yield.']}
    s.write(s.OUT/'results.json',result)
    return {'status':'COMPLETE','empirical_acceptance':result['empirical_acceptance'],'tables':tables}


def verify():
    p=s.check_protocol();result=s.read(s.OUT/'results.json');states=_states('test')
    assert result['protocol_sha256']==p['protocol_sha256']
    assert result['runs']==[s.state_metrics(state) for state in states]
    checks=0;metric_positions=0;oracle_queries=0;violations=0;soft_outside=0;hard_failures=0
    independent_mismatches=[];cache_arrays={};all_policy_checks=0
    def load_scores(case,seed,meta):
        key=(case,seed,meta['key'])
        if key not in cache_arrays:
            with np.load(s.ROOT/'cache'/f'c{case}_s{seed}'/(meta['key']+'.npz'),allow_pickle=False) as data:
                assert json.loads(str(data['metadata']))==meta
                cache_arrays[key]=data['scores'].copy()
        return cache_arrays[key]
    for state in states:
        case,seed=state['case'],state['seed']
        pool=s.read(s.ROOT/'inputs'/f'case{case}'/'pool.json');ids=pool['ids'];idx={key:i for i,key in enumerate(ids)}
        values=np.load(s.ROOT/'inputs'/f'case{case}'/'oracle.npz')['values']
        h=state['history'];assert [r['candidate_id'] for r in h[:4]]==pool['initial_ids'][str(seed)]
        assert len({r['candidate_id'] for r in h})==len(h)
        assert all(r['y']==float(values[idx[r['candidate_id']]]) for r in h)
        advice=s.read(s.ROOT/'inputs'/f'case{case}'/'advice.json')[str(seed)].get(state['regime'],{'clauses':[]})
        member=e.clause_membership(pool['z'],advice);hids=[c['id'] for c in advice['clauses']]
        for j,event in enumerate(state['decisions']):
            assert event['iteration']==j+1 and h[j+4]['candidate_id']==event['selected_id']
            scores=load_scores(case,seed,event['base_acquisition'])
            queried=np.array([key in {r['candidate_id'] for r in h[:j+4]} for key in ids])
            available=~queried;maximum=float(scores[available].max());chosen=idx[event['selected_id']]
            assert maximum-float(scores[chosen])==event['base_logei_loss']
            if state['method'] in e.GATED:
                budget=e.budget_at(state['ratio'],j+1);assert budget==event['budget_logei']
                violations+=int(maximum-scores[chosen]>budget)
            if state['method'] in ('B4_base','B4_LGBO'):
                retained,selected,witness=independent_subset(scores,member,event['budget_logei'],queried,ids,hids)
                if retained!=tuple(event['retained_ids']):independent_mismatches.append(state['run_id'])
                if state['method']=='B4_base':hard_failures+=int(selected!=chosen)
                else:soft_outside+=int(not event['selected_satisfies_retained_box'])
                checks+=1
            # Independent reconstruction of all seven decision rules from
            # frozen arrays, without calling the production policy function.
            reference=min(np.flatnonzero(available),key=lambda i:(-float(scores[i]),ids[i]))
            budget=e.budget_at(state['ratio'],j+1) if state['ratio'] else 0.
            eligible=available&((maximum-scores)<=budget)
            full=tuple(sorted(hids));kept=tuple(event['retained_ids'])
            if state['method'] in ('B4_base','B4_LGBO'):
                expected_kept,hard_choice,_=independent_subset(scores,member,budget,queried,ids,hids)
                assert kept==expected_kept
            elif state['method']=='A2_LGBO':
                assert kept==(full if (eligible&member.all(0)).any() else ())
            elif state['method']!='B0':assert kept==full
            def best(array,mask):return min(np.flatnonzero(mask),key=lambda i:(-float(array[i]),ids[i]))
            guided=scores
            if kept and state['method']!='B4_base':
                guided=load_scores(case,seed,event['guided_acquisitions'][','.join(kept)])
            if state['method']=='B0':expected=reference
            elif state['method']=='B4_base':expected=hard_choice
            elif state['method']=='LGBO_region':expected=best(guided,available)
            elif state['method']=='B3_gate':
                proposed=best(guided,available);expected=proposed if eligible[proposed] else reference
            else:expected=best(guided,eligible)
            assert expected==chosen,(state['run_id'],j+1)
            all_policy_checks+=1
        ys=[r['y'] for r in h];curve=[max(ys[:min(i,len(ys))]) for i in range(1,21)]
        assert abs(sum(curve[4:])/16*100-s.state_metrics(state)['auc'])<1e-12
        metric_positions+=20;oracle_queries+=len(ys)
    assert not violations and not hard_failures and not independent_mismatches
    selection=s.read(s.OUT/'selected_budget.json');start=s.read(s.OUT/'test_start_receipt.json')
    assert start['selected_budget_sha256']==s.filehash(s.OUT/'selected_budget.json') and start['test_states_before_start']==0
    assert start['B4_solver_source_sha256']==s.filehash('chem/decision/domain_guard.py')
    # Recompute the development choice from raw history by a separate mean.
    dev=_states('development');assert len(dev)==p['development_unique_runs'] and all(r['status']=='COMPLETE' for r in dev)
    def raw_auc(state):
        y=[v['y'] for v in state['history']]
        return sum(max(y[:i]) for i in range(5,21))/16*100
    def dev_mean(method,regime=None,ratio=None):
        values=[raw_auc(row) for row in dev if row['method']==method and (regime is None or row['regime']==regime) and (ratio is None or row['ratio']==ratio)]
        return st.mean(values)
    b0=dev_mean('B0');lg=dev_mean('LGBO_region','correct');options=[]
    for ratio in s.RATIOS:
        good=dev_mean('B4_LGBO','correct',ratio);retained=(good-b0)/(lg-b0) if lg-b0>=1 else None
        wrong=dev_mean('B4_LGBO','wrong_category',ratio);mixed=dev_mean('B4_LGBO','mixed',ratio)
        if good>=lg-1 and (retained is None or retained>=.8):options.append((ratio,min(wrong,mixed),(wrong+mixed)/2))
    expected_ratio=min(options,key=lambda row:(-row[1],-row[2],row[0]))[0] if options else 16
    assert expected_ratio==selection['ratio'] and bool(options)==selection['development_qualified']
    original=s.read('research/suzuki_official/release_inventory.json')
    for row in original['files']:assert s.filehash(row['path'])==row['sha256']
    receipt={'status':'PASS','protocol_sha256':p['protocol_sha256'],'test_trajectories':len(states),
        'completed':sum(r['status']=='COMPLETE' for r in states),'oracle_queries':oracle_queries,
        'independently_recomputed_budget_positions':metric_positions,'independent_subset_decisions':checks,
        'independently_reconstructed_policy_decisions':all_policy_checks,'development_budget_selection_recomputed':True,
        'subset_mismatches':len(independent_mismatches),'base_score_budget_violations':violations,
        'hard_B4_selection_mismatches':hard_failures,'soft_prior_queries_outside_retained_box':soft_outside,
        'operator_compiles_in_loop':sum(r['operator_compiles_in_loop'] for r in states),
        'protected_primary_files_unchanged':len(original['files']),
        'empirical_acceptance':result['empirical_acceptance'],
        'results_sha256':s.filehash(s.OUT/'results.json')}
    s.write(s.OUT/'validation_receipt.json',receipt)
    return receipt
