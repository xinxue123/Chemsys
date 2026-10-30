"""Read-only, same-prefix audit of frozen SnAr/HPLC module decisions.

No LLM requests, GP refits, new trajectories or edits of frozen experiment files.
Counterfactual objectives are looked up retrospectively, after all choices have
been reconstructed, and measure only one-step effects at the actual prefix.
"""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import csv
import json
import os
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.chdir(ROOT)
from chem.benchmarks import continuous_benchmark_study as s
from chem.benchmarks import continuous_benchmark_engine as e
from chem.benchmarks import continuous_benchmark_llm as llm
from chem.benchmarks import continuous_benchmark_data as data
from chem.benchmarks.suzuki_domain import digest

OUT=Path('research/continuous_audit_20261030')

def csv_write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def prefix_state(state,t):
    before=state['decisions'][:t-1];banks={}
    for event in before:
        if event['bank_key'] is not None and event['bank_after'] is not None:banks[event['bank_key']]=deepcopy(event['bank_after'])
    call=next((c for c in state['calls'] if c['iteration']==t),None)
    knowledge=call['knowledge_before'] if call else before[-1]['knowledge'] if before else state['initial_knowledge']
    return {**deepcopy(state),'history':deepcopy(state['history'][:t+3]),'decisions':deepcopy(before),
        'knowledge':deepcopy(knowledge),'banks':banks,
        'region_trust':deepcopy(before[-1]['region_trust_after']) if before else {},
        'calls':[deepcopy(c) for c in state['calls'] if c['iteration']<t],
        'validations':[deepcopy(v) for v in state['validations'] if v['query']<t]}

def reject_refit(*args,**kwargs):raise RuntimeError('This read-only audit refuses a GP refit.')

def audit_round(job):
    task,t=job;e.warm();pool=s.pool(task);cache=e.Cache(pool);state=data.read(s.ROOT/'runs'/(task+'_New_Agent')/'state.json')
    own=prefix_state(state,t);before_own=deepcopy(own);call=next((c for c in state['calls'] if c['iteration']==t),None)
    folder=s.ROOT/'runs'/state['run_id']/'rounds';commit=data.read(folder/f'{t:02d}_committed.json')
    with np.load(folder/f'{t:02d}_candidates.npz') as f:arrays={k:f[k].copy() for k in f.files}
    check={'task':task,'query_round':t,'initialization_and_own_prefix':True,'pre_refresh_checked':False,'GP_refits':0}
    with patch.object(e,'fit_gpytorch_mll',reject_refit):
        before=e.setup(cache,own,t)
        reasons=e.event_reasons(own,before,t)
        if call:
            assert reasons==call['reasons'],(task,t,reasons,call['reasons'])
            assert own['knowledge']==call['knowledge_before']
            answer=llm.validate_region(call['answer']['raw'],pool,own)
            assert answer==call['answer']['plan']
            context=data.read(s.ROOT/'requests'/call['context_sha256']/'context.json')
            assert context['observed_prefix']==own['history']
            assert context['public']['B4_diagnosis']==json.loads(json.dumps(before['diagnosis'].to_dict()))
            assert context['public']['trust_gate']==before['trust']
            own['calls'].append(deepcopy(call));own['knowledge']=deepcopy(answer['knowledge'])
            after=e.setup(cache,own,t);check['pre_refresh_checked']=True
        else:
            assert not reasons,(task,t,reasons)
            after=before
        assert own['knowledge']==commit['knowledge']
        assert own['region_trust']==commit['region_trust_before']
        assert after['bank']==commit['bank_before'] and after['prior']==commit['bank_prior']
        assert after['trust']==commit['trust'] and after['amplitude']==commit['amplitude']
        assert pool['ids'][after['selected']]==commit['normal_proposal_id']
        assert after['diagnosis'].to_dict()['retained_ids']==tuple(commit['retained'])
        errors={}
        for name,computed in [('base_qLogEI',after['base']),('guided_qLogEI',after['score']),('raw_shift_std',after['raw_shift_std']),('deployed_shift_std',after['shift_std'])]:
            errors[name]=float(np.max(np.abs(computed-arrays[name])));assert errors[name]<1e-8,(task,t,name,errors[name])
        check['max_actual_array_replay_error']=max(errors.values())
        available=after['available'];eligible=after['eligible'];v=after['stats'];base=after['base'];scale=pool['objective_scale']
        proxy=after['information_proxy'];ranks=after['tie_priority'];inc=max(h['y'] for h in own['history'])*scale
        def choose(scores,mask):
            js=np.flatnonzero(mask);js=js[scores[js]==scores[js].max()];js=js[proxy[js]==proxy[js].max()]
            js=js[v['sd_y'][js]==v['sd_y'][js].max()];return int(js[np.argmin(ranks[js])])
        stage=[];raw_by_stage={}
        def record(name,j,scores,shift=None,alpha=None):
            stage.append({'task':task,'query_round':t,'stage':name,'candidate_id':pool['ids'][j],
                'GP_mean':float(v['mu_y'][j])*scale,'GP_latent_SD':float(v['sd_y'][j])*scale,
                'base_qLogEI':float(base[j]),'stage_qLogEI':float(scores[j]),'base_opportunity_loss':float(base[after['reference']]-base[j]),
                'normal_gate_eligible':bool(eligible[j]),'incumbent_before':inc,'alpha':alpha,
                'deployed_shift_raw_units':float(shift[j]*v['std']*scale) if shift is not None else 0.})
            raw_by_stage[name]=j
        def field_choice(name,knowledge,retained,alpha=1.,mask=None):
            spec=e.region(pool,knowledge,retained);raw,_=cache.field(own['history'],spec)
            scores,_=cache.score(own['history'],alpha*raw) if spec and alpha else (base,after['base_meta'])
            j=choose(scores,eligible if mask is None else mask);record(name,j,scores,alpha*raw,alpha)
            return j
        record('base_BO',after['reference'],base)
        field_choice('initial_region_a1',state['initial_knowledge'],[c['id'] for c in state['initial_knowledge']['clauses']])
        field_choice('current_full_region_a1',own['knowledge'],[c['id'] for c in own['knowledge']['clauses']])
        field_choice('B4_geometry_a1',own['knowledge'],after['retained'])
        pre_alpha=after['prior']['amplitude'] if after['prior'] else 0.
        field_choice('KF_before_conflict_gate',own['knowledge'],after['retained'],pre_alpha)
        conflict=bool(after['diagnosis'].revoked_ids) or after['diagnosis'].conflict_kind!='none'
        field_choice('B4_conflict_gate',own['knowledge'],after['retained'],0. if conflict else pre_alpha)
        record('normal_after_trust',after['selected'],after['score'],after['shift_std'],after['amplitude'])
        record('before_refresh_normal',before['selected'],before['score'],before['shift_std'],before['amplitude'])
        if call:
            # Geometry change without a bank reset: transport previous deployed
            # amplitude onto new retained field, still honor the NEW B4/trust gate.
            old_alpha=before['prior']['amplitude'] if before['prior'] else 0.
            if conflict or after['trust']['trust_blocked']:old_alpha=0.
            field_choice('new_geometry_old_amplitude',own['knowledge'],after['retained'],old_alpha)
        # Remove B4 from the current geometry, preserving that full geometry's
        # own KF bank and prequery evidence latch; this is a local toggle only.
        full_spec=e.region(pool,own['knowledge'],[c['id'] for c in own['knowledge']['clauses']])
        if full_spec:
            key=e.trust.geometry_key(full_spec);bank=own['banks'].get(key,e.new_bank(t));forecast=e.forecast(bank,t)
            alpha=forecast['amplitude'];book=before_own['region_trust'].get(key)
            if book and book['blocked']:alpha=0.
            field_choice('no_B4_full_region_calibrated',own['knowledge'],[c['id'] for c in own['knowledge']['clauses']],alpha)
        else:record('no_B4_full_region_calibrated',after['reference'],base)
        # Removing only normal opportunity gating tests whether shrinking E is
        # actually binding, independently of historical GP path changes.
        normal_ungated=choose(after['score'],available)
        record('normal_without_budget_mask',normal_ungated,after['score'],after['shift_std'],after['amplitude'])
        selected=cache.ix[commit['selected_id']]
        if commit['validation']:
            prepared=e.validation_candidates(cache,own,after);assert prepared==commit['validation']['prepared']
            options={r['candidate_id']:r for r in prepared['candidates']};assert commit['selected_id'] in options
        else:assert selected==after['selected']
        record('executed',selected,after['score'],after['shift_std'],after['amplitude'])
        assert cache.counters['fits']==0
    # First access to hidden outcomes in this worker is AFTER all alternatives
    # are fixed. No counterfactual outcome enters GP, scoring or LLM validation.
    with np.load(data.ROOT/task/'oracle.npz') as f:truth=f['raw_values'].copy()
    target=int(truth.argmax());target_id=pool['ids'][target]
    for row in stage:
        j=cache.ix[row['candidate_id']];row.update(objective_after_choice=float(truth[j]),incumbent_after_one_step=max(inc,float(truth[j])))
    stages={r['stage']:r for r in stage}
    def delta(a,b,mode='objective_after_choice'):return stages[a][mode]-stages[b][mode]
    ev=state['decisions'][t-1]
    wide={'task':task,'query_round':t,'incumbent_before':inc,'executed_id':commit['selected_id'],'executed_objective':float(truth[selected]),
        'base_BO_id':pool['ids'][after['reference']],'base_BO_objective':float(truth[after['reference']]),
        'knowledge_vs_GP_delta':delta('current_full_region_a1','base_BO'),
        'B4_geometry_delta':delta('B4_geometry_a1','current_full_region_a1'),
        'KF_scaling_delta':delta('KF_before_conflict_gate','B4_geometry_a1'),
        'B4_conflict_fallback_delta':delta('B4_conflict_gate','KF_before_conflict_gate'),
        'persistent_trust_delta':delta('normal_after_trust','B4_conflict_gate'),
        'refresh_total_delta':delta('normal_after_trust','before_refresh_normal') if call else 0.,
        'controlled_override_delta':delta('executed','normal_after_trust'),
        'executed_vs_GP_delta':delta('executed','base_BO'),
        'executed_vs_GP_incumbent_delta':delta('executed','base_BO','incumbent_after_one_step'),
        'refresh_applied':bool(call),'controlled_test':bool(commit['validation']),
        'knowledge_ids':','.join(c['id'] for c in own['knowledge']['clauses']),'retained':','.join(after['retained']),
        'B4_conflict_kind':after['diagnosis'].conflict_kind,'revoked':','.join(after['diagnosis'].revoked_ids),
        'bank_key':after['bank_key'],'bank_updates_before':after['bank']['updates'] if after['bank'] else 0,
        'KF_alpha_before_gate':pre_alpha,'deployed_alpha':after['amplitude'],'trust_blocked':after['trust']['trust_blocked'],
        'trust_recovered':after['trust']['recovered'],'KF_b_after_raw':ev['bank_after']['mean'][0]*scale if ev['bank_after'] else None,
        'KF_a_after':ev['bank_after']['mean'][1] if ev['bank_after'] else None,
        'raw_region_shift_at_actual':ev['measurement']['raw_region_shift_y']*scale,
        'GP_residual_at_actual':ev['measurement']['residual']*scale,
        'innovation_z':ev['measurement']['innovation_z'],'normal_budget_size':int(eligible.sum()),
        'pool_best_id':target_id,'pool_best_objective':float(truth[target]),'pool_best_available':bool(available[target]),
        'pool_best_base_rank':int(1+np.sum(base[available]>base[target])), 'pool_best_gate_eligible':bool(eligible[target]),
        'pool_best_GP_mean':float(v['mu_y'][target])*scale,'pool_best_GP_SD':float(v['sd_y'][target])*scale,
        'KF_actual_changed_choice':stages['KF_before_conflict_gate']['candidate_id']!=stages['B4_geometry_a1']['candidate_id'],
        'budget_mask_changed_choice':stages['normal_without_budget_mask']['candidate_id']!=stages['normal_after_trust']['candidate_id']}
    refresh=None
    if call:
        oldkey=before['bank_key'];newkey=after['bank_key']
        refresh={'task':task,'query_round':t,'reasons':call['reasons'],'old_knowledge':call['knowledge_before'],'new_knowledge':own['knowledge'],
            'old_geometry_key':oldkey,'new_geometry_key':newkey,'new_region_bank_is_fresh':bool(after['bank'] and not after['bank']['updates']),
            'old_amplitude':before['amplitude'],'old_KF_amplitude':before['prior']['amplitude'] if before['prior'] else None,
            'new_amplitude':after['amplitude'],'pre_choice':stages['before_refresh_normal'],'post_choice':stages['normal_after_trust'],
            'same_new_geometry_old_amplitude':stages['new_geometry_old_amplitude'],
            'summary':call['answer']['plan']['summary'],'actions':call['answer']['plan']['actions'],'context_sha256':call['context_sha256']}
    return {'checks':check,'stages':stage,'wide':wide,'refresh':refresh}

def static_analysis():
    knowledge_rows=[];baseline_rows=[];gap_rows=[];curves={};corners=[]
    for task in s.TASKS:
        pool=s.pool(task);X=np.array(pool['z']);ix={k:i for i,k in enumerate(pool['ids'])}
        with np.load(data.ROOT/task/'oracle.npz') as f:truth=f['raw_values'].copy()
        jbest=int(truth.argmax());new=data.read(s.ROOT/'runs'/(task+'_New_Agent')/'state.json')
        variants=[(0,new['initial_knowledge'])]+[(c['iteration'],c['answer']['plan']['knowledge']) for c in new['calls']]
        for t,knowledge in variants:
            mask=e.membership(pool,knowledge).all(0);queried=[h['candidate_id'] for h in new['history'][:4 if t==0 else t+3]]
            avail=~np.isin(pool['ids'],queried);max_y=float(truth[mask].max()) if mask.any() else None
            for c in knowledge['clauses']:
                axis=pool['axes'].index(c['axis']);lo,hi=c['bounds'];l,h=pool['bounds'][c['axis']]
                knowledge_rows.append({'task':task,'effective_from_query':max(1,t),'refresh_query':t,'id':c['id'],'axis':c['axis'],
                    'lower_normalized':lo,'upper_normalized':hi,'lower_physical':l+lo*(h-l),'upper_physical':l+hi*(h-l),
                    'pool_best_satisfies_this_clause':bool(lo<=X[jbest,axis]<=hi),
                    'joint_region_pool_count':int(mask.sum()),'joint_region_unqueried_count':int((mask&avail).sum()),
                    'joint_region_best_objective_for_analysis_only':max_y,'pool_best_id':pool['ids'][jbest]})
        for method in ('New_Agent','Pure_BO','LLINBO_Justify'):
            state=data.read(s.ROOT/'runs'/(task+'_'+method)/'state.json');y=np.array([h['raw_value'] for h in state['history']]);b=np.maximum.accumulate(y)
            curves[(task,method)]={'objective':y.tolist(),'incumbent':b.tolist()}
            for event in state['decisions']:
                t=event['iteration'];j=ix[event['selected_id']];ref=ix[event['reference_id']];proposal=(event.get('LLM') or {}).get('proposed_id');pr=ix[proposal] if proposal else None
                baseline_rows.append({'task':task,'method':method,'query_round':t,'selected_id':pool['ids'][j],'objective':float(truth[j]),
                    'incumbent':float(b[t+3]),'BO_reference_id':pool['ids'][ref],'BO_reference_objective_for_analysis_only':float(truth[ref]),
                    'LLM_proposed_id':proposal,'LLM_proposal_objective_for_analysis_only':float(truth[pr]) if pr is not None else None,
                    'proposal_accepted':(event.get('LLM') or {}).get('accepted'),
                    'accepted_and_changed_choice':bool((event.get('LLM') or {}).get('accepted') and j!=ref),
                    'actual_minus_same_prefix_BO_objective':float(truth[j]-truth[ref])})
        for base in ('Pure_BO','LLINBO_Justify'):
            a=curves[(task,'New_Agent')]['incumbent'];b=curves[(task,base)]['incumbent']
            for t in range(1,17):gap_rows.append({'task':task,'baseline':base,'query_round':t,'new_incumbent':a[t+3],'baseline_incumbent':b[t+3],
                'incumbent_gap':a[t+3]-b[t+3],'contribution_to_SearchAUC_gap':(a[t+3]-b[t+3])/16})
        if task=='SnAr':
            for eq in (1.,5.):
                for temp in (30.,120.):
                    point={'tau':.5,'equiv_pldn':eq,'conc_dfnb':.5,'temperature':temp}
                    j=next(i for i,p in enumerate(pool['points']) if p==point)
                    corners.append({**point,'candidate_id':pool['ids'][j],'STY':float(truth[j])})
    return knowledge_rows,baseline_rows,gap_rows,curves,corners

def extra_analysis():
    """Independent KF algebra and target-rank tracing; no GP evaluation."""
    kf_rows=[];target_rows=[];initial_facts=0;max_error=0.
    for task in s.TASKS:
        pool=s.pool(task);scale=pool['objective_scale'];ix={k:i for i,k in enumerate(pool['ids'])}
        with np.load(data.ROOT/task/'oracle.npz') as f:truth=f['raw_values'].copy()
        target=int(truth.argmax());target_id=pool['ids'][target]
        ready=data.read(s.ROOT/'initialization'/task/'ready.json')
        context=data.read(s.ROOT/'requests'/ready['context_sha256']/'context.json');answer=data.read(s.ROOT/'requests'/ready['context_sha256']/'validated.json')
        own={'history':ready['history'],'knowledge':{'confidence':.7,'clauses':[]},'decisions':[]}
        checked=llm.validate_region(answer['raw'],pool,own);assert checked==answer['plan'] and checked['knowledge']==ready['knowledge']
        assert context['observed_prefix']==ready['history'];initial_facts+=len(checked['checked_facts'])
        for method in ('New_Agent','Pure_BO','LLINBO_Justify'):
            state=data.read(s.ROOT/'runs'/(task+'_'+method)/'state.json')
            for ev in state['decisions']:
                t=ev['iteration'];folder=s.ROOT/'runs'/state['run_id']/'rounds';c=data.read(folder/f'{t:02d}_committed.json')
                with np.load(folder/f'{t:02d}_candidates.npz') as f:arr={k:f[k].copy() for k in f.files}
                rank=int(1+np.sum(arr['base_qLogEI'][arr['available']]>arr['base_qLogEI'][target]))
                target_rows.append({'task':task,'method':method,'query_round':t,'target_id':target_id,'target_objective':float(truth[target]),
                    'target_available':bool(arr['available'][target]),'target_base_rank':rank,'target_gate_eligible':bool(arr['eligible'][target]),
                    'target_GP_mean':float(arr['mu_y'][target])*scale,'target_GP_SD':float(arr['sd_y'][target])*scale,
                    'target_base_qLogEI':float(arr['base_qLogEI'][target]),'target_base_opportunity_loss':float(np.max(arr['base_qLogEI'][arr['available']])-arr['base_qLogEI'][target]),
                    'gate_epsilon':c['budget'],'selected_id':ev['selected_id']})
                if method!='New_Agent' or c['bank_prior'] is None:continue
                prior=c['bank_prior'];j=ix[c['selected_id']];mu=float(arr['mu_y'][j]);sd=float(arr['sd_predictive_y'][j]);raw=float(arr['raw_shift_std'][j])*c['GP_scale_y']
                theta=np.array(prior['mean']);P=np.array(prior['covariance']);H=np.array([1.,raw]);R=sd**2+.0009
                y=ev['y'];r=y-mu;innovation=r-H@theta;gain=P@H/(H@P@H+R)
                mean=theta+gain*innovation;cov=P-np.outer(P@H,H@P)/(H@P@H+R)
                error=max(float(np.max(abs(mean-np.array(ev['bank_after']['mean'])))),float(np.max(abs(cov-np.array(ev['bank_after']['covariance'])))))
                assert error<1e-12;max_error=max(max_error,error)
                a=float(prior['amplitude']);deploy=c['amplitude'];b=float(theta[0]);a_unclipped=float(theta[1]);
                row={'task':task,'query_round':t,'bank_updates_before':c['bank_before']['updates'],'b_before_raw':b*scale,
                    'a_before_unclipped':a_unclipped,'a_before_clipped':a,'a_deployed':deploy,'a_after_unclipped':float(mean[1]),
                    'b_after_raw':float(mean[0])*scale,'GP_mean':mu*scale,'objective':y*scale,'raw_region_field':raw*scale,
                    'deployed_correction':deploy*raw*scale,'shadow_a_only_correction':a*raw*scale,
                    'KF_measurement_model_residual':float(H@theta)*scale,'innovation_z':ev['measurement']['innovation_z'],
                    'SE_GP':((y-mu)*scale)**2,'SE_deployed':((y-mu-deploy*raw)*scale)**2,
                    'SE_shadow_a_only':((y-mu-a*raw)*scale)**2,'SE_joint_b_a':((y-mu-b-a_unclipped*raw)*scale)**2,
                    'measurement_noise_floor_raw_SD':.03*scale,'KF_algebra_error':error}
                row['deployed_prediction_improves_GP']=row['SE_deployed']<row['SE_GP']-1e-9
                row['shadow_prediction_improves_GP']=row['SE_shadow_a_only']<row['SE_GP']-1e-9
                kf_rows.append(row)
    csv_write(OUT/'KF_prediction_diagnostics.csv',kf_rows);csv_write(OUT/'pool_best_rank_traces.csv',target_rows)
    receipt=data.read(OUT/'execution_checks.json');receipt.update(KF_updates_independently_checked=len(kf_rows),max_KF_algebra_error=max_error,
        initial_knowledge_typed_facts_checked=initial_facts)
    data.write(OUT/'execution_checks.json',receipt)
    return {'KF_updates_checked':len(kf_rows),'max_KF_error':max_error,'initial_facts_checked':initial_facts}

def b4_tie_diagnostics():
    result=[]
    for task,t in [('HPLC',1),('SnAr',4)]:
        pool=s.pool(task);folder=s.ROOT/'runs'/(task+'_New_Agent')/'rounds';c=data.read(folder/f'{t:02d}_committed.json')
        with np.load(folder/f'{t:02d}_candidates.npz') as f:base=f['base_qLogEI'].copy();available=f['available'].copy()
        knowledge=c['knowledge'];ids=[h['id'] for h in knowledge['clauses']];mem=e.membership(pool,knowledge)
        original=e.select_candidate(base,mem,c['budget'],candidate_ids=pool['ids'],hypothesis_ids=ids,queried_mask=~available)
        feasible=[v for v in original.subset_checks if v['eligible_count']>0]
        size=max(len(v['retained_ids']) for v in feasible)
        rename={'H1':'H4','H4':'H1','H2':'H2','H3':'H3'};renamed=[rename[x] for x in ids]
        alternate=e.select_candidate(base,mem,c['budget'],candidate_ids=pool['ids'],hypothesis_ids=renamed,queried_mask=~available)
        inverse={v:k for k,v in rename.items()};alternate_original_ids=sorted(inverse[x] for x in alternate.retained_ids)
        result.append({'task':task,'query_round':t,'equally_maximal_subsets':[v for v in feasible if len(v['retained_ids'])==size],
            'retained_original_ids':list(original.retained_ids),'retained_under_label_swap_mapped_back':alternate_original_ids,
            'hypothesis_id_swap':rename,'selection_priority':'maximum retained cardinality, then sorted retained IDs, then acquisition score',
            'interpretation':'Hypothesis-slot ordering changes the admitted geometry. This is specified B4 behavior, not candidate-ID tie sorting. Conflict fallback still sets current deployment to zero; no entire-trajectory effect is claimed.'})
    data.write(OUT/'B4_subset_ties.json',result);return result

def main():
    OUT.mkdir(parents=True,exist_ok=True);protocol=s.check()
    invpath=s.OUT/'release_inventory.json';receiptpath=s.OUT/'release_receipt.json'
    original_inventory_hash=data.filehash(invpath);assert data.read(receiptpath)['inventory_sha256']==original_inventory_hash
    original=data.read(invpath)['files']
    for row in original:assert data.filehash(row['path'])==row['sha256'],row['path']
    jobs=[(task,t) for task in s.TASKS for t in range(1,17)]
    with ProcessPoolExecutor(max_workers=8,initializer=e.warm) as executor:results=list(executor.map(audit_round,jobs))
    for row in original:assert data.filehash(row['path'])==row['sha256'],row['path']
    assert data.filehash(invpath)==original_inventory_hash
    csv_write(OUT/'stage_choices.csv',[r for item in results for r in item['stages']])
    csv_write(OUT/'module_round_effects.csv',[item['wide'] for item in results])
    data.write(OUT/'refresh_audit.json',[item['refresh'] for item in results if item['refresh']])
    knowledge,baselines,gaps,curves,corners=static_analysis()
    csv_write(OUT/'knowledge_quality.csv',knowledge);csv_write(OUT/'baseline_rounds.csv',baselines)
    csv_write(OUT/'incumbent_gaps.csv',gaps);csv_write(OUT/'SnAr_interaction_corners.csv',corners)
    data.write(OUT/'curves.json',{task:{m:curves[(task,m)] for m in ('New_Agent','Pure_BO','LLINBO_Justify')} for task in s.TASKS})
    checks={'status':'PASS','new_agent_rounds_reconstructed':32,'pre_refresh_actions_and_gate_states_checked':6,
        'new_LLM_calls':0,'new_GP_fits':0,'frozen_files_unchanged':len(original),'original_inventory_sha256':original_inventory_hash,
        'max_actual_replay_error':max(item['checks']['max_actual_array_replay_error'] for item in results),
        'checks':[item['checks'] for item in results],
        'interpretation':'Same-prefix module toggles and hidden-outcome lookup AFTER selection. One-step objective/incumbent differences are not additive full-trajectory causal AUC effects.'}
    data.write(OUT/'execution_checks.json',checks)
    extra_analysis()
    b4_tie_diagnostics()
    print(json.dumps({k:v for k,v in checks.items() if k!='checks'},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
