"""Paired outcome reporting and independent verification; no proposal logic."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import statistics as st

import numpy as np

from . import llinbo_guarded_study as s
from . import llinbo_guarded_engine as engine
from . import guarded_suzuki_engine as shared
from . import guarded_suzuki_study as parent
from .official_suzuki_adapter import check_sources
from .suzuki_domain import digest,CATALYSTS

CONTROLS=('B0','LGBO_region','LGBO_gate','B4_base','B4_LGBO')


def states():return [s.read(f) for f in sorted((s.ROOT/'runs').glob('*/state.json'))]


def metric(row):
    y=[v['y'] for v in row['history']]
    curve=[max(y[:min(i,len(y))])*100 for i in range(5,21)]
    return {'auc':st.mean(curve),'final':max(y)*100,'initial':max(y[:4])*100,'curve':curve}


def baseline(case,seed,regime,method):
    stage='development' if case==1 else 'test'
    key=f'c{case}_s{seed}_none_B0' if method=='B0' else f'c{case}_s{seed}_{regime}_{method}'+('' if method=='LGBO_region' else '_r4')
    return s.read(parent.ROOT/stage/'runs'/key/'state.json')


def table(rows):
    m=[metric(r) for r in rows]
    return {'n':len(rows),'completed':sum(r['status']=='COMPLETE' for r in rows),
        'AUC_mean':st.mean(v['auc'] for v in m),'AUC_sd':st.stdev(v['auc'] for v in m),
        'final_mean':st.mean(v['final'] for v in m),'initial_mean':st.mean(v['initial'] for v in m),
        'by_case':{str(c):{'n':sum(r['case']==c for r in rows),
            'AUC_mean':st.mean(metric(r)['auc'] for r in rows if r['case']==c),
            'final_mean':st.mean(metric(r)['final'] for r in rows if r['case']==c)} for c in sorted({r['case'] for r in rows})}}


def report():
    p=s.check_protocol();all_rows=states();assert len(all_rows)==80
    assert all(r['status'] in ('COMPLETE','FAILED') for r in all_rows),'study still running'
    test=[r for r in all_rows if r['case'] in (2,3,4)]
    tables={};paired={};mechanism={};raw=[]
    for regime in s.REGIMES:
        rows=[r for r in test if r['regime']==regime]
        groups={'LLINBO_matched':rows}
        groups.update({m:[baseline(r['case'],r['seed'],regime,m) for r in rows] for m in CONTROLS})
        tables[regime]={m:table(v) for m,v in groups.items()}
        paired[regime]={}
        for method,control in groups.items():
            if method=='LLINBO_matched':continue
            deltas=[metric(a)['auc']-metric(b)['auc'] for a,b in zip(rows,control)]
            paired[regime][method]={'mean_AUC_delta_pp':st.mean(deltas),
                'by_case':{str(c):st.mean(d for d,r in zip(deltas,rows) if r['case']==c) for c in (2,3,4)},
                'wins':sum(d>1e-10 for d in deltas),'losses':sum(d<-1e-10 for d in deltas),
                'ties':sum(abs(d)<=1e-10 for d in deltas),'pairs':[
                    {'case':r['case'],'seed':r['seed'],'AUC_delta_pp':d} for r,d in zip(rows,deltas)]}
        events=[v for r in rows for v in r['decisions']]
        mechanism[regime]={'decisions':len(events),'accepted':sum(v['accepted'] for v in events),
            'rejected':sum(not v['accepted'] for v in events),
            'accepted_and_changed_from_same_history_base':sum(v['accepted'] and not v['proposal_equals_reference'] for v in events),
            'mean_projection_distance':st.mean(v['projection']['normalized_numeric_distance'] for v in events),
            'max_projection_distance':max(v['projection']['normalized_numeric_distance'] for v in events),
            'max_selected_base_logei_loss':max(v['selected_logei_loss'] for v in events),
            'mean_clipped_AUC_downside_to_B0_pp':st.mean(max(metric(b)['auc']-metric(a)['auc'],0.) for a,b in zip(rows,groups['B0']))}
        # Evaluator-only diagnostics, performed after all1280 answers freeze.
        # Counterfactual one-step yields do not establish long-run policy gains.
        rejected_improves=0;rejected_better_than_executed=0;raw_repeated=0;inside=0;rejected_y_deltas=[]
        equal_b0=0
        for row,control in zip(rows,groups['B0']):
            equal_b0+=int(row['history']==control['history'])
            folder=parent.ROOT/'inputs'/f"case{row['case']}"
            pool=s.read(folder/'pool.json');lookup={v:i for i,v in enumerate(pool['ids'])}
            values=np.load(folder/'oracle.npz')['values']
            advice=s.read(folder/'advice.json')[str(row['seed'])][regime]
            membership=shared.clause_membership(pool['z'],advice)
            for event in row['decisions']:
                t=event['iteration'];prefix=row['history'][:t+3];i=lookup[event['proposed_id']]
                inside+=int(membership[:,i].all())
                proposal_raw=event['projection']['raw_point']
                raw_repeated+=int(any(proposal_raw==pool['points'][lookup[h['candidate_id']]] for h in prefix))
                if not event['accepted']:
                    proposed_y=float(values[i]);actual_y=row['history'][t+3]['y']
                    rejected_improves+=int(proposed_y>max(h['y'] for h in prefix)+1e-10)
                    rejected_better_than_executed+=int(proposed_y>actual_y+1e-10)
                    rejected_y_deltas.append((proposed_y-actual_y)*100)
        mechanism[regime].update({'proposal_inside_full_initial_preference':inside,
            'raw_proposal_exactly_repeats_observed_point':raw_repeated,
            'entire_trajectory_identical_to_B0':equal_b0,
            'posthoc_evaluator_only':{'rejected_proposal_would_improve_current_incumbent':rejected_improves,
                'rejected_proposal_y_better_than_executed':rejected_better_than_executed,
                'mean_rejected_proposal_minus_executed_y_pp':st.mean(rejected_y_deltas) if rejected_y_deltas else None,
                'boundary':'Frozen-oracle lookup after all answers; one-step counterfactual diagnostics, not new policy observations or long-run causal effects.'}})
    for row in all_rows:raw.append({k:row[k] for k in ('run_id','case','seed','regime','status')}|metric(row))
    events=[v for r in all_rows for v in r['decisions']]
    result={'study':p['study'],'protocol_sha256':p['protocol_sha256'],'test_scope':'Case2-4,5seeds,15paired runs per knowledge regime',
        'completed_runs':sum(r['status']=='COMPLETE' for r in all_rows),'planned_runs':80,
        'new_logical_LLM_responses':len(events),'logical_queries':sum(len(r['history']) for r in all_rows),
        'all_case_acceptances':sum(v['accepted'] for v in events),'all_case_rejections':sum(not v['accepted'] for v in events),
        'tables':tables,'paired_LLINBO_minus_control':paired,'mechanism':mechanism,'raw_run_metrics':raw,
        'case1_reference':{g:table([r for r in all_rows if r['case']==1 and r['regime']==g]) for g in s.REGIMES},
        'limits':['Matched adapter, not original full LLINBO UCB algorithm.',
            'Current shared assistant context is not blind, immutable or independent across tasks.',
            'Reference-aware synthetic knowledge construction; actual initial advice held fixed for regional controls.',
            'Adaptive point advice versus fixed initial region: system-level comparison with different update cadence.',
            'Three test chemistries, not60 independent chemistries; descriptive paired differences, no significance claim.']}
    s.write(s.OUT/'results.json',result)
    case_comparisons(all_rows)
    return {'completed':result['completed_runs'],'logical_LLM_responses':len(events),'AUC':{g:{m:round(v['AUC_mean'],4) for m,v in t.items()} for g,t in tables.items()},'acceptances':result['all_case_acceptances']}


def case_comparisons(all_rows=None):
    rows=states() if all_rows is None else all_rows
    result={'scope':'Case1 reference only; Case2-4 primary comparison; four-case aggregate exploratory.',
        'by_case':{},'all_four_case_means':{}}
    for case in (1,2,3,4):
        result['by_case'][str(case)]={}
        for regime in s.REGIMES:
            selected=[r for r in rows if r['case']==case and r['regime']==regime]
            methods={'LLINBO_matched':selected}|{m:[baseline(case,r['seed'],regime,m) for r in selected] for m in CONTROLS}
            result['by_case'][str(case)][regime]={m:table(group) for m,group in methods.items()}
    for regime in s.REGIMES:
        result['all_four_case_means'][regime]={m:st.mean(result['by_case'][str(c)][regime][m]['AUC_mean'] for c in (1,2,3,4)) for m in ('LLINBO_matched',*CONTROLS)}
    s.write(s.OUT/'case_comparisons.json',result)
    return result


def replay_group(job):
    """Direct uninterrupted native entrypoint; no new answers or observations."""
    rows=job;p=s.read(s.OUT/'protocol.json');case,seed=rows[0]['case'],rows[0]['seed']
    folder=parent.ROOT/'inputs'/f'case{case}';pool=s.read(folder/'pool.json');priors=s.read(folder/'advice.json')[str(seed)]
    cache=shared.ScoreCache(pool['z'],pool['ids'],s.ROOT/'cache'/f'c{case}_s{seed}',case,seed)
    count=0
    for row in rows:
        def assistant(history,prompt,t):
            assert history==row['history'][:t+3]
            event=row['decisions'][t-1];q=s.read(s.ROOT/'requests'/event['response_context_sha256']/'request.json')
            assert q['source_prompt']==prompt and q['history']==history
            response=s.read(s.ROOT/'requests'/event['response_context_sha256']/'response.json')
            return response['projection'],response
        def objective(candidate,event):
            nonlocal count
            t=event['iteration'];assert candidate==row['history'][t+3]['candidate_id']
            assert event==row['decisions'][t-1];count+=1
            return row['history'][t+3]['y']
        result=engine.execute(pool=pool,initial=row['history'][:4],advice=priors[row['regime']],case=case,seed=seed,
            ratio=p['review_budget_ratio'],cache=cache,assistant=assistant,objective=objective)
        assert len(result['histories'][0])==20
    assert cache.counters['fits']==0 and cache.counters['acquisition_evaluations']==0
    return {'decisions':count,'fits':cache.counters['fits'],'acquisition_evaluations':cache.counters['acquisition_evaluations']}


def verify():
    p=s.check_protocol();parent.check_protocol();check_sources();all_rows=states()
    assert len(all_rows)==80 and all(r['status']=='COMPLETE' for r in all_rows)
    controls_verified=0;responses=0;positions=0;budget_violations=0;floating_gate_disagreements=0;initial_score_errors=[]
    batches={s.filehash(f):s.read(f) for f in (s.ROOT/'assistant').glob('batch*.json')}
    for row in all_rows:
        case,seed,regime=row['case'],row['seed'],row['regime'];folder=parent.ROOT/'inputs'/f'case{case}'
        pool=s.read(folder/'pool.json');lookup={key:i for i,key in enumerate(pool['ids'])}
        values=np.load(folder/'oracle.npz')['values'];h=row['history']
        assert len(h)==20 and len(row['decisions'])==16 and len({v['candidate_id'] for v in h})==20
        assert [v['candidate_id'] for v in h[:4]]==pool['initial_ids'][str(seed)]
        assert all(v['y']==float(values[lookup[v['candidate_id']]]) for v in h)
        for control in CONTROLS:
            old=baseline(case,seed,regime,control);assert old['history'][:4]==h[:4];controls_verified+=1
        for t,event in enumerate(row['decisions'],1):
            prefix=h[:t+3];queried={v['candidate_id'] for v in prefix}
            file=s.ROOT/'cache'/f'c{case}_s{seed}'/(event['base_acquisition']['key']+'.npz')
            with np.load(file,allow_pickle=False) as data:scores=data['scores']
            if t==1:
                reference_state=baseline(case,seed,regime,'B0')
                old_key=reference_state['decisions'][0]['base_acquisition']['key']
                assert old_key==event['base_acquisition']['key']
                with np.load(parent.ROOT/'cache'/f'c{case}_s{seed}'/(old_key+'.npz'),allow_pickle=False) as previous:
                    error=float(np.max(np.abs(scores-previous['scores'])))
                initial_score_errors.append(error);assert error==0.
            available=[i for i,key in enumerate(pool['ids']) if key not in queried]
            reference=min(available,key=lambda i:(-float(scores[i]),pool['ids'][i]))
            proposed=lookup[event['proposed_id']];selected=lookup[event['selected_id']]
            maximum=float(scores[reference]);budget=float(np.log(4)*(16-t)/15)
            rejected=maximum>float(scores[proposed])+budget
            assert event['accepted']==(not rejected)
            assert selected==(reference if rejected else proposed)
            assert event['reference_id']==pool['ids'][reference]
            assert event['maximum_base_logei']==maximum and event['proposal_base_logei']==float(scores[proposed])
            assert event['budget_logei']==budget and event['selected_id']==h[t+3]['candidate_id']
            budget_violations+=int(maximum-float(scores[selected])>budget)
            floating_gate_disagreements+=int((maximum-float(scores[proposed])>budget)!=rejected)
            q=s.read(s.ROOT/'requests'/event['response_context_sha256']/'request.json');raw=deepcopy(q);key=raw.pop('context_sha256')
            assert digest(raw)==key and key==event['response_context_sha256']
            assert q['history']==prefix and q['public_task']==row['public_task'] and q['iteration']==t
            assert q['advice']==s.read(folder/'advice.json')[str(seed)][regime]
            response=s.read(s.ROOT/'requests'/key/'response.json')
            projection=engine.project_proposal(response['raw_point'],pool,queried)
            assert projection==response['projection']==event['projection']
            assert projection['candidate_id']==event['proposed_id'] and event['proposed_id'] not in queried
            assert all(1<=i<=len(prefix) for i in response['supporting_queries'])
            assert response['origin']=='current_assistant'
            authored=batches[response['source_batch_sha256']]['answers'][row['public_task']]
            assert authored[:4]==[CATALYSTS.index(response['raw_point']['catalyst']),
                response['raw_point']['t_res'],response['raw_point']['temperature'],response['raw_point']['catalyst_loading']]
            assert authored[4]==response['reasoning'] and authored[5]==response['supporting_queries']
            responses+=1
        positions+=20
        assert row['operator_compiles_in_loop']==0
    assert budget_violations==0 and floating_gate_disagreements==0
    # Verify all frozen parent artifacts. Its cache is read-only in this study.
    inventory=s.read(parent.OUT/'release_inventory.json')
    for record in inventory['files']:assert s.filehash(record['path'])==record['sha256'],record['path']
    old=s.read('research/suzuki_official/release_inventory.json')
    for record in old['files']:assert s.filehash(record['path'])==record['sha256'],record['path']
    groups={}
    for row in all_rows:groups.setdefault((row['case'],row['seed']),[]).append(row)
    with ProcessPoolExecutor(max_workers=s.WORKERS,initializer=s.worker_init) as pool:
        replay=list(pool.map(replay_group,groups.values()))
    result=s.read(s.OUT/'results.json')
    for regime in s.REGIMES:
        raw_mean=st.mean(sum(max(v['y'] for v in row['history'][:i])*100 for i in range(5,21))/16 for row in all_rows if row['case'] in (2,3,4) and row['regime']==regime)
        assert abs(raw_mean-result['tables'][regime]['LLINBO_matched']['AUC_mean'])<1e-10
    receipt={'integrity_status':'PASS','protocol_sha256':p['protocol_sha256'],'complete_runs':80,
        'formal_comparison_runs':60,'logical_queries':positions,'fresh_assistant_answers':responses,
        'independent_gate_decisions':responses,'source_entrypoint_uninterrupted_replay_decisions':sum(v['decisions'] for v in replay),
        'new_replay_fits':sum(v['fits'] for v in replay),'new_replay_acquisition_evaluations':sum(v['acquisition_evaluations'] for v in replay),
        'reused_control_initializations_checked':controls_verified,'budget_violations':budget_violations,
        'initial_common_score_tables_checked':len(initial_score_errors),'initial_common_score_max_abs_error':max(initial_score_errors),
        'addition_vs_subtraction_gate_disagreements':floating_gate_disagreements,'operator_compiles_in_loop':0,
        'protected_parent_artifacts_unchanged':len(inventory['files']),'protected_primary_files_unchanged':len(old['files']),
        'results_sha256':s.filehash(s.OUT/'results.json'),
        'scientific_boundary':'A completed descriptive system comparison; integrity PASS does not imply LLINBO/B4 improvement or blind evaluation.'}
    s.write(s.OUT/'validation_receipt.json',receipt)
    return receipt
