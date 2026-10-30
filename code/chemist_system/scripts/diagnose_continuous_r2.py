"""Read-only R2 dynamics diagnosis on frozen GP prefixes.

No new trajectories, LLM calls, GP fitting or outcome-provider calls. Later
shadow checks and gate bypasses below are frozen-history diagnostics only;
they are never fed back into the archived optimization experiment.
"""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch
import csv
import json
import os
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));os.chdir(ROOT)
from chem.benchmarks import continuous_benchmark_engine as base
from chem.benchmarks import continuous_r2_engine as r2
from chem.benchmarks.suzuki_domain import digest

RUN=ROOT/'data/runs/continuous_r2_20261030'
STUDY=ROOT/'research/continuous_r2_20261030'
OUT=ROOT/'research/continuous_r2_diagnosis_20261030'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def filehash(path):return sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def csv_write(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def refuse(*args,**kwargs):raise RuntimeError('Diagnosis refuses fitting or experimental outcome calls.')


class FrozenCache(base.Cache):
    """Existing checkpoint plus archived marginals; no writes or fit fallback."""
    def __init__(self,pool,c,arrays):
        self.pool=pool;self.ids=pool['ids'];self.ix={key:i for i,key in enumerate(self.ids)}
        self.X=torch.tensor(pool['z'],dtype=torch.double)
        self.c=c;self.arrays=arrays;self.fields={};self.value=None
    def model(self,history):
        assert digest(history)==self.c['history_sha256']
        if self.value is not None:return self.value
        ordered=sorted(history,key=lambda h:tuple(self.pool['z'][self.ix[h['candidate_id']]]))
        x=self.X[[self.ix[h['candidate_id']] for h in ordered]]
        y=torch.tensor([[h['y']] for h in ordered],dtype=torch.double)
        mean=self.c['GP_mean_y'];std=self.c['GP_scale_y'];meta=self.c['base_meta']
        base.seed_all(meta['fit_seed']);model=base.SingleTaskGP(x,(y-mean)/std)
        path=RUN/'cache'/self.pool['task']/(meta['fit_key']+'.pt')
        assert path.is_file()
        saved=torch.load(path,map_location='cpu',weights_only=False);assert saved['key']==meta['fit_key']
        model.load_state_dict(saved['state']);model.eval()
        self.value={'model':model,'mean':mean,'std':std,'best_std':self.c['best_std'],
            'fit_key':meta['fit_key'],'fit_seed':meta['fit_seed'],
            **{key:self.arrays[key] for key in ['mu_std','sd_std','mu_y','sd_y','sd_predictive_y']}}
        return self.value
    def score(self,history,shift=None):
        v=self.model(history)
        score=self.arrays['base_qLogEI'] if shift is None or not np.any(shift) else base.acquisition(
            v['mu_std'],v['sd_std'],shift,v['best_std'],self.c['base_meta']['mc_seed'])
        return score,self.c['base_meta']


def pick(arrays,score):
    js=np.flatnonzero(arrays['eligible']);assert len(js)
    js=js[score[js]==score[js].max()]
    js=js[arrays['information_proxy'][js]==arrays['information_proxy'][js].max()]
    js=js[arrays['sd_y'][js]==arrays['sd_y'][js].max()]
    return int(js[np.argmin(arrays['tie_priority'][js])])


def task_analysis(task):
    base.fit_gpytorch_mll=refuse;base.data.reveal=refuse;base.warm()
    pool=read(ROOT/f'data/benchmarks/continuous_20261029/{task}/pool.json')
    main=read(RUN/'runs'/(task+'_New_Agent_R2')/'state.json')
    state=read(RUN/'runs'/(task+'_New_Agent_R2_Subset')/'state.json')
    assert main['history']==state['history']
    folder=RUN/'runs'/state['run_id']/'rounds';scale=pool['objective_scale'];ids=pool['ids']
    ix={key:j for j,key in enumerate(ids)};z=np.asarray(pool['z'])
    calls=[c for c in state['calls'] if 'inheritance' in c]
    proposals=[]
    for i,call in enumerate(calls):
        t=call['iteration'];c=read(folder/f'{t:02d}_committed.json')
        knowledge=call['answer']['plan']['knowledge'];pending=state['decisions'][t-1]['pending_after']
        proposals.append({'query':t,'knowledge':knowledge,'region':pending['region'],
            'alpha':c['pending_forecast']['shadow_probe_coefficient'],
            'lifetime_last':calls[i+1]['iteration']-1 if i+1<len(calls) else 16,
            'inheritance':call['inheritance']})
    initial=[];followup=[];gates=[];rounds=[];max_error=0.;probe_counts=[]
    for t in range(r2.BOOTSTRAP[task]+1,17):
        c=read(folder/f'{t:02d}_committed.json');event=state['decisions'][t-1];before=c['state_before_decision']
        with np.load(folder/f'{t:02d}_candidates.npz') as file:arrays={key:file[key].copy() for key in file.files}
        assert c['history']==state['history'][:t+3]
        assert before['active_suspended'] or t==r2.BOOTSTRAP[task]+1
        assert c['amplitude']==0. and not c['gate_open'] and c['selected_id']==c['reference_id']
        cache=FrozenCache(pool,c,arrays);j=ix[c['selected_id']]
        mu=float(arrays['mu_y'][j]);sd=float(arrays['sd_predictive_y'][j]);y=event['y'];R=sd**2+.0009
        rounds.append({'task':task,'query_round':t,'selected_id':c['selected_id'],
            'active_suspended_before':before['active_suspended'],'active_suspended_after':event['active_suspended_after'],
            'pending_status_before':before['pending']['status'] if before.get('pending') else '',
            'pending_status_after':event['pending_after']['status'] if event.get('pending_after') else '',
            'pending_evaluated_this_round':event.get('pending_validation') is not None,
            'active_knowledge_unchanged_since_bootstrap':c['knowledge']==state['decisions'][r2.BOOTSTRAP[task]-1]['knowledge'],
            'actual_KF_feature':event['measurement']['feature'],'actual_KF_gain':event['measurement']['actual_paired_SE_gain'],
            'active_coefficient_prior':event['bank_prior']['mean'] if event['bank_prior'] else None,
            'eligible_count':int(arrays['eligible'].sum()),'active_B4_revoked_ids':','.join(c['diagnosis']['revoked_ids'])})
        for proposal in proposals:
            created=proposal['query']
            if created>t:continue
            raw,_=cache.field(c['history'],proposal['region']);h=float(raw[j]*c['GP_scale_y'])
            coefficient=proposal['alpha'];shift=coefficient*h;residual=y-mu
            gain=float(residual**2-(residual-shift)**2)
            score,_=cache.score(c['history'],coefficient*raw);guided=pick(arrays,score)
            membership=base.membership(pool,proposal['knowledge'])
            violated=[clause['id'] for k,clause in enumerate(proposal['knowledge']['clauses']) if not membership[k,j]]
            row={'task':task,'proposal_created_query':created,'evaluation_query':t,
                'within_original_lifetime':t<=proposal['lifetime_last'],'actual_selected_id':c['selected_id'],
                'GP_mean':mu*scale,'GP_predictive_SD':sd*scale,'observed':event['raw_value'],
                'probe_coefficient':coefficient,'raw_field_shift':h*scale,'shadow_shift':shift*scale,
                'shadow_mean':(mu+shift)*scale,'paired_SE_gain_y_squared':gain,
                'paired_SE_gain_raw_squared':gain*scale**2,'delta_logscore_using_KF_R':gain/(2*R),
                'absolute_shift_over_GP_SD':abs(shift)/sd,'residual_times_shift':residual*shift,
                'positive_gain':gain>1e-12,'selected_inside_proposal':not violated,
                'selected_violates_clause_ids':','.join(violated),
                'unqueried_region_points_inside_gate':int(np.sum(arrays['eligible']&membership.all(0))),
                'guided_proposal_id':ids[guided],'guidance_would_change_choice':guided!=j,
                'guided_base_opportunity_loss':float(arrays['base_qLogEI'][j]-arrays['base_qLogEI'][guided])}
            followup.append(row)
            if created==t:
                receipt=event['pending_validation'];error=max(abs(receipt['raw_shift_y']-h),
                    abs(receipt['paired_SE_gain']-gain),abs(receipt['frozen_probe_prediction_y']-(mu+shift)),
                    float(np.max(abs(raw-arrays['pending_raw_shift_std']))))
                max_error=max(max_error,error);assert error<1e-9
                assert residual*shift<0. and gain<0.
                best_a=float(np.clip(residual/h,0,1)) if h else 0.
                assert best_a==0.
                initial.append({**row,'best_nonnegative_single_point_coefficient':best_a,
                    'inheritance_sources':len(proposal['inheritance']['sources']),
                    'admission_at_validation':'negative correction direction for every positive coefficient',
                    'proposal_knowledge':proposal['knowledge']})
        conflict=bool(c['diagnosis']['revoked_ids']) or c['diagnosis']['conflict_kind']!='none'
        key=c['bank_key'];book=before['subset_evidence'].get(key,{})
        previous=book.get('observations',[]);gain_values=[r['paired_SE_gain'] for r in previous]
        evidence_ok=bool(len(previous)>=2 and sum(gain_values)>1e-12 and gain_values[-1]>1e-12)
        alpha=c['bank_prior']['amplitude'] if c['bank_prior'] else 0.
        full,_=cache.score(c['history'],alpha*arrays['raw_shift_std'])
        half,_=cache.score(c['history'],.5*alpha*arrays['raw_shift_std'])
        full_j,half_j=pick(arrays,full),pick(arrays,half)
        stable=full_j==half_j
        cleared_only_global=bool(conflict and key and c['retained'] and c['admission']['admitted'] and evidence_ok and stable)
        gate={'task':task,'query_round':t,'canonical_subset_key':key,'conflict':conflict,
            'global_pause_after_decision':event['active_suspended_after'],
            'actual_subset_permission_evaluated':'BO_shadow_count' in c['subset_recovery'],
            'past_shadow_count':len(previous),'past_shadow_queries':','.join(str(r['query']) for r in previous),
            'past_shadow_sum_gain':float(sum(gain_values)),'past_shadow_latest_gain':gain_values[-1] if gain_values else '',
            'evidence_gate_would_pass':evidence_ok,'half_full_same_candidate':stable,
            'full_candidate_id':ids[full_j],'half_candidate_id':ids[half_j],
            'current_BO_id':c['reference_id'],'eligible_count':int(arrays['eligible'].sum()),
            'would_restore_if_only_global_pause_bypassed':cleared_only_global,
            'restoration_would_change_choice':bool(cleared_only_global and full_j!=j),
            'full_candidate_base_opportunity_loss':float(arrays['base_qLogEI'][j]-arrays['base_qLogEI'][full_j])}
        gates.append(gate)
        if t==r2.BOOTSTRAP[task]+1 or (task=='SnAr' and t==8) or (task=='HPLC' and t==9):
            with patch.object(r2,'subset_permission',wraps=r2.subset_permission) as spy:
                replay=r2.setup(cache,deepcopy(before),t)
                assert replay['selected']==replay['reference'] and not replay['gate_open']
                expected_calls=0 if before['active_suspended'] else 1
                assert spy.call_count==expected_calls
                probe_counts.append({'task':task,'query':t,'active_suspended_before':before['active_suspended'],
                                     'subset_permission_calls':spy.call_count,
                                     'scope':'before current-round refresh; final decision is suspended after refresh'})
    # All hypothetical candidates above were fixed before the first unqueried
    # target lookup. These are only one-step, frozen-prefix diagnostic effects.
    with np.load(ROOT/f'data/benchmarks/continuous_20261029/{task}/oracle.npz') as file:truth=file['raw_values'].copy()
    for row in gates:
        t=row['query_round'];event=state['decisions'][t-1];target=float(truth[ix[row['full_candidate_id']]])
        row['hypothetical_full_candidate_objective_analysis_only']=target
        row['actual_objective']=event['raw_value']
        row['hypothetical_minus_actual_objective']=target-event['raw_value']
        row['actual_incumbent_before']=event['incumbent_before']*scale
        row['hypothetical_incumbent_improvement']=max(target,event['incumbent_before']*scale)-event['incumbent_after']*scale
    for row in followup:
        event=state['decisions'][row['evaluation_query']-1]
        target=float(truth[ix[row['guided_proposal_id']]])
        row['guided_candidate_objective_analysis_only']=target
        row['guided_minus_actual_objective_analysis_only']=target-event['raw_value']
        row['guided_incumbent_improvement_analysis_only']=max(target,event['incumbent_before']*scale)-event['incumbent_after']*scale
    summaries=[]
    for proposal in proposals:
        rows=[r for r in followup if r['proposal_created_query']==proposal['query']]
        lifetime=[r for r in rows if r['within_original_lifetime']]
        summaries.append({'task':task,'proposal_query':proposal['query'],'original_lifetime_last':proposal['lifetime_last'],
            'later_positive_rounds_within_original_lifetime':[r['evaluation_query'] for r in lifetime if r['evaluation_query']>proposal['query'] and r['positive_gain']],
            'later_positive_rounds_all_frozen_prefixes':[r['evaluation_query'] for r in rows if r['evaluation_query']>proposal['query'] and r['positive_gain']],
            'lifetime_sum_delta_logscore_KF_R':sum(r['delta_logscore_using_KF_R'] for r in lifetime),
            'lifetime_positive_shadow_rounds_with_changed_choice':[r['evaluation_query'] for r in lifetime if r['positive_gain'] and r['guidance_would_change_choice']]})
    raw_history=np.array([h['raw_value'] for h in state['history']]);best=np.maximum.accumulate(raw_history)
    protected_queries=4 if task=='SnAr' else 3;plateau=best[3+protected_queries]
    assert np.all(best[3+protected_queries:]==plateau)
    floor=float((best[4:4+protected_queries].sum()+(16-protected_queries)*plateau)/16)
    assert abs(floor-float(best[4:].mean()))<1e-9
    assert sum(base.JIT.values())==0
    return {'task':task,'initial':initial,'followup':followup,'gates':gates,'rounds':rounds,
            'summaries':summaries,'max_pending_replay_error':max_error,'instrumented_branch_probes':probe_counts,
            'retained_floor':{'protected_queries':protected_queries,'SearchAUC_floor':floor,
                'plateau_reached_query':protected_queries,'plateau_objective':float(plateau),
                'scope':'only the same recorded prefix and fixed candidate outcomes; no new end-to-end or other-seed guarantee'}}


def main():
    protected=read(STUDY/'protected_snapshot.json');protected.update(read(STUDY/'release_inventory.json'))
    for name in ['release_inventory.json','release_receipt.json']:
        path=STUDY/name;protected[path.relative_to(ROOT).as_posix()]=filehash(path)
    for name,expected in protected.items():assert filehash(ROOT/name)==expected,name
    with ProcessPoolExecutor(max_workers=2) as executor:results=list(executor.map(task_analysis,['SnAr','HPLC']))
    for name,expected in protected.items():assert filehash(ROOT/name)==expected,name
    OUT.mkdir(parents=True,exist_ok=True)
    for name,key in [('initial_vetoes','initial'),('proposal_shadow_followup','followup'),('subset_gate_replay','gates'),('round_state','rounds')]:
        csv_write(OUT/(name+'.csv'),[row for result in results for row in result[key]])
    initial=[row for result in results for row in result['initial']]
    gates=[row for result in results for row in result['gates']]
    facts={'unique_refresh_proposals':len(initial),'all_initial_residual_times_shift_negative':all(r['residual_times_shift']<0 for r in initial),
        'initial_validation_points_outside_proposed_joint_region':sum(not r['selected_inside_proposal'] for r in initial),
        'subset_rounds_checked':len(gates),'final_recorded_subset_permission_evaluations':sum(r['actual_subset_permission_evaluated'] for r in gates),
        'subset_evidence_gate_passes':[(r['task'],r['query_round']) for r in gates if r['evidence_gate_would_pass']],
        'subset_all_local_gates_pass_if_global_pause_removed':[(r['task'],r['query_round']) for r in gates if r['would_restore_if_only_global_pause_bypassed']],
        'subset_then_changes_choice':[(r['task'],r['query_round']) for r in gates if r['restoration_would_change_choice']],
        'prospective_followup_summaries':[row for result in results for row in result['summaries']],
        'regression_prefix_floors':{result['task']:result['retained_floor'] for result in results},
        'scope':'Retrospective frozen-prefix calculations only; no alternative full-trajectory or future performance claim.'}
    write(OUT/'findings.json',facts)
    receipt={'status':'PASS','new_trajectories':0,'LLM_calls':0,'GP_fits':0,'experimental_outcome_provider_calls':0,
        'protected_files_unchanged':len(protected),'unique_initial_vetoes_recomputed':len(initial),
        'max_pending_field_prediction_replay_error':max(r['max_pending_replay_error'] for r in results),
        'instrumented_branch_probes':[row for result in results for row in result['instrumented_branch_probes']],
        'operator_compiles_in_loop':0,'script_sha256':filehash(__file__)}
    write(OUT/'receipt.json',receipt)
    write(OUT/'inventory.json',{path.relative_to(ROOT).as_posix():filehash(path) for path in OUT.glob('*') if path.is_file() and path.name!='inventory.json'})
    print(json.dumps({'receipt':receipt,'findings':facts},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
