"""Post hoc oracle analysis after every trajectory is complete; never online."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from datetime import datetime,timezone
import json
import math
import os
import sys
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from chemical_hypothesis_bo import study as s,experts,policy,hypotheses as h
import chemical_hypothesis_active as active


def mix(values,weights,drop=None):
    indices=[k for k,i in enumerate(values['ids']) if i!=drop]
    logw=np.asarray([weights[values['ids'][k]] for k in indices]);logw-=logsumexp(logw)
    return dict(values,mixture=logsumexp(values['scores'][indices]+logw[:,None],axis=0))


def one(task):
    active.activate()
    pool=s.native.load_task(ROOT,task,seed=0);rows=[];ablation=[];weight_rows=[];refreshes=[];credits=[]
    # Counterfactual outcomes are opened ONLY here, after global completion.
    # They never enter a prompt, fit, weight update or experimental trajectory.
    with np.load(pool.source_path.with_name('oracle.npz'),allow_pickle=False) as source:
        raw=source['raw_values'].copy() if 'raw_values' in source else source['values'].copy()*pool.scale
    for arm in ('fixed','adaptive'):
        state=s.read(s.folder_for(task,arm)/'state.json');assert state['status']=='COMPLETE'
        gp=s.native.GPAdapter(pool,active.cache_for(task),seed=0)
        # Separate all diagnostic preference caches from committed run artifacts.
        gp.directory=s.REPORT/'posthoc_cache'/s.SHORT[task]
        no_feedback_weights=None;old_notes=[];old_weights=dict(base=0.)
        for q in range(1,17):
            cp=s.read(s.folder_for(task,arm)/'rounds'/f'{q:02d}_commit.json');d=cp['decision']
            notes=cp['knowledge_state']['notes'];weights=d['log_weights_before']
            with np.load(s.folder_for(task,arm)/'rounds'/f'{q:02d}_arrays.npz',allow_pickle=False) as a:
                values=dict(ids=a['expert_ids'].tolist(),scores=a['expert_logEI'].copy(),mixture=a['mixture_logEI'].copy())
                ctx=dict(eligible=a['eligible'].copy(),bo_index=pool.ix[d['BO_reference_id']],base_logei=a['base_logei'].copy(),tie_priority=a['tie_priority'].copy())
            j=pool.ix[d['selected_id']];b=ctx['bo_index'];inc=float(state['decisions'][q-1]['incumbent_before'])
            assert policy.choose(ctx,values,q)==j
            ids=values['ids'];chosen_contributions=np.exp(values['scores'][:,j]+np.asarray([weights[i] for i in ids])-values['mixture'][j])
            row=dict(task=s.SHORT[task],arm=arm,query=q,selected=d['selected_id'],BO_reference=d['BO_reference_id'],
                selected_raw=float(raw[j]),BO_reference_raw=float(raw[b]),incumbent_before=inc,
                observed_incumbent_after=max(inc,float(raw[j])),differs_from_BO=bool(j!=b),
                immediate_raw_gain_vs_same_prefix_BO=float(raw[j]-raw[b]),
                immediate_best_gain_vs_same_prefix_BO=float(max(inc,raw[j])-max(inc,raw[b])),
                native_budget_size=d['budget_size'],native_logEI_gap=d['native_logEI_gap'],
                selected_conditions=json.dumps(pool.points[j]),BO_reference_conditions=json.dumps(pool.points[b]),
                dominant_acquisition_expert=ids[int(np.argmax(chosen_contributions))],
                knowledge_mass=1-math.exp(weights['base']),refresh=bool(cp['LLM_request'] and q>1))
            for note in notes:
                without=policy.choose(ctx,mix(values,weights,drop=note['id']),q)
                if without!=j:
                    plus,minus=h.masks(pool,note)
                    context_scope={a:v for a,v in note['preferred'].items() if a not in note['focus']}
                    in_context=h.region(pool,context_scope)
                    ablation.append(dict(task=s.SHORT[task],arm=arm,query=q,id=note['id'],claim=note['claim'],
                        preferred=json.dumps(note['preferred']),comparison=json.dumps(note['comparison']),
                        full_selected=pool.ids[j],without_this_hypothesis=pool.ids[without],
                        full_raw=float(raw[j]),without_raw=float(raw[without]),
                        immediate_raw_gain=float(raw[j]-raw[without]),
                        immediate_best_gain=float(max(inc,raw[j])-max(inc,raw[without])),
                        selected_in_preferred=bool(plus[j]),selected_in_comparison=bool(minus[j]),
                        selected_inside_nonfocus_context=bool(in_context[j]),
                        without_inside_nonfocus_context=bool(in_context[without]),
                        weight=math.exp(weights[note['id']]),selection_sensitive_to_hypothesis=True))
            if arm=='adaptive':
                if q==1:no_feedback_weights=policy.initial(notes)
                elif cp['LLM_request']:no_feedback_weights=policy.edit_weights(no_feedback_weights,notes)
                equal_id=pool.ix[d['equal_prior_shadow_id']]
                nf=policy.choose(ctx,mix(values,no_feedback_weights),q)
                row.update(no_feedback_weight_shadow=pool.ids[nf],no_feedback_weight_shadow_raw=float(raw[nf]),
                    feedback_only_changes_choice=bool(nf!=j),
                    feedback_only_immediate_best_gain=float(max(inc,raw[j])-max(inc,raw[nf])),
                    equal_prior_shadow=pool.ids[equal_id],equal_prior_changes_choice=bool(equal_id!=j))
                if nf!=j:
                    credits.append(dict(task=s.SHORT[task],query=q,selected=pool.ids[j],without_forecast_updates=pool.ids[nf],
                        selected_raw=float(raw[j]),without_raw=float(raw[nf]),
                        immediate_raw_gain=float(raw[j]-raw[nf]),immediate_best_gain=row['feedback_only_immediate_best_gain'],
                        actual_weights=json.dumps(weights),without_feedback_weights=json.dumps(no_feedback_weights)))
                if q>1 and cp['LLM_request']:
                    context=gp.context(state['history'][:q+3],q)
                    previous_values=experts.evaluate(gp,context,old_notes,old_weights)
                    old_j=policy.choose(context,previous_values,q)
                    retained=set(n['id'] for n in notes)&set(n['id'] for n in old_notes)
                    refreshes.append(dict(task=s.SHORT[task],query=q,reasons=';'.join(cp['refresh']['reasons']),
                        selected=pool.ids[j],retain_old_library_selection=pool.ids[old_j],changed=bool(j!=old_j),
                        selected_raw=float(raw[j]),retain_old_raw=float(raw[old_j]),
                        immediate_best_gain=float(max(inc,raw[j])-max(inc,raw[old_j])),
                        old_hypotheses=len(old_notes),new_hypotheses=len(notes),structurally_retained=len(retained),
                        new_claims=json.dumps([n['claim'] for n in notes])))
                old_notes=deepcopy(notes);old_weights=deepcopy(state['decisions'][q-1]['feedback']['log_weights_after'])
            for item in state['decisions'][q-1]['feedback']['predictive_scores']:
                identity=item['id'];base=state['decisions'][q-1]['feedback']['predictive_scores'][0]['log_score']
                weight_rows.append(dict(task=s.SHORT[task],arm=arm,query=q,id=identity,
                    weight_before=math.exp(weights[identity]),
                    weight_after=math.exp(state['decisions'][q-1]['feedback']['log_weights_after'][identity]),
                    log_predictive_gain_vs_GP=item['log_score']-base,mean_shift_y=item['mean_shift_y'],
                    informative=bool(abs(item['mean_shift_y'])>1e-6)))
            rows.append(row);gp.release()
    return dict(decisions=rows,hypothesis_effects=ablation,weights=weight_rows,refresh_effects=refreshes,feedback_effects=credits)


def main():
    audit=s.read(s.REPORT/'audit.json');assert audit['status']=='PASS_REPLAY_AND_CUSTODY'
    p=s.check(True)
    active.activate(record=True)
    for task in s.native.TASKS:
        for arm in ('fixed','adaptive'):assert s.read(s.folder_for(task,arm)/'state.json')['status']=='COMPLETE'
    started=datetime.now(timezone.utc).isoformat()
    with ProcessPoolExecutor(max_workers=8,initializer=s.initialize) as executor:parts=list(executor.map(one,s.native.TASKS))
    tables={k:pd.DataFrame([row for part in parts for row in part[k]]) for k in parts[0]}
    for name,frame in tables.items():frame.to_csv(s.REPORT/f'posthoc_{name}.csv',index=False,encoding='utf-8-sig')
    versions=[];metrics=[];divergences=[]
    for task in s.native.TASKS:
        for method,state in [('pure_BO',s.read(ROOT/p['controls'][task]['path'])),
            ('original_strict_contrast',s.read(s.legacy.BASE/'runs'/s.SHORT[task]/'state.json')),
            ('previous_kernel_F',s.read(s.previous.folder_for(task,'feedback')/'state.json')),
            ('fixed',s.read(s.folder_for(task,'fixed')/'state.json')),
            ('adaptive',s.read(s.folder_for(task,'adaptive')/'state.json'))]:
            assert [(v['candidate_id'],v['y']) for v in state['history'][:4]]==[(v['candidate_id'],v['y']) for v in s.read(ROOT/p['controls'][task]['path'])['history'][:4]]
            r=s.legacy.summary(state,method,task)
            calculated=sum(max(v['raw_value'] for v in state['history'][:4+i]) for i in range(1,17))/16
            assert abs(calculated-r['SearchAUC'])<1e-9
            r['unit']=s.native.load_task(ROOT,task,seed=0).unit;metrics.append(r)
        fixed=s.read(s.folder_for(task,'fixed')/'state.json');adaptive=s.read(s.folder_for(task,'adaptive')/'state.json')
        first=next((q for q in range(1,17) if fixed['history'][q+3]['candidate_id']!=adaptive['history'][q+3]['candidate_id']),None)
        if first:
            cp=s.read(s.folder_for(task,'adaptive')/'rounds'/f'{first:02d}_commit.json')
            divergences.append(dict(task=s.SHORT[task],first_divergence_query=first,
                prefix_identical=fixed['history'][:first+3]==adaptive['history'][:first+3],
                refreshed_at_divergence=bool(cp['LLM_request']),
                fixed_raw=fixed['history'][first+3]['raw_value'],adaptive_raw=adaptive['history'][first+3]['raw_value']))
    for path in sorted(active.responses()):
        saved=s.read(path);req=s.read(ROOT/saved['request'])
        for note in saved['plan']['notes']:
            versions.append(dict(task=req['task'],kind=req['kind'],query=req['query'],id=note['id'],
                claim=note['claim'],rationale=note['rationale'],preferred=json.dumps(note['preferred']),comparison=json.dumps(note['comparison']),
                supporting_observations=json.dumps(note['supporting_observation_ids']),falsification=note['falsification']))
    frame=pd.DataFrame(metrics);frame.to_csv(s.REPORT/'unified_results.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(versions).to_csv(s.REPORT/'knowledge_versions.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(divergences).to_csv(s.REPORT/'first_divergences.csv',index=False,encoding='utf-8-sig')
    summary=[]
    for task in s.SHORT.values():
        m=frame[frame.task==task].set_index('method');df=tables['decisions'];df=df[(df.task==task)&(df.arm=='adaptive')]
        credit=tables['feedback_effects'];credit=credit[credit.task==task]
        refresh=tables['refresh_effects'];refresh=refresh[refresh.task==task]
        summary.append(dict(task=task,adaptive_minus_BO=m.loc['adaptive','SearchAUC']-m.loc['pure_BO','SearchAUC'],
            adaptive_minus_previous=m.loc['adaptive','SearchAUC']-m.loc['previous_kernel_F','SearchAUC'],
            adaptive_minus_fixed=m.loc['adaptive','SearchAUC']-m.loc['fixed','SearchAUC'],
            native_choice_changes=int(df.differs_from_BO.sum()),
            feedback_only_choice_changes=len(credit),feedback_positive_best_changes=int((credit.immediate_best_gain>1e-10).sum()),
            feedback_negative_best_changes=int((credit.immediate_best_gain< -1e-10).sum()),
            refresh_choice_changes=int(refresh.changed.sum()),refresh_positive_best_changes=int((refresh.immediate_best_gain>1e-10).sum()),
            refresh_negative_best_changes=int((refresh.immediate_best_gain< -1e-10).sum())))
    pd.DataFrame(summary).to_csv(s.REPORT/'mechanism_summary.csv',index=False,encoding='utf-8-sig')
    record=dict(started_UTC=started,status='COMPLETE_POSTHOC_ONLY',experimental_queries_added=0,LLM_calls_added=0,
        oracle_counterfactuals_used_only_after_all_completion=True,
        scope='Same actual prefix and GP. Each one-step counterfactual changes weights, removes one expert, or retains the preceding hypothesis library. This is not a full-trajectory causal ablation.',
        feedback_shadow='Propagate initial weights through the same actual refresh libraries using edit_weights, but never predictive feedback. Unlike equal-prior resets, this preserves the new-hypothesis-mass rule.',
        source_sha256=s.sha(__file__),protocol_sha256=p['protocol_sha256'],active_comparison_sha256=s.sha(s.REPORT/'active_comparison.json'))
    s.write(s.REPORT/'posthoc_scope.json',record,replace=True)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__=='__main__':main()
