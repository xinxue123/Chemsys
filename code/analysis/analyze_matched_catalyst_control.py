"""Outcome-blind selector custody, paired trajectories and fitted-model diagnosis."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
from statistics import mean, stdev
import csv
import json
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from matched_catalyst_control import study
from fast_hit_bo.model import make_model, digest
from fast_hit_bo.policy import choose
from llm_screen_qlogei import knowledge


def csv_save(name,rows):
    if not rows:
        return
    with (study.REPORT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        w.writeheader()
        w.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)


def metrics(state,method,threshold):
    y=[h['raw_value'] for h in state['history']]
    assert len(y)==8
    best=np.maximum.accumulate(y)
    hit=next((q for q,v in enumerate(y,1) if v>=threshold),None)
    return dict(seed=state['job']['seed'],method=method,initial_id=state['history'][0]['candidate_id'],
        SearchAUC8=float(best.mean()),final_best=float(best[-1]),first95=hit,
        hit95=hit is not None,yield_sequence=y)


def kernel_parameters(commit):
    ck=torch.load(ROOT/commit['decision']['checkpoint'],map_location='cpu',weights_only=False)
    model=make_model(torch.tensor(ck['train_X'],dtype=torch.double),
                     torch.tensor(ck['train_Y'],dtype=torch.double))
    model.load_state_dict(ck['state'])
    a,b=model.covar_module.kernels
    v1,v2=float(a.outputscale),float(b.outputscale)
    inner=a.base_kernel.kernels[1]
    vc=float(inner.outputscale)
    lc1=float(inner.base_kernel.lengthscale.item())
    lc2=float(b.base_kernel.kernels[1].lengthscale.item())
    correlation=(v1*(1+vc*np.exp(-1/lc1))+v2*np.exp(-1/lc2))/(v1*(1+vc)+v2)
    return dict(category_correlation=float(correlation),additive_category_scale=vc,
                additive_category_lengthscale=lc1,interaction_category_lengthscale=lc2)


def design_summary(pool,history):
    js=[pool.ix[h['candidate_id']] for h in history]
    x=np.asarray(pool.raw_pool['z'])[js,:3]
    cats=[pool.points[j]['catalyst'] for j in js]
    pairs=[(a,b) for a in range(len(js)) for b in range(a+1,len(js))]
    distance=[float(np.linalg.norm(x[a]-x[b])) for a,b in pairs]
    return dict(catalyst_counts=dict(Counter(cats)),distinct_catalysts=len(set(cats)),
        matched_catalyst_pairs=sum(cats[a]!=cats[b] and d<1e-8 for (a,b),d in zip(pairs,distance)),
        continuous_distance=mean(distance))


def analyze():
    torch.set_num_threads(1)
    p=study.check(True)
    assert study.read(study.BASE/'batch_receipt.json')['status']=='COMPLETE'
    pool=study.knowledge_gp.load_task(ROOT,'Reizman_Suzuki_Case_1',seed=0)
    study.POOL=pool
    with np.load(pool.source_path.with_name('oracle.npz'),allow_pickle=False) as z:
        truth=z['values']*100
    endpoint=study.read(study.BASE/'evaluation_endpoint.json')
    threshold=endpoint['threshold95']
    assert threshold==.95*float(truth.max())
    optimum=int(truth.argmax())
    all_metrics,pairs,rounds,profiles,replacements,designs=[],[],[],[],[],[]
    states={}
    for job in p['jobs']:
        folder=study.BASE/'runs'/job['id']
        state=study.read(folder/'state.json')
        source=study.read(study.SOURCE/'runs'/job['source_job']/'state.json')
        states[job['id']]=state
        assert state['status']=='COMPLETE' and len(state['history'])==8
        assert state['history'][:job['prefix_count']]==source['history'][:job['prefix_count']]
        assert len({h['candidate_id'] for h in state['history']})==8
        replay=study.initial_state(job)
        overrides=0
        for q,(obs,c) in enumerate(zip(state['history'],state['decisions']),1):
            assert c==study.read(folder/'rounds'/f'{q:02}_commit.json')
            assert c['prestate_sha256']==digest(replay)
            assert c['history_sha256']==digest(replay['history'])
            assert c['commit_sha256']==digest({k:v for k,v in c.items() if k!='commit_sha256'})
            assert c['protocol_sha256']==p['protocol_sha256']
            j=pool.ix[obs['candidate_id']]
            assert c['decision']['selected_id']==obs['candidate_id']
            assert abs(obs['y']*100-obs['raw_value'])<1e-8 and abs(obs['raw_value']-truth[j])<1e-8
            feedback=study.read(folder/'rounds'/f'{q:02}_feedback.json')
            if c['kind']=='reused_prefix':
                assert q<=job['prefix_count']
                assert study.sha(ROOT/c['source_commit_path'])==c['source_commit_sha256']
                old=study.read(ROOT/c['source_commit_path'])
                assert c['decision']==old['decision'] and c['notes_at_commit']==old['notes_at_commit']
                old_feedback=study.read(study.SOURCE/'runs'/job['source_job']/'rounds'/f'{q:02}_feedback.json')
                assert feedback['notes_after']==old_feedback['notes_after']
                notes=feedback['notes_after']
            else:
                assert c['notes_at_commit']==replay['notes']
                assert study.sha(ROOT/c['decision']['checkpoint'])==c['decision']['checkpoint_sha256']
                assert study.sha(folder/'rounds'/f'{q:02}_GP.npz')==c['decision']['GP_arrays_sha256']
                with np.load(folder/'rounds'/f'{q:02}_GP.npz',allow_pickle=False) as z:
                    ctx={key:z[key].copy() for key in ('mu','variance','logei','available','priorities')}
                    ctx['incumbent']=max(h['y'] for h in replay['history'])
                    base=choose(ctx['logei'],ctx['available'],ctx['priorities'])
                    assert c['decision']['BO_reference_id']==pool.ids[base]
                    if q==job['replacement']['query']:
                        match,mask=study.matched_choice(pool,replay['history'],ctx)
                        assert match['selected_id']==obs['candidate_id']==job['replacement']['selected_id']
                        np.testing.assert_array_equal(mask,z['matched_mask'])
                        assert c['decision']['design_override']
                        overrides+=1
                    else:
                        assert j==base and not c['decision']['design_override']
                    expected=study.forecasts(replay['notes'],ctx,j)
                    assert expected==c['decision']['forecasts']
                    notes,events=knowledge.feedback(replay['notes'],expected,obs,q)
                    assert notes==feedback['notes_after'] and events==feedback['events']
            replay['history'].append(obs)
            replay['decisions'].append(c)
            replay['notes']=notes
            replay['status']='COMPLETE' if q==8 else 'RUNNING'
        assert overrides==1 and replay==state
        new=metrics(state,'one_catalyst_contrast',threshold)
        old=metrics(source,'original_handoff',threshold)
        all_metrics.extend([old,new])
        pairs.append(dict(seed=job['seed'],physical_primary=job['physical_primary'],
            prefix_ids=[h['candidate_id'] for h in state['history'][:job['prefix_count']]],
            old_SearchAUC8=old['SearchAUC8'],new_SearchAUC8=new['SearchAUC8'],
            delta_SearchAUC8=new['SearchAUC8']-old['SearchAUC8'],
            old_final=old['final_best'],new_final=new['final_best'],delta_final=new['final_best']-old['final_best'],
            old_first95=old['first95'],new_first95=new['first95']))
        for method,prefix in [('same_LLM_start_BO','llm_start_bo'),('random_start_BO','bo')]:
            control=study.read(study.BO_SOURCE/'runs'/f'{prefix}_s{job["seed"]}'/'state.json')
            all_metrics.append(metrics(control,method,threshold))
        for method,s in [('original_handoff',source),('one_catalyst_contrast',state)]:
            best=np.maximum.accumulate([h['raw_value'] for h in s['history']])
            for q,(h,c) in enumerate(zip(s['history'],s['decisions']),1):
                rounds.append(dict(seed=job['seed'],method=method,query=q,candidate_id=h['candidate_id'],
                    yield_percent=h['raw_value'],best=float(best[q-1]),rule=c['decision']['rule'],
                    is_reused_prefix=method=='one_catalyst_contrast' and q<=job['prefix_count'],
                    **pool.points[pool.ix[h['candidate_id']]]))
        c=state['decisions'][job['replacement']['query']-1]
        a=pool.ix[c['decision']['anchor_id']]
        j=pool.ix[c['decision']['selected_id']]
        old_j=pool.ix[c['decision']['original_replaced_id']]
        replacements.append(dict(seed=job['seed'],query=job['replacement']['query'],anchor=pool.ids[a],
            anchor_yield=float(truth[a]),original_id=pool.ids[old_j],original_yield=float(truth[old_j]),
            contrast_id=pool.ids[j],contrast_yield=float(truth[j]),
            original_logei_budget=c['decision']['original_budget_epsilon'],
            actual_logei_gap=c['decision']['actual_logei_gap'],
            passes_original_budget=c['decision']['passes_original_budget'],
            retained_exp_qLogEI=c['decision']['retained_exp_qLogEI'],conditions=pool.points[j]))
        for q in range(4,9):
            with np.load(folder/'rounds'/f'{q:02}_GP.npz') as z:
                new_arrays={key:z[key].copy() for key in ('mu','variance','available','logei')}
            with np.load(study.SOURCE/'runs'/job['source_job']/'rounds'/f'{q:02}_GP.npz') as z:
                old_arrays={key:z[key].copy() for key in ('mu','variance','available','logei')}
            common=new_arrays['available'] & old_arrays['available']
            profile=dict(seed=job['seed'],query=q,observations=q-1,common_unqueried=int(common.sum()))
            for tag,s,arrays in [('old',source,old_arrays),('new',state,new_arrays)]:
                params=kernel_parameters(s['decisions'][q-1])
                mu=arrays['mu']*100
                sd=np.sqrt(arrays['variance'])*100
                profile.update({tag+'_'+key:value for key,value in params.items()})
                profile.update({tag+'_common_mean_sd':float(sd[common].mean()),
                    tag+'_common_coverage95':float(np.mean(np.abs(mu[common]-truth[common])<=1.96*sd[common])),
                    tag+'_common_RMSE':float(np.sqrt(np.mean((mu[common]-truth[common])**2))),
                    tag+'_optimum_mu':float(mu[optimum]),tag+'_optimum_sd':float(sd[optimum]),
                    tag+'_optimum_rank':1+int(np.count_nonzero(arrays['available'] &
                                                        (arrays['logei']>arrays['logei'][optimum]))),
                    tag+'_optimum_observed':not bool(arrays['available'][optimum])})
            profiles.append(profile)
            designs.append(dict(seed=job['seed'],query=q,method='old',**design_summary(pool,source['history'][:q-1])))
            designs.append(dict(seed=job['seed'],query=q,method='new',**design_summary(pool,state['history'][:q-1])))
    for job in p['jobs']:
        if job['physical_primary']!=job['id']:
            assert states[job['id']]['history']==states[job['physical_primary']]['history']
    aggregates=[]
    for method in ('original_handoff','one_catalyst_contrast','same_LLM_start_BO','random_start_BO'):
        subset=[m for m in all_metrics if m['method']==method]
        aggregates.append(dict(method=method,n=len(subset),SearchAUC8=mean(m['SearchAUC8'] for m in subset),
            SearchAUC8_sd=stdev(m['SearchAUC8'] for m in subset),final_best=mean(m['final_best'] for m in subset),
            target_hits=sum(m['hit95'] for m in subset),shared_two_observation_prefix=method in ('original_handoff','one_catalyst_contrast')))
    mechanisms=[]
    for q in range(4,9):
        rs=[r for r in profiles if r['query']==q]
        mechanisms.append(dict(query=q,n=q-1,
            old_category_merge_count=sum(r['old_category_correlation']>.99 for r in rs),
            new_category_merge_count=sum(r['new_category_correlation']>.99 for r in rs),
            **{key:mean(r[key] for r in rs) for key in (
                'old_common_mean_sd','new_common_mean_sd','old_common_coverage95','new_common_coverage95',
                'old_optimum_mu','new_optimum_mu','old_optimum_sd','new_optimum_sd')}))
    unique_pairs=[r for r in pairs if r['physical_primary']==f"contrast_s{r['seed']}"]
    receipt=dict(status='PASS',protocol_sha256=p['protocol_sha256'],paired_labels=len(pairs),
        distinct_physical_branches=len(unique_pairs),reused_prefix_observations=10,
        new_sequential_choices=30,new_LLM_calls=0,single_override_per_trajectory=True,
        all_other_new_choices_exact_native_BO=True,all_original_files_unchanged=True,
        all_replacement_gates_explicitly_overridden=all(not r['passes_original_budget'] for r in replacements),
        paired_SearchAUC8_increment=mean(r['delta_SearchAUC8'] for r in pairs),
        paired_final_increment=mean(r['delta_final'] for r in pairs),
        paired_wins=sum(r['delta_SearchAUC8']>1e-9 for r in pairs),
        paired_losses=sum(r['delta_SearchAUC8']<-1e-9 for r in pairs),
        paired_ties=sum(abs(r['delta_SearchAUC8'])<=1e-9 for r in pairs),
        three_branch_mean_increment=mean(r['delta_SearchAUC8'] for r in unique_pairs),
        operator_compiles_in_loop=0)
    csv_save('metrics.csv',all_metrics);csv_save('paired.csv',pairs);csv_save('summary.csv',aggregates)
    csv_save('rounds.csv',rounds);csv_save('replacement_choices.csv',replacements)
    csv_save('mechanisms_by_prefix.csv',profiles);csv_save('mechanism_summary.csv',mechanisms)
    csv_save('design_profiles.csv',designs);csv_save('excluded_labels.csv',p['exclusions'])
    study.write(study.REPORT/'audit.json',receipt,replace=True)
    lines=['# One catalyst contrast at a fixed Case 1 LLM prefix','',
        'Five paired labels use three distinct physical branches. The first two observations are identical; query3 is replaced once. Queries4-8 use native qLogEI in both branches. The original handoff already used that same suffix. No new LLM response or GP recipe change is introduced.','',
        '| Method, eligible labels only | SearchAUC8 | Mean final best | 95% target hits |',
        '|---|---:|---:|---:|']
    for r in aggregates:
        lines.append(f"| {r['method']} | {r['SearchAUC8']:.4f} | {r['final_best']:.4f} | {r['target_hits']}/5 |")
    lines.extend(['', 'Only the original handoff is an exact two-observation-prefix comparator. The other BO rows are frozen contextual references with different prefixes. All tables retain the eligible subset; no full-cohort control mean is substituted.', '',
        f"The paired mean SearchAUC8 change is {receipt['paired_SearchAUC8_increment']:+.4f}; final best changes by {receipt['paired_final_increment']:+.4f} percentage points. Wins/losses/ties are {receipt['paired_wins']}/{receipt['paired_losses']}/{receipt['paired_ties']}. Equal weighting of the three distinct branches gives SearchAUC8 change {receipt['three_branch_mean_increment']:+.4f}. These are descriptive, not independent-replicate significance estimates.", '',
        '## Replacement details','', '| Label | Anchor | Original third yield | Matched third yield | Retained exp(qLogEI) |', '|---|---|---:|---:|---:|'])
    for r in replacements:
        lines.append(f"| {r['seed']} | {r['anchor']} | {r['original_yield']:.4f} | {r['contrast_yield']:.4f} | {r['retained_exp_qLogEI']:.3f} |")
    lines.extend(['', 'All exact contrasts fail the original acquisition budget. This is the declared one-action experimental override, not a passed budget review. Highest native acquisition and its frozen physical tie rule choose the catalyst; archived outcomes and candidate IDs do not select it.', '',
        '## Mechanism diagnostics','', '| Observations before decision | Near-merged category kernels, old/new | Common-pool latent SD, old/new | Common-pool +/-1.96 SD coverage, old/new |', '|---|---:|---:|---:|'])
    for r in mechanisms:
        lines.append(f"| {r['n']} | {r['old_category_merge_count']}/5 / {r['new_category_merge_count']}/5 | {r['old_common_mean_sd']:.3f} / {r['new_common_mean_sd']:.3f} | {100*r['old_common_coverage95']:.1f}% / {100*r['new_common_coverage95']:.1f}% |")
    lines.extend(['', 'Near merging means full two-component covariance correlation above0.99 for different nominal catalysts at identical continuous inputs. This is a descriptive diagnostic. Coverage is measured retrospectively on the same unqueried candidates in each pair, not a formal frequentist coverage guarantee. Wider uncertainty or a retained category component alone is not optimization improvement.', '',
        'Seeds1,4,5,8,9 are excluded because no original knowledge action has an exact observed-anchor catalyst counterpart, or because no postinitial knowledge action occurs. They are not approximated or silently converted to another intervention. Seed0/7 and seed2/3 are repeated physical branches.', '',
        'Each new trajectory charges two inherited observations and six later choices: eight total. There are40 records and30 per-trajectory new choices; duplicates do not represent30 distinct physical experiments. Pre-outcome commits, model checkpoints, full score arrays and warnings are retained. Three persistent one-thread workers warm operators once, with no in-loop JIT compilation.', '',
        'The fixed BO suffix isolates this particular point intervention from downstream LLM-response randomness. It does not evaluate the full event-refresh policy after intervention. The benchmark has already been inspected: this is a post-hoc mechanistic experiment, not a blind validation or a general claim about catalyst-control strategies.','',
        'See paired.csv, rounds.csv, replacement_choices.csv, mechanisms_by_prefix.csv, mechanism_summary.csv, design_profiles.csv, excluded_labels.csv and audit.json. The separate fresh_native_check.json records independent numerical reconstruction when available.'])
    (study.REPORT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    study.check(True)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    analyze()
