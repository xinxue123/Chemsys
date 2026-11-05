"""Separate reproducible numerical traces from initialization-policy inference."""
from collections import Counter
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'experiments/runs/llm_event_start_qlogei_20261104'
OUT=ROOT/'experiments/reports/llm_initialization_audit_20261104'
REPORT=ROOT/'experiments/reports/llm_event_start_qlogei_20261104'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    reconstruction=read(OUT/'fresh_reconstruction.json')
    assert reconstruction['strict_choice_matches']==reconstruction['prefixes']==105
    assert reconstruction['maximum_logei_error']==0
    p=read(BASE/'protocol.json')
    pool_path=ROOT/'experiments/runs/suzuki_guarded/inputs/case1/pool.json'
    pool=read(pool_path)
    ids=pool['ids']
    ix={cid:j for j,cid in enumerate(ids)}
    with np.load(pool_path.with_name('oracle.npz'),allow_pickle=False) as z:
        truth=z['values']*100
    target=int(truth.argmax())
    metrics=[]
    states={}
    for job in p['jobs']:
        state=read(BASE/'runs'/job['id']/'state.json')
        states[job['id']]=state
        values=np.array([h['raw_value'] for h in state['history']])
        metrics.append(dict(job=job['id'],arm=job['arm'],initial_id=state['history'][0]['candidate_id'],
            observations=len(values),first_yield=float(values[0]),SearchAUC8=float(np.maximum.accumulate(values).sum()/len(values)),
            final_best=float(values.max())))
    with (REPORT/'metrics.csv').open(encoding='utf-8-sig',newline='') as f:
        old=list(csv.DictReader(f))
    for m in metrics:
        row=next(r for r in old if r['method']==m['arm'] and int(r['seed'])==int(m['job'].split('_s')[-1]))
        assert abs(m['SearchAUC8']-float(row['SearchAUC8']))<1e-12
        assert abs(m['final_best']-float(row['final_best']))<1e-12
    reg=read(ROOT/'experiments/runs/llm_initialization_audit_20261104/registration.json')
    for group in reg['jobs']:
        reference=states[group['reference_job']]['history']
        for key in group['source_jobs']:
            assert states[key]['history']==reference
    initial=[]
    for seed in range(10):
        c=states[f'event_s{seed}']['decisions'][0]
        req=ROOT/c['LLM']['request_relpath']
        export=read(req.parent/'export.json')
        public=read(ROOT/export['input'])['public']
        point=c['decision']['proposal']['conditions']
        initial.append(dict(seed=seed,selected_id=c['decision']['selected_id'],
            exact_copy_of_displayed_example=point in public['input_only_examples'],
            displayed_examples=len(public['input_only_examples']),
            projection_distance=c['decision']['projection_distance'],
            **point))
    details=[]
    maxima=[]
    for group in reg['jobs']:
        state=states[group['reference_job']]
        with np.load(BASE/'runs'/group['reference_job']/'rounds/02_GP.npz',allow_pickle=False) as z:
            logei=z['logei'];available=z['available']
            maximum=float(logei[available].max())
            js=np.flatnonzero(available & (logei==maximum))
            first=state['history'][0]
            chosen=ix[state['history'][1]['candidate_id']]
            details.append(dict(initial_id=group['initial_id'],reference_job=group['reference_job'],
                initial_yield=first['raw_value'],selected_id=ids[chosen],selected_yield=float(truth[chosen]),
                selected_GP_mean=float(z['mu'][chosen]*100),selected_GP_sd=float(np.sqrt(z['variance'][chosen])*100),
                selected_exp_qLogEI_pp=float(np.exp(logei[chosen])*100),
                target_id=ids[target],target_true_yield=float(truth[target]),
                target_GP_mean=float(z['mu'][target]*100),target_GP_sd=float(np.sqrt(z['variance'][target])*100),
                target_exp_qLogEI_pp=float(np.exp(logei[target])*100),
                target_ordinal_rank=1+int((logei[available]>logei[target]).sum()),
                exact_maxima=len(js),tied_yield_min=float(truth[js].min()),tied_yield_max=float(truth[js].max()),
                tied_fraction_hitting95=float((truth[js]>=.95*truth[target]).mean())))
            for j in js:
                maxima.append(dict(initial_id=group['initial_id'],reference_job=group['reference_job'],
                    candidate_id=ids[j],selected=j==chosen,qLogEI=float(logei[j]),
                    true_yield=float(truth[j]),**pool['points'][j]))
    save_csv(OUT/'metrics_recomputed.csv',metrics)
    save_csv(OUT/'initial_prompt_anchoring.csv',initial)
    save_csv(OUT/'second_query_comparison.csv',details)
    save_csv(OUT/'second_query_all_exact_maxima.csv',maxima)
    llm=[m for m in metrics if m['arm']=='llm_start_bo']
    random=[m for m in metrics if m['arm']=='bo']
    unique={m['initial_id']:m for m in llm}
    dominant=next(d for d in details if d['initial_id']=='p03260')
    comparator=next(d for d in details if d['reference_job']=='bo_s3')
    findings=dict(numerical_reconstruction='PASS',recomputed_metrics=30,
        same_initial_condition_implies_same_pure_qLogEI_trajectory=True,
        posterior_mean_injection_after_LLM_initialization=False,
        LLM_response_count=10,distinct_LLM_initial_points=len(unique),
        exact_example_copies=sum(r['exact_copy_of_displayed_example'] for r in initial),
        LLM_projected_initial_counts=dict(Counter(m['initial_id'] for m in llm)),
        random_response_weighted_SearchAUC8=float(np.mean([m['SearchAUC8'] for m in random])),
        LLM_response_weighted_SearchAUC8=float(np.mean([m['SearchAUC8'] for m in llm])),
        LLM_unique_point_unweighted_SearchAUC8_diagnostic_only=float(np.mean([m['SearchAUC8'] for m in unique.values()])),
        unique_point_weighting_is_different_estimand=True,
        GP_prior_mean_percent=100*p['GP_parameters']['prior_mean'],
        GP_hyperparameters_fixed_until_observations=p['GP_parameters']['fit_after_observations'],
        unchanged_original_observations=True,new_LLM_calls=0,new_oracle_experiments=0)
    (OUT/'findings.json').write_text(json.dumps(findings,indent=2)+'\n',encoding='utf-8')
    lines=['# Audit of the Case 1 LLM-initialization comparison','',
        'The recorded numbers reproduce exactly under the declared GP recipe. They do not '
        'establish that LLM initialization generally harms Bayesian optimization. The second '
        'choice is explained by the initial location/value and imposed GP prior. Concentrated '
        'proposals and exact acquisition ties limit the broader interpretation of this pilot.','',
        '## Independent numerical reconstruction','',
        'A standalone audit imports native BoTorch directly. It reconstructs physical input '
        'normalization, uses a separate nominal category relabeling, constructs the Mixed GP '
        'from the protocol parameters, and computes all MC512 qLogEI scores. It never loads '
        'the original fitted models or score cache for computation. Archived round arrays '
        'are opened only after the new calculation for comparison.','',
        'All 15 distinct initial conditions and 105 query prefixes were checked. There were '
        '60 fresh GP fits and 105/105 exact selected-ID matches. Maximum differences in GP '
        'mean, variance and qLogEI were all zero. All 30 SearchAUC values were independently '
        'recomputed from actual round yields using their observation count. No original '
        'scientific files changed and there were no new LLM calls or oracle experiments. '
        f"Native fitting emitted warnings on {reconstruction['prefixes_with_optimizer_warnings']} prefixes, "
        'but those prefixes also reproduced the stored fitted predictions and selections.','',
        '## What the comparison actually measures','',
        'Both columns use the same qLogEI rule and the same surrogate recipe. The treatment '
        'changes the first physical experiment. From the identical LLM-selected first point, '
        'the event arm and pure-qLogEI control have identical complete trajectories. The '
        'refresh responders all abstain, so no later LLM action altered the GP or the choice. '
        'The result is therefore not a comparison of two different BO implementations.','',
        'The recipe fixes the GP prior mean at 50% yield and prior SD at 25 percentage points. '
        'Kernel parameters remain fixed for prefixes containing one, two or three observations; '
        'marginal-likelihood fitting begins after four. This is the nondegenerate fast-hit '
        'recipe introduced for the earlier rollout comparison. It must not be conflated '
        'with every historical pure-BO recipe or described as a universal qLogEI baseline.','',
        '## A directly inspectable second choice','',
        'The dominant LLM start p03260 is P1-L5 at 330.326 s, 88.340 degC and 2.09054 mol%, '
        'with an observed yield of 28.0416%. Its second-query choices are:',
        '',
        '| Candidate | True yield (%) | GP mean (%) | GP SD (pp) | exp(qLogEI), yield pp |',
        '|---|---|---|---|---|',
        f"| Selected p02064, cold/short/low-loading | {dominant['selected_yield']:.2f} | {dominant['selected_GP_mean']:.2f} | {dominant['selected_GP_sd']:.2f} | {dominant['selected_exp_qLogEI_pp']:.2f} |",
        f"| Actual best p03053 | {dominant['target_true_yield']:.2f} | {dominant['target_GP_mean']:.2f} | {dominant['target_GP_sd']:.2f} | {dominant['target_exp_qLogEI_pp']:.2f} |",'',
        'The single warm-region result is below the imposed 50% prior mean. Conditioning '
        'therefore reduces predictions in correlated warm regions more than at distant '
        'unmeasured cold corners. The optimizer follows those predictions and uncertainty; '
        'it has no built-in chemical rule that low temperature must produce low yield. All '
        'seven exact qLogEI maxima at this second query have true yield 0%. This is not a '
        'mistaken argmax or hidden LLM mean correction. The optimum has acquisition rank3893 '
        'at this prefix. The true yields shown here are evaluator-only audit values.','',
        f"By comparison, bo_s3 starts at {comparator['initial_yield']:.4f}% in a colder region. "
        'Its next maxima are hot/long/high-loading corners. The fixed tie order selects '
        f"p03111 at {comparator['selected_yield']:.4f}% yield. Only one of its seven tied corners "
        'reaches the good-yield threshold. bo_s4 and bo_s6 also select this same high-yield '
        'corner on query2. This documents a role for the fixed tie order; it does not '
        'quantify counterfactual complete trajectories under other tie orders.','',
        '## Prompt and sampling limitations','',
        'The prompt supplies 24 input-only examples, three per catalyst. Six of ten initial '
        'proposals copy a displayed example exactly. Six projected choices are p03260 '
        '(five exact copies plus one rounded nearby proposal), and there are only five '
        'distinct initial conditions. The replies themselves describe the catalyst choice '
        'as arbitrary because catalyst identities are opaque. The experiment evaluates '
        'this particular prompt, example presentation, projection and GP combination. '
        'The observed anchoring does not prove a counterfactual causal presentation effect.','',
        f"The response-weighted LLM SearchAUC8 is {findings['LLM_response_weighted_SearchAUC8']:.4f}. "
        f"Giving its five distinct points equal weight gives {findings['LLM_unique_point_unweighted_SearchAUC8_diagnostic_only']:.4f}. "
        'The latter is a concentration diagnostic, not a corrected headline metric: it '
        'changes the estimand by replacing the observed LLM proposal frequencies. Repeated '
        'points are legitimate outcomes of independent LLM requests, but they give little '
        'coverage of alternative initial designs. Replicate labels 0..9 randomize the '
        'random-start arm; later GP numerical and tie seeds are fixed across all labels.','',
        '## Correct interpretation','',
        'Keep the archived table as the result of this specific pilot. It supports the '
        'statement that these LLM proposals, under this fixed-prior GP and tie sequence, '
        'did not improve early discovery. It does not support an intrinsic disadvantage '
        'of LLM initialization, a general BO degradation after better initial yield, or '
        'an incremental KF benefit. Testing generality would need a separately declared '
        'prompt/presentation and GP/tie sensitivity design. No such new experiment was '
        'run during this audit.','',
        'Evidence: fresh_reconstruction.json, fresh_prefix_checks.csv, '
        'metrics_recomputed.csv, initial_prompt_anchoring.csv, second_query_comparison.csv '
        'and second_query_all_exact_maxima.csv. The standalone calculation is '
        'code/analysis/audit_llm_initialization_reliability.py.']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    note=['# Reliability review of the initialization comparison','',
        'A fresh audit reconstructed 15 distinct starting conditions and all105 query prefixes '
        'with native BoTorch and no old model/score cache used for computation. All selected '
        'points and complete GP score arrays match exactly. The recorded table is reproducible.','',
        'Interpret it as a small prompt-plus-projection-plus-fixed-prior-GP experiment. Ten '
        'LLM responses produced five distinct initial points; six chose the same point, '
        'and six proposals copied displayed examples. The GP uses a 50% prior and fixed '
        'hyperparameters for the first three observations. Early exact ties and initialization '
        'location materially affect its trajectory. A higher first yield does not identify '
        'a more informative first observation for this surrogate.','',
        'The event and pure-qLogEI arms from the same LLM start are identical. These results '
        'do not establish that LLM initialization intrinsically damages BO. All original '
        'outcomes are retained. See the full audit and its independent prefix calculations.']
    (REPORT/'RELIABILITY_REVIEW.md').write_text('\n'.join(note)+'\n',encoding='utf-8')
    index=REPORT/'README.md'
    existing=index.read_text(encoding='utf-8')
    if 'RELIABILITY_REVIEW.md' not in existing:
        existing+='\n## Reliability review\n\n'
        existing+='The [independent reliability review](RELIABILITY_REVIEW.md) confirms numerical '
        existing+='reproduction and limits the initialization claim to this prompt, fixed-prior '
        existing+='GP and tie policy. The five distinct LLM starts and repeated displayed-example '
        existing+='choices should be considered when interpreting the table.\n'
        index.write_text(existing,encoding='utf-8')
    print(json.dumps(findings))


if __name__=='__main__':
    main()
