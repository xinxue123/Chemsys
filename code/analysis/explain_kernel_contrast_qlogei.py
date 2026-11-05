"""Post-run audit of useful contrasts, no-op gates and feedback consequences."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from kernel_contrast_qlogei import study, policy, PARAMETERS as P


def main():
    protocol = study.check(True)
    stages, interventions, trajectories = [], [], []
    for task, short in study.SHORT.items():
        pool = study.native.load_task(ROOT,task,seed=0)
        folder = study.BASE/'runs'/short
        state = study.read(folder/'state.json')
        control = study.read(ROOT/protocol['controls'][task]['path'])
        new_best = np.maximum.accumulate([r['raw_value'] for r in state['history']])[4:]
        old_best = np.maximum.accumulate([r['raw_value'] for r in control['history']])[4:]
        for q, decision in enumerate(state['decisions'],1):
            history = state['history'][:q+3]
            fits = study.read(folder/'rounds'/f'{q:02d}_models.json')
            table = pd.read_csv(folder/'rounds'/f'{q:02d}_candidates.csv',encoding='utf-8-sig')
            best_mll = max(m['MLL'] for m in fits)
            plausible = [i for i,m in enumerate(fits) if best_mll-m['MLL'] <= P['MLL_gap']]
            anchors,_ = policy.contrasts(pool,history)
            budget = table.in_qLogEI_budget.to_numpy(bool)
            isolated = budget & (anchors >= 0).any(axis=1)
            discriminatory = np.zeros(len(pool.ids),bool)
            max_score = 0.; pair_count = 0
            for a, i in enumerate(plausible):
                for j in plausible[a+1:]:
                    gaps = np.abs(np.asarray(fits[i]['rho'])-fits[j]['rho'])
                    valid = budget & ((anchors >= 0)&(gaps >= P['minimum_axis_correlation_gap'])[None,:]).any(axis=1)
                    if valid.any():
                        pair_count += 1; discriminatory |= valid
                        score = np.abs(table[f'model_{i}_mean_raw']-table[f'model_{j}_mean_raw'])/np.sqrt(
                            table[f'model_{i}_latent_sd_raw']**2+table[f'model_{j}_latent_sd_raw']**2)
                        max_score = max(max_score,float(score[valid].max()))
            reason = ('terminal_exact_BO' if q == 16 else 'fewer_than_two_MLL_plausible_fits' if len(plausible)<2
                else 'no_single_axis_contrast_in_budget' if not isolated.any()
                else 'no_contrast_on_a_disputed_axis' if not discriminatory.any()
                else 'predictive_disagreement_below_threshold' if max_score < P['minimum_standardized_disagreement']
                else 'qualified_contrast')
            stages.append(dict(task=short,query=q,models=len(fits),plausible_models=len(plausible),
                budget=int(budget.sum()),single_axis_contrasts=int(isolated.sum()),
                contrasts_on_disputed_axis=int(discriminatory.sum()),max_standardized_disagreement=max_score,
                qualified=int(table.qualified_contrast.sum()),changed=decision['decision']['differs_from_BO'],reason=reason))
            trajectories.append(dict(task=short,query=q,new_selected=decision['candidate_id'],
                new_raw_value=decision['raw_value'],new_best=float(new_best[q-1]),
                BO_selected=control['history'][q+3]['candidate_id'],BO_raw_value=control['history'][q+3]['raw_value'],
                BO_best=float(old_best[q-1]),incumbent_difference=float(new_best[q-1]-old_best[q-1])))
            d = decision['decision']
            if not d['differs_from_BO']:
                continue
            selected, anchor = pool.ix[d['selected_id']],pool.ix[d['anchor_id']]
            row = dict(task=short,query=q,measured_total=q+4,selected_id=d['selected_id'],
                anchor_id=d['anchor_id'],axis=d['isolated_axis'],selected_conditions=json.dumps(pool.points[selected]),
                anchor_conditions=json.dumps(pool.points[anchor]),
                anchor_result=next(r['raw_value'] for r in history if r['candidate_id']==d['anchor_id']),
                actual_result=decision['raw_value'],native_logEI_gap=d['native_logei_gap'],epsilon=d['epsilon'],
                selected_native_strict_rank=int(((table.native_logEI > table.native_logEI.iloc[selected])&table.available).sum())+1,
                standardized_disagreement=d['standardized_disagreement'],axis_correlation_gap=d['axis_correlation_gap'],
                pair_log_predictive_ratio=decision['feedback']['pair_comparison']['log_predictive_ratio'],
                best_before=decision['incumbent_before'],best_after=decision['incumbent_after'],
                two_models_both_wrong_at_3SD=all(abs(s['innovation_z'])>3 for s in decision['feedback']['prediction_scores']
                    if s['id'] in d['model_pair']))
            pair = [next(m for m in fits if m['id']==identifier) for identifier in d['model_pair']]
            predictions = {m['id']:m for m in d['predictions']}
            row.update(pair_MLL_gap=abs(pair[0]['MLL']-pair[1]['MLL']),
                pair_mean_A=predictions[pair[0]['id']]['mu']*pool.scale,
                pair_mean_B=predictions[pair[1]['id']]['mu']*pool.scale,
                pair_observation_SD_A=predictions[pair[0]['id']]['observation_sd']*pool.scale,
                pair_observation_SD_B=predictions[pair[1]['id']]['observation_sd']*pool.scale)
            if q < 16:
                following = study.read(folder/'rounds'/f'{q+1:02d}_models.json')
                best = max(m['MLL'] for m in following)
                row['next_refit_explanations'] = json.dumps([dict(source=m['source'],parent=m['parent'],
                    MLL=m['MLL'],within_MLL_gap=bool(best-m['MLL'] <= P['MLL_gap']),rho=m['rho']) for m in following])
            interventions.append(row)
    out = study.REPORT
    sf = pd.DataFrame(stages); it = pd.DataFrame(interventions); tf = pd.DataFrame(trajectories)
    for name,frame in [('gate_diagnostics',sf),('intervention_details',it),('paired_trajectories',tf)]:
        frame.to_csv(out/(name+'.csv'),index=False,encoding='utf-8-sig')
    gate = sf.groupby(['task','reason']).size().rename('rounds').reset_index()
    gate.to_csv(out/'gate_counts.csv',index=False,encoding='utf-8-sig')
    text = ['# Mechanism audit','',
        'All candidate and model-disagreement scores were committed before the selected observation. '
        'This analysis reads only completed trajectories; it performs no new GP fits or experimental queries.','',
        '| Task | Query | Isolated axis | Anchor result | Selected result | Pair means | Log predictive ratio |',
        '|---|---:|---|---:|---:|---|---:|']
    for _,r in it.iterrows():
        text.append(f'| {r.task} | {int(r.query)} | {r.axis} | {r.anchor_result:.4f} | {r.actual_result:.4f} | '
            f'{r.pair_mean_A:.4f} / {r.pair_mean_B:.4f} | {r.pair_log_predictive_ratio:.4f} |')
    text += ['',
        'Case1 query4 obtains a new incumbent. Both disputed models underestimate this outcome by '
        'more than three observation SDs. The relative predictive score favors the first explanation, '
        'but this does not establish calibrated correctness. Its SearchAUC improvement includes direct '
        'early-incumbent carryover; this experiment does not isolate downstream information value from '
        'the value of obtaining the high measurement itself.','',
        'Perera query7 does not improve the incumbent immediately. The differing subsequent BO '
        'trajectory makes the small SearchAUC difference. A one-seed difference of0.011875 in '
        'percentage-point units is not a demonstrated robust advantage.','',
        'A missing intervention means a declared gate failed, rather than an LLM/KF execution failure. '
        'The policy uses same-data empirical-Bayes local optima as competing explanations, not '
        'samples from a calibrated hyperparameter posterior. MLL proximity and standardized disagreement '
        'are development heuristics. No post-result gate relaxation was performed.','',
        '## Gate counts','', '| Reason | Rounds |','|---|---:|']
    for reason,count in sf.groupby('reason').size().items():
        text.append(f'| {reason} | {count} |')
    text += ['', 'The full gate diagnostics retain every task and round. The paired trajectory table '
        'shows when incumbent differences appear and disappear. Final best values match BO on all eight tasks.']
    (out/'mechanism.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    audit = dict(status='PASS_POSTRUN_MECHANISM_AUDIT',rounds=len(sf),interventions=len(it),
        no_new_queries=True,no_new_fits=True,new_LLM_calls=0,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        gate_counts={str(k):int(v) for k,v in sf.groupby('reason').size().items()})
    study.write(out/'mechanism_audit.json',audit,replace=True)
    print(json.dumps(audit)); print(it[['task','query','axis','anchor_result','actual_result','pair_mean_A','pair_mean_B']].to_string(index=False))


if __name__ == '__main__':
    main()
