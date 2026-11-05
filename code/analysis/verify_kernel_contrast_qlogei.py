"""Replay every decision and pre-outcome commitment without refitting."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'code/chemist_system'),str(ROOT/'code/experiment_tools')]
from kernel_contrast_qlogei import study, models, policy


def main():
    p = study.check(True)
    decisions = feedbacks = unchanged_files = fit_files = warning_count = 0
    maximum_score_error = 0.
    for task,short in study.SHORT.items():
        pool = study.native.load_task(ROOT,task,seed=0)
        folder = study.BASE/'runs'/short
        final = study.read(folder/'state.json')
        state = dict(task=task,protocol_sha256=p['protocol_sha256'],status='RUNNING',
            history=deepcopy(final['history'][:4]),decisions=[],parents=[])
        for q, saved in enumerate(final['decisions'],1):
            cp = folder/'rounds'/f'{q:02d}_commit.json'
            commit = study.read(cp)
            assert commit['commit_sha256'] == models.digest({k:v for k,v in commit.items() if k != 'commit_sha256'})
            assert commit['prestate_sha256'] == models.digest(state)
            assert commit['counters']['operator_compiles_in_loop'] == 0
            fits = study.read(folder/'rounds'/f'{q:02d}_models.json')
            csv = pd.read_csv(folder/'rounds'/f'{q:02d}_candidates.csv',encoding='utf-8-sig',float_precision='round_trip')
            assert list(csv.candidate_id) == list(pool.ids)
            with np.load(folder/'rounds'/f'{q:02d}_GP.npz',allow_pickle=False) as z:
                context = {k:z[k].copy() for k in ('base_logei','available','tie_priority')}
                available = np.flatnonzero(context['available'])
                high = available[context['base_logei'][available] == context['base_logei'][available].max()]
                context['bo_index'] = int(high[np.argmin(context['tie_priority'][high])])
                for i,m in enumerate(fits):
                    m.update(mu=csv[f'model_{i}_mean_raw'].to_numpy()/pool.scale,
                        sd=csv[f'model_{i}_latent_sd_raw'].to_numpy()/pool.scale,
                        observation_sd=csv[f'model_{i}_observation_sd_raw'].to_numpy()/pool.scale)
                    warning_count += len(m['optimizer_warnings'])
                decision,arrays = policy.select(pool,state['history'],context,fits,q)
                np.testing.assert_array_equal(arrays['budget'],z['budget'])
                np.testing.assert_array_equal(arrays['qualified'],z['qualified'])
                maximum_score_error = max(maximum_score_error,float(np.max(np.abs(arrays['score']-z['disagreement']))))
            for key in ('selected_id','BO_reference_id','reason','isolated_axis','anchor_id','model_pair','plausible_models'):
                assert decision[key] == commit['decision'][key],(short,q,key)
            for key in ('standardized_disagreement','axis_correlation_gap','native_logei_gap','epsilon'):
                assert abs(decision[key]-commit['decision'][key]) < 1e-10
            observation = final['history'][q+3]
            fb = policy.feedback(commit['decision'],observation,state['history'])
            assert fb == saved['feedback']
            receipt = study.read(folder/'rounds'/f'{q:02d}_feedback.json')
            assert receipt == dict(observation=observation,feedback=fb,commit_sha256=commit['commit_sha256'])
            assert cp.stat().st_mtime_ns <= (folder/'rounds'/f'{q:02d}_feedback.json').stat().st_mtime_ns
            state['history'].append(deepcopy(observation)); state['decisions'].append(deepcopy(saved))
            state['parents'] = commit['next_parents']; state['status'] = 'COMPLETE' if q == 16 else 'RUNNING'
            decisions += 1; feedbacks += 1
        assert state == final
        assert len({h['candidate_id'] for h in final['history']}) == 20
        fit_files += len(list((folder/'alternatives').glob('*.pt')))
    # Recheck the old study's source and diagnostic checkpoint custody.
    previous = ROOT/'experiments/reports/v13_kernel_conditions_20261104'
    for name in ('profile_source_hashes.json','contrast_source_hashes.json','fitted_checkpoint_hashes.json'):
        for rel,expected in study.read(previous/name).items():
            assert study.sha(ROOT/rel) == expected,rel
            unchanged_files += 1
    metric = pd.read_csv(study.REPORT/'results.csv',encoding='utf-8-sig')
    for task,short in study.SHORT.items():
        state = study.read(study.BASE/'runs'/short/'state.json')
        expected = np.maximum.accumulate([h['raw_value'] for h in state['history']])[4:].mean()
        got = metric[(metric.task==short)&(metric.method=='kernel_contrast')].iloc[0].SearchAUC
        assert abs(expected-got) < 1e-9
    sources = list((ROOT/'code/experiment_tools/kernel_contrast_qlogei').glob('*.py'))+[
        ROOT/'code/experiment_tools/run_kernel_contrast_qlogei.py',Path(__file__),
        ROOT/'code/analysis/explain_kernel_contrast_qlogei.py',study.DESIGN]
    study.write(study.REPORT/'final_source_hashes.json',
        {f.relative_to(ROOT).as_posix():study.sha(f) for f in sources},replace=True)
    audit = dict(status='PASS_FULL_KERNEL_CONTRAST_VERIFICATION',decision_replays=decisions,
        committed_feedback_replays=feedbacks,unique_observations_per_task=20,
        maximum_disagreement_replay_error=maximum_score_error,
        historical_file_hashes_verified=unchanged_files+len(p['protected_sha256']),
        alternative_fit_checkpoints=fit_files,optimizer_warning_messages_retained=warning_count,
        repeated_operator_compilations=0,new_LLM_calls=0,
        SearchAUC_denominator=16,SearchAUC_recomputed=True,all_final_states_reconstructed=True)
    assert decisions == feedbacks == 128 and maximum_score_error < 1e-10
    study.write(study.REPORT/'final_verification.json',audit,replace=True)
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
