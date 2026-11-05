"""Read-only scientific diagnosis of archived Case 1 initialization policies.

No optimizer, LLM request, new acquisition calculation or sequential experiment
is run. Unexecuted candidate outcomes are read ONLY by this retrospective
evaluator and never supplied to any decision policy.
"""
from pathlib import Path
from statistics import mean
import csv
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'experiments/reports/llm_start_crossstudy_diagnosis_20261104'
USED = {}


def track(path):
    path = Path(path)
    USED[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read(path):
    return json.loads(track(path).read_text(encoding='utf-8-sig'))


def rows(path):
    with track(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def csv_save(name, records):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader()
        for record in records:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v
                             for k, v in record.items()})


def outcome_summary(state, count):
    y = np.array([o['raw_value'] for o in state['history'][:count]])
    assert len(y) == count
    best = np.maximum.accumulate(y)
    return dict(initial_id=state['history'][0]['candidate_id'], initial_yield=float(y[0]),
                observations=count, SearchAUC=float(best.mean()), final_best=float(best[-1]),
                first95=next((i for i, v in enumerate(best, 1) if v >= THRESHOLD), None),
                yield_sequence=y.tolist())


def main():
    global THRESHOLD
    OUT.mkdir(parents=True, exist_ok=True)
    root = ROOT / 'experiments/runs'
    legacy = root / 'suzuki_initial_yield_20261104'
    fixed = root / 'fast_hit_bo_20261104'
    event = root / 'llm_event_start_qlogei_20261104'
    handoff = root / 'llm_bo_handoff_20261104'
    protocols = [read(p / 'protocol.json') for p in (legacy, fixed, event, handoff)]
    case1_inputs = {k: v for k, v in protocols[0]['inputs_sha256'].items() if '/case1/' in k}
    for p in protocols[1:]:
        assert all(p['inputs_sha256'][k] == v for k, v in case1_inputs.items())
    for path, digest in case1_inputs.items():
        assert hashlib.sha256(track(ROOT / path).read_bytes()).hexdigest() == digest
    pool = read(root / 'suzuki_guarded/inputs/case1/pool.json')
    ix = {cid: j for j, cid in enumerate(pool['ids'])}
    with np.load(track(root / 'suzuki_guarded/inputs/case1/oracle.npz')) as z:
        truth = z['values'].copy() * 100
    THRESHOLD = .95 * float(truth.max())
    optimum = pool['ids'][int(truth.argmax())]

    # Exact same ten physical starts; implementation recipe and budget are
    # separately labelled. The recipe contrast also changes fit/MC seeds and
    # covariance jitter, so it is NOT an isolated prior-mean ablation.
    aligned, group_summary = [], []
    for group in ('low', 'middle'):
        for seed in range(5):
            old = read(legacy / 'runs' / f'c1_s{seed}_{group}' / 'state.json')
            new = read(fixed / 'runs' / f's{seed}_{group}_qlogei' / 'state.json')
            assert old['history'][0] == new['history'][0]
            a, b, c = outcome_summary(old, 8), outcome_summary(old, 20), outcome_summary(new, 8)
            aligned.append(dict(seed=seed, group=group, initial_id=a['initial_id'],
                initial_yield=a['initial_yield'], legacy_SearchAUC8=a['SearchAUC'],
                legacy_best8=a['final_best'], legacy_first95=a['first95'],
                legacy_SearchAUC20=b['SearchAUC'], legacy_best20=b['final_best'],
                fixed_prior_SearchAUC8=c['SearchAUC'], fixed_prior_best8=c['final_best'],
                fixed_prior_first95=c['first95'], recipe_delta_SearchAUC8=c['SearchAUC']-a['SearchAUC'],
                legacy_yield_sequence8=a['yield_sequence'], fixed_prior_yield_sequence8=c['yield_sequence']))
        subset = [r for r in aligned if r['group'] == group]
        group_summary.append(dict(group=group, n=len(subset),
            **{key: mean(r[key] for r in subset) for key in (
                'initial_yield', 'legacy_SearchAUC8', 'legacy_best8', 'legacy_SearchAUC20',
                'legacy_best20', 'fixed_prior_SearchAUC8', 'fixed_prior_best8', 'recipe_delta_SearchAUC8')}))

    current, interventions, skill_examples, target_predictions = [], [], [], []
    all_native_argmaxes = 0
    for seed in range(10):
        for method, folder in [('bo', event/'runs'/f'bo_s{seed}'),
                ('llm_start_bo', event/'runs'/f'llm_start_bo_s{seed}'),
                ('handoff', handoff/'runs'/f'handoff_s{seed}')]:
            state = read(folder/'state.json')
            summary = outcome_summary(state, 8)
            current.append(dict(seed=seed, method=method, **summary))
            for observation in state['history']:
                assert abs(observation['raw_value'] - truth[ix[observation['candidate_id']]]) < 1e-8
            for q in range(2, 9):
                commit = read(folder/'rounds'/f'{q:02}_commit.json')
                decision = commit['decision']
                with np.load(track(folder/'rounds'/f'{q:02}_GP.npz')) as z:
                    score, available, priority = (z[k] for k in ('logei', 'available', 'priorities'))
                    top = np.flatnonzero(available & (score == score[available].max()))
                    bo = int(top[np.argmin(priority[top])])
                    assert pool['ids'][bo] == decision['BO_reference_id']
                    all_native_argmaxes += 1
                    j = ix[decision['selected_id']]
                    if seed == 0 and method in ('llm_start_bo', 'handoff'):
                        k = ix[optimum]
                        target_predictions.append(dict(method=method, query=q,
                            selected_id=pool['ids'][j], selected_yield=float(truth[j]),
                            target_id=optimum, target_true_yield=float(truth[k]),
                            target_GP_mean=float(z['mu'][k])*100,
                            target_GP_sd=float(np.sqrt(z['variance'][k]))*100,
                            target_acquisition_rank=1+int((available & (score > score[k])).sum()),
                            native_top_ties=len(top)))
                    if method == 'handoff' and j != bo:
                        incumbent=max(h['raw_value'] for h in state['history'][:q-1])
                        interventions.append(dict(seed=seed, query=q, selected_id=pool['ids'][j],
                            selected_yield=float(truth[j]), same_prefix_BO_id=pool['ids'][bo],
                            same_prefix_BO_yield=float(truth[bo]),
                            immediate_difference=float(truth[j]-truth[bo]),
                            improved_incumbent=bool(truth[j] > incumbent), incumbent=incumbent,
                            selected_conditions=pool['points'][j], BO_conditions=pool['points'][bo],
                            retained_exp_qLogEI=float(np.exp(score[j]-score[bo])),
                            budget_candidates=decision['budget_candidates'],
                            eligible_candidates=decision['eligible_candidates'],
                            note_diagnostics=decision['note_diagnostics']))
                if method == 'handoff' and seed in (0, 1) and q == 2:
                    feedback=read(folder/'rounds'/f'{q:02}_feedback.json')
                    skill_examples.extend(dict(seed=seed, **f) for f in feedback['events'])

    # Load fitted parameters, never optimize or calculate another acquisition.
    import torch
    torch.set_num_threads(1)
    sys.path[:0] = [str(ROOT/'code/chemist_system'), str(ROOT/'code/experiment_tools')]
    from fast_hit_bo.model import make_model
    model_file = ROOT/'code/experiment_tools/fast_hit_bo/model.py'
    track(model_file)
    parameters = []
    for method, folder in [('bo', event/'runs/bo_s3'),
            ('llm_start_bo', event/'runs/llm_start_bo_s0'), ('handoff', handoff/'runs/handoff_s0')]:
        d = read(folder/'rounds/08_commit.json')['decision']
        ck = torch.load(track(ROOT/d['checkpoint']), map_location='cpu', weights_only=False)
        model = make_model(torch.tensor(ck['train_X'], dtype=torch.double),
                           torch.tensor(ck['train_Y'], dtype=torch.double))
        model.load_state_dict(ck['state'])
        additive, interaction = model.covar_module.kernels
        ell = float(interaction.base_kernel.kernels[1].lengthscale.item())
        parameters.append(dict(method=method, query=8,
            additive_category_scale=float(additive.base_kernel.kernels[1].outputscale.item()),
            interaction_category_lengthscale=ell,
            interaction_correlation_between_distinct_categories=float(np.exp(-1/ell))))

    paired = []
    for seed in range(10):
        a = next(r for r in current if r['method']=='llm_start_bo' and r['seed']==seed)
        b = next(r for r in current if r['method']=='handoff' and r['seed']==seed)
        assert a['initial_id']==b['initial_id'] and a['initial_yield']==b['initial_yield']
        paired.append(dict(seed=seed, initial_id=a['initial_id'],
            delta_SearchAUC8=b['SearchAUC']-a['SearchAUC'], delta_final=b['final_best']-a['final_best']))
    method_summary = []
    for method in ('bo', 'llm_start_bo', 'handoff'):
        subset=[r for r in current if r['method']==method]
        method_summary.append(dict(method=method, n=len(subset),
            distinct_initials=len({r['initial_id'] for r in subset}),
            mean_initial=mean(r['initial_yield'] for r in subset),
            SearchAUC8=mean(r['SearchAUC'] for r in subset),
            final_best=mean(r['final_best'] for r in subset),
            target_hits=sum(r['first95'] is not None for r in subset)))
    summary = dict(status='PASS', new_LLM_calls=0, new_sequential_experiments=0,
        new_GP_fits=0, new_acquisition_calculations=0,
        unexecuted_outcomes_used_only_by_retrospective_evaluator=True,
        candidate_inputs_and_outcomes_same_across_studies=True, native_argmaxes_checked=all_native_argmaxes,
        changed_actions=len(interventions),
        immediate_yield_better=sum(r['immediate_difference']>1e-9 for r in interventions),
        immediate_yield_worse=sum(r['immediate_difference']<-1e-9 for r in interventions),
        improved_own_incumbent=sum(r['improved_incumbent'] for r in interventions),
        postinitial_SearchAUC8_wins=sum(r['delta_SearchAUC8']>1e-9 for r in paired),
        postinitial_SearchAUC8_losses=sum(r['delta_SearchAUC8']<-1e-9 for r in paired),
        postinitial_SearchAUC8_ties=sum(abs(r['delta_SearchAUC8'])<=1e-9 for r in paired),
        paired_SearchAUC8_increment=mean(r['delta_SearchAUC8'] for r in paired),
        paired_final_increment=mean(r['delta_final'] for r in paired),
        initialization_concentration=read(event/'runs/event_s0/state.json')['history'][0]['candidate_id'],
        current_methods=method_summary, aligned_old_initializations=group_summary)
    csv_save('same_start_recipe_comparison.csv', aligned)
    csv_save('same_prefix_interventions.csv', interventions)
    csv_save('current_paired.csv', paired)
    csv_save('current_metrics.csv', current)
    csv_save('optimum_predictions_seed0.csv', target_predictions)
    csv_save('loaded_category_parameters.csv', parameters)
    (OUT/'KF_examples.json').write_text(json.dumps(skill_examples, indent=2)+'\n', encoding='utf-8')
    for path, digest in USED.items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest, path
    track(Path(__file__))
    (OUT/'source_hashes.json').write_text(json.dumps(USED, indent=2)+'\n', encoding='utf-8')
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
