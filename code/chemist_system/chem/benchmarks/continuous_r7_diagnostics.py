"""R7-D0: read-only R6 reachability audit, recomputing ONLY prequery GP beliefs.

No future outcome or oracle is passed to a policy decision. Existing selected
outcomes may be inspected after diagnostics for an explicitly POSTHOC score.
Never modifies R6 runs, source files, or any R6 cached GP fit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import continuous_r6_core as old_core
from . import continuous_r6_engine as r6
from . import continuous_r6_study as old_study
from . import continuous_r7_core as core
from . import continuous_r7_engine as r7


def inspect_round(cache, context, history, commitment, observation=None):
    """Pure diagnostic function; history contains only PRE-ROUND observations."""
    gate = context['gate']
    assert r6.digest(history) == commitment['history_sha256']
    assert gate.bo_point_id == commitment['BO_reference_id'], 'original BO decision drift'
    results = []
    for h in commitment['hypotheses_at_commit']:
        diag = r7.diagnose_hypothesis(cache, context, history, h)
        diag['selected_point_id'] = commitment['selected_id']
        diag['R6_selection_mode'] = commitment['selection_mode']
        if observation is not None:
            # This result is a retrospectively evaluated *precommitted* score.
            # Do not use it to select or calibrate any step in this R6 path.
            assert observation['candidate_id'] == commitment['selected_id']
            j = cache.ix[commitment['selected_id']]
            d = context['d']
            md, vd, cov = r6.contrast_moments(cache, d['stats'], h, [j])
            shadow = r6._shadow_for(cache, d['stats'], h, j,
                                    commitment['iteration'], (md, vd, cov[0]))
            diag['selected_rho_squared'] = core.squared_correlation(
                shadow.gp_var_y, shadow.contrast_var, shadow.cov_y_contrast,
                shadow.preference_noise_sd)
            diag['selected_binary_sign_information_nats'] = core.sign_information_for_shadow(shadow)
            diag['posthoc_prequery_log_score_gain'] = old_core.prospective_log_score_gain(
                shadow, float(observation['y']))
        results.append(diag)
    return {'round': commitment['iteration'], 'BO_reference_id': gate.bo_point_id,
            'selected_id': commitment['selected_id'], 'original_mode': commitment['selection_mode'],
            'eligible_count': len(gate.eligible_point_ids), 'hypotheses': results}


def run(seed=0, arms=('A4', 'A4_no_exception'), r6_root=None, output=None):
    from . import continuous_benchmark_data as data
    from . import continuous_benchmark_engine as original
    from . import continuous_r7_study as study

    archive = Path(r6_root) if r6_root is not None else old_study.ROOT
    path = Path(output) if output is not None else study.OUT / f'r6_reachability_seed{seed:03d}.json'
    source = []
    for task in old_study.TASKS:
        for arm in arms:
            state_file = archive / 'runs' / f'{task}_{arm}_seed{seed:03d}' / 'state.json'
            if state_file.is_file():
                source.append(state_file)
    if not source:
        return {'status': 'MISSING_R6_RUNS', 'expected_root': str(archive),
                'message': 'Provide the original R6 run root, without moving or editing any archived files.'}
    missing_pools = [str(data.ROOT / task / 'pool.json') for task in old_study.TASKS
                     if not (data.ROOT / task / 'pool.json').is_file()]
    if missing_pools:
        return {'status': 'MISSING_DATA', 'missing_pools': missing_pools,
                'message': 'Original input pools are required, but oracle.npz is not read by diagnostics.'}
    original.warm()
    out = []
    for state_file in sorted(source):
        state = data.read(state_file)
        pool = data.read(data.ROOT / state['task'] / 'pool.json')

        class DiagnosticCache(original.Cache):
            def __init__(self, full_pool):
                super().__init__(full_pool)
                self.directory = study.ROOT / 'diagnostic_cache' / full_pool['task']
                self.directory.mkdir(parents=True, exist_ok=True)

        cache = DiagnosticCache(pool)
        rounds = []
        try:
            for t in range(1, len(state['decisions']) + 1):
                cp = state_file.parent / 'rounds' / f'{t:02d}_committed.json'
                commitment = data.read(cp)
                prefix = state['history'][:t + 3]
                observation = state['history'][t + 3]
                assert commitment['iteration'] == t
                assert commitment['selected_id'] == observation['candidate_id']
                prefix_state = dict(state, arm='A4', history=prefix,
                                    hypotheses=commitment['hypotheses_at_commit'],
                                    calls=commitment['calls_at_commit'])
                context = r7.build_context(cache, dict(prefix_state, arm='B2'), t)
                rounds.append(inspect_round(cache, context, prefix, commitment, observation))
                cache.release()
        finally:
            cache.release()
        out.append({'task': state['task'], 'arm': state['arm'], 'seed': state['seed'],
                    'archived_state_file': str(state_file), 'rounds': rounds})

    summary = []
    for row in out:
        hs = [h for rnd in row['rounds'] for h in rnd['hypotheses']]
        summary.append({'task': row['task'], 'arm': row['arm'], 'seed': row['seed'],
                        'rounds': len(row['rounds']), 'hypothesis_rounds': len(hs),
                        'geom_reachable': sum(h['geometric_eligible'] > 0 for h in hs),
                        'covariance_relevant': sum(h['covariance_relevant_eligible'] > 0 for h in hs),
                        'possible_cap_flips': sum(h['cap_flip'] for h in hs),
                        'possible_actual_weight_flips': sum(h['actual_flip'] for h in hs),
                        'counterfactual_scores_positive': sum(
                            h.get('posthoc_prequery_log_score_gain', 0.) > 0 for h in hs)})
    result = {'status': 'PASS', 'kind': 'R7_D0_DIAGNOSTIC_ONLY',
              'does_not_estimate_counterfactual_trajectory_AUC': True,
              'does_not_mutate_R6': True, 'summary': summary, 'traces': out}
    data.write(path, result)
    return {'status': 'PASS', 'output': str(path), 'summary': summary}
