"""Input-only contrast construction and pre-outcome decision rules."""
import math
import numpy as np
from . import PARAMETERS as P


def contrasts(pool, history):
    n, d = len(pool.ids), len(pool.axes)
    anchors = np.full((n, d), -1, dtype=int)
    distances = np.full((n, d), np.inf)
    for observed in history:
        anchor = pool.ix[observed['candidate_id']]
        changed = np.column_stack([
            np.abs(np.asarray([f[a] for f in pool.features], float)-pool.features[anchor][a]) > P['continuous_tolerance']
            if kind == 'continuous' else
            np.asarray([f[a] != pool.features[anchor][a] for f in pool.features])
            for a, kind in enumerate(pool.kinds)])
        isolated = changed.sum(axis=1) == 1
        for a, kind in enumerate(pool.kinds):
            js = np.flatnonzero(isolated & changed[:, a])
            if kind == 'continuous':
                delta = np.abs(np.asarray([pool.features[j][a] for j in js])-pool.features[anchor][a])
                take = delta <= distances[js, a]
                js, delta = js[take], delta[take]
                anchors[js, a], distances[js, a] = anchor, delta
            else:
                anchors[js, a], distances[js, a] = anchor, 1.
    return anchors, distances


def select(pool, history, context, models, query):
    """No oracle or retrospective labels are accepted by this interface."""
    available = context['available']
    base, logei, ties = context['bo_index'], context['base_logei'], context['tie_priority']
    epsilon = P['maximum_logei_gap']*(P['queries']-query)/(P['queries']-1)
    budget = available & (logei[base]-logei <= epsilon)
    anchors, distances = contrasts(pool, history)
    best_mll = max(m['MLL'] for m in models)
    plausible = [m for m in models if best_mll-m['MLL'] <= P['MLL_gap']]
    score = np.zeros(len(pool.ids)); axes = np.full(len(pool.ids), -1, dtype=int)
    pair = [None]*len(pool.ids)
    axis_gaps = np.zeros(len(pool.ids))
    for a, first in enumerate(plausible):
        for second in plausible[a+1:]:
            gap = np.abs(np.asarray(first['rho'])-np.asarray(second['rho']))
            allowed = (anchors >= 0) & (gap >= P['minimum_axis_correlation_gap'])[None, :]
            axis = np.argmax(np.where(allowed, gap, -1.), axis=1)
            valid = budget & allowed.any(axis=1)
            denom = np.sqrt(np.maximum(first['sd']**2+second['sd']**2, 1e-24))
            disagreement = np.abs(first['mu']-second['mu'])/denom
            improve = valid & (disagreement >= P['minimum_standardized_disagreement']) & (disagreement > score)
            for j in np.flatnonzero(improve):
                score[j], axes[j] = float(disagreement[j]), int(axis[j])
                pair[j] = [first['id'], second['id']]
                axis_gaps[j] = float(gap[axis[j]])
    qualified = budget & (score >= P['minimum_standardized_disagreement'])
    if query == P['queries']:
        selected, reason = int(base), 'terminal_exact_native_qLogEI'
    elif qualified.any():
        js = np.flatnonzero(qualified)
        selected = int(js[np.lexsort((ties[js], -logei[js], -score[js]))][0])
        reason = 'isolated_contrast_discriminates_plausible_kernels'
    else:
        selected, reason = int(base), 'no_qualified_contrast_follow_native_qLogEI'
    axis = int(axes[selected])
    anchor = int(anchors[selected, axis]) if axis >= 0 else None
    commit = dict(selected_id=pool.ids[selected], BO_reference_id=pool.ids[int(base)],
        reason=reason, query=query, epsilon=float(epsilon),
        native_logei_gap=float(logei[base]-logei[selected]), budget_size=int(budget.sum()),
        qualified_size=int(qualified.sum()), plausible_models=[m['id'] for m in plausible],
        isolated_axis=pool.axes[axis] if axis >= 0 else None,
        anchor_id=pool.ids[anchor] if anchor is not None else None,
        model_pair=pair[selected], standardized_disagreement=float(score[selected]),
        axis_correlation_gap=float(axis_gaps[selected]),
        differs_from_BO=bool(selected != base),
        predictions=[dict(id=m['id'], source=m['source'], mu=float(m['mu'][selected]),
            observation_sd=float(m['observation_sd'][selected]), MLL=float(m['MLL'])) for m in models])
    arrays = dict(budget=budget, qualified=qualified, score=score, chosen_axis=axes,
        anchors=anchors, anchor_distances=distances, axis_correlation_gap=axis_gaps)
    return commit, arrays


def feedback(commit, observation, history):
    if observation['candidate_id'] != commit['selected_id']:
        raise ValueError('Outcome does not match precommitted selection')
    y = float(observation['y']); scores = []
    for model in commit['predictions']:
        sd = max(model['observation_sd'], 1e-12)
        z = (y-model['mu'])/sd
        scores.append(dict(id=model['id'], source=model['source'], innovation_z=z,
            log_predictive_density=-math.log(sd)-0.5*math.log(2*math.pi)-0.5*z*z))
    pair = commit['model_pair']; comparison = None
    if pair:
        lookup = {r['id']:r['log_predictive_density'] for r in scores}
        llr = lookup[pair[0]]-lookup[pair[1]]
        comparison = dict(pair=pair, log_predictive_ratio=float(llr),
            preferred=pair[0] if llr > 0 else pair[1] if llr < 0 else None,
            interpretation='single-observation predictive evidence; not proof of model truth')
    anchor = next((r for r in history if r['candidate_id'] == commit['anchor_id']), None)
    return dict(prediction_scores=scores, pair_comparison=comparison,
        anchor_id=commit['anchor_id'], axis=commit['isolated_axis'],
        observed_contrast_y=y-anchor['y'] if anchor is not None else None)
