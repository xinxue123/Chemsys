"""Weighted predictive disagreement and explicit prospective feedback."""
from itertools import combinations
import math
import numpy as np
from scipy.special import logsumexp
from . import PARAMETERS as P
from .geometry import Matches


def normalize(log_weights):
    values = np.asarray(log_weights,float)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite expert log weight')
    return values-logsumexp(values)


def initial_weights(fits):
    values = np.asarray([m['MLL'] for m in fits]); values = P['initial']*(values-values.max())
    return dict(zip((m['id'] for m in fits),normalize(values).tolist()))


def weights_for(fits,arm,stored):
    if arm in ('soft_fixed','feedback'):
        if stored is None:
            stored = initial_weights(fits)
        if set(stored) != {m['id'] for m in fits}:
            raise ValueError('Expert identities changed across feedback')
        logs = normalize([stored[m['id']] for m in fits])
        return np.exp(logs),dict(zip((m['id'] for m in fits),logs.tolist()))
    best = max(m['MLL'] for m in fits)
    allowed = np.asarray([best-m['MLL'] <= P['MLL_gap'] for m in fits])
    return allowed.astype(float)/allowed.sum(),None


def pair_focus(first,second):
    focal = sorted(set(first.get('focus_axes',[])+second.get('focus_axes',[])))
    if focal:
        return focal
    # Local-optimum explanations in G can differ globally. Name one diagnostic
    # axis without requiring a fixed-probe correlation gap.
    return [int(np.argmax(np.abs(np.asarray(first['rho'])-second['rho'])))]


def rank_arrays(pool,history,context,fits,weights,matches=None):
    matches = Matches(pool,history) if matches is None else matches
    n = len(pool.ids)
    means = np.stack([m['mu'] for m in fits]); variance = np.stack([m['sd']**2 for m in fits])
    within = np.maximum(weights@variance,1e-24)
    between = np.zeros(n); raw_between = np.zeros(n); biggest = np.zeros(n)
    pair_index = np.full(n,-1,int); pairs = []
    for i,j in combinations(range(len(fits)),2):
        focal = pair_focus(fits[i],fits[j]); geometry = matches.for_axes(focal)
        term = weights[i]*weights[j]*(means[i]-means[j])**2
        raw_between += term; term = term*geometry['quality']**2; between += term
        take = term > biggest
        biggest[take] = term[take]; pair_index[take] = len(pairs)
        pairs.append(dict(indices=[i,j],focus_axes=focal))
    score = np.sqrt(2*between/within); raw_score = np.sqrt(2*raw_between/within)
    budget = context['available'] & (context['base_logei'][context['bo_index']]-context['base_logei'] <= context['epsilon'])
    qualified = budget & (score >= P['minimum_disagreement'])
    return dict(score=score,raw_score=raw_score,budget=budget,qualified=qualified,
        biggest_pair=pair_index,pairs=pairs,matches=matches,within_variance=within,
        mixture_mean=weights@means)


def choose_index(context,arrays,query):
    js = np.flatnonzero(arrays['qualified'])
    if query == P['queries'] or not len(js):
        return int(context['bo_index'])
    return int(js[np.lexsort((context['tie_priority'][js],-context['base_logei'][js],-arrays['score'][js]))][0])


def select(pool,history,context,fits,arm,stored,query,initial_stored=None):
    weights,logs = weights_for(fits,arm,stored)
    arrays = rank_arrays(pool,history,context,fits,weights)
    selected = choose_index(context,arrays,query); pair_slot = int(arrays['biggest_pair'][selected])
    pair = arrays['pairs'][pair_slot] if pair_slot >= 0 else None
    match = arrays['matches'].describe(pair['focus_axes'],selected) if pair else None
    reason = ('terminal_exact_native_BO' if query == P['queries'] else
        'weighted_kernel_contrast' if arrays['qualified'].any() else 'no_qualified_test_follow_native_BO')
    fixed_logs = initial_stored if initial_stored is not None else initial_weights(fits) if arm=='feedback' else None
    fixed_weights = np.exp(normalize([fixed_logs[m['id']] for m in fits])) if fixed_logs is not None else None
    fixed_arrays = rank_arrays(pool,history,context,fits,fixed_weights,matches=arrays['matches']) if arm=='feedback' else None
    fixed_selected = choose_index(context,fixed_arrays,query) if fixed_arrays is not None else None
    total_models = int(np.count_nonzero(weights > 0))
    commitment = dict(query=query,arm=arm,selected_id=pool.ids[selected],BO_reference_id=pool.ids[context['bo_index']],
        reason=reason,differs_from_BO=bool(selected != context['bo_index']),epsilon=float(context['epsilon']),
        native_logEI_gap=float(context['base_logei'][context['bo_index']]-context['base_logei'][selected]),
        budget_size=int(arrays['budget'].sum()),qualified_size=int(arrays['qualified'].sum()),
        score=float(arrays['score'][selected]),unmatched_score=float(arrays['raw_score'][selected]),
        maximum_budget_score=float(arrays['score'][arrays['budget']].max()),
        matching=match,representative_pair=[fits[i]['id'] for i in pair['indices']] if pair else None,
        experts_with_positive_weight=total_models,effective_expert_count=float(1/(weights@weights)),
        log_weights_before=logs,weights=dict(zip((m['id'] for m in fits),weights.tolist())),
        fixed_initial_selected_id=pool.ids[fixed_selected] if fixed_selected is not None else None,
        feedback_weights_change_selection=bool(arm=='feedback' and query>1 and fixed_selected != selected),
        predictions=[dict(id=m['id'],mu=float(m['mu'][selected]),observation_sd=float(m['observation_sd'][selected]),
            MLL=float(m['MLL']),forecast_sha256=m.get('forecast_sha256')) for m in fits])
    arrays['weights'] = weights
    return commitment,arrays


def feedback(commitment,observation):
    if observation['candidate_id'] != commitment['selected_id']:
        raise ValueError('Outcome differs from precommitted input')
    scores = []
    for m in commitment['predictions']:
        sd = max(m['observation_sd'],1e-12); z = (observation['y']-m['mu'])/sd
        logp = -.5*z*z-math.log(sd)-.5*math.log(2*math.pi)
        scores.append(dict(id=m['id'],innovation_z=float(z),log_predictive_density=float(logp)))
    if commitment['arm'] != 'feedback':
        return dict(scores=scores,log_weights_after=commitment['log_weights_before'],maximum_weight_change=0.)
    before = commitment['log_weights_before']
    logs = normalize([before[m['id']]+m['log_predictive_density'] for m in scores])
    after = dict(zip((m['id'] for m in scores),logs.tolist()))
    change = float(max(abs(math.exp(after[k])-math.exp(before[k])) for k in before))
    return dict(scores=scores,log_weights_after=after,maximum_weight_change=change,
        update='one prospective predictive density per actual observation; current MLL not reused')
