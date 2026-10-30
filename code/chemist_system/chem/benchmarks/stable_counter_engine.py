"""V3 tie-order sensitivity: local random priorities, frozen GP/KF/B4 math."""
from dataclasses import replace
import numpy as np

from . import r2_region_trust_engine as fixed
from .suzuki_domain import digest

from .stable_r2_fit import Cache
original = fixed.original
feedback = fixed.feedback
event_reasons = fixed.event_reasons
TIE_SEED = 0


def priority(n, iteration, seed=0):
    """Independent, stateless RNG; no IDs/outcomes or global RNG consumption."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(iteration)]))
    order = rng.permutation(n)
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    return rank


def ordered(scores, mask, iteration, seed=0):
    scores = np.asarray(scores)
    indices = np.flatnonzero(mask)
    ranks = priority(len(scores), iteration, seed)
    # Exact floating point equality defines ties. Distinct scores never cross.
    return indices[np.lexsort((ranks[indices], -scores[indices]))]


def competition_ranks(scores, mask):
    scores = np.asarray(scores)
    values = np.sort(scores[mask])
    return 1 + len(values) - np.searchsorted(values, scores, side='right')


def setup(cache, state, iteration):
    d = fixed.setup(cache, state, iteration)
    ids = cache.pool['ids']
    seed=state.get('tie_seed',0)
    health=cache.last_health
    if health['used_robust']:state['fit_recovery_seen']=True
    use_anchor=state.get('tie_rule')=='anchor_on_recovery' and state.get('fit_recovery_seen',False)
    def choose(scores,mask):
        if not use_anchor:return int(ordered(scores,mask,iteration,seed)[0])
        indices=np.flatnonzero(mask);maxscore=np.max(scores[indices]);indices=indices[scores[indices]==maxscore]
        anchor=max(state['history'],key=lambda h:h['y'])['candidate_id'];lookup={k:i for i,k in enumerate(ids)}
        point=cache.pool['points'][lookup[anchor]]
        seen={a:{cache.pool['points'][lookup[h['candidate_id']]][a] for h in state['history']} for a in cache.pool['axes']}
        ranks=priority(len(ids),iteration,seed)
        def key(j):
            edits=sum(cache.pool['points'][j][a]!=point[a] for a in cache.pool['axes'])
            novelty=sum(cache.pool['points'][j][a] not in seen[a] for a in cache.pool['axes'])
            return (edits,-float(d['stats']['sd_y'][j]),-novelty,ranks[j])
        return int(min(indices,key=key))
    old_ref, old_selected, dg = d['reference'], d['selected'], d['diagnosis']
    ref = choose(d['base'],d['available'])
    selected = choose(d['score'],d['eligible'])
    # Only change B4's equal-score witness, retaining its exact subset policy.
    membership = original.nominal.membership(cache.pool, state['knowledge'])
    positions = [i for i, h in enumerate(state['knowledge']['clauses']) if h['id'] in dg.retained_ids]
    feasible = d['eligible'] & (membership[positions].all(axis=0) if positions else True)
    witness = choose(d['base'],feasible)
    assert d['base'][ref] == d['base'][old_ref]
    assert d['score'][selected] == d['score'][old_selected]
    assert d['base'][witness] == d['base'][dg.candidate_index]
    d['reference'], d['selected'] = ref, selected
    d['diagnosis'] = replace(dg, candidate_index=witness, candidate_id=ids[witness])
    assert not d['fallback'] or ref == selected
    ranks = priority(len(ids), iteration, seed)
    d['tie_order'] = {
        'seed':seed,'iteration':iteration,'RNG':f'NumPy PCG64 / SeedSequence([{seed}, query_round])','anchor_active':use_anchor,
        'equality': 'exact float equality; no near-tie tolerance',
        'priority_sha256': digest(ranks.tolist()),
        'base_max_ties': int(np.sum(d['available'] & (d['base'] == d['base'][ref]))),
        'guided_max_ties': int(np.sum(d['eligible'] & (d['score'] == d['score'][selected]))),
        'ID_order_reference': ids[old_ref], 'ID_order_normal_proposal': ids[old_selected],
        'random_reference': ids[ref], 'random_normal_proposal': ids[selected],
        'B4_witness_changed': dg.candidate_index != witness,
    }
    d['GP_health']=health
    _,value=cache.model(state['history']);model=value['model'];X=model.train_inputs[0].detach().numpy()
    ls=model.covar_module.base_kernel.lengthscale.detach().reshape(-1).numpy();axes=cache.pool['axes']
    pairs={a:sum([axes[k] for k in range(len(axes)) if X[u,k]!=X[v,k]]==[a] for u in range(len(X)) for v in range(u)) for a in axes}
    seen={a:{p[a] for p in [cache.pool['points'][ids.index(h['candidate_id'])] for h in state['history']]} for a in axes}
    critical=axes[int(np.argmin(ls))]
    d['model_identifiability']={'lengthscales':dict(zip(axes,ls.tolist())),'isolated_pairs':pairs,'critical_axis':critical,
        'unseen_levels':{a:[v for v in cache.pool['levels'][a] if v not in seen[a]] for a in axes},
        'extreme_unsupported':bool(ls.min()<=1e-4 and ls.max()/ls.min()>=1e6 and pairs[critical]==0)}
    return d


def summary(cache, state):
    """Same GP-summary schema/top16; break tied cutoff using the same priority."""
    pool = cache.pool
    base, _, _ = cache.score(state['history'])
    stats = cache.stats(state['history'])
    mask = ~np.isin(pool['ids'], [h['candidate_id'] for h in state['history']])
    indices = ordered(base, mask, len(state['decisions']) + 1,state.get('tie_seed',0))[:16]
    from . import pointwise_hte_llm as llm
    return [{'point': llm.nominal.public_point(pool, pool['points'][j]),
             'GP_mean_percent': round(float(stats['mu_y'][j]) * 100, 3),
             'GP_latent_SD_pp': round(float(stats['sd_y'][j]) * 100, 3),
             'base_qLogEI': float(base[j])} for j in indices]


_original_event_reasons=event_reasons
def event_reasons(state,d,t):
    reasons=_original_event_reasons(state,d,t)
    recent=next((c for c in state['recombination_calls'] if c['iteration']==t-1),None)
    if recent and len(state['calls'])<3 and not any(c['iteration']==t for c in state['calls']):
        plan=recent['validated']['plan']
        if plan['violated_preferences'] and recent['observed_percent']>=recent['incumbent_before_percent']+.5:
            return list(dict.fromkeys(reasons+['successful_joint_counterexample']))
    return reasons
