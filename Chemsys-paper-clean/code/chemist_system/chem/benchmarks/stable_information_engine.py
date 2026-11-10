"""Exact GP ties prioritize new isolated contrasts, not candidate identifiers."""
from dataclasses import replace
import numpy as np

from . import r2_region_trust_engine as fixed
from .suzuki_domain import digest

from .stable_structure_fit import Cache
from . import stable_structure_fit as fit
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
    if state.get('fit_recovery_seen',False):cache.latched=True
    d = fixed.setup(cache, state, iteration)
    ids = cache.pool['ids']
    seed=state.get('tie_seed',0)
    health=cache.last_health
    if health['used_robust']:state['fit_recovery_seen']=True
    use_anchor=False
    information_active=not state.get('fit_recovery_seen',False)
    lookup={k:i for i,k in enumerate(ids)}
    observed=[lookup[h['candidate_id']] for h in state['history']]
    proxy,old_pairs,unseen=information_proxy(cache.X.numpy(),observed)
    if not information_active:proxy=np.zeros(len(ids))
    d['stats']={**d['stats'],'information_proxy':proxy}
    def choose(scores,mask):
        indices=np.flatnonzero(mask);indices=indices[scores[indices]==np.max(scores[indices])]
        if information_active and len(indices)>1:
            indices=indices[proxy[indices]==proxy[indices].max()]
            sd=d['stats']['sd_y'][indices];indices=indices[sd==sd.max()]
        ranks=priority(len(ids),iteration,seed)
        return int(indices[np.argmin(ranks[indices])])
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
        'random_reference':ids[int(ordered(d['base'],d['available'],iteration,seed)[0])],
        'random_normal_proposal':ids[int(ordered(d['score'],d['eligible'],iteration,seed)[0])],
        'actual_reference':ids[ref],'actual_normal_proposal':ids[selected],
        'information_active':information_active,'reference_information_proxy':float(proxy[ref]),
        'normal_information_proxy':float(proxy[selected]),'isolated_pairs':dict(zip(cache.pool['axes'],old_pairs.tolist())),
        'unseen_levels':dict(zip(cache.pool['axes'],unseen.tolist())),
        'secondary_rule':'new one-axis contrasts * log(2+unseen levels)/(1+existing isolated pairs), then latent SD, then seeded random',
        'B4_witness_changed': dg.candidate_index != witness,
    }
    d['GP_health']=health
    if health.get('model_kind')=='molecular_categorical':diag=health['identification']
    else:
        _,value=cache.model(state['history']);diag=fit.diagnose(value['model'],cache.pool['axes'])
    pool=cache.pool;seen={a:{pool['points'][lookup[h['candidate_id']]][a] for h in state['history']} for a in pool['axes']}
    d['model_identifiability']={**diag,'extreme_unsupported':diag['trigger'],
        'unseen_levels':{a:[v for v in pool['levels'][a] if v not in seen[a]] for a in pool['axes']}}
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


def information_proxy(X,observed):
    """A structural identifiability proxy, NOT exact Gaussian mutual information.

    Only equality/inequality of nominal values is used; recoding is invariant.
    A new observation can create isolated pairs with several previous points.
    """
    train=X[observed];old=train[:,None,:]!=train[None,:,:]
    isolated=(old.sum(-1)==1)&np.tril(np.ones(old.shape[:2],dtype=bool),k=-1)
    old_pairs=(old&isolated[...,None]).sum((0,1))
    unseen=np.array([len(set(X[:,j])-set(train[:,j])) for j in range(X.shape[1])])
    diff=X[:,None,:]!=train[None,:,:];new_pairs=(diff&(diff.sum(-1)==1)[...,None]).sum(1)
    proxy=(new_pairs*(np.log(2.+unseen)/(1.+old_pairs))).sum(-1)
    return proxy,old_pairs,unseen


from .stable_productive_engine import event_reasons
