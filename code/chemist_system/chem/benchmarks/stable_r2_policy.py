"""V3 counterexamples with random GP ties and shared ranks for equal scores."""
import numpy as np
from . import r2_tuning_policy_v2 as base
from . import stable_r2_engine as ties

SYSTEM = base.SYSTEM
validate = base.validate


def build_candidates(pool, state, d):
    prepared = focused_candidates(pool, state, d)
    if not prepared['candidates']:
        return prepared
    cfg = settings(state)
    iteration = len(state['decisions']) + 1
    lookup = {key: i for i, key in enumerate(pool['ids'])}
    available = ~np.isin(pool['ids'], [h['candidate_id'] for h in state['history']])
    ranks = ties.competition_ranks(d['base'], available)
    priority = ties.priority(len(pool['ids']), iteration,state.get('tie_seed',0))
    for candidate in prepared['candidates']:
        candidate['base_rank'] = int(ranks[lookup[candidate['candidate_id']]])
    rows = prepared['candidates']
    if state['validations']:
        prepared['counterexample_pool_before_shortlist'] = len(rows)
        rows = sorted(rows, key=lambda c: (-c['base_qLogEI'], priority[lookup[c['candidate_id']]]))
        prepared['candidates'] = rows[:cfg['counterexample_shortlist_size']]
        prepared['shortlist_rule'] = 'Top base qLogEI; exact-score ties use independent seed0 round-specific random priority, never candidate-ID order.'
    else:
        # Keep first test's full pool/non-tie presentation positions unchanged.
        groups = {}
        for i, candidate in enumerate(rows):
            groups.setdefault(candidate['base_qLogEI'], []).append(i)
        rows = list(rows)
        for positions in groups.values():
            shuffled = sorted((rows[i] for i in positions), key=lambda c: priority[lookup[c['candidate_id']]])
            for i, candidate in zip(positions, shuffled):
                rows[i] = candidate
        prepared['candidates'] = rows
    prepared['tie_order_rule'] = 'Seed0 stateless priorities by query round; equal GP scores share competition rank. No future outcomes or ID ranks.'
    return prepared


def settings(state):
    from pathlib import Path
    from . import reasoning_bo_data as data
    return data.read(Path('research/stable_r2_20261029/plans')/(state['variant']+'.json'))


def focused_candidates(pool,state,d):
    cfg=settings(state);prepared=base.base.build_candidates(pool,state,d)
    if max(h['y'] for h in state['history'])*100>=cfg['high_yield_threshold_percent']:
        prepared['candidates']=[];prepared['validation_skip_reason']='Observed incumbent already meets the predeclared high-yield threshold.'
        return prepared
    if not state['validations'] or not prepared['candidates']:return prepared
    lookup={k:i for i,k in enumerate(pool['ids'])}
    seen={a:{pool['points'][lookup[h['candidate_id']]][a] for h in state['history']} for a in pool['axes']}
    groups={}
    for c in prepared['candidates']:
        axis=c['changed_axis'];value=pool['points'][lookup[c['candidate_id']]][axis]
        if value not in seen[axis]:groups.setdefault(axis,[]).append(c)
    if groups:
        axis=min(groups,key=lambda a:(-len(groups[a]),-max(c['GP_latent_SD_pp'] for c in groups[a]),a))
        prepared['candidates']=groups[axis]
        prepared['coverage_focus']={'axis':axis,'unseen_counterexample_levels':{a:len(rs) for a,rs in groups.items()},
            'criterion':'Most unobserved excluded levels; GP uncertainty then axis name break ties. No unseen yields used.'}
    return prepared
