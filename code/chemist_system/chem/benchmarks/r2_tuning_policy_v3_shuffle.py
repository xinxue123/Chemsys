"""V3 counterexamples with random GP ties and shared ranks for equal scores."""
import numpy as np
from . import r2_tuning_policy_v2 as base
from . import r2_tuning_shuffle_engine as ties

SYSTEM = base.SYSTEM
validate = base.validate


def build_candidates(pool, state, d):
    prepared = base.build_candidates(pool, state, d)
    if not prepared['candidates']:
        return prepared
    cfg = base.settings(state)
    iteration = len(state['decisions']) + 1
    lookup = {key: i for i, key in enumerate(pool['ids'])}
    available = ~np.isin(pool['ids'], [h['candidate_id'] for h in state['history']])
    ranks = ties.competition_ranks(d['base'], available)
    priority = ties.priority(len(pool['ids']), iteration)
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
