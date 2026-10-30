"""Preserve complete acquisition tie classes, not arbitrary top-k members."""
import numpy as np
from . import stable_r2_policy as base
from . import stable_r2_engine as ties
SYSTEM=base.SYSTEM
validate=base.validate


def build_candidates(pool,state,d):
    prepared=base.focused_candidates(pool,state,d)
    if not prepared['candidates']:return prepared
    cfg=base.settings(state);lookup={k:i for i,k in enumerate(pool['ids'])};t=len(state['decisions'])+1
    available=~np.isin(pool['ids'],[h['candidate_id'] for h in state['history']])
    ranks=ties.competition_ranks(d['base'],available);priority=ties.priority(len(pool['ids']),t,state['tie_seed'])
    rows=prepared['candidates']
    for c in rows:c['base_rank']=int(ranks[lookup[c['candidate_id']]])
    if state['validations']:
        prepared['counterexample_pool_before_shortlist']=len(rows)
        rows=sorted(rows,key=lambda c:(-c['base_qLogEI'],priority[lookup[c['candidate_id']]]))
        k=min(cfg['counterexample_shortlist_size'],len(rows));cut=rows[k-1]['base_qLogEI']
        prepared['candidates']=[c for c in rows if c['base_qLogEI']>=cut]
        prepared['shortlist_rule']='Top5 including the complete exact-score tie class at the cutoff; no ID-priority truncation.'
    else:
        groups={}
        for i,c in enumerate(rows):groups.setdefault(c['base_qLogEI'],[]).append(i)
        rows=list(rows)
        for positions in groups.values():
            shuffled=sorted((rows[i] for i in positions),key=lambda c:priority[lookup[c['candidate_id']]])
            for i,c in zip(positions,shuffled):rows[i]=c
        prepared['candidates']=rows
    prepared['tie_order_rule']='Stateless seeded priorities by query round; equal GP scores share competition rank. No future outcomes or ID ranks.'
    return prepared
