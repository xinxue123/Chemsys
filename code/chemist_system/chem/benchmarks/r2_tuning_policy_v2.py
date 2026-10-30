"""Adaptive validation budget and feature-only coverage; baseline policy frozen."""
from copy import deepcopy
from pathlib import Path
from . import r2_tuning_policy as base
from . import reasoning_bo_data as data

SYSTEM=base.SYSTEM
validate=base.validate


def settings(state):
    return data.read(Path('research/hte_r2_tuning_20261028/plans')/(state['variant']+'.json'))


def build_candidates(pool,state,d):
    cfg=settings(state);prepared=base.build_candidates(pool,state,d)
    if max(h['y'] for h in state['history'])*100>=cfg['high_yield_threshold_percent']:
        prepared['candidates']=[];prepared['validation_skip_reason']='Observed incumbent already meets the predeclared high-yield threshold.'
        return prepared
    if not state['validations'] or not prepared['candidates']:return prepared
    lookup={k:i for i,k in enumerate(pool['ids'])}
    seen={a:{pool['points'][lookup[h['candidate_id']]][a] for h in state['history']} for a in pool['axes']}
    groups={}
    for candidate in prepared['candidates']:
        axis=candidate['changed_axis'];value=pool['points'][lookup[candidate['candidate_id']]][axis]
        if value not in seen[axis]:groups.setdefault(axis,[]).append(candidate)
    if groups:
        axis=min(groups,key=lambda a:(-len(groups[a]),-max(c['GP_latent_SD_pp'] for c in groups[a]),a))
        prepared['candidates']=groups[axis]
        prepared['coverage_focus']={'axis':axis,'unseen_counterexample_levels':{a:len(rows) for a,rows in groups.items()},
            'criterion':'Most unobserved excluded levels; GP uncertainty then axis name break ties. No unseen yields used.'}
    return prepared
