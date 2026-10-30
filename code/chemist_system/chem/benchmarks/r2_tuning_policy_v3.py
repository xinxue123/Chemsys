"""Focus later counterexamples using common BO ranking, not future yields."""
from . import r2_tuning_policy_v2 as base

SYSTEM=base.SYSTEM
validate=base.validate


def build_candidates(pool,state,d):
    prepared=base.build_candidates(pool,state,d);cfg=base.settings(state)
    if not state['validations'] or not prepared['candidates']:return prepared
    k=cfg['counterexample_shortlist_size'];rows=prepared['candidates']
    prepared['counterexample_pool_before_shortlist']=len(rows)
    prepared['candidates']=sorted(rows,key=lambda c:(-c['base_qLogEI'],c['candidate_id']))[:k]
    prepared['shortlist_rule']='Top base qLogEI among focused unobserved counterexamples; candidate-ID tie convention. No oracle sorting.'
    return prepared
