"""Original single controlled-test advice, over the full focused candidate pool."""
from . import stable_r2_policy as base
from . import stable_agent_policy as original
from . import stable_card_policy as presentation
from . import stable_structure_engine as engine

SYSTEM=original.SYSTEM
validate=original.validate
grouped_summary=presentation.grouped_summary


def build_candidates(pool,state,d):
    prepared=base.focused_candidates(pool,state,d)
    if not prepared['candidates']:return prepared
    ranks=engine.competition_ranks(d['base'],d['available']);ix={k:i for i,k in enumerate(pool['ids'])}
    for row in prepared['candidates']:row['base_rank']=int(ranks[ix[row['candidate_id']]])
    prepared['candidates']=presentation.canonical_rows(pool,prepared['candidates'])
    prepared['shortlist_rule']='Complete focused controlled pool; no top5 acquisition truncation. Presentation order is not a choice priority.'
    return prepared
