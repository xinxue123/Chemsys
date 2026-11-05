"""Explicit transfer from one LLM first point to an exact pure-BO endpoint."""
import math

VARIANT='llm_bo_handoff_20261104'
PARAMETERS=dict(total_queries=8,maximum_logei_gap=math.log(4),
    reference_KF_reliability=.4,minimum_KF_reliability=.2,
    favour_probability_threshold=.5,
    authority_rule='(total_queries-query)/(total_queries-1)',
    final_rule='exact native full-pool qLogEI maximum including its original tie rule')
