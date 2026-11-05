"""LLM screening with per-claim KF feedback and unchanged greedy qLogEI."""
import math

VARIANT='llm_screen_qlogei_20261104'
MODEL='gpt-6.1-terra'
EFFORT='medium'
PARAMETERS=dict(total_queries=8,initial=1,logei_budget=math.log(2),
    minimum_reliability=.2,minimum_weighted_probability_gap=.01,
    maximum_refreshes=2,refresh_minimum_gap=1,negative_skill_streak=2,
    deployed_nonimprovement_pause=2,maximum_notes=4,minimum_scope_candidates=4,
    minimum_yield_sd=5.,maximum_yield_sd=50.)
