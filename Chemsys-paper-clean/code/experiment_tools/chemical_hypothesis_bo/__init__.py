"""Grounded chemical contrasts, bounded preference experts and mixture EI."""
import math

VARIANT='chemical_hypothesis_bo_20261105'
ARMS=('fixed','adaptive')
MODEL='gpt-6.1-terra'
EFFORT='medium'
BACKEND='fresh_subagent_per_request'
PARAMETERS=dict(initial=4,queries=16,seed=0,maximum_hypotheses=3,
    preference_shift_SD=.5,quadrature_per_side=16,maximum_logei_gap=math.log(4),
    initial_base_mass=.5,new_hypothesis_mass=.2,maximum_LLM_calls=4,
    minimum_refresh_gap=3,reserved_refresh_query=11,last_refresh_query=14,
    predictive_gain_tolerance=.01,informative_shift_tolerance=1e-6)
