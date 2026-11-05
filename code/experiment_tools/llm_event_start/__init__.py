"""An unrestricted LLM first point and sparse confidence-triggered suggestions."""
from llm_screen_qlogei import PARAMETERS as NOTE_PARAMETERS
VARIANT='llm_event_start_qlogei_20261104'
MODEL='gpt-6.1-terra'
EFFORT='medium'
PARAMETERS=dict(NOTE_PARAMETERS,seeds=list(range(10)),first_point='LLM or uniform full-pool random',
    subsequent_policy='full-pool native qLogEI except reviewed event proposals',
    projection='nearest normalized continuous input with exactly matching nominal catalyst',
    maximum_projection_distance=.35)
