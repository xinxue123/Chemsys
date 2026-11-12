"""Grounded LLM first experiment, followed by unchanged native qLogEI."""
VARIANT='grounded_first_start_20261105'
MODEL='gpt-6.1-terra'
EFFORT='medium'
PARAMETERS=dict(initial=0,queries=20,seed=0,input_neighbors=5,
    maximum_logical_LLM_calls=16,maximum_schema_attempts=2,
    first_choice='independent grounded LLM proposal, input-only projection and confirmation',
    following_choices='native full-pool qLogEI',background_MTBD_corrected=True)
