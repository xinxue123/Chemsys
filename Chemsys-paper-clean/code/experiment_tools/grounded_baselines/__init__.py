"""Matched, grounded LGBO and Reasoning BO comparators."""
VARIANT = 'grounded_baselines_20261105'
MODEL = 'gpt-6.1-terra'
EFFORT = 'medium'
SEEDS = (0, 1, 2, 3)
ARMS = ('lgbo', 'rbo')
PARAMETERS = dict(initial=0, queries=20, maximum_schema_attempts=2,
    quadrature_cap=512, guidance_scale=3., numerical_seed=0,
    first='reuse frozen independently chosen grounded first input',
    initial_knowledge='same frozen blinded chemical hypotheses as current method',
    candidate_source='entire frozen pool; RBO receives its native qLogEI top5',
    LLM_requests_per_trajectory=19, GP_fit_unchanged=True)
