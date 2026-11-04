"""V13 zero-observation comparison; warm-start evidence remains frozen."""
VARIANT = 'cold0_20261102'
MODEL = 'gpt-6.1-terra'
EFFORT = 'medium'
BACKEND = 'fresh_subagent_per_request'
QUERIES = 20
ARMS = ('bo', 'v13', 'reasoning_bo')
