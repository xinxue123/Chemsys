"""Evidence-led revisions of the hypothesis-scope controller.

See docs/hypothesis_scope_v2_registration_20261105.md. Frozen modules are imported,
never edited. GP, acquisition, pools, budgets and prompts are unchanged.
"""
VARIANT = 'hypothesis_scope_v2_20261105'
MODEL = 'gpt-6.1-terra'
EFFORT = 'medium'
SEEDS = (0, 1, 2, 3)
EXECUTE_ARMS = ('x0', 'x1', 'ex1')
LIVE_ARMS = ('h1', 'x1', 'ex1')
EVIDENCE_FIRST_ARMS = ('ex1',)
ARMS = ('n0', 'x0', 'h1', 'x1', 'ex1')
MAXIMUM_SCHEMA_ATTEMPTS = 2
MAXIMUM_OUTPUT_TOKENS = 8192
