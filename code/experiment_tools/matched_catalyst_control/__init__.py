"""One matched catalyst experiment, followed by a frozen native BO suffix."""
VARIANT = 'matched_catalyst_control_20261104'
PARAMETERS = dict(total_queries=8, exact_continuous_tolerance=1e-12,
    intervention='earliest original postinitial knowledge action with an exact observed-anchor catalyst match',
    anchor='most recently observed eligible anchor',
    selector='maximum unchanged native qLogEI among matched unqueried catalysts; original physical tie priorities',
    acquisition_gate='one declared experimental override; record original gate rejection',
    continuation='native full-pool qLogEI after the replacement; frozen control suffix must already use this rule',
    LLM_calls=0, category_correlation_diagnostic=.99)
