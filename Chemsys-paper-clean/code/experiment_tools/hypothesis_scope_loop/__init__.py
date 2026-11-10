"""Scoped chemical hypotheses, native qLogEI and feedback-triggered LLM refresh."""
VARIANT='hypothesis_scope_loop_20261105'
MODEL='gpt-6.1-terra'
EFFORT='medium'
PARAMETERS=dict(initial=0,queries=20,seed=0,maximum_hypotheses=3,
    maximum_LLM_calls=4,maximum_schema_attempts=2,minimum_refresh_gap=3,
    reserved_late_query=15,last_refresh_query=18,minimum_scope_candidates=4,
    minimum_KF_reliability=.2,minimum_positive_probability=.5,
    negative_skill_streak=2,forecast_miss_streak=2,forecast_miss_SD=2.,
    deployed_nonimprovement_pause=2,stagnation_window=3,
    maximum_retests_per_semantic_rule=1,first_point='same frozen independent LLM first choice',
    selection='native qLogEI argmax inside intersection of eligible favour scopes minus eligible avoid scopes',
    global_qLogEI_gap_gate=False,GP_prediction_modification=False)
