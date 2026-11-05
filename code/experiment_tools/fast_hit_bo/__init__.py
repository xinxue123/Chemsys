"""A matched finite-budget qLogEI versus two-step rollout experiment."""
VARIANT = 'fast_hit_bo_20261104'
PARAMETERS = dict(
    total_queries=8,
    initial_observations=1,
    qlogei_samples=512,
    fantasy_samples=32,
    outer_qlogei=16,
    outer_uncertainty=16,
    coverage_per_catalyst=2,
    fit_after_observations=4,
    noise_variance=1e-6,
    prior_mean=.5,
    prior_outer_scale=.025,
    prior_inner_scale=.5,
    prior_continuous_lengthscale=.35,
    prior_categorical_lengthscale=1.,
    minimum_outer_scale=1e-5,
    outer_scale_prior_concentration=2.,
    outer_scale_prior_rate=80.,
)
