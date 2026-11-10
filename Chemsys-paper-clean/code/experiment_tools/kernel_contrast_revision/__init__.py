"""Pool-compatible kernel hypotheses and prequential feedback weights."""
import math

VARIANT = 'kernel_contrast_revision_20261104'
ARMS = ('geometry','hypotheses','soft_fixed','feedback')
PARAMETERS = dict(initial=4,queries=16,seed=0,maximum_logei_gap=math.log(4),
    MLL_gap=.01,matching_bandwidth=.2,local_match_RMS=.1,
    minimum_disagreement=.5,correlation_probe=.2,correlation_targets=(.2,.8),
    continuous_tolerance=1e-8,weight_likelihood='Gaussian_observation_predictive_density')
