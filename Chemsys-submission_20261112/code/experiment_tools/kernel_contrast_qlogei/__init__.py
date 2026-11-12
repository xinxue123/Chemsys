"""Condition contrasts among plausible GP fits within unchanged qLogEI."""
import math

VARIANT = 'kernel_contrast_qlogei_20261104'
PARAMETERS = dict(initial=4, queries=16, seed=0, maximum_logei_gap=math.log(4),
    MLL_gap=0.01, minimum_axis_correlation_gap=0.05,
    minimum_standardized_disagreement=0.5, continuous_tolerance=1e-8,
    maximum_alternative_parents=1)
