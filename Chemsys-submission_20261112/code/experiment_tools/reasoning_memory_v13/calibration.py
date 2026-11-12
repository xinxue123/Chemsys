"""Scalar predictive-skill KF. No recommendation lifecycle or direction here."""
from copy import deepcopy
from math import erfc, isfinite, sqrt


PARAMETERS = dict(prior_variance=.25, process_variance=.0025,
                  measurement_variance_floor=.0025, maximum_reliability=.8)


def initial():
    return dict(mean=0., variance=.25, probability_positive=.5,
                reliability=.4, updates=0)


def update(state, p_llm, p_gp, improved):
    if type(improved) is not bool or any(
        type(p) not in (int, float) or not isfinite(p) or not 0 <= p <= 1
        for p in (p_llm, p_gp)
    ):
        raise ValueError("Finite probabilities and a boolean outcome are required")
    if set(state) != set(initial()):
        raise ValueError("Calibration must not contain recommendation lifecycle fields")
    if any(not isfinite(state[k]) for k in ("mean", "variance", "reliability")) or state["variance"] <= 0:
        raise ValueError("Invalid calibration state")
    s = float(improved)
    z = (s - p_gp) ** 2 - (s - p_llm) ** 2
    r = max(PARAMETERS["measurement_variance_floor"],
            4 * (p_llm - p_gp) ** 2 * p_gp * (1 - p_gp))
    predicted = state["variance"] + PARAMETERS["process_variance"]
    gain = predicted / (predicted + r)
    mean = state["mean"] + gain * (z - state["mean"])
    variance = (1 - gain) ** 2 * predicted + gain * gain * r
    probability = .5 * erfc(-mean / sqrt(2 * variance))
    reliability = PARAMETERS["maximum_reliability"] * probability
    # Preserve V12's non-increase rule on non-positive skill feedback. This
    # number describes calibration only; it never restores a permission.
    if z <= 1e-12:
        reliability = min(reliability, state["reliability"])
    after = dict(mean=mean, variance=variance, probability_positive=probability,
                 reliability=reliability, updates=state["updates"] + 1)
    return after, dict(Brier_gain=z, measurement_variance=r,
                       predicted_variance=predicted, kalman_gain=gain,
                       before=deepcopy(state), after=deepcopy(after))
