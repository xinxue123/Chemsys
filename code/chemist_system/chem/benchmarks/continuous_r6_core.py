"""Auditable R6 building blocks for finite-pool, single-query chemical BO.

This module is deliberately independent of a particular GP/LLM backend. The
original optimizer must provide *pre-observation* GP scores and covariances.
All returned decisions preserve the original finite candidate pool.

No non-harm, regret, or chemistry-performance guarantee is implied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, isfinite, isinf, log, log1p, sqrt
from random import Random
from typing import Literal, Mapping, Sequence

from scipy.special import log_ndtr, logsumexp


FeatureType = Literal["continuous", "categorical"]


@dataclass(frozen=True)
class Candidate:
    point_id: str
    features: tuple[object, ...]
    qlogei: float
    information_tie: float
    latent_sd_tie: float
    random_tie: float


@dataclass(frozen=True)
class Gate:
    bo_point_id: str
    eligible_point_ids: frozenset[str]
    best_logei: float
    epsilon: float


@dataclass(frozen=True)
class MenuPoint:
    """The only point fields that the GP-filtered LLM is allowed to see."""

    point_id: str
    features: tuple[object, ...]


@dataclass(frozen=True)
class Comparison:
    better_point_id: str
    worse_point_id: str
    # Continuous axes use +1 or -1; categorical axes use None.
    focus_directions: Mapping[int, int | None]


@dataclass(frozen=True)
class Review:
    passed: bool
    reason: str


@dataclass(frozen=True)
class ShadowPrediction:
    """Committed before the outcome is revealed, then consumed only once."""

    hypothesis_id: str
    point_id: str
    query_round: int
    gp_mean_y: float
    gp_var_y: float  # Includes observation noise, strictly positive.
    contrast_mean: float  # GP posterior mean f(x+) - f(x-).
    contrast_var: float  # GP posterior latent variance of that difference.
    cov_y_contrast: float  # Cov[y(x), f(x+) - f(x-)].
    preference_noise_sd: float  # Noise on the qualitative inequality.


@dataclass
class EvidenceLedger:
    """Online model-evidence weight, NOT the probability chemistry is true.

    Log-score increments are exact for a *single* probit inequality and a
    Gaussian frozen predictive snapshot. The cumulative score is not an exact
    Bayes factor when GP hyperparameters are repeatedly refit.
    """

    hypothesis_id: str
    creation_round: int
    prior_model_weight: float = 0.5
    max_deployment_weight: float = 0.5
    cumulative_log_score_gain: float = 0.0
    seen_point_ids: set[str] = field(default_factory=set)
    event_log: list[tuple[int, str, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0 < self.prior_model_weight < 1:
            raise ValueError("prior_model_weight must be strictly between 0 and 1")
        if not 0 < self.max_deployment_weight <= 1:
            raise ValueError("max_deployment_weight must be in (0, 1]")

    def observe(self, prediction: ShadowPrediction, observed_y: float) -> float:
        if prediction.hypothesis_id != self.hypothesis_id:
            raise ValueError("hypothesis mismatch")
        if prediction.query_round < self.creation_round:
            raise ValueError("retrospective evidence is prohibited")
        if prediction.point_id in self.seen_point_ids:
            raise ValueError("same experimental point cannot count twice")
        if not isfinite(observed_y):
            raise ValueError("nonfinite observation")
        score = prospective_log_score_gain(prediction, observed_y)
        self.seen_point_ids.add(prediction.point_id)
        self.cumulative_log_score_gain += score
        self.event_log.append((prediction.query_round, prediction.point_id, score))
        return score

    @property
    def deployment_weight(self) -> float:
        if not self.seen_point_ids or self.cumulative_log_score_gain <= 0:
            return 0.0
        prior_log_odds = log(self.prior_model_weight / (1 - self.prior_model_weight))
        odds = prior_log_odds + self.cumulative_log_score_gain
        posterior = 1 / (1 + exp(-odds)) if odds >= 0 else exp(odds) / (1 + exp(odds))
        # Evidence must beat the *starting* model weight before deployment.
        excess = (posterior - self.prior_model_weight) / (1 - self.prior_model_weight)
        return min(self.max_deployment_weight, max(0.0, excess))


@dataclass(frozen=True)
class ContrastProbe:
    point_id: str
    predictive_var_y: float
    contrast_var: float
    cov_y_contrast: float
    preference_noise_sd: float


@dataclass(frozen=True)
class VerificationPolicy:
    max_verification_queries: int = 2
    last_verification_round: int = 10
    max_exceptions: int = 1
    max_extra_qlogei_gap: float = log(2)
    minimum_incremental_information_nats: float = 0.01


@dataclass(frozen=True)
class Decision:
    point_id: str
    mode: str
    acquisition_gap: float
    used_exception: bool = False
    information_gain_nats: float = 0.0


def _validate_candidates(candidates: Sequence[Candidate]) -> None:
    if not candidates:
        raise ValueError("candidate pool is empty")
    ids = [c.point_id for c in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate candidate IDs")
    for c in candidates:
        if not c.point_id or c.qlogei != c.qlogei or c.qlogei == float("inf"):
            raise ValueError("candidate ID or qLogEI invalid")
        if not all(isfinite(t) for t in (c.information_tie, c.latent_sd_tie, c.random_tie)):
            raise ValueError("tie priorities must be finite")


def qlogei_gate(candidates: Sequence[Candidate], query_round: int, total_rounds: int = 16) -> Gate:
    """Keep R5's qLogEI opportunity constraint, including round 16 = 0."""
    _validate_candidates(candidates)
    if total_rounds < 2 or not 1 <= query_round <= total_rounds:
        raise ValueError("query_round must be 1..total_rounds")
    epsilon = log(4) * (total_rounds - query_round) / (total_rounds - 1)
    # Non-ID tie breaking is supplied by the optimizer, not candidate numbering.
    best = max(candidates, key=lambda c: (c.qlogei, c.information_tie, c.latent_sd_tie, c.random_tie))
    if isinf(best.qlogei) and best.qlogei < 0:
        eligible = frozenset(c.point_id for c in candidates)
    else:
        eligible = frozenset(c.point_id for c in candidates if best.qlogei - c.qlogei <= epsilon + 1e-12)
    return Gate(best.point_id, eligible, best.qlogei, epsilon)


def _distance(a: tuple[object, ...], b: tuple[object, ...], kinds: Sequence[FeatureType]) -> float:
    if len(a) != len(b) or len(a) != len(kinds):
        raise ValueError("feature shape mismatch")
    cost = 0.0
    for va, vb, kind in zip(a, b, kinds):
        if kind == "categorical":
            cost += float(va != vb)
        elif kind == "continuous":
            fa, fb = float(va), float(vb)
            if not (0 <= fa <= 1 and 0 <= fb <= 1):
                raise ValueError("continuous features must be normalized to [0, 1]")
            cost += (fa - fb) ** 2
        else:
            raise ValueError(f"unknown feature type: {kind}")
    return cost


def _diverse_menu(pool: Sequence[Candidate], kinds: Sequence[FeatureType],
                  max_items: int, seed: int, required_id: str | None) -> tuple[MenuPoint, ...]:
    if max_items < 2:
        raise ValueError("menu must allow at least two candidates")
    if not pool:
        return ()
    rng = Random(seed)
    random_priority = {c.point_id: rng.random() for c in pool}
    by_id = {c.point_id: c for c in pool}
    if required_id is not None and required_id not in by_id:
        raise ValueError("required BO point absent from feasible pool")
    chosen = [required_id] if required_id is not None else [rng.choice(pool).point_id]
    while len(chosen) < min(len(pool), max_items):
        remaining = [c for c in pool if c.point_id not in chosen]
        next_point = max(
            remaining,
            key=lambda c: (
                min(_distance(c.features, by_id[s].features, kinds) for s in chosen),
                random_priority[c.point_id],
            ),
        )
        chosen.append(next_point.point_id)
    rng.shuffle(chosen)
    # Never expose acquisition scores, uncertainty, or the BO-reference label.
    return tuple(MenuPoint(p, by_id[p].features) for p in chosen)


def feasible_llm_menu(candidates: Sequence[Candidate], gate: Gate,
                      feature_types: Sequence[FeatureType], max_items: int = 24,
                      seed: int = 0) -> tuple[MenuPoint, ...]:
    """Always shows executable points; NOT a GP-independent LLM condition."""
    pool = [c for c in candidates if c.point_id in gate.eligible_point_ids]
    return _diverse_menu(pool, feature_types, max_items, seed, gate.bo_point_id)


def independent_llm_menu(candidates: Sequence[Candidate],
                         feature_types: Sequence[FeatureType], max_items: int = 24,
                         seed: int = 0) -> tuple[MenuPoint, ...]:
    """GP-blind, feature-only control: no GP-derived membership or ordering."""
    return _diverse_menu(candidates, feature_types, max_items, seed, None)


def review_comparison(comparison: Comparison, menu: Sequence[MenuPoint],
                      feature_types: Sequence[FeatureType], max_background_change: float = 0.25) -> Review:
    by_id = {c.point_id: c for c in menu}
    if comparison.better_point_id == comparison.worse_point_id:
        return Review(False, "comparison uses the same point twice")
    if comparison.better_point_id not in by_id or comparison.worse_point_id not in by_id:
        return Review(False, "comparison point is absent from the presented menu")
    focuses = comparison.focus_directions
    if not 1 <= len(focuses) <= 2 or any(not 0 <= i < len(feature_types) for i in focuses):
        return Review(False, "expect one or two valid focus axes")
    better = by_id[comparison.better_point_id].features
    worse = by_id[comparison.worse_point_id].features
    if len(better) != len(feature_types) or len(worse) != len(feature_types):
        return Review(False, "feature shape mismatch")
    for i, kind in enumerate(feature_types):
        a, b = better[i], worse[i]
        if i in focuses:
            direction = focuses[i]
            if kind == "categorical":
                if direction is not None or a == b:
                    return Review(False, f"categorical axis {i} must be a real retype")
            elif kind == "continuous":
                if direction not in (-1, 1) or direction * (float(a) - float(b)) <= 0:
                    return Review(False, f"focus-axis {i} direction inconsistent with real points")
            else:
                return Review(False, f"invalid axis type {kind}")
        elif kind == "categorical" and a != b:
            return Review(False, f"background categorical axis {i} changed")
        elif kind == "continuous" and abs(float(a) - float(b)) > max_background_change:
            return Review(False, f"background continuous axis {i} moved too much")
    return Review(True, "prospective pointwise statement is direction-consistent")


def _check_gaussian_moments(variance_y: float, variance_z: float, cov_y_z: float) -> float:
    if variance_y <= 0 or variance_z <= 0 or not all(
        isfinite(v) for v in (variance_y, variance_z, cov_y_z)
    ):
        raise ValueError("invalid Gaussian variance/covariance")
    remaining = variance_z - cov_y_z * cov_y_z / variance_y
    if remaining < -1e-8 * variance_z:
        raise ValueError("inconsistent covariance matrix")
    return max(remaining, 1e-15 * variance_z)


def conditional_preference_moments(mean_x: float, var_x: float, cov_x_contrast: float,
                                   mean_contrast: float, var_contrast: float,
                                   preference_noise_sd: float) -> tuple[float, float]:
    """Exact first two moments of f(x) | [f(x+) - f(x-) + e > 0].

    The resulting distribution is skew-normal, *not* Gaussian. Do not pass
    the two moments to a Gaussian EI formula and call it exact acquisition.
    """
    if preference_noise_sd <= 0 or var_contrast < 0:
        raise ValueError("invalid preference noise/contrast variance")
    variance_z = var_contrast + preference_noise_sd**2
    _check_gaussian_moments(var_x, variance_z, cov_x_contrast)
    alpha = mean_contrast / sqrt(variance_z)
    log_phi = -0.5 * alpha**2 - 0.5 * log(2 * 3.141592653589793)
    lam = exp(log_phi - float(log_ndtr(alpha)))
    shift = cov_x_contrast / sqrt(variance_z) * lam
    variance = var_x - (cov_x_contrast**2 / variance_z) * lam * (lam + alpha)
    return mean_x + shift, max(variance, 0.0)


def prospective_log_score_gain(prediction: ShadowPrediction, observed_y: float) -> float:
    """log p(y | GP, expert says x+>x-) - log p(y | GP).

    Correct for a single soft-probit sign given a frozen pre-outcome GP:
    the baseline Gaussian density cancels via Bayes' rule. No numeric LLM
    yield prediction or retrospective residual-fitting is required.
    """
    if prediction.preference_noise_sd <= 0 or prediction.contrast_var < 0:
        raise ValueError("invalid preference noise/variance")
    vz = prediction.contrast_var + prediction.preference_noise_sd**2
    rem = _check_gaussian_moments(prediction.gp_var_y, vz, prediction.cov_y_contrast)
    alpha = prediction.contrast_mean / sqrt(vz)
    beta = (prediction.contrast_mean + prediction.cov_y_contrast / prediction.gp_var_y
            * (observed_y - prediction.gp_mean_y)) / sqrt(rem)
    return float(log_ndtr(beta) - log_ndtr(alpha))


def information_about_contrast(probe: ContrastProbe) -> float:
    """I(y(x); contrast + noise) in nats; proxy, NOT binary truth information."""
    if probe.preference_noise_sd <= 0 or probe.contrast_var < 0:
        raise ValueError("invalid preference noise/variance")
    vz = probe.contrast_var + probe.preference_noise_sd**2
    remaining = _check_gaussian_moments(probe.predictive_var_y, vz, probe.cov_y_contrast)
    return 0.5 * log(vz / remaining)


def acquisition_gap(candidate: Candidate, gate: Gate) -> float:
    if isinf(gate.best_logei) and gate.best_logei < 0:
        return 0.0
    return gate.best_logei - candidate.qlogei


def choose_llm_rerank(candidates: Sequence[Candidate], gate: Gate,
                      displayed_menu: Sequence[MenuPoint],
                      ranked_point_ids: Sequence[str], semantic_review: Review) -> Decision:
    """A1 blind-review or A2 immediate rerank, separate from evidence deployment.

    Caller must review the exact proposed comparison *before* invoking this
    selector. A blind-menu rank may miss E; a feasible-menu rank cannot.
    """
    by_id = {c.point_id: c for c in candidates}
    displayed = {c.point_id for c in displayed_menu}
    if not semantic_review.passed or not ranked_point_ids:
        return Decision(gate.bo_point_id, "semantic_reject_or_abstain", 0.0)
    if len(ranked_point_ids) != len(set(ranked_point_ids)):
        raise ValueError("duplicate ranked point")
    if set(ranked_point_ids) - displayed or displayed - by_id.keys():
        raise ValueError("rank includes a point outside the presented finite pool")
    top_id = ranked_point_ids[0]
    if top_id not in gate.eligible_point_ids:
        return Decision(gate.bo_point_id, "GP_rejected_blind_proposal", 0.0)
    if top_id == gate.bo_point_id:
        return Decision(top_id, "LLM_agreed_with_BO", 0.0)
    return Decision(top_id, "direct_LLM_rerank", acquisition_gap(by_id[top_id], gate))


def choose_active_verification(
    candidates: Sequence[Candidate], gate: Gate, probes: Mapping[str, ContrastProbe],
    query_round: int, validation_queries_used: int, exceptions_used: int,
    policy: VerificationPolicy = VerificationPolicy(),
) -> Decision | None:
    """Spend an early capped query only for incremental contrast information.

    Uses no hidden outcomes. A negative logEI gap is never a regret guarantee.
    The BO point may still provide free shadow evidence when None is returned.
    """
    if validation_queries_used >= policy.max_verification_queries or query_round > policy.last_verification_round:
        return None
    by_id = {c.point_id: c for c in candidates}
    if gate.bo_point_id not in probes:
        return None
    bo_information = information_about_contrast(probes[gate.bo_point_id])
    room_for_exception = exceptions_used < policy.max_exceptions
    max_gap = gate.epsilon + (policy.max_extra_qlogei_gap if room_for_exception else 0.0)
    choices: list[tuple[float, float, float, Candidate, bool]] = []
    for point_id, probe in probes.items():
        if point_id not in by_id or point_id == gate.bo_point_id:
            continue
        candidate = by_id[point_id]
        gap = acquisition_gap(candidate, gate)
        info = information_about_contrast(probe)
        if gap > max_gap + 1e-12 or info - bo_information < policy.minimum_incremental_information_nats:
            continue
        exception = point_id not in gate.eligible_point_ids
        if exception and not room_for_exception:
            continue
        choices.append((info - bo_information, -gap, candidate.random_tie, candidate, exception))
    if not choices:
        return None
    chosen = max(choices, key=lambda row: row[:3])
    selected, exception = chosen[-2], chosen[-1]
    return Decision(selected.point_id, "active_verification", acquisition_gap(selected, gate),
                    exception, information_about_contrast(probes[selected.point_id]))


def choose_guided_inside_gate(candidates: Sequence[Candidate], gate: Gate,
                              guided_log_ei: Mapping[str, float],
                              deployment_weight: float) -> Decision:
    """Mix logged EI from base GP and an independently computed preference model.

    Caller must compute guided_log_ei from the *skew-normal* preference
    posterior, e.g. importance-weighted GP samples; not moment-matched EI.
    """
    by_id = {c.point_id: c for c in candidates}
    if gate.bo_point_id not in by_id:
        raise ValueError("BO point missing")
    if not 0 <= deployment_weight <= 1:
        raise ValueError("weight outside [0, 1]")
    if deployment_weight == 0:
        return Decision(gate.bo_point_id, "base_bo", 0.0)
    eligible = [c for c in candidates if c.point_id in gate.eligible_point_ids]
    if set(c.point_id for c in eligible) - guided_log_ei.keys():
        raise ValueError("missing guided EI for eligible candidate")
    for c in eligible:
        g = guided_log_ei[c.point_id]
        if g != g or g == float("inf"):
            raise ValueError("guided EI must be finite or -inf")
    if deployment_weight == 1:
        scoring = lambda c: guided_log_ei[c.point_id]
    else:
        scoring = lambda c: float(logsumexp((log1p(-deployment_weight) + c.qlogei,
                                            log(deployment_weight) + guided_log_ei[c.point_id])))
    selected = max(eligible, key=lambda c: (scoring(c), c.information_tie, c.latent_sd_tie, c.random_tie))
    return Decision(selected.point_id, "knowledge_guided" if selected.point_id != gate.bo_point_id else "base_bo",
                    acquisition_gap(selected, gate))
