"""Prospective evidence, separate recommendation permissions, and review gates.

No GP changes, candidate generation, network calls, or optimization metric claims.
The small-sample thresholds below are engineering defaults, not tuned results.
"""
from copy import deepcopy
from hashlib import sha256
from math import isfinite
import json

from . import VERSION, calibration


PARAMETERS = dict(positive_tests_required=2, unsuccessful_tests_before_suppression=2,
                  stagnation_window=3, max_review_tests=2)
USES = {"positive_support", "tentative_test", "review_test", "warning", "shadow"}


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def in_scope(scope, conditions):
    for axis, spec in scope.items():
        if axis not in conditions:
            return False
        if "allowed" in spec:
            if conditions[axis] not in spec["allowed"]:
                return False
        elif "bounds" in spec:
            x = conditions[axis]
            if type(x) not in (int, float) or not isfinite(x) or not spec["bounds"][0] <= x <= spec["bounds"][1]:
                return False
        else:
            raise ValueError("Invalid scope")
    return True


def new_note(note_id, version, effect, scope, focus, *, created_query=1, predecessors=()):
    if not isinstance(note_id, str) or not note_id or type(version) is not int or version < 1:
        raise ValueError("A versioned note identity is required")
    if effect not in ("promote", "avoid") or not focus or len(set(focus)) != len(focus):
        raise ValueError("Declare direction and relevant physical axes")
    if type(created_query) is not int or created_query < 1 or any(type(a) is not str or not a for a in focus):
        raise ValueError("Valid creation query and named physical axes required")
    if not isinstance(scope, dict):
        raise ValueError("Scope must be a physical condition mapping")
    for axis, spec in scope.items():
        if type(axis) is not str or not axis or not isinstance(spec, dict):
            raise ValueError("Invalid scope axis")
        if set(spec) == {"allowed"}:
            if not isinstance(spec["allowed"], list) or not spec["allowed"] or any(type(v) is not str for v in spec["allowed"]):
                raise ValueError("Category scope needs actual string labels")
        elif set(spec) == {"bounds"}:
            if (not isinstance(spec["bounds"], (list, tuple)) or len(spec["bounds"]) != 2
                    or any(type(v) not in (int, float) or not isfinite(v) for v in spec["bounds"])
                    or not 0 <= spec["bounds"][0] <= spec["bounds"][1] <= 1):
                raise ValueError("Continuous scope uses normalized finite bounds")
        else:
            raise ValueError("Scope must declare categorical labels or continuous bounds")
    # Carry factual suppression across an ID/focus change when the new scope
    # includes an adverse measured point. Do not replay old KF measurements.
    inherited = {}
    used_events = set()
    review_tests = 0
    for old in predecessors:
        matching = [f for f in old["recommendation"]["risk_facts"]
                    if old["id"] == note_id or in_scope(scope, f["conditions"])]
        if matching:
            inherited.update({f["fact_id"]: deepcopy(f) for f in matching})
            used_events.update(old["recommendation"]["used_review_events"])
            review_tests = max(review_tests, old["recommendation"]["review_tests_used"])
    permission = "not_applicable" if effect == "avoid" else ("suppressed" if inherited else "tentative")
    return dict(schema=VERSION, id=note_id, version=version, effect=effect,
                scope=deepcopy(scope), focus=list(focus), created_query=created_query,
                calibration=calibration.initial(),
                recommendation=dict(permission=permission, positive_tests=[], failure_streak=0,
                    risk_facts=list(inherited.values()),
                    accurate_failure_count=sum(f["accurate_failure_forecast"] for f in inherited.values()),
                    suppressed_query=created_query if inherited else None,
                    suppression_reason="inherited_adverse_facts" if inherited else None,
                    used_review_events=sorted(used_events), review_tests_used=review_tests,
                    review=None, review_positive_tests=[], transitions=[]),
                events=[], last_feedback_query=0, measured_ids=[])


def _transition(rec, permission, query, reason):
    before = rec["permission"]
    if before != permission:
        rec["transitions"].append(dict(query=query, before=before, after=permission, reason=reason))
    rec["permission"] = permission


def suppress(note, query, reason):
    result = deepcopy(note)
    rec = result["recommendation"]
    if note["effect"] != "promote" or rec["permission"] == "withdrawn":
        return result
    if rec["permission"] != "suppressed":
        rec["suppressed_query"] = query
        rec["positive_tests"] = []
        rec["review_positive_tests"] = []
    _transition(rec, "suppressed", query, reason)
    rec["suppression_reason"] = reason
    rec["review"] = None
    return result


def withdraw(note, query, reason):
    result = deepcopy(note)
    rec = result["recommendation"]
    _transition(rec, "withdrawn", query, reason)
    rec["review"] = None
    rec["positive_tests"] = []
    # Calibration is an immutable historical estimate, not zeroed by retirement.
    return result


def _history_values(history):
    if len(history) < 4 or any(type(r.get("raw_value")) not in (int, float)
                             or not isfinite(r["raw_value"]) for r in history):
        raise ValueError("Finite own observations including four initial points are required")
    return [r["raw_value"] for r in history]


def stagnation_event(history, next_query):
    values = _history_values(history)
    if len(history) != 4 + next_query - 1:
        raise ValueError("Review must use the complete pre-query history")
    window = PARAMETERS["stagnation_window"]
    if next_query <= window or max(values[-window:]) > max(values[:-window]):
        return None
    best = max(values[:4])
    last_improvement = 0
    for query, value in enumerate(values[4:], 1):
        if value > best:
            last_improvement, best = query, value
    # A continuing plateau has one identity, not a fresh allowance each round.
    return dict(kind="stagnation", event_id=f"stagnation_after_query_{last_improvement}",
                query=next_query, history_sha256=digest(history), window=window,
                candidate_conditions=None)


def condition_change_event(note, conditions, next_query, thresholds):
    """Continuous material-change thresholds must be supplied by the caller.

    Compare actual candidate inputs with an adverse measured point, never IDs,
    language changes, incumbent changes, or withheld objective values.
    """
    if not in_scope(note["scope"], conditions):
        return None
    for fact in reversed(note["recommendation"]["risk_facts"]):
        changes = {}
        for axis in note["focus"]:
            if axis not in conditions or axis not in fact["conditions"]:
                continue
            old, new = fact["conditions"][axis], conditions[axis]
            if isinstance(old, str) and isinstance(new, str):
                material = old != new
            elif type(old) in (int, float) and type(new) in (int, float):
                threshold = thresholds.get(axis)
                if threshold is None:
                    continue
                if type(threshold) not in (int, float) or not isfinite(threshold) or threshold <= 0:
                    raise ValueError("Positive material-change threshold required")
                material = abs(new - old) >= threshold
            else:
                continue
            if material:
                changes[axis] = dict(before=old, after=new)
                if type(old) in (int, float):
                    changes[axis]["minimum_change"] = threshold
        if changes:
            return dict(kind="condition_change", event_id="condition_change_" + digest(
                            dict(fact=fact["fact_id"], changes=changes)),
                        query=next_query, fact_id=fact["fact_id"], changes=changes,
                        candidate_conditions=deepcopy(conditions))
    return None


def open_review(note, history, next_query, *, conditions=None, thresholds=None):
    """Only actual stagnation or a checked material change opens a bounded test."""
    values = _history_values(history)
    if len(values) != 4 + next_query - 1 or next_query <= note["last_feedback_query"]:
        raise ValueError("Review requires the next pre-query history")
    result = deepcopy(note)
    rec = result["recommendation"]
    if rec["permission"] != "suppressed":
        return result, dict(opened=False, reason="not_suppressed")
    if rec["review_tests_used"] >= PARAMETERS["max_review_tests"]:
        return result, dict(opened=False, reason="review_budget_exhausted")
    event = None
    if conditions is not None:
        event = condition_change_event(note, conditions, next_query, thresholds or {})
    if event is None:
        event = stagnation_event(history, next_query)
    if event is None:
        return result, dict(opened=False, reason="no_stagnation_or_material_change")
    if event["event_id"] in rec["used_review_events"]:
        return result, dict(opened=False, reason="event_already_consumed")
    rec["used_review_events"].append(event["event_id"])
    rec["review"] = deepcopy(event)
    rec["review_positive_tests"] = []
    rec["positive_tests"] = []
    _transition(rec, "reviewing", next_query, event["kind"])
    return result, dict(opened=True, event=event)


def public_note(note):
    rec = note["recommendation"]
    return dict(id=note["id"], version=note["version"], effect=note["effect"],
                scope=deepcopy(note["scope"]),
                prediction_reliability=dict(note["calibration"],
                    meaning="relative probability-forecast skill, not experiment utility"),
                recommendation=dict(permission=rec["permission"],
                    positive_support_allowed=note["effect"] == "promote" and rec["permission"] == "allowed",
                    current_epoch_positive_tests=len(rec["positive_tests"]),
                    risk_warning_available=bool(rec["risk_facts"]),
                    accurate_failure_evidence=rec["accurate_failure_count"],
                    suppression_reason=rec["suppression_reason"],
                    review=deepcopy(rec["review"]), review_tests_used=rec["review_tests_used"]),
                adverse_facts=deepcopy(rec["risk_facts"][-6:]))


def validate_use(note, conditions, use):
    if use not in USES or not in_scope(note["scope"], conditions):
        raise ValueError("Unknown use or out-of-scope knowledge")
    rec = note["recommendation"]
    if rec["permission"] == "withdrawn" and use == "warning" and rec["risk_facts"]:
        return  # Historical adverse facts remain usable; no new forecast is made.
    if rec["permission"] == "withdrawn":
        raise ValueError("A withdrawn version cannot supply new forecasts or recommendations")
    if use == "positive_support" and (note["effect"] != "promote" or rec["permission"] != "allowed"):
        raise ValueError("Prediction credibility does not grant positive recommendation permission")
    if use == "tentative_test" and (note["effect"] != "promote" or rec["permission"] != "tentative"):
        raise ValueError("Suppression cannot be bypassed by relabelling a new exploratory test")
    if use == "review_test":
        if rec["permission"] != "reviewing" or rec["review"] is None:
            raise ValueError("A validated review event is required")
        target = rec["review"]["candidate_conditions"]
        if target is not None:
            # A second distinct point may confirm the same changed regime.
            # Exact point binding would make two-test recovery impossible.
            for axis, change in rec["review"]["changes"].items():
                value = conditions.get(axis)
                if "minimum_change" in change:
                    delta = value - change["before"] if type(value) in (int, float) else 0
                    if (delta * (change["after"] - change["before"]) <= 0
                            or abs(delta) < change["minimum_change"]):
                        raise ValueError("Review point does not test the declared changed regime")
                elif value != change["after"]:
                    raise ValueError("Review point changes the declared categorical regime")
        if rec["review_tests_used"] >= PARAMETERS["max_review_tests"]:
            raise ValueError("No remaining review test budget")


def commit_forecast(note, history, query, candidate_id, conditions, p_llm, p_gp, *, use="shadow"):
    values = _history_values(history)
    if note["recommendation"]["permission"] == "withdrawn":
        raise ValueError("Withdrawn versions expose historical warnings, not new forecasts")
    if (len(values) != 4 + query - 1 or query <= note["last_feedback_query"]
            or query < note["created_query"]):
        raise ValueError("Forecast must precede the next own experiment")
    if candidate_id in {r["candidate_id"] for r in history}:
        raise ValueError("Candidate was already measured")
    if any(type(p) not in (int, float) or not isfinite(p) or not 0 <= p <= 1 for p in (p_llm, p_gp)):
        raise ValueError("Invalid forecast probabilities")
    validate_use(note, conditions, use)
    packet = dict(query=query, candidate_id=candidate_id, conditions=deepcopy(conditions),
                  id=note["id"], version=note["version"], incumbent=max(values),
                  history_sha256=digest(history), state_sha256=digest(note),
                  p_LLM=float(p_llm), p_GP=float(p_gp), use=use,
                  target="strict_improvement_over_current_incumbent")
    packet["commit_sha256"] = digest(packet)
    return packet


def feedback(note, packet, history, observation):
    """Atomic feedback; actual improvement evidence and KF scores are separate."""
    if packet.get("commit_sha256") != digest({k: v for k, v in packet.items() if k != "commit_sha256"}):
        raise ValueError("Forecast commitment changed")
    if (packet["state_sha256"] != digest(note) or packet["history_sha256"] != digest(history)
            or observation.get("candidate_id") != packet["candidate_id"]
            or packet["query"] != len(history) - 3
            or packet["incumbent"] != max(_history_values(history))):
        raise ValueError("Forecast identity, prestate or reference does not match")
    value = observation.get("raw_value")
    if type(value) not in (int, float) or not isfinite(value):
        raise ValueError("A finite measured outcome is required")
    validate_use(note, packet["conditions"], packet["use"])
    result = deepcopy(note)
    improved = value > packet["incumbent"]
    result["calibration"], skill = calibration.update(
        note["calibration"], packet["p_LLM"], packet["p_GP"], improved)
    rec = result["recommendation"]
    before_permission = rec["permission"]
    tested = packet["use"] in {"positive_support", "tentative_test", "review_test"}
    accurate_failure = not improved and packet["p_LLM"] < .5 and skill["Brier_gain"] > 1e-12
    event_id = digest(dict(commit=packet["commit_sha256"], observation=observation))
    if not improved:
        fact = dict(fact_id=event_id, query=packet["query"], candidate_id=packet["candidate_id"],
                    conditions=deepcopy(packet["conditions"]), observed=value,
                    reference=packet["incumbent"], accurate_failure_forecast=accurate_failure,
                    Brier_gain=skill["Brier_gain"], source_id=note["id"], source_version=note["version"])
        rec["risk_facts"].append(fact)
        if accurate_failure:
            rec["accurate_failure_count"] += 1
    if packet["use"] == "review_test":
        rec["review_tests_used"] += 1
    if tested and improved:
        # A productive observation is not erased merely because its probability
        # was less well calibrated than the GP. It is not a causal-effect claim.
        rec["failure_streak"] = 0
        rec["positive_tests"].append(event_id)
        if before_permission == "reviewing":
            rec["review_positive_tests"].append(event_id)
        if len(rec["positive_tests"]) >= PARAMETERS["positive_tests_required"]:
            _transition(rec, "allowed", packet["query"], "new_direction_test_evidence")
            rec["review"] = None
    elif tested:
        rec["failure_streak"] += 1
    reason = "calibration_only" if not tested else "direction_test_recorded"
    if note["effect"] == "promote" and (accurate_failure or
            (tested and not improved and (before_permission == "reviewing" or
                rec["failure_streak"] >= PARAMETERS["unsuccessful_tests_before_suppression"]))):
        reason = "accurate_failure_inhibits" if accurate_failure else "unsuccessful_direction_tests"
        result = suppress(result, packet["query"], reason)
        rec = result["recommendation"]
    if rec["permission"] == "reviewing" and rec["review_tests_used"] >= PARAMETERS["max_review_tests"]:
        result = suppress(result, packet["query"], "review_budget_exhausted_without_support")
        rec = result["recommendation"]
    event = dict(event_id=event_id, query=packet["query"], candidate_id=packet["candidate_id"],
                 use=packet["use"], improved=improved, actual_gain=value - packet["incumbent"],
                 direction_test=tested, accurate_failure_forecast=accurate_failure,
                 calibration=skill, recommendation_before=before_permission,
                 recommendation_after=rec["permission"], reason=reason,
                 commit_sha256=packet["commit_sha256"])
    result["events"].append(event)
    result["last_feedback_query"] = packet["query"]
    result["measured_ids"].append(packet["candidate_id"])
    return result, event
