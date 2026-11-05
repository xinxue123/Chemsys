"""Attach separated controls to the native GP/official Reasoning BO decision.

Call public_context before the LLM, commit_decision before the oracle, and
apply_feedback after the oracle. This module has no transport or experiment IO.
"""
from copy import deepcopy
from math import isfinite

from knowledge_rank_v9 import engine as native
from reasoning_memory_v12.llm import validate_insight, official_extract, compiled_extractor
from . import control


SYSTEM_APPENDIX = """Knowledge has TWO separate channels.
prediction_reliability describes prospective probability calibration, not yield
benefit. recommendation.permission controls declared positive use. Accurate
failure forecasts strengthen a scoped warning; they never reopen promotion.
For selected-point knowledge_uses, choose positive_support only when allowed;
tentative_test only for tentative notes; review_test only for a supplied eligible
review candidate; warning for a caution. Omit notes not used in the decision.
Never relabel suppressed positive reasoning as tentative exploration or rename
the claim to escape its adverse evidence. Knowledge directions outside the note's
scope need a separately registered hypothesis, not inherited positive authority.
No candidate is globally banned by a note. Unrelated, history-based reasoning can
still choose any original GP menu point. No sixth point or GP modification.
Return insight in the official schema, knowledge_uses=[{id,version,use}], and
assessments=[{id,version,p_improve_incumbent}] for every applicable non-withdrawn
note. Withdrawn notes may only be cited as historical warnings, without a new
forecast. Forecasts target the selected point exceeding the CURRENT incumbent.
A review is a bounded new experiment inside the existing budget, not evidence
that the direction works. An accurate non-improvement forecast cannot supply
positive support. Provide concise public explanations, not hidden reasoning.
"""


def conditions(pool, candidate_id):
    if candidate_id not in pool.ix:
        raise ValueError("Unknown candidate")
    return dict(zip(pool.axes, pool.features[pool.ix[candidate_id]]))


def public_context(pool, notes, history, ids, query, *, change_thresholds=None):
    if len(ids) != 5 or len(set(ids)) != 5 or set(ids) & {r["candidate_id"] for r in history}:
        raise ValueError("Exactly five distinct unqueried GP candidates are required")
    if len(history) != query - 1:
        raise ValueError("Incorrect query index")
    rows = []
    for note in notes:
        covered = [key for key in ids if control.in_scope(note["scope"], conditions(pool, key))]
        if not covered:
            continue
        public = control.public_note(note)
        public["covered_candidate_ids"] = covered
        reviews = []
        if note["recommendation"]["permission"] == "suppressed":
            for key in covered:
                _, receipt = control.open_review(note, history, query,
                    conditions=conditions(pool, key), thresholds=change_thresholds or {})
                if receipt["opened"]:
                    reviews.append(dict(candidate_id=key, trigger=receipt["event"]["kind"],
                                        event_id=receipt["event"]["event_id"]))
        elif note["recommendation"]["permission"] == "reviewing":
            for key in covered:
                try:
                    control.validate_use(note, conditions(pool, key), "review_test")
                except ValueError:
                    continue
                grant = note["recommendation"]["review"]
                reviews.append(dict(candidate_id=key, trigger=grant["kind"], event_id=grant["event_id"]))
        public["eligible_review_candidates"] = reviews
        rows.append(public)
    return dict(knowledge=rows, current_incumbent=max((r["raw_value"] for r in history), default=None),
                review_availability_is_not_consumed=True,
                prediction_and_recommendation_are_separate=True,
                instructions=SYSTEM_APPENDIX)


def commit_decision(root, pool, notes, history, context, ids, query, raw, *, change_thresholds=None):
    """Validate the official selection and commit separate note controls.

    Invalid declared positive use stops before an oracle call; it is not silently
    changed to exploratory_transfer. The adapter never substitutes another point.
    """
    public_context(pool, notes, history, ids, query, change_thresholds=change_thresholds)
    gp_before = native.gp_hash(context)
    original, _ = native.candidates(pool, context, query)
    if set(ids) != {pool.ids[int(j)] for j in original}:
        raise ValueError("The adapter must use the unchanged native GP top-five menu")
    if (not isinstance(raw, dict) or not isinstance(raw.get("knowledge_uses", []), list)
            or not isinstance(raw.get("assessments", []), list)):
        raise ValueError("Structured knowledge uses and assessments are required")
    selected = validate_insight(root, raw.get("insight"), ids)
    if official_extract(root, raw["insight"]) != selected:
        raise ValueError("Official selection changed")
    physical = conditions(pool, selected)
    by_key = {(n["id"], n["version"]): deepcopy(n) for n in notes}
    if len(by_key) != len(notes):
        raise ValueError("Duplicate note version")
    expected = {key for key, n in by_key.items()
                if n["recommendation"]["permission"] != "withdrawn" and control.in_scope(n["scope"], physical)}
    visible = {key for key, n in by_key.items() if control.in_scope(n["scope"], physical)}
    uses = {}
    for entry in raw.get("knowledge_uses", []):
        if not isinstance(entry, dict) or set(entry) != {"id", "version", "use"}:
            raise ValueError("Invalid declared knowledge use")
        if type(entry["id"]) is not str or type(entry["version"]) is not int or type(entry["use"]) is not str:
            raise ValueError("Knowledge identity and use have invalid types")
        key = (entry["id"], entry["version"])
        if key not in visible or key in uses or entry["use"] == "shadow":
            raise ValueError("Invalid or duplicate knowledge reference")
        note = by_key[key]
        if entry["use"] == "review_test" and note["recommendation"]["permission"] == "suppressed":
            note, receipt = control.open_review(note, history, query, conditions=physical,
                thresholds=change_thresholds or {})
            if not receipt["opened"]:
                raise ValueError("Review is not eligible: " + receipt["reason"])
            by_key[key] = note
        control.validate_use(note, physical, entry["use"])
        uses[key] = entry["use"]
    probabilities = {}
    for entry in raw.get("assessments", []):
        if not isinstance(entry, dict) or set(entry) != {"id", "version", "p_improve_incumbent"}:
            raise ValueError("Invalid prospective assessment")
        if type(entry["id"]) is not str or type(entry["version"]) is not int:
            raise ValueError("Forecast identity has invalid types")
        key = (entry["id"], entry["version"])
        p = entry["p_improve_incumbent"]
        if key not in expected or key in probabilities or type(p) not in (int, float) or not isfinite(p) or not 0 <= p <= 1:
            raise ValueError("Invalid or duplicate probability")
        probabilities[key] = p
    if set(probabilities) != expected:
        raise ValueError("Every applicable non-withdrawn note needs a prospective assessment")
    scale = pool.scale
    if not isfinite(scale) or scale <= 0:
        raise ValueError("Positive native objective scale required")
    if expected and not history:
        raise ValueError("No incumbent improvement forecast exists before the first observation")
    p_gp = native.gp_probability(context, pool.ix[selected], max(r["raw_value"] for r in history) / scale) if history else None
    forecasts = [control.commit_forecast(by_key[key], history, query, selected, physical,
                    probabilities[key], p_gp, use=uses.get(key, "shadow")) for key in sorted(expected)]
    if native.gp_hash(context) != gp_before:
        raise ValueError("Knowledge mutated the GP")
    packet = dict(query=query, selected_id=selected, notes_before_sha256=control.digest(notes),
                  notes_at_commit=list(by_key.values()), history_sha256=control.digest(history),
                  GP_sha256=gp_before, original_menu_ids=list(ids),
                  insight=deepcopy(raw["insight"]), knowledge_uses=deepcopy(raw.get("knowledge_uses", [])),
                  forecasts=forecasts, parameters=deepcopy(control.PARAMETERS),
                  raw_response_sha256=control.digest(raw))
    packet["commit_sha256"] = control.digest(packet)
    return packet


def apply_feedback(notes, history, packet, observation):
    if (packet.get("commit_sha256") != control.digest({k: v for k, v in packet.items() if k != "commit_sha256"})
            or packet["notes_before_sha256"] != control.digest(notes)
            or packet["history_sha256"] != control.digest(history)
            or observation.get("candidate_id") != packet["selected_id"]):
        raise ValueError("Decision commitment mismatch")
    if (packet["query"] != len(history) + 1
            or observation["candidate_id"] in {r["candidate_id"] for r in history}
            or type(observation.get("raw_value")) not in (int, float)
            or not isfinite(observation["raw_value"])):
        raise ValueError("Finite new observation at the committed query required")
    result = deepcopy(packet["notes_at_commit"])
    index = {(n["id"], n["version"]): j for j, n in enumerate(result)}
    events = []
    for forecast in packet["forecasts"]:
        j = index[forecast["id"], forecast["version"]]
        result[j], event = control.feedback(result[j], forecast, history, observation)
        events.append(event)
    # All changes are returned only after every note has passed validation.
    return result, events
