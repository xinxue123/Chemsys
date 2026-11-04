from copy import deepcopy
from pathlib import Path
import unittest

from reasoning_memory_v13 import calibration, control, adapter
from knowledge_information_v10.tests.test_pipeline import fixture, context
from knowledge_rank_v9 import kf as frozen_kf


ROOT = Path(__file__).resolve().parents[4]


def initial_history():
    return [dict(candidate_id=f"initial{i}", raw_value=float(i + 1)) for i in range(4)]


def new_note(effect="promote"):
    return control.new_note("H001", 1, effect, {}, ["x", "category"])


def observe(note, history, value, *, p=.1, g=.6, use="shadow", x=.2, category="A"):
    q = len(history) - 3
    packet = control.commit_forecast(note, history, q, f"new{q}",
        dict(x=x, category=category), p, g, use=use)
    observation = dict(candidate_id=f"new{q}", raw_value=float(value))
    after, event = control.feedback(note, packet, history, observation)
    return after, history + [observation], event


def suppressed_after_successes():
    note, history = new_note(), initial_history()
    for value in (5, 6):
        note, history, _ = observe(note, history, value, p=.8, g=.5, use="tentative_test")
    assert note["recommendation"]["permission"] == "allowed"
    note, history, _ = observe(note, history, 3, use="positive_support")
    return note, history


def plateau():
    note, history = suppressed_after_successes()
    for value in (3, 3):
        note, history, _ = observe(note, history, value)
    return note, history


def raw_response(ids, notes, uses=None):
    return dict(insight=dict(comment="Public comparison", keywords="test", hypotheses=[
        dict(strategy="experiment", rationale="Test observed trend", confidence=c,
             parameter_sets=[dict(candidate_id=i)])
        for i, c in zip(ids[:3], ["medium", "high", "low"])]),
        knowledge_uses=uses or [], assessments=[dict(id=n["id"], version=n["version"],
            p_improve_incumbent=.2) for n in notes if n["recommendation"]["permission"] != "withdrawn"])


class CalibrationTests(unittest.TestCase):
    def test_same_numerics_as_original_kf_without_original_lifecycle(self):
        new, old = calibration.initial(), frozen_kf.initial()
        for q, (p, g, s) in enumerate([(.2, .6, False), (.8, .7, True), (.1, .9, True)], 1):
            new, event = calibration.update(new, p, g, s)
            old, legacy = frozen_kf.update(old, p, g, s, q)
            for key in ("mean", "variance", "probability_positive"):
                self.assertAlmostEqual(new[key], old[key], places=13)
            self.assertAlmostEqual(event["Brier_gain"], legacy["Brier_gain"], places=14)
        self.assertNotIn("status", new)
        self.assertNotIn("good_streak", new)

    def test_accurate_failure_raises_reliability(self):
        before = calibration.initial()
        after, _ = calibration.update(before, .1, .6, False)
        self.assertGreater(after["reliability"], before["reliability"])
        self.assertEqual(before, calibration.initial())

    def test_invalid_probabilities_cannot_mutate_state(self):
        for p in (float("nan"), float("inf"), -1., 2., True):
            state = calibration.initial()
            with self.assertRaises(ValueError):
                calibration.update(state, p, .5, True)
            self.assertEqual(state, calibration.initial())


class RecommendationTests(unittest.TestCase):
    def test_successful_tests_can_support_direction_despite_poor_calibration(self):
        note, history = new_note(), initial_history()
        for value in (5, 6):
            note, history, event = observe(note, history, value, p=.1, g=.9, use="tentative_test")
            self.assertLess(event["calibration"]["Brier_gain"], 0)
        self.assertEqual(note["recommendation"]["permission"], "allowed")
        self.assertLess(note["calibration"]["reliability"], .4)

    def test_accurate_failures_do_not_restore_old_positive_authority(self):
        note, history = suppressed_after_successes()
        self.assertEqual(note["recommendation"]["positive_tests"], [])
        previous = note["calibration"]["reliability"]
        for _ in range(4):
            note, history, _ = observe(note, history, 3)
        self.assertGreater(note["calibration"]["reliability"], previous)
        self.assertEqual(note["recommendation"]["permission"], "suppressed")
        self.assertGreater(note["recommendation"]["accurate_failure_count"], 0)
        self.assertFalse(control.public_note(note)["recommendation"]["positive_support_allowed"])

    def test_avoid_note_gains_warning_without_promotional_permission(self):
        note, history, _ = observe(new_note("avoid"), initial_history(), 2)
        public = control.public_note(note)
        self.assertTrue(public["recommendation"]["risk_warning_available"])
        self.assertFalse(public["recommendation"]["positive_support_allowed"])
        self.assertEqual(note["recommendation"]["permission"], "not_applicable")

    def test_unreferenced_success_is_not_credited_as_tested_reason(self):
        note, history = new_note(), initial_history()
        for v in (5, 6, 7):
            note, history, _ = observe(note, history, v, p=.8, g=.5, use="shadow")
        self.assertEqual(note["recommendation"]["positive_tests"], [])
        self.assertEqual(note["recommendation"]["permission"], "tentative")

    def test_unexpected_failures_also_suspend_an_unproductive_declared_direction(self):
        note, history = new_note(), initial_history()
        for value in (2, 3):
            note, history, _ = observe(note, history, value, p=.9, g=.5, use="tentative_test")
        self.assertEqual(note["recommendation"]["permission"], "suppressed")

    def test_no_event_means_no_review_despite_high_reliability(self):
        note, history = suppressed_after_successes()
        after, receipt = control.open_review(note, history, len(history) - 3)
        self.assertFalse(receipt["opened"])
        self.assertEqual(after, note)

    def test_stagnation_only_opens_review_then_two_new_tests_can_restore(self):
        note, history = plateau()
        note, receipt = control.open_review(note, history, len(history) - 3)
        self.assertTrue(receipt["opened"])
        self.assertEqual(note["recommendation"]["permission"], "reviewing")
        self.assertFalse(control.public_note(note)["recommendation"]["positive_support_allowed"])
        note, history, _ = observe(note, history, 7, p=.1, g=.9, use="review_test")
        self.assertEqual(note["recommendation"]["permission"], "reviewing")
        note, history, _ = observe(note, history, 8, p=.1, g=.9, use="review_test")
        self.assertEqual(note["recommendation"]["permission"], "allowed")
        self.assertEqual(note["recommendation"]["review_tests_used"], 2)

    def test_failed_review_reblocks_and_same_plateau_cannot_reopen(self):
        note, history = plateau()
        note, _ = control.open_review(note, history, len(history) - 3)
        note, history, _ = observe(note, history, 3, use="review_test")
        self.assertEqual(note["recommendation"]["permission"], "suppressed")
        same, receipt = control.open_review(note, history, len(history) - 3)
        self.assertEqual(receipt["reason"], "event_already_consumed")
        self.assertEqual(same, note)

    def test_condition_change_allows_distinct_confirming_points_in_new_regime(self):
        note, history = suppressed_after_successes()
        note, receipt = control.open_review(note, history, len(history) - 3,
            conditions=dict(x=.8, category="A"), thresholds={"x": .3})
        self.assertTrue(receipt["opened"])
        self.assertEqual(receipt["event"]["kind"], "condition_change")
        with self.assertRaises(ValueError):
            control.validate_use(note, dict(x=.3, category="A"), "review_test")
        note, history, _ = observe(note, history, 7, x=.8, use="review_test")
        note, history, _ = observe(note, history, 8, x=.9, use="review_test")
        self.assertEqual(note["recommendation"]["permission"], "allowed")

    def test_unrelated_or_tiny_changes_do_not_unlock(self):
        note, history = suppressed_after_successes()
        note["focus"] = ["x"]
        for cond in (dict(x=.2, category="B"), dict(x=.21, category="A")):
            after, receipt = control.open_review(note, history, len(history) - 3,
                conditions=cond, thresholds={"x": .3})
            self.assertFalse(receipt["opened"])
            self.assertEqual(after, note)

    def test_repeated_pause_does_not_reset_clock_or_erase_evidence(self):
        note, _ = suppressed_after_successes()
        after = control.suppress(note, 99, "repeat")
        self.assertEqual(after["recommendation"]["suppressed_query"], note["recommendation"]["suppressed_query"])
        self.assertEqual(after["recommendation"]["risk_facts"], note["recommendation"]["risk_facts"])

    def test_new_id_or_focus_carries_negative_facts_but_not_kf_scores(self):
        note, _ = suppressed_after_successes()
        revised = control.new_note("H099", 1, "promote", {}, ["category"], predecessors=[note])
        self.assertEqual(revised["recommendation"]["permission"], "suppressed")
        self.assertEqual(revised["calibration"], calibration.initial())
        self.assertTrue(revised["recommendation"]["risk_facts"])

    def test_revision_cannot_erase_consumed_review_event_or_quota(self):
        note, history = plateau()
        note, _ = control.open_review(note, history, len(history) - 3)
        note, history, _ = observe(note, history, 3, use="review_test")
        revised = control.new_note("H099", 1, "promote", {}, ["x"],
            created_query=len(history) - 3, predecessors=[note])
        self.assertEqual(revised["recommendation"]["review_tests_used"], 1)
        _, receipt = control.open_review(revised, history, len(history) - 3)
        self.assertEqual(receipt["reason"], "event_already_consumed")

    def test_withdrawal_does_not_zero_calibration_and_cannot_auto_restore(self):
        note, history = plateau()
        retired = control.withdraw(note, len(history) - 3, "explicit evidence review")
        self.assertEqual(retired["calibration"], note["calibration"])
        same, receipt = control.open_review(retired, history, len(history) - 3)
        self.assertFalse(receipt["opened"])
        with self.assertRaises(ValueError):
            observe(same, history, 8)
        control.validate_use(retired, dict(x=.2, category="A"), "warning")

    def test_reused_or_modified_forecast_cannot_update_twice(self):
        note, history = new_note(), initial_history()
        packet = control.commit_forecast(note, history, 1, "new1", dict(x=.2), .1, .6)
        observation = dict(candidate_id="new1", raw_value=2.)
        after, _ = control.feedback(note, packet, history, observation)
        snapshot = deepcopy(after)
        with self.assertRaises(ValueError):
            control.feedback(after, packet, history, observation)
        self.assertEqual(after, snapshot)
        modified = deepcopy(packet)
        modified["incumbent"] = 1.
        with self.assertRaises(ValueError):
            control.feedback(note, modified, history, observation)

    def test_public_context_cannot_present_reliability_as_recommendation_weight(self):
        public = control.public_note(new_note())
        self.assertNotIn("kf", public)
        self.assertIn("prediction_reliability", public)
        self.assertIn("recommendation", public)
        self.assertFalse(public["recommendation"]["positive_support_allowed"])


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.pool, self.state = fixture()
        self.notes = [control.new_note("H001", 1, "promote", {}, ["x"])]
        self.ctx = context(self.pool, self.state)
        self.ids = [self.pool.ids[j] for j in range(4, 9)]

    def test_native_selection_and_gp_are_unchanged_for_valid_inputs(self):
        raw = raw_response(self.ids, self.notes)
        before = adapter.native.gp_hash(self.ctx)
        packet = adapter.commit_decision(ROOT, self.pool, self.notes, self.state["history"], self.ctx, self.ids, 1, raw)
        self.assertEqual(packet["selected_id"], adapter.official_extract(ROOT, raw["insight"]))
        self.assertEqual(before, adapter.native.gp_hash(self.ctx))
        notes, events = adapter.apply_feedback(self.notes, self.state["history"], packet,
            dict(candidate_id=packet["selected_id"], raw_value=10.))
        self.assertEqual(events[0]["recommendation_after"], "suppressed")
        self.assertEqual(self.notes[0]["calibration"]["updates"], 0)
        self.assertEqual(notes[0]["calibration"]["updates"], 1)

    def test_illegal_positive_or_exploratory_use_stops_before_observation(self):
        self.notes[0] = control.suppress(self.notes[0], 0, "known risk")
        snapshot = deepcopy(self.notes)
        for use in ("positive_support", "tentative_test", "review_test"):
            raw = raw_response(self.ids, self.notes, [dict(id="H001", version=1, use=use)])
            with self.assertRaises(ValueError):
                adapter.commit_decision(ROOT, self.pool, self.notes, self.state["history"], self.ctx, self.ids, 1, raw)
        self.assertEqual(snapshot, self.notes)

    def test_warning_does_not_ban_an_independently_selected_candidate(self):
        self.notes[0] = control.suppress(self.notes[0], 0, "known risk")
        raw = raw_response(self.ids, self.notes, [dict(id="H001", version=1, use="warning")])
        packet = adapter.commit_decision(ROOT, self.pool, self.notes, self.state["history"], self.ctx, self.ids, 1, raw)
        self.assertEqual(packet["selected_id"], self.ids[1])

    def test_retired_facts_can_warn_without_reviving_forecasts(self):
        raw = raw_response(self.ids, self.notes)
        packet = adapter.commit_decision(ROOT, self.pool, self.notes, self.state["history"], self.ctx, self.ids, 1, raw)
        observation = dict(candidate_id=packet["selected_id"], raw_value=10., y=.1)
        notes, _ = adapter.apply_feedback(self.notes, self.state["history"], packet, observation)
        history = self.state["history"] + [observation]
        notes[0] = control.withdraw(notes[0], 2, "retain adverse facts only")
        ctx = context(self.pool, dict(history=history))
        ids = [i for i in self.pool.ids if i not in {r["candidate_id"] for r in history}][:5]
        raw = raw_response(ids, notes, [dict(id="H001", version=1, use="warning")])
        packet = adapter.commit_decision(ROOT, self.pool, notes, history, ctx, ids, 2, raw)
        self.assertEqual(packet["forecasts"], [])
        self.assertEqual(packet["knowledge_uses"][0]["use"], "warning")

    def test_missing_forecast_cannot_silently_disappear(self):
        raw = raw_response(self.ids, self.notes)
        raw["assessments"] = []
        with self.assertRaises(ValueError):
            adapter.commit_decision(ROOT, self.pool, self.notes, self.state["history"], self.ctx, self.ids, 1, raw)

    def test_official_choice_cannot_use_a_replaced_gp_menu(self):
        changed = self.ids[:-1] + ["p10"]
        with self.assertRaises(ValueError):
            adapter.commit_decision(ROOT, self.pool, self.notes, self.state["history"], self.ctx,
                changed, 1, raw_response(changed, self.notes))

    def test_boolean_version_cannot_impersonate_integer_note_version(self):
        raw = raw_response(self.ids, self.notes)
        raw["assessments"][0]["version"] = True
        with self.assertRaises(ValueError):
            adapter.commit_decision(ROOT, self.pool, self.notes, self.state["history"], self.ctx, self.ids, 1, raw)

    def test_prompt_exposes_two_channels_and_is_read_only(self):
        snapshot = deepcopy(self.notes)
        public = adapter.public_context(self.pool, self.notes, self.state["history"], self.ids, 1)
        row = public["knowledge"][0]
        self.assertIn("prediction_reliability", row)
        self.assertIn("recommendation", row)
        self.assertEqual(snapshot, self.notes)

    def test_pending_confirmation_remains_visible_as_executable_review(self):
        note, history = plateau()
        note, _ = control.open_review(note, history, len(history) - 3)
        note, history, _ = observe(note, history, 7, use="review_test")
        public = adapter.public_context(self.pool, [note], history, self.ids, len(history) - 3)
        self.assertEqual(len(public["knowledge"][0]["eligible_review_candidates"]), 5)

    def test_invalid_observation_is_rejected_even_without_any_notes(self):
        raw = raw_response(self.ids, [])
        packet = adapter.commit_decision(ROOT, self.pool, [], self.state["history"], self.ctx, self.ids, 1, raw)
        with self.assertRaises(ValueError):
            adapter.apply_feedback([], self.state["history"], packet,
                dict(candidate_id=packet["selected_id"], raw_value=float("nan")))

    def test_sixteen_synthetic_steps_commit_before_feedback_and_compile_once(self):
        adapter.compiled_extractor.cache_clear()
        notes, history = deepcopy(self.notes), deepcopy(self.state["history"])
        selected = []
        for query in range(1, 17):
            ctx = context(self.pool, dict(history=history))
            ids = [i for i in self.pool.ids if i not in {r["candidate_id"] for r in history}][:5]
            raw = raw_response(ids, notes)
            packet = adapter.commit_decision(ROOT, self.pool, notes, history, ctx, ids, query, raw)
            observation = dict(candidate_id=packet["selected_id"], raw_value=20., y=.2)
            notes, events = adapter.apply_feedback(notes, history, packet, observation)
            history.append(observation)
            selected.append(packet["selected_id"])
        self.assertEqual(len(history), 20)
        self.assertEqual(len(set(selected)), 16)
        self.assertEqual(adapter.compiled_extractor.cache_info().misses, 1)
        self.assertEqual(notes[0]["recommendation"]["permission"], "suppressed")


if __name__ == "__main__":
    unittest.main()
