"""Meaningful reference-policy checks runnable with stdlib unittest.

Run: python -m unittest discover -s tests -p test_decision.py -v
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from chem.decision.domain_guard import NoAvailableCandidates, select_candidate
from chem.decision.evaluation import run_mechanism_suite
from chem.decision.hypothesis import ParameterSpec, compile_membership, parse_hypotheses
from chem.decision.replay import ReplayOracle, audit_cealge_dataset


class GuardTests(unittest.TestCase):
    def test_joint_conflict_survives_single_constraint_checks(self):
        h = np.array([[1, 0, 1], [0, 1, 1]], dtype=bool)
        result = select_candidate([1, .99, .1], h, .02,
                                  candidate_ids=["a", "b", "c"], hypothesis_ids=["H1", "H2"])
        self.assertEqual(result.retained_ids, ("H1",))
        self.assertEqual(result.revoked_ids, ("H2",))
        self.assertEqual(result.conflict_kind, "quality_exclusion")
        self.assertEqual(result.minimal_conflict_sets, (("H1", "H2"),))
        self.assertTrue(result.subset_checks[1]["eligible_count"])
        self.assertTrue(result.subset_checks[2]["eligible_count"])

    def test_two_revisions_are_needed(self):
        result = select_candidate([1, .99, .98], np.eye(3, dtype=bool), .03,
                                  candidate_ids=["a", "b", "c"], hypothesis_ids=["H1", "H2", "H3"])
        self.assertEqual(len(result.revoked_ids), 2)

    def test_zero_budget_matches_ucb_value_and_advice_breaks_ties(self):
        result = select_candidate([1, 1, .5], np.array([[0, 1, 1]], dtype=bool), 0,
                                  candidate_ids=["a", "b", "c"], hypothesis_ids=["H1"])
        self.assertEqual(result.candidate_id, "b")
        self.assertEqual(result.opportunity_loss, 0)
        self.assertEqual(result.retained_ids, ("H1",))

    def test_hard_and_seen_points_cannot_change_the_reference_maximum(self):
        result = select_candidate([9, 8, 1, .9], np.zeros((0, 4), dtype=bool), .05,
                                  candidate_ids=list("abcd"), hypothesis_ids=[],
                                  hard_mask=np.array([0, 1, 1, 1], dtype=bool),
                                  queried_mask=np.array([0, 1, 0, 0], dtype=bool))
        self.assertEqual(result.candidate_id, "c")
        self.assertEqual(result.global_ucb, 1)

    def test_input_order_does_not_change_fixed_id_tie_breaking(self):
        h = np.array([[1, 0], [0, 1]], dtype=bool)
        first = select_candidate([1, 1], h, 0, candidate_ids=["b", "a"], hypothesis_ids=["H2", "H1"])
        second = select_candidate([1, 1], h[::-1, ::-1], 0,
                                  candidate_ids=["a", "b"], hypothesis_ids=["H1", "H2"])
        self.assertEqual(first.candidate_id, second.candidate_id)
        self.assertEqual(first.retained_ids, ("H1",))

    def test_rejects_invalid_scores_budgets_dimensions_and_exhaustion(self):
        arguments = dict(candidate_ids=["a"], hypothesis_ids=[])
        h = np.zeros((0, 1), dtype=bool)
        for scores, budget in (([float("nan")], .1), ([1], -1), ([1], float("inf"))):
            with self.subTest(scores=scores, budget=budget), self.assertRaises(ValueError):
                select_candidate(scores, h, budget, **arguments)
        with self.assertRaises(ValueError):
            select_candidate([1], np.zeros((1, 1)), 0, **arguments)
        with self.assertRaises(NoAvailableCandidates):
            select_candidate([1], h, 0, queried_mask=np.array([True]), **arguments)

    def test_exhaustive_independent_reference(self):
        receipt = run_mechanism_suite(random_states=250)
        self.assertEqual(receipt["fixture_count"], 20)
        self.assertEqual(receipt["total_decisions_checked"], 270)
        self.assertEqual(receipt["minimum_revocation_or_tie_break_mismatches"], 0)


class HypothesisTests(unittest.TestCase):
    def setUp(self):
        self.schema = {"temperature": ParameterSpec("temperature", "degC", (800, 1400)),
                       "hours": ParameterSpec("hours", "h", (0, 12)),
                       "catalyst": ParameterSpec("catalyst", "category", levels=("A", "B"))}
        self.row = {"id": "H1", "kind": "interval", "parameter": "temperature", "unit": "degC",
                    "lower": 1100, "upper": 1250, "evidence_ids": ["source1"], "observation_ids": [],
                    "rationale": "An explicitly provisional temperature-window hypothesis."}

    def parse(self, rows):
        return parse_hypotheses({"version": 1, "hypotheses": rows}, self.schema,
                                evidence_ids={"source1"}, observed_ids=set())

    def test_structured_numeric_categorical_and_interaction_predicates(self):
        category = {**self.row, "id": "H2", "kind": "allowed_values", "parameter": "catalyst",
                    "unit": "category", "values": ["A"]}
        category.pop("lower"); category.pop("upper")
        linear = {key: value for key, value in self.row.items() if key not in {"parameter", "lower", "upper"}}
        linear.update(id="H3", kind="linear_le", unit="normalized",
                      coefficients={"temperature": 1, "hours": 1}, rhs=1.25)
        hypotheses = self.parse([self.row, category, linear])
        h = compile_membership(hypotheses, [{"temperature": 1200, "hours": 3, "catalyst": "A"},
                                          {"temperature": 1300, "hours": 12, "catalyst": "B"}], self.schema)
        np.testing.assert_array_equal(h, [[1, 0], [1, 0], [1, 0]])
        self.assertFalse(h.flags.writeable)
        with self.assertRaises(FrozenInstanceError):
            hypotheses[0].version = 2

    def test_rejects_generated_code_unit_errors_and_unseen_evidence(self):
        bad = [{**self.row, "code": "raise RuntimeError('must never execute')"},
               {**self.row, "unit": "K"}, {**self.row, "evidence_ids": ["unknown"]},
               {**self.row, "observation_ids": ["future_observation"]},
               {**self.row, "lower": 700}, {**self.row, "lower": 1300, "upper": 1200}]
        for row in bad:
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.parse([row])
        with self.assertRaises(ValueError):
            self.parse([{**self.row, "id": str(j)} for j in range(5)])
        with self.assertRaises(ValueError):
            self.parse([self.row, {**self.row, "id": "duplicate_semantics"}])

    def test_missing_parameters_are_not_filled_with_midpoints(self):
        with self.assertRaises(ValueError):
            compile_membership(self.parse([self.row]), [{"temperature": 1200}], self.schema)


class ReplayTests(unittest.TestCase):
    def test_failed_fit_is_unknown_but_measured_zero_is_preserved(self):
        common = {"charge_ratio": {"Ce": 10, "Al": 70, "Ge": 30}, "yield": 0,
                  "yield_sigma": 3, "use_for_bo": True}
        rows = [{**common, "file": "failed.xy", "confidence": "failed_current_model"},
                {**common, "file": "measured_zero.xy", "confidence": "pass"},
                {**common, "file": "heated-DSC.xy", "confidence": "pass"}]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "dataset.json"
            original = json.dumps({"target": "CeAl2Ge2", "samples": rows}).encode("utf-8")
            path.write_bytes(original)
            report = audit_cealge_dataset(path)
            self.assertEqual(path.read_bytes(), original)
        self.assertIsNone(report["records"][0]["physical_yield_pct"])
        self.assertEqual(report["records"][1]["physical_yield_pct"], 0)
        self.assertEqual(report["provisionally_eligible_count"], 1)
        self.assertAlmostEqual(sum(report["records"][1]["recomputed_atom_fraction_assuming_mass_ratio"].values()), 1)
        self.assertFalse(report["ready_for_scientific_replay"])

    def test_exact_oracle_rejects_duplicates_and_unseen_candidates(self):
        oracle = ReplayOracle(["a", "b"], [.1, .9])
        self.assertEqual(oracle.query("a"), .1)
        with self.assertRaises(ValueError):
            oracle.query("a")
        with self.assertRaises(ValueError):
            oracle.query("unknown")
        external_copy = oracle.ledger
        external_copy[0]["value"] = 42
        self.assertEqual(oracle.ledger[0]["value"], .1)


class CLITests(unittest.TestCase):
    def test_new_commands_use_existing_entrypoint(self):
        from chem.cli import build_parser
        args = build_parser().parse_args(["decision", "mechanism", "--random-states", "5"])
        self.assertEqual(args.action, "mechanism")
        self.assertEqual(args.random_states, 5)


if __name__ == "__main__":
    unittest.main()
