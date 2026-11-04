"""Run local control tests and fixed-log state replay; never run optimization.

Replay holds historical forecasts, choices and outcomes fixed. It is NOT a
counterfactual optimization trajectory or evidence of SearchAUC improvement.
"""
from collections import Counter
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch
import csv
import json
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "code/chemist_system"), str(ROOT / "code/experiment_tools")]
from chem.benchmarks import knowledge_gp as gpmod
from reasoning_memory_v13 import control, adapter

OUT = ROOT / "experiments/reports/v13_state_split_20261102"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha(path):
    return sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frozen_path = ROOT / "experiments/runs/knowledge_loop_20261101/v12_reasoning_memory/protocol.json"
    frozen = read(frozen_path)
    protected = dict(frozen["source_sha256"])
    protected[frozen["design_path"]] = frozen["design_sha256"]
    protected[frozen_path.relative_to(ROOT).as_posix()] = file_sha(frozen_path)
    for name in ("paper/v12/data/evidence.json", "paper/v12_subagent/data/evidence.json"):
        protected[name] = file_sha(ROOT / name)
        for row in read(ROOT / name)["rows"]:
            protected[row["source"]] = row["source_sha256"]
    drift_before = [name for name, expected in protected.items() if file_sha(ROOT / name) != expected]
    if drift_before:
        raise ValueError("Existing frozen source drift: " + repr(drift_before))
    tests = unittest.defaultTestLoader.discover(
        str(ROOT / "code/experiment_tools/reasoning_memory_v13/tests"), pattern="test_*.py")
    split_tests = tests.countTestCases()
    original_tests = unittest.defaultTestLoader.loadTestsFromName("reasoning_memory_v12.tests.test_pipeline")
    original_count = original_tests.countTestCases()
    tests.addTests(original_tests)
    with (OUT / "unit_tests.log").open("w", encoding="utf-8") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(tests)
    if not result.wasSuccessful():
        raise RuntimeError("State-control tests failed; inspect unit_tests.log")
    evidence = read(ROOT / "paper/v12_subagent/data/evidence.json")
    rows, example, pools = [], [], {}
    # Public inputs only. Make either forbidden operation a hard failure.
    with patch.object(gpmod, "read_observation", side_effect=AssertionError("No oracle reads in replay")), \
         patch.object(gpmod, "GPAdapter", side_effect=AssertionError("No GP fits in replay")):
        runs = [r for r in evidence["rows"] if r.get("arm") == "memory"]
        for registered in runs:
            state_path = ROOT / registered["source"]
            state = read(state_path)
            task, seed = registered["task"], registered["initialization_seed"]
            if task not in pools:
                pools[task] = gpmod.load_task(ROOT, task, seed=0)
                protected[pools[task].source_path.relative_to(ROOT).as_posix()] = file_sha(pools[task].source_path)
            pool = pools[task]
            notes = {}
            definitions = sorted(state["hypotheses"], key=lambda h: (h["created_round"], h["id"], h["version"]))
            for decision in state["decisions"]:
                q = decision["query"]
                for definition in (h for h in definitions if h["created_round"] == q):
                    new = control.new_note(definition["id"], definition["version"], definition["effect"],
                        definition["scope"], definition["focus"], created_query=q, predecessors=list(notes.values()))
                    for key in list(notes):
                        if key[0] == definition["id"]:
                            notes[key] = control.withdraw(notes[key], q, "superseded_by_historical_revision")
                    notes[definition["id"], definition["version"]] = new
                history = state["history"][:4 + q - 1]
                observed = state["history"][4 + q - 1]
                references = {(r["id"], r["version"]): r for r in decision.get("memory_references", [])}
                for original in decision["KF_feedback"]:
                    key = (original["id"], original["version"])
                    note = notes[key]
                    ref = references.get(key, {})
                    positive_request = ref.get("requested_use", ref.get("use")) in ("support", "exploratory_transfer")
                    use = "warning" if ref.get("use") == "counterevidence" else "shadow"
                    blocked = False
                    if note["effect"] == "promote" and positive_request:
                        permission = note["recommendation"]["permission"]
                        if permission == "allowed":
                            use = "positive_support"
                        elif permission == "tentative":
                            use = "tentative_test"
                        else:
                            # This is a diagnostic replay of a fixed historical
                            # outcome, not permission to execute the old choice.
                            blocked = True
                            use = "shadow"
                    packet = control.commit_forecast(note, history, q, original["point_id"],
                        adapter.conditions(pool, original["point_id"]), original["p_LLM"], original["p_GP"], use=use)
                    updated, event = control.feedback(note, packet, history, observed)
                    if event["improved"] != original["success"]:
                        raise ValueError("Current-incumbent target mismatch")
                    if abs(event["calibration"]["Brier_gain"] - original["Brier_gain"]) > 1e-12:
                        raise ValueError("Pure calibration formula changed")
                    for field in ("mean", "variance", "probability_positive"):
                        if abs(updated["calibration"][field] - original["after"][field]) > 1e-12:
                            raise ValueError("Numeric KF posterior changed")
                    if not event["improved"] and updated["recommendation"]["permission"] == "allowed":
                        # A first unpredicted failure can leave existing support
                        # intact; an accurate failure must never restore it.
                        if event["accurate_failure_forecast"]:
                            raise ValueError("Accurate failure reopened positive permission")
                    notes[key] = updated
                    row = dict(task=task, seed=seed, query=q, note_id=key[0], version=key[1],
                        old_status=original["after"]["status"], old_event=original["reason"],
                        old_weight=original["after"]["weight"], p_LLM=original["p_LLM"],
                        p_GP=original["p_GP"], improved=event["improved"], Brier_gain=original["Brier_gain"],
                        prediction_reliability=updated["calibration"]["reliability"],
                        recommendation_permission=updated["recommendation"]["permission"],
                        positive_test_count=len(updated["recommendation"]["positive_tests"]),
                        accurate_failure_evidence=updated["recommendation"]["accurate_failure_count"],
                        replay_use=use, historical_positive_use_would_require_new_authorization=blocked,
                        review_authorization="none_retroactively_granted", source=registered["source"])
                    rows.append(row)
                    if task == "HPLC" and seed == 1 and key == ("H001", 1) and q in (7, 8, 9, 10):
                        example.append(dict(row=row, public_context=control.public_note(updated)))
    drift_after = [name for name, expected in protected.items() if file_sha(ROOT / name) != expected]
    if drift_after:
        raise ValueError("Frozen files changed: " + repr(drift_after))
    with (OUT / "fixed_log_state_replay.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(OUT / "hplc_recovery_example.json", example)
    report = dict(status="PASS_STATE_SEPARATION_ONLY", tests_run=result.testsRun,
        state_split_tests=split_tests, frozen_v12_regression_tests=original_count,
        failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
        replay_trajectories=len(runs), replay_forecast_events=len(rows),
        historical_auto_resumes=sum(r["old_event"] == "automatic_resume" for r in rows),
        positive_permission_after_historical_auto_resumes=[r["recommendation_permission"] for r in rows if r["old_event"] == "automatic_resume"],
        replay_permissions=dict(Counter(r["recommendation_permission"] for r in rows)),
        historical_declared_positive_uses_needing_new_authorization=sum(r["historical_positive_use_would_require_new_authorization"] for r in rows),
        frozen_files_verified=len(protected), frozen_drift=drift_after,
        parameters=control.PARAMETERS, new_experiments=0, new_LLM_requests=0, new_GP_fits=0,
        claim_boundary="Fixed-log replay validates state logic only. No new choices, optimization trajectories, SearchAUC values, or causal performance estimates were generated.",
        not_activated_in_frozen_V12=True,
        sources_sha256={p.relative_to(ROOT).as_posix(): file_sha(p) for p in sorted((ROOT / "code/experiment_tools/reasoning_memory_v13").rglob("*.py"))},
        implementation_note_sha256=file_sha(ROOT / "docs/knowledge_state_split_implementation_20261102.txt"),
        protected_sha256=protected, script_sha256=file_sha(Path(__file__)))
    write_json(OUT / "audit.json", report)
    print(json.dumps({k: report[k] for k in ("status", "tests_run", "replay_trajectories", "replay_forecast_events",
        "historical_auto_resumes", "positive_permission_after_historical_auto_resumes", "frozen_files_verified", "frozen_drift")}, indent=2))
    print(OUT)


if __name__ == "__main__":
    main()
