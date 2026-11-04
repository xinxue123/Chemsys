"""Read-only metric audit; never imports an optimizer or requests new outcomes."""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json
import math


ROOT = Path(__file__).resolve().parents[2]
INPUTS = (
    "paper/v12/data/evidence.json",
    "paper/v12_subagent/data/evidence.json",
)
OUT = ROOT / "experiments/reports/search_auc_denominator_check_20261102/audit.json"


def main():
    failures, checked, source_hashes, input_hashes = [], [], {}, {}
    for evidence_path in INPUTS:
        content = (ROOT / evidence_path).read_bytes()
        input_hashes[evidence_path] = sha256(content).hexdigest()
        evidence = json.loads(content.decode("utf-8"))
        for row in evidence["rows"]:
            source = row["source"]
            raw = (ROOT / source).read_bytes()
            digest = sha256(raw).hexdigest()
            source_hashes[source] = digest
            state = json.loads(raw.decode("utf-8"))
            history, decisions = state["history"], state["decisions"]
            values = [item.get("raw_value") for item in history]
            checks = {
                "source_matches_frozen_hash": digest == row["source_sha256"],
                "complete": state["status"] == "COMPLETE",
                "twenty_measured_points": len(values) == 20,
                "sixteen_decisions": len(decisions) == 16,
                "finite_numeric_values": all(
                    type(v) in (int, float) and math.isfinite(v) for v in values
                ),
                "unique_measured_ids": len({h["candidate_id"] for h in history}) == len(history),
                "initial_ids_match": [h["candidate_id"] for h in history[:4]] == row["initial_ids"],
                "selected_ids_match": [h["candidate_id"] for h in history[4:]] == row["selected_ids"],
                "saved_values_match": values == row["values"],
            }
            result = dict(evidence=evidence_path, source=source, task=row["task"],
                          method=row["method"], measured_points=len(values),
                          post_initial_points=len(values) - 4, decisions=len(decisions),
                          reported_SearchAUC=row["SearchAUC"])
            if checks["finite_numeric_values"] and len(values) > 4:
                initial_best = max(values[:4])
                # One independently constructed prefix maximum per actual query.
                curve = [max(values[:4 + t]) for t in range(1, len(values) - 3)]
                numerator = math.fsum(curve)
                by_actual_count = numerator / len(curve)
                by_protocol_budget = numerator / 16
                expected_plot = [initial_best] + curve
                checks.update(
                    actual_count_equals_protocol_budget=len(curve) == 16,
                    recorded_curve_has_q0_plus_queries=row["best_curve"] == expected_plot,
                    recomputed_matches_report=math.isclose(
                        by_actual_count, row["SearchAUC"], rel_tol=0, abs_tol=1e-9),
                    two_denominators_agree=by_actual_count == by_protocol_budget,
                )
                result.update(initial_best=initial_best, numerator=numerator,
                              actual_denominator=len(curve), protocol_denominator=16,
                              plot_points_including_q0=len(expected_plot),
                              recomputed_SearchAUC=by_actual_count,
                              absolute_error=abs(by_actual_count - row["SearchAUC"]),
                              zero_observations=sum(v == 0 for v in values),
                              unchanged_incumbent_queries=sum(
                                  curve[i] == ([initial_best] + curve)[i]
                                  for i in range(len(curve))))
            for check, passed in checks.items():
                if not passed:
                    failures.append(dict(evidence=evidence_path, source=source, check=check))
            result["checks"] = checks
            checked.append(result)
    report = dict(
        status="PASS" if not failures else "FAIL", scope=list(INPUTS),
        metric_rows=len(checked), unique_trajectories=len(source_hashes),
        rows_by_evidence=dict(Counter(x["evidence"] for x in checked)),
        actual_denominator_counts=dict(Counter(x.get("actual_denominator") for x in checked)),
        maximum_absolute_error=max(x.get("absolute_error", 0) for x in checked),
        failures=failures, input_sha256=input_hashes, source_sha256=source_hashes,
        definition="Mean of post-query cumulative maxima, including initial observations in each maximum; q0 is not averaged.",
        incomplete_runs="Not accepted by the manuscript record function; no missing-value filtering or padding in this audit.",
        new_experiments=0, new_LLM_requests=0, new_GP_fits=0,
        rows=checked, script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "status", "metric_rows", "unique_trajectories", "actual_denominator_counts",
        "maximum_absolute_error", "failures")}, indent=2))
    print(OUT)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
