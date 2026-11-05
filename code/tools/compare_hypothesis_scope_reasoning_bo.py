"""Compare completed historical cohorts; never run an optimizer or an LLM."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/reports/hypothesis_scope_vs_reasoning_bo_20261105"
TASKS = ("c1", "c2", "c3", "c4", "perera", "doyle", "snar", "hplc")
LABELS = dict(zip(TASKS, ("Case 1", "Case 2", "Case 3", "Case 4", "Perera", "Doyle", "SnAr", "HPLC")))
OLD = "experiments/runs/cold0_20261102"
OLD_MULTI = "experiments/runs/cold_s123_20261102"
NEW = "experiments/runs/hypothesis_scope_multiseed_20261105"
NEW_REPORT = "experiments/reports/hypothesis_scope_multiseed_20261105"


def read_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def read_csv(path):
    with (ROOT / path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def write_csv(name, rows):
    with (OUT / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def trajectory(path, cohort, method, task, seed, expected_hash=None):
    actual_hash = digest(path)
    if expected_hash is not None:
        assert actual_hash == expected_hash, path
    state = read_json(path)
    assert state["status"] == "COMPLETE", path
    history = state["history"]
    assert len(history) == 20, path
    assert len({item["candidate_id"] for item in history}) == 20, path
    values = [float(item["raw_value"]) for item in history]
    assert all(math.isfinite(value) for value in values), path
    best = []
    for value in values:
        best.append(max(value, best[-1]) if best else value)
    row = dict(cohort=cohort, method=method, task=task, seed=int(seed),
               SearchAUC=statistics.mean(best), final_best=best[-1],
               first_value=values[0], first_final_best_query=values.index(best[-1]) + 1,
               state_path=path, state_sha256=actual_hash)
    return row, best


def main():
    old = read_json(f"{OLD}/protocol.json")
    old_multi = read_json(f"{OLD_MULTI}/protocol.json")
    new = read_json(f"{NEW}/protocol.json")
    assert old["inputs_sha256"] == new["inputs_sha256"]
    assert old_multi["frozen_cold_protocol_sha256"] == old["protocol_sha256"]
    for path, expected in old["inputs_sha256"].items():
        assert digest(path) == expected, path
    source_checks = 0
    for protocol in (old, old_multi, new):
        for path, expected in protocol["sources_sha256"].items():
            assert digest(path) == expected, path
            source_checks += 1
    gp_path = "code/experiment_tools/cold_start_v13/gp.py"
    gp_core = "code/chemist_system/chem/benchmarks/knowledge_gp.py"
    assert old["sources_sha256"][gp_path] == new["sources_sha256"][gp_path]
    assert old["sources_sha256"][gp_core] == new["protected_sha256"][gp_core]
    assert old["model"] == old_multi["model"] == new["model"]
    assert old["effort"] == old_multi["effort"] == new["effort"]

    rows, curves = [], {}
    for task in TASKS:
        for seed in range(4):
            for arm in ("rbo", "bo"):
                path = (f"{OLD}/runs/{task}_{arm}/state.json" if seed == 0 else
                        f"{OLD_MULTI}/s{seed}/runs/ms{seed}_{task}_{arm}/state.json")
                row, best = trajectory(path, "historical", arm, task, seed)
                expected_protocol = old if seed == 0 else old_multi
                assert read_json(path)["protocol_sha256"] == expected_protocol["protocol_sha256"]
                rows.append(row)
                curves[("historical", arm, task, seed)] = best

    metric_errors = []
    for recorded in read_csv(f"{NEW_REPORT}/per_seed_results.csv"):
        row, best = trajectory(recorded["state_path"], "current", recorded["method"],
                               recorded["task"], recorded["seed"], recorded["state_sha256"])
        for field in ("SearchAUC", "final_best"):
            error = abs(row[field] - float(recorded[field]))
            metric_errors.append(error)
            assert error < 1e-8, (recorded["state_path"], field, error)
        rows.append(row)
        curves[("current", row["method"], row["task"], row["seed"])] = best
    assert len(rows) == 160

    def values(cohort, method, task, field):
        selected = [r[field] for r in rows if (r["cohort"], r["method"], r["task"]) == (cohort, method, task)]
        assert len(selected) == 4
        return selected

    for recorded in read_csv("experiments/reports/cold_s123_20261102/summary.csv"):
        method = {"reasoning_bo": "rbo", "bo": "bo"}.get(recorded["method"])
        if method is None:
            continue
        for field in ("SearchAUC", "final_best"):
            actual = values("historical", method, recorded["task"], field)
            assert abs(statistics.mean(actual) - float(recorded[f"{field}_mean"])) < 1e-8
            assert abs(statistics.stdev(actual) - float(recorded[f"{field}_sample_sd"])) < 1e-8

    comparison, seed0 = [], []
    for task in TASKS:
        record = dict(task=task, n_per_method=4)
        for label, cohort, method in (("historical_RBO", "historical", "rbo"),
                                     ("historical_BO", "historical", "bo"),
                                     ("current_hypothesis", "current", "hyp"),
                                     ("current_BO", "current", "bo"),
                                     ("current_same_first_BO", "current", "first")):
            for field in ("SearchAUC", "final_best"):
                actual = values(cohort, method, task, field)
                record[f"{label}_{field}_mean"] = statistics.mean(actual)
                record[f"{label}_{field}_sample_sd"] = statistics.stdev(actual)
        record["historical_RBO_minus_own_BO_AUC"] = record["historical_RBO_SearchAUC_mean"] - record["historical_BO_SearchAUC_mean"]
        record["current_hypothesis_minus_own_BO_AUC"] = record["current_hypothesis_SearchAUC_mean"] - record["current_BO_SearchAUC_mean"]
        record["historical_between_cohort_AUC_difference"] = record["current_hypothesis_SearchAUC_mean"] - record["historical_RBO_SearchAUC_mean"]
        comparison.append(record)
        zero = dict(task=task)
        for cohort, arm, label in (("historical", "rbo", "RBO"), ("current", "hyp", "new"), ("current", "bo", "BO")):
            selected = next(r for r in rows if (r["cohort"], r["method"], r["task"], r["seed"]) == (cohort, arm, task, 0))
            for field in ("SearchAUC", "final_best", "first_value", "first_final_best_query"):
                zero[f"{label}_{field}"] = selected[field]
        assert curves[("historical", "bo", task, 0)] == curves[("current", "bo", task, 0)]
        seed0.append(zero)

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv("per_trajectory_results.csv", rows)
    write_csv("comparison.csv", comparison)
    write_csv("seed0_comparison.csv", seed0)
    comparability = dict(
        strict_paired_multiseed_comparison=False,
        shared=["Identical frozen candidate pools and oracle files (16 hashes verified)",
                "Identical ColdGP and native GP source hashes; GP/MC512 numerical seed 0",
                "Zero free observations plus 20 paid evaluations", "gpt-6.1-terra, medium"],
        differences=dict(
            historical_seed="Only query1 prior/menu changes; post-first exact ties remain seed0",
            current_seed="All-query exact-tie streams change; fixed independent LLM first point and initial prior reused",
            historical_first="LLM chooses one of five data-free candidate inputs",
            current_first="Frozen independent LLM proposal from input-only physical candidate coverage",
            historical_later="LLM chooses from native qLogEI top5 on every query; official Insight prompt",
            current_later="qLogEI maximum within eligible hypothesis scopes; KF and event-driven revisions; at most three new refreshes",
            scientific_background="Current prompts add verified substrate and catalyst identities, molecular graphs and process context; older P0 context kept catalyst labels opaque and fixed substrates unspecified"),
        inference_limit="Historical descriptive comparison, not a causal KF/refresh ablation or evidence of matched-seed statistical superiority",
        metric="SearchAUC = mean of all20 post-query cumulative maxima; no observation is discarded",
        old_protocol_sha256=old["protocol_sha256"], old_multiseed_protocol_sha256=old_multi["protocol_sha256"],
        current_protocol_sha256=new["protocol_sha256"],
        verified_source_hash_entries=source_checks, verified_input_hashes=16,
        recomputed_trajectory_records=len(rows), maximum_current_metric_error=max(metric_errors),
        new_LLM_calls=0, new_experiment_queries=0,
    )
    (OUT / "comparability.json").write_text(json.dumps(comparability, indent=2) + "\n", encoding="utf-8")

    text = ["Historical comparison with Reasoning BO (0+20; seeds labelled 0,1,2,3)", "",
            "This is a descriptive comparison between completed cohorts. Seed definitions, LLM first-choice interfaces and chemical background differ. Do not interpret seed-label matches as paired randomized trials.", "",
            "SearchAUC is the mean of the20 post-query cumulative best raw objective values. Standard deviations are sample SD over four trajectories. Objectives have task-specific units and are never averaged across tasks.", "",
            "Task | historical RBO AUC mean (SD) | current hypothesis AUC mean (SD) | RBO final mean | current final mean"]
    for r in comparison:
        text.append(f"{LABELS[r['task']]} | {r['historical_RBO_SearchAUC_mean']:.3f} ({r['historical_RBO_SearchAUC_sample_sd']:.3f}) | {r['current_hypothesis_SearchAUC_mean']:.3f} ({r['current_hypothesis_SearchAUC_sample_sd']:.3f}) | {r['historical_RBO_final_best_mean']:.3f} | {r['current_hypothesis_final_best_mean']:.3f}")
    text.extend(["", "Each method relative to its own cohort's pure BO (descriptive; not causal difference-in-differences):", "Task | old BO mean AUC | current BO mean AUC | RBO minus old BO | hypothesis minus current BO"])
    for r in comparison:
        text.append(f"{LABELS[r['task']]} | {r['historical_BO_SearchAUC_mean']:.3f} | {r['current_BO_SearchAUC_mean']:.3f} | {r['historical_RBO_minus_own_BO_AUC']:+.3f} | {r['current_hypothesis_minus_own_BO_AUC']:+.3f}")
    text.extend(["", "Interpretation:",
                 "Current mean AUC exceeds historical RBO in7/8 tasks; HPLC is lower. This does not establish stable or matched-setting superiority.",
                 "Doyle has a large pure-BO cohort shift as well as substantial current method variation. Current seed2 is poor (AUC53.594, final55.566); mean improvement is not a robustness claim.",
                 "HPLC has a higher mean final best but a lower AUC than historical RBO. All four current hypothesis paths equal their same-first BO control, so there is no observed selection increment from subsequent hypothesis feedback in HPLC.",
                 "The matched current same-first BO arm is the relevant control for subsequent intervention, with8 AUC wins,16 ties and8 losses across32 task-seed pairs. The historical RBO comparison cannot isolate KF or refresh value.",
                 "Seed0 uses the same numerical/tie streams and identical pure-BO trajectory, but scientific background and first-choice interface still differ.", "",
                 "Mechanisms:", "Historical RBO: propose an Insight and choose from five GP candidates each round; no versioned KF or automated recommendation permission state.",
                 "Current method: independent chemically grounded first point and falsifiable local hypotheses; native GP remains unchanged; eligible scope constrains qLogEI; observed feedback updates KF skill and pauses/retracts unhelpful hypotheses; event-driven calls revise knowledge.", "",
                 "Provenance: per_trajectory_results.csv contains raw-state paths and SHA256 hashes. comparison.csv includes mean, SD, final best and both cohorts' BO controls. seed0_comparison.csv reports the shared-stream historical comparison. comparability.json documents limitations and checks.",
                 "No trajectories, frozen source files or original reports were modified. No new experiments, GP fitting or LLM calls were performed."])
    (OUT / "report.txt").write_text("\n".join(text) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
    for axis, task in zip(axes.flat, TASKS):
        for cohort, arm, label, color in (("historical", "rbo", "Historical RBO", "#4477AA"),
                                         ("current", "hyp", "Current hypothesis", "#CC6677")):
            series = [curves[(cohort, arm, task, seed)] for seed in range(4)]
            mean = [statistics.mean(point) for point in zip(*series)]
            sd = [statistics.stdev(point) for point in zip(*series)]
            axis.plot(range(1, 21), mean, label=label, color=color, linewidth=1.8)
            axis.fill_between(range(1, 21), [m-s for m,s in zip(mean, sd)], [m+s for m,s in zip(mean, sd)], color=color, alpha=.13)
        axis.set_title(LABELS[task])
        axis.set_xlim(1, 20)
        axis.set_xticks([1, 5, 10, 15, 20])
        axis.grid(alpha=.2)
        axis.set_xlabel("Paid query")
    axes[0, 0].set_ylabel("Cumulative best (task units)")
    axes[1, 0].set_ylabel("Cumulative best (task units)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(.5, .94), frameon=False)
    fig.suptitle("Historical 0+20 comparison: four trajectories per task", y=.99)
    fig.text(.5, .015, "Lines: means; bands: sample SD. Seed protocols, first-point interfaces and chemical context differ; not a matched paired test.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .90))
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"cumulative_best.{suffix}", dpi=170)
    plt.close(fig)
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "sha256.json"}
    (OUT / "sha256.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("\n".join(text[:14]))
    print(json.dumps({"audit": "PASS_HISTORICAL_COMPARISON", "records": len(rows), "source_hash_checks": source_checks, "new_LLM_calls": 0, "report": str(OUT)}))


if __name__ == "__main__":
    main()
