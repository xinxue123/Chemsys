"""Verify the extra data-source, S8 and Test G evidence and reproduce the two sensitivity checks."""

import argparse
from collections import defaultdict
import csv
import hashlib
from itertools import accumulate, product
import json
import math
from pathlib import Path
import statistics as st
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper/hypothesis_scope_jcim/figure_data"
RUNS = ROOT / "experiments/runs/hypothesis_scope_v2_20261105"
POOLS = ROOT / "data/pools"
sys.path.insert(0, str(ROOT / "code/analysis"))
from analyze_hypothesis_scope_method import load_npz


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same(a, b):
    return math.isclose(float(a), float(b), abs_tol=1e-8, rel_tol=1e-10)


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pool_data(task):
    entry = next(row for row in read_rows(ROOT / "data/tasks.csv") if row["task"] == task)
    return read_json(ROOT / entry["pool"]), load_npz((ROOT / entry["oracle"]).read_bytes())


def paired_interval(differences, critical):
    mean = st.mean(differences)
    half = critical * st.stdev(differences) / math.sqrt(len(differences))
    return mean, mean - half, mean + half


def screen_sources():
    source = ROOT / "data/source_tables/new_task_screen"
    manifest = read_json(source / "source_manifest.json")
    assert manifest["ref"] == "78d899a2"
    for item in manifest["files"]:
        path = source / item["path"]
        payload = path.read_bytes()
        assert len(payload) == item["bytes"] and sha(path) == item["sha256"], path
        header = f"blob {len(payload)}".encode() + b"\0"
        assert hashlib.sha1(header + payload).hexdigest() == item["git_blob_sha1"], path
    mapping = {
        "dfl": (["substrate_smiles", "fluoride_smiles", "base_smiles"], "yield", 1),
        "bor": (["Electrophile", "ligand", "solvent"], "yield", 100),
        "chx": (["Nucleophile", "Catalyst", "Base"], "Conversion", 1),
    }
    results = {}
    for task, (columns, objective, factor) in mapping.items():
        pool, arrays = pool_data(task)
        file = source / pool["source_member"]
        receipt = read_json(POOLS / task / "receipt.json")
        assert sha(file) == receipt["source_sha256"] == pool["source_sha256"]
        rows = read_rows(file)
        assert len(rows) == len(pool["ids"]) == len(arrays["raw_values"])
        points = [tuple(row[c].strip() for c in columns) for row in rows]
        assert points == [tuple(p[c] for c in pool["axes"]) for p in pool["points"]]
        assert len(set(points)) == len(points)
        raw = [factor * float(row[objective]) for row in rows]
        assert all(same(a, b) for a, b in zip(raw, arrays["raw_values"]))
        assert max(raw) == pool["objective_scale"]
        results[task] = {
            "rows": len(raw), "response": objective, "minimum": min(raw),
            "mean": st.mean(raw), "maximum": max(raw), "values_over_100": sorted(y for y in raw if y > 100),
        }
    pool, _ = pool_data("bor")
    cleaned = read_rows(source / pool["source_member"])
    for solvent, label in (("CCO", "etoh"), ("CO", "meoh")):
        original = read_rows(source / f"currybo/data-analysis/borylation/{label}_borylation_original.csv")
        a = sorted(float(row["yield"]) for row in cleaned if row["solvent"] == solvent)
        b = sorted(float(row["yield"]) / 100 for row in original)
        assert len(a) == len(b) and all(same(x, y) for x, y in zip(a, b)), solvent
    return results


def path_scores(values, top, cap=math.inf):
    best = list(accumulate((min(y, cap) for y in values), max))
    return 100 * st.mean(best) / top, 100 * best[-1] / top


def check_history(state, oracle):
    history = state["history"]
    assert state["status"] == "COMPLETE" and len(history) == 20
    assert len({item["candidate_id"] for item in history}) == 20
    for item in history:
        assert same(item["raw_value"], oracle[item["candidate_id"]])
    return [float(item["raw_value"]) for item in history]


def deoxyfluorination(out):
    pool, arrays = pool_data("dfl")
    original_top = max(arrays["raw_values"])
    assert original_top == 106
    oracle = dict(zip(pool["ids"], arrays["raw_values"]))
    pairs = []
    for stream in range(24):
        paths = {}
        for arm in ("llm", "bo"):
            s = read_json(RUNS / f"screen/dfl/runs/{arm}/s{stream}/state.json")
            assert s["seed"] == stream
            paths[arm] = check_history(s, oracle)
        pairs.append(paths)

    scenarios = (
        ("original", math.inf, original_top),
        ("cap_and_renormalize", 100, 100),
        ("cap_with_original_denominator", 100, original_top),
    )
    summaries, rows = {}, []
    for label, cap, top in scenarios:
        differences = []
        for stream, pair in enumerate(pairs):
            llm, _ = path_scores(pair["llm"], top, cap)
            bo, _ = path_scores(pair["bo"], top, cap)
            differences.append(llm - bo)
            rows.append({"scenario": label, "stream": stream, "llm_start_auc": llm,
                         "random_start_auc": bo, "difference": llm - bo})
        mean, low, high = paired_interval(differences, 1.714)
        verdict = ("LLM start leads" if low > 1 else
                   "random start leads" if high < -1 else "not separated")
        summaries[label] = {
            "response_cap": None if math.isinf(cap) else cap, "pool_denominator": top, "pairs": len(pairs),
            "difference": mean, "interval90": [low, high], "verdict": verdict,
        }
    archived = read_json(ROOT / "results/hypothesis_scope_v2_screen_20261107/screen.json")["tasks"]["dfl"]
    original = summaries["original"]
    assert round(original["difference"], 2) == archived["difference"]
    assert [round(x, 2) for x in original["interval90"]] == archived["interval90"]
    assert round(summaries["cap_and_renormalize"]["difference"], 2) == -1.68
    write_rows(out / "deoxyfluorination_pairs.csv", rows)
    return summaries


def in_bounds(x, bounds):
    lo, hi = bounds
    return (lo is None or x >= lo - 1e-9) and (hi is None or x <= hi + 1e-9)


def in_region(point, conditions):
    return all(point[name] in spec["allowed"] if "allowed" in spec else in_bounds(point[name], spec["bounds"])
               for name, spec in conditions.items())


def quantile95(values):
    values = sorted(values)
    position = 0.95 * (len(values) - 1)
    i = math.floor(position)
    return values[i] + (position - i) * (values[i + 1] - values[i])


def s8():
    folder = FIG / "table_s8_region_quality"
    archived = read_json(folder / "recomputed_statistics.json")
    records = read_rows(folder / "region_quality_69_hypotheses.csv")
    coding = RUNS / "gp_followup/claim_coding.json"
    lineage = read_json(ROOT / "provenance/claim_coding_lineage.json")
    assert sha(coding) == lineage["analysis_sha256"]
    assert archived["claim_coding_sha256"] == lineage["historical_record"]["sha256"]
    hypotheses = {r["key"]: r for r in read_json(coding)["hypotheses"] if r["group"] == "dev"}
    assert set(hypotheses) == {r["key"] for r in records} and len(records) == 69
    groups, by_task = defaultdict(list), {}
    for row in records:
        task = row["task"]
        if task not in by_task:
            pool, arrays = pool_data(task)
            y = arrays.get("raw_values", arrays["values"])
            cutoff = quantile95(y)
            assert same(cutoff, archived["pools"][task]["threshold95"])
            by_task[task] = pool["points"], y, cutoff, sum(x >= cutoff for x in y), max(y)
        points, y, cutoff, top_count, maximum = by_task[task]
        h = hypotheses[row["key"]]
        expected_group = "avoid" if h["direction"] == "avoid" else "initial_favor" if h["installed_query"] == 1 else "refreshed_favor"
        assert row["group"] == expected_group and row["semantic_key"] == h["semantic_key"]
        scoped = [v for p, v in zip(points, y) if in_region(p, h["scope"])]
        scope_top = sum(v >= cutoff for v in scoped)
        lift = (scope_top / len(scoped)) / (top_count / len(y))
        has_max = maximum in scoped
        assert (len(y), len(scoped), top_count, scope_top) == tuple(int(row[k]) for k in (
            "pool_size", "scope_count", "pool_top5_count", "scope_top5_count"))
        assert same(lift, row["lift"]) and (lift > 1) == (row["lift_above_one"] == "True")
        assert has_max == (row["contains_maximum"] == "True")
        groups[expected_group].append((lift, has_max))
    for group, rows in groups.items():
        target = archived["groups"][group]
        assert len(rows) == target["n"] and same(st.median(x for x, _ in rows), target["median_lift"])
        assert sum(x > 1 for x, _ in rows) == target["above_one"]
        assert sum(m for _, m in rows) == target["contains_maximum"]
    return {"hypotheses_recomputed_from_candidate_pools": len(records), "groups": archived["groups"]}


def sign_flip(differences):
    observed = abs(sum(differences))
    total = sum(abs(sum(s * d for s, d in zip(signs, differences))) >= observed - 1e-9
                for signs in product((-1, 1), repeat=len(differences)))
    return total / (2 ** len(differences))


def test_g():
    folder = FIG / "test_g_gate_ablation"
    ref = read_json(folder / "verification.json")
    table = folder / "gate_runs.csv"
    assert sha(table) == ref["gate_runs_csv_sha256"]
    groups, oracles = defaultdict(list), {}
    for row in read_rows(table):
        task = row["task"]
        if task not in oracles:
            candidates = POOLS / task / "candidates.csv"
            if candidates.is_file():
                oracles[task] = {r["candidate_id"]: float(r["objective_raw"]) for r in read_rows(candidates)}
            else:
                p, x = pool_data(task)
                oracles[task] = dict(zip(p["ids"], x["raw_values"]))
        oracle = oracles[task]
        top = max(oracle.values())
        assert same(top, row["pool_maximum_raw"])
        outcomes, states = {}, []
        for arm in ("gate", "persistent"):
            path = RUNS / row[f"{arm}_sealed_member"]
            assert sha(path) == row[f"{arm}_state_sha256"]
            state = read_json(path)
            states.append(state)
            y = check_history(state, oracle)
            auc, final = path_scores(y, top)
            assert same(auc, row[f"{arm}_searchauc_points"]) and same(final, row[f"{arm}_final_best_points"])
            outcomes[arm] = auc
        a, b = states
        assert a["history"][0] == b["history"][0] and a["history"][0]["candidate_id"] == row["start_candidate_id"]
        assert same(a["history"][0]["raw_value"], row["start_value_raw"])
        assert all(a[k] == b[k] for k in ("seed", "tie_seed", "numerical_seed"))
        assert len(a["calls"]) == len(b["calls"]) == 1 and a["calls"][0]["request"] == b["calls"][0]["request"]
        delta = outcomes["gate"] - outcomes["persistent"]
        assert same(delta, row["searchauc_difference_points"])
        groups[row["cohort"], task].append(delta)
    sets = {
        "development": [st.mean(v) for (cohort, _), v in groups.items() if cohort == "development"],
        "direct_arylation": groups["direct_arylation", "aryl"],
    }
    for cohort, units in sets.items():
        summary = ref[cohort]
        assert len(units) == summary["analysis_units"] == 8
        result = paired_interval(units, 1.894579)
        for key, val in zip(("mean_difference_points", "ci90_low_points", "ci90_high_points"), result):
            assert same(val, summary[key])
        assert same(sign_flip(units), summary["exact_sign_flip_p"])
    assert sum(len(v) for v in groups.values()) == ref["paired_trajectories"] == 40
    return {"pairs": 40, "state_files_checked": 80, "cohorts": {k: ref[k] for k in sets}}


def summit_sty(point, consume_product):
    """Reimplement the noise-free Summit rate equations for the selected grid."""
    from scipy.integrate import solve_ivp

    a = point["conc_dfnb"]
    b = a * point["equiv_pldn"]
    tau = point["tau"]
    inverse_temp = 1 / (point["temperature"] + 273.71) - 1 / (90 + 273.71)
    rates = [0.6 * k * math.exp(-ea / 0.008314 * inverse_temp)
             for k, ea in zip((57.9, 2.70, 0.865, 1.63), (33.3, 35.3, 38.9, 44.8))]
    ka, kb, kc, kd = rates
    if not consume_product:
        kc = 0.0

    def integrate(_, c):
        if c[0] < 1e-6 * a:
            c[0] = 0
        if c[1] < 1e-6 * b:
            c[1] = 0
        pair = c[0] * c[1]
        rc, rd = kc * c[1] * c[2], kd * c[1] * c[3]
        total = (ka + kb) * pair
        return (-total, -total - rc - rd, ka * pair - rc, kb * pair - rd, rc + rd)

    c_end = solve_ivp(integrate, (0, tau), (a, b, 0, 0, 0)).y[2, -1]
    return max(1e-6, 60 * 210.21 * c_end / tau)


def snar(out):
    import scipy

    pool, arrays = pool_data("snar")
    assert pool["oracle_noise_percent"] == 0
    values = arrays["raw_values"]
    top = max(values)
    choices = (("snar|all|H1", "temperature"), ("snar|2|H2", "equiv_pldn"))
    mechanism = ROOT / "results/hypothesis_scope_v2_mechanism_20261106"
    coding = {r["key"]: r for r in read_json(mechanism / "claim_coding.json")["hypotheses"]}
    saved = {(r["key"], r["axis"]): r for r in read_rows(mechanism / "contrasts.csv")}
    selected, masks = set(), {}
    for key, axis in choices:
        claim = coding[key]
        spec = next(c for c in claim["contrasts"] if c["axis"] == axis)
        fixed = tuple(x for x in pool["axes"] if x != axis)
        within = spec.get("within", {k: v for k, v in claim["scope"].items() if k != axis})
        groups = defaultdict(lambda: ([], []))
        for i, point in enumerate(pool["points"]):
            if not in_region(point, within):
                continue
            g = tuple(point[k] for k in fixed)
            if in_bounds(point[axis], spec["high"]):
                groups[g][0].append(i)
            if in_bounds(point[axis], spec["low"]):
                groups[g][1].append(i)
        matched = {g: (h, l) for g, (h, l) in groups.items() if h and l}
        assert len(matched) == int(saved[key, axis]["contrasts"]) == 128
        for high, low in matched.values():
            selected.update(high)
            selected.update(low)
        masks[key, axis] = fixed, matched

    off, max_error = {}, 0.0
    for i in sorted(selected):
        point = pool["points"][i]
        on = summit_sty(point, True)
        max_error = max(max_error, abs(on - values[i]))
        off[i] = summit_sty(point, False)
    assert max_error < 1e-5

    summaries, rows = {}, []
    for key, axis in choices:
        fixed, matches = masks[key, axis]
        effects, disabled = [], []
        for group, (h, l) in sorted(matches.items()):
            high = st.mean(values[i] for i in h)
            low = st.mean(values[i] for i in l)
            high_off = st.mean(off[i] for i in h)
            low_off = st.mean(off[i] for i in l)
            effect, effect_off = 100 * (high - low) / top, 100 * (high_off - low_off) / top
            effects.append(effect)
            disabled.append(effect_off)
            rows.append({"contrast": axis, "matched_inputs": json.dumps(dict(zip(fixed, group)), sort_keys=True),
                         "high_ids": " ".join(pool["ids"][i] for i in h),
                         "low_ids": " ".join(pool["ids"][i] for i in l),
                         "high_sty": high, "low_sty": low, "high_sty_kc_off": high_off, "low_sty_kc_off": low_off,
                         "effect_points": effect, "effect_kc_off_points": effect_off})
        assert same(st.mean(effects), saved[key, axis]["mean_effect"])
        assert round(st.mean(disabled), 2) == (3.19 if axis == "temperature" else 1.67)
        summaries[axis] = {"matched_groups": len(effects), "baseline": st.mean(effects), "kc_off": st.mean(disabled)}
    write_rows(out / "snar_matched_groups.csv", rows)
    return {"source": "https://gosummit.readthedocs.io/en/latest/_modules/summit/benchmarks/snar.html",
            "scipy_version": scipy.__version__, "normalization_pool_maximum_sty": top,
            "simulated_grid_points": len(selected), "largest_baseline_sty_difference": max_error,
            "contrasts": summaries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snar", action="store_true", help="Include the SNAr sensitivity check (requires SciPy).")
    parser.add_argument("--out", type=Path, help="Fresh output directory.")
    args = parser.parse_args()
    if args.out is not None:
        out = args.out.resolve()
        if out.is_relative_to(ROOT) and not out.is_relative_to(ROOT / "output"):
            parser.error("Use a fresh directory under output/ or outside the submission.")
        out.mkdir(parents=True, exist_ok=False)
    else:
        parent = ROOT / "output"
        parent.mkdir(exist_ok=True)
        out = Path(tempfile.mkdtemp(prefix="evidence-", dir=parent))
    report = {"status": "PASS", "screen_sources": screen_sources(), "table_s8": s8(), "test_g": test_g(),
              "deoxyfluorination_sensitivity": deoxyfluorination(out)}
    if args.snar:
        report["snar_sensitivity"] = snar(out)
    (out / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("PASS", out)


if __name__ == "__main__":
    main()
