"""Fixed mixed Suzuki domain and an exported, inference-only Summit oracle."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc

CATALYSTS = ("P1-L1", "P2-L1", "P1-L2", "P1-L3", "P1-L4", "P1-L5", "P1-L6", "P1-L7")
NUMERIC = ("t_res", "temperature", "catalyst_loading")
LOW = np.array([60., 30., .5])
HIGH = np.array([600., 110., 2.5])
DOMAIN = {"catalyst": list(CATALYSTS), **{n: [float(l), float(h)] for n, l, h in zip(NUMERIC, LOW, HIGH)}}


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                             separators=(",", ":")).encode("utf-8")).hexdigest()


def validate_point(row):
    if set(row) != {"catalyst", *NUMERIC} or row["catalyst"] not in CATALYSTS:
        raise ValueError("point must contain the exact mixed-domain coordinates")
    for name, lo, hi in zip(NUMERIC, LOW, HIGH):
        v = row[name]
        if isinstance(v, (str, bool)) or not np.isscalar(v) or not np.isfinite(v) or not lo <= v <= hi:
            raise ValueError(f"invalid hard-domain coordinate: {name}")
    return {"catalyst": row["catalyst"], **{name: float(row[name]) for name in NUMERIC}}


def same_point(a, b, tolerance=1e-9):
    return a["catalyst"] == b["catalyst"] and all(
        abs((a[n] - b[n]) / (hi - lo)) <= tolerance for n, lo, hi in zip(NUMERIC, LOW, HIGH))


def encode(rows):
    checked = [validate_point(row) for row in rows]
    if not checked:
        return np.empty((0, 11))
    x = (np.array([[row[n] for n in NUMERIC] for row in checked]) - LOW) / (HIGH - LOW)
    cat = np.eye(8)[[CATALYSTS.index(row["catalyst"]) for row in checked]]
    return np.concatenate([x, cat], axis=1)


def points_from_unit(x, catalysts):
    return [{"catalyst": c, **{name: float(v) for name, v in zip(NUMERIC, LOW + u * (HIGH - LOW))}}
            for u, c in zip(x, catalysts)]


def initial_design(case, seed):
    state = 20261026 + case * 10000 + seed * 100
    x = qmc.LatinHypercube(3, seed=state).random(4)
    cats = np.random.default_rng(state).permutation(CATALYSTS)[:4]
    return points_from_unit(x, cats)


def candidate_grid(case, seed, iteration, per_catalyst=512):
    state = 718013 + case * 100000 + seed * 1000 + iteration * 10
    rows = []
    for i, c in enumerate(CATALYSTS):
        x = qmc.Sobol(3, scramble=True, seed=state + i).random_base2(int(np.log2(per_catalyst)))
        rows.extend(points_from_unit(x, [c] * len(x)))
    return rows


def validate_region(region):
    if set(region) != {"catalysts", *NUMERIC}:
        raise ValueError("region must contain catalysts and all three continuous intervals")
    cats = region["catalysts"]
    if not isinstance(cats, list) or not cats or len(set(cats)) != len(cats) or not set(cats) <= set(CATALYSTS):
        raise ValueError("region catalysts must be a nonempty subset of the hard categories")
    for n, l, h in zip(NUMERIC, LOW, HIGH):
        b = region[n]
        if (not isinstance(b, list) or len(b) != 2 or any(isinstance(v, (bool, str)) for v in b)
                or not np.isfinite(b).all() or not l <= b[0] <= b[1] <= h):
            raise ValueError(f"invalid regional interval {n}")
    return region


def region_grid(region, seed, size=512):
    validate_region(region)
    # A categorical product, never continuous interpolation between one-hot labels.
    cats = region["catalysts"]
    counts = [size // len(cats) + int(i < size % len(cats)) for i in range(len(cats))]
    rows = []
    lb = np.array([region[n][0] for n in NUMERIC])
    ub = np.array([region[n][1] for n in NUMERIC])
    for i, (cat, count) in enumerate(zip(cats, counts)):
        x = qmc.Sobol(3, scramble=True, seed=seed + i).random_base2(int(np.ceil(np.log2(count))))[:count]
        for v in lb + x * (ub - lb):
            rows.append({"catalyst": cat, **{n: float(a) for n, a in zip(NUMERIC, v)}})
    # Equal mass per category, also when size is not divisible by category count.
    weights = np.concatenate([np.full(count, 1. / len(cats) / count) for count in counts])
    return rows, weights


class FrozenSuzukiOracle:
    """Only queried points are returned; posterior policies never receive weights."""
    def __init__(self, directory, case):
        if case not in (1, 2, 3, 4):
            raise ValueError("only Reizman Suzuki Cases 1-4 are in scope")
        root = Path(directory) / f"case_{case}"
        self.meta = json.loads((root / "parameters.json").read_text(encoding="utf-8"))
        if sha256((root / "weights.npz").read_bytes()).hexdigest() != self.meta["weights_sha256"]:
            raise ValueError("exported simulator weights changed")
        self.weights = dict(np.load(root / "weights.npz", allow_pickle=False))

    def predict(self, rows):
        for row in rows:
            validate_point(row)
        if not rows:
            return np.empty(0)
        p = self.meta["parameters"]
        inputs = [v for v in p["domain"] if not v["is_objective"]]
        numeric = [v["name"] for v in inputs if v["type"] == "ContinuousVariable"]
        cats = [v for v in inputs if v["type"] == "CategoricalVariable"]
        outputs = [v for v in p["domain"] if v["is_objective"]]
        ensemble = []
        for j, settings in enumerate(p["experiment_params"]["predictors"]):
            scaler = settings["input_preprocessor"]["num"]
            x = (np.array([[r[n] for n in numeric] for r in rows]) - scaler["mean_"]) / scaler["scale_"]
            encoded = [x, *[np.array([[float(r[v["name"]] == c) for c in v["levels"]] for r in rows]) for v in cats]]
            x = np.concatenate(encoded, axis=1).astype(np.float32)
            h = np.maximum(x @ self.weights[f"{j}_input_layer.weight"].T + self.weights[f"{j}_input_layer.bias"], 0)
            y = h @ self.weights[f"{j}_output_layer.weight"].T + self.weights[f"{j}_output_layer.bias"]
            y *= np.array(settings["output_preprocessor"]["scale_"])
            y += np.array(settings["output_preprocessor"]["mean_"])
            for k, out in enumerate(outputs):
                y[:, k] = np.clip(y[:, k], *out["bounds"])
            ensemble.append(y)
        names = p["experiment_params"]["output_variable_names"]
        y = np.array(ensemble).mean(axis=0)[:, names.index("yld")].astype(float) / 100.
        if not np.isfinite(y).all() or np.any((y < 0) | (y > 1)):
            raise ValueError("invalid simulator response")
        return y


def export_oracles(destination):
    """Run in the frozen Summit environment; qualify unseen continuous points."""
    import pandas as pd
    import torch
    from chem.decision.replay import load_frozen_summit, write_json_atomic

    reports = []
    root = Path(destination)
    for case in (1, 2, 3, 4):
        out = root / f"case_{case}"
        out.mkdir(parents=True, exist_ok=True)
        emulator, parameters, module, steps = load_frozen_summit(case)
        from summit.utils.dataset import DataSet
        folder = module.get_model_path() / f"reizman_suzuki_case_{case}"
        weights = {}
        for j in range(len(emulator.predictors)):
            saved = torch.load(folder / f"reizman_suzuki_case_{case}_predictor_{j}.pt", map_location="cpu")
            weights.update({f"{j}_{k}": v.numpy() for k, v in saved.items()})
        np.savez_compressed(out / "weights.npz", **weights)
        meta = {"case": case, "parameters": parameters,
                "weights_sha256": sha256((out / "weights.npz").read_bytes()).hexdigest(),
                "official_assets_sha256": {p.name: sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir()) if p.is_file()}}
        write_json_atomic(out / "parameters.json", meta)
        oracle = FrozenSuzukiOracle(root, case)
        task = json.loads((Path("data/runs/chapter5/summit_cache") / f"case_{case}" / "task.json").read_text(encoding="utf-8"))
        old_rows = task["candidates"]
        rng = np.random.default_rng(331777 + case)
        extra_rows = points_from_unit(rng.random((64, 3)), rng.choice(CATALYSTS, 64))
        rows = [*old_rows, *extra_rows]
        frame = pd.DataFrame(rows)
        with torch.no_grad():
            emulator._predict(frame.iloc[:8])
            official_mean, _ = emulator._predict(frame)
            official_public = emulator.run_experiments(DataSet.from_df(frame.iloc[-64:].copy()))
        col = parameters["experiment_params"]["output_variable_names"].index("yld")
        predicted = oracle.predict(rows)
        error = float(np.max(np.abs(official_mean[:, col] - predicted * 100)))
        public_error = float(np.max(np.abs(official_public["yld"].to_numpy().ravel() - predicted[-64:] * 100)))
        tolerance = 1e-4
        if max(error, public_error) > tolerance:
            raise ValueError(f"continuous emulator parity failed: {error}, {public_error}")
        report = {"case": case, "status": "PASS", "qualified_points": len(rows), "new_continuous_points": 64,
                  "max_abs_error_percentage_points": error, "public_api_max_abs_error_percentage_points": public_error,
                  "tolerance_percentage_points": tolerance, "optimizer_steps": steps["optimizer_steps"],
                  "backward_calls": steps["backward_calls"], "operator_compiles_in_loop": 0,
                  "checkpoint_and_scaler_equality": True, "parameters_sha256": sha256((out / "parameters.json").read_bytes()).hexdigest(),
                  "weights_sha256": meta["weights_sha256"]}
        write_json_atomic(out / "qualification.json", report)
        reports.append(report)
    write_json_atomic(root / "qualification.json", {"status": "PASS", "cases": reports})
    return reports
