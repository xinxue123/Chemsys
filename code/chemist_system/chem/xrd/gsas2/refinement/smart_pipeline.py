#!/usr/bin/env python3
"""Robust, model-selecting GSAS-II powder refinement.

The pipeline is deliberately conservative:

* a parent-only model is the baseline;
* candidate phases are added by forward BIC selection;
* every model is tried from multiple scale starts;
* nuisance parameters are released in stages;
* every stage is checkpointed and subjected to physical gates;
* trace phases are reported, but are not called quantitative below the
  configured systematic floor.

This module is standalone so it can be regression-tested alongside the
legacy workflow before being made the default backend.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from dataclasses import dataclass, asdict
from itertools import count
from .... import paths

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    cif: str


@dataclass(frozen=True)
class Recipe:
    name: str
    background_coeffs: int
    refine_parent_cell: bool = True
    refine_zero: bool = False
    refine_u: bool = False


DEFAULT_RECIPES = (
    Recipe("bg4_cell", 4, True, False, False),
    Recipe("bg6_cell", 6, True, False, False),
    Recipe("bg6_cell_zero", 6, True, True, False),
    Recipe("bg6_cell_u", 6, True, False, True),
)


def _clean(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def parse_phase(text: str) -> PhaseSpec:
    if "=" not in text:
        raise argparse.ArgumentTypeError("phase must be NAME=/path/to/file.cif")
    name, cif = text.split("=", 1)
    if not name.strip() or not cif.strip():
        raise argparse.ArgumentTypeError("phase name and CIF path are required")
    return PhaseSpec(name.strip(), cif.strip())


def _covariance(project):
    data = project.data.get("Covariance", {}).get("data", {})
    return list(data.get("varyList", [])), np.asarray(data.get("covMatrix", []), dtype=float)


def _phase_fractions(project, hist) -> List[dict]:
    phases = project.phases()
    terms = []
    for phase in phases:
        hap = phase.data["Histograms"][hist.name]
        scale = float(hap["Scale"][0])
        mass = float(phase.data["General"]["Mass"])
        volume = float(phase.get_cell()["volume"])
        terms.append((phase, scale, mass, volume, scale * mass * volume))
    denominator = sum(row[4] for row in terms)
    if not math.isfinite(denominator) or denominator <= 0:
        raise RuntimeError("non-positive phase-fraction denominator")

    vary, cov = _covariance(project)
    indices = []
    constants = []
    for phase, _scale, mass, volume, _term in terms:
        variable = f"{phase.id}:{hist.id}:Scale"
        indices.append(vary.index(variable) if variable in vary else None)
        constants.append(mass * volume)

    rows = []
    for i, (phase, scale, mass, volume, term) in enumerate(terms):
        fraction = term / denominator
        sigma = None
        if cov.ndim == 2 and cov.size and all(index is not None for index in indices):
            jac = np.zeros(len(vary), dtype=float)
            for j, index in enumerate(indices):
                jac[index] = ((constants[i] / denominator) if i == j else 0.0) - (
                    term * constants[j] / denominator**2
                )
            variance = float(jac @ cov @ jac)
            sigma = math.sqrt(max(variance, 0.0))
        rows.append(
            {
                "phase": phase.name,
                "scale": scale,
                "weight_fraction": fraction,
                "weight_percent": 100.0 * fraction,
                "weight_fraction_sigma": sigma,
                "weight_percent_sigma": None if sigma is None else 100.0 * sigma,
            }
        )
    return rows


def _profile_metrics(project, hist) -> dict:
    arrays = hist.data["data"][1]
    x = np.asarray(arrays[0], dtype=float)
    observed = np.asarray(arrays[1], dtype=float)
    weight = np.asarray(arrays[2], dtype=float)
    calculated = np.asarray(arrays[3], dtype=float)
    background = np.asarray(arrays[4], dtype=float)
    residual = observed - calculated
    finite = np.isfinite(observed) & np.isfinite(calculated) & np.isfinite(weight) & (weight > 0)
    if not finite.any() or not np.any(np.abs(calculated[finite]) > 0):
        raise RuntimeError("calculated profile is empty or non-finite")
    wrss = float(np.sum(weight[finite] * residual[finite] ** 2))
    n = int(finite.sum())
    vary, _cov = _covariance(project)
    k = len(vary)
    scaled = max(wrss / max(n, 1), np.finfo(float).tiny)
    bic = n * math.log(scaled) + k * math.log(max(n, 2))
    aic = n * math.log(scaled) + 2 * k
    aicc = aic + (2 * k * (k + 1) / max(n - k - 1, 1))
    dw_den = float(np.sum(residual[finite] ** 2))
    dw = float(np.sum(np.diff(residual[finite]) ** 2) / dw_den) if dw_den > 0 else None
    return {
        "n_points": n,
        "n_parameters": k,
        "weighted_rss": wrss,
        "bic": bic,
        "aicc": aicc,
        "durbin_watson": dw,
        "rwp": float(hist.residuals.get("wR")),
        "x": x,
        "observed": observed,
        "weight": weight,
        "calculated": calculated,
        "background": background,
        "residual": residual,
    }


def _instrument_value(hist, key: str) -> Optional[float]:
    try:
        values = hist.InstrumentParameters
        raw = values.get(key)
        if isinstance(raw, (list, tuple)):
            for item in reversed(raw):
                if isinstance(item, (int, float, np.number)):
                    return float(item)
        if isinstance(raw, (int, float, np.number)):
            return float(raw)
    except Exception:
        return None
    return None


def _physical_gate(
    fractions: Sequence[dict],
    metrics: dict,
    hist,
    parent_volume_initial: float,
    parent_volume_current: float,
    previous_rwp: Optional[float],
) -> Tuple[bool, List[str]]:
    reasons = []
    rwp = metrics["rwp"]
    if not math.isfinite(rwp):
        reasons.append("non_finite_rwp")
    if previous_rwp is not None and rwp > max(previous_rwp + 2.0, previous_rwp * 1.15):
        reasons.append(f"rwp_regression:{previous_rwp:.4f}->{rwp:.4f}")
    for row in fractions:
        if not math.isfinite(row["scale"]) or row["scale"] < 0:
            reasons.append(f"negative_or_nonfinite_scale:{row['phase']}")
        frac = row["weight_fraction"]
        if not math.isfinite(frac) or frac < -1e-8 or frac > 1.000001:
            reasons.append(f"nonphysical_fraction:{row['phase']}:{frac}")
    zero = _instrument_value(hist, "Zero")
    if zero is not None and abs(zero) > 0.30:
        reasons.append(f"zero_shift_out_of_bounds:{zero}")
    u_value = _instrument_value(hist, "U")
    if u_value is not None and not (-0.05 <= u_value <= 1.0):
        reasons.append(f"profile_u_out_of_bounds:{u_value}")
    if parent_volume_initial > 0:
        rel = abs(parent_volume_current / parent_volume_initial - 1.0)
        if rel > 0.08:
            reasons.append(f"parent_volume_change_out_of_bounds:{rel}")
    return not reasons, reasons


def _save_profile(path: Path, metrics: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["two_theta_deg", "y_observed", "weight", "y_calculated", "background", "residual"])
        writer.writerows(
            zip(
                metrics["x"], metrics["observed"], metrics["weight"], metrics["calculated"],
                metrics["background"], metrics["residual"],
            )
        )


def _json_metrics(metrics: dict) -> dict:
    return {k: _clean(v) for k, v in metrics.items() if k not in {"x", "observed", "weight", "calculated", "background", "residual"}}


class SmartGSAS2Pipeline:
    def __init__(
        self,
        xrd: str,
        parent: PhaseSpec,
        candidates: Sequence[PhaseSpec],
        instprm: str,
        output_dir: str,
        gsas2_path: str = str(paths.GSAS2_HOME),
        limits: Tuple[float, float] = (10.0, 80.0),
        systematic_floor: float = 0.09,
        min_bic_improvement: float = 10.0,
    ):
        self.xrd = str(xrd)
        self.parent = parent
        self.candidates = list(candidates)
        self.instprm = str(instprm)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gsas2_path = gsas2_path
        self.limits = limits
        self.systematic_floor = systematic_floor
        self.min_bic_improvement = min_bic_improvement
        self._attempt_counter = count(1)
        sys.path.insert(0, gsas2_path)
        from GSASII import GSASIIscriptable as G2
        self.G2 = G2

    def run_attempt(
        self,
        phases: Sequence[PhaseSpec],
        recipe: Recipe,
        impurity_scale: float,
        purpose: str,
    ) -> dict:
        attempt_id = next(self._attempt_counter)
        phase_tag = "_".join(p.name for p in phases)
        tag = f"{attempt_id:03d}_{purpose}_{phase_tag}_{recipe.name}_s{impurity_scale:g}".replace("/", "_")
        out = self.output_dir / "attempts" / tag
        out.mkdir(parents=True, exist_ok=True)
        gpx = out / "refinement.gpx"
        project = self.G2.G2Project(newgpx=str(gpx))
        hist = project.add_powder_histogram(self.xrd, self.instprm)
        hist.set_refinements({"Limits": [float(self.limits[0]), float(self.limits[1])]})
        hist.data["Sample Parameters"]["Scale"][1] = False
        g2phases = []
        for i, spec in enumerate(phases):
            phase = project.add_phase(spec.cif, phasename=spec.name, histograms=[hist])
            phase.data["Histograms"][hist.name]["Scale"][0] = 1.0 if i == 0 else impurity_scale
            phase.set_HAP_refinements({"Scale": True})
            g2phases.append(phase)
        parent_volume_initial = float(g2phases[0].get_cell()["volume"])
        project["Controls"]["data"]["max cyc"] = 40

        stages = [
            ("background_scales", lambda: hist.set_refinements({
                "Background": {"type": "chebyschev-1", "no. coeffs": recipe.background_coeffs, "refine": True}
            })),
        ]
        if recipe.refine_parent_cell:
            stages.append(("parent_cell", lambda: g2phases[0].set_refinements({"Cell": True})))
        if recipe.refine_zero:
            stages.append(("zero", lambda: hist.set_refinements({"Instrument Parameters": ["Zero"]})))
        if recipe.refine_u:
            stages.append(("profile_u", lambda: hist.set_refinements({"Instrument Parameters": ["U"]})))

        history = []
        previous_rwp = None
        accepted = []
        error = None
        for stage_index, (stage_name, activate) in enumerate(stages, start=1):
            try:
                activate()
                project.do_refinements([{}])
                fractions = _phase_fractions(project, hist)
                metrics = _profile_metrics(project, hist)
                parent_volume = float(g2phases[0].get_cell()["volume"])
                ok, reasons = _physical_gate(
                    fractions, metrics, hist, parent_volume_initial, parent_volume, previous_rwp
                )
                project.save()
                checkpoint = out / f"stage_{stage_index:02d}_{stage_name}.gpx"
                shutil.copy2(gpx, checkpoint)
                stage = {
                    "stage": stage_name,
                    "accepted": ok,
                    "rejection_reasons": reasons,
                    "metrics": _json_metrics(metrics),
                    "fractions": fractions,
                    "parent_cell": _clean(g2phases[0].get_cell()),
                    "zero_shift": _instrument_value(hist, "Zero"),
                    "profile_u": _instrument_value(hist, "U"),
                    "checkpoint": str(checkpoint),
                }
                history.append(stage)
                if not ok:
                    break
                accepted.append(stage)
                previous_rwp = metrics["rwp"]
            except Exception as exc:
                error = f"{type(exc).__name__}:{exc}"
                history.append({"stage": stage_name, "accepted": False, "rejection_reasons": [error]})
                break

        best = min(accepted, key=lambda row: row["metrics"]["bic"]) if accepted else None
        if best:
            shutil.copy2(best["checkpoint"], out / "best.gpx")
            # Profile corresponds to the last in-memory stage only when it is the best.
            # Re-open best checkpoint to make profile/report self-consistent.
            best_project = self.G2.G2Project(gpxfile=best["checkpoint"])
            best_hist = best_project.histograms()[0]
            best_metrics = _profile_metrics(best_project, best_hist)
            _save_profile(out / "best_profile.csv", best_metrics)
        report = {
            "attempt": tag,
            "purpose": purpose,
            "phases": [asdict(p) for p in phases],
            "recipe": asdict(recipe),
            "impurity_initial_scale": impurity_scale,
            "success": best is not None,
            "error": error,
            "history": history,
            "best": best,
            "attempt_dir": str(out),
        }
        (out / "attempt_report.json").write_text(json.dumps(_clean(report), indent=2), encoding="utf-8")
        return report

    @staticmethod
    def _best(attempts: Iterable[dict]) -> Optional[dict]:
        valid = [a for a in attempts if a.get("success") and a.get("best")]
        return min(valid, key=lambda a: a["best"]["metrics"]["bic"]) if valid else None

    @staticmethod
    def _phase_row(attempt: dict, name: str) -> Optional[dict]:
        if not attempt or not attempt.get("best"):
            return None
        return next((row for row in attempt["best"]["fractions"] if row["phase"] == name), None)

    def _phase_supported(self, attempt: dict, phase_name: str) -> Tuple[bool, str]:
        row = self._phase_row(attempt, phase_name)
        if not row:
            return False, "phase_fraction_missing"
        frac = float(row["weight_fraction"])
        sigma = row.get("weight_fraction_sigma")
        if frac < 0.005:
            return False, f"fraction_below_0.5pct:{frac}"
        if sigma is not None and sigma > 0 and frac < 3.0 * sigma:
            return False, f"fraction_not_3sigma_significant:{frac}/{sigma}"
        return True, "supported"

    def run(self) -> dict:
        screen_recipe = Recipe("screen_bg4_cell", 4, True, False, False)
        all_attempts = []
        baseline_attempts = [
            self.run_attempt([self.parent], screen_recipe, scale, "baseline")
            for scale in (0.02,)
        ]
        all_attempts.extend(baseline_attempts)
        current_best = self._best(baseline_attempts)
        if not current_best:
            raise RuntimeError("parent-only baseline failed all physical gates")
        selected = [self.parent]
        remaining = list(self.candidates)
        selection_log = []

        while remaining:
            trials = []
            for candidate in remaining:
                attempts = [
                    self.run_attempt(selected + [candidate], screen_recipe, scale, f"screen_{candidate.name}")
                    for scale in (0.005, 0.02, 0.08)
                ]
                all_attempts.extend(attempts)
                best = self._best(attempts)
                if best:
                    improvement = current_best["best"]["metrics"]["bic"] - best["best"]["metrics"]["bic"]
                    supported, reason = self._phase_supported(best, candidate.name)
                    trials.append((improvement, supported, candidate, best, reason))
                else:
                    selection_log.append({"phase": candidate.name, "decision": "reject", "reason": "all_attempts_failed"})
            eligible = [row for row in trials if row[1] and row[0] >= self.min_bic_improvement]
            for improvement, supported, candidate, _best, reason in trials:
                if not (supported and improvement >= self.min_bic_improvement):
                    selection_log.append({
                        "phase": candidate.name,
                        "decision": "reject",
                        "bic_improvement": improvement,
                        "reason": reason if not supported else "bic_improvement_too_small",
                    })
            if not eligible:
                break
            improvement, _supported, candidate, best, reason = max(eligible, key=lambda row: row[0])
            selected.append(candidate)
            remaining = [p for p in remaining if p.name != candidate.name]
            current_best = best
            selection_log.append({
                "phase": candidate.name,
                "decision": "accept",
                "bic_improvement": improvement,
                "reason": reason,
            })

        final_attempts = []
        for recipe in DEFAULT_RECIPES:
            for scale in (0.005, 0.02, 0.08) if len(selected) > 1 else (0.02,):
                final_attempts.append(self.run_attempt(selected, recipe, scale, "final"))
        all_attempts.extend(final_attempts)
        final_best = self._best(final_attempts)
        if not final_best:
            final_best = current_best
        if not final_best:
            raise RuntimeError("no valid final refinement model")

        final_fractions = []
        for row in final_best["best"]["fractions"]:
            row = dict(row)
            sigma = row.get("weight_fraction_sigma")
            significant = sigma is None or sigma <= 0 or row["weight_fraction"] >= 3.0 * sigma
            row["three_sigma_significant"] = bool(significant)
            row["quantitative"] = bool(significant and row["weight_fraction"] >= self.systematic_floor)
            row["reporting_class"] = "quantitative" if row["quantitative"] else (
                "qualitative_trace" if significant else "unresolved"
            )
            final_fractions.append(row)

        best_metrics = final_best["best"]["metrics"]
        quality_reasons = []
        if best_metrics.get("rwp") is None or best_metrics["rwp"] > 20.0:
            quality_reasons.append("rwp_above_20pct")
        dw = best_metrics.get("durbin_watson")
        if dw is None:
            quality_reasons.append("durbin_watson_missing")
        elif not (0.5 <= dw <= 3.0):
            quality_reasons.append(f"strongly_correlated_residuals:DW={dw:.4f}")
        unresolved = [r["phase"] for r in final_fractions if r["reporting_class"] == "unresolved"]
        if unresolved:
            quality_reasons.append("unresolved_phase_fractions:" + ",".join(unresolved))
        quality_level = "quantitative_candidate" if not quality_reasons else "interpretable_not_publication_ready"

        best_dir = Path(final_best["attempt_dir"])
        shutil.copy2(best_dir / "best.gpx", self.output_dir / "best.gpx")
        shutil.copy2(best_dir / "best_profile.csv", self.output_dir / "best_profile.csv")
        summary = {
            "success": True,
            "parent": asdict(self.parent),
            "selected_phases": [asdict(p) for p in selected],
            "rejected_candidates": [asdict(p) for p in remaining],
            "selection_log": selection_log,
            "systematic_fraction_floor": self.systematic_floor,
            "best_attempt": final_best["attempt"],
            "best_recipe": final_best["recipe"],
            "best_metrics": best_metrics,
            "phase_fractions": final_fractions,
            "quality": {
                "level": quality_level,
                "quantification_allowed": not quality_reasons,
                "reasons": quality_reasons,
                "note": "Rwp alone is not a reliability criterion; residual correlation and phase significance are mandatory gates.",
            },
            "n_attempts": len(all_attempts),
            "artifacts": {
                "gpx": str(self.output_dir / "best.gpx"),
                "profile": str(self.output_dir / "best_profile.csv"),
                "attempts_dir": str(self.output_dir / "attempts"),
            },
        }
        (self.output_dir / "smart_report.json").write_text(json.dumps(_clean(summary), indent=2), encoding="utf-8")
        with (self.output_dir / "model_trials.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["attempt", "purpose", "phases", "recipe", "initial_scale", "success", "rwp", "bic", "aicc"])
            for attempt in all_attempts:
                best = attempt.get("best") or {}
                metrics = best.get("metrics") or {}
                writer.writerow([
                    attempt["attempt"], attempt["purpose"], "+".join(p["name"] for p in attempt["phases"]),
                    attempt["recipe"]["name"], attempt["impurity_initial_scale"], attempt["success"],
                    metrics.get("rwp"), metrics.get("bic"), metrics.get("aicc"),
                ])
        return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xrd", required=True)
    parser.add_argument("--parent", required=True, type=parse_phase)
    parser.add_argument("--candidate", action="append", default=[], type=parse_phase)
    parser.add_argument("--instprm", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gsas2-path", default=str(paths.GSAS2_HOME))
    parser.add_argument("--low", type=float, default=10.0)
    parser.add_argument("--high", type=float, default=80.0)
    parser.add_argument("--systematic-floor", type=float, default=0.09)
    parser.add_argument("--min-bic-improvement", type=float, default=10.0)
    args = parser.parse_args(argv)
    pipeline = SmartGSAS2Pipeline(
        xrd=args.xrd,
        parent=args.parent,
        candidates=args.candidate,
        instprm=args.instprm,
        output_dir=args.output_dir,
        gsas2_path=args.gsas2_path,
        limits=(args.low, args.high),
        systematic_floor=args.systematic_floor,
        min_bic_improvement=args.min_bic_improvement,
    )
    result = pipeline.run()
    print(json.dumps(_clean(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
