#!/usr/bin/env python3
"""Optional soft elemental-balance posterior for GSAS-II phase fractions.

This module does not alter the Rietveld refinement or overwrite its phase
fractions. It combines the XRD-derived fractions with a Gaussian prior on the
bulk cation composition and writes a separate, auditable posterior result.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
from scipy.optimize import minimize


ATOMIC_WEIGHTS = {
    "H": 1.008, "Li": 6.94, "B": 10.81, "C": 12.011, "N": 14.007,
    "O": 15.999, "F": 18.998403, "Na": 22.989769, "Mg": 24.305,
    "Al": 26.981538, "Si": 28.085, "P": 30.973762, "S": 32.06,
    "Cl": 35.45, "K": 39.0983, "Ca": 40.078, "Sc": 44.955908,
    "Ti": 47.867, "V": 50.9415, "Cr": 51.9961, "Mn": 54.938044,
    "Fe": 55.845, "Co": 58.933194, "Ni": 58.6934, "Cu": 63.546,
    "Zn": 65.38, "Zr": 91.224, "Nb": 92.90637, "Mo": 95.95,
    "Ba": 137.327, "La": 138.90547, "Ce": 140.116,
}


def parse_composition(text: str) -> Dict[str, float]:
    """Parse `Mg=1,Ni=1,Ti=2` or a compact chemical formula."""
    text = text.strip().strip("'\"")
    if "=" in text:
        result = {}
        for token in text.replace(";", ",").split(","):
            element, value = token.split("=", 1)
            result[element.strip()] = float(value)
        return result
    compact = re.sub(r"\s+", "", text)
    pairs = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)", compact)
    if not pairs or "".join(e + n for e, n in pairs) != compact:
        raise ValueError(f"cannot parse composition: {text}")
    result: Dict[str, float] = {}
    for element, number in pairs:
        result[element] = result.get(element, 0.0) + (float(number) if number else 1.0)
    return result


def formula_from_cif(path: str | Path) -> Dict[str, float]:
    """Read chemical formula from a CIF without requiring pymatgen."""
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    keys = ("_chemical_formula_sum", "_chemical_formula_structural")
    for key in keys:
        for index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped.lower().startswith(key.lower()):
                continue
            tokens = shlex.split(stripped, comments=False)
            value = " ".join(tokens[1:]).strip() if len(tokens) > 1 else ""
            if not value and index + 1 < len(lines):
                value = lines[index + 1].strip()
            if value:
                return parse_composition(value)
    raise ValueError(f"CIF has no usable chemical formula: {path}")


def _molar_mass(formula: Dict[str, float]) -> float:
    missing = sorted(set(formula) - set(ATOMIC_WEIGHTS))
    if missing:
        raise ValueError(f"atomic weights not configured for: {','.join(missing)}")
    return sum(ATOMIC_WEIGHTS[element] * amount for element, amount in formula.items())


def _cation_fraction(weights: np.ndarray, formulas: Iterable[Dict[str, float]], elements: list[str]) -> np.ndarray:
    """Return normalized cation mole fractions for a phase weight mixture."""
    moles = np.zeros(len(elements), dtype=float)
    for phase_weight, formula in zip(weights, formulas):
        mass = _molar_mass(formula)
        for index, element in enumerate(elements):
            moles[index] += phase_weight * formula.get(element, 0.0) / mass
    total = float(moles.sum())
    if total <= 0 or not math.isfinite(total):
        raise ValueError("phase mixture contains no constrained elements")
    return moles / total


def apply_soft_element_balance(
    report: dict,
    bulk_composition: str | Dict[str, float],
    balance_sigma: float = 0.03,
    systematic_floor: float | None = None,
    cif_base_dir: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict:
    """Fuse XRD phase fractions with a soft bulk-cation composition prior."""
    if balance_sigma <= 0:
        raise ValueError("balance_sigma must be positive")
    bulk = parse_composition(bulk_composition) if isinstance(bulk_composition, str) else dict(bulk_composition)
    elements = sorted(bulk)
    target = np.asarray([bulk[element] for element in elements], dtype=float)
    target /= target.sum()

    base_dir = Path(cif_base_dir).resolve() if cif_base_dir else Path.cwd()
    phase_paths = {}
    for row in report["selected_phases"]:
        path = Path(row["cif"])
        phase_paths[row["name"]] = path if path.is_absolute() else base_dir / path
    rows = report["phase_fractions"]
    names = [row["phase"] for row in rows]
    formulas = [formula_from_cif(phase_paths[name]) for name in names]
    xrd = np.asarray([float(row["weight_fraction"]) for row in rows], dtype=float)
    xrd /= xrd.sum()
    floor = float(systematic_floor if systematic_floor is not None else report.get("systematic_fraction_floor", 0.09))
    sigma_xrd = np.asarray([
        max(float(row.get("weight_fraction_sigma") or 0.0), floor) for row in rows
    ])

    def objective(weights: np.ndarray) -> float:
        xrd_term = float(np.sum(((weights - xrd) / sigma_xrd) ** 2))
        composition = _cation_fraction(weights, formulas, elements)
        balance_term = float(np.sum(((composition - target) / balance_sigma) ** 2))
        return xrd_term + balance_term

    result = minimize(
        objective,
        xrd,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(xrd),
        constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    posterior = np.asarray(result.x if result.success else xrd, dtype=float)
    posterior /= posterior.sum()

    # Laplace uncertainty in the sum-to-one tangent space.
    if len(posterior) > 1:
        basis = np.vstack([np.eye(len(posterior) - 1), -np.ones(len(posterior) - 1)])
        step = 1e-6
        jac = np.zeros((len(elements), len(posterior)))
        for index in range(len(posterior)):
            plus = posterior.copy()
            minus = posterior.copy()
            plus[index] += step
            minus[index] = max(0.0, minus[index] - step)
            plus /= plus.sum()
            minus /= minus.sum()
            jac[:, index] = (
                _cation_fraction(plus, formulas, elements)
                - _cation_fraction(minus, formulas, elements)
            ) / (2.0 * step)
        precision = basis.T @ np.diag(1.0 / sigma_xrd**2) @ basis
        precision += (jac @ basis).T @ (jac @ basis) / balance_sigma**2
        covariance = basis @ np.linalg.pinv(precision) @ basis.T
        posterior_sigma = np.sqrt(np.maximum(0.0, np.diag(covariance)))
    else:
        posterior_sigma = sigma_xrd.copy()

    before = _cation_fraction(xrd, formulas, elements)
    after = _cation_fraction(posterior, formulas, elements)
    output = {
        "method": "xrd_fraction_gaussian_prior_plus_soft_cation_balance",
        "status": "success" if result.success else "optimizer_failed_xrd_retained",
        "message": str(result.message),
        "original_refinement_unchanged": True,
        "bulk_composition_input": bulk,
        "constrained_elements": elements,
        "target_cation_fraction": dict(zip(elements, target.tolist())),
        "balance_sigma_fraction": balance_sigma,
        "xrd_systematic_floor_fraction": floor,
        "objective_before": objective(xrd),
        "objective_after": objective(posterior),
        "cation_fraction_before": dict(zip(elements, before.tolist())),
        "cation_fraction_after": dict(zip(elements, after.tolist())),
        "phases": [
            {
                "phase": name,
                "formula": formulas[index],
                "xrd_weight_fraction": float(xrd[index]),
                "xrd_sigma": float(sigma_xrd[index]),
                "balance_posterior_weight_fraction": float(posterior[index]),
                "balance_posterior_sigma": float(posterior_sigma[index]),
            }
            for index, name in enumerate(names)
        ],
        "interpretation": (
            "Single-phase models are compositionally uninformative; the soft balance cannot distinguish R3 from R-3."
            if len(names) == 1 else
            "Posterior fractions combine XRD evidence and the stated bulk-cation prior; retain both datasets."
        ),
    }
    if output_path:
        Path(output_path).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Apply soft elemental balance to a smart GSAS-II report")
    parser.add_argument("--report", required=True)
    parser.add_argument("--bulk", required=True, help="e.g. Mg=1,Ni=1,Ti=2")
    parser.add_argument("--balance-sigma", type=float, default=0.03)
    parser.add_argument("--systematic-floor", type=float, default=None)
    parser.add_argument("--cif-base-dir", default=None,
                        help="Base directory for relative CIF paths; defaults to the report project root")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    report_path = Path(args.report).resolve()
    inferred_base = report_path.parents[2] if len(report_path.parents) >= 3 else report_path.parent
    result = apply_soft_element_balance(
        report,
        args.bulk,
        args.balance_sigma,
        args.systematic_floor,
        args.cif_base_dir or inferred_base,
        args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
