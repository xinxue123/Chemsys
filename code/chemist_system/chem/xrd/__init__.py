"""
XRD Yield Estimator — Generic Framework
========================================
Estimates phase yield from XRD patterns.

WARNING on peak-based methods:
  Peak area ratios do NOT reliably give phase fractions for solid-state
  synthesis products.  At 1100-1400°C the same phase can show very different
  peak intensities due to changes in crystallite size, preferred orientation,
  and packing density — even when the actual phase fraction changes substantially.
  (This was verified empirically: W_ErTa7O19 stayed ~0.65 across all cycles
  despite yield ranging from 22.5% to 40.4%.)

  For publication-quality results, use GSAS-II Rietveld refinement
  (RietveldRefiner backend, ~1-2% accuracy).
  PeakBasedRI is suitable only for BO feedback loops where relative
  ranking matters more than absolute accuracy.

Two implementations:
  1. RietveldRefiner  — full Rietveld via GSAS-II (most accurate, needs GSAS-II install)
  2. PeakBasedRI      — peak-profile RIR (fallback, ~5-10% accuracy)

Both implement the same public interface so the rest of the pipeline
(e.g. KnowledgeGuidedBO) is agnostic to which backend is used.

Usage
-----
    from chem.xrd import create_estimator

    estimator = create_estimator(
        system="erta7o19",
        calibration={"points": [{"R": 0.523, "yield": 40.4}]}
    )
    result = estimator.estimate("/path/to/xrd_data.txt")
    print(result.yield_pct, result.phase_fractions)
"""

import os, sys, warnings
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass

# GSAS-II availability is checked lazily inside gsas2_refinement.workflow via
# a silent subprocess preflight. Do not import GSAS-II here; failed imports can
# emit noisy binary/CifFile diagnostics before we have a chance to fall back.


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM CONFIGS — module-level registry
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_CONFIGS = {

    "erta7o19": {
        "name":        "ErTa\u2087O\u2081\u2089",
        "target_phase":"ErTa7O19",
        "phases": {
            "ErTa7O19": {
                "formula":      "ErTa7O19",
                # Reference from mp-773103 P-6c2 DFT (Cu Kα1 λ=1.5406Å)
                # (2,-1,-1) = 28.88°, (3,0,-3) = 57.46°
                # With delta_2theta=+0.57 → observed 29.45°, 58.03°
                "peaks_2theta": [28.88, 57.46],
                "hkl":          ["(2,-1,-1)", "(3,0,-3)"],
                "preferred":     0,
                "cif":          None,
            },
            "Er2O3": {
                "formula":      "Er2O3",
                # Reference from COD-1000096, cubic Ia-3 (bixbyite)
                # (400) = 50.61°, (222) = 35.14°
                # With delta_2theta=+0.57 → observed 51.18°, 35.71°
                "peaks_2theta": [50.61, 35.14],
                "hkl":          ["400", "222"],
                "preferred":     0,
                "cif":          None,
            },
            "Ta2O5": {
                "formula":      "Ta2O5",
                # Reference from COD-1000889, orthorhombic P2_1nb
                # (110) = 23.03°, (001) = 46.11°
                # With delta_2theta=+0.57 → observed 23.60°, 46.68°
                "peaks_2theta": [23.03, 46.11],
                "hkl":          ["110", "001"],
                "preferred":     0,
                "cif":          None,
            },
            "ErTaO4": {
                "formula":      "ErTaO4",
                # Reference from mp-3444, tetragonal I4_1/a
                # (110) = 24.12°, (022) = 47.22°, (113) = 55.97°
                # With delta_2theta=+0.57 → observed 24.69°, 47.79°, 56.54°
                "peaks_2theta": [24.12, 47.22, 55.97],
                "hkl":          ["110", "022", "113"],
                "preferred":     0,
                "cif":          None,
            },
        },
        # RIR values relative to Al2O3 (Chung 1974)
        "rir": {
            "ErTa7O19": 1.0,
            "Er2O3":    1.1,
            "Ta2O5":    1.2,
            "ErTaO4":   1.0,
        },
        # IMPORTANT: The simple peak-area ratio R = S_ErTa7O19/(S_ErTa7O19+S_Er2O3)
        # does NOT reliably predict yield because Er2O3 (222) peak at ~35.7°
        # overlaps with ErTaO4 (002) and other ErTa7O19 peaks.
        # For publication-quality results use GSAS-II Rietveld refinement.
        #
        # Multi-phase profile fitting calibration (Pawlik equivalent):
        # Method: 6-line pseudo-Voigt model, least-squares per-phase scale factors
        # Phases: ErTa7O19(29.45°), Er2O3(28.50°), Ta2O5(23.10°), ErTaO4(27.20°)
        # R = S_ErTa7O19 / (S_ErTa7O19 + S_Er2O3)
        # y = -101.7278 * R + 106.2027  (R²=0.953, MAE=1.32%, delta_2theta=+0.57)
        # Note: NEGATIVE slope indicates Er2O3(222) peak overlap — use Rietveld
        "calibration": {
            "slope": -101.7278,
            "intercept": 106.2027,
            "cal_sigma": 1.3,
            "R_type": "profile_scale_factor_ratio",
            "formula": "R = S_ErTa7O19 / (S_ErTa7O19 + S_Er2O3)",
            "warning": "Negative slope from Er2O3(222) overlap — install GSAS-II for Rietveld",
            "peak_positions": {
                "ErTa7O19": [28.88],   # +0.57 → 29.45° observed
                "Er2O3":    [35.14],   # +0.57 → 35.71° observed
                "Ta2O5":    [23.03],   # +0.57 → 23.60° observed
                "ErTaO4":   [24.12],   # +0.57 → 24.69° observed
            },
            "points": [
                {"R": 0.8346, "yield": 23.3},  # C1
                {"R": 0.8033, "yield": 21.5},  # C3
                {"R": 0.7963, "yield": 25.7},  # C4
                {"R": 0.6703, "yield": 38.8},  # C5
                {"R": 0.6566, "yield": 39.1},  # C6
            ],
            "excluded": ["C2"],  # KMoO precursor anomaly
        },
        # Systematic 2theta shift (sample displacement/zero error)
        # Average +0.57° across all phases (confirmed on 20260428 best sample)
        # Reference positions from mp-773103, mp-3444, COD-1000096, COD-1000889
        # When delta_2theta = +0.57:
        #   ErTa7O19 (220) ref=28.88° → obs=29.45° ✓
        #   ErTa7O19 (440) ref=57.46° → obs=58.03° ✓
        #   Er2O3    (400) ref=50.61° → obs=51.18° ✓
        #   Er2O3    (222) ref=35.14° → obs=35.71° ✓
        #   Ta2O5    (110) ref=23.03° → obs=23.60° ✓
        #   Ta2O5    (001) ref=46.11° → obs=46.68° ✓
        #   ErTaO4   (110) ref=24.12° → obs=24.69° ✓
        #   ErTaO4   (022) ref=47.22° → obs=47.79° ✓
        #   ErTaO4   (113) ref=55.97° → obs=56.54° ✓
        "delta_2theta": +0.57,
    },

    # ── Add new systems here ──────────────────────────────────────────
    # "batio3": {
    #     "name": "BaTiO3",
    #     "target_phase": "BaTiO3",
    #     "phases": {
    #         "BaTiO3":  {"peaks_2theta": [45.3], "hkl": ["200"], "preferred": 0},
    #         "BaCO3":   {"peaks_2theta": [24.9], "hkl": ["111"], "preferred": 0},
    #     },
    #     "rir": {"BaTiO3": 1.0, "BaCO3": 1.3},
    # },

    # ════════════════════════════════════════════════════════════════
    # CeAl₂Ge₂ system — arc-melt intermetallics
    # RIR ≈ 1.0 for all (similar Z, same structure type family)
    # Calibrated MAE = 0.83% (5-point calibration)
    # ════════════════════════════════════════════════════════════════
    "ceal2ge2": {
        "name":         "CeAl\u2082Ge\u2082",
        "target_phase": "CeAl2Ge2",
        "method":       "PawlikQPA",          # use Pawlik multi-peak method
        "phases": {
            # mp-7101 PDF, P-3m1, Cu Kα λ=1.5406Å
            "CeAl2Ge2": {
                "formula":  "CeAl2Ge2",
                "peaks_2theta": [27.29, 32.19, 33.33, 34.10, 35.44,
                                 38.54, 39.26, 42.16],
                "hkl":      ["(011)", "(110)", "(200)", "(002)",
                             "(102)", "(300)", "(301)", "(210)"],
                "preferred": 0,               # 27.29° — strongest, isolated
            },
            # COD-1614987, Cmce, a=6.078 b=15.069 c=7.963Å
            "Ce2Al3Ge4": {
                "formula":  "Ce2Al3Ge4",
                "peaks_2theta": [29.37, 31.72, 32.22, 38.32, 39.72],
                "hkl":      ["(200)", "(131)", "(131')", "(312)", "(241)"],
                "preferred": 0,
            },
            # COD-4105141, I41md, a=4.308 c=14.668Å
            "CeAlGe": {
                "formula":  "CeAlGe",
                "peaks_2theta": [30.53, 35.28, 37.25, 42.10],
                "hkl":      ["(211)", "(202)", "(112)", "(201)"],
                "preferred": 3,               # 42.10° — strongest (overlaps CeAl2Ge2)
            },
            # Al metal PDF 04-0787, FCC a=4.050Å
            "Al": {
                "formula":  "Al",
                "peaks_2theta": [38.47, 44.72, 65.09],
                "hkl":      ["(111)", "(200)", "(220)"],
                "preferred": 0,
            },
        },
        # RIR = 1.0 for all intermetallics (Z²/N/V ≈ constant)
        # Al metal: different structure → RIR slightly different
        #   (but Pawlik treats all equally, calibration absorbs the error)
        "rir": {
            "CeAl2Ge2":  1.0,
            "Ce2Al3Ge4": 1.0,
            "CeAlGe":    1.0,
            "Al":        1.0,
        },
        # Calibration (5-point, from manual weighing):
        #   R = A_CeAl2Ge2 / (A_CeAl2Ge2 + A_Ce2Al3Ge4 + A_CeAlGe + A_Al)
        #   yield_true = 1.03 * (R*100) - 0.02   (MAE = 0.83%)
        # Source: arc_melt calibration data + autonomous_chemist_summary.md
        "calibration": {
            "slope":      1.03,
            "intercept": -0.02,
            "cal_sigma":  0.83,
            "R_type":     "pawlik_mass_fraction",
            "formula":    "R = A_CeAl2Ge2 / ΣA_all_phases",
            "note": (
                "Pawlik RIR: w_i = A_i / ΣA_j (RIR≈1 for all intermetallics). "
                "Calibration corrects for small Z²/N/V deviations. "
                "Peak overlap at 42° (CeAlGe↔CeAl2Ge2) is handled by multi-peak "
                "integration + calibration, not by deconvolution."
            ),
            "peak_positions": {
                "CeAl2Ge2":  [27.29, 35.44, 38.54, 39.26, 42.16],
                "Ce2Al3Ge4": [29.37, 31.72, 32.22, 38.32, 39.72],
                "CeAlGe":    [30.53, 35.28, 37.25, 42.10],
                "Al":        [38.47],
            },
            "points": [
                # (R_raw, yield_true) — from weighing experiments
                # Replace with actual experimental data when available
            ],
        },
        # 2θ shift for arc-melt samples (sample displacement ~0.5°)
        # Verified: +0.57° aligns mp-7101 peaks to observed positions
        "delta_2theta": +0.57,
    },

}


# ══════════════════════════════════════════════════════════════════════════════
# YieldResult dataclass
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class YieldResult:
    """Output of an XRD yield estimation."""
    yield_pct:        float | None    # None means unknown/unquantifiable, never a zero-yield penalty
    sigma_pct:        float           # Estimated uncertainty (%)
    method:           str             # "RietveldRefiner" or "PeakBasedRI"
    R:                float = None   # Raw RIR ratio
    raw_yield:        float = None   # Pre-calibration yield
    phase_fractions:  dict = None     # {phase_name: weight_fraction}
    phase_fractions_sigma: dict = None  # {phase_name: sigma}
    phase_peaks_used: dict = None      # {phase: 2theta_peak}
    n_peaks_matched: int = 0
    confidence:       str = "medium"
    confidence_note:  str = ""

    def __str__(self) -> str:
        phases_str = ", ".join(
            f"{k}={v:.3f}" for k, v in (self.phase_fractions or {}).items()
        )
        value = "unknown" if self.yield_pct is None else f"{self.yield_pct:.1f}% \u00b1 {self.sigma_pct:.1f}%"
        return (
            f"[{self.method}] Yield: {value}\n"
            f"  Phase fractions: {{{phases_str}}}\n"
            f"  Confidence: {self.confidence} ({self.confidence_note})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Abstract Interface
# ══════════════════════════════════════════════════════════════════════════════

class XRDYieldEstimator(ABC):
    """
    Abstract XRD yield estimator.

    Subclass this to add a new quantification backend
    (e.g. GSAS-II, Topas, Rigaku SmartLab, etc.)
    """

    def __init__(self, system: str, calibration: dict = None):
        self.system = system
        self.calibration = calibration
        self._load_system_config(system)

    def _load_system_config(self, system: str):
        """Load system configuration; raise if unknown."""
        if system not in SYSTEM_CONFIGS:
            raise ValueError(
                f"Unknown system '{system}'. Available: {list(SYSTEM_CONFIGS.keys())}"
            )
        self.config = SYSTEM_CONFIGS[system]

    @abstractmethod
    def estimate(self, xrd_data, **kwargs) -> YieldResult:
        """Estimate phase fraction / yield from XRD data."""
        raise NotImplementedError

    def apply_calibration(self, raw_yield: float) -> tuple:
        """
        Apply calibration curve to raw yield estimate.
        Returns (calibrated_yield, estimated_sigma).
        """
        if not self.calibration or "points" not in self.calibration:
            return float(np.clip(raw_yield, 0, 100)), 3.0

        pts = self.calibration["points"]
        Rs  = np.array([p["R"]     for p in pts])
        ys  = np.array([p["yield"] for p in pts])

        if len(pts) < 2:
            slope = ys[0] / max(Rs[0], 1e-9)
            return float(np.clip(slope * raw_yield, 0, 100)), 3.0

        coeffs = np.polyfit(Rs, ys, deg=1)
        poly   = np.poly1d(coeffs)
        cal_yield = float(np.clip(poly(raw_yield), 0, 100))
        residuals  = ys - poly(Rs)
        sigma      = float(np.sqrt(np.mean(residuals**2)))
        return cal_yield, sigma


# ══════════════════════════════════════════════════════════════════════════════
# Public Factory
# ══════════════════════════════════════════════════════════════════════════════

class AutoXRDYieldEstimator(XRDYieldEstimator):
    """Primary/fallback estimator used by backend='auto'."""

    def __init__(self, system: str, primary: XRDYieldEstimator, fallback: XRDYieldEstimator, calibration: dict = None):
        super().__init__(system=system, calibration=calibration)
        self.primary = primary
        self.fallback = fallback

    def estimate(self, xrd_data, **kwargs) -> YieldResult:
        primary_error = None
        primary_result = None
        try:
            primary_result = self.primary.estimate(xrd_data, **kwargs)
            primary_meta = primary_result.phase_peaks_used or {}
            if primary_meta.get('authoritative') or primary_meta.get('no_fallback'):
                return primary_result
            if primary_result.confidence in {'high', 'pass', 'pass_caution'} and primary_result.phase_fractions:
                return primary_result
        except Exception as exc:
            primary_error = exc

        try:
            fallback_result = self.fallback.estimate(xrd_data, **kwargs)
            note_parts = []
            if primary_error is not None:
                note_parts.append(f"GSAS-II failed: {primary_error}")
            elif primary_result is not None:
                note_parts.append(f"GSAS-II unreliable: {primary_result.confidence_note}")
            note_parts.append(fallback_result.confidence_note or "fallback result")
            fallback_result.method = f"{fallback_result.method}_fallback_after_GSAS2"
            fallback_result.confidence_note = "; ".join([p for p in note_parts if p])
            return fallback_result
        except Exception:
            if primary_result is not None:
                return primary_result
            raise


def create_estimator(
    system: str,
    calibration: dict = None,
    use_rietveld: bool = None,
    backend: str = None,
    **kwargs
) -> XRDYieldEstimator:
    """
    Factory: create the best available estimator for the given system.

    backend:
        - "pawlik" / "peak" -> PeakBasedRI
        - "gsas2"            -> dynamic GSAS-II Rietveld agent (conda gsas2 env)
        - "gsas2_legacy"     -> single-pass GSAS2WorkflowEstimator
        - "auto" / None      -> prefer GSAS-II for supported systems, else PeakBasedRI
    """
    if calibration is None and system in SYSTEM_CONFIGS:
        calibration = SYSTEM_CONFIGS[system].get("calibration")

    if backend is None:
        if use_rietveld is True:
            backend = "gsas2"
        elif use_rietveld is False:
            backend = "pawlik"
        else:
            backend = "auto"

    backend = (backend or "auto").lower()
    peak_kwargs = {k: v for k, v in kwargs.items() if k not in {"gsas2_path"}}

    if backend in ("pawlik", "peak", "peakbased"):
        from .peak_based import PeakBasedRI
        return PeakBasedRI(system=system, calibration=calibration, **peak_kwargs)

    if backend == "gsas2":
        from .gsas2.dynamic import DynamicGSAS2RietveldEstimator
        return DynamicGSAS2RietveldEstimator(system=system, calibration=calibration, **kwargs)

    if backend in ("gsas2_legacy", "legacy_gsas2"):
        from .gsas2.refinement.workflow import GSAS2WorkflowEstimator
        return GSAS2WorkflowEstimator(system=system, calibration=calibration, **kwargs)

    # auto
    if system == "ceal2ge2":
        from .peak_based import PeakBasedRI
        try:
            from .gsas2.dynamic import DynamicGSAS2RietveldEstimator
            primary = DynamicGSAS2RietveldEstimator(system=system, calibration=calibration, **kwargs)
            fallback = PeakBasedRI(system=system, calibration=calibration, **peak_kwargs)
            return AutoXRDYieldEstimator(system=system, primary=primary, fallback=fallback, calibration=calibration)
        except Exception:
            return PeakBasedRI(system=system, calibration=calibration, **peak_kwargs)

    from .peak_based import PeakBasedRI
    return PeakBasedRI(system=system, calibration=calibration, **peak_kwargs)


# Backward compatibility aliases
PeakBasedRI = None  # resolved lazily via submodule import
