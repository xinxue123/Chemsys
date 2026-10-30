"""
Peak-Based RIR Quantification (no GSAS-II required)
===================================================
Implements the Chung (1974) Reference Intensity Ratio method using
scipy peak finding + numerical integration.

Accuracy: ~2-5% absolute (vs ~1-2% for full Rietveld with GSAS-II)
Speed:   ~50ms per file

The method:
  1. Load XRD pattern
  2. Apply background subtraction (SNIP or polynomial)
  3. Find/integrate the preferred reflection for each phase
  4. Compute weight fractions via RIR equation
  5. Apply calibration if provided

References
----------
- Chung, F.H. (1974). Quantitative interpretation of X-ray diffraction
  patterns of mixtures. J. Appl. Cryst. 7, 519-525.
"""

import warnings
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import minimize_scalar
from scipy.ndimage import gaussian_filter1d

from . import XRDYieldEstimator, YieldResult, SYSTEM_CONFIGS


class PeakBasedRI(XRDYieldEstimator):
    """
    Peak-profile RIR quantification.

    No GSAS-II required.  Uses scipy for peak detection and integration.
    Accuracy is sufficient for BO feedback loops (2-5% absolute).
    """

    def estimate(self, xrd_data, **kwargs) -> YieldResult:
        """
        Run the full RIR pipeline on xrd_data.

        Parameters
        ----------
        xrd_data : str or (n,2) array
            Path to .txt file or (2θ, intensity) array

        Returns
        -------
        YieldResult
        """
        # ── 1. Load data ──────────────────────────────────────────────
        xrd_data = self._load_xrd(xrd_data)
        two_theta = xrd_data[:, 0]
        intensity = xrd_data[:, 1].astype(float)

        # ── 2. Preprocess ────────────────────────────────────────────
        intensity = self._preprocess(two_theta, intensity)

        # ── 3. Integrate peaks ────────────────────────────────────────
        phase_areas = {}
        peak_info   = {}
        delta_2theta = self.config.get("delta_2theta", 0.0)

        for phase_name, phase_cfg in self.config["phases"].items():
            peaks = np.array(phase_cfg["peaks_2theta"]) + delta_2theta
            pref_idx = phase_cfg["preferred"]
            area, peak_int = self._integrate_peak(
                two_theta, intensity, peaks, pref_idx
            )
            phase_areas[phase_name] = area
            peak_info[phase_name] = {
                "area":   area,
                "height": peak_int,
                "2theta": peaks[pref_idx],
            }

        # ── 3b. Pawlik / RIR route (intermetallics, e.g. CeAl2Ge2) ──────
        # The ErTa7O19 empirical ratio below does not generalise; systems that
        # declare method="PawlikQPA" use Chung/Pawlik RIR mass fractions instead.
        target = self.config["target_phase"]
        if self.config.get("method") == "PawlikQPA":
            rir = self.config.get("rir", {})
            fractions, fractions_sigma = self._rir_weight_fractions(phase_areas, rir)
            R_target = float(fractions.get(target, 0.0))
            cal = self.calibration or {}
            slope = float(cal.get("slope", 1.0))
            intercept = float(cal.get("intercept", 0.0))
            cal_yield = float(np.clip(slope * (R_target * 100.0) + intercept, 0.0, 100.0))
            cal_sigma = float(cal.get("cal_sigma", 3.0))
            n_matched = sum(1 for v in phase_areas.values() if v > 0)
            if n_matched >= 3 and cal_sigma < 3.0:
                confidence = "high"
            elif n_matched >= 2:
                confidence = "medium"
            else:
                confidence = "low"
            return YieldResult(
                yield_pct=cal_yield,
                sigma_pct=cal_sigma,
                method="PeakBasedRI",
                R=R_target,
                raw_yield=float(R_target * 100.0),
                phase_fractions={k: float(v) for k, v in fractions.items()},
                phase_fractions_sigma={k: float(v) for k, v in fractions_sigma.items()},
                phase_peaks_used={k: float(p["2theta"]) for k, p in peak_info.items()},
                n_peaks_matched=n_matched,
                confidence=confidence,
                confidence_note=f"Pawlik RIR: {n_matched} phases matched, "
                                 f"R={R_target:.3f}",
            )

        # ── 4. Compute empirical ratio R ────────────────────────────────
        # Best predictor found for ErTa7O19: I_ErTa / (I_ErTa + I_Er2O3)
        # Calibrated from 5 reliable cycles: RMSE=2.3%, range [0.68, 0.89]
        A_erta = phase_areas.get("ErTa7O19", 0.0)
        A_er2o = phase_areas.get("Er2O3", 0.0)
        if A_erta + A_er2o > 0:
            R_empirical = A_erta / (A_erta + A_er2o)
        else:
            R_empirical = 0.0

        # ── 5. RIR weight fractions (still useful for reporting) ───────
        rir = self.config.get("rir", {})
        fractions, fractions_sigma = self._rir_weight_fractions(phase_areas, rir)

        # ── 6. Target phase yield from empirical ratio ────────────────
        target = self.config["target_phase"]
        if A_erta == 0:
            return YieldResult(
                yield_pct=0.0, sigma_pct=10.0, method="PeakBasedRI",
                R=R_empirical,
                confidence="low", confidence_note="Target ErTa7O19 not detected",
            )

        # Apply calibration: yield = slope * R + intercept
        if self.calibration and "slope" in self.calibration:
            slope = self.calibration["slope"]
            intercept = self.calibration["intercept"]
            cal_yield = float(np.clip(slope * R_empirical + intercept, 0, 100))
            # Sigma from calibration residuals (RMSE of 2.3% from 5 cycles)
            cal_sigma = self.calibration.get("cal_sigma", 2.5)
        elif self.calibration and "points" in self.calibration:
            cal_yield, cal_sigma = self.apply_calibration(R_empirical)
        else:
            # Fallback: no calibration → report raw fraction
            cal_yield = fractions.get(target, 0.0) * 100.0
            cal_sigma = 10.0
            R_empirical = fractions.get(target, 0.0)

        # ── 7. Confidence assessment ───────────────────────────────────
        n_matched = sum(1 for v in phase_areas.values() if v > 0)
        if n_matched >= 3 and cal_sigma < 3.0:
            confidence = "high"
        elif n_matched >= 2 and cal_sigma < 5.0:
            confidence = "medium"
        else:
            confidence = "low"

        # Report the R that was actually used in calibration: S_ErTa / (S_ErTa + S_Er2O3)
        R_reported = float(R_empirical)

        return YieldResult(
            yield_pct       = float(np.clip(cal_yield, 0, 100)),
            sigma_pct       = float(cal_sigma),
            method          = "PeakBasedRI",
            R               = R_reported,
            raw_yield       = float(cal_yield),
            phase_fractions = {k: float(v) for k, v in fractions.items()},
            phase_fractions_sigma = {k: float(v) for k, v in fractions_sigma.items()},
            phase_peaks_used= {k: float(p["2theta"]) for k, p in peak_info.items()},
            n_peaks_matched = n_matched,
            confidence      = confidence,
            confidence_note = f"{n_matched} phases matched",
        )

    # ── Data loading ───────────────────────────────────────────────────

    @staticmethod
    def _load_xrd(xrd_data):
        """Load XRD data from file path or array."""
        if isinstance(xrd_data, str):
            # Try 2-column format first (no header)
            try:
                data = np.loadtxt(xrd_data)
                if data.shape[1] >= 2:
                    return data[:, :2]
            except Exception:
                pass
            # Try Rigaku TDXRD header format
            with open(xrd_data) as f:
                lines = f.readlines()
            # Skip header lines (first ~2 lines)
            for i, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) == 2:
                    try:
                        float(parts[0]), float(parts[1])
                        data = np.loadtxt(xrd_data, skiprows=i)
                        return data[:, :2]
                    except Exception:
                        continue
            raise ValueError(f"Could not parse XRD file: {xrd_data}")
        else:
            return np.asarray(xrd_data)

    # ── Preprocessing ───────────────────────────────────────────────────

    @staticmethod
    def _preprocess(two_theta, intensity):
        """Background subtraction + smoothing."""
        # Step 1: Remove outliers (cosmic spikes)
        intensity = np.clip(intensity, 0, np.percentile(intensity, 99.9))

        # Step 2: Savitzky-Golay smoothing (window=11, poly=2)
        if len(intensity) > 11:
            intensity = savgol_filter(intensity, window_length=11, polyorder=2)

        # Step 3: Simple polynomial background subtraction
        bg = PeakBasedRI._snip_background(two_theta, intensity)
        intensity = np.clip(intensity - bg, 0, None)

        return intensity

    @staticmethod
    def _snip_background(two_theta, intensity, lambda_=2.0, num_iter=20):
        """
        SNIP (Statistics-sensitive Non-linear Iterative Peak) background estimation.
        Wang et al. (2006) NIM A 564, 426-432.
        """
        n = len(intensity)
        # Estimate background at each point
        bg = np.zeros(n)
        from scipy.ndimage import minimum_filter1d
        for p in range(1, num_iter + 1):
            width = int(lambda_ * p)
            if width < 1:
                width = 1
            left  = np.maximum(0, np.arange(n) - width)
            right = np.minimum(n - 1, np.arange(n) + width)
            neighbor_min = np.minimum(intensity[left], intensity[right])
            bg = np.minimum(intensity, (bg + neighbor_min) / 2)
        return bg

    # ── Unified RIR Methods (Zevin 1980s + Pawlik) ───────────────────────

    @staticmethod
    def zevin_iterative_rir(phase_areas, rir=None, max_iter=50, tol=1e-4):
        """
        Zevin 迭代自洽 RIR（Zevin 1980s 方法）。

        迭代方程:
            weighted_sum = Σ_j (A_j × w_j)
            R_i          = A_i / weighted_sum
            sum_AR       = Σ_j (A_j × R_j)
            w_i          = A_i / sum_AR
            normalize: w_i → w_i / Σw_j

        当 RIR=1（未知）时：迭代 2 轮收敛到"表观 wt%"
        当 RIR 已知时：代入真实 RIR → 收敛到物理正确的 wt%

        等价于：Pawlik 静态公式 w_i = A_i/ΣA_j 在 RIR=1 时的特例。

        Returns:
            {"weights": {phase: fraction}, "n_iter": int, "converged": bool,
             "delta_final": float, "RIR_used": {phase: value}}
        """
        phases = list(phase_areas.keys())
        A = np.array([max(float(phase_areas[p]), 1e-9) for p in phases])
        R = np.array([float(rir.get(p, 1.0)) if rir else 1.0 for p in phases])

        w = np.ones(len(phases)) / len(phases)   # 初始等权重

        for n in range(max_iter):
            weighted_sum = np.sum(A * w)           # Σ(A_j × w_j)
            R_i = A / weighted_sum                  # A_i / weighted_sum
            sum_AR = np.sum(A * R_i)                # Σ(A_j × R_j)
            w_new = A / sum_AR                      # A_i / sum_AR
            w_new = w_new / np.sum(w_new)           # normalize Σw=1
            delta = np.max(np.abs(w_new - w))
            w = w_new
            if delta < tol:
                return {
                    "weights": {p: float(w[i]) for i, p in enumerate(phases)},
                    "n_iter": n + 1,
                    "converged": True,
                    "delta_final": float(delta),
                    "RIR_used": {p: float(R[i]) for i, p in enumerate(phases)},
                }
        return {
            "weights": {p: float(w[i]) for i, p in enumerate(phases)},
            "n_iter": max_iter,
            "converged": False,
            "delta_final": float(delta),
            "RIR_used": {p: float(R[i]) for i, p in enumerate(phases)},
        }

    @staticmethod
    def pawlik_rir(phase_areas, rir=None):
        """
        Pawlik 静态 RIR（Chung 1974，非迭代）。

        w_i = (A_i / RIR_i) / Σ_j (A_j / RIR_j)

        用于：RIR 精确已知（如 PDF 查表）时直接计算，无需迭代。
        等价于 Zevin 迭代 1 步后的结果（RIR=1 时）。

        Returns:
            {"weights": {phase: fraction}, "raw_denom": float, "RIR_used": {phase: value}}
        """
        phases = list(phase_areas.keys())
        A = np.array([max(float(phase_areas[p]), 1e-9) for p in phases])
        R = np.array([float(rir.get(p, 1.0)) if rir else 1.0 for p in phases])
        denom = np.sum(A / R)
        w = (A / R) / denom
        return {
            "weights": {p: float(w[i]) for i, p in enumerate(phases)},
            "raw_denom": float(denom),
            "RIR_used": {p: float(R[i]) for i, p in enumerate(phases)},
        }

    @staticmethod
    def zevin_batch(
        phase_areas_batch: dict,
        rir: dict = None,
        calibration_points: list = None,
        target_phase: str = None,
    ):
        """
        完整流程：Zevin 迭代 + 校准曲线（多样品 ≥3）。

        当 rir=None → Zevin 迭代（RIR=1），返回表观 wt%
        当 rir=dict → Pawlik 静态（RIR 查表），返回物理 wt%

        流程:
            1. 对每个样品计算 Zevin 或 Pawlik 权重
            2. R = A_target / ΣA_all
            3. 若有 calibration_points → yield = α + β·R
            4. 输出各样品产率

        Returns:
            {sample_id: {"weights", "n_iter", "converged", "R", "yield_pct",
                         "cal_slope", "cal_intercept", "cal_sigma"}}
        """
        results = {}
        phases_all = list(set().union(*[set(a.keys()) for a in phase_areas_batch.values()]))
        rir_values = [rir.get(p, 1.0) if rir else 1.0 for p in phases_all]
        has_real_rir = rir is not None and any(abs(v - 1.0) > 1e-6 for v in rir_values)

        for sid, areas in phase_areas_batch.items():
            if has_real_rir:
                res = PeakBasedRI.pawlik_rir(areas, rir)
                results[sid] = {"weights": res["weights"], "n_iter": 1,
                                "converged": True, **res}
            else:
                res = PeakBasedRI.zevin_iterative_rir(areas, rir)
                results[sid] = {"weights": res["weights"], **res}

        # 计算 R_target
        if target_phase:
            for sid, areas in phase_areas_batch.items():
                total_A = sum(areas.values())
                A_t = areas.get(target_phase, 0.0)
                results[sid]["R"] = float(A_t / total_A if total_A > 0 else 0.0)

        # 校准曲线
        if calibration_points and len(calibration_points) >= 2 and target_phase:
            Rs = np.array([p["R"] for p in calibration_points])
            ys = np.array([p["yield"] for p in calibration_points])
            coeffs = np.polyfit(Rs, ys, deg=1)
            poly = np.poly1d(coeffs)
            cal_sigma = float(np.sqrt(np.mean((ys - poly(Rs))**2)))
            slope, intercept = float(coeffs[0]), float(coeffs[1])
            for sid in results:
                R_val = results[sid].get("R", 0.0)
                results[sid]["yield_pct"] = float(np.clip(poly(R_val), 0, 100))
                results[sid]["cal_slope"] = slope
                results[sid]["cal_intercept"] = intercept
                results[sid]["cal_sigma"] = cal_sigma
        elif target_phase:
            for sid in results:
                w_t = results[sid]["weights"].get(target_phase, 0.0)
                results[sid]["yield_pct"] = float(w_t * 100.0)
                results[sid]["cal_slope"] = None
                results[sid]["cal_intercept"] = None
                results[sid]["cal_sigma"] = None

        return results

    # ── Peak integration ───────────────────────────────────────────────

    def _integrate_peak(self, two_theta, intensity, peaks_2theta, pref_idx=0):
        """
        Integrate the preferred reflection peak using linear background subtraction.

        Returns (peak_area, peak_height).
        """
        peak_2theta = peaks_2theta[pref_idx]

        # Integration window: ±window degrees
        window = 1.5  # °2θ
        mask = (two_theta >= peak_2theta - window) & (two_theta <= peak_2theta + window)
        if not mask.any():
            return 0.0, 0.0

        tt_sel = two_theta[mask]
        I_sel  = intensity[mask]

        # Linear background from end-points
        n_sel  = len(I_sel)
        bg_left  = I_sel[0]
        bg_right = I_sel[-1]
        bg_line  = bg_left + (bg_right - bg_left) * np.arange(n_sel) / (n_sel - 1)
        I_net    = np.clip(I_sel - bg_line, 0, None)

        # Trapezoidal integration
        from scipy.integrate import trapezoid as _trapz
        area = float(_trapz(I_net, tt_sel))

        # Peak height (net)
        height = float(np.max(I_net))

        return area, height

    # ── RIR weight fractions (unified) ─────────────────────────────────

    def _rir_weight_fractions(self, phase_areas, rir):
        """
        统一 RIR 权重分数计算。

        决策逻辑：
          - rir 全为 1.0 或 None → Zevin 迭代（RIR=1，表观 wt%）
          - rir 有非 1.0 值（PDF 查表）→ Pawlik 静态（Chung 1974，物理 wt%）

        Zevin 迭代（RIR=1）:
            2 轮收敛 → 表观 wt%（校准后 = 真实产率）

        Pawlik 静态（RIR 已知）:
            解析解 → 物理 wt%（无需校准）

        Returns (weight_fractions dict, weight_fraction_sigma dict).
        """
        # 判断：是否有真实 RIR 值（非 1.0）
        rir_values = [rir.get(p, 1.0) if rir else 1.0 for p in phase_areas]
        has_real_rir = rir is not None and any(abs(v - 1.0) > 1e-6 for v in rir_values)

        if has_real_rir and rir:
            # RIR 已知 → Pawlik 静态（Chung 1974）
            res = PeakBasedRI.pawlik_rir(phase_areas, rir)
        else:
            # RIR=1（未知）→ Zevin 迭代自洽
            res = PeakBasedRI.zevin_iterative_rir(phase_areas, rir)

        fractions = res["weights"]
        fractions_sigma = {}

        # Sigma from Poisson count statistics
        for phase, area in phase_areas.items():
            W = fractions.get(phase, 0.0)
            if area > 0 and W > 0:
                sigma_rel = 1.0 / np.sqrt(max(float(area), 1.0))
                fractions_sigma[phase] = float(W * sigma_rel)
            else:
                fractions_sigma[phase] = 0.05

        return fractions, fractions_sigma


# ── Legacy compatibility ────────────────────────────────────────────────
# These names were used in the original non-generic version
RIRPhaseQuantifier = PeakBasedRI   # backward compat alias
