"""Autonomous chemist Dynamic GSAS-II Rietveld backend for Ce-Al-Ge arc-melt QPA.

This backend wraps the production rietveld_engine dynamic agent instead of the
older single-pass GSAS2WorkflowEstimator.  It returns the same YieldResult public
interface used by autonomous_chemist and BO code, with structured metadata for
status, residual diagnosis, artifacts, and fallback control.

Design rule: do not silently turn an unknown/unquantifiable Rietveld result into
a high Pawlik yield.  Those results set phase_peaks_used['no_fallback']=True.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from .. import YieldResult
from ... import paths


ARC_ROOT = paths.ARC_RAW.parent
RIETVELD_ENGINE = paths.RIETVELD_ENGINE
ITER_RUNS = paths.RIETVELD_RUNS / "iter"
RESIDUAL_RUNS = paths.RIETVELD_RUNS / "residual_diagnosis"
TARGET_PHASE_BY_SYSTEM = {"ceal2ge2": "CeAl2Ge2"}


class DynamicGSAS2RietveldEstimator:
    """Estimator that delegates to rietveld_engine/iter_agent.py.

    Parameters are intentionally compatible with create_estimator(...).
    """

    def __init__(
        self,
        system: str,
        calibration: Optional[dict] = None,
        engine_dir: Optional[str] = None,
        conda_env: str = None,
        target_rwp: float = 15.0,
        timeout: int = 900,
        **kwargs: Any,
    ):
        self.system = system
        self.calibration = calibration or {}
        self.engine_dir = Path(engine_dir) if engine_dir else RIETVELD_ENGINE
        self.conda_env = conda_env or paths.GSAS2_CONDA_ENV
        self.target_rwp = float(target_rwp)
        self.timeout = int(timeout)
        self.kwargs = kwargs

    def estimate(self, xrd_data, **kwargs) -> YieldResult:
        report = self._run_dynamic_agent(xrd_data, **kwargs)
        return self._report_to_yield_result(report, xrd_data=xrd_data)

    def _sample_name_from_xrd(self, xrd_data) -> str:
        p = Path(str(xrd_data))
        if p.suffix.lower() in {".raw", ".xy", ".mdi", ".txt"}:
            return p.stem
        # If caller passes an existing report-like sample key, keep it literal.
        return str(xrd_data)

    def _ensure_arc_raw_available(self, xrd_data, sample: str) -> None:
        """iter_agent expects <arc_raw>/{sample}.raw.

        For normal arc-melt runs the RAW file already exists.  If caller passes a
        different RAW path, copy it into ARC_ROOT so the legacy agent can run.
        Do not synthesize data.
        """
        src = Path(str(xrd_data))
        expected = paths.ARC_RAW / f"{sample}.raw"
        if expected.exists() or not src.exists() or src.resolve() == expected.resolve():
            return
        if src.suffix.lower() != ".raw":
            return
        expected.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, expected)

    def _target_phase(self) -> str:
        if self.system in TARGET_PHASE_BY_SYSTEM:
            return TARGET_PHASE_BY_SYSTEM[self.system]
        from .. import SYSTEM_CONFIGS
        return (SYSTEM_CONFIGS.get(self.system) or {}).get("target_phase", self.system)

    def _agent_command(self, sample: str, xrd_data) -> tuple[list[str], Path]:
        """Pick the production agent when present, else the packaged generic agent."""
        prod = self.engine_dir / "iter_agent.py"
        base = ["conda", "run", "--no-capture-output", "-n", self.conda_env, "python"]
        if prod.exists():
            return base + [str(prod), sample], self.engine_dir / "runs" / "iter" / sample / "report.json"

        out_dir = ITER_RUNS / sample
        system = "-".join(self._system_elements())
        return base + [
            "-m", "chem.xrd.gsas2.agent",
            "--sample", sample,
            "--target", self._target_phase(),
            "--xrd", str(xrd_data),
            "--system", system,
            "--output", str(out_dir),
            "--target-rwp", str(self.target_rwp),
        ], out_dir / "report.json"

    def _system_elements(self) -> list[str]:
        from ...energetics.periodic import parse_formula
        return list(parse_formula(self._target_phase()) or {})

    def _run_dynamic_agent(self, xrd_data, **kwargs) -> Dict[str, Any]:
        sample = kwargs.get("sample") or self._sample_name_from_xrd(xrd_data)
        self._ensure_arc_raw_available(xrd_data, sample)

        cmd, report_path = self._agent_command(sample, xrd_data)
        env = os.environ.copy()
        env["HOME"] = env.get("HOME", str(Path.home()))
        env["PYTHONPATH"] = os.pathsep.join(
            [str(paths.HOME), str(paths.HOME / "compat"), str(self.engine_dir),
             env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        env.setdefault("GSAS2_REFINEMENT_SUBPROCESS", "1")
        env.setdefault("CHEMIST_ARC_RAW", str(paths.ARC_RAW))
        env.setdefault("CHEMIST_CIF_DIR", str(paths.RIETVELD_CIFS))
        proc = subprocess.run(
            cmd,
            cwd=str(self.engine_dir) if (self.engine_dir / "iter_agent.py").exists() else str(paths.HOME),
            env=env,
            text=True,
            capture_output=True,
            timeout=self.timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "dynamic Rietveld agent failed "
                f"(rc={proc.returncode}): {proc.stderr[-2000:] or proc.stdout[-2000:]}"
            )
        if not report_path.exists():
            raise FileNotFoundError(f"dynamic Rietveld report not found: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report.setdefault("artifacts", {})
        report["artifacts"].update({
            "report_path": str(report_path),
            "run_dir": str(report_path.parent),
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
        })
        if not (self.engine_dir / "residual_peak_diagnose.py").exists():
            return None
        report["residual_diagnosis"] = self._run_residual_diagnosis(sample, report_path, env)
        return report

    def _run_residual_diagnosis(self, sample: str, report_path: Path, env: dict) -> Optional[Dict[str, Any]]:
        gpx = report_path.parent / "project_initial.gpx"
        if not gpx.exists():
            return None
        out_dir = RESIDUAL_RUNS / sample
        cmd = [
            "conda", "run", "--no-capture-output", "-n", self.conda_env,
            "python", str(self.engine_dir / "residual_peak_diagnose.py"),
            sample, "--gpx", str(gpx), "--report", str(report_path), "--out", str(out_dir),
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(self.engine_dir),
            env=env,
            text=True,
            capture_output=True,
            timeout=max(180, min(self.timeout, 600)),
        )
        diag_path = out_dir / "residual_diagnosis.json"
        if proc.returncode != 0 or not diag_path.exists():
            return {
                "diagnosis": "residual_diagnosis_failed",
                "error": (proc.stderr or proc.stdout)[-2000:],
                "output": str(diag_path),
            }
        return json.loads(diag_path.read_text(encoding="utf-8"))

    def _report_to_yield_result(self, report: Dict[str, Any], xrd_data=None) -> YieldResult:
        target = TARGET_PHASE_BY_SYSTEM.get(self.system, self.system)
        fracs_pct = report.get("phase_fractions_wt") or {}
        sigmas_pct = report.get("phase_sigma_wt") or {}
        phase_fractions = {k: float(v) / 100.0 for k, v in fracs_pct.items()}
        phase_sigmas = {k: float(v) / 100.0 for k, v in sigmas_pct.items()}
        target_wt_pct = float(fracs_pct.get(target, 0.0) or 0.0)
        target_sigma_pct = float(sigmas_pct.get(target, 0.0) or 0.0)
        rwp = _to_float(report.get("rwp"))
        gof = _to_float(report.get("gof"))
        residual = report.get("residual_diagnosis") or {}
        status, sigma, use_for_bo, no_fallback, note = self._classify_result(
            target_wt_pct=target_wt_pct,
            target_sigma_pct=target_sigma_pct,
            rwp=rwp,
            gof=gof,
            residual=residual,
            report=report,
        )
        yield_value = None if status in {"unknown_not_quantifiable", "failed_current_model"} else target_wt_pct
        raw_yield = target_wt_pct
        meta = {
            "backend": "dynamic_gsas2_rietveld_agent",
            "authoritative": True,
            "status": status,
            "use_for_bo": use_for_bo,
            "no_fallback": no_fallback,
            "label_type": _label_type(status),
            "target_phase": target,
            "dominant": report.get("dominant"),
            "phases_used": report.get("phases_used"),
            "fixed_phases": report.get("fixed_phases"),
            "killed_phases": report.get("killed_phases"),
            "artifacts": report.get("artifacts") or {},
            "report_path": (report.get("artifacts") or {}).get("report_path"),
            "run_dir": (report.get("artifacts") or {}).get("run_dir"),
            "residual_diagnosis": residual,
            "unknown_residual_peaks_2theta": [
                p.get("two_theta") for p in residual.get("residual_peaks", [])[:20]
                if isinstance(p, dict) and p.get("two_theta") is not None
            ],
            "metrics": {"rwp": rwp, "gof": gof},
            "note": note,
            "agent_log": report.get("log", []),
        }
        return YieldResult(
            yield_pct=yield_value,
            sigma_pct=float(sigma),
            method="GSAS2DynamicRietveld",
            R=float(target_wt_pct) / 100.0,
            raw_yield=float(raw_yield),
            phase_fractions=phase_fractions,
            phase_fractions_sigma=phase_sigmas,
            phase_peaks_used=meta,
            n_peaks_matched=len(phase_fractions),
            confidence=status,
            confidence_note=note,
        )

    def _classify_result(self, target_wt_pct: float, target_sigma_pct: float, rwp, gof, residual, report):
        diag = str((residual or {}).get("diagnosis", ""))
        sample = str(report.get("sample", ""))
        phase_names = set((report.get("phase_fractions_wt") or {}).keys()) | set(report.get("phases_used") or [])
        if self.system == "ceal2ge2" and ("Ga" in sample or any("Ga" in str(p) for p in phase_names)):
            return (
                "unknown_not_quantifiable",
                30.0,
                False,
                True,
                "Ga-containing/quaternary sample is outside the Ce-Al-Ge BO training domain; do not fallback to Pawlik",
            )
        if (not _is_finite(target_sigma_pct)) or target_sigma_pct > 50.0:
            return (
                "failed_current_model",
                20.0,
                False,
                True,
                "target phase fraction uncertainty is non-physical/sigma-blown-up; physical yield unknown, excluded from numeric BO",
            )
        n_resid = int((residual or {}).get("n_residual_peaks") or 0)
        no_candidate = "no_candidate_phase_matches_residual_peaks" in diag
        bad_fit = (rwp is None) or (rwp >= 25.0) or (gof is not None and gof >= 5.0)
        near_fit = (rwp is not None and 15.0 < rwp < 20.0) or (gof is not None and 3.0 < gof < 5.0)
        caution_fit = (rwp is not None and 12.0 < rwp <= 15.0) or (gof is not None and 2.3 < gof <= 3.0)

        if no_candidate and n_resid >= 2:
            return (
                "unknown_not_quantifiable",
                30.0,
                False,
                True,
                "significant residual peaks are not explained by available candidate phases; do not fallback to Pawlik",
            )
        if bad_fit:
            return (
                "failed_current_model",
                20.0,
                False,
                True,
                "dynamic Rietveld model failed quality thresholds; physical yield unknown, excluded from numeric BO",
            )
        if near_fit:
            return ("near", max(12.0, target_sigma_pct, 0.0), True, False, "near-threshold Rietveld fit; use low BO weight")
        if caution_fit:
            return ("pass_caution", max(8.0, target_sigma_pct, 0.0), True, False, "Rietveld fit usable but not publication-grade")
        return ("pass", max(3.0, target_sigma_pct, 0.0), True, False, "Rietveld fit passed dynamic-agent thresholds")


def _to_float(v):
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _is_finite(v) -> bool:
    try:
        return v is not None and float("-inf") < float(v) < float("inf")
    except Exception:
        return False


def _label_type(status: str) -> str:
    if status in {"unknown_not_quantifiable", "failed_current_model"}:
        return "unknown_physical_yield_no_numeric_label"
    if status == "near":
        return "low_weight_rietveld_CeAl2Ge2_wt_percent"
    return "rietveld_CeAl2Ge2_wt_percent"
