#!/usr/bin/env python3
"""Batch dynamic-Rietveld recompute for a directory of arc-melt XRD patterns.

Output is written incrementally so partial results survive failures/interruption.
This module never synthesizes data: each row is either a real dynamic-Rietveld
result or a structured error record.

Usage::

    python -m chem.cli xrd batch <raw_dir>          # writes data/runs/batch/<run_id>/
"""
from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path

from ... import paths
from .dynamic import DynamicGSAS2RietveldEstimator


def as_float(v):
    try:
        return None if v is None else float(v)
    except Exception:
        return None


def bo_training_eligible(sample: str, row: dict) -> bool:
    """Strict BO-training eligibility for Ce-Al-Ge composition BO.

    ``use_for_bo`` says a numeric label is safe at all; training additionally
    requires a nominal ``CeAlGe_*`` composition and the ternary (no Ga) domain.
    """
    return bool(row.get("use_for_bo") is True and sample.startswith("CeAlGe_")
                and "Ga" not in sample and "DSC" not in sample.upper()
                and row.get("status") in {"pass", "pass_caution", "near"}
                and row.get("yield_pct") is not None)


def result_to_row(sample: str, raw_path: Path, res, elapsed_s: float) -> dict:
    meta = res.phase_peaks_used or {}
    residual = meta.get("residual_diagnosis") or {}
    metrics = meta.get("metrics") or {}
    phase_fracs = res.phase_fractions or {}
    target = meta.get("target_phase") or "CeAl2Ge2"
    row = {
        "sample": sample,
        "raw_path": str(raw_path),
        "ok": True,
        "elapsed_s": round(elapsed_s, 2),
        "method": res.method,
        "target_phase": target,
        "yield_pct": as_float(res.yield_pct),
        "yield_sigma_pct": as_float(res.sigma_pct),
        "status": meta.get("status") or res.confidence,
        "confidence": res.confidence,
        "confidence_note": res.confidence_note,
        "use_for_bo": meta.get("use_for_bo"),
        "bo_training_eligible": None,
        "no_fallback": meta.get("no_fallback"),
        "label_type": meta.get("label_type"),
        "target_wt_frac": as_float(phase_fracs.get(target)),
        "rwp": metrics.get("rwp"),
        "gof": metrics.get("gof"),
        "dominant": meta.get("dominant"),
        "phase_fractions": phase_fracs,
        "phase_fractions_sigma": res.phase_fractions_sigma,
        "report_path": meta.get("report_path"),
        "run_dir": meta.get("run_dir"),
        "residual_diagnosis": residual.get("diagnosis"),
        "n_residual_peaks": residual.get("n_residual_peaks"),
        "unknown_residual_peaks_2theta": meta.get("unknown_residual_peaks_2theta"),
        "error": None,
    }
    row["bo_training_eligible"] = bo_training_eligible(sample, row)
    return row


def error_row(sample: str, raw_path: Path, exc: BaseException, elapsed_s: float) -> dict:
    return {
        "sample": sample, "raw_path": str(raw_path), "ok": False,
        "elapsed_s": round(elapsed_s, 2), "method": None, "target_phase": "CeAl2Ge2",
        "yield_pct": None, "yield_sigma_pct": None, "status": "error",
        "confidence": "error", "confidence_note": str(exc)[:1000],
        "use_for_bo": False, "bo_training_eligible": False, "no_fallback": True,
        "label_type": "error_no_numeric_label", "target_wt_frac": None,
        "rwp": None, "gof": None, "dominant": None, "phase_fractions": None,
        "phase_fractions_sigma": None, "report_path": None, "run_dir": None,
        "residual_diagnosis": None, "n_residual_peaks": None,
        "unknown_residual_peaks_2theta": None, "error": repr(exc),
    }


def write_outputs(rows, root_raws, run_id, summary_json: Path, summary_csv: Path) -> dict:
    payload = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n_total": len(root_raws),
        "n_done": len(rows),
        "n_ok": sum(1 for r in rows if r.get("ok")),
        "n_error": sum(1 for r in rows if not r.get("ok")),
        "n_use_for_bo": sum(1 for r in rows if r.get("use_for_bo") is True),
        "n_bo_training_eligible": sum(1 for r in rows
                                      if r.get("bo_training_eligible") is True),
        "rows": rows,
    }
    summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                            encoding="utf-8")

    flat_keys = [
        "sample", "ok", "elapsed_s", "method", "yield_pct", "yield_sigma_pct",
        "status", "use_for_bo", "bo_training_eligible", "no_fallback", "label_type",
        "rwp", "gof", "dominant", "residual_diagnosis", "n_residual_peaks",
        "report_path", "error",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=flat_keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in flat_keys})
    return payload


def main(raw_dir: str | Path | None = None, system: str = "ceal2ge2",
         out_dir: str | Path | None = None, timeout: int = 1200) -> dict:
    raw_root = Path(raw_dir) if raw_dir else paths.ARC_RAW
    root_raws = sorted(raw_root.glob("*.raw"), key=lambda p: p.name)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(out_dir) if out_dir else paths.RUNS / "batch" / run_id
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / "results.jsonl"
    summary_json = out / "summary.json"
    summary_csv = out / "summary.csv"

    print(f"[BATCH] run_id={run_id}\n[BATCH] n_root_raw={len(root_raws)}\n[BATCH] out_dir={out}",
          flush=True)

    rows = []
    est = DynamicGSAS2RietveldEstimator(system, timeout=timeout)
    for i, raw in enumerate(root_raws, start=1):
        sample = raw.stem
        print(f"[BATCH] {i}/{len(root_raws)} START {sample}", flush=True)
        t0 = time.time()
        try:
            res = est.estimate(raw, sample=sample)
            row = result_to_row(sample, raw, res, time.time() - t0)
            print(f"[BATCH] {i}/{len(root_raws)} OK {sample} status={row['status']} "
                  f"yield={row['yield_pct']}±{row['yield_sigma_pct']} rwp={row['rwp']} "
                  f"gof={row['gof']} use_for_bo={row['use_for_bo']}", flush=True)
        except Exception as exc:
            row = error_row(sample, raw, exc, time.time() - t0)
            print(f"[BATCH] {i}/{len(root_raws)} ERROR {sample}: {exc}", flush=True)
        rows.append(row)
        with jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        write_outputs(rows, root_raws, run_id, summary_json, summary_csv)

    payload = write_outputs(rows, root_raws, run_id, summary_json, summary_csv)
    print(f"[BATCH] DONE summary={summary_json}\n[BATCH] DONE csv={summary_csv}", flush=True)
    return payload


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else None)
