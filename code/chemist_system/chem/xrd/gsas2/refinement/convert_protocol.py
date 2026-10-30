#!/usr/bin/env python3
"""Convert a liquid-phase protocol JSON into the audit-compatible schema.

V2: extract temperature and dwell time from the free-text ``detail`` field.
Import-safe: all work happens in :func:`convert` / :func:`main`.
"""
from __future__ import annotations

import json
import re
from pathlib import Path


def convert(src_path: str | Path, out_path: str | Path | None = None) -> dict:
    src_file = Path(src_path)
    src = json.loads(src_file.read_text())

    thermal_profile = []
    for s in src.get("steps", []):
        detail = " ".join([s.get("detail", ""), s.get("note", ""), s.get("warning", "")])
        t_match = re.search(r"(\d{2,4})\s*°?\s*C", detail)
        h_match = re.search(r"(\d+(?:\.\d+)?)\s*h\b", detail, re.IGNORECASE)
        min_match = re.search(r"(\d+)\s*min\b", detail, re.IGNORECASE)
        ramp_match = re.search(r"(\d+(?:\.\d+)?)\s*°?\s*C\s*/\s*min", detail, re.IGNORECASE)
        if not t_match:
            continue
        temperature = int(t_match.group(1))
        if temperature >= 1000:      # filter unrealistic readings
            continue
        dwell_min = 0.0
        if h_match:
            dwell_min = float(h_match.group(1)) * 60
        elif min_match:
            dwell_min = float(min_match.group(1))
        thermal_profile.append({
            "stage": s.get("step", 0),
            "action": s.get("action", ""),
            "target_temperature_c": temperature,
            "dwell_time_min": dwell_min,
            "ramp_rate_c_per_min": float(ramp_match.group(1)) if ramp_match else 2.0,
        })

    precursors = [
        {"chemical": r.get("formula", ""), "molar_ratio": r.get("mol", 0),
         "purity": "99.99%"}
        for r in src.get("reagents", [])
    ]

    out = {
        "material": src.get("formula", "Gd4P6O21"),
        "protocol": {
            "precursors": precursors,
            "thermal_profile": thermal_profile,
            "atmosphere": src.get("atmosphere", "Air"),
            "crucible_material": src.get("crucible", "Al2O3"),
        },
        "steps": src.get("steps", []),
        "metadata": src.get("metadata", {}),
        "source_protocol": str(src_file),
    }

    target = Path(out_path) if out_path else src_file.parent / f"{src_file.stem}_audit.json"
    target.write_text(json.dumps(out, indent=2, default=str))
    return {"out_path": str(target), "n_precursors": len(precursors),
            "n_stages": len(thermal_profile), "protocol": out}


def main(argv: list[str] | None = None) -> None:
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        raise SystemExit("usage: python -m chem.xrd.gsas2.refinement.convert_protocol "
                         "<protocol.json> [out.json]")
    result = convert(argv[0], argv[1] if len(argv) > 1 else None)
    print(f"Converted: {result['out_path']}")
    print(f"Precursors: {result['n_precursors']}, Stages: {result['n_stages']}")


if __name__ == "__main__":
    main()
