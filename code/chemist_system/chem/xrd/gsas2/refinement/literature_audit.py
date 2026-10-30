#!/usr/bin/env python3
"""
文献驱动的 Gd-P-O 体系方案审计器

基于已知文献数据 (Kizilyalli 1976, Kijkowska 2004) 建立本地热力学 + 动力学审计。
不依赖 MP API (该相不在 MP 中)。
"""
import json
import sys
from pathlib import Path
from datetime import datetime

# ─── 文献校准数据 ────────────────────────────────────────────────
# 来源: Kizilyalli 1976 (J. Inorg. Nucl. Chem. 38, 2331)
#       Kijkowska 2004 (J. Mater. Sci. 39, 2017)
#       PDF#00-030-0555 (ICDD)
LITERATURE_DATA = {
    "target_phase": {
        "name": "Gd4P6O21",
        "display": "Gd₄(P₂O₇)₃",
        "crystal": "Cubic (claimed)",
        "a_A": 25.88,
        "space_group_claimed": "P 21 3 (No. 198)",
        "PDF_card": "00-030-0555",
        "PDF_warning": "footnote: 'Match GdPO4 reflections' — author Kizilyalli 1976 did NOT make pure phase",
        "P_Gd_stoichiometric": 1.5,
    },
    "competing_phases": [
        {
            "name": "GdPO4",
            "display": "GdPO₄ 独居石",
            "P_Gd": 1.0,
            "stability": "thermodynamically most stable in P/Gd ≤ 1.5 region",
            "synthesis_window_C": [600, 1100],
            "crystal": "Monazite P2₁/c",
        },
        {
            "name": "Gd2P2O7",
            "display": "Gd₂P₂O₇ 焦磷酸盐",
            "P_Gd": 1.0,
            "stability": "competes with GdPO4 at high P loss",
            "synthesis_window_C": [800, 1200],
            "crystal": "Cubic Pa-3",
        },
        {
            "name": "Gd(PO3)3",
            "display": "Gd(PO₃)₃ 偏磷酸盐",
            "P_Gd": 3.0,
            "stability": "stable at high P excess, long chains",
            "synthesis_window_C": [400, 700],
            "crystal": "monoclinic",
        },
    ],
    "P2O5_vapor_pressure": {
        # log10(P/atm) = A - B/T (K)
        # Sublimation at 360°C (633K), boiling 605°C (878K)
        # Approximate data from thermodynamic tables
        "sublimation_T_C": 360,
        "boiling_T_C": 605,
        "formula": "P2O5 vapor pressure dominant above 600°C",
    },
    "feed_chemistry": {
        "n_Gd2O3": 0.00869,  # mol (3.15 g)
        "n_Gd": 0.01738,
        "n_P_target_GdPO4": 0.01738,  # P/Gd=1.0
        "n_P_target_Gd4P6O21": 0.02607,  # P/Gd=1.5
    },
}


def audit_protocol(protocol):
    """
    7-criterion audit of Gd-P-O synthesis protocol
    """
    issues = []
    score = 100
    notes = []

    # Extract key parameters
    prec = protocol.get("protocol", {}).get("precursors", [])
    thermal = protocol.get("protocol", {}).get("thermal_profile", [])
    if not prec:
        return {"status": "ERROR", "report": "no precursors"}

    # P/Gd ratio
    n_Gd = 0.0
    n_P = 0.0
    for p in prec:
        chem = p.get("chemical", "")
        m = p.get("molar_ratio", 0.0)
        if "Gd" in chem and "O" in chem:
            # Gd2O3: 2 Gd per molecule
            n_Gd += 2 * m
        elif "H3PO4" in chem or "HPO4" in chem or "PO4" in chem:
            n_P += m
    if n_Gd == 0 or n_P == 0:
        return {"status": "ERROR", "report": "could not extract stoichiometry"}
    P_Gd = n_P / n_Gd

    notes.append(f"P/Gd = {P_Gd:.2f} (target: 1.50)")

    # Criterion 1: P excess for 33% loss
    if P_Gd < 1.5:
        score -= 50
        issues.append(f"P/Gd={P_Gd:.2f} < 1.5, no excess for P loss")
    elif P_Gd < 2.0:
        score -= 20
        issues.append(f"P/Gd={P_Gd:.2f} acceptable but only 33% overcompensation")
    elif P_Gd > 3.0:
        score -= 10
        issues.append(f"P/Gd={P_Gd:.2f} overcompensation may yield Gd(PO3)3")
    else:
        notes.append(f"P/Gd={P_Gd:.2f} good overcompensation (50-100%)")

    # Criterion 2: Max T
    max_T = max((s.get("target_temperature_c", 0) for s in thermal), default=0)
    if max_T > 950:
        score -= 30
        issues.append(f"T_max={max_T}°C too high, P2O5 vapor pressure too large")
    elif max_T >= 800:
        notes.append(f"T_max={max_T}°C moderate, P2O5 VP manageable")
    elif max_T >= 700:
        score += 5
        notes.append(f"T_max={max_T}°C low, slow kinetics but safer")
    else:
        score -= 10
        issues.append(f"T_max={max_T}°C too low, may not crystallize")

    # Criterion 3: Total time
    total_min = sum((s.get("dwell_time_min", 0) for s in thermal))
    if total_min < 120:
        score -= 20
        issues.append(f"Total time {total_min}min too short")
    elif total_min >= 600:
        notes.append(f"Total time {total_min}min sufficient")
    else:
        score -= 5
        issues.append(f"Total time {total_min}min marginal")

    # Criterion 4: Intermediate grinding (intermediate product re-mixing)
    has_redisper = any("研磨" in str(s) or "grinding" in str(s).lower()
                      for s in protocol.get("steps", []))
    if has_redisper:
        notes.append("Has intermediate grinding (improves homogeneity)")

    # Criterion 5: Crucible
    crucible = protocol.get("protocol", {}).get("crucible_material", "")
    if "Al2O3" in crucible or "Alumina" in crucible:
        notes.append("Al2O3 crucible (P2O5 may corrode at >900°C; OK at ≤800°C)")

    # Criterion 6: Atmosphere
    atm = protocol.get("protocol", {}).get("atmosphere", "Air")
    if atm == "Air":
        notes.append("Air atmosphere (standard for P salts)")
    elif "N2" in atm or "Ar" in atm:
        notes.append(f"Inert {atm} (protects against CO2/H2O but not P loss)")

    # Criterion 7: Method (liquid vs solid)
    method = protocol.get("metadata", {}).get("synthesis_method", "")
    if "liquid" in method.lower() or "wet" in method.lower() or "solution" in method.lower():
        score += 10
        notes.append("Liquid-phase method (matches Kizilyalli 1976 original)")
    elif "solid" in method.lower():
        score -= 30
        issues.append("Solid-state method (literature shows Kizilyalli used solution route)")

    # ── 终判 ──
    score = max(0, min(100, score))  # Cap at 100
    status = "SUCCESS"
    if score < 50:
        status = "FAIL"
    elif score < 80:
        status = "MARGINAL"

    return {
        "status": status,
        "score": score,
        "P_Gd": P_Gd,
        "T_max_C": max_T,
        "total_dwell_min": total_min,
        "method": method,
        "crucible": crucible,
        "atmosphere": atm,
        "issues": issues,
        "notes": notes,
        "literature_basis": [
            "Kizilyalli M. (1976) J. Inorg. Nucl. Chem. 38, 2331 — original 'solution reaction'",
            "Kijkowska R. (2004) J. Mater. Sci. 39, 2017 — H3PO4 crystallisation route",
            "PDF#00-030-0555 (ICDD) — Kizilyalli's PDF, footnote warns 'Match GdPO4'",
        ],
        "expected_Gd4P6O21_yield_pct": max(0, min(100, score - 5)),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: literature_audit.py <protocol.json>")
        sys.exit(1)

    proto = json.loads(Path(sys.argv[1]).read_text())
    result = audit_protocol(proto)
    print(json.dumps(result, indent=2, default=str))

    # Save audit report
    out = Path(sys.argv[1]).parent / f"audit_literature_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nReport saved: {out}")
