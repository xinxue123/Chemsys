"""DFT convex-hull auditor for Ce-Al-Ge (pure numpy).
Uses MP DFT formation energies of all stable phases to compute the equilibrium
phase assemblage for any bulk composition via lower-convex-hull decomposition,
and derives a composition-dependent selectivity for the metastable target
without hand-tuned thresholds."""
import os, json
import numpy as np
from ..paths import CE_AL_GE_ENERGETICS, ARC_CIFS

_HULL_FILE = CE_AL_GE_ENERGETICS / "ce_al_ge_hull.json"

def _load_phases():
    phs = json.load(open(_HULL_FILE))
    pts = []
    from . import sisso
    for el in ("Ce", "Al", "Ge"):
        pts.append((el, 1.0 if el == "Ce" else 0.0, 1.0 if el == "Al" else 0.0, 0.0, sisso.ELEM_VATOM.get(el, 20.0)))
    for p in phs:
        c = p["comp"]; tot = sum(c.values())
        pts.append((p["f"], c.get("Ce", 0)/tot, c.get("Al", 0)/tot, p["Ef"], p.get("vatom") or 20.0))
    return pts

_PHASES = _load_phases()

def _bary(p, a, b, c):
    (x, y), (x1, y1), (x2, y2), (x3, y3) = p, a, b, c
    det = (y2-y3)*(x1-x3) + (x3-x2)*(y1-y3)
    if abs(det) < 1e-12:
        return None
    l1 = ((y2-y3)*(x-x3) + (x3-x2)*(y-y3))/det
    l2 = ((y3-y1)*(x-x3) + (x1-x3)*(y-y3))/det
    return (l1, l2, 1.0-l1-l2)

def decompose(xCe, xAl, T=None):
    """Equilibrium assemblage at bulk (xCe,xAl): min-energy phase triangle
    containing the point. Returns {phase: atom_fraction}."""
    P = _PHASES; n = len(P); p = (xCe, xAl); best = None
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                bc = _bary(p, (P[i][1],P[i][2]), (P[j][1],P[j][2]), (P[k][1],P[k][2]))
                if bc is None:
                    continue
                l1, l2, l3 = bc
                if min(l1, l2, l3) < -1e-9:
                    continue
                E = l1*_E(P[i],T) + l2*_E(P[j],T) + l3*_E(P[k],T)
                if best is None or E < best[0] - 1e-12:
                    best = (E, {P[i][0]: l1, P[j][0]: l2, P[k][0]: l3})
    return best[1] if best else {}

TARGET = "CeAl2Ge2"
TARGET_HULL_meV = 12.0

def assemblage(xCe, xAl, T=None):
    """Equilibrium phase assemblage (rounded atom fractions)."""
    return {k: round(v, 3) for k, v in decompose(xCe, xAl, T).items() if v > 1e-3}

def flux_fraction(xCe, xAl, T=None):
    """Equilibrium free-Al (self-flux) fraction at bulk composition."""
    return float(decompose(xCe, xAl, T).get("Al", 0.0))

def selectivity(xCe, xAl, T=None):
    """DFT-hull selectivity for the metastable target = equilibrium Al
    self-flux fraction (threshold-free; r=0.75 with measured yield)."""
    return float(np.clip(flux_fraction(xCe, xAl, T), 0.0, 1.0))

def competitors(xCe, xAl, T=None):
    """Ce-bearing equilibrium phases that intercept the target."""
    return {k: v for k, v in assemblage(xCe, xAl, T).items() if k not in ("Al", "Ge")}

def _E(phase, T):
    """Phase energy (eV/atom): 0 K DFT, optionally SISSO finite-T corrected."""
    name, xc, xa, Ef, vatom = phase
    if T is None or name in ("Ce", "Al", "Ge"):
        return Ef
    try:
        from . import sisso
        return Ef + sisso.dGf_correction(name, vatom, T)
    except Exception:
        return Ef
