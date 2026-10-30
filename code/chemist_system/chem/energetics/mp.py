"""Materials Project DFT energetics for the auditor (new MP API, urllib only).

Reads the API key from the solid_state_audit plugin config; results are cached
to a local JSON so the auditor is reproducible offline after first fetch. The
key is never logged. If the key/network is unavailable the auditor falls back
to the self-contained Miedema model.
"""
import os, json, urllib.request, urllib.parse

from ..paths import CE_AL_GE_ENERGETICS
from ..config import mp_api_key

CACHE = str(CE_AL_GE_ENERGETICS / "mp_energetics_cache.json")

def _key():
    key = mp_api_key()
    if not key:
        raise RuntimeError("MP_API_KEY not configured")
    return key

def _load():
    try:
        return json.load(open(CACHE))
    except Exception:
        return {}

def _save(c):
    json.dump(c, open(CACHE, "w"), indent=2)

def _fetch(formula):
    base = "https://api.materialsproject.org/materials/summary/"
    qs = urllib.parse.urlencode({"formula": formula,
        "_fields": "material_id,formation_energy_per_atom,energy_above_hull"})
    req = urllib.request.Request(base + "?" + qs, headers={
        "X-API-KEY": _key(), "User-Agent": "ssa/1.0", "accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        rows = json.load(r).get("data", [])
    rows.sort(key=lambda x: x.get("energy_above_hull", 9.0))
    return rows[0] if rows else None

def energetics(formula, use_cache=True):
    """Return {material_id, Ef_eV_atom, Ehull_eV_atom, source} or None."""
    c = _load()
    if use_cache and formula in c:
        return c[formula]
    try:
        b = _fetch(formula)
        if b:
            rec = {"material_id": b["material_id"],
                   "Ef_eV_atom": round(b["formation_energy_per_atom"], 4),
                   "Ehull_eV_atom": round(b["energy_above_hull"], 4),
                   "source": "MP"}
            c[formula] = rec; _save(c); return rec
    except Exception:
        pass
    return c.get(formula)

def formation_kjmol(formula):
    e = energetics(formula)
    return e["Ef_eV_atom"] * 96.485 if e else None
