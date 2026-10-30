#!/usr/bin/env python3
"""Screen Materials Project for theoretical ordered magnetic oxides.

Run via `python -m chem.cli screen magnetocaloric`.
The API key is imported and never printed or written to output.
"""
from __future__ import annotations

import csv
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from .mp_client import MP_API_KEY
from .. import paths


BASE = "https://api.materialsproject.org/materials/summary/"
FIELDS = (
    "material_id,formula_pretty,composition_reduced,elements,nelements,nsites,density,volume,"
    "band_gap,ordering,formation_energy_per_atom,energy_above_hull,is_stable,theoretical,"
    "total_magnetization,total_magnetization_normalized_formula_units,num_magnetic_sites,"
    "num_unique_magnetic_sites,types_of_magnetic_species,symmetry,database_IDs"
)
HEADERS = {
    "X-API-KEY": MP_API_KEY,
    "accept": "application/json",
    "User-Agent": "mp-api-client/0.45 order-magnetocaloric-screen",
}

# Approximate free-ion/local-spin degeneracy. This is only an upper-bound
# screening proxy R sum(n ln g)/M, not a predicted entropy change.
DEGENERACY = {
    "Ti": 2, "V": 3, "Cr": 4, "Mn": 6, "Fe": 6, "Co": 4, "Ni": 3, "Cu": 2,
    "Ru": 4, "Rh": 3, "Pd": 2, "Os": 4, "Ir": 3,
    "Ce": 6, "Pr": 9, "Nd": 10, "Pm": 9, "Sm": 6, "Eu": 8, "Gd": 8,
    "Tb": 13, "Dy": 16, "Ho": 17, "Er": 16, "Tm": 13, "Yb": 8,
}
MASS = {
    "O": 15.999, "Li": 6.94, "Na": 22.990, "K": 39.098, "Rb": 85.468, "Cs": 132.905,
    "Be": 9.012, "Mg": 24.305, "Ca": 40.078, "Sr": 87.62, "Ba": 137.327,
    "Sc": 44.956, "Y": 88.906, "La": 138.905, "Ti": 47.867, "Zr": 91.224,
    "Hf": 178.49, "V": 50.942, "Nb": 92.906, "Ta": 180.948, "Cr": 51.996,
    "Mo": 95.95, "W": 183.84, "Mn": 54.938, "Fe": 55.845, "Co": 58.933,
    "Ni": 58.693, "Cu": 63.546, "Zn": 65.38, "Ga": 69.723, "Ge": 72.630,
    "In": 114.818, "Sn": 118.710, "Pb": 207.2, "Bi": 208.980,
    "Ce": 140.116, "Pr": 140.908, "Nd": 144.242, "Pm": 145.0, "Sm": 150.36,
    "Eu": 151.964, "Gd": 157.25, "Tb": 158.925, "Dy": 162.500, "Ho": 164.930,
    "Er": 167.259, "Tm": 168.934, "Yb": 173.045, "Lu": 174.967,
    "Ru": 101.07, "Rh": 102.906, "Pd": 106.42, "Os": 190.23, "Ir": 192.217,
}
METALS = set(MASS) - {"O"}
FORBIDDEN = {"H", "B", "C", "N", "F", "Si", "P", "S", "Cl", "As", "Se", "Br", "Te", "I",
             "Tc", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf"}
RARE_EARTH = {"Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb"}


def request_page(skip: int, limit: int = 1000):
    params = {
        "elements": "O", "theoretical": "true", "num_magnetic_sites_min": 1,
        "energy_above_hull_max": 0.03, "_limit": limit, "_skip": skip, "_fields": FIELDS,
    }
    for attempt in range(5):
        response = requests.get(BASE, params=params, headers=HEADERS, timeout=90)
        if response.status_code == 200:
            return response.json()
        if response.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** attempt)
            continue
        response.raise_for_status()
    response.raise_for_status()


def formula_mass(composition):
    try:
        return sum(float(amount) * MASS[element] for element, amount in composition.items())
    except KeyError:
        return None


def magnetic_entropy_proxy(composition):
    mass = formula_mass(composition)
    if not mass:
        return 0.0
    entropy_molar = 8.314462618 * sum(float(composition.get(el, 0)) * math.log(g)
                                     for el, g in DEGENERACY.items())
    return 1000.0 * entropy_molar / mass  # J kg^-1 K^-1


def orderability(composition):
    cations = {el: float(v) for el, v in composition.items() if el != "O"}
    values = sorted(cations.values())
    if len(values) < 3:
        return 0.0, "insufficient-cations"
    equal_pair = any(abs(values[i] - values[j]) < 1e-6
                     for i in range(len(values)) for j in range(i + 1, len(values)))
    oxygen = float(composition.get("O", 0))
    total_cat = sum(values)
    ratio = oxygen / total_cat if total_cat else 0
    label = "mixed-cation"
    bonus = 8.0
    # A2BB'O6 / AA'B2O6 includes double-perovskite and ilmenite derivatives.
    normalized = sorted(round(v / min(values), 3) for v in values)
    if len(values) == 3 and normalized == [1.0, 1.0, 2.0] and abs(oxygen / min(values) - 6) < 0.2:
        bonus, label = 28.0, "A2BBpO6-or-AApB2O6"
    elif equal_pair:
        bonus, label = 18.0, "equal-cation-pair"
    if not 1.3 <= ratio <= 3.5:
        bonus -= 5
    return bonus, label


def score_document(doc):
    comp = doc.get("composition_reduced") or {}
    elements = set(comp)
    strict_oxide = ("O" in elements and not (elements & FORBIDDEN) and
                    all(el == "O" or el in METALS for el in elements))
    if not strict_oxide or not 4 <= int(doc.get("nelements", 0)) <= 5:
        return None
    if doc.get("database_IDs") not in (None, {}):
        return None
    if int(doc.get("nsites", 999)) > 100:
        return None
    magnetic = elements & set(DEGENERACY)
    if not magnetic:
        return None
    entropy = magnetic_entropy_proxy(comp)
    order_bonus, family = orderability(comp)
    ehull = float(doc.get("energy_above_hull") or 0)
    mag_fu = abs(float(doc.get("total_magnetization_normalized_formula_units") or 0))
    rare = len(elements & RARE_EARTH)
    practical_penalty = 8.0 if elements & {"Pm"} else 0.0
    score = (40.0 * max(0.0, 1.0 - ehull / 0.03)
             + min(35.0, entropy / 8.0)
             + min(15.0, mag_fu)
             + order_bonus + 5.0 * rare - practical_penalty)
    return {
        "score": round(score, 4), "material_id": doc.get("material_id"),
        "formula": doc.get("formula_pretty"), "family_heuristic": family,
        "energy_above_hull_eV_atom": ehull, "is_stable": doc.get("is_stable"),
        "theoretical": doc.get("theoretical"), "database_IDs": doc.get("database_IDs"),
        "ordering_MP": doc.get("ordering"),
        "total_magnetization_uB_fu": doc.get("total_magnetization_normalized_formula_units"),
        "num_magnetic_sites": doc.get("num_magnetic_sites"),
        "magnetic_species_MP": "+".join(doc.get("types_of_magnetic_species") or []),
        "magnetic_elements_formula": "+".join(sorted(magnetic)),
        "magnetic_entropy_upper_proxy_J_kgK": round(entropy, 3),
        "band_gap_eV": doc.get("band_gap"), "nsites": doc.get("nsites"),
        "spacegroup": (doc.get("symmetry") or {}).get("symbol"),
        "spacegroup_number": (doc.get("symmetry") or {}).get("number"),
        "crystal_system": (doc.get("symmetry") or {}).get("crystal_system"),
        "composition_reduced": json.dumps(comp, sort_keys=True),
    }


def main():
    output = paths.MP_SCREENS / "order_magnetocaloric"
    output.mkdir(exist_ok=True)
    first = request_page(0)
    total = int(first.get("meta", {}).get("total_doc", len(first.get("data", []))))
    pages = {0: first}
    skips = list(range(1000, total, 1000))
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(request_page, skip): skip for skip in skips}
        for future in as_completed(futures):
            pages[futures[future]] = future.result()
    docs = []
    for skip in sorted(pages):
        docs.extend(pages[skip].get("data", []))
    ranked = [item for doc in docs if (item := score_document(doc)) is not None]
    ranked.sort(key=lambda x: (-x["score"], x["energy_above_hull_eV_atom"], x["formula"]))
    (output / "query_manifest.json").write_text(json.dumps({
        "endpoint": BASE, "filters": {"elements": "O", "theoretical": True,
        "num_magnetic_sites_min": 1, "energy_above_hull_max": 0.03},
        "api_meta": first.get("meta"), "documents_downloaded": len(docs),
        "strict_candidates": len(ranked), "note": "API key not stored",
    }, indent=2))
    (output / "ranked_candidates.json").write_text(json.dumps(ranked, indent=2))
    if ranked:
        with (output / "ranked_candidates.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(ranked[0]))
            writer.writeheader(); writer.writerows(ranked)
    print(json.dumps({"downloaded": len(docs), "ranked": len(ranked), "top": ranked[:30]}, indent=2))


if __name__ == "__main__":
    main()
