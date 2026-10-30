#!/usr/bin/env python3
"""Screen MP for magnetic ordered oxides accessible to furnace + lab PXRD.

Run beside mp_key.py. The key is imported only for the HTTPS request and is
never printed or persisted. The score is a synthesis-design heuristic, not a
prediction of successful synthesis or magnetocaloric performance.
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
    "material_id,formula_pretty,composition_reduced,elements,nelements,nsites,band_gap,"
    "ordering,energy_above_hull,is_stable,theoretical,total_magnetization_normalized_formula_units,"
    "num_magnetic_sites,types_of_magnetic_species,symmetry,database_IDs"
)
HEADERS = {"X-API-KEY": MP_API_KEY, "accept": "application/json",
           "User-Agent": "mp-api-client/0.45 furnace-xrd-order-screen"}

Z = {"Mg": 12, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25,
     "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Sr": 38, "Y": 39,
     "Zr": 40, "Nb": 41, "Mo": 42, "Ru": 44, "Rh": 45, "Pd": 46, "Sn": 50,
     "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Sm": 62, "Eu": 63,
     "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70,
     "Lu": 71, "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77,
     "Pt": 78, "Pb": 82, "Bi": 83, "O": 8}
MASS = {k: float(v) for k, v in {
    "Mg":24.305,"Ca":40.078,"Sc":44.956,"Ti":47.867,"V":50.942,"Cr":51.996,
    "Mn":54.938,"Fe":55.845,"Co":58.933,"Ni":58.693,"Cu":63.546,"Zn":65.38,
    "Sr":87.62,"Y":88.906,"Zr":91.224,"Nb":92.906,"Mo":95.95,"Ru":101.07,
    "Rh":102.906,"Pd":106.42,"Sn":118.71,"Ba":137.327,"La":138.905,"Ce":140.116,
    "Pr":140.908,"Nd":144.242,"Sm":150.36,"Eu":151.964,"Gd":157.25,"Tb":158.925,
    "Dy":162.5,"Ho":164.93,"Er":167.259,"Tm":168.934,"Yb":173.045,"Lu":174.967,
    "Hf":178.49,"Ta":180.948,"W":183.84,"Re":186.207,"Os":190.23,"Ir":192.217,
    "Pt":195.084,"Pb":207.2,"Bi":208.98,"O":15.999}.items()}
DEG = {"Ti":2,"V":3,"Cr":4,"Mn":6,"Fe":6,"Co":4,"Ni":3,"Cu":2,
       "Ru":4,"Rh":3,"Os":4,"Ir":3,"Ce":6,"Pr":9,"Nd":10,"Sm":6,
       "Eu":8,"Gd":8,"Tb":13,"Dy":16,"Ho":17,"Er":16,"Tm":13,"Yb":8}
RE_MCE = {"Gd", "Tb", "Dy", "Ho", "Er"}
AIR_FRIENDLY = {"Mg","Ca","Sr","Ba","Sc","Ti","Zr","Hf","Cr","Mn","Fe",
                "Co","Ni","Zn","Sn","Y","La","Pr","Nd","Sm","Gd","Tb","Dy","Ho","Er"}
PENALTY = {"Li","Na","K","Rb","Cs","Be","Cu","Pb","Bi","V","Mo","W","Ru",
           "Rh","Pd","Re","Os","Ir","Pt","Ce","Eu","Tm","Yb"}


def get_page(skip: int, max_hull: float):
    params = {"elements":"O", "theoretical":"true", "num_magnetic_sites_min":1,
              "energy_above_hull_max":max_hull, "_limit":1000, "_skip":skip,
              "_fields":FIELDS}
    for attempt in range(5):
        response = requests.get(BASE, params=params, headers=HEADERS, timeout=90)
        if response.status_code == 200:
            return response.json()
        if response.status_code in {429, 500, 502, 503, 504}:
            time.sleep(2 ** attempt)
            continue
        response.raise_for_status()
    response.raise_for_status()


def reduced_pattern(comp):
    cats = {k: float(v) for k, v in comp.items() if k != "O"}
    if len(cats) != 3:
        return None
    scale = min(cats.values())
    norm = {k: round(v / scale, 3) for k, v in cats.items()}
    if sorted(norm.values()) != [1.0, 1.0, 2.0]:
        return None
    if abs(float(comp.get("O", 0)) / scale - 6.0) > 0.15:
        return None
    pair = sorted(k for k, v in norm.items() if v == 1.0)
    doubled = next(k for k, v in norm.items() if v == 2.0)
    return pair, doubled


def entropy_proxy(comp):
    mass = sum(float(v) * MASS.get(k, 0) for k, v in comp.items())
    if not mass:
        return 0.0
    s = 8.314462618 * sum(float(comp.get(k, 0)) * math.log(g) for k, g in DEG.items())
    return 1000 * s / mass


def evaluate(doc, max_hull):
    comp = doc.get("composition_reduced") or {}
    if doc.get("database_IDs") not in (None, {}):
        return None
    if set(comp) - set(Z) or int(doc.get("nsites", 999)) > 120:
        return None
    pattern = reduced_pattern(comp)
    if pattern is None:
        return None
    pair, doubled = pattern
    dz = abs(Z[pair[0]] - Z[pair[1]])
    elements = set(comp)
    if not elements & set(DEG):
        return None
    ehull = float(doc.get("energy_above_hull") or 0)
    entropy = entropy_proxy(comp)
    air_score = len((elements - {"O"}) & AIR_FRIENDLY) / 3
    penalty = 12 * len(elements & PENALTY)
    score = (35 * max(0, 1 - ehull / max_hull) + min(30, 1.5 * dz)
             + min(22, entropy / 6) + 12 * air_score
             + 8 * bool(elements & RE_MCE) - penalty)
    return {"score":round(score,3), "material_id":doc["material_id"],
            "formula":doc["formula_pretty"], "energy_above_hull_eV_atom":ehull,
            "is_stable":doc.get("is_stable"), "ordered_pair":"/".join(pair),
            "doubled_sublattice":doubled, "xray_delta_Z":dz,
            "air_furnace_elements_only":not bool((elements-{"O"})-AIR_FRIENDLY),
            "MCE_rare_earth": "+".join(sorted(elements & RE_MCE)),
            "entropy_upper_proxy_J_kgK":round(entropy,3),
            "MP_ordering":doc.get("ordering"),
            "MP_magnetization_uB_fu":doc.get("total_magnetization_normalized_formula_units"),
            "band_gap_eV":doc.get("band_gap"),
            "spacegroup":(doc.get("symmetry") or {}).get("symbol"),
            "spacegroup_number":(doc.get("symmetry") or {}).get("number"),
            "nsites":doc.get("nsites")}


def main():
    max_hull = 0.08
    out = paths.MP_SCREENS / "furnace_xrd"
    out.mkdir(exist_ok=True)
    first = get_page(0, max_hull)
    total = int(first.get("meta", {}).get("total_doc", 0))
    pages = {0:first}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(get_page, skip, max_hull):skip for skip in range(1000,total,1000)}
        for future in as_completed(futures):
            pages[futures[future]] = future.result()
    docs = [d for skip in sorted(pages) for d in pages[skip].get("data", [])]
    ranked = [r for d in docs if (r := evaluate(d, max_hull))]
    ranked.sort(key=lambda r:(-r["score"], r["energy_above_hull_eV_atom"], r["formula"]))
    (out/"query_manifest.json").write_text(json.dumps({"max_hull_eV_atom":max_hull,
        "documents_downloaded":len(docs),"candidates":len(ranked),"note":"API key not stored"},indent=2))
    (out/"ranked_candidates.json").write_text(json.dumps(ranked,indent=2))
    with (out/"ranked_candidates.csv").open("w",newline="") as fh:
        writer=csv.DictWriter(fh,fieldnames=list(ranked[0])); writer.writeheader(); writer.writerows(ranked)
    print(json.dumps({"downloaded":len(docs),"ranked":len(ranked),"top":ranked[:40]},indent=2))


if __name__ == "__main__":
    main()
