"""Fetch the furnace-XRD shortlist structures from Materials Project.

Import-safe: the HTTP calls run in :func:`fetch` / :func:`main`, never at import.
"""
from __future__ import annotations

import json

import requests

from .. import paths
from .mp_client import MP_API_KEY

IDS = [
    "mp-aaahfohf",  # Tb2MgMnO6
    "mp-aaacrnql",  # MgTi2MnO6
    "mp-aaahhckc",  # Nd2MgCoO6
    "mp-aaacrvii",  # CaTbMn2O6
    "mp-aaaacgul",  # CaHoMn2O6
]
FIELDS = ("material_id,formula_pretty,structure,symmetry,energy_above_hull,theoretical,"
          "database_IDs,ordering,total_magnetization_normalized_formula_units,band_gap")


def fetch(ids: list[str] | None = None) -> list[dict]:
    headers = {"X-API-KEY": MP_API_KEY, "accept": "application/json",
               "User-Agent": "mp-api-client/0.45 furnace-shortlist-fetch"}
    records: list[dict] = []
    for mid in ids or IDS:
        response = requests.get(
            "https://api.materialsproject.org/materials/summary/",
            params={"material_ids": mid, "_fields": FIELDS}, headers=headers, timeout=90,
        )
        response.raise_for_status()
        records.extend(response.json().get("data", []))
    return records


def main(argv: list[str] | None = None) -> None:
    records = fetch()
    out = paths.MP_SCREENS / "furnace_xrd" / "shortlist_structures.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=2))
    print(json.dumps([{"material_id": r["material_id"], "formula": r["formula_pretty"],
                       "nsites": len(r["structure"]["sites"])} for r in records], indent=2))


if __name__ == "__main__":
    main()
