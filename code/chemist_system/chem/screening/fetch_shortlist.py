"""Fetch the magnetocaloric shortlist structures from Materials Project.

Import-safe: the HTTP calls run in :func:`fetch` / :func:`main`, never at import.
"""
from __future__ import annotations

import json

import requests

from .. import paths
from .mp_client import MP_API_KEY

IDS = [
    "mp-aaacrrwz",  # Gd2MnCoO6 known magnetocaloric control
    "mp-aaacrsmh",  # Gd2VFeO6 prospective
    "mp-aaacrskj",  # DyYMn2O6 ordered polymorph candidate
    "mp-aaahgkls",  # Eu2ZrMnO6 prospective
    "mp-aaahdpqf",  # Eu2MnSnO6 prospective
    "mp-aaacsdae",  # Dy2CrFeO6 prospective
]
FIELDS = ("material_id,formula_pretty,structure,symmetry,energy_above_hull,"
          "theoretical,database_IDs,ordering,total_magnetization_normalized_formula_units")


def fetch(ids: list[str] | None = None) -> list[dict]:
    headers = {"X-API-KEY": MP_API_KEY, "accept": "application/json",
               "User-Agent": "mp-api-client/0.45 shortlist-structure-fetch"}
    records: list[dict] = []
    for mid in ids or IDS:
        r = requests.get("https://api.materialsproject.org/materials/summary/",
                         params={"material_ids": mid, "_fields": FIELDS},
                         headers=headers, timeout=90)
        r.raise_for_status()
        records.extend(r.json().get("data", []))
    return records


def main(argv: list[str] | None = None) -> None:
    records = fetch()
    out = paths.MP_SCREENS / "order_magnetocaloric" / "shortlist_structures.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=2))
    print(json.dumps([{"material_id": x["material_id"], "formula": x["formula_pretty"],
                       "nsites": len(x["structure"]["sites"])} for x in records], indent=2))


if __name__ == "__main__":
    main()
