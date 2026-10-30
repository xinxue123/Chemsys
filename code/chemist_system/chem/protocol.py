"""Rules-based synthesis protocol generation.

This is the self-contained fallback used when the wiki and the literature search
have nothing for a material, and the schema every consumer in the system expects
(auditor, execution plan, rxn-ca, arc-melt gate).

Two routes are produced:

* ``solid_state``  – oxide/carbonate precursors, muffle furnace, air.
* ``arc_melt``     – elemental precursors, water-cooled hearth, Ti getter, anneal.

The output is deliberately explicit about being a low-confidence rule-based
guess (``confidence_score`` ~0.35); it is a starting point for the auditor, not
a literature-derived recipe.
"""
from __future__ import annotations

from .energetics.periodic import parse_formula

# element -> (precursor formula, cations per precursor formula, purity)
OXIDE_PRECURSOR: dict[str, tuple[str, int, str]] = {
    "Li": ("Li2CO3", 2, "99.99%"), "Na": ("Na2CO3", 2, "99.5%"),
    "Mg": ("MgO", 1, "99.9%"), "Al": ("Al2O3", 2, "99.99%"),
    "Si": ("SiO2", 1, "99.9%"), "Ca": ("CaCO3", 1, "99.5%"),
    "Ti": ("TiO2", 1, "99.9%"), "Mn": ("MnO2", 1, "99.9%"),
    "Fe": ("Fe2O3", 2, "99.9%"), "Co": ("Co3O4", 3, "99.9%"),
    "Ni": ("NiO", 1, "99.9%"), "Cu": ("CuO", 1, "99.9%"),
    "Zn": ("ZnO", 1, "99.9%"), "Sr": ("SrCO3", 1, "99.9%"),
    "Y": ("Y2O3", 2, "99.99%"), "Zr": ("ZrO2", 1, "99.9%"),
    "Nb": ("Nb2O5", 2, "99.99%"), "Mo": ("MoO3", 1, "99.9%"),
    "Ba": ("BaCO3", 1, "99.9%"), "La": ("La2O3", 2, "99.99%"),
    "Ce": ("CeO2", 1, "99.99%"), "Pr": ("Pr6O11", 6, "99.9%"),
    "Nd": ("Nd2O3", 2, "99.9%"), "Gd": ("Gd2O3", 2, "99.99%"),
    "Dy": ("Dy2O3", 2, "99.9%"), "Ho": ("Ho2O3", 2, "99.9%"),
    "Er": ("Er2O3", 2, "99.99%"), "Tm": ("Tm2O3", 2, "99.9%"),
    "Yb": ("Yb2O3", 2, "99.9%"), "Lu": ("Lu2O3", 2, "99.9%"),
    "Ta": ("Ta2O5", 2, "99.99%"), "W": ("WO3", 1, "99.9%"),
}

LANTHANIDES = {"La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
               "Ho", "Er", "Tm", "Yb", "Lu", "Y"}
REFRACTORY = {"Ta", "Nb", "W", "Mo", "Hf", "Zr"}


def _sinter_temperature(comp: dict[str, int]) -> int:
    els = set(comp)
    if els & LANTHANIDES and els & REFRACTORY:
        return 1300
    if els & LANTHANIDES:
        return 1100
    if els & REFRACTORY:
        return 1200
    if {"Li"} & els and {"Fe", "Co", "Ni", "Mn"} & els:
        return 750
    return 1000


def _thermal_profile(t_peak: int, t_intermediate: int = 800) -> list[dict]:
    """Ramp / dwell profile with a decomposition hold and a 6.5 h main dwell."""
    t_int = min(t_intermediate, int(t_peak * 0.6))
    return [
        {"stage": 1, "action": "ramp_up", "target_temperature_c": 100,
         "ramp_rate_c_per_min": 5, "dwell_time_min": 0, "description": "Dry sample"},
        {"stage": 2, "action": "dwell", "target_temperature_c": 100,
         "ramp_rate_c_per_min": 0, "dwell_time_min": 60, "description": "Remove adsorbed water"},
        {"stage": 3, "action": "ramp_up", "target_temperature_c": t_int,
         "ramp_rate_c_per_min": 3, "dwell_time_min": 0, "description": "Intermediate ramp"},
        {"stage": 4, "action": "dwell", "target_temperature_c": t_int,
         "ramp_rate_c_per_min": 0, "dwell_time_min": 90, "description": "Precursor decomposition"},
        {"stage": 5, "action": "ramp_up", "target_temperature_c": t_peak,
         "ramp_rate_c_per_min": 3, "dwell_time_min": 0, "description": "Ramp to sintering"},
        {"stage": 6, "action": "dwell", "target_temperature_c": t_peak,
         "ramp_rate_c_per_min": 0, "dwell_time_min": 390, "description": "Main crystallization (6.5 h)"},
        {"stage": 7, "action": "ramp_down", "target_temperature_c": 25,
         "ramp_rate_c_per_min": -5, "dwell_time_min": 0, "description": "Furnace cool"},
    ]


def _anneal_profile(t_anneal: int, hours: int) -> list[dict]:
    return [
        {"stage": 1, "action": "ramp_up", "target_temperature_c": t_anneal,
         "ramp_rate_c_per_min": 5, "dwell_time_min": 0, "description": "Ramp to homogenization"},
        {"stage": 2, "action": "dwell", "target_temperature_c": t_anneal,
         "ramp_rate_c_per_min": 0, "dwell_time_min": hours * 60,
         "description": f"Homogenization anneal ({hours} h)"},
        {"stage": 3, "action": "ramp_down", "target_temperature_c": 25,
         "ramp_rate_c_per_min": -2, "dwell_time_min": 0, "description": "Slow cool"},
    ]


def infer_route(material: str, comp: dict[str, int] | None = None) -> str:
    """Oxide (contains O) → solid-state; everything else → arc-melt."""
    comp = comp or parse_formula(material) or {}
    return "solid_state" if "O" in comp else "arc_melt"


def oxide_precursors(comp: dict[str, int]) -> list[dict]:
    """Binary oxide/carbonate precursor set with mole ratios for 1 mol product."""
    cations = {el: n for el, n in comp.items() if el != "O"}
    rows = []
    for el, n in cations.items():
        formula, per, purity = OXIDE_PRECURSOR.get(el, (f"{el}2O3", 2, "99.9%"))
        rows.append({
            "chemical": formula,
            "molar_ratio": round(n / per, 4),
            "purity": purity,
            "role": "cation source",
            "cations_per_formula": per,
            "as_oxide_of": el,
        })
    return rows


def elemental_precursors(comp: dict[str, int]) -> list[dict]:
    purity = {"Ce": "99.9%", "Al": "99.999%", "Ge": "99.999%", "Ga": "99.99%"}
    return [
        {"chemical": el, "molar_ratio": float(n), "purity": purity.get(el, "99.9%"),
         "role": "elemental", "form": "shot" if el in {"Ce", "La"} else "pieces"}
        for el, n in comp.items()
    ]


def to_arc_recipe(protocol: dict) -> dict:
    """Convert a protocol dict into the schema `PhaseGraphAuditor.audit_arc_melt` wants."""
    inner = protocol.get("protocol", {})
    precursors = inner.get("precursors", [])
    if not precursors:
        from .energetics.periodic import parse_formula
        precursors = elemental_precursors(parse_formula(protocol.get("material", "")) or {})
    anneal = inner.get("arc_melt", {}).get("anneal", {})
    return {
        "recipe_id": f"{protocol.get('material', 'unknown')}-arc-{protocol.get('generated_at', '')}",
        "target_compound": protocol.get("material"),
        "raw_materials": [
            {"element": p["chemical"], "molar_ratio": p.get("molar_ratio", 1.0),
             "purity_percent": float(str(p.get("purity", "99.9%")).rstrip("%") or 99.9),
             "form": p.get("form", "pieces")}
            for p in precursors
        ],
        "hardware": {
            "water_cooled_copper_hearth": True,
            "vacuum_base_pressure_pa": 0.001,
            "ti_getter_enabled": True,
            "centroid_arc_temp_c": 3000,
        },
        "process": {
            "flipped_melting_cycles": inner.get("arc_melt", {}).get("flip_cycles", 3),
            "initial_current_A": inner.get("arc_melt", {}).get("initial_current_a", 60),
            "atmosphere_pressure_mbar": inner.get("arc_melt", {}).get("pressure_mbar", 500),
        },
        "post_treatment": {
            "requires_annealing": bool(anneal),
            "temperature_c": anneal.get("temperature_c", 900),
            "duration_days": anneal.get("duration_days", 5),
            "protective_foil": anneal.get("protective_foil", "Ta"),
            "quench_method": anneal.get("quench_method", "furnace_cool"),
            "sealed_in_quartz": True,
        },
    }


def generate_protocol(material: str, initial_temp: float | None = None,
                      route: str | None = None) -> dict:
    """Build a complete protocol dict from stoichiometry alone."""
    formula = material.strip()
    comp = parse_formula(formula) or {}
    route = route or infer_route(formula, comp)
    notes = ["Rule-based generation (no wiki/literature hit) — treat as a starting point."]

    if route == "solid_state":
        precursors = oxide_precursors(comp)
        peak = int(initial_temp) if initial_temp else _sinter_temperature(comp)
        thermal = _thermal_profile(peak)
        inner = {
            "route": "solid_state",
            "precursors": precursors,
            "thermal_profile": thermal,
            "atmosphere": "Air",
            "crucible_material": "Alumina",
            "mechanical_processing": {"pressing_pressure_mpa": 200,
                                      "tooling": "uniaxial press"},
        }
        temps = [s["target_temperature_c"] for s in thermal if s["action"] == "dwell"]
        durations = [s["dwell_time_min"] for s in thermal if s["action"] == "dwell"]
    else:
        precursors = elemental_precursors(comp)
        anneal_t = int(initial_temp) if initial_temp else 900
        thermal = _anneal_profile(anneal_t, hours=120)
        inner = {
            "route": "arc_melt",
            "precursors": precursors,
            "thermal_profile": thermal,
            "atmosphere": "Ar",
            "crucible_material": "Water-cooled copper hearth",
            "mechanical_processing": {"pressing_pressure_mpa": 0,
                                      "tooling": "none (arc melted buttons)"},
            "arc_melt": {
                "flip_cycles": 3, "initial_current_a": 60, "pressure_mbar": 600,
                "anneal": {"temperature_c": anneal_t, "duration_days": 5,
                           "protective_foil": "Ta", "quench_method": "furnace_cool"},
            },
        }
        temps = [s["target_temperature_c"] for s in thermal
                 if s["action"] == "dwell"]
        durations = [s["dwell_time_min"] for s in thermal if s["action"] == "dwell"]
        notes.append("Arc-melt route: non-equilibrium; rxn-ca simulation not applicable.")

    precursor_ratios: dict[str, float] = {}
    for p in precursors:
        precursor_ratios[p["chemical"]] = precursor_ratios.get(p["chemical"], 0.0) + p["molar_ratio"]

    return {
        "material": formula,
        "source": "rules:stoichiometry",
        "route": route,
        "confidence_score": 0.35,
        "generated_at": __import__("datetime").datetime.now().strftime("%Y%m%dT%H%M%S"),
        "notes": notes,
        "protocol": inner,
        "precursor_ratios": precursor_ratios,
        "temperatures_c": temps,
        "durations_min": durations,
        "ramp_rate_c_per_min": 5,
        "simulation_size": 15,
        "packing_fraction": 0.55,
    }
