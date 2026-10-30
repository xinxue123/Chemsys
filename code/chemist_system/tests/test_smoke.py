"""Smoke tests for the standalone chemist system.

Run with the conda env that has numpy/scipy/pymatgen::

    /home/ucas/miniconda3/envs/python12/bin/python -m pytest -q

Tests never need a Materials Project key and never launch GSAS-II.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from chem import paths
from chem.protocol import generate_protocol, to_arc_recipe
from chem.energetics import hull, miedema
from chem.xrd import SYSTEM_CONFIGS, create_estimator
from chem.bo.llm import LLMDrivenBO


# ── paths / layout ─────────────────────────────────────────────────────────
def test_paths_are_inside_repo():
    desc = paths.describe()
    assert Path(desc["home"]).is_dir()
    assert Path(desc["ce_al_ge"]).is_dir()
    assert Path(desc["arc_raw"]).is_dir()


def test_ce_al_ge_data_is_isolated():
    assert paths.ARC_RAW.parent.parent.name == "ce_al_ge"
    raws = list(paths.ARC_RAW.glob("*.raw"))
    assert raws, "arc-melt raw patterns missing"
    assert list(paths.CE_AL_GE_DATASETS.glob("arcmelt_dataset*.json"))


def test_no_legacy_import_paths():
    bad = ("/home/ucas/.hermes/plugins_scripts",
           "from xrd_yield_estimator", "import xrd_yield_estimator",
           "from knowledge_guided_bo", "import knowledge_guided_bo",
           "import mp_key")
    offenders = []
    for py in (paths.HOME / "chem").rglob("*.py"):
        text = py.read_text(encoding="utf-8", errors="ignore")
        for token in bad:
            if token in text:
                offenders.append(f"{py.relative_to(paths.HOME)}: {token}")
    assert not offenders, offenders


# ── protocol ───────────────────────────────────────────────────────────────
def test_protocol_routes_by_chemistry():
    assert generate_protocol("CeAl2Ge2")["route"] == "arc_melt"
    assert generate_protocol("ErTa7O19")["route"] == "solid_state"


def test_arc_recipe_schema():
    protocol = generate_protocol("CeAl2Ge2")
    recipe = to_arc_recipe(protocol)
    assert recipe["target_compound"] == "CeAl2Ge2"
    assert {m["element"] for m in recipe["raw_materials"]} == {"Ce", "Al", "Ge"}
    assert recipe["hardware"]["water_cooled_copper_hearth"] is True
    assert recipe["post_treatment"]["sealed_in_quartz"] is True


def test_literature_parser_tolerates_null_abstract():
    """Semantic Scholar returns abstract=None for some records; must not crash."""
    from chem.loop import extract_synthesis_params_from_abstract
    assert extract_synthesis_params_from_abstract(None)["temperature_c"] is None
    assert extract_synthesis_params_from_abstract(
        "annealed at 900 C in Ar")["temperature_c"] == 900.0


# ── audit ──────────────────────────────────────────────────────────────────
def test_arc_melt_audit_without_mp_key():
    from chem.audit.gate import PhaseGraphAuditor

    recipe = to_arc_recipe(generate_protocol("CeAl2Ge2"))
    result = PhaseGraphAuditor(api_key=None).audit_arc_melt(recipe)
    assert result["status"] in {"APPROVED", "WARNING", "REJECTED", "HOLD"}
    assert result["checks"], "arc-melt audit produced no checks"
    if result["status"] in {"APPROVED", "WARNING"}:
        assert result["security_token"]


# ── xrd ────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("backend", ["pawlik", "auto"])
def test_estimator_factory(backend):
    est = create_estimator("ceal2ge2", backend=backend)
    assert est.system == "ceal2ge2"
    assert SYSTEM_CONFIGS["ceal2ge2"]["target_phase"] == "CeAl2Ge2"


def test_pawlik_quantifies_real_pattern():
    raw = sorted(paths.ARC_RAW.glob("CeAlGe_*.xy"))
    if not raw:
        pytest.skip("no CeAlGe pattern available")
    est = create_estimator("ceal2ge2", backend="pawlik")
    res = est.estimate(str(raw[0]))
    assert 0.0 <= res.yield_pct <= 100.0
    assert res.phase_fractions, "no phase fractions returned"
    assert 0.0 <= sum(res.phase_fractions.values()) <= 1.5


def test_auto_fallback_returns_numeric_label():
    """Auto estimator must degrade to a real Pawlik number, not 0.0/None."""
    from chem.xrd import AutoXRDYieldEstimator

    class BrokenPrimary:
        def estimate(self, xrd_data, **kwargs):
            raise RuntimeError("gsas2 unavailable")

    raw = sorted(paths.ARC_RAW.glob("CeAlGe_*.xy"))
    if not raw:
        pytest.skip("no CeAlGe pattern available")
    fallback = create_estimator("ceal2ge2", backend="pawlik")
    auto = AutoXRDYieldEstimator("ceal2ge2", primary=BrokenPrimary(), fallback=fallback)
    res = auto.estimate(str(raw[0]))
    assert res.yield_pct > 0.0
    assert "fallback" in res.method


# ── bo ─────────────────────────────────────────────────────────────────────
def test_bo_suggestion_respects_bounds():
    bo = LLMDrivenBO(system_description="CeAl2Ge2 Ce-Al-Ge intermetallic arc-melt")
    for alce, y in ((6.0, 64.5), (4.0, 10.0), (8.0, 40.0)):
        bo.add_observation(y, AlCe_ratio=alce)
    s = bo.suggest(method="EI")
    assert 1.5 <= s.params["AlCe_ratio"] <= 20.0


# ── energetics ─────────────────────────────────────────────────────────────
def test_hull_and_miedema():
    frac = hull.assemblage(0.25, 0.40)
    assert isinstance(frac, dict) and frac
    assert 0.0 <= hull.selectivity(0.25, 0.40) <= 1.0
    assert miedema.formation_enthalpy("CeAl2Ge2") < 0


# ── cli ────────────────────────────────────────────────────────────────────
def test_cli_doctor_exit_code(capsys):
    from chem.cli import main
    rc = main(["doctor"])
    out = capsys.readouterr().out
    assert "resolved paths" in out
    assert rc in (0, 1)


def test_dynamic_backend_picks_an_agent():
    from chem.xrd.gsas2.dynamic import DynamicGSAS2RietveldEstimator
    est = DynamicGSAS2RietveldEstimator("ceal2ge2")
    cmd, report = est._agent_command("demo", "demo.raw")
    assert cmd[:2] == ["conda", "run"]
    assert any("iter_agent.py" in c or "chem.xrd.gsas2.agent" in c for c in cmd)
    assert report.name == "report.json"
