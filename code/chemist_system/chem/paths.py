"""Single source of truth for every filesystem location used by the system.

No other module is allowed to hard-code an absolute path.  Everything resolves
from the repository root and can be overridden with an environment variable, so
the system can be moved, vendored, or run from a different data disk.

Layout (defaults)::

    <repo>/data/ce_al_ge/               CeAl2Ge2 arc-melt campaign data (isolated)
        arc_melt/raw/                   raw XRD patterns (.raw/.xy/.txt/.mdi)
        arc_melt/cifs/                  phase CIFs for that campaign
        datasets/                       curated yield datasets (json)
        energetics/                     DFT hull / MP caches
        bench/                          BO benchmark curves
        rietveld/                       reports, runs, GSAS artifacts
    <repo>/data/mp_screens/             Materials Project screening output
    <repo>/data/reference/              XRD reference database
    <repo>/data/runs/                   run artifacts (created on demand)
    <repo>/data/wiki/                   local wiki fallback
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


# ── repository / data roots ────────────────────────────────────────────────
HOME = _env_path("CHEMIST_HOME", ROOT)
DATA = _env_path("CHEMIST_DATA", HOME / "data")

# ── Ce-Al-Ge arc-melt campaign (the one deliberately isolated dataset) ─────
CE_AL_GE = _env_path("CHEMIST_CE_AL_GE", DATA / "ce_al_ge")
ARC_RAW = _env_path("CHEMIST_ARC_RAW", CE_AL_GE / "arc_melt" / "raw")
ARC_CIFS = _env_path("CHEMIST_ARC_CIFS", CE_AL_GE / "arc_melt" / "cifs")
CE_AL_GE_DATASETS = _env_path("CHEMIST_CE_AL_GE_DATASETS", CE_AL_GE / "datasets")
CE_AL_GE_ENERGETICS = _env_path("CHEMIST_CE_AL_GE_ENERGETICS", CE_AL_GE / "energetics")
CE_AL_GE_BENCH = _env_path("CHEMIST_CE_AL_GE_BENCH", CE_AL_GE / "bench")
CE_AL_GE_RIETVELD = _env_path("CHEMIST_CE_AL_GE_RIETVELD", CE_AL_GE / "rietveld")

# ── GSAS-II dynamic Rietveld engine (external, conda `gsas2` env) ──────────
_legacy_engine = Path("/home/ucas/workspace/arc_melt/rietveld_engine")
RIETVELD_ENGINE = _env_path(
    "CHEMIST_RIETVELD_ENGINE",
    _legacy_engine if _legacy_engine.is_dir() else CE_AL_GE_RIETVELD / "engine",
)
RIETVELD_RUNS = _env_path("CHEMIST_RIETVELD_RUNS", RIETVELD_ENGINE / "runs")
RIETVELD_CIFS = _env_path("CHEMIST_RIETVELD_CIFS", RIETVELD_ENGINE / "cifs")

# ── knowledge base ────────────────────────────────────────────────────────
_wiki_default = Path("/home/ucas/wiki") if Path("/home/ucas/wiki").is_dir() else DATA / "wiki"
WIKI = _env_path("CHEMIST_WIKI", _wiki_default)

# ── run artifacts / caches / databases ────────────────────────────────────
RUNS = _env_path("CHEMIST_RUNS", DATA / "runs")
DELIVERABLES = _env_path("CHEMIST_DELIVERABLES", DATA / "deliverables")
CACHE = _env_path("CHEMIST_CACHE", DATA / "cache")
XRD_DB = _env_path("CHEMIST_XRD_DB", DATA / "reference" / "xrd_reference.db")
MP_SCREENS = _env_path("CHEMIST_MP_SCREENS", DATA / "mp_screens")
PHASE_REGISTRY_DB = _env_path("CHEMIST_PHASE_DB", CACHE / "phase_registry.db")

# ── external toolchains ───────────────────────────────────────────────────
GSAS2_HOME = _env_path("CHEMIST_GSAS2_HOME", Path("/home/ucas/GSASIIv2"))
GSAS2_CONDA_ENV = os.environ.get("CHEMIST_GSAS2_ENV", "gsas2")
RXN_CA_SRC = _env_path(
    "CHEMIST_RXN_CA_SRC",
    Path("/home/ucas/workspace/autocell/rxn-ca/src")
    if Path("/home/ucas/workspace/autocell/rxn-ca/src").is_dir()
    else HOME / "vendor" / "rxn-ca" / "src",
)


def _find_conda_python(env_name: str) -> str | None:
    for base in (Path("/home/ucas/miniconda3/envs"), Path.home() / "miniconda3/envs",
                 Path.home() / "anaconda3/envs"):
        candidate = base / env_name / "bin" / "python"
        if candidate.exists():
            return str(candidate)
    return None


# Interpreter used for rxn-ca / pymatgen work in a child process.  Prefer the
# conda env that actually has the stack, else the current interpreter.
RXN_PYTHON = os.environ.get("CHEMIST_RXN_PYTHON") or _find_conda_python("python12") or "python3"

# Interpreter used for GSAS-II work; usually reached through `conda run -n gsas2`.
GSAS2_PYTHON = os.environ.get("CHEMIST_GSAS2_PYTHON") or _find_conda_python(GSAS2_CONDA_ENV)


def ensure(*paths: Path) -> None:
    """Create directories (idempotent)."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def ce_al_ge_dataset(name: str = "arcmelt_dataset_rietveld.json") -> Path:
    """Path to a curated CeAl2Ge2 dataset; falls back to the legacy external copy."""
    local = CE_AL_GE_DATASETS / name
    if local.exists():
        return local
    legacy = Path("/home/ucas/workspace/arc_melt") / name
    return legacy if legacy.exists() else local


def describe() -> dict[str, str]:
    """Resolved paths + which ones are missing, for `chem doctor`."""
    keys = {
        "home": HOME, "data": DATA, "ce_al_ge": CE_AL_GE, "arc_raw": ARC_RAW,
        "arc_cifs": ARC_CIFS, "datasets": CE_AL_GE_DATASETS,
        "energetics": CE_AL_GE_ENERGETICS, "rietveld_engine": RIETVELD_ENGINE,
        "rietveld_runs": RIETVELD_RUNS, "wiki": WIKI, "runs": RUNS,
        "xrd_db": XRD_DB, "gsas2_home": GSAS2_HOME, "rxn_ca_src": RXN_CA_SRC,
    }
    return {k: str(v) for k, v in keys.items()}
