"""Offline energetics: no Materials Project key required.

* `periodic`  – formula parsing and atomic masses.
* `miedema`   – semi-empirical intermetallic formation enthalpy (ranking only).
* `sisso`     – Bartel/SISSO finite-temperature Gibbs correction.
* `mp`        – optional MP DFT cache (needs a key; falls back to Miedema).
* `hull`      – convex-hull decomposition and Al-flux selectivity for Ce-Al-Ge.
"""
from __future__ import annotations

__all__ = ["periodic", "miedema", "sisso", "mp", "hull"]


def __getattr__(name):
    if name in __all__:
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(name)
