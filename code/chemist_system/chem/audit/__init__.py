"""Protocol audit gates.

`gate.PhaseGraphAuditor` provides two entry points:

* ``audit_protocol(protocol)``  – oxide / solid-state route: finite-T DFT hull,
  Tammann gate, competing-phase and driving-force checks.
* ``audit_arc_melt(recipe)``    – intermetallic / arc-melt route: 10 hardware,
  element and process interlocks; issues a ``security_token`` on approval.

`experimental/` holds the kinetic (torch/CANTERA) pathway that is not wired into
the live loop — kept for reference, import it explicitly.
"""
from __future__ import annotations

import importlib

__all__ = ["PhaseGraphAuditor", "load_api_key"]


def __getattr__(name):
    if name in __all__:
        return getattr(importlib.import_module(".gate", __name__), name)
    raise AttributeError(name)
