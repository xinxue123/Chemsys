"""Autonomous materials chemist — standalone package.

Public entry points (import lazily so a missing optional dependency in one
subsystem does not break the rest)::

    from chem import full_loop                  # chem.loop.full_loop
    from chem.xrd import create_estimator       # QPA factory
    from chem.audit import PhaseGraphAuditor    # protocol / arc-melt gate
    from chem.bo import LLMDrivenBO             # Bayesian optimization
    from chem.protocol import generate_protocol # rules-based protocol

Command line: ``python -m chem.cli --help``.
"""
from __future__ import annotations

import importlib

__version__ = "2.0.0"

__all__ = [
    "full_loop", "generate_protocol", "constrained_yield",
    "create_estimator", "YieldResult", "PhaseGraphAuditor", "LLMDrivenBO",
    "paths",
]


def _lazy(module: str, attr: str):
    mod = importlib.import_module(module, __name__)
    return getattr(mod, attr)


def __getattr__(name):
    if name == "paths":
        mod = importlib.import_module(".paths", __name__)
        globals()["paths"] = mod
        return mod
    if name == "generate_protocol":
        return _lazy(".protocol", "generate_protocol")
    if name in {"full_loop", "constrained_yield"}:
        return _lazy(".loop", name)
    if name in {"create_estimator", "YieldResult"}:
        return _lazy(".xrd", name)
    if name == "PhaseGraphAuditor":
        return _lazy(".audit", "PhaseGraphAuditor")
    if name == "LLMDrivenBO":
        return _lazy(".bo", "LLMDrivenBO")
    raise AttributeError(f"module 'chem' has no attribute {name!r}")
