"""Bayesian optimization engines.

* `llm`       – production engine: ConservativePrior → LLMGuided → GPR by n.
* `gpr`       – standalone sklearn GPR campaign driver (XRD-dir batch mode).
* `rietveld`  – sequential pool-based BO benchmark on the Ce-Al-Ge Rietveld
  yields, with a DFT-hull prior mean.
"""
from __future__ import annotations

import importlib

__all__ = ["LLMDrivenBO", "heuristic_reasoner", "Suggestion", "Observation"]


def __getattr__(name):
    if name in __all__:
        return getattr(importlib.import_module(".llm", __name__), name)
    raise AttributeError(name)
