"""GSAS-II backends.

* `dynamic`   – dynamic Rietveld agent backend (primary for CeAl2Ge2 QPA);
  delegates to the external production agent when present, else to the packaged
  generic agent (`agent.py`) in the ``gsas2`` conda environment.
* `batch`     – batch recompute of a directory of patterns.
* `refinement`– single-pass refinement package (legacy backend + pipeline).
"""
from __future__ import annotations

__all__ = ["DynamicGSAS2RietveldEstimator"]


def __getattr__(name):
    if name == "DynamicGSAS2RietveldEstimator":
        from .dynamic import DynamicGSAS2RietveldEstimator
        return DynamicGSAS2RietveldEstimator
    raise AttributeError(name)
