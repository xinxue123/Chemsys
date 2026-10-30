"""Single-pass GSAS-II refinement (legacy backend ``gsas2_legacy``).

`workflow.RietveldWorkflow` / `GSAS2WorkflowEstimator` run one refinement with a
fixed phase set; `smart_pipeline.SmartGSAS2Pipeline` adds forward BIC phase
selection and staged parameter release. Both require GSAS-II
(``CHEMIST_GSAS2_HOME``).
"""
from __future__ import annotations

import importlib

__all__ = [
    "RietveldWorkflow", "GSAS2WorkflowEstimator", "SmartGSAS2Pipeline",
    "PhaseSpec", "Recipe", "GSAS2Runner", "CifRecord", "PhaseState",
    "StepResult", "RefinementHistory", "RefinementReport",
    "run_gsas2_refinement", "get_gsas2_runner", "GSAS2_AVAILABLE", "diagnose_gsas2",
]

_MODULE_OF = {
    "RietveldWorkflow": "workflow", "GSAS2WorkflowEstimator": "workflow",
    "SmartGSAS2Pipeline": "smart_pipeline", "PhaseSpec": "smart_pipeline",
    "Recipe": "smart_pipeline",
    "CifRecord": "report", "PhaseState": "report", "StepResult": "report",
    "RefinementHistory": "report", "RefinementReport": "report",
    "GSAS2Runner": "gsas2_runner",
    "run_gsas2_refinement": "stable_launcher", "get_gsas2_runner": "stable_launcher",
    "GSAS2_AVAILABLE": "stable_launcher", "diagnose_gsas2": "stable_launcher",
}


def __getattr__(name):
    if name in _MODULE_OF:
        return getattr(importlib.import_module("." + _MODULE_OF[name], __name__), name)
    raise AttributeError(name)
