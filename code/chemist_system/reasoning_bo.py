"""Supported M2 entrypoint, with logged nullable-measurement compatibility.

The original reasoning_bo_comparison.py remains a frozen protocol artifact.
This thin entrypoint also executes on Windows worker import, before the
frozen scientific runner starts. It changes no non-null event decision.
"""
from chem.benchmarks import reasoning_bo_compat as compatibility
compatibility.engine.event_reasons=compatibility.event_reasons

if __name__=='__main__':
    from chem.benchmarks.reasoning_bo_study import main
    main()
