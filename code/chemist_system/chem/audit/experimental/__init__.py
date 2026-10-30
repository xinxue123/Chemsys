"""Experimental kinetic pathway — NOT used by the live closed loop.

Contains a torch/CUDA + CANTERA kinetic model (`engine.py`), its thermodynamics
helpers (`physics.py`), a Materials Project fetcher (`data.py`), a CLI driver
(`simulate.py`) and a diffusion/path-cost pair (`ion_transport.py`,
`path_cost.py`). Requires ``torch`` (CUDA), ``cantera`` and ``pymatgen``.

Nothing in ``chem.loop`` imports this package; treat it as a research branch.
"""
