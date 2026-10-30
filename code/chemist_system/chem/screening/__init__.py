"""Materials Project screening scripts (magnetocaloric and furnace-XRD shortlists).

Each module is a standalone CLI that queries the MP summary endpoint and writes
CSV/JSON into ``paths.MP_SCREENS``. Requires ``MP_API_KEY``.
"""
from __future__ import annotations

__all__ = ["mp_client", "magnetocaloric", "furnace_xrd",
           "fetch_shortlist", "fetch_furnace_shortlist"]
