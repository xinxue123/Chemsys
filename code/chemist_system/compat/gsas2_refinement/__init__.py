"""Compatibility shim for external GSAS-II agent scripts.

The production Ce-Al-Ge agent (``<rietveld_engine>/iter_agent.py``) imports
``gsas2_refinement`` and ``gsas2_refinement.gsas2_runner``. That package now
lives at ``chem.xrd.gsas2.refinement``; putting this directory on ``PYTHONPATH``
makes the old import name resolve to the packaged implementation without
copying code.
"""
import sys

from chem.xrd.gsas2 import refinement as _impl

sys.modules[__name__] = _impl
