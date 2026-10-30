"""Knowledge-base access (wiki).

`wiki.WikiCache` builds an in-memory index over a wiki directory of markdown
entity pages and answers synthesis-protocol lookups in O(1).  The wiki root is
``paths.WIKI`` (override with ``CHEMIST_WIKI``).
"""
from __future__ import annotations

__all__ = ["WikiCache"]


def __getattr__(name):
    if name == "WikiCache":
        from .wiki import WikiCache
        return WikiCache
    raise AttributeError(name)
