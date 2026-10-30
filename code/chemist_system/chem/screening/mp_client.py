"""Materials Project key resolution for the screening scripts.

The key comes from ``MP_API_KEY``/``MPRESTER_API_KEY`` or a key file (see
``chem.config``).  ``MP_API_KEY`` is a placeholder when unconfigured so the
modules stay importable; call :func:`require` before issuing requests.
"""
from __future__ import annotations

import warnings

from ..config import mp_api_key

_PLACEHOLDER = "SET_MP_API_KEY"


def require() -> str:
    key = mp_api_key()
    if not key:
        raise SystemExit(
            "MP_API_KEY is not configured. Export MP_API_KEY=... or write the key to "
            "<repo>/local/mp_api_key.txt."
        )
    return key


MP_API_KEY = mp_api_key() or _PLACEHOLDER
if MP_API_KEY == _PLACEHOLDER:
    warnings.warn("MP_API_KEY not configured — screening requests will fail until set",
                  RuntimeWarning, stacklevel=2)
