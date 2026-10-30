"""Runtime configuration: API keys and optional external hooks.

Keys are resolved lazily and never at import time, so importing any module in
this package works on a machine with no credentials configured.
"""
from __future__ import annotations

import os
from pathlib import Path

from . import paths

_API_KEY_FILES = (
    paths.HOME / "local" / "mp_api_key.txt",
    Path.home() / ".config" / "chemist" / "mp_api_key.txt",
    Path("/home/ucas/.hermes/plugins/solid_state_audit/api_config.py"),
)

_cached_key: str | None = None


def _read_key_file(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("MP_API_KEY"):
            _, _, value = line.partition("=")
            value = value.strip().strip("'\"")
            if value:
                return value
        elif len(line) >= 20 and not line.startswith("#") and " " not in line:
            return line
    return None


def mp_api_key(required: bool = False) -> str | None:
    """Materials Project key: env → key file → None."""
    global _cached_key
    if _cached_key:
        return _cached_key
    for var in ("MP_API_KEY", "MPRESTER_API_KEY", "PMG_MAPI_KEY"):
        value = os.environ.get(var)
        if value:
            _cached_key = value
            return value
    for path in _API_KEY_FILES:
        value = _read_key_file(path)
        if value:
            _cached_key = value
            return value
    if required:
        raise RuntimeError(
            "Materials Project API key not found. Set MP_API_KEY, or write the key to "
            f"{_API_KEY_FILES[0]}."
        )
    return None


def has_mp_key() -> bool:
    return bool(mp_api_key())


def gsas2_available() -> bool:
    return (paths.GSAS2_HOME / "GSASII").is_dir()


def rxn_stack_available() -> bool:
    try:
        import rxn_ca  # noqa: F401
        import rxn_network  # noqa: F401
        return True
    except Exception:
        return False


def pymatgen_available() -> bool:
    try:
        import pymatgen  # noqa: F401
        return True
    except Exception:
        return False
