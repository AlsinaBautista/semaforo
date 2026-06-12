# Semáforo Inteligente - Brain Module
"""SUMO_HOME bootstrap, imported before any ``sumo_rl`` import.

sumo_rl raises at import time unless ``SUMO_HOME`` is set.  When SUMO is
installed only as the ``sumo`` pip wheel (as in this project's ``.venv``), its
data directory lives next to the package.  Importing this module first points
``SUMO_HOME`` there without disturbing an existing system install.
"""

from __future__ import annotations

import os
from pathlib import Path


def ensure_sumo_home() -> str | None:
    """Set ``SUMO_HOME`` from the ``sumo`` wheel if it is not already set.

    Returns:
        The resolved ``SUMO_HOME`` path, or ``None`` if it could not be found.
    """
    existing = os.environ.get("SUMO_HOME")
    if existing:
        return existing
    try:
        import sumo  # type: ignore

        sumo_root = Path(sumo.__file__).resolve().parent
        if (sumo_root / "tools").is_dir():
            os.environ["SUMO_HOME"] = str(sumo_root)
            return str(sumo_root)
    except Exception:  # noqa: BLE001 – sumo_rl will raise a clear error instead
        pass
    return None


ensure_sumo_home()
