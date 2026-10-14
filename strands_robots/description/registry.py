"""The curated robot registry: a name or alias -> a JSON entry (data, read once)."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

_REGISTRY_DIR = Path(__file__).parent.parent / "registry"
_NAME_RE = re.compile(r"^[a-z0-9_.]+$")


def normalize_name(name: str) -> str:
    """lowercase, trimmed, dashes read as underscores — the one rule every lookup uses."""
    if not isinstance(name, str):
        raise TypeError(f"robot name must be a str, got {type(name).__name__}")
    return name.strip().lower().replace("-", "_")


@lru_cache(maxsize=1)
def registry() -> dict[str, dict[str, Any]]:
    """The curated registry: canonical name -> entry (aliases folded in as extra keys)."""
    robots: dict[str, dict[str, Any]] = json.loads((_REGISTRY_DIR / "robots.json").read_text())["robots"]
    out: dict[str, dict[str, Any]] = {}
    for key, entry in robots.items():
        entry = {**entry, "name": key}
        out[key] = entry
        for alias in entry.get("aliases", ()):
            out.setdefault(normalize_name(alias), entry)
    return out


@lru_cache(maxsize=1)
def urdf_registry() -> dict[str, dict[str, Any]]:
    return json.loads((_REGISTRY_DIR / "urdf_robots.json").read_text())["robots"]
