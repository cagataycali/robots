"""LeRobot ``semantics`` labels -> strands units, read from ``semantics.json`` (data, never an ``if``)."""

from __future__ import annotations

import json
from functools import cache
from importlib import resources
from typing import Any

from ..description import Description
from .lerobot import lerobot_units
from .scheme import Units


@cache
def table() -> dict[str, dict[str, Any]]:
    raw = json.loads(resources.files(__package__).joinpath("semantics.json").read_text())
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def units_for_semantics(semantics: str, description: Description) -> Units | None:
    """The converter for a LeRobot semantics label (None = radians), or a refusal naming the file."""
    entry = table().get(semantics)
    entry = table().get(entry["same_as"]) if entry and "same_as" in entry else entry
    if entry is None:
        raise ValueError(
            f"unknown LeRobot semantics {semantics!r}: add it to strands_robots/units/semantics.json "
            f"(known: {', '.join(sorted(table()))}) or pass units= explicitly"
        )
    if entry.get("units") == "rad":
        return None
    return lerobot_units(description, entry["lerobot_norm"])


def semantics_for(description: Description, norm: dict[str, str] | None = None) -> str:
    """The table's label for this robot's norm modes (registry default unless ``norm`` is given)."""
    norm = dict(norm if norm is not None else (description.hardware.lerobot_norm or {}))
    if not norm:
        return "radians-v1"
    hits = [k for k, v in table().items() if v.get("lerobot_norm") == norm and k.startswith(f"{description.name}-")]
    hits = hits or [k for k, v in table().items() if v.get("lerobot_norm") == norm]
    if not hits:
        raise ValueError(f"no semantics label for {description.name} with modes {norm}; add one to semantics.json")
    return sorted(hits)[0]
