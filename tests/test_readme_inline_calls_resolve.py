"""Every call the README writes in inline code resolves where a reader looks.

A reader copies `` `create_trainer("lerobot_local")` `` the way they copy
`` `Robot(...)` ``: as ``from strands_robots import <name>``. The root namespace
is the ``Robot`` surface, so a bare name must be in ``strands_robots.__all__``
or be a method of the simulation engine (an agent tool action such as
``attach_bodies``); anything else must be written with its import path,
``strands_robots.<module>.<name>(...)``, and that path must import.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import strands_robots

_README = Path(strands_robots.__file__).resolve().parent.parent / "README.md"
_INLINE_CALL = re.compile(r"`([A-Za-z_][\w.]*)\(")


def _resolves(name: str) -> bool:
    if name.startswith("strands_robots."):
        module, _, attr = name.rpartition(".")
        return hasattr(importlib.import_module(module), attr)
    if "." in name:  # an instance attribute such as robot.mesh.tell
        return True
    from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine

    return name in strands_robots.__all__ or hasattr(MuJoCoSimEngine, name)


def test_every_readme_inline_call_resolves() -> None:
    names = set(_INLINE_CALL.findall(_README.read_text(encoding="utf-8")))
    assert names, "the extractor found no inline calls in README.md"
    assert sorted(n for n in names if not _resolves(n)) == []
