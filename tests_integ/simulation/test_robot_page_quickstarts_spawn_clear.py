"""Every robot page's quickstart spawns its robot clear of the ground.

``add_robot`` warns when a model starts inside the plane and names the
``position=`` that spawns it resting on it. The registry carries that position
as ``spawn_position`` for the robots it applies to, and the robot pages put it
in their first fence, so a reader's first copy-paste runs without a warning.
This runs each such page's fence, and the bare ``Robot(name)`` beside it: an
asset re-authored to rest on the plane would make the entry stale, and the
bare spawn going quiet is what says so.

Network + MuJoCo: the assets download on first use.
"""

from __future__ import annotations

import logging
import os
import re
import sys

import pytest

os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")

pytest.importorskip("mujoco")

from tests._docs_hooks import docs_hook  # noqa: E402

_BURIED = "inside the ground"
_SPAWNED = sorted(name for name, spec in docs_hook("robot_pages").registry().items() if spec.get("spawn_position"))


def _burial_warnings(caplog: pytest.LogCaptureFixture, code: str) -> list[str]:
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        exec(compile(code, "<robot page fence>", "exec"), {})  # noqa: S102 - the page's own fence
    return [r.getMessage() for r in caplog.records if _BURIED in r.getMessage()]


@pytest.mark.parametrize("name", _SPAWNED)
def test_the_page_fence_spawns_clear_and_the_bare_spawn_does_not(name: str, caplog) -> None:
    page = docs_hook("robot_pages").robot_page(name)
    fence = re.search(r"```python\n(.*?)```", page, re.S)
    assert fence and "position=" in fence.group(1), page
    assert _burial_warnings(caplog, fence.group(1)) == [], fence.group(1)
    bare = f'from strands_robots import Robot\nRobot("{name}").cleanup()\n'
    assert _burial_warnings(caplog, bare), f"{name} now rests on the plane; drop its spawn_position"
