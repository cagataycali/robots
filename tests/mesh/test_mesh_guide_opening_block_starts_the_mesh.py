"""The mesh guide and the README Mesh row join a mesh under the environment they name.

``Mesh.start`` refuses under the default posture - mTLS auth, the built-in
permissive ACL, no acknowledgement - and logs ``PERMISSIVE_ACL_REFUSAL``, which
names three environment variables as the ways out. ``docs/learn/mesh/index.md``
opens with two robots calling ``Robot(..., mesh=True)``; on a fresh install that
block must not end in ``Mesh did NOT start``. This test replays every ``export``
the page issues before its first ``mesh=True`` fence, plus every
``os.environ.setdefault`` / ``os.environ[...] =`` that fence performs before its
first ``Robot(`` call, into a clean environment and asks the gate ``Mesh.start``
asks.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import strands_robots
from strands_robots.mesh import core

_ROOT = Path(strands_robots.__file__).resolve().parent.parent
_GUIDE = _ROOT / "docs" / "learn" / "mesh" / "index.md"
_README = _ROOT / "README.md"
_FENCE = re.compile(r"```(\w*)[^\n]*\n(.*?)```", re.S)
_EXPORT = re.compile(r"^\s*export\s+([A-Z_][A-Z0-9_]*)=(\S+)", re.M)
#: The in-fence spellings of "set this before the first Robot(mesh=True)".
_SETDEFAULT = re.compile(r"""os\.environ\.setdefault\(\s*["']([A-Z_][A-Z0-9_]*)["']\s*,\s*["']([^"']*)["']\s*\)""")
_ASSIGN = re.compile(r"""os\.environ\[\s*["']([A-Z_][A-Z0-9_]*)["']\s*\]\s*=\s*["']([^"']*)["']""")
#: An inline-code ``NAME=value`` in prose, as the README feature table writes it.
_INLINE = re.compile(r"`([A-Z_][A-Z0-9_]*)=([^`\s]+)`")


def _exports_before_the_first_mesh_true_fence() -> dict[str, str]:
    env: dict[str, str] = {}
    for lang, body in _FENCE.findall(_GUIDE.read_text(encoding="utf-8")):
        if "mesh=True" in body:
            before_first_robot = body.split("Robot(", 1)[0]
            for pattern in (_SETDEFAULT, _ASSIGN):
                env.update({name: value for name, value in pattern.findall(before_first_robot)})
            return env
        if lang in ("bash", "sh", "shell", ""):
            env.update({name: value.strip("'\"") for name, value in _EXPORT.findall(body)})
    pytest.fail(f"{_GUIDE.name} has no fence calling Robot(..., mesh=True)")


def test_the_guide_sets_something_before_it_joins() -> None:
    """The replay is not empty: the page sets at least one STRANDS_MESH variable first."""
    exports = _exports_before_the_first_mesh_true_fence()
    assert any(name.startswith("STRANDS_MESH") for name in exports), exports


def _readme_mesh_row() -> dict[str, str]:
    rows = [line for line in _README.read_text(encoding="utf-8").splitlines() if line.startswith("| **Mesh**")]
    assert len(rows) == 1, rows
    return dict(_INLINE.findall(rows[0]))


@pytest.mark.parametrize(
    "where, posture",
    [("docs/learn/mesh/index.md", _exports_before_the_first_mesh_true_fence), ("README.md Mesh row", _readme_mesh_row)],
)
def test_the_page_names_a_posture_the_start_gate_accepts(
    monkeypatch: pytest.MonkeyPatch, where: str, posture: Any
) -> None:
    """Copying ``Robot(mesh=True)`` from the page, with the variables it names, joins a mesh."""
    for name in [n for n in os.environ if n.startswith("STRANDS_MESH")]:
        monkeypatch.delenv(name, raising=False)
    exports = posture()
    for name, value in exports.items():
        monkeypatch.setenv(name, value)

    mesh: Any = core.Mesh.__new__(core.Mesh)
    core.Mesh.__init__(mesh, MagicMock(), "docs-mesh-guide")
    refused = mesh._refuse_under_permissive_default_acl()

    assert not refused, (
        f"under the environment {exports or '{}'} {where} names before its first mesh=True, "
        f"Mesh.start refuses: {core.PERMISSIVE_ACL_REFUSAL.splitlines()[0]}"
    )
