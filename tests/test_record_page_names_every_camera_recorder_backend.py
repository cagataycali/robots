"""The docs name every simulation backend that ships ``start_cameras_recording``.

``docs/learn/data/record.md`` once scoped the verb to ``[sim-mujoco]`` alone
while Isaac ships it too, and ``docs/learn/hardware/cameras.md`` said it writes
a dataset when it writes plain MP4s. The backend list is read from the source
tree, so a new backend that grows the verb fails here until the page names it.
"""

from __future__ import annotations

import ast
from pathlib import Path

from tests._package_ast import parse_file

REPO = Path(__file__).resolve().parents[1]
SIM = REPO / "strands_robots" / "simulation"
_DISPLAY = {"mujoco": "MuJoCo", "isaac": "Isaac", "newton": "Newton", "mjlab": "mjlab"}


def _backends_with_the_verb() -> set[str]:
    found = set()
    for path in SIM.glob("*/*.py"):
        tree = parse_file(path)
        if any(isinstance(n, ast.FunctionDef) and n.name == "start_cameras_recording" for n in ast.walk(tree)):
            found.add(path.parent.name)
    return found


def test_the_record_page_row_names_every_backend_that_ships_the_verb() -> None:
    backends = _backends_with_the_verb()
    assert {"mujoco", "isaac"} <= backends, backends
    rows = [
        r for r in (REPO / "docs/learn/data/record.md").read_text().splitlines() if "`start_cameras_recording()`" in r
    ]
    assert len(rows) == 1, rows
    for backend in sorted(backends):
        assert _DISPLAY.get(backend, backend) in rows[0], f"record.md does not name {backend}: {rows[0]}"
    assert "alone" not in rows[0], rows[0]


def test_the_cameras_page_says_the_verb_writes_mp4s() -> None:
    text = (REPO / "docs/learn/hardware/cameras.md").read_text()
    assert "`start_cameras_recording` writes each to an MP4" in text
    assert "`start_cameras_recording` writes them into a dataset" not in text
