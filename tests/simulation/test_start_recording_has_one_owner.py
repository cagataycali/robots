"""``start_recording`` is written once, on the shared recording mixin.

Each backend used to carry its own copy - MuJoCo, Isaac and Newton, 336 to 493
lines each - and every refusal the method makes (rate, posture flags, camera
list, collisions, a live session, the wipe ordering) had to be added three
times and graded three times. A backend now supplies only its scene's schema
(``_collect_recording_schema``) and camera names - mjlab, the fourth backend,
arrived on that contract - and a copy starting over is what this pins.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import strands_robots.simulation as simulation_pkg
from strands_robots.simulation.recording import DatasetRecordingMixin
from tests._package_ast import parse_file

_ROOT = Path(simulation_pkg.__file__).parent


def _defines(path: Path, name: str) -> bool:
    return any(isinstance(node, ast.FunctionDef) and node.name == name for node in ast.walk(parse_file(path)))


@pytest.mark.parametrize("backend", ["mujoco", "isaac", "newton", "mjlab"])
def test_no_backend_defines_its_own_start_recording(backend: str) -> None:
    offenders = [p.name for p in sorted((_ROOT / backend).glob("*.py")) if _defines(p, "start_recording")]
    assert offenders == [], f"{backend} redefines start_recording in {offenders}; supply _collect_recording_schema"
    assert _defines(_ROOT / backend / "recording.py", "_collect_recording_schema"), backend


def test_the_shared_mixin_owns_it() -> None:
    assert "start_recording" in vars(DatasetRecordingMixin)
