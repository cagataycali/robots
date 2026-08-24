"""The SO-101 reference pick example must actually lift the cube.

``examples/18_so101_pick_and_lift.py`` is the roadmap starter-kit reference:
"get an SO-101 from a cube on the table to a cube in the air". A friction pinch
does not hold the cube on the shipped ``so101`` model (strands-labs/robots#2167,
#2145), so the example uses the supported grasp-assist primitive
``attach_bodies(mode="weld")``. This test pins that the composed public-API
sequence lifts the cube, so a regression in ``move_to`` / ``set_gripper`` /
``attach_bodies`` that silently stopped lifting would fail here rather than ship
a "reference pick" that does not pick.

Headless and CPU-only: ``run_pick`` renders nothing unless a video path is
passed, so no GL/EGL backend is needed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("mujoco")

_EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "18_so101_pick_and_lift.py"


def _load_example():
    # A leading digit makes the module unimportable by name; load it by path.
    spec = importlib.util.spec_from_file_location("so101_pick_and_lift", _EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_reference_pick_lifts_the_cube():
    result = _load_example().run_pick()
    assert result["status"] == "success", result
    # The example lifts ~150 mm; require a clear lift so a friction-only
    # regression (cube stays on the table, ~0 mm) fails loudly.
    assert result["success"] is True, result
    assert result["lifted_mm"] >= 80.0, result
