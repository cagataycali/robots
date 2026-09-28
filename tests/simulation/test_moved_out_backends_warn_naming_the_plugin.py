"""A built-in backend announced to move out warns at ``create_simulation`` with its install line.

#3818 removes nothing without announcing it one minor ahead with the
replacement named. Isaac and Newton leave the package in 0.7 for the
``strands-robots-sim-extras`` plugin; the warning and each backend's page
carry the same notice, so both surfaces are graded here.
"""

import contextlib
import warnings
from pathlib import Path

import pytest

import strands_robots
from strands_robots.simulation.factory import _MOVED_OUT_IN_0_7, create_simulation, list_backends

_PAGES = Path(strands_robots.__file__).resolve().parent.parent / "docs" / "learn" / "simulation"


@pytest.mark.parametrize("backend", sorted(_MOVED_OUT_IN_0_7))
def test_create_simulation_warns_with_the_install_line(backend: str) -> None:
    assert backend in list_backends(), f"{backend!r} is announced but no longer built in"
    with pytest.warns(DeprecationWarning, match="moves out of strands-robots in 0.7") as record:
        with contextlib.suppress(Exception):  # construction may need Isaac Sim or warp; the notice may not
            create_simulation(backend)
    assert _MOVED_OUT_IN_0_7[backend] in str(record[0].message)
    assert "moves out in 0.7" in (_PAGES / f"{backend}.md").read_text().lower()


def test_mujoco_does_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        create_simulation("mujoco").cleanup()
