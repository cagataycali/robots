"""The unit suite never launches a real Isaac Sim app, installed or not.

With the ``isaacsim`` 6.x pip wheel present, tests written for an Isaac-less
host (``create_world`` "fails on a host without Isaac Sim") booted Kit instead:
they failed on ``'success' == 'error'``, hung on the EULA prompt without
``OMNI_KIT_ACCEPT_EULA``, and polluted ``sys.modules`` for the xdist worker's next
test. ``conftest._no_real_isaac_sim`` blocks the import; these pin that it does.
"""

from __future__ import annotations

import importlib.util
import sys

import pytest


def test_isaacsim_is_not_importable_here() -> None:
    with pytest.raises(ImportError):
        import isaacsim  # type: ignore[import-not-found]  # noqa: F401
    assert importlib.util.find_spec("isaacsim") is None


def test_create_world_answers_the_absent_host_error_without_booting() -> None:
    from strands_robots.simulation.isaac.simulation import IsaacSimulation

    result = IsaacSimulation(num_envs=1, headless=True).create_world()

    assert result["status"] == "error"
    assert not any(m == "omni" or m.startswith(("omni.", "carb")) for m in sys.modules if sys.modules[m] is not None)


def test_is_available_reports_absent() -> None:
    from strands_robots.simulation.isaac.simulation import IsaacSimulation

    available, reason = IsaacSimulation.is_available()
    assert available is False
    assert reason
