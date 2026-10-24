"""A robot added to a shared world costs about its compiled model in memory, not a second copy of its meshes."""

import gc
import os
import sys

import pytest

pytest.importorskip("mujoco")
from strands_robots import Robot, Simulation  # noqa: E402

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads RSS from /proc/self/statm")

BUDGET_MB = 150  # an so101 compiles to ~62 MB; before this pin each one also kept its MjSpec (~330 MB per robot)


def _rss_mb() -> float:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e6


def test_each_added_so101_stays_under_the_budget():
    Robot("so101")  # the first one pays for imports and the asset cache
    gc.collect()
    before = _rss_mb()
    for i in range(1, 4):
        Robot("so101", position=[0.5 * i, 0.0, 0.0])
    gc.collect()
    per_robot = (_rss_mb() - before) / 3
    assert per_robot < BUDGET_MB, f"{per_robot:.0f} MB per added so101 (budget {BUDGET_MB} MB)"


def test_the_world_keeps_the_compiled_model_not_the_spec():
    Robot("so101")
    assert not hasattr(Simulation.default().backend, "spec")
